"""A0f: the same tools, loop and retriever settings as A0, over entity cards instead of raw records.

* corpus: the product's serialised cards (`cards_for_product`, messages included), chunked and retrieved exactly as A0's;
* `retrieve_all` returns chunks of serialised cards;
* the metadata tools keep their extraction, selectors and grouping but return the matched cards (design F4):
  role_to_id, name_to_id, id_to_name, id_to_role -> Person cards; company_to_id, id_to_company -> Customer cards;
  pr_search -> PR cards; url_search -> Document or Url cards (a document wins a link collision, as upstream). Cards in a
  list are ordered by card id with duplicates removed; an entry with no card in the graph is left out, and an empty
  result carries the upstream "No ... found" message. The Task 7 response budget applies (200 cards, 12,000 tokens of
  serialised cards; the envelope says `truncated` when it cuts).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from baselines import paper_tools
from baselines.paper_tools import PaperTools
from inherit import agent
from inherit.graph import cards
from inherit.tools.depth1 import RESPONSE_BUDGET_TOKENS, ROW_CAP

ARM = "A0f"


def config() -> dict:
    return {**agent.arm_config("paper-cards"), "corpus": "cards", "card_version": cards.CARD_VERSION}


class CardTools(PaperTools):
    def __init__(self, loop, lookups, retriever, graph):
        super().__init__(loop, lookups, retriever, cards=True)
        self.graph, self.truncated = graph, False
        self.support = None  # answer support is the card projection alone (no free-text matching)
        self.chars = self.n_cards = 0

    def _card(self, node_id: str) -> dict | None:
        try:
            return cards.card(self.graph, node_id)
        except LookupError:
            return None

    def _fit(self, card: dict) -> bool:
        """Take the card into this response if the row and size caps allow it."""
        size = len(json.dumps(card, ensure_ascii=False, separators=(",", ":")))
        if self.n_cards >= ROW_CAP or self.chars + size > RESPONSE_BUDGET_TOKENS * 4:
            self.truncated = True
            return False
        self.n_cards += 1
        self.chars += size
        return True

    def _cards(self, node_ids: list[str]) -> list[dict]:
        found = [c for c in (self._card(i) for i in dict.fromkeys(node_ids)) if c]
        return [c for c in sorted(found, key=lambda c: c["id"]) if self._fit(c)]

    def present(self, name: str, res: Any) -> Any:
        self.chars = self.n_cards = 0
        lk = self.lk
        if name in ("role_to_id_search", "name_to_id_search"):
            out = {k: self._cards([f"Person:{e}" for e in ids]) for k, ids in res.items()}
        elif name == "company_to_id_search":
            out = {k: self._cards([f"Customer:{c}" for c in ids]) for k, ids in res.items()}
        elif name in ("id_to_name_search", "id_to_role_search"):
            out = {i: cs[0] for i in sorted(res) for cs in [self._cards([f"Person:{i}"])] if cs}
        elif name == "id_to_company_search":
            out = {i: cs[0] for i in sorted(res) for cs in [self._cards([f"Customer:{lk.cust_ids[i.lower()]}"])] if cs}
        else:
            ids = [f"Document:{r['id']}" if "document_link" in r else (f"PR:{r['link']}" if name == "pr_search"
                                                                       else f"Url:{r['link']}") for r in res]
            out = self._cards(ids)
        return {k: v for k, v in out.items() if v} if isinstance(out, dict) else out

    # ---- answer support: the shared card projection (`agent.project_depth1`), as for A4 and A4c
    def matched_rows(self, env: dict, matched: list[str]) -> list[dict]:
        found: list[dict] = []

        def walk(o: Any) -> None:
            if isinstance(o, dict) and "neighbours" in o and "label" in o:
                found.append(o)
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(env.get("result"))
        return agent.project_depth1({"rows": found}, {}) if found else []

    def chunk_rows(self, chunks: list[str]) -> list[dict]:
        rows: list[dict] = []
        for c in chunks:
            rows += chunk_rows(c)
        return rows


_STR = r'"((?:[^"\\]|\\.)*)"'
_FIELD = re.compile(r'"(id|eid|company|link)":' + _STR)
_CUSTOMER = re.compile(r'"label":"Customer","id":' + _STR + r',"name":' + _STR + "|"
                       r'"id":' + _STR + r',"label":"Customer","name":' + _STR)  # neighbour entry | card head
_SRC = re.compile(r'"src":\[(.*?)\]')
_ANSWER_LABELS = ("Person:", "PR:", "Url:")


def _unquote(raw: str) -> str:
    try:
        return json.loads(f'"{raw}"')
    except ValueError:
        return raw


def chunk_rows(chunk: str) -> list[dict]:
    """Guard rows for one retrieved chunk of serialised cards. A chunk that is a whole card goes through the shared
    projection; a fragment (the splitter cuts cards) is read for the same answer-bearing fields only: the ids of
    Person, PR and Url nodes (-> eid, link), and the card props eid, company and link. Text inside a card's
    `content` is a JSON string whose quotes are escaped, so it matches none of these."""
    try:
        card = json.loads(chunk)
    except ValueError:
        card = None
    if isinstance(card, dict) and "neighbours" in card and "label" in card:
        return agent.project_depth1({"rows": [card]}, {})
    rows: list[dict] = []
    refs: list[str] = []
    for key, raw in _FIELD.findall(chunk):
        v = _unquote(raw)
        if key == "id":
            refs.append(v)
            if v.startswith(_ANSWER_LABELS):
                rows.append({"answer": v.split(":", 1)[1]})
        else:
            rows.append({"answer": v})
    for m in _CUSTOMER.findall(chunk):  # a Customer's answer value is its company name (the display name)
        rows.append({"answer": _unquote(m[1] or m[3])})
    for group in _SRC.findall(chunk):
        rows.append({"evidence": [_unquote(x) for x in re.findall(_STR, group)]})
    return [*rows, {"refs": refs}] if refs else rows


def run(inp, *, arm: str = ARM, model: Callable | None = None, retriever=None, lookups=None, graph=None,
        clock: Callable[[], float] = time.monotonic):
    from baselines import react_paper

    def factory(loop):
        g = graph if graph is not None else agent.open_graph(inp.product)
        r = retriever if retriever is not None else react_paper.shared_retriever("cards", inp.product, graph=g)
        return CardTools(loop, lookups or react_paper.load_lookups(inp.product), r, g)

    return agent.answer(inp, toolset="paper-cards", arm=arm, model=model, clock=clock, tools_factory=factory)
