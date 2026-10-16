"""Multi-hop family classification (SPEC 6.1 visibility rule, rulings R9 to R11). No model calls.

    uv run python -m eval.families dev
    uv run python -m eval.families packet
    uv run python -m eval.families merge --codex <verdicts.json> --claude <verdicts.json>

Fixed file names under results/families/ (the input guard allows exactly these): dev_visibility.json, families.json,
packet.md, packet_ids.json, prereg_block.txt, verdicts_codex.json, verdicts_claude.json.

A family is `questions.template(text, product)` of an answerable question. Only person, company, pr and url
families are classified; content families are listed and never enter the multi-hop set.

`dev` computes, per dev question, what strict A4 can ever read: the cards of its authorised roots plus the full card
of every one-hop neighbour listed in those cards (structural only: no 200-row page, token or call limit). Roots are
the Product, every exact hit (each folded name key of the RootSet that is a substring of the question, namesakes
included) and every fallback hit over every contiguous span of two or more WORDS. Spans that start mid-word are not
enumerated. More roots only make more content visible, which moves a family toward single-hop (the conservative
direction). An answer value is visible when its `grade.normalize` form occurs in the normalised JSON of a visible
card. A family is `multi` only if every voting dev instance needs content outside the visible set.

`packet` writes the text the two classifiers get for the held-out templates not seen in dev. `merge` combines the two
verdict files: `multi` only when both say multi, `single` when both say single, anything else is excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from eval import questions as Q
from eval.grade import normalize
from inherit.config import ROOT
from inherit.graph import cards
from inherit.tools import depth1

OUT_DIR = ROOT / "results" / "families"
ID_TYPES = ("person", "company", "pr", "url")
VERDICTS = ("multi", "single", "uncertain")
CONTENT = "content (not classified)"
EXCLUDED = "excluded (disagreement)"
ROWS, TOKENS = depth1.ROW_CAP, depth1.RESPONSE_BUDGET_TOKENS

RULE_ZH = ("多跳题族的划分在看任何留出结果之前完成：按 A4 实际能看到的序列化内容判断（根实体的卡片，加上它一跳邻居的"
           "完整卡片，包括这些卡片里列出的 ID 和名字），答案需要的字段不在这些内容里的题族，算多跳题族。")
RULE_EN = ("The multi-hop families are fixed before any held-out result, by what A4 can actually see in serialised "
           "form: the cards of the root entities, plus the full cards of their one-hop neighbours, including the IDs "
           "and names listed in those cards. A family whose answers need a field that is not in that content is a "
           "multi-hop family.")


def _ser(c: dict) -> str:
    return json.dumps(c, ensure_ascii=False, separators=(",", ":"))


def sha(b: bytes | str) -> str:
    return hashlib.sha256(b.encode() if isinstance(b, str) else b).hexdigest()


# ---------------------------------------------------------------- dev visibility
def authorised_roots(graph, text: str, product: str) -> list[str]:
    """Every root strict A4 may ever read for this question, found with the frozen RootSet."""
    rs = depth1.RootSet(graph, text, product)
    rs._load()  # noqa: SLF001
    roots = {f"Product:{product}"}
    for key, ids in rs._exact.items():  # noqa: SLF001
        if key in rs._q:  # noqa: SLF001
            roots |= ids
    words = depth1.normalise(text).split()
    spans = {" ".join(words[i:j]) for i in range(len(words)) for j in range(i + 2, len(words) + 1)}
    for span in sorted(spans):
        roots.update(rs._fallback(span))  # noqa: SLF001
    return sorted(roots)


class Viewer:
    """Cards of one product graph, built once."""

    def __init__(self, graph):
        self.graph, self._cards, self._norm = graph, {}, {}

    def card(self, node_id: str) -> dict:
        if node_id not in self._cards:
            self._cards[node_id] = cards.card(self.graph, node_id)
            self._norm[node_id] = normalize(_ser(self._cards[node_id]))
        return self._cards[node_id]

    def norm(self, node_id: str) -> str:
        self.card(node_id)
        return self._norm[node_id]

    def view(self, roots: list[str]):
        """(ids of the visible cards, groups): groups maps (root, edge, dir) to the neighbour ids it lists."""
        groups: dict[tuple, list[str]] = defaultdict(list)
        for r in roots:
            for e in self.card(r)["neighbours"]:
                if e["id"] not in groups[(r, e["edge"], e["dir"])]:
                    groups[(r, e["edge"], e["dir"])].append(e["id"])
        visible = set(roots) | {i for ids in groups.values() for i in ids}
        for i in visible:
            self.card(i)
        return sorted(visible), groups

    def large(self, ids: list[str]) -> bool:
        return len(ids) > ROWS or depth1.tokens([self.card(i) for i in ids]) > TOKENS


def _paged_only(card_id: str, roots: list[str], groups: dict, viewer: Viewer) -> bool:
    """True if the card is not a root and every (root, edge, dir) group listing it is over the row or token cap."""
    mine = [ids for ids in groups.values() if card_id in ids]
    return card_id not in roots and bool(mine) and all(viewer.large(ids) for ids in mine)


def classify_question(q: Q.Question, viewer: Viewer) -> dict:
    roots = authorised_roots(viewer.graph, q.text, q.product)
    visible, groups = viewer.view(roots)
    values = []
    for v in q.gold or []:
        n = normalize(v)
        hits = [i for i in visible if n and n in viewer.norm(i)]
        paged = bool(hits) and all(_paged_only(i, roots, groups, viewer) for i in hits)
        values.append({"value": v, "visible": bool(hits), "witness": hits[:3] if hits else "missing",
                       "n_witness": len(hits), "paging_only": paged if hits else None})
    miss = [x["value"] for x in values if not x["visible"]]
    verdict = "no_gold" if not values else ("visible" if not miss else "needs_more")
    return {"family": Q.template(q.text, q.product), "type": q.type, "roots": len(roots), "visible_cards": len(visible),
            "values": values, "verdict": verdict, "missing": miss}


def family_verdict(instances: list[str]) -> str:
    """R9: multi only if every voting instance needs more; one visible instance makes the family single."""
    votes = [v for v in instances if v in ("visible", "needs_more")]
    if not votes:
        return "no_votes"
    return "single" if "visible" in votes else "multi"


def classify_dev(questions: list[Q.Question], graph_for) -> dict:
    """`graph_for(product)` gives a read-only graph handle. Returns {"questions": {...}, "families": {...}}."""
    viewers: dict[str, Viewer] = {}
    recs: dict[str, dict] = {}
    for q in questions:
        if q.type == "unanswerable" or q.type not in ID_TYPES:
            continue
        v = viewers.setdefault(q.product, Viewer(graph_for(q.product)))
        recs[str(q.key)] = classify_question(q, v)
    fams: dict[str, dict] = {}
    for q in questions:
        if q.type != "unanswerable":
            f = fams.setdefault(Q.template(q.text, q.product), {"types": set(), "inst": []})
            f["types"].add(q.type)
            if str(q.key) in recs:
                f["inst"].append(recs[str(q.key)]["verdict"])
    out = {}
    for t, f in sorted(fams.items()):
        types = "/".join(sorted(f["types"]))
        if not f["inst"]:
            out[t] = {"source": "dev", "type": types, "verdict": CONTENT}
        else:
            out[t] = {"source": "dev", "type": types, "verdict": family_verdict(f["inst"]),
                      "votes": dict(sorted(Counter(f["inst"]).items()))}
    return {"questions": recs, "families": out}


# ---------------------------------------------------------------- held-out strings
def build_packet(pairs: list[tuple[str, str]]) -> tuple[str, list[dict]]:
    """(packet text, id table). Only ID-type templates are sent; content templates are not."""
    sent = [(t, ty) for t, ty in pairs if any(x in ID_TYPES for x in ty.split("/"))]
    rows = [{"id": f"u{i:03d}", "template": t, "type": ty} for i, (t, ty) in enumerate(sent, 1)]
    text = "\n".join([
        "# Classify question templates as multi-hop or single-hop", "",
        "## Rule (SPEC 6.1, quoted)", "", RULE_ZH, "", RULE_EN, "",
        "You see only the question template and the card field list. You do not see any graph or any answer. "
        "Decide from the question text whether the answer needs a field that the visible cards do not hold: "
        "`multi` if it does, `single` if the answer is on the root card or on a one-hop neighbour card, "
        "`uncertain` if you cannot tell. Placeholders: <P> product, <E> employee id, <URL> link, <X> product-like "
        "name, <C> customer company, <N> number.", "",
        "## Card field list", "", cards.render_fields(), "",
        "## Templates", "", *[f"{r['id']} [{r['type']}] {r['template']}" for r in rows], "",
        "## Answer format", "",
        'Reply with JSON only: `{"verdicts": [{"id": "u001", "verdict": "multi" | "single" | "uncertain", '
        '"reason": "..."}]}` with one entry per id above.', ""])
    return text, rows


def read_verdicts(path: Path) -> dict[str, str]:
    """id -> verdict for the well-formed entries. A malformed file, entry, verdict or duplicated id gives nothing."""
    try:
        items = json.loads(Path(path).read_text(encoding="utf-8"))["verdicts"]
        assert isinstance(items, list)
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        return {}
    seen: dict[str, str | None] = {}
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("id"), str):
            continue
        i, v = it["id"], it.get("verdict")
        seen[i] = None if i in seen or v not in VERDICTS else v  # a repeated id is ambiguous: dropped
    return {i: v for i, v in seen.items() if v}


def merge_pair(a: str | None, b: str | None) -> str:
    return a if a == b and a in ("multi", "single") else EXCLUDED


def merge_unseen(rows: list[dict], codex: dict[str, str], claude: dict[str, str]) -> dict[str, dict]:
    return {r["template"]: {"source": "held-unseen", "type": r["type"], "id": r["id"],
                            "codex": codex.get(r["id"]), "claude": claude.get(r["id"]),
                            "verdict": merge_pair(codex.get(r["id"]), claude.get(r["id"]))} for r in rows}


# ---------------------------------------------------------------- outputs
def render_block(multi) -> str:
    """The PREREG lines read back by `stats.read_multihop_families`."""
    fams = sorted(multi)
    return "\n".join(f"multi-hop family: {t}" for t in fams) if fams else "multi-hop families: none"


def multi_set(families: dict) -> set[str]:
    return {t for t, f in families.items() if f["verdict"] == "multi"}


def _input_hashes() -> dict:
    out: dict = {"bundle_sha256": None, "facts_hash": None}
    try:
        from inherit.data.bundle import bundle_hash
        out["bundle_sha256"] = bundle_hash(ROOT / "data" / "bundle")
    except Exception:  # noqa: BLE001
        pass
    try:
        from eval.receipt import facts_hash
        out["facts_hash"] = facts_hash(ROOT / "data" / "facts")
    except Exception:  # noqa: BLE001
        pass
    return out


def build_families_doc(families: dict, hashes: dict) -> dict:
    counts = Counter(f["verdict"] for f in families.values())
    return {"families": dict(sorted(families.items())), "counts": dict(sorted(counts.items())),
            "multi_hop_set": sorted(multi_set(families)), "hashes": hashes}


def _write(out: Path, name: str, obj) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_dev(out: Path, graph_for=depth1.agent_graph, questions=None) -> dict:
    res = classify_dev(questions if questions is not None else Q.load_questions("dev"), graph_for)
    _write(out, "dev_visibility.json", res)
    _write(out, "families.json", build_families_doc(res["families"], {**_input_hashes(), "packet": None,
                                                                      "codex_verdicts": None, "claude_verdicts": None}))
    return res


def cmd_packet(out: Path) -> dict:
    text, rows = build_packet(Q.unseen_held_templates())
    _write(out, "packet.md", text)
    _write(out, "packet_ids.json", rows)
    return {"templates": len(rows), "sha256": sha(text)}


def cmd_merge(out: Path, codex: Path, claude: Path) -> dict:
    dev = _load_json(out / "dev_visibility.json")["families"]
    rows = _load_json(out / "packet_ids.json")
    unseen = merge_unseen(rows, read_verdicts(codex), read_verdicts(claude))
    hashes = {**_input_hashes(), "packet": sha((out / "packet.md").read_bytes()),
              "codex_verdicts": sha(Path(codex).read_bytes()), "claude_verdicts": sha(Path(claude).read_bytes())}
    doc = build_families_doc({**dev, **unseen}, hashes)
    pairs = Counter((u["codex"] or "missing", u["claude"] or "missing") for u in unseen.values())
    doc["agreement"] = {"agree": sum(n for (a, b), n in pairs.items() if a == b and a != "missing"),
                        "total": len(unseen), "pairs": {f"{a}/{b}": n for (a, b), n in sorted(pairs.items())}}
    _write(out, "families.json", doc)
    _write(out, "prereg_block.txt", render_block(multi_set(doc["families"])) + "\n")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval.families")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dev")
    sub.add_parser("packet")
    m = sub.add_parser("merge")
    m.add_argument("--codex", type=Path, required=True)
    m.add_argument("--claude", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "dev":
        res = cmd_dev(a.out)
        c = Counter(f["verdict"] for f in res["families"].values())
        print({k: c.get(k, 0) for k in ("multi", "single", "no_votes", CONTENT)})
    elif a.cmd == "packet":
        print(cmd_packet(a.out))
    else:
        doc = cmd_merge(a.out, a.codex, a.claude)
        print(doc["counts"], doc["agreement"])
        print(render_block(multi_set(doc["families"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
