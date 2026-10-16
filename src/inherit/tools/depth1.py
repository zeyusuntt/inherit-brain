"""The depth-1 atomic tool (arms A4 and A4c): root resolution, root cards and one-hop neighbour cards.

An `Episode` is one question. It exposes three operations, each counted against one shared budget of 12 calls and each
traced: `resolve(text)`, `get_card(id)` and `neighbors(...)`. Everything is enforced here, on the server side:

* roots come only from a substring of the question (`RootSet`), plus the episode's Product;
* A4 (`strict=True`) reads only authorised roots. A4c (`strict=False`) also reads entity ids that appeared in a card
  it received (the card itself or a neighbour entry), at hop = hop of the exposing card + 1. Record ids in `src`
  and rows the model did not receive never authorise anything;
* `neighbors` returns the cards of the distinct nodes one edge of the given type away, filtered, counted or grouped
  on the published card fields only, in id order, at most 200 rows and 12,000 tokens (chars / 4) per response.
  Rows beyond that are not returned: the envelope says `truncated` and carries an opaque `cursor`; passing it back
  (a call like any other) returns the next page. count and group run over all matches before paging.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import time
import unicodedata
from collections import defaultdict

import redis
from falkordb import FalkorDB
from falkordb.graph import Graph

from inherit.config import ROOT, load_settings
from inherit.contracts import TraceOp
from inherit.graph import cards
from inherit.ontology import SIGNATURES, label_of

RESPONSE_BUDGET_TOKENS = 12_000
ROW_CAP = 200
CALL_CAP = 12
FALLBACK_LABELS = ("Bug", "Feature", "Document", "Competitor")  # full-text fallback reaches only these
FALLBACK_TOP = 5
# Frozen per-label mention sets (stored field names; a list value contributes every item).
MENTIONS: dict[str, tuple[str, ...]] = {
    "Product": ("name", "aliases", "previous_names"),
    "Person": ("name", "eid"), "Customer": ("company", "source_id"), "Document": ("type", "source_id"),
    "Bug": ("title", "aliases", "component"), "PR": ("title", "link"), "Feature": ("title", "aliases"),
    "Competitor": ("name",), "Url": ("link",),
}
CARD_FIELDS = frozenset({"id", "label", "name"}).union(*(set(v) for v in cards.ALLOWLIST.values()))


class ToolError(ValueError):
    """A refused or invalid tool call. The message is safe to show the model."""


# ---------------------------------------------------------------- text
_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"'})


def normalise(text: str) -> str:
    """NFKC, curly quotes and apostrophes to straight, whitespace runs to one space, ends trimmed."""
    return " ".join(unicodedata.normalize("NFKC", text).translate(_QUOTES).split())


def fold(text: str) -> str:
    return normalise(text).casefold()  # case folding is for matching only


def tokens(obj) -> int:
    """Token estimate of a serialised response part: characters / 4, rounded up."""
    return math.ceil(len(json.dumps(obj, ensure_ascii=False, separators=(",", ":"))) / 4)


def _escape(span: str) -> str:
    """A full-text query that treats the span literally: every punctuation character is escaped."""
    return "".join(c if c.isalnum() or c.isspace() or c == "_" else "\\" + c for c in span)


# ---------------------------------------------------------------- reading
class _Reader:
    """GRAPH.RO_QUERY only; remembers the distinct query texts of the current call for the trace."""

    def __init__(self, graph):
        self.graph = graph
        self.queries: list[str] = []

    def ro_query(self, q, params=None):
        if q not in self.queries:
            self.queries.append(q)
        return self.graph.ro_query(q, params)


def read_only_graph(host: str, port: int, user: str, password: str, name: str) -> Graph:
    """A graph handle over a bare connection. The agent users hold only GRAPH.RO_QUERY, so the FalkorDB client
    constructor (which reads INFO) cannot connect as them."""
    conn = redis.Redis(host=host, port=port, username=user, password=password, decode_responses=True)
    client = FalkorDB.__new__(FalkorDB)
    client.connection, client.execute_command, client.flushdb = conn, conn.execute_command, conn.flushdb
    return Graph(client, name)


def agent_graph(product: str, server: str = "6.0") -> Graph:
    """herb:<product> as the read-only agent_<product> user (passwords in data/acl.json)."""
    pw = json.loads((ROOT / "data" / "acl.json").read_text())[server][f"agent_{product}"]
    kw = FalkorDB.from_url(load_settings().falkor_urls[server]).connection.connection_pool.connection_kwargs
    return read_only_graph(kw["host"], kw["port"], f"agent_{product}", pw, f"herb:{product}")


# ---------------------------------------------------------------- root resolution
class RootSet:
    """Maps a span of the question to entity ids. Refuses text that is not a substring of the question."""

    def __init__(self, graph, question: str, product: str):
        self.graph, self.question, self.product = graph, question, product
        self._q = fold(question)
        self._exact: dict[str, set[str]] | None = None
        self._present: set[str] = set()
        self.names: dict[str, tuple[str, str]] = {}  # id -> (label, display name)
        self.queries: list[str] = []

    def _load(self) -> None:
        self._exact, self._present = defaultdict(set), set()
        for label, fields in MENTIONS.items():
            cols = ", ".join(f"n.{f} AS m{i}" for i, f in enumerate(fields))
            q = f"MATCH (n:{label}) RETURN n.id, {cards._name_cols('n')}, {cols}"  # noqa: SLF001
            self.queries.append(q)
            for row in self.graph.ro_query(q).result_set:
                nid, names, vals = row[0], row[1:1 + len(cards.NAME_FIELDS)], row[1 + len(cards.NAME_FIELDS):]
                self._present.add(label)
                self.names[nid] = (label, cards.display_name(label, cards._name_row(names), nid))  # noqa: SLF001
                for v in vals:
                    for item in v if isinstance(v, list) else [v]:
                        if isinstance(item, str) and fold(item):
                            self._exact[fold(item)].add(nid)

    def resolve(self, text: str) -> list[str]:
        if not isinstance(text, str) or not normalise(text):
            raise ToolError("text must not be empty")
        span = fold(text)
        if span not in self._q:
            raise ToolError("text is not a part of the question")
        if self._exact is None:
            self._load()
        hit = self._exact.get(span)
        if hit:
            return sorted(hit)  # every exact namesake
        if len(span.split()) < 2:
            return []
        return self._fallback(normalise(text))

    def _fallback(self, span: str) -> list[str]:
        scored: list[tuple[float, str]] = []
        for label in FALLBACK_LABELS:
            if label not in self._present:
                continue
            q = f"CALL db.idx.fulltext.queryNodes('{label}', $q) YIELD node, score RETURN node.id, score"
            self.queries.append(q)
            scored += [(float(s), i) for i, s in self.graph.ro_query(q, {"q": _escape(span)}).result_set]
        scored.sort(key=lambda t: (-t[0], t[1]))
        if len(scored) > FALLBACK_TOP:
            cut = scored[FALLBACK_TOP - 1][0]
            scored = [t for t in scored if t[0] >= cut]  # ties at the cut stay
        return [i for _, i in scored]


# ---------------------------------------------------------------- episode
def _eq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return a == b


class Episode:
    def __init__(self, graph, question: str, product: str, *, strict: bool = True, max_calls: int = CALL_CAP):
        self.graph, self.question, self.product, self.strict = graph, question, product, strict
        self.max_calls = max_calls
        self.root_set = RootSet(graph, question, product)
        self._reader = _Reader(graph)
        self.product_id = f"Product:{product}"
        self.roots: set[str] = {self.product_id}
        self.exposed: dict[str, int] = {self.product_id: 0}  # entity id -> discovery hop
        self.max_hop = 0  # monotonic maximum over received cards; `received`/`exposed` keep minimum distances
        self.received: dict[str, int] = {}  # full cards the model has been given -> hop
        self.calls = 0
        self.trace: list[TraceOp] = []

    # ------------------------------------------------------------ accounting
    def stats(self) -> dict:
        return {"calls": self.calls, "distinct_cards": len(self.received),
                "max_hop": self.max_hop}

    def _call(self, tool: str, params: dict, fn):
        if self.calls >= self.max_calls:
            raise ToolError(f"call limit reached: at most {self.max_calls} calls per question")
        self.calls += 1
        self._reader.queries = []
        self.root_set.queries = []
        t0 = time.perf_counter()
        try:
            env = fn()
        except (ToolError, LookupError) as e:
            self.trace.append(TraceOp(tool, "refused", "", {**params, "error": str(e)}, 0,
                                      round((time.perf_counter() - t0) * 1000, 3)))
            if isinstance(e, ToolError):
                raise
            raise ToolError(str(e)) from e
        n = env.get("count") if env.get("count") is not None else len(env.get("groups") or env["rows"])
        queries = [*self.root_set.queries, *self._reader.queries]
        self.trace.append(TraceOp(tool, "GRAPH.RO_QUERY", "\n".join(queries), params, n,
                                  round((time.perf_counter() - t0) * 1000, 3)))
        return env

    def _authorised(self, node_id: str) -> int:
        """Hop of an id the caller may read, or raise."""
        if not isinstance(node_id, str):
            raise ToolError("id must be a string")
        if node_id in self.roots:
            return 0
        if not self.strict and node_id in self.exposed:
            return self.exposed[node_id]
        raise ToolError(f"{node_id!r} is not an authorised root"
                        + ("" if self.strict else " or a reference from a card you received"))

    def _receive(self, c: dict, hop: int) -> None:
        self.max_hop = max(self.max_hop, hop)
        self.received[c["id"]] = min(self.received.get(c["id"], hop), hop)
        self.exposed[c["id"]] = min(self.exposed.get(c["id"], hop), hop)
        for e in c["neighbours"]:
            self.exposed[e["id"]] = min(self.exposed.get(e["id"], hop + 1), hop + 1)

    # ------------------------------------------------------------ paging
    @staticmethod
    def _sig(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def _offset(self, cursor, sig: str) -> int:
        if cursor is None:
            return 0
        try:
            d = json.loads(base64.urlsafe_b64decode(str(cursor).encode()))
            ok = isinstance(d, dict) and d.get("s") == sig and isinstance(d.get("o"), int) and d["o"] >= 0
        except (ValueError, binascii.Error):
            ok = False
        if not ok:
            raise ToolError("cursor does not belong to this request")
        return d["o"]

    @staticmethod
    def _cursor(sig: str, offset: int) -> str:
        return base64.urlsafe_b64encode(json.dumps({"s": sig, "o": offset}).encode()).decode()

    @staticmethod
    def _fits(rows: list, row) -> bool:
        """Would `row` keep the serialised rows within the response budget?"""
        size = 2 + sum(len(json.dumps(r, ensure_ascii=False, separators=(",", ":"))) for r in rows) + len(rows)
        add = len(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        return size + add <= RESPONSE_BUDGET_TOKENS * 4

    def _page(self, total: int, offset: int, make, sig: str):
        """Rows [offset:] from `make(i)` until the row cap or the budget; a row over the budget alone is refused."""
        rows: list = []
        i = offset
        while i < total and len(rows) < ROW_CAP:
            row = make(i)
            if not self._fits(rows, row):
                if not rows:
                    raise ToolError(f"a single row exceeds the {RESPONSE_BUDGET_TOKENS:,}-token response budget "
                                    "and is not returned")
                break
            rows.append(row)
            i += 1
        more = i < total
        return rows, more, (self._cursor(sig, i) if more else None)

    # ------------------------------------------------------------ operations
    def resolve(self, text: str, cursor: str | None = None) -> dict:
        def run():
            sig = self._sig("resolve", text)
            offset = self._offset(cursor, sig)
            found = self.root_set.resolve(text)
            names = self.root_set.names
            make = lambda i: {"id": found[i], "label": names[found[i]][0], "name": names[found[i]][1]}  # noqa: E731
            rows, more, nxt = self._page(len(found), offset, make, sig)
            for r in rows:
                self.roots.add(r["id"])
                self.exposed[r["id"]] = 0
            return {"rows": rows, "truncated": more, "cursor": nxt}
        return self._call("resolve", {"text": text, "cursor": cursor}, run)

    def get_card(self, node_id: str) -> dict:
        def run():
            hop = self._authorised(node_id)
            c = cards.card(self._reader, node_id)
            if not self._fits([], c):
                raise ToolError(f"card {node_id} exceeds the {RESPONSE_BUDGET_TOKENS:,}-token response budget "
                                "and is not returned")
            self._receive(c, hop)
            return {"rows": [c], "truncated": False, "cursor": None}
        return self._call("get_card", {"id": node_id}, run)

    def _check_field(self, name, what: str) -> None:
        if name not in CARD_FIELDS:
            raise ToolError(f"{what} {name!r} is not a card field (id, label, name or a property of a card)")

    def _check_where(self, where) -> list[tuple[str, list]]:
        if where is None:
            return []
        if not isinstance(where, dict):
            raise ToolError("where must be an object of card field -> value or list of values")
        out = []
        for k, v in sorted(where.items()):
            self._check_field(k, "where field")
            vals = v if isinstance(v, list) else [v]
            if not all(isinstance(x, (str, int, float, bool)) for x in vals):
                raise ToolError(f"where value for {k!r} must be a string, number, boolean or a list of them")
            out.append((k, vals))
        return out

    @staticmethod
    def _field(c: dict, name: str):
        return c[name] if name in ("id", "label", "name") else c["props"].get(name)

    def neighbors(self, root_id: str, edge_type: str, direction: str, where: dict | None = None,
                  agg: str | None = None, group_field: str | None = None, cursor: str | None = None) -> dict:
        params = {"root_id": root_id, "edge_type": edge_type, "direction": direction, "where": where,
                  "agg": agg, "group_field": group_field, "cursor": cursor}

        def run():
            if edge_type not in SIGNATURES or edge_type in cards.EXCLUDED_EDGES:
                raise ToolError(f"unknown or unavailable edge type {edge_type!r}")
            if direction not in ("out", "in"):
                raise ToolError("direction must be 'out' or 'in'")
            if agg not in (None, "count", "group"):
                raise ToolError("agg must be null, 'count' or 'group'")
            if agg == "group":
                if group_field is None:
                    raise ToolError("agg 'group' needs group_field")
                self._check_field(group_field, "group_field")
            elif group_field is not None:
                raise ToolError("group_field is only for agg 'group'")
            conds = self._check_where(where)
            hop = self._authorised(root_id)
            sig = self._sig("neighbors", root_id, edge_type, direction, where, agg, group_field)
            offset = self._offset(cursor, sig)
            if cursor is not None and agg == "count":
                raise ToolError("count has no pages")

            pat = f"-[:{edge_type}]->(m)" if direction == "out" else f"<-[:{edge_type}]-(m)"
            q = f"MATCH (r:{cards._label(root_id)} {{id: $id}}){pat} RETURN DISTINCT m.id"  # noqa: SLF001
            matched = sorted(r[0] for r in self._reader.ro_query(q, {"id": root_id}).result_set)
            # id and label are known without building the card
            for k, vals in conds:
                if k == "id":
                    matched = [i for i in matched if any(_eq(i, v) for v in vals)]
                elif k == "label":
                    matched = [i for i in matched if any(_eq(label_of(i), v) for v in vals)]
            built: dict[str, dict] = {}

            def build(i: str) -> dict:
                if i not in built:
                    built[i] = cards.card(self._reader, i)
                return built[i]

            need_card = [(k, vals) for k, vals in conds if k not in ("id", "label")]
            if need_card:
                matched = [i for i in matched
                           if all(any(self._field(build(i), k) is not None and _eq(self._field(build(i), k), v)
                                      for v in vals) for k, vals in need_card)]
            if agg == "count":
                return {"rows": [], "truncated": False, "cursor": None, "count": len(matched)}
            if agg == "group":
                counts: dict[str, list] = {}
                for i in matched:
                    v = self._field(build(i), group_field)
                    counts.setdefault(json.dumps(v, sort_keys=True, ensure_ascii=False), [v, 0])[1] += 1
                groups = [{"value": v, "count": n} for _, (v, n) in sorted(counts.items(), key=lambda kv: (-kv[1][1], kv[0]))]
                page, more, nxt = self._page(len(groups), offset, lambda i: groups[i], sig)
                return {"rows": [], "truncated": more, "cursor": nxt, "groups": page}
            rows, more, nxt = self._page(len(matched), offset, lambda i: build(matched[i]), sig)
            for c in rows:
                self._receive(c, hop + 1)
            return {"rows": rows, "truncated": more, "cursor": nxt}
        return self._call("neighbors", params, run)
