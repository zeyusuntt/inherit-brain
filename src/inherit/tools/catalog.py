"""The fixed Cypher tool catalogue of the A5 agent (spec 5.4): ten tools, one per question shape.

Every query is hand-written, parameterised Cypher run with GRAPH.RO_QUERY on the product's `agent_<p>` connection.
Values travel as parameters; enums only choose among constant query fragments, so no Cypher is built from model text
and every variable-length bound is a literal. A tool aggregates over all matches first, then pages: at most 200 rows
and 12,000 tokens (chars / 4) per response (the Task 7 envelope), all ties at the top of a ranking kept, an opaque
`cursor` bound to the product, the tool and the filters.

Rows are typed, which is what the output guard checks (guard.py):
* `answer`: the graded value (employee id for people, company name, PR or URL link, or text for content);
* `refs`: label-qualified node ids the row names; `evidence`: record ids (edge `src`) along the returned path.

Property and edge names follow what Tasks 4a, 4b and 4c write (semantic-facts-contract.md; bug facts as written by 4b:
the BugEvent record id is `message_id`). The what-if tool (`if_leaves`) is a separate class of operation: it runs under
the `whatif_<p>` identity and is delegated to Task 12.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

from inherit.contracts import TraceOp
from inherit.extract.names import ROLE_NOUNS
from inherit.graph import cards
from inherit.tools.depth1 import (FALLBACK_LABELS, FALLBACK_TOP, ROW_CAP, RESPONSE_BUDGET_TOKENS, ToolError, _escape,
                                  fold, normalise, tokens)

ROLES = tuple(ROLE_NOUNS)  # canonical directory roles
REASONS = ("security", "product_direction", "introduced_bugs", "other")
BETWEENNESS_SEED, BETWEENNESS_SAMPLE = 42, 1000  # fixed; the graph holds far fewer than 1,000 people per product
PATH_CAP = 100


@dataclass(frozen=True)
class ToolResult:
    rows: list[dict]
    truncated: bool
    cursor: str | None
    ops: list[TraceOp]
    path: list[str]  # ids along the returned rows (refs then evidence), for the demo's clickable path
    ms: float


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params_schema: dict
    kind: str  # "readonly" (GRAPH.RO_QUERY on agent_<p>) or "whatif" (whatif_<p>, Task 12)
    impl: Callable

    def run(self, conn, product: str, **params) -> ToolResult:
        p = _validate(self.params_schema, params)
        if self.kind == "whatif":
            return self.impl(conn, product, **p)
        if getattr(conn, "name", None) != f"herb:{product}":
            raise ToolError(f"the connection is not the graph of product {product!r}")
        sig = _sig(product, self.name, p)
        offset = _offset(p.get("cursor"), sig)
        ctx = _Ctx(self.name, conn)
        t0 = time.perf_counter()
        rows = self.impl(ctx, p)
        page, more, cursor = _page(rows, offset, sig)
        return ToolResult(page, more, cursor, ctx.ops, _path(page), round((time.perf_counter() - t0) * 1000, 3))


# ---------------------------------------------------------------- parameters and paging
def _s(desc: str, **kw) -> dict:
    return {"type": "string", "description": desc, **kw}


def _enum(desc: str, values, default: str | None = None) -> dict:
    d = {"type": "string", "enum": list(values), "description": desc}
    if default is not None:
        d["default"] = default
    return d


def _int(desc: str, default: int, lo: int, hi: int) -> dict:
    return {"type": "integer", "description": desc, "default": default, "minimum": lo, "maximum": hi}


CURSOR = _s("Opaque cursor from a truncated result of this same call (same tool and filters); returns the next page. "
            "Each page counts as a call.")


def _schema(props: dict, required: tuple = ()) -> dict:
    return {"type": "object", "properties": {**props, "cursor": CURSOR}, "required": list(required),
            "additionalProperties": False}


def _validate(schema: dict, params: dict) -> dict:
    props = schema["properties"]
    out: dict[str, Any] = {k: v["default"] for k, v in props.items() if "default" in v}
    for k, v in params.items():
        if k not in props:
            raise ToolError(f"unknown parameter {k!r}; parameters: {', '.join(sorted(props))}")
        if v is None:
            continue
        spec = props[k]
        if spec["type"] == "string":
            if not isinstance(v, str) or not normalise(v):
                raise ToolError(f"{k} must be a non-empty string")
            if "enum" in spec and v not in spec["enum"]:
                raise ToolError(f"{k} must be one of {spec['enum']}")
        elif spec["type"] == "integer":
            if isinstance(v, bool) or not isinstance(v, int) or not spec["minimum"] <= v <= spec["maximum"]:
                raise ToolError(f"{k} must be an integer from {spec['minimum']} to {spec['maximum']}")
        elif spec["type"] == "array":
            if not isinstance(v, list) or not v or not all(isinstance(x, str) for x in v):
                raise ToolError(f"{k} must be a non-empty list of strings")
            if "enum" in spec["items"] and not set(v) <= set(spec["items"]["enum"]):
                raise ToolError(f"{k} items must be among {spec['items']['enum']}")
        out[k] = v
    missing = [k for k in schema["required"] if k not in out]
    if missing:
        raise ToolError(f"missing required parameter(s): {', '.join(missing)}")
    return out


def _sig(product: str, tool: str, p: dict) -> str:
    body = {k: v for k, v in p.items() if k != "cursor"}
    return hashlib.sha256(json.dumps([product, tool, body], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _offset(cursor, sig: str) -> int:
    if cursor is None:
        return 0
    try:
        d = json.loads(base64.urlsafe_b64decode(str(cursor).encode()))
        ok = isinstance(d, dict) and d.get("s") == sig and isinstance(d.get("o"), int) and d["o"] >= 0
    except (ValueError, binascii.Error):
        ok = False
    if not ok:
        raise ToolError("cursor does not belong to this request (same product, tool and filters)")
    return d["o"]


def _page(rows: list[dict], offset: int, sig: str):
    """Rows [offset:] until the row cap or the response budget; a row over the budget alone is refused."""
    budget = RESPONSE_BUDGET_TOKENS * 4
    page: list[dict] = []
    size = 2
    i = offset
    while i < len(rows) and len(page) < ROW_CAP:
        n = len(json.dumps(rows[i], ensure_ascii=False, separators=(",", ":"))) + 1
        if size + n > budget:
            if not page:
                raise ToolError(f"a single row exceeds the {RESPONSE_BUDGET_TOKENS:,}-token response budget "
                                "and is not returned")
            break
        page.append(rows[i])
        size += n
        i += 1
    more = i < len(rows)
    cursor = base64.urlsafe_b64encode(json.dumps({"s": sig, "o": i}).encode()).decode() if more else None
    return page, more, cursor


def _path(rows: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for key in ("refs", "evidence"):
        for r in rows:
            for x in r.get(key) or []:
                seen.setdefault(x)
    return list(seen)[:PATH_CAP]


class _Ctx:
    """GRAPH.RO_QUERY only. Records one TraceOp per executed query with the parameters actually bound."""

    def __init__(self, tool: str, conn):
        self.tool, self.conn, self.ops = tool, conn, []

    def q(self, text: str, **params) -> list:
        t0 = time.perf_counter()
        rows = self.conn.ro_query(text, params or None).result_set
        self.note("GRAPH.RO_QUERY", text, params, len(rows), t0)
        return rows

    def note(self, command: str, query: str, params: dict, n: int, t0: float) -> None:
        self.ops.append(TraceOp(self.tool, command, query, dict(params), n, round((time.perf_counter() - t0) * 1000, 3)))


def _union(*lists) -> list:
    return sorted({x for lst in lists for x in (lst or [])})


# ---------------------------------------------------------------- name resolution inside the tools
def _word_in(needle: str, hay: str) -> bool:
    return bool(needle) and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", hay) is not None


def _area_key(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", fold(text).replace("&", " and ")).split())


def _resolve_companies(ctx: _Ctx, text: str | None) -> list[str] | None:
    """None when no filter was asked; else the companies the text names (a company or a Customer id)."""
    if text is None:
        return None
    f = fold(text)
    rows = ctx.q("MATCH (c:Customer) RETURN c.id, c.source_id, c.company")
    return sorted({company for cid, sid, company in rows
                   if company and f in {fold(company), fold(cid or ""), fold(sid or "")}})


def _resolve_components(ctx: _Ctx, text: str | None) -> list[str] | None:
    if text is None:
        return None
    key = _area_key(text)
    comps = [r[0] for r in ctx.q("MATCH (b:Bug) WHERE b.component IS NOT NULL RETURN DISTINCT b.component")]
    # The intake message sometimes names only part of a product area ("backend processing"): a stored area of two or
    # more words that the question's phrase contains is that area's partial name, and counts with the exact match.
    near = sorted(c for c in comps if _area_key(c) == key
                  or (len(_area_key(c).split()) >= 2 and _word_in(_area_key(c), key)))
    if near:
        return near
    return sorted(c for c in comps if _word_in(key, _area_key(c)))


def _resolve_people(ctx: _Ctx, text: str | None) -> list[str] | None:
    """Employee ids for a canonical id or an exact directory name (namesakes all)."""
    if text is None:
        return None
    t = text[7:] if text.startswith("Person:") else text
    f = fold(t)
    rows = ctx.q("MATCH (p:Person) RETURN p.eid, p.name")
    return sorted({eid for eid, name in rows if fold(eid or "") == f or (name and fold(name) == f)})


def _resolve_bugs(ctx: _Ctx, text: str | None) -> list[str] | None:
    if text is None:
        return None
    f = fold(text)
    rows = ctx.q("MATCH (b:Bug) RETURN b.id, b.title, b.aliases")
    exact = {bid for bid, title, aliases in rows if f in {fold(bid), fold(title or ""), *map(fold, aliases or [])}}
    if exact:
        return sorted(exact)
    return sorted(bid for bid, title, aliases in rows
                  if any(_word_in(fold(n), f) or _word_in(f, fold(n)) for n in [title or "", *(aliases or [])] if n))


_STOP = frozenset("a an the of for to in on and or with by new features feature added add about related improving "
                  "improve using integration integrating".split())


def _resolve_features(ctx: _Ctx, text: str | None) -> list[str] | None:
    """Feature ids for a canonical id, an exact title/topic/alias, a title/alias inside the text (or the text inside a
    title), else the features whose words cover at least 60 percent of the text's content words."""
    if text is None:
        return None
    f = fold(text)
    rows = ctx.q("MATCH (f:Feature) RETURN f.id, f.title, f.topic, f.aliases, f.details")
    names = {fid: [fold(n) for n in [title, topic, *(aliases or [])] if n] for fid, title, topic, aliases, _ in rows}
    exact = sorted(fid for fid, ns in names.items() if f == fold(fid) or f in ns)
    if exact:
        return exact
    contained = sorted(fid for fid, ns in names.items() if any(_word_in(n, f) or _word_in(f, n) for n in ns))
    if contained:
        return contained
    words = {w for w in re.findall(r"\w+", f) if w not in _STOP and len(w) > 2}
    if not words:
        return []
    best, scored = 0.0, {}
    for fid, title, topic, aliases, details in rows:
        have = set(re.findall(r"\w+", " ".join(names[fid] + [fold(d) for d in details or []])))
        scored[fid] = len(words & have) / len(words)
        best = max(best, scored[fid])
    return sorted(fid for fid, s in scored.items() if s >= 0.6 and s == best)


def _resolve_documents(ctx: _Ctx, text: str, scope: str) -> list[str]:
    f = fold(text)
    rows = ctx.q("MATCH (d:Document) RETURN d.id, d.source_id, d.type, d.release_scope")
    hit = [r for r in rows if f in {fold(r[2] or ""), fold(r[1] or ""), fold(r[0])}]
    if not hit:
        hit = [r for r in rows if r[2] and _word_in(fold(r[2]), f)]
    if scope == "previous":
        hit = [r for r in hit if r[3] == "previous"]
    elif scope == "current":
        hit = [r for r in hit if r[3] == "current"]  # unknown scope stays unknown
    return sorted(r[0] for r in hit)


# ---------------------------------------------------------------- 1. resolve_person
def _resolve_person(ctx: _Ctx, p: dict) -> list[dict]:
    f = fold(p["name"])
    rows: list[dict] = []
    if p.get("meeting"):
        m = p["meeting"][8:] if p["meeting"].startswith("Meeting:") else p["meeting"]
        for eid, name, role, indir, speaker, amb, n, src in ctx.q(
                "MATCH (p:Person)-[s:SPOKE_IN]->(m:Meeting) WHERE m.source_id = $meeting "
                "RETURN p.eid, p.name, p.role, p.in_directory, s.speaker_name, s.ambiguous, s.n_candidates, s.src",
                meeting=m):
            if f in {fold(speaker or ""), fold(name or "")}:
                rows.append({"answer": eid, "eid": eid, "name": name, "role": role, "in_directory": bool(indir),
                             "ambiguous": bool(amb), "n_candidates": n, "match": "meeting speaker",
                             "refs": [f"Person:{eid}", f"Meeting:{m}"], "evidence": _union(src)})
    else:
        hits = [r for r in ctx.q("MATCH (p:Person) RETURN p.eid, p.name, p.role, p.in_directory")
                if fold(r[0] or "") == f or (r[1] and fold(r[1]) == f)]
        for eid, name, role, indir in hits:
            rows.append({"answer": eid, "eid": eid, "name": name, "role": role, "in_directory": bool(indir),
                         "ambiguous": len(hits) > 1, "n_candidates": len(hits),
                         "match": "eid" if fold(eid) == f else "directory name", "refs": [f"Person:{eid}"],
                         "evidence": []})
    return sorted(rows, key=lambda r: r["eid"])


# ---------------------------------------------------------------- 2. resolve_entity
KINDS = {"customer": ("Customer", ("company", "source_id")), "document": ("Document", ("type", "source_id")),
         "bug": ("Bug", ("title", "aliases", "source_id")), "pr": ("PR", ("title", "link")),
         "feature": ("Feature", ("title", "aliases", "topic")), "competitor": ("Competitor", ("name", "aliases")),
         "product": ("Product", ("name", "aliases"))}
FALLBACK_KINDS = {lb for lb in FALLBACK_LABELS}


def _resolve_entity(ctx: _Ctx, p: dict) -> list[dict]:
    f = fold(p["text"])
    wanted = [KINDS[k] for k in (p.get("kinds") or KINDS)]
    rows: dict[str, dict] = {}
    present: set[str] = set()
    for label, fields in wanted:
        cols = ", ".join(f"n.{x} AS m{i}" for i, x in enumerate(fields))
        q = f"MATCH (n:{label}) RETURN n.id, {cards._name_cols('n')}, {cols}"  # noqa: SLF001
        for row in ctx.q(q):
            nid, names, vals = row[0], row[1:1 + len(cards.NAME_FIELDS)], row[1 + len(cards.NAME_FIELDS):]
            present.add(label)
            mentions = [nid] + [x for v in vals for x in (v if isinstance(v, list) else [v]) if isinstance(x, str)]
            if f in {fold(m) for m in mentions}:
                rows[nid] = {"answer": nid, "id": nid, "label": label, "match": "exact", "refs": [nid], "evidence": [],
                             "name": cards.display_name(label, cards._name_row(names), nid)}  # noqa: SLF001
    if not rows and len(f.split()) >= 2:
        scored: list[tuple[float, str, dict]] = []
        for label, _ in wanted:
            if label not in FALLBACK_KINDS or label not in present:
                continue
            q = (f"CALL db.idx.fulltext.queryNodes('{label}', $q) YIELD node, score "
                 f"RETURN node.id, score, {cards._name_cols('node')}")  # noqa: SLF001
            for nid, score, *names in ctx.q(q, q=_escape(normalise(p["text"]))):
                scored.append((float(score), nid, {
                    "answer": nid, "id": nid, "label": label, "match": "fulltext", "refs": [nid], "evidence": [],
                    "name": cards.display_name(label, cards._name_row(names), nid)}))  # noqa: SLF001
        scored.sort(key=lambda t: (-t[0], t[1]))
        if len(scored) > FALLBACK_TOP:
            cut = scored[FALLBACK_TOP - 1][0]
            scored = [t for t in scored if t[0] >= cut]  # ties at the cut stay
        rows = {nid: r for _, nid, r in scored}
    return sorted(rows.values(), key=lambda r: (r["label"], r["id"]))


# ---------------------------------------------------------------- 3. doc_people
_SLACK_RECORD = re.compile(r"^\d{8}-\d+-[0-9a-f]+$")  # Slack message ids; transcript segments are <meeting>#<line>


def _substantive(kind: str, src) -> bool:
    """A key-reviewer contribution: a suggestion, or a review question raised in Slack. A question asked in a meeting
    transcript alone does not qualify."""
    return kind == "suggestion" or (kind == "review_question" and bool(src) and all(_SLACK_RECORD.match(r) for r in src))


def _doc_people(ctx: _Ctx, p: dict) -> list[dict]:
    docs = _resolve_documents(ctx, p["document_type"], p["scope"])
    if not docs:
        return []
    role = p.get("role")
    authors = ctx.q("MATCH (p:Person)-[au:AUTHORED]->(d:Document) WHERE d.id IN $docs AND ($role IS NULL OR p.role = $role) "
                    "RETURN p.eid, p.name, p.role, d.id, au.src", docs=docs, role=role)
    reviews = ctx.q("MATCH (p:Person)-[s:SUGGESTED]->(d:Document) WHERE d.id IN $docs "
                    "AND s.kind IN ['suggestion', 'review_question'] AND ($role IS NULL OR p.role = $role) "
                    "RETURN p.eid, p.name, p.role, d.id, s.kind, s.section, s.text, s.ambiguous, s.src",
                    docs=docs, role=role)
    reviews = [r for r in reviews if _substantive(r[4], r[8])]
    if p["return"] == "suggestions":
        out = [{"answer": text, "section": section, "kind": kind, "speaker": eid, "speaker_name": name,
                "speaker_role": prole, "anonymous": False, "ambiguous": bool(amb), "document": did,
                "refs": [f"Person:{eid}", did], "evidence": _union(src)}
               for eid, name, prole, did, kind, section, text, amb, src in reviews if text]
        if role is None:  # anonymous feedback never satisfies a role filter
            for did, sid, items in ctx.q("MATCH (d:Document) WHERE d.id IN $docs AND d.feedback_items IS NOT NULL "
                                         "RETURN d.id, d.source_id, d.feedback_items", docs=docs):
                out += [{"answer": item, "section": None, "kind": "feedback", "speaker": None, "speaker_name": None,
                         "speaker_role": None, "anonymous": True, "ambiguous": False, "document": did,
                         "refs": [did], "evidence": [sid]} for item in items]
        return sorted(out, key=lambda r: (r["document"], r["section"] or "", r["answer"], r["speaker"] or ""))
    people: dict[str, dict] = {}

    def add(eid, name, prole, did, kind, amb, src):
        r = people.setdefault(eid, {"answer": eid, "name": name, "role": prole, "roles": [], "ambiguous": False,
                                    "documents": set(), "refs": {f"Person:{eid}"}, "evidence": set()})
        if kind not in r["roles"]:
            r["roles"].append(kind)
        r["ambiguous"] = r["ambiguous"] or bool(amb)
        r["documents"].add(did)
        r["refs"].add(did)
        r["evidence"].update(src or [])

    for eid, name, prole, did, src in authors:
        add(eid, name, prole, did, "author", False, src)
    for eid, name, prole, did, kind, section, text, amb, src in reviews:
        add(eid, name, prole, did, "reviewer", amb, src)
    return [{**r, "roles": sorted(r["roles"]), "documents": sorted(r["documents"]), "refs": sorted(r["refs"]),
             "evidence": sorted(r["evidence"])} for _, r in sorted(people.items())]


# ---------------------------------------------------------------- 4. bug_board
_STATUS = {"unresolved": "b.unresolved = true", "resolved": "b.resolved = true", "active": "b.active = true",
           "reopened": "b.reopened = true", "no_fix_needed": "b.no_fix_needed = true", "any": "true"}
_AT_STATUS = {"active": "active", "resolved": "resolved", "no_fix_needed": "dismissed"}
_FILTERS = ("($components IS NULL OR b.component IN $components) "
            "AND ($companies IS NULL OR any(x IN b.companies WHERE x IN $companies))")
_SEL = f"MATCH (b:Bug) WHERE {{status}} AND {_FILTERS}"
_SEL_AT = ("MATCH (b:Bug)-[:HAS_EVENT]->(ev:BugEvent) WITH b, collect({type: ev.type, ts: ev.ts}) AS evs "
           f"WHERE herb_v1.statusAt(evs, $as_of) @AT AND {_FILTERS}")
# "now reopened": the action (resolution, fix PR, dismissal) is followed by a REOPENED event, by source timestamp
_LATER_REOPEN = " MATCH (b)-[:HAS_EVENT]->(ro:BugEvent) WHERE ro.type = 'REOPENED' AND ro.ts > {since}.ts"
_FIX_PR_RESOLVED = " MATCH (b)-[hs:HAS_EVENT]->(rs:BugEvent) WHERE rs.type = 'RESOLVED' AND pr.link IN rs.fix_pr_links"


def _bug_selector(p: dict) -> tuple[str, dict]:
    if p.get("as_of"):
        if p["status"] != "any" and p["status"] not in _AT_STATUS:
            raise ToolError("as_of supports status active, resolved, no_fix_needed or any")
        at = "= $at_status" if p["status"] != "any" else "IS NOT NULL"
        extra = {"as_of": p["as_of"]} | ({"at_status": _AT_STATUS[p["status"]]} if p["status"] != "any" else {})
        return _SEL_AT.replace("@AT", at), extra
    return _SEL.format(status=_STATUS[p["status"]]), {}


def _bug_board(ctx: _Ctx, p: dict) -> list[dict]:
    actor, ret = p["actor"], p.get("return")
    if ret == "companies" and actor != "company":
        raise ToolError("return=companies needs actor=company")
    if ret == "people" and actor == "company":
        raise ToolError("return=people needs an actor other than company")
    ret = ret or ("companies" if actor == "company" else "people")
    sel, extra = _bug_selector(p)
    comps = _resolve_components(ctx, p.get("component"))
    cos = _resolve_companies(ctx, p.get("customer"))
    base = {"components": comps, "companies": cos, **extra}
    if ret == "bugs":
        return _bug_rows(ctx, sel, base, p["status"])
    join = p["status"] == "reopened"
    if actor == "company":
        found = ctx.q(sel + " MATCH (b)-[rb:REPORTED_BY]->(c:Customer) WHERE c.company IS NOT NULL "
                      "AND ($companies IS NULL OR c.company IN $companies) RETURN c.company, c.id, b.id, rb.src", **base)
        groups: dict[str, dict] = {}
        for company, cid, bid, src in found:
            g = groups.setdefault(company, {"items": set(), "bugs": set(), "refs": set(), "evidence": set()})
            g["items"].add(bid)
            g["bugs"].add(bid)
            g["refs"].update([cid, bid])
            g["evidence"].update(src or [])
        _add_support(ctx, groups, p["status"])
        return _ranked(groups, p["rank"], lambda k, g: {"name": k})
    # every branch returns: eid, name, role, bug, pr link, the acting edge's src, the event or FIXES src, extra refs
    if actor == "assignee":
        found = ctx.q(sel + " MATCH (b)-[he:HAS_EVENT]->(ev:BugEvent)-[at:ASSIGNED_TO]->(p:Person) "
                      "WHERE ev.type = 'ASSIGNED' AND p.eid IN b.current_assignees "
                      "RETURN p.eid, p.name, p.role, b.id, null, at.src, he.src, [ev.id]", **base)
    elif actor in ("resolver", "dismisser"):
        text = (sel + " MATCH (b)-[he:HAS_EVENT]->(ev:BugEvent)-[ab:ACTED_BY]->(p:Person) WHERE ev.type = $etype"
                + (_LATER_REOPEN.format(since="ev") if join else "")
                + " RETURN p.eid, p.name, p.role, b.id, null, ab.src, he.src, [ev.id]")
        found = ctx.q(text, etype="RESOLVED" if actor == "resolver" else "DISMISSED", **base)
    else:  # fix_pr_author: counts distinct PR links
        text = (sel + " MATCH (pr:PR)-[fx:FIXES]->(b) MATCH (p:Person)-[op:OPENED]->(pr)"
                + (_FIX_PR_RESOLVED + _LATER_REOPEN.format(since="rs")
                   + " RETURN p.eid, p.name, p.role, b.id, pr.link, op.src, fx.src + hs.src, [rs.id]" if join else
                   " RETURN p.eid, p.name, p.role, b.id, pr.link, op.src, fx.src, []"))
        found = ctx.q(text, **base)
    groups = {}
    for eid, name, role, bid, link, src, src2, more in found:
        g = groups.setdefault(eid, {"items": set(), "bugs": set(), "prs": set(), "refs": {f"Person:{eid}"},
                                    "evidence": set(), "name": name, "role": role})
        g["items"].add(link if actor == "fix_pr_author" else bid)
        g["bugs"].add(bid)
        if link:
            g["prs"].add(link)
            g["refs"].add(f"PR:{link}")
        g["refs"].update([bid, *(x for x in more or [] if x and not x.startswith("http"))])
        g["evidence"].update([*(src or []), *(src2 or [])])
    _add_support(ctx, groups, p["status"])
    return _ranked(groups, p["rank"], lambda k, g: {"name": g["name"], "role": g["role"]} | (
        {"prs": sorted(g["prs"])} if actor == "fix_pr_author" else {}))


_EVENT_FOR = {"resolved": ("RESOLVED",), "no_fix_needed": ("DISMISSED",), "reopened": ("REOPENED",)}


def _bug_support(ctx: _Ctx, ids: list[str], status: str) -> dict[str, dict]:
    """The records that make each bug satisfy `status`: its resolving, dismissing or reopening events, the FIXES edges of
    its fix PRs, or its last event for active. Absence (unresolved) has none."""
    out: dict[str, dict] = defaultdict(lambda: {"refs": set(), "evidence": set()})
    if not ids or status in ("unresolved", "any"):
        return out
    events: dict[str, list] = defaultdict(list)
    for bid, eid, typ, ts, src in ctx.q("MATCH (b:Bug)-[he:HAS_EVENT]->(ev:BugEvent) WHERE b.id IN $ids "
                                        "RETURN b.id, ev.id, ev.type, ev.ts, he.src", ids=ids):
        events[bid].append((ts or "", eid, typ, src or []))
    for bid, evs in events.items():
        keep = [max(evs)] if status == "active" else [e for e in evs if e[2] in _EVENT_FOR[status]]
        for _, eid, _, src in keep:
            out[bid]["refs"].add(eid)
            out[bid]["evidence"].update(src)
    if status == "resolved":
        for bid, link, src in ctx.q("MATCH (pr:PR)-[fx:FIXES]->(b:Bug) WHERE b.id IN $ids RETURN b.id, pr.link, fx.src",
                                    ids=ids):
            out[bid]["refs"].add(f"PR:{link}")
            out[bid]["evidence"].update(src or [])
    return out


def _add_support(ctx: _Ctx, groups: dict, status: str) -> None:
    sup = _bug_support(ctx, sorted({b for g in groups.values() for b in g["bugs"]}), status)
    for g in groups.values():
        for b in g["bugs"]:
            g["refs"].update(sup[b]["refs"])
            g["evidence"].update(sup[b]["evidence"])


def _ranked(groups: dict, rank: str, extra: Callable) -> list[dict]:
    rows = [{"answer": k, **extra(k, g), "count": len(g["items"]), "bugs": sorted(g["bugs"]),
             "refs": sorted(g["refs"]), "evidence": sorted(g["evidence"])} for k, g in groups.items()]
    rows.sort(key=lambda r: (-r["count"], r["answer"]))
    if rank == "max" and rows:
        rows = [r for r in rows if r["count"] == rows[0]["count"]]  # every tie at the top
    return rows


def _bug_rows(ctx: _Ctx, sel: str, base: dict, status: str) -> list[dict]:
    found = ctx.q(sel + " RETURN b.id, b.title, b.impact_text, b.component, b.companies, b.unresolved, b.resolved, "
                  "b.active, b.reopened, b.no_fix_needed", **base)
    ids = [r[0] for r in found]
    ev: dict[str, dict] = defaultdict(lambda: {"refs": set(), "evidence": set()})
    if ids:
        for bid, cid, src in ctx.q("MATCH (b:Bug)-[rb:REPORTED_BY]->(c:Customer) WHERE b.id IN $ids "
                                   "RETURN b.id, c.id, rb.src", ids=ids):
            ev[bid]["refs"].add(cid)
            ev[bid]["evidence"].update(src or [])
    for bid, sup in _bug_support(ctx, ids, status).items():
        ev[bid]["refs"].update(sup["refs"])
        ev[bid]["evidence"].update(sup["evidence"])
    rows = []
    for bid, title, impact, comp, companies, unres, res, act, reo, nofix in sorted(found):
        rows.append({"answer": title, "id": bid, "title": title, "impact_text": impact, "component": comp,
                     "companies": companies or [],
                     "flags": {"unresolved": bool(unres), "resolved": bool(res), "active": bool(act),
                               "reopened": bool(reo), "no_fix_needed": bool(nofix)},
                     "refs": [bid, *sorted(ev[bid]["refs"])], "evidence": sorted(ev[bid]["evidence"])})
    return rows


# ---------------------------------------------------------------- 5. fix_trail
def _fix_trail(ctx: _Ctx, p: dict) -> list[dict]:
    found = ctx.q(
        "MATCH (pr:PR)-[fx:FIXES]->(b:Bug)-[rb:REPORTED_BY]->(c:Customer) "
        "WHERE ($companies IS NULL OR c.company IN $companies) AND ($bugs IS NULL OR b.id IN $bugs) "
        "MATCH (p:Person)-[op:OPENED]->(pr) WHERE ($engineers IS NULL OR p.eid IN $engineers) "
        "RETURN pr.link, pr.title, b.id, b.title, c.id, c.company, p.eid, p.name, fx.src, op.src, rb.src",
        companies=_resolve_companies(ctx, p.get("customer")), bugs=_resolve_bugs(ctx, p.get("bug")),
        engineers=_resolve_people(ctx, p.get("engineer")))
    groups: dict[tuple, dict] = {}
    for link, ptitle, bid, btitle, cid, company, eid, name, fsrc, osrc, rsrc in found:
        g = groups.setdefault((link, bid), {"answer": link, "pr": f"PR:{link}", "pr_title": ptitle, "bug": bid,
                                            "bug_title": btitle, "author": eid, "author_name": name,
                                            "customers": set(), "companies": set(), "evidence": set()})
        g["customers"].add(cid)
        g["companies"].add(company)
        g["evidence"].update([*(fsrc or []), *(osrc or []), *(rsrc or [])])
    rows = []
    for (link, bid), g in sorted(groups.items()):
        rows.append({**g, "company": ", ".join(sorted(g["companies"])), "companies": sorted(g["companies"]),
                     "refs": [g["pr"], bid, f"Person:{g['author']}", *sorted(g["customers"])],
                     "customers": sorted(g["customers"]), "evidence": sorted(g["evidence"])})
    return rows


# ---------------------------------------------------------------- 6. pr_board
def _pr_board(ctx: _Ctx, p: dict) -> list[dict]:
    match = ["MATCH (rv:PR)-[r:REVERTS]->(pr:PR)" if p["state"] == "reverted" else "MATCH (pr:PR)"]
    where = ["pr.repo_scope = 'product'"]  # product-repo PRs only; open-source look-alikes are never returned
    if p["state"] == "reverted":
        where.append("($revert_reason IS NULL OR r.reason = $revert_reason)")
    if p["approval"] != "any":
        where.append("pr.approval = $approval")
    where += {"need_merge": ["pr.pending_action = 'need_merge'"], "need_close": ["pr.pending_action = 'need_close'"],
              "merged": ["pr.merged = true"]}.get(p["state"], [])
    feats = _resolve_features(ctx, p.get("feature"))
    if feats is not None or p["kind"] == "feature_development":
        match.append("MATCH (pr)-[:ABOUT]->(f:Feature)")
        if feats is not None:
            where.append("f.id IN $features")
    if p["kind"] == "feature_development":
        where.append("NOT (pr)-[:FIXES]->(:Bug)")  # FIXES takes precedence over ABOUT
    elif p["kind"] == "bug_fix":
        where.append("(pr)-[:FIXES]->(:Bug)")
    found = ctx.q(" ".join(match) + " WHERE " + " AND ".join(where)
                  + " RETURN DISTINCT pr.link, pr.title, pr.approval, pr.merged, pr.state, pr.pending_action",
                  features=feats, approval=p["approval"], revert_reason=p.get("revert_reason"))
    links = sorted(r[0] for r in found)
    info = {r[0]: r for r in found}
    authors: dict[str, list] = defaultdict(list)
    fts: dict[str, list] = defaultdict(list)
    revs: dict[str, list] = defaultdict(list)
    ev: dict[str, set] = defaultdict(set)
    if links:
        for link, eid, name, role, src in ctx.q("MATCH (p:Person)-[op:OPENED]->(pr:PR) WHERE pr.link IN $links "
                                                "RETURN pr.link, p.eid, p.name, p.role, op.src", links=links):
            authors[link].append({"eid": eid, "name": name, "role": role})
            ev[link].update(src or [])
        for link, fid, ftitle, src in ctx.q("MATCH (pr:PR)-[a:ABOUT]->(f:Feature) WHERE pr.link IN $links "
                                            "RETURN pr.link, f.id, f.title, a.src", links=links):
            fts[link].append({"id": fid, "title": ftitle})
            ev[link].update(src or [])
        for link, by_link, reason, applied, src in ctx.q("MATCH (rv:PR)-[r:REVERTS]->(pr:PR) WHERE pr.link IN $links "
                                                         "RETURN pr.link, rv.link, r.reason, r.applied, r.src", links=links):
            revs[link].append({"by": by_link, "reason": reason, "applied": bool(applied)})
            ev[link].update(src or [])
    if p["return"] == "people":
        groups: dict[str, dict] = {}
        for link in links:
            for a in authors[link]:
                g = groups.setdefault(a["eid"], {"items": set(), "bugs": set(), "prs": set(), "name": a["name"],
                                                 "role": a["role"], "refs": {f"Person:{a['eid']}"}, "evidence": set()})
                g["items"].add(link)  # DISTINCT PR links: feature and review fan-out never inflates a count
                g["prs"].add(link)
                g["refs"].add(f"PR:{link}")
                g["evidence"].update(ev[link])
        rows = _ranked(groups, p["rank"], lambda k, g: {"name": g["name"], "role": g["role"], "prs": sorted(g["prs"])})
        return [{k: v for k, v in r.items() if k != "bugs"} for r in rows]
    rows = []
    for link in links:
        _, title, approval, merged, state, pending = info[link]
        rows.append({"answer": link, "title": title, "approval": approval, "merged": merged, "state": state,
                     "pending_action": pending, "authors": authors[link], "features": sorted(fts[link], key=lambda x: x["id"]),
                     "reverts": sorted(revs[link], key=lambda x: x["by"]),
                     "refs": [f"PR:{link}", *(f"Person:{a['eid']}" for a in authors[link]),
                              *(f["id"] for f in sorted(fts[link], key=lambda x: x["id"]))],
                     "evidence": sorted(ev[link])})
    return rows


# ---------------------------------------------------------------- 7. feature_fate
def _feature_fate(ctx: _Ctx, p: dict) -> list[dict]:
    feats = _resolve_features(ctx, p.get("topic"))
    # the five fates are the semantic layer's derived Feature properties (one definition); events only give history
    feature_rows = ctx.q("MATCH (f:Feature) RETURN f.id, f.title, f.aliases, f.details, f.implemented, f.reverted, "
                         "f.held, f.delayed_then_added, f.discussed_not_implemented")
    events: dict[str, list] = defaultdict(list)
    for fid, eid, kind, date, reason, rtext, rec in ctx.q(
            "MATCH (f:Feature)-[:HAS_EVENT]->(ev:FeatureEvent) "
            "RETURN f.id, ev.id, ev.kind, ev.date, ev.reason, ev.reason_text, ev.record_id"):
        events[fid].append({"id": eid, "kind": kind, "date": date, "reason": reason, "reason_text": rtext, "record": rec})
    impl: dict[str, list] = defaultdict(list)  # feature -> [(link, [(revert by, reason, applied, src)], src)]
    for fid, link, isrc, rvby, reason, applied, rsrc in ctx.q(
            "MATCH (f:Feature)-[ib:IMPLEMENTED_BY]->(pr:PR) OPTIONAL MATCH (rv:PR)-[r:REVERTS]->(pr) "
            "RETURN f.id, pr.link, ib.src, rv.link, r.reason, r.applied, r.src"):
        impl[fid].append((link, isrc, rvby, reason, applied, rsrc))
    about: dict[str, dict] = defaultdict(dict)
    for fid, link, title, src in ctx.q("MATCH (pr:PR)-[a:ABOUT]->(f:Feature) WHERE pr.repo_scope = 'product' "
                                       "RETURN f.id, pr.link, pr.title, a.src"):
        about[fid][link] = (title, src)
    chosen: list[dict] = []
    for fid, title, aliases, details, f_impl, f_rev, f_held, f_delayed, f_dni in sorted(feature_rows):
        if feats is not None and fid not in feats:
            continue
        evs = sorted(events[fid], key=lambda e: (e["date"] or "", e["id"]))
        fates = {"any"} | {n for n, v in (("implemented", f_impl), ("reverted", f_rev), ("held", f_held),
                                          ("delayed_then_added", f_delayed), ("unimplemented", f_dni)) if v is True}
        if p["fate"] not in fates:
            continue
        rev_reasons = [e["reason"] for e in evs if e["kind"] == "reverted" and e["reason"]]
        hold_reasons = [e["reason"] for e in evs if e["kind"] == "held" and e["reason"]]
        pool = {"reverted": rev_reasons, "held": hold_reasons, "delayed_then_added": hold_reasons}.get(
            p["fate"], rev_reasons + hold_reasons + [e["reason"] for e in evs if e["reason"]])
        if p.get("reason") and p["reason"] not in pool:
            continue
        reverts = [(rv, rs) for _, _, rv, _, ap, rs in impl[fid] if rv and ap]  # applied reverts of implementing PRs
        chosen.append({"id": fid, "title": title, "aliases": aliases or [], "details": details or [], "events": evs,
                       "fates": sorted(fates - {"any"}), "reasons": sorted(set(pool)),
                       "prs": sorted({*(l for l, *_ in impl[fid]), *about[fid]}),
                       "revert_prs": sorted({rv for rv, _ in reverts}),
                       "evidence": sorted({e["record"] for e in evs if e["record"]}
                                          | {s for _, isrc, *_r in impl[fid] for s in isrc or []}
                                          | {s for _, rs in reverts for s in rs or []})})
    if p["return"] == "features":
        return [{"answer": c["title"], "id": c["id"], "title": c["title"], "details": c["details"],
                 "aliases": c["aliases"],
                 "events": [{k: e[k] for k in ("kind", "date", "reason", "reason_text", "record")} for e in c["events"]],
                 "fates": c["fates"], "reasons": c["reasons"], "prs": c["prs"],
                 "refs": [c["id"], *(f"PR:{x}" for x in [*c["prs"], *c["revert_prs"]])],
                 "evidence": c["evidence"]} for c in chosen]
    ids = [c["id"] for c in chosen]
    if p["return"] == "prs":
        out: dict[str, dict] = {}
        for c in chosen:
            for link in c["prs"]:
                r = out.setdefault(link, {"answer": link, "title": about[c["id"]].get(link, (None,))[0],
                                          "features": [], "via": set(), "refs": {f"PR:{link}"}, "evidence": set()})
                r["features"].append(c["id"])
                r["refs"].add(c["id"])
                if link in about[c["id"]]:
                    r["via"].add("about")
                    r["evidence"].update(about[c["id"]][link][1] or [])
                for l2, isrc, *_ in impl[c["id"]]:
                    if l2 == link:
                        r["via"].add("implemented_by")
                        r["evidence"].update(isrc or [])
        return [{**r, "via": sorted(r["via"]), "refs": sorted(r["refs"]), "evidence": sorted(r["evidence"])}
                for _, r in sorted(out.items())]
    people: dict[str, dict] = {}
    if ids:
        for fid, eid, name, role, kind, amb, src in ctx.q(
                "MATCH (p:Person)-[c:CONTRIBUTED]->(f:Feature) WHERE f.id IN $ids "
                "RETURN f.id, p.eid, p.name, p.role, c.kind, c.ambiguous, c.src", ids=ids):
            r = people.setdefault(eid, {"answer": eid, "name": name, "role": role, "kinds": set(), "ambiguous": False,
                                        "features": set(), "refs": {f"Person:{eid}"}, "evidence": set()})
            r["kinds"].add(kind)
            r["ambiguous"] = r["ambiguous"] or bool(amb)
            r["features"].add(fid)
            r["refs"].add(fid)
            r["evidence"].update(src or [])
    return [{**r, "kinds": sorted(r["kinds"]), "features": sorted(r["features"]), "refs": sorted(r["refs"]),
             "evidence": sorted(r["evidence"])} for _, r in sorted(people.items())]


# ---------------------------------------------------------------- 8. who_said
_OBS_KINDS = {"competitor": {"feature": ["competitor_feature"], "weakness": ["competitor_weakness"],
                             "any": ["competitor_feature", "competitor_weakness"]},
              "previous_release": {"issue": ["customer_issue"], "strength": ["customer_strength"],
                                   "qa_issue": ["qa_issue"],
                                   "any": ["customer_issue", "customer_strength", "qa_issue"]}}
_OBS_KINDS["product"] = _OBS_KINDS["previous_release"]
_OBS_MATCH = {
    "competitor": "MATCH (o:Observation)-[:ABOUT]->(:Competitor) WHERE o.kind IN $kinds",
    "previous_release": "MATCH (o:Observation) WHERE o.kind IN $kinds AND o.release_scope = 'previous'",
    "product": ("MATCH (o:Observation)-[:ABOUT]->(:Product) WHERE o.kind IN $kinds "
                "AND coalesce(o.release_scope, 'unknown') <> 'previous'"),
}
_DEMO_MATCH = {
    "competitor": "MATCH (u:Url)-[:ABOUT]->(:Competitor) WHERE u.kind = 'demo'",
    "previous_release": "MATCH (u:Url)-[:ABOUT]->(:Product) WHERE u.kind = 'demo' AND u.release_scope = 'previous'",
    "product": ("MATCH (u:Url)-[:ABOUT]->(:Product) WHERE u.kind = 'demo' "
                "AND coalesce(u.release_scope, 'unknown') <> 'previous'"),
}


def _who_said(ctx: _Ctx, p: dict) -> list[dict]:
    subject, kind, ret, role = p["subject"], p["kind"], p["return"], p.get("role")
    if kind == "worked_on":
        if subject != "previous_release" or ret != "people":
            raise ToolError("kind=worked_on is for subject=previous_release and return=people")
        return _worked_on(ctx, role)
    if kind == "demo":
        if ret not in ("people", "urls"):
            raise ToolError("kind=demo returns people or urls")
        return _demos(ctx, subject, ret, role)
    kinds = _OBS_KINDS[subject].get(kind)
    if kinds is None:
        raise ToolError(f"kind {kind!r} does not apply to subject {subject!r}; "
                        f"use one of {sorted(_OBS_KINDS[subject])} or demo")
    if ret == "urls":
        raise ToolError("return=urls needs kind=demo")
    obs = ctx.q(_OBS_MATCH[subject] + " RETURN o.id, o.kind, o.text, o.release_scope, o.record_id", kinds=kinds)
    ids = [r[0] for r in obs]
    stated: dict[str, list] = defaultdict(list)
    comps: dict[str, list] = defaultdict(list)
    qa: dict[str, list] = defaultdict(list)
    subj: dict[str, str] = {}
    ev: dict[str, set] = defaultdict(set)
    if ids:
        for oid, eid, name, prole, src in ctx.q("MATCH (p:Person)-[s:STATED]->(o:Observation) WHERE o.id IN $ids "
                                                "RETURN o.id, p.eid, p.name, p.role, s.src", ids=ids):
            stated[oid].append((eid, name, prole))
            ev[oid].update(src or [])
        for oid, cid, company, src in ctx.q("MATCH (o:Observation)-[rb:REPORTED_BY]->(c:Customer) WHERE o.id IN $ids "
                                            "RETURN o.id, c.id, c.company, rb.src", ids=ids):
            comps[oid].append((cid, company))
            ev[oid].update(src or [])
        for oid, eid, name, prole, src in ctx.q("MATCH (o:Observation)-[rb:REPORTED_BY]->(p:Person) WHERE o.id IN $ids "
                                                "RETURN o.id, p.eid, p.name, p.role, rb.src", ids=ids):
            qa[oid].append((eid, name, prole))
            ev[oid].update(src or [])
        for oid, name in ctx.q("MATCH (o:Observation)-[:ABOUT]->(c:Competitor) WHERE o.id IN $ids RETURN o.id, c.name",
                               ids=ids):
            subj[oid] = name
    info = {r[0]: r for r in obs}
    if ret == "claims":
        rows = [{"answer": info[o][2], "kind": info[o][1], "subject": subj.get(o, "product"),
                 "release_scope": info[o][3], "stated_by": sorted({e for e, *_ in stated[o]}),
                 "reporters": sorted({e for e, *_ in qa[o]}), "companies": sorted({c for _, c in comps[o] if c}),
                 "refs": [o, *(f"Person:{e}" for e, *_ in stated[o] + qa[o]), *(c for c, _ in comps[o])],
                 "evidence": sorted(ev[o] | ({info[o][4]} if info[o][4] else set()))} for o in ids]
        return sorted(rows, key=lambda r: (r["subject"], r["refs"][0]))
    if ret == "companies":
        out: dict[str, dict] = {}
        for o in ids:
            for cid, company in comps[o]:
                if company:
                    r = out.setdefault(company, {"answer": company, "name": company, "claims": [], "refs": set(),
                                                 "evidence": set()})
                    r["claims"].append(info[o][2])
                    r["refs"].update([cid, o])
                    r["evidence"].update(ev[o])
        return [{**r, "refs": sorted(r["refs"]), "evidence": sorted(r["evidence"])} for _, r in sorted(out.items())]
    people: dict[str, dict] = {}
    for o in ids:
        for eid, name, prole in stated[o]:
            if role and prole != role:
                continue
            r = people.setdefault(eid, {"answer": eid, "name": name, "role": prole, "claims": [], "refs": {f"Person:{eid}"},
                                        "evidence": set()})
            r["claims"].append(info[o][2])
            r["refs"].add(o)
            r["evidence"].update(ev[o])
    return [{**r, "refs": sorted(r["refs"]), "evidence": sorted(r["evidence"])} for _, r in sorted(people.items())]


def _demos(ctx: _Ctx, subject: str, ret: str, role: str | None) -> list[dict]:
    urls = sorted(r[0] for r in ctx.q(_DEMO_MATCH[subject] + " RETURN u.link"))
    if not urls:
        return []
    sharers: dict[str, list] = defaultdict(list)
    for link, eid, name, prole, src in ctx.q("MATCH (p:Person)-[sh:SHARED]->(u:Url) WHERE u.link IN $links "
                                             "RETURN u.link, p.eid, p.name, p.role, sh.src", links=urls):
        if role is None or prole == role:
            sharers[link].append((eid, name, prole, src or []))
    if ret == "urls":
        return [{"answer": link, "sharers": sorted({e for e, *_ in sharers[link]}),
                 "refs": [f"Url:{link}", *(f"Person:{e}" for e, *_ in sorted(sharers[link]))],
                 "evidence": _union(*(s for *_, s in sharers[link]))} for link in urls if sharers[link] or role is None]
    people: dict[str, dict] = {}
    for link in urls:
        for eid, name, prole, src in sharers[link]:
            r = people.setdefault(eid, {"answer": eid, "name": name, "role": prole, "urls": [],
                                        "refs": {f"Person:{eid}"}, "evidence": set()})
            r["urls"].append(link)
            r["refs"].add(f"Url:{link}")
            r["evidence"].update(src)
    return [{**r, "refs": sorted(r["refs"]), "evidence": sorted(r["evidence"])} for _, r in sorted(people.items())]


def _worked_on(ctx: _Ctx, role: str | None) -> list[dict]:
    """The previous release's team: authors of documents scoped previous (R2 evidence) and explicitly named QA reporters
    (R9), by canonical directory role. Never the present-day reviewers."""
    people: dict[str, dict] = {}

    def add(eid, name, prole, via, ref, src):
        r = people.setdefault(eid, {"answer": eid, "name": name, "role": prole, "via": set(), "refs": {f"Person:{eid}"},
                                    "evidence": set()})
        r["via"].add(via)
        r["refs"].add(ref)
        r["evidence"].update(src or [])

    for eid, name, prole, did, src in ctx.q(
            "MATCH (p:Person)-[au:AUTHORED]->(d:Document) WHERE d.release_scope = 'previous' "
            "AND ($role IS NULL OR p.role = $role) RETURN p.eid, p.name, p.role, d.id, au.src", role=role):
        add(eid, name, prole, "author", did, src)
    for eid, name, prole, oid, src in ctx.q(
            "MATCH (o:Observation)-[rb:REPORTED_BY]->(p:Person) WHERE o.kind = 'qa_issue' "
            "AND o.release_scope = 'previous' AND ($role IS NULL OR p.role = $role) "
            "RETURN p.eid, p.name, p.role, o.id, rb.src", role=role):
        add(eid, name, prole, "qa_reporter", oid, src)
    return [{**r, "via": sorted(r["via"]), "refs": sorted(r["refs"]), "evidence": sorted(r["evidence"])}
            for _, r in sorted(people.items())]


# ---------------------------------------------------------------- 9. bridges
def betweenness(nodes, edges) -> dict[str, float]:
    """Brandes' betweenness on the directed projection, each ordered pair counted once (what 6.0.2 computes)."""
    adj: dict[str, list] = {n: [] for n in nodes}
    for a, b in sorted(set(edges)):
        if a in adj and b in adj:
            adj[a].append(b)
    cb = dict.fromkeys(adj, 0.0)
    for s in adj:
        stack, pred = [], {n: [] for n in adj}
        sigma, dist = dict.fromkeys(adj, 0), dict.fromkeys(adj, -1)
        sigma[s], dist[s] = 1, 0
        queue = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = dict.fromkeys(adj, 0.0)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != s:
                cb[w] += delta[w]
    return cb


_NATIVE = ("CALL algo.betweenness({nodeLabels: ['Person'], relationshipTypes: ['COLLABORATED'], "
           f"samplingSize: {BETWEENNESS_SAMPLE}, samplingSeed: {BETWEENNESS_SEED}}}) YIELD node, score "
           "RETURN node.eid, node.name, node.role, score")


def _bridges(ctx: _Ctx, p: dict) -> list[dict]:
    if str(getattr(ctx.conn, "server_version", "")).startswith("6"):
        engine = "native"
        found = ctx.q(_NATIVE)
        scored = [(eid, name, role, float(score)) for eid, name, role, score in found if eid]
    else:  # 4.22.1: the native directed scores are wrong there (Task 1b); same projection, computed here
        engine = "python"
        people = ctx.q("MATCH (p:Person) RETURN p.eid, p.name, p.role")
        edges = ctx.q("MATCH (a:Person)-[:COLLABORATED]->(b:Person) RETURN a.eid, b.eid")
        t0 = time.perf_counter()
        score = betweenness([e for e, *_ in people], [tuple(e) for e in edges])
        ctx.note("python:betweenness", "", {"nodes": len(people), "edges": len(edges), "directed": True}, len(people), t0)
        scored = [(eid, name, role, score[eid]) for eid, name, role in people]
    scored.sort(key=lambda t: (-t[3], t[0]))
    return [{"answer": eid, "name": name, "role": role, "score": round(s, 6), "engine": engine,
             "refs": [f"Person:{eid}"], "evidence": []} for eid, name, role, s in scored[:p["top_k"]]]


# ---------------------------------------------------------------- 10. if_leaves (Task 12)
def _if_leaves(conn, product: str, **params) -> ToolResult:
    try:
        from inherit import whatif  # lazy: Task 12
    except ImportError as e:
        raise NotImplementedError("if_leaves arrives with Task 12 (inherit.whatif)") from e
    return whatif.if_leaves(conn, product, **params)


# ---------------------------------------------------------------- the catalogue
def _tool(name: str, description: str, props: dict, impl: Callable, required: tuple = (), kind: str = "readonly") -> Tool:
    return Tool(name, description, _schema(props, required), kind, impl)


_NAME_FORMS = "free text as written in the question, or a canonical id from an earlier row"

TOOLS: dict[str, Tool] = {t.name: t for t in (
    _tool("resolve_person",
          "Map a person's name to employee ids with candidates and evidence. Namesakes are all returned with "
          "ambiguous=true; never pick one. With a meeting, uses the transcript-speaker resolution for that meeting. "
          "Example: who is 'Sam Kim' in meeting product_dev_X_1? -> resolve_person(name='Sam Kim', meeting='product_dev_X_1').",
          {"name": _s("A person's name as written, or an employee id."),
           "meeting": _s("Optional meeting id; restricts to the speakers resolved for that meeting.")},
          _resolve_person, ("name",)),
    _tool("resolve_entity",
          "Resolve a document, customer company, PR, bug, feature, competitor or product name to node ids: exact "
          "match on names, titles, aliases, links and keys, else a full-text fallback for multi-word text. "
          "Example: which node is 'API rate limiting'? -> resolve_entity(text='API rate limiting', kinds=['bug']).",
          {"text": _s("The name, title, alias, link or key to resolve."),
           "kinds": {"type": "array", "items": {"type": "string", "enum": list(KINDS)},
                     "description": "Optional kinds to search; default all."}},
          _resolve_entity, ("text",)),
    _tool("doc_people",
          "Authors and key reviewers of a document type (reviewers = people with an attributed substantive review "
          "suggestion or question; attendance and acknowledgements do not count), each with role and evidence. "
          "return=suggestions gives the suggestion texts with section and speaker, plus anonymous feedback when no "
          "role is given. Example: who wrote or reviewed the Market Research Report? -> doc_people(document_type="
          "'Market Research Report'); changes suggested by an Engineering Lead -> doc_people(document_type=..., "
          "role='Engineering Lead', return='suggestions').",
          {"document_type": _s(f"Document type, {_NAME_FORMS}."),
           "scope": _enum("Which release's documents.", ("any", "current", "previous"), "any"),
           "role": _enum("Canonical directory role to keep (anonymous feedback never matches).", ROLES),
           "return": _enum("people, or suggestion texts.", ("people", "suggestions"), "people")},
          _doc_people, ("document_type",)),
    _tool("bug_board",
          "Count bugs by company (Customer.company) or by person. status predicates overlap (a reopened bug with a fix "
          "PR is resolved, active and reopened, not unresolved); actor chooses who is counted: company, current "
          "assignee, resolver, dismisser, or fix_pr_author (counts distinct PR links; the others count distinct "
          "bugs). With status=reopened, resolver, dismisser and fix_pr_author must have acted before a later "
          "reopening. rank=max keeps every tie at the top. return=bugs lists the bugs themselves (title, impact "
          "text, component, companies) with no PR required. as_of (ISO date) gives the state on that date. "
          "Example: which company has the most unresolved bugs? -> bug_board(status='unresolved', rank='max'); "
          "unresolved issues reported by a company -> bug_board(status='unresolved', customer=..., return='bugs').",
          {"status": _enum("Frozen bug predicate.", ("unresolved", "resolved", "active", "reopened", "no_fix_needed",
                                                      "any"), "any"),
           "customer": _s(f"Customer company or Customer id, {_NAME_FORMS}."),
           "component": _s("Product area (component) of the bug as worded in the question."),
           "actor": _enum("Who is counted.", ("company", "assignee", "resolver", "dismisser", "fix_pr_author"), "company"),
           "rank": _enum("max: only the rows tied at the top count; all: every row.", ("max", "all"), "all"),
           "as_of": _s("ISO date-time; state as of then (status active, resolved, no_fix_needed or any)."),
           "return": _enum("companies (actor=company), people (other actors) or bugs. Default follows actor.",
                           ("companies", "people", "bugs"))},
          _bug_board),
    _tool("fix_trail",
          "Customer company -> its bugs -> the PRs that fix them -> the PR authors, with evidence (distinct fix PR "
          "links in `answer`). Example: which PRs fixed bugs reported by a company? -> fix_trail(customer=..., "
          "engineer=optional).",
          {"customer": _s(f"Customer company or Customer id, {_NAME_FORMS}."),
           "bug": _s(f"Bug title or id, {_NAME_FORMS}."),
           "engineer": _s("PR author: employee id or exact directory name.")},
          _fix_trail),
    _tool("feature_fate",
          "Features with their dated history, capability details, contributors and PRs, selected by fate: implemented "
          "(history is kept even after a later revert), unimplemented (discussed, no implementation evidence), "
          "reverted (needs an applied revert), delayed_then_added (held, then resumed or implemented), held (any hold), any; all five are the semantic layer's derived properties. reason "
          "filters the revert or hold reason. Example: features that could not be added for security reasons -> "
          "feature_fate(fate='reverted', reason='security'); who was responsible for unimplemented features -> "
          "feature_fate(fate='unimplemented', return='people').",
          {"topic": _s(f"Feature topic, {_NAME_FORMS}."),
           "fate": _enum("Feature fate.", ("any", "implemented", "unimplemented", "reverted", "delayed_then_added",
                                           "held"), "any"),
           "reason": _enum("Revert or hold reason.", REASONS),
           "return": _enum("features, contributing people, or PRs.", ("features", "people", "prs"), "features")},
          _feature_fate),
    _tool("who_said",
          "Typed observations and demo links. subject: competitor, previous_release (release-scoped records) or "
          "product (current). kind: feature, weakness (competitor); issue, strength, qa_issue (customer or QA claims); "
          "demo (links with their sharers); worked_on (previous_release only: authors of previous documents and "
          "named QA reporters, by directory role). return: people who stated or shared, claims, companies that "
          "reported, or demo urls. Example: companies that reported issues with the previous release -> "
          "who_said(subject='previous_release', kind='issue', return='companies').",
          {"subject": _enum("What the claims are about.", ("competitor", "previous_release", "product")),
           "kind": _enum("Claim kind.", ("any", "feature", "weakness", "issue", "strength", "demo", "qa_issue",
                                         "worked_on"), "any"),
           "role": _enum("Canonical directory role to keep.", ROLES),
           "return": _enum("people, claims, companies or urls.", ("people", "claims", "companies", "urls"), "people")},
          _who_said, ("subject",)),
    _tool("pr_board",
          "Product-repository PRs (open-source look-alikes are never returned). approval: approved or unapproved by "
          "review state. state: need_merge and need_close (both open, approved, unmerged; need_close is the held "
          "ones), merged, or reverted (returns the original target PR, filtered by the revert's reason). kind: "
          "feature_development (about a feature and not fixing a bug) or bug_fix. return=people groups by author "
          "and counts distinct PR links. Example: approved PRs for a feature that need to be closed -> "
          "pr_board(feature=..., approval='approved', state='need_close'); engineer with the most approved feature "
          "PRs -> pr_board(kind='feature_development', approval='approved', return='people', rank='max').",
          {"feature": _s(f"Feature, {_NAME_FORMS}."),
           "approval": _enum("Review state.", ("any", "approved", "unapproved"), "any"),
           "state": _enum("PR state.", ("any", "need_merge", "need_close", "merged", "reverted"), "any"),
           "revert_reason": _enum("Reason of the revert (state=reverted).", REASONS),
           "kind": _enum("PR kind.", ("any", "feature_development", "bug_fix"), "any"),
           "rank": _enum("max: only people tied at the top (return=people); all: every row.", ("max", "all"), "all"),
           "return": _enum("PR links, or authors with counts.", ("links", "people"), "links")},
          _pr_board),
    _tool("bridges",
          "Knowledge bridges: people ranked by betweenness on the collaboration graph (COLLABORATED projection, fixed "
          f"seed {BETWEENNESS_SEED}). Native algo.betweenness on FalkorDB 6.0; computed in Python on the same "
          "projection on 4.22 (native scores are wrong there); each row says which (`engine`). "
          "Example: who connects the most colleagues? -> bridges(top_k=5).",
          {"top_k": _int("Number of people to return.", 10, 1, 100)},
          _bridges),
    _tool("if_leaves",
          "What-if: which work has no direct owner and which people and artefacts are affected if a person leaves "
          "(runs on a copy of the graph under the whatif identity; Task 12). "
          "Example: what happens if Alice Ng leaves? -> if_leaves(person='Alice Ng').",
          {"person": _s("The person leaving: employee id or exact directory name.")},
          _if_leaves, ("person",), kind="whatif"),
)}
