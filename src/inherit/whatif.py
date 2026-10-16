"""The counterfactual "if this person leaves" (spec 5.5, Task 12), computed in FalkorDB on a copy.

    if_leaves(conn, product, person_id) -> WhatIf

`conn` is a bare connection authenticated as `whatif_<product>`. That user can GRAPH.COPY herb:<product> to
`whatif:<product>:<uuid>`, read and write only keys of that form, and read herb:<product>; it can never write the
source graph (docs/falkor-versions.md, rule W3). Steps, all in the copy:

1. materialise `RESPONSIBLE_FOR` (Person -> item) and `DEPENDS_ON` (dependent -> prerequisite) from the original
   edges, so each question below is one pattern;
2. group 1 (already unowned) and the set of items that had an owner, before the person is removed;
3. DETACH DELETE the person; group 2 (newly unowned) is the owned-before items that have no owner now;
4. group 3 (affected) is a directed `*1..4` walk (literal bounds, #3350) from group-2 items back to their dependents;
   `bound_hit` is a dependency path of exactly 4 edges from an eligible dependent to a group-2 item.
The copy is deleted in `finally`, and the deletion is verified with an allowed command (RO_QUERY on the missing key).

Open work: active bugs; PRs with state 'open' and merged = false. Features enter group 3 only, when their fate is
`discussed_not_implemented = true` (a reconciled fate). Documents are not open work (their processing status is not in
the records). A bug's owners are the people on its latest ASSIGNED event (timestamp, then record position, then
sequence); a PR's owner is its OPENED author. A feature depends on the PR that implements it (IMPLEMENTED_BY, or ABOUT
with kind 'development'); a bug depends on the PR that FIXES it.
"""

from __future__ import annotations

import json
import time
import uuid

import redis

from inherit.config import load_settings
from inherit.contracts import TraceOp
from inherit.graph import acl as acl_mod
from inherit.tools.catalog import ToolError, ToolResult, _page, _path, _sig
from inherit.tools.depth1 import fold
from inherit.whatif_schema import MAX_HOPS, WhatIf, graph_handle, person_node_id

TOOL = "if_leaves"
COPY_TRIES, COPY_WAIT = 12, 0.25  # retries of a GRAPH.COPY refused with "could not fork", seconds per step

OPEN_OWNABLE = "(({v}:Bug AND {v}.active = true) OR ({v}:PR AND {v}.state = 'open' AND {v}.merged = false))"
OPEN_ANY = ("(" + OPEN_OWNABLE + " OR ({v}:Feature AND {v}.discussed_not_implemented = true))")

# Responsibility: a bug's latest ASSIGNED event (ts, then the record position date-index of its message id, then seq).
RESPONSIBILITY_RULES = (
    "MATCH (b:Bug)-[:HAS_EVENT]->(e:BugEvent) WHERE b.active = true AND e.type = 'ASSIGNED' "
    "WITH b, e ORDER BY e.ts DESC, coalesce(toInteger(split(e.message_id, '-')[0]), -1) DESC, "
    "coalesce(toInteger(split(e.message_id, '-')[1]), -1) DESC, coalesce(e.seq, 0) DESC, e.id DESC "
    "WITH b, collect(e.id)[0] AS last MATCH (l:BugEvent {id: last})-[:ASSIGNED_TO]->(p:Person) "
    "MERGE (p)-[:RESPONSIBLE_FOR]->(b)",
    "MATCH (p:Person)-[:OPENED]->(r:PR) WHERE r.state = 'open' AND r.merged = false "
    "MERGE (p)-[:RESPONSIBLE_FOR]->(r)",
)
# Dependency (dependent -> prerequisite): only the two kinds the records state outright.
DEPENDENCY_RULES = (
    "MATCH (f:Feature)-[:IMPLEMENTED_BY]->(r:PR) MERGE (f)-[:DEPENDS_ON]->(r)",
    "MATCH (r:PR)-[a:ABOUT]->(f:Feature) WHERE a.kind = 'development' MERGE (f)-[:DEPENDS_ON]->(r)",
    "MATCH (r:PR)-[:FIXES]->(b:Bug) MERGE (b)-[:DEPENDS_ON]->(r)",
)
Q_UNOWNED = f"MATCH (w) WHERE {OPEN_OWNABLE.format(v='w')} AND NOT ()-[:RESPONSIBLE_FOR]->(w) RETURN w.id"
Q_OWNED = f"MATCH (w) WHERE {OPEN_OWNABLE.format(v='w')} AND ()-[:RESPONSIBLE_FOR]->(w) RETURN w.id"
Q_DELETE = "MATCH (p:Person {id: $pid}) DETACH DELETE p"
Q_NEWLY = (f"MATCH (w) WHERE {OPEN_OWNABLE.format(v='w')} AND w.id IN $owned AND NOT ()-[:RESPONSIBLE_FOR]->(w) "
           "RETURN w.id")
Q_AFFECTED = (f"MATCH (a)-[:DEPENDS_ON*1..{MAX_HOPS}]->(b) WHERE b.id IN $newly AND NOT a.id IN $excluded "
              f"AND {OPEN_ANY.format(v='a')} RETURN DISTINCT a.id")
Q_BOUND = (f"MATCH (a)-[:DEPENDS_ON*{MAX_HOPS}]->(b) WHERE b.id IN $newly AND NOT a.id IN $excluded "
           f"AND {OPEN_ANY.format(v='a')} RETURN a.id LIMIT 1")


def whatif_connection(product: str, server: str = "6.0") -> redis.Redis:
    """A bare connection as whatif_<product> (password in data/acl.json)."""
    pw = json.loads(acl_mod.ACL_FILE.read_text())[server][f"whatif_{product}"]
    return redis.Redis.from_url(load_settings().falkor_urls[server], username=f"whatif_{product}", password=pw,
                                decode_responses=True)


def _redis(conn):
    """The bare connection behind a connection or a graph handle."""
    return conn.client.connection if hasattr(conn, "client") and hasattr(conn, "ro_query") else conn


class _Run:
    """Runs commands on one connection and records one TraceOp per command with the parameters actually bound."""

    def __init__(self, conn):
        self.conn, self.ops = conn, []

    def note(self, command: str, query: str, params: dict, n: int, t0: float) -> None:
        self.ops.append(TraceOp(TOOL, command, query, dict(params), n, round((time.perf_counter() - t0) * 1000, 3)))

    def q(self, graph: str, text: str, ro: bool = False, **params) -> list:
        t0 = time.perf_counter()
        g = graph_handle(self.conn, graph)
        rows = (g.ro_query(text, params or None) if ro else g.query(text, params or None)).result_set
        self.note("GRAPH.RO_QUERY" if ro else "GRAPH.QUERY", text, params, len(rows), t0)
        return rows

    def copy(self, src: str, dst: str) -> None:
        """GRAPH.COPY forks the server: while another fork (a copy, a save) runs it answers "could not fork"."""
        t0 = time.perf_counter()
        for attempt in range(COPY_TRIES):
            try:
                self.conn.execute_command("GRAPH.COPY", src, dst)
                break
            except redis.exceptions.ResponseError as e:
                if "could not fork" not in str(e) or attempt == COPY_TRIES - 1:
                    raise
                time.sleep(COPY_WAIT * (attempt + 1))
        self.note("GRAPH.COPY", "", {"src": src, "dst": dst}, 0, t0)

    def delete(self, graph: str) -> None:
        t0 = time.perf_counter()
        self.conn.execute_command("GRAPH.DELETE", graph)
        self.note("GRAPH.DELETE", "", {"graph": graph}, 0, t0)


def _missing(conn, graph: str) -> bool:
    """True when the graph key is gone: RO_QUERY on it fails with the expected empty-key error."""
    try:
        conn.execute_command("GRAPH.RO_QUERY", graph, "RETURN 1")
    except redis.exceptions.NoPermissionError:
        raise
    except redis.exceptions.ResponseError as e:
        if "empty key" in str(e):
            return True
        raise
    return False


def _cleanup(run: _Run, graph: str) -> None:
    if not graph.startswith("whatif:"):
        raise RuntimeError(f"refusing to delete {graph!r}: not a what-if copy")
    if not _missing(run.conn, graph):
        run.delete(graph)
    if not _missing(run.conn, graph):
        raise RuntimeError(f"what-if copy {graph} still exists after GRAPH.DELETE")


def _ids(rows) -> list[str]:
    return sorted({r[0] for r in rows})


def _affected(run: _Run, copy: str, newly: list[str], excluded: list[str]) -> tuple[list[str], bool]:
    if not newly:
        return [], False
    affected = _ids(run.q(copy, Q_AFFECTED, newly=newly, excluded=excluded))
    return affected, bool(run.q(copy, Q_BOUND, newly=newly, excluded=excluded))


def if_leaves(conn, product: str, person_id: str) -> WhatIf:
    """The three groups for removing `person_id` (an employee id or a `Person:` node id) from herb:<product>."""
    conn = _redis(conn)
    run = _Run(conn)
    pid = person_node_id(person_id)
    src, copy = f"herb:{product}", f"whatif:{product}:{uuid.uuid4().hex}"
    if not run.q(src, "MATCH (p:Person {id: $pid}) RETURN count(p)", ro=True, pid=pid)[0][0]:
        raise ValueError(f"no such person {person_id!r} in {src}")
    made = False
    try:
        run.copy(src, copy)
        made = True
        for text in (*RESPONSIBILITY_RULES, *DEPENDENCY_RULES):
            run.q(copy, text)
        already = _ids(run.q(copy, Q_UNOWNED))
        owned = _ids(run.q(copy, Q_OWNED))
        run.q(copy, Q_DELETE, pid=pid)
        newly = _ids(run.q(copy, Q_NEWLY, owned=owned))
        affected, bound_hit = _affected(run, copy, newly, sorted({*already, *newly}))
    finally:
        if made:
            _cleanup(run, copy)
    return WhatIf(frozenset(already), frozenset(newly), frozenset(affected), bound_hit, tuple(run.ops), copy)


# ---------------------------------------------------------------- the catalogue tool
BASIS = ("Computed from the responsibility and dependency recorded in the data, not a prediction of real "
         "consequences. Documents are not counted: the records do not say whether their feedback is still open.")
GROUPS = (("newly_unowned", "newly_unowned"), ("already_unowned", "already_unowned"), ("affected", "affected"))


def _resolve(conn, product: str, text: str) -> tuple[str, str, str | None]:
    """(eid, name, role) of the one person the text names (an employee id or an exact directory name)."""
    f = fold(text[7:] if text.startswith("Person:") else text)
    rows = graph_handle(conn, f"herb:{product}").ro_query(
        "MATCH (p:Person) RETURN p.eid, p.name, p.role").result_set
    hits = sorted({(e, n, r) for e, n, r in rows if fold(e or "") == f or (n and fold(n) == f)}, key=lambda x: x[0])
    if not hits:
        raise ToolError(f"no person named {text!r} in {product}")
    if len(hits) > 1:
        cands = "; ".join(f"{e} ({n}, {r})" for e, n, r in hits)
        raise ToolError(f"ambiguous name {text!r}: {len(hits)} candidates: {cands}; call again with an employee id")
    return hits[0]


def run_tool(conn, product: str, person: str) -> ToolResult:
    t0 = time.perf_counter()
    conn = _redis(conn)
    try:
        eid, name, role = _resolve(conn, product, person)
        res = if_leaves(conn, product, eid)
    except redis.exceptions.NoPermissionError as e:
        raise ToolError("if_leaves needs the what-if identity (whatif_<product>); this connection may not copy "
                        f"the graph: {e}") from e
    ids = sorted(res.already_unowned | res.newly_unowned | res.affected)
    info = {i: (lb, title) for i, lb, title in graph_handle(conn, f"herb:{product}").ro_query(
        "MATCH (n) WHERE n.id IN $ids RETURN n.id, labels(n)[0], n.title", {"ids": ids}).result_set} if ids else {}
    counts = {"already_unowned": len(res.already_unowned), "newly_unowned": len(res.newly_unowned),
              "affected": len(res.affected)}
    rows = [{"answer": eid, "person": name, "role": role, "counts": counts, "bound_hit": res.bound_hit,
             "max_hops": MAX_HOPS, "basis": BASIS, "refs": [f"Person:{eid}"], "evidence": []}]
    for group, attr in GROUPS:
        for i in sorted(getattr(res, attr)):
            lb, title = info.get(i, (i.split(":", 1)[0], None))
            rows.append({"answer": i, "group": group, "kind": lb, "title": title, "refs": [i, f"Person:{eid}"],
                         "evidence": []})
    page, more, _ = _page(rows, 0, _sig(product, TOOL, {"person": eid}))
    return ToolResult(page, more, None, list(res.ops), _path(page), round((time.perf_counter() - t0) * 1000, 3))
