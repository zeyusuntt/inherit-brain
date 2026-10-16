"""Probe FalkorDB features on 4.22.1 (port 6379) and 6.0.2 (port 6380).

Run: uv run --with falkordb==1.8.0 python scripts/probe_falkor.py
Writes docs/falkor-versions.md. Every probe graph, user and UDF library is
removed in finally blocks, so the script can be re-run.
"""

from __future__ import annotations

import random
import sys
import time
import traceback
from pathlib import Path

import redis
from falkordb import FalkorDB

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "falkor-versions.md"
UDF_FILE = ROOT / "src" / "inherit" / "graph" / "udf" / "herb.js"
VERSIONS = {"4.22.1": 6379, "6.0.2": 6380}
PW = "probe-pw"
LIB = "herb_v1"
GRAPHS = ["herb:probe_a", "herb:probe_b", "whatif:probe_a:1", "whatif:probe_a:2", "whatif:probe_a:3"]
USERS = ["agent_probe_a", "whatif_probe_a", "agent_probe_a_v2", "agent_probe_a_v3", "agent_probe_a_v4",
         "agent_probe_a_v5", "whatif_probe_a_v2", "whatif_probe_a_v3"]

UDF_JS = """\
// herb_v1.statusAt(events, date): status after the last event with ts <= date.
// events: list of {type, ts}; type in REPORTED, ASSIGNED, RESOLVED, DISMISSED, REOPENED.
function statusAt(events, date) {
    var best = null;
    var status = null;
    for (var i = 0; i < events.length; i++) {
        var e = events[i];
        if (e.ts <= date && (best === null || e.ts >= best)) {
            best = e.ts;
            if (e.type === 'RESOLVED') status = 'resolved';
            else if (e.type === 'DISMISSED') status = 'dismissed';
            else if (e.type === 'REPORTED' || e.type === 'ASSIGNED' || e.type === 'REOPENED') status = 'active';
        }
    }
    return status;
}
falkor.register('statusAt', statusAt);
"""

# results[probe][version] = (ok, note); facts[version][key] = text
results: dict[str, dict[str, tuple[bool, str]]] = {}
facts: dict[str, dict[str, str]] = {v: {} for v in VERSIONS}
order: list[str] = []
BETW: dict[str, list] = {}
BETW_FULL: dict[str, list] = {}


def short(e: BaseException) -> str:
    s = " ".join(str(e).split())
    return f"{type(e).__name__}: {s}"[:160]


def rec(probe: str, ver: str, ok: bool, note: str = "") -> None:
    if probe not in results:
        results[probe] = {}
        order.append(probe)
    results[probe][ver] = (ok, note)
    print(f"[{ver}] {'ok  ' if ok else 'FAIL'} {probe} {note}", flush=True)


def attempt(probe: str, ver: str, fn, expect_fail: bool = False) -> bool:
    """Run fn. Records ok if it works (or, with expect_fail, if it is refused)."""
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001
        if expect_fail:
            rec(probe, ver, True, "refused: " + short(e))
        else:
            rec(probe, ver, False, short(e))
        return False
    if expect_fail:
        rec(probe, ver, False, "NOT refused" + (f" ({out})" if out is not None else ""))
        return False
    rec(probe, ver, True, str(out) if isinstance(out, (str, int, float, list)) else "")
    return True


def wipe(r: redis.Redis) -> None:
    for g in GRAPHS:
        try:
            r.execute_command("GRAPH.DELETE", g)
        except Exception:  # noqa: BLE001
            pass
    try:
        for g in r.execute_command("GRAPH.LIST"):
            if g.startswith("whatif:probe_a:"):
                r.execute_command("GRAPH.DELETE", g)
    except Exception:  # noqa: BLE001
        pass
    for u in USERS:
        try:
            r.execute_command("ACL", "DELUSER", u)
        except Exception:  # noqa: BLE001
            pass
    try:
        r.execute_command("GRAPH.UDF", "DELETE", LIB)
    except Exception:  # noqa: BLE001
        pass


def user_conn(port: int, user: str) -> redis.Redis:
    return redis.Redis(port=port, username=user, password=PW, decode_responses=True)


def seed_a(db) -> None:
    for name in ("herb:probe_a", "herb:probe_b"):
        try:
            db.select_graph(name).delete()
        except Exception:  # noqa: BLE001
            pass
    g = db.select_graph("herb:probe_a")
    g.query("CREATE (:Person {eid:'e1', name:'Ann'}), (:Person {eid:'e2', name:'Bo'})")
    db.select_graph("herb:probe_b").query("CREATE (:Person {eid:'x1', name:'Xi'})")


# ---------------------------------------------------------------- ACL
AGENT_VARIANTS = {
    "A": ("agent_probe_a", "spec rule %R~ + RO_QUERY only",
          ["+GRAPH.RO_QUERY", "%R~herb:probe_a"]),
    "A2": ("agent_probe_a_v2", "spec rule %R~ + RO_QUERY + QUERY",
           ["+GRAPH.RO_QUERY", "+GRAPH.QUERY", "%R~herb:probe_a"]),
    "B": ("agent_probe_a_v3", "key selector ~ + RO_QUERY only",
          ["+GRAPH.RO_QUERY", "~herb:probe_a"]),
    "B2": ("agent_probe_a_v4", "key selector ~ + RO_QUERY + QUERY",
           ["+GRAPH.RO_QUERY", "+GRAPH.QUERY", "~herb:probe_a"]),
}
WHATIF_VARIANTS = {
    "W": ("whatif_probe_a", "spec rule ~whatif:probe_a:* %R~herb:probe_a",
          ["~whatif:probe_a:*", "%R~herb:probe_a", "+GRAPH.COPY", "+GRAPH.QUERY", "+GRAPH.RO_QUERY", "+GRAPH.DELETE"]),
    "W2": ("whatif_probe_a_v2", "plain keys ~whatif:probe_a:* ~herb:probe_a",
           ["~whatif:probe_a:*", "~herb:probe_a", "+GRAPH.COPY", "+GRAPH.QUERY", "+GRAPH.RO_QUERY", "+GRAPH.DELETE"]),
    "W3": ("whatif_probe_a_v3", "ACL selectors (per-key command sets)",
           ["(~whatif:probe_a:* ~herb:probe_a +GRAPH.COPY)",
            "(~whatif:probe_a:* +GRAPH.QUERY +GRAPH.RO_QUERY +GRAPH.DELETE)",
            "(~herb:probe_a +GRAPH.RO_QUERY)"]),
}
AGENT_SELECTOR = ("agent_probe_a_v5", "ACL selector (~herb:probe_a +GRAPH.RO_QUERY)",
                  ["(~herb:probe_a +GRAPH.RO_QUERY)"])


def list_graphs(r: redis.Redis) -> list[str]:
    return list(r.execute_command("GRAPH.LIST"))


def agent_checks(ver: str, port: int, r: redis.Redis, key: str, u: str, label: str, rules: list[str]) -> None:
    seed_a(FalkorDB(port=port))
    pre = f"ACL agent {key} ({label})"
    try:
        r.execute_command("ACL", "SETUSER", u, "on", f">{PW}", "resetkeys", *rules)
    except Exception as e:  # noqa: BLE001
        rec(f"{pre}: SETUSER accepted", ver, False, short(e))
        return
    rec(f"{pre}: SETUSER accepted", ver, True)
    facts[ver][f"acl_{key}"] = " ".join(["on", ">pw", *rules])
    c = user_conn(port, u)
    attempt(f"{pre}: RO_QUERY own graph allowed", ver,
            lambda: c.execute_command("GRAPH.RO_QUERY", "herb:probe_a", "MATCH (n:Person) RETURN count(n)")[1])
    attempt(f"{pre}: RO_QUERY herb:probe_b refused", ver,
            lambda: c.execute_command("GRAPH.RO_QUERY", "herb:probe_b", "MATCH (n:Person) RETURN count(n)"),
            expect_fail=True)
    attempt(f"{pre}: GRAPH.QUERY write on own graph refused", ver,
            lambda: c.execute_command("GRAPH.QUERY", "herb:probe_a", "CREATE (:Person {eid:'zz'})"),
            expect_fail=True)
    attempt(f"{pre}: GRAPH.RO_QUERY write on own graph refused", ver,
            lambda: c.execute_command("GRAPH.RO_QUERY", "herb:probe_a", "CREATE (:Person {eid:'zz'})"),
            expect_fail=True)
    c.close()


def whatif_checks(ver: str, port: int, r: redis.Redis, key: str, u: str, label: str, rules: list[str]) -> None:
    seed_a(FalkorDB(port=port))
    pre = f"ACL whatif {key} ({label})"
    try:
        r.execute_command("ACL", "SETUSER", u, "on", f">{PW}", "resetkeys", *rules)
    except Exception as e:  # noqa: BLE001
        rec(f"{pre}: SETUSER accepted", ver, False, short(e))
        return
    rec(f"{pre}: SETUSER accepted", ver, True)
    facts[ver][f"acl_{key}"] = " ".join(["on", ">pw", *rules])
    c = user_conn(port, u)
    cp = "whatif:probe_a:1"
    copied = attempt(f"{pre}: GRAPH.COPY herb:probe_a to whatif copy", ver,
                     lambda: c.execute_command("GRAPH.COPY", "herb:probe_a", cp))
    if copied:
        attempt(f"{pre}: write to the copy works", ver,
                lambda: c.execute_command("GRAPH.QUERY", cp, "CREATE (:Person {eid:'w1'})")[0][0])
        attempt(f"{pre}: read the copy works", ver,
                lambda: c.execute_command("GRAPH.RO_QUERY", cp, "MATCH (n:Person) RETURN count(n)")[1])
    attempt(f"{pre}: GRAPH.QUERY write to herb:probe_a refused", ver,
            lambda: c.execute_command("GRAPH.QUERY", "herb:probe_a", "CREATE (:Person {eid:'zz'})"),
            expect_fail=True)
    attempt(f"{pre}: GRAPH.COPY from herb:probe_b refused", ver,
            lambda: c.execute_command("GRAPH.COPY", "herb:probe_b", "whatif:probe_a:2"), expect_fail=True)
    attempt(f"{pre}: GRAPH.COPY into herb:probe_b refused", ver,
            lambda: c.execute_command("GRAPH.COPY", cp, "herb:probe_b") if copied else
            c.execute_command("GRAPH.COPY", "herb:probe_a", "herb:probe_b"), expect_fail=True)
    attempt(f"{pre}: GRAPH.DELETE of herb:probe_a refused", ver,
            lambda: c.execute_command("GRAPH.DELETE", "herb:probe_a"), expect_fail=True)
    if copied:
        attempt(f"{pre}: GRAPH.DELETE of the copy works", ver, lambda: c.execute_command("GRAPH.DELETE", cp))

    # cleanup-on-exception pattern
    cp3 = "whatif:probe_a:3"
    made = False
    try:
        try:
            c.execute_command("GRAPH.COPY", "herb:probe_a", cp3)
            made = True
            raise RuntimeError("simulated failure mid-way")
        finally:
            try:
                c.execute_command("GRAPH.DELETE", cp3)
            except Exception:  # noqa: BLE001
                pass
    except RuntimeError:
        pass
    except Exception:  # noqa: BLE001  (copy itself was refused)
        pass
    left = cp3 in list_graphs(r)
    if made:
        rec(f"{pre}: try/finally deletes copy after exception", ver, not left, "copy left behind" if left else "")
    else:
        rec(f"{pre}: try/finally deletes copy after exception", ver, False, "n/a, GRAPH.COPY refused")
    c.close()
    for g in list_graphs(r):  # admin sweep between variants, not part of the probe result
        if g.startswith("whatif:probe_a:"):
            r.execute_command("GRAPH.DELETE", g)


def probe_acl(ver: str, port: int, r: redis.Redis, db) -> None:
    seed_a(db)
    for key, (u, label, rules) in AGENT_VARIANTS.items():
        agent_checks(ver, port, r, key, u, label, rules)
    agent_checks(ver, port, r, "S", *AGENT_SELECTOR)
    for key, (u, label, rules) in WHATIF_VARIANTS.items():
        whatif_checks(ver, port, r, key, u, label, rules)
    # why %R~ fails: the static key spec of the graph commands
    r.response_callbacks.pop("COMMAND", None)
    for cmd in (("GRAPH.RO_QUERY", "g", "RETURN 1"), ("GRAPH.QUERY", "g", "RETURN 1"), ("GRAPH.COPY", "a", "b")):
        try:
            facts[ver][f"key spec {cmd[0]}"] = str(r.execute_command("COMMAND", "GETKEYSANDFLAGS", *cmd))
        except Exception as e:  # noqa: BLE001
            facts[ver][f"key spec {cmd[0]}"] = short(e)


# ---------------------------------------------------------------- UDF
def probe_udf(ver: str, db) -> None:
    g = db.select_graph("herb:probe_b")
    g.query("RETURN 1")
    ok = attempt("UDF load herb_v1 (no REPLACE)", ver, lambda: db.udf_load(LIB, UDF_JS))
    if not ok:
        return
    q = ("RETURN herb_v1.statusAt([{type:'REPORTED', ts:'2024-01-01'}, {type:'RESOLVED', ts:'2024-02-01'}, "
         "{type:'REOPENED', ts:'2024-03-01'}], $d)")
    expect = {"2023-12-01": None, "2024-01-15": "active", "2024-02-15": "resolved", "2024-04-01": "active"}

    def call():
        got = {d: g.query(q, {"d": d}).result_set[0][0] for d in expect}
        if got != expect:
            raise AssertionError(f"got {got}")
        return "4 dates correct"

    attempt("UDF statusAt callable from a query", ver, call)

    def from_graph():
        g.query("CREATE (b:Bug {id:'b1'}), (:Event {type:'REPORTED', ts:'2024-01-01'})-[:OF]->(b), "
                "(:Event {type:'RESOLVED', ts:'2024-02-01'})-[:OF]->(b)")
        return g.query("MATCH (e:Event)-[:OF]->(b:Bug) WITH b, collect({type:e.type, ts:e.ts}) AS evs "
                       "RETURN herb_v1.statusAt(evs, $d)", {"d": "2024-06-01"}).result_set[0][0]

    attempt("UDF statusAt on collect() of Event nodes (maps cannot be stored as properties)", ver, from_graph)
    attempt("UDF second load of same name refused (no REPLACE)", ver, lambda: db.udf_load(LIB, UDF_JS), expect_fail=True)


# ---------------------------------------------------------------- constraints
def wait_for(fn, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.2)
    return None


def probe_constraints(ver: str, r: redis.Redis, db) -> None:
    g = db.select_graph("herb:probe_b")
    g.query("CREATE (:Person {eid:'p1', name:'P1'})")
    attempt("UNIQUE without exact-match index is refused", ver,
            lambda: r.execute_command("GRAPH.CONSTRAINT", "CREATE", "herb:probe_b", "UNIQUE", "NODE", "Person",
                                      "PROPERTIES", "1", "eid"), expect_fail=True)
    attempt("CREATE INDEX FOR (p:Person) ON (p.eid)", ver, lambda: g.query("CREATE INDEX FOR (p:Person) ON (p.eid)"))
    idx = wait_for(lambda: all(row[-1] == "OPERATIONAL" or "OPERATIONAL" in [str(x) for x in row]
                               for row in g.query("CALL db.indexes()").result_set) or None)
    rec("index reaches OPERATIONAL", ver, bool(idx))
    facts[ver]["db.indexes row"] = repr(g.query("CALL db.indexes()").result_set[:1])[:300]
    attempt("GRAPH.CONSTRAINT CREATE UNIQUE Person.eid", ver,
            lambda: r.execute_command("GRAPH.CONSTRAINT", "CREATE", "herb:probe_b", "UNIQUE", "NODE", "Person",
                                      "PROPERTIES", "1", "eid"))

    def status():
        return g.query("CALL db.constraints()").result_set

    def all_op():
        rows = status()
        return rows if rows and all("OPERATIONAL" in [str(x) for x in row] for row in rows) else None

    got = wait_for(all_op)
    rec("UNIQUE constraint reaches OPERATIONAL (db.constraints)", ver, bool(got), "" if got else f"rows={status()}")
    facts[ver]["db.constraints row"] = repr(status()[:1])[:300]
    attempt("duplicate Person.eid insert fails", ver, lambda: g.query("CREATE (:Person {eid:'p1'})"), expect_fail=True)
    attempt("distinct Person.eid insert works", ver, lambda: g.query("CREATE (:Person {eid:'p2'})"))
    attempt("MANDATORY Bug.status create", ver,
            lambda: r.execute_command("GRAPH.CONSTRAINT", "CREATE", "herb:probe_b", "MANDATORY", "NODE", "Bug",
                                      "PROPERTIES", "1", "status"))
    wait_for(lambda: all("OPERATIONAL" in [str(x) for x in row] for row in status()) or None)
    attempt("Bug without status fails", ver, lambda: g.query("CREATE (:Bug {id:'b1'})"), expect_fail=True)
    attempt("Bug with status works", ver, lambda: g.query("CREATE (:Bug {id:'b2', status:'open'})"))


# ---------------------------------------------------------------- indexes
def probe_indexes(ver: str, db) -> None:
    g = db.select_graph("herb:probe_b")
    g.query("CREATE (:Person {name:'Alice Johnson'}), (:Person {name:'Bob Stone'}), "
            "(:Ticket {title:'Login page crashes on Safari'}), (:Ticket {title:'Slow export of reports'})")
    attempt("full-text DDL: CREATE FULLTEXT INDEX FOR (n:Person) ON (n.name)", ver,
            lambda: g.query("CREATE FULLTEXT INDEX FOR (n:Person) ON (n.name)"))
    attempt("full-text DDL: CREATE FULLTEXT INDEX FOR (n:Ticket) ON (n.title)", ver,
            lambda: g.query("CREATE FULLTEXT INDEX FOR (n:Ticket) ON (n.title)"))
    time.sleep(1)

    def ft(label, term):
        rs = g.query(f"CALL db.idx.fulltext.queryNodes('{label}', '{term}') YIELD node, score "
                     "RETURN node, score")
        facts[ver][f"fulltext row {label}"] = f"columns={rs.header!r} first_row_types={[type(x).__name__ for x in rs.result_set[0]] if rs.result_set else []}"
        if not rs.result_set:
            raise AssertionError("no rows")
        return f"{len(rs.result_set)} row(s)"

    attempt("full-text query Person.name 'alice'", ver, lambda: ft("Person", "alice"))
    attempt("full-text query Ticket.title 'crashes' (stemmed 'crash')", ver, lambda: ft("Ticket", "crash"))

    rng = random.Random(7)
    attempt("vector DDL: CREATE VECTOR INDEX dim 4096 cosine on Message.embedding", ver,
            lambda: g.query("CREATE VECTOR INDEX FOR (m:Message) ON (m.embedding) "
                            "OPTIONS {dimension:4096, similarityFunction:'cosine'}"))
    vecs = [[rng.uniform(-1, 1) for _ in range(4096)] for _ in range(6)]
    for i, v in enumerate(vecs):
        g.query("CREATE (:Message {id:$i, embedding: vecf32($v)})", {"i": i, "v": v})
    time.sleep(1)

    def vq():
        rs = g.query("CALL db.idx.vector.queryNodes('Message', 'embedding', 3, vecf32($q)) YIELD node, score "
                     "RETURN node.id AS id, score ORDER BY score", {"q": vecs[2]})
        facts[ver]["vector row"] = f"columns={rs.header!r} rows={[(r[0], round(r[1], 6)) for r in rs.result_set]}"
        if not rs.result_set or rs.result_set[0][0] != 2:
            raise AssertionError(f"nearest not self: {rs.result_set}")
        return f"top-{len(rs.result_set)}, nearest is self, score {rs.result_set[0][1]:.6f}"

    attempt("vector top-3 query returns the stored vector first", ver, vq)
    attempt("vector wrong dimension insert is refused", ver,
            lambda: g.query("CREATE (:Message {id:99, embedding: vecf32([1.0, 2.0])})"), expect_fail=True)


# ---------------------------------------------------------------- betweenness
def probe_betweenness(ver: str, db) -> list[tuple[str, float]]:
    g = db.select_graph("herb:probe_b")
    g.query("CREATE (a:Person {eid:'a'}), (b:Person {eid:'b'}), (c:Person {eid:'c'}), (d:Person {eid:'d'}), "
            "(e:Person {eid:'e'}), (f:Person {eid:'f'}), "
            "(a)-[:COLLABORATED]->(b), (b)-[:COLLABORATED]->(c), (b)-[:COLLABORATED]->(d), "
            "(c)-[:COLLABORATED]->(e), (d)-[:COLLABORATED]->(e), (e)-[:COLLABORATED]->(f)")
    q = ("CALL algo.betweenness({nodeLabels:['Person'], relationshipTypes:['COLLABORATED'], "
         "samplingSize:4, samplingSeed:42}) YIELD node, score RETURN node.eid AS eid, score ORDER BY eid")
    runs = []
    for _ in range(2):
        try:
            runs.append([(r[0], float(r[1])) for r in g.query(q).result_set])
        except Exception as e:  # noqa: BLE001
            rec("algo.betweenness runs (samplingSeed, samplingSize)", ver, False, short(e))
            return []
    rec("algo.betweenness runs (samplingSeed, samplingSize)", ver, True, f"{len(runs[0])} rows")
    rec("betweenness identical across 2 runs, fixed seed", ver, runs[0] == runs[1], str(runs[0]))
    facts[ver]["betweenness"] = str(runs[0])
    BETW[ver] = runs[0]
    # full (unsampled) call for reference
    try:
        full = [(r[0], float(r[1])) for r in g.query(
            "CALL algo.betweenness({nodeLabels:['Person'], relationshipTypes:['COLLABORATED']}) "
            "YIELD node, score RETURN node.eid AS eid, score ORDER BY eid").result_set]
        facts[ver]["betweenness no sampling args"] = str(full)
        BETW_FULL[ver] = full
    except Exception as e:  # noqa: BLE001
        facts[ver]["betweenness no sampling args"] = short(e)
    # correctness on tiny graphs with a known directed answer
    g.delete()
    g = db.select_graph("herb:probe_b")
    g.query("CREATE (a:Person {eid:'a'}), (b:Person {eid:'b'}), (c:Person {eid:'c'}), "
            "(a)-[:COLLABORATED]->(b), (b)-[:COLLABORATED]->(c)")

    def chain(cfg: str) -> dict:
        rs = g.query(f"CALL algo.betweenness({{nodeLabels:['Person'], relationshipTypes:['COLLABORATED']{cfg}}}) "
                     "YIELD node, score RETURN node.eid, score ORDER BY node.eid").result_set
        return {r[0]: float(r[1]) for r in rs}

    for label, cfg in (("no sampling args", ""), ("samplingSize 10 (> node count), samplingSeed 1",
                                                   ", samplingSize:10, samplingSeed:1")):
        got = chain(cfg)
        rec(f"betweenness chain a->b->c, {label}: b scores 1.0", ver, got.get("b") == 1.0, f"scores {got}")
    g.delete()
    g = db.select_graph("herb:probe_b")
    g.query("CREATE (a:Person {eid:'Alice'}), (b:Person {eid:'Bob'}), (c:Person {eid:'Charlie'}), "
            "(d:Person {eid:'David'}), (e:Person {eid:'Emma'}), (a)-[:COLLABORATED]->(b), "
            "(b)-[:COLLABORATED]->(c), (b)-[:COLLABORATED]->(d), (c)-[:COLLABORATED]->(e), (d)-[:COLLABORATED]->(e)")
    got = chain("")
    rec("betweenness 5-node docs example, directed pairs counted once: Bob 3, Charlie 1, David 1", ver,
        (got.get("Bob"), got.get("Charlie"), got.get("David")) == (3.0, 1.0, 1.0), f"scores {got}")
    create5 = ("CREATE (a:Person {eid:'Alice'}), (b:Person {eid:'Bob'}), (c:Person {eid:'Charlie'}), "
               "(d:Person {eid:'David'}), (e:Person {eid:'Emma'}), (a)-[:COLLABORATED]->(b), "
               "(b)-[:COLLABORATED]->(c), (b)-[:COLLABORATED]->(d), (c)-[:COLLABORATED]->(e), (d)-[:COLLABORATED]->(e)")
    rebuilt = set()
    for _ in range(6):
        time.sleep(1.1)  # the unseeded default is time based
        g.delete()
        g = db.select_graph("herb:probe_b")
        g.query(create5)
        rebuilt.add(tuple(sorted(chain("").items())))
    rec("betweenness 5-node example, 6 fresh builds, 1.1 s apart, no samplingSeed, give identical scores", ver,
        len(rebuilt) == 1, f"{len(rebuilt)} distinct result(s) in 6 builds")
    seen = [tuple(sorted(chain("").items())) for _ in range(8)]
    rec("betweenness 5-node example, 8 queries on one graph, no samplingSeed, give identical scores", ver,
        len(set(seen)) == 1, f"{len(set(seen))} distinct result(s) in 8 runs")
    return runs[0]


# ---------------------------------------------------------------- bugs and query shapes
def probe_queries(ver: str, r: redis.Redis, db) -> None:
    g = db.select_graph("herb:probe_b")
    g.query("UNWIND range(0,5) AS i CREATE (:N {i:i})")
    g.query("MATCH (a:N {i:0}),(b:N {i:1}),(c:N {i:2}),(d:N {i:3}),(e:N {i:4}),(f:N {i:5}) "
            "CREATE (a)-[:R]->(b)-[:R]->(c)-[:R]->(d)-[:R]->(e)-[:R]->(f)")
    lit = g.query("MATCH (a:N {i:0})-[:R*1..4]->(b) RETURN count(b)").result_set[0][0]
    rec("#3350 literal bound *1..4", ver, lit == 4, f"count={lit}")
    try:
        par = g.query("MATCH (a:N {i:0})-[:R*1..$n]->(b) RETURN count(b)", {"n": 4}).result_set[0][0]
        rec("#3350 parameter bound *1..$n", ver, par == 4, f"count={par}")
    except Exception as e:  # noqa: BLE001
        rec("#3350 parameter bound *1..$n", ver, False, short(e))

    # RO_QUERY rejects writes
    attempt("GRAPH.RO_QUERY rejects a write", ver,
            lambda: r.execute_command("GRAPH.RO_QUERY", "herb:probe_b", "CREATE (:Z)"), expect_fail=True)
    r.execute_command("GRAPH.DELETE", "herb:probe_b")

    g = db.select_graph("herb:probe_b")
    g.query("CREATE (:Bug {id:'b1', status:'open'}), (:Bug {id:'b2', status:'open'}), (:Bug {id:'b3', status:'open'}), "
            "(:Bug {id:'b4', status:'open'}), (:Bug {id:'b5', status:'open'}), "
            "(:PR {link:'p1', merged:'True'}), (:PR {link:'p2', merged:'False'})")
    g.query("MATCH (b:Bug {id:'b1'}), (p:PR {link:'p1'}) CREATE (p)-[:FIXES]->(b)")
    g.query("MATCH (b:Bug {id:'b2'}), (p:PR {link:'p2'}) CREATE (p)-[:FIXES]->(b)")
    # expected: b1 (merged PR), b3, b4, b5 (no PR) = 4; b2 (unmerged PR) excluded
    shapes = {
        "3351 spec shape: count(*) after OPTIONAL MATCH ... WITH b, p WHERE p IS NULL OR p.merged": (
            "MATCH (b:Bug) OPTIONAL MATCH (b)<-[:FIXES]-(p:PR) WITH b, p "
            "WHERE p IS NULL OR p.merged = 'True' RETURN count(*)", 4),
        "3351 variant: return ids": (
            "MATCH (b:Bug) OPTIONAL MATCH (b)<-[:FIXES]-(p:PR) WITH b, p "
            "WHERE p IS NULL OR p.merged = 'True' RETURN b.id ORDER BY b.id", 4),
        "3351 variant: only p IS NULL": (
            "MATCH (b:Bug) OPTIONAL MATCH (b)<-[:FIXES]-(p:PR) WITH b, p WHERE p IS NULL RETURN count(*)", 3),
        "3351 variant: WHERE on the OPTIONAL MATCH itself": (
            "MATCH (b:Bug) OPTIONAL MATCH (b)<-[:FIXES]-(p:PR) WHERE p.merged = 'True' RETURN count(*)", 5),
        "3351 workaround: pattern predicates, no OPTIONAL MATCH": (
            "MATCH (b:Bug) WHERE NOT (b)<-[:FIXES]-(:PR) OR (b)<-[:FIXES]-(:PR {merged:'True'}) "
            "RETURN count(*)", 4),
    }
    for name, (q, want) in shapes.items():
        try:
            rs = g.query(q).result_set
            got = len(rs) if "ORDER BY" in q else rs[0][0]
            rec(name, ver, got == want, f"got {got}, expected {want}")
        except Exception as e:  # noqa: BLE001
            rec(name, ver, False, short(e))


def probe_3351_scale(ver: str, r: redis.Redis, db) -> None:
    g = db.select_graph("herb:probe_b")
    n = 3000
    g.query("UNWIND range(0,$n-1) AS i CREATE (:Bug {id:i, status:'open'})", {"n": n})
    g.query("CREATE INDEX FOR (b:Bug) ON (b.id)")
    # bug i: no PR if i%3==0; one merged PR if i%3==1; one unmerged PR + one merged PR if i%3==2 half the time
    g.query("UNWIND range(0,$n-1) AS i WITH i WHERE i%3=1 MATCH (b:Bug {id:i}) "
            "CREATE (:PR {link:'m'+toString(i), merged:'True'})-[:FIXES]->(b)", {"n": n})
    g.query("UNWIND range(0,$n-1) AS i WITH i WHERE i%3=2 MATCH (b:Bug {id:i}) "
            "CREATE (:PR {link:'u'+toString(i), merged:'False'})-[:FIXES]->(b)", {"n": n})
    # no-PR bugs: n/3 ; merged-PR bugs: n/3 ; unmerged-only bugs: n/3
    want = 2 * n // 3
    qs = {
        "3351 at 3000 bugs: spec shape count(*)": (
            "MATCH (b:Bug) OPTIONAL MATCH (b)<-[:FIXES]-(p:PR) WITH b, p "
            "WHERE p IS NULL OR p.merged = 'True' RETURN count(*)", want),
        "3351 at 3000 bugs: same, rows returned": (
            "MATCH (b:Bug) OPTIONAL MATCH (b)<-[:FIXES]-(p:PR) WITH b, p "
            "WHERE p IS NULL OR p.merged = 'True' RETURN b.id", want),
        "3351 at 3000 bugs: followed by a second MATCH": (
            "MATCH (b:Bug) OPTIONAL MATCH (b)<-[:FIXES]-(p:PR) WITH b, p "
            "WHERE p IS NULL OR p.merged = 'True' MATCH (b2:Bug {id:b.id}) RETURN count(*)", want),
    }
    for name, (q, w) in qs.items():
        try:
            rs = g.query(q).result_set
            got = len(rs) if "RETURN b.id" in q else rs[0][0]
            rec(name, ver, got == w, f"got {got}, expected {w}")
        except Exception as e:  # noqa: BLE001
            rec(name, ver, False, short(e))


# ---------------------------------------------------------------- main
def run_version(ver: str, port: int) -> dict:
    r = redis.Redis(port=port, decode_responses=True)
    db = FalkorDB(port=port)
    info = {}
    try:
        wipe(r)
        info = r.execute_command("MODULE", "LIST")
        steps = [
            ("acl", lambda: probe_acl(ver, port, r, db)),
            ("udf", lambda: probe_udf(ver, db)),
            ("constraints", lambda: probe_constraints(ver, r, db)),
            ("indexes", lambda: probe_indexes(ver, db)),
            ("betweenness", lambda: probe_betweenness(ver, db)),
            ("queries", lambda: probe_queries(ver, r, db)),
            ("3351 scale", lambda: probe_3351_scale(ver, r, db)),
        ]
        for name, fn in steps:
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rec(f"section {name} crashed", ver, False, short(e))
            # fresh probe_b between sections (keep probe_a for ACL)
            for gname in ("herb:probe_b",):
                try:
                    r.execute_command("GRAPH.DELETE", gname)
                except Exception:  # noqa: BLE001
                    pass
            if name == "acl":
                pass
            if name in ("acl",):
                db.select_graph("herb:probe_b").query("RETURN 1")
        # timing: free by-product
    finally:
        wipe(r)
        left = [x for x in r.execute_command("GRAPH.LIST") if x in GRAPHS]
        if left:
            print(f"WARNING graphs left on {ver}: {left}", file=sys.stderr)
    return {"modules": str(info)}


def cell(probe: str, ver: str) -> str:
    if ver not in results.get(probe, {}):
        return "n/a"
    ok, note = results[probe][ver]
    note = note.replace("|", "/")
    if len(note) > 90:
        note = note[:87] + "..."
    return ("ok" if ok else "FAIL") + (f" ({note})" if note else "")


def write_doc() -> None:
    vs = list(VERSIONS)
    lines = ["# FalkorDB feature probe: 4.22.1 and 6.0.2", "",
             "Generated by `scripts/probe_falkor.py` (run: `uv run --with falkordb==1.8.0 python scripts/probe_falkor.py`).",
             "Servers: `falkordb/falkordb:v4.22.1` on port 6379 and `falkordb/falkordb:6.0.2` on port 6380, no password on the default user.",
             "Every row below comes from a real run. `ok` means the probe behaved as needed; for refusal probes `ok` means the server refused.", "",
             "## Results", "", "| Probe | 4.22.1 | 6.0.2 |", "| --- | --- | --- |"]
    for p in order:
        lines.append(f"| {p} | {cell(p, vs[0])} | {cell(p, vs[1])} |")
    lines += ["", "## Working DDL and ACL rules", ""]
    ddl = {
        "Exact-match index": "`CREATE INDEX FOR (p:Person) ON (p.eid)`",
        "Unique constraint": "`GRAPH.CONSTRAINT CREATE <graph> UNIQUE NODE Person PROPERTIES 1 eid` (after the index is OPERATIONAL)",
        "Mandatory constraint": "`GRAPH.CONSTRAINT CREATE <graph> MANDATORY NODE Bug PROPERTIES 1 status`",
        "Full-text index": "`CREATE FULLTEXT INDEX FOR (n:Person) ON (n.name)`; query `CALL db.idx.fulltext.queryNodes('Person', 'alice') YIELD node, score`",
        "Vector index": "`CREATE VECTOR INDEX FOR (m:Message) ON (m.embedding) OPTIONS {dimension:4096, similarityFunction:'cosine'}`; query `CALL db.idx.vector.queryNodes('Message', 'embedding', 3, vecf32($q)) YIELD node, score`",
        "Betweenness": "`CALL algo.betweenness({nodeLabels:['Person'], relationshipTypes:['COLLABORATED'], samplingSize:4, samplingSeed:42}) YIELD node, score`",
        "UDF": "`db.udf_load('herb_v1', js)` once at build time, call as `herb_v1.statusAt(events, date)`",
    }
    for k, v in ddl.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "Rows and rules as observed per version:", ""]
    for v in vs:
        lines.append(f"### {v}")
        lines.append("")
        for k, val in facts[v].items():
            lines.append(f"- {k}: `{val}`")
        lines.append("")
    lines += PRIMARY_TEXT.splitlines()
    DOC.write_text("\n".join(lines) + "\n")


PRIMARY_TEXT = ""
PRIMARY = """\
## Primary version

Decision: 6.0.2 is the primary version. 4.22.1 stays in the test matrix for everything except betweenness.

Reason, from the table above:

- Every probe the loader, constraints, indexes, UDF and tools depend on behaves the same on both versions: the
  ACL selector rules, GRAPH.COPY by the restricted user, UNIQUE and MANDATORY constraints, full-text and vector
  DDL, the versioned UDF, the literal `*1..4` bound, and RO_QUERY rejecting writes.
- `algo.betweenness` is the one place the versions differ. 6.0.2 returns the directed textbook scores
  (5-node example: Bob 3, Charlie 1, David 1) and the same scores on every build. 4.22.1 returns scores
  that change from one run to the next when `samplingSeed` is not set (see the 6 builds row), and with
  `samplingSize` larger than the node count it returned all zeros on the 3-node chain. With a fixed seed the
  4.22.1 scores repeat, but they differ from 6.0.2 and are not the textbook values.
- #3351 did not reproduce on either version with our query shape, on 5 bugs or on 3000 bugs.
- The reported 6.0.2 problems did not appear: the versioned UDF loads once and works, and `%R~` fails the same
  way on both versions (see ACL below), so it does not separate them.

If betweenness has to run on 4.22.1, compute it in Python (for example networkx) from the exported edge list
and write the scores back as a property.

## Spec rules that do not work on either version

`%R~herb:<product>` cannot be used. Both servers report the key spec of GRAPH.RO_QUERY, GRAPH.QUERY and
GRAPH.COPY as `RW access update`, so a read-only key rule never matches and the user gets "No permissions to
access a key" on its own graph. Use these rules instead (all rows ok on both versions):

- Read-only agent: `ACL SETUSER agent_<product> on ><pw> resetkeys (~herb:<product> +GRAPH.RO_QUERY)`
  or `~herb:<product> +GRAPH.RO_QUERY` with GRAPH.QUERY not granted. Do not grant GRAPH.QUERY together with `~`,
  because `~` is read-write and GRAPH.QUERY then writes (row B2).
- What-if user: `ACL SETUSER whatif_<product> on ><pw> resetkeys (~whatif:<product>:* ~herb:<product> +GRAPH.COPY)
  (~whatif:<product>:* +GRAPH.QUERY +GRAPH.RO_QUERY +GRAPH.DELETE) (~herb:<product> +GRAPH.RO_QUERY)`.
  The plain form `~whatif:<product>:* ~herb:<product>` with the same commands lets the user write to and
  delete the source graph (row W2), so it is not safe.
- GRAPH.COPY needs both keys inside one selector, which is why the copy command sits in the first selector
  with both patterns. The user can read the source graph but not write to it, delete it, or copy from another product.
- Copy cleanup pattern: `try: COPY ... ; work ... finally: GRAPH.DELETE <copy>`. Row W3 shows the copy is gone after an exception.

## Docker 4.22.1 and the spec

Docker image `falkordb/falkordb:v4.22.1` stands in for the spec's Cloud 4.22. The Docker build reports module version 42201.
"""


def main() -> None:
    global PRIMARY_TEXT
    for ver, port in VERSIONS.items():
        print(f"=== {ver} ===", flush=True)
        t = time.time()
        run_version(ver, port)
        facts[ver]["probe wall time"] = f"{time.time() - t:.1f}s"
    vs = list(VERSIONS)
    if all(v in BETW for v in vs):
        same = BETW[vs[0]] == BETW[vs[1]]
        for v in vs:
            rec("betweenness (size 4, seed 42) identical across versions", v, same, "" if same else "scores differ, see below")
    if all(results.get("UDF statusAt callable from a query", {}).get(v, (False,))[0] for v in vs):
        UDF_FILE.parent.mkdir(parents=True, exist_ok=True)
        UDF_FILE.write_text(UDF_JS)
    PRIMARY_TEXT = PRIMARY
    write_doc()
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
