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
LIB_EVIL = "herb_probe_evil"  # loaded only if an ACL check wrongly lets it through
LIBS = [LIB, LIB_EVIL]
GRAPH_PREFIXES = ("herb:probe_", "whatif:probe_")
EXPECTED_MODULE_VERSION = {"4.22.1": 42201, "6.0.2": 60002}

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


def matches(e: BaseException, pats: tuple[str, ...]) -> bool:
    text = str(e).lower()
    return any(p in text for p in pats)


def attempt(probe: str, ver: str, fn, refuse: tuple[str, ...] | None = None) -> bool:
    """Run fn.

    Without `refuse`: ok if fn works. With `refuse`: ok only if fn raises an error whose text contains one of the
    given (lowercase) fragments; any other error is a FAIL that shows its text, and success is a FAIL.
    """
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001
        if refuse is None:
            rec(probe, ver, False, short(e))
        elif matches(e, refuse):
            rec(probe, ver, True, "refused: " + short(e))
        else:
            rec(probe, ver, False, "unexpected error, not the expected refusal: " + short(e))
        return False
    if refuse is not None:
        rec(probe, ver, False, "NOT refused" + (f" ({out})" if isinstance(out, (str, int, float, list)) else ""))
        return False
    rec(probe, ver, True, str(out) if isinstance(out, (str, int, float, list)) else "")
    return True


ABSENT = ("empty key", "does not exist", "not found", "no such", "not registered", "doesn't exist", "unknown library")


def wipe(r: redis.Redis) -> list[str]:
    """Remove every probe graph, user and UDF library. Returns real errors; already-absent is not an error."""
    errs: list[str] = []

    def run(what: str, *cmd) -> None:
        try:
            r.execute_command(*cmd)
        except Exception as e:  # noqa: BLE001
            if not matches(e, ABSENT):
                errs.append(f"{what}: {short(e)}")

    for g in list_graphs(r):
        if g.startswith(GRAPH_PREFIXES):
            run(f"delete graph {g}", "GRAPH.DELETE", g)
    for u in r.execute_command("ACL", "USERS"):
        if u != "default" and "probe" in u:
            run(f"delete user {u}", "ACL", "DELUSER", u)
    for lib in LIBS:
        run(f"delete UDF library {lib}", "GRAPH.UDF", "DELETE", lib)
    for k in ("probe:other", "herb:probe_z"):
        run(f"delete key {k}", "DEL", k)
    return errs


def leftovers(r: redis.Redis) -> list[str]:
    left = [f"graph {g}" for g in list_graphs(r) if g.startswith(GRAPH_PREFIXES)]
    left += [f"user {u}" for u in r.execute_command("ACL", "USERS") if u != "default" and "probe" in u]
    udf = str(r.execute_command("GRAPH.UDF", "LIST"))
    left += [f"UDF library {lib}" for lib in LIBS if lib in udf]
    left += [f"key {k}" for k in ("probe:other", "herb:probe_z") if r.exists(k)]
    return left


def drop(r: redis.Redis, name: str) -> None:
    try:
        r.execute_command("GRAPH.DELETE", name)
    except Exception as e:  # noqa: BLE001
        if not matches(e, ABSENT):
            raise


def user_conn(port: int, user: str) -> redis.Redis:
    return redis.Redis(port=port, username=user, password=PW, decode_responses=True)


def seed_a(db) -> None:
    r = db.connection
    for name in ("herb:probe_a", "herb:probe_b"):
        drop(r, name)
    g = db.select_graph("herb:probe_a")
    g.query("CREATE (:Person {eid:'e1', name:'Ann'}), (:Person {eid:'e2', name:'Bo'})")
    db.select_graph("herb:probe_b").query("CREATE (:Person {eid:'x1', name:'Xi'})")


# ---------------------------------------------------------------- ACL
PERM = ("permission",)  # NOPERM family: "No permissions to access a key", "has no permissions to run ..."
RO_ERR = ("read-only queries",)
EXISTS_OR_PERM = ("exist", "permission")


def agent_sel(p: str) -> list[str]:
    return [f"(~herb:{p} +GRAPH.RO_QUERY)"]


def whatif_sel(p: str) -> list[str]:
    return [f"(~whatif:{p}:* ~herb:{p} +GRAPH.COPY)",
            f"(~whatif:{p}:* +GRAPH.QUERY +GRAPH.RO_QUERY +GRAPH.DELETE)",
            f"(~herb:{p} +GRAPH.RO_QUERY)"]


def provision(r: redis.Redis, user: str, rules: list[str]) -> None:
    """Published provisioning: reset first, so earlier grants cannot survive a re-run."""
    r.execute_command("ACL", "SETUSER", user, "reset", "on", f">{PW}", *rules)


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
AGENT_SELECTOR = ("agent_probe_a_v5", "ACL selector (~herb:probe_a +GRAPH.RO_QUERY)", agent_sel("probe_a"))
WHATIF_VARIANTS = {
    "W": ("whatif_probe_a", "spec rule ~whatif:probe_a:* %R~herb:probe_a",
          ["~whatif:probe_a:*", "%R~herb:probe_a", "+GRAPH.COPY", "+GRAPH.QUERY", "+GRAPH.RO_QUERY", "+GRAPH.DELETE"]),
    "W2": ("whatif_probe_a_v2", "plain keys ~whatif:probe_a:* ~herb:probe_a",
           ["~whatif:probe_a:*", "~herb:probe_a", "+GRAPH.COPY", "+GRAPH.QUERY", "+GRAPH.RO_QUERY", "+GRAPH.DELETE"]),
    "W3": ("whatif_probe_a_v3", "ACL selectors (per-key command sets)", whatif_sel("probe_a")),
}

AGENT_MATRIX = [
    ("GRAPH.QUERY read", ["GRAPH.QUERY", "herb:probe_a", "MATCH (n) RETURN count(n)"]),
    ("GRAPH.QUERY write", ["GRAPH.QUERY", "herb:probe_a", "CREATE (:Person {eid:'zz'})"]),
    ("GRAPH.DELETE own graph", ["GRAPH.DELETE", "herb:probe_a"]),
    ("GRAPH.COPY own graph to any key", ["GRAPH.COPY", "herb:probe_a", "whatif:probe_a:9"]),
    ("GRAPH.CONSTRAINT CREATE", ["GRAPH.CONSTRAINT", "CREATE", "herb:probe_a", "MANDATORY", "NODE", "Person",
                                 "PROPERTIES", "1", "eid"]),
    ("GRAPH.UDF LOAD", ["GRAPH.UDF", "LOAD", LIB_EVIL, "falkor.register('x', function(){return 1})"]),
    ("SET herb:probe_a", ["SET", "herb:probe_a", "x"]),
    ("DEL herb:probe_a", ["DEL", "herb:probe_a"]),
    ("SET another key", ["SET", "probe:other", "x"]),
    ("DEL another key", ["DEL", "probe:other"]),
    ("RENAME herb:probe_a", ["RENAME", "herb:probe_a", "herb:probe_z"]),
    ("MULTI", ["MULTI"]),
    ("EXEC", ["EXEC"]),
    ("EVAL", ["EVAL", "return 1", "0"]),
    ("FUNCTION LIST", ["FUNCTION", "LIST"]),
    ("CONFIG GET", ["CONFIG", "GET", "maxmemory"]),
    ("ACL LIST", ["ACL", "LIST"]),
]


def list_graphs(r: redis.Redis) -> list[str]:
    return list(r.execute_command("GRAPH.LIST"))


def snapshot(r: redis.Redis) -> dict:
    """Contents of the protected graphs, read as admin."""
    out: dict = {}
    for g in ("herb:probe_a", "herb:probe_b"):
        out[g] = r.execute_command("GRAPH.RO_QUERY", g, "MATCH (n) RETURN n.eid ORDER BY n.eid")[1]
    out["herb graphs"] = sorted(x for x in list_graphs(r) if x.startswith("herb:"))
    out["probe:other"] = r.exists("probe:other")
    return out


def check_unchanged(probe: str, ver: str, r: redis.Redis, before: dict) -> None:
    try:
        after = snapshot(r)
        rec(probe, ver, after == before, "" if after == before else f"before {before} after {after}")
    except Exception as e:  # noqa: BLE001
        rec(probe, ver, False, "snapshot failed: " + short(e))


def agent_checks(ver: str, port: int, r: redis.Redis, key: str, u: str, label: str, rules: list[str],
                 full: bool = False) -> None:
    seed_a(FalkorDB(port=port))
    pre = f"ACL agent {key} ({label})"
    try:
        provision(r, u, rules)
    except Exception as e:  # noqa: BLE001
        rec(f"{pre}: SETUSER accepted", ver, False, short(e))
        return
    rec(f"{pre}: SETUSER accepted", ver, True)
    facts[ver][f"acl_{key}"] = "reset on >pw " + " ".join(rules)
    before = snapshot(r)
    c = user_conn(port, u)
    attempt(f"{pre}: RO_QUERY own graph allowed (count is 2)", ver,
            lambda: _expect(c.execute_command("GRAPH.RO_QUERY", "herb:probe_a", "MATCH (n:Person) RETURN count(n)")[1], [[2]]))
    attempt(f"{pre}: RO_QUERY herb:probe_b refused", ver,
            lambda: c.execute_command("GRAPH.RO_QUERY", "herb:probe_b", "MATCH (n:Person) RETURN count(n)"),
            refuse=PERM)
    attempt(f"{pre}: GRAPH.QUERY write on own graph refused", ver,
            lambda: c.execute_command("GRAPH.QUERY", "herb:probe_a", "CREATE (:Person {eid:'zz'})"), refuse=PERM)
    attempt(f"{pre}: GRAPH.RO_QUERY write on own graph refused", ver,
            lambda: c.execute_command("GRAPH.RO_QUERY", "herb:probe_a", "CREATE (:Person {eid:'zz'})"),
            refuse=PERM + RO_ERR)
    if full:
        for name, cmd in AGENT_MATRIX:
            attempt(f"{pre}: {name} refused", ver, lambda cmd=cmd: c.execute_command(*cmd), refuse=PERM)
    c.close()
    check_unchanged(f"{pre}: protected graphs unchanged after the refusals", ver, r, before)


def _expect(got, want):
    if got != want:
        raise AssertionError(f"got {got}, expected {want}")
    return got


def whatif_checks(ver: str, port: int, r: redis.Redis, key: str, u: str, label: str, rules: list[str]) -> None:
    seed_a(FalkorDB(port=port))
    pre = f"ACL whatif {key} ({label})"
    try:
        provision(r, u, rules)
    except Exception as e:  # noqa: BLE001
        rec(f"{pre}: SETUSER accepted", ver, False, short(e))
        return
    rec(f"{pre}: SETUSER accepted", ver, True)
    facts[ver][f"acl_{key}"] = "reset on >pw " + " ".join(rules)
    before = snapshot(r)
    c = user_conn(port, u)
    cp = "whatif:probe_a:1"
    attempt(f"{pre}: RO_QUERY own herb graph allowed (count is 2)", ver,
            lambda: _expect(c.execute_command("GRAPH.RO_QUERY", "herb:probe_a", "MATCH (n:Person) RETURN count(n)")[1], [[2]]))
    attempt(f"{pre}: RO_QUERY herb:probe_b refused", ver,
            lambda: c.execute_command("GRAPH.RO_QUERY", "herb:probe_b", "MATCH (n:Person) RETURN count(n)"),
            refuse=PERM)
    copied = attempt(f"{pre}: GRAPH.COPY herb:probe_a to whatif copy", ver,
                     lambda: c.execute_command("GRAPH.COPY", "herb:probe_a", cp))
    skip = "skipped: copy failed"
    if copied:
        attempt(f"{pre}: copy has the source rows (count is 2)", ver,
                lambda: _expect(c.execute_command("GRAPH.RO_QUERY", cp, "MATCH (n:Person) RETURN count(n)")[1], [[2]]))
        attempt(f"{pre}: write to the copy works", ver,
                lambda: c.execute_command("GRAPH.QUERY", cp, "CREATE (:Person {eid:'w1'})")[0][0])
        attempt(f"{pre}: copy now has 3 rows", ver,
                lambda: _expect(c.execute_command("GRAPH.RO_QUERY", cp, "MATCH (n:Person) RETURN count(n)")[1], [[3]]))
    else:
        for n in ("copy has the source rows (count is 2)", "write to the copy works", "copy now has 3 rows"):
            rec(f"{pre}: {n}", ver, False, skip)
    attempt(f"{pre}: GRAPH.QUERY write to herb:probe_a refused", ver,
            lambda: c.execute_command("GRAPH.QUERY", "herb:probe_a", "CREATE (:Person {eid:'zz'})"), refuse=PERM)
    attempt(f"{pre}: GRAPH.COPY herb:probe_a to herb:probe_a refused", ver,
            lambda: c.execute_command("GRAPH.COPY", "herb:probe_a", "herb:probe_a"), refuse=EXISTS_OR_PERM)
    attempt(f"{pre}: GRAPH.COPY herb:probe_a to another product's whatif:probe_b:1 refused", ver,
            lambda: c.execute_command("GRAPH.COPY", "herb:probe_a", "whatif:probe_b:1"), refuse=PERM)
    attempt(f"{pre}: GRAPH.COPY herb:probe_b to whatif:probe_a:2 refused", ver,
            lambda: c.execute_command("GRAPH.COPY", "herb:probe_b", "whatif:probe_a:2"), refuse=PERM)
    if copied:
        attempt(f"{pre}: GRAPH.COPY whatif copy to existing herb:probe_a refused (destination exists)", ver,
                lambda: c.execute_command("GRAPH.COPY", cp, "herb:probe_a"), refuse=EXISTS_OR_PERM)
        attempt(f"{pre}: GRAPH.COPY whatif copy to herb:probe_b refused", ver,
                lambda: c.execute_command("GRAPH.COPY", cp, "herb:probe_b"), refuse=PERM)
    else:
        rec(f"{pre}: GRAPH.COPY whatif copy to existing herb:probe_a refused (destination exists)", ver, False, skip)
        rec(f"{pre}: GRAPH.COPY whatif copy to herb:probe_b refused", ver, False, skip)
    attempt(f"{pre}: GRAPH.DELETE of herb:probe_a refused", ver,
            lambda: c.execute_command("GRAPH.DELETE", "herb:probe_a"), refuse=PERM)
    if copied:
        # herb contents are compared while the copy still exists (snapshot ignores whatif graphs)
        check_unchanged(f"{pre}: herb graphs unchanged after the refusals", ver, r, before)
        attempt(f"{pre}: GRAPH.DELETE of the copy works", ver, lambda: c.execute_command("GRAPH.DELETE", cp))
    else:
        check_unchanged(f"{pre}: herb graphs unchanged after the refusals", ver, r, before)
        rec(f"{pre}: GRAPH.DELETE of the copy works", ver, False, skip)

    # cleanup-on-exception pattern
    cp3 = "whatif:probe_a:3"
    made = False
    try:
        try:
            c.execute_command("GRAPH.COPY", "herb:probe_a", cp3)
            made = True
            raise RuntimeError("simulated failure mid-way")
        finally:
            if made:
                c.execute_command("GRAPH.DELETE", cp3)
    except RuntimeError:
        pass
    except Exception:  # noqa: BLE001  (copy or delete refused)
        pass
    left = cp3 in list_graphs(r)
    probe = f"{pre}: try/finally deletes copy after exception"
    if made:
        rec(probe, ver, not left, "copy left behind" if left else "")
    else:
        rec(probe, ver, False, "skipped: GRAPH.COPY refused")
    c.close()
    for g in list_graphs(r):  # admin sweep between variants, not part of the probe result
        if g.startswith("whatif:probe_a:"):
            r.execute_command("GRAPH.DELETE", g)


def _pairs(x):
    return x if isinstance(x, dict) else {x[i]: x[i + 1] for i in range(0, len(x), 2)}


def getuser(r: redis.Redis, user: str) -> dict:
    r.response_callbacks.pop("ACL GETUSER", None)
    info = dict(_pairs(r.execute_command("ACL", "GETUSER", user)))
    info["selectors"] = [dict(_pairs(x)) for x in info.get("selectors", [])]
    return info


EXPECTED_SELECTORS = {
    "agent": {("-@all +graph.ro_query", "~herb:probe_a")},
    "whatif": {("-@all +graph.copy", "~whatif:probe_a:* ~herb:probe_a"),
               ("-@all +graph.query +graph.ro_query +graph.delete", "~whatif:probe_a:*"),
               ("-@all +graph.ro_query", "~herb:probe_a")},
}


def probe_reprovision(ver: str, port: int, r: redis.Redis) -> None:
    """Re-applying the published commands to an over-privileged existing user leaves exactly the intended grants."""
    for kind, u, sel, nsel in (("agent", "overpriv_probe_a", agent_sel("probe_a"), 1),
                               ("whatif", "overpriv_probe_b", whatif_sel("probe_a"), 3)):
        seed_a(FalkorDB(port=port))
        pre = f"re-provision over-privileged user as {kind}"
        r.execute_command("ACL", "SETUSER", u, "reset", "on", f">{PW}", "+@all", "~*", "(~* +GRAPH.QUERY)")
        c = user_conn(port, u)
        attempt(f"{pre}: precondition, user could write herb:probe_b", ver,
                lambda: c.execute_command("GRAPH.QUERY", "herb:probe_b", "CREATE (:Person {eid:'pre'})")[0][0])
        c.close()
        seed_a(FalkorDB(port=port))
        before = snapshot(r)
        provision(r, u, sel)
        info = getuser(r, u)
        facts[ver][f"ACL GETUSER after re-provision ({kind})"] = (
            f"flags={info.get('flags')} commands={info.get('commands')!r} keys={info.get('keys')!r} "
            f"selectors={info.get('selectors')}")
        sels = info.get("selectors", [])
        got_sels = {(x.get("commands"), x.get("keys")) for x in sels}
        exact = (info.get("commands") == "-@all" and info.get("keys") == "" and len(sels) == nsel
                 and got_sels == EXPECTED_SELECTORS[kind])
        rec(f"{pre}: ACL GETUSER shows only the intended grants", ver, exact,
            f"commands={info.get('commands')!r} keys={info.get('keys')!r} selectors={sorted(got_sels)}")
        c = user_conn(port, u)
        attempt(f"{pre}: GRAPH.QUERY write on herb:probe_b now refused", ver,
                lambda: c.execute_command("GRAPH.QUERY", "herb:probe_b", "CREATE (:Person {eid:'post'})"), refuse=PERM)
        attempt(f"{pre}: GRAPH.QUERY write on herb:probe_a now refused", ver,
                lambda: c.execute_command("GRAPH.QUERY", "herb:probe_a", "CREATE (:Person {eid:'post'})"), refuse=PERM)
        attempt(f"{pre}: SET another key now refused", ver, lambda: c.execute_command("SET", "probe:other", "x"),
                refuse=PERM)
        attempt(f"{pre}: RO_QUERY own graph still allowed (count is 2)", ver,
                lambda: _expect(c.execute_command("GRAPH.RO_QUERY", "herb:probe_a", "MATCH (n:Person) RETURN count(n)")[1], [[2]]))
        c.close()
        check_unchanged(f"{pre}: protected graphs unchanged", ver, r, before)


def probe_acl(ver: str, port: int, r: redis.Redis, db) -> None:
    for key, (u, label, rules) in AGENT_VARIANTS.items():
        agent_checks(ver, port, r, key, u, label, rules)
    agent_checks(ver, port, r, "S", *AGENT_SELECTOR, full=True)
    for key, (u, label, rules) in WHATIF_VARIANTS.items():
        whatif_checks(ver, port, r, key, u, label, rules)
    probe_reprovision(ver, port, r)
    # why %R~ fails: the static key spec of the graph commands
    r.response_callbacks.pop("COMMAND", None)
    for cmd in (("GRAPH.RO_QUERY", "g", "RETURN 1"), ("GRAPH.QUERY", "g", "RETURN 1"), ("GRAPH.COPY", "a", "b")):
        try:
            facts[ver][f"key spec {cmd[0]}"] = str(r.execute_command("COMMAND", "GETKEYSANDFLAGS", *cmd))
        except Exception as e:  # noqa: BLE001
            facts[ver][f"key spec {cmd[0]}"] = short(e)


# ---------------------------------------------------------------- UDF
def lit(events: list[tuple[str, object]]) -> str:
    return "[" + ", ".join(f"{{type:'{t}', ts:{ts if isinstance(ts, int) else repr(ts)}}}" for t, ts in events) + "]"


UDF_CASES = [
    ("empty list", [], 5, None),
    ("before the first event", [("REPORTED", 5)], 4, None),
    ("exact cutoff is inclusive", [("REPORTED", 5)], 5, "active"),
    ("ASSIGNED is active", [("REPORTED", 1), ("ASSIGNED", 2)], 3, "active"),
    ("DISMISSED is dismissed", [("REPORTED", 1), ("DISMISSED", 2)], 3, "dismissed"),
    ("REOPENED after DISMISSED is active", [("REPORTED", 1), ("DISMISSED", 2), ("REOPENED", 3)], 9, "active"),
    ("RESOLVED is resolved", [("REPORTED", 1), ("RESOLVED", 2)], 9, "resolved"),
    ("event after the cutoff is ignored", [("REPORTED", 1), ("RESOLVED", 8)], 5, "active"),
    ("equal timestamps, later list entry wins (RESOLVED last)", [("REPORTED", 5), ("RESOLVED", 5)], 5, "resolved"),
    ("equal timestamps, later list entry wins (REPORTED last)", [("RESOLVED", 5), ("REPORTED", 5)], 5, "active"),
    ("unsorted input, cutoff between events", [("RESOLVED", 3), ("REPORTED", 1), ("REOPENED", 5)], 4, "resolved"),
    ("unsorted input, cutoff after all events", [("RESOLVED", 3), ("REPORTED", 1), ("REOPENED", 5)], 6, "active"),
    ("ISO date strings", [("REPORTED", "2024-01-01"), ("RESOLVED", "2024-02-01")], "2024-01-15", "active"),
]


def probe_udf(ver: str, db) -> None:
    g = db.select_graph("herb:probe_b")
    g.query("RETURN 1")
    ok = attempt("UDF load herb_v1 (no REPLACE)", ver, lambda: db.udf_load(LIB, UDF_JS))
    if not ok:
        for name, *_ in UDF_CASES:
            rec(f"UDF statusAt: {name}", ver, False, "skipped: library did not load")
        return
    for name, events, date, want in UDF_CASES:
        def call(events=events, date=date, want=want):
            got = g.query(f"RETURN herb_v1.statusAt({lit(events)}, $d)", {"d": date}).result_set[0][0]
            if got != want:
                raise AssertionError(f"got {got!r}, expected {want!r}")
            return repr(got)
        attempt(f"UDF statusAt: {name}", ver, call)

    def from_graph():
        g.query("CREATE (b:Bug {id:'b1'}), (:Event {type:'REPORTED', ts:'2024-01-01'})-[:OF]->(b), "
                "(:Event {type:'RESOLVED', ts:'2024-02-01'})-[:OF]->(b)")
        got = g.query("MATCH (e:Event)-[:OF]->(b:Bug) WITH b, collect({type:e.type, ts:e.ts}) AS evs "
                      "RETURN herb_v1.statusAt(evs, $d)", {"d": "2024-06-01"}).result_set[0][0]
        if got != "resolved":
            raise AssertionError(f"got {got!r}")
        return got

    attempt("UDF statusAt on collect() of Event nodes (maps cannot be stored as properties)", ver, from_graph)
    attempt("UDF second load of same name refused (no REPLACE)", ver, lambda: db.udf_load(LIB, UDF_JS),
            refuse=("already",))


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
                                      "PROPERTIES", "1", "eid"), refuse=("exact-match index",))
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
    attempt("duplicate Person.eid insert fails", ver, lambda: g.query("CREATE (:Person {eid:'p1'})"),
            refuse=("unique constraint violation",))
    attempt("distinct Person.eid insert works", ver, lambda: g.query("CREATE (:Person {eid:'p2'})"))
    attempt("MANDATORY Bug.status create", ver,
            lambda: r.execute_command("GRAPH.CONSTRAINT", "CREATE", "herb:probe_b", "MANDATORY", "NODE", "Bug",
                                      "PROPERTIES", "1", "status"))
    got = wait_for(all_op)
    rec("MANDATORY constraint reaches OPERATIONAL (db.constraints)", ver, bool(got),
        "" if got else f"rows={status()}")
    facts[ver]["db.constraints rows (both)"] = repr(status())[:400]
    attempt("Bug without status fails", ver, lambda: g.query("CREATE (:Bug {id:'b1'})"),
            refuse=("mandatory constraint violation",))
    attempt("Bug with status works", ver, lambda: g.query("CREATE (:Bug {id:'b2', status:'open'})"))


# ---------------------------------------------------------------- indexes
def probe_indexes(ver: str, db) -> None:
    g = db.select_graph("herb:probe_b")
    g.query("CREATE (:Person {name:'alpha one'}), (:Person {name:'beta two'}), "
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

    attempt("full-text query Person.name 'alpha'", ver, lambda: ft("Person", "alpha"))
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
            lambda: g.query("CREATE (:Message {id:99, embedding: vecf32([1.0, 2.0])})"),
            refuse=("dimension", "length", "vector"))


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
    g.query("CREATE (a:Person {eid:'v1'}), (b:Person {eid:'v2'}), (c:Person {eid:'v3'}), "
            "(d:Person {eid:'v4'}), (e:Person {eid:'v5'}), (a)-[:COLLABORATED]->(b), "
            "(b)-[:COLLABORATED]->(c), (b)-[:COLLABORATED]->(d), (c)-[:COLLABORATED]->(e), (d)-[:COLLABORATED]->(e)")
    got = chain("")
    rec("betweenness 5-node fixture, directed pairs counted once: v2 3, v3 1, v4 1", ver,
        (got.get("v2"), got.get("v3"), got.get("v4")) == (3.0, 1.0, 1.0), f"scores {got}")
    create5 = ("CREATE (a:Person {eid:'v1'}), (b:Person {eid:'v2'}), (c:Person {eid:'v3'}), "
               "(d:Person {eid:'v4'}), (e:Person {eid:'v5'}), (a)-[:COLLABORATED]->(b), "
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
            lambda: r.execute_command("GRAPH.RO_QUERY", "herb:probe_b", "CREATE (:Z)"), refuse=RO_ERR)
    drop(r, "herb:probe_b")

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
CLEANUP_PROBLEMS: list[str] = []


def run_version(ver: str, port: int) -> None:
    r = redis.Redis(port=port, decode_responses=True)
    db = FalkorDB(port=port)
    try:
        mods = r.execute_command("MODULE", "LIST")
        graph_mod = next((m for m in mods if m.get("name") == "graph"), {})
        got = graph_mod.get("ver")
        facts[ver]["graph module version (MODULE LIST)"] = str(got)
        facts[ver]["server version (INFO)"] = str(r.info("server").get("redis_version"))
        rec("server reports the expected FalkorDB version", ver, got == EXPECTED_MODULE_VERSION[ver],
            f"module ver {got}, expected {EXPECTED_MODULE_VERSION[ver]}")
        pre = wipe(r)
        if pre:
            CLEANUP_PROBLEMS.extend(f"{ver} pre-run: {x}" for x in pre)
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
            try:
                drop(r, "herb:probe_b")
            except Exception as e:  # noqa: BLE001
                rec(f"section {name} between-section cleanup", ver, False, short(e))
    finally:
        errs = wipe(r)
        left = leftovers(r)
        problems = errs + [f"left behind: {x}" for x in left]
        CLEANUP_PROBLEMS.extend(f"{ver}: {x}" for x in problems)
        rec("cleanup: no probe graph, user, UDF library or key left", ver, not problems,
            "; ".join(problems)[:200] if problems else "")


def cell(probe: str, ver: str) -> str:
    if ver not in results.get(probe, {}):
        return "FAIL (not run on this server)"
    ok, note = results[probe][ver]
    note = note.replace("|", "/")
    if len(note) > 90:
        note = note[:87] + "..."
    return ("ok" if ok else "FAIL") + (f" ({note})" if note else "")


# Rows that document a variant we do not use, a known bug, or a comparison: a FAIL there is the finding, not a defect.
INFORMATIONAL = ("ACL agent A (", "ACL agent A2 (", "ACL agent B (", "ACL agent B2 (", "ACL whatif W (",
                 "ACL whatif W2 (", "#3350 parameter", "vector wrong dimension", "betweenness", "3351 workaround")


def required(p: str) -> bool:
    return not p.startswith(INFORMATIONAL)


def passed(p: str, v: str) -> bool:
    return results.get(p, {}).get(v, (False, ""))[0]


def build_primary(vs: list[str]) -> str:
    core = {v: [p for p in order if required(p) and not passed(p, v)] for v in vs}
    betw_rows = [p for p in order if p.startswith(("betweenness chain", "betweenness 5-node"))]
    betw_ok = {v: all(passed(p, v) for p in betw_rows) for v in vs}
    eligible = [v for v in vs if not core[v]]
    L = ["## Primary version", ""]
    if not eligible:
        L += ["No primary version can be named: required probes failed on both servers.", ""]
        for v in vs:
            L += [f"- {v}: " + "; ".join(core[v][:6]) + (" ..." if len(core[v]) > 6 else "")]
        primary = None
    else:
        prefer = [v for v in eligible if betw_ok[v]] or eligible
        primary = prefer[0]
        other = [v for v in vs if v != primary]
        L += [f"Decision: {primary} is the primary version.", "", "Reason, from the table above:", ""]
        for v in vs:
            if core[v]:
                L.append(f"- {v}: {len(core[v])} required row(s) failed: " + "; ".join(core[v][:4]) + ".")
            else:
                L.append(f"- {v}: every required row passed (ACL, UDF, constraints, indexes, #3351 shapes, RO_QUERY).")
        n_pass = {v: sum(passed(p, v) for p in betw_rows) for v in vs}
        L.append("- `algo.betweenness`, on the tested fixtures only (3-node directed chain, 5-node directed fixture; "
                 "the fixture rows are listed under betweenness above): " + ", ".join(f"{v} passed {n_pass[v]} of {len(betw_rows)} fixture rows"
                                                         for v in vs) +
                 ". The expected values are the directed shortest-path counts worked out by hand for those fixtures. "
                 "No claim is made beyond them.")
        if not betw_ok[primary]:
            L.append(f"- Betweenness failed fixture rows on {primary}; compute it in Python instead.")
        elif not all(betw_ok.values()):
            bad = [v for v in vs if not betw_ok[v]]
            L.append(f"- On {', '.join(bad)} compute betweenness in Python from the exported edge list, "
                     "or do not rely on `algo.betweenness`.")
        if other and not core[other[0]]:
            L.append(f"- {other[0]} stays in the test matrix.")
    L += ["",
          "## ACL rules (identical on both versions)",
          "",
          "`%R~herb:<product>` cannot be used. Both servers report the key spec of GRAPH.RO_QUERY, GRAPH.QUERY and",
          "GRAPH.COPY as `RW access update`, so a read-only key rule never matches and the user gets \"No permissions to",
          "access a key\" on its own graph.",
          "",
          "Provisioning always starts with `reset`, so a re-run removes every earlier grant (rows `re-provision",
          "over-privileged user`). Exact commands:",
          "",
          "```",
          "ACL SETUSER agent_<p> reset on ><pw> (~herb:<p> +GRAPH.RO_QUERY)",
          "ACL SETUSER whatif_<p> reset on ><pw> (~whatif:<p>:* ~herb:<p> +GRAPH.COPY) "
          "(~whatif:<p>:* +GRAPH.QUERY +GRAPH.RO_QUERY +GRAPH.DELETE) (~herb:<p> +GRAPH.RO_QUERY)",
          "```",
          "",
          "Agent: reads its own graph only. The rows `ACL agent S` show refusal of GRAPH.QUERY, GRAPH.DELETE, GRAPH.COPY,",
          "GRAPH.CONSTRAINT, GRAPH.UDF, SET, DEL, RENAME, MULTI, EXEC, EVAL, FUNCTION, CONFIG and ACL, each on the",
          "permission error, with the protected graphs unchanged afterwards. Do not combine `~herb:<p>` with",
          "+GRAPH.QUERY: `~` is read-write (row B2).",
          "",
          "What-if user, exact guarantee. Precondition: `herb:<p>` must exist before the user is created (the loader",
          "guarantees this). Given that, the user can read `herb:<p>`, copy it into `whatif:<p>:*`, write to and delete",
          "those copies, and nothing else. Writes to `herb:<p>` and GRAPH.DELETE of it are refused on the permission",
          "error; copying into `herb:<p>` (from itself or from a copy) is refused because the destination exists;",
          "copies to another product's `whatif:` namespace, and copies from another product's `herb:` graph, are",
          "refused; the `herb:` graph contents are unchanged afterwards (rows `ACL whatif W3`).",
          "",
          "Residual risk, accepted: the GRAPH.COPY selector grants both keys without a source or destination role, so",
          "the rule cannot stop `GRAPH.COPY whatif:<p>:n herb:<p>` if `herb:<p>` does not exist at that moment. Only the",
          "destination-exists check protects it, so an administrator must not delete `herb:<p>` while the user is live.",
          "",
          "Sponsor feedback for FalkorDB: GRAPH.COPY (and GRAPH.RO_QUERY) declare `RW` on every key, so a read-only",
          "source cannot be expressed in ACL, and `%R~` rules from the ACL documentation do not work for graph commands.",
          "Copy cleanup pattern: `try: COPY ...; work ... finally: GRAPH.DELETE <copy>` (row `try/finally`).",
          "",
          "## Docker 4.22.1 and the spec",
          "",
          "Docker image `falkordb/falkordb:v4.22.1` stands in for the spec's Cloud 4.22."]
    return "\n".join(L) + "\n"


def write_doc() -> None:
    vs = list(VERSIONS)
    lines = ["# FalkorDB feature probe: 4.22.1 and 6.0.2", "",
             "Generated by `scripts/probe_falkor.py` (run: `uv run --with falkordb==1.8.0 --with redis python scripts/probe_falkor.py`).",
             "Servers: `falkordb/falkordb:v4.22.1` on port 6379 and `falkordb/falkordb:6.0.2` on port 6380, no password on the default user.",
             "Graph module versions reported by MODULE LIST: " + ", ".join(
                 f"{v} = {facts[v].get('graph module version (MODULE LIST)')}" for v in vs) + ".",
             "Every row below comes from a real run. `ok` means the probe behaved as needed; for refusal probes `ok` means the server refused",
             "with the expected error text (permission error for ACL rows, read-only error for RO_QUERY writes, the named constraint error for constraints).",
             "A row missing on one server is shown as FAIL.", "",
             "## Results", "", "| Probe | 4.22.1 | 6.0.2 |", "| --- | --- | --- |"]
    for p in order:
        lines.append(f"| {p} | {cell(p, vs[0])} | {cell(p, vs[1])} |")
    lines += ["", "## Working DDL", ""]
    ddl = {
        "Exact-match index": "`CREATE INDEX FOR (p:Person) ON (p.eid)`",
        "Unique constraint": "`GRAPH.CONSTRAINT CREATE <graph> UNIQUE NODE Person PROPERTIES 1 eid` (after the index is OPERATIONAL)",
        "Mandatory constraint": "`GRAPH.CONSTRAINT CREATE <graph> MANDATORY NODE Bug PROPERTIES 1 status`",
        "Full-text index": "`CREATE FULLTEXT INDEX FOR (n:Person) ON (n.name)`; query `CALL db.idx.fulltext.queryNodes('Person', 'alpha') YIELD node, score`",
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
    lines += build_primary(vs).splitlines()
    DOC.write_text("\n".join(lines) + "\n")


def main() -> int:
    for ver, port in VERSIONS.items():
        print(f"=== {ver} ===", flush=True)
        t = time.time()
        run_version(ver, port)
        facts[ver]["probe wall time"] = f"{time.time() - t:.1f}s"
    vs = list(VERSIONS)
    if all(v in BETW for v in vs):
        same = BETW[vs[0]] == BETW[vs[1]]
        for v in vs:
            rec("betweenness (size 4, seed 42) identical across versions", v, same,
                "" if same else "scores differ, see below")
    if all(passed("UDF statusAt: ISO date strings", v) and passed("UDF statusAt: empty list", v) for v in vs):
        UDF_FILE.parent.mkdir(parents=True, exist_ok=True)
        UDF_FILE.write_text(UDF_JS)
    write_doc()
    print(f"wrote {DOC}")
    if CLEANUP_PROBLEMS:
        print("CLEANUP PROBLEMS:", *CLEANUP_PROBLEMS, sep="\n  ", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
