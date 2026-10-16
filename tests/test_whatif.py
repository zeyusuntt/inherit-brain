"""Task 12: the counterfactual "if this person leaves" (spec 5.5), on both servers.

Every case runs on a copy made by the restricted whatif user; the Python check re-derives the groups from the exported
snapshot alone. The fixture is tests/whatiffx.py (not HERB data); the dev comparison runs on the loaded dev graphs.
"""

import ast
import inspect
import json
from pathlib import Path

import pytest
import redis

import whatiffx as fx
from inherit import whatif, whatif_check
from inherit.config import ROOT, load_settings
from inherit.graph import acl as acl_mod
from inherit.tools.catalog import TOOLS, ToolError
from inherit.tools.trace import Trace
from inherit.whatif_schema import WhatIf, export_snapshot

PID = f"Person:{fx.LEAVER}"
B = lambda *k: {f"Bug:{x}" for x in k}  # noqa: E731
R = lambda *n: {f"PR:{fx.pr(x)}" for x in n}  # noqa: E731
F = lambda *k: {f"Feature:{x}" for x in k}  # noqa: E731

TEST_RULE_GRAPH = ("MATCH (a:Feature)-[:TEST_NEEDS]->(b:Feature) MERGE (a)-[:DEPENDS_ON]->(b)",)
TEST_RULE_CHECK = (("TEST_NEEDS", "Feature", "Feature", {}, "src"),)


@pytest.fixture(params=["4.22", "6.0"])
def srv(request):
    db = fx.connect(request.param)
    fx.clean(db)
    yield db
    fx.clean(db)


@pytest.fixture
def wi(srv, tmp_path):
    """(admin db, whatif connection) on the fixture product."""
    fx.load(srv)
    return srv, fx.whatif_conn(srv, tmp_path)


def groups(r: WhatIf):
    return set(r.already_unowned), set(r.newly_unowned), set(r.affected)


def copies(db) -> list[str]:
    return [g for g in db.list_graphs() if g.startswith(f"whatif:{fx.P}:")]


# ---------------------------------------------------------------- the groups
def test_bug_owned_only_by_leaver_is_newly_unowned_and_shared_is_not(wi):
    _, conn = wi
    r = whatif.if_leaves(conn, fx.P, fx.LEAVER)
    assert {"Bug:bs", "Bug:bnew"} <= r.newly_unowned
    assert not ({"Bug:bsh", "Bug:bmulti"} & (r.newly_unowned | r.affected | r.already_unowned))


def test_latest_assignment_wins_and_never_falls_back_to_an_older_one(wi):
    _, conn = wi
    r = whatif.if_leaves(conn, fx.P, fx.LEAVER)
    assert "Bug:bold" not in r.newly_unowned  # re-assigned to Queen
    assert "Bug:bnew" in r.newly_unowned  # Queen first, the leaver last
    # leaver Queen: bold is hers alone, bnew is not (the leaver holds it), btie is hers by record position 10 > 2
    q = whatif.if_leaves(conn, fx.P, fx.QUEEN)
    assert {"Bug:bold", "Bug:btie", "Bug:bfix"} <= q.newly_unowned
    assert "Bug:bnew" not in q.newly_unowned and "Bug:bsh" not in q.newly_unowned


def test_exact_groups_for_the_leaver(wi):
    _, conn = wi
    r = whatif.if_leaves(conn, fx.P, fx.LEAVER)
    assert groups(r) == (B("bnone") | R(5),
                         B("bs", "bnew") | R(1, 6, 7),
                         B("bfix") | F("f_impl", "f_dev", "c1", "f_pend_missing", "f_pend_false"))
    assert r.bound_hit is False


def test_accepts_the_qualified_id(wi):
    _, conn = wi
    assert groups(whatif.if_leaves(conn, fx.P, PID)) == groups(whatif.if_leaves(conn, fx.P, fx.LEAVER))


def test_feature_implemented_by_an_orphaned_pr_is_affected(wi):
    _, conn = wi
    r = whatif.if_leaves(conn, fx.P, fx.LEAVER)
    assert F("f_impl", "f_dev") <= r.affected  # IMPLEMENTED_BY, and ABOUT with kind development
    assert not (F("f_unrec", "f_via_na", "f_alone", "f_other", "f_done") & (r.affected | r.newly_unowned))
    assert "Bug:bfix" in r.affected  # a bug whose fix PR is orphaned: not unowned itself (Queen), so affected


def test_in_progress_features_are_open_work_only_by_the_ruling(wi):
    _, conn = wi
    r = whatif.if_leaves(conn, fx.P, fx.LEAVER)
    assert F("f_pend_missing", "f_pend_false") <= r.affected  # implemented missing or false, development PR open
    assert not (F("f_pend_impl", "f_pend_closed", "f_nondev") & r.affected)  # implemented / PR closed / not development
    assert F("f_impl", "f_dev") <= r.affected  # the discussed-only rule still holds
    assert whatif_check.check(export_snapshot(conn, fx.P), fx.LEAVER) == r


def test_work_already_unowned_is_only_in_group_one(wi):
    _, conn = wi
    r = whatif.if_leaves(conn, fx.P, fx.ZED)
    assert {"Bug:bnone", f"PR:{fx.pr(5)}"} == r.already_unowned
    assert not (r.already_unowned & (r.newly_unowned | r.affected))
    # f_via_na is implemented by the already unowned PR 5; removing anyone leaves it out of group 3
    assert "Feature:f_via_na" not in r.affected


def test_closed_and_merged_prs_and_closed_bugs_are_not_open_work(wi):
    _, conn = wi
    r = whatif.if_leaves(conn, fx.P, fx.LEAVER)
    everything = r.already_unowned | r.newly_unowned | r.affected
    assert not ({"Bug:bclosed", f"PR:{fx.pr(3)}", f"PR:{fx.pr(4)}"} & everything)  # closed unmerged, merged, resolved
    assert f"PR:{fx.pr(2)}" not in everything  # shared by two authors


def test_groups_are_disjoint_and_features_only_in_group_three(wi):
    _, conn = wi
    for who in (fx.LEAVER, fx.QUEEN, fx.ROB, fx.ZED, fx.SAM1):
        r = whatif.if_leaves(conn, fx.P, who)
        a, n, f = groups(r)
        assert not (a & n) and not (a & f) and not (n & f)
        assert not any(x.startswith(("Feature:", "Document:")) for x in a | n)


def test_a_person_with_no_work_changes_nothing(wi):
    _, conn = wi
    r = whatif.if_leaves(conn, fx.P, fx.SAM1)
    assert not r.newly_unowned and not r.affected
    assert r.already_unowned == B("bnone") | R(5)


def test_unknown_person_is_refused_before_any_copy(wi):
    db, conn = wi
    with pytest.raises(ValueError, match="no such person"):
        whatif.if_leaves(conn, fx.P, "eid_00000000")
    assert copies(db) == []


# ---------------------------------------------------------------- the dependency bound
@pytest.mark.parametrize("chain,hit,reached", [(5, True, 4), (4, True, 4), (3, False, 3)])
def test_bound_hit_is_a_four_edge_witness_ending_at_newly_unowned_work(srv, tmp_path, monkeypatch, chain, hit, reached):
    fx.load(srv, fx.whatif_facts(chain=chain))
    conn = fx.whatif_conn(srv, tmp_path)
    monkeypatch.setattr(whatif, "DEPENDENCY_RULES", whatif.DEPENDENCY_RULES + TEST_RULE_GRAPH)
    monkeypatch.setattr(whatif_check, "DEPENDENCY_RULES", whatif_check.DEPENDENCY_RULES + TEST_RULE_CHECK)
    r = whatif.if_leaves(conn, fx.P, fx.LEAVER)
    assert r.bound_hit is hit
    assert {f"Feature:c{i}" for i in range(1, reached + 1)} <= r.affected
    assert f"Feature:c{chain}" in r.affected or chain > 4
    if chain == 5:
        assert "Feature:c5" not in r.affected  # five hops are beyond the bound
    snap = export_snapshot(conn, fx.P)
    assert whatif_check.check(snap, fx.LEAVER) == r  # the Python check reports the same bound_hit and sets


def test_a_reversed_chain_of_five_is_not_a_dependent(srv, tmp_path, monkeypatch):
    fx.load(srv, fx.whatif_facts(chain=5, reverse_chain=True))
    conn = fx.whatif_conn(srv, tmp_path)
    monkeypatch.setattr(whatif, "DEPENDENCY_RULES", whatif.DEPENDENCY_RULES + TEST_RULE_GRAPH)
    monkeypatch.setattr(whatif_check, "DEPENDENCY_RULES", whatif_check.DEPENDENCY_RULES + TEST_RULE_CHECK)
    r = whatif.if_leaves(conn, fx.P, fx.LEAVER)
    assert r.bound_hit is False
    assert F("c1") <= r.affected and not (F("c2", "c3", "c4", "c5") & r.affected)  # they are c1's prerequisites
    assert whatif_check.check(export_snapshot(conn, fx.P), fx.LEAVER) == r


def test_the_two_real_dependency_kinds_reach_one_hop_only(wi):
    _, conn = wi
    r = whatif.if_leaves(conn, fx.P, fx.LEAVER)
    assert r.bound_hit is False and "Feature:c2" not in r.affected


# ---------------------------------------------------------------- the independent check
def test_graph_and_python_check_agree_for_everyone(wi):
    _, conn = wi
    snap = export_snapshot(conn, fx.P)
    for who in (fx.LEAVER, fx.QUEEN, fx.ROB, fx.ZED, fx.SAM1, fx.SAM2):
        assert whatif_check.check(snap, who) == whatif.if_leaves(conn, fx.P, who), who


def test_the_check_has_no_view_of_the_graph_implementation():
    tree = ast.parse(Path(whatif_check.__file__).read_text())
    top = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)] + [
        a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names]
    assert "inherit.whatif" not in top and not any(m.startswith(("inherit.tools", "inherit.graph")) for m in top)
    assert "inherit.whatif_schema" in top  # the shared snapshot and schema
    assert "whatif." not in inspect.getsource(whatif_check.check)  # check() calls nothing of whatif.py


def test_check_by_hand_on_a_tiny_snapshot():
    from inherit.whatif_schema import SnapEdge, SnapNode, Snapshot

    n = lambda i, label, **p: SnapNode(i, label, p)  # noqa: E731
    e = lambda t, s, d, **p: SnapEdge(t, s, d, p, ())  # noqa: E731
    snap = Snapshot(
        nodes=(n("Person:a", "Person"), n("Person:b", "Person"), n("Bug:x", "Bug", active=True),
               n("BugEvent:x0", "BugEvent", type="ASSIGNED", ts="1", message_id="20260101-0-aaaaa", seq=0),
               n("PR:p", "PR", state="open", merged=False), n("Feature:f", "Feature", discussed_not_implemented=True),
               n("PR:q", "PR", state="open", merged=False)),
        edges=(e("HAS_EVENT", "Bug:x", "BugEvent:x0"), e("ASSIGNED_TO", "BugEvent:x0", "Person:a"),
               e("OPENED", "Person:a", "PR:p"), e("IMPLEMENTED_BY", "Feature:f", "PR:p"),
               e("FIXES", "PR:p", "Bug:x")), messages=())
    r = whatif_check.check(snap, "a")
    assert (r.already_unowned, r.newly_unowned, r.affected) == (frozenset({"PR:q"}), frozenset({"Bug:x", "PR:p"}),
                                                                 frozenset({"Feature:f"}))
    assert r.bound_hit is False
    with pytest.raises(ValueError):
        whatif_check.check(snap, "nobody")


# ---------------------------------------------------------------- the copy and the source
def test_the_copy_does_not_exist_afterwards(wi):
    db, conn = wi
    r = whatif.if_leaves(conn, fx.P, fx.LEAVER)
    assert copies(db) == []
    assert r.copy_graph.startswith(f"whatif:{fx.P}:") and r.copy_graph not in db.list_graphs()
    with pytest.raises(redis.ResponseError, match="empty key"):
        conn.execute_command("GRAPH.RO_QUERY", r.copy_graph, "RETURN 1")


def test_copy_is_retried_while_the_server_cannot_fork(wi, monkeypatch):
    db, conn = wi
    monkeypatch.setattr(whatif, "COPY_WAIT", 0.0)
    real, fails = conn.execute_command, [2]

    class Flaky:
        def __getattr__(self, name):
            return getattr(conn, name)

        def execute_command(self, *a, **k):
            if a[0] == "GRAPH.COPY" and fails[0]:
                fails[0] -= 1
                raise redis.exceptions.ResponseError("GRAPH.COPY failed, could not fork")
            return real(*a, **k)

    r = whatif.if_leaves(Flaky(), fx.P, fx.LEAVER)
    assert fails[0] == 0 and r.newly_unowned and copies(db) == []
    fails[0] = 99
    monkeypatch.setattr(whatif, "COPY_TRIES", 3)
    with pytest.raises(redis.ResponseError, match="could not fork"):
        whatif.if_leaves(Flaky(), fx.P, fx.LEAVER)
    assert copies(db) == []


def test_an_uncertain_copy_is_cleaned_and_the_original_error_kept(wi):
    db, conn = wi
    real = conn.execute_command

    class CreatesThenRaises:
        def __getattr__(self, name):
            return getattr(conn, name)

        def execute_command(self, *a, **k):
            out = real(*a, **k)
            if a[0] == "GRAPH.COPY":
                assert copies(db)  # the server did create the copy
                raise redis.exceptions.ConnectionError("response lost")
            return out

    with pytest.raises(redis.exceptions.ConnectionError, match="response lost"):
        whatif.if_leaves(CreatesThenRaises(), fx.P, fx.LEAVER)
    assert copies(db) == []


def test_a_failed_cleanup_is_reported_on_the_original_error(wi, monkeypatch):
    db, conn = wi

    def boom(*a, **k):
        raise RuntimeError("injected")

    monkeypatch.setattr(whatif, "_affected", boom)
    monkeypatch.setattr(whatif, "_cleanup", lambda run, g: (_ for _ in ()).throw(OSError("no delete")))
    with pytest.raises(RuntimeError, match="injected") as e:
        whatif.if_leaves(conn, fx.P, fx.LEAVER)
    assert any("no delete" in n for n in e.value.__notes__)
    for g in copies(db):
        db.select_graph(g).delete()


def test_the_copy_is_deleted_after_an_injected_exception(wi, monkeypatch):
    db, conn = wi

    def boom(*a, **k):
        raise RuntimeError("injected")

    monkeypatch.setattr(whatif, "_affected", boom)
    with pytest.raises(RuntimeError, match="injected"):
        whatif.if_leaves(conn, fx.P, fx.LEAVER)
    assert copies(db) == []


def test_the_source_graph_is_unchanged(wi):
    db, conn = wi
    dump = lambda: (db.select_graph(f"herb:{fx.P}").query(  # noqa: E731
        "MATCH (n) RETURN n.id, labels(n), properties(n) ORDER BY n.id").result_set,
        db.select_graph(f"herb:{fx.P}").query(
            "MATCH (a)-[e]->(b) RETURN a.id, type(e), b.id, properties(e) ORDER BY a.id, type(e), b.id").result_set)
    before = dump()
    for who in (fx.LEAVER, fx.QUEEN):
        whatif.if_leaves(conn, fx.P, who)
    assert dump() == before
    assert not db.select_graph(f"herb:{fx.P}").query("MATCH ()-[e:DEPENDS_ON|RESPONSIBLE_FOR]->() RETURN count(e)"
                                                     ).result_set[0][0]


def test_writes_to_the_source_fail_under_the_real_whatif_credentials(wi):
    db, conn = wi
    src = f"herb:{fx.P}"
    for cmd in (("GRAPH.QUERY", src, "MATCH (b:Bug) SET b.active = false"),
                ("GRAPH.QUERY", src, "MATCH (n:Person) DETACH DELETE n"), ("GRAPH.DELETE", src),
                ("GRAPH.COPY", src, f"herb:{fx.P}x"), ("GRAPH.COPY", src, f"whatif:other:{fx.P}"),
                ("GRAPH.COPY", f"whatif:{fx.P}:1", src)):
        with pytest.raises(redis.ResponseError):
            conn.execute_command(*cmd)
    assert db.select_graph(src).query("MATCH (b:Bug {active: true}) RETURN count(b)").result_set[0][0] == 8


def test_the_agent_user_cannot_run_if_leaves(srv, tmp_path):
    fx.load(srv)
    agent = fx.whatif_conn(srv, tmp_path, user="agent")
    with pytest.raises((redis.exceptions.NoPermissionError, ToolError)):
        whatif.if_leaves(agent, fx.P, fx.LEAVER)
    with pytest.raises(ToolError):
        TOOLS["if_leaves"].run(agent, fx.P, person="Pat Lee")
    assert copies(srv) == []


def test_ops_record_the_copy_the_queries_and_the_delete(wi):
    _, conn = wi
    r = whatif.if_leaves(conn, fx.P, fx.LEAVER)
    cmds = [o.command for o in r.ops]
    assert cmds.count("GRAPH.COPY") == 1 and cmds.index("GRAPH.COPY") < cmds.index("GRAPH.DELETE")
    q = "\n".join(o.query for o in r.ops)
    assert "DEPENDS_ON*1..4" in q and "RESPONSIBLE_FOR" in q and "DETACH DELETE" in q  # literal bound in the text
    assert all(o.tool == "if_leaves" for o in r.ops)


# ---------------------------------------------------------------- the tool
def test_tool_resolves_the_name_and_returns_typed_rows(wi):
    _, conn = wi
    t = Trace()
    res = t.run(conn, fx.P, "if_leaves", person="Pat Lee")
    by_group = {}
    for row in res.rows[1:]:
        by_group.setdefault(row["group"], set()).add(row["answer"])
    r = whatif.if_leaves(conn, fx.P, fx.LEAVER)
    assert by_group == {"newly_unowned": set(r.newly_unowned), "already_unowned": set(r.already_unowned),
                        "affected": set(r.affected)}
    head = res.rows[0]
    assert head["answer"] == fx.LEAVER and head["bound_hit"] is False
    assert head["counts"] == {"already_unowned": 2, "newly_unowned": 5, "affected": 6}
    assert "recorded" in head["basis"] and "prediction" in head["basis"]
    assert all(row["refs"] for row in res.rows[1:]) and any(o.command == "GRAPH.COPY" for o in t.ops)
    assert TOOLS["if_leaves"].run(conn, fx.P, person=fx.LEAVER).rows[0]["answer"] == fx.LEAVER  # by employee id


def test_tool_refuses_ambiguous_and_unknown_names_with_candidates(wi):
    db, conn = wi
    with pytest.raises(ToolError) as e:
        TOOLS["if_leaves"].run(conn, fx.P, person="Sam Kim")
    assert fx.SAM1 in str(e.value) and fx.SAM2 in str(e.value) and "ambiguous" in str(e.value)
    with pytest.raises(ToolError, match="no person"):
        TOOLS["if_leaves"].run(conn, fx.P, person="Nobody Here")
    assert copies(db) == []


# ---------------------------------------------------------------- the Task 13 audit hook
def test_audit_samples_carry_evidence_and_later_context(wi):
    _, conn = wi
    snap = export_snapshot(conn, fx.P)
    res = {fx.LEAVER: whatif_check.check(snap, fx.LEAVER)}
    a = whatif_check.audit_samples(snap, res, seed=7, n_unowned=30, n_edges=30, n_paths=20)
    assert a == whatif_check.audit_samples(snap, res, seed=7, n_unowned=30, n_edges=30, n_paths=20)  # seeded
    assert a["seed"] == 7
    items = {s["item"]: s for s in a["newly_unowned"]}
    assert set(items) == set(res[fx.LEAVER].newly_unowned)  # fewer than 30: all of them
    assert a["shortages"]["newly_unowned"] == {"requested": 30, "available": 5}
    bs = items["Bug:bs"]
    assert bs["person"] == fx.LEAVER and bs["assignment"]["message_id"] == "20260602-0-aaaa2"
    assert bs["assignment"]["records"] == ["20260602-0-aaaa2"] and bs["assignment"]["text"].startswith("@eid_a0000001")
    ctx = {m["id"] for m in bs["later_context"]}
    assert "20260603-0-m0001" in ctx  # same channel, later
    assert "20260604-0-m0003" in ctx  # another channel, names the bug by title
    assert "20260603-1-m0002" not in ctx and "20260502-0-m0004" not in ctx  # unrelated / earlier
    edges = {(s["dependent"], s["prerequisite"]): s for s in a["dependency_edges"]}
    assert ("Feature:f_impl", f"PR:{fx.pr(1)}") in edges and edges[("Feature:f_impl", f"PR:{fx.pr(1)}")]["citations"] == [fx.slug(1)]
    assert a["shortages"]["dependency_edges"]["available"] >= len(edges)
    assert any("Summary of PR 1" in t and fx.LEAVER in t for t in edges[("Feature:f_impl", f"PR:{fx.pr(1)}")]["texts"])
    # the pool is every recorded dependency, not only those leading to this person's newly unowned work
    assert ("Feature:f_via_na", f"PR:{fx.pr(5)}") in edges or len(edges) < a["shortages"]["dependency_edges"]["available"]
    paths = {p["affected"]: p for p in a["affected_paths"]}
    assert paths["Bug:bfix"]["path"] == ["Bug:bfix", f"PR:{fx.pr(1)}"] and paths["Bug:bfix"]["edges"][0]["citations"] == ["20260610-0-fff01"]
    assert a["cutoff"] == "20260609-2-p0003"
    pr1 = items[f"PR:{fx.pr(1)}"]  # a hand-over in a channel that mentioned the PR, without repeating link or title
    assert pr1["assignment"]["pr"]["authors"] == [PID] and pr1["assignment"]["pr"]["summary"] == "Summary of PR 1"
    assert {m["id"] for m in pr1["later_context"]} == {"20260609-0-p0001", "20260609-1-p0002"}


def test_audit_sample_sizes_are_capped_and_seeded(wi):
    _, conn = wi
    snap = export_snapshot(conn, fx.P)
    res = {fx.LEAVER: whatif_check.check(snap, fx.LEAVER)}
    a = whatif_check.audit_samples(snap, res, seed=1, n_unowned=2, n_edges=1, n_paths=1)
    assert (len(a["newly_unowned"]), len(a["dependency_edges"]), len(a["affected_paths"])) == (2, 1, 1)
    assert "newly_unowned" not in a["shortages"]
    assert a == whatif_check.audit_samples(snap, res, seed=1, n_unowned=2, n_edges=1, n_paths=1)


# ---------------------------------------------------------------- the runner and the dev products
def test_compare_runs_both_for_every_engineering_lead(wi):
    _, conn = wi
    rows = whatif_check.compare(conn, fx.P)
    assert sorted(r["person"] for r in rows) == sorted([fx.LEAVER, fx.QUEEN])  # the two Engineering Leads
    assert all(r["match"] for r in rows)
    lead = next(r for r in rows if r["person"] == fx.LEAVER)
    assert (lead["already_unowned"], lead["newly_unowned"], lead["affected"], lead["bound_hit"]) == (2, 5, 6, False)


def test_compare_owners_covers_everyone_with_work(wi):
    _, conn = wi
    rows = whatif_check.compare(conn, fx.P, who="owners")
    assert {r["person"] for r in rows} == {fx.LEAVER, fx.QUEEN, fx.ROB, fx.ZED} and all(r["match"] for r in rows)


def test_compare_reports_a_mismatch(wi, monkeypatch):
    _, conn = wi
    monkeypatch.setattr(whatif_check, "check", lambda s, p: WhatIf(frozenset(), frozenset(), frozenset(), False))
    rows = whatif_check.compare(conn, fx.P)
    bad = [r for r in rows if not r["match"]]
    assert bad and all("newly_unowned" in r["diff"] for r in bad)


DEV = ("ActionGenie", "AnomalyForce", "ContentForce", "KnowledgeForce", "LeadForce", "TrendForce")


@pytest.mark.parametrize("server", ["4.22", "6.0"])
def test_dev_engineering_leads_agree(server):
    acl = ROOT / "data" / "acl.json"
    assert acl.exists(), "data/acl.json missing"
    db = fx.connect(server)
    have = json.loads(acl.read_text()).get(server, {})
    missing = [p for p in DEV if f"herb:{p}" not in db.list_graphs() or f"whatif_{p}" not in have]
    assert not missing, f"dev products not loaded on {server} (graph or whatif user missing): {missing}"
    for p in DEV:
        rows = whatif_check.compare(whatif.whatif_connection(p, server), p)
        assert rows and all(r["match"] for r in rows), [r for r in rows if not r["match"]]
