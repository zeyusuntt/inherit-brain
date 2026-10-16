"""A3 (design F12): the SQLite consistency check. Three catalogue query families, SQL against the graph tools."""

import sqlite3

from toolfx import P, tg, tool_facts, tsrv  # noqa: F401  (fixtures)
from baselines import sqlite_check as sc


def test_export_keeps_every_node_and_edge_with_json_props():
    facts = tool_facts()
    db = sc.export_facts(facts)
    assert db.execute("select count(*) from nodes").fetchone()[0] == len({n.id for n in facts.nodes})
    assert db.execute("select count(*) from edges").fetchone()[0] == len({e.ekey for e in facts.edges})
    props = db.execute("select props from nodes where id = 'Bug:b2'").fetchone()[0]
    assert isinstance(props, str) and '"title"' in props


def test_sql_families_on_the_fixture():
    db = sc.export_facts(tool_facts())
    top = sc.bug_board_by_company(db, "resolved", "max")
    assert top and all(r[1] == top[0][1] for r in top)
    assert sc.bug_board_by_company(db, "any", "all")
    assert sc.fix_trail(db, customer="Nobody Inc") == []
    assert sc.pr_board_approved(db, feature="no such feature") == []
    assert sc.pr_board_approved(db, feature=P) == sc.pr_board_approved(db)  # the product name is no filter


def test_matrix_covers_empty_duplicate_and_state_dependent_cases():
    m = sc.param_matrix(sc.export_facts(tool_facts()))
    fams = {f for f, _ in m}
    assert fams == {"bug_board", "fix_trail", "pr_board"}
    statuses = {p.get("status") for f, p in m if f == "bug_board"}
    assert {"unresolved", "resolved", "active", "reopened", "no_fix_needed", "any"} <= statuses
    assert any(p.get("customer") == "Nobody Inc" for _, p in m)
    assert any(p.get("rank") == "max" for _, p in m)


def test_sql_equals_graph_tools_over_the_matrix(tg):
    db = sc.export_facts(tool_facts())
    matrix = sc.param_matrix(db)
    assert sc.compare(db, tg, P, matrix) == []
    nonempty = [1 for f, p in matrix if sc.sql_rows(db, f, p)]
    assert len(nonempty) >= 8


def test_a_deliberate_difference_is_detected(tg):
    db = sc.export_facts(tool_facts())
    matrix = sc.param_matrix(db)
    db.execute("update nodes set props = json_set(props, '$.approval', 'unapproved') where label = 'PR' "
               "and json_extract(props, '$.approval') = 'approved' and json_extract(props, '$.repo_scope') = 'product'")
    diffs = sc.compare(db, tg, P, matrix)
    assert diffs and {d["family"] for d in diffs} == {"pr_board"}
    db2 = sc.export_facts(tool_facts())
    db2.execute("delete from edges where type = 'FIXES'")
    assert {d["family"] for d in sc.compare(db2, tg, P, matrix)} >= {"fix_trail"}


def test_latency_reports_both_sides(tg):
    db = sc.export_facts(tool_facts())
    lat = sc.latency(db, tg, P, sc.param_matrix(db), repeat=2)
    assert set(lat) == {"bug_board", "fix_trail", "pr_board"}
    assert all(v["sql_ms"] >= 0 and v["graph_ms"] >= 0 for v in lat.values())


# ---------------------------------------------------------------- fan-out, exit status, single-author precondition
def fanout_facts():
    """One PR matching two features through a shared alias, and a bug reported through two customer paths."""
    from toolfx import _e
    from inherit.facts import Facts, Node

    facts = tool_facts()
    nodes = [Node(n.label, n.id, {**n.props, "aliases": [*n.props.get("aliases", []), "shared alias"]})
             if n.id in ("Feature:fA", "Feature:fB") else n for n in facts.nodes]
    edges = [*facts.edges,
             _e("REPORTED_BY", "Bug:b2", "Customer:CUST-0001", ["rec-b2-dup"]),  # second path to TechCorp
             _e("ABOUT", "PR:" + sc_pr(10), "Feature:fA", ["rec-dup-about"], key="again", kind="development")]
    return Facts(nodes, edges)


def sc_pr(n):
    from toolfx import pr
    return pr(n)


def test_fanout_is_deduplicated_canonically_against_both_engines(tsrv, tmp_path):
    from toolfx import agent_graph, load

    facts = fanout_facts()
    load(tsrv, facts)
    g = agent_graph(tsrv, tmp_path)
    g.server_version = tsrv.version_id
    db = sc.export_facts(facts)
    # the fan-out is real: the shared alias reaches two features and PR10 has three ABOUT edges
    assert len(sc._features(db, "shared alias")) == 2
    assert db.execute("select count(*) from edges where type = 'ABOUT' and src_id = ?",
                      ("PR:" + sc_pr(10),)).fetchone()[0] == 3
    cases = [("pr_board", {"feature": "shared alias"}), ("fix_trail", {"bug": "Slow export"}),
             ("fix_trail", {"customer": "TechCorp"}), ("bug_board", {"status": "any", "customer": "TechCorp"}),
             ("bug_board", {"status": "any", "rank": "max"})]
    for family, params in cases:
        a, b = sc.sql_rows(db, family, params), sc.graph_rows(g, P, family, params)
        assert a == b, (family, params)
        assert len(a) == len(set(a)), (family, params, "duplicate canonical rows")
    links = [r[0] for r in sc.sql_rows(db, "pr_board", {"feature": "shared alias"})]
    assert links.count(sc_pr(10)) == 1
    trail = sc.sql_rows(db, "fix_trail", {"bug": "Slow export"})
    assert {t[4] for t in trail} == {("Customer:CUST-0001", "Customer:CUST-0002")} and all(t[3] == ("TechCorp",) for t in trail)
    assert sc.compare(db, g, P, cases) == []


class _StubGraph:
    pass


def test_main_exits_non_zero_when_any_comparison_differs(monkeypatch, capsys):
    import pytest as _pytest

    db = sc.export_facts(tool_facts())
    monkeypatch.setattr(sc, "export_product", lambda product: db)
    monkeypatch.setattr(sc, "param_matrix", lambda d: [("pr_board", {})])
    monkeypatch.setattr(sc, "latency", lambda *a, **k: {})
    monkeypatch.setattr("inherit.tools.depth1.agent_graph", lambda product, server: _StubGraph())
    monkeypatch.setattr(sc, "DEV_PRODUCTS", ("OnlyOne",))
    monkeypatch.setattr("sys.argv", ["sqlite_check", "--products", "dev", "--servers", "x"])
    monkeypatch.setattr(sc, "compare", lambda *a, **k: [{"family": "pr_board", "params": {}, "only_sql": [1],
                                                          "only_graph": [], "sql": [1], "graph": []}])
    with _pytest.raises(SystemExit) as e:
        sc.main()
    assert e.value.code != 0
    assert "DIFF" in capsys.readouterr().out
    monkeypatch.setattr(sc, "compare", lambda *a, **k: [])
    sc.main()  # no differences: returns normally


def test_scoped_check_requires_a_single_author_per_pr_and_bug_pair():
    from toolfx import _e
    from inherit.facts import Facts

    facts = tool_facts()
    two = Facts(facts.nodes, [*facts.edges, _e("OPENED", "Person:" + "eid_c3000003", "PR:" + sc_pr(2), ["rec-two"])])
    db = sc.export_facts(two)
    import pytest as _pytest
    with _pytest.raises(AssertionError, match="several authors"):
        sc.fix_trail(db)
    sc.fix_trail(sc.export_facts(facts))  # the single-author fixture passes
