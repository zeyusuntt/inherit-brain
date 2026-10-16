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
