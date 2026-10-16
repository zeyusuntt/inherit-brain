import json

import pytest
from falkordb import FalkorDB

from conftest import fake_embed, fx_edge, fx_node
from eval import families as F
from eval import questions as Q
from eval import stats as S
from inherit.config import load_settings
from inherit.contracts import QuestionKey
from inherit.facts import Facts
from inherit.graph import load as loadmod

P = "FamFixForce"
PROD = f"Product:{P}"
PR1, PR2 = "https://github.com/famfix/pull/1", "https://github.com/famfix/pull/2"
SAM1, SAM2 = "eid_5a0000a1", "eid_5b0000a2"


def fam_facts() -> Facts:
    N, E = fx_node, fx_edge
    nodes = [
        N("Product", P, name=P),
        N("Bug", "b1", id="b1", title="Login crash", status="active"),
        N("Bug", "b2", id="b2", title="Slow export of reports", status="active"),
        N("Customer", "CUST-0001", id="CUST-0001", company="TechCorp", name="Dana Cole", role="CTO"),
        N("Customer", "CUST-0002", id="CUST-0002", company="BlueWave", name="Eli Fox", role="CIO"),
        N("PR", PR1, title="Fix login", number=1, link=PR1),
        N("PR", PR2, title="Fix export", number=2, link=PR2),
        N("Person", SAM1, eid=SAM1, name="Sam Kim", role="UX Researcher", in_directory=True),
        N("Person", SAM2, eid=SAM2, name="Sam Kim", role="Marketing Manager", in_directory=True),
    ]
    edges = [E("HAS", PROD, "Bug:b1"), E("HAS", PROD, "Bug:b2"), E("HAS", PROD, f"PR:{PR1}"),
             E("HAS", PROD, f"PR:{PR2}"),
             E("REPORTED_BY", "Bug:b1", "Customer:CUST-0001"), E("REPORTED_BY", "Bug:b2", "Customer:CUST-0002"),
             E("FIXES", f"PR:{PR1}", "Bug:b1"), E("FIXES", f"PR:{PR2}", "Bug:b2")]
    return Facts(nodes, edges)


def _clean(db):
    if f"herb:{P}" in db.list_graphs():
        db.select_graph(f"herb:{P}").delete()
    for u in (f"agent_{P}", f"whatif_{P}", f"build_{P}"):
        db.connection.execute_command("ACL", "DELUSER", u)


@pytest.fixture(params=["4.22", "6.0"])
def g(request):
    url = load_settings().falkor_urls[request.param]
    db = FalkorDB.from_url(url)
    try:
        db.connection.ping()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"FalkorDB {request.param} not reachable at {url}: {e}")
    _clean(db)
    loadmod.load_product(db, P, fam_facts(), embed=fake_embed)
    yield db.select_graph(f"herb:{P}")
    _clean(db)


def mk(i, text, gold, type_="person"):
    return Q.Question(QuestionKey(P, "ans", i), text, type_, gold, [])


def run(g, qs):
    return F.classify_dev(qs, lambda p: g)


# ---------------------------------------------------------------- instance verdicts
def test_answer_on_a_neighbour_card_is_visible_and_two_hops_away_needs_more(g):
    qs = [mk(0, "Who is the contact of the customer that reported Login crash?", ["Dana Cole"]),
          mk(1, f"Who is the customer contact behind {PR1}?", ["Dana Cole"])]
    r = run(g, qs)["questions"]
    a, b = r[str(qs[0].key)], r[str(qs[1].key)]
    assert a["verdict"] == "visible" and a["values"][0]["witness"] == ["Customer:CUST-0001"]
    assert b["verdict"] == "needs_more" and b["missing"] == ["Dana Cole"] and b["values"][0]["witness"] == "missing"
    assert b["roots"] == 2 and b["visible_cards"] >= 3  # Product, the PR, and its neighbours


def test_product_root_alone(g):
    q = mk(0, f"Which bugs does {P} have?", ["Login crash"])
    r = run(g, [q])["questions"][str(q.key)]
    assert r["roots"] == 1 and r["verdict"] == "visible"


def test_namesake_roots_are_both_authorised(g):
    q = mk(0, "Which role does Sam Kim hold?", ["Marketing Manager", "UX Researcher"])
    r = run(g, [q])["questions"][str(q.key)]
    assert r["roots"] == 3 and r["verdict"] == "visible"
    assert [v["witness"] for v in r["values"]] == [[f"Person:{SAM2}"], [f"Person:{SAM1}"]]


def test_fallback_root_from_a_two_word_span(g):
    q = mk(0, "Which customer contact reported the slow export problem?", ["Eli Fox"])
    roots = F.authorised_roots(g, q.text, P)
    assert "Bug:b2" in roots  # no exact name key is in the question; the span "slow export" found it
    assert run(g, [q])["questions"][str(q.key)]["verdict"] == "visible"


def test_empty_gold_is_no_gold(g):
    q = mk(0, f"Which bugs does {P} have?", [])
    assert run(g, [q])["questions"][str(q.key)]["verdict"] == "no_gold"


def test_paging_diagnostic_is_false_for_small_groups(g):
    q = mk(0, f"Which bugs does {P} have?", ["Login crash"])
    assert run(g, [q])["questions"][str(q.key)]["values"][0]["paging_only"] is False


# ---------------------------------------------------------------- family verdicts
def test_family_verdict_rules():
    assert F.family_verdict(["needs_more", "needs_more"]) == "multi"
    assert F.family_verdict(["needs_more", "visible"]) == "single"  # one visible instance
    assert F.family_verdict(["needs_more", "no_gold"]) == "multi"
    assert F.family_verdict(["no_gold"]) == "no_votes" and F.family_verdict([]) == "no_votes"


def test_mixed_family_is_single_and_content_is_listed_not_classified(g):
    qs = [mk(0, f"Who is the customer contact behind {PR1}?", ["Dana Cole"]),
          mk(1, f"Who is the customer contact behind {PR2}?", ["Slow export of reports"]),  # same template
          mk(2, "Where is the long form note?", ["x"], "content"),
          mk(3, "Which role does Sam Kim hold?", [])]
    fams = run(g, qs)["families"]
    t = Q.template(qs[0].text, P)
    assert Q.template(qs[1].text, P) == t
    assert fams[t]["verdict"] == "single" and fams[t]["votes"] == {"needs_more": 1, "visible": 1}
    assert fams[Q.template(qs[2].text, P)]["verdict"] == F.CONTENT
    assert fams[Q.template(qs[3].text, P)]["verdict"] == "no_votes"


# ---------------------------------------------------------------- held export
def test_unseen_export_is_only_strings_and_types(monkeypatch):
    held = [Q.Question(QuestionKey("ZZ", "ans", 0), "Who reviewed ZZ?", "person", ["SECRETGOLD"], ["cit"], False),
            Q.Question(QuestionKey("ZZ", "ans", 1), "Who reviewed ZZ?", "person", ["SECRETGOLD2"], [], False),
            Q.Question(QuestionKey("ZZ", "ans", 2), "Seen one?", "pr", ["g"], [], True),
            Q.Question(QuestionKey("ZZ", "ans", 3), "A content one?", "content", "g", [], False),
            Q.Question(QuestionKey("ZZ", "unans", 0), "Unanswerable?", "unanswerable", None, [], None)]
    monkeypatch.setattr(Q, "load_questions", lambda split, *a, **k: held)
    out = Q.unseen_held_templates()
    assert out == [("a content one", "content"), ("who reviewed <p>", "person")]
    assert all(type(x) is tuple and all(type(s) is str for s in x) for x in out)


# ---------------------------------------------------------------- packet and merge
def test_packet_sends_only_id_templates_with_stable_ids():
    text, rows = F.build_packet([("a content one", "content"), ("who reviewed <p>", "person"),
                                 ("which pr fixed <x>", "pr")])
    assert [r["id"] for r in rows] == ["u001", "u002"]
    assert "a content one" not in text and "who reviewed <p>" in text
    assert "### Bug" in text and "multi" in text and "多跳题族" in text and '"verdicts"' in text


def write_v(tmp_path, name, items):
    p = tmp_path / name
    p.write_text(json.dumps({"verdicts": [{"id": i, "verdict": v, "reason": "r"} for i, v in items]}))
    return p


ROWS = [{"id": f"u00{i}", "template": f"t{i}", "type": "person"} for i in range(1, 7)]


def test_merge_truth_table(tmp_path):
    codex = write_v(tmp_path, "c.json", [("u001", "multi"), ("u002", "single"), ("u003", "multi"), ("u004", "uncertain"),
                                         ("u005", "multi")])  # u006 missing
    claude = write_v(tmp_path, "l.json", [("u001", "multi"), ("u002", "single"), ("u003", "single"),
                                          ("u004", "uncertain"), ("u005", "bogus"), ("u006", "multi")])
    m = F.merge_unseen(ROWS, F.read_verdicts(codex), F.read_verdicts(claude))
    assert [m[f"t{i}"]["verdict"] for i in range(1, 7)] == ["multi", "single", F.EXCLUDED, F.EXCLUDED, F.EXCLUDED,
                                                            F.EXCLUDED]
    assert m["t6"]["codex"] is None and m["t5"]["claude"] is None


def test_malformed_files_and_duplicate_ids_count_as_missing(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert F.read_verdicts(bad) == {} and F.read_verdicts(tmp_path / "nope.json") == {}
    bad.write_text(json.dumps({"verdicts": "x"}))
    assert F.read_verdicts(bad) == {}
    dup = write_v(tmp_path, "d.json", [("u001", "multi"), ("u001", "multi"), ("u002", "single")])
    assert F.read_verdicts(dup) == {"u002": "single"}
    m = F.merge_unseen(ROWS[:1], {}, {})
    assert m["t1"]["verdict"] == F.EXCLUDED


# ---------------------------------------------------------------- PREREG block and R11
def test_prereg_block_round_trip():
    fams = {"who reviewed <p>", "find all <x> of <p>"}
    block = F.render_block(fams)
    assert block.splitlines() == ["multi-hop family: find all <x> of <p>", "multi-hop family: who reviewed <p>"]
    assert S.read_multihop_families("# PREREG\n" + block + "\n") == fams


def test_empty_block_is_one_line_and_parses_to_the_empty_set():
    assert F.render_block(set()) == "multi-hop families: none"
    assert S.read_multihop_families("x\nmulti-hop families: none\n") == set()


def test_parser_rejects_neither_and_both():
    with pytest.raises(ValueError):
        S.read_multihop_families("nothing")
    with pytest.raises(ValueError, match="both"):
        S.read_multihop_families("multi-hop family: a\nmulti-hop families: none\n")


def test_h3_is_not_estimable_with_an_empty_family_set():
    from test_stats import two_arm_fixture, summaries, boots
    a, b = two_arm_fixture()
    out = S.compute_bootstraps({"A5": a, "A0": b, "A1f": b, "A4": b}, set(), None, n=20, seed=1, pilot=True)
    assert out["H3"] is None and out["H3_all"].diff == 60.0
    d = S.decide(summaries(), {**boots(), "H3": None, "H3_all": (5.0, 1.0, 9.0)})
    assert d["H3"]["status"] == "not estimable" and d["H3"]["pass"] is False
    assert d["H3"]["all_questions"] == (5.0, 1.0, 9.0)
