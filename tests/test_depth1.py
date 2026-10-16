import json

import pytest

from cardfx import PRODUCT, P, srv, srv_both, PR101, agent_graph, card_facts, load
from inherit.contracts import TraceOp
from inherit.tools import depth1
from inherit.tools.depth1 import Episode, RootSet, ToolError

Q = "Find all PRs for bugs reported by TechCorp in mini"
QI = "Which bug that Ian Miller didn’t fix is called Users can’t log in, and what about Slow export?"
CUST = "Customer:CUST-0001"


@pytest.fixture
def g(srv, load_facts):
    return load(srv, card_facts(load_facts))


@pytest.fixture
def gm(srv, load_facts):
    return load(srv, card_facts(load_facts, many=250))


def ep(g, q=Q, strict=True, **kw):
    return Episode(g, q, P, strict=strict, **kw)


def ids(env):
    return [r["id"] for r in env["rows"]]


# ---------------------------------------------------------------- normalise
def test_normalise_straightens_quotes_collapses_whitespace_and_applies_nfkc():
    assert depth1.normalise("didn’t  “quote”\n\t it ‘x’") == "didn't \"quote\" it 'x'"
    assert depth1.normalise("ＡＢＣ  １") == "ABC 1"  # NFKC
    assert depth1.normalise("  Tech Corp ") == "Tech Corp"


# ---------------------------------------------------------------- RootSet
def test_resolve_techcorp_works_and_fusiontech_raises(g):
    rs = RootSet(g, Q, P)
    assert rs.resolve("TechCorp") == [CUST]
    with pytest.raises(ToolError):
        rs.resolve("FusionTech")


def test_resolve_matching_is_case_folded_and_empty_spans_raise(g):
    rs = RootSet(g, Q, P)
    assert rs.resolve("techcorp") == [CUST]
    for bad in ("", "   ", "\n"):
        with pytest.raises(ToolError):
            rs.resolve(bad)


def test_curly_apostrophes_match_straight_ones(g):
    rs = RootSet(g, QI, P)
    assert rs.resolve("didn't") == []  # substring of the question after normalising; no node has that name
    assert rs.resolve("didn’t") == []
    assert rs.resolve("Users can't log in") == ["Bug:b5"]
    assert rs.resolve("Users can’t log in") == ["Bug:b5"]


def test_namesakes_stay_plural(g):
    assert RootSet(g, QI, P).resolve("Ian Miller") == ["Person:eid_c3000003", "Person:eid_d4000004"]


def test_exact_keys_per_label(g):
    q = ("CUST-0002 eid_e5000005 Design Spec mini_design_spec Auth login failure https://github.com/mini/mini-server/"
         "pull/101 Change 101 Bulk export Rival Corp https://docs.example.com/export")
    rs = RootSet(g, q, P)
    assert rs.resolve("CUST-0002") == ["Customer:CUST-0002"]
    assert rs.resolve("eid_e5000005") == ["Person:eid_e5000005"]
    assert rs.resolve("Design Spec") == ["Document:mini_design_spec"]
    assert rs.resolve("mini_design_spec") == ["Document:mini_design_spec"]
    assert rs.resolve("Auth") == ["Bug:b5"]  # component
    assert rs.resolve("login failure") == ["Bug:b5"]  # alias
    assert rs.resolve("https://github.com/mini/mini-server/pull/101") == [PR101]
    assert rs.resolve("Change 101") == [PR101]
    assert rs.resolve("Bulk export") == ["Feature:f1"]
    assert rs.resolve("Rival Corp") == ["Competitor:Rival Corp"]
    assert rs.resolve("https://docs.example.com/export") == ["Url:https://docs.example.com/export"]


def test_full_text_fallback_only_without_exact_match_and_two_words(g):
    rs = RootSet(g, "the Slow export story, export, Alice Chen Senior (of x", P)
    assert rs.resolve("Slow export") == ["Bug:b2"]
    assert rs.resolve("export") == []  # one word: no fallback
    assert rs.resolve("Alice Chen Senior") == []  # fallback never reaches Person
    assert rs.resolve("(of x") == []  # search syntax is literal: no error
    assert rs.resolve("story, export") == []
    assert rs.resolve("Slow export story") == []  # all terms must match


def test_fallback_keeps_ties_at_the_cut(srv, load_facts):
    from conftest import fx_edge, fx_node
    from inherit.facts import Facts
    nodes = [fx_node("Bug", f"t{i}", id=f"t{i}", status="active", title=f"Sync widget crash {i}") for i in range(7)]
    edges = [fx_edge("HAS", "Product:mini", n.id) for n in nodes]
    g = load(srv, card_facts(load_facts, extra=Facts(nodes, edges)))
    got = RootSet(g, "bug Sync widget crash", P).resolve("Sync widget crash")
    assert len(got) == 7 and got == sorted(got)  # equal scores: all tied candidates are kept


# ---------------------------------------------------------------- Episode: roots and strict mode
def test_product_is_always_an_authorised_root(g):
    e = ep(g)
    env = e.get_card(PRODUCT)
    assert ids(env) == [PRODUCT] and env["truncated"] is False and env["cursor"] is None


def test_resolve_returns_ids_and_names_and_authorises_roots(g):
    e = ep(g)
    env = e.resolve("TechCorp")
    assert env["rows"] == [{"id": CUST, "label": "Customer", "name": "TechCorp"}]
    assert ids(e.get_card(CUST)) == [CUST]
    with pytest.raises(ToolError):
        e.get_card("Bug:b1")  # never resolved, not a root


def test_neighbors_returns_neighbour_cards_only(g):
    e = ep(g)
    e.resolve("TechCorp")
    env = e.neighbors(CUST, "REPORTED_BY", "in")
    assert ids(env) == ["Bug:b1", "Bug:b2", "Bug:b5"]
    assert env["rows"][0] == depth1.cards.card(g, "Bug:b1")
    assert set(env) == {"rows", "truncated", "cursor"}  # no count unless asked


def test_strict_refuses_a_pr_returned_by_an_earlier_call_and_non_strict_accepts_it(g):
    for strict in (True, False):
        e = ep(g, strict=strict)
        assert PR101 in ids(e.neighbors(PRODUCT, "HAS", "out", where={"label": "PR"}))
        if strict:
            with pytest.raises(ToolError):
                e.neighbors(PR101, "FIXES", "out")
            with pytest.raises(ToolError):
                e.get_card(PR101)
        else:
            assert ids(e.neighbors(PR101, "FIXES", "out")) == ["Bug:b1"]
            assert e.stats()["max_hop"] == 2


def test_non_strict_hop_and_distinct_card_accounting(g):
    e = ep(g, strict=False)
    e.resolve("TechCorp")
    assert e.stats() == {"calls": 1, "distinct_cards": 0, "max_hop": 0}
    e.get_card(CUST)
    e.neighbors(CUST, "REPORTED_BY", "in")
    s = e.stats()
    assert s["distinct_cards"] == 4 and s["max_hop"] == 1  # customer + three bugs
    e.neighbors("Bug:b1", "FIXES", "in")  # hop 2
    e.neighbors("Bug:b1", "FIXES", "in")  # same card again: not a new distinct card
    assert e.stats() == {"calls": 5, "distinct_cards": 5, "max_hop": 2}


def test_non_strict_exposed_references_authorise_but_record_ids_do_not(g):
    e = ep(g, strict=False)
    e.get_card(PRODUCT)
    env = e.get_card("Bug:b1")  # exposed as a neighbour entry of the Product card, hop 1
    assert ids(env) == ["Bug:b1"] and e.stats()["max_hop"] == 1
    e.get_card(CUST)  # exposed by the Bug card's REPORTED_BY entry (the Product card does not list customers): hop 2
    assert e.stats()["max_hop"] == 2
    with pytest.raises(ToolError):
        e.get_card("mini_design_1")  # a record id in src, never an entity id
    with pytest.raises(ToolError):
        e.get_card("Person:eid_e5000005")  # never exposed


def test_rows_beyond_the_page_never_authorise(gm):
    e = ep(gm, strict=False)
    page1 = e.neighbors(PRODUCT, "HAS", "out", where={"label": "Document"})
    assert page1["truncated"] is True
    first = set(ids(page1))
    beyond = "Document:d0249"
    assert beyond not in first
    with pytest.raises(ToolError):
        e.get_card(beyond)
    assert first == set(e.received)  # only the rows the model got are counted


# ---------------------------------------------------------------- the atomic tool
def test_where_filters_on_card_fields_and_membership(g):
    e = ep(g)
    got = e.neighbors(PRODUCT, "HAS", "out", where={"label": "Bug"})
    assert ids(got) == ["Bug:b1", "Bug:b2", "Bug:b3", "Bug:b4", "Bug:b5"]
    got = e.neighbors(PRODUCT, "HAS", "out", where={"label": "Bug", "component": "Auth"})
    assert ids(got) == ["Bug:b5"]
    got = e.neighbors(PRODUCT, "HAS", "out", where={"id": ["Bug:b1", "PR:nope", PR101]})
    assert ids(got) == ["Bug:b1", PR101]
    got = e.neighbors(PRODUCT, "HAS", "out", where={"label": "Bug", "unresolved": True})
    assert ids(got) == ["Bug:b5"]  # nulls never match: the other bugs have no such field


def test_where_naming_an_edge_or_unknown_field_raises(g):
    e = ep(g)
    for bad in ({"neighbours": "x"}, {"FIXES": "Bug:b1"}, {"edge": "FIXES"}, {"src": "x"}, {"nonsense": 1},
                {"name": None}, {"name": {"a": 1}}):
        with pytest.raises(ToolError):
            e.neighbors(PRODUCT, "HAS", "out", where=bad)


def test_count_is_distinct_neighbours_over_all_matches(g):
    e = ep(g, strict=False)
    e.get_card(PRODUCT)  # exposes Feature:f1
    env = e.neighbors("Feature:f1", "CONTRIBUTED", "in", agg="count")
    assert env["count"] == 1 and env["rows"] == []  # two parallel edges, one person
    env = e.neighbors(PRODUCT, "HAS", "out", where={"label": "Bug"}, agg="count")
    assert env["count"] == 5


def test_group_counts_by_one_card_field(g):
    e = ep(g)
    env = e.neighbors(PRODUCT, "HAS", "out", agg="group", group_field="label")
    groups = {x["value"]: x["count"] for x in env["groups"]}
    assert groups["Bug"] == 5 and groups["Competitor"] == 1 and env["rows"] == []
    assert [x["count"] for x in env["groups"]] == sorted((x["count"] for x in env["groups"]), reverse=True)
    env = e.neighbors(PRODUCT, "HAS", "out", where={"label": "Bug"}, agg="group", group_field="component")
    assert env["groups"] == [{"value": None, "count": 4}, {"value": "Auth", "count": 1}]
    with pytest.raises(ToolError):
        e.neighbors(PRODUCT, "HAS", "out", agg="group")  # group needs a field
    with pytest.raises(ToolError):
        e.neighbors(PRODUCT, "HAS", "out", agg="group", group_field="neighbours")


def test_aggregates_do_not_authorise_or_count_cards(g):
    e = ep(g, strict=False)
    e.neighbors(PRODUCT, "HAS", "out", agg="count")
    assert e.stats()["distinct_cards"] == 0
    with pytest.raises(ToolError):
        e.get_card("Bug:b1")


def test_message_neighbours_are_returned_when_asked_for_explicitly(g):
    e2 = ep(g, "Who is eid_e5000005?", strict=True)
    e2.resolve("eid_e5000005")
    env = e2.neighbors("Person:eid_e5000005", "POSTED", "out")
    assert {r["label"] for r in env["rows"]} == {"Message"} and len(env["rows"]) == 4
    assert all(r["neighbours"] for r in env["rows"])  # a message card lists its own neighbours


def test_bad_edge_types_directions_and_unknown_roots_raise(g):
    e = ep(g)
    for args in ((PRODUCT, "COLLABORATED", "out"), (PRODUCT, "NOPE", "out"),
                 (PRODUCT, "HAS", "sideways"), (PRODUCT, "HAS; MATCH (n) DELETE n", "out")):
        with pytest.raises(ToolError):
            e.neighbors(*args)
    with pytest.raises(ToolError):
        e.neighbors(PRODUCT, "HAS", "out", agg="avg")


def test_a_card_never_includes_further_nodes_properties(g):
    e = ep(g)
    env = e.neighbors(PRODUCT, "HAS", "out", where={"label": "PR"})
    text = json.dumps(env)
    assert "Add a test" not in text and "Looks good" not in text  # review comments sit on edges, not on cards


# ---------------------------------------------------------------- caps, budget, cursor
def test_never_more_than_200_rows_and_never_more_than_the_budget(srv, load_facts):
    from conftest import fx_edge, fx_node
    from inherit.facts import Facts
    n = 260
    nodes = [fx_node("Competitor", f"c{i:03d}", name=f"c{i:03d}") for i in range(n)]
    edges = [fx_edge("HAS", "Product:mini", x.id) for x in nodes]
    g = load(srv, card_facts(load_facts, extra=Facts(nodes, edges)))
    e = ep(g)
    env = e.neighbors(PRODUCT, "HAS", "out", where={"label": "Competitor"})
    assert len(env["rows"]) == depth1.ROW_CAP == 200 and env["truncated"] is True and env["cursor"]
    env2 = e.neighbors(PRODUCT, "HAS", "out", where={"label": "Competitor"}, cursor=env["cursor"])
    assert len(env2["rows"]) == 61  # 260 generated competitors plus the fixture's Rival Corp, less the first 200
    assert env2["truncated"] is False and env2["cursor"] is None
    assert not set(ids(env)) & set(ids(env2))


def test_the_token_budget_bounds_a_response_and_the_cursor_pages_through(gm):
    e = ep(gm)
    kw = dict(root_id=PRODUCT, edge_type="HAS", direction="out", where={"label": "Document"})
    seen, cursor, pages = [], None, 0
    while True:
        env = e.neighbors(**kw, cursor=cursor)
        pages += 1
        assert depth1.tokens(env["rows"]) <= depth1.RESPONSE_BUDGET_TOKENS
        assert len(env["rows"]) < depth1.ROW_CAP  # the budget bites before the row cap here
        seen += ids(env)
        cursor = env["cursor"]
        if not env["truncated"]:
            break
        assert cursor and pages < 10
    assert pages >= 3 and seen == sorted(seen) and len(seen) == len(set(seen)) == 250 + 3  # 250 docs + fixture docs
    assert e.stats()["calls"] == pages  # a cursor call counts as a call


def test_count_and_group_run_over_all_matches_before_paging(gm):
    e = ep(gm)
    env = e.neighbors(PRODUCT, "HAS", "out", where={"label": "Document"}, agg="count")
    assert env["count"] == 253


def test_cursor_must_belong_to_the_same_request(gm):
    e = ep(gm)
    env = e.neighbors(PRODUCT, "HAS", "out", where={"label": "Document"})
    with pytest.raises(ToolError):
        e.neighbors(PRODUCT, "HAS", "out", where={"label": "Bug"}, cursor=env["cursor"])
    with pytest.raises(ToolError):
        e.neighbors(PRODUCT, "HAS", "out", where={"label": "Document"}, cursor="garbage")
    with pytest.raises(ToolError):
        e.neighbors(PRODUCT, "HAS", "out", where={"label": "Document"}, agg="count", cursor=env["cursor"])


def test_an_oversize_single_card_is_refused_with_an_error(srv, load_facts):
    from conftest import fx_edge, fx_node
    from inherit.facts import Facts
    big = fx_node("Document", "huge", id="huge", type="Huge", content="z" * 60000)
    g = load(srv, card_facts(load_facts, extra=Facts([big], [fx_edge("HAS", "Product:mini", big.id)])))
    e = ep(g, strict=False)
    e.get_card(PRODUCT)
    with pytest.raises(ToolError, match="budget"):
        e.get_card("Document:huge")
    with pytest.raises(ToolError, match="budget"):
        e.neighbors(PRODUCT, "HAS", "out", where={"id": "Document:huge"})
    assert "Document:huge" not in e.received


def test_groups_are_paged_under_both_caps(srv, load_facts):
    from conftest import fx_edge, fx_node
    from inherit.facts import Facts
    nodes = [fx_node("Competitor", f"g{i:03d}", name=f"g{i:03d}") for i in range(260)]
    edges = [fx_edge("HAS", "Product:mini", n.id) for n in nodes]
    e = ep(load(srv, card_facts(load_facts, extra=Facts(nodes, edges))))
    kw = dict(root_id=PRODUCT, edge_type="HAS", direction="out", where={"label": "Competitor"}, agg="group",
              group_field="name")
    env = e.neighbors(**kw)
    assert len(env["groups"]) == 200 and env["truncated"] is True and env["cursor"]
    env2 = e.neighbors(**kw, cursor=env["cursor"])
    assert len(env2["groups"]) == 61 and env2["truncated"] is False and env2["cursor"] is None
    assert e.stats()["calls"] == 2
    assert not {g["value"] for g in env["groups"]} & {g["value"] for g in env2["groups"]}


def test_an_oversized_group_value_is_refused(srv, load_facts):
    from conftest import fx_edge, fx_node
    from inherit.facts import Facts
    big = fx_node("Document", "wide", id="wide", type="Wide", content="y" * 60000)
    e = ep(load(srv, card_facts(load_facts, extra=Facts([big], [fx_edge("HAS", "Product:mini", big.id)]))))
    with pytest.raises(ToolError, match="budget"):
        e.neighbors(PRODUCT, "HAS", "out", where={"id": "Document:wide"}, agg="group", group_field="content")


def test_max_hop_never_decreases_on_rediscovery_by_a_shorter_route(g):
    e = ep(g, strict=False)
    e.get_card(PRODUCT)
    e.get_card("Bug:b1")  # hop 1
    e.get_card(CUST)  # hop 2 via the bug
    assert e.stats()["max_hop"] == 2
    e.neighbors(PRODUCT, "HAS", "out", where={"id": PR101})  # PR101 hop 1
    assert e.stats()["max_hop"] == 2
    e2 = ep(g, "Find TechCorp", strict=False)
    e2.get_card(PRODUCT)
    e2.get_card("Bug:b1")
    e2.get_card(CUST)  # received at hop 2
    e2.resolve("TechCorp")  # now a root at hop 0
    e2.get_card(CUST)  # re-received at hop 0
    assert e2.stats()["max_hop"] == 2 and e2.received[CUST] == 0


def test_the_13th_call_is_refused(g):
    e = ep(g)
    for _ in range(12):
        e.get_card(PRODUCT)
    with pytest.raises(ToolError, match="12"):
        e.get_card(PRODUCT)
    assert e.stats()["calls"] == 12


def test_refused_calls_count_against_the_budget(g):
    e = ep(g)
    for _ in range(12):
        with pytest.raises(ToolError):
            e.resolve("FusionTech")
    with pytest.raises(ToolError, match="12"):
        e.get_card(PRODUCT)


# ---------------------------------------------------------------- trace
def test_every_call_is_traced_with_its_cypher(g):
    e = ep(g)
    e.resolve("TechCorp")
    e.get_card(CUST)
    e.neighbors(CUST, "REPORTED_BY", "in")
    with pytest.raises(ToolError):
        e.get_card("Bug:b1")
    ops = e.trace
    assert all(isinstance(o, TraceOp) for o in ops)
    assert [o.tool for o in ops] == ["resolve", "get_card", "neighbors", "get_card"]
    assert [o.row_count for o in ops] == [1, 1, 3, 0]
    assert all(o.command == "GRAPH.RO_QUERY" for o in ops[:3])
    assert "REPORTED_BY" in ops[2].query and "MATCH" in ops[2].query
    assert ops[2].params["root_id"] == CUST and ops[2].ms >= 0
    assert ops[3].command == "refused"
    json.dumps([o.to_dict() for o in ops])


# ---------------------------------------------------------------- servers and users
def test_responses_are_identical_on_both_servers(srv_both, load_facts):
    facts = card_facts(load_facts, many=120)
    out = []
    for db in srv_both.values():
        e = ep(load(db, facts))
        r = e.neighbors(PRODUCT, "HAS", "out", where={"label": "Document"})
        out.append((json.dumps(r, sort_keys=True), json.dumps(e.resolve("TechCorp"))))
    assert out[0] == out[1]


def test_the_episode_works_through_the_read_only_agent_user(srv, load_facts, tmp_path):
    facts = card_facts(load_facts)
    admin = load(srv, facts)
    agent = agent_graph(srv, tmp_path)
    a, b = ep(admin), ep(agent)
    assert a.resolve("TechCorp") == b.resolve("TechCorp")
    assert a.neighbors(CUST, "REPORTED_BY", "in") == b.neighbors(CUST, "REPORTED_BY", "in")


def test_product_name_and_aliases_resolve_to_the_product(srv, load_facts):
    from conftest import fx_node
    from inherit.facts import Facts
    prod = fx_node("Product", "mini", name="mini", aliases=["MiniAlias"], previous_names=["OldMini"])
    g = load(srv, card_facts(load_facts, extra=Facts([prod], [])))
    e = ep(g, f"Which bugs of {P} or MiniAlias or OldMini?")
    for span in (P, "MiniAlias", "OldMini"):
        assert [r["id"] for r in e.resolve(span)["rows"]] == [PRODUCT]
    assert ids(e.get_card(PRODUCT)) == [PRODUCT]
