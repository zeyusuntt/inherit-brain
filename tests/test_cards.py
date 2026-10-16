import json

import pytest

from cardfx import PRODUCT, P, srv, srv_both, LONG, PR101, URL1, agent_graph, card_facts, load
from inherit.graph import cards

BUG1 = "Bug:b1"


def entry(card, **kw):
    return [e for e in card["neighbours"] if all(e.get(k) == v for k, v in kw.items())]


@pytest.fixture
def g(srv, load_facts):
    return load(srv, card_facts(load_facts))


def test_card_shape_and_key_order(g):
    c = cards.card(g, BUG1)
    assert list(c) == ["id", "label", "name", "props", "neighbours"]
    assert (c["id"], c["label"], c["name"]) == (BUG1, "Bug", "Login page crashes on Safari")
    for e in c["neighbours"]:
        assert list(e) == ["edge", "dir", "label", "id", "name", "src"]
    keys = [(e["edge"], e["dir"], e["label"], e["id"]) for e in c["neighbours"]]
    assert keys == sorted(keys)


def test_bug_card_has_customer_id_and_company_but_not_contact_role(g):
    c = cards.card(g, BUG1)
    (cust,) = entry(c, edge="REPORTED_BY", dir="out")
    assert (cust["id"], cust["label"], cust["name"]) == ("Customer:CUST-0001", "Customer", "TechCorp")
    text = json.dumps(c)
    assert "CTO" not in text and "Tom Hill" not in text  # the customer's role and contact name stay on its own card
    assert entry(c, edge="FIXES", dir="in")[0]["id"] == PR101
    assert "merged" not in text  # the PR's own properties are not copied into the bug card


def test_bug_card_props_follow_the_allowlist(g):
    c = cards.card(g, "Bug:b5")
    assert c["props"] == {"title": "Users can't log in", "component": "Auth", "aliases": ["login failure"],
                          "unresolved": True, "active": True, "reopened": False, "no_fix_needed": False}
    assert "resolver" not in c["props"] and "status" not in c["props"]  # no copied relationship answers


def test_own_props_only_allowlisted_and_never_embedding(g):
    for label, nid in (("Document", "Document:mini_design_spec"), ("Message", "Message:20260610-0-a0002"),
                       ("Meeting", "Meeting:mini_design_1"), ("PR", PR101), ("Person", "Person:eid_e5000005")):
        props = cards.card(g, nid)["props"]
        assert set(props) <= set(cards.ALLOWLIST[label]), label
        assert "embedding" not in props
    assert "author" not in cards.card(g, "Document:mini_design_spec")["props"]
    assert "participants" not in cards.card(g, "Meeting:mini_design_1")["props"]


def test_values_are_logical_types_not_storage_strings(g):
    p = cards.card(g, PR101)["props"]
    assert p["merged"] is True and p["title"] == "Change 101"


def test_json_storage_properties_decode_before_the_allowlist(g, monkeypatch):
    monkeypatch.setitem(cards.ALLOWLIST, "Observation", cards.ALLOWLIST["Observation"] + ("detail", "pairs"))
    p = cards.card(g, "Observation:o1")["props"]
    assert p["detail"] == {"sev": 2, "tags": ["perf"]} and p["pairs"] == [{"a": 1}, {"b": 2}]
    assert "detail_json" not in p


def test_product_card_lists_only_six_labels_no_messages_no_counts(g):
    c = cards.card(g, PRODUCT)
    assert set(c["props"]) <= {"name", "aliases"}
    assert {e["label"] for e in c["neighbours"]} <= {"Bug", "Document", "Meeting", "Feature", "PR", "Competitor"}
    assert {e["label"] for e in c["neighbours"]} == {"Bug", "Document", "Meeting", "Feature", "PR", "Competitor"}
    assert all(e["edge"] == "HAS" and e["dir"] == "out" for e in c["neighbours"])
    text = json.dumps(c)
    assert "Message" not in text and "count" not in text and "degree" not in text


def test_person_card_omits_message_neighbours_but_message_card_lists_poster(g):
    person = cards.card(g, "Person:eid_e5000005")
    assert "Message" not in {e["label"] for e in person["neighbours"]}
    assert entry(person, edge="AUTHORED")[0]["id"] == "Document:mini_design_spec"
    msg = cards.card(g, "Message:20260610-0-a0002")
    assert entry(msg, edge="POSTED", dir="in")[0]["id"] == "Person:eid_e5000005"
    assert entry(msg, edge="IN_CHANNEL", dir="out")[0]["id"] == "Channel:ch-mini-plan"
    assert {e["id"] for e in entry(msg, edge="MENTIONS")} == {"Person:eid_a1000001", "Person:eid_f6000006"}
    chan = cards.card(g, "Channel:ch-mini-plan")
    assert chan["neighbours"] == []  # a channel's only neighbours are messages


def test_collaborated_is_never_in_a_card(g):
    c = cards.card(g, "Person:eid_c3000003")
    assert "COLLABORATED" not in {e["edge"] for e in c["neighbours"]}
    assert entry(c, edge="ATTENDED")


def test_src_lists_at_most_three_sorted_ids_then_an_ellipsis(g):
    c = cards.card(g, "Person:eid_e5000005")
    assert entry(c, edge="SHARED", id=URL1)[0]["src"] == ["rec0", "rec1", "rec2", "..."]
    pr = cards.card(g, PR101)
    assert entry(pr, edge="SHARED")[0]["src"] == ["20260615-0-b0001", "20260618-0-c0003"]  # two ids: no marker


def test_parallel_edges_become_one_entry_with_the_union_of_src(g):
    c = cards.card(g, "Feature:f1")
    es = entry(c, edge="CONTRIBUTED", dir="in")
    assert len(es) == 1 and es[0]["id"] == "Person:eid_a1000001"


def test_document_text_is_never_truncated(g):
    c = cards.card(g, "Document:long_doc")
    assert c["props"]["content"] == LONG and len(c["props"]["content"]) == 9000
    assert "[truncated]" not in json.dumps(c)


def test_display_names_are_identifiers_not_previews(g):
    assert cards.card(g, "Message:20260610-0-a0002")["name"] == "20260610-0-a0002"
    assert cards.card(g, "Observation:o1")["name"] == "o1"
    assert cards.card(g, "BugEvent:b1#0")["name"] == "b1#0"
    assert cards.card(g, "FeatureEvent:f1#0")["name"] == "f1#0"
    assert cards.card(g, URL1)["name"] == "https://docs.example.com/export"
    assert cards.card(g, "Document:mini_design_spec")["name"] == "Design Spec (mini_design_spec)"
    assert cards.card(g, "Meeting:mini_design_1")["name"] == "Design Spec meeting (mini_design_1)"
    assert cards.card(g, "Customer:CUST-0001")["name"] == "TechCorp"
    assert cards.card(g, PR101)["name"] == "Change 101"
    assert cards.card(g, "Person:eid_e5000005")["name"] == "Dana Wu"
    assert cards.card(g, "Person:EMP_555000111")["name"] == "EMP_555000111"  # nameless person: the eid
    assert cards.card(g, "Competitor:Rival Corp")["name"] == "Rival Corp"
    assert cards.card(g, "Channel:ch-mini-bug")["name"] == "bug-alicechen-mini"
    assert cards.card(g, PRODUCT)["name"] == P
    assert cards.card(g, "Feature:f1")["name"] == "Bulk export"


def test_neighbour_names_use_the_same_rule(g):
    c = cards.card(g, "Observation:o1")
    assert entry(c, edge="STATED", dir="in")[0]["name"] == "Bob Ray"
    assert entry(c, edge="ABOUT", label="Competitor")[0]["name"] == "Rival Corp"
    f = cards.card(g, "Feature:f1")
    assert entry(f, edge="HAS_EVENT")[0]["name"] == "f1#0"


def test_a_neighbours_other_properties_never_appear(g):
    c = cards.card(g, "Observation:o1")
    assert "Export is too slow" in json.dumps(c["props"])
    f = cards.card(g, "Feature:f1")
    assert "Export is too slow" not in json.dumps(f)  # the observation's text is not in the feature card


def test_every_node_has_a_card_in_deterministic_order(g):
    got = list(cards.cards_for_product(g))
    ids = [c["id"] for c in got]
    (n,) = g.ro_query("MATCH (n) RETURN count(n)").result_set[0]
    assert len(ids) == n and len(set(ids)) == n
    assert ids == sorted(ids, key=lambda i: (i.split(":", 1)[0], i))
    assert any(c["label"] == "Message" for c in got)  # the A0f corpus includes messages
    assert got == list(cards.cards_for_product(g))


def test_unknown_node_raises(g):
    with pytest.raises(LookupError):
        cards.card(g, "Bug:nope")
    with pytest.raises(ValueError):
        cards.card(g, "Nothing:x")


def test_cards_are_identical_on_both_servers(srv_both, load_facts):
    facts = card_facts(load_facts)
    out = {}
    for ver, db in srv_both.items():
        out[ver] = list(cards.cards_for_product(load(db, facts)))
    a, b = out.values()
    assert a == b and json.dumps(a) == json.dumps(b)


def test_cards_read_through_the_agent_user(srv, load_facts, tmp_path):
    facts = card_facts(load_facts)
    admin = load(srv, facts)
    agent = agent_graph(srv, tmp_path)
    assert list(cards.cards_for_product(agent)) == list(cards.cards_for_product(admin))


def test_allowlist_covers_every_ontology_label_and_bans_internal_fields():
    from inherit.ontology import NODE_KEYS
    assert set(cards.ALLOWLIST) == set(NODE_KEYS)
    for label, fields in cards.ALLOWLIST.items():
        assert not any(f.endswith("_json") or f in ("embedding", "extractor", "key", "source_id") for f in fields)
    assert cards.CARD_VERSION == "v1"
    assert cards.SRC_CAP == 3 and "COLLABORATED" in cards.EXCLUDED_EDGES


def test_policy_toggles_measure_the_exclusions(g):
    base = cards.card(g, "Person:eid_e5000005")
    with_msgs = cards.card(g, "Person:eid_e5000005", policy=cards.Policy(messages=True))
    assert "Message" in {e["label"] for e in with_msgs["neighbours"]} and with_msgs != base
    full_src = cards.card(g, "Person:eid_e5000005", policy=cards.Policy(src_cap=None))
    assert entry(full_src, edge="SHARED", id=URL1)[0]["src"] == [f"rec{i}" for i in range(5)]
    collab = cards.card(g, "Person:eid_c3000003", policy=cards.Policy(collaborated=True))
    assert "COLLABORATED" in {e["edge"] for e in collab["neighbours"]}


def test_doc_fields_section_is_generated_from_code():
    from pathlib import Path
    doc = (Path(__file__).parents[1] / "docs" / "cards.md").read_text()
    assert cards.render_fields() in doc
    for label in cards.ALLOWLIST:
        assert f"### {label}" in cards.render_fields()
    assert "v1" in doc and "12,000" in doc and "deviation" in doc.lower()
