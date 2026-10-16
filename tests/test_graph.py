import json
import math
from collections import Counter

import pytest
import redis
from falkordb import FalkorDB

from conftest import TEST_PRODUCTS, fake_embed, fake_vec, fx_edge, fx_node
from inherit.facts import Facts
from inherit.graph import acl as acl_mod
from inherit.graph import load as loadmod
from inherit.graph import schema
from inherit.graph.udf_ref import status_at

PERM = redis.exceptions.NoPermissionError
G = "herb:mini"


def load(db, facts, product="mini", **kw):
    return loadmod.load_product(db, product, facts, embed=fake_embed, **kw)


def q(db, graph, text, params=None):
    return db.select_graph(graph).query(text, params).result_set


def conn(db, user, pw):
    kw = db.connection.connection_pool.connection_kwargs
    return redis.Redis(host=kw["host"], port=kw["port"], username=user, password=pw, decode_responses=True)


def canon(rows):
    return sorted(json.dumps(list(r), sort_keys=True, default=str) for r in rows)


def other_facts():
    return Facts([fx_node("Product", "other", name="other"), fx_node("Person", "x1", eid="x1", name="Xavier")],
                 [])


# ---------------------------------------------------------------- property mapping
def test_map_props_round_trip_and_drops_none():
    props = {"a": 1, "b": "s", "c": None, "d": {"k": [1, 2], "z": None}, "e": [1, "x", 2.5, True],
             "f": [{"p": 1}, {"q": 2}], "g": [[1], [2]], "h": []}
    mapped = loadmod.map_props(props)
    assert "c" not in mapped and "d" not in mapped
    assert mapped["d_json"] == json.dumps({"k": [1, 2], "z": None}, sort_keys=True)
    assert mapped["e"] == [1, "x", 2.5, True] and mapped["h"] == []
    assert isinstance(mapped["f_json"], str) and isinstance(mapped["g_json"], str)
    back = loadmod.unmap_props(mapped)
    assert back == {k: v for k, v in props.items() if v is not None}


def test_map_props_name_collision_is_an_error():
    with pytest.raises(ValueError):
        loadmod.map_props({"d": {"a": 1}, "d_json": "x"})


def test_node_id_is_reserved_and_record_id_is_source_id():
    nd = fx_node("Bug", "b9", id="b9", status="active")
    p = loadmod.node_props(nd)
    assert p["id"] == "Bug:b9" and p["source_id"] == "b9" and p["status"] == "active"


# ---------------------------------------------------------------- layer files
def test_layers_read_in_frozen_order_and_skip_names_report(tmp_path):
    d = tmp_path / "P"
    d.mkdir()
    Facts([fx_node("Person", "a", eid="a", name="First")], []).to_jsonl(d / "structural.jsonl")
    Facts([fx_node("Person", "a", eid="a", name="Second")], []).to_jsonl(d / "rules.jsonl")
    (d / "names.jsonl").write_text('{"not": "facts"}\n')
    assert loadmod.LAYER_FILES == ("structural.jsonl", "names_facts.jsonl", "rules.jsonl", "bugs.jsonl",
                                   "semantic.jsonl")
    f = loadmod.load_layers("P", tmp_path)
    assert [n.props["name"] for n in f.nodes] == ["First"]


# ---------------------------------------------------------------- load and counts
def test_load_counts_and_ids(falkor, load_facts):
    stats = load(falkor, load_facts)
    want_nodes = Counter(n.label for n in load_facts.nodes)
    want_edges = Counter(e.type for e in load_facts.edges)
    assert stats.nodes == dict(want_nodes) and stats.edges == dict(want_edges)
    assert want_nodes["Message"] == 18 and want_nodes["Bug"] == 4 and want_edges["HAS"] == 13
    got = dict(q(falkor, G, "MATCH (n) RETURN labels(n)[0], count(n)"))
    assert got == dict(want_nodes)
    got_e = dict(q(falkor, G, "MATCH ()-[e]->() RETURN type(e), count(e)"))
    assert got_e == dict(want_edges)
    # reserved id and source_id; edges carry provenance
    assert q(falkor, G, "MATCH (b:Bug {id:'Bug:b1'}) RETURN b.source_id") == [["b1"]]
    assert q(falkor, G, "MATCH (b:Bug {source_id:'b1'})-[:HAS_EVENT]->(e) RETURN count(e)") == [[2]]
    assert q(falkor, G, "MATCH (:Product)-[e:HAS]->(:Bug {id:'Bug:b1'}) RETURN e.src, e.extractor, e.key") == \
        [[["r1"], "fixture/v1", ""]]
    # parallel keyed edges stay parallel
    assert q(falkor, G, "MATCH (:Person)-[e:CONTRIBUTED]->(:Feature) RETURN e.key, e.weight ORDER BY e.key") == \
        [["0", 1], ["1", 2]]
    # json mapped props
    assert json.loads(q(falkor, G, "MATCH (o:Observation {id:'Observation:o1'}) RETURN o.detail_json")[0][0]) == \
        {"sev": 2, "tags": ["perf"]}
    assert q(falkor, G, "MATCH (o:Observation {id:'Observation:o1'}) RETURN o.nothing") == [[None]]
    assert set(stats.seconds) >= {"indexes", "nodes", "edges", "embed"}
    assert stats.embedded == {"Message": 18, "Document": 2}


def test_build_user_is_gone_after_load(falkor, load_facts):
    load(falkor, load_facts)
    assert "build_mini" not in falkor.connection.execute_command("ACL", "USERS")


def test_reference_queries_have_explicit_answers(falkor, load_facts):
    load(falkor, load_facts)
    # 1. counts by label
    rows = q(falkor, G, "MATCH (n:Bug) RETURN n.status, count(n)")
    assert canon(rows) == canon([["resolved", 2], ["active", 2]])
    # 2. two-hop join
    per_channel = Counter()
    chan = {e.src_id: e.dst_id for e in load_facts.edges if e.type == "IN_CHANNEL"}
    for e in load_facts.edges:
        if e.type == "POSTED":
            per_channel[chan[e.dst_id]] += 1
    rows = q(falkor, G, "MATCH (p:Person)-[:POSTED]->(m:Message)-[:IN_CHANNEL]->(c:Channel) RETURN c.id, count(m)")
    assert canon(rows) == canon([[k, v] for k, v in per_channel.items()])
    # 3. OPTIONAL MATCH ... WITH ... WHERE: matched (b1, b2), unmatched (b3), filtered out (b4)
    rows = q(falkor, G, "MATCH (b:Bug) OPTIONAL MATCH (b)-[:HAS_EVENT]->(e:BugEvent) WITH b, e "
                        "WHERE e IS NULL OR e.type = 'RESOLVED' RETURN b.source_id, e.type")
    assert canon(rows) == canon([["b1", "RESOLVED"], ["b2", "RESOLVED"], ["b3", None]])


# ---------------------------------------------------------------- constraints
def test_constraints_operational(falkor, load_facts):
    load(falkor, load_facts)
    rows = {(r[0], r[1], tuple(r[2])): r[4] for r in q(falkor, G, "CALL db.constraints()")}
    assert rows == {("UNIQUE", "Person", ("eid",)): "OPERATIONAL", ("UNIQUE", "Bug", ("id",)): "OPERATIONAL",
                    ("UNIQUE", "PR", ("link",)): "OPERATIONAL", ("MANDATORY", "Bug", ("status",)): "OPERATIONAL"}


def with_nodes(facts, *nodes):
    return facts.merge(Facts(list(nodes), []))


def dup_facts(facts, *nodes):
    has = [fx_edge("HAS", "Product:mini", n.id) for n in nodes if n.label in loadmod.ROOTED]
    return Facts([*facts.nodes, *nodes], [*facts.edges, *has])


def test_duplicate_person_eid_fails(falkor, load_facts):
    bad = dup_facts(load_facts, fx_node("Person", "dup", eid="eid_a1000001", name="Dup"))
    with pytest.raises(Exception, match="(?i)unique constraint"):
        load(falkor, bad)
    assert "build_mini" not in falkor.connection.execute_command("ACL", "USERS")


def test_duplicate_bug_id_fails(falkor, load_facts):
    bad = dup_facts(load_facts, fx_node("Bug", "b1", id="b1", status="active"))
    with pytest.raises(Exception, match="(?i)unique constraint"):
        load(falkor, bad)


def test_duplicate_pr_link_fails(falkor, load_facts):
    pr = next(n for n in load_facts.nodes if n.label == "PR")
    bad = dup_facts(load_facts, fx_node("PR", "copy", link=pr.props["link"]))
    with pytest.raises(Exception, match="(?i)unique constraint"):
        load(falkor, bad)


def test_bug_without_status_fails(falkor, load_facts):
    bad = dup_facts(load_facts, fx_node("Bug", "nostatus", id="nostatus"))
    with pytest.raises(Exception, match="(?i)mandatory constraint"):
        load(falkor, bad)


# ---------------------------------------------------------------- Product root rule
def test_missing_root_is_refused_and_nothing_is_written(falkor, load_facts):
    bad = Facts([n for n in load_facts.nodes if n.label != "Product"],
                [e for e in load_facts.edges if e.type != "HAS"])
    with pytest.raises(ValueError, match="Product"):
        load(falkor, bad)
    assert G not in falkor.list_graphs()


def test_two_roots_refused(falkor, load_facts):
    with pytest.raises(ValueError, match="Product"):
        load(falkor, with_nodes(load_facts, fx_node("Product", "again", name="again")))


def test_root_must_have_every_has_edge(falkor, load_facts):
    edges = [e for e in load_facts.edges if not (e.type == "HAS" and e.dst_id == "Bug:b3")]
    with pytest.raises(ValueError, match="Bug:b3"):
        load(falkor, Facts(load_facts.nodes, edges))


def test_root_name_must_match_product(falkor, load_facts):
    with pytest.raises(ValueError, match="Product:other"):
        load(falkor, load_facts, product="other")


# ---------------------------------------------------------------- embeddings and indexes
def test_embedding_dimension_and_finiteness_checked_before_any_write(falkor, load_facts):
    def short(texts):
        return [[0.1] * 1023 for _ in texts]

    with pytest.raises(ValueError, match="1024"):
        loadmod.load_product(falkor, "mini", load_facts, embed=short)
    assert G not in falkor.list_graphs()

    def nan(texts):
        return [[float("nan")] * 1024 for _ in texts]

    with pytest.raises(ValueError, match="finite"):
        loadmod.load_product(falkor, "mini", load_facts, embed=nan)
    assert G not in falkor.list_graphs()


def test_vector_retrieval_from_message_and_document_indexes(falkor, load_facts):
    load(falkor, load_facts)
    for label, prop in (("Message", "text"), ("Document", "content")):
        texts = {n.id: n.props[prop] for n in load_facts.nodes if n.label == label}
        probe_id, probe_text = next(iter(texts.items()))
        rows = q(falkor, G, f"CALL db.idx.vector.queryNodes('{label}', 'embedding', 2, vecf32($q)) "
                            "YIELD node, score RETURN node.id, score", {"q": fake_vec(probe_text)})
        assert rows[0][0] == probe_id and abs(rows[0][1]) < 1e-3
        assert len({r[0] for r in rows}) == 2
    assert q(falkor, G, "MATCH (m:Message) WHERE m.embedding IS NOT NULL RETURN count(m)") == [[18]]


def test_vector_top5_overlap_between_versions(falkor_both, load_facts):
    tops = {}
    for ver, db in falkor_both.items():
        load(db, load_facts)
        tops[ver] = {}
        for i, n in enumerate(n for n in load_facts.nodes if n.label == "Message"):
            if i >= 5:
                break
            rows = q(db, G, "CALL db.idx.vector.queryNodes('Message', 'embedding', 5, vecf32($q)) "
                            "YIELD node, score RETURN node.id", {"q": fake_vec(n.props["text"])})
            ids = [r[0] for r in rows]
            assert len(set(ids)) == 5 and ids[0] == n.id
            tops[ver][n.id] = set(ids)
    a, b = tops.values()
    assert all(len(a[k] & b[k]) >= 4 for k in a)


def test_fulltext_entry_points(falkor, load_facts):
    load(falkor, load_facts)
    assert q(falkor, G, "CALL db.idx.fulltext.queryNodes('Bug', 'login') YIELD node RETURN node.source_id") == [["b1"]]
    assert q(falkor, G, "CALL db.idx.fulltext.queryNodes('Person', 'Alice') YIELD node RETURN node.eid") == \
        [["eid_a1000001"]]
    assert q(falkor, G, "CALL db.idx.fulltext.queryNodes('Competitor', 'rival') YIELD node RETURN node.name") == \
        [["Rival Corp"]]
    assert q(falkor, G, "CALL db.idx.fulltext.queryNodes('Customer', 'x') YIELD node RETURN node") is not None


def test_all_indexes_operational(falkor, load_facts):
    load(falkor, load_facts)
    rows = q(falkor, G, "CALL db.indexes()")
    assert rows and all("OPERATIONAL" in [str(x) for x in r] for r in rows)


# ---------------------------------------------------------------- ACL
def prov(db, tmp_path, products=("mini",)):
    return acl_mod.setup_acl(db, list(products), path=tmp_path / "acl.json")


def test_acl_requires_graph_first(falkor, tmp_path):
    with pytest.raises(RuntimeError, match="herb:mini"):
        prov(falkor, tmp_path)


def test_agent_acl(falkor, load_facts, tmp_path):
    load(falkor, load_facts)
    load(falkor, other_facts(), product="other")
    pw = prov(falkor, tmp_path, ("mini", "other"))
    a = conn(falkor, "agent_mini", pw["agent_mini"])
    assert a.execute_command("GRAPH.RO_QUERY", G, "MATCH (n:Person) RETURN count(n)")[1] == [[8]]
    with pytest.raises(PERM):
        a.execute_command("GRAPH.RO_QUERY", "herb:other", "MATCH (n) RETURN count(n)")
    with pytest.raises(PERM):
        a.execute_command("GRAPH.QUERY", G, "CREATE (:Hack)")
    with pytest.raises((PERM, redis.ResponseError)) as ei:
        a.execute_command("GRAPH.RO_QUERY", G, "CREATE (:Hack)")
    assert "permission" in str(ei.value).lower() or "read-only" in str(ei.value).lower() \
        or "read only" in str(ei.value).lower()
    for cmd in (("GRAPH.DELETE", G), ("GRAPH.COPY", G, "whatif:mini:1"), ("GRAPH.UDF", "LIST")):
        with pytest.raises(PERM):
            a.execute_command(*cmd)
    assert q(falkor, G, "MATCH (h:Hack) RETURN count(h)") == [[0]]
    assert q(falkor, G, "MATCH (n:Person) RETURN count(n)") == [[8]]


def test_whatif_acl(falkor, load_facts, tmp_path):
    load(falkor, load_facts)
    load(falkor, other_facts(), product="other")
    pw = prov(falkor, tmp_path, ("mini", "other"))
    w = conn(falkor, "whatif_mini", pw["whatif_mini"])
    before = q(falkor, G, "MATCH (n) RETURN count(n)")
    assert w.execute_command("GRAPH.RO_QUERY", G, "MATCH (n:Person) RETURN count(n)")[1] == [[8]]
    w.execute_command("GRAPH.COPY", G, "whatif:mini:1")
    assert w.execute_command("GRAPH.RO_QUERY", "whatif:mini:1", "MATCH (n:Person) RETURN count(n)")[1] == [[8]]
    w.execute_command("GRAPH.QUERY", "whatif:mini:1", "MATCH (n:Bug) DETACH DELETE n")
    assert w.execute_command("GRAPH.RO_QUERY", "whatif:mini:1", "MATCH (n:Bug) RETURN count(n)")[1] == [[0]]
    # the source is untouched and cannot be written or deleted
    assert q(falkor, G, "MATCH (n:Bug) RETURN count(n)") == [[4]]
    for cmd in (("GRAPH.QUERY", G, "CREATE (:Hack)"), ("GRAPH.DELETE", G),
                ("GRAPH.COPY", G, "whatif:other:1"), ("GRAPH.COPY", "herb:other", "whatif:mini:2"),
                ("GRAPH.RO_QUERY", "herb:other", "MATCH (n) RETURN count(n)")):
        with pytest.raises(PERM):
            w.execute_command(*cmd)
    with pytest.raises(redis.ResponseError):  # destination exists
        w.execute_command("GRAPH.COPY", "whatif:mini:1", G)
    w.execute_command("GRAPH.DELETE", "whatif:mini:1")
    assert q(falkor, G, "MATCH (n) RETURN count(n)") == before
    assert q(falkor, "herb:other", "MATCH (n) RETURN count(n)") == [[2]]


def test_reprovision_over_privileged_user(falkor, load_facts, tmp_path):
    load(falkor, load_facts)
    r = falkor.connection
    for u in ("agent_mini", "whatif_mini"):
        r.execute_command("ACL", "SETUSER", u, "reset", "on", ">old", "+@all", "~*")
    prov(falkor, tmp_path)

    def flat(x):
        return x if isinstance(x, dict) else {x[i]: x[i + 1] for i in range(0, len(x), 2)}

    def sel(u):
        d = flat(r.execute_command("ACL", "GETUSER", u))
        return d["commands"], d["keys"], [(flat(x)["commands"], flat(x)["keys"]) for x in d["selectors"]]

    c, k, s = sel("agent_mini")
    assert c == "-@all" and k == ""
    assert s == [("-@all +graph.ro_query", "~herb:mini")]
    c, k, s = sel("whatif_mini")
    assert c == "-@all" and k == ""
    assert s == [
        ("-@all +graph.copy", "~whatif:mini:* ~herb:mini"),
        ("-@all +graph.query +graph.ro_query +graph.delete", "~whatif:mini:*"),
        ("-@all +graph.ro_query", "~herb:mini")]


def test_acl_file_is_private_and_merged(falkor, load_facts, tmp_path):
    load(falkor, load_facts)
    load(falkor, other_facts(), product="other")
    pw1 = prov(falkor, tmp_path, ("mini",))
    prov(falkor, tmp_path, ("other",))
    p = tmp_path / "acl.json"
    assert (p.stat().st_mode & 0o777) == 0o600
    saved = json.loads(p.read_text())
    assert saved["local"]["agent_mini"] == pw1["agent_mini"] and "whatif_other" in saved["local"]


def test_reload_removes_users_and_whatif_copies_before_dropping(falkor, load_facts, tmp_path):
    load(falkor, load_facts)
    pw = prov(falkor, tmp_path)
    w = conn(falkor, "whatif_mini", pw["whatif_mini"])
    w.execute_command("GRAPH.COPY", G, "whatif:mini:1")
    load(falkor, load_facts)
    users = falkor.connection.execute_command("ACL", "USERS")
    assert not {"agent_mini", "whatif_mini", "build_mini"} & set(users)
    assert "whatif:mini:1" not in falkor.list_graphs()


def test_failed_load_cleans_up_build_user_and_half_graph(falkor, load_facts, tmp_path):
    load(falkor, load_facts)
    prov(falkor, tmp_path)
    bad = dup_facts(load_facts, fx_node("Bug", "b1", id="b1", status="active"))
    with pytest.raises(Exception):
        load(falkor, bad)
    users = falkor.connection.execute_command("ACL", "USERS")
    assert not {"agent_mini", "whatif_mini", "build_mini"} & set(users)
    assert G not in falkor.list_graphs()


# ---------------------------------------------------------------- UDF
UDF_CASES = [
    ([], 5), ([("REPORTED", 5)], 4), ([("REPORTED", 5)], 5), ([("REPORTED", 1), ("ASSIGNED", 2)], 3),
    ([("REPORTED", 1), ("DISMISSED", 2)], 3), ([("REPORTED", 1), ("DISMISSED", 2), ("REOPENED", 3)], 9),
    ([("REPORTED", 1), ("RESOLVED", 2)], 9), ([("REPORTED", 1), ("RESOLVED", 8)], 5),
    ([("REPORTED", 5), ("RESOLVED", 5)], 5), ([("RESOLVED", 5), ("REPORTED", 5)], 5),
    ([("RESOLVED", 3), ("REPORTED", 1), ("REOPENED", 5)], 4), ([("RESOLVED", 3), ("REPORTED", 1), ("REOPENED", 5)], 6),
    ([("REPORTED", "2024-01-01"), ("RESOLVED", "2024-02-01")], "2024-01-15"),
    ([("REPORTED", 1), ("WEIRD", 2)], 3),
]


def lit(events):
    return "[" + ", ".join(f"{{type:'{t}', ts:{ts if isinstance(ts, int) else repr(ts)}}}" for t, ts in events) + "]"


def test_status_at_reference_values():
    assert status_at([], 5) is None
    assert status_at([("RESOLVED", 5), ("REPORTED", 5)], 5) == "active"
    assert status_at([("REPORTED", 1), ("WEIRD", 2)], 3) == "active"


def test_udf_agrees_with_python_reference(falkor, load_facts):
    load(falkor, load_facts)
    schema.load_udf(falkor)
    for events, date in UDF_CASES:
        got = q(falkor, G, f"RETURN herb_v1.statusAt({lit(events)}, $d)", {"d": date})[0][0]
        assert got == status_at(events, date), (events, date)


def test_udf_active_before_resolved_after(falkor, load_facts):
    load(falkor, load_facts)
    schema.load_udf(falkor)
    for date, want in (("2026-06-15T00:00:00", "active"), ("2026-06-20T09:00:00", "resolved"),
                       ("2026-06-01T00:00:00", None)):
        got = q(falkor, G, "MATCH (b:Bug {source_id:'b1'})-[:HAS_EVENT]->(e:BugEvent) WITH b, e "
                           "ORDER BY e.ts, e.position WITH b, collect({type:e.type, ts:e.ts}) AS evs "
                           "RETURN herb_v1.statusAt(evs, $d)", {"d": date})[0][0]
        assert got == want


def test_udf_repeated_load_ok_and_mismatch_fails(falkor):
    schema.load_udf(falkor)
    schema.load_udf(falkor)
    schema.load_udf(falkor, name="herb_test", source="function f() { return 1; }\nfalkor.register('f', f);")
    with pytest.raises(RuntimeError, match="herb_test"):
        schema.load_udf(falkor, name="herb_test")


# ---------------------------------------------------------------- betweenness (6.0.2 only, hand-calculated)
def test_betweenness_hand_calculated(falkor):
    if falkor.version_id != "6.0":
        pytest.skip("4.22.1 gives wrong directed betweenness; Task 8 computes it in Python there")
    g = falkor.select_graph("herb:mini")
    g.query("CREATE (a:P {n:'a'})-[:C]->(b:P {n:'b'})-[:C]->(c:P {n:'c'})-[:C]->(d:P {n:'d'})-[:C]->(e:P {n:'e'})")
    rows = g.query("CALL algo.betweenness({nodeLabels:['P'], relationshipTypes:['C']}) YIELD node, score "
                   "RETURN node.n, score").result_set
    assert {k: v for k, v in rows} == {"a": 0.0, "b": 3.0, "c": 4.0, "d": 3.0, "e": 0.0}
