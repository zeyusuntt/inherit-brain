import pytest

from inherit.facts import Edge, Facts, Node
from inherit.ontology import check_edge


def n(id_, **props):
    return Node(id_.split(":")[0], id_, props)


def e(src, dst, type_="AUTHORED", key="", ids=("r1",), **props):
    return Edge(type_, src, dst, props, tuple(ids), "t/v1", key)


def test_merge_unions_src_and_counts_conflict():
    a = Facts([n("Person:p1", name="A", role="x")], [e("Person:p1", "Document:d1", ids=("r1",))])
    b = Facts([n("Person:p1", name="B", extra=1)], [e("Person:p1", "Document:d1", ids=("r2", "r1"))])
    m = a.merge(b)
    assert m.conflicts == 1
    assert len(m.nodes) == 1 and m.nodes[0].props == {"name": "A", "role": "x", "extra": 1}
    assert len(m.edges) == 1 and m.edges[0].src == ("r1", "r2")


def test_merge_edge_key_discriminator_keeps_parallel_edges():
    a = Facts([], [e("Person:p1", "PR:l", "REVIEWED", key="0"), e("Person:p1", "PR:l", "REVIEWED", key="1")])
    assert len(Facts([], []).merge(a).edges) == 2


def test_merge_edge_prop_conflict_counted():
    a = Facts([], [e("Person:p1", "Document:d1", w=1)])
    b = Facts([], [e("Person:p1", "Document:d1", w=2)])
    m = a.merge(b)
    assert m.conflicts == 1 and m.edges[0].props == {"w": 1}


def test_jsonl_round_trip_is_byte_identical(tmp_path):
    f = Facts(
        [n("Person:p2", name="Zed é"), n("Person:p1", name="A", list=[1, 2])],
        [e("Person:p2", "Document:d1", ids=("b", "a")), e("Person:p1", "PR:l", "REVIEWED", key="1", state="X")],
    )
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    f.to_jsonl(p1)
    g = Facts.from_jsonl(p1)
    g.to_jsonl(p2)
    assert p1.read_bytes() == p2.read_bytes()
    assert [x.id for x in g.nodes] == ["Person:p1", "Person:p2"]


def test_check_edge():
    check_edge(e("Person:p1", "Document:d1", "AUTHORED"))
    with pytest.raises(ValueError):
        check_edge(e("Document:d1", "Person:p1", "AUTHORED"))
    with pytest.raises(ValueError):
        check_edge(e("Person:p1", "Document:d1", "NOPE"))


# ---- union-merged list properties (R6) ----

def test_union_props_merge_without_conflict_and_keep_order():
    a = Facts([n("Product:p", aliases=["A", "B"], details=["x"], descriptions=["d1"], name="p")], [])
    b = Facts([n("Product:p", aliases=["B", "C"], details=["y", "x"], descriptions=["d2"], name="p")], [])
    m = a.merge(b)
    assert m.conflicts == 0
    p = m.nodes[0].props
    assert p["aliases"] == ["A", "B", "C"] and p["details"] == ["x", "y"] and p["descriptions"] == ["d1", "d2"]


def test_union_props_copy_when_missing_and_wrap_scalars():
    m = Facts([n("Product:p", name="p")], []).merge(Facts([n("Product:p", aliases=["A"])], []))
    assert m.nodes[0].props["aliases"] == ["A"] and m.conflicts == 0
    m2 = Facts([n("Product:p", aliases="A")], []).merge(Facts([n("Product:p", aliases=["B"])], []))
    assert m2.nodes[0].props["aliases"] == ["A", "B"]


def test_union_props_keep_unhashable_items_once():
    a = Facts([n("Url:u", details=[{"k": 1}])], [])
    m = a.merge(Facts([n("Url:u", details=[{"k": 1}, {"k": 2}])], []))
    assert m.nodes[0].props["details"] == [{"k": 1}, {"k": 2}]


def test_other_list_props_stay_first_value_wins():
    a = Facts([n("Product:p", previous_names=["X"])], [])
    m = a.merge(Facts([n("Product:p", previous_names=["Y"])], []))
    assert m.nodes[0].props["previous_names"] == ["X"] and m.conflicts == 1


def test_union_props_also_apply_to_edges():
    a = Facts([], [e("Person:p1", "Document:d1", aliases=["A"])])
    m = a.merge(Facts([], [e("Person:p1", "Document:d1", aliases=["B"])]))
    assert m.edges[0].props["aliases"] == ["A", "B"] and m.conflicts == 0
