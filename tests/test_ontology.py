import pytest

from inherit.facts import Edge
from inherit.ontology import NODE_KEYS, SIGNATURES, check_edge


def edge(type_, src, dst):
    return Edge(type_, src, dst, {}, ("r1",), "t/v1", "")


# Every signature the Task 4 design (4a, 4b, 4c) uses. Each has a reason comment in ontology.py.
DESIGN = [
    ("HAS", "Product", t) for t in ("Bug", "Document", "Meeting", "Feature", "PR", "Competitor")
] + [
    ("REVERTS", "PR", "PR"),
    ("ABOUT", "PR", "Feature"), ("ABOUT", "Url", "Competitor"), ("ABOUT", "Url", "Product"),
    ("ABOUT", "Observation", "Competitor"), ("ABOUT", "Observation", "Product"),
    ("ABOUT", "Observation", "Feature"), ("ABOUT", "Observation", "Bug"), ("ABOUT", "Observation", "Person"),
    ("ABOUT", "Message", "PR"),
    ("STATED", "Person", "Observation"),
    ("REPORTED_BY", "Bug", "Customer"), ("REPORTED_BY", "Observation", "Customer"),
    ("REPORTED_BY", "Observation", "Person"),
    ("CONTRIBUTED", "Person", "Feature"),
    ("SUGGESTED", "Person", "Document"),
    ("HAS_EVENT", "Bug", "BugEvent"), ("HAS_EVENT", "Feature", "FeatureEvent"),
    ("IMPLEMENTED_BY", "Feature", "PR"),
    ("DISCUSSES", "Meeting", "Feature"), ("DISCUSSES", "Meeting", "Document"),
    ("DISCUSSES", "Message", "Feature"), ("DISCUSSES", "Message", "Document"),
    ("ASSIGNED_TO", "BugEvent", "Person"), ("ACTED_BY", "BugEvent", "Person"),
    ("FIXES", "PR", "Bug"), ("SIMILAR_TO", "ExternalBug", "Bug"),
]


@pytest.mark.parametrize("type_,src,dst", DESIGN)
def test_design_signature_is_registered(type_, src, dst):
    check_edge(edge(type_, f"{src}:a", f"{dst}:b"))


def test_new_labels_are_keyed_by_id():
    assert NODE_KEYS["Observation"] == "id" and NODE_KEYS["FeatureEvent"] == "id"


def test_reported_by_rejects_other_pairs():
    with pytest.raises(ValueError):
        check_edge(edge("REPORTED_BY", "Person:a", "Customer:b"))
    with pytest.raises(ValueError):
        check_edge(edge("REVERTS", "Person:a", "PR:b"))


def test_every_signature_label_has_a_key():
    for pairs in SIGNATURES.values():
        for s, d in pairs:
            assert s in NODE_KEYS and d in NODE_KEYS
