import json
from collections import Counter

from eval import pilot


def test_allocation_is_proportional_and_exact():
    assert pilot.allocate({"person": 46, "company": 26, "pr": 22, "url": 5, "content": 57}, 36) == {
        "person": 11, "company": 6, "pr": 5, "url": 1, "content": 13}


def test_sample_is_deterministic_dev_only_and_pinned():
    a, b = pilot.build_sample(), pilot.build_sample()
    assert [q.key for q in a] == [q.key for q in b]
    c = Counter(q.type for q in a)
    assert sum(c[t] for t in pilot.TYPES) == 36 and c["unanswerable"] == 12
    assert {q.product for q in a} == set(pilot.A1F_TOKENS)
    rec = json.loads(pilot.SAMPLE_PATH.read_text())
    assert rec == pilot.sample_record(a)
