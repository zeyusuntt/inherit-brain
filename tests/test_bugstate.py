import pytest

from inherit.extract.bugstate import derive_bug, official_states, sort_events

KEYS = {"unresolved", "active", "reopened", "no_fix_needed"}


def st(events, prs=()):
    return official_states(events, list(prs))


def test_keys_and_empty():
    s = st([])
    assert set(s) == KEYS
    assert s["unresolved"] and not s["active"] and not s["reopened"] and not s["no_fix_needed"]


def test_resolved_without_pr_is_unresolved():
    s = st(["REPORTED", "ASSIGNED", "RESOLVED"])
    assert s["unresolved"] is True
    assert s["active"] is False


def test_resolved_with_pr():
    s = st(["REPORTED", "RESOLVED"], ["https://github.com/salesforce/X/pull/1"])
    assert s == {"unresolved": False, "active": False, "reopened": False, "no_fix_needed": False}


def test_reopened_after_fix_is_active_and_reopened_and_resolved():
    s = st(["REPORTED", "RESOLVED", "REOPENED"], ["https://github.com/salesforce/X/pull/1"])
    assert s["active"] and s["reopened"] and not s["unresolved"] and not s["no_fix_needed"]


def test_dismissed_is_no_fix_needed_not_active_but_unresolved():
    s = st(["REPORTED", "DISMISSED"])
    assert s["no_fix_needed"] and not s["active"] and s["unresolved"]


@pytest.mark.parametrize("last", ["REPORTED", "ASSIGNED", "REOPENED"])
def test_active_last_event(last):
    assert st(["REPORTED", last])["active"] is True


def test_dismissed_then_reopened_is_both():
    s = st(["REPORTED", "DISMISSED", "REOPENED"])
    assert s["no_fix_needed"] and s["reopened"] and s["active"]


def test_events_as_dicts_and_objects():
    class E:
        type = "REOPENED"

    assert st([{"type": "REPORTED"}, E()])["active"] is True


def test_sort_uses_timestamps_not_ids():
    # TrendForce shape: the record id order disagrees with the real time order.
    evs = [{"type": "REOPENED", "ts": "2026-09-27T11:00:00", "id": "20260927-0-8c21b", "seq": 1},
           {"type": "RESOLVED", "ts": "2026-09-27T09:00:00", "id": "20260927-1-8d6b4", "seq": 0}]
    assert [e["type"] for e in sort_events(evs)] == ["RESOLVED", "REOPENED"]


def test_sort_ties_keep_sequence():
    evs = [{"type": "ASSIGNED", "ts": "2026-01-01T00:00:00", "seq": 1},
           {"type": "REPORTED", "ts": "2026-01-01T00:00:00", "seq": 0}]
    assert [e["type"] for e in sort_events(evs)] == ["REPORTED", "ASSIGNED"]


def test_derive_bug():
    evs = [{"type": "REPORTED", "ts": "2026-01-01T00:00:00", "seq": 0, "assignee_eids": []},
           {"type": "ASSIGNED", "ts": "2026-01-01T00:01:00", "seq": 1, "assignee_eids": ["eid_a"]},
           {"type": "RESOLVED", "ts": "2026-01-02T00:00:00", "seq": 2, "assignee_eids": []},
           {"type": "REOPENED", "ts": "2026-01-03T00:00:00", "seq": 3, "assignee_eids": []},
           {"type": "ASSIGNED", "ts": "2026-01-03T00:05:00", "seq": 4, "assignee_eids": ["eid_b", "eid_c"]}]
    d = derive_bug(list(reversed(evs)), ["https://github.com/salesforce/X/pull/1"])
    assert d["status"] == "ASSIGNED"
    assert d["current_assignees"] == ["eid_b", "eid_c"]
    assert d["unresolved"] is False and d["resolved"] is True
    assert d["active"] and d["reopened"] and not d["no_fix_needed"]
    assert d["event_types"] == ["REPORTED", "ASSIGNED", "RESOLVED", "REOPENED", "ASSIGNED"]
    assert d["derived_by"] == "bug_state/v1"


def test_derive_without_events():
    d = derive_bug([], [])
    assert d["status"] == "REPORTED"  # a bug always has its REPORTED event; the fallback is the mandatory default
    assert d["current_assignees"] == []
