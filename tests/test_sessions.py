from dataclasses import dataclass

from inherit.extract.sessions import sessions


@dataclass
class M:
    id: str
    ts: str


def test_split_at_exactly_six_hours():
    ms = [M("a", "2026-01-01T00:00:00"), M("b", "2026-01-01T05:59:59"), M("c", "2026-01-01T11:59:58")]
    assert [[m.id for m in s] for s in sessions(ms)] == [["a", "b", "c"]]
    ms = [M("a", "2026-01-01T00:00:00"), M("b", "2026-01-01T06:00:00")]
    assert [[m.id for m in s] for s in sessions(ms)] == [["a"], ["b"]]


def test_sorts_by_ts_and_handles_empty():
    ms = [M("b", "2026-01-02T00:00:00"), M("a", "2026-01-01T00:00:00")]
    assert [[m.id for m in s] for s in sessions(ms)] == [["a"], ["b"]]
    assert sessions([]) == []


def test_gap_hours_parameter():
    ms = [M("a", "2026-01-01T00:00:00"), M("b", "2026-01-01T02:00:00")]
    assert len(sessions(ms, gap_hours=2)) == 2
    assert len(sessions(ms, gap_hours=3)) == 1
