import multiprocessing as mp
import threading

import pytest

from inherit.meter import BudgetExceeded, Meter
from inherit import meter as meter_mod

import json

from inherit.config import EMBED_MODEL, MODEL, PROJECT_CAP_USD

PRICE_IN = 1e-7
PRICE_OUT = 3e-7


def make_meter(tmp_path, cap=1.0, **kw):
    return Meter(cap_usd=cap, path=tmp_path / "meter.jsonl", **kw)


def test_cost_matches_price(tmp_path):
    m = make_meter(tmp_path)
    r = m.reserve(0.001, "a")
    cost = m.settle(r, 224, 36)
    assert cost == pytest.approx(3.32e-05, rel=1e-9)
    assert m.spent("a") == pytest.approx(3.32e-05, rel=1e-9)
    assert m.spent() == pytest.approx(3.32e-05, rel=1e-9)


def test_refuses_over_cap(tmp_path):
    m = make_meter(tmp_path, cap=0.001)
    with pytest.raises(BudgetExceeded):
        m.reserve(0.002, "a")
    assert m.spent() == 0


def test_open_reservation_counts_against_cap(tmp_path):
    m = make_meter(tmp_path, cap=0.05)
    m.reserve(0.03, "a")
    with pytest.raises(BudgetExceeded):
        m.reserve(0.03, "a")


def test_concurrent_reservations_never_exceed_cap(tmp_path):
    m = make_meter(tmp_path, cap=0.05)
    ok, bad = [], []

    def go():
        try:
            ok.append(m.reserve(0.01, "a"))
        except BudgetExceeded:
            bad.append(1)

    ts = [threading.Thread(target=go) for _ in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(ok) == 5
    assert len(bad) == 15


def _proc(path, q):
    m = Meter(cap_usd=0.05, path=path)
    n = 0
    for _ in range(5):
        try:
            m.reserve(0.01, "a")
            n += 1
        except BudgetExceeded:
            pass
    q.put(n)


def test_multiprocess_reservations_never_exceed_cap(tmp_path):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    path = tmp_path / "meter.jsonl"
    ps = [ctx.Process(target=_proc, args=(path, q)) for _ in range(4)]
    [p.start() for p in ps]
    [p.join(60) for p in ps]
    total = sum(q.get(timeout=5) for _ in ps)
    assert total == 5
    assert Meter(cap_usd=0.05, path=path).spent() == pytest.approx(0.05)


def test_two_meters_same_file_see_each_other(tmp_path):
    a = make_meter(tmp_path, cap=0.02)
    b = make_meter(tmp_path, cap=0.02)
    a.reserve(0.015, "x")
    with pytest.raises(BudgetExceeded):
        b.reserve(0.015, "y")


def test_arm_subcap(tmp_path):
    m = make_meter(tmp_path, cap=1.0, arm_caps={"a": 0.01})
    m.reserve(0.006, "a")
    with pytest.raises(BudgetExceeded):
        m.reserve(0.006, "a")
    m.reserve(0.006, "b")
    m.set_arm_cap("b", 0.007)
    with pytest.raises(BudgetExceeded):
        m.reserve(0.006, "b")


def test_meter_survives_restart(tmp_path):
    m = make_meter(tmp_path)
    m.settle(m.reserve(0.01, "a"), 1000, 100)
    m.reserve(0.02, "a")  # never settled: stays counted
    m2 = make_meter(tmp_path)
    assert m2.spent("a") == pytest.approx(1000 * PRICE_IN + 100 * PRICE_OUT + 0.02)


def test_double_settle_rejected(tmp_path):
    m = make_meter(tmp_path)
    r = m.reserve(0.01, "a")
    m.settle(r, 1, 1)
    with pytest.raises(ValueError):
        m.settle(r, 1, 1)


def events(tmp_path):
    return [json.loads(l) for l in (tmp_path / "meter.jsonl").read_text().splitlines()]


def test_over_reservation_recorded(tmp_path):
    m = make_meter(tmp_path, cap=1.0)
    r = m.reserve(1e-6, "a")
    cost = m.settle(r, 1000, 0)
    assert cost == pytest.approx(1e-4)
    assert m.spent("a") == pytest.approx(1e-4)
    assert events(tmp_path)[-1]["over_reservation"] is True


def test_budget_above_project_cap_rejected(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    monkeypatch.setenv("NEBIUS_BUDGET_USD", str(PROJECT_CAP_USD + 1))
    from inherit import config

    config.load_settings.cache_clear()
    try:
        with pytest.raises(ValueError):
            config.load_settings()
    finally:
        config.load_settings.cache_clear()


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_amounts_rejected(tmp_path, bad):
    m = make_meter(tmp_path)
    with pytest.raises(ValueError):
        m.reserve(bad, "a")
    r = m.reserve(0.01, "a")
    with pytest.raises(ValueError):
        m.settle(r, -1, 0)
    with pytest.raises(ValueError):
        m.settle(r, 0, -5)
    assert m.spent() == pytest.approx(0.01)


def test_forged_reservation_rejected(tmp_path):
    m = make_meter(tmp_path, arm_caps={"capped": 0.02})
    r = m.reserve(0.01, "capped")
    from dataclasses import replace

    with pytest.raises(ValueError):
        m.settle_full(replace(r, amount=0))
    with pytest.raises(ValueError):
        m.settle(replace(r, arm="free"), 1, 1)
    assert m.spent("capped") == pytest.approx(0.01)


def test_arm_cap_shared_across_meters(tmp_path):
    a = make_meter(tmp_path)
    b = make_meter(tmp_path)
    a.set_arm_cap("x", 0.01)
    with pytest.raises(BudgetExceeded):
        b.reserve(0.02, "x")


def test_refresh_failure_does_not_leak_locks(tmp_path, monkeypatch):
    m = make_meter(tmp_path)
    m.reserve(0.01, "a")
    real = m._refresh
    monkeypatch.setattr(m, "_refresh", lambda: (_ for _ in ()).throw(OSError("induced")))
    with pytest.raises(OSError):
        m.spent()
    monkeypatch.setattr(m, "_refresh", real)
    assert m.spent() == pytest.approx(0.01)  # would hang if a lock leaked


def test_torn_trailing_line(tmp_path):
    m = make_meter(tmp_path)
    m.reserve(0.01, "a")
    with open(tmp_path / "meter.jsonl", "ab") as f:
        f.write(b'{"type": "reserve", "id": "zz", "ar')  # crash mid-write
    m2 = make_meter(tmp_path)
    m2.reserve(0.01, "a")  # must not crash or merge into the fragment
    m3 = make_meter(tmp_path)
    assert m3.spent("a") == pytest.approx(0.02)
    assert m3.corrupt_lines == 1


def test_garbage_line_skipped_and_counted(tmp_path):
    (tmp_path / "meter.jsonl").write_text("not json\n[1]\n")
    m = make_meter(tmp_path)
    assert m.spent() == 0
    assert m.corrupt_lines == 2


def test_legacy_ledger_replays(tmp_path):
    lines = [
        {"id": "a1", "arm": "x", "amount": 20_000_000_000},
        {"id": "a1", "arm": "x", "cost": 15_000_000_000, "prompt_tokens": 1, "completion_tokens": 1},
        {"id": "a2", "arm": "x", "amount": 10_000_000_000},  # open
        {"id": "a3", "arm": "y", "amount": 20_000_000_000},
        {"id": "a3", "arm": "y", "cost": 10_000_000_000, "prompt_tokens": 1, "completion_tokens": 1},
    ]
    (tmp_path / "meter.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    m = make_meter(tmp_path)
    assert m.spent() == pytest.approx(0.035)
    assert m.spent("x") == pytest.approx(0.025)
    assert m.corrupt_lines == 0


def test_meter_has_no_price_parameter(tmp_path):
    with pytest.raises(TypeError):
        Meter(cap_usd=1.0, path=tmp_path / "m.jsonl", prices={MODEL: (0, 0)})


@pytest.mark.parametrize("bad", [(-1, 0), (0, -5), (1.5, 0), (float("nan"), 0)])
def test_bad_price_table_rejected(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(meter_mod, "PRICES_PICO", {MODEL: bad})
    with pytest.raises(ValueError):
        make_meter(tmp_path)


def test_invalid_line_changes_nothing(tmp_path):
    lines = [
        {"type": "reserve", "id": "r1", "arm": "x", "amount": 10_000_000_000, "model": MODEL},
        {"type": "settle", "id": "r1", "cost": 5},  # missing arm
        {"type": "settle", "id": "r1", "arm": "x", "cost": -3},  # negative
    ]
    (tmp_path / "meter.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
    m = make_meter(tmp_path)
    assert m.spent() == pytest.approx(0.01)
    assert m.corrupt_lines == 2  # reservation still open


@pytest.mark.parametrize("arm", ["", None, 5])
def test_invalid_arm_writes_nothing(tmp_path, arm):
    m = make_meter(tmp_path)
    with pytest.raises(ValueError):
        m.reserve(0.01, arm)
    with pytest.raises(ValueError):
        m.set_arm_cap(arm, 0.01)
    assert not (tmp_path / "meter.jsonl").exists() or (tmp_path / "meter.jsonl").read_text() == ""
    assert m.spent() == 0 and m.corrupt_lines == 0


