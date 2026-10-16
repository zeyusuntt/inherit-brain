import multiprocessing as mp
import threading
from types import SimpleNamespace

import pytest

from inherit.llm import BudgetExceeded, Meter, chat, embed

PRICE_IN = 1e-7
PRICE_OUT = 3e-7


def make_meter(tmp_path, cap=1.0, **kw):
    return Meter(
        cap_usd=cap, path=tmp_path / "meter.jsonl", price_in=PRICE_IN, price_out=PRICE_OUT, **kw
    )


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
    m = Meter(cap_usd=0.05, path=path, price_in=PRICE_IN, price_out=PRICE_OUT)
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
    assert Meter(cap_usd=0.05, path=path, price_in=PRICE_IN, price_out=PRICE_OUT).spent() == pytest.approx(0.05)


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


class FakeChat:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail
        self.completions = SimpleNamespace(create=self.create)
        self.chat = SimpleNamespace(completions=self.completions)

    def create(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise self.fail
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=224, completion_tokens=36),
        )


def test_chat_settles_real_usage(tmp_path):
    m = make_meter(tmp_path)
    c = FakeChat()
    res = chat([{"role": "user", "content": "hello"}], arm="t", meter=m, client=c, model="fake")
    assert c.calls[0]["temperature"] == 0
    assert res.prompt_tokens == 224 and res.completion_tokens == 36
    assert res.cost_usd == pytest.approx(3.32e-05, rel=1e-9)
    assert m.spent("t") == pytest.approx(3.32e-05, rel=1e-9)


def test_chat_refused_before_request(tmp_path):
    m = make_meter(tmp_path, cap=1e-6)
    c = FakeChat()
    with pytest.raises(BudgetExceeded):
        chat([{"role": "user", "content": "hello"}], arm="t", max_tokens=800, meter=m, client=c, model="fake")
    assert c.calls == []


def test_chat_error_settles_full_reservation(tmp_path):
    m = make_meter(tmp_path)
    c = FakeChat(fail=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        chat([{"role": "user", "content": "x" * 300}], arm="t", max_tokens=800, meter=m, client=c, model="fake")
    expected = 300 / 3 * PRICE_IN + 800 * PRICE_OUT
    assert m.spent("t") == pytest.approx(expected, rel=0.05)
    assert m.spent("t") > 0


class FakeEmbed:
    def __init__(self):
        self.calls = []
        self.embeddings = SimpleNamespace(create=self.create)

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in kw["input"]],
            usage=SimpleNamespace(prompt_tokens=7),
        )


def test_embed_metered(tmp_path):
    m = make_meter(tmp_path)
    c = FakeEmbed()
    res = embed(["a", "b"], arm="t", meter=m, client=c, price_in=1e-8, model="fake")
    assert res.vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert res.prompt_tokens == 7
    assert res.cost_usd == pytest.approx(7e-8, rel=1e-9)
    assert m.spent("t") == pytest.approx(7e-8, rel=1e-9)
