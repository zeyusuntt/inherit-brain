import multiprocessing as mp
import threading
from types import SimpleNamespace

import pytest

from inherit.llm import BudgetExceeded, Meter, chat, embed

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
    res = chat([{"role": "user", "content": "hello"}], arm="t", meter=m, client=c, model=MODEL)
    assert c.calls[0]["temperature"] == 0
    assert res.prompt_tokens == 224 and res.completion_tokens == 36
    assert res.cost_usd == pytest.approx(3.32e-05, rel=1e-9)
    assert m.spent("t") == pytest.approx(3.32e-05, rel=1e-9)


def test_chat_refused_before_request(tmp_path):
    m = make_meter(tmp_path, cap=1e-6)
    c = FakeChat()
    with pytest.raises(BudgetExceeded):
        chat([{"role": "user", "content": "hello"}], arm="t", max_tokens=800, meter=m, client=c, model=MODEL)
    assert c.calls == []


def test_chat_error_settles_full_reservation(tmp_path):
    m = make_meter(tmp_path)
    c = FakeChat(fail=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        chat([{"role": "user", "content": "x" * 300}], arm="t", max_tokens=800, meter=m, client=c, model=MODEL)
    # 5xx / unknown error: charged at the full reservation, which is a byte-bound ceiling
    assert m.spent("t") >= 300 * PRICE_IN + 800 * PRICE_OUT
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
    res = embed(["a", "b"], arm="t", meter=m, client=c, model=EMBED_MODEL)
    assert res.vectors == [[0.1, 0.2], [0.1, 0.2]]
    assert res.prompt_tokens == 7
    assert res.cost_usd == pytest.approx(7e-8, rel=1e-9)
    assert m.spent("t") == pytest.approx(7e-8, rel=1e-9)


# ---- fix round 1 ----------------------------------------------------------

import httpx2
import openai

from inherit import llm
from inherit.config import Settings


def events(tmp_path):
    return [json.loads(l) for l in (tmp_path / "meter.jsonl").read_text().splitlines()]


def test_reservation_is_a_byte_ceiling(tmp_path):
    m = make_meter(tmp_path)
    c = FakeChat()
    msgs = [{"role": "user", "content": "é" * 100}]  # 2 bytes per char
    tools = [{"type": "function", "function": {"name": "f"}}]
    chat(msgs, arm="t", tools=tools, response_format={"type": "json_object"}, max_tokens=10, meter=m, client=c, model=MODEL)
    res = [e for e in events(tmp_path) if e["type"] == "reserve"][0]
    nbytes = len(json.dumps(msgs, ensure_ascii=False).encode()) + len(json.dumps(tools).encode())
    nbytes += len(json.dumps({"type": "json_object"}).encode())
    assert res["amount"] == (nbytes + 16 + 512) * 100_000 + 10 * 300_000


def test_over_reservation_recorded(tmp_path):
    m = make_meter(tmp_path, cap=1.0)
    r = m.reserve(1e-6, "a")
    cost = m.settle(r, 1000, 0)
    assert cost == pytest.approx(1e-4)
    assert m.spent("a") == pytest.approx(1e-4)
    assert events(tmp_path)[-1]["over_reservation"] is True


def test_real_sdk_one_http_request_per_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    hits = []

    def handler(request):
        hits.append(1)
        return httpx2.Response(500, json={"error": {"message": "boom"}})

    sdk = openai.OpenAI(
        api_key="k", base_url="http://x/v1", http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )  # default max_retries=2 must be overridden by chat()
    m = make_meter(tmp_path)
    with pytest.raises(openai.InternalServerError):
        chat([{"role": "user", "content": "hi"}], arm="t", meter=m, client=sdk, model=MODEL, attempts=3)
    reserves = [e for e in events(tmp_path) if e["type"] == "reserve"]
    assert len(hits) == len(reserves) == 3
    assert m.spent("t") > 0  # 5xx charged in full


def test_4xx_settles_zero_and_429_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    codes = iter([429, 400])

    def handler(request):
        return httpx2.Response(next(codes), json={"error": {"message": "no"}})

    sdk = openai.OpenAI(
        api_key="k", base_url="http://x/v1", http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    m = make_meter(tmp_path)
    with pytest.raises(openai.BadRequestError):
        chat([{"role": "user", "content": "hi"}], arm="t", meter=m, client=sdk, model=MODEL)
    settles = [e for e in events(tmp_path) if e["type"] == "settle"]
    assert [s["status_code"] for s in settles] == [429, 400]
    assert m.spent("t") == 0


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


def test_unknown_model_refused(tmp_path):
    m = make_meter(tmp_path)
    with pytest.raises(ValueError):
        chat([{"role": "user", "content": "x"}], arm="t", meter=m, client=FakeChat(), model="other/model")
    with pytest.raises(ValueError):
        embed(["x"], arm="t", meter=m, client=FakeEmbed(), model="other/model")


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


def test_missing_usage_settles_full_once(tmp_path):
    m = make_meter(tmp_path)
    c = FakeChat()
    good = c.create

    def bad(**kw):
        c.calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=None)], usage=None)

    c.completions.create = bad
    with pytest.raises(Exception):
        chat([{"role": "user", "content": "x"}], arm="t", max_tokens=100, meter=m, client=c, model=MODEL)
    evs = events(tmp_path)
    assert [e["type"] for e in evs] == ["reserve", "settle"]
    assert evs[1]["cost"] == evs[0]["amount"]
