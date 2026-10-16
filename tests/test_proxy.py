"""Offline tests for the metering proxy. The upstream is a scripted fake transport."""

import json
import logging
import os
import socket
import threading
import time
from collections import deque
from pathlib import Path

import httpx2
import pytest
from fastapi.testclient import TestClient

from inherit import llm
from inherit.config import EMBED_MODEL, MODEL
from inherit.meter import BudgetExceeded, Meter
from inherit.proxy import create_app

KEY = "sk-secret-upstream-key-123"
ROOT = Path(__file__).resolve().parents[1]
OK = {
    "id": "c1",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 224, "completion_tokens": 36},
}
EMB_OK = {"data": [{"index": 0, "embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 7, "total_tokens": 7}}


class Upstream:
    def __init__(self, *responses, default=(200, OK)):
        self.queue = deque(responses)
        self.default = default
        self.requests = []
        self.lock = threading.Lock()
        self.delay = 0.0

    def __call__(self, request):
        with self.lock:
            self.requests.append(request)
            item = self.queue.popleft() if self.queue else self.default
        if self.delay:
            time.sleep(self.delay)
        if isinstance(item, Exception):
            raise item
        status, body = item
        if isinstance(body, bytes):
            return httpx2.Response(status, content=body)
        return httpx2.Response(status, json=body)

    @property
    def transport(self):
        return httpx2.MockTransport(self)


def make(tmp_path, up, cap=1.0, **kw):
    meter = Meter(cap_usd=cap, path=tmp_path / "meter.jsonl")
    app = create_app(
        upstream_base_url="http://upstream.test/v1/",
        upstream_key=KEY,
        transport=up.transport,
        meter=meter,
        sleep=lambda s: None,
        **kw,
    )
    return TestClient(app), meter


def events(tmp_path):
    return [json.loads(l) for l in (tmp_path / "meter.jsonl").read_text().splitlines()]


def body(**kw):
    b = {"model": MODEL, "messages": [{"role": "user", "content": "hello"}]}
    b.update(kw)
    return b


H = {"X-Inherit-Arm": "t"}


def test_missing_arm_400(tmp_path):
    up = Upstream()
    c, _ = make(tmp_path, up)
    assert c.post("/v1/chat/completions", json=body()).status_code == 400
    assert c.post("/v1/chat/completions", json=body(), headers={"X-Inherit-Arm": ""}).status_code == 400
    assert c.post("/v1/embeddings", json={"model": EMBED_MODEL, "input": "x"}).status_code == 400
    assert up.requests == []


def test_bad_requests_400(tmp_path):
    up = Upstream()
    c, _ = make(tmp_path, up)
    assert c.post("/v1/chat/completions", json=body(model="other/m"), headers=H).status_code == 400
    assert c.post("/v1/chat/completions", json=body(stream=True), headers=H).status_code == 400
    assert c.post("/v1/chat/completions", json=body(n=2), headers=H).status_code == 400
    assert c.post("/v1/embeddings", json={"model": "other/m", "input": "x"}, headers=H).status_code == 400
    assert c.post("/v1/chat/completions", content=b"not json", headers=H).status_code == 400
    assert up.requests == []


def test_max_tokens_injected_and_reservation_is_formula(tmp_path):
    up = Upstream()
    c, _ = make(tmp_path, up, default_max_tokens=1024)
    msgs = [{"role": "user", "content": "é" * 50}]
    r = c.post("/v1/chat/completions", json=body(messages=msgs), headers=H)
    assert r.status_code == 200
    sent = up.requests[0].content
    assert json.loads(sent)["max_tokens"] == 1024
    res = [e for e in events(tmp_path) if e["type"] == "reserve"][0]
    assert res["amount"] == (len(sent) + 16 * 1 + 512) * 100_000 + 1024 * 300_000
    assert r.headers["X-Inherit-Reserved-USD"] == str(res["amount"] / 10**12)
    assert float(r.headers["X-Inherit-Cost-USD"]) == pytest.approx(3.32e-05)
    assert r.json() == OK


def test_explicit_max_tokens_kept(tmp_path):
    up = Upstream()
    c, _ = make(tmp_path, up)
    c.post("/v1/chat/completions", json=body(max_completion_tokens=50), headers=H)
    sent = json.loads(up.requests[0].content)
    assert "max_tokens" not in sent and sent["max_completion_tokens"] == 50


def test_temperature_forced_zero(tmp_path):
    up = Upstream()
    c, _ = make(tmp_path, up)
    c.post("/v1/chat/completions", json=body(temperature=0.9), headers=H)
    assert json.loads(up.requests[0].content)["temperature"] == 0
    assert events(tmp_path)[-1]["temperature_overridden"] is True
    c.post("/v1/chat/completions", json=body(), headers=H)
    assert "temperature_overridden" not in events(tmp_path)[-1]


def test_over_cap_402_no_upstream(tmp_path):
    up = Upstream()
    c, _ = make(tmp_path, up, cap=0.0001)
    r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 402
    assert r.json()["error"]["type"] == "budget_exceeded"
    assert up.requests == []


def test_arm_cap_via_admin(tmp_path):
    up = Upstream()
    c, m = make(tmp_path, up)
    assert c.post("/admin/arm_cap", json={"arm": "t", "usd": 0.0001}).status_code == 200
    assert c.post("/v1/chat/completions", json=body(), headers=H).status_code == 402
    assert c.post("/v1/chat/completions", json=body(), headers={"X-Inherit-Arm": "other"}).status_code == 200
    assert any(e["type"] == "cap" for e in events(tmp_path))
    assert c.post("/admin/arm_cap", json={"arm": "", "usd": 1}).status_code == 400
    assert up.requests and len(up.requests) == 1


def test_429_then_200_two_reservations(tmp_path):
    up = Upstream((429, {"error": {"message": "slow"}}))
    c, m = make(tmp_path, up)
    r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 200
    assert len(up.requests) == 2
    evs = events(tmp_path)
    assert [e["type"] for e in evs] == ["reserve", "settle", "reserve", "settle"]
    assert evs[1]["cost"] == 0 and evs[1]["status_code"] == 429
    assert evs[3]["cost"] == 224 * 100_000 + 36 * 300_000


def test_500_settles_full_and_retries_twice(tmp_path):
    up = Upstream(default=(500, {"error": {"message": "boom"}}))
    c, m = make(tmp_path, up)
    r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 500
    assert len(up.requests) == 3
    evs = events(tmp_path)
    reserves = [e for e in evs if e["type"] == "reserve"]
    settles = [e for e in evs if e["type"] == "settle"]
    assert len(reserves) == len(settles) == 3
    assert [s["cost"] for s in settles] == [r_["amount"] for r_ in reserves]


def test_400_from_upstream_unbilled_passthrough(tmp_path):
    up = Upstream((400, {"error": {"message": "bad"}}))
    c, _ = make(tmp_path, up)
    r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 400 and r.json()["error"]["message"] == "bad"
    assert len(up.requests) == 1
    assert events(tmp_path)[-1]["cost"] == 0


def test_200_without_usage_is_502_full_settle(tmp_path):
    up = Upstream((200, {"choices": []}))
    c, m = make(tmp_path, up)
    r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 502
    evs = events(tmp_path)
    assert evs[-1]["cost"] == evs[0]["amount"]


def test_usage_over_reservation_is_a_bug(tmp_path, caplog):
    huge = {**OK, "usage": {"prompt_tokens": 10_000_000, "completion_tokens": 10}}
    up = Upstream((200, huge))
    c, m = make(tmp_path, up)
    with caplog.at_level(logging.ERROR):
        r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 200
    assert events(tmp_path)[-1]["over_reservation"] is True
    bugs = (tmp_path / "meter_bugs.jsonl").read_text().splitlines()
    assert len(bugs) == 1 and json.loads(bugs[0])["arm"] == "t"
    assert any(rec.levelno == logging.ERROR for rec in caplog.records)
    assert m.spent("t") == pytest.approx(10_000_000 * 1e-7 + 10 * 3e-7)


def test_embeddings(tmp_path):
    up = Upstream(default=(200, EMB_OK))
    c, m = make(tmp_path, up)
    payload = {"model": EMBED_MODEL, "input": ["a", "bb"]}
    r = c.post("/v1/embeddings", json=payload, headers=H)
    assert r.status_code == 200 and r.json() == EMB_OK
    assert up.requests[0].url.path.endswith("/embeddings")
    res = events(tmp_path)[0]
    assert res["amount"] == (len(up.requests[0].content) + 16 * 2) * 10_000
    assert events(tmp_path)[1]["cost"] == 7 * 10_000
    assert float(r.headers["X-Inherit-Cost-USD"]) == pytest.approx(7e-8)


def test_key_sent_upstream_but_never_leaked(tmp_path, caplog):
    up = Upstream((500, {"error": {"message": "boom"}}), (200, OK))
    c, _ = make(tmp_path, up)
    with caplog.at_level(logging.DEBUG):
        r = c.post(
            "/v1/chat/completions", json=body(), headers={**H, "Authorization": "Bearer client-secret"}
        )
    assert up.requests[0].headers["authorization"] == f"Bearer {KEY}"
    blob = r.text + str(dict(r.headers)) + caplog.text + (tmp_path / "meter.jsonl").read_text()
    assert KEY not in blob and "client-secret" not in blob


def test_models_health_spend(tmp_path):
    up = Upstream()
    c, _ = make(tmp_path, up)
    assert c.get("/health").status_code == 200
    ids = {m["id"] for m in c.get("/v1/models").json()["data"]}
    assert ids == {MODEL, EMBED_MODEL}
    c.post("/v1/chat/completions", json=body(), headers=H)
    s = c.get("/spend").json()
    assert s["cap_usd"] == 1.0
    assert s["per_arm"]["t"]["committed_usd"] == pytest.approx(3.32e-05)
    assert s["open_reservations"] == 0
    assert up.requests and len(up.requests) == 1  # none of these routes call upstream again


def test_concurrent_requests_never_exceed_cap(tmp_path):
    up = Upstream()
    # one request reserves about 0.0003 + 0.0001; cap fits only a few
    c, m = make(tmp_path, up, cap=0.002)
    codes = []

    def go():
        codes.append(c.post("/v1/chat/completions", json=body(), headers=H).status_code)

    ts = [threading.Thread(target=go) for _ in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert codes.count(200) + codes.count(402) == 20
    assert 0 < codes.count(200) < 20
    assert len(up.requests) == codes.count(200)
    assert m.spent() <= 0.002


def test_no_module_but_proxy_holds_the_key():
    allowed = {ROOT / "src/inherit/proxy.py", ROOT / "src/inherit/config.py"}
    for f in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        if f in allowed:
            continue
        text = f.read_text()
        assert "NEBIUS_API_KEY" not in text and "nebius_api_key" not in text, f
    from inherit import config

    assert "api_key" not in config.Settings.__dataclass_fields__


# ---- client through a live in-process uvicorn ------------------------------


@pytest.fixture
def live(tmp_path):
    import uvicorn

    up = Upstream()
    c, meter = make(tmp_path, up)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(c.app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1", up, meter
    server.should_exit = True
    th.join(5)


def test_chat_through_live_proxy(live):
    url, up, meter = live
    res = llm.chat([{"role": "user", "content": "hello"}], arm="t", base_url=url)
    assert res.message.content == "hi"
    assert (res.prompt_tokens, res.completion_tokens) == (224, 36)
    assert res.cost_usd == pytest.approx(3.32e-05)
    assert meter.spent("t") == pytest.approx(3.32e-05)  # metered once, by the proxy only
    assert json.loads(up.requests[0].content)["temperature"] == 0


def test_embed_through_live_proxy(live):
    url, up, meter = live
    up.default = (200, EMB_OK)
    res = llm.embed(["a"], arm="t", base_url=url)
    assert res.vectors == [[0.1, 0.2]] and res.prompt_tokens == 7
    assert res.cost_usd == pytest.approx(7e-8)


def test_chat_budget_exceeded_raises(live):
    url, up, meter = live
    meter.set_arm_cap("t", 1e-9)
    with pytest.raises(BudgetExceeded):
        llm.chat([{"role": "user", "content": "hello"}], arm="t", base_url=url)
    assert up.requests == []


def test_chat_empty_arm_raises_before_any_request(live):
    url, up, _ = live
    with pytest.raises(ValueError):
        llm.chat([{"role": "user", "content": "x"}], arm="", base_url=url)
    assert up.requests == []


def test_proxy_not_running_message():
    with pytest.raises(RuntimeError, match="inherit.proxy"):
        llm.chat([{"role": "user", "content": "x"}], arm="t", base_url="http://127.0.0.1:1/v1")


def test_proxy_openai_kwargs():
    kw = llm.proxy_openai_kwargs("a1")
    assert kw["default_headers"] == {"X-Inherit-Arm": "a1"}
    assert kw["base_url"].startswith("http://127.0.0.1:") and kw["api_key"]


# ---- proxy fix round 1 ------------------------------------------------------

from inherit import config


@pytest.fixture
def dummy_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "NEBIUS_API_KEY=sk-dummy-from-dotenv\nNEBIUS_BUDGET_USD=5\nNEBIUS_BASE_URL=http://x/v1/\n"
    )
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_BUDGET_USD", raising=False)
    config.load_settings.cache_clear()
    yield tmp_path
    config.load_settings.cache_clear()


def test_clients_never_see_the_key(dummy_env):
    s = config.load_settings()
    assert s.budget_usd == 5
    assert "NEBIUS_API_KEY" not in os.environ
    assert not hasattr(s, "nebius_api_key") and "sk-dummy" not in repr(s)
    llm._client("a", None)
    assert "NEBIUS_API_KEY" not in os.environ
    assert config.load_upstream_secret() == "sk-dummy-from-dotenv"
    assert "NEBIUS_API_KEY" not in os.environ


def test_upstream_401_echoing_key_is_redacted(tmp_path):
    up = Upstream((401, {"error": {"message": f"bad key {KEY}"}}))
    c, _ = make(tmp_path, up)
    r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 401 and KEY not in r.text and "[redacted]" in r.text


def test_non_json_5xx_wrapped_in_json(tmp_path):
    up = Upstream(default=(502, b"<html>Bad Gateway</html>"))
    c, _ = make(tmp_path, up)
    r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_error" and "Bad Gateway" in r.json()["error"]["message"]


@pytest.mark.parametrize("exc", [httpx2.ReadTimeout("slow"), httpx2.ConnectError("down")])
def test_timeout_and_connection_error_settle_full(tmp_path, exc):
    up = Upstream(default=exc)
    c, _ = make(tmp_path, up)
    r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 502 and r.json()["error"]["type"] == "upstream_unreachable"
    evs = events(tmp_path)
    assert [e["type"] for e in evs] == ["reserve", "settle"] and evs[1]["cost"] == evs[0]["amount"]
    assert len(up.requests) == 1


def fail_settles(meter, times):
    real = meter._append
    n = {"left": times}

    def flaky(ev):
        if ev["type"] == "settle" and n["left"] > 0:
            n["left"] -= 1
            raise OSError("disk full")
        return real(ev)

    meter._append = flaky


def test_settle_fails_once_falls_back_to_full(tmp_path):
    up = Upstream()
    c, m = make(tmp_path, up)
    fail_settles(m, 1)
    r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 500 and r.json()["error"]["type"] == "ledger_error"
    evs = events(tmp_path)
    assert [e["type"] for e in evs] == ["reserve", "settle"] and evs[1]["cost"] == evs[0]["amount"]


@pytest.mark.parametrize("upstream", [(200, OK), (400, {"error": {"message": "no"}}), (500, {"error": {}})])
def test_settle_always_fails_keeps_reservation_after_restart(tmp_path, upstream):
    up = Upstream(default=upstream)
    c, m = make(tmp_path, up)
    fail_settles(m, 99)
    r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 500 and r.json()["error"]["type"] == "ledger_error"
    reserve = events(tmp_path)[0]["amount"]
    replay = Meter(cap_usd=1.0, path=tmp_path / "meter.jsonl")
    assert replay.spent("t") == pytest.approx(reserve / 10**12)  # still counted at its ceiling
    assert len(up.requests) == 1  # no retry once the ledger is broken


def test_bug_file_failure_still_logs_error_and_answers(tmp_path, caplog):
    huge = {**OK, "usage": {"prompt_tokens": 10_000_000, "completion_tokens": 10}}
    up = Upstream((200, huge))
    bad = tmp_path / "is_a_dir"
    bad.mkdir()
    c, _ = make(tmp_path, up, bug_path=bad)
    with caplog.at_level(logging.ERROR):
        r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 200
    assert "usage above reservation" in caplog.text and "could not write" in caplog.text


def test_embeddings_reject_stream_and_n(tmp_path):
    up = Upstream(default=(200, EMB_OK))
    c, _ = make(tmp_path, up)
    for extra in ({"stream": True}, {"n": 2}):
        r = c.post("/v1/embeddings", json={"model": EMBED_MODEL, "input": "x", **extra}, headers=H)
        assert r.status_code == 400
    assert up.requests == [] and not (tmp_path / "meter.jsonl").exists()


def test_deadline_stops_new_attempts(tmp_path):
    up = Upstream(default=(500, {"error": {}}))
    up.delay = 0.2
    c, _ = make(tmp_path, up, deadline=0.3)
    r = c.post("/v1/chat/completions", json=body(), headers=H)
    assert r.status_code == 504 and r.json()["error"]["type"] == "deadline_exceeded"
    assert len(up.requests) == 2  # third attempt never started
    evs = events(tmp_path)
    assert [e["type"] for e in evs].count("reserve") == [e["type"] for e in evs].count("settle") == 2


def test_client_timeout_is_distinct_from_unreachable(live, monkeypatch):
    url, up, _ = live
    up.delay = 1.0
    monkeypatch.setattr(llm, "_timeout", lambda: 0.2)
    with pytest.raises(TimeoutError, match="/spend"):
        llm.chat([{"role": "user", "content": "x"}], arm="t", base_url=url)


def test_client_timeout_exceeds_proxy_deadline():
    assert llm._timeout() >= config.load_settings().proxy_deadline_s + 30
