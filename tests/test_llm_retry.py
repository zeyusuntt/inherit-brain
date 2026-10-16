"""429 quota errors are retried with quota-window backoff in the shared client, the same for every arm."""

import httpx
import openai
import pytest

from inherit import llm

OK = {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
      "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
      "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}


def install(monkeypatch, statuses, retry_after=None):
    seen = {"n": 0, "sleeps": []}

    def handler(request):
        i = seen["n"]
        seen["n"] += 1
        status = statuses[min(i, len(statuses) - 1)]
        if status == 200:
            return httpx.Response(200, json=OK, headers={"X-Inherit-Cost-USD": "0.001"})
        h = {"Retry-After": str(retry_after)} if retry_after is not None else {}
        return httpx.Response(status, json={"error": {"message": "quota"}}, headers=h)

    def client(arm, base_url):
        return openai.OpenAI(base_url="http://x/v1", api_key="k", max_retries=0,
                             http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(llm, "_client", client)
    monkeypatch.setattr(llm, "_sleep", lambda s: seen["sleeps"].append(s))
    return seen


def test_a_429_is_retried_with_quota_window_backoff(monkeypatch):
    seen = install(monkeypatch, [429, 429, 200])
    r = llm.chat([{"role": "user", "content": "q"}], arm="t")
    assert r.message.content == "hi" and seen["n"] == 3 and seen["sleeps"] == [20, 40]


def test_at_most_three_retries_then_the_call_fails_as_before(monkeypatch):
    seen = install(monkeypatch, [429])
    with pytest.raises(openai.RateLimitError):
        llm.chat([{"role": "user", "content": "q"}], arm="t")
    assert seen["n"] == 4 and seen["sleeps"] == [20, 40, 80]


def test_retry_after_is_honoured(monkeypatch):
    seen = install(monkeypatch, [429, 200], retry_after=7)
    llm.chat([{"role": "user", "content": "q"}], arm="t")
    assert seen["sleeps"] == [7]


def test_other_errors_are_not_retried(monkeypatch):
    seen = install(monkeypatch, [500])
    with pytest.raises(openai.APIStatusError):
        llm.chat([{"role": "user", "content": "q"}], arm="t")
    assert seen["n"] == 1 and seen["sleeps"] == []


def test_a_402_still_raises_budget_exceeded_without_retry(monkeypatch):
    seen = install(monkeypatch, [402])
    with pytest.raises(llm.BudgetExceeded):
        llm.chat([{"role": "user", "content": "q"}], arm="t")
    assert seen["n"] == 1
