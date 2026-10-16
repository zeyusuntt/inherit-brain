"""Thin client of the metering proxy (inherit.proxy). The proxy meters; this module does not."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import openai

from inherit.config import load_settings
from inherit.meter import BudgetExceeded  # noqa: F401  re-exported


@dataclass
class ChatResult:
    message: Any
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    prompt_tokens: int
    cost_usd: float


def proxy_url() -> str:
    return f"http://127.0.0.1:{load_settings().proxy_port}/v1"


def proxy_openai_kwargs(arm: str) -> dict:
    """base_url, api_key and headers for any OpenAI-compatible client (llama-index, GraphRAG-SDK)."""
    if not isinstance(arm, str) or not arm.strip():
        raise ValueError("arm must be a non-empty string")
    return {"base_url": proxy_url(), "api_key": "proxy", "default_headers": {"X-Inherit-Arm": arm}}


def _timeout() -> float:
    # Outlast the proxy's total deadline so a client never gives up while attempts are still billed.
    return load_settings().proxy_deadline_s + 30.0


def _client(arm: str, base_url: str | None) -> openai.OpenAI:
    kw = proxy_openai_kwargs(arm)
    if base_url:
        kw["base_url"] = base_url
    # The proxy retries and meters; the SDK must not resend anything.
    return openai.OpenAI(**kw, max_retries=0, timeout=_timeout())


def _call(fn, **kw):
    try:
        raw = fn(**kw)
    except openai.APITimeoutError as e:
        raise TimeoutError(
            "metering proxy did not answer in time; it may still be finishing billed attempts, check GET /spend"
        ) from e
    except openai.APIConnectionError as e:
        raise RuntimeError("metering proxy is not reachable; start it with `uv run python -m inherit.proxy`") from e
    except openai.APIStatusError as e:
        if e.status_code == 402:
            raise BudgetExceeded(str(e)) from e
        raise
    return raw, float(raw.headers["X-Inherit-Cost-USD"])


def chat(
    messages: list[dict],
    *,
    arm: str,
    tools: list[dict] | None = None,
    max_tokens: int = 800,
    response_format: dict | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> ChatResult:
    client = _client(arm, base_url)
    kw: dict[str, Any] = dict(
        model=model or load_settings().model, messages=messages, max_tokens=max_tokens, temperature=0
    )
    if tools:
        kw["tools"] = tools
    if response_format is not None:
        kw["response_format"] = response_format
    t0 = time.monotonic()
    raw, cost = _call(client.chat.completions.with_raw_response.create, **kw)
    latency = time.monotonic() - t0
    resp = raw.parse()
    return ChatResult(resp.choices[0].message, resp.usage.prompt_tokens, resp.usage.completion_tokens, cost, latency)


def embed(
    texts: list[str], *, arm: str, model: str | None = None, base_url: str | None = None
) -> EmbedResult:
    client = _client(arm, base_url)
    raw, cost = _call(
        client.embeddings.with_raw_response.create, model=model or load_settings().embed_model, input=texts
    )
    resp = raw.parse()
    return EmbedResult([d.embedding for d in resp.data], resp.usage.prompt_tokens, cost)
