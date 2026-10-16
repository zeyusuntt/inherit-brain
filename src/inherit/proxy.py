"""Local metering proxy: the only code that holds the upstream key.

Every model call in the project (our chat()/embed(), llama-index, GraphRAG-SDK)
points here. The proxy reserves a proven ceiling on the exact forwarded bytes,
forwards to Nebius, and settles on the shared Meter ledger. Binds 127.0.0.1 only.

Run: uv run python -m inherit.proxy
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

import httpx2
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from inherit.config import load_settings, load_upstream_secret
from inherit.meter import (
    FIXED_OVERHEAD_TOKENS,
    PER_MESSAGE_TOKENS,
    PICO,
    BudgetExceeded,
    Meter,
)

log = logging.getLogger("inherit.proxy")
ARM_HEADER = "X-Inherit-Arm"
TAG_HEADER = "X-Inherit-Tag"  # optional per-attempt label, recorded on the ledger and never forwarded
MAX_TAG_LEN = 300
MAX_RETRIES = 2  # upstream 429 / 5xx, each attempt with a fresh reservation


def _err(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status)


class _DeadlineHit(Exception):
    pass


def _pos_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def create_app(
    *,
    upstream_base_url: str,
    upstream_key: str,
    meter: Meter,
    transport: httpx2.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    default_max_tokens: int = 1024,
    timeout: float = 300.0,
    deadline: float = 600.0,
    bug_path: Any = None,
) -> FastAPI:
    app = FastAPI(title="inherit metering proxy")
    http = httpx2.Client(transport=transport, timeout=timeout)
    base = upstream_base_url.rstrip("/")
    bug_file = bug_path or meter.path.with_name("meter_bugs.jsonl")

    key_bytes = upstream_key.encode()

    def redact(b: bytes) -> bytes:
        return b.replace(key_bytes, b"[redacted]") if key_bytes else b

    def finish(status: int, content: bytes, cost_pu: int, reserved_pu: int) -> Response:
        """Pass an upstream body through, redacted. Non-JSON bodies are wrapped in a JSON error."""
        content = redact(content)
        try:
            json.loads(content)
        except ValueError:
            content = json.dumps(
                {"error": {"type": "upstream_error", "message": content.decode("utf-8", "replace")[:500]}}
            ).encode()
        return Response(
            content=content,
            status_code=status,
            media_type="application/json",
            headers={
                "X-Inherit-Cost-USD": repr(cost_pu / PICO),
                "X-Inherit-Reserved-USD": repr(reserved_pu / PICO),
            },
        )

    def ledger_error() -> JSONResponse:
        return _err(500, "ledger_error", "could not record the cost; the reservation stays counted")

    def safe(fn: Callable, *args: Any) -> bool:
        try:
            fn(*args)
            return True
        except Exception as e:
            log.error("ledger write failed: %s", type(e).__name__)
            return False

    def forward(
        path: str, content: bytes, model: str, ceiling: int, arm: str, extra: dict, tag: str | None = None
    ) -> Response:
        start = time.monotonic()
        for attempt in range(MAX_RETRIES + 1):
            remaining = deadline - (time.monotonic() - start)
            if attempt > 0 and remaining <= 0:
                return _err(504, "deadline_exceeded", "proxy deadline reached; no new attempt started")
            try:
                r = meter.reserve_pu(ceiling, arm, model, tag)
            except BudgetExceeded as e:
                return _err(402, "budget_exceeded", str(e))
            try:
                # Wall-clock bound on the whole attempt: each network operation is limited to
                # min(per-attempt timeout, time left), and the clock is checked between chunks.
                with http.stream(
                    "POST",
                    base + path,
                    content=content,
                    headers={"Authorization": f"Bearer {upstream_key}", "Content-Type": "application/json"},
                    timeout=min(timeout, max(remaining, 0.05)),
                ) as resp:
                    chunks = []
                    for chunk in resp.iter_bytes():
                        chunks.append(chunk)
                        if time.monotonic() - start >= deadline:
                            raise _DeadlineHit()
                    data = b"".join(chunks)
                    status = resp.status_code
            except _DeadlineHit:
                log.warning("upstream %s aborted at the deadline", path)
                if not safe(meter.settle_full, r):  # billing unknown: keep the whole ceiling
                    return ledger_error()
                return _err(504, "deadline_exceeded", "proxy deadline reached during the upstream response")
            except httpx2.HTTPError as e:
                log.warning("upstream %s failed: %s", path, type(e).__name__)
                if not safe(meter.settle_full, r):  # sent, outcome unknown: keep the whole ceiling
                    return ledger_error()
                return _err(502, "upstream_unreachable", type(e).__name__)
            if 400 <= status < 500:
                if not safe(meter.settle_unbilled, r, status):
                    return ledger_error()
                if status == 429 and attempt < MAX_RETRIES:
                    sleep(2**attempt)
                    continue
                return finish(status, data, 0, r.amount)
            if status != 200:
                if not safe(meter.settle_full, r):
                    return ledger_error()
                if status >= 500 and attempt < MAX_RETRIES:
                    sleep(2**attempt)
                    continue
                if status >= 500:
                    return finish(status, data, r.amount, r.amount)
                return _err(502, "unexpected_upstream_status", str(status))
            try:
                usage = json.loads(data)["usage"]
                pt = usage["prompt_tokens"]
                ct = usage.get("completion_tokens", 0) if path.endswith("/embeddings") else usage["completion_tokens"]
                cost_pu = meter.cost_pu(model, pt, ct)
            except Exception:
                if not safe(meter.settle_full, r):
                    return ledger_error()
                return _err(502, "bad_upstream_response", "upstream 200 without valid usage")
            if not safe(lambda: meter.settle(r, pt, ct, **extra)):
                safe(meter.settle_full, r)  # else the open reservation still counts at its ceiling
                return ledger_error()
            if cost_pu > r.amount:
                log.error("usage above reservation arm=%s model=%s reserved=%s cost=%s", arm, model, r.amount, cost_pu)
                try:
                    with open(bug_file, "a") as f:
                        f.write(json.dumps({"ts": time.time(), "arm": arm, "model": model, "path": path,
                                            "reserved": r.amount, "cost": cost_pu}) + "\n")
                except Exception as e:
                    log.error("could not write %s: %s", bug_file, type(e).__name__)
            return finish(200, data, cost_pu, r.amount)
        raise AssertionError("unreachable")

    def tag_of(request: Request) -> str | None:
        return request.headers.get(TAG_HEADER, "").strip()[:MAX_TAG_LEN] or None

    def parse(raw: bytes, request: Request) -> tuple[str, dict] | JSONResponse:
        arm = request.headers.get(ARM_HEADER, "").strip()
        if not arm:
            return _err(400, "missing_arm", f"header {ARM_HEADER} is required")
        try:
            payload = json.loads(raw)
        except ValueError:
            return _err(400, "invalid_json", "body is not JSON")
        if not isinstance(payload, dict):
            return _err(400, "invalid_json", "body must be an object")
        if payload.get("model") not in meter.prices:
            return _err(400, "unknown_model", "model is not in the price table")
        if payload.get("stream"):
            return _err(400, "unsupported", "stream is not supported")
        if payload.get("n", 1) != 1:
            return _err(400, "unsupported", "n must be 1")
        return arm, payload

    def chat_sync(raw: bytes, request: Request) -> Response:
        parsed = parse(raw, request)
        if isinstance(parsed, JSONResponse):
            return parsed
        arm, body = parsed
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return _err(400, "invalid_request", "messages must be a non-empty list")
        limits = [body[k] for k in ("max_tokens", "max_completion_tokens") if body.get(k) is not None]
        if any(not _pos_int(v) for v in limits):
            return _err(400, "invalid_request", "max_tokens must be a positive integer")
        if not limits:
            body["max_tokens"] = default_max_tokens
            limits = [default_max_tokens]
        extra: dict = {}
        if "temperature" in body and body["temperature"] not in (0, None):
            extra["temperature_overridden"] = True
        body["temperature"] = 0
        content = json.dumps(body, ensure_ascii=False).encode("utf-8")
        price_in, price_out = meter.prices[body["model"]]
        in_bound = len(content) + PER_MESSAGE_TOKENS * len(msgs) + FIXED_OVERHEAD_TOKENS
        ceiling = in_bound * price_in + max(limits) * price_out
        return forward("/chat/completions", content, body["model"], ceiling, arm, extra, tag_of(request))

    def embed_sync(raw: bytes, request: Request) -> Response:
        parsed = parse(raw, request)
        if isinstance(parsed, JSONResponse):
            return parsed
        arm, body = parsed
        inp = body.get("input")
        if isinstance(inp, str):
            n = 1
        elif isinstance(inp, list) and inp and all(isinstance(t, str) for t in inp):
            n = len(inp)
        else:
            return _err(400, "invalid_request", "input must be a string or a non-empty list of strings")
        price_in, _ = meter.prices[body["model"]]
        ceiling = (len(raw) + PER_MESSAGE_TOKENS * n) * price_in
        return forward("/embeddings", raw, body["model"], ceiling, arm, {}, tag_of(request))

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await run_in_threadpool(chat_sync, await request.body(), request)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await run_in_threadpool(embed_sync, await request.body(), request)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": m, "object": "model"} for m in meter.prices]}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/spend")
    def spend():
        return meter.snapshot()

    @app.post("/admin/arm_cap")
    async def arm_cap(request: Request):
        try:
            b = await request.json()
            await run_in_threadpool(meter.set_arm_cap, b["arm"], b["usd"])
        except (ValueError, KeyError, TypeError) as e:
            return _err(400, "invalid_request", f"need {{arm, usd}}: {type(e).__name__}")
        return {"ok": True}

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    s = load_settings()
    app = create_app(
        upstream_base_url=s.nebius_base_url,
        upstream_key=load_upstream_secret(),
        meter=Meter(s.budget_usd, s.meter_path),
        default_max_tokens=s.default_max_tokens,
        deadline=s.proxy_deadline_s,
    )
    uvicorn.run(app, host="127.0.0.1", port=s.proxy_port, log_level="info")


if __name__ == "__main__":
    main()
