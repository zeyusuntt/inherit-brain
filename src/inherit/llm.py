"""The only door to the model. Every call is reserved and settled on a Meter.

The Meter keeps an append-only log (runs/meter.jsonl) shared by threads and
processes. Money is integer pico-USD so sums are exact.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PICO = 10**12  # pico-USD per USD


class BudgetExceeded(RuntimeError):
    pass


def _pu(usd: float) -> int:
    return round(usd * PICO)


@dataclass(frozen=True)
class Reservation:
    id: str
    arm: str
    amount: int  # pico-USD


class Meter:
    def __init__(
        self,
        cap_usd: float,
        path: str | Path,
        price_in: float,
        price_out: float,
        arm_caps: dict[str, float] | None = None,
    ):
        self.cap = _pu(cap_usd)
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.price_in = price_in
        self.price_out = price_out
        self.arm_caps = {a: _pu(v) for a, v in (arm_caps or {}).items()}
        self._tlock = threading.Lock()
        self._offset = 0
        self._open: dict[str, tuple[str, int]] = {}  # id -> (arm, amount)
        self._settled: dict[str, int] = {}  # arm -> pico-USD
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def set_arm_cap(self, arm: str, usd: float) -> None:
        with self._tlock:
            self.arm_caps[arm] = _pu(usd)

    # State is rebuilt from the log; call only with the locks held (or for reads).
    def _refresh(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            data = f.read()
        end = data.rfind(b"\n") + 1  # ignore a partial trailing line
        for line in data[:end].splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            if "cost" in ev:
                self._open.pop(ev["id"], None)
                self._settled[ev["arm"]] = self._settled.get(ev["arm"], 0) + ev["cost"]
            else:
                self._open[ev["id"]] = (ev["arm"], ev["amount"])
        self._offset += end

    def _append(self, ev: dict[str, Any]) -> None:
        with open(self.path, "ab") as f:
            f.write((json.dumps(ev) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())

    class _Locked:
        def __init__(self, m: "Meter"):
            self.m = m

        def __enter__(self):
            self.m._tlock.acquire()
            self.fh = open(self.m.lock_path, "a")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
            self.m._refresh()

        def __exit__(self, *exc):
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
            self.m._tlock.release()

    def _arm_total(self, arm: str) -> int:
        return self._settled.get(arm, 0) + sum(a for ar, a in self._open.values() if ar == arm)

    def _total(self) -> int:
        return sum(self._settled.values()) + sum(a for _, a in self._open.values())

    def reserve(self, max_cost: float, arm: str) -> Reservation:
        amount = _pu(max_cost)
        with self._Locked(self):
            if self._total() + amount > self.cap:
                raise BudgetExceeded(
                    f"cap {self.cap / PICO} USD: committed {self._total() / PICO} + {amount / PICO}"
                )
            sub = self.arm_caps.get(arm)
            if sub is not None and self._arm_total(arm) + amount > sub:
                raise BudgetExceeded(
                    f"arm {arm!r} cap {sub / PICO} USD: committed {self._arm_total(arm) / PICO} + {amount / PICO}"
                )
            r = Reservation(uuid.uuid4().hex, arm, amount)
            self._append({"id": r.id, "arm": arm, "amount": amount})
            self._refresh()
            return r

    def settle(
        self,
        r: Reservation,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float | None = None,
    ) -> float:
        """Replace a reservation with the real cost. Returns the cost in USD."""
        if cost_usd is None:
            cost_usd = prompt_tokens * self.price_in + completion_tokens * self.price_out
        cost = _pu(cost_usd)
        with self._Locked(self):
            if r.id not in self._open:
                raise ValueError(f"reservation {r.id} is not open")
            self._append(
                {
                    "id": r.id,
                    "arm": r.arm,
                    "cost": cost,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
            )
            self._refresh()
        return cost / PICO

    def settle_full(self, r: Reservation) -> float:
        """Charge the whole reservation (request may have been billed, outcome unknown)."""
        return self.settle(r, 0, 0, cost_usd=r.amount / PICO)

    def spent(self, arm: str | None = None) -> float:
        """Settled cost plus open reservations, in USD."""
        with self._Locked(self):
            return (self._total() if arm is None else self._arm_total(arm)) / PICO


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


_meter: Meter | None = None
_client: Any = None
_meter_lock = threading.Lock()


def _settings():
    from inherit.config import load_settings

    return load_settings()


def default_meter() -> Meter:
    global _meter
    with _meter_lock:
        if _meter is None:
            s = _settings()
            _meter = Meter(s.budget_usd, s.meter_path, s.price_in, s.price_out)
        return _meter


def _default_client():
    global _client
    with _meter_lock:
        if _client is None:
            from openai import OpenAI

            s = _settings()
            # max_retries=0: the SDK must never resend a request behind the meter's back.
            _client = OpenAI(api_key=s.nebius_api_key, base_url=s.nebius_base_url, max_retries=0)
        return _client


def _retryable(e: Exception) -> bool:
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    return isinstance(e, (openai.RateLimitError, openai.InternalServerError))


def chat(
    messages: list[dict],
    *,
    arm: str,
    tools: list[dict] | None = None,
    max_tokens: int = 800,
    response_format: dict | None = None,
    meter: Meter | None = None,
    client: Any = None,
    model: str | None = None,
    attempts: int = 3,
) -> ChatResult:
    meter = meter or default_meter()
    client = client or _default_client()
    price_in = meter.price_in
    chars = len(json.dumps(messages)) + (len(json.dumps(tools)) if tools else 0)
    estimate = chars / 3 * price_in + max_tokens * meter.price_out
    model = model or _settings().model
    kw: dict[str, Any] = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=0)
    if tools:
        kw["tools"] = tools
    if response_format is not None:
        kw["response_format"] = response_format
    for attempt in range(attempts):
        r = meter.reserve(estimate, arm)  # a fresh reservation per attempt
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(**kw)
        except Exception as e:
            meter.settle_full(r)
            if _retryable(e) and attempt + 1 < attempts:
                time.sleep(2**attempt)
                continue
            raise
        latency = time.monotonic() - t0
        pt, ct = resp.usage.prompt_tokens, resp.usage.completion_tokens
        cost = meter.settle(r, pt, ct)
        return ChatResult(resp.choices[0].message, pt, ct, cost, latency)
    raise AssertionError("unreachable")


def embed(
    texts: list[str],
    *,
    arm: str,
    model: str | None = None,
    meter: Meter | None = None,
    client: Any = None,
    price_in: float | None = None,
) -> EmbedResult:
    meter = meter or default_meter()
    client = client or _default_client()
    model = model or _settings().embed_model
    price_in = price_in if price_in is not None else _settings().embed_price_in
    estimate = sum(len(t) for t in texts) / 3 * price_in
    r = meter.reserve(estimate, arm)
    try:
        resp = client.embeddings.create(model=model, input=texts)
    except Exception:
        meter.settle_full(r)
        raise
    pt = resp.usage.prompt_tokens
    cost = meter.settle(r, pt, 0, cost_usd=pt * price_in)
    return EmbedResult([d.embedding for d in resp.data], pt, cost)
