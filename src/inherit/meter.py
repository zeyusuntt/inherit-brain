"""The spend meter: an append-only ledger shared by threads and processes.

The Meter keeps an append-only log (runs/meter.jsonl) shared by threads and
processes. Money is integer pico-USD (1 USD = 10**12) so sums are exact, and
prices come from the table in config, never from the caller.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from inherit.config import MODEL, PRICES_PICO

PICO = 10**12  # pico-USD per USD
PER_MESSAGE_TOKENS = 16
FIXED_OVERHEAD_TOKENS = 512  # chat template and tool rendering


class BudgetExceeded(RuntimeError):
    pass


def _pu_ceil(usd: float) -> int:
    """USD to pico-USD, rounded up. Rejects negative, NaN and infinite input."""
    if isinstance(usd, bool) or not isinstance(usd, (int, float)) or not math.isfinite(usd) or usd < 0:
        raise ValueError(f"invalid amount: {usd!r}")
    return int((Decimal(repr(float(usd))) * PICO).to_integral_value(rounding=ROUND_CEILING))


def _count(n: Any, name: str) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"invalid {name}: {n!r}")
    return n


def _str(s: Any) -> str:
    if not isinstance(s, str) or not s:
        raise ValueError(f"invalid string: {s!r}")
    return s


@dataclass(frozen=True)
class Reservation:
    id: str
    arm: str
    amount: int  # pico-USD
    model: str = MODEL
    tag: str | None = None  # opaque attempt label from X-Inherit-Tag, copied to the ledger events


class Meter:
    def __init__(
        self,
        cap_usd: float,
        path: str | Path,
        arm_caps: dict[str, float] | None = None,
    ):
        self.cap = _pu_ceil(cap_usd)
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.prices = dict(PRICES_PICO)
        for m, pr in self.prices.items():
            if len(pr) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in pr):
                raise ValueError(f"invalid price for {m!r}: {pr!r}")
        self.arm_caps: dict[str, int] = {}
        self.corrupt_lines = 0
        self._tlock = threading.Lock()
        self._offset = 0
        self._open: dict[str, tuple[str, int, str]] = {}  # id -> (arm, amount, model)
        self._settled: dict[str, int] = {}  # arm -> pico-USD
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for arm, usd in (arm_caps or {}).items():
            self.set_arm_cap(arm, usd)

    # -- ledger plumbing ---------------------------------------------------

    def _refresh(self) -> None:
        """Apply events appended since the last read. Call with the locks held."""
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            data = f.read()
        end = data.rfind(b"\n") + 1  # a partial trailing line waits for its newline
        for line in data[:end].splitlines():
            if not line.strip():
                continue
            try:
                ev = self._parse(json.loads(line))
            except (ValueError, KeyError, TypeError):
                self.corrupt_lines += 1
                continue
            self._apply(ev)  # only fully validated events change state
        self._offset += end

    @staticmethod
    def _parse(ev: dict) -> tuple:
        """Validate one ledger line into a typed event without touching state."""
        if not isinstance(ev, dict):
            raise ValueError("not an object")
        kind = ev.get("type")
        if kind is None:  # lines written before the "type" field existed
            kind = "settle" if "cost" in ev else "reserve" if "amount" in ev else None
        if kind == "reserve":
            return ("reserve", _str(ev["id"]), _str(ev["arm"]), _count(ev["amount"], "amount"),
                    _str(ev.get("model", MODEL)))
        if kind == "settle":
            _count(ev.get("prompt_tokens", 0), "prompt_tokens")
            _count(ev.get("completion_tokens", 0), "completion_tokens")
            return ("settle", _str(ev["id"]), _str(ev["arm"]), _count(ev["cost"], "cost"))
        if kind == "cap":
            return ("cap", _str(ev["arm"]), _count(ev["amount"], "amount"))
        raise ValueError(f"unknown event {kind!r}")

    def _apply(self, ev: tuple) -> None:
        if ev[0] == "reserve":
            self._open[ev[1]] = (ev[2], ev[3], ev[4])
        elif ev[0] == "settle":
            self._open.pop(ev[1], None)
            self._settled[ev[2]] = self._settled.get(ev[2], 0) + ev[3]
        else:
            self.arm_caps[ev[1]] = ev[2]

    def _append(self, ev: dict[str, Any]) -> None:
        self._parse(ev)  # write only what replay can read; raises ValueError otherwise
        line = (json.dumps(ev) + "\n").encode()
        with open(self.path, "ab") as f:
            if f.tell() > 0:
                with open(self.path, "rb") as r:
                    r.seek(-1, os.SEEK_END)
                    if r.read(1) != b"\n":
                        line = b"\n" + line  # close a torn trailing line first
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    @contextmanager
    def _locked(self):
        with self._tlock:
            with open(self.lock_path, "a") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    self._refresh()
                    yield
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)

    def _arm_total(self, arm: str) -> int:
        return self._settled.get(arm, 0) + sum(a for ar, a, _ in self._open.values() if ar == arm)

    def _total(self) -> int:
        return sum(self._settled.values()) + sum(a for _, a, _ in self._open.values())

    # -- public API --------------------------------------------------------

    def set_arm_cap(self, arm: str, usd: float) -> None:
        amount = _pu_ceil(usd)
        with self._locked():
            self._append({"type": "cap", "arm": arm, "amount": amount})
            self._refresh()

    def reserve(self, max_cost: float, arm: str, model: str = MODEL) -> Reservation:
        return self.reserve_pu(_pu_ceil(max_cost), arm, model)

    def reserve_pu(self, amount: int, arm: str, model: str = MODEL, tag: str | None = None) -> Reservation:
        _count(amount, "amount")
        if model not in self.prices:
            raise ValueError(f"no price for model {model!r}")
        with self._locked():
            if self._total() + amount > self.cap:
                raise BudgetExceeded(
                    f"cap {self.cap / PICO} USD: committed {self._total() / PICO} + {amount / PICO}"
                )
            sub = self.arm_caps.get(arm)
            if sub is not None and self._arm_total(arm) + amount > sub:
                raise BudgetExceeded(
                    f"arm {arm!r} cap {sub / PICO} USD: committed {self._arm_total(arm) / PICO} + {amount / PICO}"
                )
            r = Reservation(uuid.uuid4().hex, arm, amount, model, tag)
            ev = {"type": "reserve", "id": r.id, "arm": arm, "amount": amount, "model": model}
            if tag:
                ev["tag"] = tag
            self._append(ev)
            self._refresh()
            return r

    def _settle(self, r: Reservation, cost: int, pt: int, ct: int, **extra: Any) -> float:
        with self._locked():
            rec = self._open.get(r.id)
            if rec is None:
                raise ValueError(f"reservation {r.id} is not open")
            if rec != (r.arm, r.amount, r.model):
                raise ValueError(f"reservation {r.id} does not match the ledger")
            ev = {
                **extra,
                "type": "settle",
                "id": r.id,
                "arm": rec[0],
                "cost": cost,
                "prompt_tokens": pt,
                "completion_tokens": ct,
            }
            if r.tag:
                ev["tag"] = r.tag
            if cost > rec[1]:
                ev["over_reservation"] = True
            self._append(ev)
            self._refresh()
        return cost / PICO

    def settle(
        self, r: Reservation, prompt_tokens: int, completion_tokens: int, **extra: Any
    ) -> float:
        """Replace a reservation with the real cost (table price x real tokens). Returns USD."""
        pt, ct = _count(prompt_tokens, "prompt_tokens"), _count(completion_tokens, "completion_tokens")
        price_in, price_out = self.prices[r.model]
        return self._settle(r, pt * price_in + ct * price_out, pt, ct, **extra)

    def cost_pu(self, model: str, prompt_tokens: int, completion_tokens: int) -> int:
        price_in, price_out = self.prices[model]
        return _count(prompt_tokens, "prompt_tokens") * price_in + _count(
            completion_tokens, "completion_tokens"
        ) * price_out

    def snapshot(self) -> dict[str, Any]:
        """Totals in USD for /spend."""
        with self._locked():
            arms = set(self._settled) | {a for a, _, _ in self._open.values()} | set(self.arm_caps)
            return {
                "cap_usd": self.cap / PICO,
                "spent_usd": sum(self._settled.values()) / PICO,
                "open_reserved_usd": sum(a for _, a, _ in self._open.values()) / PICO,
                "open_reservations": len(self._open),
                "committed_usd": self._total() / PICO,
                "per_arm": {
                    a: {
                        "committed_usd": self._arm_total(a) / PICO,
                        "cap_usd": self.arm_caps[a] / PICO if a in self.arm_caps else None,
                    }
                    for a in sorted(arms)
                },
                "corrupt_lines": self.corrupt_lines,
            }

    def settle_full(self, r: Reservation) -> float:
        """Charge the whole reservation: the request may have been billed, outcome unknown."""
        return self._settle(r, r.amount, 0, 0)

    def settle_unbilled(self, r: Reservation, status_code: int) -> float:
        """The provider rejected the request (HTTP 4xx): nothing was billed."""
        return self._settle(r, 0, 0, 0, status_code=status_code)

    def spent(self, arm: str | None = None) -> float:
        """Settled cost plus open reservations, in USD."""
        with self._locked():
            return (self._total() if arm is None else self._arm_total(arm)) / PICO
