"""Cached, metered JSON calls for the model extraction layers (shared by 4b and 4c).

Every call goes through `inherit.llm.chat` (the metering proxy). Responses are cached on disk under
`<cache_root>/<prompt version>/<sha256 of the exact request>.json`, so a rerun costs nothing. Output cut off at
max_tokens is retried once with double max_tokens; a second failure is recorded, never silently dropped. In `dry`
mode nothing is sent or written: misses are counted with a token estimate, which is what `--estimate` prints.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from inherit.config import MODEL, PRICES_PICO, ROOT

CACHE_ROOT = ROOT / "data" / "cache" / "extract"
RESPONSE_FORMAT = {"type": "json_object"}
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def request_sha(messages: list[dict], max_tokens: int) -> str:
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "response_format": RESPONSE_FORMAT,
               "temperature": 0}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def parse_json_object(text) -> dict | None:
    t = (text or "").strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1)
    try:
        data = json.loads(t)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


class Caller:
    def __init__(self, chat, cache_root, version: str, *, arm: str = "extract", dry: bool = False):
        self.chat, self.arm, self.dry = chat, arm, dry
        self.dir = Path(cache_root if cache_root is not None else CACHE_ROOT) / version
        self.calls: Counter = Counter()  # live requests by kind, retries included
        self.cached: Counter = Counter()
        self.prompt_tokens = self.completion_tokens = 0
        self.cost_usd = 0.0
        self.est_in = self.est_out = 0  # dry mode token estimates for the misses

    def json_call(self, kind: str, messages: list[dict], max_tokens: int,
                  check=None) -> tuple[dict | None, str | None]:
        """(parsed object, None) or (None, reason). reason: truncated, invalid_json, invalid_shape. `check(data)` returns
        an error string for a response of the wrong shape: such an answer is never cached or returned, and a cached one
        that fails it is asked again."""
        path = self.dir / f"{request_sha(messages, max_tokens)}.json"
        if path.exists():
            hit = json.loads(path.read_text(encoding="utf-8"))["response"]
            if check is None or not check(hit):
                self.cached[kind] += 1
                return hit, None
        if self.dry:
            self.calls[kind] += 1
            self.est_in += len(json.dumps(messages, ensure_ascii=False)) // 4
            self.est_out += max_tokens // 2
            return {}, None
        reason = None
        for attempt, mt in enumerate((max_tokens, 2 * max_tokens)):
            self.calls[kind] += 1
            res = self.chat(messages, arm=self.arm, response_format=RESPONSE_FORMAT, max_tokens=mt)
            self.prompt_tokens += res.prompt_tokens
            self.completion_tokens += res.completion_tokens
            self.cost_usd += res.cost_usd
            if res.completion_tokens >= mt:
                reason = "truncated"
                continue
            data = parse_json_object(res.message.content)
            if data is None:
                return None, "invalid_json"
            if check is not None and check(data):
                return None, "invalid_shape"
            self.dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"response": data, "attempts": attempt + 1, "max_tokens": mt,
                                        "completion_tokens": res.completion_tokens}, ensure_ascii=False,
                                       sort_keys=True), encoding="utf-8")
            return data, None
        return None, reason

    def projected_usd(self) -> float:
        pin, pout = PRICES_PICO[MODEL]
        return (self.est_in * pin + self.est_out * pout) / 1e12
