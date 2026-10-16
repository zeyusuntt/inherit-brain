"""Group a channel's messages into sessions split at long silences."""

from __future__ import annotations

from datetime import datetime, timedelta


def sessions(messages, gap_hours: float = 6) -> list[list]:
    """Sort by `ts` (ISO string) and start a new session when the gap to the previous message is >= gap_hours."""
    gap = timedelta(hours=gap_hours)
    ordered = sorted(messages, key=lambda m: datetime.fromisoformat(m.ts))
    out: list[list] = []
    prev = None
    for m in ordered:
        t = datetime.fromisoformat(m.ts)
        if prev is None or t - prev >= gap:
            out.append([])
        out[-1].append(m)
        prev = t
    return out
