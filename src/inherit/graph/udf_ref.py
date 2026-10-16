"""Python reference of herb.js statusAt, used to test the UDF. Same rules, same quirks.

Status after the last event with ts <= date. Equal timestamps: the later list entry wins. An event of an unknown
type still advances the cutoff but leaves the status unchanged.
"""

from __future__ import annotations

STATUS = {"RESOLVED": "resolved", "DISMISSED": "dismissed", "REPORTED": "active", "ASSIGNED": "active",
          "REOPENED": "active"}


def status_at(events, date):
    best = None
    status = None
    for typ, ts in events:
        if ts <= date and (best is None or ts >= best):
            best = ts
            status = STATUS.get(typ, status)
    return status
