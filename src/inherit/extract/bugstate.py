"""Official bug states derived from a bug's ordered events (Global Constraints, spec 5.3). No model calls.

unresolved    = no linked fix PR
active        = the last event is REPORTED, ASSIGNED or REOPENED
reopened      = at least one REOPENED
no_fix_needed = at least one DISMISSED
`resolved` is `not unresolved`. These are independent flags, not one exclusive status. Order is by the real record
timestamp (never by record id), ties by extraction sequence.
"""

from __future__ import annotations

DERIVED_BY = "bug_state/v1"
ACTIVE_TYPES = frozenset({"REPORTED", "ASSIGNED", "REOPENED"})


def _type(e) -> str:
    return e if isinstance(e, str) else e["type"] if isinstance(e, dict) else e.type


def sort_events(events: list[dict]) -> list[dict]:
    return sorted(events, key=lambda e: (e["ts"], e.get("seq", 0)))


def official_states(events, fix_prs) -> dict[str, bool]:
    """`events` in time order (strings, dicts with `type`, or objects with `.type`); `fix_prs` the linked fix PRs."""
    types = [_type(e) for e in events]
    return {
        "unresolved": not fix_prs,
        "active": bool(types) and types[-1] in ACTIVE_TYPES,
        "reopened": "REOPENED" in types,
        "no_fix_needed": "DISMISSED" in types,
    }


def derive_bug(events: list[dict], fix_prs) -> dict:
    """Derived props of a Bug node. `events`: dicts with type, ts, seq, assignee_eids (any order)."""
    ordered = sort_events(events)
    types = [e["type"] for e in ordered]
    assigned = [e for e in ordered if e["type"] == "ASSIGNED"]
    states = official_states(types, list(fix_prs))
    return {
        **states,
        "resolved": not states["unresolved"],
        "status": types[-1] if types else "REPORTED",
        "current_assignees": list(assigned[-1].get("assignee_eids") or []) if assigned else [],
        "event_types": types,
        "derived_by": DERIVED_BY,
    }
