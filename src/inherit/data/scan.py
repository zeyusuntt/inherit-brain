"""One recursive scanner for label keys, shared by make_bundle and the audit."""

from __future__ import annotations

FORBIDDEN_KEYS = frozenset({
    "team", "customers", "answerable_questions", "unanswerable_questions", "ground_truth", "citations", "question",
})


def find_forbidden_keys(obj, where: str) -> list[str]:
    """Messages for every forbidden key found at any depth in dicts and lists."""
    found: list[str] = []
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k in FORBIDDEN_KEYS:
                    found.append(f"{where}: forbidden key {k!r}")
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)
    return found
