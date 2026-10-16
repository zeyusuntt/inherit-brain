"""Deterministic helpers for the semantic layer (Task 4c). No model calls.

Feedback bullets, competitor demo links, the routing rules that decide which Slack sessions go to the model, session
splitting, the previous-release cue, and PR reference resolution against the catalogue.
"""

from __future__ import annotations

import re

from inherit.extract.cite import normalise

BULLET = re.compile(r"^\s*(?:[-*•])\s+(.*\S)\s*$")
DEMO_DESC = re.compile(r"\bdemo showing how (.+?) works\b", re.IGNORECASE)
# Evidence in a session that a customer or QA remark concerns the previous release. A channel or product name never
# counts (R2); the words must be in the text.
PREV_CUE = re.compile(
    r"\b(?:last|previous|prior|latest|older) (?:release|version|team)\b|\bdifferent team\b"
    r"|\b(?:didn't|did not) develop this release\b", re.IGNORECASE)
URL_IN_TEXT = re.compile(r"https?://")
PR_NUMBER = re.compile(r"\bPRs?\s*#?\s*\d+", re.IGNORECASE)
PR_LINK = re.compile(r"https?://github\.com/([^/\s]+)/([^/\s]+)/pull/\d+")
DOC_WORDS = r"(?:Market Research Report|Product Vision Document|Product Requirements Document" \
            r"|Technical Specifications Document|System Design Document)"
# Ordered: the first cue that matches names the routing reason. `demo` is not a cue: demo links are rules.
CUES = (
    ("competitor", re.compile(r"competitor|\brivals?\b", re.IGNORECASE)),
    ("pr_ref", re.compile(PR_NUMBER.pattern + r"|/pull/\d+", re.IGNORECASE)),
    ("previous_release", re.compile(r"\b(?:last|previous|prior|latest) (?:release|version)\b", re.IGNORECASE)),
    ("qa", re.compile(r"\bQA\b")),
    ("revert", re.compile(r"\brevert", re.IGNORECASE)),
    ("merge", re.compile(r"\bmerg(?:e|ed|ing)\b", re.IGNORECASE)),
    ("hold", re.compile(r"\bhold(?:ing)? (?:off|on|the|this|it|back)\b|\bon hold\b", re.IGNORECASE)),
    ("product_direction", re.compile(r"product direction", re.IGNORECASE)),
    ("customer", re.compile(r"\bCUST-\d+")),
    ("document", re.compile(r"archives/docs/|" + DOC_WORDS, re.IGNORECASE)),
)


def parse_feedback_items(feedback) -> list[str]:
    """Anonymous feedback bullets of a document. Text with no bullets is one item."""
    text = (feedback or "").strip()
    if not text:
        return []
    items = [m.group(1) for m in (BULLET.match(line) for line in text.splitlines()) if m]
    return items or [text]


def competitor_of_demo(description: str, own_names) -> str | None:
    """The product a `A demo showing how <X> works.` description names, unless X is this product or one of its aliases."""
    m = DEMO_DESC.search(description or "")
    if not m:
        return None
    name = m.group(1).strip()
    return None if normalise(name) in {normalise(n) for n in own_names} else name


def route_session(msgs) -> tuple[bool, str]:
    """(send to the model, reason). Planning channels always go (R10); bug channels never (4b); other channels go when
    any message carries a cue."""
    real = [m for m in msgs if not m.is_bot]
    name = real[0].channel_name if real else ""
    if name.startswith("bug-"):
        return False, "bug_channel"
    if name.startswith("planning-"):
        return True, "planning"
    for reason, rx in CUES:
        if any(rx.search(m.text) for m in real):
            return True, reason
    return False, "no_cue"


def split_session(msgs, max_chars: int) -> list[list]:
    """Consecutive chunks of whole messages, each at most `max_chars` of message text (a single longer message stays
    alone)."""
    out: list[list] = []
    size = 0
    for m in msgs:
        n = len(m.text)
        if not out or size + n > max_chars:
            out.append([])
            size = 0
        out[-1].append(m)
        size += n
    return out


def pr_ref_number(ref) -> str | None:
    """The PR number in a reference like `8`, `PR# 8` or `#12`; None for a link or text with no number."""
    ref = str(ref or "")
    if PR_LINK.search(ref):
        return None
    m = re.search(r"\d+", ref)
    return m.group(0) if m else None


def resolve_pr(ref, cat):
    """(PrEntry, 'ok') for a reference to one product PR; (None, why) otherwise. why: unknown, external, ambiguous.
    A number resolves only inside the product's repositories (R3); a number found in several of them is ambiguous."""
    ref = str(ref or "").strip()
    m = PR_LINK.search(ref)
    if m:
        p = cat.pr(m.group(0))
        if p is None:
            return None, "unknown"
        return (p, "ok") if p.repo_scope == "product" else (None, "external")
    n = pr_ref_number(ref)
    if n is None:
        return None, "unknown"
    hits = [p for p in cat.prs if p.number == n and p.repo_scope == "product"]
    if len(hits) == 1:
        return hits[0], "ok"
    return None, ("ambiguous" if hits else "unknown")
