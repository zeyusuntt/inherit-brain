"""Citation validation shared by the model layers (R11). No model calls.

A citation is `{"record_id", "quote"}` plus an optional `field` and, where an ID repeats, a `position`.
It is valid when the normalised quote is a non-empty substring of the normalised text of a record that was
supplied to that call, with an allowed (source type, field). The supporting span is kept. Whether the quote
actually supports the claim is left to the acceptance audit.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from inherit.data.records import ProductRecords

_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
})
_WS = re.compile(r"\s+")


def normalise(text) -> str:
    """Quotes straightened, whitespace collapsed to single spaces and trimmed, case-folded."""
    return _WS.sub(" ", str(text or "").translate(_QUOTES)).strip().casefold()


@dataclass(frozen=True)
class SourceRecord:
    """Text a call may cite. `source_type`: message, segment, chat, document, pr, url. `position` is the stable
    position of a record whose ID repeats (a URL record, a review of a PR)."""

    id: str
    source_type: str
    field: str
    text: str
    position: int | None = None


@dataclass(frozen=True)
class Citation:
    record_id: str
    quote: str
    field: str | None = None
    position: int | None = None


@dataclass(frozen=True)
class CiteResult:
    ok: bool
    reason: str  # ok, empty_quote, record_not_supplied, field_not_supplied, field_not_allowed,
    #              position_not_supplied, quote_not_found
    span: tuple[int, int] | None = None  # offsets of the quote inside normalise(record.text)
    position: int | None = None
    record: SourceRecord | None = None


def _parts(c) -> tuple[str, object, str | None, int | None]:
    if isinstance(c, Citation):
        return c.record_id, c.quote, c.field, c.position
    return c.get("record_id"), c.get("quote"), c.get("field"), c.get("position")


def validate_citation(citation, supplied, allowed) -> CiteResult:
    """`supplied`: the SourceRecords given to the call. `allowed`: set of (source_type, field)."""
    rid, quote, field, position = _parts(citation)
    nq = normalise(quote)
    if not nq:
        return CiteResult(False, "empty_quote")
    cands = [r for r in supplied if r.id == rid]
    if not cands:
        return CiteResult(False, "record_not_supplied")
    if field is not None:
        cands = [r for r in cands if r.field == field]
        if not cands:
            return CiteResult(False, "field_not_supplied")
    cands = [r for r in cands if (r.source_type, r.field) in allowed]
    if not cands:
        return CiteResult(False, "field_not_allowed")
    if position is not None:
        cands = [r for r in cands if r.position == position]
        if not cands:
            return CiteResult(False, "position_not_supplied")
    for r in cands:
        i = normalise(r.text).find(nq)
        if i >= 0:
            return CiteResult(True, "ok", (i, i + len(nq)), r.position, r)
    return CiteResult(False, "quote_not_found")


def validate_items(items, supplied, allowed, key: str = "citations") -> tuple[list[dict], dict[str, int]]:
    """Keep each item with at least one valid citation, trimmed to its valid citations (each gains `span` and, where
    the ID repeats, `position`). Dropped items are counted by the reason of their first failing citation."""
    kept: list[dict] = []
    dropped: Counter = Counter()
    for item in items:
        cites = item.get(key) or []
        good, first_bad = [], None
        for c in cites:
            r = validate_citation(c, supplied, allowed)
            if r.ok:
                d = dict(c) if isinstance(c, dict) else {"record_id": c.record_id, "quote": c.quote}
                d["span"] = list(r.span)
                if r.position is not None:
                    d["position"] = r.position
                good.append(d)
            elif first_bad is None:
                first_bad = r.reason
        if good:
            kept.append({**item, key: good})
        else:
            dropped[first_bad or "no_citation"] += 1
    return kept, dict(dropped)


def source_records(rec: ProductRecords) -> list[SourceRecord]:
    """Every citable text of one product: Slack messages, transcript segments, meeting chats, document content and
    feedback, PR title, summary and each review comment, URL descriptions."""
    out: list[SourceRecord] = []
    out += [SourceRecord(m.id, "message", "text", m.text) for m in rec.messages]
    for mt in rec.meetings:
        out += [SourceRecord(s.id, "segment", "text", s.text) for s in mt.segments]
    out += [SourceRecord(c.id, "chat", "text", c.text) for c in rec.chats]
    for d in rec.documents:
        out.append(SourceRecord(d.id, "document", "content", d.content))
        if d.feedback:
            out.append(SourceRecord(d.id, "document", "feedback", d.feedback))
    for p in rec.prs:
        out.append(SourceRecord(p.id, "pr", "title", p.title))
        out.append(SourceRecord(p.id, "pr", "summary", p.summary))
        out += [SourceRecord(p.id, "pr", "review", r.comment, r.position) for r in p.reviews]
    for u in rec.urls:
        out += [SourceRecord(u.id, "url", "description", desc, pos) for desc, pos in zip(u.descriptions, u.positions)]
    return out
