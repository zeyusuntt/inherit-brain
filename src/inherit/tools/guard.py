"""The output guard: no ID in a final answer may be one the tools did not return. Two exact rules.

Rows are typed (see catalog.py): `answer` is the graded value (employee id, company name, PR or URL link, or text),
`refs` are label-qualified node ids a row names, `evidence` are record ids (edge `src`). Rows nest; the category of a
key is inherited by everything below it.

1. `guard(text, ids, results)`: every proposed answer id must be exactly one of the rows' `answer` values; others are
   dropped and listed (a qualified id, a ref or an evidence id is not an answer).
2. In the text, every ID-like token (employee eid, label-qualified node id, CUST- id, Slack record id, http(s) link)
   must equal a returned value or ref exactly, with complete token boundaries (no prefix or substring support). Complete
   returned ids are recognised whole first, so Markdown punctuation and multiword ids such as
   `Feature:p:voice assistant` pass. An unsupported token is removed from the text (not the sentence) and listed.

Free-text person and company names are not policed: the answer prompt tells the model to name only entities that
appear in tool results, and the README discloses that the guard validates ids and structured answers, not prose.
`guard_evidence(ids, results)` keeps the evidence ids that appear in some returned row.
"""

from __future__ import annotations

import re

from inherit.ontology import NODE_KEYS

_LABELS = "|".join(sorted(NODE_KEYS, key=len, reverse=True))
_STOP_CHARS = r"\s,;()`'\"*\[\]<>"
ID_RE = re.compile(
    rf"\b(?:{_LABELS}):[^{_STOP_CHARS}]+"  # label-qualified node id
    rf"|https?://[^{_STOP_CHARS}]+"  # link
    r"|\beid_[0-9a-f]{8}\b|\bEMP_\d+\b|\bCUST-\d+\b"  # employee and customer ids
    r"|\b\d{8}-\d+-[0-9a-f]{5}\b")  # Slack record id
_TRAIL = ".,;:!?"
_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CATEGORY = {"answer": "answer", "refs": "refs", "evidence": "evidence"}


def _rows(results) -> list[dict]:
    out: list[dict] = []
    for r in results:
        out.extend(r.rows if hasattr(r, "rows") else r)
    return out


def _walk(obj, bucket: dict, mode: str | None = None) -> None:
    """Collect every string under its category (answer, refs, evidence; anything else is 'other'). A key sets the
    category for its whole subtree; a container inherits the category of the key above it."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, bucket, _CATEGORY.get(k, mode))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, bucket, mode)
    elif isinstance(obj, str):
        bucket[mode or "other"].add(obj)


class _Support:
    def __init__(self, results):
        b = {"answer": set(), "refs": set(), "evidence": set(), "other": set()}
        for row in _rows(results):
            _walk(row, b)
        self.answers, self.evidence = b["answer"], b["evidence"]
        self.refs = b["refs"] | {r.split(":", 1)[1] for r in b["refs"] if ":" in r}  # a ref's bare key is the same id
        self.tokens = b["answer"] | self.refs | b["evidence"] | b["other"]


def _mask(sentence: str, sup: _Support) -> str:
    """Blank out the complete returned ids in the sentence, longest first, so what remains holds only unknown ids."""
    for tok in sorted((t for t in sup.tokens if len(t) > 3 and (":" in t or "/" in t or "_" in t or "-" in t)),
                      key=len, reverse=True):
        sentence = re.sub(rf"(?<![\w:/-]){re.escape(tok)}(?![\w:/-])", lambda m: "\x00" * len(m.group(0)), sentence)
    return sentence


def _remove(sentence: str, value: str) -> str:
    """Take one token out of a sentence together with its list punctuation."""
    v = re.escape(value)
    for pat in (rf",\s*{v}(?=\s+(?:and|or)\b)", rf"\s+(?:and|or)\s+{v}", rf",\s*{v}", rf"{v}\s*(?:,|\s+and|\s+or)\s+"):
        new = re.sub(pat, "", sentence, count=1)
        if new != sentence:
            return new
    return sentence.replace(value, "", 1)


def guard(text: str, ids: list[str], results) -> tuple[str, list[str], list[dict]]:
    """(text, kept answer ids, dropped). `dropped` lists {"value", "where": "ids"|"text", "reason"}."""
    sup = _Support(results)
    dropped: list[dict] = []
    kept = []
    for i in ids:
        if i in sup.answers:
            kept.append(i)
        else:
            dropped.append({"value": i, "where": "ids", "reason": "not an answer value"})
    out_lines = []
    for line in text.split("\n"):
        out = []
        for sent in (s for s in _SPLIT.split(line) if s.strip()):
            bad = [m.group(0).rstrip(_TRAIL) for m in ID_RE.finditer(_mask(sent, sup))]
            bad = [t for t in dict.fromkeys(bad) if t not in sup.tokens]
            for t in bad:
                dropped.append({"value": t, "where": "text", "reason": "not in any returned row"})
                sent = _remove(sent, t)
            sent = " ".join(re.sub(r"`\s*`|\*\*\s*\*\*|\(\s*\)", "", sent).split())
            sent = re.sub(r"\s+([.,;:!?])", r"\1", sent)
            if sent.strip(" ,;:.!?-"):
                out.append(sent)
        out_lines.append(" ".join(out))
    return "\n".join(out_lines).strip(), kept, dropped


def guard_evidence(ids: list[str], results) -> tuple[list[str], list[str]]:
    """(kept, dropped): an evidence id survives when some returned row names it as a ref, an evidence id or an answer."""
    sup = _Support(results)
    ok = sup.answers | sup.refs | sup.evidence
    return [i for i in ids if i in ok], [i for i in ids if i not in ok]
