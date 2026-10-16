"""The output guard: nothing in a final answer may be an entity the tools did not return.

Rows are typed (see catalog.py): `answer` is the graded value (employee id, company name, PR or URL link, or text),
`refs` are label-qualified node ids a row names, `evidence` are record ids (edge `src`). Rows nest (claims, bugs,
authors, reverts); the category of a key is inherited by everything below it.

* `guard(text, ids, results, *, context="", names=())` checks the proposed answer ids against the `answer` values
  exactly (a qualified id, a ref or an evidence id is not an answer). In the text it checks, always:
  every complete returned id (masked first, so a multiword id such as `Feature:p:voice assistant` or an id in Markdown
  backticks is recognised whole), every other id-shaped token, and every name-like mention (two or more capitalised
  words, or a CamelCase word such as a company) against the strings the rows contain; names in `context` (the
  question) are allowed too. `names` adds explicit names to check. A sentence with an unsupported entity loses just
  that entity when a supported one remains (`A, B and C` -> `A and C`); a sentence left with nothing supported is
  removed. Both namesakes of an ambiguous row are supported, because each is a row of its own.
* `guard_evidence(ids, results)` keeps the evidence ids that appear in some returned row.
"""

from __future__ import annotations

import re
import unicodedata

from inherit.ontology import NODE_KEYS

_LABELS = "|".join(sorted(NODE_KEYS, key=len, reverse=True))
_STOP_CHARS = r"\s,;()`'\"*\[\]<>"
ID_RE = re.compile(
    rf"\b(?:{_LABELS}):[^{_STOP_CHARS}]+"  # label-qualified node id (complete ones are masked before this runs)
    rf"|https?://[^{_STOP_CHARS}]+"  # link
    r"|\beid_[0-9a-f]{8}\b|\bEMP_\d+\b|\bCUST-\d+\b"  # employee and customer ids
    r"|\b\d{8}-\d+-[0-9a-f]{5}\b")  # Slack record id
NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b")
_TRAIL = ".,;:!?"
_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CATEGORY = {"answer": "answer", "refs": "refs", "evidence": "evidence"}


def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).split()).casefold()


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
    def __init__(self, results, context: str = ""):
        b = {"answer": set(), "refs": set(), "evidence": set(), "other": set()}
        for row in _rows(results):
            _walk(row, b)
        self.answers, self.evidence = b["answer"], b["evidence"]
        self.refs = b["refs"] | {r.split(":", 1)[1] for r in b["refs"] if ":" in r}
        self.tokens = b["answer"] | self.refs | b["evidence"] | b["other"]  # every returned string
        self.blob = _fold(" \n ".join(sorted(self.tokens))) + " \n " + _fold(context)

    def has_token(self, tok: str) -> bool:
        return tok in self.tokens or _fold(tok) in self.blob

    def has_name(self, name: str) -> bool:
        return _fold(name) in self.blob


def _mask(sentence: str, sup: _Support) -> tuple[str, list[str]]:
    """Blank out the complete returned ids found in the sentence (longest first); return the masked text and them."""
    found = []
    for tok in sorted((t for t in sup.tokens if len(t) > 3 and (":" in t or "/" in t or "_" in t or "-" in t)),
                      key=len, reverse=True):
        pat = re.compile(rf"(?<![\w:/]){re.escape(tok)}(?![\w])")
        if pat.search(sentence):
            found.append(tok)
            sentence = pat.sub(lambda m: "\x00" * len(m.group(0)), sentence)
    return sentence, found


def _remove(sentence: str, value: str) -> str:
    """Take one entity out of a sentence together with its list punctuation."""
    v = re.escape(value)
    for pat in (rf",\s*{v}(?=\s+(?:and|or)\b)", rf"\s+(?:and|or)\s+{v}\b", rf",\s*{v}", rf"{v}\s*(?:,|\s+and|\s+or)\s+"):
        new = re.sub(pat, "", sentence, count=1, flags=re.I)
        if new != sentence:
            return new
    return re.sub(v, "", sentence, count=1, flags=re.I)


def _entities(masked: str, names) -> list[tuple[str, str]]:
    found = [(m.group(0).rstrip(_TRAIL), "id") for m in ID_RE.finditer(masked)]
    found += [(m.group(0), "name") for m in NAME_RE.finditer(masked)]
    low = _fold(masked)
    for n in names:
        if re.search(rf"(?<!\w){re.escape(_fold(n))}(?!\w)", low) and (n, "name") not in found:
            found.append((n, "name"))
    return found


def guard(text: str, ids: list[str], results, *, context: str = "", names=()) -> tuple[str, list[str], list[dict]]:
    """(text, kept answer ids, dropped). `dropped` lists {"value", "where": "ids"|"text", "reason"}."""
    sup = _Support(results, context)
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
            masked, whole = _mask(sent, sup)
            ents = _entities(masked, names)
            bad = [(v, k) for v, k in ents if not (sup.has_token(v) if k == "id" else sup.has_name(v))]
            good = [e for e in ents if e not in bad] + [(w, "id") for w in whole]
            if not bad:
                out.append(sent)
                continue
            for v, _ in bad:
                dropped.append({"value": v, "where": "text", "reason": "not in any returned row"})
            if not good:
                continue  # nothing supported is left to say
            rebuilt = sent
            for v, _ in bad:
                rebuilt = _remove(rebuilt, v)
            out.append(" ".join(rebuilt.split()))
        out_lines.append(" ".join(out))
    return "\n".join(out_lines).strip(), kept, dropped


def guard_evidence(ids: list[str], results) -> tuple[list[str], list[str]]:
    """(kept, dropped): an evidence id survives when some returned row names it as a ref, an evidence id or an answer."""
    sup = _Support(results)
    ok = sup.answers | sup.refs | sup.evidence
    return [i for i in ids if i in ok], [i for i in ids if i not in ok]
