"""The output guard: nothing in a final answer may be an entity the tools did not return.

Rows are typed (see catalog.py): `answer` is the graded value (employee id, company name, PR or URL link, or text),
`refs` are label-qualified node ids a row names, `evidence` are record ids (Slack message, transcript segment, chat,
document). All three may also sit in nested dicts and lists of a row.

* `guard(text, ids, results, names=())` checks the proposed answer ids against the `answer` values exactly (a qualified
  id, a ref or an evidence id is not an answer), and checks every id-shaped token and every name in `names` that the
  text mentions against answers, refs, evidence and row names. A sentence that names an unsupported entity loses just
  that entity when it still names a supported one (`A, B and C` -> `A and C`); a sentence left with nothing supported
  is removed. Both namesakes of an ambiguous row are supported, because each is a row of its own.
* `guard_evidence(ids, results)` keeps the evidence ids that appear in some returned row.
"""

from __future__ import annotations

import re
import unicodedata

from inherit.ontology import NODE_KEYS

_LABELS = "|".join(sorted(NODE_KEYS, key=len, reverse=True))
ID_RE = re.compile(
    rf"\b(?:{_LABELS}):[^\s,;()]+"  # label-qualified node id
    r"|https?://[^\s<>\"')\]]+"  # link
    r"|\beid_[0-9a-f]{8}\b|\bEMP_\d+\b|\bCUST-\d+\b"  # employee and customer ids
    r"|\b\d{8}-\d+-[0-9a-f]{5}\b")  # Slack record id
_TRAIL = ".,;:!?"
_SPLIT = re.compile(r"(?<=[.!?])\s+")
_KEYS_REFS, _KEYS_EVIDENCE, _KEYS_ANSWER, _KEYS_NAME = {"refs"}, {"evidence"}, {"answer"}, {"name", "company"}


def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).split()).casefold()


def _rows(results) -> list[dict]:
    out: list[dict] = []
    for r in results:
        out.extend(r.rows if hasattr(r, "rows") else r)
    return out


def _walk(obj, bucket: dict, mode: str | None = None) -> None:
    """Collect strings under `answer`, `refs`, `evidence` and `name`/`company` keys at any depth. A list or dict under
    such a key is wholly that kind; nested rows are walked on their own."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, bucket, "answer" if k in _KEYS_ANSWER else "refs" if k in _KEYS_REFS else
                  "evidence" if k in _KEYS_EVIDENCE else "name" if k in _KEYS_NAME else None)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, bucket, mode)
    elif isinstance(obj, str) and mode:
        bucket[mode].add(obj)


class _Support:
    def __init__(self, results):
        b = {"answer": set(), "refs": set(), "evidence": set(), "name": set()}
        for row in _rows(results):
            _walk(row, b)
        self.answers = b["answer"]
        self.refs = b["refs"] | {r.split(":", 1)[1] for r in b["refs"] if ":" in r}
        self.evidence = b["evidence"]
        self.names = {_fold(n) for n in b["name"] | b["answer"]}
        self.tokens = self.answers | self.refs | self.evidence

    def has_token(self, tok: str) -> bool:
        return tok in self.tokens or tok in self.names or _fold(tok) in self.names

    def has_name(self, name: str) -> bool:
        return _fold(name) in self.names


def _remove(sentence: str, value: str) -> str:
    """Take one entity out of a sentence together with its list punctuation."""
    v = re.escape(value)
    for pat in (rf",\s*{v}(?=\s+(?:and|or)\b)", rf"\s+(?:and|or)\s+{v}\b", rf",\s*{v}", rf"{v}\s*(?:,|\s+and|\s+or)\s+"):
        new = re.sub(pat, "", sentence, count=1, flags=re.I)
        if new != sentence:
            return new
    return re.sub(v, "", sentence, count=1, flags=re.I)


def _entities(sentence: str, names) -> list[tuple[str, str]]:
    found = [(m.group(0).rstrip(_TRAIL), "id") for m in ID_RE.finditer(sentence)]
    low = _fold(sentence)
    for n in names:
        if re.search(rf"(?<!\w){re.escape(_fold(n))}(?!\w)", low):
            found.append((n, "name"))
    return found


def guard(text: str, ids: list[str], results, *, names=()) -> tuple[str, list[str], list[dict]]:
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
            ents = _entities(sent, names)
            bad = [(v, k) for v, k in ents if not (sup.has_token(v) if k == "id" else sup.has_name(v))]
            good = [e for e in ents if e not in bad]
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
    kept = [i for i in ids if i in sup.tokens]
    return kept, [i for i in ids if i not in sup.tokens]
