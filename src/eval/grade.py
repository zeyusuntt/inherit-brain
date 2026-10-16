"""The deterministic grader, with the judge for content and unanswerable questions.

Ported from HERB's `evaluate.py` (Salesforce AI Research, the benchmark's own scoring code), behaviour unchanged:
`normalize`, `f1_score_sets`, the `unanswerability_eval` and `answer_likert_score` prompts, and their score
mappings. tests/test_grade.py runs the originals next to these and compares them.

What differs from HERB, all disclosed in the report:
- ID types (person, company, pr, url): HERB extracts IDs from free text with GPT-4o. Here the structured
  `answer_ids` field is scored directly, so extraction is deterministic.
- An empty or failed output scores 0 on every type. HERB treats `answer=None` as a correct refusal.
- The judge is model M through the metering proxy (arm "judge"), not GPT-4o.
- Judge errors (transport, malformed JSON, out-of-range score) are persisted per slot and retried on a later
  grading pass. They never become an arm failure.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from eval import run as R
from eval.questions import Question, snapshot_hash
from inherit.config import MODEL
from inherit import llm
from inherit.contracts import QuestionKey, RunRecord

PROMPT_VERSION = "herb-evaluate-v1"
SCORER_VERSION = "eval.grade/2"
JUDGE_MAX_TOKENS = 400


# ---------------------------------------------------------------- ported from HERB evaluate.py
def normalize(answer: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def f1_score_sets(y_true, y_pred):
    y_true_set = set([normalize(e) for e in y_true])
    y_pred_set = set([normalize(e) for e in y_pred])

    tp = len(y_true_set & y_pred_set)
    fp = len(y_pred_set - y_true_set)
    fn = len(y_true_set - y_pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return f1


def unanswerability_prompt(question, answer) -> str:
    """The prompt of HERB's `unanswerability_eval`."""
    prompt = """Given a question and its corresponding answer, determine whether the answer provides sufficient information to fully or even partially respond to the question.

- Reply "Yes" if the answer directly addresses the question, either fully or partially, using information relevant to the question.
- Reply "No" if the answer does not address the question due to missing, vague, or insufficient information — even if it explicitly states that the information is not available.

Question: Which products were affected by the bug in March?
Answer:  The context does not provide the specific information required to determine the products.
Output: No

Question: Find all authors and reviewers of the PRD?
Answer: John authored the PRD
Output: Yes

Question: Find all authors and reviewers of the PRD?
Answer: John authored the PRD. Yuan and Jessica reviewed the PRD.
Output: Yes
"""
    prompt += (
        f'\n\nQuestion: {question}\n\nAnswer: {answer}\n\nRespond using the below json format:\n{{"done": "yes/no", "reason": "justification for the decision"}}'
    )
    return prompt


def likert_prompt(question, reference, candidate) -> str:
    """The prompt of HERB's `answer_likert_score`."""
    return (
        "You are an expert evaluator. Given a question, a reference answer, "
        "and a candidate answer, your task is to evaluate how well the candidate "
        "answer aligns with the reference.\n\nFor your evaluation:"
        "\nFocus on accuracy, completeness, and relevance."
        "\nIf the candidate includes extra information, check if it's correct and appropriate."
        "\nIf it omits key points from the reference, mention that."
        "\nQuestion: {question}\nReference Answer: {reference}\nCandidate Answer: {candidate}"
        "\nRespond in json format: {{'score': 'Your overall rating between 0-100', "
        "'reason': 'a brief justification'}}'.").format(question=question, reference=reference, candidate=candidate)


# ---------------------------------------------------------------- judge
class JudgeError(RuntimeError):
    """The judge call failed or its answer was unusable. The slot stays ungraded and is retried later."""


class IncompleteGrading(RuntimeError):
    def __init__(self, errors: dict[str, str], missing: list[str] | None = None):
        self.errors, self.missing = errors, missing or []
        super().__init__(f"{len(errors)} judge errors and {len(self.missing)} ungraded slots; run the grading step again")


def likert_score_from(parsed: Any) -> int:
    """HERB maps the judge's `score` with int(); a result outside 0-100 is rejected."""
    try:
        if not isinstance(parsed, dict) or isinstance(parsed["score"], bool):
            raise ValueError
        score = int(parsed["score"])
    except (KeyError, ValueError, TypeError):
        raise JudgeError(f"no usable score in {str(parsed)[:120]!r}") from None
    if not 0 <= score <= 100:
        raise JudgeError(f"score {score} is outside 0-100")
    return score


def refusal_from(parsed: Any) -> str:
    """'no' means the answer does not address the question, which is the correct response to an unanswerable one."""
    done = parsed.get("done") if isinstance(parsed, dict) else None
    # HERB lowercases and does not strip. A value other than exactly yes/no after lowercasing (for example
    # " no ") is a judge error here, retried on the next pass, so no score depends on whitespace.
    if not (isinstance(done, str) and done.lower() in ("yes", "no")):
        raise JudgeError(f"no usable 'done' in {str(parsed)[:120]!r}")
    return done.lower()


class Judge:
    """Model M behind the metering proxy, cached by exact request. The cache key includes the prompt version."""

    def __init__(self, cache_path: Path | None, chat: Callable = llm.chat, arm: str = "judge", model: str | None = None):
        self.chat, self.arm, self.model = chat, arm, model
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    e = json.loads(line)
                    self.cache[e["k"]] = e["v"]
                except (ValueError, KeyError, TypeError):
                    continue

    def identity(self) -> dict[str, str]:
        return {"model": self.model or MODEL, "prompt_version": PROMPT_VERSION, "scorer_version": SCORER_VERSION}

    def _key(self, prompt: str) -> str:
        blob = json.dumps([PROMPT_VERSION, self.model, JUDGE_MAX_TOKENS, prompt], ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _ask(self, prompt: str, parse: Callable[[Any], Any]) -> Any:
        key = self._key(prompt)
        content = self.cache.get(key)
        if content is None:
            try:
                res = self.chat([{"role": "user", "content": prompt}], arm=self.arm, max_tokens=JUDGE_MAX_TOKENS,
                                response_format={"type": "json_object"}, model=self.model)
                content = res.message.content
            except llm.BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                raise JudgeError(f"{type(e).__name__}: {e}") from e
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            raise JudgeError(f"judge answer is not JSON: {str(content)[:120]!r}") from None
        value = parse(parsed)  # validate before caching
        if key not in self.cache:
            self.cache[key] = content
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": key, "v": content}) + "\n")
        return value

    def likert(self, question, reference, candidate) -> int:
        return self._ask(likert_prompt(question, reference, candidate), likert_score_from)

    def refusal(self, question, answer) -> str:
        return self._ask(unanswerability_prompt(question, answer), refusal_from)


# ---------------------------------------------------------------- grading
@dataclass(frozen=True)
class Graded:
    key: QuestionKey
    arm: str
    type: str
    score: float  # 0-100
    failed: bool  # run failure or empty output; the slot still counts in the denominator, at 0
    outcome: str = "path"
    seen: bool | None = None
    decision: Any = None  # judge decision: "yes"/"no" (unanswerable) or the 0-100 integer (content)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "key": self.key.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Graded":
        return cls(QuestionKey.from_dict(d["key"]), d["arm"], d["type"], d["score"], d["failed"], d["outcome"],
                   d["seen"], d["decision"])


def grade(record: RunRecord, q: Question, judge: Judge | None) -> Graded:
    f = record.final
    base = dict(key=q.key, arm=record.arm, type=q.type, outcome=f.outcome, seen=q.seen)
    if f.outcome == "run_failed" or record.failure:
        return Graded(score=0.0, failed=True, **base)
    if not f.answer_text.strip() and not f.answer_ids:  # empty output
        return Graded(score=0.0, failed=True, **base)
    if q.type in ("person", "company", "pr", "url"):
        return Graded(score=100.0 * f1_score_sets(q.gold, f.answer_ids), failed=False, **base)
    if q.type == "content":
        if not f.answer_text.strip():
            return Graded(score=0.0, failed=True, **base)
        s = judge.likert(q.text, q.gold, f.answer_text)
        return Graded(score=float(s), failed=False, decision=s, **base)
    if q.type == "unanswerable":
        if not f.answer_text.strip():
            return Graded(score=0.0, failed=True, **base)
        d = judge.refusal(q.text, f.answer_text)
        return Graded(score=100.0 if d == "no" else 0.0, failed=False, decision=d, **base)
    raise ValueError(f"unknown question type {q.type!r}")


def _grade_lines(run_dir: Path) -> dict[str, dict]:
    p = Path(run_dir) / "grades.jsonl"
    out: dict[str, dict] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["key"]] = row  # the latest line of a slot wins
    return out


def read_grading_manifest(run_dir: Path) -> dict[str, str]:
    return json.loads((Path(run_dir) / "grading.json").read_text(encoding="utf-8"))


def _sha(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()


def grade_run(run_dir: Path, questions: list[Question], judge: Judge) -> list[Graded]:
    """Grade every slot of a finished run, resumably. An unfinished run is refused: a missing terminal record is
    not a grade. Each persisted grade is bound to the sha256 of its source record and is redone if that changes.
    The judge identity is frozen in grading.json on the first pass; a different judge or prompt version is
    refused. Judge errors are written to grades.jsonl and retried on the next call; arms are never re-run.
    Raises IncompleteGrading unless every slot ends up graded."""
    run_dir = Path(run_dir)
    manifest = R.read_manifest(run_dir)
    lines = R.read_terminal_lines(run_dir)
    unfinished = [s for s in manifest["slots"] if s not in lines]
    if unfinished:
        raise R.RunStateError(f"{len(unfinished)} slots have no terminal record (first: {unfinished[0]}); "
                              "finish the run before grading")
    by_slot = {str(q.key): q for q in questions}
    absent = [s for s in manifest["slots"] if s not in by_slot]
    if absent:
        raise ValueError(f"no question supplied for slots {absent[:3]}")
    if snapshot_hash([by_slot[s] for s in manifest["slots"]]) != manifest["question_sha"]:
        raise R.RunStateError("the questions differ from the snapshot this run was started with")
    ident = judge.identity()
    gpath = run_dir / "grading.json"
    if gpath.exists():
        if read_grading_manifest(run_dir) != ident:
            raise R.RunStateError(f"run was graded with {read_grading_manifest(run_dir)}, not {ident}; refusing to mix judges")
    else:
        tmp = gpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(ident, indent=1), encoding="utf-8")
        tmp.replace(gpath)
    records = R.read_records(run_dir)
    rows = _grade_lines(run_dir)
    errors: dict[str, str] = {}
    with open(run_dir / "grades.jsonl", "a", encoding="utf-8") as out:
        for slot in manifest["slots"]:
            sha = _sha(lines[slot])
            if rows.get(slot, {}).get("status") == "ok" and rows[slot].get("record_sha256") == sha:
                continue
            try:
                g = grade(records[slot], by_slot[slot], judge)
                row = {"key": slot, "status": "ok", **{k: v for k, v in g.to_dict().items() if k != "key"},
                       "record_sha256": sha}
            except JudgeError as e:
                errors[slot] = str(e)
                row = {"key": slot, "status": "error", "error": str(e), "record_sha256": sha}
            except llm.BudgetExceeded as e:
                out.write(json.dumps({"key": slot, "status": "error", "error": f"BudgetExceeded: {e}",
                                      "record_sha256": sha}) + "\n")
                raise
            out.write(json.dumps(row) + "\n")
            out.flush()
    if errors:
        raise IncompleteGrading(errors)
    return load_graded(run_dir)


def load_graded(run_dir: Path) -> list[Graded]:
    """The graded slots in manifest order. Raises IncompleteGrading unless every slot is graded from its current
    terminal record, and RunStateError if a grade belongs to another arm or the run is unfinished."""
    run_dir = Path(run_dir)
    manifest = R.read_manifest(run_dir)
    lines = R.read_terminal_lines(run_dir)
    rows = _grade_lines(run_dir)
    slots = manifest["slots"]
    if any(s not in lines for s in slots):
        raise R.RunStateError("run is unfinished; its grades are not final")
    errors = {s: rows[s].get("error", "") for s in slots if rows.get(s, {}).get("status") == "error"}
    stale = [s for s in slots if s in rows and rows[s].get("status") == "ok" and rows[s].get("record_sha256") != _sha(lines[s])]
    missing = [s for s in slots if s not in rows or s in stale]
    if errors or missing:
        raise IncompleteGrading(errors, missing)
    out = [Graded.from_dict({**rows[s], "key": QuestionKey.parse(s).to_dict()}) for s in slots]
    if any(g.arm != manifest["arm"] for g in out):
        raise R.RunStateError(f"grades of another arm found in {run_dir.name}")
    return out


# ---------------------------------------------------------------- grader audit sample (T14)
def audit_sample(keyed_types: list[tuple[QuestionKey, str]], n: int = 50, seed: int = 20261015) -> list[QuestionKey]:
    """A seeded sample of n question keys stratified by type, proportional to type size (largest remainder),
    at least one per type when n allows. Returned in the input order."""
    if n >= len(keyed_types):
        return [k for k, _ in keyed_types]
    rng = random.Random(seed)
    by_type: dict[str, list[QuestionKey]] = {}
    for k, t in keyed_types:
        by_type.setdefault(t, []).append(k)
    total = len(keyed_types)
    quota = {t: n * len(v) / total for t, v in by_type.items()}
    take = {t: max(1, int(quota[t])) if n >= len(by_type) else int(quota[t]) for t in by_type}
    order = sorted(by_type, key=lambda t: (quota[t] - int(quota[t]), t), reverse=True)
    i = 0
    while sum(take.values()) < n:
        t = order[i % len(order)]
        if take[t] < len(by_type[t]):
            take[t] += 1
        i += 1
    while sum(take.values()) > n:
        t = max(take, key=lambda x: (take[x], x))
        take[t] -= 1
    chosen = set()
    for t in sorted(by_type):
        chosen |= set(rng.sample(by_type[t], take[t]))
    return [k for k, _ in keyed_types if k in chosen]
