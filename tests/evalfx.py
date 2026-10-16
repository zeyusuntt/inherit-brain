"""Fixtures for the eval harness tests: fake questions, fake arms, a fixed run environment."""

from __future__ import annotations

from eval.arms import Arm, ArmResult
from eval.questions import Question
from eval.receipt import RunEnv
from inherit.contracts import CallUsage, FinalAnswer, QuestionKey, TraceOp, Usage

TYPES = ["person", "company", "pr", "url", "content"]
FACTS = "f" * 64


def mk_questions(products=("P1", "P2"), n_ans=5, n_unans=2) -> list[Question]:
    qs = []
    for p in products:
        for i in range(n_ans):
            t = TYPES[i % len(TYPES)]
            gold = "reference text" if t == "content" else [f"{p}_g{i}a", f"{p}_g{i}b"]
            qs.append(Question(QuestionKey(p, "ans", i), f"What about {p} item {i}?", t, gold, [f"c{i}"]))
        for i in range(n_unans):
            qs.append(Question(QuestionKey(p, "unans", i), f"Unanswerable {p} {i}?", "unanswerable", None, []))
    return qs


def final(text="an answer", ids=(), outcome="path", prompt=100, completion=10, n_trace=2, evidence=()) -> FinalAnswer:
    trace = [TraceOp("graph", "q", "MATCH", {}, 1, 1.0) for _ in range(n_trace)]
    return FinalAnswer(outcome, text, list(ids), list(evidence), trace, Usage(prompt, completion, 0.0001, 1))


def result(**kw) -> ArmResult:
    f = final(**kw)
    call = CallUsage("fake", "m", f.usage.prompt_tokens, f.usage.completion_tokens, 0.0001, 0.5)
    return ArmResult(f, [call], [{"tool": "graph", "rows": [[1], [2]]}])


def make_arm(fn=None, name="fake", **kw) -> Arm:
    return Arm(name, fn or (lambda ei: result(ids=["x"])), **kw)


def env(**kw) -> RunEnv:
    base = dict(code_commit="c0ffee", facts_hash=FACTS, bundle_sha="b" * 64, model="m1")
    base.update(kw)
    return RunEnv(**base)
