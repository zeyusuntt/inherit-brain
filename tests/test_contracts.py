import json

import pytest

from inherit.contracts import (
    GRADED_FIELD,
    OUTCOMES,
    CallUsage,
    ExecInput,
    FinalAnswer,
    QuestionKey,
    RunRecord,
    TraceOp,
    Usage,
)


def roundtrip(obj):
    cls = type(obj)
    again = cls.from_dict(json.loads(json.dumps(obj.to_dict())))
    assert again == obj
    return again


def make_final(outcome="path"):
    op = TraceOp("neighbors", "MATCH", "q", {"id": "x", "n": 2}, 3, 1.5)
    return FinalAnswer(outcome, "text", ["a", "b"], ["e1"], [op], Usage(10, 5, 0.001, 2))


def test_simple_roundtrips():
    roundtrip(ExecInput("mini", "who owns it?"))
    roundtrip(TraceOp("t", "c", "q", {"k": [1, 2]}, 0, 0.5))
    roundtrip(CallUsage("A4", "m", 1, 2, 0.1, 0.2))
    roundtrip(Usage(1, 2, 0.3, 4))
    roundtrip(QuestionKey("mini", "ans", 3))


def test_final_answer_and_run_record_roundtrip():
    f = roundtrip(make_final())
    assert f.trace[0].params == {"id": "x", "n": 2}
    rec = RunRecord(
        QuestionKey("mini", "unans", 0), "A4", make_final("not_in_graph"),
        [CallUsage("A4", "m", 1, 2, 0.1, 0.2)], [TraceOp("t", "c", "q", {}, 1, 1.0)], 2.5, None,
    )
    again = roundtrip(rec)
    assert isinstance(again.key, QuestionKey) and isinstance(again.final, FinalAnswer)
    failed = RunRecord(QuestionKey("mini", "ans", 1), "A1", make_final("run_failed"), [], [], 0.0, "timeout")
    assert roundtrip(failed).failure == "timeout"


def test_outcomes_closed_set():
    assert OUTCOMES == {"path", "memory", "not_in_graph", "unknown_name", "ambiguous_name", "partial", "run_failed"}
    for o in OUTCOMES:
        make_final(o)
    with pytest.raises(ValueError):
        make_final("maybe")
    with pytest.raises(ValueError):
        FinalAnswer.from_dict({**make_final().to_dict(), "outcome": "bogus"})


def test_frozen():
    k = QuestionKey("mini", "ans", 0)
    with pytest.raises(Exception):
        k.index = 5  # type: ignore[misc]


def test_question_key_string_form():
    k = QuestionKey("mini", "ans", 12)
    assert str(k) == "mini/ans/12"
    assert QuestionKey.parse("mini/ans/12") == k
    assert QuestionKey.parse(str(QuestionKey("A", "unans", 0))) == QuestionKey("A", "unans", 0)


@pytest.mark.parametrize("bad", ["mini/ans", "mini/foo/1", "mini/ans/x", "mini/ans/-1", "a/b/ans/1", ""])
def test_question_key_rejects_bad(bad):
    with pytest.raises(ValueError):
        QuestionKey.parse(bad)


def test_question_key_validates_fields():
    with pytest.raises(ValueError):
        QuestionKey("mini", "other", 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        QuestionKey("mini", "ans", -1)
    with pytest.raises(ValueError):
        QuestionKey("a/b", "ans", 0)


def test_graded_field():
    assert GRADED_FIELD == {
        "person": "answer_ids", "company": "answer_ids", "pr": "answer_ids", "url": "answer_ids",
        "content": "answer_text", "unanswerable": "answer_text",
    }
    assert "evidence_ids" not in GRADED_FIELD.values()
