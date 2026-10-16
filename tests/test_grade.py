import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import grade as G
from eval import run as R
from eval.arms import ARMS
from eval.questions import Question
from inherit import llm
from inherit.contracts import QuestionKey, RunRecord

from evalfx import env, final, make_arm, mk_questions, result

ROOT = Path(__file__).resolve().parents[1]
EVALUATE = ROOT / "data" / "raw" / "code" / "evaluate.py"


class FakeChat:
    """Stands in for inherit.llm.chat. `script` maps a substring of the prompt to a content string or an Exception."""

    def __init__(self, script=None, default='{"score": 80, "reason": "r"}'):
        if isinstance(script, str):  # FakeChat('<content>') answers every prompt with it
            script, default = None, script
        self.script, self.default, self.calls = script or {}, default, []

    def __call__(self, messages, **kw):
        self.calls.append((messages, kw))
        prompt = messages[0]["content"]
        for sub, out in self.script.items():
            if sub in prompt:
                if isinstance(out, Exception):
                    raise out
                return SimpleNamespace(message=SimpleNamespace(content=out))
        return SimpleNamespace(message=SimpleNamespace(content=self.default))


def rec(q, **kw):
    return RunRecord(q.key, "fake", final(**kw))


def failed_rec(q):
    return R.failed_record(q.key, "fake", "boom")


def Q(type_, gold, i=0, kind="ans", text="What?"):
    return Question(QuestionKey("P", kind, i), text, type_, gold, [])


# ---------------------------------------------------------------- deterministic ID scoring
def test_two_golds_one_hit_scores_66_67():
    g = G.grade(rec(Q("person", ["eid_a", "eid_b"]), ids=["eid_a"]), Q("person", ["eid_a", "eid_b"]), None)
    assert g.score == pytest.approx(66.67, abs=0.01) and not g.failed


def test_ids_are_normalised_like_herb_and_scored_as_sets():
    q = Q("pr", ["https://github.com/x/y/pull/1", "https://github.com/x/y/pull/2"])
    r = rec(q, ids=["HTTPS://github.com/x/y/pull/1", "https://github.com/x/y/pull/1", "https://github.com/x/y/pull/9"])
    assert G.grade(r, q, None).score == pytest.approx(50.0)
    q2 = Q("company", ["Acme Corp"])
    assert G.grade(rec(q2, ids=["the acme corp"]), q2, None).score == 100.0


def test_empty_ids_on_an_answerable_question_score_zero_but_are_not_a_failure_when_text_exists():
    q = Q("person", ["eid_a"])
    g = G.grade(rec(q, text="I found nothing", ids=[], outcome="not_in_graph"), q, None)
    assert g.score == 0 and not g.failed and g.outcome == "not_in_graph"


def test_id_types_never_call_the_judge():
    chat = FakeChat()
    j = G.Judge(None, chat=chat)
    for t in ("person", "company", "pr", "url"):
        q = Q(t, ["a"])
        G.grade(rec(q, ids=["a"]), q, j)
    assert chat.calls == []


# ---------------------------------------------------------------- failures and blanks
def test_run_failed_scores_zero_on_both_kinds_and_counts_as_failed_without_a_judge_call():
    chat = FakeChat()
    j = G.Judge(None, chat=chat)
    for q in (Q("person", ["a"]), Q("content", "ref"), Q("unanswerable", None, kind="unans")):
        g = G.grade(failed_rec(q), q, j)
        assert g.score == 0 and g.failed and g.outcome == "run_failed"
    assert chat.calls == []


def test_empty_output_on_an_unanswerable_question_scores_zero_not_100():
    chat = FakeChat('{"done": "no", "reason": "r"}')  # a judge would call empty text a refusal
    j = G.Judge(None, chat=chat)
    q = Q("unanswerable", None, kind="unans")
    g = G.grade(rec(q, text="  ", ids=[]), q, j)
    assert g.score == 0 and g.failed and chat.calls == []


def test_blank_content_candidate_scores_zero_without_a_judge_call():
    chat = FakeChat()
    q = Q("content", "ref")
    g = G.grade(rec(q, text=" \n"), q, G.Judge(None, chat=chat))
    assert g.score == 0 and g.failed and chat.calls == []


# ---------------------------------------------------------------- judged types
def test_unanswerable_refusal_is_100_and_an_answer_is_0():
    q = Q("unanswerable", None, kind="unans", text="Who left?")
    j = G.Judge(None, chat=FakeChat('{"done": "no", "reason": "r"}'))
    g = G.grade(rec(q, text="The context does not say."), q, j)
    assert g.score == 100 and g.decision == "no"
    j = G.Judge(None, chat=FakeChat('{"done": "Yes", "reason": "r"}'))
    g = G.grade(rec(q, text="It was John."), q, j)
    assert g.score == 0 and g.decision == "yes"


def test_content_score_is_the_judges_integer_0_to_100():
    q = Q("content", "reference", text="Summarise?")
    for raw, want in (('{"score": 85, "reason": ""}', 85), ('{"score": "70", "reason": ""}', 70),
                      ('{"score": 0, "reason": ""}', 0), ('{"score": 100, "reason": ""}', 100)):
        assert G.grade(rec(q, text="cand"), q, G.Judge(None, chat=FakeChat(raw))).score == want


@pytest.mark.parametrize("raw", ['{"score": 101}', '{"score": -1}', '{"score": "high"}', '{"score": true}',
                                 'not json', '[1]', '{"nope": 1}'])
def test_malformed_or_out_of_range_content_judgement_is_a_judge_error(raw):
    q = Q("content", "reference")
    with pytest.raises(G.JudgeError):
        G.grade(rec(q, text="cand"), q, G.Judge(None, chat=FakeChat(raw)))


@pytest.mark.parametrize("raw", ['{"done": "maybe"}', '{"done": 1}', '{}', 'xx'])
def test_malformed_unanswerable_judgement_is_a_judge_error(raw):
    q = Q("unanswerable", None, kind="unans")
    with pytest.raises(G.JudgeError):
        G.grade(rec(q, text="t"), q, G.Judge(None, chat=FakeChat(raw)))


def test_transport_errors_become_judge_errors():
    q = Q("content", "reference")
    with pytest.raises(G.JudgeError, match="TimeoutError"):
        G.grade(rec(q, text="c"), q, G.Judge(None, chat=FakeChat({"Candidate": TimeoutError("slow")})))


def test_judge_goes_through_chat_as_arm_judge_with_json_format():
    chat = FakeChat()
    G.Judge(None, chat=chat).likert("q", "r", "c")
    msgs, kw = chat.calls[0]
    assert kw["arm"] == "judge" and kw["response_format"] == {"type": "json_object"} and msgs[0]["role"] == "user"
    assert G.Judge(None).chat is llm.chat
    chat2 = FakeChat(default='{"done": "no"}')
    G.Judge(None, chat=chat2, arm="judge-smoke").refusal("q", "a")
    assert chat2.calls[0][1]["arm"] == "judge-smoke"


def test_judge_cache_hits_on_exact_request_across_instances_and_keys_on_prompt_version(tmp_path, monkeypatch):
    cache = tmp_path / "cache.jsonl"
    c1 = FakeChat()
    assert G.Judge(cache, chat=c1).likert("q", "r", "c") == 80
    assert G.Judge(cache, chat=c1).likert("q", "r", "c") == 80 and len(c1.calls) == 1
    G.Judge(cache, chat=c1).likert("q", "r", "c2")
    assert len(c1.calls) == 2
    monkeypatch.setattr(G, "PROMPT_VERSION", "herb-evaluate-v2")
    G.Judge(cache, chat=c1).likert("q", "r", "c")
    assert len(c1.calls) == 3


def test_invalid_judgements_are_not_cached(tmp_path):
    cache = tmp_path / "cache.jsonl"
    bad = FakeChat(default='{"score": 500}')
    for _ in range(2):
        with pytest.raises(G.JudgeError):
            G.Judge(cache, chat=bad).likert("q", "r", "c")
    assert len(bad.calls) == 2


# ---------------------------------------------------------------- the grading step over a run
@pytest.fixture
def finished_run(tmp_path, monkeypatch):
    n = {"calls": 0}

    def arm(ei):
        n["calls"] += 1
        return result(text=f"cand for {ei.text}", ids=["x"])

    monkeypatch.setitem(ARMS, "fake", make_arm(arm))
    qs = mk_questions()
    d = R.run_arm("fake", "dev", None, root=tmp_path, questions=qs, env=env(), cap_setter=lambda a, u: None)
    return d, qs, n


def test_judge_error_is_persisted_then_resumed_without_rerunning_arms(finished_run):
    d, qs, n = finished_run
    arm_calls = n["calls"]
    bad_text = "Unanswerable P2 0?"
    chat = FakeChat({bad_text: TimeoutError("slow"), "Candidate Answer": '{"score": 60}'},
                    default='{"done": "no", "reason": "r"}')
    with pytest.raises(G.IncompleteGrading) as e:
        G.grade_run(d, qs, G.Judge(None, chat=chat))
    assert list(e.value.errors) == ["P2/unans/0"]
    rows = [json.loads(l) for l in (d / "grades.jsonl").read_text().splitlines()]
    assert sum(r["status"] == "error" for r in rows) == 1 and sum(r["status"] == "ok" for r in rows) == len(qs) - 1
    with pytest.raises(G.IncompleteGrading):
        G.load_graded(d)
    calls_after_first = len(chat.calls)
    good = FakeChat(default='{"done": "no", "reason": "r"}')
    graded = G.grade_run(d, qs, G.Judge(None, chat=good))
    assert len(good.calls) == 1  # only the slot that errored is judged again
    assert [g.key for g in graded] == [q.key for q in qs]
    assert n["calls"] == arm_calls and calls_after_first > 1
    assert [g.score for g in G.load_graded(d)] == [g.score for g in graded]
    again = FakeChat()
    G.grade_run(d, qs, G.Judge(None, chat=again))
    assert again.calls == []


def test_budget_exhaustion_stops_grading_and_keeps_what_was_done(finished_run):
    d, qs, _ = finished_run

    def chat(messages, **kw):
        raise llm.BudgetExceeded("cap")

    with pytest.raises(llm.BudgetExceeded):
        G.grade_run(d, qs, G.Judge(None, chat=chat))
    rows = [json.loads(l) for l in (d / "grades.jsonl").read_text().splitlines()]
    assert any(r["status"] == "error" and "BudgetExceeded" in r["error"] for r in rows)
    with pytest.raises(G.IncompleteGrading):
        G.load_graded(d)


def _interrupted(tmp_path, monkeypatch):
    qs = mk_questions()

    class Stop(KeyboardInterrupt):
        pass

    def arm(ei):
        if ei.text == qs[3].text:
            raise Stop()
        return result(ids=["x"])

    monkeypatch.setitem(ARMS, "fake", make_arm(arm))
    with pytest.raises(Stop):
        R.run_arm("fake", "dev", None, root=tmp_path, questions=qs, env=env(), workers=1, cap_setter=lambda a, u: None)
    return tmp_path / "runs/eval/fake-dev", qs


def test_an_unfinished_run_is_not_graded_and_nothing_is_persisted(tmp_path, monkeypatch):
    d, qs = _interrupted(tmp_path, monkeypatch)
    with pytest.raises(R.RunStateError, match="finish the run"):
        G.grade_run(d, qs, G.Judge(None, chat=FakeChat()))
    assert not (d / "grades.jsonl").exists() and not (d / "grading.json").exists()
    with pytest.raises(R.RunStateError):
        G.load_graded(d)


def test_grades_are_bound_to_their_source_record_and_redone_when_it_changes(finished_run):
    d, qs, _ = finished_run
    chat = FakeChat(default='{"done": "no", "score": 70}')
    first = G.grade_run(d, qs, G.Judge(None, chat=chat))
    # the record of slot 0 changes (as after a re-run); its old grade must not survive
    lines = (d / "records.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    rec["final"]["answer_ids"] = ["P1_g0a", "P1_g0b"] if rec["key"]["index"] == 0 else rec["final"]["answer_ids"]
    lines[0] = json.dumps(rec)
    (d / "records.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(G.IncompleteGrading):
        G.load_graded(d)
    again = G.grade_run(d, qs, G.Judge(None, chat=chat))
    assert again[0].score == 100.0 and first[0].score != 100.0
    assert [g.score for g in again[1:]] == [g.score for g in first[1:]]


def test_a_different_judge_or_prompt_version_cannot_resume_a_grading(finished_run, monkeypatch):
    d, qs, _ = finished_run
    bad = FakeChat({"Unanswerable P2 0?": TimeoutError("slow")}, default='{"done": "no", "score": 70}')
    with pytest.raises(G.IncompleteGrading):
        G.grade_run(d, qs, G.Judge(None, chat=bad))
    assert json.loads((d / "grading.json").read_text())["prompt_version"] == G.PROMPT_VERSION
    with pytest.raises(R.RunStateError, match="mix"):
        G.grade_run(d, qs, G.Judge(None, chat=FakeChat(default='{"done": "no"}'), model="other-model"))
    monkeypatch.setattr(G, "PROMPT_VERSION", "herb-evaluate-v2")
    with pytest.raises(R.RunStateError, match="mix"):
        G.grade_run(d, qs, G.Judge(None, chat=FakeChat(default='{"done": "no"}')))
    monkeypatch.undo()
    G.grade_run(d, qs, G.Judge(None, chat=FakeChat(default='{"done": "no", "score": 70}')))


def test_grading_refuses_questions_that_differ_from_the_run_snapshot(finished_run):
    from dataclasses import replace
    d, qs, _ = finished_run
    changed = [replace(qs[0], gold=["different"])] + qs[1:]
    with pytest.raises(R.RunStateError, match="snapshot"):
        G.grade_run(d, changed, G.Judge(None, chat=FakeChat()))


def test_load_graded_rejects_grades_of_another_arm(finished_run):
    d, qs, _ = finished_run
    G.grade_run(d, qs, G.Judge(None, chat=FakeChat(default='{"done": "no", "score": 70}')))
    rows = [json.loads(l) for l in (d / "grades.jsonl").read_text().splitlines()]
    rows[0]["arm"] = "other"
    (d / "grades.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(R.RunStateError, match="another arm"):
        G.load_graded(d)


def test_graded_round_trips():
    g = G.Graded(QuestionKey("P", "ans", 1), "a", "person", 66.5, False, "path", True, None)
    assert G.Graded.from_dict(g.to_dict()) == g


# ---------------------------------------------------------------- grader audit sample (F10)
def test_audit_sample_is_seeded_stratified_by_type_and_sized():
    qs = mk_questions(products=("P1", "P2", "P3"), n_ans=20, n_unans=10)
    typed = [(q.key, q.type) for q in qs]
    a = G.audit_sample(typed, 50, 1)
    assert len(a) == 50 and a == G.audit_sample(typed, 50, 1) and a != G.audit_sample(typed, 50, 2)
    by_type = {}
    for k in a:
        by_type[dict(typed)[k]] = by_type.get(dict(typed)[k], 0) + 1
    assert set(by_type) == {"person", "company", "pr", "url", "content", "unanswerable"}
    assert by_type["unanswerable"] > by_type["person"]  # proportional: unanswerable has 30 of 90 questions
    assert len(G.audit_sample(typed[:10], 50, 1)) == 10


# ---------------------------------------------------------------- the port of evaluate.py
@pytest.fixture(scope="module")
def herb():
    if not EVALUATE.exists():
        pytest.skip("data/raw/code/evaluate.py is absent")
    sent = []
    stub = types.ModuleType("utils")
    stub.reply = {}
    stub.sent = sent

    def get_gpt4_response(messages, **kw):
        sent.append(messages[0]["content"])
        return stub.reply

    stub.get_gpt4_response = get_gpt4_response
    sys.modules["utils"] = stub
    try:
        spec = importlib.util.spec_from_file_location("herb_evaluate", EVALUATE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        del sys.modules["utils"]
    mod.stub = stub
    return mod


SAMPLES = ["The Quick, brown FOX!", "  an  apple a day  ", "eid_9f3a2b", "https://x.y/pull/12", "", "A", "Café — au lait"]


def test_normalize_matches_herb(herb):
    for s in SAMPLES:
        assert G.normalize(s) == herb.normalize(s)


def test_f1_matches_herb(herb):
    cases = [([], []), (["a"], []), ([], ["a"]), (["a", "b"], ["a"]), (["a", "A"], ["a", "c", "c"]),
             (["The X", "y"], ["x", "Y."]), (["a", "b", "c"], ["a", "b", "c"]), (["a"], ["b"])]
    for t, p in cases:
        assert G.f1_score_sets(t, p) == herb.f1_score_sets(t, p)


def test_prompts_are_byte_identical_to_herbs(herb):
    herb.stub.sent.clear()
    herb.stub.reply = {"score": "50"}
    herb.answer_likert_score("Q {x}?", ["ref", "list"], "cand {y}")
    herb.stub.reply = {"done": "yes"}
    herb.unanswerability_eval("Q {x}?", "A: {y}")
    likert, unans = herb.stub.sent
    assert G.likert_prompt("Q {x}?", ["ref", "list"], "cand {y}") == likert
    assert G.unanswerability_prompt("Q {x}?", "A: {y}") == unans


def test_score_mappings_match_herb(herb):
    for raw in ("85", 85, 85.0, "0", 100):
        herb.stub.reply = {"score": raw}
        want = int(herb.answer_likert_score("q", "r", "c"))
        assert G.likert_score_from({"score": raw}) == want
    # HERB does not strip: " no " is not "no" there. Here such a value is a judge error, never a score.
    herb.stub.reply = {"done": " no "}
    assert herb.unanswerability_eval("q", "a") is True  # HERB: answered (score 0)
    with pytest.raises(G.JudgeError):
        G.refusal_from({"done": " no "})
    for done in ("no", "No", "NO", "yes", "Yes"):
        herb.stub.reply = {"done": done}
        herb_says_answered = herb.unanswerability_eval("q", "a")
        assert (G.refusal_from({"done": done}) == "no") == (not herb_says_answered)


def test_default_judge_model_is_resolved_once_and_keys_the_cache(tmp_path, monkeypatch):
    cache = tmp_path / "c.jsonl"
    chat = FakeChat()
    j = G.Judge(cache, chat=chat)
    assert j.model == G.MODEL and j.identity()["model"] == G.MODEL
    j.likert("q", "r", "c")
    assert chat.calls[0][1]["model"] == G.MODEL
    monkeypatch.setattr(G, "MODEL", "other/model")
    j2 = G.Judge(cache, chat=chat)
    j2.likert("q", "r", "c")  # a different default model must not reuse the old cached score
    assert len(chat.calls) == 2 and chat.calls[1][1]["model"] == "other/model"
