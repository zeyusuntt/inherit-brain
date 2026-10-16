import gzip
import hashlib
import json
import threading
import time

import pytest

from eval import run as R
from eval.arms import ARMS, Arm, ArmResult
from eval.receipt import facts_hash, ledger_cost_for_run
from inherit import llm
from inherit.contracts import QuestionKey, RunRecord

from evalfx import FACTS, env, final, make_arm, mk_questions, result


@pytest.fixture
def reg(monkeypatch):
    def _reg(arm):
        monkeypatch.setitem(ARMS, arm.name, arm)
        return arm
    return _reg


def go(tmp_path, arm="fake", split="dev", questions=None, limit=None, **kw):
    kw.setdefault("env", env())
    kw.setdefault("cap_setter", lambda a, usd: None)
    return R.run_arm(arm, split, limit, root=tmp_path, questions=questions or mk_questions(), **kw)


def lines(run_dir, name="records.jsonl"):
    p = run_dir / name
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_fresh_run_writes_manifest_one_terminal_record_per_slot_and_receipt(tmp_path, reg):
    reg(make_arm())
    qs = mk_questions()
    d = go(tmp_path, questions=qs)
    assert d == tmp_path / "runs" / "eval" / "fake-dev"
    m = json.loads((d / "manifest.json").read_text())
    for k in ("arm", "split", "code_commit", "arm_config_sha", "facts_hash", "bundle_sha", "prompt_versions",
              "model", "slots", "limit", "seed"):
        assert k in m
    assert m["slots"] == [str(q.key) for q in qs]
    recs = lines(d)
    assert sorted(r["key"]["product"] + str(r["key"]["index"]) + r["key"]["kind"] for r in recs) == sorted(
        q.key.product + str(q.key.index) + q.key.kind for q in qs)
    assert len(recs) == len(qs)
    rec = json.loads((d / "receipt.json").read_text())
    assert rec["status"] == "complete" and rec["counts"]["slots"] == len(qs) and rec["counts"]["run_failed"] == 0


def test_arm_receives_exec_input_only(tmp_path, reg):
    seen = []
    reg(make_arm(lambda ei: (seen.append(ei), result(ids=["x"]))[1]))
    go(tmp_path)
    assert seen and all(type(s).__name__ == "ExecInput" and set(s.to_dict()) == {"product", "text"} for s in seen)


def test_started_attempts_are_logged_before_the_arm_runs(tmp_path, reg):
    d = tmp_path / "runs" / "eval" / "fake-dev"

    def arm(ei):
        assert any(s["slot"].startswith(ei.product + "/") for s in lines(d, "started.jsonl"))
        return result()

    reg(make_arm(arm))
    go(tmp_path, workers=1)
    assert len(lines(d, "started.jsonl")) == len(mk_questions())


def test_raising_arm_is_a_run_failed_record_and_the_others_continue(tmp_path, reg):
    def arm(ei):
        if "item 1" in ei.text:
            raise RuntimeError("boom")
        return result(ids=["x"])

    reg(make_arm(arm))
    d = go(tmp_path)
    recs = [RunRecord.from_dict(r) for r in lines(d)]
    bad = [r for r in recs if r.final.outcome == "run_failed"]
    assert len(bad) == 2 and all("RuntimeError: boom" in r.failure for r in bad)
    assert len(recs) == len(mk_questions())


@pytest.mark.parametrize("bad", ["text", None, ArmResult(final(ids=[1]), [], [])])
def test_invalid_arm_output_is_run_failed(tmp_path, reg, bad):
    reg(make_arm(lambda ei: bad))
    d = go(tmp_path, questions=mk_questions(products=("P1",), n_ans=1, n_unans=0))
    (r,) = [RunRecord.from_dict(x) for x in lines(d)]
    assert r.final.outcome == "run_failed" and r.failure


def test_arm_reported_failure_keeps_the_arms_own_record(tmp_path, reg):
    f = final(text="", outcome="run_failed")
    reg(make_arm(lambda ei: ArmResult(f, [], [], failure="deadline 300 s")))
    d = go(tmp_path, questions=mk_questions(products=("P1",), n_ans=1, n_unans=0))
    (r,) = [RunRecord.from_dict(x) for x in lines(d)]
    assert r.failure == "deadline 300 s" and len(lines(d)) == 1


def test_at_most_four_questions_in_flight(tmp_path, reg):
    live, peak, lock = [0], [0], threading.Lock()

    def arm(ei):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.02)
        with lock:
            live[0] -= 1
        return result()

    reg(make_arm(arm))
    go(tmp_path, questions=mk_questions(n_ans=10))
    assert 1 <= peak[0] <= 4
    with pytest.raises(ValueError):
        go(tmp_path, run_id="w", workers=5)


def test_attempt_tag_is_set_for_the_arm_and_reset_after(tmp_path, reg):
    tags = []
    reg(make_arm(lambda ei: (tags.append(llm.ATTEMPT_TAG.get()), result())[1]))
    go(tmp_path, questions=mk_questions(products=("P1",), n_ans=1, n_unans=0), workers=1)
    assert tags == ["fake-dev/P1/ans/0/1"]
    assert llm.ATTEMPT_TAG.get() is None


def test_tool_envelopes_saved_beside_the_record(tmp_path, reg):
    reg(make_arm())
    d = go(tmp_path, questions=mk_questions(products=("P1",), n_ans=1, n_unans=0))
    with gzip.open(d / "tools" / "P1__ans__0.json.gz", "rt") as f:
        assert json.load(f) == [{"tool": "graph", "rows": [[1], [2]]}]


def test_sub_cap_is_set_through_the_hook_before_dispatch(tmp_path, reg):
    calls = []
    reg(make_arm(sub_cap_usd=2.5))
    go(tmp_path, cap_setter=lambda a, usd: calls.append((a, usd, len(lines(tmp_path / "runs/eval/fake-dev")))))
    assert calls == [("fake", 2.5, 0)]


# ---------------------------------------------------------------- resume and recovery (F6)
class Stop(KeyboardInterrupt):
    pass


def stop_on(q, calls=None):
    """An arm that dies (as a killed process would) when it reaches question q."""

    def arm(ei):
        if calls is not None:
            calls.append(ei.text)
        if ei.text == q.text:
            raise Stop()
        return result(ids=["x"])

    return arm


def test_resume_skips_finished_keys_and_reruns_the_unfinished_slot_once(tmp_path, reg):
    qs = mk_questions()
    reg(make_arm(stop_on(qs[4])))
    with pytest.raises(Stop):
        go(tmp_path, questions=qs, workers=1)
    d = tmp_path / "runs" / "eval" / "fake-dev"
    assert len(lines(d)) == 4 and json.loads((d / "receipt.json").read_text())["status"] == "interrupted"
    unfinished = qs[4]
    calls2 = []
    reg(make_arm(lambda ei: (calls2.append(ei.text), result(ids=["x"]))[1]))
    go(tmp_path, questions=qs, workers=1)
    assert len(calls2) == len(qs) - 4 and unfinished.text in calls2
    assert len(lines(d)) == len(qs)
    starts = [s for s in lines(d, "started.jsonl") if s["slot"] == str(unfinished.key)]
    assert [s["attempt"] for s in starts] == [1, 2]
    # a completed run reruns nothing
    calls2.clear()
    go(tmp_path, questions=qs, workers=1)
    assert calls2 == []


def test_slot_started_twice_without_a_terminal_record_is_not_run_a_third_time(tmp_path, reg):
    qs = mk_questions()
    for _ in range(2):
        reg(make_arm(stop_on(qs[4])))
        with pytest.raises(Stop):
            go(tmp_path, questions=qs, workers=1)
    # the second attempt of slot 4 died too; the third run abandons it instead of paying again
    calls = []
    reg(make_arm(lambda ei: (calls.append(ei.text), result())[1]))
    d = go(tmp_path, questions=qs, workers=1)
    assert qs[4].text not in calls
    rec = {r["key"]["product"] + "/" + r["key"]["kind"] + "/" + str(r["key"]["index"]): r for r in lines(d)}
    r = rec[str(qs[4].key)]
    assert r["final"]["outcome"] == "run_failed" and "without a terminal record" in r["failure"]


def test_incompatible_resume_is_refused(tmp_path, reg):
    reg(make_arm())
    qs = mk_questions()
    go(tmp_path, questions=qs)
    for kw in (
        dict(env=env(code_commit="other")),
        dict(env=env(facts_hash="0" * 64)),
        dict(env=env(bundle_sha="0" * 64)),
        dict(env=env(model="m2")),
        dict(questions=qs[:-1]),
        dict(limit=3),
    ):
        with pytest.raises(R.IncompatibleResume):
            go(tmp_path, **{"questions": qs, **kw})
    reg(make_arm(config={"max_turns": 9}))
    with pytest.raises(R.IncompatibleResume, match="arm_config_sha"):
        go(tmp_path, questions=qs)
    reg(make_arm(prompt_versions={"sys": "v2"}))
    with pytest.raises(R.IncompatibleResume, match="prompt_versions"):
        go(tmp_path, questions=qs)


def test_duplicate_terminal_record_is_an_error(tmp_path, reg):
    reg(make_arm())
    qs = mk_questions()
    d = go(tmp_path, questions=qs)
    first = (d / "records.jsonl").read_text().splitlines()[0]
    with open(d / "records.jsonl", "a") as f:
        f.write(first + "\n")
    with pytest.raises(R.DuplicateTerminalRecord):
        go(tmp_path, questions=qs)
    with pytest.raises(R.DuplicateTerminalRecord):
        R.read_records(d)


def test_record_for_a_slot_outside_the_manifest_is_an_error(tmp_path, reg):
    reg(make_arm())
    d = go(tmp_path)
    rec = json.loads((d / "records.jsonl").read_text().splitlines()[0])
    rec["key"]["product"] = "Z"
    with open(d / "records.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    with pytest.raises(R.RunStateError, match="not in the manifest"):
        R.read_records(d)


def test_limit_is_refused_on_held_before_anything_is_written(tmp_path, reg):
    reg(make_arm())
    with pytest.raises(ValueError, match="dev"):
        go(tmp_path, split="held", limit=5, expected_facts_hash=FACTS)
    assert not (tmp_path / "runs").exists()


def test_held_requires_the_pre_registered_facts_hash(tmp_path, reg):
    reg(make_arm())
    with pytest.raises(R.RunStateError, match="facts"):  # no PREREG.md and no hash given
        go(tmp_path, split="held")
    with pytest.raises(R.RunStateError, match="facts"):
        go(tmp_path, split="held", expected_facts_hash="0" * 64)
    (tmp_path / "PREREG.md").write_text(f"facts hash: {FACTS}\n")
    d = go(tmp_path, split="held")  # read from PREREG.md
    assert (d / "receipt.json").exists()
    with pytest.raises(R.RunStateError, match="facts"):
        go(tmp_path, split="dev", run_id="d2", expected_facts_hash="0" * 64)


def test_unknown_arm_split_and_run_id(tmp_path):
    with pytest.raises(KeyError):
        go(tmp_path, arm="nope")
    with pytest.raises(ValueError):
        go(tmp_path, split="test")
    with pytest.raises(ValueError):
        go(tmp_path, run_id="a/b")


def test_limit_sample_is_seeded_stratified_and_in_frozen_order(tmp_path):
    qs = mk_questions(products=("P1", "P2", "P3"), n_ans=10, n_unans=4)
    a = R.sample_slots(qs, 6, 7)
    assert a == R.sample_slots(qs, 6, 7) and a != R.sample_slots(qs, 6, 8)
    assert {q.type for q in a} == {"person", "company", "pr", "url", "content", "unanswerable"}
    order = {q.key: i for i, q in enumerate(qs)}
    assert [order[q.key] for q in a] == sorted(order[q.key] for q in a)
    b = R.sample_slots(qs, 18, 7)
    assert {q.product for q in b} == {"P1", "P2", "P3"} and len(b) == 18
    assert R.sample_slots(qs, 999, 7) == qs


def test_reconcile_counts_never_started_slots_as_run_failed(tmp_path, reg):
    qs = mk_questions()
    reg(make_arm(stop_on(qs[2])))
    with pytest.raises(Stop):
        go(tmp_path, questions=qs, workers=1)
    d = tmp_path / "runs" / "eval" / "fake-dev"
    full = R.reconcile(d)
    assert [r.key for r in full] == [q.key for q in qs]
    missing = [r for r in full if r.final.outcome == "run_failed"]
    assert len(missing) == len(qs) - 2 and all("no terminal record" in r.failure for r in missing)


# ---------------------------------------------------------------- receipts (F8, F9)
def test_receipt_has_required_fields_and_hashes(tmp_path, reg):
    reg(make_arm(sub_cap_usd=1.0, config={"max_turns": 8}))
    d = go(tmp_path)
    rec = json.loads((d / "receipt.json").read_text())
    for k in ("run_id", "arm", "split", "status", "started_utc", "ended_utc", "code_commit", "bundle_sha",
              "facts_hash", "split_facts_hash", "arm_config_sha", "model", "prompt_versions", "output_sha256",
              "counts", "cost_usd", "ledger_cost_usd", "manifest_sha256", "records_sha256"):
        assert k in rec, k
    assert rec["facts_hash"] == FACTS and rec["model"] == "m1"
    assert rec["started_utc"].endswith("Z") and rec["ended_utc"] >= rec["started_utc"]
    stored = {json.loads(l)["key"]["index"]: l for l in (d / "records.jsonl").read_text().splitlines()
              if json.loads(l)["key"]["product"] == "P1" and json.loads(l)["key"]["kind"] == "ans"}
    assert rec["output_sha256"]["P1/ans/0"] == hashlib.sha256(stored[0].encode()).hexdigest()
    assert len(rec["output_sha256"]) == len(mk_questions())
    assert rec["records_sha256"] == hashlib.sha256((d / "records.jsonl").read_bytes()).hexdigest()
    assert rec["manifest_sha256"] == hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest()
    assert rec["cost_usd"] == pytest.approx(0.0001 * len(mk_questions()))


def test_ledger_cost_for_run_sums_tagged_settles_including_earlier_attempts(tmp_path):
    m = tmp_path / "meter.jsonl"
    ev = [
        {"type": "reserve", "id": "1", "arm": "a", "amount": 10**9, "tag": "r1/P/ans/0/1"},
        {"type": "settle", "id": "1", "arm": "a", "cost": 2 * 10**9, "tag": "r1/P/ans/0/1"},
        {"type": "settle", "id": "2", "arm": "a", "cost": 3 * 10**9, "tag": "r1/P/ans/0/2"},
        {"type": "settle", "id": "3", "arm": "a", "cost": 5 * 10**9, "tag": "r10/P/ans/0/1"},
        {"type": "settle", "id": "4", "arm": "a", "cost": 7 * 10**9},
    ]
    m.write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    assert ledger_cost_for_run(m, "r1") == pytest.approx(5e-3)
    assert ledger_cost_for_run(tmp_path / "none.jsonl", "r1") is None


def test_interrupted_receipt_is_written(tmp_path, reg):
    qs = mk_questions()
    reg(make_arm(stop_on(qs[1])))
    with pytest.raises(Stop):
        go(tmp_path, workers=1)
    rec = json.loads((tmp_path / "runs/eval/fake-dev/receipt.json").read_text())
    assert rec["status"] == "interrupted" and rec["counts"]["finished"] == 1


def test_facts_hash_is_over_sorted_paths_lengths_and_bytes(tmp_path):
    f = tmp_path / "facts"
    (f / "A").mkdir(parents=True)
    (f / "B").mkdir()
    (f / "A" / "x.jsonl").write_bytes(b"ab")
    (f / "B" / "y.jsonl").write_bytes(b"c")
    h0 = facts_hash(f)
    assert h0 == facts_hash(f) and len(h0) == 64
    (f / ".DS_Store").write_bytes(b"junk")
    assert facts_hash(f) == h0
    (f / "A" / "x.jsonl").write_bytes(b"abc")
    assert facts_hash(f) != h0
    # moving a byte across a file boundary changes the hash
    (f / "A" / "x.jsonl").write_bytes(b"a")
    (f / "B" / "y.jsonl").write_bytes(b"bc")
    assert facts_hash(f) != h0 and facts_hash(f) != facts_hash(f, ["A"])
    assert facts_hash(f, ["A"]) != facts_hash(f, ["B"])
    assert facts_hash(tmp_path / "absent") == facts_hash(tmp_path / "absent")
