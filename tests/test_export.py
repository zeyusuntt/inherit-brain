import hashlib
import json

import pytest

from eval import grade as G
from eval import receipt as rc
from eval import run as R
from eval.arms import ARMS
from inherit.audit import check_receipts

from evalfx import FACTS, env, make_arm, mk_questions, result
from test_grade import FakeChat


def finished(tmp_path, monkeypatch, stop=False):
    monkeypatch.setitem(ARMS, "fake", make_arm(lambda ei: result(text=f"ans {ei.text}", ids=["x"], evidence=["e0"])))
    qs = mk_questions()
    d = R.run_arm("fake", "dev", None, root=tmp_path, questions=qs, env=env(), cap_setter=lambda a, u: None)
    graded = G.grade_run(d, qs, G.Judge(None, chat=FakeChat(default='{"done": "no", "score": 70}')))
    return d, qs, graded


def test_export_writes_a_table_and_a_receipt_bound_by_hashes(tmp_path, monkeypatch):
    d, qs, graded = finished(tmp_path, monkeypatch)
    out = tmp_path / "results"
    table, receipt = rc.export_run(d, graded, out, judge_model="m-judge")
    rows = [json.loads(l) for l in table.read_text().splitlines()]
    assert [r["key"] for r in rows] == [str(q.key) for q in qs]
    for k in ("key", "arm", "type", "outcome", "answer_ids", "evidence_ids", "answer_text", "failure", "usage", "latency_s",
              "decision", "score", "failed", "seen"):
        assert k in rows[0]
    rec = json.loads(receipt.read_text())
    assert receipt == out / "receipts" / "fake-dev.json" and table == out / "tables" / "fake-dev.jsonl"
    assert rec["facts_hash"] == FACTS and rec["status"] == "complete"
    assert rec["table_sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
    assert rec["judge"] == {"model": "m-judge", "prompt_version": G.PROMPT_VERSION}
    assert rec["records_sha256"] == hashlib.sha256((d / "records.jsonl").read_bytes()).hexdigest()
    prereg = tmp_path / "PREREG.md"
    prereg.write_text(f"facts hash: {FACTS}\n")
    assert check_receipts([out / "receipts"], prereg).ok


def test_export_never_writes_gold_or_citations(tmp_path, monkeypatch):
    d, qs, graded = finished(tmp_path, monkeypatch)
    table, receipt = rc.export_run(d, graded, tmp_path / "results", judge_model="m")
    text = table.read_text() + receipt.read_text()
    for q in qs:
        for g in (q.gold if isinstance(q.gold, list) else [q.gold]):
            if g:
                assert g not in text
        for c in q.citations:
            assert f'"{c}"' not in text


def test_export_refuses_an_unfinished_run_or_incomplete_grading(tmp_path, monkeypatch):
    d, qs, graded = finished(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="every slot"):
        rc.export_run(d, graded[:-1], tmp_path / "results", judge_model="m")
    rcpt = json.loads((d / "receipt.json").read_text())
    rcpt["status"] = "interrupted"
    (d / "receipt.json").write_text(json.dumps(rcpt))
    with pytest.raises(ValueError, match="complete"):
        rc.export_run(d, graded, tmp_path / "results", judge_model="m")
