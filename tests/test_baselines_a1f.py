"""Task 11, unit A1f / A1: long context. The model is scripted; the token counter is injected where the real
tokenizer is not the point. One test uses the real Qwen3 tokenizer file when it is cached."""

import json
from types import SimpleNamespace

import pytest

from baselines import long_context as lc
from inherit import agent
from eval import a1_fields
from eval.arms import ARMS
from eval.run import _validate
from inherit.contracts import ExecInput
from inherit.llm import ChatResult

EMPLOYEES = {"eid_b2": {"employee_id": "eid_b2", "name": "Bob Ray", "role": "Product Manager"},
             "eid_a1": {"employee_id": "eid_a1", "name": "Alice Chen", "role": "Software Engineer"}}
CUSTOMERS = [{"name": "Tom", "role": "CTO", "company": "TechCorp", "id": "CUST-0002"},
             {"name": "Lee", "role": "PM", "company": "BlueWave", "id": "CUST-0001"}]


def msg(mid, uid, text, ts, ch="ch1", name="general"):
    return {"Channel": {"name": name, "channelID": ch}, "id": mid,
            "Message": {"User": {"userId": uid, "timestamp": ts, "text": text, "utterranceID": mid}}}


def bundle():
    return {
        "slack": [msg("m2", "eid_a1", "second", "2026-06-10T10:00:00"), msg("m1", "eid_b2", "first", "2026-06-10T09:00:00")],
        "documents": [{"id": "D1", "document_link": "https://d/1", "date": "2026-04-01", "author": "eid_a1",
                       "type": "Report", "content": "doc body"}],
        "meeting_transcripts": [{"id": "T1", "date": "2026-05-01", "participants": ["eid_a1"], "transcript": "talk"}],
        "meeting_chats": [{"id": "C1", "text": "chat"}],
        "urls": [{"link": "https://u/1", "description": "a site"}],
        "prs": [{"title": "Fix", "user": {"login": "alice"}, "created_at": "2026-01-01", "state": "closed",
                 "mergeable": True, "merged": True, "link": "https://github.com/a/b/pull/1", "summary": "sum",
                 "reviews": [{"state": "APPROVED", "user": {"login": "bob"}, "submitted_at": "2026-01-02",
                               "comment": "ok"}]}],
    }


def heads(ctx):
    order = ["EMPLOYEES:", "CUSTOMERS:", "TEAM STRUCTURE:", "SLACK CHANNELS", "MEETING TRANSCRIPTS", "MEETING CHATS",
             "DOCUMENTS", "URLs", "PULL REQUESTS"]
    return [h for _, h in sorted((ctx.find(h), h) for h in order if h in ctx)]


def test_a1f_context_has_no_team_or_customers_fields_and_has_both_directories():
    ctx = lc.build_context_a1f(bundle(), EMPLOYEES, CUSTOMERS)
    assert "TEAM STRUCTURE" not in ctx
    assert "EMPLOYEES:\neid_a1: Alice Chen (Software Engineer)\neid_b2: Bob Ray (Product Manager)\n" in ctx
    assert "CUSTOMERS:\nCUST-0001: BlueWave\nCUST-0002: TechCorp\n" in ctx


def test_slack_lines_carry_message_ids_in_time_order():
    ctx = lc.build_context_a1f(bundle(), EMPLOYEES, CUSTOMERS)
    assert "Channel: general\n[m1] eid_b2: first\n[m2] eid_a1: second" in ctx


def test_section_order_matches_the_papers_builder_with_directories_first():
    ctx = lc.build_context_a1f(bundle(), EMPLOYEES, CUSTOMERS)
    assert heads(ctx) == ["EMPLOYEES:", "CUSTOMERS:", "SLACK CHANNELS", "MEETING TRANSCRIPTS", "MEETING CHATS",
                          "DOCUMENTS", "URLs", "PULL REQUESTS"]


def test_a1_context_is_the_papers_builder_with_team_and_customers_and_no_ids():
    ctx = lc.build_context_a1(bundle(), EMPLOYEES, {"team": ["eid_a1", "eid_b2"], "customers": ["CUST-0001"]},
                              {"CUST-0001": "BlueWave"})
    assert ctx.startswith("TEAM STRUCTURE:\nSoftware Engineer: eid_a1 (Alice Chen)\nProduct Manager: eid_b2 (Bob Ray)\n")
    assert "CUSTOMERS:\nCUST-0001: BlueWave\n" in ctx
    assert "EMPLOYEES:" not in ctx and "[m1]" not in ctx and "eid_b2: first" in ctx
    assert heads(ctx) == ["TEAM STRUCTURE:", "CUSTOMERS:", "SLACK CHANNELS", "MEETING TRANSCRIPTS", "MEETING CHATS",
                          "DOCUMENTS", "URLs", "PULL REQUESTS"]


# ---------------------------------------------------------------- the run
def scripted(content, prompt=500, completion=30, seen=None):
    def model(messages, *, arm, max_tokens, response_format=None, **kw):
        if seen is not None:
            seen.append({"messages": messages, "arm": arm, "max_tokens": max_tokens})
        return ChatResult(SimpleNamespace(content=content), prompt, completion, 0.01, 0.1)
    return model


GOOD = json.dumps({"answer_text": "eid_a1 wrote it", "answer_ids": ["eid_a1"], "evidence_ids": ["m2"],
                   "outcome": "path"})


@pytest.fixture
def bdir(tmp_path):
    from inherit.data.bundle import make_bundle
    raw = tmp_path / "raw"
    (raw / "products").mkdir(parents=True)
    (raw / "metadata").mkdir()
    (raw / "products" / "ActionGenie.json").write_text(json.dumps({**bundle(), "team": ["x"], "customers": ["y"]}))
    (raw / "metadata" / "employee.json").write_text(json.dumps(EMPLOYEES))
    (raw / "metadata" / "customers_data.json").write_text(json.dumps(CUSTOMERS))
    (raw / "metadata" / "salesforce_team.json").write_text("{}")
    out = tmp_path / "bundle"
    make_bundle(raw, out)
    return out


def test_a1f_output_validates_against_the_shared_schema(bdir):
    seen = []
    res = _validate(lc.run(ExecInput("ActionGenie", "who wrote second?"), model=scripted(GOOD, seen=seen),
                           bundle_dir=bdir, count=lambda s: 10))
    assert res.final.outcome == "path" and res.final.answer_ids == ["eid_a1"] and res.failure is None
    assert len(res.calls) == 1 and res.calls[0].arm == "A1f"
    user = seen[0]["messages"][-1]["content"]
    assert "who wrote second?" in user and "EMPLOYEES:" in user and "TEAM STRUCTURE" not in user
    assert seen[0]["max_tokens"] == lc.OUTPUT_HEADROOM or seen[0]["max_tokens"] <= lc.OUTPUT_HEADROOM


def test_invalid_model_output_is_a_run_failure(bdir):
    res = _validate(lc.run(ExecInput("ActionGenie", "q"), model=scripted("nope"), bundle_dir=bdir, count=lambda s: 10))
    assert res.final.outcome == "run_failed" and res.failure


def test_oversized_request_fails_without_a_model_call_or_truncation(bdir):
    seen = []
    res = _validate(lc.run(ExecInput("ActionGenie", "q"), model=scripted(GOOD, seen=seen), bundle_dir=bdir,
                           count=lambda s: lc.LIMIT - lc.OUTPUT_HEADROOM + 1))
    assert seen == [] and res.final.outcome == "run_failed" and "262144" in res.failure and res.calls == []


def test_request_exactly_at_the_limit_is_sent(bdir):
    seen = []
    res = lc.run(ExecInput("ActionGenie", "q"), model=scripted(GOOD, seen=seen), bundle_dir=bdir,
                 count=lambda s: lc.LIMIT - lc.OUTPUT_HEADROOM)
    assert len(seen) == 1 and res.failure is None


def test_rendered_template_is_counted_whole_with_the_generation_prefix():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    seen = []
    lc.count_rendered(lambda s: seen.append(s) or 0, msgs)
    assert seen == ["<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"]


def test_real_template_boundary_counts_the_framing_tokens():
    if not lc.TOKENIZER_PATH.exists():
        pytest.skip("tokenizer.json not cached")
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    # 2 content tokens + 13 framing tokens (role markers, newlines, end markers, assistant prefix) = 15, not 2 + 2 * 4
    assert lc.request_tokens(msgs, lc.count_tokens) == 15


def refusal_run(bdir, content):
    return _validate(lc.run(ExecInput("ActionGenie", "q"), model=scripted(content), bundle_dir=bdir,
                            count=lambda s: 10))


@pytest.mark.parametrize("outcome", sorted(agent.REFUSALS))
def test_refusal_prose_is_replaced_by_the_canonical_text(bdir, outcome):
    res = refusal_run(bdir, json.dumps({"answer_text": "Nobody, I guess.", "answer_ids": [], "evidence_ids": [],
                                        "outcome": outcome}))
    assert res.final.outcome == outcome and res.final.answer_ids == []
    assert res.final.answer_text == agent.refusal_text([], [])


def test_refusal_carrying_supported_ids_becomes_path(bdir):
    res = refusal_run(bdir, json.dumps({"answer_text": "Alice wrote it", "answer_ids": ["eid_a1"],
                                        "evidence_ids": ["m2"], "outcome": "not_in_graph"}))
    assert res.final.outcome == "path" and res.final.answer_ids == ["eid_a1"]
    assert res.final.answer_text.startswith("eid_a1. ")


def test_a1_refuses_a_held_out_product_and_never_calls_the_model(bdir):
    seen = []
    res = _validate(lc.run(ExecInput("SomeHeldProduct", "q"), arm="A1", model=scripted(GOOD, seen=seen),
                           bundle_dir=bdir, count=lambda s: 10))
    assert seen == [] and res.final.outcome == "run_failed" and "dev" in res.failure


def test_a1_fields_loader_refuses_held_out_and_reads_only_two_keys(tmp_path):
    raw = tmp_path / "raw" / "products"
    raw.mkdir(parents=True)
    (raw / "ActionGenie.json").write_text(json.dumps({"team": ["a"], "customers": ["c"], "slack": [1]}))
    (raw / "HeldOne.json").write_text(json.dumps({"team": ["a"], "customers": ["c"]}))
    assert a1_fields.load_fields("ActionGenie", tmp_path / "raw") == {"team": ["a"], "customers": ["c"]}
    with pytest.raises(ValueError, match="dev"):
        a1_fields.load_fields("HeldOne", tmp_path / "raw")


def test_arms_are_registered():
    for n in ("A1f", "A1", "dev-a1f", "dev-a1"):
        assert n in ARMS
    assert ARMS["A1f"].config["limit_tokens"] == 262144


def test_real_tokenizer_counts_when_cached():
    if not lc.TOKENIZER_PATH.exists():
        pytest.skip("tokenizer.json not cached")
    assert 0 < lc.count_tokens("Hello world, this is a test.") < 20
