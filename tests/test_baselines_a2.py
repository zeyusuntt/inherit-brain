"""Task 11, unit A2: GraphRAG-SDK 1.4.0 arm. Offline: the SDK facade is replaced by a stub, the model by a script."""

import json
from types import SimpleNamespace

import pytest

from baselines import graphrag_sdk_arm as a2
from eval.arms import ARMS
from inherit import agent
from inherit.contracts import CallUsage, ExecInput, FinalAnswer
from inherit.llm import ChatResult

INP = ExecInput("mini", "Who reviewed PR 101 in mini?")


def _srcs(mini_bundle):
    return a2.source_texts("mini", mini_bundle)


# ---------------------------------------------------------------- source texts (F7)
def test_sources_cover_every_category(mini_bundle):
    ids = [s.doc_id for s in _srcs(mini_bundle)]
    kinds = {i.split(":", 1)[0] for i in ids}
    assert kinds == {"slack", "document", "meeting", "pr", "url", "employees", "customers"}
    assert len(ids) == len(set(ids))
    assert sum(i.startswith("slack:") for i in ids) >= 1


def test_sources_have_stable_ids_and_timestamps(mini_bundle):
    a, b = _srcs(mini_bundle), _srcs(mini_bundle)
    assert [(s.doc_id, s.text) for s in a] == [(s.doc_id, s.text) for s in b]
    by = {s.doc_id: s.text for s in a}
    slack = "\n".join(t for i, t in by.items() if i.startswith("slack:"))
    assert "2026-06-" in slack and "20260609-0-" in slack  # timestamp and message id on every line
    meeting = next(t for i, t in by.items() if i.startswith("meeting:"))
    assert "Meeting ID:" in meeting and "Chat" in meeting  # the transcript carries its chat
    pr = by["pr:https://github.com/mini/mini-server/pull/101"]
    assert "APPROVED" in pr and "eid_a1000001" in pr and "2026-06-16T09:00:00" in pr  # reviews with time
    assert any(i.startswith("url:") for i in by)


def test_sources_hold_directories_and_no_team_or_questions(mini_bundle):
    by = {s.doc_id: s.text for s in _srcs(mini_bundle)}
    emp = by["employees"]
    assert "eid_a1000001: Alice Chen (" in emp
    assert "CUST-" in by["customers"]
    blob = "\n".join(by.values())
    assert "answerable" not in blob and "TEAM STRUCTURE" not in blob


# ---------------------------------------------------------------- answer prompt (F8)
def test_answer_prompt_asks_for_shared_schema():
    for key in ("answer_text", "answer_ids", "evidence_ids", "outcome"):
        assert key in a2.ANSWER_SYSTEM
    assert "{context}" in a2.ANSWER_TEMPLATE and "{question}" in a2.ANSWER_TEMPLATE
    a2.ANSWER_TEMPLATE.format(context="c", question="q")  # no stray braces
    for o in agent.MODEL_OUTCOMES:
        assert o in a2.ANSWER_SYSTEM


# ---------------------------------------------------------------- the answer path
class StubRag:
    def __init__(self, content, ctx="Record 20260609-0-d019a: eid_a1000001 reviewed https://github.com/mini/mini-server/pull/101"):
        self.content, self.ctx, self.calls = content, ctx, []

    async def completion(self, question, **kw):
        self.calls.append((question, kw))
        items = [SimpleNamespace(content=self.ctx)]
        return SimpleNamespace(answer=self.content, retriever_result=SimpleNamespace(items=items))


class Meter:
    def __init__(self, calls=()):
        self.calls = list(calls)


def good(**kw):
    d = {"answer_text": "Alice reviewed it.", "answer_ids": ["eid_a1000001"],
         "evidence_ids": ["20260609-0-d019a"], "outcome": "path"}
    d.update(kw)
    return json.dumps(d)


class FakeModel:
    def __init__(self, *contents):
        self.contents, self.calls = list(contents), []

    def __call__(self, messages, *, arm, tools=None, max_tokens=800, response_format=None, **kw):
        self.calls.append({"messages": messages, "arm": arm, "max_tokens": max_tokens, "rf": response_format})
        msg = SimpleNamespace(content=self.contents.pop(0), tool_calls=None)
        return ChatResult(msg, 50, 10, 0.001, 0.1)


def run(rag, model=None):
    return a2.answer(INP, opener=lambda product: (rag, Meter()), model=model or FakeModel())


def test_valid_json_needs_no_conversion():
    rag, model = StubRag(good()), FakeModel()
    res = a2.answer(INP, opener=lambda p: (rag, Meter()), model=model)
    assert model.calls == []  # no conversion call
    f = res.final
    assert (f.outcome, f.answer_ids, f.evidence_ids) == ("path", ["eid_a1000001"], ["20260609-0-d019a"])
    q, kw = rag.calls[0]
    assert q == INP.text and kw["prompt_template"] == a2.ANSWER_TEMPLATE
    assert kw["history"][0]["role"] == "system" and kw["return_context"] is True
    assert res.failure is None


def test_conversion_only_on_parse_failure_and_keeps_both_outputs():
    prose = "Alice Chen (eid_a1000001) reviewed it."
    model = FakeModel(good())
    res = run(StubRag(prose), model)
    assert len(model.calls) == 1 and model.calls[0]["arm"] == "a2"
    sys_or_user = json.dumps(model.calls[0]["messages"])
    assert prose in sys_or_user and "do not add" in sys_or_user.lower()
    assert res.final.answer_ids == ["eid_a1000001"]
    assert len(res.calls) == 1
    env = [e for e in res.envelopes if "a2" in e][0]["a2"]
    assert env["sdk_answer"] == prose and env["converted"] is True and env["conversion"] == good()


def test_conversion_that_still_fails_is_run_failed():
    res = run(StubRag("prose"), FakeModel("still not json"))
    assert res.final.outcome == "run_failed" and res.failure


def test_unsupported_ids_are_dropped_and_refusal_with_ids_becomes_path():
    raw = good(answer_ids=["eid_a1000001", "eid_deadbeef"], evidence_ids=["20260609-0-d019a", "20990101-9-fffff"])
    f = run(StubRag(raw)).final
    assert f.answer_ids == ["eid_a1000001"] and f.evidence_ids == ["20260609-0-d019a"]
    f = run(StubRag(good(outcome="not_in_graph"))).final
    assert f.outcome == "path"
    f = run(StubRag(good(answer_ids=[], answer_text="", outcome="path"))).final
    assert f.outcome == "run_failed"


def test_qualified_ids_become_bare():
    f = run(StubRag(good(answer_ids=["Person:eid_a1000001"]))).final
    assert f.answer_ids == ["eid_a1000001"]


def test_output_validates_against_shared_schema():
    res = run(StubRag(good()))
    again = FinalAnswer.from_dict(res.final.to_dict())
    assert again == res.final
    assert res.final.trace == []


def test_sdk_calls_and_conversion_are_accounted():
    sdk = [CallUsage("a2", "m", 10, 5, 0.002, 0.2)]
    model = FakeModel(good())
    res = a2.answer(INP, opener=lambda p: (StubRag("prose"), Meter(sdk)), model=model)
    assert len(res.calls) == 2 and res.final.usage.n_calls == 2
    assert res.final.usage.prompt_tokens == 60


def test_opener_failure_is_a_terminal_run_failed():
    def boom(product):
        raise RuntimeError("graph a2:mini is not built")

    res = a2.answer(INP, opener=boom, model=FakeModel())
    assert res.final.outcome == "run_failed" and "not built" in res.failure


# ---------------------------------------------------------------- providers through the proxy
def test_proxy_llm_goes_through_chat_once_and_records_usage(monkeypatch):
    seen = []

    def fake_chat(messages, *, arm, max_tokens=800, response_format=None, **kw):
        seen.append((arm, max_tokens, response_format))
        return ChatResult(SimpleNamespace(content="hi"), 7, 3, 0.5, 0.1)

    monkeypatch.setattr(a2.llm, "chat", fake_chat)
    m = a2.UsageLog()
    llm_ = a2.ProxyLLM("a2-build", m)
    assert llm_.invoke("p").content == "hi"
    assert [(c.arm, c.prompt_tokens, c.cost_usd) for c in m.calls] == [("a2-build", 7, 0.5)]
    assert seen[0][0] == "a2-build"


def test_proxy_llm_json_mode_only_for_message_calls(monkeypatch):
    seen = []
    monkeypatch.setattr(a2.llm, "chat", lambda messages, *, arm, max_tokens=800, response_format=None, **kw:
                        (seen.append(response_format), ChatResult(SimpleNamespace(content="{}"), 1, 1, 0.0, 0.0))[1])
    import asyncio
    from graphrag_sdk import ChatMessage

    llm_ = a2.ProxyLLM("a2", a2.UsageLog(), json_messages=True)
    asyncio.run(llm_.ainvoke("keywords?"))
    asyncio.run(llm_.ainvoke_messages([ChatMessage(role="user", content="q")]))
    assert seen == [None, {"type": "json_object"}]


def test_proxy_embedder_requests_1024_dims_and_records(monkeypatch):
    got = {}

    def fake_embed(texts, *, arm, dimensions=None, **kw):
        got.update(arm=arm, dims=dimensions, n=len(texts))
        return SimpleNamespace(vectors=[[0.0] * 1024 for _ in texts], prompt_tokens=11, cost_usd=0.001)

    monkeypatch.setattr(a2.llm, "embed", fake_embed)
    m = a2.UsageLog()
    e = a2.ProxyEmbedder("a2-build", m)
    assert len(e.embed_query("x")) == 1024 and got["dims"] == 1024 and got["arm"] == "a2-build"
    assert len(e.embed_documents(["a", "b"])) == 2
    assert sum(c.prompt_tokens for c in m.calls) == 22


# ---------------------------------------------------------------- caps and registration
def test_caps_are_set_before_use(monkeypatch):
    posted = []
    monkeypatch.setattr(a2, "_post_cap", lambda arm, usd: posted.append((arm, usd)))
    a2.set_caps()
    assert posted == [("a2-build", 1.0), ("a2", 0.3)]


def test_arm_registered():
    arm = ARMS["A2"]
    assert arm.config["graph"] == "a2:<product>" and arm.config["model"]
    assert arm.sub_cap_usd == 0.3
