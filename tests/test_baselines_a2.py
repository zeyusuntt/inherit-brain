"""Task 11, unit A2: GraphRAG-SDK 1.4.0 arm. Offline: the SDK facade is replaced by a stub, the model by a script."""

import asyncio
import json
import random
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
    assert kinds - {"chat"} == {"slack", "document", "meeting", "prs", "urls", "employees", "customers"}
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
    pr = by["prs"]
    assert "pull/101" in pr and "APPROVED" in pr and "eid_a1000001" in pr and "2026-06-16T09:00:00" in pr
    assert "https://docs.example.com/export" in by["urls"]  # the duplicated link keeps both records


def test_sources_hold_directories_and_no_team_or_questions(mini_bundle):
    by = {s.doc_id: s.text for s in _srcs(mini_bundle)}
    emp = by["employees"]
    assert "eid_a1000001: Alice Chen (" in emp
    assert "CUST-" in by["customers"]
    blob = "\n".join(by.values())
    assert "answerable" not in blob and "TEAM STRUCTURE" not in blob


def test_documents_carry_feedback_and_every_chat_is_a_source(mini_bundle):
    by = {s.doc_id: s.text for s in _srcs(mini_bundle)}
    assert "Specify the retry limit." in by["document:mini_design_spec"]  # feedback
    chats = {i: t for i, t in by.items() if i.startswith("chat:")}
    assert "chat:orphan_chat" in chats  # unmatched chat, original id
    assert "orphan_chat" in chats["chat:orphan_chat"] and "Unrelated chat." in chats["chat:orphan_chat"]
    assert not any(i.startswith("chat:mini_design_1") for i in chats)  # a matched chat stays in its meeting


def test_every_allowed_record_of_the_product_is_in_some_source(mini_bundle):
    blob = "\n".join(s.text for s in _srcs(mini_bundle))
    raw = json.loads((mini_bundle / "products" / "mini.json").read_text(encoding="utf-8"))
    emp, cust = a2.load_metadata(mini_bundle)
    missing = []
    for r in raw["slack"]:
        u = r["Message"]["User"]
        missing += [v for v in (u["utterranceID"], u["timestamp"], u["userId"], u["text"]) if v not in blob]
        missing += [x for x in r["ThreadReplies"] if json.dumps(x) not in blob]
    for d in raw["documents"]:
        missing += [v for v in (d["id"], d["document_link"], d["content"], d["author"], d["date"], d["type"],
                                d.get("feedback") or "") if v not in blob]
    for m in raw["meeting_transcripts"]:
        missing += [v for v in (m["id"], m["transcript"], m["date"], *m["participants"]) if v not in blob]
    for c in raw["meeting_chats"]:
        missing += [v for v in (c["id"], c["text"]) if v not in blob]
    for p in raw["prs"]:
        missing += [v for v in (p["link"], p["title"], p["summary"], p["user"]["login"], p["created_at"], p["number"])
                    if v not in blob]
        for rv in p["reviews"]:
            missing += [v for v in (rv["state"], rv["user"]["login"], rv["comment"], rv["submitted_at"]) if v not in blob]
    for u in raw["urls"]:
        missing += [v for v in (u["link"], u["description"]) if v not in blob]
    missing += [e for e in emp if e not in blob] + [c["id"] for c in cust if c["id"] not in blob]
    assert missing == []


def test_discovery_sample_is_small_deterministic_and_covers_categories(mini_bundle):
    srcs = _srcs(mini_bundle)
    sub = a2.discovery_sources(srcs)
    assert sub == a2.discovery_sources(srcs)
    kinds = lambda xs: {s.doc_id.split(":", 1)[0] for s in xs}  # noqa: E731
    assert kinds(sub) == kinds(srcs)
    for k in kinds(sub):
        assert sum(s.doc_id.split(":", 1)[0] == k for s in sub) <= a2.DISCOVERY_PER_CATEGORY


# ---------------------------------------------------------------- answer prompt (F8)
def test_answer_prompt_asks_for_shared_schema():
    for key in ("answer_text", "answer_ids", "evidence_ids", "outcome"):
        assert key in a2.ANSWER_SYSTEM
    assert "{context}" in a2.ANSWER_TEMPLATE and "{question}" in a2.ANSWER_TEMPLATE
    a2.ANSWER_TEMPLATE.format(context="c", question="q")  # no stray braces
    for o in agent.MODEL_OUTCOMES:
        assert o in a2.ANSWER_SYSTEM


# ---------------------------------------------------------------- the answer path
CTX = "Record 20260609-0-d019a: eid_a1000001 reviewed https://github.com/mini/mini-server/pull/101"


class StubRag:
    """Stands in for the SDK facade. `sdk_calls` are (final, prompt tokens) model calls it makes through the
    question's budget before it answers, as the SDK's retrieval and answer prompts do."""

    def __init__(self, content, ctx=CTX, sdk_calls=()):
        self.content, self.ctx, self.calls, self.sdk_calls, self.budget = content, ctx, [], list(sdk_calls), None

    async def completion(self, question, **kw):
        self.calls.append((question, kw))
        keep = self.budget.model
        for final, tokens in self.sdk_calls:
            self.budget.model = lambda *a, _t=tokens, **k: ChatResult(SimpleNamespace(content="x"), _t, 1, 0.001, 0.1)
            self.budget.chat([{"role": "user", "content": "p"}], arm=a2.ARM_ANSWER, max_tokens=8, final=final)
        self.budget.model = keep
        items = [SimpleNamespace(content=self.ctx)] if self.ctx is not None else []
        return SimpleNamespace(answer=self.content, retriever_result=SimpleNamespace(items=items))


@pytest.fixture(autouse=True)
def _mini_is_allowed(monkeypatch):
    monkeypatch.setattr(a2, "allowed_products", lambda *a, **k: frozenset({"mini"}))


def opener(rag):
    def open_(product, budget):
        rag.budget = budget
        return rag
    return open_


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


def run(rag, model=None, inp=INP, **kw):
    return a2.answer(inp, opener=opener(rag), model=model or FakeModel(), **kw)


def test_valid_json_needs_no_conversion():
    rag, model = StubRag(good()), FakeModel()
    res = run(rag, model)
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
    assert len(model.calls) == 1 and model.calls[0]["arm"] == a2.ARM_ANSWER == "A2"
    sys_or_user = json.dumps(model.calls[0]["messages"])
    assert prose in sys_or_user and "do not add" in sys_or_user.lower()
    assert res.final.answer_ids == ["eid_a1000001"]
    assert len(res.calls) == 1
    env = [e for e in res.envelopes if "a2" in e][0]["a2"]
    assert env["sdk_answer"] == prose and env["converted"] is True and env["conversion"] == good()
    assert env["model_fields"]["answer_ids"] == ["eid_a1000001"]  # the model's own fields stay in the diagnostics


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


def test_question_only_id_is_not_supported_by_the_question():
    inp = ExecInput("mini", "Which role does eid_b2000002 have?")
    raw = good(answer_ids=["eid_b2000002"], answer_text="eid_b2000002 is the QA lead.", outcome="path")
    res = run(StubRag(raw, ctx="Nothing about that person."), inp=inp)
    assert res.final.answer_ids == [] and "eid_b2000002" not in res.final.answer_text
    diag = [e for e in res.envelopes if "a2" in e][0]["a2"]
    assert diag["model_fields"]["answer_ids"] == ["eid_b2000002"]
    assert {"value": "eid_b2000002", "where": "ids", "reason": "not an answer value"} in diag["dropped"]


def test_a_link_is_supported_only_by_the_complete_token():
    ctx = "PR https://github.com/mini/mini-server/pull/10 was reviewed."
    raw = good(answer_ids=["https://github.com/mini/mini-server/pull/1"], evidence_ids=[],
               answer_text="See https://github.com/mini/mini-server/pull/1.")
    f = run(StubRag(raw, ctx=ctx)).final
    assert f.answer_ids == [] and "pull/1." not in f.answer_text and "pull/1 " not in f.answer_text
    raw = good(answer_ids=["https://github.com/mini/mini-server/pull/10"], evidence_ids=[], answer_text="PR 10.")
    assert run(StubRag(raw, ctx=ctx)).final.answer_ids == ["https://github.com/mini/mini-server/pull/10"]


def test_refusal_is_the_canonical_text_and_the_original_is_kept():
    raw = good(answer_ids=[], evidence_ids=[], outcome="not_in_graph", answer_text="Sorry, I do not know that.")
    res = run(StubRag(raw))
    assert res.final.outcome == "not_in_graph" and res.final.answer_text == agent.REFUSAL_SENTENCE  # the shared sentence
    diag = [e for e in res.envelopes if "a2" in e][0]["a2"]
    assert diag["model_fields"]["answer_text"] == "Sorry, I do not know that."
    assert diag["searched"] == "graphrag_retrieval"
    res = run(StubRag(raw, ctx=None))  # nothing retrieved
    assert res.final.answer_text == agent.REFUSAL_SENTENCE
    assert [e for e in res.envelopes if "a2" in e][0]["a2"]["searched"] == "no tool returned results"


def test_qualified_ids_become_bare():
    f = run(StubRag(good(answer_ids=["Person:eid_a1000001"]))).final
    assert f.answer_ids == ["eid_a1000001"]


def test_output_validates_against_shared_schema():
    res = run(StubRag(good()))
    again = FinalAnswer.from_dict(res.final.to_dict())
    assert again == res.final
    assert res.final.trace == []


def test_sdk_calls_and_conversion_are_accounted():
    rag = StubRag("prose", sdk_calls=[(False, 10), (True, 50)])
    res = run(rag, FakeModel(good()))
    assert len(res.calls) == 3 and res.final.usage.n_calls == 3
    assert res.final.usage.prompt_tokens == 110
    diag = [e for e in res.envelopes if "a2" in e][0]["a2"]
    assert diag["model_calls"] == 3 and diag["input_tokens"] == 110


def test_opener_failure_is_a_terminal_run_failed():
    def boom(product, budget):
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

    llm_ = a2.ProxyLLM(a2.ARM_ANSWER, a2.UsageLog(), json_messages=True)
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
    assert posted == [("a2-build", 1.7), ("A2", 0.3)]


def test_one_meter_name_for_answering(monkeypatch):
    arm = ARMS["A2"]
    assert arm.name == a2.ARM_ANSWER  # the name the runner caps is the name every answering call is metered under
    seen = {}
    monkeypatch.setattr(a2, "_facade", lambda product, arm_, log, **kw: seen.update(arm=arm_, log=log) or "rag")
    monkeypatch.setattr(a2, "build_problems", lambda product, sources: [])
    monkeypatch.setattr(a2, "_expected_sources", lambda product: [])
    budget = a2.QuestionBudget()
    assert a2.open_graph("mini", budget) == "rag" and seen["arm"] == "A2" and seen["log"] is budget


def test_arm_registered():
    arm = ARMS["A2"]
    assert arm.config["graph"] == "a2:<product>" and arm.config["model"]
    assert arm.sub_cap_usd == 0.3
    assert arm.config["held_out_sample_seed"] == 20261015 and arm.config["held_out_sample_size"] == 6


def test_registration_needs_no_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(a2, "BUNDLE_DIR", tmp_path / "missing")
    monkeypatch.setattr(a2, "frozen_sample", lambda *a, **k: (_ for _ in ()).throw(ValueError("no bundle")))
    assert a2.arm_config()["held_out_sample_size"] == 6  # config does not draw the sample


def test_without_a_bundle_the_arm_fails_the_question_not_the_import(monkeypatch):
    monkeypatch.setattr(a2, "allowed_products", lambda: (_ for _ in ()).throw(FileNotFoundError("no bundle")))
    res = run(StubRag(good()))
    assert res.final.outcome == "run_failed" and "held-out sample" in res.failure


# ---------------------------------------------------------------- one budget per question (F2)
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def scripted(tokens=10, clock=None, dt=0.0):
    def model(messages, *, arm, max_tokens, response_format=None, **kw):
        if clock:
            clock.t += dt
        return ChatResult(SimpleNamespace(content="x"), tokens, 1, 0.001, 0.1)
    return model


def call(b, final=False, text="p"):
    return b.chat([{"role": "user", "content": text}], arm="A2", max_tokens=8, final=final)


def test_budget_limits_calls_and_reserves_the_last_two_for_finalisation():
    b = a2.QuestionBudget(model=scripted())
    for _ in range(a2.agent.MAX_TOTAL_MODEL_CALLS - a2.agent.RESERVED_FINAL_CALLS):
        call(b)
    with pytest.raises(a2.BudgetStop):
        call(b)  # the 23rd non-final call is refused
    call(b, final=True)
    call(b, final=True)  # the answer and the conversion may use the reserve
    with pytest.raises(a2.BudgetStop):
        call(b, final=True)
    assert len(b.calls) == 24 == b.started


def test_budget_input_policy_stops_non_final_calls_only():
    b = a2.QuestionBudget(model=scripted(tokens=59_000))
    call(b)
    with pytest.raises(a2.BudgetStop):
        call(b, text="x" * 4000)  # 59,000 used + 2,000 estimated > 60,000
    call(b, text="short")  # 59,000 + 2 stays within the budget
    call(b, final=True, text="x" * 4000)  # finalisation is not stopped by the input policy
    assert b.input_tokens == 59_000 * 3


def test_budget_deadline_is_checked_before_and_after_each_call():
    clock = Clock()
    b = a2.QuestionBudget(model=scripted(clock=clock, dt=200.0), clock=clock)
    call(b)
    with pytest.raises(a2.DeadlineExceeded):  # the second call's response arrives after 400 s
        call(b)
    assert len(b.calls) == 2 and b.deadline_hit  # its usage is still recorded
    n = len(b.calls)
    with pytest.raises(a2.DeadlineExceeded):  # expired: no call is made
        call(b)
    assert len(b.calls) == n


def test_proxy_llm_calls_share_the_question_budget(monkeypatch):
    b = a2.QuestionBudget(model=scripted())
    llm_ = a2.ProxyLLM(a2.ARM_ANSWER, b, json_messages=True)
    for _ in range(22):
        llm_.invoke("p")
    with pytest.raises(a2.BudgetStop):
        llm_.invoke("p")
    from graphrag_sdk import ChatMessage
    asyncio.run(llm_.ainvoke_messages([ChatMessage(role="user", content="q")]))  # the SDK's answer call is final
    assert len(b.calls) == 23


def test_conversion_is_charged_to_the_same_budget():
    rag = StubRag("prose", sdk_calls=[(False, 5)] * 22 + [(True, 5)])  # 23 calls, one reserved call left
    model = FakeModel(good())
    res = run(rag, model)
    assert res.failure is None and len(res.calls) == 24 and res.final.answer_ids == ["eid_a1000001"]
    rag = StubRag("prose", sdk_calls=[(False, 5)] * 22 + [(True, 5)] * 2)  # all 24 used by the SDK
    res = run(rag, FakeModel(good()))
    assert res.final.outcome == "run_failed" and "model call limit" in res.failure


def test_conversion_after_the_deadline_is_refused():
    clock = Clock()
    rag = StubRag("prose")
    inner = rag.completion

    async def slow(question, **kw):
        out = await inner(question, **kw)
        clock.t += 301.0
        return out

    rag.completion = slow
    model = FakeModel(good())
    res = run(rag, model, clock=clock)
    assert res.final.outcome == "run_failed" and "deadline" in res.failure and model.calls == []


def test_a_provider_failure_the_sdk_swallowed_is_reported(monkeypatch):
    rag = StubRag(good())
    inner = rag.completion

    async def soft(question, **kw):
        try:
            rag.budget.model = lambda *a, **k: (_ for _ in ()).throw(a2.llm.BudgetExceeded("cap"))
            call(rag.budget)
        except Exception:  # noqa: BLE001 - the SDK's soft-fail
            pass
        return await inner(question, **kw)

    rag.completion = soft
    res = run(rag)
    assert res.final.outcome == "run_failed" and res.failure == "budget_exceeded"


# ---------------------------------------------------------------- complete builds only (C1)
class FakeStore:
    def __init__(self, rag):
        self.rag = rag

    async def get_document_record(self, doc_id):
        return object() if doc_id in self.rag.docs else None


class FakeBuildRag:
    """Records what the build does. `ingest_hook(document_id)` may raise or swallow provider failures."""

    def __init__(self, log, hook=None, docs=()):
        self.log, self.hook, self.docs, self.deleted, self.finalized = log, hook, set(docs), [], 0
        self.llm, self._graph_store = None, FakeStore(self)
        self.order = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def set_ontology(self, o):
        pass

    async def delete_document(self, doc_id, if_missing="error"):
        self.deleted.append(doc_id)
        self.docs.discard(doc_id)

    async def ingest(self, *, text, document_id, extractor):
        self.docs.add(document_id)  # the SDK writes the document record before extraction
        self.order.append(document_id)
        if self.hook:
            await self.hook(self, document_id)
        return SimpleNamespace(chunks_indexed=2)

    async def finalize(self):
        self.finalized += 1
        if self.fin_hook:
            await self.fin_hook(self)
        return SimpleNamespace(model_dump=lambda: {"entities_embedded": 3})

    fin_hook = None

    async def get_statistics(self):
        return {"nodes": 1}


@pytest.fixture
def build_env(monkeypatch, tmp_path, mini_bundle):
    holder = {}

    def facade(product, arm, log, **kw):
        rag = holder["rag"] = FakeBuildRag(log, holder.get("hook"), holder.get("docs", ()))
        rag.fin_hook = holder.get("fin_hook")
        return rag

    async def ontology(product, sources, log, report):
        report["entity_types"], report["relation_types"] = [], []
        return SimpleNamespace(entities=[], relations=[])

    monkeypatch.setattr(a2, "_facade", facade)
    monkeypatch.setattr(a2, "_load_ontology", ontology)
    monkeypatch.setattr(a2, "_extractor", lambda rag, ontology: None)
    monkeypatch.setattr(a2, "state_path", lambda product: tmp_path / f"{product}.build.json")
    monkeypatch.setattr(a2, "set_caps", lambda: None)
    holder["sources"] = a2.source_texts("mini", mini_bundle)
    monkeypatch.setattr(a2, "_expected_sources", lambda product: holder["sources"])
    holder["bundle"] = mini_bundle
    return holder


def run_build(env, **kw):
    return a2.build("mini", env["bundle"], **kw)


def test_clean_build_completes_every_source_and_finalizes(build_env):
    rep = run_build(build_env)
    ids = [s.doc_id for s in build_env["sources"]]
    assert sorted(build_env["rag"].order) == sorted(ids) and build_env["rag"].finalized == 1
    assert a2.build_problems("mini", build_env["sources"]) == []
    assert rep["sources_complete"] == len(ids)


def test_ner_soft_failures_are_counted_in_the_manifest_not_rejected(build_env):
    import logging

    async def noisy(rag, doc_id):
        logging.getLogger("graphrag_sdk.x").warning("LLM NER returned invalid JSON, skipping chunk abc")
    build_env["hook"] = noisy
    rep = run_build(build_env)
    n = len(build_env["sources"])
    assert rep["ner_soft_failures_this_run"] == n and a2.read_state("mini")["ner_soft_failures"] == n
    build_env["docs"] = set(build_env["rag"].docs)
    rep = run_build(build_env)  # a resumed run with nothing to ingest adds nothing
    assert rep["ner_soft_failures_this_run"] == 0 and rep["ner_soft_failures_total"] == n


def test_build_problems_name_missing_sources_and_missing_finalize(build_env):
    assert a2.build_problems("mini", build_env["sources"])  # nothing built yet
    async def die_on_last(rag, doc_id):
        if doc_id == "urls":
            raise RuntimeError("402")
    build_env["hook"] = die_on_last
    with pytest.raises(a2.BuildIncomplete):
        run_build(build_env)
    problems = " ".join(a2.build_problems("mini", build_env["sources"]))
    assert "1 of" in problems and "urls" in problems and "finalize" in problems


def test_a_failed_source_is_not_complete_and_is_not_finalized(build_env):
    async def die(rag, doc_id):
        if doc_id == "prs":
            raise RuntimeError("provider down")
    build_env["hook"] = die
    with pytest.raises(a2.BuildIncomplete) as e:
        run_build(build_env)
    assert "prs" in str(e.value) and build_env["rag"].finalized == 0
    assert "prs" in build_env["rag"].deleted  # the half-written document is removed
    assert "prs" not in a2.read_state("mini")["sources"]


def test_a_provider_failure_the_sdk_swallowed_rejects_the_source(build_env, monkeypatch):
    monkeypatch.setattr(a2.llm, "chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("402 payment")))

    async def soft(rag, doc_id):
        if doc_id.startswith("slack:") or doc_id == "employees":
            try:  # the SDK extraction catches the provider error and goes on with an empty result
                await a2.ProxyLLM(a2.ARM_BUILD, rag.log).ainvoke("extract")
            except Exception:  # noqa: BLE001
                pass
    build_env["hook"] = soft
    with pytest.raises(a2.BuildIncomplete) as e:
        run_build(build_env)
    state = a2.read_state("mini")["sources"]
    swallowed = {s.doc_id for s in build_env["sources"] if s.doc_id.startswith("slack:") or s.doc_id == "employees"}
    assert swallowed and not swallowed & set(state) and "employees" in str(e.value)
    assert set(state) == {s.doc_id for s in build_env["sources"]} - swallowed
    assert build_env["rag"].finalized == 0


def test_an_interrupted_source_is_deleted_and_ingested_again(build_env):
    ok = {s.doc_id for s in build_env["sources"]} - {"prs"}
    run_build(build_env)
    st = a2.read_state("mini")
    st["sources"].pop("prs"), st.update(finalized=False)
    a2.write_state("mini", st)  # as after a crash: the record exists, completion was never written
    build_env["docs"] = ok | {"prs"}
    rep = run_build(build_env)
    rag = build_env["rag"]
    assert rag.order == ["prs"] and "prs" in rag.deleted and rag.finalized == 1
    assert a2.build_problems("mini", build_env["sources"]) == [] and rep["sources_complete"] == len(ok) + 1


def test_a_finalize_failure_leaves_the_build_unfinished(build_env, monkeypatch):
    async def fin(rag):
        try:
            monkeypatch.setattr(a2.llm, "embed", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("embed down")))
            a2.ProxyEmbedder(a2.ARM_BUILD, rag.log).embed_documents(["x"])
        except RuntimeError:
            pass  # swallowed by the SDK
    build_env["fin_hook"] = fin
    with pytest.raises(a2.BuildIncomplete):
        run_build(build_env)
    assert any("finalize" in p for p in a2.build_problems("mini", build_env["sources"]))


def test_a_provider_failure_in_discovery_rejects_the_draft_before_ingesting(build_env, monkeypatch, tmp_path):
    async def ontology(product, sources, log, report):
        log.fail("chat", RuntimeError("402"))
        report["entity_types"] = report["relation_types"] = []
        return SimpleNamespace(entities=[], relations=[])

    monkeypatch.setattr(a2, "_load_ontology", ontology)
    monkeypatch.setattr(a2, "ontology_path", lambda product: tmp_path / "o.json")
    with pytest.raises(a2.BuildIncomplete, match="discovery"):
        run_build(build_env)
    assert "rag" not in build_env  # the graph was never opened


def test_open_graph_refuses_an_incomplete_build(build_env):
    with pytest.raises(a2.BuildIncomplete):
        a2.open_graph("mini", a2.QuestionBudget())
    res = a2.answer(INP, model=FakeModel())  # the real opener
    assert res.final.outcome == "run_failed" and "incomplete" in res.failure


# ---------------------------------------------------------------- the pre-registered held-out sample
def test_held_out_sample_is_the_seeded_draw():
    held = [f"P{i:02d}" for i in range(24)]
    assert a2.held_out_sample(held) == random.Random(20261015).sample(sorted(held), 6)
    assert a2.held_out_sample(list(reversed(held))) == a2.held_out_sample(held)  # input order does not matter


def test_allowed_products_are_dev_plus_the_sample(tmp_path):
    (tmp_path / "products").mkdir()
    names = ["ActionGenie"] + [f"Prod{i:02d}" for i in range(29)]
    for n in names:
        (tmp_path / "products" / f"{n}.json").write_text("{}")
    dev, held = a2.split(names)
    allowed = a2.products_allowed(tmp_path)
    assert allowed == frozenset(dev) | frozenset(a2.held_out_sample(held)) and len(allowed) == 12


def test_the_arm_refuses_held_out_products_outside_the_sample(monkeypatch):
    monkeypatch.setattr(a2, "allowed_products", lambda *a, **k: frozenset({"ActionGenie", "EdgeForce"}))
    rag = StubRag(good())
    res = run(rag, inp=ExecInput("PitchForce", "q"))
    assert res.final.outcome == "run_failed" and "not in the pre-registered sample" in res.failure and rag.calls == []
    assert run(StubRag(good()), inp=ExecInput("ActionGenie", "q")).failure is None
    assert run(StubRag(good()), inp=ExecInput("EdgeForce", "q")).failure is None


@pytest.mark.skipif(not (a2.BUNDLE_DIR / "products").exists(), reason="no local bundle")
def test_the_recorded_sample_on_the_real_split():
    assert a2.frozen_sample() == ["EdgeForce", "WorkFlowGenie", "PitchForce", "ProposalForce", "AutoTuneForce",
                                  "CoachForce"]
