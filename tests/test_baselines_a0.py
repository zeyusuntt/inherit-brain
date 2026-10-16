"""Task 11, unit A0 / A0f / A0v: the paper's retriever and tools in the shared agent loop, over raw records or cards,
and the single-retrieval hybrid RAG. The model is scripted; the embeddings are the deterministic test vectors.
Card tests run on the Task 8 fixture graph on both FalkorDB servers."""

import copy
import json
from types import SimpleNamespace

import pytest

from baselines import cards_react, hybrid_once, paper_retriever, paper_tools, react_paper
from baselines.paper_retriever import QwenEmbedding, build_retriever, paper_texts, union_nodes
from baselines.paper_tools import Lookups, PaperTools
from conftest import fake_embed, fake_vec
from eval.arms import ARMS
from eval.run import _validate
from inherit import agent
from inherit.contracts import ExecInput
from inherit.llm import ChatResult
from toolfx import ALICE, BOB, P, pr, tg, tsrv  # noqa: F401  (fixtures)

ALICE_ID, BOB_ID = "eid_a1000001", "eid_b2000002"
EMPLOYEES = {
    ALICE_ID: {"employee_id": ALICE_ID, "name": "Alice Chen", "role": "Software Engineer"},
    BOB_ID: {"employee_id": BOB_ID, "name": "Bob Ray", "role": "Product Manager"},
    "eid_c3000003": {"employee_id": "eid_c3000003", "name": "Carl Ray", "role": "Software Engineer"},
}
CUSTOMERS = [{"name": "Tom", "role": "CTO", "company": "TechCorp", "id": "CUST-0001"},
             {"name": "Sara", "role": "PM", "company": "TechCorp", "id": "CUST-0002"},
             {"name": "Lee", "role": "PM", "company": "BlueWave", "id": "CUST-0003"}]
LINK_A, LINK_B = "https://github.com/acme/app/pull/1", "https://github.com/other/app/pull/9"


def bundle(prs=(), urls=(), docs=(), slack=()):
    return {"slack": list(slack), "documents": list(docs), "meeting_transcripts": [], "meeting_chats": [],
            "urls": list(urls), "prs": list(prs)}


def slack_msg(uid, text, mid):
    return {"Channel": {"name": "chan", "channelID": "ch"}, "ThreadReplies": [{"x": 1}], "id": mid,
            "Message": {"User": {"userId": uid, "timestamp": "2026-06-10T09:00:00", "text": text,
                                 "utterranceID": mid}, "Reactions": []}}


def doc(i, link, content="body"):
    return {"id": i, "document_link": link, "date": "2026-04-01", "author": "eid_a1000001", "type": "Report",
            "content": content}


# ---------------------------------------------------------------- scripted model
class T:
    def __init__(self, *calls, prompt=100, completion=20):
        self.calls, self.prompt, self.completion = calls, prompt, completion


class Say:
    def __init__(self, content, prompt=100, completion=20):
        self.content, self.prompt, self.completion = content, prompt, completion


def J(text="", ids=(), evidence=(), outcome="path", **kw):
    return Say(json.dumps({"answer_text": text, "answer_ids": list(ids), "evidence_ids": list(evidence),
                           "outcome": outcome}), **kw)


def D(**kw):
    return Say(json.dumps(kw))


class FakeModel:
    def __init__(self, steps, on_call=None):
        self.steps, self.calls, self.on_call = list(steps), [], on_call

    def __call__(self, messages, *, arm, tools=None, max_tokens=800, response_format=None, **kw):
        self.calls.append({"messages": copy.deepcopy(messages), "arm": arm, "tools": tools,
                           "max_tokens": max_tokens, "response_format": response_format})
        if self.on_call:
            self.on_call(len(self.calls))
        step = self.steps.pop(0)
        if isinstance(step, T):
            tcs = [SimpleNamespace(id=f"c{len(self.calls)}_{i}", type="function",
                                   function=SimpleNamespace(name=n, arguments=json.dumps(a)))
                   for i, (n, a) in enumerate(step.calls)]
            msg = SimpleNamespace(content=None, tool_calls=tcs)
        else:
            msg = SimpleNamespace(content=step.content, tool_calls=None)
        return ChatResult(msg, step.prompt, step.completion, 0.001, 0.1)


class Retriever:
    def __init__(self, chunks):
        self.chunks, self.queries = chunks, []

    def retrieve(self, q):
        self.queries.append(q)
        return list(self.chunks)


def lookups(**kw):
    return Lookups(bundle(**kw), EMPLOYEES, CUSTOMERS)


def run_a0(steps, chunks=("Message ID: m1\n\neid_a1000001: hello",), lk=None, q="Who greeted?", **kw):
    m = FakeModel(steps)
    r = Retriever(list(chunks))
    res = react_paper.run(ExecInput("acme", q), arm=kw.pop("arm", "A0"), model=m, retriever=r,
                          lookups=lk or lookups(), **kw)
    return res, m, r


# ---------------------------------------------------------------- the retriever
def test_splitter_settings_are_recorded_and_are_the_library_defaults():
    assert paper_retriever.SPLITTER == {"chunk_size": 1024, "chunk_overlap": 200}
    assert paper_retriever.effective_splitter() == paper_retriever.SPLITTER


def test_corpus_is_the_papers_documents_in_react_order(tmp_path):
    b = bundle(slack=[slack_msg("eid_a1000001", "hi", "20260610-0-aaaaa")], docs=[doc("d1", "https://x/d1")],
               prs=[{"link": LINK_A}], urls=[{"link": "https://u"}])
    b["meeting_transcripts"] = [{"id": "t1", "date": "2026-06-11", "participants": ["a", "b"], "transcript": "talk"}]
    b["meeting_chats"] = [{"id": "c1", "text": "chat text"}]
    texts = paper_texts(b)
    assert texts[0] == ("Message ID: 20260610-0-aaaaa\nChannel Name: chan\nTimestamp: 2026-06-10T09:00:00\n\n"
                        "eid_a1000001: hi")
    assert texts[1].startswith("Doc ID: d1\nLink: https://x/d1\n2026-04-01\nAuthor: eid_a1000001\n\nReport\nbody")
    assert texts[2] == "Meeting ID: t1\n2026-06-11\nParticipants: ['a', 'b']\n\ntalk"
    assert texts[3] == "Meeting ID: c1\n\nChats:\nchat text"
    assert len(texts) == 4  # no PRs, URLs or thread replies


def _node(i):
    return SimpleNamespace(node=SimpleNamespace(node_id=i), score=1.0)


def test_union_is_bm25_first_then_vector_deduplicated_by_node():
    got = union_nodes([_node("b1"), _node("b2"), _node("s1")], [_node("s1"), _node("v1"), _node("b1"), _node("v2")])
    assert [n.node.node_id for n in got] == ["b1", "b2", "s1", "v1", "v2"]


def _embedding(calls=None):
    def corpus(texts):
        if calls is not None:
            calls.append(len(texts))
        return fake_embed(texts)

    return QwenEmbedding(corpus_fn=corpus, query_fn=fake_vec)


def test_retriever_takes_20_and_20_with_bm25_first_and_persists(tmp_path):
    texts = [f"Message ID: m{i}\n\neid_a1000001: topic{i % 7} word{i} filler text" for i in range(80)]
    calls: list[int] = []
    r = build_retriever(texts, tmp_path / "idx", _embedding(calls))
    assert calls  # embedded on the first build
    nodes = r.retrieve_nodes("topic3 word5")
    bm25 = [n.node.node_id for n in r.bm25.retrieve("topic3 word5")]
    ids = [n.node.node_id for n in nodes]
    assert len(bm25) == 20 and ids[:20] == bm25
    assert len(ids) == len(set(ids)) and 20 < len(ids) <= 40
    plain = paper_retriever.SentenceSplitter().get_nodes_from_documents(
        [paper_retriever.Document(text=t) for t in texts])
    assert len(r.nodes) == len(plain)  # the stock splitter, default settings

    def boom(texts):
        raise AssertionError("must load the persisted index, not embed again")

    again = build_retriever(texts, tmp_path / "idx", QwenEmbedding(corpus_fn=boom, query_fn=fake_vec))
    assert [n.node.get_content() for n in again.retrieve_nodes("topic3 word5")][:20] == \
        [n.node.get_content() for n in nodes][:20]


# ---------------------------------------------------------------- the paper's tools
def test_a0_offers_the_papers_nine_tools():
    tools = PaperTools(None, lookups(), Retriever([]))
    names = [s["function"]["name"] for s in tools.specs()]
    assert names == ["retrieve_all", "role_to_id_search", "name_to_id_search", "company_to_id_search",
                     "id_to_company_search", "id_to_name_search", "id_to_role_search", "pr_search", "url_search"]
    res, m, _ = run_a0([J("x", outcome="not_in_graph"), D(done="yes")])
    assert [s["function"]["name"] for s in m.calls[0]["tools"]] == names
    assert "BM25" not in m.calls[0]["messages"][0]["content"]  # the paper's context prompt, not the graph prompt
    assert "knowledge graph" not in m.calls[0]["messages"][0]["content"]


def loop_stub(*replies):
    seen = []
    queue = list(replies)

    def nested_chat(messages, **kw):
        seen.append(messages[0]["content"])
        return SimpleNamespace(message=SimpleNamespace(content=queue.pop(0)))

    return SimpleNamespace(nested_chat=nested_chat, prompts=seen)


def test_role_to_id_search_formats_its_prompt_and_returns_ids():
    loop = loop_stub(json.dumps({"roles": ["Software Engineer"]}))
    tools = PaperTools(loop, lookups(), Retriever([]))
    env, _ = tools.call("role_to_id_search", {"text": "Who are the Software Engineers?"})  # upstream: KeyError
    assert "Who are the Software Engineers?" in loop.prompts[0] and "{text}" not in loop.prompts[0]
    assert env["result"] == {"software engineer": [ALICE_ID, "eid_c3000003"]}


def test_id_to_role_search_returns_roles_not_ids():
    loop = loop_stub(json.dumps({"ids": [ALICE_ID, "eid_unknown"]}))
    env, _ = PaperTools(loop, lookups(), Retriever([])).call("id_to_role_search", {"content": ALICE_ID})
    assert env["result"] == {ALICE_ID: "Software Engineer"}


def test_lookups_give_names_companies_and_empty_messages():
    loop = loop_stub(json.dumps({"names": ["Bob Ray", "Ray"]}), json.dumps({"ids": ["cust-0003", "CUST-9"]}),
                     json.dumps({"names": ["Nobody"]}), "not json")
    tools = PaperTools(loop, lookups(), Retriever([]))
    assert tools.call("name_to_id_search", {"content": "x"})[0]["result"] == {
        "bob ray": [BOB_ID], "ray": [BOB_ID, "eid_c3000003"]}
    assert tools.call("id_to_company_search", {"content": "x"})[0]["result"] == {"cust-0003": "BlueWave"}
    env = tools.call("company_to_id_search", {"content": "x"})[0]
    assert env["result"] == {} and env["message"].startswith("No IDs found")
    env = tools.call("pr_search", {"content": "x"})[0]  # an extraction that is not JSON is an empty result
    assert env["result"] == [] and env["message"].startswith("No PR details found")


def test_metadata_never_returns_another_products_entries():
    other = lookups(prs=[{"link": LINK_B, "title": "other"}], urls=[{"link": "https://other.example/u"}],
                    docs=[doc("od", "https://other.example/doc")])
    mine = lookups(prs=[{"link": LINK_A, "title": "mine"}])
    assert LINK_B.lower() in other.pr_metadata and LINK_B.lower() not in mine.pr_metadata
    loop = loop_stub(json.dumps({"links": [LINK_B, LINK_A]}), json.dumps({"links": ["https://other.example/u",
                                                                                      "https://other.example/doc"]}))
    tools = PaperTools(loop, mine, Retriever([]))
    assert [p["title"] for p in tools.call("pr_search", {"content": "x"})[0]["result"]] == ["mine"]
    assert tools.call("url_search", {"content": "x"})[0]["result"] == []


def test_url_search_prefers_a_document_on_a_link_collision():
    lk = lookups(urls=[{"link": "https://x/same", "description": "url"}], docs=[doc("d1", "https://x/same", "DOC")])
    assert lk.url_metadata["https://x/same"]["content"] == "DOC"


def test_an_answer_id_must_be_in_returned_text_or_it_is_dropped():
    steps = [T(("retrieve_all", {"query": "who greeted"})),
             J("Alice greeted.", ids=[ALICE_ID, "eid_deadbeef"], evidence=["m1", "m2"]), D(done="yes")]
    res, m, r = run_a0(steps)
    assert r.queries == ["who greeted"]
    assert res.final.answer_ids == [ALICE_ID]
    assert res.final.evidence_ids == ["m1"]
    assert res.final.outcome == "path"


# ---------------------------------------------------------------- one counter for every model call (F2)
def test_nested_calls_are_labelled_with_the_arm_and_counted_with_the_agent_turns():
    steps = [T(("name_to_id_search", {"content": "Alice Chen"})), D(names=["Alice Chen"]),
             J("Alice", ids=[ALICE_ID]), D(done="yes")]
    res, m, _ = run_a0(steps, lk=lookups(), arm="dev-a0")
    assert [c["arm"] for c in m.calls] == ["dev-a0"] * 4
    assert len(res.calls) == 4 and {c.arm for c in res.calls} == {"dev-a0"}
    assert res.final.usage.n_calls == 4
    assert m.calls[1]["response_format"] == {"type": "json_object"} and m.calls[1]["tools"] is None
    assert res.final.answer_ids == [ALICE_ID]  # ALICE_ID is in the lookup result
    assert [t.tool for t in res.final.trace] == ["name_to_id_search"]


def test_total_model_calls_and_tool_calls_are_limited_with_nested_calls_included():
    turns = []
    for _ in range(11):
        turns += [T(("name_to_id_search", {"content": "Alice"})), D(names=["Alice"])]
    steps = turns + [J("Alice", ids=[ALICE_ID]), D(done="no")]  # verification says no, but no call is left
    res, m, r = run_a0(steps)
    assert len(m.calls) == 24 == agent.MAX_TOTAL_MODEL_CALLS
    assert r.queries == []  # the fallback did not fit
    assert res.final.answer_ids == [ALICE_ID] and res.final.outcome == "path"
    diag = res.envelopes[-1]["agent"]
    assert diag["model_calls"] == 24 and diag["tool_calls"] == 11


def test_the_tool_call_limit_of_12_holds():
    steps = [T(("retrieve_all", {"query": str(i)})) for i in range(12)]
    steps += [J("done", ids=[], outcome="not_in_graph"), D(done="yes")]
    res, m, r = run_a0(steps)
    assert len(r.queries) == 12 and res.envelopes[-1]["agent"]["tool_calls"] == 12


def test_a_failed_verification_runs_the_fallback_once_with_a_retrieval_over_the_question():
    steps = [J("Bob maybe", ids=[BOB_ID]), D(done="no"),
             J("Alice per records", ids=[ALICE_ID], evidence=["m1"])]
    res, m, r = run_a0(steps, q="Who greeted?")
    assert r.queries == ["Who greeted?"] and len(m.calls) == 3
    assert "Given the context information and not prior knowledge" in m.calls[2]["messages"][0]["content"]
    assert res.final.answer_ids == [ALICE_ID]  # the fallback's answer replaces the first
    tools = [e["tool"] for e in res.envelopes if "tool" in e]
    assert tools == ["finish_agent", "fallback_answer"]


def test_a_deadline_during_a_nested_call_ends_the_question_as_run_failed():
    now = [0.0]

    def on_call(n):
        if n == 2:
            now[0] = 1000.0  # the extraction call returns after the 300 s deadline

    m = FakeModel([T(("name_to_id_search", {"content": "Alice"})), D(names=["Alice"])], on_call=on_call)
    res = react_paper.run(ExecInput("acme", "q"), model=m, retriever=Retriever([]), lookups=lookups(),
                          clock=lambda: now[0])
    assert res.final.outcome == "run_failed" and "deadline" in res.failure
    assert len(res.calls) == 2  # the late call's usage is still recorded


# ---------------------------------------------------------------- A0f: the same tools, returning cards (F4)
def card_lookups():
    return Lookups(
        bundle(prs=[{"link": pr(1), "title": "t"}, {"link": pr(2), "title": "u"}],
               urls=[{"link": "https://demo.example.com/v1"}, {"link": "https://demo.example.com/current"}],
               docs=[doc("latest_mrr", "https://demo.example.com/v1")]),
        {ALICE: {"employee_id": ALICE, "name": "Alice Ng", "role": "Software Engineer"},
         BOB: {"employee_id": BOB, "name": "Bob Roy", "role": "Software Engineer"},
         "eid_zz000000": {"employee_id": "eid_zz000000", "name": "Zed Nobody", "role": "Software Engineer"}},
        [{"name": "T", "role": "CTO", "company": "TechCorp", "id": "CUST-0001"},
         {"name": "S", "role": "PM", "company": "TechCorp", "id": "CUST-0002"}])


def card_tools(tg, *replies):
    loop = loop_stub(*[json.dumps(r) for r in replies])
    return cards_react.CardTools(loop, card_lookups(), Retriever([]), tg)


def ids(cards_):
    return [c["id"] for c in cards_]


def test_a0f_person_tools_return_person_cards(tg):
    t = card_tools(tg, {"roles": ["software engineer"]}, {"names": ["Alice Ng"]}, {"ids": [ALICE, "eid_zz000000"]},
                   {"ids": [BOB]})
    got = t.call("role_to_id_search", {"text": "x"})[0]["result"]
    assert ids(got["software engineer"]) == sorted([f"Person:{ALICE}", f"Person:{BOB}"])  # no card for Zed: left out
    assert all(c["label"] == "Person" and "neighbours" in c for c in got["software engineer"])
    assert ids(t.call("name_to_id_search", {"content": "x"})[0]["result"]["alice ng"]) == [f"Person:{ALICE}"]
    byid = t.call("id_to_name_search", {"content": "x"})[0]["result"]
    assert list(byid) == [ALICE] and byid[ALICE]["props"]["name"] == "Alice Ng"
    role = t.call("id_to_role_search", {"content": "x"})[0]["result"]
    assert role[BOB]["props"]["role"] == "Software Engineer"


def test_a0f_customer_pr_and_url_tools_return_cards(tg):
    t = card_tools(tg, {"names": ["techcorp"]}, {"ids": ["cust-0001"]}, {"links": [pr(2), pr(1), pr(1)]},
                   {"links": ["https://demo.example.com/v1", "https://demo.example.com/current"]},
                   {"links": ["https://nowhere.example"]})
    got = t.call("company_to_id_search", {"content": "x"})[0]["result"]
    assert ids(got["techcorp"]) == ["Customer:CUST-0001", "Customer:CUST-0002"]
    assert ids([t.call("id_to_company_search", {"content": "x"})[0]["result"]["cust-0001"]]) == ["Customer:CUST-0001"]
    prs = t.call("pr_search", {"content": "x"})[0]["result"]
    assert ids(prs) == [f"PR:{pr(1)}", f"PR:{pr(2)}"]  # ordered by card id, duplicates removed
    urls = t.call("url_search", {"content": "x"})[0]["result"]
    assert sorted(ids(urls)) == ["Document:latest_mrr", "Url:https://demo.example.com/current"]
    assert "Document:latest_mrr" in ids(urls)  # the document wins the collision with the url of the same link
    assert "Url:https://demo.example.com/v1" not in ids(urls)
    env = t.call("url_search", {"content": "x"})[0]
    assert env["result"] == [] and env["message"].startswith("No URL details found")


def test_a0f_response_budget_truncates(tg, monkeypatch):
    monkeypatch.setattr(cards_react, "ROW_CAP", 1)
    t = card_tools(tg, {"roles": ["software engineer"]})
    env = t.call("role_to_id_search", {"text": "x"})[0]
    assert len(env["result"]["software engineer"]) == 1 and env["truncated"] is True


def test_a0f_answer_ids_come_from_returned_cards(tg):
    loop_replies = [D(names=["Alice Ng"])]
    m = FakeModel([T(("name_to_id_search", {"content": "Alice Ng"})), *loop_replies,
                   J("Alice", ids=[f"Person:{ALICE}", "eid_deadbeef"]), D(done="yes")])
    res = cards_react.run(ExecInput(P, "Who is Alice Ng?"), model=m, retriever=Retriever(["{}"]),
                          lookups=card_lookups(), graph=tg, arm="A0f")
    assert res.final.answer_ids == [ALICE]
    assert {c.arm for c in res.calls} == {"A0f"}
    assert "entity cards" in m.calls[0]["messages"][0]["content"]


# ---------------------------------------------------------------- A0v: one retrieval, one answer call
def test_a0v_makes_exactly_one_retrieval_and_one_answer_call():
    m = FakeModel([J("Alice greeted.", ids=[ALICE_ID], evidence=["m1"])])
    r = Retriever(["Message ID: m1\n\neid_a1000001: hello", "Message ID: m2\n\neid_b2000002: bye"])
    res = hybrid_once.run(ExecInput("acme", "Who greeted?"), model=m, retriever=r)
    assert r.queries == ["Who greeted?"] and len(m.calls) == 1 and m.calls[0]["arm"] == "A0v"
    assert m.calls[0]["tools"] is None and m.calls[0]["response_format"] == {"type": "json_object"}
    assert "eid_b2000002: bye" in m.calls[0]["messages"][1]["content"]
    assert res.final.answer_ids == [ALICE_ID] and res.final.evidence_ids == ["m1"]
    assert res.final.usage.n_calls == 1 and len(res.calls) == 1


def test_a0v_invalid_json_is_run_failed_without_a_second_call():
    m = FakeModel([Say("not json")])
    res = hybrid_once.run(ExecInput("acme", "q"), model=m, retriever=Retriever(["c"]))
    assert res.final.outcome == "run_failed" and "invalid final answer" in res.failure and len(m.calls) == 1


def test_a0v_refusal_label_cannot_carry_answer_ids():
    m = FakeModel([J("Alice", ids=[ALICE_ID], outcome="not_in_graph")])
    res = hybrid_once.run(ExecInput("acme", "q"), model=m, retriever=Retriever(["eid_a1000001 said hi"]))
    assert res.final.outcome == "path" and res.final.answer_ids == [ALICE_ID]


# ---------------------------------------------------------------- registry and schema
def test_arms_are_registered_with_distinct_configs():
    for name in ("A0", "A0f", "A0v", "dev-a0", "dev-a0f", "dev-a0v"):
        assert name in ARMS
    assert len({ARMS[n].config_sha() for n in ("A0", "A0f", "A0v")}) == 3
    assert ARMS["A0"].config["max_model_calls"] == 24 and ARMS["A0v"].config["max_model_calls"] == 1


def test_every_arms_output_validates_against_the_shared_schema(tg):
    results = [
        run_a0([J("Alice", ids=[ALICE_ID]), D(done="yes")])[0],
        run_a0([Say("garbage"), Say("still garbage")])[0],  # a failed question is still a valid ArmResult
        cards_react.run(ExecInput(P, "q"), model=FakeModel([J("none", outcome="not_in_graph"), D(done="yes")]),
                        retriever=Retriever(["{}"]), lookups=card_lookups(), graph=tg),
        hybrid_once.run(ExecInput("acme", "q"), model=FakeModel([J("Alice", ids=[ALICE_ID])]),
                        retriever=Retriever(["eid_a1000001"])),
        hybrid_once.run(ExecInput("acme", "q"), model=FakeModel([Say("garbage")]), retriever=Retriever(["c"])),
    ]
    for res in results:
        _validate(res)
        assert isinstance(res.final.answer_text, str) and res.final.outcome
