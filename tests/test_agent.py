"""Task 9: the tool-calling agent shared by A5, A4 and A4c, its outcomes and its in-session memory.

A scripted fake model drives the real tool layer on the Task 8 fixture graph, on both FalkorDB servers.
"""

import copy
import json
from types import SimpleNamespace

import pytest

from toolfx import ALICE, BOB, P, pr, tg, tsrv  # noqa: F401  (fixtures)
from eval.arms import ARMS
from inherit import agent
from inherit.contracts import ExecInput, QuestionKey, TraceOp
from inherit.llm import ChatResult
from inherit.memory import Finding, Session

CANON = "I could not find information in the records to answer this question."
QB = f"Which company has the most unresolved bugs in {P}?"
QP = f"Who opened {pr(1)} in {P}?"
TOOLSETS = ("full", "depth1", "depth1_chain")


# ---------------------------------------------------------------- scripted model
class T:
    """A model turn that calls tools: T(("bug_board", {...}), ...)."""

    def __init__(self, *calls, prompt=100, completion=20):
        self.calls, self.prompt, self.completion = calls, prompt, completion


class Say:
    """A model turn that answers in text (a JSON object or anything else)."""

    def __init__(self, content, prompt=100, completion=20):
        self.content, self.prompt, self.completion = content, prompt, completion


def J(text="", ids=(), evidence=(), outcome="path", **kw):
    return Say(json.dumps({"answer_text": text, "answer_ids": list(ids), "evidence_ids": list(evidence),
                           "outcome": outcome}), **kw)


class FakeModel:
    def __init__(self, steps, on_call=None):
        self.steps, self.calls, self.on_call = list(steps), [], on_call

    def __call__(self, messages, *, arm, tools=None, max_tokens=800, response_format=None, **kw):
        self.calls.append({"messages": copy.deepcopy(messages), "arm": arm, "tools": tools,
                           "max_tokens": max_tokens, "response_format": response_format})
        if self.on_call:
            self.on_call(len(self.calls))
        step = self.steps.pop(0)  # an exhausted script is a test failure (IndexError)
        if isinstance(step, T):
            tcs = [SimpleNamespace(id=f"c{len(self.calls)}_{i}", type="function",
                                   function=SimpleNamespace(name=n, arguments=json.dumps(a)))
                   for i, (n, a) in enumerate(step.calls)]
            msg = SimpleNamespace(content=None, tool_calls=tcs)
        else:
            msg = SimpleNamespace(content=step.content, tool_calls=None)
        return ChatResult(msg, step.prompt, step.completion, 0.001, 0.1)


def run(g, steps, q=QB, toolset="full", arm="t", **kw):
    m = kw.pop("model", None) or FakeModel(steps)
    res = agent.answer(ExecInput(P, q), toolset=toolset, arm=arm, graph=g, model=m, **kw)
    return res, m


def board(**kw):
    return ("bug_board", {"status": "unresolved", "rank": "max", **kw})


# ---------------------------------------------------------------- the model input
@pytest.mark.parametrize("toolset", TOOLSETS)
def test_model_input_is_system_prompt_question_and_tool_traffic_only(tg, toolset):
    call = board() if toolset == "full" else ("resolve", {"text": P})
    res, m = run(tg, [T(call), J("x")], toolset=toolset)
    assert res.final.outcome != "run_failed"
    for c in m.calls:
        msgs = c["messages"]
        assert msgs[0] == {"role": "system", "content": agent.SYSTEM_PROMPT}
        assert msgs[1] == {"role": "user", "content": QB}
        assert {x["role"] for x in msgs[2:]} <= {"assistant", "tool"}
        blob = json.dumps(msgs)
        assert str(QuestionKey(P, "ans", 3)) not in blob and '"ans"' not in blob
        assert P not in agent.SYSTEM_PROMPT
        assert c["arm"] == "t" and c["max_tokens"] == agent.MAX_OUTPUT_TOKENS == 1024


def test_the_prompt_is_the_same_for_every_toolset_and_only_the_tools_differ(tg):
    seen = {}
    for ts in TOOLSETS:
        call = board() if ts == "full" else ("resolve", {"text": P})
        _, m = run(tg, [T(call), J("x")], toolset=ts)
        seen[ts] = m.calls[0]
    assert len({json.dumps(c["messages"][:2]) for c in seen.values()}) == 1
    names = {ts: {t["function"]["name"] for t in c["tools"]} for ts, c in seen.items()}
    assert names["depth1"] == names["depth1_chain"] == {"resolve", "get_card", "neighbors"}
    assert "bug_board" in names["full"] and "if_leaves" not in names["full"]
    assert names["full"].isdisjoint(names["depth1"])
    for c in seen.values():
        assert all(t["type"] == "function" and t["function"]["parameters"]["type"] == "object" for t in c["tools"])


def test_unknown_toolset_is_refused(tg):
    with pytest.raises(ValueError):
        agent.answer(ExecInput(P, QB), toolset="bogus", arm="t", graph=tg, model=FakeModel([]))


# ---------------------------------------------------------------- outcomes
def test_unknown_person_gives_unknown_name_and_no_query_follows_the_failed_resolve(tg):
    q = f"What did Zed Nobody work on in {P}?"
    res, m = run(tg, [T(("resolve_person", {"name": "Zed Nobody"})),
                      J("I cannot find Zed Nobody in the graph.", outcome="unknown_name")], q=q)
    assert res.final.outcome == "unknown_name" and res.final.answer_ids == []
    assert [op.tool for op in res.final.trace] == ["resolve_person"]
    assert res.envelopes[0]['rows'] == []  # the tool returned no rows


def test_unknown_person_through_depth1(tg):
    q = f"What did Zed Nobody work on in {P}?"
    res, _ = run(tg, [T(("resolve", {"text": "Zed Nobody"})), J("No Zed Nobody.", outcome="unknown_name")],
                 q=q, toolset="depth1")
    assert res.final.outcome == "unknown_name"
    assert [op.tool for op in res.final.trace] == ["resolve"]


def test_all_zero_rows_gives_not_in_graph_listing_what_was_searched(tg):
    q = f"Which engineers fixed bugs reported by BlueWave in {P}?"
    text = "I searched the Customer nodes for BlueWave and the FIXES edges on its bugs: no rows."
    res, _ = run(tg, [T(("fix_trail", {"customer": "BlueWave"})), J(text, outcome="not_in_graph")], q=q)
    assert res.final.outcome == "not_in_graph" and res.final.answer_ids == []
    assert res.final.answer_text == CANON  # the model's own wording is replaced
    assert res.envelopes[-1]["agent"]["searched"].startswith("fix_trail")
    assert res.envelopes[0]['rows'] == []


def test_partial_is_kept_when_some_ids_are_supported(tg):
    res, _ = run(tg, [T(board()), J("BlueWave has the most; the assignee is not recorded.", ["BlueWave"],
                                    outcome="partial")])
    assert res.final.outcome == "partial" and res.final.answer_ids == ["BlueWave"]


def test_namesakes_are_named_and_the_ambiguous_label_is_kept(tg):
    q = f"What did Sam Kim review in {P}?"
    res, _ = run(tg, [T(("resolve_person", {"name": "Sam Kim"})),
                      J("Two people are named Sam Kim: eid_5a000007 and eid_5b000008.",
                        ["eid_5a000007", "eid_5b000008"], outcome="ambiguous_name")], q=q)
    assert res.final.outcome == "ambiguous_name"
    assert res.final.answer_ids == ["eid_5a000007", "eid_5b000008"]
    assert "eid_5a000007" in res.final.answer_text and "eid_5b000008" in res.final.answer_text


def test_a_failed_exploratory_resolve_before_a_correct_answer_stays_path(tg):
    res, _ = run(tg, [T(("resolve_person", {"name": "Nobody Atall"})), T(board()),
                      J("BlueWave.", ["BlueWave"])])
    assert res.final.outcome == "path" and res.final.answer_ids == ["BlueWave"]
    assert res.envelopes[0]['rows'] == [] and res.envelopes[1]['rows']


def test_refusal_label_with_supported_ids_becomes_path_with_the_values_up_front(tg):
    res, _ = run(tg, [T(board()), J("Nothing found.", ["BlueWave"], outcome="not_in_graph")])
    f = res.final
    assert f.outcome == "path" and f.answer_ids == ["BlueWave"]
    assert f.answer_text.startswith("BlueWave") and "Nothing found." in f.answer_text


def test_refusal_label_whose_ids_are_all_unsupported_stays_a_refusal(tg):
    res, _ = run(tg, [T(board()), J("Not there.", ["Ghost Corp"], outcome="not_in_graph")])
    assert res.final.outcome == "not_in_graph" and res.final.answer_ids == []


def test_no_text_and_no_ids_is_run_failed(tg):
    res, _ = run(tg, [T(board()), J("", [], outcome="path")])
    assert res.final.outcome == "run_failed" and res.failure


def test_content_answer_with_no_ids_is_path(tg):
    res, _ = run(tg, [T(board(**{"return": "bugs"})), J("Avatar upload fails and Dark mode flicker.")])
    assert res.final.outcome == "path" and res.final.answer_ids == []


def test_model_label_memory_or_run_failed_is_invalid_and_repaired(tg):
    res, m = run(tg, [T(board()), J("x", ["BlueWave"], outcome="memory"), J("BlueWave", ["BlueWave"])])
    assert res.final.outcome == "path" and len(m.calls) == 3


# ---------------------------------------------------------------- the guard, the same for every toolset
def _script_for(toolset):
    if toolset == "full":
        calls = [("pr_board", {"state": "merged"}), ("bug_board", {"status": "resolved", "actor": "resolver"})]
    else:
        calls = [("resolve", {"text": pr(1)}), ("get_card", {"node_id": f"PR:{pr(1)}"})]
    return [T(calls[0]), T(calls[1]),
            J(f"{ALICE} opened {pr(1)}; eid_deadbeef also did.", [pr(1), ALICE, "eid_deadbeef", f"Person:{ALICE}"])]


@pytest.mark.parametrize("toolset", TOOLSETS)
def test_the_same_supported_employee_and_pr_survive_and_unsupported_values_go_in_every_toolset(tg, toolset):
    res, _ = run(tg, _script_for(toolset), q=QP, toolset=toolset)
    f = res.final
    assert f.outcome == "path"
    assert sorted(f.answer_ids) == sorted([pr(1), ALICE])
    assert "eid_deadbeef" not in f.answer_text and ALICE in f.answer_text


def test_depth1_projection_maps_qualified_ids_to_bare_answer_values(tg):
    rows = agent.project_depth1({"rows": [{"id": f"Person:{ALICE}", "label": "Person", "name": "Alice Ng"},
                                          {"id": f"PR:{pr(1)}", "label": "PR", "name": "t"},
                                          {"id": "Url:https://x.example/a", "label": "Url", "name": "u"},
                                          {"id": "Customer:CUST-0001", "label": "Customer", "name": "TechCorp"},
                                          {"id": "Bug:b1", "label": "Bug", "name": "Login crash"}]}, {})
    answers = {r["answer"] for r in rows if "answer" in r}
    assert answers == {ALICE, pr(1), "https://x.example/a", "TechCorp"}


def test_depth1_projection_counts_card_props_neighbour_entries_and_group_values(tg):
    card = {"id": "Document:d1", "label": "Document", "name": "Doc", "props": {"link": "https://d.example/1"},
            "neighbours": [{"edge": "AUTHORED", "dir": "in", "label": "Person", "id": f"Person:{BOB}", "name": "Bob Roy",
                            "src": ["rec-1", "..."]}]}
    rows = agent.project_depth1({"rows": [card]}, {})
    assert {r["answer"] for r in rows if "answer" in r} == {"https://d.example/1", BOB}
    assert "rec-1" in {e for r in rows for e in r.get("evidence", [])} and "..." not in {
        e for r in rows for e in r.get("evidence", [])}
    g = agent.project_depth1({"rows": [], "groups": [{"value": "TechCorp", "count": 2}]}, {"group_field": "company"})
    assert {r["answer"] for r in g} == {"TechCorp"}
    assert agent.project_depth1({"rows": [], "groups": [{"value": "open", "count": 2}]}, {"group_field": "state"}) == []


def test_evidence_ids_not_returned_by_a_tool_are_dropped(tg):
    res, _ = run(tg, [T(("fix_trail", {"customer": "TechCorp"})),
                      J("PRs.", [pr(1)], evidence=["rec-b1-0", "rec-made-up"])])
    assert "rec-made-up" not in res.final.evidence_ids
    assert res.final.answer_ids == [pr(1)]


# ---------------------------------------------------------------- budgets
def test_the_13th_tool_call_is_refused_and_the_agent_then_answers_without_tools(tg):
    calls = [("resolve_person", {"name": f"Nobody {i}"}) for i in range(13)]
    res, m = run(tg, [T(*calls), J("Nothing.", outcome="not_in_graph")])
    assert len(res.final.trace) == 12 and all(op.command != "refused" for op in res.final.trace)
    tool_msgs = [x for x in m.calls[1]["messages"] if x["role"] == "tool"]
    assert len(tool_msgs) == 13 and "call limit" in tool_msgs[-1]["content"]
    assert m.calls[1]["tools"] is None and m.calls[1]["response_format"] == {"type": "json_object"}
    assert res.final.outcome == "not_in_graph"


@pytest.mark.parametrize("toolset", ["depth1", "depth1_chain"])
def test_the_13th_tool_call_is_refused_in_the_depth1_toolsets_too(tg, toolset):
    calls = [("resolve", {"text": P})] * 13
    res, m = run(tg, [T(*calls), J("Done.")], toolset=toolset)
    tool_msgs = [x for x in m.calls[1]["messages"] if x["role"] == "tool"]
    assert len(tool_msgs) == 13 and "call limit" in tool_msgs[-1]["content"]
    assert len(res.final.trace) == 12 and m.calls[1]["tools"] is None


def test_twelve_tool_turns_end_with_one_final_call_without_tools(tg):
    steps = [T(("resolve_person", {"name": f"Nobody {i}"})) for i in range(12)] + [J("None.", outcome="unknown_name")]
    res, m = run(tg, steps)
    assert len(m.calls) == 13 and m.calls[12]["tools"] is None
    assert all(c["tools"] for c in m.calls[:12])
    assert res.final.outcome == "unknown_name"


def test_at_most_14_model_calls_final_then_one_repair_then_run_failed(tg):
    steps = [T(("resolve_person", {"name": f"Nobody {i}"})) for i in range(12)] + [Say("not json"), Say("{still not")]
    res, m = run(tg, steps)
    assert len(m.calls) == 14 and res.final.outcome == "run_failed" and "invalid" in res.failure
    assert m.calls[13]["tools"] is None and m.calls[13]["response_format"] == {"type": "json_object"}
    assert res.final.usage.n_calls == 14


def test_invalid_final_json_is_repaired_once(tg):
    res, m = run(tg, [T(board()), Say("BlueWave, I think"), J("BlueWave.", ["BlueWave"])])
    assert res.final.outcome == "path" and res.final.answer_ids == ["BlueWave"]
    assert len(m.calls) == 3
    last = m.calls[2]["messages"]
    assert last[-1]["role"] == "user" and last[-2] == {"role": "assistant", "content": "BlueWave, I think"}
    assert m.calls[2]["tools"] is None and m.calls[2]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("bad", ['{"answer_text": "x"}', '{"answer_text": 1, "answer_ids": [], "evidence_ids": [], '
                                 '"outcome": "path"}', '{"answer_text": "x", "answer_ids": "a", "evidence_ids": [], '
                                 '"outcome": "path"}', '{"answer_text": "x", "answer_ids": [], "evidence_ids": [], '
                                 '"outcome": "maybe"}', "[1]"])
def test_final_json_of_the_wrong_shape_is_invalid(tg, bad):
    res, m = run(tg, [T(board()), Say(bad), Say(bad)])
    assert res.final.outcome == "run_failed" and len(m.calls) == 3


def test_input_token_budget_stops_tool_calls_and_forces_the_final_answer(tg):
    res, m = run(tg, [T(board(), prompt=45_000), J("BlueWave.", ["BlueWave"])])
    assert len(m.calls) == 2 and m.calls[1]["tools"] is None
    assert res.final.outcome == "path" and res.final.answer_ids == ["BlueWave"]
    assert res.envelopes[-1]["agent"]["budget_stop"] is True


def test_token_budget_does_not_bind_when_the_history_is_small(tg):
    res, m = run(tg, [T(board(), prompt=2_000), T(board(), prompt=2_500), J("BlueWave.", ["BlueWave"], prompt=3_000)])
    assert m.calls[2]["tools"] is not None
    assert res.envelopes[-1]["agent"]["budget_stop"] is False


def test_the_budget_estimate_uses_actual_usage_plus_the_new_tool_result(tg):
    # 59,000 used, and the next call would carry the last prompt plus the new result: over 60,000
    res, m = run(tg, [T(board(), prompt=59_000), J("BlueWave.", ["BlueWave"])])
    assert len(m.calls) == 2 and m.calls[1]["tools"] is None


# ---------------------------------------------------------------- deadline
def test_deadline_is_checked_before_every_call_in_the_loop(tg):
    now = [0.0]
    model = FakeModel([T(board()), T(board()), J("late")], on_call=lambda n: now.__setitem__(0, now[0] + 200.0))
    res, m = run(tg, None, model=model, clock=lambda: now[0])
    assert res.final.outcome == "run_failed" and "deadline" in res.failure
    assert len(m.calls) == 2 and len(res.final.trace) == 1  # call 2 ran at 200 s, its tool call at 400 s did not


def test_no_call_starts_after_the_deadline(tg):
    now = [0.0]
    model = FakeModel([T(board()), J("late")], on_call=lambda n: now.__setitem__(0, 301.0))
    res, m = run(tg, None, model=model, clock=lambda: now[0])
    assert res.final.outcome == "run_failed" and "deadline" in res.failure
    assert len(m.calls) == 1 and res.final.trace == []  # the tool call after expiry never ran
    assert res.final.usage.n_calls == 1


# ---------------------------------------------------------------- result shape for the runner
def test_arm_result_carries_calls_envelopes_and_usage(tg):
    res, m = run(tg, [T(board()), J("BlueWave.", ["BlueWave"], evidence=[])])
    assert len(res.calls) == 2 and all(c.arm == "t" for c in res.calls)
    assert res.final.usage.prompt_tokens == 200 and res.final.usage.completion_tokens == 40
    assert res.final.usage.n_calls == 2 and res.final.usage.cost_usd == pytest.approx(0.002)
    env = res.envelopes[0]
    assert env["tool"] == "bug_board" and env["params"]["status"] == "unresolved" and env["rows"]
    diag = res.envelopes[-1]["agent"]
    assert set(diag) >= {"model_outcome", "budget_stop", "repaired", "tool_calls", "model_calls", "dropped"}
    assert diag["model_outcome"] == "path" and diag["repaired"] is False
    assert all(isinstance(op, TraceOp) for op in res.final.trace)


def test_a_tool_error_is_shown_to_the_model_and_the_loop_goes_on(tg):
    res, m = run(tg, [T(("bug_board", {"status": "nonsense"})), T(board()), J("BlueWave.", ["BlueWave"])])
    err = [x for x in m.calls[1]["messages"] if x["role"] == "tool"][0]["content"]
    assert "error" in err.lower()
    assert res.final.outcome == "path"
    assert res.final.trace[0].command == "refused"


def test_bad_tool_arguments_and_unknown_tools_do_not_crash_the_loop(tg):
    m = FakeModel([T(("nope", {}), ("bug_board", {"status": 5})), T(board()), J("BlueWave.", ["BlueWave"])])
    res, _ = run(tg, None, model=m)
    msgs = [x for x in m.calls[1]["messages"] if x["role"] == "tool"]
    assert len(msgs) == 2 and all("error" in x["content"].lower() for x in msgs)
    assert res.final.outcome == "path"


def test_a_model_or_tool_exception_becomes_run_failed_with_the_calls_so_far(tg):
    def boom(n):
        if n == 2:
            raise RuntimeError("proxy down")

    res, m = run(tg, None, model=FakeModel([T(board()), J("x")], on_call=boom))
    assert res.final.outcome == "run_failed" and "RuntimeError" in res.failure and "proxy down" in res.failure
    assert len(res.calls) == 1 and len(res.final.trace) >= 1


def test_budget_exceeded_returns_the_failed_result_with_the_usage_collected(tg):
    from inherit.llm import BudgetExceeded

    def boom(n):
        if n == 2:
            raise BudgetExceeded("cap")

    res, _ = run(tg, None, model=FakeModel([T(board()), J("x")], on_call=boom))
    assert res.final.outcome == "run_failed" and res.failure == "budget_exceeded"
    assert len(res.calls) == 1 and res.envelopes[0]["tool"] == "bug_board" and len(res.final.trace) >= 1


# ---------------------------------------------------------------- the same loop with each toolset
def test_depth1_strict_refuses_a_card_that_is_not_a_root_and_chain_allows_it(tg):
    q = f"Who opened {pr(1)} in {P}?"
    steps = lambda: [T(("resolve", {"text": pr(1)})), T(("get_card", {"node_id": f"PR:{pr(1)}"})),
                     T(("get_card", {"node_id": f"Person:{ALICE}"})), J(f"{ALICE}.", [ALICE])]
    strict, ms = run(tg, steps(), q=q, toolset="depth1")
    chain, mc = run(tg, steps(), q=q, toolset="depth1_chain")
    last_strict = [x for x in ms.calls[3]["messages"] if x["role"] == "tool"][-1]["content"]
    last_chain = [x for x in mc.calls[3]["messages"] if x["role"] == "tool"][-1]["content"]
    assert "not an authorised root" in last_strict
    assert '"Person:' in last_chain and "not an authorised root" not in last_chain
    assert strict.final.answer_ids == [ALICE]  # the neighbour entry of the PR card carried her
    assert chain.final.answer_ids == [ALICE]


# ---------------------------------------------------------------- memory
def _first(tg):
    res, _ = run(tg, [T(board()), J("BlueWave.", ["BlueWave"])])
    return res


def test_a_confirmed_finding_is_returned_as_memory_in_the_same_session_and_not_in_a_new_one(tg):
    res = _first(tg)
    inp = ExecInput(P, QB)
    s = Session()
    s.confirm(Finding.from_final(inp, res.final, confirmed_by="zeyu", confirmed_at="2026-10-16T10:00:00Z"))
    again, m = run(tg, [], session=s)
    f = again.final
    assert f.outcome == "memory" and f.answer_ids == ["BlueWave"] and m.calls == []
    assert "2026-10-16T10:00:00Z" in f.answer_text and "memory" in f.answer_text.lower() and "zeyu" in f.answer_text
    assert f.trace == res.final.trace and f.evidence_ids == res.final.evidence_ids  # the original path
    assert f.usage.n_calls == 0 and again.calls == []
    fresh, m2 = run(tg, [T(board()), J("BlueWave.", ["BlueWave"])], session=Session())
    assert fresh.final.outcome == "path" and len(m2.calls) == 2


def test_memory_matches_the_normalised_question_text_only_and_the_same_product(tg):
    res = _first(tg)
    s = Session()
    s.confirm(Finding.from_final(ExecInput(P, QB), res.final, confirmed_by="zeyu", confirmed_at="t"))
    shouting = "  WHICH company  has the most unresolved bugs in %s?" % P
    assert run(tg, [], q=shouting, session=s)[0].final.outcome == "memory"
    other_q = f"Which company has the most resolved bugs in {P}?"
    other, m = run(tg, [T(board(status="resolved")), J("TechCorp.", ["TechCorp"])], q=other_q, session=s)
    assert other.final.outcome == "path" and len(m.calls) == 2
    s2 = Session()
    s2.confirm(Finding(product="other", question=QB, text="x", answer_ids=["x"], evidence_ids=[], trace=[],
                       confirmed_at="t", confirmed_by="z"))
    assert run(tg, [T(board()), J("BlueWave.", ["BlueWave"])], session=s2)[0].final.outcome == "path"


def test_session_find_rules():
    f = Finding(product="p", question="Who is the best?", text="Ann", answer_ids=["e1"], evidence_ids=["r1"], trace=[],
                confirmed_at="t", confirmed_by="z", roots=("Person:e1",), predicate="best")
    s = Session()
    assert s.find("p", "who is the best?") is None
    s.confirm(f)
    assert s.find("p", "  WHO is  the best? ") is f
    assert s.find("q", "who is the best?") is None
    assert s.find("p", "who is the worst?") is None
    assert s.find("p", "a different wording", predicate="best", roots=("Person:e1",)) is f
    assert s.find("p", "a different wording", predicate="best", roots=("Person:e2",)) is None
    assert s.find("p", "a different wording", predicate="worst", roots=("Person:e1",)) is None
    assert s.find("p", "a different wording", roots=("Person:e1",)) is None
    assert Session().find("p", "who is the best?") is None


def test_session_keeps_everything_in_process():
    import inspect

    import inherit.memory as mem

    src = inspect.getsource(mem)
    assert "open(" not in src and "write_text" not in src and "falkor" not in src.lower()


# ---------------------------------------------------------------- graph handle and arm registration
def test_open_graph_sets_server_version_after_agent_graph(monkeypatch):
    seen = {}

    def fake(product, server="6.0"):
        seen["args"] = (product, server)
        return SimpleNamespace(name=f"herb:{product}")

    monkeypatch.setattr(agent.depth1, "agent_graph", fake)
    g = agent.open_graph("ActionGenie")
    assert seen["args"] == ("ActionGenie", agent.SERVER) and g.server_version == agent.SERVER
    assert agent.SERVER.startswith("6")


def test_the_six_arms_are_registered_with_identical_budgets_and_prompt(monkeypatch):
    names = {"A5": "full", "A4": "depth1", "A4c": "depth1_chain",
             "dev-a5": "full", "dev-a4": "depth1", "dev-a4c": "depth1_chain"}
    assert set(names) <= set(ARMS)
    cfgs = {n: {k: v for k, v in ARMS[n].config.items() if k != "toolset"} for n in names}
    assert all(c == cfgs["A5"] for c in cfgs.values()) and cfgs["A5"]["max_tool_calls"] == 12
    assert {ARMS[n].config["toolset"] for n in names} == set(names.values())
    assert len({json.dumps(ARMS[n].prompt_versions, sort_keys=True) for n in names}) == 1
    assert ARMS["A5"].prompt_versions == {"agent": agent.PROMPT_VERSION}
    got = {}
    monkeypatch.setattr(agent, "answer", lambda inp, **kw: got.update(inp=inp, **kw) or "r")
    for n, ts in names.items():
        assert ARMS[n].run(ExecInput("p", "q")) == "r"
        assert got["toolset"] == ts and got["arm"] == n


def test_question_tokens_are_not_support_for_text_or_evidence(tg):
    q = f"What did eid_deadbeef work on in {P}?"
    res, _ = run(tg, [J("eid_deadbeef worked on nothing.", [], evidence=["eid_deadbeef"], outcome="path")], q=q)
    assert "eid_deadbeef" not in res.final.answer_text and res.final.evidence_ids == []


@pytest.mark.parametrize("toolset", TOOLSETS)
def test_qualified_employee_and_link_ids_are_graded_as_bare_values_in_every_toolset(tg, toolset):
    steps = _script_for(toolset)[:2] + [J("Done.", [f"PR:{pr(1)}", f"Person:{ALICE}", pr(1)])]
    res, _ = run(tg, steps, q=QP, toolset=toolset)
    assert res.final.answer_ids == [pr(1), ALICE]


def test_the_depth1_resolve_text_says_where_the_product_node_is(tg):
    specs = {s["function"]["name"]: s["function"]["description"] for s in agent._Depth1(tg, P, QB, True).specs()}
    assert "Product:" in specs["resolve"]


# ---------------------------------------------------------------- fix round 1
def _resolution_script(toolset):
    if toolset == "full":
        calls = [("resolve_entity", {"text": "TechCorp"}), ("resolve_entity", {"text": f"{pr(1)}"})]
    else:
        calls = [("resolve", {"text": "TechCorp"}), ("resolve", {"text": pr(1)})]
    return [T(calls[0]), T(calls[1]), J("TechCorp and the PR.", ["TechCorp", pr(1), f"Customer:CUST-0001"])]


@pytest.mark.parametrize("toolset", TOOLSETS)
def test_resolution_alone_supports_the_same_company_and_link_in_every_toolset(tg, toolset):
    res, _ = run(tg, _resolution_script(toolset), q=f"Tell me about TechCorp and {pr(1)} in {P}", toolset=toolset)
    assert res.final.answer_ids == ["TechCorp", pr(1)], toolset


@pytest.mark.parametrize("toolset", TOOLSETS)
def test_twelve_invalid_calls_use_up_the_budget_in_every_toolset(tg, toolset):
    bad = [("nope", {})] * 6 + [("resolve_person" if toolset == "full" else "resolve", {})] * 6
    good = board() if toolset == "full" else ("resolve", {"text": P})
    res, m = run(tg, [T(*bad), T(good), J("x", outcome="not_in_graph")], toolset=toolset)
    last = [x for x in m.calls[2]["messages"] if x["role"] == "tool"][-1]["content"]
    assert "call limit" in last and m.calls[2]["tools"] is None
    assert all(op.command == "refused" for op in res.final.trace) and len(res.final.trace) <= 12
    assert res.envelopes[-1]["agent"]["tool_calls"] == 12


def test_a_final_response_arriving_after_the_deadline_is_not_accepted(tg):
    now = [0.0]
    model = FakeModel([J("BlueWave.", ["BlueWave"])], on_call=lambda n: now.__setitem__(0, 301.0))
    res, m = run(tg, None, model=model, clock=lambda: now[0])
    assert res.final.outcome == "run_failed" and "deadline" in res.failure
    assert res.final.usage.n_calls == 1 and len(res.calls) == 1


def test_a_repair_response_arriving_after_the_deadline_is_not_accepted(tg):
    now = [0.0]
    model = FakeModel([Say("junk"), J("BlueWave.", ["BlueWave"])], on_call=lambda n: now.__setitem__(0, 301.0 * (n - 1)))
    res, m = run(tg, None, model=model, clock=lambda: now[0])
    assert res.final.outcome == "run_failed" and "deadline" in res.failure and len(res.calls) == 2


@pytest.mark.parametrize("bad", ['{"answer_text": "x", "answer_ids": [], "evidence_ids": [], "outcome": []}',
                                 '{"answer_text": "x", "answer_ids": [], "evidence_ids": [], "outcome": {}}'])
def test_unhashable_outcome_gets_the_single_repair(tg, bad):
    res, m = run(tg, [T(board()), Say(bad), J("BlueWave.", ["BlueWave"])])
    assert res.final.outcome == "path" and len(m.calls) == 3


@pytest.mark.parametrize("comp", [P, f"Product:{P}", P.upper()])
def test_bug_board_ignores_the_product_name_as_a_component(tg, comp):
    from inherit.tools.catalog import TOOLS

    base = TOOLS["bug_board"].run(tg, P, status="unresolved", rank="max")
    got = TOOLS["bug_board"].run(tg, P, status="unresolved", rank="max", component=comp)
    assert got.rows and [r["answer"] for r in got.rows] == [r["answer"] for r in base.rows]
    real = TOOLS["bug_board"].run(tg, P, status="unresolved", component="Authentication")
    assert real.rows is not None


def test_the_prompt_says_the_graph_is_already_scoped_to_the_product():
    assert "already in scope" in agent.SYSTEM_PROMPT and "component" in agent.SYSTEM_PROMPT


@pytest.mark.parametrize("toolset", TOOLSETS)
def test_refusals_use_one_canonical_text_in_every_toolset(tg, toolset):
    call = ("resolve_person", {"name": "Zed Nobody"}) if toolset == "full" else ("resolve", {"text": "Zed Nobody"})
    res, _ = run(tg, [T(call), J("No one named Zed Nobody ever worked on anything.", outcome="unknown_name")],
                 q=f"What did Zed Nobody work on in {P}?", toolset=toolset)
    t = res.final.answer_text
    assert res.final.outcome == "unknown_name" and t == CANON
    assert "No one named" not in t and call[0] in res.envelopes[-1]["agent"]["searched"]


def test_a_refusal_with_no_tool_results_still_gets_the_template(tg):
    res, _ = run(tg, [J("Nothing.", outcome="not_in_graph")] * 3)
    assert res.final.answer_text == CANON
    assert res.envelopes[-1]["agent"]["searched"] == "no tool returned results"


def test_the_template_names_the_node_types_searched(tg):
    res, _ = run(tg, [T(("resolve_entity", {"text": "TechCorp"})), J("x", outcome="not_in_graph"),
                      J("x", outcome="not_in_graph")])
    assert "resolve_entity (node types: Customer)" in res.envelopes[-1]["agent"]["searched"]


def test_ambiguous_and_partial_keep_the_models_own_text(tg):
    res, _ = run(tg, [T(board()), J("BlueWave has the most; the rest is missing.", ["BlueWave"], outcome="partial")])
    assert res.final.answer_text.startswith("BlueWave has the most")


def test_the_common_prompt_has_the_refusal_and_list_rules():
    p = agent.SYSTEM_PROMPT
    assert "do not state a negative fact" in p and "not_in_graph" in p
    assert "tied at the top, answer_ids lists every tied row" in p and "detail fields" in p


# ---------------------------------------------------------------- refusal needs a relation query (SPEC 5.3)
def _resolver_only(toolset):
    return ("resolve_entity", {"text": "TechCorp"}) if toolset == "full" else ("resolve", {"text": "TechCorp"})


def _relation(toolset):
    if toolset == "full":
        return board()
    return ("neighbors", {"root_id": "Customer:CUST-0001", "edge_type": "REPORTED_BY", "direction": "in"})


@pytest.mark.parametrize("toolset", TOOLSETS)
def test_a_refusal_after_resolver_calls_only_gets_one_reminder_and_the_loop_continues(tg, toolset):
    q = f"Which bugs did TechCorp report in {P}?"
    steps = [T(_resolver_only(toolset)), J("Nothing.", outcome="not_in_graph"), T(_relation(toolset)),
             J("BlueWave.", ["BlueWave"] if toolset == "full" else [])]
    res, m = run(tg, steps, q=q, toolset=toolset)
    assert len(m.calls) == 4 and res.final.outcome == "path"
    assert m.calls[2]["messages"][-1] == {"role": "user", "content": agent.REMINDER}
    assert m.calls[2]["messages"][-2]["role"] == "assistant" and m.calls[2]["tools"]
    assert res.envelopes[-1]["agent"]["reminded"] is True


@pytest.mark.parametrize("toolset", TOOLSETS)
def test_at_most_two_reminders_then_the_refusal_is_accepted(tg, toolset):
    q = f"Which bugs did TechCorp report in {P}?"
    res, m = run(tg, [T(_resolver_only(toolset))] + [J("x", outcome="not_in_graph")] * 3, q=q, toolset=toolset)
    assert len(m.calls) == 4 and res.final.outcome == "not_in_graph"
    assert res.envelopes[-1]["agent"]["reminders"] == 2


@pytest.mark.parametrize("toolset", TOOLSETS)
def test_a_failed_resolution_allows_the_refusal_without_a_reminder(tg, toolset):
    call = ("resolve_person", {"name": "Zed Nobody"}) if toolset == "full" else ("resolve", {"text": "Zed Nobody"})
    res, m = run(tg, [T(call), J("x", outcome="unknown_name")], q=f"What did Zed Nobody do in {P}?", toolset=toolset)
    assert len(m.calls) == 2 and res.final.outcome == "unknown_name"


@pytest.mark.parametrize("toolset", TOOLSETS)
def test_a_relation_query_with_zero_rows_allows_the_refusal(tg, toolset):
    if toolset == "full":
        call = ("fix_trail", {"customer": "BlueWave"})
    else:
        call = ("neighbors", {"root_id": "Customer:CUST-0003", "edge_type": "FIXES", "direction": "in"})
    steps = [T(("resolve", {"text": "BlueWave"})), T(call), J("x", outcome="not_in_graph")] if toolset != "full" else \
        [T(call), J("x", outcome="not_in_graph")]
    res, m = run(tg, steps, q=f"Which PRs fixed BlueWave bugs in {P}?", toolset=toolset)
    assert res.final.outcome == "not_in_graph" and len(m.calls) == len(steps) + 0


def test_a_refusal_with_no_tool_call_at_all_gets_the_reminder(tg):
    res, m = run(tg, [J("x", outcome="not_in_graph"), T(board()), J("BlueWave.", ["BlueWave"])])
    assert res.final.outcome == "path" and len(m.calls) == 3


def test_the_v3_prompt_rules():
    p = agent.SYSTEM_PROMPT
    for frag in ("never a filter and never an entity to resolve", "Add a filter only when the question states it",
                 "full phrase from the question", "all statuses", "do not replace them with values from unrelated"):
        assert frag in p


def test_a_tool_note_is_shown_to_the_model_and_kept_in_the_envelope(tg, monkeypatch):
    from inherit.tools import catalog

    real = catalog.TOOLS["bug_board"]

    def with_note(conn, product, **params):
        r = real.run(conn, product, **params)
        return catalog.ToolResult(r.rows, r.truncated, r.cursor, r.ops, r.path, r.ms, "Try the full phrase.")

    wrapped = SimpleNamespace(name="bug_board", kind="readonly", description="d",
                              params_schema=real.params_schema, run=with_note)
    monkeypatch.setitem(agent.TOOLS, "bug_board", wrapped)
    res, m = run(tg, [T(board()), J("BlueWave.", ["BlueWave"])])
    tool_msg = [x for x in m.calls[1]["messages"] if x["role"] == "tool"][0]["content"]
    assert json.loads(tool_msg)["note"] == "Try the full phrase."
    assert res.envelopes[0]["note"] == "Try the full phrase."


def test_no_note_key_when_the_tool_gave_none(tg):
    res, m = run(tg, [T(board()), J("BlueWave.", ["BlueWave"])])
    assert "note" not in json.loads([x for x in m.calls[1]["messages"] if x["role"] == "tool"][0]["content"])


# ---------------------------------------------------------------- hardened refusal rule, ties, shared consistency
@pytest.mark.parametrize("toolset", TOOLSETS)
def test_a_resolver_that_errors_or_looks_up_the_product_does_not_license_a_refusal(tg, toolset):
    bad = ("resolve_entity", {"text": "X", "kinds": ["nonsense"]}) if toolset == "full" else ("resolve", {})
    prod = ("resolve_entity", {"text": P}) if toolset == "full" else ("resolve", {"text": P})
    res, m = run(tg, [T(bad), T(prod), J("x", outcome="not_in_graph"), T(_relation(toolset)),
                      J("BlueWave.", ["BlueWave"] if toolset == "full" else [])], toolset=toolset)
    assert len(m.calls) == 5 and res.final.outcome == "path"
    assert m.calls[3]["messages"][-1] == {"role": "user", "content": agent.REMINDER}


def test_a_refusal_produced_inside_the_json_repair_is_rechecked(tg):
    res, m = run(tg, [T(("resolve_entity", {"text": "TechCorp"})), Say("not json"),
                      J("x", outcome="not_in_graph"), T(board()), J("BlueWave.", ["BlueWave"])])
    assert res.final.outcome == "path" and len(m.calls) == 5
    assert m.calls[3]["messages"][-1] == {"role": "user", "content": agent.REMINDER}
    assert res.envelopes[-1]["agent"]["repaired"] is True


def test_ties_of_one_rank_max_call_are_completed_and_other_calls_are_not_copied(tg):
    tied = ("bug_board", {"status": "resolved", "actor": "fix_pr_author", "rank": "max"})
    other = ("bug_board", {"status": "any", "actor": "company", "rank": "all"})
    res, _ = run(tg, [T(tied), T(other), J("Alice.", [ALICE])])
    assert res.final.answer_ids == [ALICE, BOB]
    assert res.envelopes[-1]["agent"]["tie_completed"] == [BOB]
    assert "TechCorp" not in res.final.answer_ids  # values of the other call are never copied


def test_no_tie_completion_without_an_answer_value_from_the_top_tie(tg):
    res, _ = run(tg, [T(board()), J("x", [], outcome="partial")])
    assert res.final.answer_ids == [] and res.envelopes[-1]["agent"]["tie_completed"] == []


def test_apply_consistency_is_the_one_rule():
    assert agent.apply_consistency("not_in_graph", "No X.", []) == ("not_in_graph", agent.REFUSAL_SENTENCE, None)
    assert agent.apply_consistency("unknown_name", "", []) == ("unknown_name", agent.REFUSAL_SENTENCE, None)
    assert agent.apply_consistency("not_in_graph", "n", ["a", "b"]) == ("path", "a, b. n", None)
    assert agent.apply_consistency("partial", "", [])[2] and agent.apply_consistency("path", " ", [])[2]
    assert agent.apply_consistency("partial", "p", ["a"]) == ("partial", "p", None)
    assert agent.REFUSAL_SENTENCE == "I could not find information in the records to answer this question."


def test_every_arm_that_uses_the_shared_format_shares_the_helper():
    import inspect

    from baselines import hybrid_once, long_context

    for mod in (hybrid_once, long_context):
        assert "apply_consistency" in inspect.getsource(mod) and "refusal_text" not in inspect.getsource(mod)


def test_every_arm_carries_the_identical_answer_instructions_and_no_second_copy():
    import hashlib
    import inspect
    import re

    from baselines import graphrag_sdk_arm, hybrid_once, long_context, paper_tools

    want = hashlib.sha256(agent.ANSWER_INSTRUCTIONS.encode()).hexdigest()
    prompts = {"agent": agent.SYSTEM_PROMPT, "a0v": hybrid_once.SYSTEM, "a1f": long_context.SYSTEM,
               "a2": graphrag_sdk_arm.ANSWER_SYSTEM, "paper_fallback": paper_tools.FALLBACK_PROMPT.format(
                   context="c", question="q", format=paper_tools.ANSWER_FORMAT)}
    for fn in [f for _, f in inspect.getmembers(paper_tools, inspect.isfunction) if "cards" in inspect.signature(f).parameters]:
        try:
            prompts["paper_" + fn.__name__] = fn(cards=False)
        except TypeError:
            pass
    assert len(prompts) >= 5
    for name, text in prompts.items():
        assert text.count(agent.ANSWER_INSTRUCTIONS) == 1, name
        i = text.index(agent.ANSWER_INSTRUCTIONS)
        assert hashlib.sha256(text[i:i + len(agent.ANSWER_INSTRUCTIONS)].encode()).hexdigest() == want
    marker = "answer_ids: the answer values themselves"
    for mod in (agent, graphrag_sdk_arm, hybrid_once, long_context, paper_tools):
        assert inspect.getsource(mod).count(marker) == (1 if mod is agent else 0), mod.__name__
    assert re.search(r"CUST-", agent.ANSWER_INSTRUCTIONS)


# ---------------------------------------------------------------- agent/v5: answer format, parser, schema guidance
V5_HEAD = '{"answer_text": "The answer.", "answer_ids": ["a1"], "outcome": "path", "evidence_ids": '


def test_the_v5_answer_format_is_ordered_and_capped():
    t = agent.ANSWER_INSTRUCTIONS
    assert '{"answer_text": "...", "answer_ids": ["..."], "outcome": "...", "evidence_ids": ["..."]}' in t
    assert "at most 10 distinct record ids" in t
    assert t.index("- outcome, one of") < t.index("- evidence_ids")
    assert agent.PROMPT_VERSION == "agent/v5"


def test_valid_replies_are_unchanged_and_evidence_is_deduplicated_and_capped():
    ev = [f"r{i}" for i in range(12)] + ["r0", "r1"]
    f, err = agent._parse(V5_HEAD + json.dumps(ev) + "}")
    assert err == "" and f["evidence_ids"] == ev[:10] and "salvaged_evidence" not in f
    g, _ = agent._parse(V5_HEAD + '["x", "x", "y"]}')
    assert g["evidence_ids"] == ["x", "y"] and g["answer_ids"] == ["a1"]


def test_an_unfinished_final_evidence_list_is_salvaged_without_the_cut_item():
    # the A1f failure shape: repetition cut mid-item at the token cap
    cut = V5_HEAD + '["20260611-3-5f298", "20260611-3-5f299", "20260611-3-5f298", "20260611-3-5f2'
    f, err = agent._parse(cut)
    assert f is not None and f["evidence_ids"] == ["20260611-3-5f298", "20260611-3-5f299"]
    assert f["salvaged_evidence"] is True and f["answer_ids"] == ["a1"] and f["outcome"] == "path"
    f2, _ = agent._parse(V5_HEAD + '["r1", "r2", ')
    assert f2["evidence_ids"] == ["r1", "r2"] and f2["salvaged_evidence"] is True
    f3, _ = agent._parse(V5_HEAD + "[")
    assert f3["evidence_ids"] == [] and f3["salvaged_evidence"] is True
    f4, _ = agent._parse("```json\n" + V5_HEAD + '["r1", "r')
    assert f4["evidence_ids"] == ["r1"]


@pytest.mark.parametrize("bad", [
    '{"answer_text": "The answer.", "answer_ids": ["a',                            # cut inside the first id
    '{"answer_text": "The ans',                                                     # cut inside answer_text
    '{"answer_text": "x", "answer_ids": ["a1"], "evidence_ids": ["r1", "r',          # outcome missing (old order)
    '{"answer_text": "x", "answer_ids": ["a1"], "outcome": "bogus", "evidence_ids": ["r1", "r',
    V5_HEAD + '["r1"]',                                                             # list closed, object not
    V5_HEAD + '["r1"]} trailing',
    V5_HEAD + '["r1" "r2"]}',                                                       # missing separator, closed
    V5_HEAD + '["r1", "r2"] extra',
    V5_HEAD + '["r1", 5, "r',                                                       # malformed item before the cut
    V5_HEAD + '["r1", ]}',
    V5_HEAD + '["r1", "r2"}',                                                       # wrong closer
    V5_HEAD + '["bad\\q", "r2", "r',
    '{"answer_text": "x", "answer_ids": ["a1" "a2"], "outcome": "path", "evidence_ids": ["r1", "r',  # broken earlier
])
def test_nothing_else_is_salvaged(bad):
    f, err = agent._parse(bad)
    assert f is None and err


def test_the_loop_records_the_salvage_in_its_envelope(tg):
    res, _ = run(tg, [T(board()), Say(V5_HEAD.replace("a1", "BlueWave") + '["rec-b3-0", "rec-b')])
    assert res.final.answer_ids == ["BlueWave"] and res.envelopes[-1]["agent"]["salvaged_evidence"] is True
    ok, _ = run(tg, [T(board()), J("BlueWave.", ["BlueWave"])])
    assert ok.envelopes[-1]["agent"]["salvaged_evidence"] is False


def test_every_arm_parses_through_the_shared_parser_and_records_the_salvage():
    import inspect

    from baselines import graphrag_sdk_arm, hybrid_once, long_context, paper_tools

    for mod in (hybrid_once, long_context, graphrag_sdk_arm):
        src = inspect.getsource(mod)
        assert "agent._parse" in src and "salvaged_evidence" in src, mod.__name__
    assert "_parse" in inspect.getsource(paper_tools)  # the fallback answer uses the same parser


def test_the_shared_loop_budget_is_60k():
    assert agent.INPUT_TOKEN_BUDGET == 60_000
    assert agent.arm_config("full")["input_token_budget"] == 60_000


def test_versions_are_bumped():
    from baselines import graphrag_sdk_arm, long_context, paper_tools

    assert (paper_tools.PROMPT_VERSION, long_context.PROMPT_VERSION, graphrag_sdk_arm.PROMPT_VERSION) == (
        "baselines/a0/v3", "baselines/a1f/v3", "a2/v4")


def test_the_depth1_schema_guidance_is_generated_and_identical_for_both_toolsets(tg):
    from inherit.graph import cards
    from inherit.ontology import SIGNATURES

    texts = {}
    for strict in (True, False):
        specs = {s["function"]["name"]: s["function"]["description"] for s in agent._Depth1(tg, P, QB, strict).specs()}
        texts[strict] = specs["neighbors"]
    guide = agent.schema_guidance()
    assert guide in texts[True] and guide in texts[False]
    for label in cards.ALLOWLIST:
        assert label in guide
    for edge, pairs in SIGNATURES.items():
        if edge in cards.EXCLUDED_EDGES:
            assert edge not in guide
            continue
        line = next(l for l in guide.splitlines() if l.startswith(f"{edge}:"))
        for src, dst in pairs:
            assert src in line and dst in line
    assert "HAS: Product -> " in guide and "PR" in next(l for l in guide.splitlines() if l.startswith("HAS:"))
    assert "PullRequest" not in guide


def test_a_decoding_error_in_salvage_is_ordinary_invalid_json_never_an_exception():
    for t in (V5_HEAD + '["bad\\q", "r2", "r',):
        f, err = agent._parse(t)
        assert f is None and err == "not valid JSON"


def test_salvage_accepts_complete_items_with_valid_separators_only_when_truncated():
    f, _ = agent._parse(V5_HEAD + '["r1", "r2", "r3')
    assert f["evidence_ids"] == ["r1", "r2"]
    f, _ = agent._parse(V5_HEAD + '["r1",\n "r2",')
    assert f["evidence_ids"] == ["r1", "r2"] and f["salvaged_evidence"]


# ---------------------------------------------------------------- truncation inside answer_ids (repetition at the cap)
AI_HEAD = '{"answer_text": "The answer.", "answer_ids": ['


def test_a_repeating_cycle_cut_mid_item_inside_answer_ids_is_salvaged_as_partial():
    cycle = ", ".join(['"eid_aaaaaaaa"', '"eid_bbbbbbbb"', '"eid_cccccccc"'] * 40)
    f, err = agent._parse(AI_HEAD + cycle + ', "eid_aaaa')
    assert err == "" and f["answer_ids"] == ["eid_aaaaaaaa", "eid_bbbbbbbb", "eid_cccccccc"]
    assert f["outcome"] == "partial" and f["evidence_ids"] == [] and f["answer_text"] == "The answer."
    assert f["salvaged_answer_ids"] is True and "salvaged_evidence" not in f
    assert agent.apply_consistency(f["outcome"], f["answer_text"], f["answer_ids"]) == (
        "partial", "The answer.", None)


def test_a_cut_at_an_item_boundary_inside_answer_ids_is_salvaged_too():
    f, _ = agent._parse(AI_HEAD + '"a1", "a2", ')
    assert f["answer_ids"] == ["a1", "a2"] and f["salvaged_answer_ids"] is True


@pytest.mark.parametrize("bad", [
    AI_HEAD + '"eid_aaaa',                                  # cut inside the first id: nothing complete
    AI_HEAD,                                                # empty list so far
    '{"answer_text": "The ans',                             # cut inside answer_text
    '{"answer_text": "x", "answer_ids": ["a1"], "outcome": "pa',   # cut after answer_ids closed
    AI_HEAD + '"a1" "a2", "a',                              # missing separator
    '{"answer_text": 5, "answer_ids": ["a1", "a',           # answer_text invalid
    '{"outcome": "path", "answer_text": "x", "answer_ids": ["a1", "a',   # other fields before the list
])
def test_other_truncations_stay_invalid(bad):
    f, err = agent._parse(bad)
    assert f is None and err


def test_the_evidence_case_is_unchanged_by_the_answer_ids_rule():
    f, _ = agent._parse(V5_HEAD + '["r1", "r2", "r')
    assert f["salvaged_evidence"] is True and "salvaged_answer_ids" not in f and f["outcome"] == "path"


def test_the_loop_records_the_answer_ids_salvage(tg):
    cut = '{"answer_text": "x", "answer_ids": ["BlueWave", "BlueWave", "Blue'
    res, _ = run(tg, [T(board()), Say(cut)])
    assert res.final.answer_ids == ["BlueWave"] and res.final.outcome == "partial"
    assert res.envelopes[-1]["agent"]["salvaged_answer_ids"] is True
    assert res.envelopes[-1]["agent"]["salvaged_evidence"] is False
