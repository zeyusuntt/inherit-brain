"""The MCP server over the A5 tools, and the `inherit.ask --replay` mode. No model call anywhere."""

import asyncio
import gzip
import json

from mcp import Client

from toolfx import P, tg, tsrv  # noqa: F401  (fixtures)
from inherit import agent, ask, llm, mcp_server
from inherit.tools.catalog import TOOLS

CALL = ("bug_board", {"status": "unresolved", "rank": "max"})


def _run(coro):
    return asyncio.run(coro)


def test_tool_list_equals_the_agents_full_toolset():
    async def go():
        async with Client(mcp_server.build_server(P, graph=object())) as c:
            return (await c.list_tools()).tools

    tools = _run(go())
    want = {t["function"]["name"]: t["function"] for t in agent._Full(None, P).specs()}  # noqa: SLF001
    assert [t.name for t in tools] == list(want)
    assert set(want) == {n for n, t in TOOLS.items() if t.kind == "readonly"}
    for t in tools:
        assert t.description == want[t.name]["description"]
        assert t.input_schema == want[t.name]["parameters"]


def test_call_returns_the_direct_envelope(tg):  # noqa: F811
    direct, _ = agent._Full(tg, P).call(*CALL)  # noqa: SLF001

    async def go():
        async with Client(mcp_server.build_server(P, graph=tg)) as c:
            ok = await c.call_tool(*CALL)
            bad = await c.call_tool("no_such_tool", {})
            return ok, bad

    ok, bad = _run(go())
    assert not ok.is_error and json.loads(ok.content[0].text) == json.loads(json.dumps(direct))
    assert direct["rows"]
    assert bad.is_error


def test_list_products_resource_has_the_loaded_graphs():
    async def go():
        async with Client(mcp_server.build_server(P, graph=object())) as c:
            res = await c.read_resource(mcp_server.PRODUCTS_URI)
            return json.loads(res.contents[0].text)

    body = _run(go())
    assert body["bound"] == P and len(body["products"]) == 30 and "ActionGenie" in body["products"]


def test_replay_reexecutes_recorded_calls_without_a_model(tg, tmp_path, monkeypatch):  # noqa: F811
    def boom(*a, **k):
        raise AssertionError("a model call was made")

    monkeypatch.setattr(llm, "chat", boom)
    monkeypatch.setattr(agent, "open_graph", boom)
    run = tmp_path / "run"
    (run / "tools").mkdir(parents=True)
    rec_ops = [{"tool": CALL[0], "command": "GRAPH.RO_QUERY", "query": "q", "params": CALL[1], "row_count": 1, "ms": 99.0}]
    (run / "records.jsonl").write_text(json.dumps({"key": {"product": P, "kind": "ans", "index": 0},
                                                   "tool_ops": rec_ops}) + "\n")
    envs = [{"tool": CALL[0], "params": CALL[1], "rows": [], "truncated": False, "cursor": None}, {"agent": {}}]
    (run / "tools" / f"{P}__ans__0.json.gz").write_bytes(gzip.compress(json.dumps(envs).encode()))

    items = ask.replay(run, f"{P}/ans/0", graph=tg)
    assert len(items) == 1 and items[0]["rows"] >= 1 and items[0]["error"] is None
    fresh = items[0]["fresh"]
    assert fresh and all(f["ms"] > 0 and f["row_count"] >= 0 for f in fresh)
    assert items[0]["recorded"] == rec_ops[:1] and fresh[0]["ms"] != 99.0
    text = ask.format_replay(items)
    assert "fresh" in text and "recorded rows=1 ms=99.0" in text


def test_ask_prints_answer_trace_and_evidence(tg):  # noqa: F811
    from types import SimpleNamespace

    calls = []

    def model(messages, *, arm, tools=None, **kw):
        calls.append(arm)
        if len(calls) == 1:
            tc = SimpleNamespace(id="c1", type="function",
                                 function=SimpleNamespace(name=CALL[0], arguments=json.dumps(CALL[1])))
            msg = SimpleNamespace(content=None, tool_calls=[tc])
        else:
            msg = SimpleNamespace(content=json.dumps({"answer_text": "x", "answer_ids": [], "evidence_ids": [],
                                                      "outcome": "path"}), tool_calls=None)
        return SimpleNamespace(message=msg, prompt_tokens=10, completion_tokens=5, cost_usd=0.0, model="m",
                               latency_s=0.0)

    out = ask.ask(P, f"Which company has the most unresolved bugs in {P}?", model=model, graph=tg)
    assert "tool calls:" in out and "bug_board" in out and "evidence ids:" in out
