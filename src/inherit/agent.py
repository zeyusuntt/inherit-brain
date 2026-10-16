"""The tool-calling agent shared by A5 (toolset "full"), A4 ("depth1", strict) and A4c ("depth1_chain").

One loop, one system prompt, one set of budgets and one answer handler; only the tool list differs, so a difference
between the arms is a difference between their tools. The model sees the system prompt, the question text and tool
traffic, nothing else (the product only selects the graph connection).

Budgets, identical for every toolset: at most 12 tool calls (enforced by the tool layer; the 13th is refused), at most
14 model calls (12 tool turns, one final-answer call, one JSON repair), at most 1,024 output tokens per call, a
cumulative input budget of 40,000 prompt tokens (actual usage so far plus an estimate of the next call: the previous
prompt and completion plus the new tool-result characters / 2; when that would exceed the budget the agent stops
calling tools and makes the final-answer call with its history as is), and a cooperative 300 s deadline checked before
every model and tool call.

After the model's final JSON: the output guard removes unsupported answer values and unsupported ids from the text
(`project_depth1` is the one canonical answer-value projection for the depth-1 cards), then one consistency rule fixes
the outcome label for every arm: refusal label with supported answer ids becomes `path`; no ids and no text becomes
`run_failed`. The trace never decides the outcome.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from inherit import llm
from inherit.config import load_settings
from inherit.contracts import CallUsage, ExecInput, FinalAnswer, TraceOp, Usage
from inherit.graph import cards
from inherit.llm import BudgetExceeded
from inherit.memory import Finding, Session
from inherit.ontology import SIGNATURES
from inherit.tools import depth1
from inherit.tools.catalog import TOOLS
from inherit.tools.depth1 import CALL_CAP, Episode, ToolError
from inherit.tools.guard import guard, guard_evidence
from inherit.tools.trace import Trace

PROMPT_VERSION = "agent/v1"
TOOLSETS = ("full", "depth1", "depth1_chain")
SERVER = "6.0"  # the FalkorDB server the agents read (bridges uses the native algorithm there)
MAX_TOOL_CALLS = CALL_CAP  # 12
MAX_TOOL_TURNS = 12
MAX_MODEL_CALLS = 14  # 12 tool turns + 1 final answer + 1 JSON repair
MAX_OUTPUT_TOKENS = 1024
INPUT_TOKEN_BUDGET = 40_000
DEADLINE_S = 300.0

MODEL_OUTCOMES = frozenset({"path", "not_in_graph", "unknown_name", "ambiguous_name", "partial"})
REFUSALS = frozenset({"not_in_graph", "unknown_name"})
JSON_FORMAT = {"type": "json_object"}

SYSTEM_PROMPT = """You answer questions about one company's product from a knowledge graph. You can read the graph only through the tools you are given.

How to work
1. Find facts with the tools. Never answer from memory or guess: every person, company, link and fact in your answer must come from a tool result.
2. Look up each person, company, document, bug, feature or PR the question names before you rely on it. A name the tools cannot find is unknown; do not substitute a similar one. The graph is already scoped to the product the question names: the product is not an entity to look up, and its name is not a filter or a component.
3. If one name matches several people (namesakes), name every candidate and say the name is ambiguous. Do not pick one.
4. You have at most 12 tool calls and a limited reading budget, so ask for what you need and no more. A result marked truncated has a cursor for the next page; a page counts as a call.
5. Stop when you can answer, or when the tools show the answer is not in the graph.

Final answer: reply with only one JSON object, and nothing else:
{"answer_text": "...", "answer_ids": ["..."], "evidence_ids": ["..."], "outcome": "..."}
- answer_text: the answer in plain sentences. Name only people, companies and items that appear in tool results; put every value that is graded (employee ids, company names, PR links, URLs) in answer_ids as well.
- answer_ids: the answer values themselves, bare: employee ids such as eid_xxxxxxxx, company names, PR links or URLs. No label prefixes (write eid_xxxxxxxx, not Person:eid_xxxxxxxx). Use [] when you give no such values.
- evidence_ids: record ids (messages, transcript segments, documents) that the tools returned and that support the answer.
- outcome, one of:
  path: you answered from tool results.
  unknown_name: a name the question depends on could not be found. answer_ids must be [].
  not_in_graph: the entities were found but every relation the question needs returned nothing. Say which node labels and relations you searched. answer_ids must be [].
  ambiguous_name: a name matches several candidates that the tools cannot tell apart. Name all of them.
  partial: some of what was asked was found and some was not. Give what you found and say what is missing."""

FINAL_NUDGE = ("The tool budget is used up. Give your final answer now, as the single JSON object described in the "
               "instructions, using only what the tools returned.")
REPAIR = ("Your reply was not accepted ({err}). Reply again with only the JSON object described in the instructions.")


class _Deadline(Exception):
    pass


# ---------------------------------------------------------------- configuration seen by the harness
def arm_config(toolset: str) -> dict:
    """Every knob that changes behaviour; identical across toolsets except the toolset itself."""
    return {"toolset": toolset, "max_tool_calls": MAX_TOOL_CALLS, "max_model_calls": MAX_MODEL_CALLS,
            "max_output_tokens": MAX_OUTPUT_TOKENS, "input_token_budget": INPUT_TOKEN_BUDGET,
            "deadline_s": DEADLINE_S, "server": SERVER, "temperature": 0}


def open_graph(product: str):
    """herb:<product> as the read-only agent user. `server_version` is set after `agent_graph` so bridges takes the
    native path on 6.0 (Task 8)."""
    g = depth1.agent_graph(product, SERVER)
    g.server_version = SERVER
    return g


# ---------------------------------------------------------------- the canonical answer-value projection
ANSWER_FIELDS = ("eid", "company", "link", "name")  # card fields whose values can be answers
_PROPS = ("eid", "company", "link")


def _node_value(node_id: str, label: str, name: Any, props: dict) -> str | None:
    """The bare answer value of a node: employee id, PR or URL link, company name."""
    if label == "Person":
        return node_id.split(":", 1)[1]
    if label in ("PR", "Url"):
        return node_id.split(":", 1)[1]
    if label == "Customer":
        return props.get("company") or (name if isinstance(name, str) else None)
    return None


def _node_rows(d: dict) -> list[dict]:
    nid, label = d.get("id"), d.get("label")
    if not isinstance(nid, str) or not isinstance(label, str):
        return []
    props = d.get("props") or {}
    name = d.get("name")
    row: dict[str, Any] = {"refs": [nid], "evidence": [s for s in d.get("src") or [] if s != cards.ELLIPSIS]}
    if isinstance(name, str):
        row["name"] = name
    value = _node_value(nid, label, name, props)
    if value:
        row["answer"] = value
    rows = [row]
    for k in _PROPS:  # a value the card itself carries (Document.link, Customer.company)
        v = props.get(k)
        if isinstance(v, str) and v and v != value:
            rows.append({"answer": v})
    return rows


def project_depth1(env: dict, params: dict) -> list[dict]:
    """Guard rows for a depth-1 response, one projection for every card, neighbour entry and resolve row:
    `Person:<eid>` yields eid, `PR:<link>` and `Url:<link>` yield link, Customer yields company, props eid / company /
    link count, and group values count when the grouped field can hold an answer. Nothing is fetched."""
    rows: list[dict] = []
    for r in env.get("rows") or []:
        rows += _node_rows(r)
        for e in r.get("neighbours") or []:
            rows += _node_rows(e)
    if params.get("group_field") in ANSWER_FIELDS:
        rows += [{"answer": g["value"]} for g in env.get("groups") or [] if isinstance(g.get("value"), str)]
    return rows


# ---------------------------------------------------------------- tool adapters
def _fn(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


class _Full:
    """A5: the fixed Cypher catalogue (what-if is a separate class of operation, Task 12, and is not offered)."""

    def __init__(self, conn, product: str):
        self.conn, self.product, self.trace = conn, product, Trace(MAX_TOOL_CALLS)

    def specs(self) -> list[dict]:
        return [_fn(t.name, t.description, t.params_schema) for t in TOOLS.values() if t.kind == "readonly"]

    @property
    def ops(self) -> list[TraceOp]:
        return self.trace.ops

    def call(self, name: str, args: dict) -> tuple[dict, list[dict]]:
        res = self.trace.run(self.conn, self.product, name, **args)
        rows = res.rows
        if name == "resolve_entity":  # a resolver row names a node: the same canonical projection as depth-1 cards
            rows = project_depth1({"rows": rows}, {})
        return {"rows": res.rows, "truncated": res.truncated, "cursor": res.cursor}, rows


class _Depth1:
    """A4 (strict) and A4c (chained): resolve, get_card, neighbors."""

    def __init__(self, conn, product: str, question: str, strict: bool):
        self.strict = strict
        self.episode = Episode(conn, question, product, strict=strict, max_calls=MAX_TOOL_CALLS)

    def specs(self) -> list[dict]:
        ref = ("a node returned by resolve or the product node (Product:<product name as written in the question>)"
               if self.strict else
               "a node returned by resolve, the product node (Product:<product name as written in the question>), "
               "or any node id listed on a card you have received")
        cursor = {"type": "string", "description": "Cursor from a truncated result of this same call; returns the next "
                                                    "page. Each page counts as a call."}
        edges = sorted(set(SIGNATURES) - cards.EXCLUDED_EDGES)
        return [
            _fn("resolve", "Find graph nodes for a name or identifier that appears in the question (a person, company, "
                "document, bug, feature, PR, competitor or link). `text` must be copied from the question. Returns "
                "candidate nodes (id, label, name). Several rows are namesakes or close matches; no rows means the "
                "graph has no such entity. The product itself is not resolved here: its node is "
                "Product:<product name as written in the question>, and its card lists the product's bugs, documents, "
                "meetings, features, PRs and competitors.",
                {"type": "object", "properties": {"text": {"type": "string"}, "cursor": cursor},
                 "required": ["text"], "additionalProperties": False}),
            _fn("get_card", "Read one node's card: its properties and one entry per connected node (edge type, "
                f"direction, neighbour id, label, name, source record ids). `node_id` must be {ref}.",
                {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"],
                 "additionalProperties": False}),
            _fn("neighbors", "List the cards of the nodes one edge away from a root, optionally filtered, counted or "
                f"grouped on card fields. `root_id` must be {ref}. `direction` out: the root is the edge source; in: "
                "the root is the edge target. `where` maps a card field (id, label, name or a property) to a value "
                "or a list of values. `agg` count returns only the number of matches; group needs `group_field` and "
                "returns the distinct values with counts. The product is a valid root by its name (resolve the name as written "
                "in the question). Use `where` on neighbour card fields to narrow large expansions.",
                {"type": "object", "properties": {
                    "root_id": {"type": "string"}, "edge_type": {"type": "string", "enum": edges},
                    "direction": {"type": "string", "enum": ["out", "in"]}, "where": {"type": "object"},
                    "agg": {"type": "string", "enum": ["count", "group"]}, "group_field": {"type": "string"},
                    "cursor": cursor},
                 "required": ["root_id", "edge_type", "direction"], "additionalProperties": False}),
        ]

    @property
    def ops(self) -> list[TraceOp]:
        return self.episode.trace

    def call(self, name: str, args: dict) -> tuple[dict, list[dict]]:
        ep = self.episode
        if name == "resolve":
            env = ep.resolve(**args)
        elif name == "get_card":
            env = ep.get_card(**args)
        elif name == "neighbors":
            env = ep.neighbors(**args)
        else:
            raise ToolError(f"unknown tool {name!r}; tools: resolve, get_card, neighbors")
        return env, project_depth1(env, args)


# ---------------------------------------------------------------- answer parsing
def _parse(content: str) -> tuple[dict | None, str]:
    """(fields, "") or (None, reason)."""
    text = (content or "").strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if m:
        text = m.group(1)
    try:
        d = json.loads(text)
    except ValueError:
        return None, "not valid JSON"
    if not isinstance(d, dict):
        return None, "not a JSON object"
    if not isinstance(d.get("answer_text"), str):
        return None, "answer_text must be a string"
    for k in ("answer_ids", "evidence_ids"):
        if not (isinstance(d.get(k), list) and all(isinstance(x, str) for x in d[k])):
            return None, f"{k} must be a list of strings"
    if not isinstance(d.get("outcome"), str) or d["outcome"] not in MODEL_OUTCOMES:
        return None, f"outcome must be one of {sorted(MODEL_OUTCOMES)}"
    return d, ""


_QUALIFIED = re.compile(r"^(?:Person|PR|Url):(?=\S)")


def _bare(ids: list[str]) -> list[str]:
    """A qualified employee id or link (`Person:eid_..`, `PR:https://..`) is the bare value the answer is graded on;
    the model copies them from cards. Same for every toolset. Order kept, duplicates removed."""
    return list(dict.fromkeys(_QUALIFIED.sub("", i) for i in ids))


def _from_memory(f: Finding) -> "ArmResult":
    from eval.arms import ArmResult

    text = f"From memory: confirmed in this session at {f.confirmed_at} by {f.confirmed_by}. {f.text}".strip()
    final = FinalAnswer("memory", text, list(f.answer_ids), list(f.evidence_ids), list(f.trace), Usage(0, 0, 0.0, 0))
    return ArmResult(final, [], [{"agent": {"memory": True}}])


# ---------------------------------------------------------------- the loop
class _Loop:
    def __init__(self, inp: ExecInput, toolset: str, arm: str, tools, model: Callable, clock: Callable[[], float]):
        self.inp, self.toolset, self.arm, self.tools, self.model, self.clock = inp, toolset, arm, tools, model, clock
        self.deadline = clock() + DEADLINE_S
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "user", "content": inp.text}]
        self.calls: list[CallUsage] = []
        self.envelopes: list[Any] = []
        self.guard_rows: list[list[dict]] = []
        self.cap_hit = self.budget_stop = self.repaired = False
        self.tool_calls = 0
        self.model_outcome: str | None = None
        self.dropped: list[dict] = []

    def check_deadline(self) -> None:
        if self.clock() >= self.deadline:
            raise _Deadline

    def chat(self, tools=None, response_format=None):
        self.check_deadline()
        r = self.model(self.messages, arm=self.arm, tools=tools, max_tokens=MAX_OUTPUT_TOKENS,
                       response_format=response_format)
        self.calls.append(CallUsage(self.arm, load_settings().model, r.prompt_tokens, r.completion_tokens,
                                    r.cost_usd, r.latency_s))
        self.check_deadline()  # a response that arrives after expiry is not accepted (its usage is recorded)
        return r

    def run_tool(self, tc) -> str:
        self.check_deadline()
        name = tc.function.name
        try:
            # every attempt counts, valid or not, before any argument check or adapter dispatch (same for all toolsets)
            if self.tool_calls >= MAX_TOOL_CALLS:
                raise ToolError(f"call limit reached: at most {MAX_TOOL_CALLS} calls per question")
            self.tool_calls += 1
            args = json.loads(tc.function.arguments or "{}")
            if not isinstance(args, dict):
                raise ToolError("arguments must be a JSON object")
            env, rows = self.tools.call(name, args)
            self.guard_rows.append(rows)
            self.envelopes.append({"tool": name, "params": args, **env})
            return json.dumps(env, ensure_ascii=False, separators=(",", ":"))
        except (ToolError, TypeError, ValueError) as e:  # a bad call is shown to the model, not fatal
            msg = str(e)
            if msg.startswith("call limit reached"):
                self.cap_hit = True
                msg += ". Answer now with what you have."
            self.envelopes.append({"tool": name, "params": tc.function.arguments, "error": msg})
            return json.dumps({"error": msg}, ensure_ascii=False)

    def go(self) -> "ArmResult":
        specs = self.tools.specs()
        for _ in range(MAX_TOOL_TURNS):
            r = self.chat(tools=specs)
            tcs = getattr(r.message, "tool_calls", None) or []
            if not tcs:
                return self.finish(r.message.content or "")
            self.messages.append({"role": "assistant", "content": r.message.content or "", "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tcs]})
            chars = 0
            for tc in tcs:
                text = self.run_tool(tc)
                chars += len(text)
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
            used = sum(c.prompt_tokens for c in self.calls)
            if self.cap_hit:
                break
            if used + r.prompt_tokens + r.completion_tokens + chars // 2 > INPUT_TOKEN_BUDGET:
                self.budget_stop = True
                break
        self.messages.append({"role": "user", "content": FINAL_NUDGE})
        return self.finish(self.chat(response_format=JSON_FORMAT).message.content or "")

    def finish(self, content: str) -> "ArmResult":
        fields, err = _parse(content)
        if fields is None:
            self.repaired = True
            self.messages.append({"role": "assistant", "content": content})
            self.messages.append({"role": "user", "content": REPAIR.format(err=err)})
            content = self.chat(response_format=JSON_FORMAT).message.content or ""
            fields, err = _parse(content)
            if fields is None:
                return self.fail(f"invalid final answer after one repair: {err}")
        self.model_outcome = fields["outcome"]
        asked = _bare(fields["answer_ids"])
        text, ids, dropped = guard(fields["answer_text"], asked, self.guard_rows)
        evidence, ev_dropped = guard_evidence(fields["evidence_ids"], self.guard_rows)
        self.dropped = dropped + [{"value": v, "where": "evidence", "reason": "not in any returned row"}
                                  for v in ev_dropped]
        outcome = self.model_outcome
        if ids and outcome in REFUSALS:
            outcome, text = "path", f"{', '.join(ids)}. {text}".strip()
        elif not ids and not text.strip():
            return self.fail("empty final answer: no answer ids and no text")
        return self.result(FinalAnswer(outcome, text, ids, evidence, list(self.tools.ops), self.usage()))

    def usage(self) -> Usage:
        return Usage(sum(c.prompt_tokens for c in self.calls), sum(c.completion_tokens for c in self.calls),
                     sum(c.cost_usd for c in self.calls), len(self.calls))

    def fail(self, why: str) -> "ArmResult":
        return self.result(FinalAnswer("run_failed", "", [], [], list(self.tools.ops), self.usage()), why)

    def result(self, final: FinalAnswer, failure: str | None = None) -> "ArmResult":
        from eval.arms import ArmResult

        diag = {"toolset": self.toolset, "model_outcome": self.model_outcome, "budget_stop": self.budget_stop,
                "cap_hit": self.cap_hit, "repaired": self.repaired, "tool_calls": self.tool_calls,
                "model_calls": len(self.calls), "dropped": self.dropped}
        return ArmResult(final, list(self.calls), [*self.envelopes, {"agent": diag}], failure)


def answer(inp: ExecInput, *, toolset: str, arm: str, session: Session | None = None, graph=None,
           model: Callable | None = None, clock: Callable[[], float] = time.monotonic) -> "ArmResult":
    """Answer one question. `arm` is the metering name sent to the proxy; `graph` and `model` are for tests."""
    if toolset not in TOOLSETS:
        raise ValueError(f"toolset must be one of {TOOLSETS}, got {toolset!r}")
    if session is not None:
        found = session.find(inp.product, inp.text)
        if found is not None:
            return _from_memory(found)
    loop = None
    try:
        conn = graph if graph is not None else open_graph(inp.product)
        tools = (_Full(conn, inp.product) if toolset == "full"
                 else _Depth1(conn, inp.product, inp.text, strict=toolset == "depth1"))
        loop = _Loop(inp, toolset, arm, tools, model or llm.chat, clock)
        return loop.go()
    except _Deadline:
        return loop.fail(f"deadline exceeded: {DEADLINE_S:.0f} s per question")
    except BudgetExceeded:
        if loop is None:
            raise
        return loop.fail("budget_exceeded")
    except Exception as e:  # noqa: BLE001 - any failure is this question's terminal run_failed
        if loop is None:
            from eval.arms import ArmResult

            return ArmResult(FinalAnswer("run_failed", "", [], [], [], Usage(0, 0, 0.0, 0)), [], [],
                             f"{type(e).__name__}: {e}")
        return loop.fail(f"{type(e).__name__}: {e}")
