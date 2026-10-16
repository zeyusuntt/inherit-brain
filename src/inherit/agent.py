"""The tool-calling agent shared by A5 (toolset "full"), A4 ("depth1", strict) and A4c ("depth1_chain").

One loop, one system prompt, one set of budgets and one answer handler; only the tool list differs, so a difference
between the arms is a difference between their tools. The model sees the system prompt, the question text and tool
traffic, nothing else (the product only selects the graph connection).

Budgets, identical for every toolset and every agentic arm (A5, A4, A4c, A0, A0f): at most 12 tool calls (enforced by
the tool layer; the 13th is refused), at most 24 model calls in total counted by one counter (agent turns plus any call
a custom tool set makes itself; the last 2 are kept for the final answer and one verification or repair; the built-in
toolsets make no calls of their own, so they use at most 14: 12 tool turns, the final answer, one JSON repair), at most
1,024 output tokens per call, a cumulative input budget of 40,000 prompt tokens (actual usage so far plus an estimate of
the next call: the previous prompt and completion plus the new tool-result characters / 2; when that would exceed the
budget the agent stops calling tools and makes the final-answer call with its history as is; a call a tool makes in
the middle of a tool batch is checked against the same budget before it is sent), and a cooperative 300 s deadline
checked before every model and tool call.

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
from inherit.ontology import NODE_KEYS, SIGNATURES
from inherit.tools import depth1
from inherit.tools.catalog import TOOLS
from inherit.tools.depth1 import CALL_CAP, Episode, ToolError, fold
from inherit.tools.guard import guard, guard_evidence
from inherit.tools.trace import Trace

PROMPT_VERSION = "agent/v4"
TOOLSETS = ("full", "depth1", "depth1_chain")
SERVER = "6.0"  # the FalkorDB server the agents read (bridges uses the native algorithm there)
MAX_TOOL_CALLS = CALL_CAP  # 12
MAX_TOOL_TURNS = 12
MAX_REMINDERS = 2
# One per-question counter for every model call of every agentic arm: at most 24 calls in total. A custom tool set
# (tools_factory) makes calls of its own (lookup extraction, answer verification, fallback); the last two calls are kept
# for the final answer and one verification or repair. The built-in toolsets never exceed 14 (12 turns + answer + repair).
MAX_TOTAL_MODEL_CALLS = 24
RESERVED_FINAL_CALLS = 2
MAX_OUTPUT_TOKENS = 1024
INPUT_TOKEN_BUDGET = 40_000
DEADLINE_S = 300.0

MODEL_OUTCOMES = frozenset({"path", "not_in_graph", "unknown_name", "ambiguous_name", "partial"})
REFUSALS = frozenset({"not_in_graph", "unknown_name"})
JSON_FORMAT = {"type": "json_object"}

SYSTEM_PROMPT_HEAD = """You answer questions about one company's product from a knowledge graph. You can read the graph only through the tools you are given.

How to work
1. Find facts with the tools. Never answer from memory or guess: every person, company, link and fact in your answer must come from a tool result.
2. Look up each person, company, document, bug, feature or PR the question names before you rely on it. A name the tools cannot find is unknown; do not substitute a similar one. The product the question names is already in scope: it is never a filter and never an entity to resolve, and its name is not a component, feature, topic or customer.
3. Add a filter only when the question states it: no current or previous release scope, no merged or pending state, no status restriction unless the question asks for one (for a list of companies or people affected by issues in a component, use all statuses). Pass the full phrase from the question to feature, component and topic parameters, not a shortened phrase.
3b. If one name matches several people (namesakes), name every candidate and say the name is ambiguous. Do not pick one.
4. Build the final answer from the rows of the call that answers the relation the question asks about; do not replace them with values from unrelated later lookups. You have at most 12 tool calls and a limited reading budget, so ask for what you need and no more. A result marked truncated has a cursor for the next page; a page counts as a call.
5. Stop when you can answer, or when the tools show the answer is not in the graph."""

ANSWER_INSTRUCTIONS = """Final answer: reply with only one JSON object, and nothing else:
{"answer_text": "...", "answer_ids": ["..."], "evidence_ids": ["..."], "outcome": "..."}
- answer_text: the answer in plain sentences. Name only people, companies and items that appear in the records or tool results you were given; put every value that is graded (employee ids, company names, PR links, URLs) in answer_ids as well.
- answer_ids: the answer values themselves, bare: employee ids such as eid_xxxxxxxx, company names, PR links or URLs. A company is answered by its name, never by its CUST- id: when you only have a customer id, turn it into the company name. No label prefixes (write eid_xxxxxxxx, not Person:eid_xxxxxxxx). Use [] when you give no such values.
- Lists and ties: when a ranking returns several rows tied at the top, answer_ids lists every tied row (every tied answer value). For a list question, answer_ids holds every returned answer value, not a sample. For a question that asks for content (features, suggestions, issues, claims), answer_text includes the detail fields you were given (feature details, suggestion texts, claim texts), not only titles.
- Nothing found: when the records or tool results contain nothing that answers what was asked, do not state a negative fact (not "No X reported ..." or "X has no ...") and do not add related facts, people or ids. Set outcome to not_in_graph (or unknown_name when a name could not be found), leave answer_ids empty, and say the records do not contain the information, naming what you searched.
- evidence_ids: record ids (messages, transcript segments, documents) that you were given and that support the answer.
- outcome, one of:
  path: you answered from the records or tool results.
  unknown_name: a name the question depends on could not be found. answer_ids must be [].
  not_in_graph: the entities were found but every relation the question needs returned nothing. Say what you searched. answer_ids must be [].
  ambiguous_name: a name matches several candidates that you cannot tell apart. Name all of them.
  partial: some of what was asked was found and some was not. Give what you found and say what is missing."""

SYSTEM_PROMPT = SYSTEM_PROMPT_HEAD + "\n\n" + ANSWER_INSTRUCTIONS

FINAL_NUDGE = ("The tool budget is used up. Give your final answer now, as the single JSON object described in the "
               "instructions, using only what the tools returned.")
REMINDER = ("Reminder: the product is already in scope. Query the relation the question asks about before concluding "
            "there is no information.")
RESOLVERS = frozenset({"resolve_person", "resolve_entity", "resolve"})
NOT_RELATION = RESOLVERS | {"get_card"}
REPAIR = ("Your reply was not accepted ({err}). Reply again with only the JSON object described in the instructions.")


class _Deadline(Exception):
    pass


class AnswerRejected(Exception):
    """Raised by a tool set's `review` when the final answer is rejected and no valid replacement exists."""


# ---------------------------------------------------------------- configuration seen by the harness
def arm_config(toolset: str) -> dict:
    """Every knob that changes behaviour; identical across toolsets except the toolset itself."""
    return {"toolset": toolset, "max_tool_calls": MAX_TOOL_CALLS, "max_model_calls": MAX_TOTAL_MODEL_CALLS,
            "reserved_final_calls": RESERVED_FINAL_CALLS,
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
        env = {"rows": res.rows, "truncated": res.truncated, "cursor": res.cursor}
        if getattr(res, "note", None):  # a one-sentence hint from the tool, shown to the model as part of the result
            env["note"] = res.note
        return env, rows


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


REFUSAL_SENTENCE = "I could not find information in the records to answer this question."
_LABEL_PREFIX = re.compile(rf"^({'|'.join(sorted(NODE_KEYS, key=len, reverse=True))}):")


def refusal_text(*_unused) -> str:
    """The one refusal sentence, identical for every arm. What was searched goes to diagnostics (`searched`)."""
    return REFUSAL_SENTENCE


def searched(tools_used: list[str], rows: list[list[dict]]) -> str:
    """A diagnostic summary of what a run searched: tools in call order and the node labels seen in returned refs."""
    names = list(dict.fromkeys(tools_used))
    labels = sorted({m.group(1) for group in rows for r in group for ref in r.get("refs") or []
                     if isinstance(ref, str) and (m := _LABEL_PREFIX.match(ref))})
    out = ", ".join(names) if names else "no tool returned results"
    return out + (f" (node types: {', '.join(labels)})" if labels else "")


def apply_consistency(outcome: str, text: str, ids: list[str]) -> tuple[str, str, str | None]:
    """The one consistency rule, shared by every arm that answers through the common JSON format. Returns
    (outcome, text, failure): a refusal with no ids gets the canonical sentence; a refusal with supported ids becomes
    `path` with the values up front; no ids and no text is a failure (the caller returns run_failed)."""
    if outcome in REFUSALS and not ids:
        text = REFUSAL_SENTENCE
    if ids and outcome in REFUSALS:
        outcome, text = "path", f"{', '.join(ids)}. {text}".strip()
    elif not ids and not text.strip():
        return outcome, text, "empty final answer: no answer ids and no text"
    return outcome, text, None


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
        self.reminders = 0
        self.ties: list[list[str]] = []  # answer values of each rank=max result (its top tie)
        self.tie_completed: list[str] = []
        self.model_outcome: str | None = None
        self.dropped: list[dict] = []
        self.max_calls = MAX_TOTAL_MODEL_CALLS  # the one declared limit of every agentic arm

    def use_tools(self, tools) -> None:
        """Install a custom tool set. It provides specs(), call(name, args) -> (envelope, guard rows) and ops, and
        optionally system_prompt, review(loop, fields) -> fields (raise AnswerRejected to reject), rescue(loop) -> fields
        | None (the one fallback answer, used when the agent ends without a valid answer) and
        support(answer_ids, evidence_ids) -> rows. It may call `nested_chat`; every such call is counted and limited
        with the agent's own."""
        self.tools = tools
        self.messages[0]["content"] = getattr(tools, "system_prompt", SYSTEM_PROMPT)

    def room(self, final: bool = False) -> bool:
        return len(self.calls) < self.max_calls - (0 if final else RESERVED_FINAL_CALLS)

    def input_stop(self, next_chars: int) -> bool:
        """The shared input stopping policy for a call about to be made: tokens already spent plus an estimate of the
        call's own prompt (its characters / 2) must stay within the cumulative budget."""
        return sum(c.prompt_tokens for c in self.calls) + next_chars // 2 > INPUT_TOKEN_BUDGET

    def nested_chat(self, messages: list[dict], *, response_format=None, final: bool = False):
        """A model call made by a tool or a finaliser: same deadline, counter and usage list as the agent turns.
        Refused with a ToolError when the per-question limit leaves no room (`final` may use the reserved calls) or,
        for a call that is not a finalisation, when the input stopping policy says stop (the agent then answers)."""
        if not self.room(final):
            raise ToolError(f"model call limit reached: at most {self.max_calls} model calls per question")
        if not final and self.input_stop(sum(len(str(m.get("content") or "")) for m in messages)):
            self.budget_stop = True
            raise ToolError("input token budget reached: answer now with what you have")
        return self.chat(response_format=response_format, messages=messages)

    def check_deadline(self) -> None:
        if self.clock() >= self.deadline:
            raise _Deadline

    def chat(self, tools=None, response_format=None, messages=None):
        self.check_deadline()
        if len(self.calls) >= self.max_calls:
            raise ToolError(f"model call limit reached: at most {self.max_calls} model calls per question")
        r = self.model(self.messages if messages is None else messages, arm=self.arm, tools=tools, max_tokens=MAX_OUTPUT_TOKENS,
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
            if args.get("rank") == "max":
                top = [r["answer"] for r in env.get("rows") or [] if isinstance(r.get("answer"), str)]
                if top:
                    self.ties.append(top)
            return json.dumps(env, ensure_ascii=False, separators=(",", ":"))
        except (ToolError, TypeError, ValueError) as e:  # a bad call is shown to the model, not fatal
            msg = str(e)
            if msg.startswith("call limit reached"):
                self.cap_hit = True
                msg += ". Answer now with what you have."
            self.envelopes.append({"tool": name, "params": tc.function.arguments, "error": msg})
            return json.dumps({"error": msg}, ensure_ascii=False)

    def unqueried_refusal(self, fields: dict) -> bool:
        """SPEC 5.3: a refusal is allowed only if a needed entity failed to resolve (a resolver call that ran and found
        nothing; an erroring call, an empty result that carries a tool note (such as an unsupported kind) or a lookup of the product itself does not count) or a relation
        tool (anything but a resolver or get_card; neighbors counts) ran and returned zero rows."""
        if fields["outcome"] not in REFUSALS or fields["answer_ids"]:
            return False
        product = {fold(self.inp.product), fold(f"Product:{self.inp.product}")}
        for e in self.envelopes:
            name = e.get("tool")
            if name is None or "error" in e:
                continue
            if e.get("rows") or e.get("groups") or e.get("count"):
                continue
            if e.get("note"):  # an empty result with a tool hint (unsupported kind, retry with the full phrase) is not proof
                continue
            if name in RESOLVERS:
                params = e.get("params") or {}
                target = params.get("text") if isinstance(params.get("text"), str) else params.get("name")
                if not (isinstance(target, str) and fold(target) in product):
                    return False
            elif name not in NOT_RELATION:
                return False
        return True

    def complete_ties(self, ids: list[str]) -> list[str]:
        """If the answer holds a value from one rank=max result's top tie, add every value tied at the top in that same
        call. Values of other calls are never copied."""
        out = list(ids)
        for top in self.ties:
            if set(top) & set(ids):
                add = [v for v in top if v not in out]
                out += add
                self.tie_completed += add
        return out

    def go(self) -> "ArmResult":
        specs = self.tools.specs()
        for _ in range(MAX_TOOL_TURNS):
            if not self.room():
                break
            r = self.chat(tools=specs)
            tcs = getattr(r.message, "tool_calls", None) or []
            if not tcs:
                out = self.finish(r.message.content or "", may_remind=True)
                if out is not None:
                    return out
                continue  # a reminder was sent; the tool loop goes on
            self.messages.append({"role": "assistant", "content": r.message.content or "", "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in tcs]})
            chars = 0
            for tc in tcs:
                text = self.run_tool(tc)
                chars += len(text)
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})
            used = sum(c.prompt_tokens for c in self.calls)
            if self.cap_hit or self.budget_stop:
                break
            if used + r.prompt_tokens + r.completion_tokens + chars // 2 > INPUT_TOKEN_BUDGET:
                self.budget_stop = True
                break
        self.messages.append({"role": "user", "content": FINAL_NUDGE})
        return self.finish(self.chat(response_format=JSON_FORMAT).message.content or "")

    def finish(self, content: str, may_remind: bool = False) -> "ArmResult | None":
        """The final answer from the model's `content`. With `may_remind` a refusal that breaks the relation rule (also
        one produced inside the JSON repair) gets a reminder instead and None is returned, up to MAX_REMINDERS."""
        fields, err = _parse(content)
        rescued = False
        if fields is None:
            self.repaired = True
            self.messages.append({"role": "assistant", "content": content})
            self.messages.append({"role": "user", "content": REPAIR.format(err=err)})
            content = self.chat(response_format=JSON_FORMAT).message.content or ""
            fields, err = _parse(content)
            if fields is None:  # an unsuccessful completion gets the one permitted fallback, if the tool set has one
                rescue = getattr(self.tools, "rescue", None)
                fields = rescue(self) if rescue is not None else None
                rescued = fields is not None
                if fields is None:
                    return self.fail(f"invalid final answer after one repair: {err}")
        if (may_remind and not rescued and self.reminders < MAX_REMINDERS and self.room()
                and self.unqueried_refusal(fields)):
            self.reminders += 1
            self.messages.append({"role": "assistant", "content": content})
            self.messages.append({"role": "user", "content": REMINDER})
            return None
        review = getattr(self.tools, "review", None)
        if review is not None and not rescued:  # the fallback answer is the paper's last resort: not verified again
            try:
                fields = review(self, fields) or fields
            except AnswerRejected as e:
                return self.fail(f"final answer rejected: {e}")
        self.model_outcome = fields["outcome"]
        asked = _bare(fields["answer_ids"])
        rows = self.guard_rows
        support = getattr(self.tools, "support", None)
        if support is not None:
            rows = [*rows, support(asked, fields["evidence_ids"])]
        text, ids, dropped = guard(fields["answer_text"], asked, rows)
        evidence, ev_dropped = guard_evidence(fields["evidence_ids"], rows)
        self.dropped = dropped + [{"value": v, "where": "evidence", "reason": "not in any returned row"}
                                  for v in ev_dropped]
        ids = self.complete_ties(ids)
        outcome, text, failure = apply_consistency(self.model_outcome, text, ids)
        if failure:
            return self.fail(failure)
        return self.result(FinalAnswer(outcome, text, ids, evidence, list(self.tools.ops), self.usage()))

    def usage(self) -> Usage:
        return Usage(sum(c.prompt_tokens for c in self.calls), sum(c.completion_tokens for c in self.calls),
                     sum(c.cost_usd for c in self.calls), len(self.calls))

    def fail(self, why: str) -> "ArmResult":
        ops = list(self.tools.ops) if self.tools is not None else []
        return self.result(FinalAnswer("run_failed", "", [], [], ops, self.usage()), why)

    def result(self, final: FinalAnswer, failure: str | None = None) -> "ArmResult":
        from eval.arms import ArmResult

        diag = {"toolset": self.toolset, "model_outcome": self.model_outcome, "budget_stop": self.budget_stop,
                "cap_hit": self.cap_hit, "reminded": self.reminders > 0, "reminders": self.reminders, "tie_completed": self.tie_completed,
                "searched": searched([e["tool"] for e in self.envelopes if "error" not in e and "tool" in e], self.guard_rows), "repaired": self.repaired, "tool_calls": self.tool_calls,
                "model_calls": len(self.calls), "dropped": self.dropped}
        return ArmResult(final, list(self.calls), [*self.envelopes, {"agent": diag}], failure)


def answer(inp: ExecInput, *, toolset: str, arm: str, session: Session | None = None, graph=None,
           model: Callable | None = None, clock: Callable[[], float] = time.monotonic,
           tools_factory: Callable | None = None) -> "ArmResult":
    """Answer one question. `arm` is the metering name sent to the proxy; `graph` and `model` are for tests.

    `tools_factory(loop)` replaces the built-in tool sets (then `toolset` is only a label for the diagnostics): the
    paper baselines A0 and A0f run their tools in this same loop. See `_Loop.use_tools`."""
    if tools_factory is None and toolset not in TOOLSETS:
        raise ValueError(f"toolset must be one of {TOOLSETS}, got {toolset!r}")
    if session is not None:
        found = session.find(inp.product, inp.text)
        if found is not None:
            return _from_memory(found)
    loop = None
    try:
        if tools_factory is not None:
            loop = _Loop(inp, toolset, arm, None, model or llm.chat, clock)
            loop.use_tools(tools_factory(loop))
            return loop.go()
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
