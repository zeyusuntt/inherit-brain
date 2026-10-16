"""A1f and A1: the HERB paper's long-context baseline (long_context_eval.py `build_long_context`).

A1f: the paper's builder without `team` and `customers`, plus the full employee and customer directories first
(`ID: name (role)`, `ID: company`, sorted by ID, the same lookup domains the metadata tools use), and every Slack line
tagged with its message ID (`[id] user: text`) so sources can be cited. Section order is the paper's: Slack by channel
sorted by time, meeting transcripts, meeting chats, documents, URLs, pull requests. One answer call with the shared
JSON format. The complete rendered chat request (with the assistant prefix) is counted with the Qwen3 tokenizer; if it plus 2,048 output tokens exceeds
262,144 the question fails, nothing is truncated.

A1: the paper's builder as written (team and customers from the dev-only loader, no directories, no IDs), dev
products only; a composite field-sensitivity diagnostic, since fields, directories and IDs all differ from A1f.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from typing import Callable

from baselines.paper_tools import ANSWER_FORMAT, Seen
from inherit import agent, llm
from inherit.config import ROOT, load_settings
from inherit.contracts import CallUsage, FinalAnswer, Usage
from inherit.data.bundle import BUNDLE_DIR, load_bundle, load_metadata
from inherit.llm import BudgetExceeded
from inherit.tools.guard import guard, guard_evidence

LIMIT = 262_144
OUTPUT_HEADROOM = 2_048
TOKENIZER_PATH = ROOT / "data" / "cache" / "tokenizer" / "tokenizer.json"
TOKENIZER_REPO = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PROMPT_VERSION = "baselines/a1f/v2"

SYSTEM = ("You are a helpful AI assistant. Use the following product context to answer the given multi-hop question. "
          "Use reasoning to find the answer in the context. If the answer cannot be found in the context, say so "
          "with the outcome not_in_graph. Never answer from memory or guess. Slack messages are tagged [message id]; "
          "documents, meetings and pull requests carry their own ids and links.\n\n" + ANSWER_FORMAT)

_TOKENIZER = None


def count_tokens(text: str) -> int:
    global _TOKENIZER
    if _TOKENIZER is None:
        from tokenizers import Tokenizer

        _TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
    return len(_TOKENIZER.encode(text, add_special_tokens=False).ids)


def config(arm: str = "A1f") -> dict:
    return {"max_model_calls": 1, "max_output_tokens": agent.MAX_OUTPUT_TOKENS, "deadline_s": agent.DEADLINE_S,
            "temperature": 0, "limit_tokens": LIMIT, "output_headroom": OUTPUT_HEADROOM,
            "tokenizer": TOKENIZER_REPO, "variant": arm}


# ---------------------------------------------------------------- context builders
def _slack(product_data: dict, ids: bool) -> str | None:
    channels: dict = defaultdict(list)
    for item in product_data.get("slack") or []:
        u = item["Message"]["User"]
        channels[item["Channel"]["channelID"]].append(
            (datetime.fromisoformat(u["timestamp"]), item["Channel"]["name"], u["userId"], u["text"], u["utterranceID"]))
    parts = []
    for msgs in channels.values():
        msgs.sort(key=lambda m: m[0])
        lines = [(f"[{m[4]}] " if ids else "") + f"{m[2]}: {m[3]}" for m in msgs]
        parts.append(f"Channel: {msgs[0][1]}\n" + "\n".join(lines))
    return "SLACK CHANNELS\n\n" + "\n\n".join(parts) if parts else None


def _records(d: dict, ids: bool) -> list[str]:
    """Every section after team, customers and Slack, exactly as the paper's builder writes them."""
    out = []
    if d.get("meeting_transcripts"):
        out.append("MEETING TRANSCRIPTS\n" + "\n\n".join(
            f"Meeting ID: {t['id']}\n{t['date']}\nParticipants: {t['participants']}\n\n{t['transcript']}"
            for t in d["meeting_transcripts"]))
    if d.get("meeting_chats"):
        out.append("MEETING CHATS\n" + "\n\n".join(
            f"Meeting ID: {c['id']}\nChats:\n{c['text']}" for c in d["meeting_chats"]))
    if d.get("documents"):
        out.append("DOCUMENTS\n" + "\n\n".join(
            f"Doc ID: {x['id']}\nLink: {x['document_link']}\n{x['date']}\nAuthor: {x['author']}\n\n{x['type']}\n"
            f"{x['content']}" for x in d["documents"]))
    if d.get("urls"):
        out.append("URLs\n" + "\n\n".join(f"URL: {u['link']}\n\nDescription: {u['description']}" for u in d["urls"]))
    if d.get("prs"):
        prs = []
        for pr in d["prs"]:
            s = (f"Title: {pr['title']}\nAuthor: {pr['user']['login']}\nCreated At: {pr['created_at']}\n"
                 f"State: {pr['state']}\nMergeable: {pr['mergeable']}\nMerged: {pr['merged']}\nLink: {pr['link']}\n"
                 f"Summary: {pr['summary']}\n\nReviews:\n")
            for r in pr["reviews"]:
                s += f"- {r['state']} by {r['user']['login']} at {r['submitted_at']}: {r['comment']}\n"
            prs.append(s)
        out.append("PULL REQUESTS\n" + "\n\n".join(prs))
    return out


def build_context_a1f(bundle: dict, employees: dict, customers: list) -> str:
    emp = "EMPLOYEES:\n" + "".join(
        f"{e['employee_id']}: {e['name']} ({e['role']})\n" for e in sorted(employees.values(), key=lambda e: e["employee_id"]))
    cust = "CUSTOMERS:\n" + "".join(f"{c['id']}: {c['company']}\n" for c in sorted(customers, key=lambda c: c["id"]))
    slack = _slack(bundle, ids=True)
    return "\n\n".join([emp, cust] + ([slack] if slack else []) + _records(bundle, ids=True))


def build_context_a1(bundle: dict, employees: dict, fields: dict, customer_names: dict) -> str:
    sections = []
    if fields.get("team"):
        by_role: dict = {}
        for eid in fields["team"]:
            e = employees.get(eid)
            if e is not None:
                by_role.setdefault(e["role"], []).append(f"{eid} ({e['name']})")
        sections.append("TEAM STRUCTURE:\n" + "".join(f"{r}: {', '.join(v)}\n" for r, v in by_role.items()))
    if fields.get("customers"):
        sections.append("CUSTOMERS:\n" + "".join(f"{c}: {customer_names[c]}\n" for c in fields["customers"]))
    slack = _slack(bundle, ids=False)
    return "\n\n".join(sections + ([slack] if slack else []) + _records(bundle, ids=False))


# ---------------------------------------------------------------- the run
def _failed(why: str, calls: list[CallUsage]) -> "ArmResult":
    from eval.arms import ArmResult  # late: eval.arms imports this module while registering

    use = Usage(sum(c.prompt_tokens for c in calls), sum(c.completion_tokens for c in calls),
                sum(c.cost_usd for c in calls), len(calls))
    return ArmResult(FinalAnswer("run_failed", "", [], [], [], use), calls, [], why)


def render_chat(messages: list[dict]) -> str:
    """The Qwen3 chat template as the server applies it: role markers, end markers and the assistant generation prefix."""
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"


def count_rendered(count: Callable[[str], int], messages: list[dict]) -> int:
    return count(render_chat(messages))


def request_tokens(messages: list[dict], count: Callable[[str], int]) -> int:
    return count_rendered(count, messages)


def run(inp, *, arm: str = "A1f", model: Callable | None = None, bundle_dir=BUNDLE_DIR, count: Callable | None = None,
        raw_dir=None, clock: Callable[[], float] = time.monotonic) -> "ArmResult":
    from eval.arms import ArmResult

    deadline = clock() + agent.DEADLINE_S
    calls: list[CallUsage] = []
    variant = "A1" if "a1f" not in arm.lower() else "A1f"
    try:
        fields = None
        if variant == "A1":  # refuse a held-out product before anything is loaded
            from eval import a1_fields

            try:
                fields = a1_fields.load_fields(inp.product, *([raw_dir] if raw_dir else []))
            except ValueError as e:
                return _failed(str(e), calls)
        bundle = load_bundle(bundle_dir, inp.product)
        employees, customers = load_metadata(bundle_dir)
        if variant == "A1":
            context = build_context_a1(bundle, employees, fields, {c["id"]: c["company"] for c in customers})
        else:
            context = build_context_a1f(bundle, employees, customers)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Product Context:\n{context}\n\nQuestion:\n{inp.text}\n\nAnswer:\n"}]
        n = request_tokens(messages, count or count_tokens)
        if n + OUTPUT_HEADROOM > LIMIT:
            return _failed(f"context too long: {n} request tokens + {OUTPUT_HEADROOM} output headroom exceeds "
                           f"{LIMIT}", calls)
        res = (model or llm.chat)(messages, arm=arm, max_tokens=agent.MAX_OUTPUT_TOKENS,
                                  response_format=agent.JSON_FORMAT)
        calls.append(CallUsage(arm, load_settings().model, res.prompt_tokens, res.completion_tokens, res.cost_usd,
                               res.latency_s))
        if clock() >= deadline:
            return _failed(f"deadline exceeded: {agent.DEADLINE_S:.0f} s per question", calls)
        fields_out, err = agent._parse(res.message.content or "")
        if fields_out is None:
            return _failed(f"invalid final answer: {err}", calls)
        seen = Seen()
        rows = seen.add([context])
        asked = agent._bare(fields_out["answer_ids"])
        rows = [rows, seen.support(asked, fields_out["evidence_ids"])]
        text, ids, _ = guard(fields_out["answer_text"], asked, rows)
        evidence, _ = guard_evidence(fields_out["evidence_ids"], rows)
        outcome, text, failure = agent.apply_consistency(fields_out["outcome"], text, ids)
        if failure:
            return _failed(failure, calls)
        use = Usage(res.prompt_tokens, res.completion_tokens, res.cost_usd, 1)
        return ArmResult(FinalAnswer(outcome, text, ids, evidence, [], use), calls,
                         [{"agent": {"toolset": "long_context", "variant": variant, "request_tokens": n,
                                     "model_outcome": fields_out["outcome"], "model_calls": 1}}])
    except BudgetExceeded:
        return _failed("budget_exceeded", calls)
    except Exception as e:  # noqa: BLE001 - any failure is this question's terminal run_failed
        return _failed(f"{type(e).__name__}: {e}", calls)
