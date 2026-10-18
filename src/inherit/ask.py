"""Ask the A5 agent one question live, or replay the recorded tool calls of a held-out slot.

    uv run python -m inherit.ask --product ActionGenie "Who reviewed the market research report?"
    uv run python -m inherit.ask --replay runs/eval/A5-held --slot AutoTuneForce/ans/2

Live mode makes model calls through the metering proxy (inherit.llm), so the proxy must be running first:
`uv run python -m inherit.proxy`. Replay mode makes no model call: it re-executes the recorded tool calls (tool name and
parameters from the run's tools/ file) against the live read-only graph and prints fresh timings next to the recorded ones.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Callable

from inherit import agent
from inherit.contracts import ExecInput, TraceOp
from inherit.tools.catalog import ToolError
from inherit.tools.trace import Trace

ARM = "A5-ask"  # metering name of interactive questions, apart from the evaluation arms


def _q(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def format_ops(ops: list[TraceOp]) -> str:
    out = []
    for i, op in enumerate(ops, 1):
        out.append(f"  {i}. {op.tool} {json.dumps(op.params, ensure_ascii=False)}\n"
                   f"     {op.command}: {_q(op.query, 400)}\n     rows={op.row_count} ms={op.ms}")
    return "\n".join(out) or "  (no tool calls)"


def ask_run(product: str, question: str, *, model: Callable | None = None, graph=None) -> tuple[str, str]:
    """The printed text and the outcome of one live question."""
    res = agent.answer(ExecInput(product, question), toolset="full", arm=ARM, model=model, graph=graph)
    f = res.final
    lines = [f"outcome: {f.outcome}", f"answer: {f.answer_text}", f"answer ids: {', '.join(f.answer_ids) or '-'}",
             "tool calls:", format_ops(f.trace), f"evidence ids: {', '.join(f.evidence_ids) or '-'}",
             f"usage: {f.usage.n_calls} model calls, {f.usage.prompt_tokens}+{f.usage.completion_tokens} tokens, "
             f"${f.usage.cost_usd:.6f}"]
    if res.failure:
        lines.append(f"failure: {res.failure}")
    return "\n".join(lines), f.outcome


def ask(product: str, question: str, *, model: Callable | None = None, graph=None) -> str:
    return ask_run(product, question, model=model, graph=graph)[0]


# ---------------------------------------------------------------- replay (no model call)
def _slot_file(run_dir: Path, key: str) -> Path:
    product, kind, index = key.split("/")
    return run_dir / "tools" / f"{product}__{kind}__{index}.json.gz"


def _recorded_ops(run_dir: Path, key: str) -> list[dict]:
    for line in (run_dir / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        k = r["key"]
        if f"{k['product']}/{k['kind']}/{k['index']}" == key:
            return r.get("tool_ops") or []
    raise KeyError(f"slot {key!r} is not in {run_dir / 'records.jsonl'}")


def _as_params(params):
    """The recorded arguments as a dict, or None if they were not a JSON object (a string is decoded first)."""
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            return None
    return params if isinstance(params, dict) else None


def _take(recorded: list[dict], used: set[int], tool: str, *, query: str | None = None, refused: bool = False):
    """The first unused recorded op of `tool` with this query text (or, if refused, with command "refused")."""
    for i, o in enumerate(recorded):
        if i in used or o.get("tool") != tool:
            continue
        if (o.get("command") == "refused") if refused else (o.get("query") == query and o.get("command") != "refused"):
            used.add(i)
            return o
    return None


def replay(run_dir: Path, key: str, graph=None) -> list[dict]:
    """Re-run each recorded tool call. Returns one dict per call: tool, params, rows, fresh ops and the recorded ops
    that line up with them (same tool and query text; a refused call takes a recorded "refused" op). Never calls a model."""
    run_dir = Path(run_dir)
    envs = [e for e in json.loads(gzip.decompress(_slot_file(run_dir, key).read_bytes())) if "tool" in e]
    recorded = _recorded_ops(run_dir, key)
    conn = graph if graph is not None else agent.open_graph(key.split("/")[0])
    product = key.split("/")[0]
    trace, out, used = Trace(max(len(envs), 1)), [], set()
    for e in envs:
        params = _as_params(e["params"])
        item = {"tool": e["tool"], "params": e["params"] if params is None else params, "error": None, "rows": None,
                "fresh": [], "recorded": []}
        if params is None:  # the model's arguments were not a JSON object: nothing ran then
            item["error"] = "recorded arguments were not a JSON object; skipped"
            r = _take(recorded, used, e["tool"], refused=True)
            item["recorded"] = [r] if r else []
            out.append(item)
            continue
        before = len(trace.ops)
        try:
            item["rows"] = len(trace.run(conn, product, e["tool"], **params).rows)
        except (ToolError, TypeError, ValueError, NotImplementedError) as ex:
            item["error"] = str(ex)
        item["fresh"] = [op.to_dict() for op in trace.ops[before:]]
        if not item["fresh"]:  # refused before the trace noted it (call limit)
            r = _take(recorded, used, e["tool"], refused=True)
            item["recorded"] = [r] if r else []
        else:  # a refused fresh op takes a recorded refusal; any other takes the recorded op with the same query
            item["recorded"] = [_take(recorded, used, e["tool"], refused=f["command"] == "refused", query=f["query"])
                                for f in item["fresh"]]
        out.append(item)
    return out


def format_replay(items: list[dict]) -> str:
    out = []
    for i, it in enumerate(items, 1):
        head = f"{i}. {it['tool']} {json.dumps(it['params'], ensure_ascii=False)}"
        out.append(head + (f"\n   error: {it['error']}" if it["error"] else f"\n   rows={it['rows']}"))
        for j, f in enumerate(it["fresh"]):
            r = it["recorded"][j] if j < len(it["recorded"]) else None
            rec = f"recorded rows={r['row_count']} ms={r['ms']}" if r else "recorded: -"
            out.append(f"   query {j + 1}: fresh rows={f['row_count']} ms={f['ms']} | {rec}\n"
                       f"     {_q(f['query'], 400)}")
    tf = sum(f["ms"] for it in items for f in it["fresh"])
    tr = sum(r["ms"] for it in items for r in it["recorded"] if r)
    out.append(f"total: fresh {tf:.3f} ms, recorded {tr:.3f} ms (no model call)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?", help="the question (live mode)")
    ap.add_argument("--product", help="the product graph (live mode)")
    ap.add_argument("--replay", metavar="RUN_DIR", help="replay the recorded tool calls of a run, e.g. runs/eval/A5-held")
    ap.add_argument("--slot", help="with --replay: the slot key, e.g. AutoTuneForce/ans/2")
    a = ap.parse_args(argv)
    if a.replay:
        if not a.slot:
            ap.error("--replay needs --slot")
        try:
            items = replay(Path(a.replay), a.slot)
        except KeyError:
            raise
        except Exception as e:  # noqa: BLE001 - the graph could not be opened (or read)
            print(f"replay failed: {e}", file=sys.stderr)
            return 2
        print(format_replay(items))
        return 0
    if not (a.product and a.question):
        ap.error("give --product and a question, or --replay with --slot")
    text, outcome = ask_run(a.product, a.question)
    print(text)
    return 1 if outcome == "run_failed" else 0


if __name__ == "__main__":
    sys.exit(main())
