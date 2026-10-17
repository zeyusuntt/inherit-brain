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


def ask(product: str, question: str, *, model: Callable | None = None, graph=None) -> str:
    res = agent.answer(ExecInput(product, question), toolset="full", arm=ARM, model=model, graph=graph)
    f = res.final
    lines = [f"outcome: {f.outcome}", f"answer: {f.answer_text}", f"answer ids: {', '.join(f.answer_ids) or '-'}",
             "tool calls:", format_ops(f.trace), f"evidence ids: {', '.join(f.evidence_ids) or '-'}",
             f"usage: {f.usage.n_calls} model calls, {f.usage.prompt_tokens}+{f.usage.completion_tokens} tokens, "
             f"${f.usage.cost_usd:.6f}"]
    if res.failure:
        lines.append(f"failure: {res.failure}")
    return "\n".join(lines)


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


def replay(run_dir: Path, key: str, graph=None) -> list[dict]:
    """Re-run each recorded tool call. Returns one dict per call: tool, params, rows, fresh ops and the recorded ops
    that line up with them (same order, same tool). Never calls a model."""
    run_dir = Path(run_dir)
    envs = [e for e in json.loads(gzip.decompress(_slot_file(run_dir, key).read_bytes())) if "tool" in e]
    recorded = _recorded_ops(run_dir, key)
    conn = graph if graph is not None else agent.open_graph(key.split("/")[0])
    product = key.split("/")[0]
    trace, out, at = Trace(max(len(envs), 1)), [], 0
    for e in envs:
        params = e["params"]
        item = {"tool": e["tool"], "params": params, "error": None, "rows": None, "fresh": [], "recorded": []}
        if not isinstance(params, dict):  # the model's arguments were not a JSON object: nothing ran then
            item["error"] = "recorded arguments were not a JSON object; skipped"
            out.append(item)
            continue
        before = len(trace.ops)
        try:
            item["rows"] = len(trace.run(conn, product, e["tool"], **params).rows)
        except (ToolError, TypeError, ValueError, NotImplementedError) as ex:
            item["error"] = str(ex)
        item["fresh"] = [op.to_dict() for op in trace.ops[before:]]
        n = len(item["fresh"])
        item["recorded"] = [o for o in recorded[at:at + n] if o.get("tool") == e["tool"]]
        at += n
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
    tr = sum(r["ms"] for it in items for r in it["recorded"])
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
        print(format_replay(replay(Path(a.replay), a.slot)))
        return 0
    if not (a.product and a.question):
        ap.error("give --product and a question, or --replay with --slot")
    print(ask(a.product, a.question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
