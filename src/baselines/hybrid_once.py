"""A0v: the paper's hybrid RAG with a single retrieval. Retrieve once with the question (BM25 top 20, then vector top
20, deduplicated), make one answer call with those chunks and the shared JSON answer format. No tools, no repair, no
verification: exactly one retrieval and one model call.
"""

from __future__ import annotations

import time
from typing import Callable

from baselines import paper_retriever
from baselines.paper_tools import ANSWER_FORMAT, Seen
from baselines.react_paper import shared_retriever
from inherit import agent, llm
from inherit.contracts import CallUsage, FinalAnswer, Usage
from inherit.config import load_settings
from inherit.llm import BudgetExceeded
from inherit.tools.guard import guard, guard_evidence

ARM = "A0v"
SYSTEM = ("You answer questions about one company's product from retrieved records: Slack messages, documents, "
          "meeting transcripts and meeting chats. Use only the records given; never answer from memory or guess. "
          "If the records do not contain the answer, say so with the outcome not_in_graph.\n\n" + ANSWER_FORMAT)


def config() -> dict:
    return {"max_model_calls": 1, "max_output_tokens": agent.MAX_OUTPUT_TOKENS, "deadline_s": agent.DEADLINE_S,
            "temperature": 0, "corpus": "raw", "top_k": paper_retriever.TOP_K, "splitter": paper_retriever.SPLITTER}


def _failed(why: str, calls: list[CallUsage]) -> "ArmResult":
    from eval.arms import ArmResult  # late: eval.arms imports this module while registering

    use = Usage(sum(c.prompt_tokens for c in calls), sum(c.completion_tokens for c in calls),
                sum(c.cost_usd for c in calls), len(calls))
    return ArmResult(FinalAnswer("run_failed", "", [], [], [], use), calls, [], why)


def run(inp, *, arm: str = ARM, model: Callable | None = None, retriever=None,
        clock: Callable[[], float] = time.monotonic) -> "ArmResult":
    from eval.arms import ArmResult

    deadline = clock() + agent.DEADLINE_S
    calls: list[CallUsage] = []
    try:
        r = retriever if retriever is not None else shared_retriever("raw", inp.product)
        chunks = r.retrieve(inp.text)
        if clock() >= deadline:
            return _failed(f"deadline exceeded: {agent.DEADLINE_S:.0f} s per question", calls)
        context = "\n\n".join(chunks)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Records:\n---------------------\n{context}\n---------------------\n"
                                                f"Question: {inp.text}"}]
        res = (model or llm.chat)(messages, arm=arm, max_tokens=agent.MAX_OUTPUT_TOKENS,
                                  response_format=agent.JSON_FORMAT)
        calls.append(CallUsage(arm, load_settings().model, res.prompt_tokens, res.completion_tokens, res.cost_usd,
                               res.latency_s))
        if clock() >= deadline:
            return _failed(f"deadline exceeded: {agent.DEADLINE_S:.0f} s per question", calls)
        fields, err = agent._parse(res.message.content or "")
        if fields is None:
            return _failed(f"invalid final answer: {err}", calls)
        seen = Seen()
        rows = seen.add(list(chunks))
        asked = agent._bare(fields["answer_ids"])
        rows = [rows, seen.support(asked, fields["evidence_ids"])]
        text, ids, _ = guard(fields["answer_text"], asked, rows)
        evidence, _ = guard_evidence(fields["evidence_ids"], rows)
        outcome, text, failure = agent.apply_consistency(fields["outcome"], text, ids)
        if failure:
            return _failed(failure, calls)
        use = Usage(res.prompt_tokens, res.completion_tokens, res.cost_usd, 1)
        return ArmResult(FinalAnswer(outcome, text, ids, evidence, [], use), calls,
                         [{"agent": {"toolset": "hybrid_once", "model_outcome": fields["outcome"],
                                     "chunks": len(chunks), "model_calls": 1}}])
    except BudgetExceeded:
        return _failed("budget_exceeded", calls)
    except Exception as e:  # noqa: BLE001 - any failure is this question's terminal run_failed
        return _failed(f"{type(e).__name__}: {e}", calls)
