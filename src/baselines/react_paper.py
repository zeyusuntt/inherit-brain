"""A0: the HERB paper's tools (`retrieve_all` plus eight metadata tools) in the shared agent loop.

Named "HERB tools in the shared agent loop", not a reproduction of the paper's ReAct run (design F1). Same model as
every arm, temperature 0, native function calling, the loop's limits (12 tool calls, 24 model calls in total counting
the tools' own, the 40,000-token stopping policy, a 300 s deadline), one JSON answer. The retriever is the paper's
(`paper_retriever`), the tools are `paper_tools`.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from baselines import paper_retriever, paper_tools
from baselines.paper_tools import PaperTools
from inherit import agent
from inherit.data.bundle import BUNDLE_DIR, load_bundle, load_metadata

ARM = "A0"
_RETRIEVERS: dict[tuple, paper_retriever.PaperRetriever] = {}
_LOCK = threading.Lock()


def config() -> dict:
    return {**agent.arm_config("paper"), "max_model_calls": agent.MAX_TOTAL_MODEL_CALLS,
            "reserved_final_calls": agent.RESERVED_FINAL_CALLS, "corpus": "raw", "top_k": paper_retriever.TOP_K,
            "splitter": paper_retriever.SPLITTER}


def load_lookups(product: str, bundle_dir: Path = BUNDLE_DIR) -> paper_tools.Lookups:
    employees, customers = load_metadata(bundle_dir)
    return paper_tools.Lookups(load_bundle(bundle_dir, product), employees, customers)


def shared_retriever(kind: str, product: str, *, graph=None, bundle_dir: Path = BUNDLE_DIR):
    """One retriever per (kind, product) in this process; the index itself is persisted under data/cache/baselines."""
    with _LOCK:
        key = (kind, product)
        if key not in _RETRIEVERS:
            _RETRIEVERS[key] = (paper_retriever.raw_retriever(product, load_bundle(bundle_dir, product))
                                if kind == "raw" else paper_retriever.cards_retriever(product, graph))
        return _RETRIEVERS[key]


def run(inp, *, arm: str = ARM, model: Callable | None = None, retriever=None, lookups=None,
        clock: Callable[[], float] = time.monotonic):
    def factory(loop):
        r = retriever if retriever is not None else shared_retriever("raw", inp.product)
        return PaperTools(loop, lookups or load_lookups(inp.product), r)

    return agent.answer(inp, toolset="paper", arm=arm, model=model, clock=clock, tools_factory=factory)
