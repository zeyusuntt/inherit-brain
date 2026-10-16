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
from inherit.graph import cards
from inherit.data.bundle import BUNDLE_DIR, load_bundle, load_metadata

ARM = "A0"
_RETRIEVERS: dict[tuple, paper_retriever.PaperRetriever] = {}
_IDENTITIES: dict[tuple, str] = {}
_LOCK = threading.RLock()


def config() -> dict:
    return {**agent.arm_config("paper"), "corpus": "raw", "top_k": paper_retriever.TOP_K,
            "splitter": paper_retriever.SPLITTER}


def load_lookups(product: str, bundle_dir: Path = BUNDLE_DIR) -> paper_tools.Lookups:
    employees, customers = load_metadata(bundle_dir)
    return paper_tools.Lookups(load_bundle(bundle_dir, product), employees, customers)


def _identity(kind: str, product: str, graph, bundle_dir: Path):
    """(texts or a callable producing them, content identity). The caller holds the lock."""
    if kind == "raw":
        texts = paper_retriever.paper_texts(load_bundle(bundle_dir, product))
        return texts, paper_retriever.texts_identity(texts)
    return (lambda: paper_retriever.card_texts(graph)), paper_retriever.graph_identity(graph)


def invalidate(kind: str | None = None, product: str | None = None) -> None:
    """Forget the memoized content identity and retriever of (kind, product); None matches all. Call it after a graph
    reload or a bundle change; the next question recomputes the identity and builds or loads the matching index."""
    with _LOCK:
        for table in (_IDENTITIES, _RETRIEVERS):
            for k in [k for k in table if kind in (None, k[0]) and product in (None, k[1])]:
                del table[k]


def shared_retriever(kind: str, product: str, *, graph=None, bundle_dir: Path = BUNDLE_DIR):
    """One retriever per (kind, product) in this process. The content identity (hash of the bundle text, or of the
    frozen graph and card version) is computed once per product per process and reused by later questions; the
    persisted index is keyed by it, so a changed corpus never loads a stale index. After a reload call `invalidate`."""
    with _LOCK:
        mkey = (kind, product, cards.CARD_VERSION if kind == "cards" else "")
        if mkey not in _IDENTITIES:
            _IDENTITIES[mkey] = _identity(kind, product, graph, bundle_dir)[1]
        identity = _IDENTITIES[mkey]
        key = (*mkey, identity)
        if key not in _RETRIEVERS:
            texts = (lambda: paper_retriever.card_texts(graph)) if kind == "cards" \
                else (lambda: paper_retriever.paper_texts(load_bundle(bundle_dir, product)))
            _RETRIEVERS[key] = paper_retriever.build_retriever(
                texts, paper_retriever.CACHE_ROOT / kind / product, identity=identity)
        return _RETRIEVERS[key]


def run(inp, *, arm: str = ARM, model: Callable | None = None, retriever=None, lookups=None,
        clock: Callable[[], float] = time.monotonic):
    def factory(loop):
        r = retriever if retriever is not None else shared_retriever("raw", inp.product)
        return PaperTools(loop, lookups or load_lookups(inp.product), r)

    return agent.answer(inp, toolset="paper", arm=arm, model=model, clock=clock, tools_factory=factory)
