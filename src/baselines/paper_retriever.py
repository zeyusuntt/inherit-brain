"""The HERB paper's retriever (react.py), over a corpus from the bundle, with our Qwen embeddings.

Corpus: one Document per Slack message, document, meeting transcript and meeting chat, formatted exactly as
`MyRAG._setup_data_loaders` builds them (URLs and PRs are not in the corpus; the paper reaches them through its
pr_search and url_search tools; thread replies are not in it either).

Splitter: react.py calls `SentenceSplitter()` with no arguments, so the settings that apply are the library defaults.
`Settings.chunk_size = 1024` and `chunk_overlap = 96` set in the constructor are not read by an explicit splitter.
In llama-index-core 0.14 the defaults are chunk_size 1024 and chunk_overlap 200; `effective_splitter()` reads them
from the library and a test pins them, so a library upgrade that changes them fails loudly.

Retrieval: `VectorIndexRetriever` top 20 and `BM25Retriever` top 20, joined as react.py's `EnsembleRetriever`: BM25
nodes first, then vector nodes, deduplicated by node id, no rank fusion. Up to 40 chunks. The vectors are Qwen3-Embedding
through `inherit.embed` (1024 dims, disk cached); the paper used text-embedding-3-large (disclosed).

The index is persisted per product under data/cache/baselines/<kind>/<product>/ and reused while the corpus text, the
splitter settings and the embedding model are unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.retrievers.bm25 import BM25Retriever

from inherit import embed
from inherit.config import EMBED_MODEL, ROOT

TOP_K = 20  # react.py similarity_top_k for both retrievers
SPLITTER = {"chunk_size": 1024, "chunk_overlap": 200}  # SentenceSplitter() defaults, llama-index-core 0.14
CACHE_ROOT = ROOT / "data" / "cache" / "baselines"


def effective_splitter() -> dict:
    """The settings `SentenceSplitter()` really uses, read from the library."""
    s = SentenceSplitter()
    return {"chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap}


# ---------------------------------------------------------------- corpus
def paper_texts(bundle: dict) -> list[str]:
    """The paper's documents for one product, in react.py order: Slack, documents, transcripts, meeting chats."""
    out: list[str] = []
    for m in bundle.get("slack", []):
        u = m["Message"]["User"]
        out.append(f"Message ID: {u['utterranceID']}\nChannel Name: {m['Channel']['name']}\n"
                   f"Timestamp: {u['timestamp']}\n\n{u['userId']}: {u['text']}")
    for d in bundle.get("documents", []):
        out.append(f"Doc ID: {d['id']}\nLink: {d['document_link']}\n{d['date']}\nAuthor: {d['author']}\n\n"
                   f"{d['type']}\n{d['content']}")
    for t in bundle.get("meeting_transcripts", []):
        out.append(f"Meeting ID: {t['id']}\n{t['date']}\nParticipants: {t['participants']}\n\n{t['transcript']}")
    for c in bundle.get("meeting_chats", []):
        out.append(f"Meeting ID: {c['id']}\n\nChats:\n{c['text']}")
    return out


def card_texts(graph) -> list[str]:
    """A0f's corpus: every card of the product (messages included) serialised as compact JSON, ordered by (label, id)."""
    from inherit.graph import cards

    return [json.dumps(c, ensure_ascii=False, separators=(",", ":")) for c in cards.cards_for_product(graph)]


# ---------------------------------------------------------------- embeddings
class QwenEmbedding(BaseEmbedding):
    """llama-index adapter over `inherit.embed`: documents through the cached corpus encoder, the query with the
    model's instruction prefix. The two callables are replaceable for tests."""

    embed_batch_size: int = embed.BATCH_ITEMS
    _corpus_fn: Any = PrivateAttr()
    _query_fn: Any = PrivateAttr()

    def __init__(self, corpus_fn: Callable | None = None, query_fn: Callable | None = None, **kw: Any):
        super().__init__(model_name=EMBED_MODEL, **kw)
        self._corpus_fn = corpus_fn or embed.encode_corpus
        self._query_fn = query_fn or embed.encode_query

    @classmethod
    def class_name(cls) -> str:
        return "QwenEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._query_fn(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._query_fn(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._corpus_fn([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._corpus_fn(texts)


# ---------------------------------------------------------------- retrieval
def union_nodes(bm25: list[NodeWithScore], vector: list[NodeWithScore]) -> list[NodeWithScore]:
    """react.py `EnsembleRetriever._retrieve`: BM25 nodes first, then vector nodes, deduplicated by node id."""
    out: list[NodeWithScore] = []
    seen: set[str] = set()
    for n in [*bm25, *vector]:
        if n.node.node_id not in seen:
            out.append(n)
            seen.add(n.node.node_id)
    return out


class PaperRetriever:
    """BM25 top k plus vector top k over one product's nodes. `retrieve` returns the chunk texts."""

    def __init__(self, nodes: list[TextNode], index: VectorStoreIndex, top_k: int = TOP_K):
        self.nodes, self.top_k = nodes, top_k
        self.vector = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        self.bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    def retrieve_nodes(self, query: str) -> list[NodeWithScore]:
        return union_nodes(self.bm25.retrieve(query), self.vector.retrieve(query))

    def retrieve(self, query: str) -> list[str]:
        return [n.node.get_content() for n in self.retrieve_nodes(query)]


def _fingerprint(texts: list[str]) -> str:
    h = hashlib.sha256(json.dumps({"splitter": SPLITTER, "embed": EMBED_MODEL, "dim": embed.DIM}).encode())
    for t in texts:
        h.update(hashlib.sha256(t.encode("utf-8")).digest())
    return h.hexdigest()


def build_retriever(texts: list[str], cache_dir: Path, embed_model: BaseEmbedding | None = None) -> PaperRetriever:
    """Split, embed and index `texts` (or load the persisted index when its fingerprint matches)."""
    embed_model = embed_model or QwenEmbedding()
    cache_dir = Path(cache_dir)
    fp = _fingerprint(texts)
    meta = cache_dir / "meta.json"
    if meta.exists() and json.loads(meta.read_text())["fingerprint"] == fp:
        sc = StorageContext.from_defaults(persist_dir=str(cache_dir))
        index = load_index_from_storage(sc, embed_model=embed_model)
        return PaperRetriever(list(sc.docstore.docs.values()), index)
    nodes = SentenceSplitter().get_nodes_from_documents([Document(text=t) for t in texts])
    sc = StorageContext.from_defaults()
    sc.docstore.add_documents(nodes)
    index = VectorStoreIndex(nodes, storage_context=sc, embed_model=embed_model)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sc.persist(persist_dir=str(cache_dir))
    meta.write_text(json.dumps({"fingerprint": fp, "documents": len(texts), "nodes": len(nodes),
                                "splitter": effective_splitter(), "embed_model": EMBED_MODEL}))
    return PaperRetriever(nodes, index)


def raw_retriever(product: str, bundle: dict, *, cache_root: Path = CACHE_ROOT,
                  embed_model: BaseEmbedding | None = None) -> PaperRetriever:
    """A0 and A0v: the paper's raw-record corpus of one product."""
    return build_retriever(paper_texts(bundle), cache_root / "raw" / product, embed_model)


def cards_retriever(product: str, graph, *, cache_root: Path = CACHE_ROOT,
                    embed_model: BaseEmbedding | None = None) -> PaperRetriever:
    """A0f: the product's serialised cards. Cards are read from the graph only when the cached index does not match
    the graph's node and edge counts and the card definition version."""
    from inherit.graph import cards

    nodes_n = graph.ro_query("MATCH (n) RETURN count(n)").result_set[0][0]
    edges_n = graph.ro_query("MATCH ()-[e]->() RETURN count(e)").result_set[0][0]
    cache_dir = cache_root / "cards" / product
    stamp = cache_dir / "graph.json"
    want = {"card_version": cards.CARD_VERSION, "nodes": nodes_n, "edges": edges_n}
    if stamp.exists() and json.loads(stamp.read_text()) == want and (cache_dir / "meta.json").exists():
        sc = StorageContext.from_defaults(persist_dir=str(cache_dir))
        index = load_index_from_storage(sc, embed_model=embed_model or QwenEmbedding())
        return PaperRetriever(list(sc.docstore.docs.values()), index)
    r = build_retriever(card_texts(graph), cache_dir, embed_model)
    stamp.write_text(json.dumps(want))
    return r
