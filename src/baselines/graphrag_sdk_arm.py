"""A2: GraphRAG-SDK 1.4.0 with its automatic ontology, on the product's allowed records.

Lifecycle, as the SDK ships it: `Ontology.from_sources` drafts an ontology from the source texts, `GraphRAG.ingest`
extracts a graph per source into FalkorDB (graph `a2:<product>` on FalkorDB 4.22.1), `finalize()` deduplicates and embeds, and
`completion()` answers each question (no history, SDK retrieval defaults). Nothing about the SDK is tuned.

Every model and embedding call goes through the metering proxy (`inherit.llm`), meter name `a2-build` while building
and `A2` (the registered arm name, which the runner caps) while answering. The SDK's own retries are switched off (the
proxy retries and meters). The answer is asked for as the shared JSON through the SDK's answer prompt; one conversion
call runs only when that reply does not parse, and it may only restate the SDK's answer.

One `QuestionBudget` per question carries the declared limits of every agentic arm (24 model calls with the last two
reserved for finalisation, the 40,000-token cumulative input policy, the 300 s deadline checked before and after each
call) over every SDK model call and the conversion call. The answer then goes through the shared output guard
(complete-token support in the retrieved context only), the canonical refusal text and the consistency rule.

A build counts only when every source finished ingesting without a provider failure (the SDK swallows some) and
`finalize()` succeeded; completion is written to `runs/a2/<product>.build.json` after each source, and `open_graph`
refuses a graph without it. A2 answers dev products and a pre-registered seeded sample of 6 held-out products.

Source texts (F7), built from the bundle only: one per Slack channel, document (with its feedback) and meeting (with
its chat), one per meeting chat that matches no meeting, one for all PRs with their reviews, one for all URLs, plus
the employee and customer directories. Each carries stable record ids and timestamps.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import json
import logging
import random
import threading
import time
from typing import Any, Callable, NamedTuple

import httpx

from graphrag_sdk import (
    ChatMessage, ConnectionConfig, Embedder, GraphRAG, LLMInterface, Ontology,
)
from graphrag_sdk.core.models import LLMResponse
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError  # noqa: F401
from graphrag_sdk.ingestion.extraction_strategies import GraphExtraction, LLMExtractor
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.core.models import DocumentInfo, DocumentOutput

from baselines.paper_tools import Seen
from eval.arms import Arm, ArmResult, register
from eval.questions import DEV_PRODUCTS
from inherit import agent, llm
from inherit.config import ROOT, load_settings
from inherit.contracts import CallUsage, ExecInput, FinalAnswer, Usage
from inherit.data.bundle import BUNDLE_DIR, load_bundle, load_metadata
from inherit.data.split import split
from inherit.tools.guard import guard, guard_evidence

PROMPT_VERSION = "a2/v2"
ARM_BUILD, ARM_ANSWER = "a2-build", "A2"  # the one meter name per phase; ARM_ANSWER is also the registered arm name
SAMPLE_SEED, SAMPLE_SIZE = 20261015, 6  # the pre-registered draw of held-out products A2 runs on
BUILD_CAP_USD, ANSWER_CAP_USD = 1.45, 0.3  # build cap raised from 1.0 by the controller to finish the AnomalyForce dev build
FALKOR_HOST, FALKOR_PORT = "localhost", 6379  # FalkorDB 4.22.1: the SDK's fulltext DDL (db.idx.fulltext.createNodeIndex
# with a label and fields) is rejected by 6.0.2, which then trips the SDK's circuit breaker mid-ingest
EMBED_DIM = 1024
BUILD_MAX_TOKENS, ANSWER_MAX_TOKENS = 4096, agent.MAX_OUTPUT_TOKENS
DEADLINE_S = agent.DEADLINE_S
DISCOVERY_SAMPLE_CHUNKS = 1  # SDK default is 3
DISCOVERY_PER_CATEGORY = 1  # sources per category the ontology is drafted from (F7: discovery sampling is cut first)
DISCOVERY_SEED = 20261015
DISCOVERY_CONCURRENCY = 12  # SDK default 4; wall-clock only, the calls are the same
INGEST_CONCURRENCY = 3  # the SDK's own batch default
GRAPH_PATTERN = "a2:{product}"
EMBED_BATCH = 64

logger = logging.getLogger(__name__)


def discovery_sources(sources: list[Source]) -> list[Source]:
    """The sources the ontology is drafted from: the first DISCOVERY_PER_CATEGORY of each category, in id order.
    Drafting from all of them gave a 100-type un-normalized draft (the SDK's normalization pass could not return
    it), and every extraction prompt carries the ontology, so the draft size sets the whole build cost."""
    seen: dict[str, int] = {}
    out = []
    for s in sources:
        kind = s.doc_id.split(":", 1)[0]
        seen[kind] = seen.get(kind, 0) + 1
        if seen[kind] <= DISCOVERY_PER_CATEGORY:
            out.append(s)
    return out


def ontology_path(product: str):
    return ROOT / "runs" / "a2" / f"{product}.ontology.json"


def graph_name(product: str) -> str:
    return GRAPH_PATTERN.format(product=product)


# ---------------------------------------------------------------- source texts (F7)
class Source(NamedTuple):
    doc_id: str
    text: str


def _slack(product_data: dict) -> list[Source]:
    channels: dict[str, dict] = {}
    for item in product_data.get("slack") or []:
        ch, user = item["Channel"], item["Message"]["User"]
        entry = channels.setdefault(ch["channelID"], {"name": ch["name"], "msgs": []})
        entry["msgs"].append((user["timestamp"], user["utterranceID"], user["userId"], user["text"]))
    out = []
    for cid in sorted(channels):
        e = channels[cid]
        lines = [f"[{mid}] {ts} {uid}: {text}" for ts, mid, uid, text in sorted(e["msgs"])]
        out.append(Source(f"slack:{cid}", f"Slack channel {e['name']} ({cid})\n" + "\n".join(lines)))
    return out


def _documents(product_data: dict) -> list[Source]:
    out = []
    for d in sorted(product_data.get("documents") or [], key=lambda d: d["id"]):
        text = (f"Document {d['id']}\nType: {d['type']}\nDate: {d['date']}\nAuthor: {d['author']}\n"
                f"Link: {d['document_link']}\n\n{d['content']}")
        if d.get("feedback"):
            text += f"\n\nFeedback on document {d['id']}:\n{d['feedback']}"
        out.append(Source(f"document:{d['id']}", text))
    return out


def _meetings(product_data: dict) -> list[Source]:
    """Each meeting with its chat (the chat whose id is `<meeting id>_chat`); every other chat is its own source
    under its original id, so no chat is lost."""
    chats = {c["id"]: c["text"] for c in product_data.get("meeting_chats") or []}
    out = []
    for m in sorted(product_data.get("meeting_transcripts") or [], key=lambda m: m["id"]):
        text = (f"Meeting ID: {m['id']}\nDate: {m['date']}\nType: {m.get('document_type', '')}\n"
                f"Participants: {', '.join(m['participants'])}\n\n{m['transcript']}")
        chat = chats.pop(f"{m['id']}_chat", None)
        if chat:
            text += f"\n\nChat {m['id']}_chat of meeting {m['id']}:\n{chat}"
        out.append(Source(f"meeting:{m['id']}", text))
    out += [Source(f"chat:{cid}", f"Meeting chat {cid}\n{text}") for cid, text in sorted(chats.items())]
    return out


def _prs(product_data: dict) -> list[Source]:
    """All of the product's PRs with their reviews as one source: a source per PR would be a hundred tiny
    documents, each paying its own discovery summary and ingestion overhead."""
    blocks = []
    for pr in sorted(product_data.get("prs") or [], key=lambda p: p["link"]):
        lines = [f"Pull request {pr['link']}", f"Number: {pr['number']}", f"Title: {pr['title']}", f"Author: {pr['user']['login']}",
                 f"Created at: {pr['created_at']}", f"State: {pr['state']}", f"Mergeable: {pr['mergeable']}",
                 f"Merged: {pr['merged']}", f"Summary: {pr['summary']}", "Reviews:"]
        lines += [f"- {r['state']} by {r['user']['login']} at {r['submitted_at']}: {r['comment']}"
                  for r in pr.get("reviews") or []]
        blocks.append("\n".join(lines))
    return [Source("prs", "Pull requests\n\n" + "\n\n".join(blocks))] if blocks else []


def _urls(product_data: dict) -> list[Source]:
    blocks = [f"URL {u['link']}\nDescription: {u['description']}"
              for u in sorted(product_data.get("urls") or [], key=lambda u: (u["link"], u["description"]))]
    return [Source("urls", "URLs\n\n" + "\n\n".join(blocks))] if blocks else []


def source_texts(product: str, bundle_dir=BUNDLE_DIR) -> list[Source]:
    """The product's allowed records as source texts; deterministic, bundle only."""
    data = load_bundle(bundle_dir, product)
    employees, customers = load_metadata(bundle_dir)
    emp = "Employee directory (employee id: name (role))\n" + "\n".join(
        f"{eid}: {e['name']} ({e['role']})" for eid, e in sorted(employees.items()))
    cust = "Customer directory (customer id: company)\n" + "\n".join(
        f"{c['id']}: {c['company']}" for c in sorted(customers, key=lambda c: c["id"]))
    return [*_slack(data), *_documents(data), *_meetings(data), *_prs(data), *_urls(data),
            Source("employees", emp), Source("customers", cust)]


# ---------------------------------------------------------------- providers through the proxy
CURRENT_SOURCE: contextvars.ContextVar[str | None] = contextvars.ContextVar("a2_source", default=None)


class Failure(NamedTuple):
    source: str | None  # the source being ingested when the call failed, if known
    call: str
    error: str


class BudgetStop(Exception):
    """A model call refused by the per-question limits."""


class DeadlineExceeded(Exception):
    """The per-question deadline passed before or after a call."""


class UsageLog:
    """Every metered call the SDK made, as CallUsage, and every provider failure (the SDK swallows some of them, so
    the failures are recorded here, tagged with the source being ingested)."""

    def __init__(self, model: Callable | None = None) -> None:
        self.calls: list[CallUsage] = []
        self.failures: list[Failure] = []
        self.model = model  # tests; default `inherit.llm.chat`
        self._lock = threading.Lock()

    def add(self, c: CallUsage) -> None:
        with self._lock:
            self.calls.append(c)

    def fail(self, call: str, e: Exception) -> None:
        with self._lock:
            self.failures.append(Failure(CURRENT_SOURCE.get(), call, f"{type(e).__name__}: {e}"))

    def chat(self, messages: list[dict], *, arm: str, max_tokens: int, response_format: dict | None = None,
             final: bool = False):
        try:
            r = (self.model or llm.chat)(messages, arm=arm, max_tokens=max_tokens, response_format=response_format)
        except Exception as e:  # noqa: BLE001 - recorded, then the caller decides
            self.fail("chat", e)
            raise
        self.add(CallUsage(arm, load_settings().model, r.prompt_tokens, r.completion_tokens, r.cost_usd, r.latency_s))
        return r

    def embed(self, texts: list[str], *, arm: str, dimensions: int):
        try:
            r = llm.embed(texts, arm=arm, dimensions=dimensions)
        except Exception as e:  # noqa: BLE001
            self.fail("embed", e)
            raise
        self.add(CallUsage(arm, load_settings().embed_model, r.prompt_tokens, 0, r.cost_usd, 0.0))
        return r


class QuestionBudget(UsageLog):
    """The one budget of a question, shared by every SDK model call and the conversion call: at most
    `agent.MAX_TOTAL_MODEL_CALLS` model calls with the last `RESERVED_FINAL_CALLS` kept for finalisation (`final=True`:
    the SDK's answer call and the conversion), the cumulative input policy (prompt tokens so far plus the next call's
    characters / 2 within `INPUT_TOKEN_BUDGET`, not applied to finalisation), and the deadline, checked before and
    after each call (a response that arrives late is recorded but not accepted). The same rules as `agent._Loop`."""

    def __init__(self, model: Callable | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(model)
        self.clock, self.deadline = clock, clock() + DEADLINE_S
        self.started = self.input_tokens = 0
        self.stops: list[str] = []
        self.deadline_hit = False

    def check_deadline(self) -> None:
        if self.clock() >= self.deadline:
            self.deadline_hit = True
            raise DeadlineExceeded

    def chat(self, messages, *, arm, max_tokens, response_format=None, final=False):
        self.check_deadline()
        chars = sum(len(str(m.get("content") or "")) for m in messages)
        with self._lock:
            if self.started >= agent.MAX_TOTAL_MODEL_CALLS - (0 if final else agent.RESERVED_FINAL_CALLS):
                self.stops.append("calls")
                raise BudgetStop(f"model call limit reached: at most {agent.MAX_TOTAL_MODEL_CALLS} model calls per question")
            if not final and self.input_tokens + chars // 2 > agent.INPUT_TOKEN_BUDGET:
                self.stops.append("input")
                raise BudgetStop("input token budget reached")
            self.started += 1
        r = super().chat(messages, arm=arm, max_tokens=max_tokens, response_format=response_format, final=final)
        with self._lock:
            self.input_tokens += r.prompt_tokens
        self.check_deadline()
        return r

    def embed(self, texts, *, arm, dimensions):
        self.check_deadline()
        r = super().embed(texts, arm=arm, dimensions=dimensions)
        self.check_deadline()
        return r


class ProxyLLM(LLMInterface):
    """The SDK's LLM, routed through `inherit.llm.chat` by way of a `UsageLog` (a `QuestionBudget` while answering).
    One attempt per call: the proxy retries and meters.

    `json_messages` asks for a JSON object on `ainvoke_messages` only, the call `completion()` uses for the answer
    (a finalisation call for the budget); the SDK's retrieval-side prompts (keywords, text-to-Cypher) stay free text."""

    def __init__(self, arm: str, log: UsageLog, *, max_tokens: int = BUILD_MAX_TOKENS, json_messages: bool = False):
        super().__init__(model_name=load_settings().model)
        self.arm, self.log, self.max_tokens, self.json_messages = arm, log, max_tokens, json_messages

    def _chat(self, messages: list[dict], response_format: dict | None = None, final: bool = False) -> LLMResponse:
        r = self.log.chat(messages, arm=self.arm, max_tokens=self.max_tokens, response_format=response_format,
                          final=final)
        return LLMResponse(content=r.message.content or "")

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return self._chat([{"role": "user", "content": prompt}])

    async def ainvoke(self, prompt: str, *, max_retries: int = 3, timeout: float | None = None, **kw: Any):
        return await super().ainvoke(prompt, max_retries=1, timeout=timeout, **kw)

    async def ainvoke_messages(self, messages: list[ChatMessage], *, max_retries: int = 3,
                               timeout: float | None = None, **kw: Any) -> LLMResponse:
        rf = {"type": "json_object"} if self.json_messages else None
        payload = [{"role": m.role, "content": m.content} for m in messages]
        return await asyncio.to_thread(self._chat, payload, rf, self.json_messages)


class ProxyEmbedder(Embedder):
    """Qwen3-Embedding-8B at 1024 dimensions through the proxy (the SDK is told the same dimension)."""

    def __init__(self, arm: str, log: UsageLog):
        self.arm, self.log = arm, log
        self._model = load_settings().embed_model

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            out += self.log.embed(texts[i:i + EMBED_BATCH], arm=self.arm, dimensions=EMBED_DIM).vectors
        return out

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_documents([text])[0]


class _DictLoader(LoaderStrategy):
    """Serves source texts by id, so ontology discovery needs no files."""

    def __init__(self, texts: dict[str, str]):
        self.texts = texts

    async def load(self, source: str, ctx: Context) -> DocumentOutput:
        return DocumentOutput(text=self.texts[source], document_info=DocumentInfo(path=source))


# ---------------------------------------------------------------- caps
def _post_cap(arm: str, usd: float) -> None:
    url = llm.proxy_url().removesuffix("/v1") + "/admin/arm_cap"
    httpx.post(url, json={"arm": arm, "usd": usd}, timeout=10.0).raise_for_status()


def set_caps() -> None:
    """The hard sub-caps, set before any build or answer."""
    _post_cap(ARM_BUILD, BUILD_CAP_USD)
    _post_cap(ARM_ANSWER, ANSWER_CAP_USD)


# ---------------------------------------------------------------- the held-out sample
def held_out_sample(held_out: list[str]) -> list[str]:
    """The pre-registered draw: SAMPLE_SIZE of the held-out products, seeded, from the sorted names."""
    return random.Random(SAMPLE_SEED).sample(sorted(held_out), SAMPLE_SIZE)


def _frozen_split(bundle_dir) -> tuple[list[str], list[str]]:
    return split(sorted(p.stem for p in (bundle_dir / "products").glob("*.json")))


def frozen_sample(bundle_dir=BUNDLE_DIR) -> list[str]:
    return held_out_sample(_frozen_split(bundle_dir)[1])


def products_allowed(bundle_dir=BUNDLE_DIR) -> frozenset[str]:
    """The dev products and the sampled held-out products."""
    dev, held = _frozen_split(bundle_dir)
    return frozenset(dev) | frozenset(held_out_sample(held))


@functools.lru_cache(maxsize=1)
def allowed_products() -> frozenset[str]:
    return products_allowed()


# ---------------------------------------------------------------- build
class BuildIncomplete(RuntimeError):
    """The graph is not a complete, finalized build of every source; A2 must not answer on it."""

    def __init__(self, why: str, report: dict | None = None):
        super().__init__(why)
        self.report = report or {}


def state_path(product: str):
    return ROOT / "runs" / "a2" / f"{product}.build.json"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_state(product: str) -> dict:
    """{"sources": {source id: sha256 of the text that finished ingesting}, "finalized": bool, ...}."""
    p = state_path(product)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"sources": {}, "finalized": False}


def write_state(product: str, state: dict) -> None:
    p = state_path(product)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


def build_problems(product: str, sources: list[Source]) -> list[str]:
    """Why the product's graph may not be answered on; empty when every expected source finished and finalize ran."""
    done = read_state(product)
    missing = [s.doc_id for s in sources if done["sources"].get(s.doc_id) != _sha(s.text)]
    problems = []
    if missing:
        problems.append(f"{len(missing)} of {len(sources)} sources not ingested ({', '.join(missing[:5])}"
                        f"{', ...' if len(missing) > 5 else ''})")
    if not done.get("finalized"):
        problems.append("finalize has not completed")
    return problems


@functools.lru_cache(maxsize=8)
def _expected_sources(product: str) -> list[Source]:
    return source_texts(product)


def _facade(product: str, arm: str, log: UsageLog, **llm_kw) -> GraphRAG:
    conn = ConnectionConfig(host=FALKOR_HOST, port=FALKOR_PORT, graph_name=graph_name(product),
                            query_timeout_ms=120_000)
    return GraphRAG(conn, ProxyLLM(arm, log, **llm_kw), ProxyEmbedder(arm, log), embedding_dimension=EMBED_DIM)


async def _load_ontology(product: str, sources: list[Source], log: UsageLog, report: dict) -> Ontology:
    texts = {s.doc_id: s.text for s in discovery_sources(sources)}
    cache = ontology_path(product)  # a resumed build reuses the draft instead of paying for discovery again
    if cache.exists():
        ontology = Ontology.from_file(str(cache))
    else:
        ontology = await Ontology.from_sources(
            list(texts), ProxyLLM(ARM_BUILD, log), loader=_DictLoader(texts),
            sample_chunks_per_doc=DISCOVERY_SAMPLE_CHUNKS, concurrency=DISCOVERY_CONCURRENCY, seed=DISCOVERY_SEED)
        cache.parent.mkdir(parents=True, exist_ok=True)
        ontology.save_to_file(str(cache))
    report["entity_types"] = [e.label for e in ontology.entities]
    report["relation_types"] = [r.label for r in ontology.relations]
    return ontology


def _extractor(rag: GraphRAG, ontology: Ontology) -> GraphExtraction:
    # The SDK's default step-1 NER is a local GLiNER model whose tokenizer fails to load under the pinned
    # transformers 5.x; the SDK's own LLM NER (LLMExtractor) is its documented alternative.
    return GraphExtraction(rag.llm, entity_extractor=LLMExtractor(rag.llm),
                           entity_types=[e.label for e in ontology.entities])


async def _build(product: str, sources: list[Source], log: UsageLog, report: dict) -> None:
    ontology = await _load_ontology(product, sources, log, report)
    if log.failures:  # discovery swallowed a provider failure: the draft is not trustworthy, nothing is ingested on it
        ontology_path(product).unlink(missing_ok=True)
        report["failures"] = [f._asdict() for f in log.failures[:20]]
        raise BuildIncomplete("provider failure during ontology discovery", report)
    state = read_state(product)
    async with _facade(product, ARM_BUILD, log) as rag:
        await rag.set_ontology(ontology)
        sem = asyncio.Semaphore(INGEST_CONCURRENCY)
        extractor = _extractor(rag, ontology)
        ingested_now: list[str] = []

        async def drop(doc_id: str) -> None:
            """Remove a document with whatever an interrupted or failed attempt left behind."""
            if await rag._graph_store.get_document_record(doc_id) is not None:
                await rag.delete_document(doc_id, if_missing="ignore")

        async def one(s: Source) -> int:
            async with sem:
                digest = _sha(s.text)
                if state["sources"].get(s.doc_id) == digest and await rag._graph_store.get_document_record(s.doc_id):
                    return 0  # finished in an earlier run
                state["sources"].pop(s.doc_id, None)
                await drop(s.doc_id)  # the SDK writes the document record first: a record alone proves nothing
                token = CURRENT_SOURCE.set(s.doc_id)
                chunks = None
                try:
                    chunks = (await rag.ingest(text=s.text, document_id=s.doc_id, extractor=extractor)).chunks_indexed
                except Exception as e:  # noqa: BLE001 - this source is not complete
                    log.fail("ingest", e)
                finally:
                    CURRENT_SOURCE.reset(token)
                if chunks is None or any(f.source == s.doc_id for f in log.failures):
                    await drop(s.doc_id)
                    return 0
                state["sources"][s.doc_id] = digest
                state["finalized"] = False
                ingested_now.append(s.doc_id)
                write_state(product, state)
                return chunks

        report["chunks"] = sum(await asyncio.gather(*[one(s) for s in sources]))
        if any(f.source is None for f in log.failures):  # cannot be tied to a source: none of this run's can be trusted
            for doc_id in ingested_now:
                state["sources"].pop(doc_id, None)
                await drop(doc_id)
            write_state(product, state)
            raise BuildIncomplete("provider failure outside any source; this run's sources were removed", report)
        failed = sorted({f.source for f in log.failures})
        report["sources_complete"] = len(state["sources"])
        if failed:
            report["failed_sources"], report["failures"] = failed, [f._asdict() for f in log.failures[:20]]
            raise BuildIncomplete(f"{len(failed)} sources failed ingestion: {', '.join(failed[:5])}", report)
        if ingested_now or not state.get("finalized"):
            n0 = len(log.failures)
            fin = await rag.finalize()
            if len(log.failures) > n0:
                report["failures"] = [f._asdict() for f in log.failures[n0:][:20]]
                raise BuildIncomplete("provider failure during finalize", report)
            state["finalized"], state["finalize"] = True, fin.model_dump()
            write_state(product, state)
        report["finalize"] = state["finalize"]
        report["stats"] = await rag.get_statistics()


NER_SOFT_FAILURE = "LLM NER returned invalid JSON"


class _SoftFailureCounter(logging.Handler):
    """Counts the SDK's NER replies that were not JSON (the chunk is skipped). Disclosed, not a reason to reject."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if NER_SOFT_FAILURE in record.getMessage():
            self.count += 1


def _stats(log: UsageLog, report: dict, t0: float) -> None:
    chat_calls = [c for c in log.calls if c.completion_tokens or c.model == load_settings().model]
    report.update(seconds=round(time.monotonic() - t0, 1), calls=len(log.calls),
                  chat_calls=len(chat_calls), prompt_tokens=sum(c.prompt_tokens for c in log.calls),
                  completion_tokens=sum(c.completion_tokens for c in log.calls),
                  cost_usd=round(sum(c.cost_usd for c in log.calls), 4))


def _record_soft_failures(product: str, counter: _SoftFailureCounter, report: dict) -> None:
    """Add this run's count to the build manifest (a resumed build keeps the earlier runs' counts)."""
    state = read_state(product)
    state["ner_soft_failures"] = state.get("ner_soft_failures", 0) + counter.count
    write_state(product, state)
    report["ner_soft_failures_this_run"], report["ner_soft_failures_total"] = counter.count, state["ner_soft_failures"]


def build(product: str, bundle_dir=BUNDLE_DIR, *, replace: bool = False) -> dict:
    """Build `a2:<product>` from the bundle. Returns a report: sources, chunks, calls, tokens, cost, seconds.
    Raises `BuildIncomplete` (its `report` holds the same numbers) unless every source finished and finalize ran."""
    set_caps()
    sources = source_texts(product, bundle_dir)
    log, report = UsageLog(), {"product": product, "graph": graph_name(product), "sources": len(sources),
                               "source_chars": sum(len(s.text) for s in sources)}
    t0 = time.monotonic()
    if replace:
        async def wipe():
            async with _facade(product, ARM_BUILD, log) as rag:
                await rag.delete_all()
        asyncio.run(wipe())
        ontology_path(product).unlink(missing_ok=True)
        state_path(product).unlink(missing_ok=True)
    counter = _SoftFailureCounter()
    logging.getLogger().addHandler(counter)
    try:
        asyncio.run(_build(product, sources, log, report))
    except BuildIncomplete as e:
        _record_soft_failures(product, counter, e.report)
        _stats(log, e.report, t0)
        raise
    finally:
        logging.getLogger().removeHandler(counter)
    _record_soft_failures(product, counter, report)
    _stats(log, report, t0)
    return report


# ---------------------------------------------------------------- answering
ANSWER_SYSTEM = """You answer questions about one company's product using only the context the user message gives you. The context was retrieved from a knowledge graph built from the product's Slack channels, documents, meetings, pull requests, URLs and the employee and customer directories. Never answer from memory; every person, company, link and fact must come from the context. A name the context does not contain is unknown; do not substitute a similar one. If one name matches several people, name every candidate and say it is ambiguous.

Reply with only one JSON object, and nothing else:
{"answer_text": "...", "answer_ids": ["..."], "evidence_ids": ["..."], "outcome": "..."}
- answer_text: the answer in plain sentences, naming only people, companies and items that appear in the context.
- answer_ids: the answer values themselves, bare: employee ids such as eid_xxxxxxxx, company names, PR links or URLs. No label prefixes. Use [] when you give no such values.
- evidence_ids: record ids (Slack message ids such as 20260611-3-5f298, meeting ids, document ids) that appear in the context and support the answer.
- outcome, one of:
  path: you answered from the context.
  unknown_name: a name the question depends on is not in the context. answer_ids must be [].
  not_in_graph: the entities were found but the context holds nothing that answers the question. answer_ids must be [].
  ambiguous_name: a name matches several candidates the context cannot tell apart. Name all of them.
  partial: some of what was asked was found and some was not. Give what you found and say what is missing."""

ANSWER_TEMPLATE = "<context>\n{context}\n</context>\n\nQuestion: {question}\n\nReply with only the JSON object."

CONVERT = ("Rewrite the answer below as the JSON object described. Restate only what the answer says: do not add "
           "facts, ids or names, and do not look anything up. Reply with only the JSON object.\n\n"
           'Fields: {{"answer_text": string, "answer_ids": [bare employee ids, company names, PR links or URLs '
           'that the answer names], "evidence_ids": [record ids the answer cites, else []], "outcome": one of '
           "{outcomes}}}\n\nQuestion: {question}\n\nAnswer to convert:\n{answer}")


def arm_config() -> dict:
    s = load_settings()
    try:
        sample = frozen_sample()
    except OSError:  # no local bundle: the config then records no sample and the arm refuses every held-out product
        sample = None
    return {"graph": "a2:<product>", "model": s.model, "embed_model": s.embed_model, "embed_dim": EMBED_DIM,
            "sdk": "graphrag-sdk 1.4.0", "falkordb_port": FALKOR_PORT, "ner": "LLMExtractor",
            "discovery_per_category": DISCOVERY_PER_CATEGORY, "discovery_sample_chunks": DISCOVERY_SAMPLE_CHUNKS,
            "max_output_tokens": ANSWER_MAX_TOKENS, "deadline_s": DEADLINE_S, "temperature": 0,
            "max_model_calls": agent.MAX_TOTAL_MODEL_CALLS, "reserved_final_calls": agent.RESERVED_FINAL_CALLS,
            "input_token_budget": agent.INPUT_TOKEN_BUDGET,
            "held_out_sample_seed": SAMPLE_SEED, "held_out_sample_size": SAMPLE_SIZE, "held_out_sample": sample}


def open_graph(product: str, budget: QuestionBudget) -> GraphRAG:
    """The product's graph for one question, its SDK calls charged to the question's budget. Refuses a graph that is
    not a complete, finalized build of every source."""
    problems = build_problems(product, _expected_sources(product))
    if problems:
        raise BuildIncomplete(f"graph {graph_name(product)} is an incomplete build: {'; '.join(problems)}")
    return _facade(product, ARM_ANSWER, budget, max_tokens=ANSWER_MAX_TOKENS, json_messages=True)


def _failed(why: str, budget: UsageLog, envelopes: list) -> ArmResult:
    calls = list(budget.calls)
    use = Usage(sum(c.prompt_tokens for c in calls), sum(c.completion_tokens for c in calls),
                sum(c.cost_usd for c in calls), len(calls))
    return ArmResult(FinalAnswer("run_failed", "", [], [], [], use), calls, envelopes, why)


async def _ask(rag, question: str, budget: QuestionBudget):
    hist = [{"role": "system", "content": ANSWER_SYSTEM}]
    return await asyncio.wait_for(
        rag.completion(question, history=hist, prompt_template=ANSWER_TEMPLATE, return_context=True),
        timeout=max(budget.deadline - budget.clock(), 0.001))


def answer(inp: ExecInput, *, opener: Callable | None = None, model: Callable | None = None,
           clock: Callable[[], float] = time.monotonic) -> ArmResult:
    """Answer one question. `opener`, `model` and `clock` are for tests."""
    budget = QuestionBudget(model, clock)
    if inp.product not in allowed_products():
        return _failed(f"product {inp.product} is not in the pre-registered sample (dev products and "
                       f"{SAMPLE_SIZE} held-out products drawn with seed {SAMPLE_SEED})", budget, [])
    try:
        rag = (opener or open_graph)(inp.product, budget)
    except Exception as e:  # noqa: BLE001 - this question's terminal result
        return _failed(f"{type(e).__name__}: {e}", budget, [])

    async def run():
        try:
            return await _ask(rag, inp.text, budget)
        finally:
            closer = getattr(rag, "close", None)
            if closer:
                await closer()

    try:
        res = asyncio.run(run())
        budget.check_deadline()  # a deadline the SDK swallowed
    except (asyncio.TimeoutError, DeadlineExceeded):
        return _failed(f"deadline exceeded: {DEADLINE_S:.0f} s per question", budget, [])
    except llm.BudgetExceeded:
        return _failed("budget_exceeded", budget, [])
    except Exception as e:  # noqa: BLE001
        return _failed(f"{type(e).__name__}: {e}", budget, [])
    if any("BudgetExceeded" in f.error for f in budget.failures):  # the SDK swallowed a spend-cap refusal
        return _failed("budget_exceeded", budget, [])
    items = [i.content for i in (res.retriever_result.items if res.retriever_result else [])]
    sdk_answer = res.answer or ""
    diag: dict[str, Any] = {"sdk_answer": sdk_answer, "converted": False, "context_items": len(items),
                            "provider_failures": [f._asdict() for f in budget.failures], "budget_stops": budget.stops}

    def done(why: str) -> ArmResult:
        diag.update(model_calls=budget.started, input_tokens=sum(c.prompt_tokens for c in budget.calls))
        return _failed(why, budget, [{"a2": diag}])

    fields, err = agent._parse(sdk_answer)
    if fields is None:
        diag["converted"] = True
        msg = [{"role": "user", "content": CONVERT.format(
            outcomes=", ".join(sorted(agent.MODEL_OUTCOMES)), question=inp.text, answer=sdk_answer)}]
        try:
            r = budget.chat(msg, arm=ARM_ANSWER, max_tokens=ANSWER_MAX_TOKENS, response_format=agent.JSON_FORMAT,
                            final=True)
        except DeadlineExceeded:
            return done(f"deadline exceeded: {DEADLINE_S:.0f} s per question")
        except llm.BudgetExceeded:
            return done("budget_exceeded")
        except BudgetStop as e:
            return done(str(e))
        except Exception as e:  # noqa: BLE001
            return done(f"conversion call failed: {type(e).__name__}: {e}")
        diag["conversion"] = r.message.content or ""
        fields, err = agent._parse(diag["conversion"])
        if fields is None:
            return done(f"SDK answer not convertible to the shared schema: {err}")
    diag["model_fields"] = dict(fields)
    # Shared answer handling: support only from the retrieved context (never the question), complete tokens only,
    # the output guard on the text, then the canonical refusal and the consistency rule.
    seen = Seen()
    context_rows = seen.add(items)
    asked = agent._bare(fields["answer_ids"])
    rows = [context_rows, seen.support(asked, fields["evidence_ids"])]
    text, ids, dropped = guard(fields["answer_text"], asked, rows)
    evidence, ev_dropped = guard_evidence(fields["evidence_ids"], rows)
    diag["dropped"] = dropped + [{"value": v, "where": "evidence", "reason": "not in the retrieved context"}
                                 for v in ev_dropped]
    outcome = fields["outcome"]
    if outcome in agent.REFUSALS and not ids:
        text = agent.refusal_text(["graphrag_retrieval"] if items else [], [])
    if ids and outcome in agent.REFUSALS:
        outcome, text = "path", f"{', '.join(ids)}. {text}".strip()
    elif not ids and not text.strip():
        return done("empty final answer: no answer ids and no text")
    diag.update(model_calls=budget.started, input_tokens=sum(c.prompt_tokens for c in budget.calls))
    calls = list(budget.calls)
    use = Usage(sum(c.prompt_tokens for c in calls), sum(c.completion_tokens for c in calls),
                sum(c.cost_usd for c in calls), len(calls))
    return ArmResult(FinalAnswer(outcome, text, ids, evidence, [], use), calls, [{"a2": diag}])


register(Arm(ARM_ANSWER, answer, sub_cap_usd=ANSWER_CAP_USD, config=arm_config(),
             prompt_versions={"a2": PROMPT_VERSION}))
