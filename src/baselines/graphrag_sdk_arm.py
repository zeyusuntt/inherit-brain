"""A2: GraphRAG-SDK 1.4.0 with its automatic ontology, on the product's allowed records.

Lifecycle, as the SDK ships it: `Ontology.from_sources` drafts an ontology from the source texts, `GraphRAG.ingest`
extracts a graph per source into FalkorDB 6.0.2 (graph `a2:<product>`), `finalize()` deduplicates and embeds, and
`completion()` answers each question (no history, SDK retrieval defaults). Nothing about the SDK is tuned.

Every model and embedding call goes through the metering proxy (`inherit.llm`), arm name `a2-build` while building
and `a2` while answering. The SDK's own retries are switched off (the proxy retries and meters). The answer is asked
for as the shared JSON through the SDK's answer prompt; one conversion call runs only when that reply does not parse,
and it may only restate the SDK's answer.

Source texts (F7), built from the bundle only: one per Slack channel, document and meeting (with its chat), one for
all PRs with their reviews, one for all URLs, plus the employee and customer directories. Each carries stable record ids and timestamps.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Any, Callable, NamedTuple

import httpx

from graphrag_sdk import (
    ChatMessage, ConnectionConfig, Embedder, GraphRAG, LLMInterface, Ontology,
)
from graphrag_sdk.core.models import LLMResponse
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import LatencyBudgetExceededError  # noqa: F401
from graphrag_sdk.ingestion.loaders.base import LoaderStrategy
from graphrag_sdk.core.models import DocumentInfo, DocumentOutput

from eval.arms import Arm, ArmResult, register
from inherit import agent, llm
from inherit.config import ROOT, load_settings
from inherit.contracts import CallUsage, ExecInput, FinalAnswer, Usage
from inherit.data.bundle import BUNDLE_DIR, load_bundle, load_metadata

PROMPT_VERSION = "a2/v1"
ARM_BUILD, ARM_ANSWER = "a2-build", "a2"
BUILD_CAP_USD, ANSWER_CAP_USD = 1.0, 0.3
FALKOR_HOST, FALKOR_PORT = "localhost", 6380  # FalkorDB 6.0.2
EMBED_DIM = 1024
BUILD_MAX_TOKENS, ANSWER_MAX_TOKENS = 4096, agent.MAX_OUTPUT_TOKENS
DEADLINE_S = agent.DEADLINE_S
DISCOVERY_SAMPLE_CHUNKS = 1  # SDK default is 3
DISCOVERY_PER_CATEGORY = 3  # sources per category the ontology is drafted from (F7: discovery sampling is cut first)
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
    return [Source(f"document:{d['id']}",
                   f"Document {d['id']}\nType: {d['type']}\nDate: {d['date']}\nAuthor: {d['author']}\n"
                   f"Link: {d['document_link']}\n\n{d['content']}")
            for d in sorted(product_data.get("documents") or [], key=lambda d: d["id"])]


def _meetings(product_data: dict) -> list[Source]:
    chats = {c["id"]: c["text"] for c in product_data.get("meeting_chats") or []}
    out = []
    for m in sorted(product_data.get("meeting_transcripts") or [], key=lambda m: m["id"]):
        text = (f"Meeting ID: {m['id']}\nDate: {m['date']}\nType: {m.get('document_type', '')}\n"
                f"Participants: {', '.join(m['participants'])}\n\n{m['transcript']}")
        chat = chats.get(f"{m['id']}_chat")
        if chat:
            text += f"\n\nChat of meeting {m['id']}:\n{chat}"
        out.append(Source(f"meeting:{m['id']}", text))
    return out


def _prs(product_data: dict) -> list[Source]:
    """All of the product's PRs with their reviews as one source: a source per PR would be a hundred tiny
    documents, each paying its own discovery summary and ingestion overhead."""
    blocks = []
    for pr in sorted(product_data.get("prs") or [], key=lambda p: p["link"]):
        lines = [f"Pull request {pr['link']}", f"Title: {pr['title']}", f"Author: {pr['user']['login']}",
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
class UsageLog:
    """Thread-safe list of every metered call the SDK made, as CallUsage."""

    def __init__(self) -> None:
        self.calls: list[CallUsage] = []
        self._lock = threading.Lock()

    def add(self, c: CallUsage) -> None:
        with self._lock:
            self.calls.append(c)


class ProxyLLM(LLMInterface):
    """The SDK's LLM, routed through `inherit.llm.chat`. One attempt per call: the proxy retries and meters.

    `json_messages` asks for a JSON object on `ainvoke_messages` only, the call `completion()` uses for the answer;
    the SDK's retrieval-side prompts (keywords, text-to-Cypher) stay free text."""

    def __init__(self, arm: str, log: UsageLog, *, max_tokens: int = BUILD_MAX_TOKENS, json_messages: bool = False):
        super().__init__(model_name=load_settings().model)
        self.arm, self.log, self.max_tokens, self.json_messages = arm, log, max_tokens, json_messages

    def _chat(self, messages: list[dict], response_format: dict | None = None) -> LLMResponse:
        r = llm.chat(messages, arm=self.arm, max_tokens=self.max_tokens, response_format=response_format)
        self.log.add(CallUsage(self.arm, self.model_name, r.prompt_tokens, r.completion_tokens, r.cost_usd,
                               r.latency_s))
        return LLMResponse(content=r.message.content or "")

    def invoke(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return self._chat([{"role": "user", "content": prompt}])

    async def ainvoke(self, prompt: str, *, max_retries: int = 3, timeout: float | None = None, **kw: Any):
        return await super().ainvoke(prompt, max_retries=1, timeout=timeout, **kw)

    async def ainvoke_messages(self, messages: list[ChatMessage], *, max_retries: int = 3,
                               timeout: float | None = None, **kw: Any) -> LLMResponse:
        rf = {"type": "json_object"} if self.json_messages else None
        payload = [{"role": m.role, "content": m.content} for m in messages]
        return await asyncio.to_thread(self._chat, payload, rf)


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
            r = llm.embed(texts[i:i + EMBED_BATCH], arm=self.arm, dimensions=EMBED_DIM)
            self.log.add(CallUsage(self.arm, self._model, r.prompt_tokens, 0, r.cost_usd, 0.0))
            out += r.vectors
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


# ---------------------------------------------------------------- build
def _facade(product: str, arm: str, log: UsageLog, **llm_kw) -> GraphRAG:
    conn = ConnectionConfig(host=FALKOR_HOST, port=FALKOR_PORT, graph_name=graph_name(product),
                            query_timeout_ms=120_000)
    return GraphRAG(conn, ProxyLLM(arm, log, **llm_kw), ProxyEmbedder(arm, log), embedding_dimension=EMBED_DIM)


async def _build(product: str, sources: list[Source], log: UsageLog, report: dict) -> None:
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
    async with _facade(product, ARM_BUILD, log) as rag:
        await rag.set_ontology(ontology)
        sem = asyncio.Semaphore(INGEST_CONCURRENCY)

        async def one(s: Source) -> int:
            async with sem:
                if await rag._graph_store.get_document_record(s.doc_id) is not None:
                    return 0  # already ingested by an interrupted build
                return (await rag.ingest(text=s.text, document_id=s.doc_id)).chunks_indexed

        report["chunks"] = sum(await asyncio.gather(*[one(s) for s in sources]))
        fin = await rag.finalize()
        report["finalize"] = fin.model_dump()
        report["stats"] = await rag.get_statistics()


def build(product: str, bundle_dir=BUNDLE_DIR, *, replace: bool = False) -> dict:
    """Build `a2:<product>` from the bundle. Returns a report: sources, chunks, calls, tokens, cost, seconds."""
    import time

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
    asyncio.run(_build(product, sources, log, report))
    chat_calls = [c for c in log.calls if c.completion_tokens or c.model == load_settings().model]
    report.update(seconds=round(time.monotonic() - t0, 1), calls=len(log.calls),
                  chat_calls=len(chat_calls), prompt_tokens=sum(c.prompt_tokens for c in log.calls),
                  completion_tokens=sum(c.completion_tokens for c in log.calls),
                  cost_usd=round(sum(c.cost_usd for c in log.calls), 4))
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
    return {"graph": "a2:<product>", "model": s.model, "embed_model": s.embed_model, "embed_dim": EMBED_DIM,
            "sdk": "graphrag-sdk 1.4.0", "falkordb_port": FALKOR_PORT, "discovery_sample_chunks": DISCOVERY_SAMPLE_CHUNKS,
            "max_output_tokens": ANSWER_MAX_TOKENS, "deadline_s": DEADLINE_S, "temperature": 0}


def open_graph(product: str) -> tuple[GraphRAG, UsageLog]:
    log = UsageLog()
    return _facade(product, ARM_ANSWER, log, max_tokens=ANSWER_MAX_TOKENS, json_messages=True), log


def _supported(values: list[str], haystack: str) -> list[str]:
    return [v for v in values if v and v in haystack]


def _failed(why: str, calls: list[CallUsage], envelopes: list) -> ArmResult:
    use = Usage(sum(c.prompt_tokens for c in calls), sum(c.completion_tokens for c in calls),
                sum(c.cost_usd for c in calls), len(calls))
    return ArmResult(FinalAnswer("run_failed", "", [], [], [], use), calls, envelopes, why)


async def _ask(rag, question: str):
    hist = [{"role": "system", "content": ANSWER_SYSTEM}]
    return await asyncio.wait_for(
        rag.completion(question, history=hist, prompt_template=ANSWER_TEMPLATE, return_context=True),
        timeout=DEADLINE_S)


def answer(inp: ExecInput, *, opener: Callable | None = None, model: Callable | None = None) -> ArmResult:
    """Answer one question. `opener` and `model` are for tests."""
    model = model or llm.chat
    calls: list[CallUsage] = []
    try:
        rag, log = (opener or open_graph)(inp.product)
    except Exception as e:  # noqa: BLE001 - this question's terminal result
        return _failed(f"{type(e).__name__}: {e}", [], [])

    async def run():
        try:
            return await _ask(rag, inp.text)
        finally:
            closer = getattr(rag, "close", None)
            if closer:
                await closer()

    try:
        res = asyncio.run(run())
    except asyncio.TimeoutError:
        return _failed(f"deadline exceeded: {DEADLINE_S:.0f} s per question", list(log.calls), [])
    except llm.BudgetExceeded:
        return _failed("budget_exceeded", list(log.calls), [])
    except Exception as e:  # noqa: BLE001
        return _failed(f"{type(e).__name__}: {e}", list(log.calls), [])
    calls += log.calls
    sdk_answer = res.answer or ""
    context = "\n".join(i.content for i in (res.retriever_result.items if res.retriever_result else []))
    diag: dict[str, Any] = {"sdk_answer": sdk_answer, "converted": False, "context_items":
                            len(res.retriever_result.items) if res.retriever_result else 0}
    fields, err = agent._parse(sdk_answer)
    if fields is None:
        diag["converted"] = True
        msg = [{"role": "user", "content": CONVERT.format(
            outcomes=", ".join(sorted(agent.MODEL_OUTCOMES)), question=inp.text, answer=sdk_answer)}]
        try:
            r = model(msg, arm=ARM_ANSWER, max_tokens=ANSWER_MAX_TOKENS, response_format=agent.JSON_FORMAT)
        except llm.BudgetExceeded:
            return _failed("budget_exceeded", calls, [{"a2": diag}])
        except Exception as e:  # noqa: BLE001
            return _failed(f"conversion call failed: {type(e).__name__}: {e}", calls, [{"a2": diag}])
        calls.append(CallUsage(ARM_ANSWER, load_settings().model, r.prompt_tokens, r.completion_tokens, r.cost_usd,
                               r.latency_s))
        diag["conversion"] = r.message.content or ""
        fields, err = agent._parse(diag["conversion"])
        if fields is None:
            return _failed(f"SDK answer not convertible to the shared schema: {err}", calls, [{"a2": diag}])
    haystack = context + "\n" + inp.text
    asked = agent._bare(fields["answer_ids"])
    ids = _supported(asked, haystack)
    evidence = _supported(list(dict.fromkeys(fields["evidence_ids"])), context)
    diag["dropped"] = [v for v in asked + fields["evidence_ids"] if v not in ids and v not in evidence]
    outcome, text = fields["outcome"], fields["answer_text"]
    if ids and outcome in agent.REFUSALS:
        outcome, text = "path", f"{', '.join(ids)}. {text}".strip()
    elif not ids and not text.strip():
        return _failed("empty final answer: no answer ids and no text", calls, [{"a2": diag}])
    use = Usage(sum(c.prompt_tokens for c in calls), sum(c.completion_tokens for c in calls),
                sum(c.cost_usd for c in calls), len(calls))
    return ArmResult(FinalAnswer(outcome, text, ids, evidence, [], use), calls, [{"a2": diag}])


register(Arm("A2", answer, sub_cap_usd=ANSWER_CAP_USD, config=arm_config(),
             prompt_versions={"a2": PROMPT_VERSION}))
