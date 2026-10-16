"""The HERB paper's tools in the shared agent loop (arm A0): `retrieve_all` and the eight metadata tools of
metadata_search_functions.py, plus `finish_agent` and `fallback_answer`.

Ported as in the paper: each metadata tool asks the model to pull names, roles, ids or links out of the text it is
given, then looks them up in a dictionary. Differences, all disclosed:

* two upstream bugs fixed: `role_to_id_search` formatted its prompt with `question=` while the template says `{text}`
  (a KeyError on every call), and `id_to_role_search` returned the ids instead of the roles;
* every lookup is scoped to the current product's bundle: PRs, URLs and documents come from this product only
  (upstream scans every product file at import); the employee and customer directories are the bundle's;
* every model call (extraction, `finish_agent`, the fallback synthesis) goes through `chat()` with the calling arm's
  name, under the loop's per-question counter and deadline (`_Loop.nested_chat`);
* a failed verification triggers the paper's fallback answer once (retrieval over the question, one synthesis call);
  there is no second agent attempt;
* tool outputs are JSON envelopes `{"result": ..., "message": ...}`; the message is the upstream "No ... found" text;
* answer support comes only from matched records and retrieved chunks (never from the "No ... found" message or the
  model's own extraction); identifiers and links are supported by exact complete-token equality, other values (company
  names) by a whole-word match in a returned string; a rejected final answer is terminal unless the one fallback
  replaces it.

`CardTools` (A0f, in cards_react.py) overrides `present` so the same tools return entity cards.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from typing import Any

from inherit import agent
from inherit.agent import JSON_FORMAT, AnswerRejected, _parse
from inherit.contracts import TraceOp
from inherit.tools.depth1 import ToolError
from inherit.tools.guard import _TRAIL, ID_RE

PROMPT_VERSION = "baselines/a0/v3"
PAPER_ROLES = ("Marketing Manager, Engineering Lead, VP of Engineering, Software Engineer, QA Specialist, "
               "Product Manager, Technical Architect, UX Researcher, Marketing Research Analyst, Chief Product Officer")

RETRIEVE_DESCRIPTION = ("This tool uses vector search to retrieve information from all available unstructured sources "
                        "including messages, internal documents, meetings and meeting chats. The retrieved information "
                        "will include relevant metadata and content based on the query.")

# (name, argument name, argument description, tool description): verbatim from metadata_search_functions.py
METADATA_TOOLS = (
    ("role_to_id_search", "text", "A text containing some role names such as 'Engineer Lead'.",
     'Given a text snippet, this tool extracts any role names (e.g., "Product Manager") mentioned and returns the '
     'employee IDs associated with those roles.'),
    ("name_to_id_search", "content", "A text snippet containing some employee names such as 'Xiangyu Peng'.",
     "Given a text snippet that contains employee names, this tool returns a list of the corresponding employee IDs "
     "for those names."),
    ("company_to_id_search", "content", "A text snippet containing some company names such as 'Google'.",
     "Given a text chunk that contains customer company names, this tool returns a list of the corresponding "
     "customer IDs for those names."),
    ("id_to_company_search", "content", "A text snippet containing some customer IDs such as 'CUST-0012567'.",
     "Given a text snippet that contains customer IDs, this tool returns the corresponding company's names."),
    ("id_to_name_search", "content", "A text snippet containing some employee IDs such as eid_123456.",
     "Given a text snippet that contains employee IDs, this tool returns the corresponding employee's names."),
    ("id_to_role_search", "content", "A text snippet containing some employee IDs such as 'eid_qwehjweh'.",
     "Given a text snippet that contains employee IDs, this tool returns the corresponding employee's role, e.g., "
     "'Product Manager'."),
    ("pr_search", "content", "A text snippet containing one or more GitHub PR links.",
     "Given a text snippet containing one or more GitHub PR links, this tool returns metadata such as the PR status "
     "and other relevant details for each found PR."),
    ("url_search", "content", "A text snippet containing one or more URLs.",
     "Given a text snippet containing one or more URLs, this tool returns metadata such as the content description "
     "and other relevant information from each found URL."),
)
TOOL_NAMES = ("retrieve_all", *(t[0] for t in METADATA_TOOLS))

EXTRACT_ROLES = ("Extract role names (e.g., ['Engineer Lead', 'UX Designer']) from the given text. "
                 "If no roles are mentioned, return []."
                 "\n\nAvailable roles: " + PAPER_ROLES +
                 "\n\nText: {text}"
                 "\n\nRespond in json format: {{'roles': ['list of roles']}}")


EXTRACT = {
    "name_to_id_search": ("names", "Extract employee names (e.g., 'Becky Peng') from the given text. "
                          "If no names are mentioned, return []."
                          "\n\nText: {content}"
                          "\n\nRespond in json format: {{'names': ['list of employee names']}}"),
    "company_to_id_search": ("names", "Extract company names (e.g., 'Google') from the given text. "
                             "If no names are mentioned, return []."
                             "\n\nText: {content}"
                             "\n\nRespond in json format: {{'names': ['list of company names']}}"),
    "id_to_company_search": ("ids", "Extract customer Ids from the given text. If no IDs are mentioned, return []."
                             "\n\nText: {content}"
                             "\n\nRespond in json format: {{'ids': ['list of customer IDs']}}"),
    "id_to_name_search": ("ids", "Extract employee Ids from the given text. If no IDs are mentioned, return []."
                          "\n\nText: {content}"
                          "\n\nRespond in json format: {{'ids': ['list of employee IDs']}}"),
    "id_to_role_search": ("ids", "Extract employee Ids from the given text. If no IDs are mentioned, return []."
                          "\n\nText: {content}"
                          "\n\nRespond in json format: {{'ids': ['list of employee names']}}"),
    "pr_search": ("links", "Extract GitHub PR links from the given text. If no PR links are found, return []."
                  "\n\nText: {content}"
                  "\n\nRespond in json format: {{'links': ['list of PR links']}}"),
    "url_search": ("links", "Extract URLs from the given text. If no URLs are found, return []."
                   "\n\nText: {content}"
                   "\n\nRespond in json format: {{'links': ['list of URLs']}}"),
}

FINISH_PROMPT = ("Given the question and answer, determine whether the given answer successfully answers the given "
                 "question.\nReply \"Yes\" if the answer sufficiently addresses the given question with detailed "
                 "information.  \nReply \"No\" if the answer fails to address the question due to insufficient context.")
FINISH_SUFFIX = ("\n\nQuestion: {question}\n\nAnswer: {answer}\n\nRespond using the below json format:\n"
                 "{{\"done\": \"yes/no\", \"reason\": \"justification for the decision\"}}")

# llama-index's default question-answering prompt (what RetrieverQueryEngine uses), plus the shared answer format.
FALLBACK_PROMPT = ("Context information is below.\n---------------------\n{context}\n---------------------\n"
                   "Given the context information and not prior knowledge, answer the query.\n"
                   "Query: {question}\n\n{format}")

ANSWER_FORMAT = agent.ANSWER_INSTRUCTIONS  # the one answer-instruction text of every arm; never a copy

CONTEXT = """You are an intelligent assistant designed to answer questions about products using a variety of enterprise data sources.

You will have access to the following information:

* **Employee directory**: Names, employee IDs, and roles.
* **Slack conversations**: Messages exchanged across different channels, including timestamps, employee IDs, and document URLs.
* **Meeting records**: Natural language transcripts with timestamps, meeting IDs, and participant employee IDs.
* **Referenced documents**: Each includes the author's name and ID, timestamp, title, and full content.
* **GitHub pull request descriptions**: Summaries of PRs related to feature development or bug fixes across different products.
* **Web URL summaries**: Descriptions of content retrieved from external URLs.

Your goal is to retrieve and synthesize relevant information from conversations, meetings, documents, and other sources to answer the given product-related question.

To support this:

* **Unstructured search** uses semantic similarity to retrieve relevant Slack messages, documents, meeting transcripts and chats.
* **Structured search** enables precise extraction of metadata related to employees, customers, pull requests, and URLs.
"""

HOW_TO_WORK = """
How to work
1. Find facts with the tools. Never answer from memory or guess: every person, company, link and fact in your answer must come from retrieved text or a tool result.
2. The sources are scoped to the product the question names. A name or identifier the tools cannot find is unknown; do not substitute a similar one.
3. If one name matches several people (namesakes), name every candidate and say the name is ambiguous. Do not pick one.
4. You have at most 12 tool calls and a limited reading budget, so ask for what you need and no more.
5. Stop when you can answer, or when the tools show the answer is not in the sources.

"""


def system_prompt(cards: bool = False) -> str:
    context = CONTEXT
    if cards:
        context += ("\nIn this deployment every source is available as entity cards: JSON objects with `id`, `label`, "
                    "`name`, `props` (the node's own fields) and `neighbours` (the connected nodes with their ids, "
                    "names and the record ids that support each link). `retrieve_all` returns cards, and the "
                    "structured search tools return the matching cards.\n")
    return context + HOW_TO_WORK + ANSWER_FORMAT


# ---------------------------------------------------------------- lookups (one product's bundle plus the directories)
class Lookups:
    """The paper's dictionaries, built from one product's bundle and the bundle's employee and customer files."""

    def __init__(self, bundle: dict, employees: dict, customers: list):
        self.all_roles = set()
        self.role2id: dict[str, list[str]] = defaultdict(list)
        self.name2id: dict[str, list[str]] = defaultdict(list)
        self.id2role: dict[str, str] = {}
        self.id2name: dict[str, str] = {}
        for e in employees.values():
            eid = e["employee_id"]
            self.all_roles.add(e["role"])
            self.role2id[e["role"].lower()].append(eid)
            self.id2role[eid] = e["role"]
            self.id2name[eid] = e["name"]
            self.name2id[e["name"].lower()].append(eid)
            for part in e["name"].split():
                self.name2id[part.lower()].append(eid)
        self.cust2id: dict[str, list[str]] = defaultdict(list)
        self.id2cust: dict[str, str] = {}
        self.cust_ids: dict[str, str] = {}  # lower-case id -> id as written in the directory
        for c in customers:
            self.cust2id[c["company"].lower()].append(c["id"])
            self.id2cust[c["id"].lower()] = c["company"]
            self.cust_ids[c["id"].lower()] = c["id"]
        self.pr_metadata = {p["link"].lower(): p for p in bundle.get("prs", [])}
        self.url_metadata = {u["link"].lower(): u for u in bundle.get("urls", [])}
        for d in bundle.get("documents", []):  # a document wins over a URL with the same link, as upstream
            self.url_metadata[d["document_link"].lower()] = d


# ---------------------------------------------------------------- answer-support helpers
def id_tokens(text: str) -> list[str]:
    """The id-like tokens of a text (the output guard's own pattern)."""
    return list(dict.fromkeys(m.group(0).rstrip(_TRAIL) for m in ID_RE.finditer(text)))


def leaves(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in leaves(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in leaves(v)]
    return []


class Seen:
    """Every string a matched record or a retrieved chunk held this question: the evidence an answer value is checked
    against. Diagnostics ("No ... found" with the model's extraction) are never added."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.tokens: set[str] = set()

    def add(self, strings: list[str]) -> list[dict]:
        """Record the strings; return guard rows naming the ids they contain."""
        self.texts += strings
        toks = list(dict.fromkeys(t for s in strings for t in id_tokens(s)))
        self.tokens.update(toks)
        return [{"refs": toks}] if toks else []

    def has(self, value: str) -> bool:
        """An identifier or link (the guard's own id pattern) is supported only by an identical complete token; any
        other value (a company name) by a whole-word match, with '-', '/' and ':' joined to a word not a boundary."""
        if ID_RE.fullmatch(value):
            return value in self.tokens
        pat = re.compile(rf"(?<![\w/:-]){re.escape(value)}(?![\w]|[-/:]\w)")
        return any(pat.search(t) for t in self.texts)

    def support(self, answer_ids: list[str], evidence_ids: list[str]) -> list[dict]:
        return ([{"answer": a} for a in answer_ids if a and self.has(a)]
                + [{"evidence": e} for e in evidence_ids if e and self.has(e)])


# ---------------------------------------------------------------- the tools
class PaperTools:
    """A0's tool set for `agent.answer(tools_factory=...)`."""

    def __init__(self, loop, lookups: Lookups, retriever, *, cards: bool = False):
        self.loop, self.lk, self.retriever = loop, lookups, retriever
        self.system_prompt = system_prompt(cards)
        self.ops: list[TraceOp] = []
        self.seen = Seen()
        self.support = self.seen.support

    # ---- specs
    def specs(self) -> list[dict]:
        def fn(name, desc, arg, arg_desc):
            return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
                "type": "object", "properties": {arg: {"type": "string", "description": arg_desc}},
                "required": [arg], "additionalProperties": False}}}

        return [fn("retrieve_all", RETRIEVE_DESCRIPTION, "query", "The search query."),
                *(fn(n, d, a, ad) for n, a, ad, d in METADATA_TOOLS)]

    # ---- one call
    def call(self, name: str, args: dict) -> tuple[dict, list[dict]]:
        if name not in TOOL_NAMES:
            raise ToolError(f"unknown tool {name!r}; tools: {', '.join(TOOL_NAMES)}")
        arg = "query" if name == "retrieve_all" else dict((n, a) for n, a, _, _ in METADATA_TOOLS)[name]
        value = args.get(arg)
        if not isinstance(value, str):
            raise ToolError(f"{name} needs a string argument {arg!r}")
        t0 = time.monotonic()
        if name == "retrieve_all":
            env = self.retrieve(value)
            rows = self.chunk_rows(env["chunks"])
        else:
            env, matched = self.lookup(name, value)
            rows = self.matched_rows(env, matched)
        count = len(env.get("chunks") or env.get("result") or [])
        self.ops.append(TraceOp(name, "paper", "", dict(args), count, (time.monotonic() - t0) * 1000))
        return env, rows

    # ---- what supports an answer: matched records and retrieved chunks only
    def chunk_rows(self, chunks: list[str]) -> list[dict]:
        return self.seen.add(list(chunks))

    def matched_rows(self, env: dict, matched: list[str]) -> list[dict]:
        """`matched` holds the strings of the records the lookup really matched (never the extraction or a message)."""
        return self.seen.add(matched)

    def retrieve(self, query: str) -> dict:
        return {"chunks": self.retriever.retrieve(query)}

    # ---- the eight metadata tools
    def extract(self, prompt: str) -> Any:
        """One model call that maps text to a JSON object (upstream `get_gpt4_response(..., response_format="json")`)."""
        r = self.loop.nested_chat([{"role": "user", "content": prompt}], response_format=JSON_FORMAT)
        try:
            return json.loads(r.message.content or "")
        except ValueError:
            return None

    def lookup(self, name: str, text: str) -> tuple[dict, list[str]]:
        """(envelope, strings of the matched records)."""
        lk = self.lk
        if name == "role_to_id_search":
            got = self.extract(EXTRACT_ROLES.format(text=text))  # upstream formatted with question=, a KeyError
            key, table = "roles", lk.role2id
        else:
            key, prompt = EXTRACT[name]
            got = self.extract(prompt.format(content=text))
            table = {"name_to_id_search": lk.name2id, "company_to_id_search": lk.cust2id}.get(name)
        items = got.get(key, []) if isinstance(got, dict) else None
        res: Any
        try:
            if name in ("role_to_id_search", "name_to_id_search", "company_to_id_search"):
                res = {}
                for item in items:
                    ids = table.get(item.lower(), [])
                    if ids:
                        res[item.lower()] = ids
            elif name in ("id_to_name_search", "id_to_role_search"):
                src = lk.id2name if name == "id_to_name_search" else lk.id2role
                res = {i: src[i] for i in items if src.get(i)}  # id_to_role returned the ids upstream
            elif name == "id_to_company_search":
                res = {i: lk.id2cust[i.lower()] for i in items if lk.id2cust.get(i.lower())}
            else:
                src = lk.pr_metadata if name == "pr_search" else lk.url_metadata
                res = [src[x.lower()] for x in items if src.get(x.lower())]
        except (AttributeError, TypeError):
            res = {} if name not in ("pr_search", "url_search") else []
        matched = self.matched_strings(name, res)
        shown = self.present(name, res)
        env: dict = {"result": shown}
        if getattr(self, "truncated", False):
            env["truncated"] = True
            self.truncated = False
        if not shown:
            env["message"] = {"role_to_id_search": "No roles found", "name_to_id_search": "No IDs found",
                              "company_to_id_search": "No IDs found", "id_to_company_search": "No companies found",
                              "id_to_name_search": "No names found", "id_to_role_search": "No roles found",
                              "pr_search": "No PR details found", "url_search": "No URL details found"}[name] \
                + f": {got}"
        return env, matched

    def matched_strings(self, name: str, res: Any) -> list[str]:
        """The strings that support answers: ids and records the dictionaries returned. For the inverse lookups the
        matched employee or customer id (the dictionary key) is a returned id too."""
        lk = self.lk
        if name in ("id_to_name_search", "id_to_role_search"):
            return [s for i, v in res.items() for s in (i, v)]
        if name == "id_to_company_search":
            return [s for i, v in res.items() for s in (lk.cust_ids[i.lower()], v)]
        if name in ("pr_search", "url_search"):
            return leaves(res)
        return leaves(list(res.values()))  # role / name / company to id: the ids (the keys echo the extraction)

    def present(self, name: str, res: Any) -> Any:
        """What the model sees for a lookup result. A0: the paper's records and dictionaries as they are."""
        return res

    # ---- finalisation (product_rag.py: finish_agent, then the fallback query engine)
    def finish_agent(self, question: str, answer: str) -> bool:
        prompt = FINISH_PROMPT + FINISH_SUFFIX.format(question=question, answer=answer)
        r = self.loop.nested_chat([{"role": "user", "content": prompt}], response_format=JSON_FORMAT, final=True)
        try:
            return str(json.loads(r.message.content or "")["done"]).lower() != "no"
        except (ValueError, KeyError, TypeError, AttributeError):
            return False  # upstream raised here and the attempt counted as failed

    def fallback_answer(self, question: str) -> dict | None:
        if not self.loop.room(final=True):  # no call left: do not retrieve for nothing
            raise ToolError("model call limit reached")
        chunks = self.retriever.retrieve(question)
        self.loop.guard_rows.append(self.chunk_rows(chunks))
        prompt = FALLBACK_PROMPT.format(context="\n\n".join(chunks), question=question, format=ANSWER_FORMAT)
        r = self.loop.nested_chat([{"role": "user", "content": prompt}], response_format=JSON_FORMAT, final=True)
        fields, _ = _parse(r.message.content or "")
        return fields

    def rescue(self, loop) -> dict | None:
        """The one fallback answer: a valid answer, or None when no call was left or its reply was not valid."""
        try:
            fb = self.fallback_answer(loop.inp.text)
        except ToolError:
            loop.envelopes.append({"tool": "fallback_answer", "params": {}, "error": "model call limit reached"})
            return None
        loop.envelopes.append({"tool": "fallback_answer", "params": {}, "used": fb is not None})
        return fb

    def review(self, loop, fields: dict) -> dict | None:
        """Verify the final answer. A rejection is terminal unless the fallback answer replaces it (AnswerRejected);
        if no call is left to verify, the answer stands unverified (that is not a rejection)."""
        answer = fields["answer_text"]
        if fields["answer_ids"]:
            answer += "\nAnswer values: " + ", ".join(fields["answer_ids"])
        try:
            ok = self.finish_agent(loop.inp.text, answer)
        except ToolError:
            loop.envelopes.append({"tool": "finish_agent", "params": {}, "error": "model call limit reached"})
            return None
        loop.envelopes.append({"tool": "finish_agent", "params": {}, "done": ok})
        if ok:
            return None
        fb = self.rescue(loop)
        if fb is None:
            raise AnswerRejected("finish_agent said no and no valid fallback answer was available")
        return fb
