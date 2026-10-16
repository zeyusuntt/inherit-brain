"""Layer 2c: semantic extraction from meetings and ordinary Slack sessions (Task 4c).

Order (R14): documents (feedback bullets by rule), meetings (one call each), Slack sessions (planning channels always,
other channels routed by rules), competitor demo links (rule), PR association (exact meeting titles first, then batches
of unmatched PRs), then derived outcomes in code (pending PR actions, feature events from holds and reverts, feature
fate predicates, the completeness manifest).

The model returns verbatim citations and references, never employee ids: a meeting speaker comes from the parsed speaker
of the referenced segment through names.lookup (R12), a Slack contributor from the message author. Every item must carry
a citation that passes cite.validate_items against the records given to that call (R11); losses are counted by reason.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from inherit import llm
from inherit.extract.catalog import _topic, build_catalog
from inherit.extract.cite import SourceRecord, normalise, validate_citation, validate_items
from inherit.extract.llm_extract import Caller
from inherit.extract.names import chat_lines, lookup, resolve_meeting_names
from inherit.extract.rules import named_qa_eids, parse_revert_title, pr_approval, run_rules
from inherit.extract.semantic_prompts import (DOCUMENT_SYSTEM, FEATURE_GROUP_SYSTEM, MEETING_SYSTEM, PR_MEMBERSHIP_SYSTEM,
                                              SEMANTIC_PROMPT_VERSION, SESSION_SYSTEM)
from inherit.extract import semantic_rules as sr
from inherit.extract.sessions import sessions
from inherit.extract.structural import _links
from inherit.facts import Edge, Facts, Node
from inherit.ontology import check_edge

EXTRACTOR = "semantic/v2"
FATE_VERSION = "feature_fate/v1"
PENDING_VERSION = "pending/v1"
PROMPT_VERSION = SEMANTIC_PROMPT_VERSION
SESSION_CHARS = 18000  # about 6,000 input tokens of message text per call
MEMBERSHIP_BATCH = 10
MEETING_ALLOWED = {("segment", "text"), ("chat", "text")}
SESSION_ALLOWED = {("message", "text")}
PR_ALLOWED = {("pr", "title"), ("pr", "summary")}
REVIEW_KINDS = {"suggestion", "review_question"}
CONTRIB_KINDS = {"lead", "assigned", "design", "support"}
EVENT_KINDS = {"discussed", "implemented", "held", "resumed", "reverted"}  # the model's; documents add "proposed"
REASONS = {"security", "product_direction", "introduced_bugs", "other"}
COMPETITOR_KINDS = {"competitor_feature", "competitor_weakness"}
CLAIM_KINDS = COMPETITOR_KINDS | {"customer_issue", "customer_strength", "qa_issue"}
CUSTOMER_KINDS = {"customer_issue", "customer_strength"}
ACTIONS = {"withhold", "resume", "merge", "revert", "question"}
CUST_RE = re.compile(r"\bCUST-\d+\b")
DOC_TYPE_RE = re.compile(sr.DOC_WORDS, re.IGNORECASE)
PRODUCT_PLANNING = "product-planning"
DOC_ALLOWED = {("document", "content")}
# Completed-action evidence (C1): a completion word in a sentence with no promise, plan, condition or negation.
_COMPLETE = re.compile(r"\b(?:completed?|merged|finished|done|shipped|delivered|deployed|launched|implemented|released"
                       r"|resolved|wrapped up|went live|is live|are live|now live|in production)\b", re.I)
_NOT_DONE = re.compile(r"\b(?:will|shall|going to|plan(?:s|ning|ned)? to|to be|should|would|could|can|need(?:s)? to|want"
                       r" to|hope|aim|next (?:week|sprint|month)|soon|once|when|if|after|before|until|unless|yet|never"
                       r"|not|ready for)\b|'ll\b|n't\b", re.I)
_SENTENCES = re.compile(r"(?<=[.!?])\s+")
_POLITE = re.compile(r"^\W*(?:and |so |but )?(?:could|can|would|will) you\b", re.I)
_Q_START = re.compile(r"^\W*(?:and |so |but )?(?:should|shall|do|does|did|is|are|what|how|why|any|has|have|"
                      r"(?:can|could|would) we|is there)\b", re.I)


def completed(text: str) -> bool:
    """True when a sentence states finished work: a completion word and no promise, plan, condition or negation."""
    return any(_COMPLETE.search(x) and not _NOT_DONE.search(x) for x in _SENTENCES.split(text or ""))


# A document sentence that says the capability exists or was added is conflicting evidence for "not implemented".
_EXISTS = re.compile(r"\b(?:has|have|had) been (?:added|implemented|introduced|integrated|delivered)\b|\bnow "
                     r"(?:supports?|includes?|offers?|provides?)\b|\bincludes\b|\bwas (?:added|implemented|introduced)\b",
                     re.I)
# Documents shared as the latest or new draft describe the product as it is, so present-tense capability verbs count.
_DESCRIBES = re.compile(r"\b(?:offers?|provides?|features?|supports?|leverages?|enables?)\b", re.I)
_ROLLED_BACK = re.compile(r"\b(?:reverted|rolled back|backed out|was undone|were undone)\b", re.I)
_REVIEW_CUE = re.compile(
    r"\b(?:should|could|would|add|include|consider|suggest|might|recommend|needs?|perhaps|maybe|expand|clarify|"
    r"elaborate|mention|highlight|detail|specify|emphasi[sz]e|improve|update|replace|remove|worth|how about|"
    r"what about)\b", re.I)


def quote_sentences(text: str, quote: str) -> str:
    """The sentence(s) of `text` that hold the cited quote; the whole quote when it is not found (C1)."""
    q = normalise(quote)
    hit = [x for x in _SENTENCES.split(text or "") if q and q in normalise(x)]
    if not hit and q:
        head = " ".join(q.split()[:4])
        hit = [x for x in _SENTENCES.split(text or "") if head in normalise(x)]
    return " ".join(hit) or quote


def rolled_back(text: str) -> bool:
    """A sentence states a finished rollback: a reverted/rolled back word and no promise, plan, condition or negation."""
    head = lambda x: re.sub(r"\b(?:after|before)\b.*", "", x, flags=re.I)  # noqa: E731  "...yesterday after the audit"
    return any(_ROLLED_BACK.search(x) and not _NOT_DONE.search(head(x)) for x in _SENTENCES.split(text or ""))


def _is_question(sentence: str) -> bool:
    x = sentence.strip()
    return False if _POLITE.match(x) else (x.endswith("?") or bool(_Q_START.match(x)))


def sentence_mood(text: str, cue) -> str:
    """plain, question or conditional for the sentences of `text` that carry `cue` (I4). Any plain sentence makes it plain;
    a polite request ("could you please ensure it is not merged?") is plain."""
    moods = set()
    for x in _SENTENCES.split(text or ""):
        if cue.search(x):
            moods.add("question" if _is_question(x) else "conditional" if re.search(r"\bif\b", x, re.I) else "plain")
    return "plain" if "plain" in moods or not moods else ("question" if "question" in moods else "conditional")


def _shape(arrays, required=()):
    """A response check (I1): each named field is a list of objects whose citations are lists of objects; a required
    field must be present, and a response with none of its fields at all is not an answer."""
    def check(data):
        for key in required:
            if key not in data or data[key] is None:
                return f"{key} is missing"
        if arrays and not any(k in data for k in arrays):
            return "none of the expected fields is present"
        for key in arrays:
            v = data.get(key)
            if v is None:
                continue
            if not isinstance(v, list) or any(not isinstance(i, dict) for i in v):
                return f"{key} is not a list of objects"
            for i in v:
                c = i.get("citations")
                if c is not None and (not isinstance(c, list) or any(not isinstance(x, dict) for x in c)):
                    return f"{key} citations are malformed"
        return None
    return check


MEETING_SHAPE = _shape(("document_reviews", "features", "contributions", "development_links", "feature_events"),
                       required=("features", "contributions", "development_links", "feature_events"))
SESSION_SHAPE = _shape(("reviews", "claims", "pr_decisions"))
DOCUMENT_SHAPE = _shape(("features", "claims"))
MEMBERSHIP_SHAPE = _shape(("assignments",), required=("assignments",))
GROUP_SHAPE = _shape(("groups",), required=("groups",))
# Wording a cited record must contain for a model feature event of that kind (the model over-reads plans as results).
EVENT_CUES = {
    "held": re.compile(r"\bhold\b|on hold|delay|postpon|paus|put off|defer|\bdrop|shelve|halt|cancel|scrap|\bstop"
                       r"|not (?:to )?(?:merge|proceed|move forward|go ahead)|won't|hold off", re.I),
    "resumed": re.compile(r"resum|move forward|proceed|go ahead|restart|reinstat|back on|pick (?:it )?back|continue"
                          r"|reopen", re.I),
    "reverted": re.compile(r"revert|roll(?:ed)? back|\bundo|backed out", re.I),
}
COMP_CUE = re.compile(r"competitor|\brivals?\b|competing", re.IGNORECASE)
# A PR decision's action needs its own wording in the deciding message; a reason needs support in the session so far.
ACTION_CUES = {
    "withhold": re.compile(r"not (?:be )?merg|n't (?:be )?merg|\bhold|unmerged|best not|withhold|not proceed|not going to"
                           r"|shouldn't|should not|\bclose|\bstop|\bpaus", re.I),
    "resume": re.compile(r"merg|go ahead|proceed|move forward|resum|continue|reinstat|back on", re.I),
    "merge": re.compile(r"merg|go ahead|proceed|move forward", re.I),
    "revert": re.compile(r"revert|roll(?:ed)? back|\bundo", re.I),
}
REASON_CUES = {
    "product_direction": re.compile(r"direction|priorit|strateg|roadmap|pivot|\bfocus|shift|no longer", re.I),
    "security": re.compile(r"secur|vulnerab|exploit|breach", re.I),
    "introduced_bugs": re.compile(r"\bbugs?\b|crash|fail|regress|degrad|errors?\b", re.I),
}
# A lead or assignment cited on the segment where someone is asked, not on the answer, is the requester's line.
REQUEST_CUE = re.compile(r"\b(?:can|could|would|will) you\b|\byou(?:'re| are) (?:assigned|on|responsible|going to|the one)"
                         r"|\bcan we get you\b", re.I)


@dataclass
class SemanticResult:
    facts: Facts
    report: dict


def _str(x) -> str:
    return x.strip() if isinstance(x, str) else ""


def _list(x) -> list:
    """A model field that should be a list: a lone string or number becomes a one-item list, anything else empty."""
    return x if isinstance(x, list) else ([x] if isinstance(x, (str, int)) and not isinstance(x, bool) else [])


def _items(data, key) -> list[dict]:
    v = data.get(key) if isinstance(data, dict) else None
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


# ---------------------------------------------------------------- prompts

def _display(emp: dict, eid: str) -> str:
    return (emp.get(eid) or {}).get("name") or "unknown"


def _header(rec, cat) -> list[str]:
    also = f" (also called: {', '.join(cat.aliases)})" if cat.aliases else ""
    return [f"PRODUCT: {rec.product}{also}"]


def build_meeting_request(mt, rec, cat, chats, known_titles) -> tuple[list[dict], int]:
    lines = _header(rec, cat) + [f"MEETING: {mt.id}", f"DATE: {mt.date}"]
    if mt.document_type and mt.document_type != PRODUCT_PLANNING:
        lines.append(f"This is a DOCUMENT REVIEW meeting about the {mt.document_type}.")
        docs = cat.docs_of_type(mt.document_type)
        if docs:
            lines.append("DOCUMENTS OF THIS TYPE: " + "; ".join(f"{d.id} ({d.link})" for d in docs))
    else:
        lines.append("This is a PRODUCT PLANNING meeting.")
    lines.append("KNOWN FEATURES: " + ("; ".join(known_titles) if known_titles else "none"))
    lines += ["", "SEGMENTS:"]
    for s in mt.segments:
        lines.append(f"[{s.id}] {s.speaker}: {s.text}" if s.speaker else f"[{s.id}] {s.text}")
    clines = [(f"{c.id}@{k}", sp, tx) for c in chats for k, (sp, tx) in enumerate(chat_lines(c.text), 1)]
    if clines:
        lines += ["", "CHAT LINES:"] + [f"[{ref}] {sp}: {tx}" for ref, sp, tx in clines]
    max_tokens = min(8000, 2400 + 80 * (len(mt.segments) + len(clines)))
    return [{"role": "system", "content": MEETING_SYSTEM}, {"role": "user", "content": "\n".join(lines)}], max_tokens


def build_session_request(chunk, rec, cat, competitors, anchors) -> tuple[list[dict], int]:
    real = [m for m in chunk if not m.is_bot]
    lines = _header(rec, cat)
    if cat.previous_names:
        lines.append("NAMES OF THE PREVIOUS RELEASE: " + ", ".join(cat.previous_names))
    lines += [f"CHANNEL: {real[0].channel_name}", "DOCUMENTS:"]
    lines += [f"- {d.id} | {d.type} | {d.link}" for d in cat.documents] or ["- none"]
    lines.append("KNOWN COMPETITORS: " + (", ".join(competitors) if competitors else "none"))
    flagged = [m.id for m in real if m.id in anchors]
    if flagged:
        lines.append("RULES FLAGGED (these messages say last or previous release): " + ", ".join(flagged))
    lines += ["", "MESSAGES (time order):"]
    lines += [f"[{m.id}] {m.ts} {_display(rec.employees, m.user_id)} ({m.user_id}): {m.text}" for m in real]
    return ([{"role": "system", "content": SESSION_SYSTEM}, {"role": "user", "content": "\n".join(lines)}],
            min(4000, 800 + 80 * len(real)))


def build_document_request(d, rec, cat) -> tuple[list[dict], int]:
    lines = _header(rec, cat) + [f"DOCUMENT: {d.id}", f"TYPE: {d.type}", f"DATE: {d.date}", "", "CONTENT:", d.content]
    return ([{"role": "system", "content": DOCUMENT_SYSTEM}, {"role": "user", "content": "\n".join(lines)}],
            min(4000, 800 + len(d.content) // 4))


def build_membership_request(batch, features, examples=()) -> tuple[list[dict], int]:
    lines = ["FEATURES:"]
    for i, f in enumerate(features, 1):
        det = "; ".join(f["details"][:6])
        lines.append(f"F{i}: {f['title']}" + (f" | {det}" if det else ""))
    if examples:
        lines += ["", "EXAMPLES (PRs already assigned from the meetings):"] + [f"- {t} -> F{k}" for t, k in examples]
    lines += ["", "PRS:"]
    for i, p in enumerate(batch, 1):
        lines.append(f"P{i}: {p['title']} | {' '.join(p['summary'].split())[:200]}")
    return ([{"role": "system", "content": PR_MEMBERSHIP_SYSTEM}, {"role": "user", "content": "\n".join(lines)}],
            min(3000, 400 + 90 * len(batch)))


def _item_text(item: dict, supplied) -> str:
    """Text a lost item (failed validation, unresolved reference) speaks about: its own words and the cited records."""
    own = " ".join(str(item.get(k) or "") for k in ("title", "text", "pr"))
    quotes = " ".join(str(c.get("quote") or "") for c in (item.get("citations") if isinstance(item.get("citations"), list)
                                                           else []) if isinstance(c, dict))
    extra = ""
    if item.get("kind") == "implemented":  # a record that states completed work names the feature it is about
        ids = {c.get("record_id") for c in (item.get("citations") if isinstance(item.get("citations"), list) else [])
               if isinstance(c, dict)}
        extra = " ".join(x for r in supplied if r.id in ids for x in _SENTENCES.split(r.text) if completed(x))
    return own + " " + quotes + " " + extra


def _could_carry_implementation(item: dict, cat) -> bool:
    """A rejected event of kind implemented or reverted, or a link to a merged PR, could have carried evidence."""
    if item.get("kind") in ("implemented", "reverted"):
        return True
    t = _str(item.get("pr"))
    return bool(t) and any(p.merged is True for p in cat.prs if p.title == t or p.link == t)


def _validated(kind, items, supplied, allowed, drops, note_loss, ref_text=None, impl=None) -> list[dict]:
    """validate_items item by item: valid items are returned, each lost item is counted and noted (R8). `ref_text`
    names the feature an item refers to, so the loss counts against that feature."""
    ok = []
    for it in items:
        good, bad = validate_items([it], supplied, allowed)
        if good:
            ok += good
        else:
            for reason, n in bad.items():
                drops[kind][reason] += n
            note_loss(_item_text(it, supplied) + " " + (ref_text(it) if ref_text else ""),
                      impl=bool(impl and impl(it)))
    return ok


def _title_context(mt, title: str) -> tuple[str, str]:
    """(the segment where a PR title is spoken, the segment before it): the task it is discussed under (C4)."""
    t = normalise(title)
    for i, sg in enumerate(mt.segments):
        if t and t in normalise(sg.text):
            return sg.text, mt.segments[i - 1].text if i else ""
    return "", ""


def _context_score(ctx: tuple[str, str], tokens: set[str]) -> int:
    """Feature-name words in the title's own segment count double against the segment before it."""
    words = lambda x: set(re.findall(r"[a-z0-9]{4,}", x.casefold()))  # noqa: E731
    return 2 * len(words(ctx[0]) & tokens) + len(words(ctx[1]) & tokens)


# Agenda and assignment wording: the segment says which task is being discussed (13f).
_TASK_CUE = re.compile(
    r"\btasks?\b|\bassign\w*|\bpending\b|\blastly\b|\bfirst up\b|\bnext up\b|\bmoving on\b|\brelated pr\b"
    r"|\bpr (?:for|on|to)\b|\bresponsible\b|\bwork(?:ing)? on\b", re.I)


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z0-9]+", text.casefold()) if len(w) >= 4}


def explicit_task_features(title: str, meetings, names_of: dict[str, list[str]]) -> list[dict]:
    """Features a meeting segment names as the task a PR implements. For each segment that speaks the PR title, the
    segment without the title words must carry task wording (task, assigned, pending, lastly, related PR ...) and name
    exactly one best feature: at least two of its stems and two thirds of a feature name's stems appear. The words of
    the title itself are removed first, so they cannot pick a feature by overlap. One row per feature found, with the
    segment that says it. `names_of`: feature id -> its title and aliases."""
    nt = normalise(title)
    out: dict[str, dict] = {}
    for mt in meetings:
        for sg in mt.segments:
            text = normalise(sg.text)
            if not nt or nt not in text:
                continue
            rest = text.replace(nt, " ")
            if not _TASK_CUE.search(rest):
                continue
            have = _stems(rest)
            scored = []
            for fid, names in names_of.items():
                cover = [len(st & have) / len(st) if len(st) >= 2 else 0.0 for st in map(_stems, names)]
                best = max(cover, default=0.0)
                if best >= 2 / 3:  # a feature named by its title outranks one named only by an alias of it
                    scored.append(((best, cover[0]), fid))
            scored.sort(reverse=True)
            if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                out.setdefault(scored[0][1], {"feature": scored[0][1], "meeting": mt.id, "date": mt.date,
                                              "context": (sg.text, ""),
                                              "citations": [{"record_id": sg.id, "quote": title, "field": "text"}]})
    return list(out.values())


def _names_in(text: str, names) -> set[str]:
    t = " " + normalise(text) + " "
    return {n for n in names if re.search(r"(?<![a-z0-9])" + re.escape(normalise(n)) + r"(?![a-z0-9])", t)}


# A first-person or collaborative commitment to do a task, gather input or work together: it proposes nothing about
# the document's content ("I'll handle the X section", "I'll gather feedback", "maybe we can collaborate on Y").
_TASK_TAKING = re.compile(
    r"\b(?:i'?ll|i will|i can|i'?m going to|i am going to|i'?ve got|we'?ll|we will|we can|let'?s|(?:maybe|perhaps) we can"
    r"|(?:can|could|would|will) you)\b[^.!?]{0,40}?\b(?:handle|work(?:ing)? on|draft(?:ing)?|gather(?:ing)?|collect(?:ing)?"
    r"|prepare|take (?:the lead|care|charge|on|over|this|that|it)|start(?:ed|ing)?|focus(?:ing)? on|writ(?:e|ing)"
    r"|put together|collaborat\w*|sync(?:ing)?|assist|help|pair|research|cover|look(?:ing)? into|set up|schedule"
    r"|own)\b", re.I)
# Wording that proposes a change to content. A purpose clause ("...to include in the document") describes the
# gathering, not a change, and does not count.
_CONTENT_CUE = re.compile(
    r"\b(?:should|could|would|consider|suggest\w*|recommend\w*|add(?:s|ed|ing)?|includ(?:e|es|ing)|replace|remove|expand"
    r"|clarif\w+|elaborate|mention|highlight|detail|specify|emphasi[sz]e|improve|update|worth|how about|what about)\b", re.I)
_PURPOSE = re.compile(r"\bto (?:includ\w*|add|incorporate|mention|put|use)\b[^.!?]*", re.I)


def _task_statement(sentence: str) -> bool:
    """True for a sentence that only takes on, gathers or shares a task and proposes no change to the content."""
    return bool(_TASK_TAKING.search(sentence)) and not _CONTENT_CUE.search(_PURPOSE.sub(" ", sentence))


def cited_remark(text: str, citations, record_id: str) -> str:
    """The sentence(s) of the record that the extracted remark cites; the whole text when nothing in it is cited. A
    review is judged on what was cited, not on an unrelated sentence of the same message."""
    hits = [quote_sentences(text, c["quote"]) for c in citations or []
            if isinstance(c, dict) and c.get("record_id") == record_id and c.get("quote")]
    return " ".join(dict.fromkeys(hits)) or text


def is_review(kind: str, text: str) -> bool:
    """A review question is asked (a question mark); a suggestion carries suggesting wording. An opener, thanks, or a
    statement that only takes on or shares a task is neither. The text is the cited remark, which is often one sentence
    of a longer message, so it counts when it is either a suggestion or a question: the model's label between the two
    does not decide it."""
    for sentence in _SENTENCES.split(text or ""):
        if _task_statement(sentence):
            continue
        if "?" in sentence:
            return True
        if kind != "review_question" and (_REVIEW_CUE.search(sentence) or _TASK_TAKING.search(sentence)):
            return True  # "I'll start by adding a roadmap section"
        if kind == "review_question" and _REVIEW_CUE.search(sentence):
            return True
    return False


# ---- whose document a review is about (13f): the subject is read before the type
_DOC_I = "(?i:" + sr.DOC_WORDS + ")"
_NAME = r"[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*"
_LEAD_SUBJECT = re.compile(rf"(?<![\w-])({_NAME})(?:['\u2019]s)?\s+({_DOC_I})")  # X-Genie Product Vision Document
_TRAIL_SUBJECT = re.compile(rf"({_DOC_I})\s+(?:of|from|by|for)\s+(?:the\s+)?({_NAME})")  # Product Vision Document of X-Genie
_OUR_DOC = re.compile(rf"\b(?:our|my|your)\s+(?:(?:own|final|latest|new|updated|current|draft|revised)\s+)*({_DOC_I})", re.I)
_NOT_A_SUBJECT = frozenset(
    "the our this that these those new latest final updated draft initial current first next previous prior last old older "
    "revised full complete entire whole existing proposed original overall main my your their its each every any all a an "
    "review update discuss let hi hey please thanks check read share shared see send finalize finish approve slack "
    "salesforce tableau twitter github google microsoft".split())


def _alnum(x: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", (x or "").casefold()))


def _doc_subjects(text: str, own: set[str], known_external: set[str]) -> list[tuple[int, str, str]]:
    """(position, document type, 'external'|'ours') for each mention of a document type in `text` that names whose it
    is: an external product ("the X-Genie Product Vision Document", "the Product Vision Document of X-Genie"), our own
    product by name, or "our ...". A mention that names nobody is not listed."""
    out: list[tuple[int, str, str]] = []

    def judge(name: str, words_before: list[str], words_after: list[str], initial: bool) -> str | None:
        words = [*words_before[-4:]] if words_before else []
        if any(_alnum(" ".join(words[i:])) in own for i in range(len(words))):
            return "ours"
        if any(_alnum(" ".join(words_after[:n])) in own for n in range(1, 5) if words_after):
            return "ours"
        key = _alnum(name)
        if key in known_external:
            return "external"
        if (name.lower() in _NOT_A_SUBJECT or len(name) < 3 or not re.search(r"[a-z]", name)
                or (initial and re.fullmatch(r"[A-Z][a-z]+", name))):
            return None  # a verb opening the sentence, a version word, an acronym: not a product name
        return "external"

    for m in _LEAD_SUBJECT.finditer(text):
        before = re.findall(r"[A-Za-z0-9-]+", text[:m.end(1)])
        prefix = text[:m.start(1)]
        initial = not prefix.strip() or bool(re.search(r"[.!?:\n]\s*$", prefix))
        who = judge(m.group(1), before, [], initial)
        if who:
            out.append((m.start(2), _alnum(m.group(2)), who))
    for m in _TRAIL_SUBJECT.finditer(text):
        after = re.findall(r"[A-Za-z0-9-]+", text[m.start(2):])[:4]
        who = judge(m.group(2), [], after, False)
        if who:
            out.append((m.start(1), _alnum(m.group(1)), who))
    for m in _OUR_DOC.finditer(text):
        out.append((m.start(1), _alnum(m.group(1)), "ours"))
    return sorted(out)


def resolve_review_docs(cat, texts_upto: list[str], ref: str, prev=frozenset(), *, later_versions: bool = False):
    """(documents, why) for a review (I2). A document link in the session, or a model reference the text supports,
    identifies one document. Otherwise the document's subject comes first: a thread about an external product's
    document (a competitor's, named in the text and not switched back to ours) is not about any document of ours.
    Then the type the text names gives every document of that type the text could mean, narrowed only by a product
    stem the text names. Release-scope words ("latest release", "new version") say which release a remark is about,
    not which document version, so they never narrow; several candidates are marked `version_ambiguous` by the caller.
    `later_versions` (a meeting): a linked document is the version shared in the meeting, and the revisions of the
    same type dated after it carry the remarks, so they are candidates too. `prev` is kept for callers and unused."""
    doc = None
    for t in texts_upto:
        for link in sorted(_links(t)):
            doc = cat.doc_by_link(link) or doc
    if doc is not None:
        if later_versions:
            later = sorted((d for d in cat.documents if normalise(d.type) == normalise(doc.type) and d.date > doc.date),
                           key=lambda d: (d.date, d.id))
            if later:
                return [doc, *later], "link_and_later_versions"
        return [doc], "link"
    text = " ".join(texts_upto)
    cand = cat.doc(ref) if ref else None
    if cand is not None and (cand.id in text or cand.link in text):
        return [cand], "reference"
    dtype = None
    for t in reversed(texts_upto):
        hit = list(DOC_TYPE_RE.finditer(t))
        if hit:
            dtype = hit[-1].group(0)
            break
    if dtype is None:
        return [], "no_document_named"
    own = {_alnum(n) for n in (cat.product, *cat.aliases) if _alnum(n)}
    known_external = {_alnum(e["name"]) for e in cat.competitors.entries.values()}
    subject: dict[str, str] = {}
    for t in texts_upto:
        for _, dt, who in _doc_subjects(t, own, known_external):
            subject[dt] = who
    if subject.get(_alnum(dtype)) == "external":
        return [], "external_product_document"
    cands = [d for d in cat.documents if normalise(d.type) == normalise(dtype)]
    low = " " + normalise(text) + " "
    named = {d.stem for d in cands if d.stem and re.search(r"(?<![a-z0-9])" + re.escape(d.stem) + r"(?![a-z0-9])", low)}
    cands = [d for d in cands if d.stem in named] or cands
    return cands, ("only_document_of_type" if len(cands) == 1 else "version_ambiguous")


# ---------------------------------------------------------------- the run

def run_semantic(rec, *, chat=llm.chat, cache_root=None, dry: bool = False) -> SemanticResult:
    product = rec.product
    rules_facts, rules_rep = run_rules(rec)
    cat = build_catalog(rec).with_previous(rules_rep["previous_names"])
    anchors = set(rules_rep["anchor_messages"])
    _, name_rows = resolve_meeting_names(rec)
    caller = Caller(chat, cache_root, PROMPT_VERSION, dry=dry)
    own_names = [product, *cat.aliases]
    own_keys = {normalise(n) for n in own_names}
    doc_prev = {n.id.split(":", 1)[1] for n in rules_facts.nodes
                if n.label == "Document" and n.props.get("release_scope") == "previous"}

    prs_raw = {p.link: p for p in rec.prs}

    drops: dict[str, Counter] = defaultdict(Counter)
    failures: list[dict] = []
    unresolved_speakers: list[dict] = []
    nodes: list[Node] = []
    edges: list[Edge] = []
    people: set[str] = set()
    customers_seen: set[str] = set()
    state = {"unresolved_feature_refs": 0, "split_sessions": 0, "scope_downgraded": 0}
    units_lost: list[dict] = []  # R8: failed units and validation losses, each with the text it speaks about

    def note_loss(text, *, kind="loss", unit=None, everywhere=False, impl=False):
        """`impl`: the lost item could have carried implementation evidence (it can flip "not implemented")."""
        units_lost.append({"kind": kind, "unit": unit, "raw": normalise(text), "flat": _topic(text),
                           "everywhere": everywhere, "impl": impl})

    unresolved_reviews: list[dict] = []

    def edge(type_, src, dst, ids, key="", **props):
        e = Edge(type_, src, dst, props, tuple(ids), EXTRACTOR, key)
        check_edge(e)
        edges.append(e)

    def person(eid):
        if eid not in people:
            people.add(eid)
            emp = rec.employees.get(eid)
            nodes.append(Node("Person", f"Person:{eid}", {
                "eid": eid, "name": emp["name"] if emp else None, "role": emp["role"] if emp else None,
                "in_directory": emp is not None}))
        return f"Person:{eid}"

    def customer(cid):
        if cid not in customers_seen:
            customers_seen.add(cid)
            c = rec.customers.get(cid, {})
            nodes.append(Node("Customer", f"Customer:{cid}", {"id": cid, "company": c.get("company"),
                                                              "name": c.get("name"), "role": c.get("role")}))
        return f"Customer:{cid}"

    def lose(kind, bad: dict):
        for reason, n in bad.items():
            drops[kind][reason] += n

    # collected facts
    feature_info: dict[str, dict] = {}  # feature id -> {title, details, citations, records}
    discusses: set[tuple[str, str]] = set()  # (Meeting|Message id, Feature|Document id)
    contributions: list[dict] = []
    reviews: list[dict] = []
    fevents: list[dict] = []
    dev_links: dict[str, list[dict]] = defaultdict(list)  # pr link -> [{feature, basis, citations, record}]
    implemented_pr: dict[tuple[str, str], str] = {}  # (feature id, pr link) -> basis
    observations: list[dict] = []
    decisions: list[dict] = []
    competitor_alias: dict[str, set] = defaultdict(set)
    failed_sessions: list[dict] = []  # {first_message, reason, text}
    session_sums = {"planning": Counter(), "other": Counter(), "excluded": Counter()}
    excluded_ids: dict[str, list[str]] = defaultdict(list)

    # ---- 1. documents: anonymous feedback bullets
    n_feedback_docs = n_items = 0
    for d in rec.documents:
        items = sr.parse_feedback_items(d.feedback)
        if items:
            n_feedback_docs += 1
            n_items += len(items)
            nodes.append(Node("Document", f"Document:{d.id}", {"id": d.id, "feedback_items": items}))

    # ---- 2. meetings, one call each, in time order
    chats_of: dict[str, list] = defaultdict(list)
    for c in rec.chats:
        if c.meeting_id:
            chats_of[c.meeting_id].append(c)
    meeting_failed: list[str] = []

    def register_feature(title, details, cites, record, source):
        fid = cat.register_feature(title, citations=cites)
        e = feature_info.setdefault(fid, {"title": cat.features.entries[fid]["title"], "details": [], "records": [],
                                          "sources": set()})
        e["sources"].add(source)
        for dt in details:
            if dt not in e["details"]:
                e["details"].append(dt)
        if record not in e["records"]:
            e["records"].append(record)
        return fid

    def pr_entries(ref, text_norm):
        """Product PRs a meeting item names: a link, else every PR with exactly that title (R3)."""
        ref = _str(ref)
        if sr.PR_LINK.search(ref):
            p, why = sr.resolve_pr(ref, cat)
            return ([p] if p else []), why
        if normalise(ref) not in text_norm:
            return [], "title_not_in_meeting"
        hits = [p for p in cat.prs_by_title(ref, "product") if parse_revert_title(p.title) is None]
        return hits, ("ok" if hits else "unknown")

    for mt in sorted(rec.meetings, key=lambda m: (m.date, m.id)):
        mchats = chats_of.get(mt.id, [])
        known = [e["title"] for e in cat.features.entries.values()]
        req, mt_tokens = build_meeting_request(mt, rec, cat, mchats, known)
        data, reason = caller.json_call("meeting", req, mt_tokens, check=MEETING_SHAPE)
        if data is None:
            failures.append({"kind": "meeting", "id": mt.id, "reason": reason})
            meeting_failed.append(mt.id)
            note_loss(" ".join(sg.text for sg in mt.segments) + " " + " ".join(c.text for c in mchats),
                      kind="meeting", unit=mt.id)
            continue
        supplied = [SourceRecord(s.id, "segment", "text", s.text) for s in mt.segments]
        supplied += [SourceRecord(c.id, "chat", "text", c.text) for c in mchats]
        speaker_of = {s.id: s.speaker for s in mt.segments}
        record_of = {s.id: s.id for s in mt.segments}
        for c in mchats:
            for k, (sp, _) in enumerate(chat_lines(c.text), 1):
                speaker_of[f"{c.id}@{k}"] = sp
                record_of[f"{c.id}@{k}"] = c.id
        text_norm = normalise(" ".join(r.text for r in supplied))
        local: dict[str, str] = {}

        def ref_text(item):
            fid_ = local.get(_str(item.get("feature_ref"))) or cat.resolve_feature(_str(item.get("feature_ref")))
            if fid_ and fid_ in cat.features.entries:
                return " ".join([cat.features.entries[fid_]["title"], *cat.features.entries[fid_]["aliases"]])
            return _str(item.get("feature_ref"))

        def feature_ref(ref, item=None):
            fid = local.get(_str(ref)) or (cat.resolve_feature(_str(ref)) if _str(ref) else None)
            if fid is None:
                state["unresolved_feature_refs"] += 1
                note_loss(_item_text(item or {}, supplied), impl=_could_carry_implementation(item or {}, cat))
            return fid

        ok = []
        for f in _items(data, "features"):
            good, bad = validate_items([f], supplied, MEETING_ALLOWED)
            if good:
                ok += good
            elif bad.get("no_citation") and cat.resolve_feature(_str(f.get("title"))) and _str(f.get("local_id")):
                # the model listed a known feature without evidence from this meeting: a reference, not a mention
                local[_str(f["local_id"])] = cat.resolve_feature(_str(f.get("title")))
                state["known_feature_references"] = state.get("known_feature_references", 0) + 1
            else:
                lose("features", bad)
                state["unresolved_feature_refs"] += 1
                note_loss(_item_text(f, supplied), impl=True)  # the feature itself is missing: its evidence is too
        for f in ok:
            title = _str(f.get("title"))
            if not title:
                drops["features"]["no_title"] += 1
                continue
            details = [_str(x) for x in _list(f.get("details")) if _str(x)]
            fid = register_feature(title, details, f["citations"], mt.id, "meeting")
            if _str(f.get("local_id")):
                local[_str(f["local_id"])] = fid
            discusses.add((f"Meeting:{mt.id}", fid))

        ok = _validated("contributions", _items(data, "contributions"), supplied, MEETING_ALLOWED, drops, note_loss, ref_text)
        for c in ok:
            fid = feature_ref(c.get("feature_ref"), c)
            ref = _str(c.get("segment_ref"))
            if c.get("kind") not in CONTRIB_KINDS or ref not in speaker_of:
                drops["contributions"]["bad_kind_or_segment"] += 1
                continue
            if fid is None:
                drops["contributions"]["feature_unresolved"] += 1
                continue
            if c["kind"] in ("lead", "assigned") and all(REQUEST_CUE.search(x["quote"]) for x in c["citations"]):
                drops["contributions"]["request_not_acceptance"] += 1
                continue
            sp = speaker_of[ref]
            res = lookup(name_rows, mt.id, sp) if sp else None
            if res is None or res.status == "unresolved":
                unresolved_speakers.append({"meeting": mt.id, "segment": ref, "speaker": sp})
                continue
            contributions.append({"feature": fid, "eids": res.ids, "ambiguous": res.ambiguous, "kind": c["kind"],
                                  "ref": ref, "record": record_of[ref], "speaker": sp, "citations": c["citations"],
                                  "meeting": mt.id})

        ok = _validated("development_links", _items(data, "development_links"), supplied, MEETING_ALLOWED, drops, note_loss, ref_text,
                        lambda it: _could_carry_implementation(it, cat))
        for lk in ok:
            fid = feature_ref(lk.get("feature_ref"), lk)
            if fid is None:
                drops["development_links"]["feature_unresolved"] += 1
                continue
            entries, why = pr_entries(lk.get("pr"), text_norm)
            if not entries:
                drops["development_links"][why] += 1
                continue
            ctx = _title_context(mt, _str(lk.get("pr")))
            for p in entries:
                dev_links[p.link].append({"feature": fid, "basis": "meeting_title", "citations": lk["citations"],
                                          "record": mt.id, "context": ctx, "date": mt.date})

        ok = _validated("feature_events", _items(data, "feature_events"), supplied, MEETING_ALLOWED, drops, note_loss, ref_text,
                        lambda it: _could_carry_implementation(it, cat))
        for ev in ok:
            fid = feature_ref(ev.get("feature_ref"), ev)
            ref = _str(ev.get("segment_ref"))
            if ev.get("kind") not in EVENT_KINDS:
                drops["feature_events"]["bad_kind"] += 1
                continue
            if fid is None:
                drops["feature_events"]["feature_unresolved"] += 1
                continue
            cited = " ".join(r.text for r in supplied if r.id in {c["record_id"] for c in ev["citations"]} | {record_of.get(ref)})
            # the span that mentions the feature decides, not the rest of the record (C1)
            spans = " ".join(quote_sentences(next((r_.text for r_ in supplied if r_.id == c["record_id"]), ""), c["quote"])
                             for c in ev["citations"])
            if ev["kind"] == "implemented" and not completed(spans):
                drops["feature_events"]["not_completed_work"] += 1  # a promise, plan, condition or negation (C1)
                note_loss(ref_text(ev) + " " + spans, impl=False)  # read and found not done: no hidden evidence
                continue
            if ev["kind"] == "reverted" and not rolled_back(spans):
                drops["feature_events"]["not_completed_rollback"] += 1
                note_loss(ref_text(ev) + " " + spans, impl=False)
                continue
            if ev["kind"] in EVENT_CUES and not EVENT_CUES[ev["kind"]].search(cited):
                drops["feature_events"]["no_evidence_for_kind"] += 1
                note_loss(ref_text(ev) + " " + spans, impl=ev["kind"] == "reverted")
                continue
            reason_ = ev.get("reason") if ev.get("reason") in REASONS else None
            if reason_ is None and ev["kind"] in ("held", "reverted"):
                reason_ = "other"
            rid = record_of.get(ref) or ev["citations"][0]["record_id"]
            prs_ = []
            if _str(ev.get("pr")):
                prs_, _ = pr_entries(ev["pr"], text_norm)
            fevents.append({"feature": fid, "kind": ev["kind"], "reason": reason_ if ev["kind"] in ("held", "reverted")
                            else None, "date": mt.date, "record": rid, "citations": ev["citations"],
                            "prs": [p.link for p in prs_], "source": "meeting"})
            if ev["kind"] == "implemented":
                for p in prs_:
                    implemented_pr.setdefault((fid, p.link), "meeting_event")

        ok = _validated("document_reviews", _items(data, "document_reviews"), supplied, MEETING_ALLOWED, drops, note_loss)
        docs, why_doc = [], "no_document"
        if mt.document_type and mt.document_type != PRODUCT_PLANNING:
            docs, why_doc = resolve_review_docs(cat, [c.text for c in mchats] + [mt.document_type], "", doc_prev,
                                                later_versions=True)
        for d_ in docs:
            discusses.add((f"Meeting:{mt.id}", f"Document:{d_.id}"))
        for rv in ok:
            ref = _str(rv.get("segment_ref"))
            if rv.get("kind") not in REVIEW_KINDS or ref not in speaker_of:
                drops["document_reviews"]["bad_kind_or_segment"] += 1
                continue
            seg_text = next((r_.text for r_ in supplied if r_.id == record_of[ref]), "")
            if not is_review(rv["kind"], cited_remark(seg_text, rv["citations"], record_of[ref])):
                drops["document_reviews"]["not_a_review"] += 1  # an opener or thanks suggests nothing
                continue
            if not docs:
                drops["document_reviews"]["document_unresolved"] += 1
                unresolved_reviews.append({"record": record_of.get(ref), "why": why_doc, "text": _str(rv.get("text"))})
                continue
            sp = speaker_of[ref]
            res = lookup(name_rows, mt.id, sp) if sp else None
            if res is None or res.status == "unresolved":
                unresolved_speakers.append({"meeting": mt.id, "segment": ref, "speaker": sp})
                continue
            reviews.append({"docs": [d_.id for d_ in docs], "eids": res.ids, "ambiguous": res.ambiguous,
                            "kind": rv["kind"], "section": _str(rv.get("section")) or None,
                            "text": _str(rv.get("text")), "record": record_of[ref], "citations": rv["citations"],
                            "speaker": sp, "source_text": seg_text[:600], "doc_basis": why_doc})

    # ---- competitors named by demo links (rule, no model): known before any session is read
    demo_links = []
    for u in rec.urls:
        name = next((n for n in (sr.competitor_of_demo(d, own_names) for d in u.descriptions) if n), None)
        if name is None:
            continue
        cid = cat.register_competitor(name, citations=[{"record_id": u.id, "quote": u.descriptions[0]}])
        competitor_alias[cid].add(name)
        for link in u.links:
            nodes.append(Node("Url", f"Url:{link}", {"link": link, "kind": "demo", "subject": cid,
                                                     "source_competitor": name}))
            edge("ABOUT", f"Url:{link}", cid, [u.id], kind="demo")
            demo_links.append({"link": link, "competitor": cid})

    # ---- 2b. reconcile feature identities (R6): one call per product groups names of the same area
    merged_features: dict[str, str] = {}
    if len(feature_info) > 1:
        fl = list(feature_info)
        lines = [f"F{i}: {feature_info[f]['title']}" + (" | " + "; ".join(feature_info[f]["details"][:5])
                                                       if feature_info[f]["details"] else "") for i, f in enumerate(fl, 1)]
        req = [{"role": "system", "content": FEATURE_GROUP_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]
        data, reason = caller.json_call("feature_group", req, min(3000, 300 + 60 * len(fl)), check=GROUP_SHAPE)
        if data is None:
            failures.append({"kind": "feature_group", "id": product, "reason": reason})
            state["unresolved_feature_refs"] += 1
            note_loss("", kind="feature_group", unit=product, everywhere=True)
        else:
            for g in _items(data, "groups"):
                refs = [m_ for m_ in _list(g.get("members")) if isinstance(m_, str) and re.fullmatch(r"F\d+", m_)]
                head = _str(g.get("title"))
                if not re.fullmatch(r"F\d+", head) or not refs:
                    drops["feature_group"]["bad_group"] += 1
                    continue
                idx = [int(r[1:]) for r in refs + [head] if 1 <= int(r[1:]) <= len(fl)]
                canon = fl[int(head[1:]) - 1] if 1 <= int(head[1:]) <= len(fl) else None
                for i in idx:
                    if canon and fl[i - 1] != canon and fl[i - 1] not in merged_features and canon not in merged_features:
                        merged_features[fl[i - 1]] = canon
        for old, new in merged_features.items():
            src, dst = feature_info.pop(old), feature_info[new]
            for dt in src["details"]:
                if dt not in dst["details"]:
                    dst["details"].append(dt)
            dst["records"] += [r_ for r_ in src["records"] if r_ not in dst["records"]]
            dst["sources"] |= src["sources"]
            ent = cat.features.entries[new]
            for a in (src["title"], *cat.features.entries[old]["aliases"]):
                if a != ent["title"] and a not in ent["aliases"]:
                    ent["aliases"].append(a)
            for k, v in list(cat.features._index.items()):
                if v == old:
                    cat.features._index[k] = new
            del cat.features.entries[old]
        remap = lambda f: merged_features.get(f, f)  # noqa: E731
        for c in contributions:
            c["feature"] = remap(c["feature"])
        for ls in dev_links.values():
            for l_ in ls:
                l_["feature"] = remap(l_["feature"])
        for ev in fevents:
            ev["feature"] = remap(ev["feature"])
        discusses = {(a, remap(b)) for a, b in discusses}
        remapped: dict[tuple[str, str], str] = {}
        for (f_, l_), basis in implemented_pr.items():  # I5: implementation evidence follows the merge
            remapped.setdefault((remap(f_), l_), basis)
            if basis == "merged":
                remapped[(remap(f_), l_)] = basis
        implemented_pr = remapped
    # integrity after grouping: no structure may point at a feature that no longer exists
    live = set(feature_info)
    for name, rows, key in (("contributions", contributions, "feature"), ("feature_events", fevents, "feature")):
        bad_rows = [r_ for r_ in rows if r_[key] not in live]
        if bad_rows:
            drops["after_grouping"][name] += len(bad_rows)
            rows[:] = [r_ for r_ in rows if r_[key] in live]
    for l_, ls_ in list(dev_links.items()):
        ls_[:] = [x for x in ls_ if x["feature"] in live]
    implemented_pr = {k_: v_ for k_, v_ in implemented_pr.items() if k_[0] in live}
    discusses = {(a, b) for a, b in discusses if not b.startswith("Feature:") or b in live}

    # C4: a PR title keeps the task it is discussed under. Every spoken occurrence of the title in any meeting is read
    # with the two segments before it; the feature whose name fits that context best wins, and both are cited.
    ftoks = lambda fid: set(re.findall(r"[a-z0-9]{4,}", " ".join(  # noqa: E731
        [feature_info[fid]["title"], *cat.features.entries[fid]["aliases"]]).casefold()))
    tokens = {fid: ftoks(fid) for fid in feature_info}
    conflicts = []
    winner_of: dict[str, str] = {}
    pre_c4 = {link: list(ls_) for link, ls_ in dev_links.items()}  # 13f re-reads these after the membership calls
    for link in list(dev_links):
        ls_ = dev_links[link]
        if not ls_ or any(x["basis"] != "meeting_title" for x in ls_):
            continue
        title = normalise(prs_raw[link].title)
        cands: dict[str, dict] = {}

        def offer(fid, score, date, cites, rec_id, ctx):
            if fid not in cands or (score, date) > (cands[fid]["score"], cands[fid]["date"]):
                cands[fid] = {"feature": fid, "score": score, "date": date, "citations": cites, "record": rec_id,
                              "context": ctx}

        for x in ls_:
            offer(x["feature"], _context_score(x["context"], tokens[x["feature"]]), x["date"], x["citations"],
                  x["record"], x["context"])
        for mt_ in rec.meetings:
            for k, sg in enumerate(mt_.segments):
                if title and title in normalise(sg.text):
                    ctx = (sg.text, mt_.segments[k - 1].text if k else "")
                    sc = sorted(((_context_score(ctx, t_), f_) for f_, t_ in tokens.items()), reverse=True)
                    if sc and sc[0][0] > 0 and (len(sc) == 1 or sc[0][0] > sc[1][0]):
                        offer(sc[0][1], sc[0][0], mt_.date, [{"record_id": sg.id, "quote": prs_raw[link].title,
                                                              "field": "text"}], mt_.id, ctx)
        if len(cands) < 2:
            continue
        best = max(cands.values(), key=lambda x: (x["score"], x["date"]))
        others = [x for x in cands.values() if x["feature"] != best["feature"]]
        conflicts.append({"pr": link, "kept": best["feature"], "dropped": sorted(x["feature"] for x in others),
                          "citations": [c for x in cands.values() for c in x["citations"]]})
        keep = next((x for x in ls_ if x["feature"] == best["feature"]), None) or {
            "feature": best["feature"], "basis": "meeting_title", "citations": best["citations"],
            "record": best["record"], "context": best["context"], "date": best["date"]}
        keep["also_cited"] = [c for x in others for c in x["citations"]]
        dev_links[link] = [keep]
        winner_of[link] = best["feature"]

    # ---- 3. Slack sessions: planning channels always, other channels routed by rules
    by_channel: dict[str, list] = defaultdict(list)
    for m in rec.messages:
        by_channel[m.channel_id].append(m)
    units = []
    for msgs in by_channel.values():
        for sess in sessions(msgs):
            real = [m for m in sess if not m.is_bot]
            if not real:
                continue
            group = "planning" if real[0].channel_name.startswith("planning-") else "other"
            if real[0].channel_name.startswith("bug-"):
                continue
            session_sums[group]["total"] += 1
            send, why = sr.route_session(sess)
            if not send:
                session_sums["excluded"][why] += 1
                excluded_ids[why].append(real[0].id)
                continue
            if group == "other":
                session_sums["other"]["routed"] += 1
            chunks = sr.split_session(real, SESSION_CHARS)
            state["split_sessions"] += len(chunks) > 1
            units.append((real[0].ts, real[0].id, group, chunks))
    units.sort(key=lambda u: (u[0], u[1]))


    for _, first_id, group, chunks in units:
        for chunk in chunks:
            comps = [e["name"] for e in cat.competitors.entries.values()]
            req, tokens = build_session_request(chunk, rec, cat, comps, anchors)
            data, reason = caller.json_call("session", req, tokens, check=SESSION_SHAPE)
            if data is None:
                failures.append({"kind": "session", "id": chunk[0].id, "reason": reason})
                failed_sessions.append({"first_message": chunk[0].id, "reason": reason,
                                        "text": _topic(" ".join(m.text for m in chunk))})
                note_loss(" ".join(m.text for m in chunk), kind="session", unit=chunk[0].id)
                continue
            session_sums[group]["extracted"] += 1
            _process_session(data, chunk, rec, cat, anchors, own_keys, drops, state, reviews, observations,
                             decisions, discusses, competitor_alias, note_loss, unresolved_reviews, doc_prev)

    # ---- demo links whose cited message names a known competitor (rule, after competitor discovery)
    done_links = {d_["link"] for d_ in demo_links}
    comp_names = {e["name"]: cid_ for cid_, e in cat.competitors.entries.items()}
    for u in rec.urls:
        looks_demo = any("demo" in x.lower() for x in u.descriptions) or any("/demo" in x for x in u.links)
        for link in u.links:
            if link in done_links or not looks_demo or _names_in(" ".join(u.descriptions), own_names):
                continue
            for m in rec.messages:
                if link not in m.text or m.is_bot:
                    continue
                hit = _names_in(m.text, comp_names)
                if len(hit) == 1:
                    name = next(iter(hit))
                    cid = comp_names[name]
                    quote = next((x for x in _SENTENCES.split(m.text) if _names_in(x, [name])), m.text)[:200]
                    nodes.append(Node("Url", f"Url:{link}", {"link": link, "kind": "demo", "subject": cid,
                                                             "source_competitor": name}))
                    edge("ABOUT", f"Url:{link}", cid, [m.id], kind="demo", basis="message", quote=quote)
                    demo_links.append({"link": link, "competitor": cid})
                    done_links.add(link)
                    break

    # ---- 5. PR association: meeting titles first, then batches of the unmatched
    bugfix = {m_.group(0) for m in rec.messages if m.bug_channel_owner for m_ in sr.PR_LINK.finditer(m.text)}
    product_prs = [p for p in cat.prs if p.repo_scope == "product" and parse_revert_title(p.title) is None
                   and p.link not in bugfix]  # a PR a bug channel cites fixes a bug; it is not feature development
    product_prs.sort(key=lambda p: (p.repo or "", int(p.number) if p.number.isdigit() else 0))
    feats = [{"id": fid, "title": e["title"], "details": e["details"]} for fid, e in feature_info.items()]
    unmatched = [p for p in product_prs if p.link not in dev_links]
    batches = [unmatched[i:i + MEMBERSHIP_BATCH] for i in range(0, len(unmatched), MEMBERSHIP_BATCH)]
    for batch in batches:
        if not feats:
            break
        rows = [{"title": p.title, "summary": prs_raw[p.link].summary, "entry": p} for p in batch]
        fidx = {f["id"]: k for k, f in enumerate(feats, 1)}
        examples = [(prs_raw[link].title, fidx[ls[0]["feature"]]) for link, ls in dev_links.items()
                    if ls[0]["basis"] == "meeting_title" and ls[0]["feature"] in fidx][:12]
        req, tokens = build_membership_request(rows, feats, examples)
        data, reason = caller.json_call("membership", req, tokens, check=MEMBERSHIP_SHAPE)
        if data is None:
            failures.append({"kind": "membership", "id": batch[0].link, "reason": reason})
            note_loss(" ".join(p.title + " " + prs_raw[p.link].summary for p in batch), kind="membership",
                      unit=batch[0].link, everywhere=True)  # which feature those PRs belong to is unknown (R8)
            continue
        for a in _items(data, "assignments"):
            pref, fref = _str(a.get("pr")), a.get("feature")
            m = re.fullmatch(r"P(\d+)", pref)
            if not m or not 1 <= int(m.group(1)) <= len(rows):
                drops["pr_membership"]["bad_pr_ref"] += 1
                continue
            if fref is None or fref == "":
                continue
            fm = re.fullmatch(r"F(\d+)", _str(fref))
            if not fm or not 1 <= int(fm.group(1)) <= len(feats):
                drops["pr_membership"]["bad_feature_ref"] += 1
                pm = rows[int(m.group(1)) - 1]["entry"]
                if prs_raw[pm.link].merged is True:
                    # a merged PR we could not file: it blocks the feature its task context names, else every feature
                    ttl = normalise(pm.title)
                    best: list[tuple[int, str]] = []
                    for mt_ in rec.meetings:
                        for k_, sg_ in enumerate(mt_.segments):
                            if ttl in normalise(sg_.text):
                                cx = (sg_.text, mt_.segments[k_ - 1].text if k_ else "")
                                best += [(_context_score(cx, set(re.findall(r"[a-z0-9]{4,}", " ".join(
                                    [e_["title"]]).casefold()))), f_) for f_, e_ in feature_info.items()]
                    best.sort(reverse=True)
                    if best and best[0][0] > 0 and (len(best) == 1 or best[0][0] > best[1][0]):
                        note_loss(feature_info[best[0][1]]["title"] + " " + pm.title, impl=True)
                    else:
                        note_loss(pm.title, impl=True, everywhere=True)
                continue
            p = rows[int(m.group(1)) - 1]["entry"]
            pr_src = [SourceRecord(prs_raw[p.link].id, "pr", "title", prs_raw[p.link].title),
                      SourceRecord(prs_raw[p.link].id, "pr", "summary", prs_raw[p.link].summary)]
            cite = {"record_id": prs_raw[p.link].id, "quote": a.get("quote")}
            r = validate_citation(cite, pr_src, PR_ALLOWED)
            if not r.ok:
                drops["pr_membership"][r.reason] += 1
                note_loss(feats[int(fm.group(1)) - 1]["title"] + " " + prs_raw[p.link].title + " " + p.link,
                          impl=prs_raw[p.link].merged is True)
                continue
            dev_links[p.link].append({"feature": feats[int(fm.group(1)) - 1]["id"], "basis": "batch",
                                      "citations": [{**cite, "field": r.record.field, "span": list(r.span)}],
                                      "record": p.link})
    # 13f: an explicit task-to-PR statement outranks the title-word reading above. That reading stays in place until
    # here because the membership requests carry its results as examples (the cached requests must not change).
    names_of = {fid: [feature_info[fid]["title"], *cat.features.entries[fid]["aliases"]] for fid in feature_info}
    for link, old_ls in pre_c4.items():
        if not old_ls or any(x["basis"] != "meeting_title" for x in old_ls):
            continue
        found = explicit_task_features(prs_raw[link].title, rec.meetings, names_of)
        if not found:
            continue
        seen = {x["feature"] for x in old_ls} | {x["feature"] for x in dev_links.get(link, [])}
        cites = [c for x in found for c in x["citations"]]
        if len(found) > 1:  # explicit statements disagree: no association
            conflicts[:] = [c_ for c_ in conflicts if c_["pr"] != link]
            conflicts.append({"pr": link, "kept": None, "dropped": sorted(seen | {x["feature"] for x in found}),
                              "citations": cites, "basis": "explicit_task_conflict"})
            dev_links.pop(link, None)
            winner_of.pop(link, None)
            if prs_raw[link].merged is True:  # a merged PR we cannot file blocks the features it might implement
                note_loss(" ".join(feature_info[x["feature"]]["title"] for x in found) + " " + prs_raw[link].title,
                          impl=True)
            continue
        win = found[0]
        if seen == {win["feature"]}:
            continue  # the same feature either way
        keep = next((x for x in old_ls if x["feature"] == win["feature"]), None) or {
            "feature": win["feature"], "basis": "meeting_title", "citations": win["citations"], "record": win["meeting"],
            "context": win["context"], "date": win["date"]}
        keep = {**keep, "also_cited": [c for x in old_ls if x["feature"] != win["feature"] for c in x["citations"]]}
        conflicts[:] = [c_ for c_ in conflicts if c_["pr"] != link]
        conflicts.append({"pr": link, "kept": win["feature"], "dropped": sorted(seen - {win["feature"]}),
                          "citations": [*win["citations"], *keep["also_cited"]], "basis": "explicit_task"})
        dev_links[link] = [keep]
        winner_of[link] = win["feature"]
    unassigned = [p.link for p in product_prs if p.link not in dev_links]

    # ---- 6. derived outcomes
    feat_of_pr: dict[str, list[str]] = defaultdict(list)
    for link, ls in dev_links.items():
        for l_ in ls:
            if l_["feature"] not in feat_of_pr[link]:
                feat_of_pr[link].append(l_["feature"])
    # C4: an event that names PRs follows the corrected feature of those PRs; PRs that disagree leave it unattached
    events_unattached = 0
    for ev in fevents:
        if ev["source"] != "meeting" or not ev["prs"] or ev["kind"] == "discussed":
            continue
        fs = {f for l_ in ev["prs"] for f in feat_of_pr.get(l_, [])}
        if len(fs) == 1:
            ev["feature"] = next(iter(fs))
        elif len(fs) > 1:
            ev["feature"] = None
            events_unattached += 1
            note_loss(" ".join(ev["prs"]), kind="loss", impl=ev["kind"] in ("implemented", "reverted"))
    implemented_pr = {k_: v_ for k_, v_ in implemented_pr.items() if v_ != "meeting_event"}
    for ev in fevents:
        if ev["kind"] == "implemented" and ev["feature"] and ev["source"] == "meeting":
            for l_ in ev["prs"]:
                if not feat_of_pr.get(l_) or ev["feature"] in feat_of_pr[l_]:
                    implemented_pr.setdefault((ev["feature"], l_), "meeting_event")
    for link, fl in feat_of_pr.items():
        entry = cat.pr(link)
        if entry and prs_raw[link].merged is True:
            for fid in fl:
                implemented_pr[(fid, link)] = "merged"

    # PR decisions -> pending action and feature events
    decisions.sort(key=lambda d: (d["ts"], d["message"], d["idx"]))
    effective: dict[str, list[dict]] = defaultdict(list)
    for d in decisions:
        if d["action"] in ("withhold", "resume", "merge") and not d["conditional"]:
            effective[d["pr"]].append(d)
    pending_counts: Counter = Counter()
    pr_props: dict[str, dict] = {}
    for p in cat.prs:
        if p.repo_scope != "product":
            continue
        raw = prs_raw[p.link]
        eff = effective.get(p.link, [])
        held = bool(eff) and eff[-1]["action"] == "withhold"
        action = reason = None
        basis = []
        if raw.state == "open" and raw.merged is False and pr_approval(raw.reviews) == "approved":
            action = "need_close" if held else "need_merge"
            if held:
                reason = eff[-1]["reason"]
            basis = [eff[-1]["message"]] if eff else []
            pending_counts[action] += 1
        pr_props[p.link] = {"link": p.link, "pending_action": action, "pending_reason": reason,
                            "pending_basis": basis, "pending_version": PENDING_VERSION}
    held_before: set[str] = set()
    for d in decisions:
        if d["conditional"] or d["action"] not in ("withhold", "resume", "merge"):
            continue
        kind = "held" if d["action"] == "withhold" else "resumed"
        if kind == "resumed" and d["pr"] not in held_before:
            continue
        if kind == "held":
            held_before.add(d["pr"])
        for fid in feat_of_pr.get(d["pr"], []):
            fevents.append({"feature": fid, "kind": kind, "reason": d["reason"] or ("other" if kind == "held" else None),
                            "date": d["ts"], "record": d["message"], "citations": d["citations"], "prs": [d["pr"]],
                            "source": "slack"})
            discusses.add((f"Message:{d['message']}", fid))
    # a completed rollback stated in Slack ("we reverted the rollout")
    for d in decisions:
        spans = " ".join(quote_sentences(d["text"], c_["quote"]) for c_ in d["citations"])
        about_pr = d["pr"] in spans or re.search(r"\bPRs?\s*#?\s*" + re.escape(prs_raw[d["pr"]].number) + r"\b", spans, re.I)
        if d["action"] == "revert" and not d["conditional"] and rolled_back(spans) and about_pr:
            for fid in feat_of_pr.get(d["pr"], []):
                fevents.append({"feature": fid, "kind": "reverted", "reason": d["reason"] or "other",
                                "date": d["ts"], "record": d["message"], "citations": d["citations"],
                                "prs": [d["pr"]], "source": "slack"})
    # applied reverts of a feature's PR
    for rv in rules_rep["reverts"]:
        if not rv["applied"] or not rv["original"] or rv["repo_scope"] != "product":
            continue
        rpr = prs_raw.get(rv["revert"])
        for fid in feat_of_pr.get(rv["original"], []):
            if (fid, rv["original"]) not in implemented_pr:
                continue  # reverting work that was never implemented reverts nothing (C2)
            fevents.append({"feature": fid, "kind": "reverted", "reason": rv["reason"], "reason_text": rv["clause"],
                            "date": rpr.created_at, "record": rpr.id, "citations": [], "prs": [rv["original"]],
                            "source": "revert"})

    # ---- documents: features and competitor claims in the content (one call per document); a feature found only here
    # is "proposed", never "discussed"
    doc_calls_failed: list[str] = []
    doc_obs_n: Counter = Counter()
    doc_observations: list[dict] = []
    for d in rec.documents:
        req = build_document_request(d, rec, cat)
        data, reason = caller.json_call("document", req[0], req[1], check=DOCUMENT_SHAPE)
        if data is None:
            failures.append({"kind": "document", "id": d.id, "reason": reason})
            doc_calls_failed.append(d.id)
            note_loss(d.content, kind="document", unit=d.id)
            continue
        supplied = [SourceRecord(d.id, "document", "content", d.content)]
        for f in _validated("doc_features", _items(data, "features"), supplied, DOC_ALLOWED, drops, note_loss):
            title = _str(f.get("title"))
            if not title:
                drops["doc_features"]["no_title"] += 1
                continue
            fid = register_feature(title, [_str(x) for x in _list(f.get("details")) if _str(x)], f["citations"],
                                   d.id, "document")
            fevents.append({"feature": fid, "kind": "proposed", "reason": None, "date": d.date, "record": d.id,
                            "citations": f["citations"], "prs": [], "source": "document"})
        for n_, c in enumerate(_validated("doc_claims", _items(data, "claims"), supplied, DOC_ALLOWED, drops, note_loss)):
            kind, subject = c.get("kind"), _str(c.get("subject_ref"))
            if kind not in COMPETITOR_KINDS or not _str(c.get("text")) or not subject:
                drops["doc_claims"]["bad_kind_or_text"] += 1
                continue
            cited = " ".join(x["quote"] for x in c["citations"])
            if normalise(subject) in own_keys or not _names_in(cited, [subject]) or not COMP_CUE.search(d.content):
                drops["doc_claims"]["subject_not_grounded"] += 1
                continue
            cid = cat.register_competitor(subject, citations=c["citations"])
            competitor_alias[cid].add(subject)
            doc_observations.append({
                "id": f"Observation:{product}:{d.id}:{doc_obs_n[d.id]}", "kind": kind, "text": _str(c["text"]),
                "release_scope": "previous" if d.id in doc_prev else "unknown",
                "scope_basis": "document_scope" if d.id in doc_prev else "none",
                "subject": cat.competitors.entries[cid]["name"], "record": d.id, "customer_ids": [],
                "citations": c["citations"], "author": None, "target": cid, "reporters": []})
            doc_obs_n[d.id] += 1
    observations += doc_observations

    # number events per record (R13) and build event nodes
    per_record: Counter = Counter()
    for ev in fevents:
        ev["n"] = per_record[ev["record"]]
        per_record[ev["record"]] += 1
        ev["id"] = f"FeatureEvent:{product}:{ev['record']}:{ev['n']}"

    # fate predicates (C2, C3)
    pr_ids = defaultdict(set)  # feature -> {(link, number, normalised title)} of the PRs linked to it
    for link, fl_ in feat_of_pr.items():
        for fid_ in fl_:
            pr_ids[fid_].add((link, prs_raw[link].number, normalise(prs_raw[link].title)))
    held_prs = defaultdict(set)  # PR link -> dates of a hold followed by a resume or merge decision
    seen_hold: dict[str, str] = {}
    for d in decisions:
        if d["conditional"]:
            continue
        if d["action"] == "withhold":
            seen_hold.setdefault(d["pr"], d["ts"])
        elif d["action"] in ("resume", "merge") and d["pr"] in seen_hold:
            held_prs[d["pr"]].add(seen_hold[d["pr"]])

    blockers: dict[str, dict] = {}

    def reconciled(fid):
        topics = [fid.split(":", 2)[2], *(_topic(a) for a in cat.features.entries[fid]["aliases"])]
        for u in units_lost:
            if u["kind"] == "loss" and not u["impl"]:
                continue  # a lost contribution, review, discussion or observation cannot flip "not implemented"
            hit = u["everywhere"] or any(t and t in u["flat"] for t in topics)
            if not hit:
                hit = any(link in u["raw"] or title in u["raw"] or re.search(r"\bpr\s*#?\s*" + number + r"\b", u["raw"])
                          for link, number, title in pr_ids[fid])
            if hit:
                blockers[fid] = {"kind": u["kind"], "unit": u["unit"], "text": u["raw"][:120]}
                return False
        return True

    doc_sentences = [(d.id, x) for d in rec.documents if d.id not in doc_prev
                     for x in re.split(r"(?<=[.!?])\s+|\n+", d.content)
                     if (_EXISTS.search(x) or (_DESCRIBES.search(x) and d.id.startswith(("latest_", "new_"))))
                     and not _NOT_DONE.search(x)]

    def _words(text, parts=False):
        whole = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.casefold())
        out = {w for w in whole if len(w) >= 4}
        if parts:  # a hyphenated word also counts by its parts
            out |= {x for w in whole for x in w.split("-") if len(x) >= 4}
        return out

    def document_conflicts(fid):
        out = set()
        for name in [feature_info[fid]["title"], *cat.features.entries[fid]["aliases"]]:
            need = _words(name)
            if not need:
                continue
            k = -(-2 * len(need) // 3)  # at least two thirds of the feature name's words
            out |= {d_ for d_, x in doc_sentences if len(need & _words(x, True)) >= k}
        return sorted(out)

    fate: dict[str, dict] = {}
    for fid in feature_info:
        evs = [e for e in fevents if e["feature"] == fid]
        kinds = Counter(e["kind"] for e in evs)
        implemented = kinds["implemented"] > 0 or any(f == fid for (f, _) in implemented_pr)
        # reverted: an applied revert of a PR that implemented this feature (those are the only revert events kept)
        reverted = any(e["kind"] == "reverted" for e in evs)  # rule reverts, meeting rollbacks, Slack rollbacks
        held_dates = sorted(e["date"] for e in evs if e["kind"] == "held")
        positive_after = [e for e in evs if held_dates and e["kind"] == "implemented" and e["date"] > held_dates[0]]
        merged_after_hold = any(prs_raw[l_].merged is True and l_ in held_prs
                                for (f_, l_) in implemented_pr if f_ == fid)
        discussed = bool(feature_info[fid]["sources"] - {"document"}) or kinds["discussed"] > 0
        rec_ok = reconciled(fid)
        conflict = document_conflicts(fid) if not implemented else []
        fate[fid] = {"implemented": implemented, "reverted": reverted, "held": kinds["held"] > 0,
                     "resumed": kinds["resumed"] > 0,
                     "delayed_then_added": bool(held_dates and (positive_after or merged_after_hold)),
                     "discussed": discussed, "proposed": kinds["proposed"] > 0, "reconciled": rec_ok,
                     "document_conflict": conflict,
                     "discussed_not_implemented": (((not implemented) if rec_ok and not conflict else None)
                                                   if discussed else False),
                     "derived_by": FATE_VERSION}

    # ---- facts
    pid = f"Product:{product}"
    for fid, info in feature_info.items():
        ent = cat.features.entries[fid]
        nodes.append(Node("Feature", fid, {"id": fid, "title": info["title"], "topic": fid.split(":", 2)[2],
                                           "aliases": ent["aliases"], "details": info["details"], "product": product,
                                           "citations": ent["citations"], **fate[fid]}))
        edge("HAS", pid, fid, info["records"])
    for cid, ent in cat.competitors.entries.items():
        nodes.append(Node("Competitor", cid, {"name": ent["name"], "aliases": sorted(competitor_alias[cid] - {ent["name"]}),
                                              "product": product, "citations": ent["citations"]}))
        edge("HAS", pid, cid, [c["record_id"] for c in ent["citations"]])
    for src, dst in sorted(discusses):
        if src.startswith("Message:"):
            nodes.append(Node("Message", src, {"id": src.split(":", 1)[1]}))
        edge("DISCUSSES", src, dst, [src.split(":", 1)[1]])
    for c in contributions:
        for eid in c["eids"]:
            edge("CONTRIBUTED", person(eid), c["feature"], [c["record"]], f"{c['ref']}:{c['kind']}", kind=c["kind"],
                 ambiguous=c["ambiguous"], speaker_name=c["speaker"], n_candidates=len(c["eids"]))
    remark_n: Counter = Counter()
    for r in reviews:
        for doc_id in r["docs"]:
            n = remark_n[(r["record"], doc_id)]  # several remarks of one message get their own edge
            remark_n[(r["record"], doc_id)] += 1
            for eid in r["eids"]:
                edge("SUGGESTED", person(eid), f"Document:{doc_id}", [r["record"]],
                     r["record"] if n == 0 else f"{r['record']}#{n + 1}", section=r["section"], text=r["text"],
                     kind=r["kind"], ambiguous=r["ambiguous"], source_text=r["source_text"],
                     doc_basis=r.get("doc_basis"), version_ambiguous=len(r["docs"]) > 1)
    for link, fl in feat_of_pr.items():
        for fid in fl:
            first = next(x for x in dev_links[link] if x["feature"] == fid)
            edge("ABOUT", f"PR:{link}", fid, [prs_raw[link].id], kind="development", basis=first["basis"])
    for (fid, link), basis in sorted(implemented_pr.items()):
        edge("IMPLEMENTED_BY", fid, f"PR:{link}", [prs_raw[link].id], basis=basis)
    for ev in fevents:
        nodes.append(Node("FeatureEvent", ev["id"], {
            "id": ev["id"], "kind": ev["kind"], "reason": ev["reason"], "reason_text": ev.get("reason_text")
            or (ev["citations"][0]["quote"] if ev["citations"] else None), "date": ev["date"],
            "record_id": ev["record"], "feature": ev["feature"], "attached": bool(ev["feature"]), "product": product, "pr_links": ev["prs"],
            "source": ev["source"], "citations": ev["citations"]}))
        if ev["feature"]:
            edge("HAS_EVENT", ev["feature"], ev["id"], [ev["record"]])
    obs_counts: dict[str, Counter] = defaultdict(Counter)
    for o in observations:
        oid = o["id"]
        obs_counts[o["kind"]][o["release_scope"]] += 1
        nodes.append(Node("Observation", oid, {
            "id": oid, "kind": o["kind"], "text": o["text"], "release_scope": o["release_scope"],
            "scope_basis": o["scope_basis"], "subject": o["subject"], "record_id": o["record"], "product": product,
            "customer_ids": o["customer_ids"], "citations": o["citations"]}))
        if o["author"]:
            edge("STATED", person(o["author"]), oid, [o["record"]])
        edge("ABOUT", oid, o["target"], [o["record"]])
        for cid in o["customer_ids"]:
            edge("REPORTED_BY", oid, customer(cid), [o["record"]])
        for eid in o["reporters"]:
            edge("REPORTED_BY", oid, person(eid), [o["record"]])
    for d in decisions:
        nodes.append(Node("Message", f"Message:{d['message']}", {"id": d["message"]}))
        edge("ABOUT", f"Message:{d['message']}", f"PR:{d['pr']}", [d["message"]], str(d["idx"]), action=d["action"],
             reason=d["reason"], conditional=d["conditional"])
    for props in pr_props.values():
        nodes.append(Node("PR", f"PR:{props['link']}", props))

    decision_prs = Counter(d["action"] for d in decisions)
    failed_ids = [{"first_message": f["first_message"], "reason": f["reason"]} for f in failed_sessions]
    sessions_failed_planning = sum(1 for f in failed_sessions if f["first_message"] in
                                   {m.id for m in rec.messages if m.channel_name.startswith("planning-")})
    manifest = {
        "documents": {"total": len(rec.documents), "with_feedback": n_feedback_docs,
                      "content_extracted": len(rec.documents) - len(doc_calls_failed), "content_failed": doc_calls_failed},
        "meetings": {"total": len(rec.meetings), "extracted": len(rec.meetings) - len(meeting_failed),
                     "failed": meeting_failed},
        "sessions": {
            "planning": {"total": session_sums["planning"]["total"],
                         "extracted": session_sums["planning"]["total"] - sessions_failed_planning},
            "other": {"total": session_sums["other"]["total"], "routed": session_sums["other"]["routed"],
                      "excluded": dict(session_sums["excluded"]), "excluded_first_messages": dict(excluded_ids),
                      "extracted": session_sums["other"]["routed"] - (len(failed_ids) - sessions_failed_planning)},
            "failed": failed_ids, "split_sessions": state["split_sessions"]},
        "unresolved_feature_refs": state["unresolved_feature_refs"],
        "validation_losses": sum(1 for u in units_lost if u["kind"] == "loss"),
        "failed_units": [{"kind": u["kind"], "unit": u["unit"]} for u in units_lost if u["kind"] != "loss"],
        "features_not_reconciled": [f for f, v in fate.items() if not v["reconciled"]],
        "fate_blockers": blockers,
        "complete": not [u for u in units_lost if u["kind"] != "loss"] and state["unresolved_feature_refs"] == 0,
    }
    report = {
        "product": product, "prompt_version": PROMPT_VERSION, "fate_version": FATE_VERSION,
        "features": len(feature_info), "feature_titles": [e["title"] for e in feature_info.values()],
        "feature_events": dict(Counter(e["kind"] for e in fevents)),
        "contributions": dict(Counter(c["kind"] for c in contributions)),
        "contributors": len({e for c in contributions for e in c["eids"]}),
        "pr_conflicts": conflicts, "events_unattached": events_unattached, "unresolved_reviews": unresolved_reviews,
        "pr_membership": {"bugfix_prs_excluded": len(bugfix), "by_basis": dict(Counter(l_["basis"] for ls in dev_links.values() for l_ in ls[:1])),
                          "unassigned": unassigned, "features_with_prs": len({f for fl in feat_of_pr.values() for f in fl})},
        "pr_decisions": {"by_action": dict(decision_prs), "unresolved_prs": state.get("unresolved_prs", [])},
        "pending": dict(pending_counts),
        "observations": {k: dict(v) for k, v in sorted(obs_counts.items())},
        "scope_downgraded": state["scope_downgraded"],
        "demo_links": demo_links,
        "suggestions": len(reviews),
        "feedback_items": n_items,
        "unresolved_speakers": unresolved_speakers,
        "unresolved_feature_refs": state["unresolved_feature_refs"],
        "extraction_failures": failures,
        "dropped": {k: dict(v) for k, v in drops.items()},
        "manifest": manifest,
        "features_merged": len(merged_features),
        "calls": {"meeting": caller.calls["meeting"], "document": caller.calls["document"], "feature_group": caller.calls["feature_group"], "session": caller.calls["session"],
                  "membership": caller.calls["membership"], "live": sum(caller.calls.values()),
                  "cached": sum(caller.cached.values())},
        "usage": {"prompt_tokens": caller.prompt_tokens, "completion_tokens": caller.completion_tokens,
                  "cost_usd": round(caller.cost_usd, 6)},
        "membership_calls_max": len(batches),
        "estimate": {"prompt_tokens": caller.est_in, "completion_tokens": caller.est_out,
                     "projected_usd": round(caller.projected_usd(), 6)} if dry else None,
    }
    return SemanticResult(Facts([], []).merge(Facts(nodes, edges)), report)


def estimate_semantic(rec, *, cache_root=None) -> dict:
    """Calls a live run would make now (cache misses), without sending anything. Membership calls depend on the
    features the meeting calls return, so `membership_calls_max` is the count when every PR is still unmatched."""
    res = run_semantic(rec, chat=None, cache_root=cache_root, dry=True)
    r = res.report
    projected = r["estimate"]["projected_usd"]
    return {"meeting_calls": r["calls"]["meeting"], "session_calls": r["calls"]["session"],
            "membership_calls": r["calls"]["membership"], "membership_calls_max": r["membership_calls_max"],
            "cached": r["calls"]["cached"], "projected_usd": projected}


# ---------------------------------------------------------------- one Slack session chunk

def _process_session(data, chunk, rec, cat, anchors, own_keys, drops, state, reviews, observations, decisions,
                     discusses, competitor_alias, note_loss, unresolved_reviews, doc_prev) -> None:
    msgs = {m.id: m for m in chunk if not m.is_bot}
    supplied = [SourceRecord(m.id, "message", "text", m.text) for m in msgs.values()]
    text_all = "\n".join(m.text for m in msgs.values())
    text_norm = normalise(text_all)
    order = {m.id: i for i, m in enumerate(msgs.values())}
    session_prev = bool(sr.PREV_CUE.search(text_all))
    anchor_session = any(mid in anchors for mid in msgs)

    def lose(kind, bad):
        for reason, n in bad.items():
            drops[kind][reason] += n

    # reviews
    ok = _validated("reviews", _items(data, "reviews"), supplied, SESSION_ALLOWED, drops, note_loss)
    for rv in ok:
        mid = _str(rv.get("message_id"))
        if mid not in msgs or rv.get("kind") not in REVIEW_KINDS:
            drops["reviews"]["bad_message_or_kind"] += 1
            continue
        if not is_review(rv["kind"], cited_remark(msgs[mid].text, rv["citations"], mid)):
            drops["reviews"]["not_a_review"] += 1  # an opener, thanks or plan suggests nothing about the document
            continue
        upto = [m.text for m in msgs.values() if order[m.id] <= order[mid]]
        docs, why = resolve_review_docs(cat, upto, _str(rv.get("document_ref")), doc_prev)
        if not docs:
            drops["reviews"]["document_unresolved"] += 1
            unresolved_reviews.append({"record": mid, "why": why, "text": _str(rv.get("text")),
                                       "document_ref": _str(rv.get("document_ref")) or None})
            continue
        reviews.append({"docs": [d_.id for d_ in docs], "eids": [msgs[mid].user_id], "ambiguous": False,
                        "kind": rv["kind"], "section": _str(rv.get("section")) or None,
                        "text": _str(rv.get("text")), "record": mid, "citations": rv["citations"], "speaker": None,
                        "source_text": msgs[mid].text[:600], "doc_basis": why})
        for d_ in docs:
            discusses.add((f"Message:{mid}", f"Document:{d_.id}"))

    # claims
    raw_claims, pos = [], Counter()
    for c in _items(data, "claims"):  # R13: n is the claim's position in its record's output, dropped ones included
        raw_claims.append({**c, "_n": pos[_str(c.get("message_id"))]})
        pos[_str(c.get("message_id"))] += 1
    ok = _validated("claims", raw_claims, supplied, SESSION_ALLOWED, drops, note_loss)
    for c in ok:
        mid, kind = _str(c.get("message_id")), c.get("kind")
        if mid not in msgs or kind not in CLAIM_KINDS or not _str(c.get("text")):
            drops["claims"]["bad_message_or_kind"] += 1
            continue
        subject = _str(c.get("subject_ref"))
        if kind in COMPETITOR_KINDS:
            key = normalise(subject)
            if not key or key in own_keys:
                drops["claims"]["subject_is_product"] += 1
                continue
            known = cat.resolve_competitor(subject) is not None
            if key not in text_norm and not known:
                drops["claims"]["subject_not_in_session"] += 1
                continue
            if not known and not any(COMP_CUE.search(m.text) and key in normalise(m.text) for m in msgs.values()):
                drops["claims"]["subject_not_a_competitor"] += 1  # named, but never called a competitor or rival
                continue
            if sr.URL_IN_TEXT.search(msgs[mid].text) and "demo" in msgs[mid].text.lower():
                drops["claims"]["demo_share_message"] += 1  # sharing a demo link states no feature or weakness
                continue
            # I3: the cited discussion must name the competitor; a thread reply may rely on the opener when the
            # session names exactly one competitor
            names = {n for e in cat.competitors.entries.values() for n in [e["name"]]}
            names |= {a for al in competitor_alias.values() for a in al}
            cited_msgs = " ".join(msgs[r_["record_id"]].text for r_ in c["citations"] if r_["record_id"] in msgs)
            if not _names_in(cited_msgs + " " + msgs[mid].text, [subject]):
                before = " ".join(m.text for m in msgs.values() if order[m.id] <= order[mid])
                if not _names_in(before, [subject]) or len({normalise(n) for n in _names_in(text_all, names)}) > 1:
                    drops["claims"]["subject_not_grounded"] += 1
                    continue
            cid = cat.register_competitor(subject, citations=c["citations"])
            competitor_alias[cid].add(subject)
            target, subj = cid, cat.competitors.entries[cid]["name"]
        else:
            target, subj = f"Product:{rec.product}", rec.product
        cust = []
        if kind in CUSTOMER_KINDS:
            cust = [x for x in dict.fromkeys(_list(c.get("customer_ids"))) if isinstance(x, str)]
            cited_text = " ".join(msgs[r["record_id"]].text for r in c["citations"] if r["record_id"] in msgs)
            if any(CUST_RE.fullmatch(x) is None or x not in text_all for x in cust):
                drops["claims"]["customer_not_in_session"] += 1
                continue
            cust = [x for x in cust if x in cited_text or x in msgs[mid].text]
            if not cust:
                drops["claims"]["no_customer_in_cited_message"] += 1
                continue
        reporters = []
        if kind == "qa_issue":
            for m in [m for m in msgs.values() if order[m.id] <= order[mid]][::-1]:
                valid, _, _ = named_qa_eids(m.text, rec.employees)
                if valid:
                    reporters = valid
                    break
        scope, basis = "unknown", "none"
        if kind not in COMPETITOR_KINDS:
            # R2: only record-specific evidence. The claim's own message says last/previous release (rules anchor), or
            # the session it belongs to does (a rules anchor, or the cue words in its text). The model's own scope
            # label is not evidence either way.
            if mid in anchors:
                scope, basis = "previous", "phrase"
            elif anchor_session:
                scope, basis = "previous", "anchor_session"
            elif session_prev:
                scope, basis = "previous", "session_cue"
            elif c.get("release_scope") == "previous":
                state["scope_downgraded"] += 1
        observations.append({"id": f"Observation:{rec.product}:{mid}:{c['_n']}", "kind": kind, "text": _str(c["text"]),
                             "release_scope": scope, "scope_basis": basis, "subject": subj, "record": mid,
                             "customer_ids": cust, "citations": c["citations"], "author": msgs[mid].user_id,
                             "target": target, "reporters": reporters})

    # PR decisions
    ok = _validated("pr_decisions", _items(data, "pr_decisions"), supplied, SESSION_ALLOWED, drops, note_loss)
    unresolved = state.setdefault("unresolved_prs", [])
    ok.sort(key=lambda d: order.get(_str(d.get("message_id")), 0))
    for d in ok:
        mid = _str(d.get("message_id"))
        if mid not in msgs or d.get("action") not in ACTIONS:
            drops["pr_decisions"]["bad_message_or_action"] += 1
            continue
        act = d["action"]
        if act in ACTION_CUES and not ACTION_CUES[act].search(msgs[mid].text):
            drops["pr_decisions"]["no_evidence_for_action"] += 1
            continue
        conditional = bool(d.get("conditional"))
        if act in ACTION_CUES:  # I4: a question or a condition is not an effective decision
            mood = sentence_mood(msgs[mid].text, ACTION_CUES[act])
            if mood == "question":
                act = "question"
            elif mood == "conditional":
                conditional = True
        reason = d.get("reason") if d.get("reason") in REASONS else None
        if reason in REASON_CUES and not REASON_CUES[reason].search(
                " ".join(m.text for m in msgs.values() if order[m.id] <= order[mid])):
            drops["pr_decisions"]["reason_unsupported"] += 1
            reason = None
        for idx, ref in enumerate(_list(d.get("prs"))):
            p, why = sr.resolve_pr(ref, cat)
            if p is None:
                unresolved.append({"message": mid, "pr": str(ref), "why": why})
                continue
            num_ok = p.link in text_all or re.search(r"(?<!\d)" + re.escape(p.number) + r"(?!\d)", text_all)
            if not num_ok:
                drops["pr_decisions"]["pr_not_in_session"] += 1
                continue
            if act in ("merge", "resume"):
                # a decision to merge answers a pending question or hold; an author's routine "let's get this
                # merged" after review does not (the extraction contract)
                num = re.compile(r"(?<!\d)" + re.escape(p.number) + r"(?!\d)")
                asked = any("?" in m.text and re.search(r"merg|hold|revert|direction|close", m.text, re.I)
                            and (p.link in m.text or num.search(m.text))
                            for m in msgs.values() if order[m.id] < order[mid])
                pending = any(x["pr"] == p.link and x["action"] in ("question", "withhold") for x in decisions)
                if not (asked or pending):
                    drops["pr_decisions"]["routine_request"] += 1
                    continue
            decisions.append({"message": mid, "pr": p.link, "action": act, "reason": reason,
                              "conditional": conditional, "ts": msgs[mid].ts, "idx": idx,
                              "citations": d["citations"], "text": msgs[mid].text})
