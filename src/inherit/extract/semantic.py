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
from inherit.extract.semantic_prompts import FEATURE_GROUP_SYSTEM, MEETING_SYSTEM, PR_MEMBERSHIP_SYSTEM, SEMANTIC_PROMPT_VERSION, SESSION_SYSTEM
from inherit.extract import semantic_rules as sr
from inherit.extract.sessions import sessions
from inherit.extract.structural import _links
from inherit.facts import Edge, Facts, Node
from inherit.ontology import check_edge

EXTRACTOR = "semantic/v1"
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
EVENT_KINDS = {"discussed", "implemented", "held", "resumed", "reverted"}
REASONS = {"security", "product_direction", "introduced_bugs", "other"}
COMPETITOR_KINDS = {"competitor_feature", "competitor_weakness"}
CLAIM_KINDS = COMPETITOR_KINDS | {"customer_issue", "customer_strength", "qa_issue"}
CUSTOMER_KINDS = {"customer_issue", "customer_strength"}
ACTIONS = {"withhold", "resume", "merge", "revert", "question"}
CUST_RE = re.compile(r"\bCUST-\d+\b")
DOC_TYPE_RE = re.compile(sr.DOC_WORDS, re.IGNORECASE)
PRODUCT_PLANNING = "product-planning"
# Wording a cited record must contain for a model feature event of that kind (the model over-reads plans as results).
EVENT_CUES = {
    "implemented": re.compile(r"complet|\bmerged\b|finish|\bdone\b|shipp|deliver|deploy|launch|implemented|released"
                              r"|wrapped up|resolved|went live|in production", re.I),
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
    doc_feedback = {d.id for d in rec.documents if d.feedback}

    def pick_doc(type_name):
        """One document of a type named in a session: not the previous release's, then one that has feedback."""
        c = [d for d in cat.documents if normalise(d.type) == normalise(type_name)]
        c = [d for d in c if d.id not in doc_prev] or c
        return ([d for d in c if d.id in doc_feedback] or c or [None])[0]

    prs_raw = {p.link: p for p in rec.prs}

    drops: dict[str, Counter] = defaultdict(Counter)
    failures: list[dict] = []
    unresolved_speakers: list[dict] = []
    nodes: list[Node] = []
    edges: list[Edge] = []
    people: set[str] = set()
    customers_seen: set[str] = set()
    state = {"unresolved_feature_refs": 0, "split_sessions": 0, "scope_downgraded": 0}

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

    def register_feature(title, details, cites, record):
        fid = cat.register_feature(title, citations=cites)
        e = feature_info.setdefault(fid, {"title": cat.features.entries[fid]["title"], "details": [], "records": []})
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
        data, reason = caller.json_call("meeting", req, mt_tokens)
        if data is None:
            failures.append({"kind": "meeting", "id": mt.id, "reason": reason})
            meeting_failed.append(mt.id)
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

        def feature_ref(ref):
            fid = local.get(_str(ref)) or (cat.resolve_feature(_str(ref)) if _str(ref) else None)
            if fid is None:
                state["unresolved_feature_refs"] += 1
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
        for f in ok:
            title = _str(f.get("title"))
            if not title:
                drops["features"]["no_title"] += 1
                continue
            details = [_str(x) for x in _list(f.get("details")) if _str(x)]
            fid = register_feature(title, details, f["citations"], mt.id)
            if _str(f.get("local_id")):
                local[_str(f["local_id"])] = fid
            discusses.add((f"Meeting:{mt.id}", fid))

        ok, bad = validate_items(_items(data, "contributions"), supplied, MEETING_ALLOWED)
        lose("contributions", bad)
        for c in ok:
            fid = feature_ref(c.get("feature_ref"))
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

        ok, bad = validate_items(_items(data, "development_links"), supplied, MEETING_ALLOWED)
        lose("development_links", bad)
        for lk in ok:
            fid = feature_ref(lk.get("feature_ref"))
            if fid is None:
                drops["development_links"]["feature_unresolved"] += 1
                continue
            entries, why = pr_entries(lk.get("pr"), text_norm)
            if not entries:
                drops["development_links"][why] += 1
                continue
            for p in entries:
                dev_links[p.link].append({"feature": fid, "basis": "meeting_title", "citations": lk["citations"],
                                          "record": mt.id})

        ok, bad = validate_items(_items(data, "feature_events"), supplied, MEETING_ALLOWED)
        lose("feature_events", bad)
        for ev in ok:
            fid = feature_ref(ev.get("feature_ref"))
            ref = _str(ev.get("segment_ref"))
            if ev.get("kind") not in EVENT_KINDS:
                drops["feature_events"]["bad_kind"] += 1
                continue
            if fid is None:
                drops["feature_events"]["feature_unresolved"] += 1
                continue
            cited = " ".join(r.text for r in supplied if r.id in {c["record_id"] for c in ev["citations"]} | {record_of.get(ref)})
            if ev["kind"] in EVENT_CUES and not EVENT_CUES[ev["kind"]].search(cited):
                drops["feature_events"]["no_evidence_for_kind"] += 1
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

        ok, bad = validate_items(_items(data, "document_reviews"), supplied, MEETING_ALLOWED)
        lose("document_reviews", bad)
        doc = None
        for c in mchats:
            for link in sorted(_links(c.text)):
                doc = doc or cat.doc_by_link(link)
        if doc is None and mt.document_type and mt.document_type != PRODUCT_PLANNING:
            types = cat.docs_of_type(mt.document_type)
            doc = types[0] if types else None
        if doc is not None and mt.document_type != PRODUCT_PLANNING:
            discusses.add((f"Meeting:{mt.id}", f"Document:{doc.id}"))
        for rv in ok:
            ref = _str(rv.get("segment_ref"))
            if rv.get("kind") not in REVIEW_KINDS or ref not in speaker_of:
                drops["document_reviews"]["bad_kind_or_segment"] += 1
                continue
            if doc is None:
                drops["document_reviews"]["document_unresolved"] += 1
                continue
            sp = speaker_of[ref]
            res = lookup(name_rows, mt.id, sp) if sp else None
            if res is None or res.status == "unresolved":
                unresolved_speakers.append({"meeting": mt.id, "segment": ref, "speaker": sp})
                continue
            reviews.append({"doc": doc.id, "eids": res.ids, "ambiguous": res.ambiguous, "kind": rv["kind"],
                            "section": _str(rv.get("section")) or None, "text": _str(rv.get("text")),
                            "record": record_of[ref], "citations": rv["citations"], "speaker": sp,
                            "source_text": next((r_.text for r_ in supplied if r_.id == record_of[ref]), "")[:600]})

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
        data, reason = caller.json_call("feature_group", req, min(3000, 300 + 60 * len(fl)))
        if data is None:
            failures.append({"kind": "feature_group", "id": product, "reason": reason})
            state["unresolved_feature_refs"] += 1
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
            data, reason = caller.json_call("session", req, tokens)
            if data is None:
                failures.append({"kind": "session", "id": chunk[0].id, "reason": reason})
                failed_sessions.append({"first_message": chunk[0].id, "reason": reason,
                                        "text": _topic(" ".join(m.text for m in chunk))})
                continue
            session_sums[group]["extracted"] += 1
            _process_session(data, chunk, rec, cat, anchors, own_keys, drops, state, reviews, observations,
                             decisions, discusses, competitor_alias, pick_doc)

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
        data, reason = caller.json_call("membership", req, tokens)
        if data is None:
            failures.append({"kind": "membership", "id": batch[0].link, "reason": reason})
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
                continue
            p = rows[int(m.group(1)) - 1]["entry"]
            pr_src = [SourceRecord(prs_raw[p.link].id, "pr", "title", prs_raw[p.link].title),
                      SourceRecord(prs_raw[p.link].id, "pr", "summary", prs_raw[p.link].summary)]
            cite = {"record_id": prs_raw[p.link].id, "quote": a.get("quote")}
            r = validate_citation(cite, pr_src, PR_ALLOWED)
            if not r.ok:
                drops["pr_membership"][r.reason] += 1
                continue
            dev_links[p.link].append({"feature": feats[int(fm.group(1)) - 1]["id"], "basis": "batch",
                                      "citations": [{**cite, "field": r.record.field, "span": list(r.span)}],
                                      "record": p.link})
    unassigned = [p.link for p in product_prs if p.link not in dev_links]

    # ---- 6. derived outcomes
    feat_of_pr: dict[str, list[str]] = defaultdict(list)
    for link, ls in dev_links.items():
        for l_ in ls:
            if l_["feature"] not in feat_of_pr[link]:
                feat_of_pr[link].append(l_["feature"])
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
    # applied reverts of a feature's PR
    for rv in rules_rep["reverts"]:
        if not rv["applied"] or not rv["original"] or rv["repo_scope"] != "product":
            continue
        rpr = prs_raw.get(rv["revert"])
        for fid in feat_of_pr.get(rv["original"], []):
            fevents.append({"feature": fid, "kind": "reverted", "reason": rv["reason"], "reason_text": rv["clause"],
                            "date": rpr.created_at, "record": rpr.id, "citations": [], "prs": [rv["original"]],
                            "source": "revert"})

    # number events per record (R13) and build event nodes
    per_record: Counter = Counter()
    for ev in fevents:
        ev["n"] = per_record[ev["record"]]
        per_record[ev["record"]] += 1
        ev["id"] = f"FeatureEvent:{product}:{ev['record']}:{ev['n']}"

    # fate predicates
    feature_text = {fid: [fid.split(":", 2)[2], *(_topic(a) for a in cat.features.entries[fid]["aliases"])]
                    for fid in feature_info}
    meetings_ok = not meeting_failed
    refs_ok = state["unresolved_feature_refs"] == 0
    fate: dict[str, dict] = {}
    for fid in feature_info:
        evs = [e for e in fevents if e["feature"] == fid]
        kinds = Counter(e["kind"] for e in evs)
        implemented = kinds["implemented"] > 0 or any(f == fid for (f, _) in implemented_pr)
        reverted = kinds["reverted"] > 0
        held_dates = sorted(e["date"] for e in evs if e["kind"] == "held")
        later = [e for e in evs if held_dates and e["kind"] in ("resumed", "implemented") and e["date"] > held_dates[0]]
        blocked_by_session = any(any(t and t in fs["text"] for t in feature_text[fid]) for fs in failed_sessions)
        # a feature lost to a validation failure is absent, so its own unresolved items cannot affect another feature
        complete = meetings_ok and not blocked_by_session
        fate[fid] = {"implemented": implemented, "reverted": reverted, "held": kinds["held"] > 0,
                     "resumed": kinds["resumed"] > 0, "delayed_then_added": bool(held_dates and later),
                     "discussed": True,
                     "discussed_not_implemented": (not implemented) if complete else None,
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
        n = remark_n[(r["record"], r["doc"])]  # several remarks of one message get their own edge
        remark_n[(r["record"], r["doc"])] += 1
        for eid in r["eids"]:
            edge("SUGGESTED", person(eid), f"Document:{r['doc']}", [r["record"]],
                 r["record"] if n == 0 else f"{r['record']}#{n + 1}", section=r["section"], text=r["text"],
                 kind=r["kind"], ambiguous=r["ambiguous"], source_text=r["source_text"])
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
            "record_id": ev["record"], "feature": ev["feature"], "product": product, "pr_links": ev["prs"],
            "source": ev["source"], "citations": ev["citations"]}))
        edge("HAS_EVENT", ev["feature"], ev["id"], [ev["record"]])
    obs_counts: dict[str, Counter] = defaultdict(Counter)
    for o in observations:
        oid = o["id"]
        obs_counts[o["kind"]][o["release_scope"]] += 1
        nodes.append(Node("Observation", oid, {
            "id": oid, "kind": o["kind"], "text": o["text"], "release_scope": o["release_scope"],
            "scope_basis": o["scope_basis"], "subject": o["subject"], "record_id": o["record"], "product": product,
            "customer_ids": o["customer_ids"], "citations": o["citations"]}))
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
        "documents": {"total": len(rec.documents), "with_feedback": n_feedback_docs, "content_calls": 0},
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
        "complete": meetings_ok and refs_ok and not failed_ids,
    }
    report = {
        "product": product, "prompt_version": PROMPT_VERSION, "fate_version": FATE_VERSION,
        "features": len(feature_info), "feature_titles": [e["title"] for e in feature_info.values()],
        "feature_events": dict(Counter(e["kind"] for e in fevents)),
        "contributions": dict(Counter(c["kind"] for c in contributions)),
        "contributors": len({e for c in contributions for e in c["eids"]}),
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
        "calls": {"meeting": caller.calls["meeting"], "feature_group": caller.calls["feature_group"], "session": caller.calls["session"],
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
                     discusses, competitor_alias, pick_doc) -> None:
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
    ok, bad = validate_items(_items(data, "reviews"), supplied, SESSION_ALLOWED)
    lose("reviews", bad)
    for rv in ok:
        mid = _str(rv.get("message_id"))
        if mid not in msgs or rv.get("kind") not in REVIEW_KINDS:
            drops["reviews"]["bad_message_or_kind"] += 1
            continue
        doc = None
        upto = [m for m in msgs.values() if order[m.id] <= order[mid]]
        for m in upto:  # the document shared in this session, latest at or before the remark
            for link in sorted(_links(m.text)):
                doc = cat.doc_by_link(link) or doc
        if doc is None:  # else the document type named closest before the remark
            for m in reversed(upto):
                hit = list(DOC_TYPE_RE.finditer(m.text))
                if hit:
                    doc = pick_doc(hit[-1].group(0))
                    break
        if doc is None:
            drops["reviews"]["document_unresolved"] += 1
            continue
        reviews.append({"doc": doc.id, "eids": [msgs[mid].user_id], "ambiguous": False, "kind": rv["kind"],
                        "section": _str(rv.get("section")) or None, "text": _str(rv.get("text")), "record": mid,
                        "citations": rv["citations"], "speaker": None, "source_text": msgs[mid].text[:600]})
        discusses.add((f"Message:{mid}", f"Document:{doc.id}"))

    # claims
    raw_claims, pos = [], Counter()
    for c in _items(data, "claims"):  # R13: n is the claim's position in its record's output, dropped ones included
        raw_claims.append({**c, "_n": pos[_str(c.get("message_id"))]})
        pos[_str(c.get("message_id"))] += 1
    ok, bad = validate_items(raw_claims, supplied, SESSION_ALLOWED)
    lose("claims", bad)
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
    ok, bad = validate_items(_items(data, "pr_decisions"), supplied, SESSION_ALLOWED)
    lose("pr_decisions", bad)
    unresolved = state.setdefault("unresolved_prs", [])
    for d in ok:
        mid = _str(d.get("message_id"))
        if mid not in msgs or d.get("action") not in ACTIONS:
            drops["pr_decisions"]["bad_message_or_action"] += 1
            continue
        act = d["action"]
        if act in ACTION_CUES and not ACTION_CUES[act].search(msgs[mid].text):
            drops["pr_decisions"]["no_evidence_for_action"] += 1
            continue
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
            decisions.append({"message": mid, "pr": p.link, "action": d["action"], "reason": reason,
                              "conditional": bool(d.get("conditional")), "ts": msgs[mid].ts, "idx": idx,
                              "citations": d["citations"]})
