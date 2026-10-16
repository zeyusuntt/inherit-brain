"""Layer 2a: deterministic rules over the typed records. No model calls.

Produces the Product root with aliases and previous names, release scope from record-specific evidence only (R2),
PR repository scope and approval, REVERTS with the reason clause classified (R4), demo links, and literal QA eids
named in templated sentences (R9). Each rule has a version constant; the report returned beside the facts says what
was found and what was left for the semantic layer (4c).

Scope values are `previous` or `unknown`. A channel name or an alias never sets scope by itself: LeadForce's
`planning-AIX` channel later shares `new_aix_*` documents.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from inherit.data.records import ProductRecords, Review
from inherit.extract.catalog import build_catalog, doc_stem, repo_of
from inherit.extract.sessions import sessions
from inherit.extract.structural import _links
from inherit.facts import Edge, Facts, Node
from inherit.ontology import check_edge

EXTRACTOR = "rules/v1"
APPROVAL_VERSION = "pr_approval/v1"
REVERT_VERSION = "revert/v1"
SCOPE_VERSION = "scope/v1"
NAMED_QA_VERSION = "named_qa/v1"
DEMO_VERSION = "demo/v1"
VERSIONS = {"approval": APPROVAL_VERSION, "revert": REVERT_VERSION, "scope": SCOPE_VERSION,
            "named_qa": NAMED_QA_VERSION, "demo": DEMO_VERSION}

# ------------------------------------------------------------------ PR approval

def pr_approval(reviews) -> str:
    """approved: at least one APPROVED review and no CHANGES_REQUESTED after the last APPROVED one (by submitted_at,
    then position). unapproved: it has reviews and is not approved. unreviewed: no reviews. Review states are
    authoritative; comments never change the result."""
    if not reviews:
        return "unreviewed"
    ordered = sorted(reviews, key=lambda r: (r.submitted_at, r.position))
    last = max((i for i, r in enumerate(ordered) if r.state == "APPROVED"), default=None)
    if last is None:
        return "unapproved"
    return "unapproved" if any(r.state == "CHANGES_REQUESTED" for r in ordered[last + 1:]) else "approved"


_APPROVES = re.compile(r"\b(approved|lgtm)\b", re.IGNORECASE)
_ASKS_CHANGES = re.compile(
    r"\b(changes? (?:are |is )?(?:requested|required|needed)|needs? (?:more )?(?:work|changes|fixes)"
    r"|not (?:yet )?approved|do not merge|don't merge)\b", re.IGNORECASE)


def _comment_says_approved(text: str) -> bool:
    for m in _APPROVES.finditer(text or ""):
        before = text[max(0, m.start() - 8):m.start()].lower()
        if "not " in before or "n't " in before or before.endswith("un"):
            continue
        return True
    return False


def review_contradiction(r: Review) -> str | None:
    if r.state != "APPROVED" and _comment_says_approved(r.comment):
        return "comment_says_approved_state_is_not"
    if r.state == "APPROVED" and _ASKS_CHANGES.search(r.comment or ""):
        return "state_approved_comment_asks_for_changes"
    return None


# ------------------------------------------------------------------ reverts

_REVERT_TITLE = re.compile(r"^\s*revert(?:s|ed)?\s+(?:pr\s*)?#\s*(\d+)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9'\"])")
_DIRECTION = re.compile(
    r"product direction|product strateg|change in direction|(?:new|updated|current) (?:strategy|roadmap)|roadmap"
    r"|strategic|pivot|shift(?:ed|ing)? (?:in|of|away|to|towards)\b|moving away|no longer aligned", re.IGNORECASE)
_SECURITY = re.compile(
    r"security|vulnerab|insecure|unauthori[sz]ed|data (?:breach|exposure|leak)|sensitive (?:data|information)|exploit",
    re.IGNORECASE)
_BUGS = re.compile(
    r"\bbugs?\b|crash|critical issues?|regression|degradation|defects?|\bfailures?\b|broken|malfunction|\berrors?\b"
    r"|outage", re.IGNORECASE)
_REASONS = (("product_direction", _DIRECTION), ("security", _SECURITY), ("introduced_bugs", _BUGS))


def parse_revert_title(title: str) -> int | None:
    m = _REVERT_TITLE.match(title or "")
    return int(m.group(1)) if m else None


def classify_revert_reason(summary: str) -> tuple[str, str]:
    """(reason, clause). The clause is the first sentence of the revert PR's own summary; the target feature's title is
    never read. The earliest cue in the clause wins. No cue means `other`; the clause is kept either way."""
    clause = (_SENTENCE_END.split((summary or "").strip(), maxsplit=1)[0]).strip()
    best = None
    for reason, rx in _REASONS:
        m = rx.search(clause)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), reason)
    return (best[1] if best else "other"), clause


# ------------------------------------------------------------------ named QA eids

_BARE_EID = re.compile(r"(?<![@\w])(?<!@')eid_[0-9a-f]{8}\b")
_QA_BEFORE = re.compile(r"\bQA\b[^.!?\n]{0,40}$")


def named_qa_eids(text: str, directory) -> tuple[list[str], list[str], list[str]]:
    """Bare `eid_xxxxxxxx` text (no @) in a sentence where a QA role word shortly precedes it: (valid, invalid, other).
    valid are in the employee directory; invalid are QA-templated but unknown; other are bare ids outside a QA
    template (left for 4c). A mention (`@eid_...`) is not a bare id."""
    valid: list[str] = []
    invalid: list[str] = []
    other: list[str] = []
    for m in _BARE_EID.finditer(text or ""):
        eid = m.group(0)
        start = max((text.rfind(c, 0, m.start()) for c in ".!?\n"), default=-1) + 1
        if _QA_BEFORE.search(text[start:m.start()]):
            bucket = valid if eid in directory else invalid
        else:
            bucket = other
        if eid not in bucket:
            bucket.append(eid)
    return valid, invalid, other


# ------------------------------------------------------------------ release scope

_PREV_PHRASE = re.compile(r"\b(?:last|previous|prior|latest)\s+release\b|\b(?:last|previous|prior)\s+version\b",
                          re.IGNORECASE)
_PREV_DOCS = re.compile(r"\bdocuments?\s+from\s+the\s+(?:last|previous|prior)\s+release\b", re.IGNORECASE)
_DEMO = re.compile(r"\bdemo\b", re.IGNORECASE)
_URL_PATH_ALIAS = re.compile(r"/archives/([^/]+)/demo_\d+")


def _is_bug_channel(name: str) -> bool:
    return name.startswith("bug-")


def _names_in(text: str, names: dict[str, str]) -> list[str]:
    """Distinct product names (product and aliases, display casing) that occur as whole words in the text."""
    found = []
    for low, display in names.items():
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(low) + r"(?![A-Za-z0-9])", text or "", re.IGNORECASE):
            found.append(display)
    return found


def run_rules(rec: ProductRecords) -> tuple[Facts, dict]:
    product = rec.product
    cat = build_catalog(rec)
    names = {product.lower(): product, **{a.lower(): a for a in cat.aliases}}
    nodes: list[Node] = []
    edges: list[Edge] = []

    def edge(type_, src, dst, ids, key="", **props):
        e = Edge(type_, src, dst, props, tuple(ids), EXTRACTOR, key)
        check_edge(e)
        edges.append(e)

    # ---- documents shared as the previous release, from the Slack sessions that say so
    doc_by_link = {d.link: d for d in rec.documents}
    by_channel = defaultdict(list)
    for m in rec.messages:
        by_channel[m.channel_id].append(m)
    doc_scope: dict[str, tuple[str, list[str]]] = {}  # doc id -> (basis, anchor message ids)
    anchors: dict[str, str] = {}  # message id -> channel id, messages whose own text says last/previous release
    for msgs in by_channel.values():
        for sess in sessions(msgs):
            doc_anchors = [m for m in sess if not m.is_bot and _PREV_DOCS.search(m.text)
                           and not _is_bug_channel(m.channel_name)]
            for m in sess:
                if not m.is_bot and not _is_bug_channel(m.channel_name) and _PREV_PHRASE.search(m.text):
                    anchors[m.id] = m.channel_id
            if not doc_anchors:
                continue
            aid = sorted(m.id for m in doc_anchors)
            for m in sess:
                for link in sorted(_links(m.text)):
                    d = doc_by_link.get(link)
                    if d is None:
                        continue
                    basis = "anchor_link" if m in doc_anchors else "anchor_session"
                    old = doc_scope.get(d.id)
                    if old is None or (old[0] == "anchor_session" and basis == "anchor_link"):
                        doc_scope[d.id] = (basis, [m.id] if basis == "anchor_link" else aid)
                    elif old[0] == basis == "anchor_link" and m.id not in old[1]:
                        old[1].append(m.id)

    # ---- URL records: demo kind, subject, scope
    url_links = {link: u for u in rec.urls for link in u.links}
    sharers: dict[str, list[tuple[str, str, bool]]] = defaultdict(list)  # link -> (message id, text, is previous phrase)
    for m in rec.messages:
        if m.is_bot:
            continue
        for link in _links(m.text):
            if link in url_links:
                sharers[link].append((m.id, m.text, m.id in anchors))
    for c in rec.chats:
        for link in _links(c.text):
            if link in url_links:
                sharers[link].append((c.id, c.text, False))
    url_info: dict[str, dict] = {}
    for link, u in url_links.items():
        desc = " ".join(u.descriptions)
        shared = sharers.get(link, [])
        if not (_DEMO.search(desc) or re.search(r"/demo(?:_\d+)?(?:[/?#]|$)", link)
                or any(_DEMO.search(t) for _, t, _ in shared)):
            continue
        found = sorted({*_names_in(desc, names), *_names_in(" ".join(_URL_PATH_ALIAS.findall(link)), names)})
        prev_msgs = sorted(i for i, _, p in shared if p)
        info = {"id": u.id, "release_scope": "previous" if prev_msgs else "unknown",
                "scope_basis": "shared_in_previous_message" if prev_msgs else "none", "scope_evidence": prev_msgs,
                "sharers": sorted({i for i, _, _ in shared})}
        if len(found) == 1:
            info["subject"] = f"Product:{product}"
            info["source_product_name"] = found[0]
        url_info[link] = info

    # ---- previous names: an alias is previous when every document id and demo path that carries it is previous
    carriers: dict[str, list[tuple[str, bool, list[str]]]] = defaultdict(list)
    for d in rec.documents:
        stem = doc_stem(d.id)
        if stem and stem in names and names[stem] != product:
            carriers[names[stem]].append((d.id, d.id in doc_scope, doc_scope.get(d.id, ("", []))[1]))
    for link, info in url_info.items():
        for a in _URL_PATH_ALIAS.findall(link):
            if a.lower() in names and names[a.lower()] != product:
                carriers[names[a.lower()]].append((info["id"], info["release_scope"] == "previous", info["scope_evidence"]))
    previous_evidence = {}
    for alias, cs in carriers.items():
        if cs and all(p for _, p, _ in cs):
            previous_evidence[alias] = sorted({*(i for i, _, _ in cs), *(e for _, _, ev in cs for e in ev)})
    previous_names = sorted(previous_evidence, key=str.lower)

    nodes.append(Node("Product", f"Product:{product}", {
        "name": product, "aliases": list(cat.aliases), "previous_names": previous_names,
        "alias_evidence": cat.alias_evidence, "previous_evidence": previous_evidence}))

    # ---- documents and meetings: scope props and HAS
    scope_counts = {"Document": Counter(), "Message": Counter(), "Url": Counter(), "Meeting": Counter()}
    for d in rec.documents:
        basis, ev = doc_scope.get(d.id, ("none", []))
        p = {"id": d.id, "release_scope": "previous" if d.id in doc_scope else "unknown",
             "scope_basis": basis, "scope_evidence": ev}
        stem = doc_stem(d.id)
        if stem and stem in names:
            p["source_product_name"] = names[stem]
        nodes.append(Node("Document", f"Document:{d.id}", p))
        edge("HAS", f"Product:{product}", f"Document:{d.id}", [d.id])
        scope_counts["Document"][p["release_scope"]] += 1
    for mt in rec.meetings:
        nodes.append(Node("Meeting", f"Meeting:{mt.id}", {"id": mt.id, "release_scope": "unknown"}))
        edge("HAS", f"Product:{product}", f"Meeting:{mt.id}", [mt.id])
        scope_counts["Meeting"]["unknown"] += 1

    # ---- messages: phrase scope and named QA eids
    directory = rec.employees
    named, left = [], []
    for m in rec.messages:
        p: dict = {}
        if m.id in anchors:
            p.update(release_scope="previous", scope_basis="phrase", scope_evidence=[m.id])
            found = _names_in(m.text, names)
            if len(found) == 1:
                p["source_product_name"] = found[0]
        valid, invalid, other = named_qa_eids(m.text, directory)
        if valid:
            p["named_eids"] = valid
            named += [{"message": m.id, "eid": e} for e in valid]
        left += [{"message": m.id, "eid": e, "why": "not_in_directory"} for e in invalid]
        left += [{"message": m.id, "eid": e, "why": "not_qa_template"} for e in other]
        if p:
            nodes.append(Node("Message", f"Message:{m.id}", {"id": m.id, **p}))
        scope_counts["Message"]["previous" if m.id in anchors else "unknown"] += 1

    # ---- URL nodes and ABOUT
    for link, info in sorted(url_info.items()):
        p = {"link": link, "kind": "demo", "release_scope": info["release_scope"], "scope_basis": info["scope_basis"],
             "scope_evidence": info["scope_evidence"]}
        for k in ("subject", "source_product_name"):
            if k in info:
                p[k] = info[k]
        nodes.append(Node("Url", f"Url:{link}", p))
        if "subject" in info:
            edge("ABOUT", f"Url:{link}", info["subject"], [info["id"]], kind="demo")
    for link, u in url_links.items():
        scope_counts["Url"]["previous" if url_info.get(link, {}).get("release_scope") == "previous" else "unknown"] += 1

    # ---- PRs: repo scope, approval, reverts
    by_link = {p.link: p for p in rec.prs}
    pr_counts: dict[str, Counter] = defaultdict(Counter)
    contradictions, reverts, unparsed = [], [], []
    for p in rec.prs:
        entry = cat.pr(p.link)
        approval = pr_approval(p.reviews)
        props = {"link": p.link, "repo": f"{entry.owner}/{entry.repo}", "repo_scope": entry.repo_scope,
                 "approval": approval, "approval_rule": APPROVAL_VERSION,
                 "open_unmerged": p.state == "open" and p.merged is False}
        pr_counts[entry.repo_scope][approval] += 1
        flagged = [{"position": r.position, "state": r.state, "login": r.login, "why": why,
                    "comment": r.comment[:200]}
                   for r in p.reviews if (why := review_contradiction(r))]
        if flagged:
            contradictions.append({"pr": p.link, "approval": approval, "reviews": flagged})
        n = parse_revert_title(p.title)
        if n is not None:
            reason, clause = classify_revert_reason(p.summary)
            props["reverts_number"] = n
            target = re.sub(r"/pull/\d+$", f"/pull/{n}", p.link)
            applied = p.merged is True
            rv = {"revert": p.link, "original": target if target in by_link else None, "target_number": n,
                  "reason": reason, "clause": clause, "applied": applied, "repo_scope": entry.repo_scope}
            reverts.append(rv)
            if rv["original"]:
                edge("REVERTS", f"PR:{p.link}", f"PR:{target}", [p.id], reason=reason, reason_text=clause,
                     applied=applied, rule=REVERT_VERSION)
        elif re.search(r"\brevert", p.title, re.IGNORECASE):
            unparsed.append(p.link)
        nodes.append(Node("PR", f"PR:{p.link}", props))
        edge("HAS", f"Product:{product}", f"PR:{p.link}", [p.id])

    by_reason: dict[str, dict[str, int]] = {}
    for r in reverts:
        d = by_reason.setdefault(r["reason"], {"applied": 0, "not_applied": 0})
        d["applied" if r["applied"] else "not_applied"] += 1

    demo_list = [{"link": k, "subject": v.get("subject"), "release_scope": v["release_scope"],
                  "source_product_name": v.get("source_product_name"), "sharers": v["sharers"]}
                 for k, v in sorted(url_info.items())]
    in_anchor_session = 0
    for msgs in by_channel.values():
        for sess in sessions(msgs):
            if any(m.id in anchors for m in sess):
                in_anchor_session += sum(1 for m in sess if not m.is_bot and m.id not in anchors)

    report = {
        "product": product,
        "versions": VERSIONS,
        "aliases": {a: {"evidence": cat.alias_evidence[a], "previous": a in previous_evidence} for a in cat.aliases},
        "previous_names": previous_names,
        "previous_evidence": previous_evidence,
        "scope_counts": {k: dict(v) for k, v in scope_counts.items()},
        "anchor_messages": sorted(anchors),
        "pr_counts": {k: dict(v) for k, v in pr_counts.items()},
        "approval_contradictions": contradictions,
        # PRs whose review comments say approved while the review-state predicate says not approved
        "approval_flips": [c["pr"] for c in contradictions if c["approval"] != "approved"
                           and any(r["why"] == "comment_says_approved_state_is_not" for r in c["reviews"])],
        "reverts": reverts,
        "reverts_by_reason": by_reason,
        "revert_titles_unparsed": unparsed,
        "demo_links": demo_list,
        "demos_without_subject": [d["link"] for d in demo_list if d["subject"] is None],
        "named_qa_eids": named,
        "bare_eids_left": left,
        "left_for_4c": {
            "demos_without_subject": sum(1 for d in demo_list if d["subject"] is None),
            "messages_in_previous_phrase_sessions_without_phrase": in_anchor_session,
            "bare_eids_left": len(left),
            "revert_titles_unparsed": len(unparsed),
            "reverts_missing_target": sum(1 for r in reverts if r["original"] is None),
        },
    }
    return Facts([], []).merge(Facts(nodes, edges)), report
