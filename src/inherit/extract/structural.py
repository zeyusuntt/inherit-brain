"""Layer 1: deterministic facts straight from the typed records. No model calls.

Node ids and edge signatures are defined in `inherit.ontology`. Every edge's `src` holds the IDs of the
records it came from (document id, meeting id, utterance id, PR record id).
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from inherit.data.records import ProductRecords, build_records
from inherit.extract.sessions import sessions
from inherit.facts import Edge, Facts, Node
from inherit.ontology import check_edge

EXTRACTOR = "structural/v1"
CUST_RE = re.compile(r"\bCUST-\d+\b")
URL_RE = re.compile(r"https?://[^\s<>|\]]+")
KINDS = ("co_attend", "pr_review", "mention", "session")


def _links(text: str) -> set[str]:
    # Trailing sentence punctuation and a closing paren are not part of a link.
    return {m.rstrip(".,;:!?)'\"") for m in URL_RE.findall(text)}


def structural_facts(bundle_product: dict, employees: dict, customers: list, product: str) -> Facts:
    """structural_facts(bundle_product, employees, customers, product). The product name is needed to parse
    bug channel names (`bug-<owner>-<Product>`)."""
    return structural_facts_from_records(build_records(bundle_product, employees, customers, product))


def structural_facts_from_records(rec: ProductRecords) -> Facts:
    nodes: list[Node] = []
    edges: list[Edge] = []
    # pair -> kind -> set of (record id, dedupe token). The set makes each event count once.
    collab: dict[tuple[str, str], dict[str, set]] = defaultdict(lambda: {k: set() for k in KINDS})
    cust_ids: set[str] = set()

    def person(eid: str) -> str:
        emp = rec.employees.get(eid)
        nodes.append(Node("Person", f"Person:{eid}", {
            "eid": eid, "name": emp["name"] if emp else None, "role": emp["role"] if emp else None,
            "in_directory": emp is not None}))
        return f"Person:{eid}"

    def edge(type_, src, dst, ids, key="", **props):
        e = Edge(type_, src, dst, props, tuple(ids), EXTRACTOR, key)
        check_edge(e)
        edges.append(e)

    def pair(a: str, b: str, kind: str, token, ids) -> None:
        if a == b:
            return
        p = (a, b) if a < b else (b, a)
        collab[p][kind].add((token, tuple(ids)))

    def scan_customers(*texts) -> None:
        for t in texts:
            cust_ids.update(CUST_RE.findall(t or ""))

    # Documents
    doc_by_link: dict[str, str] = {}
    for d in rec.documents:
        nodes.append(Node("Document", f"Document:{d.id}", {
            "id": d.id, "type": d.type, "link": d.link, "date": d.date, "author": d.author,
            "content": d.content, "feedback": d.feedback}))
        edge("AUTHORED", person(d.author), f"Document:{d.id}", [d.id])
        doc_by_link[d.link] = d.id
        scan_customers(d.content, d.feedback)

    # Meetings
    for m in rec.meetings:
        nodes.append(Node("Meeting", f"Meeting:{m.id}", {
            "id": m.id, "date": m.date, "document_type": m.document_type, "participants": list(m.participants)}))
        who = sorted(set(m.participants))
        for eid in who:
            edge("ATTENDED", person(eid), f"Meeting:{m.id}", [m.id])
        for a, b in combinations(who, 2):
            pair(f"Person:{a}", f"Person:{b}", "co_attend", m.id, [m.id])
        scan_customers(*(s.text for s in m.segments))
    for c in rec.chats:
        scan_customers(c.text)

    # PRs
    pr_by_link: dict[str, str] = {}
    for p in rec.prs:
        pid = f"PR:{p.link}"
        nodes.append(Node("PR", pid, {
            "link": p.link, "number": p.number, "title": p.title, "summary": p.summary, "merged": p.merged,
            "merged_raw": p.merged_raw, "mergeable": p.mergeable, "state": p.state, "created_at": p.created_at}))
        pr_by_link[p.link] = pid
        author = person(p.author)
        edge("OPENED", author, pid, [p.id])
        scan_customers(p.title, p.summary)
        for r in p.reviews:
            reviewer = person(r.login)
            edge("REVIEWED", reviewer, pid, [p.id], str(r.position),
                 state=r.state, submitted_at=r.submitted_at, comment=r.comment)
            pair(author, reviewer, "pr_review", p.id, [p.id])
            scan_customers(r.comment)

    # URLs: one node per link; records with the same link are merged, all descriptions kept.
    url_ids: dict[str, list[str]] = {}
    url_desc: dict[str, list[str]] = {}
    for u in rec.urls:
        for link in u.links:
            ids = url_ids.setdefault(link, [])
            if u.id not in ids:
                ids.append(u.id)
            for desc in u.descriptions:
                if desc not in url_desc.setdefault(link, []):
                    url_desc[link].append(desc)
        scan_customers(*u.descriptions)
    for link in url_ids:
        nodes.append(Node("Url", f"Url:{link}", {"link": link, "ids": url_ids[link], "descriptions": url_desc[link]}))

    # Slack
    for ch_id in dict.fromkeys(m.channel_id for m in rec.messages):
        first = next(m for m in rec.messages if m.channel_id == ch_id)
        nodes.append(Node("Channel", f"Channel:{ch_id}", {
            "id": ch_id, "name": first.channel_name, "bug_owner": first.bug_channel_owner}))
    for m in rec.messages:
        mid = f"Message:{m.id}"
        nodes.append(Node("Message", mid, {
            "id": m.id, "ts": m.ts, "text": m.text, "channel_id": m.channel_id, "user_id": m.user_id,
            "kind": "slack"}))
        edge("IN_CHANNEL", mid, f"Channel:{m.channel_id}", [m.id])
        scan_customers(m.text)
        poster = None if m.is_bot else person(m.user_id)
        if poster:
            edge("POSTED", poster, mid, [m.id])
            for link in sorted(_links(m.text)):
                for table, label in ((doc_by_link, "Document"), (pr_by_link, "PR")):
                    if link in table:
                        target = f"Document:{table[link]}" if label == "Document" else table[link]
                        edge("SHARED", poster, target, [m.id])
                if link in url_ids:
                    edge("SHARED", poster, f"Url:{link}", [m.id])
        for eid in dict.fromkeys(m.mentions):
            target = person(eid)
            edge("MENTIONS", mid, target, [m.id])
            if poster:
                pair(poster, target, "mention", m.id, [m.id])
    by_channel: dict[str, list] = defaultdict(list)
    for m in rec.messages:
        by_channel[m.channel_id].append(m)
    for ch_id, msgs in by_channel.items():
        for si, sess in enumerate(sessions(msgs)):
            posters: dict[str, list[str]] = defaultdict(list)
            for m in sess:
                if not m.is_bot:
                    posters[f"Person:{m.user_id}"].append(m.id)
            for a, b in combinations(sorted(posters), 2):
                pair(a, b, "session", f"{ch_id}/{si}", posters[a] + posters[b])

    for cid in sorted(cust_ids):
        c = rec.customers.get(cid, {})
        nodes.append(Node("Customer", f"Customer:{cid}", {
            "id": cid, "company": c.get("company"), "name": c.get("name"), "role": c.get("role")}))

    for (a, b), kinds in collab.items():
        counts = {k: len(kinds[k]) for k in KINDS}
        ids = sorted({i for k in KINDS for _, tup in kinds[k] for i in tup})
        edge("COLLABORATED", a, b, ids, weight=sum(counts.values()), counts=counts)

    return Facts([], []).merge(Facts(nodes, edges))
