"""Dev acceptance audit tooling (Task 13 step C, rulings R4 to R8).

    uv run python -m eval.audit_extraction packets --round 1
    uv run python -m eval.audit_extraction batches --round 1 --size 10
    uv run python -m eval.audit_extraction collect --round 1
    uv run python -m eval.audit_extraction third --round 1
    uv run python -m eval.audit_extraction gates --round 1

No model is called here. The fan-out (Codex and a Claude subagent per batch, then a third Codex pass) reads the batch
files and writes verdict files; this module builds the packets and prompts, collects and adjudicates the verdicts and
computes the gates. Dev products only. Evidence is hydrated from the allowlisted bundle (never from the graph, never
from question files). Recall packets carry no extraction; the comparison with extraction is made in code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from inherit.config import ROOT
from inherit.data.records import ProductRecords, load_records
from inherit.data.split import SEED as SPLIT_SEED
from inherit.data.split import split
from inherit.extract.sessions import sessions

BASE_SEED = SPLIT_SEED  # 20261015; round N samples with BASE_SEED + N
AUDIT_DIR = ROOT / "runs" / "audit"
BUNDLE_DIR = ROOT / "data" / "bundle"
FACTS_DIR = ROOT / "data" / "facts"
PROMPT_VERSION = "audit-prompts/v1"

PRECISION_QUOTAS = {"bug_event": 50, "review_relation": 50, "name_resolution": 50,
                    "whatif_unowned": 30, "whatif_edge": 30, "whatif_path": 20}
RECALL_QUOTAS = {"recall_bug": 3, "recall_reviewer": 3}
KINDS = tuple(PRECISION_QUOTAS) + tuple(RECALL_QUOTAS)
LAYERS = ("bugs", "semantic", "names_facts", "structural", "rules")
BUG_FIELDS = ("identity", "customer", "assignee", "status", "time")
EVENT_TYPES = ("REPORTED", "ASSIGNED", "RESOLVED", "DISMISSED", "REOPENED")
VERDICTS = ("correct", "incorrect", "uncertain")
GATE_PRECISION = 0.90
GATE_RECALL = 0.85
MAX_LATER = 60  # later-context messages per what-if packet; larger contexts are cut and the cut is recorded
EID_RE = re.compile(r"\beid_[A-Za-z0-9_]+")
CUST_RE = re.compile(r"\bCUST-\d+\b")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- the raw side of one product
class Source:
    """One dev product: typed raw records (bundle) and its extracted fact rows (facts layers, names)."""

    def __init__(self, product: str, records: ProductRecords, facts: dict[str, list[dict]], names: list[dict]):
        self.product, self.records, self.facts, self.names = product, records, facts, names
        r = records
        self.msg = {m.id: m for m in r.messages}
        self.pr = {p.id: p for p in r.prs}
        self.pr_by_link = {p.link: p for p in r.prs}
        self.meeting = {m.id: m for m in r.meetings}
        self.seg = {s.id: s for m in r.meetings for s in m.segments}
        self.doc = {d.id: d for d in r.documents}
        self.chat = {c.id: c for c in r.chats}
        self.url = {u.id: u for u in r.urls}
        self.channel = defaultdict(list)
        for m in r.messages:
            self.channel[m.channel_id].append(m)

    def rows(self, layer: str, *, label: str | None = None, type: str | None = None) -> list[dict]:
        out = []
        for row in self.facts.get(layer, []):
            if label is not None and row.get("label") != label:
                continue
            if type is not None and row.get("type") != type:
                continue
            out.append(row)
        return out

    def node(self, layer: str, node_id: str) -> dict | None:
        return next((r for r in self.facts.get(layer, []) if r.get("kind") == "node" and r["id"] == node_id), None)

    def person(self, eid: str) -> str:
        e = self.records.employees.get(eid)
        return f"{e['name']} ({eid})" if e else f"{eid} (not in directory)"

    def name_of(self, who: str) -> str:
        """Normalised display name of an eid or of a name as written."""
        e = self.records.employees.get(who)
        return (e["name"] if e else who).casefold().strip()


def load_source(product: str, bundle: Path = BUNDLE_DIR, facts_dir: Path = FACTS_DIR) -> Source:
    facts = {layer: read_jsonl(facts_dir / product / f"{layer}.jsonl") for layer in LAYERS}
    return Source(product, load_records(bundle, product), facts, read_jsonl(facts_dir / product / "names.jsonl"))


def dev_products(bundle: Path = BUNDLE_DIR) -> list[str]:
    return split(json.loads((bundle / "manifest.json").read_text())["products"])[0]


# ---------------------------------------------------------------- rendering raw records as text
def fmt_message(src: Source, m) -> str:
    return f"{m.ts} {src.person(m.user_id) if m.user_id in src.records.employees else m.user_id} in #{m.channel_name}: {m.text}"


def message_session_records(src: Source, message_ids) -> list[dict]:
    """Every message of each session (6 hour silence gap) that holds one of the messages."""
    wanted = {i for i in message_ids if i in src.msg}
    out, seen = [], set()
    for ch in sorted({src.msg[i].channel_id for i in wanted}):
        for sess in sessions(src.channel[ch]):
            if wanted & {m.id for m in sess}:
                for m in sess:
                    if m.id not in seen:
                        seen.add(m.id)
                        out.append({"id": m.id, "text": fmt_message(src, m)})
    return out


def meeting_records(src: Source, meeting_id: str) -> list[dict]:
    m = src.meeting.get(meeting_id)
    if m is None:
        return []
    who = "; ".join(src.person(e) for e in m.participants)
    out = [{"id": m.id, "text": f"Meeting {m.id}, document type {m.document_type}, date {m.date}. Participants: {who}. "
                                f"Attendee names listed: {', '.join(m.attendees) or 'none'}."}]
    out += [{"id": s.id, "text": f"{s.speaker or '(no speaker)'}: {s.text}"} for s in m.segments]
    out += [{"id": c.id, "text": f"Meeting chat ({c.date}): {c.text}"} for c in src.records.chats if c.meeting_id == m.id]
    return out


def pr_record(src: Source, p) -> dict:
    reviews = "; ".join(f"{r.login} {r.state} {r.submitted_at}: {r.comment}" for r in p.reviews) or "none"
    author = src.person(p.author) if p.author in src.records.employees else p.author
    return {"id": p.id, "text": f"Pull request {p.link} '{p.title}': {p.summary} Author {author}. state {p.state}, "
                                f"merged {p.merged_raw}, mergeable {p.mergeable_raw}, created {p.created_at}. "
                                f"Reviews: {reviews}"}


def doc_record(src: Source, d, *, feedback: bool = True) -> dict:
    fb = f" Feedback: {d.feedback}" if feedback and d.feedback else ""
    return {"id": d.id, "text": f"Document {d.id}, type {d.type}, author {d.author}, date {d.date}, link {d.link}.{fb}"}


def record_text(src: Source, rid: str) -> dict | None:
    """The raw record behind any record id the facts cite, or None."""
    if rid in src.msg:
        return {"id": rid, "text": fmt_message(src, src.msg[rid])}
    if rid in src.pr:
        return pr_record(src, src.pr[rid])
    if rid in src.seg:
        s = src.seg[rid]
        return {"id": rid, "text": f"Meeting {s.meeting_id} line {s.line_no}, {s.speaker or '(no speaker)'}: {s.text}"}
    if rid in src.meeting:
        return meeting_records(src, rid)[0]
    if rid in src.doc:
        return doc_record(src, src.doc[rid])
    if rid in src.chat:
        return {"id": rid, "text": f"Meeting chat: {src.chat[rid].text}"}
    if rid in src.url:
        u = src.url[rid]
        return {"id": rid, "text": f"URL {u.link}: {' / '.join(u.descriptions)}"}
    return None


def directory(src: Source, texts: list[str], extra_eids=(), names=()) -> list[str]:
    """Directory lines for every eid in the texts, the extra eids and every directory person named like one of `names`."""
    eids = set(extra_eids)
    for t in texts:
        eids.update(EID_RE.findall(t))
    wanted = {n.casefold().strip() for n in names}
    for eid, e in src.records.employees.items():
        if e["name"].casefold().strip() in wanted:
            eids.add(eid)
    lines = []
    for eid in sorted(eids):
        e = src.records.employees.get(eid)
        lines.append(f"{eid} | {e['name']} | {e['role']}" if e else f"{eid} | not in the directory")
    for cid in sorted({c for t in texts for c in CUST_RE.findall(t)}):
        c = src.records.customers.get(cid)
        if c:
            lines.append(f"{cid} | {c['name']} | {c['role']} at {c['company']}")
    return lines


def make_packet(kind: str, key: str, src: Source, *, claim: dict | None = None, task: str | None = None,
                records: list[dict], meta: dict | None = None, names=()) -> dict:
    seen, recs = set(), []
    for r in records:
        if r["id"] not in seen:
            seen.add(r["id"])
            recs.append(r)
    pid = f"{kind}-{sha256_text(src.product + '|' + key)[:10]}"
    texts = [r["text"] for r in recs] + [canon(claim or "") + (task or "")]
    dirlines = directory(src, texts, names=names)
    body = [f"PACKET {pid}  kind: {kind}  product: {src.product}"]
    if claim is not None:
        body.append("CLAIM (an extracted fact to check):")
        body += [f"  {k}: {v}" for k, v in claim.items()]
    else:
        body += ["TASK:", f"  {task}"]
    body.append("RAW RECORDS (each line starts with its record id):")
    body += [f"  [{r['id']}] {r['text']}" for r in recs]
    body.append("DIRECTORY (employee id | name | role):")
    body += [f"  {line}" for line in dirlines] or ["  (none)"]
    packet = {"id": pid, "kind": kind, "product": src.product, "records": recs, "directory": dirlines,
              "meta": meta or {}, "text": "\n".join(body)}
    packet["claim" if claim is not None else "task"] = claim if claim is not None else task
    return packet


# ---------------------------------------------------------------- seeded sampling with frozen quotas
def rng_for(round_n: int, kind: str) -> random.Random:
    return random.Random(f"{BASE_SEED + round_n}:{kind}")


def sample_strata(pool: list[tuple[str, str, object]], n: int, rng: random.Random) -> tuple[list, dict]:
    """Even allocation over strata, capped by stratum size, leftover shares redistributed (water-filling); within a
    stratum a seeded draw. `pool` items are (ident, stratum, payload). A pool of n or fewer is a census (no padding).
    Returns (picked items sorted by ident, {'requested', 'available'})."""
    groups: dict[str, list] = defaultdict(list)
    for item in sorted(pool, key=lambda x: x[0]):
        groups[item[1]].append(item)
    order = sorted(groups)
    rng.shuffle(order)  # who gets the odd extra slot is seeded, not alphabetical
    alloc, left = {s: 0 for s in order}, min(n, len(pool))
    while left > 0:
        active = [s for s in order if alloc[s] < len(groups[s])]
        share, extra = divmod(left, len(active))
        if share == 0:
            for s in active[:extra]:
                alloc[s] += 1
            break
        for s in active:
            add = min(share, len(groups[s]) - alloc[s])
            alloc[s] += add
            left -= add
    picked = []
    for s in sorted(order):
        picked += rng.sample(groups[s], alloc[s]) if alloc[s] < len(groups[s]) else groups[s]
    return sorted(picked, key=lambda x: x[0]), {"requested": n, "available": len(pool)}


# ---------------------------------------------------------------- precision packets
def bug_event_packets(sources: dict[str, Source], n: int, rng: random.Random) -> tuple[list[dict], dict]:
    pool = []
    for p, src in sorted(sources.items()):
        for ev in src.rows("bugs", label="BugEvent"):
            pool.append((f"{p}|{ev['id']}", f"{p}|{ev['props']['type']}", (src, ev)))
    picked, info = sample_strata(pool, n, rng)
    out = []
    for ident, _, (src, ev) in picked:
        ep = ev["props"]
        bug = src.node("bugs", ep["bug"])["props"]
        acted = [e["dst_id"].split(":", 1)[1] for e in src.rows("bugs", type="ACTED_BY") if e["src_id"] == ev["id"]]
        mids = {e["props"]["message_id"] for e in src.rows("bugs", label="BugEvent") if e["props"]["bug"] == ep["bug"]}
        recs = message_session_records(src, mids)
        for link in ep.get("fix_pr_links") or []:
            pr = src.pr_by_link.get(link)
            recs += [pr_record(src, pr)] if pr else []
        claim = {
            "event": f"{ep['type']} event {ev['id']} on message {ep['message_id']}",
            "bug identity": f"'{bug['title']}' (component {bug.get('component')}, channel {bug['channel_name']})",
            "time": ep["ts"],
            "actor": ", ".join(src.person(a) for a in acted) or "none",
            "assignee(s) of this event": ", ".join(src.person(a) for a in ep.get("assignee_eids") or []) or "none",
            "customer(s) of the bug": ", ".join(bug.get("customer_ids") or []) or "none",
            "bug status now": f"{bug.get('status')} (active {bug.get('active')}, resolved {bug.get('resolved')}, "
                              f"reopened {bug.get('reopened')}, no fix needed {bug.get('no_fix_needed')})",
            "bug current assignee(s)": ", ".join(src.person(a) for a in bug.get("current_assignees") or []) or "none",
        }
        out.append(make_packet("bug_event", ident, src, claim=claim, records=recs,
                               meta={"event_id": ev["id"], "bug": ep["bug"]}))
    return out, info


def _source_type(src: Source, rid: str) -> str:
    return ("segment" if rid in src.seg else "message" if rid in src.msg else "document" if rid in src.doc
            else "chat" if rid in src.chat else "other")


def review_packets(sources: dict[str, Source], n: int, rng: random.Random) -> tuple[list[dict], dict]:
    pool = []
    for p, src in sorted(sources.items()):
        for e in src.rows("semantic", type="SUGGESTED"):
            rid = (e["src"] or [""])[0]
            ident = f"{p}|{e['src_id']}|{e['dst_id']}|{e['key']}"
            pool.append((ident, f"{p}|{_source_type(src, rid)}|{e['props'].get('version_ambiguous')}", (src, e)))
    picked, info = sample_strata(pool, n, rng)
    out = []
    for ident, _, (src, e) in picked:
        ep, rid = e["props"], (e["src"] or [""])[0]
        eid = e["src_id"].split(":", 1)[1]
        doc = src.doc.get(e["dst_id"].split(":", 1)[1])
        siblings = [d for d in src.records.documents if doc and d.type == doc.type]
        st = _source_type(src, rid)
        recs = (meeting_records(src, src.seg[rid].meeting_id) if st == "segment"
                else message_session_records(src, [rid]) if st == "message"
                else [doc_record(src, src.doc[rid])] if st == "document"
                else [r for r in [record_text(src, rid)] if r])
        recs += [doc_record(src, d, feedback=False) for d in siblings]
        level = ("document TYPE (several versions exist; the relation is correct if the source is about this type of "
                 f"document: {doc.type if doc else '?'}; candidate versions: {', '.join(d.id for d in siblings)})"
                 if ep.get("version_ambiguous") else f"the single document {doc.id if doc else e['dst_id']} "
                 f"(type {doc.type if doc else '?'})")
        claim = {"relation": f"{src.person(eid)} SUGGESTED a change/question about {level}",
                 "kind": ep.get("kind"), "section": ep.get("section"), "text": ep.get("text"),
                 "source record": f"{rid} ({st})", "judged at": "document type" if ep.get("version_ambiguous") else "document"}
        out.append(make_packet("review_relation", ident, src, claim=claim, records=recs,
                               meta={"source_type": st, "version_ambiguous": bool(ep.get("version_ambiguous"))},
                               names=[src.records.employees[eid]["name"]] if eid in src.records.employees else []))
    return out, info


def name_packets(sources: dict[str, Source], n: int, rng: random.Random) -> tuple[list[dict], dict]:
    pool = [(f"{p}|{r['meeting_id']}|{r['name']}", f"{p}|{r['status']}", (src, r))
            for p, src in sorted(sources.items()) for r in src.names]
    picked, info = sample_strata(pool, n, rng)
    out = []
    for ident, _, (src, r) in picked:
        recs = meeting_records(src, r["meeting_id"])
        shown = r["ids"] if r["status"] == "resolved" else r["candidates"]
        claim = {"speaker label": f"'{r['name']}' in meeting {r['meeting_id']}", "status": r["status"],
                 ("resolved to" if r["status"] == "resolved" else "ambiguous among"):
                     ", ".join(src.person(e) for e in shown) or "nobody"}
        out.append(make_packet("name_resolution", ident, src, claim=claim, records=recs,
                               meta={"status": r["status"]}, names=[r["name"]]))
    return out, info


# ---------------------------------------------------------------- what-if packets (R4: hydrate from the bundle)
def node_title(src: Source, nid: str) -> str:
    for layer in ("bugs", "semantic", "rules", "structural"):
        n = src.node(layer, nid)
        if n and n["props"].get("title"):
            return n["props"]["title"]
    return nid


def item_records(src: Source, nid: str) -> list[dict]:
    """The raw evidence of an endpoint or item: a PR's original record, a bug's event sessions, a feature's cited records."""
    label, rest = nid.split(":", 1)
    if label == "PR":
        p = src.pr_by_link.get(rest)
        return [pr_record(src, p)] if p else []
    if label == "Bug":
        mids = {e["props"]["message_id"] for e in src.rows("bugs", label="BugEvent") if e["props"]["bug"] == nid}
        return message_session_records(src, mids)
    if label == "Feature":
        n = src.node("semantic", nid)
        rids = [c["record_id"] for c in (n["props"].get("citations") or [])] if n else []
        for ev in src.rows("semantic", label="FeatureEvent"):
            if ev["props"].get("feature") == nid:
                rids += [c["record_id"] for c in ev["props"].get("citations") or []]
        return [r for r in (record_text(src, x) for x in dict.fromkeys(rids)) if r]
    return []


def _hydrate(src: Source, rids) -> tuple[list[dict], int]:
    got = [record_text(src, r) for r in dict.fromkeys(rids)]
    return [g for g in got if g], sum(1 for g in got if g is None)


def whatif_packets(sources: dict[str, Source], whatif_fn: Callable[[str], dict], quotas: dict, rng_of: Callable
                   ) -> tuple[dict[str, list[dict]], dict]:
    pools: dict[str, list] = {"whatif_unowned": [], "whatif_edge": [], "whatif_path": []}
    for p, src in sorted(sources.items()):
        wi = whatif_fn(p)  # identifiers and pools only; evidence below comes from the bundle
        pools["whatif_unowned"] += [(f"{p}|{x['person']}|{x['item']}", f"{p}|{x['label']}", (src, x))
                                    for x in wi["newly_unowned"]]
        pools["whatif_edge"] += [(f"{p}|{x['dependent']}|{x['prerequisite']}", f"{p}|{x['rule']}", (src, x))
                                 for x in wi["dependency_edges"]]
        pools["whatif_path"] += [(f"{p}|{x['person']}|{x['affected']}", f"{p}|{len(x['path'])}", (src, x))
                                 for x in wi["affected_paths"]]
    out, info = {}, {}
    for kind, pool in pools.items():
        picked, info[kind] = sample_strata(pool, quotas[kind], rng_of(kind))
        out[kind] = [globals()[f"_{kind}_packet"](ident, src, x) for ident, _, (src, x) in picked]
    return out, info


def _cap_later(src: Source, ids: list[str], names: list[str]) -> list[str]:
    """At most MAX_LATER later messages: the ones that name the item (link or title) first, then the earliest others;
    kept in record order."""
    if len(ids) <= MAX_LATER:
        return ids
    keys = [k.split(":", 1)[-1].casefold() for k in names if k]
    named = [i for i in ids if i in src.msg and any(k in src.msg[i].text.casefold() for k in keys)]
    rest = [i for i in ids if i not in set(named)]
    keep = set((named + rest)[:MAX_LATER])
    return [i for i in ids if i in keep]


def _whatif_unowned_packet(ident: str, src: Source, x: dict) -> dict:
    a = x["assignment"]
    recs = item_records(src, x["item"])
    later_ids = _cap_later(src, [m["id"] for m in x["later_context"]], [x["item"], x["title"]])
    later, miss = _hydrate(src, later_ids)
    got, miss2 = _hydrate(src, [i for i in [a.get("message_id"), *(a.get("records") or [])] if i])
    claim = {"statement": f"If {src.person(x['person'])} leaves, the {x['label']} '{x['title']}' ({x['item']}) is newly "
                          "unowned: it is open work and this person is its only remaining owner",
             "owner basis": ("latest ASSIGNED event of the bug (assignees of that event)" if x["label"] == "Bug"
                             else "the PR author"),
             "assignment evidence": f"{a['kind']} {a.get('event') or ''} {a.get('ts') or ''}".strip()}
    cut = max(0, len(x["later_context"]) - MAX_LATER)
    return make_packet("whatif_unowned", ident, src, claim=claim, records=got + recs + later,
                       meta={"item": x["item"], "person": x["person"], "later_context_cut": cut,
                             "unhydrated": miss + miss2}, names=[])


def _edge_records(src: Source, dependent: str, prereq: str, cites) -> tuple[list[dict], int]:
    got, miss = _hydrate(src, cites)
    return got + item_records(src, dependent) + item_records(src, prereq), miss


def _whatif_edge_packet(ident: str, src: Source, x: dict) -> dict:
    recs, miss = _edge_records(src, x["dependent"], x["prerequisite"], x["citations"])
    claim = {"statement": f"'{node_title(src, x['dependent'])}' ({x['dependent']}) depends on "
                          f"'{node_title(src, x['prerequisite'])}' ({x['prerequisite']}) "
                          f"(the prerequisite must be done first)",
             "rule": x["rule"], "cited records": ", ".join(x["citations"]) or "none"}
    return make_packet("whatif_edge", ident, src, claim=claim, records=recs,
                       meta={"rule": x["rule"], "unhydrated": miss})


def _whatif_path_packet(ident: str, src: Source, x: dict) -> dict:
    recs, miss = [], 0
    for e in x["edges"]:
        r, m = _edge_records(src, e["dependent"], e["prerequisite"], e["citations"])
        recs, miss = recs + r, miss + m
    steps = [f"{e['dependent']} depends on {e['prerequisite']} (rule {e['rule']}, cited {', '.join(e['citations']) or 'none'})"
             for e in x["edges"]]
    claim = {"statement": f"If {src.person(x['person'])} leaves, '{node_title(src, x['affected'])}' ({x['affected']}) is "
                          "affected: it is open work or a pending feature and it depends, through the path below, "
                          "on an item that becomes newly unowned",
             "path (affected item first, newly unowned item last)": " -> ".join(x["path"]),
             "dependency steps": " | ".join(steps)}
    return make_packet("whatif_path", ident, src, claim=claim, records=recs,
                       meta={"path_length": len(x["path"]), "unhydrated": miss})


# ---------------------------------------------------------------- recall packets (no extraction shown)
def _spread_products(cands: dict[str, list], n: int, rng: random.Random) -> list[tuple[str, object]]:
    """Seeded choice of n items, one per product first (round robin over shuffled products and shuffled candidates)."""
    prods = sorted(p for p in cands if cands[p])
    rng.shuffle(prods)
    shuffled = {p: rng.sample(sorted(cands[p], key=str), len(cands[p])) for p in prods}
    chosen = []
    for round_i in range(max((len(v) for v in shuffled.values()), default=0)):
        for p in prods:
            if len(chosen) < n and round_i < len(shuffled[p]):
                chosen.append((p, shuffled[p][round_i]))
    return chosen


BUG_TASK = ("List every bug event in channel {name} (product {product}). Use only the records below. Event types: "
            "REPORTED (a bug is introduced or reported), ASSIGNED (someone is asked to take it or takes it), RESOLVED "
            "(reported fixed or closed), DISMISSED (looked at and judged not a bug or no fix needed), REOPENED (comes "
            "back after resolution). One item per bug per event; a message can hold several events. Give the type, the "
            "actor (who performed the event), the time (the message timestamp) and the message id.")
REVIEW_TASK = ("List every reviewer of document {doc} (type {type}, product {product}): each person who gives a "
               "suggestion, a requested change or a review question about this document, or about a document of this "
               "type when the source names no version. Sources are the document feedback, the meetings of this "
               "document type and the Slack sessions that share the document link. One item per person per source "
               "record. Give the reviewer as the name written in the record (or the eid when the record uses one) and "
               "the id of the source record (a meeting line id like meeting#12, a message id, or the document id).")


def recall_bug_packets(sources: dict[str, Source], n: int, rng: random.Random) -> tuple[list[dict], dict]:
    cands = {p: [c for c in sorted(s.channel) if s.channel[c][0].bug_channel_owner is not None]
             for p, s in sources.items()}
    out = []
    for p, ch in _spread_products(cands, n, rng):
        src = sources[p]
        msgs = sorted(src.channel[ch], key=lambda m: (m.ts, m.id))
        recs = [{"id": m.id, "text": fmt_message(src, m)} for m in msgs]
        task = BUG_TASK.format(name=msgs[0].channel_name, product=p)
        out.append(make_packet("recall_bug", f"{p}|{ch}", src, task=task, records=recs,
                               meta={"channel_id": ch, "channel_name": msgs[0].channel_name, "messages": len(msgs)}))
    return out, {"requested": n, "available": sum(len(v) for v in cands.values())}


def recall_reviewer_packets(sources: dict[str, Source], n: int, rng: random.Random) -> tuple[list[dict], dict]:
    def has_sources(src, d):
        return bool(d.feedback or any(m.document_type == d.type for m in src.records.meetings)
                    or any(d.link in m.text for m in src.records.messages))
    cands = {p: [d.id for d in s.records.documents if has_sources(s, d)] for p, s in sources.items()}
    out = []
    for p, did in _spread_products(cands, n, rng):
        src, d = sources[p], sources[p].doc[did]
        recs = [doc_record(src, d)]
        for m in src.records.meetings:
            if m.document_type == d.type:
                recs += meeting_records(src, m.id)
        linked = [m.id for m in src.records.messages if d.link in m.text]
        recs += message_session_records(src, linked)
        task = REVIEW_TASK.format(doc=d.id, type=d.type, product=p)
        names = [s.speaker for m in src.records.meetings if m.document_type == d.type for s in m.segments if s.speaker]
        out.append(make_packet("recall_reviewer", f"{p}|{did}", src, task=task, records=recs,
                               meta={"document_id": did, "document_type": d.type}, names=names))
    return out, {"requested": n, "available": sum(len(v) for v in cands.values())}


# ---------------------------------------------------------------- prompts (versioned and hashed)
COMMON_PRECISION = """You are auditing extracted facts against raw company records. Work only from the records and the
directory inside each packet. Do not use outside knowledge and do not guess. For each packet decide:
  correct    every part of the claim is supported by the records
  incorrect  some part is contradicted by the records or has no support in them
  uncertain  the records cannot decide it either way (use rarely)
Output only a JSON array with one object per packet:
  {"packet_id": "<id>", "verdict": "correct|incorrect|uncertain", "reason": "<one or two sentences naming record ids>"
   [, "fields": {"<field>": "correct|incorrect|na"}]}
Do not add text outside the JSON."""

RUBRICS = {
    "bug_event": """Kind bug_event. The claim is one bug event and the bug it belongs to. It is correct only if ALL of
these hold (joint correctness): identity (the title and component are the bug the records describe at that event),
customer (the customer ids belong to that bug), assignee (the event's assignees and the bug's current assignees match
the records), status (the stated status follows from the records, including later messages), time (the event time is
the timestamp of the cited message). Report each as "fields": {"identity","customer","assignee","status","time"} with
correct, incorrect or na. The verdict is correct only if no field is incorrect.""",
    "review_relation": """Kind review_relation. The claim says a person suggested a change, or asked a review question,
about a document. Correct if the person (resolved through the directory) did say it in the source record and it is about
that document. When the claim is judged at document-type level (several versions exist), it is correct if the source is
about a document of that type; do not require a specific version. Incorrect if the speaker is another person, the text
is not a suggestion or review question, or the source is about another document type.""",
    "name_resolution": """Kind name_resolution. The claim maps a speaker name in one meeting to employee ids. For a
resolved claim: correct if the records (participants, attendee list, chat, what is said) support that this speaker is
that employee and no other directory person of the same name is equally likely. For an ambiguous claim: correct if the
records genuinely do not decide between the listed candidates and the candidates are the directory people who fit the
name and the meeting; incorrect if the records do decide it, or a plausible candidate is missing.""",
    "whatif_unowned": """Kind whatif_unowned. Open work is a bug that has not been resolved, dismissed or closed, or a
pull request that is open and not merged. The owner of a bug is the assignee of its latest ASSIGNED event; the owner of
a PR is its author. Correct only if the item is open work, the named person is its only owner, and no later record hands
it over to someone else or closes it.""",
    "whatif_edge": """Kind whatif_edge. A dependency means the dependent item cannot be finished before the
prerequisite. Rules: IMPLEMENTED_BY (a feature depends on the PR that implements it), ABOUT with kind development (a
feature depends on a development PR about it), FIXES (a bug depends on the PR that fixes it). Correct only if the
records show that relation between exactly these two items (the cited records name both, or the PR record and the other
item's records make the link clear).""",
    "whatif_path": """Kind whatif_path. The claim gives a chain of dependency steps from an affected item to an item that
becomes newly unowned. Correct only if every step is supported by the records (see the whatif_edge rule), the chain
joins the steps in order, and the affected item is open work or a pending feature (a bug not yet resolved, an open
unmerged PR, or a feature discussed but not implemented or with an open development PR).""",
}

COMMON_RECALL = """You are listing every item of one kind in the raw company records of a packet. Work only from the
records and the directory in the packet. Be complete and do not invent items. Output only JSON:
  {"packet_id": "<id>", "items": [ ... ]}
Do not add text outside the JSON."""

RECALL_RULES = {
    "recall_bug": """Each item: {"type": "REPORTED|ASSIGNED|RESOLVED|DISMISSED|REOPENED", "actor": "<eid or name of who
performed it>", "time": "<message timestamp>", "message_id": "<id>", "bug": "<bug title>"}. List one item per bug per
event. A bug is reported once when it is introduced; it is assigned when someone is asked to take it or takes it; resolved
when reported fixed or closed; dismissed when judged not a bug; reopened when it returns after resolution.""",
    "recall_reviewer": """Each item: {"reviewer": "<name as written in the record, or the eid>", "source": "<source record
id>"}. A reviewer is a person who gives a suggestion, a requested change or a review question about the document (or its
document type when no version is named). One item per person per source record.""",
}

THIRD_PRECISION = """You are the third and final auditor. Two auditors disagreed, or one gave no usable answer. Read the
packet and both reasoned verdicts, check the records yourself, and decide. Your verdict must be correct or incorrect
(uncertain only if the records truly cannot decide). Use the rubric below. Output only a JSON array with one object per
packet in the same format as the first auditors."""

THIRD_RECALL = """You are reconciling two auditors' lists. Read the packet and both lists, check the records yourself, and
output the single complete list: keep items that the records support, drop items they do not, add items both missed.
Output only {"packet_id": "<id>", "items": [...]} in the same item format. You have not been shown any extraction."""


def prompt_hashes() -> dict[str, str]:
    texts = {"common_precision": COMMON_PRECISION, "common_recall": COMMON_RECALL, "third_precision": THIRD_PRECISION,
             "third_recall": THIRD_RECALL, **{f"rubric:{k}": v for k, v in RUBRICS.items()},
             **{f"recall_rules:{k}": v for k, v in RECALL_RULES.items()}}
    return {k: sha256_text(v) for k, v in sorted(texts.items())} | {"version": PROMPT_VERSION}


# ---------------------------------------------------------------- build a round
def build_packets(sources: dict[str, Source], round_n: int, whatif_fn: Callable[[str], dict],
                  quotas: dict | None = None) -> tuple[list[dict], dict]:
    q = quotas or (PRECISION_QUOTAS | RECALL_QUOTAS)
    packets, sizes = [], {}
    builders = {"bug_event": bug_event_packets, "review_relation": review_packets, "name_resolution": name_packets,
                "recall_bug": recall_bug_packets, "recall_reviewer": recall_reviewer_packets}
    for kind, fn in builders.items():
        got, sizes[kind] = fn(sources, q[kind], rng_for(round_n, kind))
        packets += got
    wi, info = whatif_packets(sources, whatif_fn, q, lambda kind: rng_for(round_n, kind))
    for kind, got in wi.items():
        packets += got
    sizes.update(info)
    for kind in KINDS:
        sizes[kind]["actual"] = sum(1 for p in packets if p["kind"] == kind)
    return packets, sizes


def round_dir(round_n: int, root: Path = AUDIT_DIR) -> Path:
    return root / f"round-{round_n}"


def write_round(packets: list[dict], sizes: dict, round_n: int, bundle_sha: str | None, root: Path = AUDIT_DIR) -> dict:
    d = round_dir(round_n, root)
    pdir = d / "packets"
    pdir.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in packets:
        body = json.dumps(p, ensure_ascii=False, indent=1, sort_keys=True)
        (pdir / f"{p['id']}.json").write_text(body, encoding="utf-8")
        entries.append({"id": p["id"], "kind": p["kind"], "product": p["product"], "sha256": sha256_text(body)})
    entries.sort(key=lambda e: e["id"])
    manifest = {"round": round_n, "seed": BASE_SEED + round_n, "bundle_sha256": bundle_sha,
                "quotas": PRECISION_QUOTAS | RECALL_QUOTAS, "sizes": sizes,
                "shortages": {k: v for k, v in sizes.items() if v["available"] < v["requested"]},
                "prompt_hashes": prompt_hashes(), "packets": entries,
                "sample_hash": sha256_text("\n".join(f"{e['id']}:{e['sha256']}" for e in entries))}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def load_packets(round_n: int, root: Path = AUDIT_DIR) -> dict[str, dict]:
    pdir = round_dir(round_n, root) / "packets"
    return {p["id"]: p for p in (json.loads(f.read_text(encoding="utf-8")) for f in sorted(pdir.glob("*.json")))}


def is_recall(kind: str) -> bool:
    return kind in RECALL_QUOTAS


def _chunks(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _batch_plan(packets: dict[str, dict], size: int) -> dict[str, list[str]]:
    """Homogeneous batches: precision kinds in chunks of `size`, recall packets one per batch (they are large)."""
    plan = {}
    for kind in KINDS:
        ids = sorted(i for i, p in packets.items() if p["kind"] == kind)
        for n, chunk in enumerate(_chunks(ids, 1 if is_recall(kind) else size), 1):
            plan[f"{kind}-{n:02d}"] = chunk
    return plan


def batch_header(kind: str, third: bool = False) -> str:
    if is_recall(kind):
        return (THIRD_RECALL if third else COMMON_RECALL) + "\n\n" + RECALL_RULES[kind]
    return (THIRD_PRECISION if third else COMMON_PRECISION) + "\n\n" + RUBRICS[kind]


def write_batches(round_n: int, size: int = 10, root: Path = AUDIT_DIR) -> list[str]:
    packets = load_packets(round_n, root)
    bdir = round_dir(round_n, root) / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    plan = _batch_plan(packets, size)
    for name, ids in plan.items():
        kind = packets[ids[0]]["kind"]
        parts = [batch_header(kind)] + [f"\n==== {packets[i]['id']} ====\n{packets[i]['text']}" for i in ids]
        (bdir / f"{name}.txt").write_text("\n".join(parts) + "\n", encoding="utf-8")
    (bdir / "index.json").write_text(json.dumps(plan, indent=1, sort_keys=True), encoding="utf-8")
    return sorted(plan)


# ---------------------------------------------------------------- verdicts
def validate_verdict(obj, packet: dict, third: bool = False) -> str | None:
    """None when the object is a valid answer for the packet, else the reason it is not."""
    if not isinstance(obj, dict) or obj.get("packet_id") != packet["id"]:
        return "not an object for this packet"
    if is_recall(packet["kind"]):
        items = obj.get("items")
        if not isinstance(items, list):
            return "items is not a list"
        need = (("type", "actor", "time", "message_id") if packet["kind"] == "recall_bug" else ("reviewer", "source"))
        for it in items:
            if not isinstance(it, dict) or any(not isinstance(it.get(k), str) or not it[k].strip() for k in need):
                return "item missing a required string field"
            if packet["kind"] == "recall_bug" and it["type"].upper() not in EVENT_TYPES:
                return "unknown event type"
        return None
    v = obj.get("verdict")
    if v not in VERDICTS:
        return "bad verdict"
    if not isinstance(obj.get("reason"), str) or not obj["reason"].strip():
        return "no reason"
    f = obj.get("fields")
    if f is not None:
        if not isinstance(f, dict) or any(v2 not in ("correct", "incorrect", "na") for v2 in f.values()):
            return "bad fields"
        if packet["kind"] == "bug_event" and v == "correct" and "incorrect" in f.values():
            return "correct verdict with an incorrect field"
    return None


def load_verdicts(vdir: Path, packets: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    """Valid answers by packet id from every json file of the directory (a list, {'verdicts': [...]} or one object).
    The first valid answer per packet wins; the rest are counted as duplicates."""
    good, stats = {}, Counter()
    for f in sorted(vdir.glob("*.json")) if vdir.exists() else []:
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            stats["unreadable_files"] += 1
            continue
        for obj in (raw if isinstance(raw, list) else raw.get("verdicts", [raw]) if isinstance(raw, dict) else []):
            pid = obj.get("packet_id") if isinstance(obj, dict) else None
            if pid not in packets:
                stats["unknown_packet"] += 1
            elif validate_verdict(obj, packets[pid]) is not None:
                stats["malformed"] += 1
            elif pid in good:
                stats["duplicate"] += 1
            else:
                good[pid] = obj
    return good, dict(stats)


def resolve(a: str | None, b: str | None, third: str | None) -> dict:
    """The R5 truth table. a, b: the two auditors' verdicts (None: missing or malformed). third: the third pass."""
    if a == b == "correct":
        return {"final": "correct", "route": "agree", "needs_third": False}
    if a == b == "incorrect":
        return {"final": "incorrect", "route": "agree", "needs_third": False}
    if third in ("correct", "incorrect"):
        return {"final": third, "route": "third", "needs_third": False}
    if third is None:
        return {"final": None, "route": "pending", "needs_third": True}
    return {"final": "unresolved", "route": "third_uncertain", "needs_third": False}


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


# ---------------------------------------------------------------- recall: truth keys and extraction keys
def truth_keys(kind: str, items: list[dict], src: Source) -> Counter:
    if kind == "recall_bug":
        return Counter((i["type"].upper().strip(), i["message_id"].strip()) for i in items)
    return Counter({(src.name_of(i["reviewer"]), i["source"].strip()) for i in items})


def extracted_keys(packet: dict, src: Source) -> Counter:
    if packet["kind"] == "recall_bug":
        ch = packet["meta"]["channel_id"]
        return Counter((ev["props"]["type"], ev["props"]["message_id"]) for ev in src.rows("bugs", label="BugEvent")
                       if ev["props"]["channel_id"] == ch)
    dst = f"Document:{packet['meta']['document_id']}"
    return Counter({(src.name_of(e["src_id"].split(":", 1)[1]), rid)
                    for e in src.rows("semantic", type="SUGGESTED") if e["dst_id"] == dst for rid in e["src"]})


def compare_recall(truth: Counter, extracted: Counter) -> dict:
    matched = sum(min(n, extracted[k]) for k, n in truth.items())
    missed = sorted(f"{k[0]}|{k[1]}" for k, n in truth.items() for _ in range(n - min(n, extracted[k])))
    return {"truth": sum(truth.values()), "matched": matched, "missed": missed,
            "extra_in_extraction": sum(extracted.values()) - sum(min(n, extracted[k]) for k, n in truth.items())}


# ---------------------------------------------------------------- collect
def _third_stage(round_n: int, root: Path):
    packets = load_packets(round_n, root)
    rd = round_dir(round_n, root)
    v = {who: load_verdicts(rd / "verdicts" / who, packets) for who in ("codex", "claude", "third")}
    return packets, v


def pending_third(round_n: int, root: Path = AUDIT_DIR) -> list[str]:
    """Packet ids that need the third pass and have no usable third answer yet."""
    packets, v = _third_stage(round_n, root)
    out = []
    for pid, p in sorted(packets.items()):
        if is_recall(p["kind"]):
            if pid not in v["third"][0]:
                out.append(pid)
        else:
            g = lambda who: v[who][0].get(pid, {}).get("verdict")  # noqa: E731
            if resolve(g("codex"), g("claude"), g("third"))["needs_third"]:
                out.append(pid)
    return out


def _verdict_text(obj: dict | None, who: str, recall: bool) -> str:
    if obj is None:
        return f"{who}: no usable answer (missing or malformed after one retry)"
    if recall:
        return f"{who} list: {json.dumps(obj['items'], ensure_ascii=False)}"
    return f"{who}: {obj['verdict']}. Reason: {obj['reason']}" + (f" Fields: {obj['fields']}" if obj.get("fields") else "")


def write_third(round_n: int, size: int = 10, root: Path = AUDIT_DIR) -> list[str]:
    packets, v = _third_stage(round_n, root)
    need = pending_third(round_n, root)
    tdir = round_dir(round_n, root) / "third_batches"
    tdir.mkdir(parents=True, exist_ok=True)
    for f in tdir.glob("*"):
        f.unlink()
    names = []
    for kind in KINDS:
        ids = [i for i in need if packets[i]["kind"] == kind]
        for n, chunk in enumerate(_chunks(ids, 1 if is_recall(kind) else size), 1):
            parts = [batch_header(kind, third=True)]
            for pid in chunk:
                rec = is_recall(kind)
                parts.append(f"\n==== {pid} ====\n{packets[pid]['text']}\nFIRST AUDITORS:\n"
                             f"{_verdict_text(v['codex'][0].get(pid), 'auditor 1', rec)}\n"
                             f"{_verdict_text(v['claude'][0].get(pid), 'auditor 2', rec)}")
            name = f"third-{kind}-{n:02d}"
            (tdir / f"{name}.txt").write_text("\n".join(parts) + "\n", encoding="utf-8")
            names.append(name)
    return names


def collect(round_n: int, sources: dict[str, Source], root: Path = AUDIT_DIR) -> dict:
    rd = round_dir(round_n, root)
    manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    packets, v = _third_stage(round_n, root)
    cats: dict[str, dict] = {}
    recall: dict[str, dict] = {}
    pairs: dict[str, list] = defaultdict(list)
    for pid, p in sorted(packets.items()):
        kind = p["kind"]
        if is_recall(kind):
            r = recall.setdefault(kind, {"packets": 0, "truth": 0, "matched": 0, "pending": 0, "lists_equal": 0, "detail": []})
            r["packets"] += 1
            a, b, t = (v[w][0].get(pid) for w in ("codex", "claude", "third"))
            if t is None:
                r["pending"] += 1
                continue
            src = sources[p["product"]]
            cmp = compare_recall(truth_keys(kind, t["items"], src), extracted_keys(p, src))
            r["truth"] += cmp["truth"]
            r["matched"] += cmp["matched"]
            same = bool(a and b and truth_keys(kind, a["items"], src) == truth_keys(kind, b["items"], src))
            r["lists_equal"] += same
            r["detail"].append({"packet_id": pid, "truth": cmp["truth"], "matched": cmp["matched"],
                                "missed": cmp["missed"], "extra_in_extraction": cmp["extra_in_extraction"],
                                "auditor_lists_equal": same})
            continue
        c = cats.setdefault(kind, {"n": 0, "correct": 0, "incorrect": 0, "unresolved": 0, "pending": 0,
                                   "double_positive": 0, "agree": 0, "to_third": 0, "third_decided": 0})
        g = lambda who: v[who][0].get(pid, {}).get("verdict")  # noqa: E731
        a, b, t = g("codex"), g("claude"), g("third")
        res = resolve(a, b, t)
        c["n"] += 1
        c["double_positive"] += a == b == "correct"
        c["agree"] += a is not None and a == b
        c["to_third"] += res["route"] != "agree"
        c["third_decided"] += res["route"] == "third"
        c[res["final"] if res["final"] in ("correct", "incorrect", "unresolved") else "pending"] += 1
        pairs[kind].append((a or "missing", b or "missing"))
    for kind, c in cats.items():
        c["precision"] = round(c["correct"] / c["n"], 4) if c["n"] else None
        c["double_positive_rate"] = round(c["double_positive"] / c["n"], 4) if c["n"] else None
        c["agreement_before_adjudication"] = round(c["agree"] / c["n"], 4) if c["n"] else None
        c["kappa"] = cohen_kappa(pairs[kind])
    for kind, r in recall.items():
        r["recall"] = round(r["matched"] / r["truth"], 4) if r["truth"] else None
    pend = len(pending_third(round_n, root))
    rdirs = {w: sha256_text("".join(sha256_text(f.read_text(encoding="utf-8")) for f in sorted((rd / "verdicts" / w).glob("*.json"))))
             if (rd / "verdicts" / w).exists() else None for w in ("codex", "claude", "third")}
    aud = rd / "auditors.json"
    summary = {"round": round_n, "complete": pend == 0, "pending_third": pend, "sample_hash": manifest["sample_hash"],
               "prompt_hashes": manifest["prompt_hashes"], "verdict_hashes": rdirs,
               "verdict_file_stats": {w: v[w][1] for w in v}, "sizes": manifest["sizes"], "shortages": manifest["shortages"],
               "precision": cats, "recall": recall,
               "auditors": json.loads(aud.read_text(encoding="utf-8")) if aud.exists() else None}
    (rd / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    (rd / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def render_report(s: dict) -> str:
    L = [f"# Dev acceptance audit, round {s['round']}", "",
         "Model-based dev acceptance. Two model auditors, a third pass for disagreements. Agreement between models "
         "does not establish equivalence to human labels. No source text appears in this report.", "",
         f"- complete: {s['complete']} (third passes still pending: {s['pending_third']})",
         f"- sample hash: {s['sample_hash']}", f"- prompt version: {s['prompt_hashes']['version']}",
         f"- verdict hashes: {s['verdict_hashes']}", ""]
    if s["shortages"]:
        L.append("Pools smaller than the quota (census, no padding): " + ", ".join(
            f"{k} {v['available']}/{v['requested']}" for k, v in sorted(s["shortages"].items())) + "\n")
    L += ["## Precision (adjudicated)", "", "| category | correct/n | precision | double-positive | agreement | kappa | to third | "
          "unresolved |", "|---|---|---|---|---|---|---|---|"]
    for k, c in sorted(s["precision"].items()):
        L.append(f"| {k} | {c['correct']}/{c['n']} | {c['precision']} | {c['double_positive']}/{c['n']} | "
                 f"{c['agree']}/{c['n']} | {c['kappa']} | {c['to_third']} | {c['unresolved']} |")
    L += ["", "## Recall", "", "| kind | matched/truth | recall | packets | auditor lists equal |", "|---|---|---|---|---|"]
    for k, r in sorted(s["recall"].items()):
        L.append(f"| {k} | {r['matched']}/{r['truth']} | {r['recall']} | {r['packets']} | {r['lists_equal']} |")
    L += ["", "## Prompt hashes", ""] + [f"- {k}: {h}" for k, h in s["prompt_hashes"].items() if k != "version"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- gates
def gates(summary: dict) -> list[tuple[str, bool, str]]:
    out = [("complete", summary["complete"], f"{summary['pending_third']} third passes pending")]
    for kind in PRECISION_QUOTAS:
        c = summary["precision"].get(kind)
        ok = bool(c and c["n"] and c["precision"] is not None and c["precision"] >= GATE_PRECISION and not c["pending"])
        out.append((f"precision:{kind}", ok, f"{c['correct']}/{c['n']}" if c else "no packets"))
    for kind in RECALL_QUOTAS:
        r = summary["recall"].get(kind)
        ok = bool(r and r["truth"] and not r["pending"] and r["recall"] >= GATE_RECALL)
        out.append((f"recall:{kind}", ok, f"{r['matched']}/{r['truth']}" if r else "no packets"))
    return out


# ---------------------------------------------------------------- live what-if pools and the CLI
def live_whatif(product: str, server: str = "6.0") -> dict:
    """Pools from the audit hook over the live graph (identifiers and pools only; census sized draws)."""
    from inherit import whatif
    from inherit.whatif_check import audit_samples, check, work_owners
    from inherit.whatif_schema import export_snapshot

    snap = export_snapshot(whatif.whatif_connection(product, server), product)
    results = {p: check(snap, p) for p in work_owners(snap)}
    big = 10 ** 6
    return audit_samples(snap, results, seed=BASE_SEED, n_unowned=big, n_edges=big, n_paths=big)


def main(argv: list[str] | None = None, *, root: Path = AUDIT_DIR, whatif_fn: Callable | None = None,
         sources: dict[str, Source] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval.audit_extraction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("packets", "batches", "collect", "third", "gates"):
        sp = sub.add_parser(name)
        sp.add_argument("--round", type=int, required=True)
        if name in ("batches", "third"):
            sp.add_argument("--size", type=int, default=10)
    args = ap.parse_args(argv)
    n = args.round
    if args.cmd == "packets":
        srcs = sources or {p: load_source(p) for p in dev_products()}
        sha = json.loads((BUNDLE_DIR / "manifest.json").read_text())["sha256"] if (BUNDLE_DIR / "manifest.json").exists() else None
        packets, sizes = build_packets(srcs, n, whatif_fn or live_whatif)
        m = write_round(packets, sizes, n, sha, root)
        print(f"round {n}: {len(packets)} packets, sample hash {m['sample_hash']}")
        for k in KINDS:
            print(f"  {k}: {sizes[k]['actual']} (requested {sizes[k]['requested']}, pool {sizes[k]['available']})")
    elif args.cmd == "batches":
        names = write_batches(n, args.size, root)
        print(f"{len(names)} batches in {round_dir(n, root) / 'batches'}")
    elif args.cmd == "third":
        names = write_third(n, args.size, root)
        print(f"{len(names)} third-pass batches ({len(pending_third(n, root))} packets)")
    elif args.cmd == "collect":
        packets = load_packets(n, root)
        srcs = sources or {p: load_source(p) for p in sorted({q["product"] for q in packets.values()})}
        s = collect(n, srcs, root)
        print(f"complete: {s['complete']}, pending third passes: {s['pending_third']}")
        print((round_dir(n, root) / "report.md").read_text(encoding="utf-8"))
    else:
        sf = round_dir(n, root) / "summary.json"
        if not sf.exists():
            print("no summary: run collect first")
            return 1
        rows = gates(json.loads(sf.read_text(encoding="utf-8")))
        for g, ok, detail in rows:
            print(f"{'PASS' if ok else 'FAIL'} {g} {detail}")
        return 0 if all(ok for _, ok, _ in rows) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
