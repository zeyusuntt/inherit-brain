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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from inherit.config import ROOT
from inherit.data.records import ProductRecords, load_records, parse_mentions
from eval import audit_prompts_v1, audit_prompts_v2, audit_prompts_v3
from inherit.data.split import SEED as SPLIT_SEED
from inherit.data.split import split
from inherit.extract.sessions import sessions

BASE_SEED = SPLIT_SEED  # 20261015; round N samples with BASE_SEED + N
AUDIT_DIR = ROOT / "runs" / "audit"
BUNDLE_DIR = ROOT / "data" / "bundle"
FACTS_DIR = ROOT / "data" / "facts"
PROMPT_VERSION = "audit-prompts/v4"  # v1 to v3 are frozen in audit_prompts_v1.py .. _v3.py

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
BATCH_MAX_PACKETS, BATCH_MAX_CHARS = 10, 120_000  # a batch is scheduling only; one oversized packet goes alone
AUDITOR_KEYS = ("model", "version", "settings", "prompt_hash")
PR_LINK_RE = re.compile(r"https?://github\.com/[^\s)>\]]+/pull/\d+")
WINDOW = timedelta(days=3)
NOT_EXTRACTED = "not extracted"
NOT_APPLICABLE = "not applicable for REPORTED"
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
        m = src.meeting[s.meeting_id]
        i = m.segments.index(s)
        around = " | ".join(f"line {x.line_no}, {x.speaker or '(no speaker)'}: {x.text}" for x in m.segments[max(0, i - 3):i + 4])
        return {"id": rid, "text": f"Meeting {m.id}, document type {m.document_type}, date {m.date}. Cited line "
                                   f"{s.line_no}, {s.speaker or '(no speaker)'}: {s.text} Surrounding lines: {around}"}
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


def product_aliases(src: Source) -> set[str]:
    """Other names of the product, from the bundle: the last token of a channel name that is not the product's own name."""
    names = {m.channel_name for m in src.records.messages}
    return {n.rsplit("-", 1)[-1].casefold() for n in names if "-" in n} - {src.product.casefold()}


def product_identity_records(src: Source) -> list[dict]:
    """Neutral provenance from the bundle: the channels of the product file (ids and names) and the first two messages
    (creation or announcement notices) of every channel whose name ends in a different name. Nothing says what those
    names mean; the auditor decides from the raw text."""
    chans = {}
    for m in src.records.messages:
        chans.setdefault(m.channel_id, m.channel_name)
    aliases = product_aliases(src)
    head = (f"Product records (provenance). Product file: {src.product}. Channels in the file (id: name): "
            + "; ".join(f"{c}: {n}" for c, n in sorted(chans.items())) + ". Document ids: "
            + ", ".join(d.id for d in src.records.documents) + ".")
    out = [{"id": f"product:{src.product}", "text": head}]
    for c, n in sorted(chans.items()):
        if n.rsplit("-", 1)[-1].casefold() in aliases:
            first = sorted(src.channel[c], key=lambda m: (m.ts, m.id))[:2]
            out.append({"id": f"channel:{c}", "text": f"Product records (provenance): channel {c}, named {n}; its first "
                                                     "messages are listed as records."})
            out += [{"id": m.id, "text": fmt_message(src, m)} for m in first]
    return out


def make_packet(kind: str, key: str, src: Source, *, claim: dict | None = None, task: str | None = None,
                records: list[dict], meta: dict | None = None, names=(), identity: str = "if_alias") -> dict:
    """identity: 'always' adds the product identity records; 'if_alias' adds them when a product alias occurs in the
    records."""
    al = product_aliases(src)
    if identity == "always" or (al and any(a in r["text"].casefold() for r in records for a in al)):
        records = list(records) + product_identity_records(src)
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
def bug_context_records(src: Source, bug_id: str, since_ts: str) -> list[dict]:
    """Evidence for a bug's events, independent of what was extracted: the sessions of its event messages; every later
    session of the bug's channel that mentions its title, a customer's company or any PR link; and the raw record of
    every PR linked anywhere in the bug's history."""
    bug = src.node("bugs", bug_id)["props"]
    events = [e["props"] for e in src.rows("bugs", label="BugEvent") if e["props"]["bug"] == bug_id]
    recs = message_session_records(src, {e["message_id"] for e in events})
    keys = [k.casefold() for k in [bug.get("title"), *(bug.get("aliases") or []), *(bug.get("companies") or [])] if k]
    seen = {r["id"] for r in recs}
    for sess in sessions(src.channel.get(bug.get("channel_id"), [])):
        if sess[-1].ts >= since_ts and any(
                any(k in m.text.casefold() for k in keys) or PR_LINK_RE.search(m.text) for m in sess):
            recs += [{"id": m.id, "text": fmt_message(src, m)} for m in sess if m.id not in seen]
            seen.update(m.id for m in sess)
    links = {l for e in events for l in e.get("fix_pr_links") or []}
    links |= {e["src_id"].split(":", 1)[1] for e in src.rows("bugs", type="FIXES") if e["dst_id"] == bug_id}
    links |= {l for r in recs for l in PR_LINK_RE.findall(r["text"])}
    return recs + [pr_record(src, src.pr_by_link[l]) for l in sorted(links) if l in src.pr_by_link]


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
        recs = bug_context_records(src, ep["bug"], ep["ts"])
        claim = {
            "event": f"{ep['type']} event {ev['id']} on message {ep['message_id']}",
            "bug identity": f"'{bug['title']}' (component {bug.get('component') or NOT_EXTRACTED}, channel {bug['channel_name']})",
            "time": ep["ts"],
            "actor": ", ".join(src.person(a) for a in acted) or "none",
            "assignee(s) of this event": (NOT_APPLICABLE if ep["type"] == "REPORTED" else
                                          ", ".join(src.person(a) for a in ep.get("assignee_eids") or []) or NOT_EXTRACTED),
            "customer(s) of the bug": ", ".join(bug.get("customer_ids") or []) or NOT_EXTRACTED,
            "bug status (frozen historical flags)": (
                f"unresolved {not bug.get('resolved')} (no linked fix PR), active {bug.get('active')} (last event REPORTED, "
                f"ASSIGNED or REOPENED), reopened {bug.get('reopened')} (at least one REOPENED), no fix needed "
                f"{bug.get('no_fix_needed')} (at least one DISMISSED, ever); label {bug.get('status')}"),
            "bug current assignee(s)": ", ".join(src.person(a) for a in bug.get("current_assignees") or []) or NOT_EXTRACTED,
        }
        out.append(make_packet("bug_event", ident, src, claim=claim, records=recs,
                               meta={"event_id": ev["id"], "bug": ep["bug"]}))
    return out, info


def _source_type(src: Source, rid: str) -> str:
    return ("segment" if rid in src.seg else "message" if rid in src.msg else "document" if rid in src.doc
            else "chat" if rid in src.chat else "other")


def review_units(src: Source) -> list[dict]:
    """Sampling units of the SUGGESTED edges: one per (remark, person), where a version-ambiguous remark counts once per
    product document type (it carries its candidate versions) and a remark whose name resolution kept several candidates
    is one unit carrying the candidate set."""
    docs_type = lambda dst: getattr(src.doc.get(dst.split(":", 1)[1]), "type", dst)  # noqa: E731
    groups = defaultdict(list)
    for e in src.rows("semantic", type="SUGGESTED"):
        unit_doc = docs_type(e["dst_id"]) if e["props"].get("version_ambiguous") else e["dst_id"]
        groups[(tuple(e["src"]), e["key"], unit_doc)].append(e)
    units = []
    for (srcs, key, unit_doc), edges in sorted(groups.items()):
        persons = sorted({x["src_id"].split(":", 1)[1] for x in edges})
        versions = sorted({x["dst_id"].split(":", 1)[1] for x in edges})
        if any(x["props"].get("ambiguous") for x in edges):
            people = [persons]
        else:
            people = [[x] for x in persons]
        for ppl in people:
            who = "+".join(ppl)
            units.append({"ident": f"{src.product}|{who}|{'+'.join(srcs)}|{key}|{unit_doc}", "edge": edges[0],
                          "persons": ppl, "versions": versions,
                          "ambiguous": len(ppl) > 1 or bool(edges[0]["props"].get("ambiguous"))})
    return units


def review_packets(sources: dict[str, Source], n: int, rng: random.Random) -> tuple[list[dict], dict]:
    pool = []
    for p, src in sorted(sources.items()):
        for u in review_units(src):
            e = u["edge"]
            rid = (e["src"] or [""])[0]
            pool.append((u["ident"], f"{p}|{_source_type(src, rid)}|{e['props'].get('version_ambiguous')}", (src, u)))
    picked, info = sample_strata(pool, n, rng)  # info.available counts distinct units
    out = []
    for ident, _, (src, u) in picked:
        e = u["edge"]
        ep, rid = e["props"], (e["src"] or [""])[0]
        cands = u["persons"] if u["ambiguous"] else None
        speaker = ep.get("speaker_name") or (src.seg[rid].speaker if rid in src.seg else None)
        eid = u["persons"][0]
        doc = src.doc.get(e["dst_id"].split(":", 1)[1])
        siblings = [d for d in src.records.documents if doc and d.type == doc.type]
        st = _source_type(src, rid)
        recs = (meeting_records(src, src.seg[rid].meeting_id) if st == "segment"
                else message_session_records(src, [rid]) if st == "message"
                else [doc_record(src, src.doc[rid])] if st == "document"
                else [r for r in [record_text(src, rid)] if r])
        recs += [doc_record(src, d, feedback=False) for d in siblings]
        if st == "segment" and speaker and sum(
                1 for e2 in src.records.employees.values() if e2["name"].casefold().strip() == speaker.casefold().strip()) > 1:
            # namesakes in the directory: the name-rule evidence for every same-name person, resolved or not
            recs += candidate_records(src, {"meeting_id": src.seg[rid].meeting_id, "name": speaker})
        level = ("document TYPE (several versions exist; the relation is correct if the source is about this type of "
                 f"document: {doc.type if doc else '?'}; versions the extraction linked: {', '.join(u['versions'])}; "
                 f"all versions: {', '.join(d.id for d in siblings)})"
                 if ep.get("version_ambiguous") else f"the single document {doc.id if doc else e['dst_id']} "
                 f"(type {doc.type if doc else '?'})")
        who = (f"one of the candidate speakers ({', '.join(src.person(c) for c in cands)}); the name resolution kept several "
               f"candidates for the speaker name '{speaker or '?'}'" if cands else src.person(eid))
        claim = {"relation": f"{who} SUGGESTED a change/question about {level}",
                 "kind": ep.get("kind"), "section": ep.get("section"), "text": ep.get("text"),
                 **({"ambiguous": True, "judged as": "a candidate set (the complete tied set), not one asserted person"}
                    if cands else {}),
                 "source record": f"{rid} ({st})", "judged at": "document type" if ep.get("version_ambiguous") else "document"}
        out.append(make_packet("review_relation", ident, src, claim=claim, records=recs, identity="always",
                               meta={"source_type": st, "version_ambiguous": bool(ep.get("version_ambiguous")),
                                     "candidates": cands or [eid], "versions": u["versions"]},
                               names=[src.records.employees[c]["name"] for c in (cands or [eid]) if c in src.records.employees]))
    return out, info


def candidate_records(src: Source, row: dict) -> list[dict]:
    """Evidence for the frozen rule, for every directory person with the speaker's exact name: in the meeting, @-mentioned
    in the meeting (transcript or chat), and posts within +-3 days with their full sessions. A post is related when its
    session text names the meeting's document type or the link of a document of that type (names.py)."""
    m = src.meeting[row["meeting_id"]]
    mdt = datetime.fromisoformat(m.date)
    chats = [c for c in src.records.chats if c.meeting_id == m.id]
    mentioned_eids = {e for sg in m.segments for e in parse_mentions(sg.text)} | {e for c in chats for e in c.mentions}
    terms = ([m.document_type.lower()] if m.document_type else []) + [d.link.lower() for d in src.records.documents
                                                                         if d.type == m.document_type]
    out = []
    for eid, e in sorted(src.records.employees.items()):
        if e["name"].casefold().strip() != row["name"].casefold().strip():
            continue
        posts = [x for x in src.records.messages if x.user_id == eid and not x.is_bot
                 and abs(datetime.fromisoformat(x.ts) - mdt) <= WINDOW]
        sess = message_session_records(src, [x.id for x in posts])
        hit = None
        for x in posts:
            for sx in sessions(src.channel[x.channel_id]):
                if x.id in {y.id for y in sx}:
                    text = "\n".join(y.text for y in sx).lower()
                    hit = hit or next((t for t in terms if t in text), None)
        listing = "; ".join(f"[{x.id}] {x.ts} #{x.channel_name}" for x in posts) or "none"
        out.append({"id": f"candidate:{eid}", "text": (
            f"Candidate {e['name']} ({eid}), directory role {e['role']}. In the meeting's participants: "
            f"{'yes' if eid in m.participants else 'no'}. @-mentioned in this meeting: {'yes' if eid in mentioned_eids else 'no'}. "
            f"Slack posts within 3 days of {m.date}: {len(posts)} ({listing}); the full session of each post follows below. "
            f"Related post under the rule: {'yes' if hit else 'no'}"
            + (f" (the session names '{hit}')" if hit else "")
            + f". Terms checked: document type '{m.document_type}' and the links of documents of that type: "
              f"{', '.join(terms[1:]) or 'none'}.")})
        out += sess
    return out


def name_packets(sources: dict[str, Source], n: int, rng: random.Random) -> tuple[list[dict], dict]:
    pool = [(f"{p}|{r['meeting_id']}|{r['name']}", f"{p}|{r['status']}", (src, r))
            for p, src in sorted(sources.items()) for r in src.names]
    picked, info = sample_strata(pool, n, rng)
    out = []
    for ident, _, (src, r) in picked:
        recs = meeting_records(src, r["meeting_id"])
        recs += candidate_records(src, r)
        claim = {"speaker label": f"'{r['name']}' in meeting {r['meeting_id']}", "status": r["status"],
                 "eligible candidates the extractor found": ", ".join(src.person(e) for e in r["candidates"]) or "nobody",
                 "returned (the unique winner if resolved, the tied winners if ambiguous)":
                     ", ".join(src.person(e) for e in r["ids"]) or "nobody"}
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


def assignment_message(src: Source, item: str) -> str | None:
    """The message of a bug's latest ASSIGNED event (latest by time, then sequence); None for other items."""
    evs = [e["props"] for e in src.rows("bugs", label="BugEvent") if e["props"]["bug"] == item and e["props"]["type"] == "ASSIGNED"]
    return max(evs, key=lambda e: (e["ts"], e.get("seq") or 0))["message_id"] if evs else None


def responsibility_context(src: Source, item: str, person: str, start_mid: str | None) -> list[dict]:
    """Every message after the assignment (bug) or opening (PR) that could show a hand-over, with no cap: all later
    messages of each channel where the item was discussed (and the bug's own channel), plus all later messages anywhere
    in the product that name the item (id, title, aliases, PR link or number) or the person (eid or name)."""
    label, rest = item.split(":", 1)
    node = src.node("bugs", item) or src.node("rules", item) or src.node("structural", item) or {"props": {}}
    terms = [t.casefold() for t in [item, node_title(src, item), *(node["props"].get("aliases") or [])] if t]
    pat = None
    if label == "PR":
        terms.append(rest.casefold())
        num = rest.rsplit("/", 1)[-1]
        pat = re.compile(rf"pull/{num}(?!\d)|\bPR\s*#?\s*{num}(?!\d)", re.I) if num.isdigit() else None
    emp = src.records.employees.get(person, {})
    who = [person.casefold()] + ([emp["name"].casefold()] if emp else [])
    pr = src.pr_by_link.get(rest) if label == "PR" else None
    if start_mid in src.msg:
        ts0, id0 = src.msg[start_mid].ts, start_mid
    else:
        ts0, id0 = (pr.created_at if pr else ""), None
    named = lambda m: any(t in m.text.casefold() for t in terms) or bool(pat and pat.search(m.text))  # noqa: E731
    chans = {m.channel_id for m in src.records.messages if named(m)} | ({node["props"]["channel_id"]} if node["props"].get("channel_id") else set())
    keep = [m for m in src.records.messages if (m.ts > ts0 or (m.ts == ts0 and m.id != id0)) and (
        m.channel_id in chans or named(m) or any(w in m.text.casefold() for w in who))]
    return [{"id": m.id, "text": fmt_message(src, m)} for m in sorted(keep, key=lambda m: (m.ts, m.id))]


def _chars(recs: list[dict]) -> int:
    return sum(len(r["text"]) for r in recs)


def _whatif_unowned_packet(ident: str, src: Source, x: dict) -> dict:
    a = x["assignment"]
    recs = item_records(src, x["item"])
    start = a.get("message_id") or assignment_message(src, x["item"])
    later = responsibility_context(src, x["item"], x["person"], start)
    got, miss2 = _hydrate(src, [i for i in [a.get("message_id"), *(a.get("records") or [])] if i])
    claim = {"statement": f"If {src.person(x['person'])} leaves, the {x['label']} '{x['title']}' ({x['item']}) is newly "
                          "unowned: it is open work and this person is its only remaining owner",
             "owner basis": ("latest ASSIGNED event of the bug (assignees of that event)" if x["label"] == "Bug"
                             else "the PR author"),
             "assignment evidence": f"{a['kind']} {a.get('event') or ''} {a.get('ts') or ''}".strip()}
    out = make_packet("whatif_unowned", ident, src, claim=claim, records=got + recs + later,
                      meta={"item": x["item"], "person": x["person"], "later_messages": len(later), "unhydrated": miss2})
    out["meta"]["chars"] = len(out["text"])
    return out


def _edge_records(src: Source, dependent: str, prereq: str, cites) -> tuple[list[dict], int]:
    got, miss = _hydrate(src, cites)
    for c in cites:  # keep the task enumeration: the whole meeting or session behind each cited line
        if c in src.seg:
            got += meeting_records(src, src.seg[c].meeting_id)
        elif c in src.msg:
            got += message_session_records(src, [c])
    return got + item_records(src, dependent) + item_records(src, prereq), miss


def _whatif_edge_packet(ident: str, src: Source, x: dict) -> dict:
    recs, miss = _edge_records(src, x["dependent"], x["prerequisite"], x["citations"])
    claim = {"statement": f"'{node_title(src, x['prerequisite'])}' ({x['prerequisite']}) is development work of "
                          f"'{node_title(src, x['dependent'])}' ({x['dependent']}) (membership)",
             "rule": x["rule"], "cited records": ", ".join(x["citations"]) or "none"}
    return make_packet("whatif_edge", ident, src, claim=claim, records=recs,
                       meta={"rule": x["rule"], "unhydrated": miss})


def _whatif_path_packet(ident: str, src: Source, x: dict) -> dict:
    recs, miss = [], 0
    for e in x["edges"]:
        r, m = _edge_records(src, e["dependent"], e["prerequisite"], e["citations"])
        recs, miss = recs + r, miss + m
    end = x["path"][-1]
    resp = responsibility_context(src, end, x["person"], assignment_message(src, end))
    recs += item_records(src, end) + resp
    steps = [f"{e['prerequisite']} belongs to the development work of {e['dependent']} (rule {e['rule']}, cited {', '.join(e['citations']) or 'none'})"
             for e in x["edges"]]
    claim = {"statement": f"If {src.person(x['person'])} leaves, '{node_title(src, x['affected'])}' ({x['affected']}) is "
                          "affected: it is open work or a pending feature and, through the membership steps below, the development work "
                          "of each item includes the next, ending at the last item, which becomes newly unowned (open work whose "
                          "only owner is this person)",
             "path (affected item first, newly unowned item last)": " -> ".join(x["path"]),
             "membership steps": " | ".join(steps)}
    out = make_packet("whatif_path", ident, src, claim=claim, records=recs,
                      meta={"path_length": len(x["path"]), "unhydrated": miss, "later_messages": len(resp)})
    out["meta"]["chars"] = len(out["text"])
    return out


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


ASSIGNED_PLACEMENT = (
    "Placement of ASSIGNED events (frozen convention): exactly one ASSIGNED event per assignment, on the message that hands the "
    "bug to the assignee (it names or @-mentions them with an assignment cue) when that person accepts before the next bug "
    "introduction, and no second event at the acceptance; if nobody hands it over and a person takes it, the one event is on "
    "the message where they take it.")

AUTHOR_WORDING = (
    "An author's acknowledgement, explanation, repetition, or promise to implement another person's feedback is not a "
    "reviewer contribution. Independently proposed content changes or substantive review questions are judged by the same "
    "rule regardless of authorship.")

REVIEW_BOUNDARY = (
    "A review relation needs a proposal about the document's content or a review question about it. Pure work allocation "
    "(taking a section, gathering input, proposing to collaborate) is not a review relation; a concrete content suggestion "
    "phrased \"I'll add ...\" is one. " + AUTHOR_WORDING)

BUG_TASK = ("List every bug event in channel {name} (product {product}). Use only the records below. Event types: "
            "REPORTED (a bug is introduced or reported), ASSIGNED (someone is asked to take it or takes it), RESOLVED "
            "(reported fixed or closed), DISMISSED (looked at and judged not a bug or no fix needed), REOPENED (comes "
            "back after resolution). One item per bug per event; a message can hold several events. Give the bug "
            "(the id of the message where that bug was first reported), the type, the actor (who performed the event), "
            "the time (the message timestamp) and the message id of the event. " + ASSIGNED_PLACEMENT)
REVIEW_TASK = ("List every reviewer contribution about documents of type {type} in product {product}, across ALL versions "
               "of the type ({versions}): each person who gives a suggestion, a requested change or a review question about "
               "a document of this type. " + REVIEW_BOUNDARY + " Sources are the feedback of each version, the meetings of "
               "this document type, the Slack sessions that share any version's link and the same-product Slack discussions "
               "of the document type that name no version and share no exact link. One item per person per source record. "
               "Give the reviewer as the name written in the record (or the eid when the record uses one), the id of the "
               "source record (a meeting line id like meeting#12, a message id, or the document id) and, only when the "
               "record names one version, that version's id as \"document\".")


BUG_WORDS = re.compile(r"\b(?:bug|issue|problem|vulnerabilit\w*|leak|crash\w*|error|fail\w*)\b", re.I)


def _human(m) -> bool:
    return m.in_directory and not m.is_bot


def bug_channel_eligible(src: Source, channel_id: str) -> bool:
    """Frozen before sampling, from the raw inventory only: the channel has a bug-channel identity (bug-<owner>-<Product>)
    and at least one human message that reports content (a customer id or a bug word); setup-only channels are out."""
    msgs = src.channel.get(channel_id, [])
    return bool(msgs) and msgs[0].bug_channel_owner is not None and any(
        _human(m) and (CUST_RE.search(m.text) or BUG_WORDS.search(m.text)) for m in msgs)


GENERIC_TYPE_TAIL = ("document", "report")
SUBSTANTIVE_WORDS = 10  # a substantive message has at least this many words (an acknowledgement or a one-line aside does not)


def doc_terms(d) -> list[str]:
    """Lower-case terms that mean the document or its type: the type, the type without a generic last word (document,
    report) when two or more words remain, and the document link."""
    t = d.type.casefold().strip()
    words = t.split()
    out = [t]
    if len(words) > 2 and words[-1] in GENERIC_TYPE_TAIL:
        out.append(" ".join(words[:-1]))
    return out + [d.link.casefold()]


def attributes_a_person(src: Source, feedback: str) -> bool:
    """A feedback line that names its author: it starts with a directory name or eid followed by ':' / '-' / '(' , or reads
    'from <name>' / 'by <name>'."""
    names = [re.escape(e["name"]) for e in src.records.employees.values()] + [r"eid_[A-Za-z0-9_]+"]
    alt = "|".join(sorted(names, key=len, reverse=True))
    pat = re.compile(rf"^\W*(?:{alt})\s*[:\-\u2013\u2014(]|\b(?:from|by)\s+(?:{alt})\b", re.I | re.M)
    return bool(pat.search(feedback))


def reviewer_eligible(src: Source, d) -> bool:
    """Frozen before sampling, from the raw inventory only: an attributable review opportunity. Any of:
    - feedback with a line that attributes a person (name or eid at the line start before ':' '-' '(' , or 'from/by <name>');
    - a meeting of the document's type with a named speaker other than the author in a segment that mentions the document
      or its type (the type, the type without a generic last word, or the link);
    - a Slack session in which a message names the document or its type and that message or a later one in the session is a
      substantive message (at least SUBSTANTIVE_WORDS words) by a human participant other than the author."""
    author, terms = src.name_of(d.author), doc_terms(d)
    if d.feedback and attributes_a_person(src, d.feedback):
        return True
    for m in src.records.meetings:
        if m.document_type == d.type and any(sg.speaker and sg.speaker.casefold().strip() != author
                                             and any(t in sg.text.casefold() for t in terms) for sg in m.segments):
            return True
    named = lambda m: any(t in m.text.casefold() for t in terms)  # noqa: E731
    for ch in src.channel.values():
        for sess in sessions(ch):
            first = next((i for i, m in enumerate(sess) if named(m)), None)
            if first is not None and any(_human(m) and m.user_id != d.author and len(m.text.split()) >= SUBSTANTIVE_WORDS
                                         for m in sess[first:]):
                return True
    return False


def recall_bug_packets(sources: dict[str, Source], n: int, rng: random.Random) -> tuple[list[dict], dict]:
    ident = {p: [c for c in sorted(s.channel) if s.channel[c][0].bug_channel_owner is not None] for p, s in sources.items()}
    cands = {p: [c for c in chs if bug_channel_eligible(sources[p], c)] for p, chs in ident.items()}
    out = []
    for p, ch in _spread_products(cands, n, rng):
        src = sources[p]
        msgs = sorted(src.channel[ch], key=lambda m: (m.ts, m.id))
        recs = [{"id": m.id, "text": fmt_message(src, m)} for m in msgs]
        task = BUG_TASK.format(name=msgs[0].channel_name, product=p)
        out.append(make_packet("recall_bug", f"{p}|{ch}", src, task=task, records=recs,
                               meta={"channel_id": ch, "channel_name": msgs[0].channel_name, "messages": len(msgs)}))
    ne = sum(len(v) for v in cands.values())
    return out, {"requested": n, "available": ne, "eligible": ne, "ineligible": sum(len(v) for v in ident.values()) - ne}


def _types(src: Source) -> list[str]:
    return sorted({d.type for d in src.records.documents})


def recall_reviewer_packets(sources: dict[str, Source], n: int, rng: random.Random) -> tuple[list[dict], dict]:
    """Targets are (product, document type): eligible when any version is (the frozen rule, at type level)."""
    cands = {p: [t for t in _types(s) if any(reviewer_eligible(s, d) for d in s.records.documents if d.type == t)]
             for p, s in sources.items()}
    out = []
    for p, typ in _spread_products(cands, n, rng):
        src = sources[p]
        versions = [d for d in src.records.documents if d.type == typ]
        recs = [doc_record(src, d) for d in versions]
        for m in src.records.meetings:
            if m.document_type == typ:
                recs += meeting_records(src, m.id)
        terms = sorted({t for d in versions for t in doc_terms(d)})
        linked = [m.id for m in src.records.messages if any(t in m.text.casefold() for t in terms)]
        recs += message_session_records(src, linked)
        task = REVIEW_TASK.format(type=typ, product=p, versions=", ".join(d.id for d in versions))
        names = [s.speaker for m in src.records.meetings if m.document_type == typ for s in m.segments if s.speaker]
        out.append(make_packet("recall_reviewer", f"{p}|{typ}", src, task=task, records=recs, identity="always",
                               meta={"document_type": typ, "document_ids": [d.id for d in versions]}, names=names))
    ne, total = sum(len(v) for v in cands.values()), sum(len(_types(s)) for s in sources.values())
    return out, {"requested": n, "available": ne, "eligible": ne, "ineligible": total - ne}


# ---------------------------------------------------------------- prompts (versioned and hashed)
COMMON_PRECISION = """You are auditing extracted facts against raw company records. Work only from the records and the
directory inside each packet. Do not use outside knowledge and do not guess. For each packet decide:
  correct    every part of the claim is supported by the records
  incorrect  the records contradict the claim, or the records given contain no support for it
  uncertain  the records support conflicting readings that the packet cannot settle
Output only a JSON array with one object per packet:
  {"packet_id": "<id>", "verdict": "correct|incorrect|uncertain", "reason": "<one or two sentences naming record ids>"
   [, "fields": {"<field>": "correct|incorrect|na"}]}
Do not add text outside the JSON."""

OPEN_WORK = """Open work (the frozen definition): an ACTIVE bug, or a PR whose state is "open" and merged is "False".
A bug is active when its last event, in time order, is REPORTED, ASSIGNED or REOPENED (a reopened bug is active again even
though it was resolved before; RESOLVED and DISMISSED last means not active). A feature is pending work when it was
discussed but not implemented, or when it is not implemented (a missing value counts as not implemented) and a PR about
its development is open and unmerged. Owners: a bug's owners are the assignees of its latest ASSIGNED event (an
earlier assignment is replaced); a PR's owner is its author."""

RUBRICS = {
    "bug_event": """Kind bug_event. The claim is one bug event and the bug it belongs to. Event model (frozen): an event's
assignees are the people that event assigns. A REPORTED event carries no assignees by definition (the claim shows
"not applicable for REPORTED"). An assignment is a separate ASSIGNED event. """ + ASSIGNED_PLACEMENT + """ The bug's current assignee is judged separately, as the bug's current
ownership after the latest assignment and any later reassignment. Status uses the frozen historical definitions, not
"status now" in a looser sense: unresolved = the bug has no linked fix PR; active = the last event in time order is REPORTED,
ASSIGNED or REOPENED; reopened = at least one REOPENED event ever; no fix needed = at least one DISMISSED event ever (even
if the bug was later reopened). A component shown as "not extracted" is a missing value.
Judge five fields and report all five in "fields": {"identity","customer","assignee","status","time"}, each "correct",
"incorrect" or "na". identity: the title, component and event type describe the bug and the event the records show at that
message. customer: the customer ids belong to that bug. assignee: the event's assignees follow the event model above, and
the bug's current assignee follows from the records including later reassignment, each judged on its own; the field is
correct only if both are. status: the frozen flags above follow from the records, including later sessions and the linked
pull request records. time: the event time is the timestamp of the cited message. "na" is allowed only for customer or
assignee when the claim shows "not extracted" or "not applicable" for every value of that field and the records also give
none; identity, status and time are never "na". A missing required field in your answer is a failure. The verdict is
"correct" only if every field is "correct" (or such an "na"); otherwise "incorrect" (or "uncertain" under the general
boundary).""",
    "review_relation": """Kind review_relation. The claim says a person suggested a change, or asked a review question,
about a document. """ + REVIEW_BOUNDARY + """ Judge the cited content proposal independently of adjacent allocation language; a message containing both can support a review relation. When the claim lists several candidate speakers (the name resolution kept several
candidates and the claim shows ambiguous: True), judge it as a candidate set, not as an assertion about one person. It is
correct only when the candidates are exactly the complete tied set under the frozen name rule (the eligible candidates are
the directory people with exactly that name who are in the meeting, posted in the product Slack within 3 days of it, or are
@-mentioned in it; scored 3 for a role cue matching the candidate's role in the speaker's own lines, plus 2 for a related post
within 3 days, plus 1 for being mentioned; ties at the top score are kept): no unsupported extra candidate and no omitted
tied candidate, using the candidate evidence records in the packet. That the records cannot say which candidate spoke does
not by itself make the relation incorrect. With a single person, correct if
that person did say it in the source record and it is about the document. When the claim is judged at document-type level
(several versions exist), it is correct if the source is about a document of that type OF THIS PRODUCT (the same subject);
a discussion of another product's document, such as a competitor's, is not. Do not require a specific version. Incorrect if
the speaker is not among the candidates, the text is not a suggestion or review question, or the source is about another
document type or another product's document.""",
    "name_resolution": """Kind name_resolution. The claim maps a speaker name in one meeting to employee ids. The frozen
rule: the eligible candidates are the directory people with exactly that name who are in the meeting's participants, or
posted in the product Slack within 3 days of the meeting date, or are @-mentioned in the meeting chat (candidate records
give these facts). One eligible candidate resolves without scoring. Several are scored: 3 if the speaker's own lines
state a role cue matching the candidate's directory role (the speaker says their own role, such as "as QA I checked"),
plus 2 if the candidate has a related post within 3 days (a post in a session that names the meeting's document type or
document link), plus 1 if the candidate is mentioned in the meeting. A unique top score resolves to that candidate; a tie
for the top score is ambiguous and the tied candidates are returned. The claim lists the eligible candidates the
extractor found and what it returned. Correct only if the eligible candidates are right and the returned ids follow from
the rule applied to the records.""",
    "whatif_unowned": OPEN_WORK + """

Kind whatif_unowned. Correct only if the item is open work, the named person is an owner, no other owner remains, and no
later record in the later context hands it over to someone else, reassigns it, merges or resolves it, or otherwise
changes who owns it or whether it is open.""",
    "whatif_edge": """Kind whatif_edge. A dependency edge records development membership. Rules: IMPLEMENTED_BY (a feature
is implemented by the PR), ABOUT with kind development (a development PR belongs to the feature), FIXES (a PR fixes the
bug). An explicit task-to-PR statement establishes development membership, including component, testing, monitoring and
documentation work. Hard necessity is not claimed; membership or merged status alone does not establish a hard prerequisite.
Correct only if the records show that membership between exactly these two items (the cited records, with the surrounding
task enumeration, name both or make the link clear). Topical association alone (the PR merely mentions the subject) is
not membership.""",
    "whatif_path": OPEN_WORK + """

Kind whatif_path. The claim gives a chain of dependency steps from an affected item to the last item, which becomes newly
unowned when the person leaves. Check every component and answer correct only if all hold: (1) each dependency step is
supported by the records as development membership (rules: IMPLEMENTED_BY, ABOUT with kind development, FIXES): an explicit
task-to-PR statement establishes membership, including component, testing, monitoring and documentation work; hard
necessity is not claimed and topical association alone is not membership; (2) the steps join in order; (3) the affected item is open work or pending work as defined
above; (4) the last item is open work, the person is an owner and no other owner remains after the later context.""",
}

COMMON_RECALL = """You are listing every item of one kind in the raw company records of a packet. Work only from the
records and the directory in the packet. Be complete and do not invent items. Output only JSON:
  {"packet_id": "<id>", "items": [ ... ]}
Do not add text outside the JSON."""

RECALL_RULES = {
    "recall_bug": """Each item: {"bug": "<id of the message where this bug was first reported>", "bug_title": "<optional title, ignored>", "type": "REPORTED|ASSIGNED|RESOLVED|DISMISSED|REOPENED", "actor": "<eid or name of who
performed it>", "time": "<message timestamp>", "message_id": "<id>"}. List one item per bug per
event. The "bug" field is always the id of the message where the bug was first reported, never a title.
Example item: {"bug": "20261117-1-a77c3", "bug_title": "Duplicate Log Entries", "type": "ASSIGNED", "actor": "eid_0ac476e4", "time": "2026-11-17T07:06:00", "message_id": "20261117-1-a77c3"}
A bug is reported once when it is introduced; it is assigned when someone is asked to take it or takes it; resolved
when reported fixed or closed; dismissed when judged not a bug; reopened when it returns after resolution.
""" + ASSIGNED_PLACEMENT + """ A REPORTED event has no assignees.""",
    "recall_reviewer": """Each item: {"reviewer": "<name as written in the record, or the eid>", "source": "<source record
id>", "document": "<optional: the version id, only when the record names one version>"}. The target is a document type of
one product: list every reviewer contribution about any version of the type, from named feedback, meetings of the type,
Slack sessions that share any version's link, and same-product Slack discussions of the type that name no version and
share no exact link. A reviewer is a person who gives a suggestion, a requested change or a review question about the
document type. """ + REVIEW_BOUNDARY + """ One item per person per source record.""",
}

THIRD_PRECISION = """You are the third and final auditor. Two auditors disagreed, or one gave no usable answer. Read the
packet and both reasoned verdicts, check the records yourself, and decide. Your verdict must be correct or incorrect
(uncertain only if the records truly cannot decide). Use the rubric below. Output only a JSON array with one object per
packet in the same format as the first auditors."""

THIRD_RECALL = """You are reconciling two auditors' lists. Read the packet and both lists, check the records yourself, and
output the single complete list: keep items that the records support, drop items they do not, add items both missed.
Output only {"packet_id": "<id>", "items": [...]} in the same item format. You have not been shown any extraction."""


PROMPT_CHANGES_V2 = (
    "bug_event: event model defined (REPORTED carries no assignees, ASSIGNED placement, current assignee judged separately); "
    "status uses the frozen historical predicates (unresolved, active, reopened, no fix needed); missing values render "
    "'not extracted'",
    "recall_bug: the same ASSIGNED placement convention; the duplicate 'bug' key in the item format removed",
    "review_relation: document type of THIS product (same subject); ambiguous claims judged as a candidate set and shown with "
    "the candidate people and the ambiguous flag",
    "whatif_edge and whatif_path: a dependency needs evidence of implementation or necessity, not topical association, for "
    "code, test and documentation PRs alike",
)

PROMPT_CHANGES_V3 = (
    "product identity: review_relation and recall_reviewer packets (and name, what-if and other packets whenever a product "
    "alias appears in their records) show the product's channel ids and names and the creation notices from the bundle, "
    "never an extracted alias list",
    "review_relation: namesake evidence (directory matches, participants, mentions, nearby posts with full sessions) for "
    "every meeting speaker with same-name people, not only ambiguous claims",
    "whatif_edge and whatif_path: the whole meeting or session behind each cited line is kept, so task enumerations and "
    "boundaries are visible; the rubric judges development membership (an explicit task-to-PR statement, including "
    "component, testing, monitoring and documentation work), not hard necessity",
    "review_relation: the cited content proposal is judged independently of adjacent allocation language",
    "recall sampling: reviewer documents need an attributable review opportunity and bug channels a bundle bug-channel "
    "identity with human report content (frozen before sampling); same-product type-level discussions are included in "
    "recall_reviewer packets; eligible and ineligible pool sizes are recorded",
)
PROMPT_CHANGES_BY_VERSION = {audit_prompts_v2.VERSION: PROMPT_CHANGES_V2}
PROMPT_CHANGES_V4 = (
    "recall_reviewer: targets are (product, document type); the packet holds the review evidence of all versions of the "
    "type (each version's feedback, same-type meetings, sessions sharing any version's link, same-product type-level "
    "discussions); items may name a version (optional 'document')",
    "recall comparison: (canonical reviewer, source record) matched against SUGGESTED edges to any version of the type, "
    "deduplicated across versions (the gate); strict per-document recall and distinct-reviewer coverage reported alongside",
    "author wording: an author's acknowledgement, explanation, repetition or promise to implement another person's feedback "
    "is not a reviewer contribution; independent content proposals count regardless of authorship (recall rules, task text "
    "and the review precision rubric)",
)
PROMPT_CHANGES_BY_VERSION[PROMPT_VERSION] = PROMPT_CHANGES_V4
PROMPT_CHANGES_BY_VERSION[audit_prompts_v3.VERSION] = PROMPT_CHANGES_V3
PROMPT_CHANGES = PROMPT_CHANGES_V4


def prompt_set(version: str) -> dict:
    """The prompt texts of a version: v1 is frozen in audit_prompts_v1 (round 1), v2 is this module."""
    frozen = {m.VERSION: m for m in (audit_prompts_v1, audit_prompts_v2, audit_prompts_v3)}
    if version in frozen:
        m = frozen[version]
        return {"common_precision": m.COMMON_PRECISION, "common_recall": m.COMMON_RECALL, "third_precision": m.THIRD_PRECISION,
                "third_recall": m.THIRD_RECALL, "open_work": m.OPEN_WORK, "rubrics": m.RUBRICS, "recall_rules": m.RECALL_RULES}
    if version != PROMPT_VERSION:
        raise ValueError(f"unknown prompt version {version!r}")
    return {"common_precision": COMMON_PRECISION, "common_recall": COMMON_RECALL, "third_precision": THIRD_PRECISION,
            "third_recall": THIRD_RECALL, "open_work": OPEN_WORK, "rubrics": RUBRICS, "recall_rules": RECALL_RULES}


def prompt_hashes(version: str = PROMPT_VERSION) -> dict[str, str]:
    ps = prompt_set(version)
    texts = {k: ps[k] for k in ("common_precision", "common_recall", "third_precision", "third_recall", "open_work")}
    texts |= {f"rubric:{k}": v for k, v in ps["rubrics"].items()} | {f"recall_rules:{k}": v for k, v in ps["recall_rules"].items()}
    return {k: sha256_text(v) for k, v in sorted(texts.items())} | {"version": version}


def round_version(round_n: int, root: Path = AUDIT_DIR) -> str:
    mf = round_dir(round_n, root) / "manifest.json"
    return json.loads(mf.read_text(encoding="utf-8"))["prompt_hashes"]["version"] if mf.exists() else PROMPT_VERSION


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
    if d.exists():
        raise FileExistsError(f"{d} exists: a round is never overwritten (failed rounds stay); use the next round number")
    pdir = d / "packets"
    pdir.mkdir(parents=True)
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
    rd = round_dir(round_n, root)
    mf = rd / "manifest.json"
    listed = {e["id"] for e in json.loads(mf.read_text(encoding="utf-8"))["packets"]} if mf.exists() else None
    out = {}
    for f in sorted((rd / "packets").glob("*.json")):  # strays outside the manifest are an integrity problem, not input
        if listed is None or f.stem in listed:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return out


def integrity_problems(round_n: int, root: Path = AUDIT_DIR) -> list[str]:
    """Differences between the manifest and the packet files on disk, and missing auditor metadata."""
    rd = round_dir(round_n, root)
    if not (rd / "manifest.json").exists():
        return ["manifest missing"]
    manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    want = {e["id"]: e["sha256"] for e in manifest["packets"]}
    problems = []
    have = {f.stem: sha256_text(f.read_text(encoding="utf-8")) for f in (rd / "packets").glob("*.json")}
    problems += [f"packet missing: {i}" for i in sorted(set(want) - set(have))]
    problems += [f"packet not in the manifest: {i}" for i in sorted(set(have) - set(want))]
    problems += [f"packet changed: {i}" for i in sorted(set(want) & set(have)) if want[i] != have[i]]
    if sha256_text("\n".join(f"{i}:{want[i]}" for i in sorted(want))) != manifest["sample_hash"]:
        problems.append("sample hash does not match the manifest packet list")
    aud = rd / "auditors.json"
    meta = json.loads(aud.read_text(encoding="utf-8")) if aud.exists() else {}
    for who in ("codex", "claude", "third"):
        m = meta.get(who)
        if not isinstance(m, dict) or any(not isinstance(m.get(k), str) or not m[k].strip() for k in AUDITOR_KEYS):
            problems.append(f"auditor metadata missing or incomplete for {who} (needs {', '.join(AUDITOR_KEYS)})")
    return problems


def _verdict_hashes(rd: Path) -> dict:
    return {w: sha256_text("".join(sha256_text(f.read_text(encoding="utf-8")) for f in sorted((rd / "verdicts" / w).glob("*.json"))))
            if (rd / "verdicts" / w).exists() else None for w in ("codex", "claude", "third")}


def is_recall(kind: str) -> bool:
    return kind in RECALL_QUOTAS


def _batch_plan(packets: dict[str, dict], size: int = BATCH_MAX_PACKETS) -> dict[str, list[str]]:
    """Scheduling only. Batches are homogeneous by kind (the auditor holds one rubric and nothing else), at most `size`
    packets and BATCH_MAX_CHARS characters; a packet over the limit goes alone. Recall packets go one per batch."""
    plan = {}
    for kind in KINDS:
        ids = sorted(i for i, p in packets.items() if p["kind"] == kind)
        cap = 1 if is_recall(kind) else min(size, BATCH_MAX_PACKETS)
        n, cur, chars = 0, [], 0
        for i in ids:
            c = len(packets[i]["text"])
            if cur and (len(cur) >= cap or chars + c > BATCH_MAX_CHARS):
                n += 1
                plan[f"{kind}-{n:02d}"], cur, chars = cur, [], 0
            cur.append(i)
            chars += c
        if cur:
            plan[f"{kind}-{n + 1:02d}"] = cur
    return plan


def batch_header(kind: str, third: bool = False, version: str = PROMPT_VERSION) -> str:
    ps = prompt_set(version)
    if is_recall(kind):
        return (ps["third_recall"] if third else ps["common_recall"]) + "\n\n" + ps["recall_rules"][kind]
    return (ps["third_precision"] if third else ps["common_precision"]) + "\n\n" + ps["rubrics"][kind]


def write_batches(round_n: int, size: int = 10, root: Path = AUDIT_DIR) -> list[str]:
    packets = load_packets(round_n, root)
    bdir = round_dir(round_n, root) / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    plan = _batch_plan(packets, size)
    for name, ids in plan.items():
        kind = packets[ids[0]]["kind"]
        parts = [batch_header(kind, version=round_version(round_n, root))] + [f"\n==== {packets[i]['id']} ====\n{packets[i]['text']}" for i in ids]
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
        need = (("bug", "type", "actor", "time", "message_id") if packet["kind"] == "recall_bug" else ("reviewer", "source"))
        for it in items:
            if not isinstance(it, dict) or any(not isinstance(it.get(k), str) or not it[k].strip() for k in need):
                return "item missing a required string field"
            if packet["kind"] == "recall_bug" and it["type"].upper() not in EVENT_TYPES:
                return "unknown event type"
            if packet["kind"] == "recall_reviewer" and "document" in it and not isinstance(it["document"], str):
                return "document must be a string"
        return None
    v = obj.get("verdict")
    if v not in VERDICTS:
        return "bad verdict"
    if not isinstance(obj.get("reason"), str) or not obj["reason"].strip():
        return "no reason"
    f = obj.get("fields")
    if packet["kind"] != "bug_event":
        return None if f is None or isinstance(f, dict) else "bad fields"
    if not isinstance(f, dict) or set(f) != set(BUG_FIELDS) or any(x not in ("correct", "incorrect", "na") for x in f.values()):
        return "a bug event verdict needs all five fields, each correct, incorrect or na"
    claim = packet["claim"]
    empty = ("none", NOT_EXTRACTED, NOT_APPLICABLE)  # "none" is the round 1 rendering
    none_ok = {"customer": claim.get("customer(s) of the bug") in empty,
               "assignee": claim.get("assignee(s) of this event") in empty and claim.get("bug current assignee(s)") in empty}
    if any(f[k] == "na" and not none_ok.get(k) for k in BUG_FIELDS):
        return "na is only allowed where the claim has no value (customer, assignee)"
    if v == "correct" and any(x != "correct" for x in f.values() if x != "na"):
        return "correct verdict with a field that is not correct"
    if v == "incorrect" and all(x != "incorrect" for x in f.values()):
        return "incorrect verdict with no incorrect field"
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
        if isinstance(raw, dict) and "verdicts" in raw:
            raw = raw["verdicts"]
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            stats["malformed"] += 1
            continue
        for obj in raw:
            pid = obj.get("packet_id") if isinstance(obj, dict) else None
            if not isinstance(pid, str):
                stats["malformed"] += 1
            elif pid not in packets:
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
        return Counter((i["bug"].strip(), i["type"].upper().strip(), i["message_id"].strip()) for i in items)
    return Counter({(src.name_of(i["reviewer"]), i["source"].strip()) for i in items})


def extracted_keys(packet: dict, src: Source) -> Counter:
    if packet["kind"] == "recall_bug":
        ch = packet["meta"]["channel_id"]
        evs = [ev["props"] for ev in src.rows("bugs", label="BugEvent") if ev["props"]["channel_id"] == ch]
        first = {}  # a bug's report message: the message of its earliest REPORTED event
        for e in sorted((e for e in evs if e["type"] == "REPORTED"), key=lambda e: (e["ts"], e.get("seq") or 0)):
            first.setdefault(e["bug"], e["message_id"])
        return Counter((first.get(e["bug"], ""), e["type"], e["message_id"]) for e in evs)
    ids = packet["meta"].get("document_ids") or [packet["meta"]["document_id"]]  # rounds before 5 name one document
    dst = {f"Document:{i}" for i in ids}
    return Counter({(src.name_of(e["src_id"].split(":", 1)[1]), rid)
                    for e in src.rows("semantic", type="SUGGESTED") if e["dst_id"] in dst for rid in e["src"]})


def compare_reviewer_recall(packet: dict, items: list[dict], src: Source) -> dict:
    """Type-level recall (the gate: (canonical reviewer, source) against edges to any version, deduplicated), strict
    per-document recall (an item that names a version is matched against that version's edges only) and distinct-reviewer
    coverage (people)."""
    truth = truth_keys("recall_reviewer", items, src)
    ex = extracted_keys(packet, src)
    cmp = compare_recall(truth, ex)
    per_doc = {d: extracted_keys({"kind": "recall_reviewer", "meta": {"document_id": d}}, src)
               for d in packet["meta"].get("document_ids", [])}
    strict = 0
    for it in {(i["reviewer"], i["source"], i.get("document")) for i in items}:
        key = (src.name_of(it[0]), it[1].strip())
        pool = per_doc.get(it[2], ex) if it[2] else ex
        strict += key in pool
    people = {k[0] for k in truth}
    return {**cmp, "strict_matched": strict, "strict_truth": len(truth),
            "people_truth": len(people), "people_covered": len(people & {k[0] for k in ex})}


def shared_report_messages(src: Source, channel_id: str) -> int:
    """First-report messages of the channel that carry more than one bug in the extraction."""
    first = {}
    evs = [e["props"] for e in src.rows("bugs", label="BugEvent") if e["props"]["channel_id"] == channel_id]
    for e in sorted((e for e in evs if e["type"] == "REPORTED"), key=lambda e: (e["ts"], e.get("seq") or 0)):
        first.setdefault(e["bug"], e["message_id"])
    return sum(1 for n in Counter(first.values()).values() if n > 1)


def compare_recall(truth: Counter, extracted: Counter) -> dict:
    matched = sum(min(n, extracted[k]) for k, n in truth.items())
    missed = sorted("|".join(k) for k, n in truth.items() for _ in range(n - min(n, extracted[k])))
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
        sub = {i: packets[i] for i in need if packets[i]["kind"] == kind}
        for n, chunk in enumerate(_batch_plan(sub, size).values(), 1):
            parts = [batch_header(kind, third=True, version=round_version(round_n, root))]
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
            if kind == "recall_bug":
                r["shared_report_messages"] = r.get("shared_report_messages", 0) + shared_report_messages(
                    sources[p["product"]], p["meta"]["channel_id"])
            a, b, t = (v[w][0].get(pid) for w in ("codex", "claude", "third"))
            if t is None:
                r["pending"] += 1
                continue
            src = sources[p["product"]]
            if kind == "recall_reviewer" and "document_ids" in p["meta"]:
                cmp = compare_reviewer_recall(p, t["items"], src)
                for k in ("strict_matched", "strict_truth", "people_truth", "people_covered"):
                    r[k] = r.get(k, 0) + cmp[k]
            else:
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
    problems = integrity_problems(round_n, root)
    rdirs = _verdict_hashes(rd)
    aud = rd / "auditors.json"
    summary = {"round": round_n, "complete": pend == 0 and not problems, "pending_third": pend,
               "integrity_problems": problems, "sample_hash": manifest["sample_hash"],
               "prompt_hashes": manifest["prompt_hashes"], "verdict_hashes": rdirs,
               "verdict_file_stats": {w: v[w][1] for w in v}, "sizes": manifest["sizes"], "shortages": manifest["shortages"],
               "precision": cats, "recall": recall,
               "auditors": json.loads(aud.read_text(encoding="utf-8")) if aud.exists() else None,
               "packet_chars": _char_stats(packets)}
    (rd / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    (rd / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def _char_stats(packets: dict[str, dict]) -> dict:
    out = {}
    for kind in KINDS:
        n = sorted(len(p["text"]) for p in packets.values() if p["kind"] == kind)
        out[kind] = {"n": len(n), "median": n[len(n) // 2] if n else 0, "max": n[-1] if n else 0}
    return out


def render_report(s: dict) -> str:
    L = [f"# Dev acceptance audit, round {s['round']}", "",
         "Model-based dev acceptance. Two model auditors, a third pass for disagreements. Agreement between models "
         "does not establish equivalence to human labels. No source text appears in this report.", "",
         f"- complete: {s['complete']} (third passes still pending: {s['pending_third']})",
         f"- sample hash: {s['sample_hash']}", f"- prompt version: {s['prompt_hashes']['version']}",
         f"- verdict hashes: {s['verdict_hashes']}",
         f"- integrity problems: {s['integrity_problems'] or 'none'}", ""]
    pv = s["prompt_hashes"]["version"]
    if pv != "audit-prompts/v1":
        L += ["## Prompt changes", "", f"This round uses {pv}; round 1 used audit-prompts/v1 and round 2 audit-prompts/v2 "
              "(both kept unchanged). Changes in this version:"]
        L += [f"- {c}" for c in PROMPT_CHANGES_BY_VERSION.get(pv, ())] + [""]
    L += ["## Method", "",
          "Each packet is judged on its own and answered per packet. Packets are scheduled in batches for the fan-out: a "
          f"batch holds only packets of one kind and the rubric, at most {BATCH_MAX_PACKETS} packets and about "
          f"{BATCH_MAX_CHARS} characters (a larger packet goes alone). The third pass shares a model family with "
          "auditor 1 (Codex).", "", "## Auditors", ""]
    for who in ("codex", "claude", "third"):
        m = (s.get("auditors") or {}).get(who) or {}
        L.append(f"- {who}: " + ", ".join(f"{k} {m.get(k)}" for k in AUDITOR_KEYS))
    L.append("")
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
    rr = s["recall"].get("recall_reviewer") or {}
    if "strict_matched" in rr:
        L += ["", f"Reviewer recall alongside the gate: strict per-document recall {rr['strict_matched']}/{rr['strict_truth']} "
              f"(an item that names a version is matched against that version only); distinct-reviewer coverage "
              f"{rr['people_covered']}/{rr['people_truth']} people."]
    if s["prompt_hashes"]["version"] >= "audit-prompts/v4":
        L += ["", "Disclosure: After inspecting dev rounds 1-4, the reviewer-recall endpoint was amended: targets are document "
              "types (all versions), author acknowledgements are excluded by act, and strict per-document recall and "
              "distinct-reviewer coverage are reported alongside. Earlier rounds' results are unchanged and remain failed."]
    empty = sorted(k for k, v in (s.get("sizes") or {}).items() if v.get("available") == 0)
    if empty:
        L += ["", f"Not estimable (empty dev pool, excluded from pass and fail): {', '.join(empty)}."]
    sh = (s["recall"].get("recall_bug") or {}).get("shared_report_messages")
    L += ["", f"Disclosure: {sh} first-report messages carrying more than one bug in the extraction are in the bug recall "
          "packets; those bugs share one identity key in the comparison.",
          "Disclosure: later sessions that refer to a bug only by anaphora (\"this one\"), with no title, customer "
          "company or PR link, are not in the bug precision packets."]
    L += ["", "## Prompt hashes", ""] + [f"- {k}: {h}" for k, h in s["prompt_hashes"].items() if k != "version"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- gates
NOT_ESTIMABLE = "not estimable (empty dev pool)"


def _gate_rows(summary: dict) -> list[dict]:
    """One row per category. A category whose pool in the manifest is empty (available == 0) is not estimable: status says
    so, 0/0, and it neither passes nor fails the round (pass is True so it never blocks). Any other category with a zero
    adjudicated denominator still fails."""
    sizes = summary.get("sizes") or {}
    rows = []
    for kind, thr, cat in [(k, GATE_PRECISION, "precision") for k in PRECISION_QUOTAS] + [(k, GATE_RECALL, "recall") for k in RECALL_QUOTAS]:
        name = f"{cat}:{kind}"
        if sizes.get(kind, {}).get("available") == 0:
            rows.append({"name": name, "numerator": 0, "denominator": 0, "threshold": thr, "pass": True, "status": NOT_ESTIMABLE})
            continue
        c = summary[cat].get(kind)
        if cat == "precision":
            n, d = (c["correct"], c["n"]) if c else (0, 0)
            ok = bool(c and d and c["precision"] is not None and c["precision"] >= thr and not c["pending"])
        else:
            n, d = (c["matched"], c["truth"]) if c else (0, 0)
            ok = bool(c and d and not c["pending"] and c["recall"] >= thr)
        rows.append({"name": name, "numerator": n, "denominator": d, "threshold": thr, "pass": ok, "status": "evaluated"})
    return rows


def gates(summary: dict) -> list[tuple[str, bool, str]]:
    out = [("complete", summary["complete"], f"{summary['pending_third']} third passes pending, "
                                            f"{len(summary.get('integrity_problems') or [])} integrity problems")]
    return out + [(g["name"], g["pass"], g["status"] if g["status"] != "evaluated" else
                   f"{g['numerator']}/{g['denominator']}" if g["denominator"] else "no packets")
                  for g in _gate_rows(summary)]


def current_problems(rd: Path, summary: dict) -> list[str]:
    """Completeness recomputed from the files now, never from the cached summary: packets against the manifest, auditor
    metadata, pending third passes, and the verdict files against the hashes the summary recorded."""
    n, root = int(rd.name.split("-")[1]), rd.parent
    out = integrity_problems(n, root)
    out += [f"third pass missing: {i}" for i in pending_third(n, root)]
    now = _verdict_hashes(rd)
    out += [f"verdict files changed since collect: {w}" for w in now if now[w] != (summary.get("verdict_hashes") or {}).get(w)]
    return [x + " (re-run collect)" for x in out]


def gates_report(round_dir: Path) -> dict:
    """The one entry point for the acceptance report: gate rows, completeness and the disclosure fields of a round."""
    round_dir = Path(round_dir)
    sf = round_dir / "summary.json"
    if not sf.exists():
        return {"complete": False, "gates": [], "pass": False, "auditors": None, "prompt_hashes": None,
                "sample_hash": None, "summary_sha256": None, "report_md": str(round_dir / "report.md")}
    raw = sf.read_text(encoding="utf-8")
    summary = json.loads(raw)
    rows = _gate_rows(summary)
    reasons = current_problems(round_dir, summary)
    complete = not reasons
    return {"complete": complete, "reasons": reasons, "gates": rows, "pass": complete and all(g["pass"] for g in rows),
            "auditors": summary.get("auditors"), "prompt_hashes": summary.get("prompt_hashes"),
            "sample_hash": summary.get("sample_hash"), "summary_sha256": sha256_text(raw),
            "report_md": str(round_dir / "report.md")}


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
        try:
            m = write_round(packets, sizes, n, sha, root)
        except FileExistsError as e:
            print(e)
            return 2
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
        rep = gates_report(round_dir(n, root))
        print("audit gates only; the dev score gate is in the acceptance report")
        if not (round_dir(n, root) / "summary.json").exists():
            print("no summary: run collect first")
            return 1
        print(f"{'PASS' if rep['complete'] else 'FAIL'} complete")
        for g in rep["gates"]:
            print(f"{'PASS' if g['pass'] else 'FAIL'} {g['name']} {g['numerator']}/{g['denominator']} (threshold {g['threshold']})"
                  + ("" if g["status"] == "evaluated" else f" [{g['status']}]"))
        return 0 if rep["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
