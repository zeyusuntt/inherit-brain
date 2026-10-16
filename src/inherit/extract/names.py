"""Layer 1b: resolve the speaker names in meeting transcripts and meeting chats to employee IDs.

HERB names speakers only by name and most names are shared by several employees, so a wrong pick would
quietly corrupt every person answer. The rule (frozen in the spec): candidates are directory people with that
exact name who are in the meeting, active in the product's Slack within +-3 days, or @-mentioned there. One
candidate resolves without scoring. Several are scored 3 x role cue + 2 x related post + 1 x mentioned (each
term 0 or 1). A unique top score wins; a tie keeps every tied candidate and is flagged ambiguous. Never guess.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from inherit.data.records import SPEAKER_RE, Meeting, ProductRecords, parse_mentions
from inherit.extract.sessions import sessions
from inherit.facts import Edge, Facts, Node
from inherit.ontology import check_edge

EXTRACTOR = "names/v2"
WEIGHTS = {"role_cue": 3, "related_post": 2, "mentioned": 1}
WINDOW = timedelta(days=3)
ROLE_CUES_VERSION = "v2"

# Role cue table. A cue must say the speaker's OWN role, not a topic that every role talks about
# ("the QA team", "engineering team", "testing" alone are not cues). Three shapes:
#   "as|speaking as [a|an|the|our] <role>" followed by a first-person word ("as QA I checked", "as the product
#   manager, we ..."), "I am [a|an|the] <role>" and "in my role as <role>" use ROLE_NOUNS. "As the product
#   manager suggested" is a reference to someone else and is not a cue;
#   "from [a|an|my|our] <x> perspective|standpoint|angle|viewpoint" uses ROLE_NOUNS plus PERSPECTIVE_EXTRA;
#   ARCHITECT_OFFER is the one fixed phrase taken from the dev transcripts.
ROLE_NOUNS: dict[str, tuple[str, ...]] = {
    "Software Engineer": ("software engineer", "developer", "engineer"),
    "QA Specialist": ("qa specialist", "qa engineer", "quality assurance specialist", "quality assurance", "qa",
                      "tester"),
    "Engineering Lead": ("engineering lead", "tech lead", "technical lead", "lead engineer", "team lead"),
    "Product Manager": ("product manager", "product owner", "pm"),
    "Technical Architect": ("technical architect", "solution architect", "architect"),
    "VP of Engineering": ("vp of engineering", "vp engineering", "vice president of engineering"),
    "UX Researcher": ("ux researcher", "user researcher", "ux research", "user research", "ux"),
    "Marketing Research Analyst": ("marketing research analyst", "market research analyst", "research analyst"),
    "Chief Product Officer": ("chief product officer", "cpo"),
    "Marketing Manager": ("marketing manager",),
}
PERSPECTIVE_EXTRA: dict[str, tuple[str, ...]] = {
    "Product Manager": ("product",),
    "Technical Architect": ("architecture",),
    "UX Researcher": ("user experience",),
    "Marketing Research Analyst": ("market research",),
    "Marketing Manager": ("marketing",),
}
ARCHITECT_OFFER = (r"\bi(?:'ll| will| can) (?:be available (?:for|to provide)|provide|offer) (?:any |some )?"
                   r"(?:architectural|architecture) (?:guidance|input|advice|insights?)")


def _alt(nouns) -> str:
    return "|".join(re.escape(n) for n in sorted(nouns, key=len, reverse=True))


def _compile() -> list[tuple[str, re.Pattern]]:
    out = []
    for role, nouns in ROLE_NOUNS.items():
        own = _alt(nouns)
        persp = _alt(nouns + PERSPECTIVE_EXTRA.get(role, ()))
        parts = [
            rf"\b(?:as|speaking as) (?:a |an |the |our )?(?:{own})\b,? (?:i|i'm|i'll|i've|i'd|my|me|we|we're|we'll|let me)\b",
            rf"\bin my (?:role|capacity) as (?:a |an |the )?(?:{own})\b",
            rf"\bi(?:'m| am) (?:a |an |the |our )(?:{own})\b(?! team)(?!'s)",
            rf"\bfrom (?:a |an |my |our )(?:{persp})(?: team's)? (?:perspective|standpoint|angle|viewpoint|point of view)\b",
        ]
        if role == "Technical Architect":
            parts.append(ARCHITECT_OFFER)
        out.append((role, re.compile("|".join(parts), re.I)))
    return out


_CUES = _compile()
URL_RE = re.compile(r"https?://[^\s<>|\]]+")


def role_cue_hits(text: str) -> dict[str, list[str]]:
    """Role -> matched phrases (lowercased) found in the text. Empty dict when there is no cue."""
    text = text.replace("’", "'")
    out: dict[str, list[str]] = {}
    for role, pat in _CUES:
        hits = [m.group(0).lower() for m in pat.finditer(text)]
        if hits:
            out[role] = hits
    return out


def norm(name: str) -> str:
    return " ".join(name.split())


def _links(text: str) -> set[str]:
    return {m.rstrip(".,;:!?)'\"") for m in URL_RE.findall(text)}


@dataclass(frozen=True)
class Resolution:
    ids: list[str]
    ambiguous: bool
    status: Literal["resolved", "ambiguous", "unresolved"]
    evidence: list[str]
    scores: dict[str, int] = field(default_factory=dict)


@dataclass
class NameContext:
    employees: dict[str, dict]
    by_name: dict[str, list[str]]  # normalised directory name -> eids, sorted
    posts: dict[str, list[tuple[datetime, tuple[str, int]]]]  # eid -> (time, session key) of each post
    session_text: dict[tuple[str, int], str]  # session key -> lowercased text of the whole session
    mentions: dict[str, set[str]]  # meeting id -> eids @-mentioned in its chat or transcript
    doc_terms: dict[str, list[str]]  # document type -> lowercased links of documents of that type
    lines: dict[tuple[str, str], list[str]]  # (meeting id, name) -> that speaker's transcript and chat lines

    @classmethod
    def build(cls, rec: ProductRecords) -> NameContext:
        by_name: dict[str, list[str]] = defaultdict(list)
        for eid in sorted(rec.employees):
            by_name[norm(rec.employees[eid]["name"])].append(eid)

        posts: dict[str, list] = defaultdict(list)
        session_text: dict[tuple[str, int], str] = {}
        by_channel: dict[str, list] = defaultdict(list)
        for m in rec.messages:
            by_channel[m.channel_id].append(m)
        for ch, msgs in by_channel.items():
            for i, sess in enumerate(sessions(msgs)):
                session_text[(ch, i)] = "\n".join(m.text for m in sess).lower()
                for m in sess:
                    if m.in_directory and not m.is_bot:
                        posts[m.user_id].append((datetime.fromisoformat(m.ts), (ch, i)))

        mentions: dict[str, set[str]] = defaultdict(set)
        lines: dict[tuple[str, str], list[str]] = defaultdict(list)
        for m in rec.meetings:
            for s in m.segments:
                mentions[m.id].update(parse_mentions(s.text))
                if s.speaker is not None:
                    lines[(m.id, norm(s.speaker))].append(s.text)
        for c in rec.chats:
            if c.meeting_id is None:
                continue
            mentions[c.meeting_id].update(c.mentions)
            for sp, tx in chat_lines(c.text):
                lines[(c.meeting_id, norm(sp))].append(tx)

        doc_terms: dict[str, list[str]] = defaultdict(list)
        for d in rec.documents:
            doc_terms[d.type].append(d.link.lower())
        return cls(rec.employees, dict(by_name), dict(posts), session_text, dict(mentions), dict(doc_terms), dict(lines))


def chat_lines(text: str) -> list[tuple[str, str]]:
    """(speaker, text) for each `Name: text` line after the date line of a meeting chat."""
    out = []
    for line in text.split("\n")[1:]:
        m = SPEAKER_RE.match(line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _evidence_role(role: str, hits: list[str]) -> str:
    return f"role cue: {role} ({hits[0]!r})"


def resolve_speaker(name: str, meeting: Meeting, context: NameContext) -> Resolution:
    name = norm(name)
    mdt = datetime.fromisoformat(meeting.date)
    mentioned = context.mentions.get(meeting.id, set())
    same_name = context.by_name.get(name, [])
    cands = [eid for eid in same_name if eid in meeting.participants]  # SPEC 5.3: co-attendees come first
    if not cands:  # nobody of that name attended: fall back to people active nearby or mentioned
        for eid in same_name:
            near = any(abs(t - mdt) <= WINDOW for t, _ in context.posts.get(eid, []))
            if near or eid in mentioned:
                cands.append(eid)
    cue_roles = role_cue_hits("\n".join(context.lines.get((meeting.id, name), [])))
    if cands and not any(context.employees[e].get("role") in cue_roles for e in cands):
        # a same-name person who did not attend but is active nearby or mentioned, and whose role the speaker's own
        # words match, reopens the name: attendance alone is no longer enough (a related post or mention does not)
        late = [e for e in same_name if e not in cands and context.employees[e].get("role") in cue_roles
                and (e in mentioned or any(abs(t - mdt) <= WINDOW for t, _ in context.posts.get(e, [])))]
        if late:
            ids = [*cands, *late]
            ev = [f"{e} role cue: {context.employees[e].get('role')} ({cue_roles[context.employees[e].get('role')][0]!r}), "
                  "not an attendee" for e in late]
            return Resolution(ids, True, "ambiguous", ev + [f"attendees without the cue: {', '.join(cands)}"],
                              {e: (WEIGHTS["role_cue"] if e in late else 0) for e in ids})
    if not cands:
        return Resolution([], False, "unresolved", [f"no candidate for {name!r}"], {})
    if len(cands) == 1:
        return Resolution(cands, False, "resolved", ["single candidate"], {})

    terms = [meeting.document_type.lower()] if meeting.document_type else []
    terms += context.doc_terms.get(meeting.document_type, [])
    scores: dict[str, int] = {}
    evidence: list[str] = []
    for eid in cands:
        role = context.employees[eid].get("role")
        s = 0
        if role in cue_roles:
            s += WEIGHTS["role_cue"]
            evidence.append(f"{eid} {_evidence_role(role, cue_roles[role])}")
        keys = {k for t, k in context.posts.get(eid, []) if abs(t - mdt) <= WINDOW}
        if any(term in context.session_text[k] for k in keys for term in terms):
            s += WEIGHTS["related_post"]
            evidence.append(f"{eid} related post within 3 days")
        if eid in mentioned:
            s += WEIGHTS["mentioned"]
            evidence.append(f"{eid} mentioned in the meeting")
        scores[eid] = s
    top = max(scores.values())
    winners = [e for e in cands if scores[e] == top]
    if len(winners) == 1:
        return Resolution(winners, False, "resolved", evidence, scores)
    evidence.append(f"tie at score {top} among {len(winners)} candidates")
    return Resolution(winners, True, "ambiguous", evidence, scores)


def resolve_meeting_names(rec: ProductRecords) -> tuple[Facts, list[dict]]:
    ctx = NameContext.build(rec)
    by_id = {m.id: m for m in rec.meetings}
    seg_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    chat_n: dict[tuple[str, str], int] = defaultdict(int)
    for m in rec.meetings:
        for s in m.segments:
            if s.speaker is not None:
                seg_ids[(m.id, norm(s.speaker))].append(s.id)
    chat_by_meeting: dict[str, list] = defaultdict(list)
    for c in rec.chats:
        if c.meeting_id is not None:
            chat_by_meeting[c.meeting_id].append(c)
            for sp, _ in chat_lines(c.text):
                chat_n[(c.meeting_id, norm(sp))] += 1

    keys = sorted(set(seg_ids) | set(chat_n))
    rows: list[dict] = []
    res: dict[tuple[str, str], Resolution] = {}
    for mid, name in keys:
        r = resolve_speaker(name, by_id[mid], ctx)
        res[(mid, name)] = r
        basis = ("none" if r.status == "unresolved" else "single" if r.evidence == ["single candidate"]
                 else "score" if r.status == "resolved" else "tie")
        cands = sorted(r.scores) if r.scores else list(r.ids)
        hits = role_cue_hits("\n".join(ctx.lines.get((mid, name), [])))
        rows.append({"meeting_id": mid, "name": name, "status": r.status, "basis": basis, "ids": r.ids,
                     "candidates": cands, "scores": r.scores, "evidence": r.evidence,
                     "lines": len(seg_ids.get((mid, name), [])) + chat_n.get((mid, name), 0),
                     "chat_lines": chat_n.get((mid, name), 0),
                     "cue_hits": hits, "cue_version": ROLE_CUES_VERSION})

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []

    def person(eid: str) -> str:
        e = rec.employees[eid]
        nodes[f"Person:{eid}"] = Node("Person", f"Person:{eid}", {
            "eid": eid, "name": e["name"], "role": e["role"], "in_directory": True})
        return f"Person:{eid}"

    def edge(type_, src, dst, ids, key="", **props):
        e = Edge(type_, src, dst, props, tuple(ids), EXTRACTOR, key)
        check_edge(e)
        edges.append(e)

    for (mid, name), r in res.items():
        if r.status == "unresolved" or not seg_ids.get((mid, name)):
            continue
        for eid in r.ids:
            edge("SPOKE_IN", person(eid), f"Meeting:{mid}", seg_ids[(mid, name)], speaker_name=name,
                 ambiguous=r.ambiguous, score=r.scores.get(eid, 0), n_candidates=len(r.scores) or 1)

    doc_by_link = {d.link: d.id for d in rec.documents}
    pr_by_link = {p.link: f"PR:{p.link}" for p in rec.prs}
    url_links = {link for u in rec.urls for link in u.links}
    for mid, chats in chat_by_meeting.items():
        for c in chats:
            for sp, tx in chat_lines(c.text):
                r = res[(mid, norm(sp))]
                if r.status == "unresolved":
                    continue
                for link in sorted(_links(tx)):
                    targets = []
                    if link in doc_by_link:
                        targets.append(f"Document:{doc_by_link[link]}")
                    if link in pr_by_link:
                        targets.append(pr_by_link[link])
                    if link in url_links:
                        targets.append(f"Url:{link}")
                    for dst in targets:
                        for eid in r.ids:
                            edge("SHARED", person(eid), dst, [c.id], c.id, speaker_name=norm(sp), ambiguous=r.ambiguous)

    return Facts([], []).merge(Facts(list(nodes.values()), edges)), rows


def lookup(rows: list[dict], meeting_id: str, name: str) -> Resolution:
    """The computed resolution for (meeting, name); unresolved when that pair was never a speaker."""
    name = norm(name)
    for r in rows:
        if r["meeting_id"] == meeting_id and r["name"] == name:
            return Resolution(list(r["ids"]), r["status"] == "ambiguous", r["status"], list(r["evidence"]),
                              dict(r["scores"]))
    return Resolution([], False, "unresolved", [f"no speaker {name!r} in {meeting_id}"], {})
