"""Typed adapters over one bundle product. Task 3 extraction consumes these.

Unknown or malformed values are kept as found and counted in `ProductRecords.anomalies`.
Nothing is silently fixed.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from inherit.data.bundle import load_bundle, load_metadata

MENTION_RE = re.compile(r"@(eid_[A-Za-z0-9_]+)")
# Bot IDs seen in the real data (only slack_admin_bot occurs; the other non-directory IDs are malformed eids).
KNOWN_BOTS = frozenset({"slack_admin_bot"})
BUG_CHANNEL_RE = re.compile(r"^bug-(?P<owner>.+)-(?P<product>[^-]+)$")
SPEAKER_RE = re.compile(r"^([^:\n]{1,60}): (.*)$")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


def parse_mentions(text: str) -> tuple[str, ...]:
    """Employee IDs after '@', in order. Trailing punctuation is never part of the ID."""
    return tuple(MENTION_RE.findall(text or ""))


def parse_bug_channel(channel_name: str, product: str) -> str | None:
    """Owner part of `bug-<owner>-<Product>`, else None. The owner is not a bug's assignee."""
    m = BUG_CHANNEL_RE.match(channel_name or "")
    if not m or m.group("product") != product:
        return None
    return m.group("owner")


@dataclass(frozen=True)
class SlackMessage:
    id: str
    channel_id: str
    channel_name: str
    user_id: str
    ts: str
    text: str
    mentions: tuple[str, ...]
    is_bot: bool
    in_directory: bool
    bug_channel_owner: str | None


@dataclass(frozen=True)
class Document:
    id: str
    author: str
    type: str
    link: str
    date: str
    content: str
    feedback: str | None


@dataclass(frozen=True)
class Segment:
    """One transcript line. `line_no` is 1-based in the raw transcript text; id is `<meeting id>#<line_no>`."""

    id: str
    meeting_id: str
    line_no: int
    speaker: str | None
    text: str


@dataclass(frozen=True)
class Meeting:
    id: str
    date: str
    document_type: str
    participants: tuple[str, ...]
    attendees: tuple[str, ...]
    segments: tuple[Segment, ...]


@dataclass(frozen=True)
class MeetingChat:
    id: str
    text: str
    date: str | None
    mentions: tuple[str, ...]
    meeting_id: str | None
    link_method: str | None  # "id", "date" or None


@dataclass(frozen=True)
class Review:
    position: int
    state: str
    login: str
    comment: str
    submitted_at: str
    in_directory: bool


@dataclass(frozen=True)
class PullRequest:
    id: str
    link: str
    number: str
    title: str
    summary: str
    author: str
    in_directory: bool
    merged: bool | None
    mergeable: bool | None
    merged_raw: str
    mergeable_raw: str
    state: str
    created_at: str
    reviews: tuple[Review, ...]


@dataclass(frozen=True)
class UrlRecord:
    """URL records merged by id. `positions` are the indexes of the source records in the product's urls list."""

    id: str
    link: str
    links: tuple[str, ...]
    descriptions: tuple[str, ...]
    positions: tuple[int, ...]


@dataclass
class ProductRecords:
    product: str
    messages: list[SlackMessage] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    meetings: list[Meeting] = field(default_factory=list)
    chats: list[MeetingChat] = field(default_factory=list)
    prs: list[PullRequest] = field(default_factory=list)
    urls: list[UrlRecord] = field(default_factory=list)
    employees: dict[str, dict] = field(default_factory=dict)
    customers: dict[str, dict] = field(default_factory=dict)
    anomalies: Counter = field(default_factory=Counter)


def _bool_str(value, anomalies: Counter, counter_key: str) -> bool | None:
    # The data stores "True"/"False" strings. bool("False") is True, so never call bool().
    if value == "True":
        return True
    if value == "False":
        return False
    anomalies[counter_key] += 1
    return None


def _slack(data: dict, product: str, employees: dict, an: Counter) -> list[SlackMessage]:
    out = []
    for r in data["slack"]:
        ch, msg = r["Channel"], r["Message"]["User"]
        uid = msg["userId"]
        is_bot = uid in KNOWN_BOTS
        in_directory = uid in employees
        if not in_directory and not is_bot:
            an["slack_user_unknown"] += 1
        if r.get("id") != msg["utterranceID"]:
            an["slack_id_mismatch"] += 1
        out.append(SlackMessage(
            id=msg["utterranceID"], channel_id=ch["channelID"], channel_name=ch["name"], user_id=uid,
            ts=msg["timestamp"], text=msg["text"], mentions=parse_mentions(msg["text"]), is_bot=is_bot,
            in_directory=in_directory,
            bug_channel_owner=parse_bug_channel(ch["name"], product),
        ))
    return out


def _documents(data: dict) -> list[Document]:
    return [
        Document(id=d["id"], author=d["author"], type=d["type"], link=d["document_link"], date=d["date"],
                 content=d["content"], feedback=d.get("feedback"))
        for d in data["documents"]
    ]


def _meeting(m: dict, an: Counter) -> Meeting:
    lines = m["transcript"].split("\n")
    attendees: tuple[str, ...] = ()
    if "Transcript" in lines:
        marker = lines.index("Transcript")
        if lines and lines[0] == "Attendees" and marker >= 2:
            attendees = tuple(n.strip() for n in ", ".join(lines[1:marker]).split(",") if n.strip())
    else:
        an["meeting_no_transcript_marker"] += 1
        marker = -1
    segments = []
    for i in range(marker + 1, len(lines)):
        if not lines[i].strip():
            continue
        sm = SPEAKER_RE.match(lines[i])
        if sm:
            speaker, text = sm.group(1), sm.group(2)
        else:
            an["segment_no_speaker"] += 1
            speaker, text = None, lines[i]
        segments.append(Segment(id=f"{m['id']}#{i + 1}", meeting_id=m["id"], line_no=i + 1, speaker=speaker, text=text))
    return Meeting(id=m["id"], date=m["date"], document_type=m["document_type"],
                   participants=tuple(m["participants"]), attendees=attendees, segments=tuple(segments))


def _chats(data: dict, meetings: list[Meeting], an: Counter) -> list[MeetingChat]:
    by_id = {m.id for m in meetings}
    by_date: dict[str, list[str]] = {}
    for m in meetings:
        by_date.setdefault(m.date, []).append(m.id)
    out = []
    for c in data["meeting_chats"]:
        first = c["text"].split("\n", 1)[0].strip()
        date = first if ISO_RE.match(first) else None
        if date is None:
            an["chat_no_date"] += 1
        meeting_id = method = None
        stem = c["id"][:-5] if c["id"].endswith("_chat") else None
        if stem is not None and stem in by_id:
            meeting_id, method = stem, "id"
        elif date is not None and len(by_date.get(date, [])) == 1:
            meeting_id, method = by_date[date][0], "date"
        elif date is not None and len(by_date.get(date, [])) > 1:
            an["chat_link_ambiguous"] += 1
        if meeting_id is None:
            an["chat_unlinked"] += 1
        out.append(MeetingChat(id=c["id"], text=c["text"], date=date, mentions=parse_mentions(c["text"]),
                               meeting_id=meeting_id, link_method=method))
    return out


def _prs(data: dict, employees: dict, an: Counter) -> list[PullRequest]:
    out = []
    for p in data["prs"]:
        login = p["user"]["login"]
        reviews = tuple(
            Review(position=i, state=r["state"], login=r["user"]["login"], comment=r.get("comment", ""),
                   submitted_at=r.get("submitted_at", ""), in_directory=r["user"]["login"] in employees)
            for i, r in enumerate(p["reviews"])
        )
        if login not in employees:
            an["pr_author_not_in_directory"] += 1
        out.append(PullRequest(
            id=p["id"], link=p["link"], number=p["number"], title=p["title"], summary=p["summary"], author=login,
            in_directory=login in employees,
            merged=_bool_str(p["merged"], an, "pr_merged_unparsed"),
            mergeable=_bool_str(p["mergeable"], an, "pr_mergeable_unparsed"),
            merged_raw=p["merged"], mergeable_raw=p["mergeable"],
            state=p["state"], created_at=p["created_at"], reviews=reviews,
        ))
    return out


def _urls(data: dict, an: Counter) -> list[UrlRecord]:
    order: list[str] = []
    groups: dict[str, list[tuple[int, dict]]] = {}
    for pos, u in enumerate(data["urls"]):
        if u["id"] not in groups:
            order.append(u["id"])
            groups[u["id"]] = []
        groups[u["id"]].append((pos, u))
    out = []
    for uid in order:
        g = groups[uid]
        if len(g) > 1:
            an["url_duplicate_id"] += 1
        links = tuple(dict.fromkeys(u["link"] for _, u in g))
        if len(links) > 1:
            an["url_link_conflict"] += 1
        out.append(UrlRecord(id=uid, link=links[0], links=links,
                             descriptions=tuple(u["description"] for _, u in g), positions=tuple(p for p, _ in g)))
    return out


def build_records(data: dict, employees: dict, customers: list, name: str) -> ProductRecords:
    an: Counter = Counter()
    meetings = [_meeting(m, an) for m in data["meeting_transcripts"]]
    return ProductRecords(
        product=name,
        messages=_slack(data, name, employees, an),
        documents=_documents(data),
        meetings=meetings,
        chats=_chats(data, meetings, an),
        prs=_prs(data, employees, an),
        urls=_urls(data, an),
        employees=employees,
        customers={c["id"]: c for c in customers},
        anomalies=an,
    )


def load_records(bundle: Path, product: str) -> ProductRecords:
    employees, customers = load_metadata(bundle)
    return build_records(load_bundle(bundle, product), employees, customers, product)
