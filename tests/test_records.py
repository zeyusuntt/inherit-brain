from pathlib import Path

import pytest

from inherit.data.bundle import load_bundle
from inherit.data.records import (
    build_records,
    load_records,
    parse_bug_channel,
    parse_mentions,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "bundle"


@pytest.fixture
def rec(mini_bundle):
    return load_records(mini_bundle, "mini")


def test_parse_mentions_strips_trailing_punctuation():
    t = "Thanks @eid_a1000001, @eid_f6000006, and @eid_c3000003. Also (@eid_d4000004)! @channel"
    assert parse_mentions(t) == ("eid_a1000001", "eid_f6000006", "eid_c3000003", "eid_d4000004")
    assert parse_mentions("no mentions") == ()


def test_parse_bug_channel():
    assert parse_bug_channel("bug-alicechen-mini", "mini") == "alicechen"
    assert parse_bug_channel("bug-ianmiller-ActionGenie", "ActionGenie") == "ianmiller"
    assert parse_bug_channel("planning-mini", "mini") is None
    assert parse_bug_channel("develop-alicechen-mini", "mini") is None


def test_slack_messages(rec):
    assert len(rec.messages) == 18
    by_id = {m.id: m for m in rec.messages}
    m = by_id["20260610-0-a0002"]
    assert m.channel_id == "ch-mini-plan" and m.channel_name == "planning-mini"
    assert m.user_id == "eid_e5000005" and m.ts == "2026-06-10T09:00:00"
    assert m.mentions == ("eid_a1000001", "eid_f6000006")
    assert not m.is_bot and m.bug_channel_owner is None
    bot = by_id["20260609-0-a0001"]
    assert bot.is_bot and bot.mentions == ("eid_e5000005",)
    assert by_id["20260616-1-b0003"].is_bot
    bug = by_id["20260617-0-c0001"]
    assert bug.bug_channel_owner == "alicechen"
    assert not by_id["20260610-0-a0002"].is_bot
    assert "slack_user_not_employee_non_bot" not in rec.anomalies


def test_documents(rec):
    docs = {d.id: d for d in rec.documents}
    d = docs["mini_design_spec"]
    assert d.author == "eid_e5000005" and d.type == "Design Spec" and d.date.startswith("2026-06-10")
    assert d.link.endswith("/mini_design_spec") and d.content and d.feedback.startswith("- Add numbers")
    assert docs["mini_release_notes"].feedback is None


def test_meetings_segments_and_attendees(rec):
    m = {x.id: x for x in rec.meetings}["mini_design_1"]
    assert m.date == "2026-06-13T17:07:00" and m.document_type == "Design Spec"
    assert m.participants == ("eid_e5000005", "eid_c3000003", "eid_f6000006", "eid_d4000004")
    assert m.attendees == ("Dana Wu", "Ian Miller", "Eve Park", "Ian Miller")
    assert [s.id for s in m.segments] == [f"mini_design_1#{n}" for n in (4, 5, 6, 7, 8, 9)]
    s5 = m.segments[1]
    assert s5.speaker == "Ian Miller" and s5.text.startswith("As QA I checked")
    assert s5.meeting_id == "mini_design_1" and s5.line_no == 5
    assert m.segments[3].speaker == "Bob Ray"
    # Bob Ray is a speaker but not a participant (participants are IDs; names resolve via directory)
    names = {rec.employees[p]["name"] for p in m.participants}
    assert "Bob Ray" not in names
    assert {s.speaker for s in m.segments} - names == {"Bob Ray"}


def test_segment_ids_are_raw_line_numbers(mini_bundle):
    raw = load_bundle(mini_bundle, "mini")["meeting_transcripts"][0]["transcript"].split("\n")
    rec = load_records(mini_bundle, "mini")
    for s in rec.meetings[0].segments:
        assert raw[s.line_no - 1] == f"{s.speaker}: {s.text}"


def test_meeting_chats(rec):
    c = {x.id: x for x in rec.chats}
    a = c["mini_design_1_chat"]
    assert a.date == "2026-06-13T17:07:00" and a.mentions == ("eid_a1000001",)
    assert a.meeting_id == "mini_design_1" and a.link_method == "id"
    b = c["retro_notes_chat"]
    assert b.meeting_id == "mini_review_1" and b.link_method == "date"
    o = c["orphan_chat"]
    assert o.meeting_id is None and o.link_method is None
    assert rec.anomalies["chat_unlinked"] == 1


def test_prs(rec):
    prs = {p.number: p for p in rec.prs}
    p = prs["101"]
    assert p.merged is True and p.mergeable is True and p.state == "closed"
    assert p.author == "eid_b2000002" and p.in_directory
    assert [(r.position, r.state, r.login, r.in_directory) for r in p.reviews] == [
        (0, "APPROVED", "eid_a1000001", True), (1, "COMMENTED", "eid_f6000006", True)]
    assert p.reviews[0].comment == "Looks good." and p.reviews[0].submitted_at
    q = prs["102"]
    assert q.merged is False and q.state == "closed" and q.reviews == ()
    r = prs["103"]
    assert r.author == "EMP_555000111" and not r.in_directory and r.merged is False and r.state == "open"
    assert not r.reviews[0].in_directory
    assert p.link.endswith("/pull/101") and p.title and p.summary and p.created_at


def test_pr_string_booleans_never_bool(mini_bundle):
    d = load_bundle(mini_bundle, "mini")
    d["prs"][0]["merged"] = "False"
    d["prs"][1]["merged"] = "maybe"
    d["prs"][2]["mergeable"] = ""
    emp = {"eid_b2000002": {}}
    rec = build_records(d, emp, [], name="mini")
    prs = {p.number: p for p in rec.prs}
    assert prs["101"].merged is False
    assert prs["102"].merged is None
    assert prs["103"].mergeable is None
    assert rec.anomalies["pr_merged_unparsed"] == 1 and rec.anomalies["pr_mergeable_unparsed"] == 1


def test_urls_merged_by_id(rec):
    assert len(rec.urls) == 2
    u = {x.id: x for x in rec.urls}["docs_example_com_export"]
    assert u.positions == (0, 1)
    assert u.descriptions == ("Guide to streaming exports.", "Vendor reference for chunked CSV export.")
    assert u.link == "https://docs.example.com/export" and u.links == (u.link,)
    assert rec.anomalies["url_duplicate_id"] == 1
    assert {x.id: x for x in rec.urls}["blog_example_com_retries"].positions == (2,)


def test_metadata_indexed(rec):
    assert rec.employees["eid_c3000003"]["name"] == "Ian Miller"
    assert rec.employees["eid_d4000004"]["role"] == "QA Specialist"
    assert rec.customers["CUST-0001"]["company"] == "TechCorp"


def test_malformed_values_kept_and_counted(mini_bundle):
    d = load_bundle(mini_bundle, "mini")
    d["slack"][1]["Message"]["User"]["userId"] = "eid_unknown1"
    d["meeting_transcripts"][1]["transcript"] = "Attendees\nA B\nTranscript\nno speaker here"
    rec = build_records(d, {"eid_e5000005": {}}, [], name="mini")
    assert rec.anomalies["slack_user_not_employee_non_bot"] >= 1
    assert rec.anomalies["segment_no_speaker"] == 1
    seg = [m for m in rec.meetings if m.id == "mini_review_1"][0].segments[0]
    assert seg.speaker is None and seg.text == "no speaker here"


@pytest.mark.skipif(not (BUNDLE / "manifest.json").exists(), reason="data/bundle absent")
@pytest.mark.parametrize("product", ["ActionGenie", "TrendForce", "LeadForce"])
def test_real_products_field_presence_and_counts(product):
    raw = load_bundle(BUNDLE, product)
    rec = load_records(BUNDLE, product)
    assert len(rec.messages) == len(raw["slack"])
    assert len(rec.documents) == len(raw["documents"])
    assert len(rec.meetings) == len(raw["meeting_transcripts"])
    assert len(rec.chats) == len(raw["meeting_chats"])
    assert len(rec.prs) == len(raw["prs"])
    assert len(rec.urls) == len({u["id"] for u in raw["urls"]})
    assert sum(len(u.positions) for u in rec.urls) == len(raw["urls"])
    assert all(m.segments for m in rec.meetings)
    assert all(p.merged is not None and p.mergeable is not None for p in rec.prs)
