import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from inherit.data.bundle import load_bundle, load_metadata
from inherit.data.records import Document, Meeting, MeetingChat, ProductRecords, Segment, SlackMessage, \
    UrlRecord, parse_mentions
from inherit.extract.names import (ROLE_CUES_VERSION, WEIGHTS, NameContext, Resolution, lookup, resolve_meeting_names,
                                   resolve_speaker, role_cue_hits)
from inherit.ontology import check_edge

ROOT = Path(__file__).resolve().parents[1]
DOC_LINK = "https://sf-internal.slack.com/archives/docs/spec1"


def emp(eid, name, role):
    return {"employee_id": eid, "name": name, "role": role}


EMPLOYEES = {
    "eid_qa": emp("eid_qa", "Ian Miller", "QA Specialist"),
    "eid_pm": emp("eid_pm", "Ian Miller", "Product Manager"),
    "eid_solo": emp("eid_solo", "Dana Wu", "Marketing Manager"),
    "eid_other": emp("eid_other", "Zed Active", "Software Engineer"),
}


def msg(i, user, ts, text="hello", channel="c1"):
    return SlackMessage(id=i, channel_id=channel, channel_name=channel, user_id=user, ts=ts, text=text,
                        mentions=parse_mentions(text), is_bot=False, in_directory=user in EMPLOYEES,
                        bug_channel_owner=None)


def meeting(lines, participants=("eid_qa", "eid_pm", "eid_solo"), date="2026-06-13T10:00:00", mid="m1"):
    segs = tuple(Segment(f"{mid}#{i + 3}", mid, i + 3, sp, tx) for i, (sp, tx) in enumerate(lines))
    return Meeting(mid, date, "Design Spec", tuple(participants), (), segs)


def records(meetings, messages=(), chats=(), employees=None):
    doc = Document("spec1", "eid_solo", "Design Spec", DOC_LINK, "2026-06-01T00:00:00", "", None)
    return ProductRecords(product="p", messages=list(messages), documents=[doc], meetings=list(meetings),
                          chats=list(chats), prs=[], urls=[], employees=employees or EMPLOYEES)


def resolve(rec, name="Ian Miller", mid="m1"):
    ctx = NameContext.build(rec)
    m = next(x for x in rec.meetings if x.id == mid)
    return resolve_speaker(name, m, ctx)


QA_LINES = [("Ian Miller", "As QA I checked the export path.")]


def test_weights_and_version_frozen():
    assert WEIGHTS == {"role_cue": 3, "related_post": 2, "mentioned": 1}
    assert ROLE_CUES_VERSION == "v1"


def test_role_cue_picks_qa_over_pm():
    r = resolve(records([meeting(QA_LINES)]))
    assert r.status == "resolved" and not r.ambiguous
    assert r.ids == ["eid_qa"] and r.scores == {"eid_qa": 3, "eid_pm": 0}
    assert any("role cue" in e for e in r.evidence)


def test_no_cues_is_ambiguous_with_both_ids():
    r = resolve(records([meeting([("Ian Miller", "Looks fine to me.")])]))
    assert r.status == "ambiguous" and r.ambiguous
    assert sorted(r.ids) == ["eid_pm", "eid_qa"] and r.scores == {"eid_qa": 0, "eid_pm": 0}


def test_cue_uses_all_lines_of_the_speaker():
    lines = [("Ian Miller", "Looks fine."), ("Dana Wu", "As the product manager I disagree."),
             ("Ian Miller", "From a QA perspective it needs tests.")]
    assert resolve(records([meeting(lines)])).ids == ["eid_qa"]


def test_cue_from_another_speaker_does_not_count():
    lines = [("Ian Miller", "Fine."), ("Dana Wu", "As QA I checked it.")]
    assert resolve(records([meeting(lines)])).ambiguous


def test_related_post_within_three_days_adds_two():
    rec = records([meeting([("Ian Miller", "Fine.")])],
                  [msg("a", "eid_qa", "2026-06-12T09:00:00", f"see {DOC_LINK}")])
    r = resolve(rec)
    assert r.scores == {"eid_qa": 2, "eid_pm": 0} and r.ids == ["eid_qa"]


def test_related_post_matches_document_type_case_insensitively():
    rec = records([meeting([("Ian Miller", "Fine.")])], [msg("a", "eid_qa", "2026-06-12T09:00:00", "the design spec")])
    assert resolve(rec).scores["eid_qa"] == 2


def test_related_post_four_days_away_adds_nothing():
    rec = records([meeting([("Ian Miller", "Fine.")])],
                  [msg("a", "eid_qa", "2026-06-09T09:00:00", f"see {DOC_LINK}")])
    r = resolve(rec)
    assert r.scores == {"eid_qa": 0, "eid_pm": 0} and r.ambiguous


def test_post_must_be_in_a_session_with_the_document():
    # The qa post is in the window but its session never mentions the document.
    rec = records([meeting([("Ian Miller", "Fine.")])], [msg("a", "eid_qa", "2026-06-12T09:00:00", "lunch?")])
    assert resolve(rec).scores["eid_qa"] == 0


def test_session_text_counts_for_every_poster_in_the_session():
    rec = records([meeting([("Ian Miller", "Fine.")])],
                  [msg("a", "eid_other", "2026-06-12T09:00:00", f"see {DOC_LINK}"),
                   msg("b", "eid_qa", "2026-06-12T10:00:00", "ok")])
    assert resolve(rec).scores == {"eid_qa": 2, "eid_pm": 0}


def test_mention_in_meeting_chat_adds_one():
    chat = MeetingChat("m1_chat", "2026-06-13T10:00:00\nDana Wu: thanks @eid_qa", "2026-06-13T10:00:00",
                       ("eid_qa",), "m1", "id")
    rec = records([meeting([("Ian Miller", "Fine.")])], chats=[chat])
    assert resolve(rec).scores == {"eid_qa": 1, "eid_pm": 0}


def test_mention_in_transcript_adds_one():
    rec = records([meeting([("Ian Miller", "Fine."), ("Dana Wu", "Thanks @eid_pm.")])])
    assert resolve(rec).scores == {"eid_qa": 0, "eid_pm": 1}


def test_role_cue_beats_related_post():
    rec = records([meeting(QA_LINES)], [msg("a", "eid_pm", "2026-06-12T09:00:00", f"see {DOC_LINK}")])
    r = resolve(rec)
    assert r.scores == {"eid_qa": 3, "eid_pm": 2} and r.ids == ["eid_qa"] and not r.ambiguous


def test_related_post_plus_mention_ties_role_cue():
    rec = records([meeting(QA_LINES + [("Dana Wu", "Thanks @eid_pm.")])],
                  [msg("a", "eid_pm", "2026-06-12T09:00:00", f"see {DOC_LINK}")])
    r = resolve(rec)
    assert r.scores == {"eid_qa": 3, "eid_pm": 3}
    assert r.ambiguous and r.status == "ambiguous" and sorted(r.ids) == ["eid_pm", "eid_qa"]


def test_tie_keeps_only_the_tied_top_candidates():
    emps = dict(EMPLOYEES, eid_x=emp("eid_x", "Ian Miller", "UX Researcher"))
    rec = records([meeting([("Ian Miller", "Fine.")], participants=("eid_qa", "eid_pm", "eid_x"))],
                  [msg("a", "eid_qa", "2026-06-12T09:00:00", f"see {DOC_LINK}"),
                   msg("b", "eid_pm", "2026-06-12T09:30:00", "ok")], employees=emps)
    r = resolve(rec)
    assert r.scores == {"eid_qa": 2, "eid_pm": 2, "eid_x": 0}
    assert sorted(r.ids) == ["eid_pm", "eid_qa"] and r.ambiguous


def test_single_candidate_resolves_without_scoring():
    r = resolve(records([meeting([("Dana Wu", "Hi.")])]), name="Dana Wu")
    assert r.status == "resolved" and r.ids == ["eid_solo"] and not r.ambiguous
    assert r.evidence == ["single candidate"] and r.scores == {}


def test_participant_without_matching_name_is_unresolved_and_keeps_raw_name():
    r = resolve(records([meeting([("Bob Nobody", "Hi.")])]), name="Bob Nobody")
    assert r.status == "unresolved" and r.ids == [] and not r.ambiguous
    assert any("Bob Nobody" in e for e in r.evidence)


def test_same_name_but_outside_all_evidence_is_not_a_candidate():
    # eid_pm is not a participant, has no post in the window and no mention: only eid_qa is a candidate.
    rec = records([meeting([("Ian Miller", "Hi.")], participants=("eid_qa",))])
    r = resolve(rec)
    assert r.ids == ["eid_qa"] and r.evidence == ["single candidate"]


def test_speaker_not_in_participants_but_active_nearby_is_a_candidate():
    rec = records([meeting([("Zed Active", "Hi.")], participants=("eid_solo",))],
                  [msg("a", "eid_other", "2026-06-15T09:00:00", "hello")])
    r = resolve(rec, name="Zed Active")
    assert r.ids == ["eid_other"] and r.status == "resolved"


def test_speaker_active_four_days_away_is_unresolved():
    rec = records([meeting([("Zed Active", "Hi.")], participants=("eid_solo",))],
                  [msg("a", "eid_other", "2026-06-17T09:00:00", "hello")])
    assert resolve(rec, name="Zed Active").status == "unresolved"


def test_speaker_not_in_participants_but_mentioned_is_a_candidate():
    rec = records([meeting([("Zed Active", "Hi."), ("Dana Wu", "Thanks @eid_other")], participants=("eid_solo",))])
    assert resolve(rec, name="Zed Active").ids == ["eid_other"]


def test_whitespace_in_name_is_normalised():
    rec = records([meeting([("Dana  Wu ", "Hi.")])])
    assert resolve(rec, name=" Dana   Wu").ids == ["eid_solo"]


@pytest.mark.parametrize("text,role", [
    ("As QA I checked it", "QA Specialist"),
    ("From a QA perspective this is risky", "QA Specialist"),
    ("As the product manager I own this", "Product Manager"),
    ("From a UX standpoint it is clumsy", "UX Researcher"),
    ("As the architect I would split it", "Technical Architect"),
    ("In my role as engineering lead I agree", "Engineering Lead"),
    ("As a software engineer I can build it", "Software Engineer"),
    ("As the VP of Engineering I approve", "VP of Engineering"),
    ("As the CPO I care about the roadmap", "Chief Product Officer"),
    ("As the marketing manager I will plan the launch", "Marketing Manager"),
    ("As a market research analyst I pulled the numbers", "Marketing Research Analyst"),
])
def test_role_cues_hit(text, role):
    assert role in role_cue_hits(text)


@pytest.mark.parametrize("text", [
    "Coordinate with the QA team for testing.",
    "The architecture needs work.",
    "Thanks for the engineering team update.",
    "Alice, do you have any architectural insights?",
    "We should improve the user experience.",
    "It works as expected.",
])
def test_topic_words_are_not_role_cues(text):
    assert role_cue_hits(text) == {}


def test_resolution_is_a_plain_value():
    r = Resolution(ids=[], ambiguous=False, status="unresolved", evidence=[], scores={})
    assert r.status == "unresolved"


# ---- facts, rows, lookup ----

def run_names(rec):
    return resolve_meeting_names(rec)


def test_rows_one_per_meeting_name_with_status():
    rec = records([meeting(QA_LINES + [("Dana Wu", "Hi."), ("Bob Nobody", "Hi.")])])
    facts, rows = run_names(rec)
    by = {r["name"]: r for r in rows}
    assert set(by) == {"Ian Miller", "Dana Wu", "Bob Nobody"} and len(rows) == 3
    assert by["Ian Miller"]["status"] == "resolved" and by["Ian Miller"]["ids"] == ["eid_qa"]
    assert by["Ian Miller"]["scores"] == {"eid_qa": 3, "eid_pm": 0} and by["Ian Miller"]["meeting_id"] == "m1"
    assert by["Ian Miller"]["evidence"] and by["Ian Miller"]["lines"] == 1
    assert by["Dana Wu"]["basis"] == "single" and by["Ian Miller"]["basis"] == "score"
    assert by["Bob Nobody"]["status"] == "unresolved" and by["Bob Nobody"]["basis"] == "none"


def test_spoke_in_edges_and_src():
    rec = records([meeting([("Ian Miller", "As QA I checked."), ("Dana Wu", "Hi."), ("Ian Miller", "More.")])])
    facts, _ = run_names(rec)
    spoke = {(e.src_id, e.dst_id): e for e in facts.edges if e.type == "SPOKE_IN"}
    assert set(spoke) == {("Person:eid_qa", "Meeting:m1"), ("Person:eid_solo", "Meeting:m1")}
    e = spoke[("Person:eid_qa", "Meeting:m1")]
    assert e.src == ("m1#3", "m1#5") and e.extractor == "names/v1"
    assert e.props == {"speaker_name": "Ian Miller", "ambiguous": False, "score": 3, "n_candidates": 2}
    for x in facts.edges:
        check_edge(x)


def test_ambiguous_name_gives_one_flagged_edge_per_tied_candidate():
    facts, _ = run_names(records([meeting([("Ian Miller", "Fine.")])]))
    spoke = [e for e in facts.edges if e.type == "SPOKE_IN"]
    assert sorted(e.src_id for e in spoke) == ["Person:eid_pm", "Person:eid_qa"]
    assert all(e.props["ambiguous"] and e.props["n_candidates"] == 2 and e.src == ("m1#3",) for e in spoke)


def test_unresolved_gives_no_edge():
    facts, _ = run_names(records([meeting([("Bob Nobody", "Hi.")])]))
    assert [e for e in facts.edges if e.type == "SPOKE_IN"] == []


def test_segments_without_speaker_are_skipped():
    m = meeting([("Dana Wu", "Hi.")])
    m = Meeting(m.id, m.date, m.document_type, m.participants, m.attendees,
                m.segments + (Segment("m1#9", "m1", 9, None, "stray"),))
    _, rows = run_names(records([m]))
    assert [r["name"] for r in rows] == ["Dana Wu"]


def test_chat_link_lines_give_shared_edges_with_chat_id_as_src():
    chat = MeetingChat("m1_chat", f"2026-06-13T10:00:00\nIan Miller: As QA I posted {DOC_LINK}\nDana Wu: no link here",
                       "2026-06-13T10:00:00", (), "m1", "id")
    rec = records([meeting([("Dana Wu", "Hi.")])], chats=[chat])
    facts, rows = run_names(rec)
    shared = [e for e in facts.edges if e.type == "SHARED"]
    assert [(e.src_id, e.dst_id, e.src) for e in shared] == [("Person:eid_qa", "Document:spec1", ("m1_chat",))]
    assert shared[0].props["ambiguous"] is False
    # The chat cue ("As QA") counts toward the role cue, and the chat-only name still gets a row without SPOKE_IN.
    row = next(r for r in rows if r["name"] == "Ian Miller")
    assert row["ids"] == ["eid_qa"] and row["chat_lines"] == 1
    assert not [e for e in facts.edges if e.type == "SPOKE_IN" and e.src_id == "Person:eid_qa"]


def test_chat_shared_edges_flag_ambiguity():
    chat = MeetingChat("m1_chat", f"2026-06-13T10:00:00\nIan Miller: {DOC_LINK}", "2026-06-13T10:00:00", (), "m1", "id")
    facts, _ = run_names(records([meeting([("Dana Wu", "Hi.")])], chats=[chat]))
    shared = [e for e in facts.edges if e.type == "SHARED"]
    assert len(shared) == 2 and all(e.props["ambiguous"] and e.src == ("m1_chat",) for e in shared)


def test_chat_link_to_url_record_and_unknown_link():
    rec = records([meeting([("Dana Wu", "Hi.")])], chats=[MeetingChat(
        "m1_chat", "2026-06-13T10:00:00\nDana Wu: see https://x.example/a and https://x.example/zzz",
        "2026-06-13T10:00:00", (), "m1", "id")])
    rec.urls.append(UrlRecord("u1", "https://x.example/a", ("https://x.example/a",), ("d",), (0,)))
    facts, _ = run_names(rec)
    assert [(e.dst_id) for e in facts.edges if e.type == "SHARED"] == ["Url:https://x.example/a"]


def test_unlinked_chat_is_ignored():
    chat = MeetingChat("orphan", "2026-07-01T09:00:00\nDana Wu: hi", "2026-07-01T09:00:00", (), None, None)
    _, rows = run_names(records([meeting([("Dana Wu", "Hi.")])], chats=[chat]))
    assert len(rows) == 1


def test_lookup_returns_resolution_from_rows():
    _, rows = run_names(records([meeting(QA_LINES)]))
    r = lookup(rows, "m1", "Ian  Miller")
    assert r.ids == ["eid_qa"] and r.status == "resolved"
    missing = lookup(rows, "m1", "Nobody")
    assert missing.status == "unresolved" and missing.ids == []


# ---- real fixture and runner ----

def test_mini_fixture_resolves_qa_ian(mini_bundle):
    from inherit.data.records import build_records

    emps, custs = load_metadata(mini_bundle)
    rec = build_records(load_bundle(mini_bundle, "mini"), emps, custs, "mini")
    facts, rows = resolve_meeting_names(rec)
    ian = next(r for r in rows if r["meeting_id"] == "mini_design_1" and r["name"] == "Ian Miller")
    assert ian["ids"] == ["eid_d4000004"] and ian["status"] == "resolved"
    assert ian["scores"]["eid_d4000004"] >= 3
    bob = next(r for r in rows if r["name"] == "Bob Ray")  # not in participants, posts nearby
    assert bob["status"] == "resolved" and bob["ids"] == ["eid_b2000002"]
    for e in facts.edges:
        check_edge(e)


def test_runner_names_layer_writes_rows_and_facts(tmp_path, mini_bundle):
    root = tmp_path / "root"
    shutil.copytree(mini_bundle, root / "data" / "bundle")
    (root / "src").mkdir()
    r = subprocess.run([sys.executable, "-m", "inherit.extract.run", "--layer", "names", "--products", "mini",
                        "--repo-root", str(root)], capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    d = root / "data" / "facts" / "mini"
    rows = [json.loads(x) for x in (d / "names.jsonl").read_text().splitlines()]
    assert rows and {"meeting_id", "name", "status", "ids", "scores", "evidence"} <= set(rows[0])
    assert (d / "names_facts.jsonl").read_text().count("SPOKE_IN") > 0
    assert "resolved" in r.stdout and "ambiguous" in r.stdout
