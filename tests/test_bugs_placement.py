"""Task 13f: where an ASSIGNED event sits, and where a bug's product area may be read from. Synthetic records."""

import json

import bugfx as fx
from inherit.extract import bugs

M_INTRO = ("20260111-0-a0001", fx.IAN, "2026-01-11T09:00:00", fx.B_INTRO)
M_B1 = ("20260111-1-a0002", fx.IAN, "2026-01-11T09:02:00",
        f"First up, we have an issue with Memory Leak in Suggestion Rendering. It slows the UI for customer CUST-0064. "
        f"@{fx.EMMA}, can you take this one?")
M_A1 = ("20260111-2-a0003", fx.EMMA, "2026-01-11T09:05:00", "Got it! I'll start on the memory leak right away.")
M_B2 = ("20260111-3-a0004", fx.IAN, "2026-01-11T09:10:00",
        f"Next, we have a Data Race in Export Jobs. Customer CUST-0108 is affected. cc @{fx.BOB}")
M_A2 = ("20260111-4-a0005", fx.BOB, "2026-01-11T09:12:00", "I'll take this one.")
M_B3 = ("20260111-5-a0006", fx.IAN, "2026-01-11T09:20:00",
        "Lastly, we have a Stale Cache in Search. Customer CUST-0047 is hit. Can you take this one?")
M_A3 = ("20260111-6-a0007", fx.EMMA, "2026-01-11T09:24:00", "Sure, I'll check the cache layers.")
SESSION1 = [M_INTRO, M_B1, M_A1, M_B2, M_A2, M_B3, M_A3]

LATER = [
    ("20260120-0-b0001", fx.IAN, "2026-01-20T10:00:00",
     "The Memory Leak in Suggestion Rendering has been reassigned, please see the thread."),
    ("20260120-1-b0002", fx.EMMA, "2026-01-20T10:03:00",
     "Understood, I'll take another look at the Memory Leak in Suggestion Rendering."),
]


def bug(lid, title, mid, text, cust):
    return {"local_id": lid, "title": title, "aliases": [], "component": None, "impact_text": text,
            "customer_ids": [cust], "message_id": mid, "citations": [fx.cite(mid, title)]}


def assigned(ref, mid, who, quote, mention=None):
    return {"type": "ASSIGNED", "bug_ref": ref, "bug_mention": mention, "actor_message_id": mid,
            "assignee_eids": [who], "fix_pr_links": [], "citations": [fx.cite(mid, quote)]}


BUGS = [bug("x1", "Memory Leak in Suggestion Rendering", M_B1[0], M_B1[3], "CUST-0064"),
        bug("x2", "Data Race in Export Jobs", M_B2[0], M_B2[3], "CUST-0108"),
        bug("x3", "Stale Cache in Search", M_B3[0], M_B3[3], "CUST-0047")]
ACCEPTANCES = [assigned("x1", M_A1[0], fx.EMMA, "I'll start on the memory leak"),
               assigned("x2", M_A2[0], fx.BOB, "I'll take this one"),
               assigned("x3", M_A3[0], fx.EMMA, "I'll check the cache layers")]


def run(events=ACCEPTANCES, slack=SESSION1, extra_script=None, tmp=None):
    script = {M_B1[0]: fx.session_json(bugs=BUGS, events=events), **(extra_script or {})}
    chat = fx.ScriptedChat(script=script, link=fx.link_json())
    return bugs.run_bugs(fx.records(slack=slack), chat=chat, cache_root=tmp), chat


def assigned_events(out, title_slug):
    evs = [n for n in out.facts.nodes if n.label == "BugEvent" and n.props["type"] == "ASSIGNED"
           and n.props["bug"].endswith(title_slug)]
    return sorted((n.props["message_id"], tuple(n.props["assignee_eids"])) for n in evs)


def test_assignment_sits_on_the_handoff_message_not_on_the_acceptance(tmp_path):
    out, _ = run(tmp=tmp_path)
    # @EMMA is asked on the intro message and accepts on the next one: one event, on the hand-off
    assert assigned_events(out, "memory-leak-in-suggestion-rendering") == [(M_B1[0], (fx.EMMA,))]
    ev = next(n for n in out.facts.nodes if n.id == f"BugEvent:Mini:{M_B1[0]}:1")
    assert ev.props["type"] == "ASSIGNED" and ev.props["citations"][0]["record_id"] == M_B1[0]
    assert "can you take this one" in ev.props["citations"][0]["quote"]
    assert ev.props["ts"] == M_B1[2]
    assert out.report["assignment_placement"]["moved_to_handoff"] >= 1


def test_reported_event_never_carries_assignees(tmp_path):
    out, _ = run(tmp=tmp_path)
    rep = [n for n in out.facts.nodes if n.label == "BugEvent" and n.props["type"] == "REPORTED"]
    assert rep and all(n.props["assignee_eids"] == [] for n in rep)


def test_current_assignee_is_still_derived_from_the_events(tmp_path):
    out, _ = run(tmp=tmp_path)
    b = next(n for n in out.facts.nodes if n.id.endswith("memory-leak-in-suggestion-rendering"))
    assert b.props["current_assignees"] == [fx.EMMA]


def test_a_bare_mention_is_not_a_handoff(tmp_path):
    out, _ = run(tmp=tmp_path)
    # "cc @BOB" carries no assignment cue: BOB took the bug himself, the event stays where he took it
    assert assigned_events(out, "data-race-in-export-jobs") == [(M_A2[0], (fx.BOB,))]


def test_nobody_hands_the_bug_over_and_a_person_takes_it(tmp_path):
    out, _ = run(tmp=tmp_path)
    assert assigned_events(out, "stale-cache-in-search") == [(M_A3[0], (fx.EMMA,))]


def test_a_second_event_for_the_same_assignment_is_dropped(tmp_path):
    both = [assigned("x1", M_B1[0], fx.EMMA, "can you take this one"), ACCEPTANCES[0], *ACCEPTANCES[1:]]
    out, _ = run(events=both, tmp=tmp_path)
    assert assigned_events(out, "memory-leak-in-suggestion-rendering") == [(M_B1[0], (fx.EMMA,))]
    assert out.report["assignment_placement"]["duplicate_dropped"] == 1


def test_a_later_reassignment_is_not_backfilled_onto_the_first_handoff(tmp_path):
    again = assigned(None, LATER[1][0], fx.EMMA, "I'll take another look",
                     mention="Memory Leak in Suggestion Rendering")
    out, _ = run(slack=SESSION1 + LATER, extra_script={LATER[1][0]: fx.session_json(events=[again])}, tmp=tmp_path)
    got = assigned_events(out, "memory-leak-in-suggestion-rendering")
    assert got == [(M_B1[0], (fx.EMMA,)), (LATER[1][0], (fx.EMMA,))]  # the second assignment stays on its own message


def test_events_by_type_in_the_report_match_the_nodes(tmp_path):
    out, _ = run(tmp=tmp_path)
    n = sum(1 for x in out.facts.nodes if x.label == "BugEvent" and x.props["type"] == "ASSIGNED")
    assert out.report["events_by_type"]["ASSIGNED"] == n == 3


def test_placement_changes_no_model_request(tmp_path):
    out, chat = run(tmp=tmp_path)
    users = [c["user"] for c in chat.calls]
    out2, chat2 = run(tmp=tmp_path)  # a rerun is fully cached: every request is identical
    assert chat2.calls == [] and len(users) == 1


# ---------------------------------------------------------------- component named in the acceptance

def test_area_named_in_the_uniquely_linked_acceptance_is_accepted(tmp_path):
    first = (f"First up, we have 'Misconfigured Export Permissions'. Users export more than they may. Customer "
             f"CUST-0047. @{fx.EMMA}, you're on this one too.")
    accept = "Understood. I'll check the Data Export Module and fix the permissions."
    slack = [("20260111-0-b0001", fx.IAN, "2026-01-11T09:00:00", fx.B_INTRO),
             ("20260111-1-b0002", fx.IAN, "2026-01-11T09:02:00", first),
             ("20260111-2-b0003", fx.EMMA, "2026-01-11T09:05:00", accept)]
    answer = {"bugs": [], "events": [], "external": [], "components": [
        {"bug_ref": "rb1", "component": "Data Export Module", "citations": [fx.cite("20260111-2-b0003", "Data Export Module")]}]}
    chat = fx.ScriptedChat(script={"20260111-2-b0003": json.dumps(answer)})
    out = bugs.run_bugs(fx.records(slack=slack), chat=chat, cache_root=tmp_path)
    b = next(n for n in out.facts.nodes if n.label == "Bug")
    assert b.props["component"] == "Data Export Module"


def test_area_in_a_later_bugs_acceptance_is_not_borrowed(tmp_path):
    first = (f"First up, we have 'Misconfigured Export Permissions'. Users export more than they may. Customer "
             f"CUST-0047. @{fx.EMMA}, you're on this one too.")
    second = (f"Next, we have 'Improper Access Control'. Customer CUST-0108 is hit. @{fx.BOB}, can you take this one?")
    slack = [("20260111-0-b0001", fx.IAN, "2026-01-11T09:00:00", fx.B_INTRO),
             ("20260111-1-b0002", fx.IAN, "2026-01-11T09:02:00", first),
             ("20260111-2-b0003", fx.EMMA, "2026-01-11T09:05:00", "Understood. I'll look into it."),
             ("20260111-3-b0004", fx.IAN, "2026-01-11T09:10:00", second),
             ("20260111-4-b0005", fx.BOB, "2026-01-11T09:12:00", "Sure. I'll check the Data Export Module.")]
    answer = {"bugs": [], "events": [], "external": [], "components": [
        {"bug_ref": "rb1", "component": "Data Export Module", "citations": [fx.cite("20260111-4-b0005", "Data Export Module")]}]}
    chat = fx.ScriptedChat(script={"20260111-1-b0002": json.dumps(answer)})
    out = bugs.run_bugs(fx.records(slack=slack), chat=chat, cache_root=tmp_path)
    b = next(n for n in out.facts.nodes if n.label == "Bug" and n.props["title"] == "Misconfigured Export Permissions")
    assert b.props["component"] is None  # the area sits in the acceptance of the next bug
    assert out.report["dropped"]["components"] == {"not_in_message": 1}


def test_a_quote_cited_on_the_intro_but_written_in_the_acceptance_is_cited_on_the_acceptance(tmp_path):
    first = (f"First up, we have 'Misconfigured Export Permissions'. Users export more than they may. Customer "
             f"CUST-0047. @{fx.EMMA}, you're on this one too.")
    accept = "Understood. I'll check the Data Export Module and fix the permissions."
    slack = [("20260111-0-b0001", fx.IAN, "2026-01-11T09:00:00", fx.B_INTRO),
             ("20260111-1-b0002", fx.IAN, "2026-01-11T09:02:00", first),
             ("20260111-2-b0003", fx.EMMA, "2026-01-11T09:05:00", accept)]
    answer = {"bugs": [], "events": [], "external": [], "components": [
        {"bug_ref": "rb1", "component": "Data Export Module",
         "citations": [fx.cite("20260111-1-b0002", "I'll check the Data Export Module and fix the permissions")]}]}
    chat = fx.ScriptedChat(script={"20260111-2-b0003": json.dumps(answer)})
    out = bugs.run_bugs(fx.records(slack=slack), chat=chat, cache_root=tmp_path)
    b = next(n for n in out.facts.nodes if n.label == "Bug")
    assert b.props["component"] == "Data Export Module"
    assert out.report["repaired_citations"] == 1


def test_a_component_read_from_the_acceptance_does_not_change_the_link_request(tmp_path):
    text2 = ("Next, we have 'Inefficient Database Query Optimization'. Customer CUST-0108 is affected. "
             "Can you take this one?")
    slack = [r if r[0] != "20260111-3-b0004" else (r[0], r[1], r[2], text2) for r in fx.SLACK]
    slack = [r if r[0] != "20260111-4-b0005" else (r[0], r[1], r[2], "Sure thing! I'll dive into the Query Planner Core.")
             for r in slack]
    answer = {"bugs": [], "events": [], "external": [], "components": [
        {"bug_ref": "rb2", "component": "Query Planner Core", "citations": [fx.cite("20260111-4-b0005", "Query Planner Core")]}]}

    def go(components, tmp):
        first = json.loads(fx.session_json(bugs=[fx.LEAK_BUG]))  # the same session asks for the unquoted intake too
        script = {"20260111-5-b0006": json.dumps({**first, "components": answer["components"] if components else []}),
                  "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])}
        chat = fx.ScriptedChat(script=script, link=fx.link_json(("I1", "B1")))
        return bugs.run_bugs(fx.records(slack=slack), chat=chat, cache_root=tmp), chat

    with_c, chat_c = go(True, tmp_path / "a")
    without, chat_n = go(False, tmp_path / "b")
    b = next(n for n in with_c.facts.nodes if n.label == "Bug" and n.props["title"] == "Inefficient Database Query Optimization")
    assert b.props["component"] == "Query Planner Core"
    assert [c["user"] for c in chat_c.link_calls] == [c["user"] for c in chat_n.link_calls] and chat_c.link_calls
