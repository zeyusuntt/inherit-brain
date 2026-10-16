import bugfx as fx
from inherit.extract.bug_rules import parse_session, norm_title
from inherit.extract.catalog import build_catalog
from inherit.extract.sessions import sessions


def parsed():
    rec = fx.records()
    cat = build_catalog(rec)
    ss = sessions(rec.messages)
    return {s[0].id[:8] + s[0].id[-5:-4]: parse_session(s, cat, rec.customers) for s in ss}, ss


def by_day(day):
    out, ss = parsed()
    for s, key in zip(ss, out):
        if s[0].id.startswith(day):
            return out[key]
    raise KeyError(day)


def test_norm_title():
    assert norm_title("The 'Duplicate  Log Entries' ") == "duplicate log entries"
    assert norm_title("the Session Hijacking Vulnerability in Mini", ("Mini",)) == "session hijacking vulnerability"
    assert norm_title("Blocking JavaScript Execution in Mini", ("Mini",)) == "blocking javascript execution"
    assert norm_title("Memory Leak in Task Processing Module", ("Mini",)) == "memory leak in task processing module"
    assert norm_title("API's CORS policy") == "api s cors policy"


def test_bot_session_is_empty():
    p = by_day("20260110")
    assert not p.bugs and not p.events and not p.residual


def test_intake_quoted_titles_customers_component_and_assignees():
    p = by_day("20260111")
    titles = [b["title"] for b in p.bugs]
    assert titles == ["Duplicate Log Entries for Manual Actions", "Inefficient Database Query Optimization"]
    b1, b2 = p.bugs
    assert b1["customer_ids"] == ["CUST-0047"] and b2["customer_ids"] == ["CUST-0108"]
    assert b1["component"] == "Database Management & Logging System"
    assert b2["component"] == "Backend Processing & Task Queue Management"
    assert b1["message_id"] == "20260111-1-b0002"
    assert b1["citations"][0]["record_id"] == "20260111-1-b0002"
    assert b1["citations"][0]["quote"] in fx.B_TITLE1
    assigned = [e for e in p.events if e["type"] == "ASSIGNED"]
    # b1: @mention by the author. b2: no mention, the next non-author responder self-assigns.
    assert (assigned[0]["message_id"], assigned[0]["assignee_eids"], assigned[0]["local_bug"]) == \
        ("20260111-1-b0002", [fx.EMMA], b1["local_id"])
    assert (assigned[1]["message_id"], assigned[1]["assignee_eids"], assigned[1]["local_bug"]) == \
        ("20260111-4-b0005", [fx.BOB], b2["local_id"])
    assert len(assigned) == 2  # the mentioned engineer's acknowledgement is not a second assignment


def test_unquoted_intake_is_residual_not_guessed():
    p = by_day("20260111")
    assert p.residual == ["20260111-5-b0006"]
    assert "20260111-5-b0006" not in p.handled
    assert {"20260111-1-b0002", "20260111-4-b0005"} <= p.handled


def test_open_source_look_alike():
    p = by_day("20260113")
    assert not p.bugs and not p.residual
    [x] = p.externals
    assert x["link"] == "https://github.com/postgres/postgres/pull/3"
    assert x["title"] == "Duplicate Log Entries for Transaction Rollbacks"
    assert x["similar_mention"] == "Duplicate Log Entries for Manual Actions"
    assert x["project"] == "postgres/postgres"
    assert x["message_id"] == "20260113-1-c0002"


def test_resolved_quoted_with_pr_link():
    p = by_day("20260115")
    [e] = p.events
    assert e["type"] == "RESOLVED" and e["mention"] == "Inefficient Database Query Optimization"
    assert e["message_id"] == "20260115-0-d0001"
    assert e["fix_pr_links"] == ["https://github.com/salesforce/Mini/pull/31"]  # trailing full stop stripped
    assert not p.residual


def test_resolved_paraphrase_keeps_mention_and_takes_pr_link_from_later_message_of_same_author():
    p = by_day("20260117")
    [e] = p.events
    assert e["mention"] == "duplicate manual action log rows"
    assert e["fix_pr_links"] == ["https://github.com/salesforce/Mini/pull/32"]
    assert {c["record_id"] for c in e["citations"]} == {"20260117-0-e0001", "20260117-1-e0002"}
    assert not p.residual


def test_dismissed_by_investigator_cue_message():
    p = by_day("20260120")
    [e] = p.events
    assert e["type"] == "DISMISSED" and e["mention"] == "Memory Leak in Suggestion Rendering"
    assert e["message_id"] == "20260120-1-f0002"  # the first message of the investigator that says no fix


def test_dismissal_cue_from_other_people_does_not_count():
    rec = fx.records(slack=[
        ("20260120-0-f0001", fx.EMMA, "2026-01-20T13:00:00", "Hi team, I looked into the reported issue about Foo Bar."),
        ("20260120-1-f0002", fx.BOB, "2026-01-20T13:02:00", "Good to know it's not a bug on our end."),
    ])
    p = parse_session(sessions(rec.messages)[0], build_catalog(rec), rec.customers)
    assert not p.events
    assert p.residual == ["20260120-0-f0001"]  # the opener is left for the model, nothing is guessed


def test_by_design_counts_as_no_fix():
    rec = fx.records(slack=[
        ("20260120-0-f0001", fx.EMMA, "2026-01-20T13:00:00", "Hi team, I looked into the reported issue about 'Token Expiry'."),
        ("20260120-1-f0002", fx.EMMA, "2026-01-20T13:02:00", "I found that this behavior is actually by design."),
    ])
    [e] = parse_session(sessions(rec.messages)[0], build_catalog(rec), rec.customers).events
    assert e["type"] == "DISMISSED" and e["mention"] == "Token Expiry"


def test_reopened_quoted_and_session_flagged_for_follow_up():
    p = by_day("20260125")
    [e] = p.events
    assert e["type"] == "REOPENED" and e["mention"] == "Duplicate Log Entries for Manual Actions"
    assert e["message_id"] == "20260125-0-g0001"
    assert p.reopen is True
    assert p.residual == []  # the follow-up is asked of the model because of the reopen flag, not a residual message


def test_reopened_unquoted_other_phrasing_is_residual():
    rec = fx.records(slack=[
        ("20260125-0-g0001", fx.IAN, "2026-01-25T14:00:00", "Hi team, the customer reopened the data sync ticket again."),
    ])
    p = parse_session(sessions(rec.messages)[0], build_catalog(rec), rec.customers)
    assert not p.events and p.residual == ["20260125-0-g0001"]


def test_reopened_by_customer_id_phrase():
    rec = fx.records(slack=[
        ("20260125-0-g0001", fx.IAN, "2026-01-25T14:00:00", "Hi team, the 'API Authentication Failure' bug has been reopened by CUST-0047."),
    ])
    [e] = parse_session(sessions(rec.messages)[0], build_catalog(rec), rec.customers).events
    assert e["type"] == "REOPENED" and e["mention"] == "API Authentication Failure"


def test_non_product_pr_link_is_not_a_fix_link():
    rec = fx.records(slack=[
        ("20260115-0-d0001", fx.BOB, "2026-01-15T11:00:00",
         "Hi team, the bug titled 'Foo Bar Baz' has been resolved. PR https://github.com/django/django/pull/9 and "
         "https://github.com/salesforce/Mini/pull/31"),
    ])
    [e] = parse_session(sessions(rec.messages)[0], build_catalog(rec), rec.customers).events
    assert e["fix_pr_links"] == ["https://github.com/salesforce/Mini/pull/31"]
