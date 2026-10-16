import json

import pytest

import semfx as fx
from inherit.extract import semantic
from inherit.extract.rules import run_rules
from inherit.extract.structural import structural_facts_from_records
from inherit.facts import Facts
from inherit.ontology import check_edge

P = "https://github.com/salesforce/Mini/pull/"
ALERTS = "Feature:Mini:real time alerts"
PRED = "Feature:Mini:predictive analytics"


def run(chat=None, rec=None, tmp=None, **kw):
    chat = chat or fx.standard_chat()
    return semantic.run_semantic(rec or fx.records(), chat=chat, cache_root=tmp, **kw), chat


def nodes(facts, label):
    return {n.id: n for n in facts.nodes if n.label == label}


def edges(facts, type_):
    return [e for e in facts.edges if e.type == type_]


def pairs(facts, type_):
    return {(e.src_id, e.dst_id) for e in edges(facts, type_)}


@pytest.fixture
def res(tmp_path):
    out, chat = run(tmp=tmp_path)
    return out, chat


# ---------------------------------------------------------------- documents, meetings, features

def test_feedback_bullets_become_anonymous_items(res):
    out, _ = res
    d = nodes(out.facts, "Document")["Document:final_mini_market_research_report"]
    assert d.props["feedback_items"] == ["Add industry specifics to the Executive Summary.",
                                         "Include regulatory changes in Market Trends."]
    assert "Document:mini_market_research_report" not in nodes(out.facts, "Document")  # no feedback, no node


def test_features_have_canonical_ids_details_and_product_edges(res):
    out, _ = res
    f = nodes(out.facts, "Feature")
    assert set(f) == {ALERTS, PRED}
    assert f[ALERTS].props["title"] == "real-time alerts" and f[ALERTS].props["topic"] == "real time alerts"
    assert f[ALERTS].props["details"] == ["WebSockets for pushing alerts"] and f[ALERTS].props["product"] == "Mini"
    assert (f"Product:Mini", ALERTS) in pairs(out.facts, "HAS")
    assert (f"Meeting:{fx.M1}", ALERTS) in pairs(out.facts, "DISCUSSES")


def test_contribution_person_comes_from_the_referenced_segment_speaker(res):
    out, _ = res
    c = {(e.src_id, e.dst_id, e.props["kind"]): e for e in edges(out.facts, "CONTRIBUTED")}
    assert (f"Person:{fx.BEN}", ALERTS, "lead") in c  # Ben speaks in segment 5; Ann (who asked) is not the lead
    assert (f"Person:{fx.ANN}", ALERTS, "lead") not in c
    assert (f"Person:{fx.CY}", ALERTS, "support") in c and not c[(f"Person:{fx.CY}", ALERTS, "support")].props["ambiguous"]
    assert c[(f"Person:{fx.BEN}", ALERTS, "lead")].key == f"{fx.M1}#5:lead"


def test_ambiguous_speaker_keeps_both_candidates_and_unresolved_is_counted(res):
    out, _ = res
    amb = [e for e in edges(out.facts, "CONTRIBUTED") if e.dst_id == PRED]
    assert {e.src_id for e in amb} == {f"Person:{fx.SAM1}", f"Person:{fx.SAM2}"}
    assert all(e.props["ambiguous"] and e.props["kind"] == "assigned" for e in amb)
    assert out.report["unresolved_speakers"] == [{"meeting": fx.M1, "segment": f"{fx.M1}#9", "speaker": "Zed Nobody"}]


def test_meeting_prompt_carries_catalogue_and_names_no_employee_ids(res):
    _, chat = res
    first = chat.kind("meeting")[0]["user"]
    assert f"MEETING: {fx.M1}" in first
    assert "PR LIST" not in first and "Add Metrics Export" in first  # only titles the speakers say; no catalogue to copy
    assert "Support RNN Streams" not in first and "Revert PR# 1" not in first
    assert fx.BEN not in first.split("SEGMENTS")[1]
    second = chat.kind("meeting")[1]["user"]
    assert "real-time alerts" in second  # KNOWN FEATURES from the first meeting


def test_meeting_citation_that_is_not_a_quote_drops_the_item(tmp_path):
    bad = json.loads(fx.M1_ANSWER)
    bad["features"][0]["citations"] = [fx.cite(f"{fx.M1}#5", "this was never said")]
    chat = fx.standard_chat()
    chat.script[f"MEETING: {fx.M1}"] = json.dumps(bad)
    out, _ = run(chat, tmp=tmp_path)
    assert ALERTS not in nodes(out.facts, "Feature")
    assert out.report["dropped"]["features"] == {"quote_not_found": 1}
    assert out.report["unresolved_feature_refs"] >= 3  # its contributions, events and links lost their feature
    assert out.report["manifest"]["complete"] is False  # a validation loss on a feature-bearing record


def test_event_without_the_wording_of_its_kind_is_dropped(tmp_path):
    bad = json.loads(fx.M1_ANSWER)
    bad["feature_events"].append({"feature_ref": "f1", "kind": "implemented", "reason": None, "pr": None,
                                  "segment_ref": f"{fx.M1}#5", "citations": [fx.cite(f"{fx.M1}#5", "I'll lead real-time alerts")]})
    chat = fx.standard_chat()
    chat.script[f"MEETING: {fx.M1}"] = json.dumps(bad)
    out, _ = run(chat, tmp=tmp_path)
    assert out.report["dropped"]["feature_events"] == {"no_evidence_for_kind": 1}
    assert nodes(out.facts, "Feature")[ALERTS].props["implemented"] is False


def test_lead_cited_on_the_asking_segment_is_dropped(tmp_path):
    bad = json.loads(fx.M1_ANSWER)
    bad["contributions"].append({"feature_ref": "f1", "segment_ref": f"{fx.M1}#4", "kind": "lead",
                                 "citations": [fx.cite(f"{fx.M1}#4", "Ben, can you lead real-time alerts?")]})
    chat = fx.standard_chat()
    chat.script[f"MEETING: {fx.M1}"] = json.dumps(bad)
    out, _ = run(chat, tmp=tmp_path)
    assert out.report["dropped"]["contributions"] == {"request_not_acceptance": 1}
    assert (f"Person:{fx.ANN}", ALERTS) not in pairs(out.facts, "CONTRIBUTED")


def test_citation_to_a_record_outside_the_call_is_dropped(tmp_path):
    bad = json.loads(fx.M1_ANSWER)
    bad["features"][1]["citations"] = [fx.cite("20260305-1-a0002", "elaborate")]
    chat = fx.standard_chat()
    chat.script[f"MEETING: {fx.M1}"] = json.dumps(bad)
    out, _ = run(chat, tmp=tmp_path)
    assert PRED not in nodes(out.facts, "Feature")
    assert out.report["dropped"]["features"] == {"record_not_supplied": 1}


# ---------------------------------------------------------------- documents reviews

def test_slack_review_is_a_suggested_edge_from_the_message_author_to_the_shared_document(res):
    out, _ = res
    e = {(x.src_id, x.dst_id): x for x in edges(out.facts, "SUGGESTED")}
    ben = e[(f"Person:{fx.BEN}", "Document:mini_market_research_report")]
    assert ben.key == "20260305-1-a0002" and ben.props["section"] == "Product Description"
    assert ben.props["kind"] == "suggestion" and ben.props["text"] == "Elaborate on the advanced algorithms"
    assert ben.src == ("20260305-1-a0002",) and ben.props["ambiguous"] is False
    assert (f"Person:{fx.DEE}", "Document:mini_market_research_report") in e  # the doc given as a link resolves too
    assert ("Message:20260305-1-a0002", "Document:mini_market_research_report") in pairs(out.facts, "DISCUSSES")


def test_meeting_review_attaches_to_the_document_linked_in_the_meeting_chat(res):
    out, _ = res
    e = {(x.src_id, x.dst_id): x for x in edges(out.facts, "SUGGESTED")}
    dee = e[(f"Person:{fx.DEE}", "Document:final_mini_market_research_report")]
    assert dee.key == f"{fx.M2}#5" and dee.props["section"] == "Competitive Analysis"
    assert (f"Person:{fx.BEN}", "Document:final_mini_market_research_report") in e
    assert (f"Meeting:{fx.M2}", "Document:final_mini_market_research_report") in pairs(out.facts, "DISCUSSES")


# ---------------------------------------------------------------- PR membership

def test_exact_title_link_returns_every_pr_with_that_title_and_ignores_external_and_revert(res):
    out, _ = res
    about = {(e.src_id, e.dst_id): e for e in edges(out.facts, "ABOUT") if e.src_id.startswith("PR:")}
    assert about[(f"PR:{P}1", PRED)].props["kind"] == "development"
    assert (f"PR:{P}7", PRED) in about  # same title, other author and unapproved: stays separate but included
    assert (f"PR:{P}2", PRED) in about
    assert not [k for k in about if k[0] in (f"PR:{P}5", "PR:https://github.com/apache/kafka/pull/9001")]
    assert about[(f"PR:{P}1", PRED)].props["basis"] == "meeting_title"


def test_batch_call_assigns_unmatched_prs_and_only_with_a_valid_quote(res):
    out, chat = res
    about = {(e.src_id, e.dst_id): e for e in edges(out.facts, "ABOUT") if e.src_id.startswith("PR:")}
    assert about[(f"PR:{P}6", PRED)].props["basis"] == "batch"
    assert (f"PR:{P}3", ALERTS) in about and (f"PR:{P}4", ALERTS) in about
    assert not [k for k in about if k[0] == f"PR:{P}8"]  # nothing answered for it
    prompt = chat.kind("membership")[0]["user"]
    prs_part = prompt.split("PRS:")[1]
    assert "Predictive Model Evaluation" in prs_part and "Add Metrics Export" not in prs_part  # matched by title already
    assert "Add Metrics Export -> F2" in prompt  # shown as an example of an assignment
    assert "Support RNN Streams" not in prompt and "Revert PR# 1" not in prompt
    assert "Fix Crash In Export" not in prompt  # a bug channel cites it: a bug fix, not feature development
    assert out.report["pr_membership"]["bugfix_prs_excluded"] == 1
    assert out.report["pr_membership"]["unassigned"] == [f"{P}8"]


def test_batch_assignment_with_a_bad_quote_is_dropped(tmp_path):
    chat = fx.standard_chat()
    good = chat.membership

    def bad(user):
        data = json.loads(good(user))
        for a in data["assignments"]:
            a["quote"] = "words that are not in the pull request"
        return json.dumps(data)

    chat.membership = bad
    out, _ = run(chat, tmp=tmp_path)
    assert not [e for e in edges(out.facts, "ABOUT") if e.src_id == f"PR:{P}6"]
    assert out.report["dropped"]["pr_membership"]["quote_not_found"] >= 1


def test_implemented_by_needs_a_merged_pr(res):
    out, _ = res
    ib = pairs(out.facts, "IMPLEMENTED_BY")
    assert (PRED, f"PR:{P}1") in ib and (PRED, f"PR:{P}2") in ib
    assert (PRED, f"PR:{P}7") not in ib  # approved nowhere, not merged
    assert (ALERTS, f"PR:{P}3") not in ib


# ---------------------------------------------------------------- claims and observations

def test_competitor_claims_create_cited_competitor_and_observations(res):
    out, _ = res
    assert "Competitor:RivalCo" in nodes(out.facts, "Competitor")
    assert nodes(out.facts, "Competitor")["Competitor:RivalCo"].props["name"] == "RivalCo"
    obs = nodes(out.facts, "Observation")
    f, w = obs["Observation:Mini:20260310-0-b0001:0"], obs["Observation:Mini:20260310-0-b0001:1"]
    assert (f.props["kind"], w.props["kind"]) == ("competitor_feature", "competitor_weakness")
    assert f.props["subject"] == "RivalCo" and f.props["release_scope"] == "unknown"
    assert f.props["record_id"] == "20260310-0-b0001" and f.props["text"] == "RivalCo offers live dashboards"
    st = pairs(out.facts, "STATED")
    assert (f"Person:{fx.EVE}", f.id) in st and (f"Person:{fx.EVE}", w.id) in st  # the author of the message
    ab = pairs(out.facts, "ABOUT")
    assert (f.id, "Competitor:RivalCo") in ab
    assert ("Product:Mini", "Competitor:RivalCo") in pairs(out.facts, "HAS")


def test_competitor_not_named_in_the_session_is_dropped(tmp_path):
    bad = json.loads(fx.session_json(claims=fx.RIVAL_CLAIMS))
    bad["claims"][0]["subject_ref"] = "GhostCo"
    chat = fx.standard_chat()
    chat.script["[20260310-0-b0001]"] = json.dumps(bad)
    out, _ = run(chat, tmp=tmp_path)
    assert "Competitor:GhostCo" not in nodes(out.facts, "Competitor")
    assert "Observation:Mini:20260310-0-b0001:0" not in nodes(out.facts, "Observation")
    assert out.report["dropped"]["claims"]["subject_not_in_session"] == 1


def test_competitor_demo_link_is_a_rule_not_a_model_call(tmp_path):
    out, chat = run(tmp=tmp_path, dry=True)  # dry mode answers nothing from the model
    u = nodes(out.facts, "Url")["Url:https://www.rivalco.com/demo"]
    assert u.props["subject"] == "Competitor:RivalCo" and u.props["kind"] == "demo"
    assert ("Url:https://www.rivalco.com/demo", "Competitor:RivalCo") in pairs(out.facts, "ABOUT")
    assert "Url:https://sf-internal.slack.com/archives/Mini/demo_1" not in nodes(out.facts, "Url")  # own demo: 4a's
    assert out.report["demo_links"] == [{"link": "https://www.rivalco.com/demo", "competitor": "Competitor:RivalCo"}]


def test_previous_release_scope_needs_record_specific_evidence(res):
    out, _ = res
    obs = nodes(out.facts, "Observation")
    qa = obs["Observation:Mini:20260315-0-c0001:0"]
    assert qa.props["release_scope"] == "previous" and qa.props["scope_basis"] == "phrase"
    issue = obs["Observation:Mini:20260316-0-c0003:0"]
    assert issue.props["release_scope"] == "previous" and issue.props["scope_basis"] == "session_cue"
    cur = obs["Observation:Mini:20260317-0-c0006:0"]  # the model said previous, the session has no such cue
    assert cur.props["release_scope"] == "unknown"
    assert out.report["scope_downgraded"] == 1
    assert out.report["observations"]["customer_issue"] == {"previous": 1, "unknown": 1}


def test_customer_claims_link_to_the_company_record_and_qa_to_the_named_person(res):
    out, _ = res
    rb = pairs(out.facts, "REPORTED_BY")
    assert ("Observation:Mini:20260316-0-c0003:0", "Customer:CUST-0001") in rb
    assert ("Observation:Mini:20260316-2-c0005:0", "Customer:CUST-0002") in rb
    assert nodes(out.facts, "Customer")["Customer:CUST-0001"].props["company"] == "TechCorp"
    assert ("Observation:Mini:20260315-0-c0001:0", f"Person:{fx.CY}") in rb  # literal eid_ text in the QA sentence
    assert (f"Person:{fx.ANN}", "Observation:Mini:20260315-0-c0001:0") in pairs(out.facts, "STATED")  # poster != reporter
    ab = pairs(out.facts, "ABOUT")
    assert ("Observation:Mini:20260316-0-c0003:0", "Product:Mini") in ab


def test_customer_id_not_in_the_session_is_dropped(tmp_path):
    bad = json.loads(fx.session_json(claims=fx.CUST_CLAIMS))
    bad["claims"][0]["customer_ids"] = ["CUST-0099"]
    chat = fx.standard_chat()
    chat.script["[20260316-0-c0003]"] = json.dumps(bad)
    out, _ = run(chat, tmp=tmp_path)
    assert not [e for e in edges(out.facts, "REPORTED_BY") if e.src_id == "Observation:Mini:20260316-0-c0003:0"]
    assert out.report["dropped"]["claims"]["customer_not_in_session"] == 1


# ---------------------------------------------------------------- PR decisions

def test_decisions_are_message_edges_with_action_reason_and_conditional(res):
    out, _ = res
    e = {x.src_id: x for x in edges(out.facts, "ABOUT") if x.dst_id == f"PR:{P}3" and x.src_id.startswith("Message:")}
    assert e["Message:20260320-1-d0002"].props == {"action": "withhold", "reason": "product_direction",
                                                  "conditional": False}
    assert e["Message:20260320-0-d0001"].props["action"] == "question"
    assert e["Message:20260320-2-d0003"].props["conditional"] is True


def test_pending_action_follows_the_latest_effective_decision(res):
    out, _ = res
    pr = nodes(out.facts, "PR")
    assert pr[f"PR:{P}3"].props["pending_action"] == "need_close"  # approved, open, unmerged, held
    assert pr[f"PR:{P}3"].props["pending_reason"] == "product_direction"
    assert pr[f"PR:{P}4"].props["pending_action"] == "need_merge"  # held, then resumed
    assert pr[f"PR:{P}6"].props["pending_action"] is None  # open but changes requested
    assert pr[f"PR:{P}1"].props["pending_action"] is None  # merged
    assert out.report["pending"] == {"need_close": 1, "need_merge": 1}


def test_open_approved_pr_without_any_decision_needs_merge(tmp_path):
    rec = fx.records(prs=[fx.pr(1, "Alone")])
    out, _ = run(fx.ScriptedChat(), rec, tmp_path)
    assert nodes(out.facts, "PR")[f"PR:{P}1"].props["pending_action"] == "need_merge"


def test_hold_and_resume_become_dated_feature_events_after_membership(res):
    out, _ = res
    ev = nodes(out.facts, "FeatureEvent")
    held = ev["FeatureEvent:Mini:20260320-1-d0002:0"]
    assert held.props["kind"] == "held" and held.props["reason"] == "product_direction"
    assert held.props["date"] == "2026-03-20T09:05:00" and held.props["record_id"] == "20260320-1-d0002"
    assert ev["FeatureEvent:Mini:20260325-1-f0002:0"].props["kind"] == "held"
    resumed = ev["FeatureEvent:Mini:20260402-0-f0003:0"]
    assert resumed.props["kind"] == "resumed" and resumed.props["date"] == "2026-04-02T09:00:00"
    assert (ALERTS, held.id) in pairs(out.facts, "HAS_EVENT")
    assert "FeatureEvent:Mini:20260320-0-d0001:0" not in ev  # a question is not an event
    assert "FeatureEvent:Mini:20260320-2-d0003:0" not in ev  # nor is a conditional request


def test_unresolvable_pr_number_is_counted_not_linked(tmp_path):
    chat = fx.standard_chat()
    bad = json.loads(fx.session_json(pr_decisions=fx.PR4_HOLD))
    bad["pr_decisions"][0]["prs"] = ["404"]
    chat.script["[20260325-1-f0002]"] = json.dumps(bad)
    out, _ = run(chat, tmp=tmp_path)
    assert out.report["pr_decisions"]["unresolved_prs"] == [{"message": "20260325-1-f0002", "pr": "404", "why": "unknown"}]


# ---------------------------------------------------------------- feature events and derived fate

def test_meeting_events_and_reverted_event_from_an_applied_revert(res):
    out, _ = res
    ev = nodes(out.facts, "FeatureEvent")
    d = ev[f"FeatureEvent:Mini:{fx.M1}#4:0"]
    assert d.props["kind"] == "discussed" and d.props["date"] == "2026-03-01T10:00:00"
    impl = ev[f"FeatureEvent:Mini:{fx.M1}#8:0"]
    assert impl.props["kind"] == "implemented"
    rev = ev["FeatureEvent:Mini:github_com_salesforce_Mini_pull_5:0"]
    assert rev.props["kind"] == "reverted" and rev.props["reason"] == "product_direction"
    assert (PRED, rev.id) in pairs(out.facts, "HAS_EVENT")  # attached through the reverted PR's feature


def test_fate_predicates(res):
    out, _ = res
    f = nodes(out.facts, "Feature")
    p = f[PRED].props
    assert p["implemented"] is True and p["reverted"] is True and p["delayed_then_added"] is False
    assert p["discussed_not_implemented"] is False and p["derived_by"] == "feature_fate/v1"
    a = f[ALERTS].props
    assert a["implemented"] is False and a["held"] is True and a["delayed_then_added"] is True
    assert a["discussed_not_implemented"] is True


def test_discussed_not_implemented_needs_a_complete_manifest(tmp_path):
    chat = fx.standard_chat()
    chat.script[f"MEETING: {fx.M2}"] = "this is not json"
    out, _ = run(chat, tmp=tmp_path)
    assert out.report["manifest"]["meetings"]["failed"] == [fx.M2]
    assert out.report["manifest"]["complete"] is False
    assert nodes(out.facts, "Feature")[ALERTS].props["discussed_not_implemented"] is None


def test_failed_session_blocks_only_features_it_mentions(tmp_path):
    chat = fx.standard_chat()
    chat.script["[20260310-0-b0001]"] = "not json"  # the RivalCo session: no feature mentioned there
    out, _ = run(chat, tmp=tmp_path)
    assert out.report["manifest"]["sessions"]["failed"] == [{"first_message": "20260310-0-b0001",
                                                              "reason": "invalid_json"}]
    assert nodes(out.facts, "Feature")[ALERTS].props["discussed_not_implemented"] is True
    chat2 = fx.standard_chat()
    chat2.script["[20260320-0-d0001]"] = "not json"  # PR#3 session: its text does not name a feature either
    out2, _ = run(chat2, tmp=tmp_path / "b")
    assert nodes(out2.facts, "Feature")[ALERTS].props["discussed_not_implemented"] is True


# ---------------------------------------------------------------- routing, manifest, splitting

def test_routing_sends_planning_always_and_develop_by_cue_and_never_bug_channels(res):
    out, chat = res
    prompts = "\n".join(c["user"] for c in chat.kind("session"))
    assert "[20260310-1-b0002]" in prompts  # planning
    assert "[20260320-0-d0001]" in prompts and "[20260404-0-f0004]" in prompts  # develop with PR refs
    assert "[20260322-0-e0001]" not in prompts and "[20260405-0-g0001]" not in prompts and "[20260312-0-b0003]" not in prompts
    assert "[20260406-0-h0001]" not in prompts and "slack_admin_bot" not in prompts
    s = out.report["manifest"]["sessions"]
    assert s["planning"] == {"total": 5, "extracted": 5} and s["other"]["routed"] == 4
    assert s["other"]["excluded"] == {"no_cue": 3}
    assert len(chat.kind("session")) == 9 and len(chat.kind("meeting")) == 2 and len(chat.kind("membership")) == 1


def test_long_session_is_split_at_message_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic, "SESSION_CHARS", 300)
    out, chat = run(tmp=tmp_path)
    firsts = [c["user"] for c in chat.kind("session") if "[20260305-0-a0001]" in c["user"]]
    assert firsts and "[20260305-3-a0004]" not in firsts[0]
    assert out.report["manifest"]["sessions"]["split_sessions"] >= 1


def test_competitor_registry_is_passed_to_later_sessions(res):
    _, chat = res
    later = [c["user"] for c in chat.kind("session") if "[20260315-0-c0001]" in c["user"]][0]
    assert "KNOWN COMPETITORS: RivalCo" in later
    earlier = [c["user"] for c in chat.kind("session") if "[20260305-0-a0001]" in c["user"]][0]
    assert "KNOWN COMPETITORS: RivalCo" in earlier  # named by its demo link before any session is read


# ---------------------------------------------------------------- cache, failures, estimate, graph integrity

def test_rerun_with_cache_makes_no_calls(tmp_path):
    out1, chat1 = run(tmp=tmp_path)
    assert len(chat1.calls) == 12
    out2, chat2 = run(tmp=tmp_path)
    assert chat2.calls == []
    assert out2.report["calls"]["live"] == 0 and out2.report["calls"]["cached"] == 12
    assert sorted(out1.facts.nodes, key=lambda n: n.id) == sorted(out2.facts.nodes, key=lambda n: n.id)


def test_truncated_output_is_retried_then_counted_as_a_failure(tmp_path):
    chat = fx.standard_chat()
    chat.script[f"MEETING: {fx.M2}"] = (fx.M2_ANSWER, 10_000)  # always at max_tokens: truncated twice
    out, _ = run(chat, tmp=tmp_path)
    assert out.report["manifest"]["meetings"]["failed"] == [fx.M2]
    assert out.report["extraction_failures"][0]["reason"] == "truncated"
    assert len([c for c in chat.calls if f"MEETING: {fx.M2}" in c["user"]]) == 2
    assert [c["max_tokens"] for c in chat.calls if f"MEETING: {fx.M2}" in c["user"]][1] == 2 * [
        c["max_tokens"] for c in chat.calls if f"MEETING: {fx.M2}" in c["user"]][0]


def test_estimate_counts_cache_misses_without_sending_anything(tmp_path):
    est = semantic.estimate_semantic(fx.records(), cache_root=tmp_path)
    assert est["meeting_calls"] == 2 and est["session_calls"] == 9 and est["membership_calls_max"] == 1
    assert est["cached"] == 0 and est["projected_usd"] > 0
    run(tmp=tmp_path)
    again = semantic.estimate_semantic(fx.records(), cache_root=tmp_path)
    assert again["session_calls"] == 0 and again["meeting_calls"] == 0 and again["cached"] >= 11


def test_every_edge_passes_check_edge_and_layers_merge_without_conflicts(res):
    out, _ = res
    for e in out.facts.edges:
        check_edge(e)
    rec = fx.records()
    merged = structural_facts_from_records(rec).merge(run_rules(rec)[0]).merge(out.facts)
    assert merged.conflicts == 0
    ids = {n.id for n in merged.nodes}
    dangling = [e for e in merged.edges if e.src_id not in ids or e.dst_id not in ids]
    assert dangling == []


def test_report_counts(res):
    out, _ = res
    r = out.report
    assert r["features"] == 2 and r["feature_events"]["held"] == 2 and r["feature_events"]["resumed"] == 1
    assert r["suggestions"] == 4 and r["contributions"]["lead"] == 1
    assert r["pr_membership"]["by_basis"] == {"meeting_title": 3, "batch": 3}
    assert r["prompt_version"] == "semantic-v2"
    assert r["extraction_failures"] == []


# ---------------------------------------------------------------- runner

def test_write_semantic_writes_facts_and_report_and_estimate_writes_nothing(tmp_path, capsys):
    from inherit.extract import run

    root = tmp_path / "root"
    est = run.write_semantic(root, "Mini", fx.records(), True, cache_root=tmp_path / "c")
    assert est["meeting_calls"] == 2 and not (root / "data").exists()
    assert "meeting calls 2 session calls 9" in capsys.readouterr().out
    r = run.write_semantic(root, "Mini", fx.records(), False, chat=fx.standard_chat(), cache_root=tmp_path / "c")
    d = root / "data" / "facts" / "Mini"
    facts = Facts.from_jsonl(d / "semantic.jsonl")
    assert {n.id for n in facts.nodes if n.label == "Feature"} == {ALERTS, PRED}
    assert json.loads((d / "semantic_report.json").read_text())["prompt_version"] == "semantic-v2"
    assert r["manifest"]["complete"] is True
    assert "complete True" in capsys.readouterr().out


def test_runner_semantic_estimate_from_the_cli(tmp_path, mini_bundle):
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    root_repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "root"
    shutil.copytree(mini_bundle, root / "data" / "bundle")
    (root / "src").mkdir()
    env = {"PYTHONPATH": str(root_repo / "src"), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-m", "inherit.extract.run", "--layer", "semantic", "--products", "mini",
                        "--repo-root", str(root), "--estimate"], capture_output=True, text=True, cwd=root_repo, env=env)
    assert r.returncode == 0, r.stderr
    assert "meeting calls" in r.stdout and "projected" in r.stdout and "total projected" in r.stdout
    assert not (root / "data" / "facts" / "mini" / "semantic.jsonl").exists()


def test_two_remarks_in_one_message_and_two_kinds_on_one_segment_do_not_collide(tmp_path):
    two = json.loads(fx.session_json(reviews=fx.REVIEWS))
    two["reviews"].append({**two["reviews"][0], "section": "Market Trends", "text": "Second remark",
                           "citations": [fx.cite("20260305-1-a0002", "advanced algorithms")]})
    chat = fx.standard_chat()
    chat.script["[20260305-1-a0002]"] = json.dumps(two)
    m1 = json.loads(fx.M1_ANSWER)
    m1["contributions"].append({"feature_ref": "f1", "segment_ref": f"{fx.M1}#5", "kind": "design",
                                "citations": [fx.cite(f"{fx.M1}#5", "use WebSockets for pushing alerts")]})
    chat.script[f"MEETING: {fx.M1}"] = json.dumps(m1)
    out, _ = run(chat, tmp=tmp_path)
    assert out.facts.conflicts == 0
    keys = sorted(e.key for e in edges(out.facts, "SUGGESTED") if e.src_id == f"Person:{fx.BEN}"
                  and e.dst_id == "Document:mini_market_research_report")
    assert keys == ["20260305-1-a0002", "20260305-1-a0002#2"]
    assert {e.props["kind"] for e in edges(out.facts, "CONTRIBUTED") if e.src_id == f"Person:{fx.BEN}"} == {"lead", "design"}


def test_a_name_that_is_never_called_a_competitor_is_not_one(tmp_path):
    bad = json.loads(fx.session_json(claims=fx.RIVAL_CLAIMS))
    for c in bad["claims"]:
        c["subject_ref"] = "Interesting"  # in the session text, but not introduced as a competitor and no demo link
    chat = fx.standard_chat()
    chat.script["[20260310-0-b0001]"] = json.dumps(bad)
    out, _ = run(chat, tmp=tmp_path)
    assert out.report["dropped"]["claims"]["subject_not_a_competitor"] == 2
    assert "Competitor:Interesting" not in nodes(out.facts, "Competitor")


def test_a_demo_share_message_states_no_claim(tmp_path):
    claim = dict(fx.RIVAL_CLAIMS[0], message_id="20260312-0-b0003",
                 citations=[fx.cite("20260312-0-b0003", "see how RivalCo works")])
    chat = fx.standard_chat()
    chat.script["[20260312-0-b0003]"] = fx.session_json(claims=[claim])
    # the share is routed only when another cue is present: a "competitor" word in the same message
    slack = [(i, a, t, tx.replace("see how", "a competitor demo: see how") if i == "20260312-0-b0003" else tx, c, n)
             for i, a, t, tx, c, n in fx.SLACK]
    out, _ = run(chat, fx.records(slack=slack), tmp_path)
    assert out.report["dropped"]["claims"]["demo_share_message"] == 1
    assert "Observation:Mini:20260312-0-b0003:0" not in nodes(out.facts, "Observation")


def test_withhold_needs_hold_wording_in_the_deciding_message(tmp_path):
    chat = fx.standard_chat()
    dec = json.loads(fx.session_json(pr_decisions=fx.PR3_DECISIONS))
    dec["pr_decisions"].append({"message_id": "20260320-2-d0003", "prs": ["3"], "action": "withhold", "reason": None,
                                "conditional": False, "citations": [fx.cite("20260320-2-d0003", "please revert it")]})
    chat.script["[20260320-0-d0001]"] = json.dumps(dec)
    out, _ = run(chat, tmp=tmp_path)
    assert out.report["dropped"]["pr_decisions"]["no_evidence_for_action"] == 1


def test_a_reason_the_session_does_not_support_is_dropped(tmp_path):
    chat = fx.standard_chat()
    dec = json.loads(fx.session_json(pr_decisions=fx.PR3_DECISIONS))
    dec["pr_decisions"][1]["reason"] = "security"  # the session never mentions security
    chat.script["[20260320-0-d0001]"] = json.dumps(dec)
    out, _ = run(chat, tmp=tmp_path)
    assert out.report["dropped"]["pr_decisions"]["reason_unsupported"] == 1
    held = [e for e in edges(out.facts, "ABOUT") if e.src_id == "Message:20260320-1-d0002"]
    assert held and held[0].props["reason"] is None


def test_model_scope_label_is_not_evidence(tmp_path):
    chat = fx.standard_chat()
    cs = json.loads(fx.session_json(claims=fx.CUST_CLAIMS))
    for c in cs["claims"]:
        c["release_scope"] = "current"  # the model calls it current; the session says the previous team
    chat.script["[20260316-0-c0003]"] = json.dumps(cs)
    out, _ = run(chat, tmp=tmp_path)
    o = nodes(out.facts, "Observation")["Observation:Mini:20260316-0-c0003:0"]
    assert o.props["release_scope"] == "previous" and o.props["scope_basis"] == "session_cue"


def test_known_feature_listed_without_evidence_is_a_reference_not_a_loss(tmp_path):
    again = json.loads(fx.M1_ANSWER)
    m2 = json.loads(fx.meeting_json(
        features=[{"local_id": "g1", "title": "real-time alerts", "details": [], "citations": []}],
        contributions=[{"feature_ref": "g1", "segment_ref": f"{fx.M2}#5", "kind": "support",
                        "citations": [fx.cite(f"{fx.M2}#5", "add a comparison table")]}]))
    chat = fx.standard_chat()
    chat.script[f"MEETING: {fx.M2}"] = json.dumps(m2)
    out, _ = run(chat, tmp=tmp_path)
    assert out.report["unresolved_feature_refs"] == 0 and out.report["manifest"]["complete"] is True
    assert (f"Person:{fx.DEE}", ALERTS) in pairs(out.facts, "CONTRIBUTED")
    assert again  # the first meeting is unchanged


def test_suggested_edge_keeps_the_source_message_text(res):
    out, _ = res
    e = next(x for x in edges(out.facts, "SUGGESTED") if x.key == "20260305-1-a0002")
    assert e.props["source_text"] == fx.R_BEN
