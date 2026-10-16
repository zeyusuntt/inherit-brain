"""Task 13f: deterministic review-relation and PR-association corrections. Synthetic records, no model calls."""

import json

import pytest

import semfx as fx
from inherit.data.records import Document
from inherit.extract import semantic
from inherit.extract.catalog import build_catalog
from inherit.extract.semantic import cited_remark, is_review, resolve_review_docs

P = "https://github.com/salesforce/Mini/pull/"
ALERTS = "Feature:Mini:real time alerts"
PRED = "Feature:Mini:predictive analytics"
MR = "mini_market_research_report"
FINAL = "final_mini_market_research_report"
LATEST = "latest_mini_market_research_report"
BOTH = {MR, FINAL}


def run(chat=None, rec=None, tmp=None):
    chat = chat or fx.standard_chat()
    return semantic.run_semantic(rec or fx.records(), chat=chat, cache_root=tmp), chat


def nodes(facts, label):
    return {n.id: n for n in facts.nodes if n.label == label}


def edges(facts, type_):
    return [e for e in facts.edges if e.type == type_]


# ---------------------------------------------------------------- self-commitments are not reviews

@pytest.mark.parametrize("text", [
    "I'll handle the technical feasibility section.",
    "I'll also gather some user feedback from our initial research to include in the document.",
    "I'll work on the competitive analysis.",
    "For the UX part, I'll focus on user personas and journey mapping.",
    "Maybe we can collaborate on the technical architecture?",
    "I can help draft the testing strategies section.",
    "I'll start drafting the risk management section.",
    "Let's sync up tomorrow to divide the tasks.",
])
def test_task_taking_gathering_and_collaboration_are_not_suggestions(text):
    assert is_review("suggestion", text) is False


@pytest.mark.parametrize("text", [
    "Maybe we can collaborate on the technical architecture?",
    "Can you take the lead on the UX section?",
])
def test_task_requests_are_not_review_questions(text):
    assert is_review("review_question", text) is False


@pytest.mark.parametrize("text", [
    "I'll add a section on regulatory risks to the Executive Summary.",
    "I'll start by adding a roadmap section with dates.",
    "We should include metrics in the Competitive Analysis.",
    "In the Market Overview, could we add more context on growth?",
    "It might be useful to include some personas or case studies.",
    "I'll handle the roadmap section. We need to outline the dates and owners.",
])
def test_concrete_content_suggestions_survive(text):
    assert is_review("suggestion", text) is True


def test_a_question_or_a_suggestion_counts_whatever_the_label_says():
    assert is_review("review_question", "Could the summary say more about pricing?") is True
    assert is_review("review_question", "The summary is fine.") is False
    assert is_review("review_question", "We should be careful not to make it too lengthy.") is True  # a suggestion
    assert is_review("suggestion", "In the Competitive Analysis, can we get a clarification on the metrics?") is True
    assert is_review("suggestion", "Thanks, everyone, for the feedback.") is False


def test_the_cited_remark_is_validated_not_the_whole_message():
    text = "Sure. I'll handle the Roadmap section. We need broader coverage later."
    cites = [{"record_id": "m1", "quote": "I'll handle the Roadmap section"}]
    assert cited_remark(text, cites, "m1", "Handle the roadmap section") == "I'll handle the Roadmap section."
    assert is_review("suggestion", text) is True  # the whole message has "need": the old, wrong test
    assert is_review("suggestion", cited_remark(text, cites, "m1", "Handle the roadmap section")) is False
    other = [{"record_id": "m2", "quote": "broader coverage"}]
    assert cited_remark(text, other, "m1") == text  # nothing cited in this record: the message stands in
    two = cites + [{"record_id": "m1", "quote": "need broader coverage"}]
    assert cited_remark(text, two, "m1") == "I'll handle the Roadmap section. We need broader coverage later."


def test_a_cited_task_statement_is_dropped_as_not_a_review_in_a_slack_session(tmp_path):
    msg = "20260601-1-z0002"
    extra = [("20260601-0-z0001", fx.ANN, "2026-06-01T09:00:00", "Let's review the Market Research Report today.") + fx.PLAN,
             (msg, fx.BEN, "2026-06-01T09:05:00",
              "Sure. I'll handle the Roadmap section. We need broader coverage later.") + fx.PLAN]
    chat = fx.standard_chat()
    chat.script[f"[{msg}]"] = fx.session_json(reviews=[{
        "message_id": msg, "document_ref": "Market Research Report", "kind": "suggestion", "section": "Roadmap",
        "text": "Handle the roadmap section", "citations": [fx.cite(msg, "I'll handle the Roadmap section")]}])
    out, _ = run(chat, fx.records(slack=fx.SLACK + extra), tmp_path)
    assert out.report["dropped"]["reviews"]["not_a_review"] == 1
    assert not [e for e in edges(out.facts, "SUGGESTED") if e.key == msg]


# ---------------------------------------------------------------- documents of another product

def _cat():
    return build_catalog(fx.records())


def test_a_thread_about_an_external_products_document_does_not_resolve_to_ours():
    cat = _cat()
    texts = ["Hi team, I wanted to discuss the X-Genie Market Research Report. It's quite interesting.",
             "They emphasize privacy and security in their document. We should ensure our security is up to par."]
    assert resolve_review_docs(cat, texts, "") == ([], "external_product_document")
    d, why = resolve_review_docs(cat, ["the Market Research Report of X-Genie"], "")
    assert d == [] and why == "external_product_document"
    d, why = resolve_review_docs(cat, ["Let's read the Market Research Report for X-Genie"], "")
    assert d == [] and why == "external_product_document"
    assert resolve_review_docs(cat, ["We read X-Genie's Market Research Report"], "")[1] == "external_product_document"


def test_an_explicit_switch_back_to_our_document_resolves_again():
    cat = _cat()
    texts = ["I wanted to discuss the X-Genie Market Research Report.", "Now back to our Market Research Report."]
    d, why = resolve_review_docs(cat, texts, "")
    assert {x.id for x in d} == BOTH and why == "version_ambiguous"
    d, why = resolve_review_docs(cat, ["Compare X-Genie Market Research Report with the Market Research Report for Mini"], "")
    assert {x.id for x in d} == BOTH


def test_our_product_name_or_a_sentence_opening_verb_is_not_an_external_subject():
    cat = _cat()
    for t in ("Please review the Mini Market Research Report", "Review Market Research Report today",
              "Update the Market Research Report for Mini", "Update Market Research Report", "the Market Research Report"):
        d, why = resolve_review_docs(cat, [t], "")
        assert {x.id for x in d} == BOTH and why == "version_ambiguous", t


def test_a_document_link_still_settles_the_document():
    d, why = resolve_review_docs(_cat(), [f"X-Genie Market Research Report? no, see {fx.FINAL_LINK}"], "")
    assert [x.id for x in d] == [FINAL] and why == "link"


# ---------------------------------------------------------------- release scope is not a document version

@pytest.mark.parametrize("text", [
    "Review the Market Research Report from the last release",
    "Feedback on the Market Research Report for the latest release",
    "Review the new version of the Market Research Report",
    "The current draft of the Market Research Report needs work",
    "Comments on the previous version of the Market Research Report",
])
def test_release_scope_words_do_not_pick_a_document_version(text):
    prev = {MR}
    d, why = resolve_review_docs(_cat(), [text], "", prev)
    assert {x.id for x in d} == BOTH and why == "version_ambiguous"


def test_a_session_about_the_latest_release_marks_every_version_ambiguous(tmp_path):
    msg = "20260601-1-z0002"
    extra = [("20260601-0-z0001", fx.ANN, "2026-06-01T09:00:00",
              "Let's review the Market Research Report with feedback from the latest release.") + fx.PLAN,
             (msg, fx.BEN, "2026-06-01T09:05:00", "We should add a pricing section.") + fx.PLAN]
    chat = fx.standard_chat()
    chat.script[f"[{msg}]"] = fx.session_json(reviews=[{
        "message_id": msg, "document_ref": "Market Research Report", "kind": "suggestion", "section": "Pricing",
        "text": "Add a pricing section", "citations": [fx.cite(msg, "We should add a pricing section")]}])
    out, _ = run(chat, fx.records(slack=fx.SLACK + extra), tmp_path)
    es = [e for e in edges(out.facts, "SUGGESTED") if e.key == msg]
    assert {e.dst_id for e in es} == {f"Document:{d}" for d in BOTH}
    assert all(e.props["version_ambiguous"] is True and e.props["doc_basis"] == "version_ambiguous" for e in es)


# ---------------------------------------------------------------- a meeting's chat link and later versions

def _rec_with_later_version():
    rec = fx.records()
    rec.documents.append(Document(LATEST, fx.ANN, "Market Research Report",
                                  "https://sf-internal.slack.com/archives/docs/" + LATEST, "2026-03-09T00:00:00",
                                  "Executive Summary: Mini is a product. Latest.", "- Add pricing."))
    return rec


def test_a_meeting_chat_link_names_the_shared_version_and_the_revisions_that_follow_it():
    cat = build_catalog(_rec_with_later_version())
    chat = f"2026-03-02T10:30:00\nAnn Lee: {fx.FINAL_LINK}"
    d, why = resolve_review_docs(cat, [chat, "Market Research Report"], "", later_versions=True)
    assert [x.id for x in d] == [FINAL, LATEST] and why == "link_and_later_versions"
    d, why = resolve_review_docs(cat, [chat, "Market Research Report"], "")  # a Slack session: the link settles it
    assert [x.id for x in d] == [FINAL] and why == "link"
    d, why = resolve_review_docs(build_catalog(fx.records()), [chat, "Market Research Report"], "", later_versions=True)
    assert [x.id for x in d] == [FINAL] and why == "link"  # nothing was revised after it


def test_a_meeting_remark_reaches_a_later_revision_only_when_its_feedback_shows_the_remark(tmp_path):
    rec = _rec_with_later_version()
    rec.documents[-1] = Document(LATEST, fx.ANN, "Market Research Report", rec.documents[-1].link, "2026-03-09T00:00:00",
                                 "Executive Summary: Mini is a product. Latest.",
                                 "- The Executive Summary needs more detail on the algorithms.")
    out, _ = run(rec=rec, tmp=tmp_path)
    shown = [e for e in edges(out.facts, "SUGGESTED") if e.key == f"{fx.M2}#6"]
    assert {e.dst_id for e in shown} == {f"Document:{FINAL}", f"Document:{LATEST}"}
    assert all(e.props["version_ambiguous"] is True and e.props["doc_basis"] == "link_and_later_versions" for e in shown)
    other = [e for e in edges(out.facts, "SUGGESTED") if e.key == f"{fx.M2}#5"]
    assert {e.dst_id for e in other} == {f"Document:{FINAL}"}  # nothing in the revision's feedback shows this remark
    assert all(e.props["version_ambiguous"] is False and e.props["doc_basis"] == "link" for e in other)
    older = [e for e in edges(out.facts, "SUGGESTED") if e.dst_id == f"Document:{MR}" and e.key.startswith(fx.M2)]
    assert older == []  # a version older than the shared one is not what the meeting looked at


def test_revision_evidence_is_the_share_of_distinctive_remark_words_in_the_feedback():
    from inherit.extract.semantic import shows_remark
    fb = "- The Executive Summary needs more detail on the algorithms."
    assert shows_remark("needs more detail on the algorithms", fb) == 1.0
    assert shows_remark("Add a comparison table", fb) == 0.0  # fewer than three distinctive words: no evidence
    assert shows_remark("pricing tiers and regional compliance rules", fb) == 0.0


# ---------------------------------------------------------------- explicit task-to-PR statements

PR11 = "Real-time Alerts Forecast Rollout"


def _rec_explicit(*statements, slack=()):
    rec = fx.records(slack=fx.SLACK + [(f"2026060{i}-0-y000{i}", fx.EVE, f"2026-06-0{i}T09:00:00", t) + fx.PLAN
                                       for i, t in enumerate(slack, 1)],
                     prs=[*fx.PRS, fx.pr(11, PR11, merged=True, state="closed", summary="Rolls out forecasts.")])
    segs = list(fx.M1_SEGS) + [(10 + i, "Ben Ray", s) for i, s in enumerate(statements)]
    rec.meetings[0] = fx.meeting(fx.M1, "2026-03-01T10:00:00", "product-planning",
                                 [fx.ANN, fx.BEN, fx.CY, fx.SAM1, fx.SAM2, fx.EVE], segs)
    return rec


def _chat_explicit(n_segments=1):
    m1 = json.loads(fx.M1_ANSWER)
    m1["development_links"].append({"feature_ref": "f2", "pr": PR11,
                                    "citations": [fx.cite(f"{fx.M1}#10", f"'{PR11}'")]})
    chat = fx.standard_chat()
    chat.script[f"MEETING: {fx.M1}"] = json.dumps(m1)
    return chat


STATE_PRED = f"The task here is predictive analytics. The related PR is titled '{PR11}'."
STATE_ALERTS = f"Our pending task is real-time alerts. The related PR is titled '{PR11}'."


def _about(out, n=11):
    return {(e.src_id, e.dst_id) for e in edges(out.facts, "ABOUT") if e.src_id == f"PR:{P}{n}"}


def test_an_explicit_task_statement_outranks_title_word_overlap(tmp_path):
    out, _ = run(_chat_explicit(), _rec_explicit(STATE_PRED), tmp_path)
    # the title shares "real-time alerts" with another feature; the segment says which task the PR implements
    assert _about(out) == {(f"PR:{P}11", PRED)}
    ib = {(e.src_id, e.dst_id) for e in edges(out.facts, "IMPLEMENTED_BY")}
    assert (PRED, f"PR:{P}11") in ib and (ALERTS, f"PR:{P}11") not in ib


def test_conflicting_explicit_statements_leave_the_association_unresolved(tmp_path):
    out, _ = run(_chat_explicit(), _rec_explicit(STATE_PRED, STATE_ALERTS), tmp_path)
    assert _about(out) == set()
    ib = {(e.src_id, e.dst_id) for e in edges(out.facts, "IMPLEMENTED_BY")}
    assert not [x for x in ib if x[1] == f"PR:{P}11"]
    assert f"{P}11" in out.report["pr_membership"]["unassigned"]
    assert out.report["pr_conflicts"][-1]["kept"] is None and out.report["pr_conflicts"][-1]["pr"] == f"{P}11"


def test_without_an_explicit_statement_the_context_rule_stands(tmp_path):
    # the title is spoken with no task named: the old C4 reading applies (words of the title and its neighbours)
    out, _ = run(_chat_explicit(), _rec_explicit(f"Also the '{PR11}' PR."), tmp_path)
    assert _about(out)  # still associated with some feature, as before


def test_the_membership_requests_do_not_change(tmp_path):
    out, chat = run(_chat_explicit(), _rec_explicit(STATE_PRED), tmp_path)
    users = [c["user"] for c in chat.kind("membership")]
    # PR 11 is already filed from the meeting, so only the other unmatched PRs go to the model, with the examples
    # computed from the previous reading of the meeting links
    assert users and PR11 not in "\n".join(users).split("PRS:")[-1]


# ---------------------------------------------------------------- fix round 1 (review of f6cdd55 and 23363ae)

@pytest.mark.parametrize("kind", ["suggestion", "review_question"])
@pytest.mark.parametrize("text", [
    "Can you take the integration part?",
    "Hi team, I want to discuss the Product Requirements Document. Let's brainstorm some ideas and outline the key "
    "sections we need to include.",
    "Does that work for everyone?",
    "Thoughts?",
])
def test_ownership_scheduling_and_openers_are_not_reviews_whatever_the_label(kind, text):
    assert is_review(kind, text) is False


@pytest.mark.parametrize("kind", ["suggestion", "review_question"])
def test_a_content_suggestion_or_question_is_a_review_whatever_the_label(kind):
    assert is_review(kind, "We should add a pricing section.") is True
    assert is_review(kind, "In the Competitive Analysis, can we get a clarification on the metrics?") is True


def test_adjacent_sentences_of_the_same_remark_stay_with_the_cited_one():
    text = "The current design is too complex for new users. We might need to simplify the layout. Thanks all, great chat."
    cites = [{"record_id": "m1", "quote": "too complex for new users"}]
    remark = "The current design is too complex for new users; simplify the layout"
    got = cited_remark(text, cites, "m1", remark)
    assert got == "The current design is too complex for new users. We might need to simplify the layout."
    assert is_review("suggestion", cited_remark(text, cites, "m1")) is False  # alone, the complaint carries no proposal
    assert is_review("suggestion", got) is True


def test_a_competitor_thread_worded_with_on_is_external_and_a_plain_topic_is_not():
    cat = _cat()
    d, why = resolve_review_docs(cat, ["Hi team, let's discuss the Market Research Report on X-Genie.",
                                       "How would our onboarding compare?"], "")
    assert d == [] and why == "external_product_document"  # talk of "our onboarding" is not a switch to our document
    d, why = resolve_review_docs(cat, ["Comments on the Market Research Report on Pricing"], "")
    assert {x.id for x in d} == BOTH


def test_an_explicit_slack_statement_outranks_title_overlap_too(tmp_path):
    rec = _rec_explicit(f"Also the '{PR11}' PR.", slack=[STATE_PRED])
    out, _ = run(_chat_explicit(), rec, tmp_path)
    assert _about(out) == {(f"PR:{P}11", PRED)}


def test_a_meeting_and_a_slack_statement_that_disagree_leave_the_pr_unassociated(tmp_path):
    rec = _rec_explicit(STATE_ALERTS, slack=[STATE_PRED])
    out, _ = run(_chat_explicit(), rec, tmp_path)
    assert _about(out) == set()
    assert out.report["pr_conflicts"][-1]["kept"] is None


@pytest.mark.parametrize("text", [
    "I'll handle the testing strategies section. We should aim to have a draft by next week. How does that sound?",
    "Could you schedule a meeting?",
    "Can you take the integration part?",
    "We should aim to have a draft by next week.",
    "I'll send the draft over by Friday.",
])
def test_workflow_requests_are_not_reviews(text):
    assert is_review("suggestion", text) is False
    assert is_review("review_question", text) is False


@pytest.mark.parametrize("text", [
    "Can you explain the authentication flow?",
    "Can you explain why the algorithm is quadratic?",
    "I'll add a section on MFA.",
    "Could you clarify the pricing assumptions?",
    "We should add a meeting-notes section.",
])
def test_requests_about_the_content_are_reviews(text):
    assert is_review("suggestion", text) is True
    assert is_review("review_question", text) is True


# ---------------------------------------------------------------- round-2 fixes: completion and merged-feature quarantine

from inherit.extract.semantic import completed  # noqa: E402


@pytest.mark.parametrize("text", [
    "We've reviewed and updated the OAuth 2.0 implementation to align with the latest security standards. "
    "Proper token handling and secure storage are in place, and we're compliant.",
    "The rollout is now in place.",
    "The migration has been completed.",
    "We upgraded the cache layer last week.",
])
def test_affirmative_completed_updates_count_as_completion(text):
    assert completed(text) is True


@pytest.mark.parametrize("text", [
    "The documentation needs to be updated.",
    "We will update the OAuth flow once QA signs off.",
    "The update is not in place yet.",
    "Is the secure storage in place?",
    "We plan to have it in place by Friday.",
])
def test_plans_negations_and_questions_are_not_completion(text):
    assert completed(text) is False


def test_a_completion_sentence_alone_does_not_create_an_implemented_event(tmp_path):
    # sentence-level reconciliation was removed: only the model's events (validated by completed()) count
    rec = fx.records()
    segs = list(fx.M1_SEGS) + [(10, "Ben Ray", "We've reviewed and updated the real-time alerts delivery. Everything is in place.")]
    rec.meetings[0] = fx.meeting(fx.M1, "2026-03-01T10:00:00", "product-planning",
                                 [fx.ANN, fx.BEN, fx.CY, fx.SAM1, fx.SAM2, fx.EVE], segs)
    out, _ = run(None, rec, tmp_path)
    assert nodes(out.facts, "Feature")[ALERTS].props["implemented"] is False
    assert not [e for e in nodes(out.facts, "FeatureEvent").values() if e.props["record_id"] == f"{fx.M1}#10"]


def test_a_model_implemented_event_with_a_completed_update_sentence_is_accepted(tmp_path):
    text = "We've reviewed and updated the real-time alerts delivery. Everything is in place."
    m1 = json.loads(fx.M1_ANSWER)
    m1["feature_events"].append({"feature_ref": "f1", "kind": "implemented", "reason": None, "pr": None,
                                 "segment_ref": f"{fx.M1}#10", "citations": [fx.cite(f"{fx.M1}#10", "We've reviewed and updated")]})
    chat = fx.standard_chat()
    chat.script[f"MEETING: {fx.M1}"] = json.dumps(m1)
    rec = fx.records()
    rec.meetings[0] = fx.meeting(fx.M1, "2026-03-01T10:00:00", "product-planning",
                                 [fx.ANN, fx.BEN, fx.CY, fx.SAM1, fx.SAM2, fx.EVE], list(fx.M1_SEGS) + [(10, "Ben Ray", text)])
    out, _ = run(chat, rec, tmp_path)
    f = nodes(out.facts, "Feature")[ALERTS].props
    assert f["implemented"] is True and f["discussed_not_implemented"] is False  # PR 3 for alerts is still open


def test_a_planned_update_does_not_make_the_feature_implemented(tmp_path):
    rec = fx.records()
    segs = list(fx.M1_SEGS) + [(10, "Ben Ray", "We will update the real-time alerts delivery next week.")]
    rec.meetings[0] = fx.meeting(fx.M1, "2026-03-01T10:00:00", "product-planning",
                                 [fx.ANN, fx.BEN, fx.CY, fx.SAM1, fx.SAM2, fx.EVE], segs)
    out, _ = run(None, rec, tmp_path)
    assert nodes(out.facts, "Feature")[ALERTS].props["implemented"] is False


ENUM = "We have three main tasks to discuss: real-time alerts, predictive analytics, and data exports."
PLACE = "Next, we have predictive analytics. Dee, you're assigned to build Predictive Model Evaluation. Can you confirm?"


def _rec_enum(*extra):
    rec = fx.records()
    segs = list(fx.M1_SEGS) + [(10 + i, "Ann Lee", t) for i, t in enumerate(extra)]
    rec.meetings[0] = fx.meeting(fx.M1, "2026-03-01T10:00:00", "product-planning",
                                 [fx.ANN, fx.BEN, fx.CY, fx.SAM1, fx.SAM2, fx.EVE], segs)
    return rec


def _wrong_membership():
    chat = fx.standard_chat()
    orig = chat.membership

    def wrong(user):
        data = json.loads(orig(user))
        fref = next(ln.split(":")[0] for ln in user.splitlines() if ln.startswith("F") and "alerts" in ln)
        for a in data["assignments"]:
            if a["quote"].startswith("Evaluate"):
                a["feature"] = fref  # the model files the PR under the sibling task "real-time alerts"
        return json.dumps(data)

    chat.membership = wrong
    return chat


def test_a_pr_placed_under_a_different_enumerated_sibling_task_loses_the_association(tmp_path):
    out, _ = run(_wrong_membership(), _rec_enum(ENUM, PLACE), tmp_path)
    assert _about(out, 6) == set()
    ib = {(e.src_id, e.dst_id) for e in edges(out.facts, "IMPLEMENTED_BY")}
    assert not [x for x in ib if x[1] == f"PR:{P}6"]
    q = out.report["pr_quarantined"]
    assert [(x["pr"], x["feature"], x["placed_under"]) for x in q] == [(f"{P}6", ALERTS, "predictive analytics")]


def test_without_a_placing_statement_the_association_stays(tmp_path):
    out, _ = run(_wrong_membership(), _rec_enum(ENUM), tmp_path)
    assert _about(out, 6) == {(f"PR:{P}6", ALERTS)}
    assert out.report["pr_quarantined"] == []


def test_a_placement_under_the_features_own_task_is_kept(tmp_path):
    out, _ = run(None, _rec_enum(ENUM, PLACE), tmp_path)  # standard membership already files PR 6 under predictive analytics
    assert _about(out, 6) == {(f"PR:{P}6", PRED)}
    assert out.report["pr_quarantined"] == []


PR12 = "Predictive Model Evaluation Status Cards"


def test_a_statement_naming_a_longer_pr_title_does_not_identify_the_shorter_one(tmp_path):
    rec = _rec_enum(ENUM, "Next, we have predictive analytics. Dee, you're assigned to build Predictive Model Evaluation "
                          "Status Cards. Can you confirm?")
    rec.prs.append(fx.pr(12, PR12, summary="Cards."))
    out, _ = run(_wrong_membership(), rec, tmp_path)
    assert _about(out, 6) == {(f"PR:{P}6", ALERTS)}  # the statement is about PR 12; nothing is said about PR 6
    assert out.report["pr_quarantined"] == []


def test_a_pr_number_or_link_identifies_the_pr_and_articles_in_a_title_are_ignored(tmp_path):
    out, _ = run(_wrong_membership(), _rec_enum(ENUM, "Next, we have predictive analytics. Dee, you're assigned to PR#6."),
                 tmp_path)
    assert _about(out, 6) == set()
    out2, _ = run(_wrong_membership(), _rec_enum(ENUM, "Next, we have predictive analytics. Dee, you're assigned to the "
                                                      "predictive model evaluation."), tmp_path / "b")
    assert _about(out2, 6) == set()


# ---------------------------------------------------------------- round-3 recall: type-level review discussions

def _brainstorm(opener, replies, day="20260601"):
    rows = [(f"{day}-0-q0000", fx.ANN, f"2026-06-01T09:00:00", opener) + fx.PLAN]
    for i, (who, text) in enumerate(replies, 1):
        rows.append((f"{day}-{i}-q000{i}", who, f"2026-06-01T09:{i:02d}:00", text) + fx.PLAN)
    return fx.records(slack=fx.SLACK + rows)


def _suggested_in(out, day="20260601"):
    return {(e.key, e.dst_id) for e in edges(out.facts, "SUGGESTED") if e.key.startswith(day)}


def test_a_brainstorm_about_a_document_type_reviews_every_version_for_replies_with_content_proposals(tmp_path):
    rec = _brainstorm("Hi team, I want to discuss the Market Research Report for Mini. Let's brainstorm some ideas!", [
        (fx.BEN, "Great initiative! We should ensure the report clearly outlines the pricing model."),
        (fx.DEE, "Sure thing, I'll draft a roadmap and share it by next week."),
        (fx.CY, "Absolutely. I'll focus on gathering feedback. We should include a section on user personas."),
        (fx.EVE, "Let's aim to have a draft ready by the end of next week. Thanks, everyone!")])
    out, _ = run(None, rec, tmp_path)
    got = _suggested_in(out)
    assert got == {(k, f"Document:{d}") for k in ("20260601-1-q0001", "20260601-3-q0003") for d in BOTH}
    es = [e for e in edges(out.facts, "SUGGESTED") if e.key == "20260601-3-q0003"]
    assert all(e.props["version_ambiguous"] is True and e.props["doc_basis"] == "version_ambiguous" for e in es)
    assert es[0].props["text"] == "We should include a section on user personas."
    assert out.report["type_level_rule_reviews"] == 2


def test_a_rejected_allocation_sentence_stays_rejected_while_its_content_proposal_is_accepted(tmp_path):
    rec = _brainstorm("Hi team, we need to nail down the key sections of the Market Research Report for Mini. Thoughts?", [
        (fx.BEN, "I'll handle the roadmap section. We need to outline the dates and owners.")])
    chat = fx.standard_chat()
    chat.script["[20260601-1-q0001]"] = fx.session_json(reviews=[{
        "message_id": "20260601-1-q0001", "document_ref": "Market Research Report", "kind": "suggestion", "section": None,
        "text": "Handle the roadmap section", "citations": [fx.cite("20260601-1-q0001", "I'll handle the roadmap section")]}])
    out, _ = run(chat, rec, tmp_path)
    es = [e for e in edges(out.facts, "SUGGESTED") if e.key == "20260601-1-q0001"]
    assert {e.dst_id for e in es} == {f"Document:{d}" for d in BOTH}
    assert {e.props["text"] for e in es} == {"We need to outline the dates and owners."}  # not the allocation sentence
    assert all(e.props["review_origin"] == "type_level_rule" for e in es)


def test_a_message_with_an_accepted_model_review_is_not_read_again(tmp_path):
    rec = _brainstorm(OPEN, [(fx.BEN, "We should add a pricing section. We should also include a roadmap section.")])
    chat = fx.standard_chat()
    chat.script["[20260601-1-q0001]"] = fx.session_json(reviews=[{
        "message_id": "20260601-1-q0001", "document_ref": "Market Research Report", "kind": "suggestion", "section": "Pricing",
        "text": "Add a pricing section", "citations": [fx.cite("20260601-1-q0001", "We should add a pricing section")]}])
    out, _ = run(chat, rec, tmp_path)
    es = [e for e in edges(out.facts, "SUGGESTED") if e.key == "20260601-1-q0001"]
    assert es and all(e.props["review_origin"] == "model" for e in es)
    assert out.report["type_level_rule_reviews"] == 0


def test_a_brainstorm_about_another_products_document_or_without_a_type_adds_nothing(tmp_path):
    ext = _brainstorm("Hi team, I want to discuss the X-Genie Market Research Report. Let's brainstorm!",
                      [(fx.BEN, "We should highlight their privacy story.")], day="20260602")
    out, _ = run(None, ext, tmp_path / "a")
    assert _suggested_in(out, "20260602") == set()
    plain = _brainstorm("Hi team, let's brainstorm the launch plan!", [(fx.BEN, "We should include a launch checklist.")],
                        day="20260603")
    out2, _ = run(None, plain, tmp_path / "b")
    assert _suggested_in(out2, "20260603") == set()


# ---------------------------------------------------------------- round-3 review: admission, anchoring, gate, provenance

OPEN = "Hi team, I want to discuss the Market Research Report for Mini. Let's brainstorm some ideas!"


@pytest.mark.parametrize("reply", [
    "I'll coordinate with the engineering team to ensure our timelines are realistic. @eid_439a052b, let's sync up tomorrow?",
    "Thanks, George! I'll update everyone on the QA progress by mid-week. Let's do this!",
    "Sounds like a solid plan, @Bob Garcia. Let's get these changes implemented and ensure Mini is on the right track.",
    "Those suggestions will add value.",
    "It will add credibility to our vision.",
    "We should make sure everyone is staffed for the work.",
])
def test_workflow_and_evaluative_replies_are_not_read_as_reviews(tmp_path, reply):
    out, _ = run(None, _brainstorm(OPEN, [(fx.BEN, reply)]), tmp_path)
    assert _suggested_in(out) == set()


def test_concrete_additions_are_still_read(tmp_path):
    out, _ = run(None, _brainstorm(OPEN, [(fx.BEN, "I'll add a section on regulatory risks to the report."),
                                         (fx.DEE, "We should explain how pricing tiers work.")]), tmp_path)
    assert {k for k, _ in _suggested_in(out)} == {"20260601-1-q0001", "20260601-2-q0002"}


def test_the_openers_document_type_is_the_only_target(tmp_path):
    rec = _brainstorm("Hi team, I want to discuss the Market Research Report for Mini. Let's brainstorm!", [
        (fx.BEN, "We should include a comparison table using our Product Vision Document as a source."),
        (fx.DEE, "We should include a comparison table of the competitors.")])
    out, _ = run(None, rec, tmp_path)
    got = _suggested_in(out)
    assert {k for k, _ in got} == {"20260601-2-q0002"}  # the first reply turns to another document type: skipped
    assert {d for _, d in got} == {f"Document:{d}" for d in BOTH}


def test_a_cancelled_review_opener_starts_no_discussion(tmp_path):
    rec = _brainstorm("Hi team, we are cancelling the Market Research Report review; instead let's talk QA staffing.",
                      [(fx.BEN, "We should include a staffing table.")])
    out, _ = run(None, rec, tmp_path)
    assert _suggested_in(out) == set()


def test_edges_carry_their_origin(tmp_path):
    rec = _brainstorm(OPEN, [(fx.BEN, "We should include a section on pricing.")])
    out, _ = run(None, rec, tmp_path)
    rule = [e for e in edges(out.facts, "SUGGESTED") if e.key == "20260601-1-q0001"]
    assert rule and all(e.props["review_origin"] == "type_level_rule" for e in rule)
    model = [e for e in edges(out.facts, "SUGGESTED") if e.key == "20260305-1-a0002"]
    assert model and all(e.props["review_origin"] == "model" for e in model)


R4_ACCEPT = [
    "@eid_469598db Sounds good, Hannah! I think we should start with defining the core functionalities of the AI engine. "
    "What exactly do we want it to recommend based on customer conversations?",
    "Agreed, George. We should also consider how the AI will integrate with existing CRM systems. That could be a key section in the PRD.",
    "From a QA perspective, we need to ensure we have clear testing criteria for the recommendations. Maybe we can include a section on performance metrics?",
    "Hey all, don't forget about the user experience! We need to make sure the recommendations are easily accessible and actionable for the sales reps. I'll draft some initial thoughts on the UX part.",
    "Good point, Charlie! We should also think about how to handle edge cases where the AI might not have enough data to make a recommendation.",
    "From a UX perspective, the 'User Experience' section is solid, but I think we should emphasize user testing. Adding a line about iterative testing with sales teams could highlight our focus on usability and continuous improvement.",
    "Hey, sounds good! I think we should start with a section on the current market landscape for AI-driven productivity tools. What do you think?",
    "For the technical section, we should explain how the AI suggests actions in Slack. Maybe a flowchart or diagram would help? Thoughts?",
    "Let's also include a section on potential use cases and benefits for different user personas. This will help in marketing the product effectively.",
    "Let's not forget about potential risks and how we plan to mitigate them. It's always good to be prepared.",
]
R4_REJECT = [
    "Great idea, George! A visual representation will definitely make it clearer. I'll draft something and share it here for feedback.",
    "I can help draft the section on competitive analysis. It'll be important to show how we stand out from other lead scoring solutions.",
]


@pytest.mark.parametrize("reply", R4_ACCEPT)
def test_round_4_content_proposals_are_read_at_type_level(tmp_path, reply):
    out, _ = run(None, _brainstorm(OPEN, [(fx.BEN, reply)]), tmp_path)
    assert {k for k, _ in _suggested_in(out)} == {"20260601-1-q0001"}


@pytest.mark.parametrize("reply", R4_REJECT)
def test_round_4_allocation_and_endorsement_stay_unextracted(tmp_path, reply):
    out, _ = run(None, _brainstorm(OPEN, [(fx.BEN, reply)]), tmp_path)
    assert _suggested_in(out) == set()


# ---------------------------------------------------------------- round 5: author acts, summaries and openers

@pytest.mark.parametrize("who,reply", [
    (fx.ANN, "Good catch, Emma. I'll add a note on our initial target regions for expansion. Thanks for the input, everyone!"),
    (fx.ANN, "Thanks for the feedback! Let's incorporate these changes into the document."),
    (fx.ANN, "Great suggestions, Bob. Let's update the 'System Architecture' section to highlight modularity and resource "
             "allocation for CRM integrations."),
    (fx.BEN, "Great suggestions, everyone! Here's a quick summary: - Enhance the training data strategy. - Incorporate "
             "user feedback and UX research for UI improvements. - Expand the modular architecture."),
    (fx.BEN, "To summarize, we'll add sections on pricing, onboarding and regional expansion."),
])
def test_author_acknowledgements_and_summaries_are_not_reviews(tmp_path, who, reply):
    out, _ = run(None, _brainstorm(OPEN, [(who, reply)]), tmp_path)
    assert _suggested_in(out) == set()


def test_an_authors_independent_content_proposal_is_still_a_review(tmp_path):
    out, _ = run(None, _brainstorm(OPEN, [(fx.ANN, "We should add a section on pricing tiers.")]), tmp_path)
    assert {k for k, _ in _suggested_in(out)} == {"20260601-1-q0001"}


def test_a_model_reviewed_authors_ack_is_dropped_too(tmp_path):
    msg = "20260601-1-q0001"
    chat = fx.standard_chat()
    chat.script[f"[{msg}]"] = fx.session_json(reviews=[{
        "message_id": msg, "document_ref": "Market Research Report", "kind": "suggestion", "section": None,
        "text": "Add a note on regions", "citations": [fx.cite(msg, "I'll add a note on our initial target regions")]}])
    rec = _brainstorm(OPEN, [(fx.ANN, "Good catch, Emma. I'll add a note on our initial target regions for expansion.")])
    out, _ = run(chat, rec, tmp_path)
    assert _suggested_in(out) == set()
    assert out.report["dropped"]["reviews"]["author_response"] == 1


def test_a_session_opener_without_its_own_content_proposal_is_not_a_review(tmp_path):
    opener = "Hi team, I want to discuss about the Market Research Report for Mini. We need to outline the key sections and assign responsibilities."
    first = "20260601-0-q0000"
    chat = fx.standard_chat()
    chat.script[f"[{first}]"] = fx.session_json(reviews=[{
        "message_id": first, "document_ref": "Market Research Report", "kind": "suggestion", "section": None,
        "text": "Outline the key sections", "citations": [fx.cite(first, "We need to outline the key sections and assign responsibilities")]}])
    out, _ = run(chat, _brainstorm(opener, [(fx.BEN, "We should include a section on pricing.")]), tmp_path)
    assert {k for k, _ in _suggested_in(out)} == {"20260601-1-q0001"}
    assert out.report["dropped"]["reviews"]["opener"] == 1


def test_an_opener_with_its_own_content_proposal_is_kept(tmp_path):
    opener = "Hi team, let's review the Market Research Report for Mini. I think we should add a section on competitors' pricing."
    first = "20260601-0-q0000"
    chat = fx.standard_chat()
    chat.script[f"[{first}]"] = fx.session_json(reviews=[{
        "message_id": first, "document_ref": "Market Research Report", "kind": "suggestion", "section": None,
        "text": "Add a competitor pricing section", "citations": [fx.cite(first, "we should add a section on competitors' pricing")]}])
    out, _ = run(chat, _brainstorm(opener, [(fx.BEN, "Agreed.")]), tmp_path)
    assert first in {k for k, _ in _suggested_in(out)}
