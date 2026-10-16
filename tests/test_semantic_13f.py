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
