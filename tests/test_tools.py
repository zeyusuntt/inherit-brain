"""Task 8: the fixed Cypher tool catalogue, the output guard and the trace. Each tool runs on both servers."""

import importlib.util
import json

import pytest

from toolfx import (ALICE, BOB, CARA, DAN, EVE, FINN, GUS, P, SAM1, SAM2, agent_graph, clean, load, pr, tg, tool_facts,
                    tsrv, tsrv_both)
from inherit.contracts import TraceOp
from inherit.ontology import check_edge
from inherit.tools import catalog, depth1
from inherit.tools.catalog import TOOLS, ToolError, ToolResult
from inherit.tools.guard import guard, guard_evidence
from inherit.tools.trace import Trace


def call(g, tool, **params) -> ToolResult:
    return TOOLS[tool].run(g, P, **params)


def answers(res) -> list:
    return sorted(r["answer"] for r in res.rows)


def by(res, key="answer") -> dict:
    return {r[key]: r for r in res.rows}


def counts(res) -> dict:
    return {r["answer"]: r["count"] for r in res.rows}


def links(*ns) -> list:
    return sorted(pr(n) for n in ns)


# ---------------------------------------------------------------- the fixture and the catalogue
def test_fixture_edges_pass_the_ontology():
    for e in tool_facts().edges:
        check_edge(e)


def test_catalogue_has_the_ten_tools_and_a_model_facing_contract():
    assert sorted(TOOLS) == sorted(["resolve_person", "resolve_entity", "doc_people", "bug_board", "fix_trail",
                                    "feature_fate", "who_said", "pr_board", "bridges", "if_leaves"])
    for name, t in TOOLS.items():
        assert t.name == name and t.description.strip()
        assert "Example:" in t.description, name  # one short routing example each
        props = t.params_schema["properties"]
        assert "cursor" in props, name  # every tool pages
        assert t.params_schema["additionalProperties"] is False
        for p in props.values():
            assert "type" in p and "description" in p
    assert TOOLS["if_leaves"].kind == "whatif"
    assert {t.kind for n, t in TOOLS.items() if n != "if_leaves"} == {"readonly"}
    assert TOOLS["bug_board"].params_schema["properties"]["status"]["enum"] == [
        "unresolved", "resolved", "active", "reopened", "no_fix_needed", "any"]
    assert TOOLS["bug_board"].params_schema["properties"]["rank"]["default"] == "all"


def test_bad_parameters_are_refused_with_a_model_safe_message(tg):
    for tool, params in (("bug_board", {"status": "closed"}), ("bug_board", {"colour": "red"}),
                         ("bug_board", {"status": 3}), ("who_said", {"subject": "competitor", "role": "Janitor"}),
                         ("pr_board", {"approval": "maybe"}), ("bridges", {"top_k": 0}),
                         ("bug_board", {"status": "unresolved", "actor": "assignee", "return": "companies"})):
        with pytest.raises(ToolError):
            call(tg, tool, **params)


def test_a_graph_of_another_product_is_refused(tg):
    with pytest.raises(ToolError, match="product"):
        TOOLS["bug_board"].run(tg, "other", status="any")


def test_tools_only_use_graph_ro_query(tg):
    class OnlyRO:
        name = tg.name

        def __init__(self, g):
            self._g = g

        def ro_query(self, q, params=None):
            return self._g.ro_query(q, params)

    res = call(OnlyRO(tg), "bug_board", status="unresolved")  # an AttributeError on .query would fail here
    assert answers(res) == ["BlueWave"]


# ---------------------------------------------------------------- resolve_person
def test_resolve_person_keeps_namesakes_ambiguous(tg):
    res = call(tg, "resolve_person", name="Sam Kim")
    assert answers(res) == sorted([SAM1, SAM2])
    assert all(r["ambiguous"] and r["in_directory"] for r in res.rows)
    assert {r["role"] for r in res.rows} == {"UX Researcher", "Marketing Manager"}


def test_resolve_person_unique_name_and_unknown_name(tg):
    res = call(tg, "resolve_person", name="alice  ng")
    assert answers(res) == [ALICE] and res.rows[0]["ambiguous"] is False and res.rows[0]["name"] == "Alice Ng"
    assert call(tg, "resolve_person", name="Zed Quinn").rows == []


def test_resolve_person_within_a_meeting_uses_the_task5_resolution(tg):
    res = call(tg, "resolve_person", name="Sam Kim", meeting="m1")
    assert answers(res) == sorted([SAM1, SAM2]) and all(r["ambiguous"] for r in res.rows)
    assert all("m1#1" in r["evidence"] for r in res.rows)
    one = call(tg, "resolve_person", name="Alice Ng", meeting="m1")
    assert answers(one) == [ALICE] and one.rows[0]["ambiguous"] is False
    assert call(tg, "resolve_person", name="Cara Diaz", meeting="m1").rows == []  # not a speaker there


# ---------------------------------------------------------------- resolve_entity
def test_resolve_entity_exact_matches_on_names_titles_aliases_links_and_keys(tg):
    r = call(tg, "resolve_entity", text="techcorp")
    assert answers(r) == ["Customer:CUST-0001", "Customer:CUST-0002"]
    assert all(x["label"] == "Customer" and x["match"] == "exact" and x["name"] == "TechCorp" for x in r.rows)
    assert answers(call(tg, "resolve_entity", text="Login  crash")) == ["Bug:b1"]
    assert answers(call(tg, "resolve_entity", text="single sign-on")) == ["Feature:fE"]
    assert answers(call(tg, "resolve_entity", text=pr(10))) == [f"PR:{pr(10)}"]
    assert answers(call(tg, "resolve_entity", text="Rival")) == ["Competitor:Rival Corp"]
    assert answers(call(tg, "resolve_entity", text="Market Research Report")) == [
        "Document:latest_mrr", "Document:odd_mrr", "Document:prox_mrr"]
    assert answers(call(tg, "resolve_entity", text="Bug:b2")) == ["Bug:b2"]


def test_resolve_entity_kinds_filter_fulltext_fallback_and_no_match(tg):
    assert answers(call(tg, "resolve_entity", text="techcorp", kinds=["pr"])) == []
    fb = call(tg, "resolve_entity", text="mode flicker")
    assert answers(fb) == ["Bug:b4"] and fb.rows[0]["match"] == "fulltext"
    assert call(tg, "resolve_entity", text="unicorn").rows == []  # one word: no fallback
    assert call(tg, "resolve_entity", text="purple unicorn parade").rows == []


# ---------------------------------------------------------------- bug_board
def board(g, **kw):
    kw.setdefault("rank", "max")
    return call(g, "bug_board", **kw)


def test_company_rankings_group_by_customer_company_not_contact_id(tg):
    assert counts(board(tg, status="unresolved")) == {"BlueWave": 2}
    assert counts(board(tg, status="resolved")) == {"TechCorp": 3}  # b1 + b6 (CUST-0001) + b2 (CUST-0002)
    assert counts(board(tg, status="reopened")) == {"TechCorp": 2}
    assert counts(board(tg, status="active")) == {"TechCorp": 2}
    r = board(tg, status="resolved")
    assert r.rows[0]["bugs"] == ["Bug:b1", "Bug:b2", "Bug:b6"] and r.rows[0]["name"] == "TechCorp"


def test_rank_max_returns_every_tie_and_rank_all_everything(tg):
    assert counts(board(tg, status="no_fix_needed")) == {"BlueWave": 1, "CloudSync": 1}
    allr = call(tg, "bug_board", status="any")  # default rank is all
    assert counts(allr) == {"TechCorp": 3, "BlueWave": 2, "CloudSync": 1}
    assert [r["answer"] for r in allr.rows] == ["TechCorp", "BlueWave", "CloudSync"]  # count desc


def test_status_predicates_overlap_a_reopened_bug_with_a_fix_pr_is_active_and_reopened_not_unresolved(tg):
    rows = {r["id"]: r for r in call(tg, "bug_board", **{"status": "any", "return": "bugs"}).rows}
    f = rows["Bug:b2"]["flags"]
    assert f == {"unresolved": False, "resolved": True, "active": True, "reopened": True, "no_fix_needed": False}
    assert rows["Bug:b3"]["flags"]["no_fix_needed"] and rows["Bug:b3"]["flags"]["unresolved"]
    assert not rows["Bug:b3"]["flags"]["active"]


def test_component_filter_matches_question_wording(tg):
    r = call(tg, "bug_board", status="any", component="backend processing and task queue management")
    assert counts(r) == {"TechCorp": 2}
    r2 = call(tg, "bug_board", status="resolved", component="Backend Processing & Task Queue Management",
              actor="resolver")
    assert counts(r2) == {ALICE: 1, BOB: 1}
    assert call(tg, "bug_board", status="any", component="no such area").rows == []


def test_resolver_assignee_dismisser_actors(tg):
    assert counts(board(tg, status="resolved", actor="resolver")) == {ALICE: 3}
    assert counts(board(tg, status="unresolved", actor="assignee")) == {BOB: 1, ALICE: 1}  # current assignees, tie
    assert counts(board(tg, status="active", actor="assignee")) == {ALICE: 3}
    assert counts(board(tg, status="no_fix_needed", actor="dismisser")) == {CARA: 2}
    top = board(tg, status="resolved", actor="resolver").rows[0]
    assert top["name"] == "Alice Ng" and top["role"] == "Software Engineer" and top["evidence"]


def test_now_reopened_attribution_joins_the_action_to_a_later_reopening(tg):
    # b5: dismissed 06-16, reopened 06-30, resolved with PR4 on 07-05 (after the reopening): only the dismissal counts.
    assert counts(board(tg, status="reopened", actor="dismisser")) == {CARA: 1}  # not b3 (never reopened)
    # fix PRs resolved before a reopening: PR2 + PR3 (b2, Bob) and PR5 (b6, Alice); PR4 is after the reopening.
    r = call(tg, "bug_board", status="reopened", actor="fix_pr_author")
    assert counts(r) == {BOB: 2, ALICE: 1}
    assert board(tg, status="reopened", actor="fix_pr_author").rows[0]["answer"] == BOB
    assert by(r)[ALICE]["prs"] == [pr(5)]


def test_fix_pr_author_counts_distinct_prs_other_actors_count_bugs(tg):
    r = call(tg, "bug_board", status="resolved", actor="fix_pr_author")
    assert counts(r) == {ALICE: 3, BOB: 3}  # Bob: PR2 and PR3 on one bug, plus PR17
    assert by(r)[BOB]["bugs"] == ["Bug:b1", "Bug:b2"]
    rr = call(tg, "bug_board", status="resolved", actor="resolver")
    assert counts(rr) == {ALICE: 3, BOB: 1}  # one count per bug


def test_return_bugs_lists_unresolved_and_dismissed_bugs_without_any_pr(tg):
    r = call(tg, "bug_board", status="unresolved", customer="BlueWave", **{"return": "bugs"})
    assert answers(r) == ["Avatar upload fails", "Dark mode flicker"]
    b = by(r, "id")["Bug:b3"]
    assert b["impact_text"] == "Avatar upload fails hurts BlueWave." and b["component"] == "Authentication"
    assert b["companies"] == ["BlueWave"] and b["refs"][0] == "Bug:b3" and b["evidence"]
    by_id = call(tg, "bug_board", status="no_fix_needed", customer="CUST-0004", **{"return": "bugs"})
    assert answers(by_id) == ["Report email bounce"]


def test_customer_filter_by_company_name_and_unknown_customer(tg):
    assert counts(call(tg, "bug_board", status="resolved", customer="TechCorp")) == {"TechCorp": 3}
    assert call(tg, "bug_board", status="any", customer="Nobody Inc").rows == []


def test_as_of_uses_the_status_udf(tg):
    a = call(tg, "bug_board", status="active", as_of="2026-06-15T00:00:00")
    assert counts(a) == {"TechCorp": 2, "BlueWave": 2, "CloudSync": 1}  # b6 is not reported yet
    r = call(tg, "bug_board", status="resolved", as_of="2026-06-25T00:00:00")
    assert counts(r) == {"TechCorp": 2}  # b1 and b2; b6 resolves at 09:00 on 06-25
    assert counts(call(tg, "bug_board", status="no_fix_needed", as_of="2026-07-30T00:00:00")) == {"BlueWave": 1}
    with pytest.raises(ToolError):
        call(tg, "bug_board", status="unresolved", as_of="2026-06-25T00:00:00")


# ---------------------------------------------------------------- fix_trail
def test_fix_trail_customer_to_bugs_to_pr_links_to_authors(tg):
    r = call(tg, "fix_trail", customer="TechCorp")
    assert answers(r) == links(1, 2, 3, 5, 17)
    row = by(r)[pr(2)]
    assert row["bug"] == "Bug:b2" and row["author"] == BOB and row["company"] == "TechCorp"
    assert row["refs"] == [f"PR:{pr(2)}", "Bug:b2", f"Person:{BOB}", "Customer:CUST-0002"]
    assert row["evidence"]


def test_fix_trail_engineer_bug_and_name_filters(tg):
    assert answers(call(tg, "fix_trail", customer="TechCorp", engineer=BOB)) == links(2, 3, 17)
    assert answers(call(tg, "fix_trail", customer="techcorp", engineer="Bob Roy")) == links(2, 3, 17)
    assert answers(call(tg, "fix_trail", bug="Login crash")) == links(1, 17)
    assert answers(call(tg, "fix_trail", bug="Bug:b5")) == links(4)
    assert call(tg, "fix_trail", customer="BlueWave").rows == []  # its bugs have no fix PR
    assert call(tg, "fix_trail", customer="Nobody Inc").rows == []


# ---------------------------------------------------------------- pr_board
def test_pr_board_selectors_are_product_repo_only(tg):
    a = call(tg, "pr_board", feature="predictive analytics", approval="approved")
    assert answers(a) == links(10, 11)  # PR20 is approved and about the feature but an external look-alike
    assert answers(call(tg, "pr_board", feature="integrating predictive analytics", approval="unapproved")) == links(12)
    assert answers(call(tg, "pr_board", feature="Feature:fA")) == links(10, 11, 12)
    assert call(tg, "pr_board", feature="no such feature").rows == []
    r = by(a)[pr(10)]
    assert r["approval"] == "approved" and r["authors"][0]["eid"] == ALICE
    assert {f["id"] for f in r["features"]} == {"Feature:fA", "Feature:fB"} and r["evidence"]


def test_pr_board_need_merge_and_need_close_share_conditions_and_differ_by_hold(tg):
    assert answers(call(tg, "pr_board", feature="predictive analytics", state="need_merge")) == links(10)
    assert answers(call(tg, "pr_board", feature="predictive analytics", approval="approved",
                        state="need_close")) == links(11)
    assert answers(call(tg, "pr_board", feature="predictive analytics", approval="unapproved",
                        state="need_close")) == []


def test_pr_board_reverted_returns_the_original_target_and_filters_the_reason(tg):
    r = call(tg, "pr_board", feature="Dark mode", state="reverted", revert_reason="product_direction")
    assert answers(r) == links(13)  # not the revert PR 14
    assert r.rows[0]["reverts"][0]["reason"] == "product_direction"
    assert r.rows[0]["reverts"][0]["by"] == pr(14)
    assert answers(call(tg, "pr_board", feature="Dark mode", state="reverted", revert_reason="introduced_bugs")) == []
    assert answers(call(tg, "pr_board", feature="Export v2", state="reverted",
                        revert_reason="introduced_bugs")) == links(15)
    assert answers(call(tg, "pr_board", state="reverted")) == links(13, 15, 22, 24)  # any reason, applied or not


def test_pr_board_kind_separates_feature_development_from_bug_fixes(tg):
    assert answers(call(tg, "pr_board", kind="bug_fix")) == links(1, 2, 3, 4, 5, 17)
    dev = call(tg, "pr_board", kind="feature_development")
    assert pr(17) not in answers(dev)  # FIXES takes precedence over ABOUT
    assert pr(20) not in answers(dev) and pr(14) not in answers(dev)
    assert pr(10) in answers(dev) and pr(21) in answers(dev)


def test_pr_board_people_count_distinct_prs_with_all_ties_at_the_top(tg):
    r = call(tg, "pr_board", kind="feature_development", approval="approved", **{"return": "people", "rank": "max"})
    assert counts(r) == {ALICE: 3}  # PR10 (two features) counts once
    assert by(r)[ALICE]["prs"] == links(10, 11, 15)
    u = call(tg, "pr_board", kind="feature_development", approval="unapproved", **{"return": "people", "rank": "max"})
    assert counts(u) == {BOB: 1, CARA: 1}
    allr = call(tg, "pr_board", kind="feature_development", approval="approved", **{"return": "people"})
    assert counts(allr)[BOB] == 1 and len(allr.rows) == 5


# ---------------------------------------------------------------- feature_fate
def test_feature_fate_unimplemented_features_and_their_people(tg):
    f = call(tg, "feature_fate", fate="unimplemented")
    assert answers(f) == ["Voice assistant"] and f.rows[0]["details"] == ["Hands-free report queries"]
    p = call(tg, "feature_fate", fate="unimplemented", **{"return": "people"})
    assert answers(p) == sorted([DAN, EVE]) and by(p)[DAN]["kinds"] == ["lead"]


def test_feature_fate_reverted_by_reason_needs_an_applied_revert(tg):
    sec = call(tg, "feature_fate", fate="reverted", reason="security")
    assert answers(sec) == ["Third-party scanner"]  # via an applied REVERTS; Gamification's revert was not applied
    assert sec.rows[0]["reasons"] == ["security"]
    assert answers(call(tg, "feature_fate", fate="reverted", reason="product_direction")) == ["Dark mode"]
    assert answers(call(tg, "feature_fate", fate="reverted", reason="introduced_bugs")) == ["Export v2"]
    assert answers(call(tg, "feature_fate", fate="reverted")) == ["Dark mode", "Export v2", "Third-party scanner"]


def test_feature_fate_delayed_then_added_held_and_implemented_keep_history(tg):
    d = call(tg, "feature_fate", fate="delayed_then_added", reason="product_direction")
    # selected from the derived property: Offline mode was held and resumed without implementation evidence
    assert answers(d) == ["Offline mode", "SSO login"]
    assert [e["kind"] for e in by(d)["SSO login"]["events"]] == ["held", "resumed", "implemented"]
    assert by(d)["Offline mode"]["fates"] == ["delayed_then_added", "held"]
    assert answers(call(tg, "feature_fate", fate="held")) == [
        "Blockchain audit", "Offline mode", "SSO login"]  # derived: any hold
    imp = call(tg, "feature_fate", fate="implemented", topic="export")
    assert answers(imp) == ["Export v2"]  # added, then reverted: the implementation history is kept
    assert [e["kind"] for e in imp.rows[0]["events"]][-1] == "reverted"
    assert answers(call(tg, "feature_fate", fate="implemented")) == sorted(
        ["Dark mode", "Export v2", "SSO login", "Third-party scanner", "Gamification"])


def test_feature_fate_topic_alias_and_prs(tg):
    assert answers(call(tg, "feature_fate", topic="single sign-on")) == ["SSO login"]
    assert answers(call(tg, "feature_fate", topic="dark mode")) == ["Dark mode"]
    assert call(tg, "feature_fate", topic="hovercraft").rows == []
    prs = call(tg, "feature_fate", fate="delayed_then_added", **{"return": "prs"})
    assert answers(prs) == [pr(21)]


# ---------------------------------------------------------------- who_said
def test_who_said_competitor_people_claims_and_demos(tg):
    assert answers(call(tg, "who_said", subject="competitor", kind="feature", **{"return": "people"})) == [ALICE]
    assert answers(call(tg, "who_said", subject="competitor", kind="weakness", **{"return": "people"})) == [BOB]
    both = call(tg, "who_said", subject="competitor", **{"return": "people"})
    assert answers(both) == sorted([ALICE, BOB])
    c = call(tg, "who_said", subject="competitor", kind="feature", **{"return": "claims"})
    assert answers(c) == ["Rival has live dashboards"] and c.rows[0]["subject"] == "Rival Corp"
    assert c.rows[0]["stated_by"] == [ALICE] and c.rows[0]["evidence"]
    d = call(tg, "who_said", subject="competitor", kind="demo", **{"return": "people"})
    assert answers(d) == sorted([ALICE, BOB])
    u = call(tg, "who_said", subject="competitor", kind="demo", **{"return": "urls"})
    assert answers(u) == ["https://demo.example.com/rival"] and sorted(u.rows[0]["sharers"]) == sorted([ALICE, BOB])


def test_who_said_previous_release_is_scoped_by_release_scope(tg):
    assert answers(call(tg, "who_said", subject="previous_release", kind="issue",
                        **{"return": "companies"})) == ["BlueWave"]  # CloudSync's issue is current
    assert answers(call(tg, "who_said", subject="previous_release", kind="strength",
                        **{"return": "companies"})) == ["TechCorp"]
    q = call(tg, "who_said", subject="previous_release", kind="qa_issue", **{"return": "claims"})
    assert answers(q) == ["Memory leak in sync"] and q.rows[0]["reporters"] == [DAN]
    iss = call(tg, "who_said", subject="previous_release", kind="issue", **{"return": "claims"})
    assert answers(iss) == ["Reports failed to export"]
    pdemo = call(tg, "who_said", subject="previous_release", kind="demo", **{"return": "people"})
    assert answers(pdemo) == [EVE]
    assert answers(call(tg, "who_said", subject="previous_release", kind="demo",
                        **{"return": "urls"})) == ["https://demo.example.com/v1"]
    cur = call(tg, "who_said", subject="product", kind="demo", **{"return": "urls"})
    assert answers(cur) == ["https://demo.example.com/current"]


def test_who_said_worked_on_previous_release_uses_authors_and_named_qa_by_directory_role(tg):
    w = lambda role=None: answers(call(tg, "who_said", subject="previous_release", kind="worked_on",  # noqa: E731
                                       role=role, **{"return": "people"}))
    assert w("QA Specialist") == [DAN]  # the named QA reporter, not the poster Eve
    assert w("Product Manager") == [EVE]  # author of the previous Product Vision Document
    assert w("Technical Architect") == [FINN]
    assert w("Software Engineer") == []  # Alice only reviewed a present-day document
    assert w() == sorted([DAN, EVE, FINN])
    with pytest.raises(ToolError):
        call(tg, "who_said", subject="competitor", kind="worked_on")


# ---------------------------------------------------------------- doc_people
def test_doc_people_authors_plus_substantive_reviewers_not_attendees(tg):
    r = call(tg, "doc_people", document_type="Market Research Report")
    # Bob raised a review question in Slack; Dan's question came from a meeting transcript line, Alice only attended
    assert answers(r) == sorted([GUS, FINN, CARA, SAM1, SAM2, BOB])
    roles = {x["answer"]: x["roles"] for x in r.rows}
    assert roles[GUS] == ["author"] and roles[CARA] == ["reviewer"] and roles[FINN] == ["author"]
    assert by(r)[SAM1]["ambiguous"] and by(r)[SAM2]["ambiguous"] and not by(r)[CARA]["ambiguous"]
    assert by(r)[CARA]["evidence"] == ["rec-s1"] or "rec-s1" in by(r)[CARA]["evidence"]


def test_doc_people_scope_and_type_matching(tg):
    assert answers(call(tg, "doc_people", document_type="market research report", scope="previous")) == [FINN]
    cur = call(tg, "doc_people", document_type="Market Research Report", scope="current")
    assert FINN not in answers(cur) and GUS in answers(cur)
    assert answers(call(tg, "doc_people", document_type="Product Vision Document")) == sorted([EVE, ALICE])
    assert call(tg, "doc_people", document_type="No Such Document").rows == []


def test_doc_people_role_filter_keeps_ambiguity_provenance(tg):
    r = call(tg, "doc_people", document_type="Market Research Report", role="UX Researcher")
    assert answers(r) == [SAM1] and r.rows[0]["ambiguous"] is True
    assert answers(call(tg, "doc_people", document_type="Market Research Report", role="Marketing Manager")) == [SAM2]


def test_doc_people_suggestions_with_sections_speakers_and_anonymous_feedback(tg):
    s = call(tg, "doc_people", document_type="Market Research Report", role="Engineering Lead",
             **{"return": "suggestions"})
    assert answers(s) == ["Add pricing tiers"]
    assert s.rows[0]["section"] == "Pricing" and s.rows[0]["speaker"] == CARA and s.rows[0]["evidence"] == ["rec-s1"]
    every = call(tg, "doc_people", document_type="Market Research Report", **{"return": "suggestions"})
    assert "Make it shorter" in answers(every) and by(every)["Make it shorter"]["speaker"] is None
    assert by(every)["Make it shorter"]["anonymous"] is True
    assert "What about the EU?" not in answers(every) and "Is pricing final?" in answers(every)
    roled = call(tg, "doc_people", document_type="Market Research Report", role="QA Specialist",
                 **{"return": "suggestions"})
    assert roled.rows == []  # Dan's only contribution is a transcript question; anonymous feedback never matches a role
    slack_q = call(tg, "doc_people", document_type="Market Research Report", role="Software Engineer",
                   **{"return": "suggestions"})
    assert answers(slack_q) == ["Is pricing final?"] and slack_q.rows[0]["evidence"] == ["20260401-0-abcde"]


def test_review_questions_alone_do_not_make_a_key_reviewer(tg):
    people = call(tg, "doc_people", document_type="Market Research Report")
    assert DAN not in answers(people)  # a transcript-line question
    assert by(people)[BOB]["roles"] == ["reviewer"]  # a Slack review question does


def test_scope_current_does_not_include_unknown_scope(tg):
    anyp = call(tg, "doc_people", document_type="Market Research Report", scope="any")
    cur = call(tg, "doc_people", document_type="Market Research Report", scope="current")
    assert by(anyp)[GUS]["documents"] == ["Document:latest_mrr", "Document:odd_mrr"]
    assert by(cur)[GUS]["documents"] == ["Document:latest_mrr"]


# ---------------------------------------------------------------- bridges
def test_brandes_on_the_directed_projection_matches_hand_computation():
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("c", "e")]
    s = catalog.betweenness(["a", "b", "c", "d", "e"], edges)
    assert s == {"a": 0.0, "b": 3.0, "c": 4.0, "d": 0.0, "e": 0.0}
    diamond = catalog.betweenness(list("abcd"), [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    assert diamond == {"a": 0.0, "b": 0.5, "c": 0.5, "d": 0.0}  # two shortest paths a-d share the credit


def test_bridges_native_on_6_0_python_on_4_22_same_scores(tg):
    # a1 -> b2 -> c3 -> {d4, e5 -> ghost -> f6}: c 8, e 6, b 5, ghost 4 (paths through the nameless placeholder count)
    r = call(tg, "bridges", top_k=4)
    assert [x["answer"] for x in r.rows] == [CARA, EVE, BOB, "eid_e6000010"]
    assert [x["score"] for x in r.rows] == [8.0, 6.0, 5.0, 4.0]
    want = "native" if tg.server_version == "6.0" else "python"
    assert {x["engine"] for x in r.rows} == {want}
    assert r.rows[0]["name"] == "Cara Diaz" and r.rows[3]["name"] is None
    if want == "native":
        assert "algo.betweenness" in r.ops[0].query and "samplingSeed" in r.ops[0].query
    else:
        assert any(op.command == "python:betweenness" for op in r.ops)


# ---------------------------------------------------------------- if_leaves
@pytest.mark.skipif(importlib.util.find_spec("inherit.whatif") is not None, reason="Task 12 has landed")
def test_if_leaves_is_registered_and_not_implemented_until_task_12():
    with pytest.raises(NotImplementedError):
        TOOLS["if_leaves"].run(None, P, person="Alice Ng")


# ---------------------------------------------------------------- envelope, paging, trace
def test_every_result_carries_cypher_row_count_and_typed_rows(tg):
    calls = [("resolve_person", {"name": "Alice Ng"}), ("resolve_entity", {"text": "techcorp"}),
             ("doc_people", {"document_type": "Market Research Report"}), ("bug_board", {"status": "any"}),
             ("fix_trail", {"customer": "TechCorp"}), ("feature_fate", {"fate": "any"}),
             ("who_said", {"subject": "competitor"}), ("pr_board", {"approval": "approved"}),
             ("bridges", {"top_k": 2})]
    for tool, params in calls:
        r = call(tg, tool, **params)
        assert r.rows and r.ops and r.ms >= 0, tool
        assert isinstance(r.truncated, bool) and (r.cursor is None or isinstance(r.cursor, str))
        ro = [op for op in r.ops if op.command == "GRAPH.RO_QUERY"]
        assert ro and all(op.query.strip() and isinstance(op.row_count, int) and op.tool == tool for op in ro), tool
        for row in r.rows:
            assert "answer" in row and isinstance(row["refs"], list) and isinstance(row["evidence"], list), tool
        json.dumps(r.rows)  # serialisable


def test_trace_ops_carry_the_actual_bound_parameters(tg):
    r = call(tg, "bug_board", status="resolved", customer="TechCorp", component="Authentication")
    flat = {k: v for op in r.ops for k, v in op.params.items()}
    assert flat["companies"] == ["TechCorp"] and flat["components"] == ["Authentication"]
    assert all(isinstance(op, TraceOp) for op in r.ops)


def test_paging_cursor_is_bound_to_product_tool_and_filters(tg, monkeypatch):
    monkeypatch.setattr(catalog, "ROW_CAP", 2)
    first = call(tg, "bug_board", status="any", **{"return": "bugs"})
    assert len(first.rows) == 2 and first.truncated and first.cursor
    second = call(tg, "bug_board", status="any", cursor=first.cursor, **{"return": "bugs"})
    third = call(tg, "bug_board", status="any", cursor=second.cursor, **{"return": "bugs"})
    got = [r["id"] for r in first.rows + second.rows + third.rows]
    assert len(got) == 6 and len(set(got)) == 6 and not third.truncated and third.cursor is None
    with pytest.raises(ToolError, match="cursor"):
        call(tg, "bug_board", status="resolved", cursor=first.cursor, **{"return": "bugs"})
    with pytest.raises(ToolError, match="cursor"):
        call(tg, "pr_board", cursor=first.cursor)
    with pytest.raises(ToolError, match="cursor"):
        call(tg, "bug_board", status="any", cursor="garbage", **{"return": "bugs"})


def test_a_single_page_stays_within_the_row_cap_and_token_budget(tg, monkeypatch):
    monkeypatch.setattr(catalog, "RESPONSE_BUDGET_TOKENS", 400)
    r = call(tg, "bug_board", status="any", **{"return": "bugs"})
    assert r.truncated and r.cursor and 0 < len(r.rows) < 6 and catalog.tokens(r.rows) <= 400


def test_trace_collector_orders_ops_counts_pages_and_enforces_the_call_cap(tg):
    tr = Trace(max_calls=3)
    a = tr.run(tg, P, "bug_board", status="unresolved")
    b = tr.run(tg, P, "resolve_person", name="Alice Ng")
    assert tr.calls == 2 and [op.tool for op in tr.ops] == [op.tool for op in a.ops + b.ops]
    with pytest.raises(ToolError):
        tr.run(tg, P, "bug_board", status="bogus")  # a refused call counts and is traced
    assert tr.calls == 3 and tr.ops[-1].command == "refused" and tr.ops[-1].params["error"]
    with pytest.raises(ToolError, match="call limit"):
        tr.run(tg, P, "bug_board", status="any")
    assert tr.calls == 3
    d = tr.to_dicts()
    assert d[0]["tool"] == "bug_board" and d[0]["query"] and "row_count" in d[0]
    assert tr.max_calls == 3 and Trace().max_calls == depth1.CALL_CAP


@pytest.mark.skipif(importlib.util.find_spec("inherit.whatif") is not None, reason="Task 12 has landed")
def test_trace_separates_the_what_if_tool(tg):
    tr = Trace()
    with pytest.raises(NotImplementedError):
        tr.run(None, P, "if_leaves", person="Alice Ng")
    assert tr.calls == 1 and tr.ops[-1].tool == "if_leaves" and tr.ops[-1].command == "refused"


def test_rows_are_identical_on_both_servers_except_bridges(tsrv_both, tmp_path):
    graphs = {}
    for ver, db in tsrv_both.items():
        load(db)
        (tmp_path / ver).mkdir()
        graphs[ver] = agent_graph(db, tmp_path / ver)
        graphs[ver].server_version = ver
    calls = [("resolve_person", {"name": "Sam Kim", "meeting": "m1"}), ("resolve_entity", {"text": "mode flicker"}),
             ("doc_people", {"document_type": "Market Research Report"}),
             ("doc_people", {"document_type": "Market Research Report", "return": "suggestions"}),
             ("bug_board", {"status": "reopened", "actor": "fix_pr_author"}),
             ("bug_board", {"status": "active", "as_of": "2026-06-15T00:00:00"}),
             ("bug_board", {"status": "any", "return": "bugs"}), ("fix_trail", {"customer": "TechCorp"}),
             ("feature_fate", {"fate": "any"}), ("feature_fate", {"fate": "reverted", "return": "prs"}),
             ("who_said", {"subject": "previous_release", "kind": "worked_on"}),
             ("who_said", {"subject": "competitor", "kind": "demo", "return": "urls"}),
             ("pr_board", {"kind": "feature_development", "return": "people"}), ("pr_board", {"state": "reverted"})]
    for tool, params in calls:
        rows = {v: call(g, tool, **params).rows for v, g in graphs.items()}
        assert rows["4.22"] == rows["6.0"], (tool, params)
        assert rows["6.0"], (tool, params)
    b = {v: call(g, "bridges", top_k=2).rows for v, g in graphs.items()}
    assert {r["engine"] for r in b["4.22"]} == {"python"} and {r["engine"] for r in b["6.0"]} == {"native"}
    assert [(r["answer"], r["score"]) for r in b["4.22"]] == [(r["answer"], r["score"]) for r in b["6.0"]]


# ---------------------------------------------------------------- guard
def rows_result(rows):
    return ToolResult(rows=rows, truncated=False, cursor=None, ops=[], path=[], ms=0.0)


ROW_A = {"answer": ALICE, "name": "Alice Ng", "refs": [f"Person:{ALICE}", "Bug:b1"], "evidence": ["20260325-0-1281f"]}
ROW_B = {"answer": "TechCorp", "name": "TechCorp", "refs": ["Customer:CUST-0001"],
         "bugs": [{"id": "Bug:b1", "evidence": ["20260326-1-abcde"]}], "evidence": []}
RES = rows_result([ROW_A, ROW_B])


def test_guard_strips_an_invented_employee_id_and_the_sentence_naming_it():
    text, ids, dropped = guard(f"{ALICE} fixed it. eid_deadbeef also fixed it.", [ALICE, "eid_deadbeef"], [RES])
    assert ids == [ALICE] and "eid_deadbeef" not in text and ALICE in text
    assert any(d["value"] == "eid_deadbeef" for d in dropped)


def test_guard_validates_answers_exactly_and_keeps_evidence_only_ids_as_evidence_not_answers():
    text, ids, dropped = guard("See 20260325-0-1281f for details.", [ALICE, "20260325-0-1281f", "Bug:b1"], [RES])
    assert ids == [ALICE]
    assert {d["value"] for d in dropped} == {"20260325-0-1281f", "Bug:b1"}
    assert all(d["reason"] == "not an answer value" for d in dropped if d["where"] == "ids")
    assert "20260325-0-1281f" in text  # the text may cite evidence
    kept, gone = guard_evidence(["20260325-0-1281f", "20260326-1-abcde", "20269999-9-fffff", "Bug:b1",
                                 f"Person:{ALICE}"], [RES])
    assert kept == ["20260325-0-1281f", "20260326-1-abcde", "Bug:b1", f"Person:{ALICE}"]  # nested evidence counts
    assert gone == ["20269999-9-fffff"]


def test_guard_accepts_label_qualified_ids_in_text_when_a_row_names_them():
    t1, _, d1 = guard("Person:eid_a1000001 owns Bug:b1. It also owns Bug:b9.", [], [RES])
    assert t1 == "Person:eid_a1000001 owns Bug:b1. It also owns." and [d["value"] for d in d1] == ["Bug:b9"]
    t2, _, d2 = guard("Customer:CUST-0001 and CUST-0001 are the same.", [], [RES])
    assert t2 == "Customer:CUST-0001 and CUST-0001 are the same." and d2 == []  # a bare suffix of a ref is fine

def test_guard_keeps_both_namesakes_of_an_ambiguous_row():
    sam = rows_result([{"answer": SAM1, "name": "Sam Kim", "ambiguous": True, "refs": [], "evidence": []},
                       {"answer": SAM2, "name": "Sam Kim", "ambiguous": True, "refs": [], "evidence": []}])
    text, ids, dropped = guard(f"Sam Kim is {SAM1} or {SAM2}.", [SAM1, SAM2], [sam])
    assert ids == [SAM1, SAM2] and dropped == [] and text == f"Sam Kim is {SAM1} or {SAM2}."

def test_guard_removes_only_the_invented_id_and_keeps_the_rest_of_the_sentence():
    text, ids, dropped = guard(f"{ALICE} and eid_deadbeef fixed the bug. Nothing else happened.",
                               [ALICE, "eid_deadbeef"], [RES])
    assert text == f"{ALICE} fixed the bug. Nothing else happened."
    assert ids == [ALICE] and {(d["where"], d["value"]) for d in dropped} == {("ids", "eid_deadbeef"),
                                                                              ("text", "eid_deadbeef")}
    t2, _, d2 = guard("The bug was fixed by eid_deadbeef last week.", [], [RES])
    assert t2 == "The bug was fixed by last week." and [d["value"] for d in d2] == ["eid_deadbeef"]

def test_guard_drops_unsupported_links_and_ids_when_results_are_empty():
    pr_row = rows_result([{"answer": pr(1), "refs": [f"PR:{pr(1)}"], "evidence": []}])
    t2, ids2, d2 = guard(f"Fixed in {pr(1)}. Also {pr(99)}.", [pr(1), pr(99)], [pr_row])
    assert ids2 == [pr(1)] and pr(99) not in t2 and pr(1) in t2
    t3, ids3, d3 = guard(f"{ALICE} did it.", [ALICE], [rows_result([])])
    assert t3 == "did it." and ids3 == [] and d3

# ---------------------------------------------------------------- component matching (extraction leaves partial areas)
class _Comps:
    def __init__(self, comps):
        self.comps = comps

    def q(self, text, **params):
        return [[c] for c in self.comps]


def test_component_resolution_adds_partial_areas_of_two_or_more_words_but_never_merges_siblings():
    comps = ["Backend Processing & Task Queue Management", "backend processing", "User Interface & Frontend Performance",
             "frontend performance", "User Interface & Data Visualization", "Security"]
    rc = lambda t: catalog._resolve_components(_Comps(comps), t)  # noqa: E731, SLF001
    assert rc("backend processing and task queue management") == ["Backend Processing & Task Queue Management",
                                                                  "backend processing"]
    assert rc("User Interface & Frontend Performance") == ["User Interface & Frontend Performance", "frontend performance"]
    assert rc("user interface") == ["User Interface & Data Visualization", "User Interface & Frontend Performance"]
    assert rc("API security and access control") == []  # a one-word area ("Security") is not a partial name
    assert rc(None) is None


# ---------------------------------------------------------------- fates come from the derived Feature properties
def test_feature_fate_selects_by_the_derived_properties_not_by_events(tsrv, tmp_path):
    facts = tool_facts()
    nodes = [n for n in facts.nodes]
    from dataclasses import replace
    flipped = [replace(n, props={**n.props, "reverted": False, "implemented": False, "discussed_not_implemented": True})
               if n.id == "Feature:fB" else n for n in nodes]
    load(tsrv, type(facts)(flipped, facts.edges))
    g = agent_graph(tsrv, tmp_path)
    assert "Dark mode" not in answers(call(g, "feature_fate", fate="reverted"))  # events still say reverted
    assert "Dark mode" not in answers(call(g, "feature_fate", fate="implemented"))
    assert "Dark mode" in answers(call(g, "feature_fate", fate="unimplemented"))
    assert call(g, "feature_fate", topic="dark mode").rows[0]["fates"] == ["unimplemented"]


# ---------------------------------------------------------------- evidence names the whole qualifying path
def test_company_ranking_evidence_includes_the_status_events(tg):
    r = board(tg, status="reopened")
    row = r.rows[0]
    assert row["answer"] == "TechCorp"
    assert {"rec-b2-0", "rec-b6-0", "msg-b2-3", "msg-b6-3"} <= set(row["evidence"])  # reporting edges + REOPENED events
    assert "BugEvent:b2#3" in row["refs"] and "BugEvent:b6#3" in row["refs"]
    res = board(tg, status="resolved").rows[0]
    assert {"msg-b1-fix", "msg-b2-fix", "msg-b6-fix", "msg-b1-2", "msg-b2-2", "msg-b6-2"} <= set(res["evidence"])
    assert f"PR:{pr(5)}" in res["refs"]


def test_reopened_fix_pr_author_evidence_has_opened_fixes_resolution_and_the_later_reopening(tg):
    bob = by(call(tg, "bug_board", status="reopened", actor="fix_pr_author"))[BOB]
    assert {"rec-pr2", "rec-pr3", "msg-b2-fix", "msg-b2-2", "msg-b2-3"} <= set(bob["evidence"])
    assert {"BugEvent:b2#2", "BugEvent:b2#3", f"PR:{pr(2)}", f"PR:{pr(3)}", "Bug:b2"} <= set(bob["refs"])
    cara = by(call(tg, "bug_board", status="reopened", actor="dismisser"))[CARA]
    assert {"msg-b5-2", "msg-b5-3"} <= set(cara["evidence"])  # the dismissal and the reopening after it
    assert "msg-b3-2" not in cara["evidence"]


def test_bug_rows_and_reverted_features_carry_their_supporting_records(tg):
    b = by(call(tg, "bug_board", status="no_fix_needed", **{"return": "bugs"}), "id")["Bug:b3"]
    assert {"rec-b3-0", "msg-b3-2"} <= set(b["evidence"]) and "BugEvent:b3#2" in b["refs"]
    f = by(call(tg, "feature_fate", fate="reverted", reason="security", **{"return": "features"}), "id")["Feature:fF"]
    assert "rec-revert23" in f["evidence"] and f"PR:{pr(23)}" in f["refs"] and "rec-impl22" in f["evidence"]


def test_derived_fates_are_reported_on_the_row(tg):
    row = by(call(tg, "feature_fate"), "id")["Feature:fE"]
    assert row["fates"] == ["delayed_then_added", "held", "implemented"]


# ---------------------------------------------------------------- guard: whole ids, real row shapes
def test_guard_does_not_police_prose_names_and_keeps_supported_sentences_intact():
    # NB2: a title before a supported name must not delete the sentence; names are not policed at all
    t, _, d = guard("Software Engineer Alice Ng fixed the bug. Zed Quinn helped.", [], [RES])
    assert t == "Software Engineer Alice Ng fixed the bug. Zed Quinn helped." and d == []


def test_guard_ids_need_exact_membership_not_prefix_or_substring():
    rows = rows_result([{"answer": "Bug b10", "refs": ["Bug:b10"], "evidence": []}])
    t, _, d = guard("See Bug:b10, Bug:b1 and Bug:b10:invented.", [], [rows])
    assert t == "See Bug:b10." and [x["value"] for x in d] == ["Bug:b1", "Bug:b10:invented"]
    t2, _, d2 = guard(f"Mail {ALICE} or eid_a1000002.", [], [rows_result([{"answer": ALICE}])])
    assert t2 == f"Mail {ALICE}." and [x["value"] for x in d2] == ["eid_a1000002"]

def test_guard_recognises_complete_returned_ids_with_markdown_and_spaces():
    rows = rows_result([{"answer": "Voice assistant", "refs": ["Feature:mini:voice assistant", "Bug:b1"],
                         "evidence": []}])
    for sent in ("See `Bug:b1` for details.", "The feature is **Feature:mini:voice assistant**.",
                 "Feature:mini:voice assistant is not implemented.", "(Bug:b1) is open."):
        t, _, d = guard(sent, [], [rows])
        assert t == sent and d == [], sent
    t, _, d = guard("See `Bug:b9` for details.", [], [rows])
    assert t == "See for details." and d[0]["value"] == "Bug:b9"
    t, _, d = guard("Feature:mini:other thing is open.", [], [rows])
    assert d and "Feature:mini:other" not in t

def test_guard_supports_the_fields_the_tools_emit():
    sug = rows_result([{"answer": "Add pricing tiers", "speaker": CARA, "speaker_name": "Cara Diaz",
                        "speaker_role": "Engineering Lead", "document": "Document:latest_mrr",
                        "refs": [f"Person:{CARA}"], "evidence": ["20260401-0-abcde"]}])
    trail = rows_result([{"answer": pr(2), "author": BOB, "author_name": "Bob Roy", "company": "TechCorp",
                          "companies": ["TechCorp"], "bug_title": "Slow export", "refs": [f"PR:{pr(2)}"],
                          "evidence": []}])
    claim = rows_result([{"answer": "Rival has live dashboards", "subject": "Rival Corp", "stated_by": [ALICE],
                          "refs": [], "evidence": [],
                          "claims": [{"evidence": {"records": ["20260402-1-fffff"]}}]}])
    text = ("Cara Diaz suggested pricing tiers. Bob Roy opened the fix for TechCorp. "
            "Rival Corp has live dashboards, said by " + ALICE + ". See 20260402-1-fffff.")
    t, _, d = guard(text, [], [sug, trail, claim])
    assert t == text and d == []
    kept, gone = guard_evidence(["20260402-1-fffff"], [claim])
    assert kept == ["20260402-1-fffff"] and gone == []  # dictionary-shaped nested evidence keeps its category


# ---------------------------------------------------------------- the oracle never prints gold
def test_oracle_stdout_has_no_gold_values(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    from eval.questions import Question
    from inherit.contracts import QuestionKey

    spec = importlib.util.spec_from_file_location("dev_oracle_plans", Path(__file__).parents[1] / "scripts" /
                                                  "dev_oracle_plans.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    gold_id, gold_text = "GOLD_SENTINEL_ID", "GOLD_SENTINEL_PHRASE words"
    qs = [Question(QuestionKey("ActionGenie", "ans", 0), "Find the name of company that has the maximum number of "
                   "unresolved bugs in ActionGenie?", "company", [gold_id, "ANOTHER_SENTINEL"]),
          Question(QuestionKey("ActionGenie", "ans", 1), "Find all unresolved issues reported by TechCorp in the "
                   "ActionGenie product?", "content", gold_text)]
    monkeypatch.setattr(oracle, "load_questions", lambda split: qs)
    monkeypatch.setattr(oracle, "agent_graph", lambda p, server: SimpleNamespace())
    monkeypatch.setattr(oracle, "present", lambda g, needs: True)
    monkeypatch.setattr(oracle.Plan, "call", lambda self, tool, **kw: [{"answer": "TechCorp"}])
    oracle.main(["--misses"])
    out = capsys.readouterr().out
    assert "ActionGenie/ans/0 F1 0" in out and "extra 1" in out and "missing 2" in out
    assert "SENTINEL" not in out
