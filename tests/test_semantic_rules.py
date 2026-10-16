import semfx as fx
from inherit.extract import semantic_rules as sr
from inherit.extract.catalog import build_catalog


def test_feedback_items_are_the_bullets():
    assert sr.parse_feedback_items(fx.FEEDBACK) == [
        "Add industry specifics to the Executive Summary.", "Include regulatory changes in Market Trends."]
    assert sr.parse_feedback_items(None) == []
    assert sr.parse_feedback_items("Plain sentence without bullets.") == ["Plain sentence without bullets."]


def test_competitor_of_demo_reads_the_description_and_skips_own_names():
    own = ["Mini", "Old Mini"]
    assert sr.competitor_of_demo("A demo showing how RivalCo works.", own) == "RivalCo"
    assert sr.competitor_of_demo("A demo showing how Sales Mate AI works.", own) == "Sales Mate AI"
    assert sr.competitor_of_demo("A demo video of Mini.", own) is None
    assert sr.competitor_of_demo("A demo showing how Mini works.", own) is None
    assert sr.competitor_of_demo("A blog post about demos.", own) is None


def test_route_planning_always_and_develop_by_cue():
    msg = lambda text, name: type("M", (), {"text": text, "channel_name": name, "is_bot": False})()  # noqa: E731
    assert sr.route_session([msg("hello", "planning-Mini")]) == (True, "planning")
    assert sr.route_session([msg("PR#3 is approved", "develop-ben-Mini")]) == (True, "pr_ref")
    assert sr.route_session([msg("I was reading about X, a competitor", "develop-ben-Mini")]) == (True, "competitor")
    assert sr.route_session([msg("QA found issues", "develop-ben-Mini")]) == (True, "qa")
    assert sr.route_session([msg("good morning", "develop-ben-Mini")]) == (False, "no_cue")
    assert sr.route_session([msg("see https://www.x.com/demo", "develop-ben-Mini")]) == (False, "no_cue")
    assert sr.route_session([msg("PR#3", "bug-ben-Mini")]) == (False, "bug_channel")


def test_split_session_at_message_boundaries():
    msgs = [type("M", (), {"text": "x" * 100, "id": str(i)})() for i in range(10)]
    parts = sr.split_session(msgs, max_chars=350)
    assert [len(p) for p in parts] == [3, 3, 3, 1]
    assert [m.id for p in parts for m in p] == [str(i) for i in range(10)]
    assert sr.split_session(msgs, max_chars=10_000) == [msgs]


def test_scope_cue():
    assert sr.PREV_CUE.search("QA found issues in the last release of Mini")
    assert sr.PREV_CUE.search("Kudos to the previous team")
    assert sr.PREV_CUE.search("developed by a different team")
    assert not sr.PREV_CUE.search("dashboards are slow")


def test_pr_numbers_from_refs():
    assert sr.pr_ref_number("8") == "8"
    assert sr.pr_ref_number("PR# 8") == "8"
    assert sr.pr_ref_number("#12") == "12"
    assert sr.pr_ref_number("https://github.com/salesforce/Mini/pull/31") is None  # a link, not a number
    assert sr.pr_ref_number("the audit PR") is None


def test_resolve_pr_by_number_or_link_inside_the_product_repo():
    cat = build_catalog(fx.records())
    p, why = sr.resolve_pr("3", cat)
    assert (p.link, why) == ("https://github.com/salesforce/Mini/pull/3", "ok")
    p, why = sr.resolve_pr("https://github.com/salesforce/Mini/pull/4", cat)
    assert (p.number, why) == ("4", "ok")
    assert sr.resolve_pr("9001", cat) == (None, "unknown")  # the external PR is not a product PR
    assert sr.resolve_pr("https://github.com/apache/kafka/pull/9001", cat) == (None, "external")
    assert sr.resolve_pr("999", cat) == (None, "unknown")
    assert sr.resolve_pr("the audit PR", cat) == (None, "unknown")


def test_resolve_pr_number_in_two_product_repos_is_ambiguous():
    from types import SimpleNamespace

    from inherit.extract.catalog import PrEntry

    def entry(repo, n):
        return PrEntry(f"{repo}_{n}", f"https://github.com/salesforce/{repo}/pull/{n}", str(n), "t", "a", "salesforce",
                       repo, "product", False, "open")

    prs = [entry("AIX", 3), entry("LeadForce", 3), entry("AIX", 4)]
    cat = SimpleNamespace(prs=prs, pr=lambda link: next((p for p in prs if p.link == link), None))
    assert sr.resolve_pr("3", cat) == (None, "ambiguous")
    assert sr.resolve_pr("4", cat)[1] == "ok"
