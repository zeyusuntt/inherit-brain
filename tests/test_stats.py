import numpy as np
import pytest

from eval import stats as S
from eval.grade import Graded
from eval.questions import Question
from inherit.contracts import QuestionKey, RunRecord

from evalfx import final, mk_questions


def G(product, idx, type_, score, arm="A", kind=None, failed=False, outcome="path", seen=None):
    kind = kind or ("unans" if type_ == "unanswerable" else "ans")
    return Graded(QuestionKey(product, kind, idx), arm, type_, float(score), failed, outcome, seen)


# ---------------------------------------------------------------- summarize
def held_like(n_failed=3):
    """659 answerable and 544 unanswerable slots; the first n_failed answerable ones failed (score 0)."""
    types = ["person"] * 260 + ["company"] * 130 + ["pr"] * 167 + ["url"] * 20 + ["content"] * 82
    assert len(types) == 659
    out = [G("P", i, t, 0 if i < n_failed else 100, failed=i < n_failed, outcome="run_failed" if i < n_failed else "path")
           for i, t in enumerate(types)]
    out += [G("P", i, "unanswerable", 100) for i in range(544)]
    return out


def test_denominator_stays_659_and_544_when_three_runs_fail():
    s = S.summarize(held_like(3), expected_n=S.HELD_N)
    assert s["n"] == {"answerable": 659, "unanswerable": 544}
    assert s["failed"]["answerable"] == 3
    assert s["per_type"]["person"]["n"] == 260 and s["per_type"]["person"]["mean"] == pytest.approx(100 * 257 / 260)
    assert s["failure_rate"] == pytest.approx(100 * 3 / 1203)
    # losing the failed rows instead of keeping them at 0 is an error, not a smaller denominator
    with pytest.raises(ValueError, match="659"):
        S.summarize(held_like(3)[3:], expected_n=S.HELD_N)


def test_primary_endpoint_is_the_mean_of_three_type_means_not_the_mean_over_questions():
    gs = [G("P", i, "person", 100) for i in range(10)]
    gs += [G("P", 10, "company", 0)]
    gs += [G("P", 11, "pr", 100), G("P", 12, "pr", 100), G("P", 13, "url", 0)]  # artifact pools pr and url
    gs += [G("P", 14 + i, "content", 0) for i in range(10)]
    s = S.summarize(gs)
    assert s["groups"] == {"person": 100.0, "company": 0.0, "artifact": pytest.approx(200 / 3), "content": 0.0}
    assert s["primary"] == pytest.approx((100 + 0 + 200 / 3) / 3)
    assert s["paper_macro"] == pytest.approx((100 + 0 + 200 / 3 + 0) / 4)
    assert s["primary"] != pytest.approx(np.mean([g.score for g in gs]))
    assert s["per_type"]["pr"]["mean"] == 100.0 and s["per_type"]["url"]["mean"] == 0.0


def test_primary_is_none_when_a_group_is_absent():
    s = S.summarize([G("P", 0, "person", 50), G("P", 1, "company", 50)])
    assert s["primary"] is None and s["groups"]["artifact"] is None


def test_unanswerable_rate_false_refusals_and_failures():
    gs = [G("P", 0, "person", 100), G("P", 1, "person", 0, outcome="not_in_graph"),
          G("P", 2, "company", 0, outcome="unknown_name"), G("P", 3, "pr", 0, failed=True, outcome="run_failed"),
          G("P", 0, "unanswerable", 100), G("P", 1, "unanswerable", 0), G("P", 2, "unanswerable", 0, failed=True)]
    s = S.summarize(gs)
    assert s["unanswerable_rate"] == pytest.approx(100 / 3)
    assert s["false_refusal_rate"] == pytest.approx(50.0)  # 2 of 4 answerable
    assert s["failure_rate"] == pytest.approx(100 * 2 / 7)
    assert s["failed"] == {"answerable": 1, "unanswerable": 1}


def test_tables_are_also_split_by_seen_and_unseen():
    gs = [G("P", 0, "person", 100, seen=True), G("P", 1, "person", 0, seen=False), G("P", 2, "person", 50, seen=False),
          G("P", 0, "unanswerable", 100, seen=True), G("P", 1, "unanswerable", 0, seen=False)]
    s = S.summarize(gs)
    assert s["by_seen"]["seen"]["per_type"]["person"] == {"n": 1, "mean": 100.0, "failed": 0}
    assert s["by_seen"]["unseen"]["per_type"]["person"]["mean"] == 25.0
    assert s["by_seen"]["seen"]["unanswerable_rate"] == 100.0 and s["by_seen"]["unseen"]["unanswerable_rate"] == 0.0
    assert "by_seen" not in S.summarize([G("P", 0, "person", 1)])


# ---------------------------------------------------------------- usage and citations
def test_usage_summary_percentiles_tokens_and_tool_calls():
    recs = [RunRecord(QuestionKey("P", "ans", i), "A", final(prompt=100 * (i + 1), completion=10 * (i + 1), n_trace=i),
                      latency_s=float(i + 1)) for i in range(10)]
    recs.append(RunRecord(QuestionKey("P", "unans", 0), "A", final(prompt=100000, completion=5), latency_s=100.0))
    u = S.usage_summary(recs)
    assert u["median_input_tokens"] == 550.0  # answerable only
    assert u["mean_output_tokens"] == pytest.approx(np.mean([10 * (i + 1) for i in range(10)] + [5]))
    assert u["mean_tool_calls"] == pytest.approx(np.mean(list(range(10)) + [2]))
    assert u["latency_p50"] == pytest.approx(np.percentile([float(i + 1) for i in range(10)] + [100.0], 50))
    assert u["latency_p95"] == pytest.approx(np.percentile([float(i + 1) for i in range(10)] + [100.0], 95))


def test_citation_overlap_is_reported_and_separate_from_scores():
    q = [Question(QuestionKey("P", "ans", 0), "q", "person", ["a"], ["c1", "c2"]),
         Question(QuestionKey("P", "ans", 1), "q", "person", ["a"], ["c3"]),
         Question(QuestionKey("P", "unans", 0), "q", "unanswerable", None, [])]
    recs = [RunRecord(q[0].key, "A", final(evidence=["c1", "zz"])), RunRecord(q[1].key, "A", final(evidence=[])),
            RunRecord(q[2].key, "A", final(evidence=["c9"]))]
    o = S.citation_overlap(recs, q)
    assert o == {"n": 2, "mean_recall": 0.25, "mean_precision": 0.25, "n_empty_evidence": 1}


# ---------------------------------------------------------------- bootstrap
def two_arm_fixture():
    """Three products with different numbers of questions. A is better than B on every question."""
    a, b = [], []
    for p, n in (("X", 1), ("Y", 3), ("Z", 2)):
        i = 0
        for t in ("person", "company", "pr"):
            for _ in range(n):
                a.append(G(p, i, t, 80, arm="A"))
                b.append(G(p, i, t, 20, arm="B"))
                i += 1
    return a, b


def test_bootstrap_with_a_seed_is_reproducible_and_seeds_differ():
    a, b = two_arm_fixture()
    r1 = S.paired_bootstrap(a, b, None, n=300, seed=5)
    assert r1 == S.paired_bootstrap(a, b, None, n=300, seed=5)
    assert len(r1) == 3 and r1[0] == pytest.approx(60.0)
    assert r1 != S.paired_bootstrap(a, b, None, n=300, seed=6) or r1[1] == r1[2]  # constant diff has no spread


def test_bootstrap_default_seed_is_the_preregistered_one():
    import inspect
    sig = inspect.signature(S.paired_bootstrap)
    assert sig.parameters["seed"].default == 20261015 and sig.parameters["n"].default == 10000


def test_identical_arms_give_a_zero_interval():
    a, _ = two_arm_fixture()
    assert S.paired_bootstrap(a, a, None, n=200, seed=1) == (0.0, 0.0, 0.0)


def test_bootstrap_interval_excludes_zero_for_a_real_gap_and_straddles_it_for_noise():
    rng = np.random.default_rng(0)
    a, b = [], []
    for p in range(24):
        for i, t in enumerate(("person", "company", "pr", "url")):
            for j in range(5):
                base = rng.normal(60, 20)
                a.append(G(f"p{p}", i * 5 + j, t, np.clip(base + 25, 0, 100), arm="A"))
                b.append(G(f"p{p}", i * 5 + j, t, np.clip(base, 0, 100), arm="B"))
    diff, lo, hi = S.paired_bootstrap(a, b, None, n=2000, seed=3)
    assert lo > 0 and lo < diff < hi
    c = [G(g.key.product, g.key.index, g.type, np.clip(g.score + rng.normal(0, 10), 0, 100), arm="C") for g in b]
    d2, lo2, hi2 = S.paired_bootstrap(c, b, None, n=2000, seed=3)
    assert lo2 < 0 < hi2


def test_pairing_is_by_question_key_not_by_position():
    a, b = two_arm_fixture()
    assert S.paired_bootstrap(a, list(reversed(b)), None, n=200, seed=2) == S.paired_bootstrap(a, b, None, n=200, seed=2)
    with pytest.raises(ValueError, match="same questions"):
        S.paired_bootstrap(a, b[:-1], None, n=10, seed=2)
    # a pair that is split across products would give a different answer if matched by position
    sw = [G(g.key.product, g.key.index, g.type, g.score, arm="B") for g in b]
    sw[0], sw[-1] = sw[-1], sw[0]
    assert S.paired_bootstrap(a, sw, None, n=200, seed=2) == S.paired_bootstrap(a, b, None, n=200, seed=2)


def test_replicate_keeps_multiplicity_and_weights_by_question():
    """Product X has 1 question per type, Y has 3. Drawing [X, X] counts X's questions twice, [Y] three times."""
    a, b = two_arm_fixture()
    sa, ca, products = S.tables(a, None)
    sb, _, _ = S.tables(b, None)
    assert products == ["X", "Y", "Z"] and ca.shape == (3, 4)
    mask = S.group_mask(S.PRIMARY_GROUPS)
    # person scores: X 80, Y 80, Z 80 in A; draw X twice and Y once -> weighted mean is still 80, counts 1+1+3
    idx = np.array([[0, 0, 1]])
    tot_c = S.replicate_counts(ca, idx)
    assert tot_c[0, 0] == 5 and tot_c[0, 1] == 5
    ep_a, miss = S.replicate_endpoints(sa, ca, idx, mask)
    assert ep_a[0] == pytest.approx(80.0) and miss == 0
    # make X's person score differ so multiplicity matters: weighted mean = (2*1*100 + 3*40)/5 = 64
    a2 = [G(g.key.product, g.key.index, g.type, 100 if (g.key.product == "X" and g.type == "person") else (
        40 if (g.key.product == "Y" and g.type == "person") else g.score), arm="A") for g in a]
    sa2, ca2, _ = S.tables(a2, None)
    group_means, _ = S._group_means(sa2, ca2, idx)
    assert group_means[0, 0] == pytest.approx(64.0)
    # the average of per-product means (100, 100, 40) would be 80: not what we compute
    assert group_means[0, 0] != pytest.approx(80.0)


def test_replicate_with_a_missing_stratum_uses_the_remaining_groups_and_is_counted():
    a = [G("X", 0, "person", 100), G("X", 1, "company", 0), G("X", 2, "pr", 50),
         G("Y", 0, "person", 20), G("Y", 1, "company", 40)]  # Y has no artifact question
    sa, ca, products = S.tables(a, None)
    mask = S.group_mask(S.PRIMARY_GROUPS)
    ep, miss = S.replicate_endpoints(sa, ca, np.array([[1, 1], [0, 1]]), mask)
    assert ep[0] == pytest.approx((20 + 40) / 2)  # artifact is missing: the other two groups are averaged
    assert ep[1] == pytest.approx(((100 + 20) / 2 + (0 + 40) / 2 + 50) / 3)
    assert miss == 1
    res = S.paired_bootstrap_full(a, a, None, n=50, seed=1)
    assert res.n == 50 and 0 <= res.n_missing_stratum <= 50 and res.n_missing_stratum > 0


def test_bootstrap_on_a_subset_of_keys():
    a, b = two_arm_fixture()
    keys = {str(g.key) for g in a if g.key.product in ("X", "Y")}
    full = S.paired_bootstrap_full(a, b, None, n=100, seed=1)
    sub = S.paired_bootstrap_full(a, b, None, n=100, seed=1, keys=keys)
    assert full.diff == sub.diff == 60.0
    with pytest.raises(ValueError):
        S.paired_bootstrap_full(a, b, None, n=10, seed=1, keys={"nope/ans/0"})


def test_clusters_argument_lists_the_products_to_resample():
    a, b = two_arm_fixture()
    with pytest.raises(ValueError, match="cluster"):
        S.paired_bootstrap(a, b, ["X", "Y"], n=10, seed=1)  # Z has questions but is not a cluster
    assert S.paired_bootstrap(a, b, ["X", "Y", "Z"], n=50, seed=1) == S.paired_bootstrap(a, b, None, n=50, seed=1)


# ---------------------------------------------------------------- hypotheses as code
def summaries(a5_rate=90.0, a1f_rate=80.0, a5_tok=1000, a1f_tok=10000):
    return {"A5": {"unanswerable_rate": a5_rate, "false_refusal_rate": 4.0, "median_input_tokens": a5_tok},
            "A1f": {"unanswerable_rate": a1f_rate, "false_refusal_rate": 1.0, "median_input_tokens": a1f_tok}}


def boots(**over):
    b = {"H1": (20.0, 12.0, 28.0), "H2": (-1.0, -2.5, 0.5), "H3": (11.0, 2.0, 19.0), "H3_all": (5.0, 1.0, 9.0)}
    b.update(over)
    return b


def test_decisions_follow_the_preregistered_thresholds():
    d = S.decide(summaries(), boots())
    assert d["H1"]["pass"] and d["H1"]["prior_expectation_met"]
    assert d["H2"]["pass"] and d["H2"]["interval_ok"] and d["H2"]["tokens_ok"]
    assert d["H3"]["pass"] and d["H3"]["prior_expectation_met"] and d["H3"]["all_questions"] == (5.0, 1.0, 9.0)
    assert d["H4"]["pass"] and d["H4"]["false_refusal_rate"] == {"A5": 4.0, "A1f": 1.0}


def test_thresholds_are_strict_where_the_spec_says_greater_than():
    assert not S.decide(summaries(), boots(H1=(20, 0.0, 28)))["H1"]["pass"]
    assert not S.decide(summaries(), boots(H3=(20, 0.0, 28)))["H3"]["pass"]
    assert not S.decide(summaries(), boots(H2=(-1, -3.0, 1)))["H2"]["pass"]
    assert S.decide(summaries(), boots(H2=(-1, -2.99, 1)))["H2"]["pass"]
    assert not S.decide(summaries(), boots(H1=(10, 5.0, 28)))["H1"]["prior_expectation_met"]
    assert S.decide(summaries(), boots(H1=(10, 5.0, 28)))["H1"]["pass"]


def test_h2_needs_both_the_interval_and_the_token_ratio():
    d = S.decide(summaries(a5_tok=1000, a1f_tok=10000), boots())
    assert d["H2"]["tokens_ok"]  # exactly 10 percent passes
    d = S.decide(summaries(a5_tok=1001, a1f_tok=10000), boots())
    assert not d["H2"]["tokens_ok"] and not d["H2"]["pass"]


def test_h4_is_a_point_comparison_of_refusal_rates():
    assert S.decide(summaries(a5_rate=80.0, a1f_rate=80.0), boots())["H4"]["pass"]
    assert not S.decide(summaries(a5_rate=79.9, a1f_rate=80.0), boots())["H4"]["pass"]


def test_bootstrap_results_may_be_full_results():
    a, b = two_arm_fixture()
    r = S.paired_bootstrap_full(a, b, None, n=20, seed=1)
    d = S.decide(summaries(), boots(H1=r))
    assert d["H1"]["pass"]


def test_compute_bootstraps_covers_h1_to_h3():
    a, b = two_arm_fixture()
    by_arm = {"A5": a, "A0": b, "A1f": b, "A4": b}
    keys = {str(g.key) for g in a if g.key.product == "Y"}
    out = S.compute_bootstraps(by_arm, keys, None, n=40, seed=1)
    assert set(out) == {"H1", "H2", "H3", "H3_all"}
    assert out["H1"].diff == 60.0 and out["H3"].diff == 60.0


def test_multihop_families_come_from_prereg_text():
    text = "# Pre-registration\n\nmulti-hop family: find all <x> of <p>\nmulti-hop family: who reviewed <p>\n\nother\n"
    assert S.read_multihop_families(text) == {"find all <x> of <p>", "who reviewed <p>"}
    with pytest.raises(ValueError, match="multi-hop"):
        S.read_multihop_families("nothing here")
    qs = [Question(QuestionKey("P", "ans", 0), "Who reviewed P?", "person", ["a"], []),
          Question(QuestionKey("P", "ans", 1), "Something else?", "person", ["a"], []),
          Question(QuestionKey("P", "unans", 0), "Who reviewed P?", "unanswerable", None, [])]
    assert S.multihop_keys(qs, {"who reviewed <p>"}) == {"P/ans/0"}
