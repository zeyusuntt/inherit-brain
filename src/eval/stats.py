"""Summaries, the product-clustered paired bootstrap, and the H1-H4 decisions (SPEC 6.2 and 6.3).

Scores are 0-100. Rates are percentages. Denominators are fixed: a failed slot is a 0 in the table, never a
missing row. Frozen choices (design review F11):
- Bootstrap resamples products with replacement and keeps each drawn product's questions, so a product drawn
  twice counts twice. Every replicate recomputes question-weighted type means and the endpoint, for both arms on
  the same draw. Arms are paired by question key. The interval is the 2.5 and 97.5 percentiles.
- A group with no question in a replicate is left out of that replicate's endpoint (no imputation). Such
  replicates are counted and reported.
- H3 uses the same endpoint restricted to the pre-registered multi-hop question set.
- H2's token population is the per-question total input tokens on answerable questions.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

import numpy as np

from eval.grade import Graded
from eval.questions import Question, template
from inherit.contracts import RunRecord

SEED = 20261015
HELD_N = (659, 544)  # answerable, unanswerable
ANSWERABLE_TYPES = ("person", "company", "pr", "url", "content")
GROUPS = ("person", "company", "artifact", "content")
PRIMARY_GROUPS = ("person", "company", "artifact")
GROUP_OF = {"person": "person", "company": "company", "pr": "artifact", "url": "artifact", "content": "content"}
REFUSAL_OUTCOMES = frozenset({"not_in_graph", "unknown_name"})  # an answerable question refused: a false refusal


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def _macro(groups: dict[str, float | None], names: tuple[str, ...]) -> float | None:
    vals = [groups[g] for g in names]
    return None if any(v is None for v in vals) else float(np.mean(vals))


def _core(graded: list[Graded]) -> dict[str, Any]:
    ans = [g for g in graded if g.type != "unanswerable"]
    un = [g for g in graded if g.type == "unanswerable"]
    per_type = {}
    for t in (*ANSWERABLE_TYPES, "unanswerable"):
        xs = [g for g in graded if g.type == t]
        per_type[t] = {"n": len(xs), "mean": _mean([g.score for g in xs]), "failed": sum(g.failed for g in xs)}
    groups = {name: _mean([g.score for g in ans if GROUP_OF[g.type] == name]) for name in GROUPS}
    return {
        "n": {"answerable": len(ans), "unanswerable": len(un)},
        "failed": {"answerable": sum(g.failed for g in ans), "unanswerable": sum(g.failed for g in un)},
        "per_type": per_type,
        "groups": groups,
        "primary": _macro(groups, PRIMARY_GROUPS),  # person, company, artifact (pr plus url), equal weight
        "paper_macro": _macro(groups, GROUPS),  # adds content, as the HERB paper does
        "unanswerable_rate": _mean([g.score for g in un]),  # correct refusals
        "false_refusal_rate": (100.0 * sum(g.outcome in REFUSAL_OUTCOMES for g in ans) / len(ans)) if ans else None,
        "failure_rate": (100.0 * sum(g.failed for g in graded) / len(graded)) if graded else None,
    }


def _check_unique(graded: list[Graded]) -> None:
    keys = [str(g.key) for g in graded]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate question keys among the graded slots")


def summarize(graded: list[Graded], expected_n: tuple[int, int] | None = None, *, pilot: bool = False) -> dict[str, Any]:
    """Per type means, the primary endpoint, the paper macro, refusal and failure rates, and the same split by
    seen and unseen question strings. `expected_n` is the fixed (answerable, unanswerable) denominator; without
    it the call must say `pilot=True`, so a partial population is never summarised by accident. Use
    summarize_split for a real split."""
    _check_unique(graded)
    if expected_n is None and not pilot:
        raise ValueError("summarize needs expected_n (or summarize_split); pass pilot=True for a partial population")
    out = _core(graded)
    if expected_n is not None and (out["n"]["answerable"], out["n"]["unanswerable"]) != tuple(expected_n):
        raise ValueError(
            f"expected {expected_n[0]} answerable and {expected_n[1]} unanswerable slots, "
            f"got {out['n']['answerable']} and {out['n']['unanswerable']}; a failed slot is scored 0, never dropped")
    if any(g.seen is not None for g in graded):
        out["by_seen"] = {
            "seen": _core([g for g in graded if g.seen is True]),
            "unseen": _core([g for g in graded if g.seen is False]),
        }
    return out


def _canonical(split: str, questions=None) -> set[str]:
    from eval.questions import load_questions

    return {str(q.key) for q in (questions if questions is not None else load_questions(split))}


def summarize_split(graded: list[Graded], split: str, questions=None) -> dict[str, Any]:
    """summarize() over exactly the canonical key set of the split: no missing and no duplicate key, which also
    gives the frozen denominator (held 659/544, dev 156/155)."""
    from eval.questions import FROZEN_COUNTS

    keys = [str(g.key) for g in graded]
    canon = _canonical(split, questions)
    if len(keys) != len(set(keys)) or set(keys) != canon:
        raise ValueError(f"graded slots are not the canonical {split} key set: "
                         f"{len(canon - set(keys))} missing, {len(set(keys) - canon)} extra, "
                         f"{len(keys) - len(set(keys))} duplicated")
    return summarize(graded, expected_n=FROZEN_COUNTS[split])


# ---------------------------------------------------------------- usage and citations
def usage_summary(records: list[RunRecord]) -> dict[str, float | None]:
    """Tokens, tool calls and latency per question. The H2 token population is answerable questions only."""
    ans = [r for r in records if r.key.kind == "ans"]
    lat = [r.latency_s for r in records]
    return {
        "median_input_tokens": float(np.median([r.final.usage.prompt_tokens for r in ans])) if ans else None,
        "mean_input_tokens": _mean([r.final.usage.prompt_tokens for r in records]),
        "mean_output_tokens": _mean([r.final.usage.completion_tokens for r in records]),
        "mean_tool_calls": _mean([len(r.final.trace) for r in records]),
        "latency_p50": float(np.percentile(lat, 50)) if lat else None,
        "latency_p95": float(np.percentile(lat, 95)) if lat else None,
    }


def citation_overlap(records: list[RunRecord], questions: list[Question]) -> dict[str, Any]:
    """How the cited records (evidence_ids) overlap HERB's citations on answerable questions. Reported on its
    own: it says which records were cited, not that the evidence suffices, and it is never graded."""
    by_key = {q.key: q for q in questions}
    rec, prec, empty = [], [], 0
    for r in records:
        q = by_key.get(r.key)
        if q is None or q.type == "unanswerable" or not q.citations or r.final.outcome == "run_failed":
            continue
        ev, cit = set(r.final.evidence_ids), set(q.citations)
        hit = len(ev & cit)
        rec.append(hit / len(cit))
        prec.append(hit / len(ev) if ev else 0.0)
        empty += not ev
    return {"n": len(rec), "mean_recall": _mean(rec), "mean_precision": _mean(prec), "n_empty_evidence": empty}


# ---------------------------------------------------------------- bootstrap
class BootResult(NamedTuple):
    diff: float
    lo: float
    hi: float
    n_missing_stratum: int  # replicates where a group had no question and was left out of the endpoint
    n: int


def group_mask(groups: tuple[str, ...]) -> np.ndarray:
    return np.array([g in groups for g in GROUPS])


def tables(graded: list[Graded], clusters: list[str] | None = None, keys: set[str] | None = None):
    """Per product and group: score sums and question counts. Returns (sums, counts, products)."""
    ans = [g for g in graded if g.type != "unanswerable"]
    seen_products = {g.key.product for g in ans}
    products = sorted(set(clusters)) if clusters else sorted(seen_products)
    if seen_products - set(products):
        raise ValueError(f"products {sorted(seen_products - set(products))} are not in the cluster list")
    row = {p: i for i, p in enumerate(products)}
    col = {g: i for i, g in enumerate(GROUPS)}
    s = np.zeros((len(products), len(GROUPS)))
    c = np.zeros((len(products), len(GROUPS)))
    for g in ans:
        if keys is not None and str(g.key) not in keys:
            continue
        s[row[g.key.product], col[GROUP_OF[g.type]]] += g.score
        c[row[g.key.product], col[GROUP_OF[g.type]]] += 1
    return s, c, products


def _multiplicity(idx: np.ndarray, n_products: int) -> np.ndarray:
    m = np.zeros((idx.shape[0], n_products))
    np.add.at(m, (np.arange(idx.shape[0])[:, None], idx), 1)
    return m


def replicate_counts(c: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return _multiplicity(idx, c.shape[0]) @ c


def _group_means(s: np.ndarray, c: np.ndarray, idx: np.ndarray):
    m = _multiplicity(idx, s.shape[0])
    tot_s, tot_c = m @ s, m @ c
    means = np.where(tot_c > 0, tot_s / np.maximum(tot_c, 1), np.nan)
    return means, tot_c


def replicate_endpoints(s: np.ndarray, c: np.ndarray, idx: np.ndarray, mask: np.ndarray):
    """Endpoint per replicate (one row of product draws each) and the number of replicates with a missing group."""
    means, tot_c = _group_means(s, c, idx)
    sel = means[:, mask]
    valid = ~np.isnan(sel)
    n_valid = valid.sum(axis=1)
    ep = np.where(n_valid > 0, np.nansum(sel, axis=1) / np.maximum(n_valid, 1), np.nan)
    return ep, int(np.any(tot_c[:, mask] == 0, axis=1).sum())


def paired_bootstrap_full(
    a: list[Graded], b: list[Graded], clusters: list[str] | None = None, n: int = 10000, seed: int = SEED,
    groups: tuple[str, ...] = PRIMARY_GROUPS, keys: set[str] | None = None,
) -> BootResult:
    def keyset(gs):
        return {str(g.key) for g in gs if g.type != "unanswerable" and (keys is None or str(g.key) in keys)}

    ka, kb = keyset(a), keyset(b)
    if not ka or ka != kb:
        raise ValueError("a and b must grade the same questions (paired by key), and there must be at least one")
    sa, ca, products = tables(a, clusters, keys)
    sb, cb, _ = tables(b, clusters or products, keys)
    mask = group_mask(groups)
    full = np.arange(len(products))[None, :]
    obs_a, _ = replicate_endpoints(sa, ca, full, mask)
    obs_b, _ = replicate_endpoints(sb, cb, full, mask)
    diff = float(obs_a[0] - obs_b[0])
    if np.isnan(diff):
        raise ValueError("no group has a question; the endpoint is undefined")
    idx = np.random.default_rng(seed).integers(0, len(products), size=(n, len(products)))
    ea, missing = replicate_endpoints(sa, ca, idx, mask)
    eb, _ = replicate_endpoints(sb, cb, idx, mask)
    d = (ea - eb)[~np.isnan(ea - eb)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return BootResult(diff, float(lo), float(hi), missing, n)


def paired_bootstrap(
    a: list[Graded], b: list[Graded], clusters: list[str] | None = None, n: int = 10000, seed: int = SEED, **kw: Any,
) -> tuple[float, float, float]:
    """(observed difference a-b of the primary endpoint, 2.5 percentile, 97.5 percentile), product-clustered."""
    r = paired_bootstrap_full(a, b, clusters, n, seed, **kw)
    return r.diff, r.lo, r.hi


def compute_bootstraps(
    by_arm: dict[str, list[Graded]], multihop: set[str], clusters: list[str] | None = None, n: int = 10000,
    seed: int = SEED, *, canonical_keys: set[str] | None = None, pilot: bool = False,
) -> dict[str, BootResult]:
    """The three pre-registered intervals. H3_all is the all-questions difference, reported next to H3.
    Unless pilot=True, every arm must grade exactly the canonical held answerable key set (no missing, no
    duplicate key); canonical_keys defaults to the held split's."""
    if not pilot:
        canon = canonical_keys if canonical_keys is not None else {
            k for k in _canonical("held") if "/ans/" in k}
        for arm, gs in by_arm.items():
            keys = [str(g.key) for g in gs if g.type != "unanswerable"]
            if len(keys) != len(set(keys)) or set(keys) != canon:
                raise ValueError(f"arm {arm} does not grade exactly the {len(canon)} canonical answerable slots")
    return {
        "H1": paired_bootstrap_full(by_arm["A5"], by_arm["A0"], clusters, n, seed),
        "H2": paired_bootstrap_full(by_arm["A5"], by_arm["A1f"], clusters, n, seed),
        "H3": paired_bootstrap_full(by_arm["A5"], by_arm["A4"], clusters, n, seed, keys=multihop),
        "H3_all": paired_bootstrap_full(by_arm["A5"], by_arm["A4"], clusters, n, seed),
    }


# ---------------------------------------------------------------- pre-registered multi-hop family
def read_multihop_families(prereg_text: str) -> set[str]:
    """Families are listed in PREREG.md, one per line: `multi-hop family: <normalised question template>`,
    where the template is `eval.questions.template` output."""
    fams = {m.group(1).strip() for m in re.finditer(r"^multi-hop family:\s*(.+)$", prereg_text, re.IGNORECASE | re.MULTILINE)}
    if not fams:
        raise ValueError("PREREG has no 'multi-hop family:' lines")
    return fams


def multihop_keys(questions: list[Question], families: set[str]) -> set[str]:
    return {str(q.key) for q in questions if q.type != "unanswerable" and template(q.text, q.product) in families}


# ---------------------------------------------------------------- H1-H4
def _ci(b: Any) -> tuple[float, float, float]:
    return float(b[0]), float(b[1]), float(b[2])


def decide(summary_by_arm: dict[str, dict], boot: dict[str, Any]) -> dict[str, dict]:
    """SPEC 6.2 as code. `summary_by_arm[arm]` holds summarize() output merged with usage_summary() output.
    `boot` holds H1 (A5-A0), H2 (A5-A1f), H3 (A5-A4 on multi-hop) and optionally H3_all."""
    a5, a1f = summary_by_arm["A5"], summary_by_arm["A1f"]
    d1, l1, _ = _ci(boot["H1"])
    d2, l2, _ = _ci(boot["H2"])
    d3, l3, _ = _ci(boot["H3"])
    tokens_ok = a5["median_input_tokens"] * 10 <= a1f["median_input_tokens"]
    return {
        "H1": {"diff": d1, "lo": l1, "pass": l1 > 0, "prior_expectation_met": d1 >= 15},
        "H2": {"diff": d2, "lo": l2, "interval_ok": l2 > -3, "tokens_ok": tokens_ok, "pass": l2 > -3 and tokens_ok,
               "token_ratio": a5["median_input_tokens"] / a1f["median_input_tokens"]},
        "H3": {"diff": d3, "lo": l3, "pass": l3 > 0, "prior_expectation_met": d3 >= 10,
               "all_questions": _ci(boot["H3_all"]) if "H3_all" in boot else None},
        "H4": {"pass": a5["unanswerable_rate"] >= a1f["unanswerable_rate"],
               "unanswerable_rate": {"A5": a5["unanswerable_rate"], "A1f": a1f["unanswerable_rate"]},
               "false_refusal_rate": {"A5": a5["false_refusal_rate"], "A1f": a1f["false_refusal_rate"]}},
    }
