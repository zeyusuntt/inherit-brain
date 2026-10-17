"""Held-out grading and statistics (Task 14). Outside the repo; calls frozen code only.

    uv run --project /Users/sunzeyu/dev/inherit-brain python analyze_held.py grade   # grade + export every run
    uv run --project /Users/sunzeyu/dev/inherit-brain python analyze_held.py stats   # results/results.json

Comparisons follow PREREG.md section 7 and 8: H1 (A5-A0), H2 (A5-A1f, plus the token criterion), H3 (A5-A4 on the
multi-hop family questions), H3_all (A5-A4 on every question A4 answered), H4 (refusal rates); secondary contrasts
on the arms' pre-registered product subsets, each paired with A5 or A0 restricted to the same questions. Every
interval is stats.paired_bootstrap_full (product-clustered, 10,000 replicates, seed 20261015). The post-hoc
sensitivity analysis drops provider-side 502 slots from both arms of a pair; it is labelled post hoc.
"""

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path("/Users/sunzeyu/dev/inherit-brain")
sys.path.insert(0, str(ROOT / "src"))

from eval import grade as G  # noqa: E402
from eval import receipt as rc  # noqa: E402
from eval import run as R  # noqa: E402
from eval import stats  # noqa: E402
from eval.questions import load_questions, template  # noqa: E402
from inherit import llm  # noqa: E402

ARMS = ["A5", "A4", "A0", "A1f", "A0f", "A4c", "A0v", "A2"]
RUNS = {a: ROOT / "runs" / "eval" / f"{a}-held" for a in ARMS}


def spend() -> dict:
    return json.loads(urllib.request.urlopen("http://127.0.0.1:8787/spend", timeout=10).read())


def held_by_key() -> dict:
    return {str(q.key): q for q in load_questions("held")}


def grade_all() -> None:
    qmap = held_by_key()
    judge_now = (spend()["per_arm"].get("judge") or {}).get("committed_usd", 0.0)
    R._push_cap("judge", judge_now + 0.8)
    for arm, run_dir in RUNS.items():
        if not (run_dir / "receipt.json").exists():
            print(f"{arm}: no finished run, skipped", flush=True)
            continue
        slots = R.read_manifest(run_dir)["slots"]
        qs = [qmap[s] for s in slots]
        judge = G.Judge(run_dir / "judge_cache.jsonl")
        token = llm.ATTEMPT_TAG.set(f"{run_dir.name}/judge/1")
        try:
            graded = G.grade_run(run_dir, qs, judge)
        finally:
            llm.ATTEMPT_TAG.reset(token)
        rc.export_run(run_dir, graded, ROOT / "results")
        print(f"{arm}: graded {len(graded)}; spend {spend()['spent_usd']:.3f}", flush=True)


def _boot(a, b, keys=None, groups=stats.PRIMARY_GROUPS):
    if keys is not None:
        a = [g for g in a if str(g.key) in keys]
        b = [g for g in b if str(g.key) in keys]
    r = stats.paired_bootstrap_full(a, b, None, 10000, stats.SEED, groups=groups)
    return {"diff": r.diff, "lo": r.lo, "hi": r.hi, "missing_group_replicates": r.n_missing_stratum, "n": r.n}


def _ans_keys(gs):
    return {str(g.key) for g in gs if g.type != "unanswerable"}


def stats_all() -> None:
    qmap = held_by_key()
    graded = {a: G.load_graded(d) for a, d in RUNS.items() if (d / "grades.jsonl").exists()}
    records = {a: list(R.read_records(d).values()) for a, d in RUNS.items() if a in graded}
    full = {"A5", "A0", "A1f", "A0v"}  # every held question; the others run on pre-registered product subsets
    summary = {a: {**(stats.summarize(graded[a], expected_n=stats.HELD_N) if a in full
                      else stats.summarize(graded[a], pilot=True)), **stats.usage_summary(records[a])}
               for a in graded}
    fams = stats.read_multihop_families((ROOT / "PREREG.md").read_text(encoding="utf-8"))
    multihop = stats.multihop_keys(list(qmap.values()), fams)
    boot = {"H1": _boot(graded["A5"], graded["A0"]), "H2": _boot(graded["A5"], graded["A1f"])}
    a4k = _ans_keys(graded["A4"])
    boot["H3"] = _boot(graded["A5"], graded["A4"], keys=multihop & a4k) if multihop & a4k else None
    boot["H3_all"] = _boot(graded["A5"], graded["A4"], keys=a4k)
    paper = {"H1": _boot(graded["A5"], graded["A0"], groups=stats.GROUPS),
             "H2": _boot(graded["A5"], graded["A1f"], groups=stats.GROUPS)}
    sub = _ans_keys(graded["A0f"])
    secondary = {
        "A0f-A0 (subset)": _boot(graded["A0f"], graded["A0"], keys=sub),
        "A4-A0f (subset)": _boot(graded["A4"], graded["A0f"], keys=sub),
        "A5-A4c (subset)": _boot(graded["A5"], graded["A4c"], keys=_ans_keys(graded["A4c"])),
        "A0v-A0": _boot(graded["A0v"], graded["A0"]),
        "A5-A2 (A2 products)": _boot(graded["A5"], graded["A2"], keys=_ans_keys(graded["A2"])),
        "A0-A2 (A2 products)": _boot(graded["A0"], graded["A2"], keys=_ans_keys(graded["A2"])),
    }
    tup = {k: (v["diff"], v["lo"], v["hi"]) for k, v in boot.items() if v is not None}
    if boot["H3"] is None:
        tup["H3"] = None
    decisions = stats.decide({a: summary[a] for a in ("A5", "A1f")}, tup)

    # post hoc: drop provider-side 502 slots (infrastructure, not the arm) from both arms of each pair
    def infra(a):
        return {s for s, r in R.read_records(RUNS[a]).items()
                if r.final.outcome == "run_failed" and "502" in str(r.failure)}
    sens = {}
    for name, (x, y) in {"H1": ("A5", "A0"), "H2": ("A5", "A1f")}.items():
        drop = infra(x) | infra(y)
        keep = _ans_keys(graded[x]) - drop
        sens[name] = {**_boot(graded[x], graded[y], keys=keep), "dropped_slots": len(drop)}

    # post hoc: company answers given as CUST- ids, mapped to the company name through the customer directory
    # (the same mapping for every arm), rescored with the frozen grader's F1; everything else unchanged
    import dataclasses
    from inherit.data.bundle import BUNDLE_DIR, load_metadata
    _, customers = load_metadata(BUNDLE_DIR)
    cmap = {c["id"]: c["company"] for c in customers}

    def mapped(arm):
        recs = R.read_records(RUNS[arm])
        out_g, changed = [], 0
        for g in graded[arm]:
            r = recs.get(str(g.key))
            if g.type == "company" and not g.failed and r is not None and any(x in cmap for x in r.final.answer_ids):
                ids = [cmap.get(x, x) for x in r.final.answer_ids]
                g = dataclasses.replace(g, score=100.0 * G.f1_score_sets(qmap[str(g.key)].gold, ids))
                changed += 1
            out_g.append(g)
        return out_g, changed
    mg = {a: mapped(a) for a in ("A5", "A0", "A1f")}
    cust = {"changed_company_answers": {a: mg[a][1] for a in mg},
            "primary_endpoint": {a: stats.summarize(mg[a][0], expected_n=stats.HELD_N).get("primary") for a in mg},
            "H1": _boot(mg["A5"][0], mg["A0"][0]), "H2": _boot(mg["A5"][0], mg["A1f"][0])}

    out = {"summary": summary, "boot": boot, "paper_style": paper, "secondary": secondary,
           "decisions": decisions, "multihop_questions": len(multihop),
           "post_hoc_sensitivity_excluding_provider_502": sens,
           "post_hoc_sensitivity_company_cust_ids_mapped": cust}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("boot", "decisions", "post_hoc_sensitivity_excluding_provider_502")},
                     indent=1, default=float))


if __name__ == "__main__":
    {"grade": grade_all, "stats": stats_all}[sys.argv[1]]()
