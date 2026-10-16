"""Run one arm on dev questions, grade the run, print a per-type summary and the ledger cost.

    uv run python scripts/run_dev.py --arm A5 --products ActionGenie [--types person,company]
        [--sample N --seed S] [--run-id ID]

Dev only: held-out products are refused before any question file is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval import grade as G  # noqa: E402
from eval import run as R  # noqa: E402
from eval import receipt as rc  # noqa: E402
from eval import stats  # noqa: E402
from eval.questions import DEV_PRODUCTS, Question, load_questions  # noqa: E402
from inherit import llm  # noqa: E402
from inherit.config import ROOT  # noqa: E402

TYPES = ("person", "company", "pr", "url", "content", "unanswerable")
JUDGE_CAP_USD, ARM_CAP_USD = 0.5, 2.5  # the arm cap is cumulative over all runs of that arm name


def build_questions(products: list[str], types: list[str] | None = None, sample: int | None = None,
                    seed: int = R.SEED) -> list[Question]:
    """The explicit ordered dev question list. Refuses any product outside the six dev products."""
    bad = [p for p in products if p not in DEV_PRODUCTS]
    if bad:
        raise SystemExit(f"refusing non-dev products {bad}; dev products are {list(DEV_PRODUCTS)}")
    unknown = [t for t in types or [] if t not in TYPES]
    if unknown:
        raise SystemExit(f"unknown question types {unknown}; known types are {list(TYPES)}")
    qs = [q for q in load_questions("dev") if q.product in products and (not types or q.type in types)]
    if sample is not None:
        qs = R.sample_slots(qs, sample, seed)
    if not qs:
        raise SystemExit("no questions selected")
    return qs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--products", default=",".join(DEV_PRODUCTS), help="comma-separated dev products")
    ap.add_argument("--types", help="comma-separated question types")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--seed", type=int, default=R.SEED)
    ap.add_argument("--run-id")
    ap.add_argument("--pilot-sample", action="store_true", help="use the pinned pilot list (results/pilot/sample.json)")
    ap.add_argument("--arm-cap", type=float, help="cumulative arm cap in USD (default 2.5)")
    ap.add_argument("--judge-cap", type=float, help="cumulative judge cap in USD (default 0.5)")
    a = ap.parse_args(argv)
    if a.pilot_sample:
        from eval import pilot

        qs = pilot.load_pinned()
    else:
        qs = build_questions(a.products.split(","), a.types.split(",") if a.types else None, a.sample, a.seed)
    run_id = a.run_id or f"{a.arm}-dev-{'+'.join(sorted({q.product for q in qs}))}"
    R._push_cap("judge", a.judge_cap or JUDGE_CAP_USD)
    R._push_cap(a.arm, a.arm_cap or ARM_CAP_USD)
    run_dir = R.run_arm(a.arm, "dev", questions=qs, run_id=run_id)
    judge = G.Judge(run_dir / "judge_cache.jsonl")
    token = llm.ATTEMPT_TAG.set(f"{run_id}/judge/1")  # judge spend lands under the run's ledger tag
    try:
        graded = G.grade_run(run_dir, qs, judge)
    finally:
        llm.ATTEMPT_TAG.reset(token)
    print(json.dumps(stats.summarize(graded, pilot=True)["per_type"], indent=1))
    print("run dir:", run_dir)
    cost = rc.ledger_cost_for_run(ROOT / "runs" / "meter.jsonl", run_id)
    print(f"ledger cost USD (run + judge, tag {run_id}/): {cost}")


if __name__ == "__main__":
    main()
