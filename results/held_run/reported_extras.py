"""The two 'also reported' items of PREREG section 10 that RESULTS.md first left out: H1 and H2 split by seen and
unseen question strings, and the overlap of cited records with HERB's citations. Frozen code only, no model calls.

    uv run python results/held_run/reported_extras.py   # writes results/reported_extras.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from eval import grade as G  # noqa: E402
from eval import run as R  # noqa: E402
from eval import stats  # noqa: E402
from eval.questions import load_questions  # noqa: E402

ARMS = ["A5", "A4", "A0", "A1f", "A0f", "A4c", "A0v", "A2"]
RUNS = {a: ROOT / "runs" / "eval" / f"{a}-held" for a in ARMS}
PAIRS = {"H1": ("A5", "A0"), "H2": ("A5", "A1f")}


def main() -> None:
    graded = {a: G.load_graded(d) for a, d in RUNS.items()}
    held = load_questions("held")
    out = {"by_seen": {}, "citation_overlap": {}}
    for split, flag in (("seen", True), ("unseen", False)):
        keys = {str(g.key) for g in graded["A5"] if g.seen is flag}
        out["by_seen"][split] = {"n_questions": len(keys)}
        for name, (a, b) in PAIRS.items():
            xa = [g for g in graded[a] if str(g.key) in keys]
            xb = [g for g in graded[b] if str(g.key) in keys]
            r = stats.paired_bootstrap_full(xa, xb, None, 10000, stats.SEED, groups=stats.PRIMARY_GROUPS)
            out["by_seen"][split][name] = {"diff": r.diff, "lo": r.lo, "hi": r.hi,
                                           "missing_group_replicates": r.n_missing_stratum}
    for a, d in RUNS.items():
        out["citation_overlap"][a] = stats.citation_overlap(list(R.read_records(d).values()), held)
    (ROOT / "results" / "reported_extras.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
