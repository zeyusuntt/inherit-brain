"""Held-out run driver (Task 14). Lives outside the repo so the frozen tree stays clean; it only calls frozen code.

    uv run --project /Users/sunzeyu/dev/inherit-brain python run_held.py A5 [A4 ...]

Each arm goes through eval.run.run_arm(arm, "held"), which runs the launch preflight, filters questions by
results/freeze/arms.json and pushes the arm's cap relative to its spend so far. Before each arm the driver checks the
pre-registered stop rule (spend over 45 USD stops the run); the proxy's 50 USD hard cap stays in force regardless.
"""

import json
import sys
import time
import urllib.request

sys.path.insert(0, "/Users/sunzeyu/dev/inherit-brain/src")

from eval import run as R  # noqa: E402

STOP_USD = 45.0
WORKERS = {"A1f": 3}


def spend() -> dict:
    return json.loads(urllib.request.urlopen("http://127.0.0.1:8787/spend", timeout=10).read())


def main(arms: list[str]) -> None:
    for arm in arms:
        s = spend()["spent_usd"]
        if s > STOP_USD:
            print(f"STOP: spend {s:.3f} USD is over the {STOP_USD} USD stop rule before {arm}", flush=True)
            return
        t = time.time()
        print(f"== {arm}: start (spend {s:.3f} USD)", flush=True)
        run_dir = R.run_arm(arm, "held", workers=WORKERS.get(arm, R.MAX_WORKERS))
        recs = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines() if line.strip()]
        failed = sum(1 for r in recs if (r.get("final") or {}).get("outcome") == "run_failed")
        s2 = spend()
        arm_spend = (s2.get("per_arm", {}).get(arm) or {}).get("committed_usd")
        print(f"== {arm}: done in {(time.time() - t) / 60:.1f} min; {len(recs)} records, {failed} run_failed; "
              f"spend {s2['spent_usd']:.3f} USD (arm cumulative {arm_spend}); run dir {run_dir}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
