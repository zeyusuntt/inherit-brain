"""The pinned Task 13 pilot sample (F10, R12): built eval-side from dev questions only, hashed, recorded.

36 answerable questions allocated to the types in proportion to the dev type counts (largest remainder), 12
unanswerable. Within each type the allocation is split between the short and the long half of the six dev
products (by A1f context tokens) as evenly as the pools allow, then drawn with a seeded shuffle.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from eval.questions import DEV_PRODUCTS, Question, load_questions

SEED = 20261015
N_ANSWERABLE, N_UNANSWERABLE = 36, 12
TYPES = ("person", "company", "pr", "url", "content")
# A1f context tokens per dev product (task-11-a1f-report): the three smallest are the short half.
A1F_TOKENS = {"ActionGenie": 133_714, "AnomalyForce": 113_249, "ContentForce": 117_633,
              "KnowledgeForce": 124_864, "LeadForce": 137_033, "TrendForce": 111_704}
SHORT = tuple(sorted(sorted(A1F_TOKENS, key=A1F_TOKENS.get)[:3]))
LONG = tuple(sorted(set(A1F_TOKENS) - set(SHORT)))
SAMPLE_PATH = Path(__file__).resolve().parents[2] / "results" / "pilot" / "sample.json"


def allocate(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder proportional allocation; ties go to the earlier name."""
    n = sum(counts.values())
    quota = {k: total * v / n for k, v in counts.items()}
    out = {k: int(q) for k, q in quota.items()}
    for k in sorted(counts, key=lambda k: (-(quota[k] - out[k]), k))[: total - sum(out.values())]:
        out[k] += 1
    return out


def _draw(pool_short: list[Question], pool_long: list[Question], n: int, rng: random.Random) -> list[Question]:
    ns, nl = n // 2, n - n // 2
    if n % 2 and rng.random() < 0.5:
        ns, nl = nl, ns
    ns, nl = min(ns, len(pool_short)), min(nl, len(pool_long))
    ns = min(len(pool_short), n - nl)
    nl = min(len(pool_long), n - ns)
    rng.shuffle(pool_short)
    rng.shuffle(pool_long)
    return pool_short[:ns] + pool_long[:nl]


def build_sample(dev: list[Question] | None = None, seed: int = SEED) -> list[Question]:
    dev = dev if dev is not None else load_questions("dev")
    assert set(q.product for q in dev) <= set(DEV_PRODUCTS)
    rng = random.Random(seed)
    ans = [q for q in dev if q.type != "unanswerable"]
    quota = allocate({t: sum(q.type == t for q in ans) for t in TYPES}, N_ANSWERABLE)
    picked: list[Question] = []
    for t in (*TYPES, "unanswerable"):
        pool = [q for q in dev if q.type == t]
        n = N_UNANSWERABLE if t == "unanswerable" else quota[t]
        picked += _draw([q for q in pool if q.product in SHORT], [q for q in pool if q.product in LONG], n, rng)
    order = {q.key: i for i, q in enumerate(dev)}
    return sorted(picked, key=lambda q: order[q.key])


def sample_record(qs: list[Question]) -> dict:
    keys = [str(q.key) for q in qs]
    return {"seed": SEED, "short_products": list(SHORT), "long_products": list(LONG), "keys": keys,
            "sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest()}


def write_sample(path: Path = SAMPLE_PATH) -> dict:
    rec = sample_record(build_sample())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


def load_pinned(path: Path = SAMPLE_PATH) -> list[Question]:
    """The pinned sample as Question objects (dev only), verifying the recorded hash."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if hashlib.sha256("\n".join(rec["keys"]).encode()).hexdigest() != rec["sha256"]:
        raise ValueError("pilot sample hash mismatch")
    by_key = {str(q.key): q for q in load_questions("dev")}
    return [by_key[k] for k in rec["keys"]]


if __name__ == "__main__":
    r = write_sample()
    print(len(r["keys"]), r["sha256"])
