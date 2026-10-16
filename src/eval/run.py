"""Run one arm over a split in fixed slots, with no silent loss and no mixed protocols.

Layout of runs/eval/<run id>/ (git-ignored, large):
  manifest.json   written once before dispatch: arm, split, code, facts, bundle, prompts, model, ordered slots
  started.jsonl   one line per attempt, written before the arm is called
  records.jsonl   exactly one terminal RunRecord line per slot (plus the attempt number)
  tools/*.gz      the tool response envelopes of each slot
  receipt.json    written at the end and on interruption
  grades.jsonl    written by the separate grading step (eval.grade)

Recovery policy (frozen): a slot started but without a terminal record is run again once on resume. A slot
that has been started twice with no terminal record is closed as run_failed and not paid for again. The spend
of earlier attempts stays in the ledger and in the receipt's ledger cost. The arm owns its 300 s deadline and
writes its own terminal result; the runner never writes a second record for a slot.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from eval import receipt as rc
from eval.arms import ARMS, ArmResult
from eval.questions import Question, load_questions, snapshot_hash
from inherit import llm
from inherit.config import ROOT
from inherit.contracts import FinalAnswer, QuestionKey, RunRecord, Usage

SEED = 20261015
MAX_WORKERS = 4
MAX_STARTS = 2  # first attempt plus one re-run
PREREG_FACTS_RE = re.compile(r"facts[ _-]?hash\W{0,10}([0-9a-f]{64})", re.IGNORECASE)


class RunStateError(RuntimeError):
    pass


class IncompatibleResume(RunStateError):
    pass


class DuplicateTerminalRecord(RunStateError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fname(slot: str) -> str:
    return slot.replace("/", "__")


def sample_slots(questions: list[Question], limit: int, seed: int = SEED) -> list[Question]:
    """A seeded dev pilot sample stratified by type and product, returned in the frozen question order.

    Types are taken in turn, and within a type the products take turns, so a small limit still covers every type
    and spreads over products."""
    if limit >= len(questions):
        return list(questions)
    rng = random.Random(seed)
    by_type: dict[str, dict[str, list[Question]]] = {}
    for q in questions:
        by_type.setdefault(q.type, {}).setdefault(q.product, []).append(q)
    lanes: list[list[Question]] = []
    for t in sorted(by_type):
        order = sorted(by_type[t])
        rng.shuffle(order)
        pools = []
        for p in order:
            rng.shuffle(by_type[t][p])
            pools.append(by_type[t][p])
        lane: list[Question] = []
        while any(pools):
            lane += [pool.pop() for pool in pools if pool]
        lanes.append(lane)
    picked: list[Question] = []
    while len(picked) < limit:
        for lane in lanes:
            if lane and len(picked) < limit:
                picked.append(lane.pop(0))
    position = {q.key: i for i, q in enumerate(questions)}
    return sorted(picked, key=lambda q: position[q.key])


# ---------------------------------------------------------------- state on disk
def read_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))


def _read_lines(path: Path) -> list[str]:
    return [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def read_terminal_lines(run_dir: Path) -> dict[str, str]:
    """slot -> stored line. Raises on a duplicate terminal record or a slot outside the manifest."""
    run_dir = Path(run_dir)
    slots = set(read_manifest(run_dir)["slots"])
    out: dict[str, str] = {}
    for line in _read_lines(run_dir / "records.jsonl"):
        slot = str(QuestionKey.from_dict(json.loads(line)["key"]))
        if slot not in slots:
            raise RunStateError(f"record for {slot} is not in the manifest of {run_dir.name}")
        if slot in out:
            raise DuplicateTerminalRecord(f"two terminal records for {slot} in {run_dir.name}")
        out[slot] = line
    return out


def read_records(run_dir: Path) -> dict[str, RunRecord]:
    return {s: RunRecord.from_dict(json.loads(l)) for s, l in read_terminal_lines(run_dir).items()}


def read_starts(run_dir: Path) -> Counter:
    return Counter(json.loads(l)["slot"] for l in _read_lines(Path(run_dir) / "started.jsonl"))


def failed_record(key: QuestionKey, arm: str, failure: str, latency_s: float = 0.0) -> RunRecord:
    final = FinalAnswer("run_failed", "", [], [], [], Usage(0, 0, 0.0, 0))
    return RunRecord(key, arm, final, [], [], latency_s, failure)


def reconcile(run_dir: Path) -> list[RunRecord]:
    """One record per manifest slot, in slot order. A slot with no terminal record is counted as run_failed."""
    run_dir = Path(run_dir)
    m = read_manifest(run_dir)
    have = read_records(run_dir)
    return [have.get(s) or failed_record(QuestionKey.parse(s), m["arm"], "no terminal record") for s in m["slots"]]


def _append(path: Path, line: str, lock: threading.Lock) -> None:
    with lock, open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------- run
def _validate(res: Any) -> ArmResult:
    if not isinstance(res, ArmResult):
        raise TypeError(f"arm returned {type(res).__name__}, expected ArmResult")
    f = res.final
    if not isinstance(f, FinalAnswer):
        raise TypeError("ArmResult.final is not a FinalAnswer")
    if not isinstance(f.answer_text, str):
        raise TypeError("answer_text is not a string")
    for name in ("answer_ids", "evidence_ids"):
        v = getattr(f, name)
        if not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
            raise TypeError(f"{name} is not a list of strings")
    return res


def _check_facts(split: str, env: rc.RunEnv, expected: str | None, root: Path) -> None:
    if expected is None and split == "held":
        prereg = root / "PREREG.md"
        m = PREREG_FACTS_RE.search(prereg.read_text(encoding="utf-8")) if prereg.exists() else None
        if not m:
            raise RunStateError("a held-out run needs the pre-registered facts hash: PREREG.md has no 'facts hash:' line")
        expected = m.group(1).lower()
    if expected is not None and env.facts_hash != expected:
        raise RunStateError(f"facts hash {env.facts_hash} does not match the pre-registered {expected}")


def _push_cap(arm: str, usd: float) -> None:
    from inherit.llm import proxy_url

    req = urllib.request.Request(
        proxy_url().removesuffix("/v1") + "/admin/arm_cap", data=json.dumps({"arm": arm, "usd": usd}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=10).read()


def _launch_check(root: Path) -> None:
    from eval import prereg

    res = prereg.launch_check(root)
    if not res["pass"]:
        raise RunStateError("held-out launch refused:\n  " + "\n  ".join(res["failures"]))


def _read_arm_spend(arm: str) -> float:
    """The arm's committed spend in USD from the proxy's /spend (0 if the proxy has not seen the arm)."""
    from inherit.llm import proxy_url

    with urllib.request.urlopen(proxy_url().removesuffix("/v1") + "/spend", timeout=10) as r:
        per = json.loads(r.read()).get("per_arm", {})
    return float(per.get(arm, {}).get("committed_usd", 0.0))


def _held_arm_spec(root: Path, arm: str) -> dict:
    """The arm's entry in results/freeze/arms.json: {"products": list or None (all), "sub_cap_usd": x or None}."""
    f = root / "results" / "freeze" / "arms.json"
    try:
        spec = json.loads(f.read_text(encoding="utf-8"))["arms"].get(arm)
    except (OSError, ValueError, KeyError, AttributeError) as e:
        raise RunStateError(f"results/freeze/arms.json is unreadable: {e}") from e
    if spec is None:
        raise RunStateError(f"arm {arm} is not listed in results/freeze/arms.json; a held-out run follows that file")
    prods = spec.get("products", "all")
    if prods != "all" and not (isinstance(prods, list) and prods and all(isinstance(p, str) for p in prods)):
        raise RunStateError(f"arms.json: arm {arm} has a bad product list")
    return {"products": None if prods == "all" else sorted(prods), "sub_cap_usd": spec.get("sub_cap_usd")}


def run_arm(
    arm: str,
    split: Literal["dev", "held"],
    limit: int | None = None,
    seed: int = SEED,
    *,
    root: Path = ROOT,
    questions: list[Question] | None = None,
    env: rc.RunEnv | None = None,
    run_id: str | None = None,
    workers: int = MAX_WORKERS,
    expected_facts_hash: str | None = None,
    cap_setter: Callable[[str, float], None] = _push_cap,
    meter_path: Path | None = None,
    clean_check: Callable[[Path], None] = rc.require_clean_tree,
    launch_check: Callable[[Path], None] = _launch_check,
    arm_spec: Callable[[Path, str], dict] = _held_arm_spec,
    spend_reader: Callable[[str], float] = _read_arm_spend,
) -> Path:
    """Run `arm` over every slot of `split` and return the run directory. Resumes a compatible earlier run."""
    root = Path(root)
    if split not in ("dev", "held"):
        raise ValueError(f"split must be 'dev' or 'held', got {split!r}")
    if limit is not None and split != "dev":
        raise ValueError("limit is allowed only on the dev split")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer")
    if not (1 <= workers <= MAX_WORKERS):
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    run_id = run_id or f"{arm}-{split}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError(f"bad run id {run_id!r}")
    a = ARMS[arm]
    if split == "held":
        try:
            clean_check(root)
        except RuntimeError as e:
            raise RunStateError(str(e)) from e
        launch_check(root)  # before any held question is loaded
        spec = arm_spec(root, arm)
    else:
        spec = {"products": None, "sub_cap_usd": None}
    env = env or rc.current_env(root)
    _check_facts(split, env, expected_facts_hash, root)

    run_dir = root / "runs" / "eval" / run_id
    mpath = run_dir / "manifest.json"
    cap_base = None
    if split == "held":  # the proxy's arm cap is cumulative; a held run's budget starts at the spend so far
        cap_base = read_manifest(run_dir).get("cap_base_usd") if mpath.exists() else spend_reader(arm)
    qs = list(questions) if questions is not None else load_questions(split)
    if spec["products"] is not None:
        qs = [q for q in qs if q.product in spec["products"]]
        if not qs:
            raise RunStateError(f"arm {arm}: no held-out question in its arms.json products")
    if limit is not None:
        qs = sample_slots(qs, limit, seed)
    manifest = {
        "run_id": run_id, "arm": arm, "split": split, "code_commit": env.code_commit,
        "arm_config_sha": a.config_sha(), "facts_hash": env.facts_hash,
        "split_facts_hash": rc.facts_hash(root / "data" / "facts", sorted({q.product for q in qs})),
        "bundle_sha": env.bundle_sha, "prompt_versions": a.prompt_versions, "model": env.model,
        "limit": limit, "seed": seed, "cap_base_usd": cap_base,
        "products": sorted({q.product for q in qs}) if split == "held" else None, "question_sha": snapshot_hash(qs), "slots": [str(q.key) for q in qs],
    }
    if mpath.exists():
        old = read_manifest(run_dir)
        diff = sorted(k for k in set(old) | set(manifest) if old.get(k) != manifest.get(k))
        if diff:
            raise IncompatibleResume(f"run {run_id} was started with a different {', '.join(diff)}; refusing to resume")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        tmp = mpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        tmp.replace(mpath)
    (run_dir / "tools").mkdir(exist_ok=True)

    done = read_terminal_lines(run_dir)  # raises on a duplicate terminal record
    starts = read_starts(run_dir)
    prior = run_dir / "receipt.json"
    started_utc = json.loads(prior.read_text())["started_utc"] if prior.exists() else _now()
    lock = threading.Lock()

    def finish(rec: RunRecord, attempt: int, res: ArmResult | None) -> None:
        slot = str(rec.key)
        if res is not None:
            with gzip.open(run_dir / "tools" / f"{_fname(slot)}.json.gz", "wt", encoding="utf-8") as f:
                json.dump(res.envelopes, f, default=str)
        _append(run_dir / "records.jsonl", json.dumps({**rec.to_dict(), "attempt": attempt}), lock)

    abort = threading.Event()  # set on KeyboardInterrupt or any non-Exception failure: no slot starts after it

    def work(q: Question) -> None:
        if abort.is_set():
            return
        try:
            _work(q)
        except BaseException:
            abort.set()
            raise

    def _work(q: Question) -> None:
        slot = str(q.key)
        if starts[slot] >= MAX_STARTS:
            finish(failed_record(q.key, arm, f"started {starts[slot]} times without a terminal record; not run again"),
                   starts[slot], None)
            return
        attempt = starts[slot] + 1
        tag = f"{run_id}/{slot}/{attempt}"
        _append(run_dir / "started.jsonl", json.dumps({"slot": slot, "attempt": attempt, "tag": tag, "ts": _now()}), lock)
        token = llm.ATTEMPT_TAG.set(tag)
        t0 = time.monotonic()
        res: ArmResult | None = None
        try:
            res = _validate(a.run(q.exec_input()))
            f = res.final
            failure = res.failure or ("arm reported run_failed" if f.outcome == "run_failed" else None)
            rec = RunRecord(q.key, arm, f, list(res.calls), list(f.trace), time.monotonic() - t0, failure)
        except Exception as e:  # noqa: BLE001 - any arm error is this slot's terminal record
            rec = failed_record(q.key, arm, f"{type(e).__name__}: {e}", time.monotonic() - t0)
            res = None
        finally:
            llm.ATTEMPT_TAG.reset(token)
        finish(rec, attempt, res)

    pending = [q for q in qs if str(q.key) not in done]
    cap = spec["sub_cap_usd"] if spec["sub_cap_usd"] is not None else a.sub_cap_usd  # arms.json wins on held
    if pending and cap is not None:
        cap_setter(arm, cap + (cap_base or 0.0))
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        for fut in as_completed([ex.submit(work, q) for q in pending]):
            fut.result()
    except BaseException:
        abort.set()
        raise
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
        _write_receipt(run_dir, manifest, started_utc, meter_path or root / "runs" / "meter.jsonl")
    return run_dir


def _write_receipt(run_dir: Path, manifest: dict, started_utc: str, meter_path: Path) -> dict:
    lines = read_terminal_lines(run_dir)
    recs = {s: RunRecord.from_dict(json.loads(l)) for s, l in lines.items()}
    starts = read_starts(run_dir)
    counts = {
        "slots": len(manifest["slots"]), "finished": len(recs),
        "run_failed": sum(r.final.outcome == "run_failed" for r in recs.values()),
        "rerun_slots": sum(1 for s in manifest["slots"] if starts[s] >= 2),
        "attempts": sum(starts.values()),
    }
    status = "complete" if len(recs) == len(manifest["slots"]) else "interrupted"
    receipt = rc.build_receipt(
        manifest, rc.sha256_file(run_dir / "manifest.json"),
        rc.sha256_file(run_dir / "records.jsonl") if (run_dir / "records.jsonl").exists() else "",
        {s: hashlib.sha256(l.encode()).hexdigest() for s, l in lines.items()}, counts, status,
        started_utc, _now(), sum(c.cost_usd for r in recs.values() for c in r.calls),
        rc.ledger_cost_for_run(meter_path, manifest["run_id"]))
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    return receipt
