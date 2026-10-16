"""Run receipts: the facts anyone needs to check a run, and the export of a graded run to results/."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inherit.config import MODEL, ROOT
from inherit.graph.load import LAYER_FILES


@dataclass(frozen=True)
class RunEnv:
    """What a run depends on besides its arm and questions. Frozen into the manifest."""

    code_commit: str
    facts_hash: str  # over the frozen facts of every product
    bundle_sha: str
    model: str


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def facts_hash(facts_dir: Path, products: list[str] | None = None) -> str:
    """sha256 over the sorted relative paths of the fact-layer files the loader reads (its LAYER_FILES), each as
    path, length and bytes. Reports and other files are not part of the facts.

    With `products`, only those product directories are hashed (the split-specific hash); with None it is the
    frozen all-product hash the pre-registration records.
    """
    base = Path(facts_dir)
    h = hashlib.sha256()
    wanted = None if products is None else set(products)
    rels = sorted(
        p.relative_to(base) for p in base.glob("*/*")
        if p.is_file() and p.name in LAYER_FILES and (wanted is None or p.parent.name in wanted)
    )
    for rel in rels:
        data = (base / rel).read_bytes()
        h.update(rel.as_posix().encode("utf-8") + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()


def require_clean_tree(root: Path = ROOT) -> None:
    """Raise unless HEAD is a valid commit and nothing is modified, staged, or untracked under src, scripts, tests."""
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=root, capture_output=True, check=True, text=True).stdout

    try:
        git("rev-parse", "--verify", "HEAD")
        tracked = [x for x in git("status", "--porcelain", "--untracked-files=no").splitlines() if x.strip()]
        untracked = [x for x in git("status", "--porcelain", "--untracked-files=all", "--", "src", "scripts", "tests").splitlines()
                     if x.startswith("??")]
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"cannot establish a clean committed tree at {root}: {e}") from e
    if tracked or untracked:
        raise RuntimeError(f"working tree is not clean ({len(tracked)} modified, {len(untracked)} untracked in "
                           "src/scripts/tests); commit first. A held-out run needs a committed protocol")


def code_version(root: Path = ROOT) -> str:
    """HEAD, plus a short digest of uncommitted changes when the tree is dirty."""
    def git(*a: str) -> bytes:
        return subprocess.run(["git", *a], cwd=root, capture_output=True, check=True).stdout

    try:
        head = git("rev-parse", "HEAD").decode().strip()
        dirty = git("diff", "HEAD") + git("ls-files", "--others", "--exclude-standard")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + (f"-dirty-{hashlib.sha256(dirty).hexdigest()[:12]}" if dirty.strip() else "")


def current_env(root: Path = ROOT) -> RunEnv:
    from inherit.data.bundle import bundle_hash

    root = Path(root)
    return RunEnv(code_version(root), facts_hash(root / "data" / "facts"), bundle_hash(root / "data" / "bundle"), MODEL)


def ledger_cost_for_run(meter_path: Path, run_id: str) -> float | None:
    """USD settled on the ledger for every attempt tagged `<run id>/...`, earlier attempts included.
    None when there is no ledger file."""
    p = Path(meter_path)
    if not p.exists():
        return None
    pico = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict) and e.get("type") == "settle" and str(e.get("tag", "")).startswith(run_id + "/"):
            pico += int(e.get("cost", 0))
    return pico / 10**12


def build_receipt(
    manifest: dict[str, Any], manifest_sha: str, records_sha: str, output_sha: dict[str, str], counts: dict[str, int],
    status: str, started_utc: str, ended_utc: str, cost_usd: float, ledger_cost_usd: float | None,
) -> dict[str, Any]:
    return {
        "run_id": manifest["run_id"], "arm": manifest["arm"], "split": manifest["split"], "status": status,
        "started_utc": started_utc, "ended_utc": ended_utc,
        "code_commit": manifest["code_commit"], "bundle_sha": manifest["bundle_sha"],
        "facts_hash": manifest["facts_hash"], "split_facts_hash": manifest["split_facts_hash"],
        "arm_config_sha": manifest["arm_config_sha"], "model": manifest["model"],
        "prompt_versions": manifest["prompt_versions"],
        "manifest_sha256": manifest_sha, "records_sha256": records_sha,
        "output_sha256": output_sha, "counts": counts,
        "cost_usd": cost_usd, "ledger_cost_usd": ledger_cost_usd,
    }


def export_run(run_dir: Path, graded: list, out_root: Path) -> tuple[Path, Path]:
    """Write the tracked per-slot table and the receipt for a finished, fully graded run.

    results/tables/<run id>.jsonl holds, per slot, the arm's answer, failure, usage, latency and the judge
    decision and score: enough to recompute every published number. It holds no gold answer and no citation.
    results/receipts/<run id>.json is the run receipt plus the table hash and the judge identity, taken from the
    run's grading manifest. Nothing is written unless the manifest and record hashes match the receipt and the
    grades are exactly the persisted grades of this arm over these records.
    """
    from eval import grade as G
    from eval import run as R

    run_dir, out_root = Path(run_dir), Path(out_root)
    manifest = R.read_manifest(run_dir)
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    if receipt["status"] != "complete":
        raise ValueError(f"run {manifest['run_id']} is {receipt['status']}, not complete; finish it before exporting")
    lines = R.read_terminal_lines(run_dir)
    if receipt["manifest_sha256"] != sha256_file(run_dir / "manifest.json"):
        raise ValueError("manifest.json does not match the receipt")
    if receipt["records_sha256"] != sha256_file(run_dir / "records.jsonl"):
        raise ValueError("records.jsonl does not match the receipt")
    if receipt["output_sha256"] != {s: hashlib.sha256(l.encode()).hexdigest() for s, l in lines.items()}:
        raise ValueError("per-question output hashes do not match the receipt")
    if [str(g.key) for g in graded] != manifest["slots"]:
        raise ValueError("graded must cover every slot of the manifest, in manifest order")
    if graded != G.load_graded(run_dir):  # also checks arm and source-record hashes of every grade
        raise ValueError("graded does not match the persisted grades of this run")
    judge = G.read_grading_manifest(run_dir)
    records = R.read_records(run_dir)
    rows = []
    for g in graded:
        r = records[str(g.key)]
        rows.append({
            "key": str(g.key), "arm": g.arm, "type": g.type, "seen": g.seen, "outcome": g.outcome,
            "answer_ids": r.final.answer_ids, "evidence_ids": r.final.evidence_ids, "answer_text": r.final.answer_text,
            "failure": r.failure, "usage": r.final.usage.to_dict(), "latency_s": r.latency_s,
            "decision": g.decision, "score": g.score, "failed": g.failed,
        })
    table = out_root / "tables" / f"{manifest['run_id']}.jsonl"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    receipt.update(table_sha256=sha256_file(table), judge=judge)
    path = out_root / "receipts" / f"{manifest['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    return table, path
