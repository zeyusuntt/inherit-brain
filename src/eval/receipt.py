"""Run receipts: the facts anyone needs to check a run, and the export of a graded run to results/."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inherit.config import MODEL, ROOT


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
    """sha256 over the sorted relative paths of every fact-layer file, each as path, length and bytes.

    Dotfiles are skipped. With `products`, only files under those product directories are hashed
    (the split-specific hash); with None it is the frozen all-product hash the pre-registration records.
    """
    base = Path(facts_dir)
    h = hashlib.sha256()
    wanted = None if products is None else set(products)
    rels = sorted(
        p.relative_to(base) for p in base.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(base).parts)
    )
    for rel in rels:
        if wanted is not None and rel.parts[0] not in wanted:
            continue
        data = (base / rel).read_bytes()
        h.update(rel.as_posix().encode("utf-8") + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()


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
