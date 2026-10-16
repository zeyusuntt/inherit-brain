"""`make audit`: three checks anyone can run. A missing input is a FAIL, never a skip.

1. The bundle manifest exists, its hash matches the files, and no bundle file has a forbidden key.
2. At least one extraction access log exists, every event was allowed, and every in-repo path
   is under an allowed data dir.
3. Run receipts all carry the same facts hash, equal to the one in PREREG.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from inherit.config import ROOT
from inherit.data.scan import FORBIDDEN_KEYS, find_forbidden_keys  # noqa: F401
from inherit.ioguard import _real, _under, repo_dir_allowed, _is_write_mode

PREREG_HASH_RE = re.compile(r"facts[ _-]?hash\W{0,10}([0-9a-f]{64})", re.IGNORECASE)


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: list[str] = field(default_factory=list)


def check_bundle(bundle: Path, expected_products: int | None = None) -> CheckResult:
    from inherit.data.bundle import ALLOWED_METADATA, MANIFEST, bundle_hash

    name = "1. bundle has no forbidden keys"
    bundle = Path(bundle)
    if not bundle.is_dir():
        return CheckResult(name, False, [f"bundle directory {bundle} does not exist; run make bundle"])
    problems: list[str] = []
    manifest_path = bundle / MANIFEST
    declared: list[str] | None = None
    recorded = None
    if not manifest_path.exists():
        problems.append(f"bundle manifest {manifest_path} is missing; run make bundle")
    else:
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            recorded = m["sha256"]
            declared = m["products"]
            dk = m["dropped_keys"]
            if not (isinstance(recorded, str) and re.fullmatch(r"[0-9a-f]{64}", recorded)):
                raise ValueError("sha256 is not a 64-char hex string")
            if not (isinstance(declared, list) and declared and all(isinstance(x, str) for x in declared)
                    and len(set(declared)) == len(declared)):
                raise ValueError("products must be a non-empty list of unique names")
            if not (isinstance(dk, dict) and set(dk) == set(declared)
                    and all(isinstance(v, list) for v in dk.values())):
                raise ValueError("dropped_keys must map every declared product to a list")
        except (ValueError, KeyError, TypeError) as e:
            problems.append(f"manifest.json is invalid: {e}")
            declared = None
    if declared is not None and expected_products is not None and len(declared) != expected_products:
        problems.append(f"manifest declares {len(declared)} products, expected {expected_products}")

    n_scanned = 0
    seen_products: set[str] = set()
    seen_meta: set[str] = set()
    for p in sorted(x for x in bundle.rglob("*") if x.is_file()):
        rel = p.relative_to(bundle).as_posix()
        if rel == MANIFEST:
            kind = None
        elif rel.startswith("products/") and rel.count("/") == 1 and rel.endswith(".json"):
            kind = "product"
            seen_products.add(rel[len("products/"):-len(".json")])
        elif rel.startswith("metadata/") and rel.count("/") == 1 and rel.split("/")[1] in ALLOWED_METADATA:
            kind = "meta"
            seen_meta.add(rel.split("/")[1])
        else:
            problems.append(f"{rel}: unexpected file in bundle")
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            problems.append(f"{rel}: not valid JSON ({e})")
            continue
        problems += find_forbidden_keys(data, rel)
        if kind:
            n_scanned += 1
    for f in ALLOWED_METADATA:
        if f not in seen_meta:
            problems.append(f"metadata/{f} is missing from the bundle")
    if declared is not None:
        for missing in sorted(set(declared) - seen_products):
            problems.append(f"declared product {missing} has no file in the bundle")
        for extra in sorted(seen_products - set(declared)):
            problems.append(f"products/{extra}.json is not declared in the manifest")
    if recorded is not None and not problems:
        actual = bundle_hash(bundle)
        if actual != recorded:
            problems.append(f"bundle sha256 {actual} does not match manifest {recorded}")
    if problems:
        return CheckResult(name, False, problems)
    return CheckResult(name, True, [f"{len(seen_products)} products and {len(seen_meta)} metadata files scanned, manifest hash matches"])


def check_access_log(access_dir: Path, repo_root: Path) -> CheckResult:
    name = "2. extraction touched only allowed paths"
    access_dir = Path(access_dir)
    logs = sorted(access_dir.glob("*.jsonl")) if access_dir.is_dir() else []
    if not logs:
        return CheckResult(name, False, [f"no access log under {access_dir}; run extraction under ioguard"])
    root = _real(repo_root)
    problems: list[str] = []
    n_events = 0
    for log in logs:
        count = 0
        for i, line in enumerate(log.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                path, allowed = e["path"], e["allowed"]
            except (ValueError, KeyError, TypeError):
                problems.append(f"{log.name}:{i}: unreadable log line")
                continue
            count += 1
            if allowed is not True:
                problems.append(f"{log.name}:{i}: refused access to {path}")
            if str(e.get("mode", "")).startswith(("os.", "subprocess.")):
                continue
            real = _real(path)
            if _under(real, root) and not repo_dir_allowed(real, root, write=_is_write_mode(e.get("mode"))):
                problems.append(f"{log.name}:{i}: in-repo path outside allowed dirs: {path}")
        if count == 0:
            problems.append(f"{log.name}: log has no events")
        n_events += count
    if problems:
        return CheckResult(name, False, problems)
    return CheckResult(name, True, [f"{len(logs)} logs, {n_events} events, 0 refused"])


def check_receipts(receipt_dirs: list[Path], prereg: Path) -> CheckResult:
    name = "3. receipts share the pre-registered facts hash"
    problems: list[str] = []
    prereg = Path(prereg)
    expected = None
    if not prereg.exists():
        problems.append(f"{prereg.name} is missing")
    else:
        m = PREREG_HASH_RE.search(prereg.read_text(encoding="utf-8"))
        if not m:
            problems.append(f"{prereg.name} has no 'facts hash: <sha256>' line")
        else:
            expected = m.group(1).lower()
    receipts = sorted(p for d in receipt_dirs if Path(d).is_dir() for p in Path(d).rglob("*.json"))
    if not receipts:
        problems.append("no receipts under " + " or ".join(str(d) for d in receipt_dirs))
    hashes: dict[str, list[str]] = {}
    for p in receipts:
        try:
            h = json.loads(p.read_text(encoding="utf-8"))["facts_hash"]
        except (ValueError, KeyError, TypeError):
            problems.append(f"{p.name}: no readable facts_hash")
            continue
        hashes.setdefault(str(h), []).append(p.name)
    if len(hashes) > 1:
        problems.append(f"receipts carry {len(hashes)} different facts hashes: " + ", ".join(sorted(hashes)))
    if expected is not None and hashes and set(hashes) != {expected}:
        problems.append(f"receipt facts hash does not match {prereg.name} ({expected})")
    if problems:
        return CheckResult(name, False, problems)
    return CheckResult(name, True, [f"{len(receipts)} receipts, one facts hash, matches {prereg.name}"])


def audit(root: Path = ROOT, expected_products: int | None = None) -> list[CheckResult]:
    root = Path(root)
    return [
        check_bundle(root / "data" / "bundle", expected_products),
        check_access_log(root / "runs" / "access", root),
        check_receipts([root / "results" / "receipts", root / "runs" / "receipts"], root / "PREREG.md"),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the three audit checks.")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--expect-products", type=int, default=30)
    args = ap.parse_args(argv)
    results = audit(Path(args.root), args.expect_products)
    for r in results:
        print(f"{'PASS' if r.ok else 'FAIL'}  {r.name}")
        for d in r.details[:20]:
            print(f"      {d}")
        if len(r.details) > 20:
            print(f"      ... {len(r.details) - 20} more")
    failed = [r for r in results if not r.ok]
    print(f"audit: {len(results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
