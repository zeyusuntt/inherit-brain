"""Run an extraction layer under the file guard.

    uv run python -m inherit.extract.run --layer structural|names --products dev|all|P1,P2

Reads the bundle only, writes data/facts/<product>/<layer>.jsonl, prints node and edge counts.
The names layer writes names.jsonl (one row per meeting and speaker name) and names_facts.jsonl (the facts),
and prints speaker pairs and lines by resolution status.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from inherit.config import ROOT
from inherit.data.bundle import load_metadata
from inherit.data.records import build_records
from inherit.data.split import split
from inherit.extract.names import resolve_meeting_names
from inherit.extract.structural import structural_facts_from_records
from inherit.facts import Facts
from inherit import ioguard

LAYERS = {"structural": structural_facts_from_records, "names": None}

BASES = ("single", "score", "tie", "none")


def select_products(spec: str, available: list[str]) -> list[str]:
    if spec == "all":
        return sorted(available)
    if spec == "dev":
        return split(available)[0]
    names = spec.split(",")
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"unknown products: {unknown}")
    return names


def counts(f: Facts) -> tuple[Counter, Counter]:
    return Counter(n.label for n in f.nodes), Counter(e.type for e in f.edges)


def write_names(root: Path, product: str, records) -> None:
    facts, rows = resolve_meeting_names(records)
    out = root / "data" / "facts" / product
    out.mkdir(parents=True, exist_ok=True)
    (out / "names.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    facts.to_jsonl(out / "names_facts.jsonl")
    pairs, lines = Counter(r["basis"] for r in rows), Counter()
    for r in rows:
        lines[r["basis"]] += r["lines"]
    label = {"single": "resolved single", "score": "resolved by score", "tie": "ambiguous", "none": "unresolved"}
    print(f"{product}: pairs/lines " + ", ".join(f"{label[b]} {pairs[b]}/{lines[b]}" for b in BASES)
          + f"; edges {dict(sorted(counts(facts)[1].items()))}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True, choices=sorted(LAYERS))
    ap.add_argument("--products", required=True)
    ap.add_argument("--repo-root", default=str(ROOT))
    ap.add_argument("--bundle", default=None)
    args = ap.parse_args(argv)
    root = Path(args.repo_root)
    bundle = Path(args.bundle) if args.bundle else root / "data" / "bundle"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ioguard.install(root, root / "runs" / "access" / f"extract-{stamp}.jsonl")

    employees, customers = load_metadata(bundle)
    available = json.loads((bundle / "manifest.json").read_text())["products"] if (bundle / "manifest.json").exists() \
        else sorted(p.stem for p in (bundle / "products").glob("*.json"))
    for product in select_products(args.products, available):
        data = json.loads((bundle / "products" / f"{product}.json").read_text(encoding="utf-8"))
        records = build_records(data, employees, customers, product)
        if args.layer == "names":
            write_names(root, product, records)
            continue
        facts = LAYERS[args.layer](records)
        out = root / "data" / "facts" / product / f"{args.layer}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        facts.to_jsonl(out)
        nc, ec = counts(facts)
        print(f"{product}: nodes {dict(sorted(nc.items()))} edges {dict(sorted(ec.items()))} "
              f"conflicts {facts.conflicts} non_directory_people "
              f"{sum(1 for n in facts.nodes if n.label == 'Person' and not n.props['in_directory'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
