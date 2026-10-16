"""Run an extraction layer under the file guard.

    uv run python -m inherit.extract.run --layer structural|names|rules --products dev|all|P1,P2

Reads the bundle only, writes data/facts/<product>/<layer>.jsonl, prints node and edge counts.
The names layer writes names.jsonl (one row per meeting and speaker name) and names_facts.jsonl (the facts),
and prints speaker pairs and lines by resolution status.
The rules layer writes rules.jsonl and rules_report.json (aliases and evidence, scope counts, PR approval counts,
approval contradictions, reverts, demos, named QA eids, what is left for 4c) and prints a summary line.
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
from inherit.extract.rules import run_rules
from inherit.extract.structural import structural_facts_from_records
from inherit.facts import Facts
from inherit import ioguard

LAYERS = {"structural": structural_facts_from_records, "names": None, "rules": None}

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


def write_rules(root: Path, product: str, records) -> None:
    facts, rep = run_rules(records)
    out = root / "data" / "facts" / product
    out.mkdir(parents=True, exist_ok=True)
    facts.to_jsonl(out / "rules.jsonl")
    (out / "rules_report.json").write_text(json.dumps(rep, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
                                           encoding="utf-8")
    prs = {k: sum(v.values()) for k, v in sorted(rep["pr_counts"].items())}
    appr = Counter()
    for v in rep["pr_counts"].values():
        appr.update(v)
    sc = {k: v.get("previous", 0) for k, v in rep["scope_counts"].items()}
    rv = {k: f"{v['applied']}/{v['applied'] + v['not_applied']}" for k, v in sorted(rep["reverts_by_reason"].items())}
    print(f"{product}: aliases {list(rep['aliases'])} previous {rep['previous_names']}; previous scope {sc}; "
          f"prs {prs} approval {dict(sorted(appr.items()))} contradictions {len(rep['approval_contradictions'])}; "
          f"reverts applied/total {rv}; demos {len(rep['demo_links'])} "
          f"(no subject {len(rep['demos_without_subject'])}); named QA eids {len(rep['named_qa_eids'])}; "
          f"left for 4c {rep['left_for_4c']}; conflicts {facts.conflicts}")


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
        if args.layer == "rules":
            write_rules(root, product, records)
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
