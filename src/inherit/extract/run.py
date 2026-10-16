"""Run an extraction layer under the file guard.

    uv run python -m inherit.extract.run --layer structural|names|rules|bugs|semantic --products dev|all|P1,P2 [--estimate]

Reads the bundle only, writes data/facts/<product>/<layer>.jsonl, prints node and edge counts.
The names layer writes names.jsonl (one row per meeting and speaker name) and names_facts.jsonl (the facts),
and prints speaker pairs and lines by resolution status.
The rules layer writes rules.jsonl and rules_report.json (aliases and evidence, scope counts, PR approval counts,
approval contradictions, reverts, demos, named QA eids, what is left for 4c) and prints a summary line.
The bugs layer writes bugs.jsonl and bugs_report.json. It calls the model through the metering proxy (start it with
`uv run python -m inherit.proxy &`); `--estimate` prints the calls and projected cost without sending anything.
The semantic layer (meetings, document reviews, Slack claims, PR decisions) writes semantic.jsonl and
semantic_report.json the same way and also supports `--estimate`.
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
from inherit.extract.bugs import estimate_bugs, run_bugs
from inherit.extract.names import resolve_meeting_names
from inherit.extract.rules import run_rules
from inherit.extract.semantic import estimate_semantic, run_semantic
from inherit.extract.structural import structural_facts_from_records
from inherit.facts import Facts
from inherit import ioguard

LAYERS = {"structural": structural_facts_from_records, "names": None, "rules": None, "bugs": None, "semantic": None}

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


def write_bugs(root: Path, product: str, records, estimate: bool) -> dict:
    if estimate:
        est = estimate_bugs(records)
        print(f"{product}: session calls {est['session_calls']} link calls (max) {est['link_calls_max']} "
              f"cached {est['cached']} projected {est['projected_usd']:.4f} USD")
        return est
    res = run_bugs(records)
    out = root / "data" / "facts" / product
    out.mkdir(parents=True, exist_ok=True)
    res.facts.to_jsonl(out / "bugs.jsonl")
    (out / "bugs_report.json").write_text(json.dumps(res.report, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
                                          encoding="utf-8")
    r = res.report
    print(f"{product}: bugs {r['bugs']} events {r['events_by_type']} linked/unmatched "
          f"{r['events']['linked']}/{r['events']['unmatched']} external {r['external_look_alikes']} "
          f"calls {r['calls']} cost {r['usage']['cost_usd']:.4f} USD failures {len(r['extraction_failures'])} "
          f"conflicts {res.facts.conflicts}")
    return r


def write_semantic(root: Path, product: str, records, estimate: bool, **kw) -> dict:
    if estimate:
        est = estimate_semantic(records, **kw)
        print(f"{product}: meeting calls {est['meeting_calls']} session calls {est['session_calls']} membership calls "
              f"(max) {est['membership_calls_max']} cached {est['cached']} projected {est['projected_usd']:.4f} USD")
        return est
    res = run_semantic(records, **kw)
    out = root / "data" / "facts" / product
    out.mkdir(parents=True, exist_ok=True)
    res.facts.to_jsonl(out / "semantic.jsonl")
    (out / "semantic_report.json").write_text(json.dumps(res.report, ensure_ascii=False, sort_keys=True, indent=1)
                                              + "\n", encoding="utf-8")
    r = res.report
    print(f"{product}: features {r['features']} events {r['feature_events']} contributors {r['contributors']} "
          f"pr links {r['pr_membership']['by_basis']} decisions {r['pr_decisions']['by_action']} "
          f"observations { {k: sum(v.values()) for k, v in r['observations'].items()} } demos {len(r['demo_links'])} "
          f"suggestions {r['suggestions']} calls {r['calls']} cost {r['usage']['cost_usd']:.4f} USD "
          f"failures {len(r['extraction_failures'])} complete {r['manifest']['complete']} "
          f"conflicts {res.facts.conflicts}")
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True, choices=sorted(LAYERS))
    ap.add_argument("--products", required=True)
    ap.add_argument("--repo-root", default=str(ROOT))
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--estimate", action="store_true", help="bugs and semantic layers: print calls and projected cost, send nothing")
    args = ap.parse_args(argv)
    root = Path(args.repo_root)
    bundle = Path(args.bundle) if args.bundle else root / "data" / "bundle"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    ioguard.install(root, root / "runs" / "access" / f"extract-{stamp}.jsonl")

    employees, customers = load_metadata(bundle)
    available = json.loads((bundle / "manifest.json").read_text())["products"] if (bundle / "manifest.json").exists() \
        else sorted(p.stem for p in (bundle / "products").glob("*.json"))
    totals: list[dict] = []
    for product in select_products(args.products, available):
        data = json.loads((bundle / "products" / f"{product}.json").read_text(encoding="utf-8"))
        records = build_records(data, employees, customers, product)
        if args.layer == "names":
            write_names(root, product, records)
            continue
        if args.layer == "bugs":
            totals.append(write_bugs(root, product, records, args.estimate))
            continue
        if args.layer == "semantic":
            totals.append(write_semantic(root, product, records, args.estimate))
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
    if args.layer == "semantic" and args.estimate:
        print(f"total projected {sum(t['projected_usd'] for t in totals):.4f} USD, "
              f"meeting calls {sum(t['meeting_calls'] for t in totals)}, "
              f"session calls {sum(t['session_calls'] for t in totals)}, "
              f"membership calls (max) {sum(t['membership_calls_max'] for t in totals)}")
    elif args.layer == "semantic":
        print(f"total cost {sum(t['usage']['cost_usd'] for t in totals):.4f} USD")
    elif args.layer == "bugs" and args.estimate:
        print(f"total projected {sum(t['projected_usd'] for t in totals):.4f} USD, "
              f"session calls {sum(t['session_calls'] for t in totals)}, "
              f"link calls (max) {sum(t['link_calls_max'] for t in totals)}")
    elif args.layer == "bugs":
        print(f"total cost {sum(t['usage']['cost_usd'] for t in totals):.4f} USD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
