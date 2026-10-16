"""The independent check of the counterfactual (spec 5.5 program layer), the comparison runner and the audit hook.

    uv run python -m inherit.whatif_check --products dev [--servers 4.22,6.0]

`check(snapshot, person_id)` recomputes the three groups and `bound_hit` in plain Python from the exported snapshot
(whatif_schema.Snapshot: nodes with their stored properties and the original typed edges). It shares with whatif.py the
snapshot and the schema (names of labels, edge types and properties) and nothing else: it selects the open items,
derives owners and dependency direction from the original edges, and walks the dependencies itself. It does not
import the graph implementation, and it never sees a projected edge, an owner set or a result row of it.

Definitions (frozen in .superpowers/sdd/PLAN/task-12-decisions.md): open ownable work is an active bug or a PR with
state 'open' and merged False; a feature is eligible for group 3 when discussed_not_implemented is True, or when implemented is not True
and an ABOUT-development PR is open and unmerged.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict

from inherit.config import ROOT, load_settings
from inherit.whatif_schema import MAX_HOPS, Snapshot, WhatIf, export_snapshot, person_node_id

# (edge type, source label, target label, required edge properties, which end depends on the other)
DEPENDENCY_RULES = (
    ("IMPLEMENTED_BY", "Feature", "PR", {}, "src"),
    ("ABOUT", "PR", "Feature", {"kind": "development"}, "dst"),
    ("FIXES", "PR", "Bug", {}, "dst"),
)


def _position(message_id) -> tuple[int, int]:
    """Record position of a message id 'YYYYMMDD-index-hash'; unparseable parts sort first."""
    parts = (message_id or "").split("-")
    return tuple(int(parts[i]) if len(parts) > i and parts[i].isdigit() else -1 for i in (0, 1))


def check(snap: Snapshot, person_id: str) -> WhatIf:
    pid = person_node_id(person_id)
    node = {n.id: n for n in snap.nodes}
    if pid not in node or node[pid].label != "Person":
        raise ValueError(f"no such person {person_id!r}")
    ownable = {n.id for n in snap.nodes if (n.label == "Bug" and n.props.get("active") is True)
               or (n.label == "PR" and n.props.get("state") == "open" and n.props.get("merged") is False)}
    pending = {e.dst for e in snap.edges if e.type == "ABOUT" and e.props.get("kind") == "development"
               and e.src in ownable and node[e.src].label == "PR"}  # features with a development PR still open
    eligible = ownable | {n.id for n in snap.nodes if n.label == "Feature" and (
        n.props.get("discussed_not_implemented") is True or (n.props.get("implemented") is not True and n.id in pending))}
    # owners: a bug's people on its latest ASSIGNED event; a PR's OPENED authors
    events, assigned, owners = defaultdict(list), defaultdict(set), defaultdict(set)
    for e in snap.edges:
        if e.type == "HAS_EVENT" and e.src in ownable and node[e.dst].props.get("type") == "ASSIGNED":
            events[e.src].append(node[e.dst])
        elif e.type == "ASSIGNED_TO":
            assigned[e.src].add(e.dst)
        elif e.type == "OPENED" and e.dst in ownable and node[e.dst].label == "PR":
            owners[e.dst].add(e.src)
    for bug, evs in events.items():
        last = max(evs, key=lambda n: (n.props.get("ts") or "", *_position(n.props.get("message_id")),
                                       n.props.get("seq") or 0, n.id))
        owners[bug] = set(assigned[last.id])
    already = {i for i in ownable if not owners[i]}
    newly = {i for i in ownable if owners[i] and owners[i] <= {pid}}
    # dependents of each prerequisite, from the original edges
    dependents = defaultdict(set)
    for e in snap.edges:
        for typ, sl, dl, need, end in DEPENDENCY_RULES:
            if (e.type == typ and node[e.src].label == sl and node[e.dst].label == dl
                    and all(e.props.get(k) == v for k, v in need.items())):
                dependents[e.dst if end == "src" else e.src].add(e.src if end == "src" else e.dst)
    skip = already | newly
    reached, frontier = set(), set(newly)
    for _ in range(MAX_HOPS):
        frontier = {d for x in frontier for d in dependents[x]}
        reached |= frontier
    affected = {i for i in reached if i in eligible and i not in skip}

    def trails(at, left, used):  # dependency paths of exactly `left` more edges, no dependency edge twice
        if left == 0:
            return at in eligible and at not in skip
        return any(trails(d, left - 1, used | {(at, d)}) for d in dependents[at] if (at, d) not in used)

    return WhatIf(frozenset(already), frozenset(newly), frozenset(affected),
                  any(trails(g, MAX_HOPS, frozenset()) for g in newly))


# ---------------------------------------------------------------- the runner
def engineering_leads(snap: Snapshot) -> list[str]:
    return sorted(n.props["eid"] for n in snap.nodes if n.label == "Person" and n.props.get("role") == "Engineering Lead"
                  and n.props.get("in_directory") is True)


def work_owners(snap: Snapshot) -> list[str]:
    """Everyone with an OPENED or ASSIGNED_TO edge: a wider set than the leads, for a stronger comparison."""
    node = {n.id: n for n in snap.nodes}
    return sorted({node[e.src].props["eid"] for e in snap.edges if e.type == "OPENED"}
                  | {node[e.dst].props["eid"] for e in snap.edges if e.type == "ASSIGNED_TO"})


def _diff(a: WhatIf, b: WhatIf) -> str:
    parts = [f"{f} differs (graph-only {sorted(set(getattr(a, f)) - set(getattr(b, f)))[:5]}, "
             f"python-only {sorted(set(getattr(b, f)) - set(getattr(a, f)))[:5]})"
             for f in ("already_unowned", "newly_unowned", "affected") if getattr(a, f) != getattr(b, f)]
    if a.bound_hit != b.bound_hit:
        parts.append(f"bound_hit differs (graph {a.bound_hit}, python {b.bound_hit})")
    return "; ".join(parts)


def compare(conn, product: str, people: list[str] | None = None, who: str = "leads") -> list[dict]:
    """Run the graph counterfactual and the Python check for every Engineering Lead of the product (`who='owners'`:
    everyone who opened a PR or was assigned a bug; or the given `people`) from one snapshot, and report counts and
    any difference item by item."""
    from inherit import whatif  # the runner is the only place the two meet

    snap = export_snapshot(conn, product)
    rows = []
    for pid in people or (work_owners(snap) if who == "owners" else engineering_leads(snap)):
        g, p = whatif.if_leaves(conn, product, pid), check(snap, pid)
        rows.append({"product": product, "person": pid, "already_unowned": len(g.already_unowned),
                     "newly_unowned": len(g.newly_unowned), "affected": len(g.affected), "bound_hit": g.bound_hit,
                     "match": g == p, "diff": "" if g == p else _diff(g, p)})
    return rows


# ---------------------------------------------------------------- the Task 13 audit hook
def audit_samples(snap: Snapshot, results: dict[str, WhatIf], *, seed: int, n_unowned: int = 30, n_edges: int = 30,
                  n_paths: int = 20) -> dict:
    """Seeded samples for the raw-record audit, drawn from the results of one snapshot (person eid -> WhatIf):
    `n_unowned` newly unowned items with the assignment (or OPENED) evidence and the later raw messages that could
    show a hand-over; `n_edges` dependency edges drawn from all recorded dependencies (not only a person's) with their citations; `n_paths`
    affected items with a witness path and each edge's citations. A pool smaller than asked is returned whole and
    listed in `shortages`. Later context is every message after the evidence record, up to the snapshot's last
    message, in the item's channel or naming the item (title, alias or link)."""
    node = {n.id: n for n in snap.nodes}
    msgs = sorted(snap.messages, key=lambda m: (_position(m["id"]), m["id"]))
    by_id = {m["id"]: m for m in msgs}
    cutoff = msgs[-1]["id"] if msgs else None
    dep = {}  # (dependent, prerequisite) -> edge
    for e in snap.edges:
        for typ, sl, dl, need, end in DEPENDENCY_RULES:
            if (e.type == typ and node[e.src].label == sl and node[e.dst].label == dl
                    and all(e.props.get(k) == v for k, v in need.items())):
                dep[(e.src, e.dst) if end == "src" else (e.dst, e.src)] = e
    rng, shortages = random.Random(seed), {}

    def draw(name, pool, n):
        pool = sorted(pool, key=str)
        if len(pool) < n:
            shortages[name] = {"requested": n, "available": len(pool)}
        return rng.sample(pool, min(n, len(pool)))

    def mentions(n) -> "re.Pattern | None":
        """Raw-text mentions of a PR: its link, 'pull/N', or 'PR N' / 'PR #N' (not N as a prefix of a longer number)."""
        link, num = n.props.get("link"), n.props.get("number") or (n.props.get("link") or "").rsplit("/", 1)[-1]
        alts = [re.escape(link) + r"(?!\d)"] if link else []
        if num and str(num).isdigit():
            alts += [rf"pull/{num}(?!\d)", rf"\bPR\s*#?\s*{num}(?!\d)"]
        return re.compile("|".join(alts), re.I) if alts else None

    def later(item: str, after_id: str | None) -> list[dict]:
        n = node[item]
        names = [x.casefold() for x in [n.props.get("title"), n.props.get("link"), *(n.props.get("aliases") or [])] if x]
        start = _position(after_id) if after_id else (-1, -1)
        if n.label == "PR":  # PR nodes have no channel: the channels that mention the PR, from its first mention on
            pat = mentions(n)

            def named(m):  # link / number pattern, or the title or an alias (the same matcher as other items)
                t = m["text"] or ""
                return bool(pat and pat.search(t)) or any(x in t.casefold() for x in names)

            hit = [m for m in msgs if named(m)]
            chans = {m["channel_id"] for m in hit}
            first = min((_position(m["id"]) for m in hit), default=None)
            return [m for m in msgs if first is not None and _position(m["id"]) >= first and m["channel_id"] in chans
                    or named(m)]
        return [m for m in msgs if _position(m["id"]) > start
                and ((n.props.get("channel_id") and m["channel_id"] == n.props["channel_id"])
                     or any(x in (m["text"] or "").casefold() for x in names))]

    def slug(link: str) -> str:
        return re.sub(r"[^\w-]+", "_", link.split("://", 1)[-1])

    prs_by_record = {slug(n.props["link"]): n for n in snap.nodes if n.label == "PR" and n.props.get("link")}

    def record_text(rid: str) -> str | None:
        """Raw text behind a record id: a Slack message, or the cited PR record (author, title, summary, state)."""
        if rid in by_id:
            return by_id[rid]["text"]
        if rid in prs_by_record:
            return pr_record(prs_by_record[rid])["text"]
        return None

    def pr_record(n) -> dict:
        authors = sorted(e.src for e in snap.edges if e.type == "OPENED" and e.dst == n.id)
        pp = n.props
        return {"link": pp.get("link"), "title": pp.get("title"), "summary": pp.get("summary"), "state": pp.get("state"),
                "merged": pp.get("merged"), "authors": authors,
                "text": f"{pp.get('title')}: {pp.get('summary')} (state {pp.get('state')}, merged {pp.get('merged')}, "
                        f"opened by {', '.join(authors) or 'nobody'})"}

    def evidence(item: str, who: str) -> dict:
        if node[item].label == "Bug":
            evs = [node[e.dst] for e in snap.edges if e.type == "HAS_EVENT" and e.src == item
                   and node[e.dst].props.get("type") == "ASSIGNED"]
            last = max(evs, key=lambda n: (n.props.get("ts") or "", *_position(n.props.get("message_id")),
                                           n.props.get("seq") or 0, n.id))
            mid = last.props.get("message_id")
            return {"kind": "assigned", "event": last.id, "ts": last.props.get("ts"), "message_id": mid,
                    "records": [r for e in snap.edges if e.type == "ASSIGNED_TO" and e.src == last.id for r in e.records],
                    "text": (by_id.get(mid) or {}).get("text")}
        rec = [r for e in snap.edges if e.type == "OPENED" and e.src == who and e.dst == item for r in e.records]
        return {"kind": "opened", "records": rec, "message_id": None, "text": None, "pr": pr_record(node[item])}

    def witness(item: str, targets: set) -> list[str]:
        queue, prev = [item], {item: None}
        for _ in range(MAX_HOPS):
            nxt = []
            for x in queue:
                for (d, p) in dep:
                    if d == x and p not in prev:
                        prev[p] = x
                        nxt.append(p)
                        if p in targets:
                            out, cur = [], p
                            while cur is not None:
                                out.append(cur)
                                cur = prev[cur]
                            return out[::-1]
            queue = nxt
        return [item]

    pool = [(w, i) for w, r in sorted(results.items()) for i in sorted(r.newly_unowned)]
    unowned = []
    for w, i in draw("newly_unowned", pool, n_unowned):
        ev = evidence(i, person_node_id(w))
        unowned.append({"person": w, "item": i, "label": node[i].label, "title": node[i].props.get("title"),
                        "assignment": ev, "later_context": [
                            {"id": m["id"], "channel_id": m["channel_id"], "ts": m["ts"], "text": m["text"]}
                            for m in later(i, ev["message_id"])]})
    edge_pool = sorted(dep)  # the complete recorded dependency pool, independent of any person's result
    edges = [{"dependent": d, "prerequisite": p, "rule": dep[(d, p)].type, "citations": list(dep[(d, p)].records),
              "texts": [t for t in map(record_text, dep[(d, p)].records) if t is not None]}
             for d, p in draw("dependency_edges", edge_pool, n_edges)]
    path_pool = [(w, i) for w, r in sorted(results.items()) for i in sorted(r.affected)]
    paths = []
    for w, i in draw("affected_paths", path_pool, n_paths):
        path = witness(i, set(results[w].newly_unowned))
        paths.append({"person": w, "affected": i, "path": path,
                      "edges": [{"dependent": a, "prerequisite": b, "rule": dep[(a, b)].type,
                                 "citations": list(dep[(a, b)].records)} for a, b in zip(path, path[1:])]})
    return {"seed": seed, "cutoff": cutoff, "newly_unowned": unowned, "dependency_edges": edges,
            "affected_paths": paths, "shortages": shortages}


def main(argv: list[str] | None = None) -> int:
    from inherit import whatif
    from inherit.data.split import split

    ap = argparse.ArgumentParser()
    ap.add_argument("--products", required=True, help="dev, or a comma-separated list")
    ap.add_argument("--who", choices=("leads", "owners"), default="leads")
    ap.add_argument("--servers", default=",".join(load_settings().falkor_urls))
    args = ap.parse_args(argv)
    if args.products == "dev":
        manifest = json.loads((ROOT / "data" / "bundle" / "manifest.json").read_text())
        products = split(manifest["products"])[0]
    else:
        products = args.products.split(",")
    bad = 0
    print("server product leads already_unowned(sum,max) newly_unowned(sum,max) affected(sum,max) bound_hit mismatches")
    for ver in args.servers.split(","):
        for p in products:
            rows = compare(whatif.whatif_connection(p, ver), p, who=args.who)
            col = lambda k: f"{sum(r[k] for r in rows)},{max((r[k] for r in rows), default=0)}"  # noqa: E731
            miss = [r for r in rows if not r["match"]]
            bad += len(miss)
            print(f"{ver} {p} {len(rows)} {col('already_unowned')} {col('newly_unowned')} {col('affected')} "
                  f"{sum(r['bound_hit'] for r in rows)} {len(miss)}", flush=True)
            for r in rows:
                print(f"    {r['person']} {r['already_unowned']}/{r['newly_unowned']}/{r['affected']} "
                      f"bound_hit={r['bound_hit']} {'OK' if r['match'] else 'MISMATCH ' + r['diff']}")
    print("mismatches:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
