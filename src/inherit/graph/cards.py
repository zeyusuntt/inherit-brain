"""Entity cards: the one projection of a graph node that A0f retrieves over and A4 and A4c read.

A card is the node's own allowlisted properties plus, for every incident edge, the neighbour's label, id and
display name and the edge's `src`. It never holds a neighbour's other properties, a count or an aggregate. The
definition is frozen and published (docs/cards.md); change anything here and CARD_VERSION must change too.

    uv run python -m inherit.graph.cards --doc [--product P ...]   # regenerate docs/cards.md (measured on dev)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

CARD_VERSION = "v2"
SRC_CAP = 3  # record ids listed per neighbour entry; a longer list ends with "..." and no count
ELLIPSIS = "..."
EXCLUDED_EDGES = frozenset({"COLLABORATED"})  # derived analytic projection, not a recorded relation
PRODUCT_LABELS = ("Bug", "Document", "Meeting", "Feature", "PR", "Competitor")  # spec 6.1: all a Product card lists

# Logical field allowlist per label, frozen with CARD_VERSION. Names are logical: `_json` storage is decoded first.
# Nothing here is a copied relationship answer (resolver, fix author, assignee, counts) or aggregate: those stay
# traversals. Bug helper lists (companies, customer_ids, current_assignees, event_types) are left out for that reason.
ALLOWLIST: dict[str, tuple[str, ...]] = {
    "Product": ("name", "aliases"),
    "Person": ("eid", "name", "role", "in_directory"),
    "Customer": ("id", "company", "name", "role"),
    "Channel": ("name", "bug_owner"),
    "Message": ("ts", "kind", "user_id", "text", "release_scope"),
    "Document": ("type", "date", "link", "content", "feedback", "feedback_items", "release_scope",
                 "source_product_name"),
    "Meeting": ("document_type", "date", "release_scope"),
    "PR": ("title", "number", "summary", "state", "merged", "mergeable", "created_at", "repo", "repo_scope",
           "approval", "open_unmerged", "pending_action", "pending_reason"),
    "Url": ("link", "descriptions", "kind", "release_scope", "source_product_name"),
    "Bug": ("title", "aliases", "component", "impact_text", "first_reported", "status", "unresolved", "active",
            "reopened", "no_fix_needed", "resolved"),
    "BugEvent": ("ts", "type", "seq"),
    "Feature": ("title", "topic", "aliases", "details", "implemented", "reverted", "held", "delayed_then_added",
                "discussed_not_implemented"),
    "FeatureEvent": ("kind", "reason", "reason_text", "date", "record_id"),
    "Competitor": ("name", "aliases"),
    "ExternalBug": ("title", "link"),
    "Observation": ("kind", "text", "release_scope", "record_id"),
}

# Stored fields a display name may need (Task 6 storage: record id in `source_id`, dicts under `<name>_json`).
NAME_FIELDS = ("name", "company", "title", "type", "document_type", "source_id", "link", "eid")


@dataclass(frozen=True)
class Policy:
    """The frozen card is `Policy()`. The flags exist so the measured savings in docs/cards.md come from the same code."""
    messages: bool = False  # True: non-Message cards list their Message neighbours (spec 6.1 literal)
    collaborated: bool = False  # True: COLLABORATED edges are listed
    src_cap: int | None = SRC_CAP  # None: the full src list
    extra_fields: frozenset = frozenset()  # {(label, stored field)} added to the allowlist


FROZEN = Policy()


def display_name(label: str, p: dict, node_id: str) -> str:
    """Stable identifier or domain name; never a content preview. `p` holds the stored NAME_FIELDS values."""
    g = p.get
    if label == "Person":
        v = g("name") or g("eid")
    elif label == "Customer":
        v = g("company")
    elif label == "Document":
        v = f"{g('type')} ({g('source_id')})" if g("type") and g("source_id") else None
    elif label == "Meeting":
        v = f"{g('document_type')} meeting ({g('source_id')})" if g("document_type") and g("source_id") else None
    elif label in ("PR", "Bug", "Feature", "ExternalBug"):
        v = g("title")
    elif label in ("Message", "Observation", "BugEvent", "FeatureEvent"):
        v = g("source_id")
    elif label == "Url":
        v = g("link")
    else:  # Product, Competitor, Channel
        v = g("name")
    return str(v) if v not in (None, "") else node_id


def _label(node_id: str) -> str:
    label = node_id.split(":", 1)[0]
    if label not in ALLOWLIST:
        raise ValueError(f"unknown node label in id {node_id!r}")
    return label


def fields_for(label: str, policy: Policy = FROZEN) -> tuple[str, ...]:
    return ALLOWLIST[label] + tuple(f for lb, f in sorted(policy.extra_fields) if lb == label)


def _cols(var: str, fields, prefix: str) -> str:
    return ", ".join(f"{var}.{f} AS {prefix}{i}" for i, f in enumerate(fields))


def _name_cols(var: str) -> str:
    return _cols(var, NAME_FIELDS, "nm")


def _name_row(row) -> dict:
    return dict(zip(NAME_FIELDS, row))


def _props(graph, label: str, node_id: str, policy: Policy):
    fields = fields_for(label, policy)
    cols = _cols("n", [x for f in fields for x in (f, f + "_json")], "p")
    q = f"MATCH (n:{label} {{id: $id}}) RETURN {_name_cols('n')}, {cols}"
    rows = graph.ro_query(q, {"id": node_id}).result_set
    if not rows:
        raise LookupError(f"no node {node_id}")
    row = rows[0]
    names = _name_row(row[:len(NAME_FIELDS)])
    rest = row[len(NAME_FIELDS):]
    props = {}
    for i, f in enumerate(fields):
        plain, js = rest[2 * i], rest[2 * i + 1]
        v = plain if plain is not None else (json.loads(js) if js is not None else None)
        if v is not None:
            props[f] = v
    return names, props


def _entries(graph, label: str, node_id: str, policy: Policy) -> list[dict]:
    skip_edges = [] if policy.collaborated else sorted(EXCLUDED_EDGES)
    skip_labels = [] if (policy.messages or label == "Message") else ["Message"]
    merged: dict[tuple, dict] = {}
    for direction, pat in (("out", "-[e]->(m)"), ("in", "<-[e]-(m)")):
        q = (f"MATCH (n:{label} {{id: $id}}){pat} WHERE NOT type(e) IN $skip_edges "
             f"AND NOT labels(m)[0] IN $skip_labels RETURN type(e), labels(m)[0], m.id, e.src, {_name_cols('m')}")
        for row in graph.ro_query(q, {"id": node_id, "skip_edges": skip_edges, "skip_labels": skip_labels}).result_set:
            typ, mlabel, mid, src = row[:4]
            if label == "Product" and not (direction == "out" and typ == "HAS" and mlabel in PRODUCT_LABELS):
                continue
            ent = merged.setdefault((typ, direction, mlabel, mid), {
                "edge": typ, "dir": direction, "label": mlabel, "id": mid,
                "name": display_name(mlabel, _name_row(row[4:]), mid), "src": set()})
            ent["src"].update(src or [])
    out = []
    for key in sorted(merged):
        ent = merged[key]
        src = sorted(ent["src"])
        if policy.src_cap is not None and len(src) > policy.src_cap:
            src = src[:policy.src_cap] + [ELLIPSIS]
        if label == "Product":
            del ent["src"]
        else:
            ent["src"] = src
        out.append(ent)
    return out


def card(graph, node_id: str, *, policy: Policy = FROZEN) -> dict:
    """The card of one node. `graph` is a falkordb Graph (read through GRAPH.RO_QUERY, so an agent user works)."""
    label = _label(node_id)
    names, props = _props(graph, label, node_id, policy)
    return {"id": node_id, "label": label, "name": display_name(label, names, node_id), "props": props,
            "neighbours": _entries(graph, label, node_id, policy)}


def node_ids(graph) -> list[str]:
    rows = graph.ro_query("MATCH (n) RETURN labels(n)[0], n.id").result_set
    for label, _ in rows:
        if label not in ALLOWLIST:
            raise ValueError(f"graph holds a node with unknown label {label!r}")
    return sorted((i for _, i in rows), key=lambda i: (i.split(":", 1)[0], i))


def cards_for_product(graph, *, policy: Policy = FROZEN) -> Iterator[dict]:
    """Every node's card (messages included), ordered by (label, id): the A0f corpus."""
    for node_id in node_ids(graph):
        yield card(graph, node_id, policy=policy)


# ---------------------------------------------------------------- documentation (docs/cards.md is generated from here)
NAME_RULES = {
    "Product": "`name`", "Person": "`name`, else the eid", "Customer": "`company`",
    "Channel": "`name`", "Message": "the message id", "Document": "`type` + \" (\" + document id + \")\"",
    "Meeting": "`document_type` + \" meeting (\" + meeting id + \")\"", "PR": "`title`", "Url": "the link",
    "Bug": "`title`", "BugEvent": "the event id", "Feature": "`title`", "FeatureEvent": "the event id",
    "Competitor": "`name`", "ExternalBug": "`title`", "Observation": "the observation id",
}
# Fields measured as extra cost if they were added back (they repeat an edge, or are internal bookkeeping).
MEASURED_FIELDS = (("Meeting", "participants"), ("Document", "author"), ("Message", "channel_id"),
                   ("PR", "reverts_number"))
FIELDS_BEGIN, FIELDS_END = "<!-- fields:begin -->", "<!-- fields:end -->"
MEASURES_BEGIN, MEASURES_END = "<!-- measures:begin -->", "<!-- measures:end -->"


def render_fields() -> str:
    lines = [FIELDS_BEGIN, ""]
    for label, fields in ALLOWLIST.items():
        lines += [f"### {label}", "",
                  f"- display name: {NAME_RULES[label]}",
                  f"- props: {', '.join(f'`{f}`' for f in fields)}"]
        if label == "Product":
            lines.append("- neighbours: only `HAS` out-edges to " + ", ".join(PRODUCT_LABELS)
                         + " (`edge`, `dir`, `label`, `id`, `name`; no `src`, no messages, no counts)")
        elif label == "Message":
            lines.append("- neighbours: all incident edges, including other Messages' edges to it (the poster, "
                         "channel, mentioned people, PRs, features, documents)")
        else:
            lines.append("- neighbours: every incident edge except `COLLABORATED`, and no Message neighbours")
        lines.append("")
    lines.append(FIELDS_END)
    return "\n".join(lines)


def _pctl(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def tokens_of(obj) -> int:
    return -(-len(json.dumps(obj, ensure_ascii=False, separators=(",", ":"))) // 4)


def measure(graphs: dict[str, object]) -> dict:
    """Card sizes (tokens = characters / 4) per label under the frozen policy, and what each exclusion saves, over
    the given {product: graph}. Every number comes from building the cards with and without the exclusion."""
    variants = {
        "messages": Policy(messages=True), "collaborated": Policy(collaborated=True), "src_cap": Policy(src_cap=None),
        "fields": Policy(extra_fields=frozenset(MEASURED_FIELDS)),
    }
    sizes: dict[str, list[int]] = {}
    saved: dict[str, dict[str, int]] = {k: {} for k in variants}
    n_cards = 0
    for graph in graphs.values():
        for nid in node_ids(graph):
            label = nid.split(":", 1)[0]
            base = tokens_of(card(graph, nid))
            sizes.setdefault(label, []).append(base)
            n_cards += 1
            for name, pol in variants.items():
                saved[name][label] = saved[name].get(label, 0) + tokens_of(card(graph, nid, policy=pol)) - base
    return {"products": sorted(graphs), "cards": n_cards,
            "sizes": {lb: {"n": len(v), "median": _pctl(v, 0.5), "p95": _pctl(v, 0.95), "max": max(v),
                           "total": sum(v)} for lb, v in sorted(sizes.items())},
            "saved": saved}


def render_measures(m: dict | None) -> str:
    if not m:
        return f"{MEASURES_BEGIN}\nNot measured: no dev graphs were loaded when this file was generated.\n{MEASURES_END}"
    lines = [MEASURES_BEGIN, "",
             f"Measured on {m['cards']} cards of {len(m['products'])} dev products ({', '.join(m['products'])}). "
             "Tokens are serialised characters / 4.", "",
             "Card size under the frozen definition:", "",
             "| label | cards | median | p95 | max | total |", "|---|---|---|---|---|---|"]
    for lb, s in m["sizes"].items():
        lines.append(f"| {lb} | {s['n']} | {s['median']} | {s['p95']} | {s['max']} | {s['total']} |")
    total = sum(s["total"] for s in m["sizes"].values())
    labels = list(m["sizes"])
    lines += ["", f"Whole corpus under the frozen definition: {total} tokens.", "",
              "Tokens each exclusion removes (the cost of the card if it were included), by label of the card "
              "that carries it:", "",
              "| exclusion | " + " | ".join(labels) + " | all |", "|---|" + "---|" * (len(labels) + 1)]
    names = {"messages": "Message neighbours on non-Message cards", "collaborated": "COLLABORATED entries",
             "src_cap": f"src lists beyond {SRC_CAP} ids",
             "fields": "fields that repeat an edge (" + ", ".join(f"{a}.{b}" for a, b in MEASURED_FIELDS) + ")"}
    for k, per in m["saved"].items():
        lines.append(f"| {names[k]} | " + " | ".join(str(per.get(lb, 0)) for lb in labels)
                     + f" | {sum(per.values())} |")
    lines.append(MEASURES_END)
    return "\n".join(lines)


def render_doc(m: dict | None = None) -> str:
    from inherit.tools import depth1  # constants of the tool live with the tool
    return f"""# Entity cards

Card definition version: **{CARD_VERSION}**. This file is generated by `uv run python -m inherit.graph.cards --doc`
from `src/inherit/graph/cards.py` and `src/inherit/tools/depth1.py`. The definition is frozen and published with the
pre-registration; any change to it changes the version.

## What a card is

A card is a JSON object with a fixed key order: `id`, `label`, `name`, `props`, `neighbours`.

- `props`: the node's own properties, restricted to the per-label field list below, with storage encoding undone
  (`_json` properties are decoded) and null values left out. Nothing is computed for the card: no count, no degree,
  no aggregate, and no relationship answer copied onto a node (resolver, fix author, assignee stay traversals).
- `neighbours`: one entry per distinct (edge type, direction, neighbour) of the node's incident edges:
  `{{edge, dir: "out"|"in", label, id, name, src}}`. `name` is the neighbour's display name. A neighbour's other
  properties are never in the card. `src` is the edge's record ids in sorted order, at most {SRC_CAP}, followed by
  the literal `"..."` when more exist (no count). Parallel edges of one type to one neighbour become one entry whose
  `src` is the union. Edge properties (review state and comment, suggestion text, contribution kind, weights,
  reasons) are not on cards. Entries are sorted by (edge, dir, label, id).
- Display names are domain names or stable identifiers, never content previews (a message or observation is named by
  its id, a URL by its link).
- Document `content` and `feedback` are complete: no card text is truncated. Cost is bounded by the response budget
  below, not by cutting text.

## Fields per label

{render_fields()}

## Registered deviations from spec 6.1

1. **Messages are left out of every non-Message card.** Spec 6.1 excludes messages only from the Product card. Here a
   Person card would list up to 157 posts and a Channel card up to 290 messages, so every card except a Message's
   own omits Message neighbours. A Message card lists its neighbours normally, and `neighbors(root, POSTED, out)`,
   `neighbors(channel, IN_CHANNEL, in)` and the like return Message cards when asked explicitly. The same projection
   is used for A0f, A4 and A4c. The measured saving is in the table below.
2. **`COLLABORATED` is on no card.** It is a derived analytic projection (used by the bridges tool), not a recorded
   relation, and its provenance made one meeting's attendee cards about 264,000 tokens. The edge type is also refused
   by the atomic tool.
3. **`src` is capped at {SRC_CAP} record ids** per neighbour entry (spec: "each edge's src"), then `"..."`.
4. **Fields that repeat an edge or are internal bookkeeping are not on cards** (for example `Meeting.participants`
   repeats `ATTENDED`, `Document.author` repeats `AUTHORED`): see the table below.

## Measured costs

{render_measures(m)}

## Response budget (the atomic tool)

- At most {depth1.CALL_CAP} calls per question, shared by `resolve`, `get_card` and `neighbors`; a refused call counts.
- At most {depth1.ROW_CAP} rows and {depth1.RESPONSE_BUDGET_TOKENS:,} tokens (characters / 4) of serialised rows per
  response. Rows beyond that are not returned; the response says `truncated: true` and carries an opaque `cursor`.
  Passing the cursor returns the next page and counts as a call. A single card (or group value) larger than the
  budget is refused with an explicit error, never returned; no card in the dev data comes close.
- Response envelope: `{{rows, truncated, cursor}}`, plus `count` for `agg: "count"` or `groups` for `agg: "group"`.
  Neighbours are distinct nodes in id order. `where` filters on card fields only (`id`, `label`, `name` and the
  props above), by equality or membership in a list; a missing field never matches. `count` and `group` run over all
  matches before paging; groups are then paged under the same caps. No further edge is followed.

## Embeddings (Task 6)

Message text and Document content are embedded whole, one vector per node and no chunking, and the vector lives on the
graph node. Cards never contain it, and A0f does not retrieve with it: Task 11 freezes A0f as the paper's retriever
over the serialised cards.
"""


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    from inherit.config import ROOT
    from inherit.tools.depth1 import agent_graph

    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", action="store_true", help="write docs/cards.md")
    ap.add_argument("--products", default="ActionGenie,AnomalyForce,ContentForce,KnowledgeForce,LeadForce,TrendForce")
    ap.add_argument("--server", default="6.0")
    ap.add_argument("--measure-json", help="also write the raw measurements here")
    args = ap.parse_args(argv)
    m = None
    try:
        m = measure({p: agent_graph(p, args.server) for p in args.products.split(",")})
    except Exception as e:  # noqa: BLE001
        print(f"not measured: {type(e).__name__}: {e}")
    if args.measure_json and m:
        Path(args.measure_json).write_text(json.dumps(m, indent=1))
    if args.doc:
        (ROOT / "docs" / "cards.md").write_text(render_doc(m), encoding="utf-8")
        print("wrote docs/cards.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
