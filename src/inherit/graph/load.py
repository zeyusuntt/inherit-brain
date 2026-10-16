"""Load facts into one FalkorDB graph per product (`herb:<product>`).

    uv run python -m inherit.graph.load --products dev|all|P1,P2 [--servers 4.22,6.0]

Loading is a build step, never incremental: the graph is dropped and rebuilt. Order: validate and embed (nothing
is touched if that fails), drop users and graph, then as a temporary `build_<p>` user: exact-match indexes,
constraints, nodes, edges, full-text and vector indexes. A graph property `id` is the Task 3 node ID; a record's
own `id` is stored as `source_id`. FalkorDB stores scalars and arrays of scalars only, so dicts and lists of
non-scalars are JSON strings under `<name>_json`, and None is dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from falkordb import FalkorDB

from inherit import embed as embed_mod
from inherit.config import ROOT, load_settings
from inherit.facts import Facts
from inherit.graph import acl, schema

# Frozen order: Facts.merge keeps the first value on a conflict. names.jsonl is a report, not Facts.
LAYER_FILES = ("structural.jsonl", "names_facts.jsonl", "rules.jsonl", "bugs.jsonl", "semantic.jsonl")
ROOTED = ("Bug", "Document", "Meeting", "Feature", "PR", "Competitor")
CHUNK = 1000
EMBED_CHUNK = 100
EDGE_RESERVED = ("src", "extractor", "key")


# ---------------------------------------------------------------- property mapping
def _scalar(v) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _dumps(v) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def map_props(props: dict) -> dict:
    out: dict = {}
    for k, v in props.items():
        if v is None:
            continue
        if _scalar(v) or (isinstance(v, list) and all(_scalar(x) for x in v)):
            name, val = k, v
        else:
            name, val = f"{k}_json", _dumps(v)
        if name in out or (name == k and f"{k}_json" in props):
            raise ValueError(f"property name collision on {k!r}")
        out[name] = val
    return out


def unmap_props(mapped: dict) -> dict:
    return {(k[:-5] if k.endswith("_json") else k): (json.loads(v) if k.endswith("_json") else v)
            for k, v in mapped.items()}


def node_props(node) -> dict:
    props = dict(node.props)
    out = map_props({k: v for k, v in props.items() if k != "id"})
    if "id" in props and props["id"] is not None:
        out["source_id"] = props["id"]
    out["id"] = node.id
    return out


def edge_props(edge) -> dict:
    clash = [k for k in EDGE_RESERVED if k in edge.props]
    if clash:
        raise ValueError(f"edge {edge.type} uses reserved property names {clash}")
    return {**map_props(edge.props), "src": list(edge.src), "extractor": edge.extractor, "key": edge.key}


# ---------------------------------------------------------------- layers
def load_layers(product: str, facts_dir: Path | None = None) -> Facts:
    d = Path(facts_dir or ROOT / "data" / "facts") / product
    facts = Facts()
    for name in LAYER_FILES:
        if (d / name).exists():
            facts = facts.merge(Facts.from_jsonl(d / name))
    return facts


# ---------------------------------------------------------------- stats
@dataclass
class LoadStats:
    product: str
    nodes: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    embedded: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (f"{self.product}: nodes {dict(sorted(self.nodes.items()))} edges {dict(sorted(self.edges.items()))} "
                f"embedded {self.embedded} seconds {({k: round(v, 2) for k, v in self.seconds.items()})}")


# ---------------------------------------------------------------- validation
def check_root(product: str, facts: Facts) -> None:
    roots = [n.id for n in facts.nodes if n.label == "Product"]
    if roots != [f"Product:{product}"]:
        raise ValueError(f"need exactly one root Product:{product}, found {roots}")
    has = {e.dst_id for e in facts.edges if e.type == "HAS" and e.src_id == roots[0]}
    missing = sorted(n.id for n in facts.nodes if n.label in ROOTED and n.id not in has)
    if missing:
        raise ValueError(f"{len(missing)} nodes lack a HAS edge from {roots[0]}: {missing[:5]}")


def embed_targets(facts: Facts) -> list[tuple[str, str, str]]:
    """(node id, label, text) for every Message with text and Document with content."""
    out = []
    for n in facts.nodes:
        prop = {"Message": "text", "Document": "content"}.get(n.label)
        text = n.props.get(prop) if prop else None
        if isinstance(text, str) and text.strip():
            out.append((n.id, n.label, text))
    return out


def check_vectors(vecs, n: int) -> None:
    if len(vecs) != n:
        raise ValueError(f"expected {n} embeddings, got {len(vecs)}")
    for v in vecs:
        if len(v) != schema.VECTOR_DIM:
            raise ValueError(f"embedding must have {schema.VECTOR_DIM} components, got {len(v)}")
        if not all(math.isfinite(x) for x in v):
            raise ValueError("embedding has non-finite values")


# ---------------------------------------------------------------- writing
def _chunks(rows: list, n: int):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def _write_nodes(g, facts: Facts, vectors: dict) -> None:
    by_label: dict[str, list] = defaultdict(list)
    for n in facts.nodes:
        by_label[n.label].append(n)
    for label, nodes in by_label.items():
        plain = [{"props": node_props(n)} for n in nodes if n.id not in vectors]
        vec = [{"props": node_props(n), "emb": vectors[n.id]} for n in nodes if n.id in vectors]
        # A mandatory property must be present at CREATE time; the full map follows in SET.
        mand = ", ".join(f"{pr}: r.props.{pr}" for k, lb, pr in schema.CONSTRAINTS if k == "MANDATORY" and lb == label)
        create = f"(n:{label} {{{mand}}})" if mand else f"(n:{label})"
        for rows, size, extra in ((plain, CHUNK, ""), (vec, EMBED_CHUNK, ", n.embedding = vecf32(r.emb)")):
            for part in _chunks(rows, size):
                res = g.query(f"UNWIND $rows AS r CREATE {create} SET n = r.props{extra}", {"rows": part})
                if res.nodes_created != len(part):
                    raise RuntimeError(f"{label}: created {res.nodes_created} of {len(part)} nodes")


def _write_edges(g, facts: Facts) -> None:
    label = {n.id: n.label for n in facts.nodes}
    groups: dict[tuple, list] = defaultdict(list)
    for e in facts.edges:
        groups[(e.type, label[e.src_id], label[e.dst_id])].append({"s": e.src_id, "d": e.dst_id,
                                                                    "props": edge_props(e)})
    for (typ, sl, dl), rows in groups.items():
        for part in _chunks(rows, CHUNK):
            res = g.query(f"UNWIND $rows AS r MATCH (a:{sl} {{id: r.s}}), (b:{dl} {{id: r.d}}) "
                          f"CREATE (a)-[e:{typ}]->(b) SET e = r.props", {"rows": part})
            if res.relationships_created != len(part):
                raise RuntimeError(f"{typ} {sl}->{dl}: created {res.relationships_created} of {len(part)} edges")


def _build_db(admin, user: str, password: str) -> FalkorDB:
    kw = admin.connection.connection_pool.connection_kwargs
    return FalkorDB(host=kw["host"], port=kw["port"], username=user, password=password)


def load_product(db, product: str, facts: Facts, *, embed=None) -> LoadStats:
    """Replace graph herb:<product> with `facts`. `embed(texts) -> vectors` defaults to the metered Qwen3 helper."""
    stats = LoadStats(product)
    t0 = time.monotonic()

    def lap(phase: str) -> None:
        nonlocal t0
        now = time.monotonic()
        stats.seconds[phase] = round(now - t0, 3)
        t0 = now

    check_root(product, facts)
    lap("validate")
    targets = embed_targets(facts)
    vecs = (embed or embed_mod.encode_corpus)([t for _, _, t in targets])
    check_vectors(vecs, len(targets))
    vectors = {i: v for (i, _, _), v in zip(targets, vecs)}
    stats.embedded = dict(Counter(label for _, label, _ in targets))
    lap("embed")

    name = f"herb:{product}"
    acl.drop_product_users(db, product)
    if name in db.list_graphs():
        db.select_graph(name).delete()
    build_user, pw = f"build_{product}", secrets.token_urlsafe(24)
    acl.set_user(db, build_user, pw, acl.build_rules(product))
    lap("reset")
    try:
        g = _build_db(db, build_user, pw).select_graph(name)
        labels = {n.label for n in facts.nodes}
        schema.create_exact_indexes(g, labels)
        schema.create_constraints(g)
        lap("indexes")
        _write_nodes(g, facts, vectors)
        lap("nodes")
        _write_edges(g, facts)
        lap("edges")
        schema.create_search_indexes(g, labels, [lb for lb in schema.VECTOR_LABELS if lb in stats.embedded])
        lap("search_indexes")
        unrooted = g.query("MATCH (n) WHERE any(l IN labels(n) WHERE l IN $rooted) "
                           "AND NOT ()-[:HAS]->(n) RETURN count(n)", {"rooted": list(ROOTED)}).result_set[0][0]
        if unrooted:
            raise RuntimeError(f"{unrooted} nodes lack a HAS edge in the loaded graph")
        stats.nodes = {k: v for k, v in g.query("MATCH (n) RETURN labels(n)[0], count(n)").result_set}
        stats.edges = {k: v for k, v in g.query("MATCH ()-[e]->() RETURN type(e), count(e)").result_set}
        lap("verify")
    except BaseException:
        if name in db.list_graphs():
            db.select_graph(name).delete()
        raise
    finally:
        acl.del_user(db, build_user)
    return stats


def main(argv: list[str] | None = None) -> int:
    from inherit.extract.run import select_products

    ap = argparse.ArgumentParser()
    ap.add_argument("--products", required=True)
    ap.add_argument("--servers", default=",".join(load_settings().falkor_urls))
    ap.add_argument("--facts", default=str(ROOT / "data" / "facts"))
    args = ap.parse_args(argv)
    available = sorted(p.name for p in Path(args.facts).iterdir() if p.is_dir())
    products = select_products(args.products, available)
    for ver in args.servers.split(","):
        db = FalkorDB.from_url(load_settings().falkor_urls[ver])
        for p in products:
            print(f"[{ver}] {load_product(db, p, load_layers(p, Path(args.facts)))}", flush=True)
        acl.setup_acl(db, products, label=ver)
        print(f"[{ver}] ACL users provisioned for {len(products)} products (passwords in data/acl.json)")
    st = embed_mod.STATE
    print(f"embedding path {st['path']}, calls {st['calls']}, tokens {st['prompt_tokens']}, "
          f"cost ${st['cost_usd']:.4f}, cache hits {st['cache_hits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
