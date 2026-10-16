"""The freeze: extract, load, verify and hash all 30 products before any held-out question is run.

    uv run python -m inherit.freeze extract --products held|all|P1,P2 [--layers structural,names,...] [--estimate]
    uv run python -m inherit.freeze load [--products all] [--servers 6.0,4.22]
    uv run python -m inherit.freeze verify-whatif [--products all] [--servers 6.0,4.22]
    uv run python -m inherit.freeze manifest
    uv run python -m inherit.freeze check

`manifest` writes results/freeze/manifest.json (task-13-decisions R3): exactly the 30 bundle products x the five
layer files (a file may be empty, may not be absent), the split, bundle, facts and lock hashes, the effective config
(settings, models, prices, caps, prompt and protocol versions, and the arms and product subsets of
results/freeze/arms.json), the version sidecars, extraction reports and the access logs of the freeze extraction runs,
the card corpus and embedding cache hashes, the retrieval indexes the arms read (fingerprint recomputed, files hashed),
A2 ontology, build and export identities for its sampled products, and a content identity of every loaded graph on
every server, verified against the facts. Every one of these is required; nothing is optional. It refuses a dirty tree. `check` recomputes all of it and prints
every difference. No subcommand opens a question file.

R1: a layer's code or prompt version is recorded next to each product's facts (data/facts/<p>/freeze_versions.json).
`extract` refuses to run a layer for a subset of products while any other product has facts made under another
version of it: a version change means re-running that layer for all 30.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from inherit import embed as embed_mod
from inherit.config import EMBED_MODEL, MODEL, PRICES_PICO, PROJECT_CAP_USD, ROOT, load_settings
from inherit.data.bundle import bundle_hash
from inherit.data.split import split, split_hash
from inherit.graph import acl, cards
from inherit.graph import load as loadmod

LAYERS = ("structural", "names", "rules", "bugs", "semantic")
EXPECTED_PRODUCTS = 30
SUMMARY_FILES = {"rules": "rules_report.json", "bugs": "bugs_report.json", "semantic": "semantic_report.json"}
VERSIONS_FILE = "freeze_versions.json"


class FreezeError(RuntimeError):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def canon_hash(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))


# ---------------------------------------------------------------- environment
def git_commit(root: Path) -> str:
    """HEAD of a clean tree (raises otherwise)."""
    from eval.receipt import require_clean_tree

    require_clean_tree(root)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, check=True, text=True).stdout.strip()


def git_drift(root: Path, commit: str) -> list[str]:
    """Files whose code differs between `commit` and the working tree, and any dirty or untracked code."""
    from eval.receipt import require_clean_tree

    try:
        require_clean_tree(root)
        out = subprocess.run(["git", "diff", "--name-only", commit, "--", "src", "scripts", "tests", "pyproject.toml",
                              "uv.lock"], cwd=root, capture_output=True, check=True, text=True).stdout
    except (RuntimeError, OSError, subprocess.CalledProcessError) as e:
        return [f"cannot compare the code with {commit}: {e}"]
    return [f"code changed since {commit[:12]}: {f}" for f in out.splitlines() if f.strip()]


def _default_graph(product: str, server: str):
    from inherit.tools.depth1 import agent_graph

    return agent_graph(product, server)


def _default_conn(product: str, server: str):
    from inherit import whatif

    return whatif.whatif_connection(product, server)


def _default_db(server: str):
    from falkordb import FalkorDB

    return FalkorDB.from_url(load_settings().falkor_urls[server])


@dataclass
class Env:
    root: Path = ROOT
    servers: tuple = ("6.0", "4.22")  # the first is primary
    expected: int = EXPECTED_PRODUCTS
    graph: Callable = _default_graph  # (product, server) -> read-only graph
    conn: Callable = _default_conn  # (product, server) -> whatif connection
    db: Callable = _default_db  # (server) -> admin client
    split: Callable = split  # (products) -> (dev, held)
    commit: Callable = git_commit  # (root) -> HEAD of a clean tree
    drift: Callable = git_drift  # (root, commit) -> differences
    embed: Callable | None = None  # texts -> vectors; None: the metered cached helper
    a2_graph: Callable = lambda p: _default_a2_graph(p)  # (product) -> the A2 graph
    extra: dict = field(default_factory=dict)

    @property
    def bundle(self) -> Path:
        return self.root / "data" / "bundle"

    @property
    def facts(self) -> Path:
        return self.root / "data" / "facts"

    @property
    def out(self) -> Path:
        return self.root / "results" / "freeze"


def inventory(env: Env) -> list[str]:
    return sorted(json.loads((env.bundle / "manifest.json").read_text())["products"])


def select(spec: str, inv: list[str], splitter: Callable = split) -> list[str]:
    if spec == "all":
        return sorted(inv)
    if spec in ("held", "dev"):
        dev, held = splitter(inv)
        return held if spec == "held" else dev
    names = spec.split(",")
    unknown = [n for n in names if n not in inv]
    if unknown:
        raise SystemExit(f"unknown products: {unknown}")
    return names


# ---------------------------------------------------------------- layer versions (R1)
SOURCE_ROOTS = ("extract", "data")  # directories under src/inherit hashed whole
SOURCE_FILES = ("facts.py", "ontology.py")


def extraction_source_digest() -> str:
    """One digest of every source file the extraction outputs depend on: src/inherit/extract/**, src/inherit/data/**,
    facts.py and ontology.py. Any change to any of them is a code change of every layer."""
    src = Path(__file__).resolve().parent
    files = sorted(f for d in SOURCE_ROOTS for f in (src / d).rglob("*.py")) + [src / n for n in SOURCE_FILES]
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(src).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def declared_versions(layer: str) -> dict:
    from inherit.extract import bug_rules, names, prompts, rules, semantic, semantic_prompts

    return {
        "structural": {},
        "names": {"role_cues": names.ROLE_CUES_VERSION},
        "rules": dict(rules.VERSIONS),
        "bugs": {"prompt": prompts.BUGS_PROMPT_VERSION, "rules": bug_rules.RULES_VERSION},
        "semantic": {"prompt": semantic_prompts.SEMANTIC_PROMPT_VERSION, "fate": semantic.FATE_VERSION,
                     "pending": semantic.PENDING_VERSION},
    }[layer]


def layer_versions(layer: str) -> dict:
    """Declared prompt and rule versions plus the shared extraction-source digest."""
    return {**declared_versions(layer), "code": extraction_source_digest()}


def recorded_versions(env: Env, layer: str, product: str) -> dict | None:
    """What the product's facts for `layer` were made under: the sidecar, else the versions its report states."""
    d = env.facts / product
    side = d / VERSIONS_FILE
    if side.exists():
        rec = json.loads(side.read_text(encoding="utf-8")).get(layer)
        if rec is not None:
            return rec
    try:
        if layer == "bugs":
            r = json.loads((d / "bugs_report.json").read_text(encoding="utf-8"))
            return {"prompt": r["prompt_version"], "rules": r["rules_version"]}
        if layer == "semantic":
            r = json.loads((d / "semantic_report.json").read_text(encoding="utf-8"))
            return {"prompt": r["prompt_version"], "fate": r["fate_version"]}
        if layer == "rules":
            return dict(json.loads((d / "rules_report.json").read_text(encoding="utf-8"))["versions"])
        if layer == "names":
            rows = (d / "names.jsonl").read_text(encoding="utf-8").splitlines()
            return {"role_cues": json.loads(rows[0])["cue_version"]} if rows else None
    except (OSError, KeyError, ValueError):
        return None
    return None


def stale_products(env: Env, layer: str, running: list[str], inv: list[str]) -> list[str]:
    """Products outside this run whose `layer` facts were made under another version than the current one."""
    cur = layer_versions(layer)
    out = []
    for p in inv:
        if p in running:
            continue
        rec = recorded_versions(env, layer, p)
        if rec is not None and any(cur.get(k) != v for k, v in rec.items()):
            out.append(p)
    return out


def _stamp(env: Env, layer: str, product: str) -> None:
    side = env.facts / product / VERSIONS_FILE
    data = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
    data[layer] = layer_versions(layer)
    side.write_text(json.dumps(data, sort_keys=True, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- extract
def extract(env: Env, products: list[str], layers=LAYERS, *, estimate: bool = False, runner: Callable | None = None,
            records_for: Callable | None = None, run_id: str = "norun") -> dict:
    """Run the layers in order for `products`. A fatal error (budget, proxy, timeout) propagates and stops the run;
    per-unit failures are counted per product and layer. Returns {product: {layer: {failures, complete}}}."""
    from inherit.extract import run as run_mod

    inv = inventory(env)
    runner = runner or run_mod.run_layer
    if not estimate:
        for layer in layers:
            bad = stale_products(env, layer, products, inv)
            if bad:
                raise FreezeError(f"layer {layer!r} changed version since {len(bad)} other products were extracted "
                                  f"({bad[:5]}...): re-run it for all {len(inv)} products, not a subset")
    if records_for is None:
        from inherit.data.bundle import load_metadata
        from inherit.data.records import build_records

        employees, customers = load_metadata(env.bundle)

        def records_for(p):
            data = json.loads((env.bundle / "products" / f"{p}.json").read_text(encoding="utf-8"))
            return build_records(data, employees, customers, p)

    summary: dict = {}
    total = 0.0
    for layer in layers:
        if estimate and layer not in ("bugs", "semantic"):
            print(f"{layer}: no model calls")
            continue
        for p in products:
            try:
                rep = runner(env.root, layer, p, records_for(p), estimate)
            except BaseException as e:
                print(f"FATAL {p} {layer}: {type(e).__name__}: {e}", file=sys.stderr)
                raise
            if estimate:
                total += (rep or {}).get("projected_usd", 0.0)
                continue
            _stamp(env, layer, p)
            fails = len((rep or {}).get("extraction_failures", []))
            complete = (rep or {}).get("manifest", {}).get("complete", True)
            summary.setdefault(p, {})[layer] = {"failures": fails, "complete": complete, "run_id": run_id}
    if estimate:
        print(f"estimate total projected {total:.4f} USD for {len(products)} products")
        return {}
    for p in products:
        print(f"{p}: " + ", ".join(f"{layer} failures {summary[p][layer]['failures']}"
                                   f"{'' if summary[p][layer]['complete'] else ' INCOMPLETE'}"
                                   for layer in layers if layer in summary[p]))
    env.out.mkdir(parents=True, exist_ok=True)
    path = env.out / "extraction_summary.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    old.setdefault("runs", {})[run_id] = {"access_log": f"extract-{run_id}.jsonl"}
    for p, v in summary.items():
        old.setdefault("products", {}).setdefault(p, {}).update(v)
    path.write_text(json.dumps(old, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------- load and what-if check
def load_all(env: Env, products: list[str]) -> dict:
    """Load every product on every server, create the ACL users (the loader also registers the UDF)."""
    stats: dict = {}
    for ver in env.servers:
        db = env.db(ver)
        for p in products:
            st = loadmod.load_product(db, p, loadmod.load_layers(p, env.facts), embed=env.embed)
            stats.setdefault(ver, {})[p] = {"nodes": st.nodes, "edges": st.edges, "embedded": st.embedded}
            print(f"[{ver}] {st}", flush=True)
        acl.setup_acl(db, products, label=ver)
    env.out.mkdir(parents=True, exist_ok=True)
    (env.out / "load_report.json").write_text(json.dumps(stats, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return stats


def verify_whatif(env: Env, products: list[str]) -> int:
    """Graph counterfactual against the Python check for every Engineering Lead, every product, every server.
    Returns the number of mismatches; writes results/freeze/whatif_check.json (with the graph identities checked)."""
    from inherit import whatif_check

    res: dict = {}
    bad = 0
    for ver in env.servers:
        for p in products:
            rows = whatif_check.compare(env.conn(p, ver), p)
            miss = [r for r in rows if not r["match"]]
            bad += len(miss)
            res.setdefault(ver, {})[p] = {
                "leads": len(rows), "mismatches": len(miss), "identity": graph_identity(env.graph(p, ver))["identity"],
                "diffs": [{"person": r["person"], "diff": r["diff"]} for r in miss]}
            print(f"[{ver}] {p}: {len(rows)} leads, {len(miss)} mismatches"
                  f"{' (no Engineering Lead: nothing to compare)' if not rows else ''}", flush=True)
            for r in miss:
                print(f"    MISMATCH {r['person']}: {r['diff']}")
    env.out.mkdir(parents=True, exist_ok=True)
    (env.out / "whatif_check.json").write_text(json.dumps(res, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    print("mismatches:", bad)
    return bad


# ---------------------------------------------------------------- content identities
def _line(row) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)


def _identity(nodes: list, edges: list) -> dict:
    lines = ["N" + _line(r) for r in nodes] + ["E" + _line(r) for r in edges]
    return {"identity": sha256_bytes("\n".join(sorted(lines)).encode("utf-8")), "nodes": len(nodes), "edges": len(edges)}


def facts_identity(facts) -> dict:
    """The identity a loaded graph must have: every node as the loader stores it (embeddings left out: they are
    pinned by the embedding cache hash) and every edge with its stored properties."""
    nodes = [[n.label, n.id, loadmod.node_props(n)] for n in facts.nodes]
    edges = [[e.src_id, e.type, e.dst_id, loadmod.edge_props(e)] for e in facts.edges]
    return _identity(nodes, edges)


def graph_identity(graph) -> dict:
    nodes = [[lb, i, {k: v for k, v in props.items() if k != "embedding"}] for lb, i, props in
             graph.ro_query("MATCH (n) RETURN labels(n)[0], n.id, properties(n)").result_set]
    edges = [list(r) for r in graph.ro_query("MATCH (a)-[e]->(b) RETURN a.id, type(e), b.id, properties(e)").result_set]
    return _identity(nodes, edges)


# ---------------------------------------------------------------- the manifest state
ARM_INDEX = {"A0": "raw", "A0v": "raw", "A0f": "cards"}  # which retrieval index an arm reads
ARMS_FILE = "arms.json"


def load_arms(env: Env, inv: list[str], problems: list[str]) -> dict | None:
    """results/freeze/arms.json (written by the controller after the pilot decision):
    {"arms": {"A5": {"products": "all", "sub_cap_usd": 3.0}, "A2": {"products": [6 names]}, ...}}.
    Returns {arm: {"products": [...], "sub_cap_usd": x}} for registered arms, or None."""
    raw = _read_json(env.out / ARMS_FILE)
    if raw is None or not raw.get("arms"):
        problems.append(f"results/freeze/{ARMS_FILE} missing or empty: the arm and product-subset decision is an input")
        return None
    from eval.arms import ARMS

    out = {}
    for name, spec in sorted(raw["arms"].items()):
        if name not in ARMS or name.startswith("dev-"):
            problems.append(f"{ARMS_FILE}: unknown arm {name!r}")
            continue
        prods = spec.get("products", "all")
        prods = list(inv) if prods == "all" else sorted(prods)
        if not prods or [x for x in prods if x not in inv]:
            problems.append(f"{ARMS_FILE}: arm {name} has an empty or unknown product list")
            continue
        out[name] = {"products": prods, "sub_cap_usd": spec.get("sub_cap_usd")}
    return out


def effective_config(arms: dict | None) -> dict:
    import dataclasses

    from eval import grade
    from eval.arms import ARMS
    from inherit import agent
    from inherit.extract import bug_rules, names, prompts, rules, semantic, semantic_prompts

    settings = {k: v for k, v in dataclasses.asdict(load_settings()).items() if k != "meter_path"}
    return {
        "settings": settings,  # budget, endpoint, models, proxy limits, server URLs
        "models": {"chat": MODEL, "embed": EMBED_MODEL, "judge": MODEL},
        "prices_pico_usd_per_token": {k: list(v) for k, v in PRICES_PICO.items()},
        "project_cap_usd": PROJECT_CAP_USD,
        "arms": {n: {**a, "config": ARMS[n].config, "config_sha": ARMS[n].config_sha(),
                     "prompt_versions": ARMS[n].prompt_versions} for n, a in (arms or {}).items()},
        "versions": {
            "bugs_prompt": prompts.BUGS_PROMPT_VERSION, "bug_rules": bug_rules.RULES_VERSION,
            "semantic_prompt": semantic_prompts.SEMANTIC_PROMPT_VERSION, "feature_fate": semantic.FATE_VERSION,
            "pending": semantic.PENDING_VERSION, "role_cues": names.ROLE_CUES_VERSION,
            "approval_predicate": rules.APPROVAL_VERSION, "rules": dict(rules.VERSIONS),
            "card": cards.CARD_VERSION, "agent_prompt": agent.PROMPT_VERSION,
            "judge_prompt": grade.PROMPT_VERSION, "scorer": grade.SCORER_VERSION,
        },
        "extraction_source": extraction_source_digest(),
    }


def _card_corpus_hash(graph) -> str:
    h = hashlib.sha256()
    for t in __import__("baselines.paper_retriever", fromlist=["x"]).card_texts(graph):
        h.update(t.encode("utf-8") + b"\n")
    return h.hexdigest()


RUN_ID_RE = re.compile(r"[A-Za-z0-9._-]+")


def _access_log(env: Env, run_id) -> Path | None:
    """runs/access/extract-<run id>.jsonl for a valid run id, else None. The summary never supplies a path."""
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id) or run_id.startswith("."):
        return None
    base = (env.root / "runs" / "access").resolve()
    f = (base / f"extract-{run_id}.jsonl").resolve()
    return f if f.parent == base else None


def _log_problem(f: Path, run_id: str) -> str | None:
    if not f.is_file():
        return "missing"
    with f.open(encoding="utf-8") as fh:
        first = fh.readline()
    try:
        head = json.loads(first)
    except ValueError:
        head = None
    if not isinstance(head, dict) or head.get("run_id") != run_id:
        return "empty, or its first line does not name this run"
    return None


def collect(env: Env) -> tuple[dict, list[str]]:
    """The freeze state as it is now, and every problem that stops it being a valid freeze."""
    problems: list[str] = []
    inv = inventory(env)
    if len(inv) != env.expected:
        problems.append(f"bundle lists {len(inv)} products, expected {env.expected}")
    dev, held = env.split(inv)
    arms = load_arms(env, inv, problems)
    cfg = effective_config(arms)
    state: dict = {
        "bundle_sha256": bundle_hash(env.bundle), "split_hash": split_hash(dev, held),
        "dev_products": dev, "held_products": held, "products": inv,
        "upstream_herb_commit": _read(env.root / "data" / "raw" / "COMMIT", problems, "upstream HERB commit").strip(),
        "uv_lock_sha256": sha256_file(env.root / "uv.lock") if (env.root / "uv.lock").exists() else None,
        "arms": arms, "arms_sha256": sha256_file(env.out / ARMS_FILE) if (env.out / ARMS_FILE).is_file() else None,
        "config": cfg, "config_hash": canon_hash(cfg),
    }
    if state["uv_lock_sha256"] is None:
        problems.append("uv.lock missing")
    extra = sorted(p.name for p in env.facts.iterdir() if p.is_dir() and p.name not in inv) if env.facts.is_dir() else []
    if extra:
        problems.append(f"fact directories for products outside the bundle: {extra}")
    layers: dict = {}
    reports: dict = {}
    sidecars: dict = {}
    summary = _read_json(env.out / "extraction_summary.json") or {}
    if not summary:
        problems.append("results/freeze/extraction_summary.json missing: run the freeze extraction")
    runs = summary.get("runs", {})
    used_runs: set = set()
    for p in inv:
        d = env.facts / p
        if not d.is_dir():
            problems.append(f"{p}: no fact directory")
            continue
        layers[p] = {}
        for f in loadmod.LAYER_FILES:
            if not (d / f).is_file():
                problems.append(f"{p}: layer file {f} is absent (an empty file is allowed, a missing one is not)")
            else:
                layers[p][f] = sha256_file(d / f)
        for layer, rf in SUMMARY_FILES.items():
            if (d / rf).is_file():
                reports.setdefault(p, {})[rf] = sha256_file(d / rf)
            else:
                problems.append(f"{p}: extraction report {rf} is absent")
        side = d / VERSIONS_FILE
        sidecars[p] = sha256_file(side) if side.is_file() else None
        rec = json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}
        for layer in LAYERS:
            if rec.get(layer) != layer_versions(layer):
                problems.append(f"{p}: layer {layer} was not extracted under the current code and prompt versions "
                                "(re-run extract --products all)")
            ent = summary.get("products", {}).get(p, {}).get(layer)
            if ent is None or ent.get("run_id") not in runs:
                problems.append(f"{p}: layer {layer} has no recorded freeze extraction run")
            else:
                used_runs.add(ent["run_id"])
    state["layers"] = layers
    state["facts_hash"] = _facts_hash(env.facts, inv)
    state["extraction_reports"] = reports
    state["version_sidecars"] = sidecars
    state["extraction_summary_sha256"] = (sha256_file(env.out / "extraction_summary.json")
                                          if (env.out / "extraction_summary.json").is_file() else None)
    logs = {}
    for rid in sorted(used_runs):
        f = _access_log(env, rid)
        why = "has an invalid run id" if f is None else _log_problem(f, rid)
        if why:
            problems.append(f"access log of extraction run {rid!r}: {why}")
        else:
            logs[f.name] = sha256_file(f)
    state["access_logs"] = logs
    lr = _read_json(env.out / "load_report.json") or {}
    for ver in env.servers:
        miss = [p for p in inv if p not in lr.get(ver, {})]
        if miss:
            problems.append(f"load report has no entry on server {ver} for {len(miss)} products ({miss[:3]}...): "
                            "run load --products all")
    state["load_report_sha256"] = (sha256_file(env.out / "load_report.json")
                                   if (env.out / "load_report.json").is_file() else None)
    state["embedding_cache_sha256"] = _embedding_hash(env, inv, problems)
    graphs, corpus = {}, {}
    wi = _read_json(env.out / "whatif_check.json")
    if wi is None:
        problems.append("results/freeze/whatif_check.json missing: run verify-whatif")
    primary = {}
    for p in inv:
        if p not in layers or len(layers[p]) != len(loadmod.LAYER_FILES):
            continue
        try:
            want = facts_identity(loadmod.load_layers(p, env.facts))
        except Exception as e:  # noqa: BLE001
            problems.append(f"{p}: facts unreadable: {type(e).__name__}: {e}")
            continue
        graphs[p] = {}
        for i, ver in enumerate(env.servers):
            try:
                g = env.graph(p, ver)
                got = graph_identity(g)
            except Exception as e:  # noqa: BLE001
                problems.append(f"{p} [{ver}]: graph unreadable: {type(e).__name__}: {e}")
                continue
            graphs[p][ver] = {**got, "a0f_identity": _a0f_identity(g)}
            if got["identity"] != want["identity"]:
                problems.append(f"{p} [{ver}]: graph content identity differs from the facts "
                                f"(graph {got['nodes']} nodes {got['edges']} edges, facts {want['nodes']}/{want['edges']})")
            if i == 0:
                corpus[p] = _card_corpus_hash(g)
                primary[p] = g
            if wi is not None:
                w = wi.get(ver, {}).get(p)
                if w is None:
                    problems.append(f"{p} [{ver}]: no what-if check recorded")
                elif w["mismatches"] or w["identity"] != got["identity"]:
                    problems.append(f"{p} [{ver}]: what-if check has {w['mismatches']} mismatches or is stale "
                                    "(graph changed since it ran)")
    state["graphs"] = graphs
    state["card_corpus_sha256"] = corpus
    state["whatif_check_sha256"] = sha256_file(env.out / "whatif_check.json") if wi is not None else None
    state["a0_indexes"] = _indexes(env, arms or {}, primary, problems)
    state["a2"] = _a2(env, arms or {}, problems)
    return state, problems


def _a0f_identity(graph) -> str:
    from baselines.paper_retriever import graph_identity as a0f

    return a0f(graph)


def _facts_hash(facts_dir: Path, inv: list[str]) -> str:
    from eval.receipt import facts_hash

    return facts_hash(facts_dir, inv)


def _read(path: Path, problems: list[str], what: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        problems.append(f"{what} missing ({path})")
        return ""


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _embedding_hash(env: Env, inv: list[str], problems: list[str]) -> str | None:
    cache = env.root / "data" / "cache" / "embed"
    files: dict[str, str] = {}
    missing = 0
    for p in inv:
        if not (env.facts / p).is_dir():
            continue
        try:
            targets = loadmod.embed_targets(loadmod.load_layers(p, env.facts))
        except Exception:  # noqa: BLE001  # reported once, by the identity check
            continue
        for _, _, text in targets:
            f = embed_mod._cache_file(cache, text)
            if f.exists():
                files[f.name] = sha256_file(f)
            else:
                missing += 1
    if missing:
        problems.append(f"{missing} embedding texts have no cache file under data/cache/embed")
    return canon_hash(files)


def _dir_hash(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(x for x in d.rglob("*") if x.is_file()):
        h.update(f.relative_to(d).as_posix().encode() + b"\0" + sha256_file(f).encode() + b"\0")
    return h.hexdigest()


def _indexes(env: Env, arms: dict, graphs: dict, problems: list[str]) -> dict:
    """The retrieval indexes the selected arms read, one per (kind, product): the persisted fingerprint must equal the
    one recomputed from the current corpus and settings, and every persisted file is hashed."""
    need: dict[str, set] = {}
    for name, a in arms.items():
        if name in ARM_INDEX:
            need.setdefault(ARM_INDEX[name], set()).update(a["products"])
    if not need:
        return {}
    from baselines import paper_retriever as pr
    from inherit.data.bundle import load_bundle

    out: dict = {}
    for kind in sorted(need):
        for p in sorted(need[kind]):
            d = env.root / "data" / "cache" / "baselines" / kind / p
            if not (d / "meta.json").is_file():
                problems.append(f"{kind} retrieval index of {p} is missing ({d})")
                continue
            try:
                ident = (pr.texts_identity(pr.paper_texts(load_bundle(env.bundle, p))) if kind == "raw"
                         else _a0f_identity(graphs[p]))
            except Exception as e:  # noqa: BLE001
                problems.append(f"{kind} index of {p}: corpus identity not computable: {type(e).__name__}: {e}")
                continue
            got = json.loads((d / "meta.json").read_text(encoding="utf-8"))["fingerprint"]
            if got != pr.fingerprint(ident):
                problems.append(f"{kind} index of {p}: persisted fingerprint does not match the current corpus and settings")
            out.setdefault(kind, {})[p] = {"fingerprint": got, "files_sha256": _dir_hash(d)}
    return out


def _default_a2_graph(product: str):
    from falkordb import FalkorDB

    from baselines import graphrag_sdk_arm as a2

    return FalkorDB(host=a2.FALKOR_HOST, port=a2.FALKOR_PORT).select_graph(a2.graph_name(product))


def a2_export_identity(graph) -> dict:
    """Content identity of an A2 graph: nodes (labels, properties without vectors) and edges (type, properties, the
    two endpoint rows), independent of internal ids."""
    def clean(props):
        return {k: v for k, v in props.items()
                if "embedding" not in k.lower() and not (isinstance(v, list) and len(v) >= 64
                                                         and all(isinstance(x, float) for x in v))}

    rows = graph.ro_query("MATCH (n) RETURN id(n), labels(n), properties(n)").result_set
    line = {i: _line([sorted(lb), clean(pr)]) for i, lb, pr in rows}
    nodes = list(line.values())
    edges = [[t, clean(pr), line[a], line[b]] for a, t, pr, b in
             graph.ro_query("MATCH (a)-[e]->(b) RETURN id(a), type(e), properties(e), id(b)").result_set]
    return _identity(nodes, edges)


def _a2(env: Env, arms: dict, problems: list[str]) -> dict:
    """A2's ontology, finalized build manifest and graph export identity for each of its sampled products."""
    out: dict = {}
    for p in arms.get("A2", {}).get("products", []):
        row = {}
        for k in ("ontology", "build"):
            f = env.root / "runs" / "a2" / f"{p}.{k}.json"
            if f.is_file():
                row[k] = sha256_file(f)
            else:
                problems.append(f"A2 {k} file of {p} is missing ({f.name})")
        b = _read_json(env.root / "runs" / "a2" / f"{p}.build.json")
        if b is not None and b.get("finalized") is not True:
            problems.append(f"A2 build of {p} is not finalized")
        try:
            row["export"] = a2_export_identity(env.a2_graph(p))
            if not row["export"]["nodes"]:
                problems.append(f"A2 graph of {p} is empty")
        except Exception as e:  # noqa: BLE001
            problems.append(f"A2 graph of {p} unreadable: {type(e).__name__}: {e}")
        out[p] = row
    return out


def build_manifest(env: Env) -> dict:
    state, problems = collect(env)
    if problems:
        raise FreezeError("freeze manifest refused:\n  " + "\n  ".join(problems))
    commit = env.commit(env.root)
    manifest = {"code_commit": commit, **state}
    env.out.mkdir(parents=True, exist_ok=True)
    (env.out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1, default=str) + "\n",
                                           encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- check
def diff(a, b, path: str = "") -> list[str]:
    """Every difference between a recorded value `a` and the current value `b`."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: not recorded, now {_short(b[k])}")
            elif k not in b:
                out.append(f"{path}/{k}: recorded {_short(a[k])}, now absent")
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
        return out
    return [] if a == b else [f"{path}: recorded {_short(a)}, now {_short(b)}"]


def _short(v) -> str:
    s = json.dumps(v, sort_keys=True, default=str)
    return s if len(s) <= 120 else s[:117] + "..."


def check(env: Env) -> list[str]:
    path = env.out / "manifest.json"
    if not path.is_file():
        return [f"{path} does not exist"]
    rec = json.loads(path.read_text(encoding="utf-8"))
    state, problems = collect(env)
    state_rec = {k: v for k, v in rec.items() if k != "code_commit"}
    out = list(problems) + diff(state_rec, state)
    out += env.drift(env.root, rec["code_commit"])
    return out


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None, env: Env | None = None) -> int:
    ap = argparse.ArgumentParser(prog="inherit.freeze")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "load", "verify-whatif"):
        sp = sub.add_parser(name)
        sp.add_argument("--products", default="all" if name != "extract" else None, required=name == "extract")
        sp.add_argument("--servers", default=None)
        if name == "extract":
            sp.add_argument("--layers", default=",".join(LAYERS))
            sp.add_argument("--estimate", action="store_true")
    sub.add_parser("manifest")
    sub.add_parser("check")
    ap.add_argument("--repo-root", default=None, help="tests only")
    args = ap.parse_args(argv)
    env = env or Env(root=Path(args.repo_root) if args.repo_root else ROOT)
    if getattr(args, "servers", None):
        env.servers = tuple(args.servers.split(","))
    try:
        if args.cmd == "extract":
            from inherit import ioguard

            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            ioguard.install(env.root, env.root / "runs" / "access" / f"extract-{stamp}.jsonl")
            env.extra["run_id"] = stamp
            layers = tuple(args.layers.split(","))
            if any(x not in LAYERS for x in layers):
                raise SystemExit(f"unknown layers: {layers}")
            extract(env, select(args.products, inventory(env), env.split), layers, estimate=args.estimate,
                    run_id=env.extra.get("run_id", "norun"))
        elif args.cmd == "load":
            load_all(env, select(args.products, inventory(env), env.split))
        elif args.cmd == "verify-whatif":
            return 1 if verify_whatif(env, select(args.products, inventory(env), env.split)) else 0
        elif args.cmd == "manifest":
            m = build_manifest(env)
            print(f"wrote {env.out / 'manifest.json'}: {len(m['products'])} products, facts_hash {m['facts_hash']}, "
                  f"code {m['code_commit']}")
        else:
            diffs = check(env)
            for d in diffs:
                print("DIFF", d)
            print(f"{len(diffs)} differences")
            return 1 if diffs else 0
    except FreezeError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
