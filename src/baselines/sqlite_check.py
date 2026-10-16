"""A3 (design F12): a consistency check, not an answering arm.

The frozen facts of a product (the loader's LAYER_FILES) are exported to SQLite as `nodes` and `edges` tables with JSON
props. Three catalogue query families are re-written in SQL with the semantics of inherit/tools/catalog.py and their
canonical rows are compared with the graph tools on a small parameter matrix:

* bug_board by company (`actor=company`; each status; rank all or max with every tie kept; customer filter);
* fix_trail (customer -> bugs -> fix PR links -> authors; customer, bug and engineer filters);
* pr_board approved by feature (`approval=approved`, links).

Name resolution is shared code (`fold`, `_area_key`) plus the exact forms only: customer = company, Customer id or
source id; bug = id, title or alias; engineer = employee id or directory name; feature = id, title, topic or alias. The
tools' fuzzy fallbacks (containment, stemmed token overlap) are not re-implemented, so the matrix stays on exact forms
and on texts that match nothing. The claim is limited to these three families.

    uv run python -m baselines.sqlite_check --products dev [--servers 4.22,6.0]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from pathlib import Path

from inherit.config import ROOT
from inherit.facts import Facts
from inherit.graph.load import load_layers
from inherit.tools.catalog import TOOLS, _area_key
from inherit.tools.depth1 import fold

DEV_PRODUCTS = ("ActionGenie", "AnomalyForce", "ContentForce", "KnowledgeForce", "LeadForce", "TrendForce")
STATUSES = ("unresolved", "resolved", "active", "reopened", "no_fix_needed", "any")
FAMILY_TOOL = {"bug_board": "bug_board", "fix_trail": "fix_trail", "pr_board": "pr_board"}


# ---------------------------------------------------------------- export
def export_facts(facts: Facts, path: str = ":memory:") -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.create_function("fold", 1, lambda s: fold(s) if isinstance(s, str) else None, deterministic=True)
    db.executescript("""
        create table nodes (id text primary key, label text not null, props text not null);
        create table edges (type text not null, src_id text not null, dst_id text not null, key text not null,
                            props text not null, src text not null, primary key (type, src_id, dst_id, key));
        create index edges_src on edges (type, src_id);
        create index edges_dst on edges (type, dst_id);
        create index nodes_label on nodes (label);""")
    dump = lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False)  # noqa: E731
    db.executemany("insert into nodes values (?, ?, ?)", [(n.id, n.label, dump(n.props)) for n in facts.nodes])
    db.executemany("insert into edges values (?, ?, ?, ?, ?, ?)",
                   [(e.type, e.src_id, e.dst_id, e.key, dump(e.props), dump(list(e.src))) for e in facts.edges])
    db.commit()
    return db


def export_product(product: str, facts_dir: Path | None = None, path: str = ":memory:") -> sqlite3.Connection:
    return export_facts(load_layers(product, facts_dir), path)


# ---------------------------------------------------------------- name resolution (exact forms)
def _is_product_name(db, key: str) -> bool:
    for pid, props in db.execute("select id, props from nodes where label = 'Product'"):
        p = json.loads(props)
        own = {_area_key(x) for x in [pid, p.get("name"), *(p.get("aliases") or []), *(p.get("previous_names") or [])]
               if isinstance(x, str)}
        if key in own or key == _area_key(f"Product:{pid.split(':', 1)[-1]}"):
            return True
    return False


def _companies(db, text):
    if text is None or _is_product_name(db, _area_key(text)):
        return None
    return [r[0] for r in db.execute(
        "select distinct json_extract(props, '$.company') c from nodes where label = 'Customer' and c is not null "
        "and ? in (fold(c), fold(id), fold(json_extract(props, '$.id'))) order by c", (fold(text),))]


def _bugs(db, text):
    if text is None:
        return None
    return [r[0] for r in db.execute(
        "select id from nodes where label = 'Bug' and (? in (fold(id), fold(json_extract(props, '$.title'))) or exists "
        "(select 1 from json_each(props, '$.aliases') j where fold(j.value) = ?)) order by id", (fold(text), fold(text)))]


def _engineers(db, text):
    if text is None:
        return None
    t = fold(text[7:] if text.startswith("Person:") else text)
    return [r[0] for r in db.execute(
        "select distinct json_extract(props, '$.eid') e from nodes where label = 'Person' and e is not null "
        "and (fold(e) = ? or fold(json_extract(props, '$.name')) = ?) order by e", (t, t))]


def _features(db, text):
    if text is None or _is_product_name(db, _area_key(text)):
        return None
    f = fold(text)
    return [r[0] for r in db.execute(
        "select id from nodes where label = 'Feature' and (fold(id) = ? or ? in (fold(json_extract(props, '$.title')), "
        "fold(json_extract(props, '$.topic'))) or exists (select 1 from json_each(props, '$.aliases') j "
        "where fold(j.value) = ?)) order by id", (f, f, f))]


def _in(col: str, values) -> tuple[str, list]:
    """`col in (...)` for a resolved list; None means no filter and [] matches nothing."""
    if values is None:
        return "1", []
    return f"{col} in (select value from json_each(?))", [json.dumps(values)]


# ---------------------------------------------------------------- the three SQL families
def bug_board_by_company(db, status: str = "any", rank: str = "all", customer: str | None = None) -> list[tuple]:
    """[(company, distinct bug count, sorted bug ids)], count descending then company; rank=max keeps every tie."""
    flag = "1" if status == "any" else f"json_extract(b.props, '$.{status}') = 1"
    cos = _companies(db, customer)
    w_co, a_co = _in("json_extract(c.props, '$.company')", cos)
    w_bc, a_bc = _in("j.value", cos)
    pairs = f"""select distinct json_extract(c.props, '$.company') company, b.id bug
        from nodes b join edges e on e.type = 'REPORTED_BY' and e.src_id = b.id
        join nodes c on c.id = e.dst_id and c.label = 'Customer'
        where b.label = 'Bug' and {flag} and json_extract(c.props, '$.company') is not null and {w_co}
        and (? or exists (select 1 from json_each(b.props, '$.companies') j where {w_bc}))"""
    args = [*a_co, cos is None, *a_bc]
    top = "where n = (select max(n) from counts)" if rank == "max" else ""
    rows = db.execute(f"""with pairs as ({pairs}),
        counts as (select company, count(*) n from pairs group by company)
        select p.company, k.n, p.bug from pairs p join (select * from counts {top}) k using (company)
        order by k.n desc, p.company, p.bug""", args).fetchall()
    out: dict[str, list] = {}
    for company, n, bug in rows:
        out.setdefault(company, [n, []])[1].append(bug)
    return [(c, n, tuple(b)) for c, (n, b) in out.items()]


def fix_trail(db, customer: str | None = None, bug: str | None = None, engineer: str | None = None) -> list[tuple]:
    """[(fix PR link, bug id, authors, companies, customer ids)] ordered by (link, bug)."""
    w_co, a_co = _in("json_extract(c.props, '$.company')", _companies(db, customer))
    w_bu, a_bu = _in("b.id", _bugs(db, bug))
    w_en, a_en = _in("json_extract(p.props, '$.eid')", _engineers(db, engineer))
    rows = db.execute(f"""select distinct json_extract(pr.props, '$.link') link, b.id, json_extract(p.props, '$.eid'),
            json_extract(c.props, '$.company'), c.id
        from edges fx join nodes pr on pr.id = fx.src_id and pr.label = 'PR'
        join nodes b on b.id = fx.dst_id and b.label = 'Bug'
        join edges rb on rb.type = 'REPORTED_BY' and rb.src_id = b.id
        join nodes c on c.id = rb.dst_id and c.label = 'Customer'
        join edges op on op.type = 'OPENED' and op.dst_id = pr.id
        join nodes p on p.id = op.src_id and p.label = 'Person'
        where fx.type = 'FIXES' and {w_co} and {w_bu} and {w_en}
        order by 1, 2""", [*a_co, *a_bu, *a_en]).fetchall()
    groups: dict[tuple, list[set]] = {}
    for link, bid, eid, company, cid in rows:
        g = groups.setdefault((link, bid), [set(), set(), set()])
        g[0].add(eid), g[1].add(company), g[2].add(cid)
    return [(link, bid, tuple(sorted(a)), tuple(sorted(co, key=str)), tuple(sorted(cu)))
            for (link, bid), (a, co, cu) in sorted(groups.items())]


def pr_board_approved(db, feature: str | None = None) -> list[tuple]:
    """[(link, title, approval)] of approved product-repo PRs, about the feature when one is given."""
    w_f, a_f = _in("a.dst_id", _features(db, feature))
    about = "" if feature is None or _features(db, feature) is None else \
        f"and exists (select 1 from edges a where a.type = 'ABOUT' and a.src_id = pr.id and {w_f})"
    return db.execute(f"""select distinct json_extract(pr.props, '$.link') link, json_extract(pr.props, '$.title'),
            json_extract(pr.props, '$.approval')
        from nodes pr where pr.label = 'PR' and json_extract(pr.props, '$.repo_scope') = 'product'
        and json_extract(pr.props, '$.approval') = 'approved' {about} order by link""", a_f if about else []).fetchall()


def sql_rows(db, family: str, params: dict) -> list[tuple]:
    if family == "bug_board":
        return bug_board_by_company(db, params.get("status", "any"), params.get("rank", "all"), params.get("customer"))
    if family == "fix_trail":
        return fix_trail(db, params.get("customer"), params.get("bug"), params.get("engineer"))
    return pr_board_approved(db, params.get("feature"))


# ---------------------------------------------------------------- the graph side
def graph_rows(graph, product: str, family: str, params: dict) -> list[tuple]:
    """The catalogue tool's rows (all pages) in the same canonical form."""
    kw = {k: v for k, v in params.items()}
    if family == "bug_board":
        kw["actor"] = "company"
    if family == "pr_board":
        kw["approval"] = "approved"
    rows, cursor = [], None
    while True:
        res = TOOLS[FAMILY_TOOL[family]].run(graph, product, **kw, **({"cursor": cursor} if cursor else {}))
        rows += res.rows
        cursor = res.cursor
        if not res.truncated or not cursor:
            break
    if family == "bug_board":
        return [(r["answer"], r["count"], tuple(r["bugs"])) for r in rows]
    if family == "fix_trail":
        return [(r["answer"], r["bug"], (r["author"],), tuple(r["companies"]), tuple(r["customers"])) for r in rows]
    return [(r["answer"], r["title"], r["approval"]) for r in rows]


# ---------------------------------------------------------------- matrix, comparison, latency
def param_matrix(db) -> list[tuple[str, dict]]:
    """Parameters read from the data: every status and rank, and for each family the empty result, a product name used
    as a filter (ignored), a customer id and a company name (same customers), and the shared-feature (duplicate) case."""
    pname = (db.execute("select json_extract(props, '$.name') from nodes where label = 'Product'").fetchone() or ["P"])[0]
    cos = [r[0] for r in db.execute("select distinct json_extract(props, '$.company') c from nodes "
                                    "where label = 'Customer' and c is not null order by c limit 3")]
    cid = db.execute("select json_extract(props, '$.id') from nodes where label = 'Customer' and "
                     "json_extract(props, '$.company') = ?", (cos[0],)).fetchone()[0] if cos else None
    m: list[tuple[str, dict]] = []
    for s in STATUSES:
        for rank in ("all", "max"):
            m.append(("bug_board", {"status": s, "rank": rank}))
    for c in ([*cos[:2], cid, pname, "Nobody Inc"] if cos else [pname, "Nobody Inc"]):
        m += [("bug_board", {"status": "any", "customer": c}), ("bug_board", {"status": "resolved", "customer": c})]
    trail = fix_trail(db)
    m += [("fix_trail", {}), ("fix_trail", {"customer": pname}), ("fix_trail", {"customer": "Nobody Inc"})]
    m += [("fix_trail", {"customer": c}) for c in cos]
    if trail:
        link, bid, authors, companies, _ = trail[0]
        title = db.execute("select json_extract(props, '$.title') from nodes where id = ?", (bid,)).fetchone()[0]
        name = db.execute("select json_extract(props, '$.name') from nodes where json_extract(props, '$.eid') = ?",
                          (authors[0],)).fetchone()[0]
        m += [("fix_trail", {"bug": bid}), ("fix_trail", {"bug": title}), ("fix_trail", {"engineer": authors[0]}),
              ("fix_trail", {"engineer": name}), ("fix_trail", {"customer": companies[0], "engineer": authors[0]}),
              ("fix_trail", {"customer": "Nobody Inc", "engineer": authors[0]}), ("fix_trail", {"bug": "no such bug"})]
    feats = db.execute("""select f.id, json_extract(f.props, '$.title'), count(distinct a.src_id) n from nodes f
        left join edges a on a.type = 'ABOUT' and a.dst_id = f.id and a.src_id like 'PR:%'
        where f.label = 'Feature' group by f.id order by n desc, f.id limit 4""").fetchall()
    m += [("pr_board", {}), ("pr_board", {"feature": pname}), ("pr_board", {"feature": "no such feature"})]
    for fid, title, _ in feats:
        m += [("pr_board", {"feature": fid})] + ([("pr_board", {"feature": title})] if title else [])
    return m


def compare(db, graph, product: str, matrix: list[tuple[str, dict]] | None = None) -> list[dict]:
    """Every matrix entry whose canonical rows differ (empty list: equal everywhere)."""
    diffs = []
    for family, params in matrix or param_matrix(db):
        a, b = sql_rows(db, family, params), graph_rows(graph, product, family, params)
        if a != b:
            diffs.append({"family": family, "params": params, "sql": a, "graph": b,
                          "only_sql": [r for r in a if r not in b], "only_graph": [r for r in b if r not in a]})
    return diffs


def latency(db, graph, product: str, matrix: list[tuple[str, dict]], repeat: int = 3) -> dict[str, dict]:
    """Mean milliseconds per query over the matrix, per family: SQLite and the graph tool (round trip, all pages)."""
    out: dict[str, dict] = {}
    for family in FAMILY_TOOL:
        ps = [p for f, p in matrix if f == family]
        t = {"sql_ms": 0.0, "graph_ms": 0.0}
        for key, fn in (("sql_ms", lambda p: sql_rows(db, family, p)), ("graph_ms", lambda p: graph_rows(graph, product, family, p))):
            samples = []
            for _ in range(repeat):
                for p in ps:
                    t0 = time.perf_counter()
                    fn(p)
                    samples.append((time.perf_counter() - t0) * 1000)
            t[key] = round(statistics.mean(samples), 3) if samples else 0.0
        out[family] = {**t, "queries": len(ps)}
    return out


def main() -> None:
    from inherit.config import load_settings
    from inherit.tools.depth1 import agent_graph

    ap = argparse.ArgumentParser()
    ap.add_argument("--products", default="dev")
    ap.add_argument("--servers", default=",".join(load_settings().falkor_urls))
    a = ap.parse_args()
    if a.products != "dev":
        raise SystemExit("A3 runs on dev products only")
    products = list(DEV_PRODUCTS)
    loc = sum(1 for ln in Path(__file__).read_text().splitlines() if ln.strip())
    print(f"lines of code (non-blank): {loc}")
    for product in products:
        db = export_product(product)
        matrix = param_matrix(db)
        for server in a.servers.split(","):
            g = agent_graph(product, server)
            diffs = compare(db, g, product, matrix)
            lat = latency(db, g, product, matrix)
            print(f"{product} {server}: {len(matrix)} queries, {len(diffs)} differences, latency {json.dumps(lat)}")
            for d in diffs[:3]:
                print("  DIFF", d["family"], d["params"], "only_sql", d["only_sql"][:2], "only_graph", d["only_graph"][:2])


if __name__ == "__main__":
    main()
