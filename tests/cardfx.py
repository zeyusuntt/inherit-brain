"""Fixture helpers shared by the card and depth-1 tests: the load fixture plus the shapes Task 7 needs."""

import dataclasses

import pytest
from falkordb import FalkorDB

from conftest import fake_embed, fx_edge, fx_node
from inherit.config import load_settings
from inherit.facts import Edge, Facts
from inherit.graph import acl as acl_mod
from inherit.graph import load as loadmod
from inherit.tools.depth1 import read_only_graph

# These tests use their own product so a concurrent run of the other graph tests (which clean "mini") cannot reset
# their graphs or users.
P = "cardmini"
PRODUCT = f"Product:{P}"
PR101 = "PR:https://github.com/mini/mini-server/pull/101"
URL1 = "Url:https://docs.example.com/export"
LONG = "Longword " * 1000  # 9,000 characters


def card_facts(load_facts, many: int = 0, extra: Facts | None = None) -> Facts:
    """`many` > 0 adds that many long documents under the product (the 200-row and budget cases)."""
    N, E = fx_node, fx_edge
    nodes = [
        N("Bug", "b5", id="b5", status="active", title="Users can't log in", component="Auth",
          aliases=["login failure"], unresolved=True, active=True, reopened=False, no_fix_needed=False,
          resolver="Person:eid_b2000002"),
        N("Document", "long_doc", id="long_doc", type="Long Report", content=LONG, feedback="- one\n- two"),
    ]
    edges = [
        E("HAS", "Product:mini", "Bug:b5"), E("HAS", "Product:mini", "Document:long_doc"),
        E("REPORTED_BY", "Bug:b1", "Customer:CUST-0001"), E("REPORTED_BY", "Bug:b2", "Customer:CUST-0001"),
        E("REPORTED_BY", "Bug:b3", "Customer:CUST-0002"), E("REPORTED_BY", "Bug:b5", "Customer:CUST-0001"),
        E("FIXES", PR101, "Bug:b1"),
        E("ABOUT", "Observation:o1", "Competitor:Rival Corp"),
        Edge("SHARED", "Person:eid_e5000005", URL1, {}, tuple(f"rec{i}" for i in range(5)), "fixture/v1"),
    ]
    if many:
        for i in range(many):
            nodes.append(N("Document", f"d{i:04d}", id=f"d{i:04d}", type="Spec", date="2026-07-01T00:00:00",
                           content=f"doc {i} " + "w" * 300))
            edges.append(E("HAS", "Product:mini", f"Document:d{i:04d}"))
    return rename(load_facts.merge(Facts(nodes, edges)).merge(extra or Facts()))


def rename(facts: Facts) -> Facts:
    """Product:mini -> Product:cardmini."""
    nodes = [dataclasses.replace(n, id=PRODUCT, props={**n.props, "name": P}) if n.id == "Product:mini" else n
             for n in facts.nodes]
    swap = lambda i: PRODUCT if i == "Product:mini" else i  # noqa: E731
    edges = [dataclasses.replace(e, src_id=swap(e.src_id), dst_id=swap(e.dst_id)) for e in facts.edges]
    return Facts(nodes, edges)


def load(db, facts, product=P):
    loadmod.load_product(db, product, facts, embed=fake_embed)
    return db.select_graph(f"herb:{product}")


def agent_graph(db, tmp_path, product=P):
    """The graph as the read-only agent_<product> user sees it."""
    pw = acl_mod.setup_acl(db, [product], path=tmp_path / "acl.json", label="t")[f"agent_{product}"]
    kw = db.connection.connection_pool.connection_kwargs
    return read_only_graph(kw["host"], kw["port"], f"agent_{product}", pw, f"herb:{product}")


def clean(db) -> None:
    for g in db.list_graphs():
        if g in (f"herb:{P}",) or g.startswith(f"whatif:{P}:"):
            db.select_graph(g).delete()
    for u in (f"agent_{P}", f"whatif_{P}", f"build_{P}"):
        db.connection.execute_command("ACL", "DELUSER", u)


def _connect(ver):
    url = load_settings().falkor_urls[ver]
    db = FalkorDB.from_url(url)
    try:
        db.connection.ping()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"FalkorDB {ver} not reachable at {url}: {e}")
    return db


@pytest.fixture(params=["4.22", "6.0"])
def srv(request):
    db = _connect(request.param)
    clean(db)
    yield db
    clean(db)


@pytest.fixture
def srv_both():
    dbs = {v: _connect(v) for v in load_settings().falkor_urls}
    for db in dbs.values():
        clean(db)
    yield dbs
    for db in dbs.values():
        clean(db)
