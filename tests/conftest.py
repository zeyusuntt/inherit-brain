import json
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_raw(tmp_path):
    """A raw-layout tree (products/, metadata/) holding the mini fixture."""
    raw = tmp_path / "raw"
    (raw / "products").mkdir(parents=True)
    (raw / "metadata").mkdir()
    shutil.copy(FIXTURES / "mini_product.json", raw / "products" / "mini.json")
    shutil.copy(FIXTURES / "mini_metadata" / "employee.json", raw / "metadata" / "employee.json")
    shutil.copy(FIXTURES / "mini_metadata" / "customers_data.json", raw / "metadata" / "customers_data.json")
    (raw / "metadata" / "salesforce_team.json").write_text(json.dumps({"x": 1}))
    return raw


@pytest.fixture
def mini_bundle(mini_raw, tmp_path):
    from inherit.data.bundle import make_bundle

    out = tmp_path / "bundle"
    make_bundle(mini_raw, out)
    return out


# ---------------------------------------------------------------- FalkorDB (Task 6)
import hashlib  # noqa: E402

import numpy as np  # noqa: E402
import redis  # noqa: E402
from falkordb import FalkorDB  # noqa: E402

from inherit.config import load_settings  # noqa: E402
from inherit.data.bundle import load_bundle, load_metadata  # noqa: E402
from inherit.data.records import build_records  # noqa: E402
from inherit.extract.names import resolve_meeting_names  # noqa: E402
from inherit.extract.structural import structural_facts_from_records  # noqa: E402
from inherit.facts import Edge, Facts, Node  # noqa: E402

TEST_PRODUCTS = ("mini", "other")


def fake_vec(text: str) -> list[float]:
    """Deterministic unit vector of 1024 components for a text."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
    v = np.random.RandomState(seed).randn(1024)
    return (v / np.linalg.norm(v)).tolist()


def fake_embed(texts):
    return [fake_vec(t) for t in texts]


def _clean(db) -> None:
    r = db.connection
    for g in db.list_graphs():
        if g.split(":")[1:2] and g.split(":")[1] in TEST_PRODUCTS and g.split(":")[0] in ("herb", "whatif"):
            db.select_graph(g).delete()
    for u in r.execute_command("ACL", "USERS"):
        if u.endswith(tuple("_" + p for p in TEST_PRODUCTS)) or u.startswith("overpriv_"):
            r.execute_command("ACL", "DELUSER", u)
    for lib in ("herb_test",):
        try:
            r.execute_command("GRAPH.UDF", "DELETE", lib)
        except redis.ResponseError:
            pass


@pytest.fixture(params=["4.22", "6.0"])
def falkor(request):
    """A FalkorDB client for one server version. A server that is down fails the test, it does not skip it."""
    url = load_settings().falkor_urls[request.param]
    db = FalkorDB.from_url(url)
    try:
        db.connection.ping()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"FalkorDB {request.param} not reachable at {url}: {e}")
    db.version_id = request.param
    _clean(db)
    yield db
    _clean(db)


@pytest.fixture
def falkor_both():
    dbs = {}
    for ver, url in load_settings().falkor_urls.items():
        db = FalkorDB.from_url(url)
        try:
            db.connection.ping()
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"FalkorDB {ver} not reachable at {url}: {e}")
        _clean(db)
        dbs[ver] = db
    yield dbs
    for db in dbs.values():
        _clean(db)


def fx_node(label, key, **props):
    return Node(label, f"{label}:{key}", props)


def fx_edge(type_, src, dst, key="", **props):
    return Edge(type_, src, dst, props, ("r1",), "fixture/v1", key)


@pytest.fixture
def load_facts(mini_bundle):
    """Structural + names facts of the mini product, plus a synthetic layer (Product root, bugs and events,
    features, competitor, observations, parallel keyed edges). Built directly: Facts does not validate."""
    emp, cust = load_metadata(mini_bundle)
    rec = build_records(load_bundle(mini_bundle, "mini"), emp, cust, "mini")
    base = structural_facts_from_records(rec).merge(resolve_meeting_names(rec)[0])
    N, E = fx_node, fx_edge
    prs = sorted(n.id for n in base.nodes if n.label == "PR")
    nodes = [
        N("Product", "mini", name="mini"),
        N("Bug", "b1", id="b1", status="resolved", title="Login page crashes on Safari"),
        N("Bug", "b2", id="b2", status="resolved", title="Slow export of reports"),
        N("Bug", "b3", id="b3", status="active", title="Avatar upload fails"),
        N("Bug", "b4", id="b4", status="active", title="Dark mode flicker"),
        N("BugEvent", "b1#0", id="b1#0", ts="2026-06-10T09:00:00", type="REPORTED", position=0),
        N("BugEvent", "b1#1", id="b1#1", ts="2026-06-20T09:00:00", type="RESOLVED", position=1),
        N("BugEvent", "b2#0", id="b2#0", ts="2026-06-11T09:00:00", type="REPORTED", position=0),
        N("BugEvent", "b2#1", id="b2#1", ts="2026-06-12T09:00:00", type="ASSIGNED", position=1),
        N("BugEvent", "b2#2", id="b2#2", ts="2026-06-25T09:00:00", type="RESOLVED", position=2),
        N("BugEvent", "b4#0", id="b4#0", ts="2026-06-13T09:00:00", type="REPORTED", position=0),
        N("Feature", "f1", id="f1", title="Bulk export", topic="bulk export", implemented=True,
          reverted=False, held=False, delayed_then_added=False, discussed_not_implemented=False),
        N("FeatureEvent", "f1#0", id="f1#0", ts="2026-06-14T09:00:00", type="PROPOSED"),
        N("Competitor", "Rival Corp", name="Rival Corp"),
        N("Observation", "o1", id="o1", text="Export is too slow", detail={"sev": 2, "tags": ["perf"]},
          pairs=[{"a": 1}, {"b": 2}], nothing=None),
        N("Observation", "o2", id="o2", text="Alice owns the export"),
    ]
    roots = [n.id for n in [*nodes, *base.nodes] if n.label in
             ("Bug", "Document", "Meeting", "Feature", "PR", "Competitor")]
    edges = [E("HAS", "Product:mini", t) for t in roots]
    edges += [
        E("HAS_EVENT", "Bug:b1", "BugEvent:b1#0"), E("HAS_EVENT", "Bug:b1", "BugEvent:b1#1"),
        E("HAS_EVENT", "Bug:b2", "BugEvent:b2#0"), E("HAS_EVENT", "Bug:b2", "BugEvent:b2#1"),
        E("HAS_EVENT", "Bug:b2", "BugEvent:b2#2"), E("HAS_EVENT", "Bug:b4", "BugEvent:b4#0"),
        E("HAS_EVENT", "Feature:f1", "FeatureEvent:f1#0"),
        E("ABOUT", "Observation:o1", "Feature:f1"), E("ABOUT", "Observation:o1", "Bug:b2"),
        E("ABOUT", "Observation:o2", "Person:eid_a1000001"),
        E("STATED", "Person:eid_b2000002", "Observation:o1"),
        E("CONTRIBUTED", "Person:eid_a1000001", "Feature:f1", "0", weight=1),
        E("CONTRIBUTED", "Person:eid_a1000001", "Feature:f1", "1", weight=2),
        E("REVERTS", prs[0], prs[1]),
    ]
    return base.merge(Facts(nodes, edges))
