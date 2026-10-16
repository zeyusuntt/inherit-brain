"""Indexes, constraints and the UDF library of a product graph. DDL exactly as measured in docs/falkor-versions.md."""

from __future__ import annotations

import time
from pathlib import Path

UDF_FILE = Path(__file__).parent / "udf" / "herb.js"
UDF_LIB = "herb_v1"
VECTOR_DIM = 1024
VECTOR_LABELS = ("Message", "Document")

# Exact-match indexes that back the constraints, created whatever labels the facts hold.
CONSTRAINT_INDEXES = (("Person", "eid"), ("PR", "link"), ("Bug", "id"))
CONSTRAINTS = (("UNIQUE", "Person", "eid"), ("UNIQUE", "Bug", "id"), ("UNIQUE", "PR", "link"),
               ("MANDATORY", "Bug", "status"))
FULLTEXT = {"Person": ("name",), "Customer": ("company",), "Bug": ("title",), "PR": ("title",),
            "Document": ("type",), "Feature": ("title",), "Competitor": ("name",)}


def _operational(rows) -> bool:
    return bool(rows) and all("OPERATIONAL" in [str(x) for x in r] for r in rows)


def wait_operational(graph, what: str, timeout: float = 60.0) -> None:
    """Block until every index (what='indexes') or constraint (what='constraints') is OPERATIONAL."""
    end = time.monotonic() + timeout
    while True:
        rows = graph.query(f"CALL db.{what}()").result_set
        if _operational(rows):
            return
        if time.monotonic() > end:
            raise TimeoutError(f"{what} not OPERATIONAL after {timeout}s: {rows}")
        time.sleep(0.1)


def create_exact_indexes(graph, labels) -> None:
    pairs = [(label, "id") for label in sorted(labels)] + list(CONSTRAINT_INDEXES)
    for label, prop in dict.fromkeys(pairs):
        graph.query(f"CREATE INDEX FOR (n:{label}) ON (n.{prop})")
    wait_operational(graph, "indexes")


def create_constraints(graph) -> None:
    for kind, label, prop in CONSTRAINTS:
        graph.client.connection.execute_command("GRAPH.CONSTRAINT", "CREATE", graph.name, kind, "NODE", label,
                                         "PROPERTIES", "1", prop)
    wait_operational(graph, "constraints")


def create_search_indexes(graph, labels, vector_labels) -> None:
    for label in sorted(labels):
        for prop in FULLTEXT.get(label, ()):
            graph.query(f"CREATE FULLTEXT INDEX FOR (n:{label}) ON (n.{prop})")
    for label in sorted(vector_labels):
        graph.query(f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.embedding) "
                    f"OPTIONS {{dimension:{VECTOR_DIM}, similarityFunction:'cosine'}}")
    wait_operational(graph, "indexes")


def load_udf(db, name: str = UDF_LIB, source: str | None = None) -> None:
    """Load the library once (the reply is a list or a dict depending on the client connection), never REPLACE. If it exists, its source must equal ours or this raises."""
    source = UDF_FILE.read_text(encoding="utf-8") if source is None else source
    found = db.udf_list(name, with_code=True)
    if not found:
        db.udf_load(name, source)
        return
    row = found[0] if isinstance(found[0], dict) else dict(zip(found[0][::2], found[0][1::2]))
    if row.get("library_code") != source:
        raise RuntimeError(f"UDF library {name} exists with different source; delete it deliberately and reload")
