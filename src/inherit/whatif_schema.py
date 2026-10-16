"""What the counterfactual (whatif.py) and its independent check (whatif_check.py) share, and nothing else.

* `WhatIf`: the three disjoint groups and `bound_hit`.
* `Snapshot`: an immutable export of the nodes and original typed edges of one product graph (read once with
  GRAPH.RO_QUERY): the work-item and person nodes with their stored properties, every edge between them with its
  properties and source records, and the raw messages (for the audit hook). No filter, owner set or projected
  edge is computed here; the two implementations derive those themselves.
* `MAX_HOPS`: the dependency bound (the Cypher side writes it as a literal, #3350).
"""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_HOPS = 4
SNAPSHOT_LABELS = ("Bug", "BugEvent", "PR", "Person", "Feature")


@dataclass(frozen=True)
class WhatIf:
    already_unowned: frozenset
    newly_unowned: frozenset
    affected: frozenset
    bound_hit: bool
    ops: tuple = field(default=(), compare=False)  # TraceOps of the graph run (empty for the Python check)
    copy_graph: str = field(default="", compare=False)  # the name of the copy that was made and deleted


@dataclass(frozen=True)
class SnapNode:
    id: str
    label: str
    props: dict


@dataclass(frozen=True)
class SnapEdge:
    type: str
    src: str
    dst: str
    props: dict
    records: tuple  # the edge's `src` record ids


@dataclass(frozen=True)
class Snapshot:
    nodes: tuple
    edges: tuple
    messages: tuple  # dicts: id, channel_id, ts, text, user_id


def person_node_id(person_id: str) -> str:
    return person_id if person_id.startswith("Person:") else f"Person:{person_id}"


def export_snapshot(conn, product: str) -> Snapshot:
    """Read the snapshot from herb:<product> over a read-only command (the whatif user may RO_QUERY it)."""
    name = f"herb:{product}"

    nodes = []
    for label in SNAPSHOT_LABELS:
        for nid, props in _rows(conn, name, f"MATCH (n:{label}) RETURN n.id, properties(n)"):
            nodes.append(SnapNode(nid, label, dict(props)))
    either = " OR ".join(f"{{v}}:{lb}" for lb in SNAPSHOT_LABELS)
    edges = []
    for typ, src, dst, props in _rows(
            conn, name, f"MATCH (a)-[e]->(b) WHERE ({either.format(v='a')}) AND ({either.format(v='b')}) "
                        "RETURN type(e), a.id, b.id, properties(e)"):
        props = dict(props)
        edges.append(SnapEdge(typ, src, dst, props, tuple(props.pop("src", None) or ())))
    msgs = [{"id": i, "channel_id": c, "ts": t, "text": x, "user_id": u}
            for i, c, t, x, u in _rows(conn, name, "MATCH (m:Message) RETURN m.source_id, m.channel_id, m.ts, m.text, "
                                                   "m.user_id")]
    return Snapshot(tuple(sorted(nodes, key=lambda n: n.id)),
                    tuple(sorted(edges, key=lambda e: (e.type, e.src, e.dst, str(e.props)))),
                    tuple(sorted(msgs, key=lambda m: m["id"])))


def graph_handle(conn, name: str):
    """A graph handle over a bare connection: the whatif and agent users hold no INFO right, so the FalkorDB
    client constructor cannot connect as them (same shim as tools.depth1.read_only_graph)."""
    from falkordb.graph import Graph

    client = type("Client", (), {})()
    client.connection, client.execute_command, client.flushdb = conn, conn.execute_command, conn.flushdb
    return Graph(client, name)


def _rows(conn, graph: str, text: str):
    return graph_handle(conn, graph).ro_query(text).result_set
