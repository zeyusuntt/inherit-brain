"""Fact model: nodes and edges with provenance, merged by key.

Node key is its id. Edge key is (type, src_id, dst_id, key), where `key` is an optional discriminator
(for example a review position). Merge unions `src`, keeps the first value on a property conflict and counts it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Node:
    label: str
    id: str
    props: dict


@dataclass(frozen=True)
class Edge:
    type: str
    src_id: str
    dst_id: str
    props: dict
    src: tuple[str, ...]
    extractor: str
    key: str = ""

    @property
    def ekey(self) -> tuple[str, str, str, str]:
        return (self.type, self.src_id, self.dst_id, self.key)


def _merge_props(base: dict, other: dict) -> tuple[dict, int]:
    out, conflicts = dict(base), 0
    for k, v in other.items():
        if k not in out:
            out[k] = v
        elif out[k] != v:
            conflicts += 1
    return out, conflicts


@dataclass(frozen=True)
class Facts:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    conflicts: int = 0

    def merge(self, other: Facts) -> Facts:
        conflicts = self.conflicts + other.conflicts
        nodes: dict[str, Node] = {}
        for nd in [*self.nodes, *other.nodes]:
            if nd.id in nodes:
                props, c = _merge_props(nodes[nd.id].props, nd.props)
                conflicts += c
                nodes[nd.id] = Node(nd.label, nd.id, props)
            else:
                nodes[nd.id] = nd
        edges: dict[tuple, Edge] = {}
        for ed in [*self.edges, *other.edges]:
            if ed.ekey in edges:
                old = edges[ed.ekey]
                props, c = _merge_props(old.props, ed.props)
                conflicts += c
                edges[ed.ekey] = Edge(old.type, old.src_id, old.dst_id, props,
                                      tuple(sorted({*old.src, *ed.src})), old.extractor, old.key)
            else:
                edges[ed.ekey] = Edge(ed.type, ed.src_id, ed.dst_id, ed.props, tuple(sorted(set(ed.src))),
                                      ed.extractor, ed.key)
        return Facts(list(nodes.values()), list(edges.values()), conflicts)

    def to_jsonl(self, path) -> None:
        """Nodes sorted by id, then edges sorted by edge key, keys sorted: the file hashes the same every run."""
        lines = []
        for nd in sorted(self.nodes, key=lambda x: x.id):
            lines.append({"kind": "node", "label": nd.label, "id": nd.id, "props": nd.props})
        for ed in sorted(self.edges, key=lambda x: x.ekey):
            lines.append({"kind": "edge", "type": ed.type, "src_id": ed.src_id, "dst_id": ed.dst_id, "key": ed.key,
                          "props": ed.props, "src": sorted(ed.src), "extractor": ed.extractor})
        text = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in lines)
        Path(path).write_text(text, encoding="utf-8")

    @classmethod
    def from_jsonl(cls, path) -> Facts:
        nodes, edges = [], []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["kind"] == "node":
                nodes.append(Node(r["label"], r["id"], r["props"]))
            else:
                edges.append(Edge(r["type"], r["src_id"], r["dst_id"], r["props"], tuple(r["src"]),
                                  r["extractor"], r["key"]))
        return cls(nodes, edges)
