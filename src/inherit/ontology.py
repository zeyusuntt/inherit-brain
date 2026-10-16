"""Single source of truth for node labels, their domain keys and allowed edge signatures.

Node id = `<Label>:<domain key>`. Every edge produced anywhere must pass `check_edge`.
"""

from __future__ import annotations

# label -> property holding the domain key. Review has no label: a review is a REVIEWED edge
# keyed by PR plus position. Meeting transcript segments are citation strings, not nodes.
NODE_KEYS: dict[str, str] = {
    "Product": "name",
    "Person": "eid",
    "Customer": "id",
    "Channel": "id",
    "Message": "id",
    "Document": "id",
    "Meeting": "id",
    "PR": "link",
    "Url": "link",
    "Bug": "id",
    "BugEvent": "id",
    "Feature": "id",
    "Competitor": "name",
    "ExternalBug": "link",
}

# edge type -> allowed (src label, dst label) pairs. Types from spec 5.1 are added by the task that
# produces them. IN_CHANNEL is not in spec 5.1: it groups messages by channel (bug channels, cards).
# COLLABORATED is stored once per unordered pair with src_id < dst_id. REPLIED_TO is dropped (no thread data).
SIGNATURES: dict[str, frozenset[tuple[str, str]]] = {
    "AUTHORED": frozenset({("Person", "Document")}),
    "ATTENDED": frozenset({("Person", "Meeting")}),
    "POSTED": frozenset({("Person", "Message")}),
    "IN_CHANNEL": frozenset({("Message", "Channel")}),
    "MENTIONS": frozenset({("Message", "Person")}),
    "OPENED": frozenset({("Person", "PR")}),
    "REVIEWED": frozenset({("Person", "PR")}),
    "SHARED": frozenset({("Person", "Url"), ("Person", "Document"), ("Person", "PR")}),
    "COLLABORATED": frozenset({("Person", "Person")}),
}


def label_of(node_id: str) -> str:
    return node_id.split(":", 1)[0]


def check_edge(edge) -> None:
    """Raise ValueError unless the edge type exists and its endpoint labels (from the ids) are allowed."""
    allowed = SIGNATURES.get(edge.type)
    if allowed is None:
        raise ValueError(f"unknown edge type {edge.type!r}")
    sig = (label_of(edge.src_id), label_of(edge.dst_id))
    if sig not in allowed:
        raise ValueError(f"{edge.type} does not allow {sig[0]} -> {sig[1]}: {edge.src_id} -> {edge.dst_id}")
    if edge.type == "COLLABORATED" and not edge.src_id < edge.dst_id:
        raise ValueError(f"COLLABORATED must have src_id < dst_id: {edge.src_id} -> {edge.dst_id}")
