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
    "Observation": "id",  # a cited claim (competitor feature or weakness, customer issue or strength, QA issue); 4c
    "FeatureEvent": "id",  # dated feature history: discussed, implemented, held, resumed, reverted; 4c
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
    "SPOKE_IN": frozenset({("Person", "Meeting")}),
    "SHARED": frozenset({("Person", "Url"), ("Person", "Document"), ("Person", "PR")}),
    "COLLABORATED": frozenset({("Person", "Person")}),
    # Task 4 design (4a rules, 4b bugs, 4c semantic). One reason per signature.
    "HAS": frozenset({("Product", t) for t in ("Bug", "Document", "Meeting", "Feature", "PR", "Competitor")}),
    # HAS: the Product root of spec 6.1; the graph loader requires it to reach every node of these labels.
    "REVERTS": frozenset({("PR", "PR")}),  # revert PR -> original PR, same repository; props reason, applied (4a)
    "ABOUT": frozenset({
        ("PR", "Feature"),  # development association, independent of approval and merge state (4c)
        ("Url", "Competitor"), ("Url", "Product"),  # demo links: competitor demos (4c), own and previous demos (4a)
        ("Observation", "Competitor"), ("Observation", "Product"),  # competitor claims, release observations (4c)
        ("Observation", "Feature"), ("Observation", "Bug"),  # a customer claim about one feature or bug (4c)
        ("Observation", "Person"),  # a claim about a person; no extractor yet, the load fixture uses it
        ("Message", "PR"),  # a Slack decision about a PR: withhold, resume, request_merge, request_revert (4c)
    }),
    "STATED": frozenset({("Person", "Observation")}),  # who provided the insight: the Slack author or speaker (4c)
    "REPORTED_BY": frozenset({
        ("Bug", "Customer"),  # the affected customer company of a bug (4b)
        ("Observation", "Customer"),  # customer issue or strength (4c)
        ("Observation", "Person"),  # R9: QA reporter named by literal eid_ text, distinct from the poster (4a ids, 4c)
    }),
    "CONTRIBUTED": frozenset({("Person", "Feature")}),  # kind lead, assigned, design, support (4c)
    "SUGGESTED": frozenset({("Person", "Document")}),  # attributed review suggestion or question (4c)
    "HAS_EVENT": frozenset({("Bug", "BugEvent"), ("Feature", "FeatureEvent")}),  # event history, state derived (4b, 4c)
    "IMPLEMENTED_BY": frozenset({("Feature", "PR")}),  # only with merged PR or explicit completion evidence (4c)
    "DISCUSSES": frozenset({
        ("Meeting", "Feature"), ("Meeting", "Document"),  # meeting segments about a feature or document (4c)
        ("Message", "Feature"), ("Message", "Document"),  # Slack sessions about the same (4c)
    }),
    "ASSIGNED_TO": frozenset({("BugEvent", "Person")}),  # assignee of an ASSIGNED event; current assignee derived (4b)
    "ACTED_BY": frozenset({("BugEvent", "Person")}),  # actor = message author; resolver, dismisser by traversal (4b)
    "FIXES": frozenset({("PR", "Bug")}),  # a resolution message cites this PR; fix-PR author by traversal (4b)
    "SIMILAR_TO": frozenset({("ExternalBug", "Bug")}),  # open-source look-alike of a customer bug (4b)
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
