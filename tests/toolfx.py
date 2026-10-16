"""A product graph with Task 4-shaped facts for the Task 8 tool tests. Not HERB data.

Traps built in (each one is asserted in tests/test_tools.py):
* company grouping uses Customer.company: b1, b2 and b6 come from three CUST ids of TechCorp or two of them;
* b2 is resolved, active and reopened at once (two fix PRs by one engineer); b5 is dismissed, reopened, then
  resolved after the reopening; b3 is dismissed and never reopened;
* PR17 both FIXES a bug and is ABOUT a feature; PR10 is ABOUT two features; PR20 is an external look-alike;
* PR14/16/23 are revert PRs whose targets are PR13/15/22; the revert 25 -> 24 was never applied;
* two employees named Sam Kim, and a Slack/meeting attendee with no review contribution;
* a current-release customer observation next to previous-release ones, and a QA reporter who is not the poster.
"""

from __future__ import annotations

import zlib

import pytest
from falkordb import FalkorDB

from conftest import fake_embed, fx_edge, fx_node
from inherit.config import load_settings
from inherit.facts import Edge, Facts
from inherit.graph import acl as acl_mod
from inherit.graph import load as loadmod
from inherit.tools.depth1 import read_only_graph

P = "toolmini"
PROD = f"Product:{P}"
ALICE, BOB, CARA, DAN, EVE, FINN = ("eid_a1000001", "eid_b2000002", "eid_c3000003", "eid_d4000004",
                                    "eid_e5000005", "eid_f6000006")
SAM1, SAM2, GUS = "eid_5a000007", "eid_5b000008", "eid_a9000009"
GHOST = "eid_e6000010"  # a placeholder person (no name, not in the directory) on a collaboration path


def pr(n: int) -> str:
    return f"https://github.com/salesforce/{P}/pull/{n}"


def _e(typ, src, dst, rec=None, key="", **props):
    rec = rec or [f"rec-{typ.lower()}-{zlib.crc32(f'{src}|{dst}|{key}'.encode()) % 10**6}"]
    return Edge(typ, src, dst, props, tuple(rec), "fixture/v1", key)


def tool_facts() -> Facts:
    N = fx_node
    nodes = [N("Product", P, name=P)]
    edges: list[Edge] = []

    def has(*ids):
        edges.extend(_e("HAS", PROD, i, ["root"]) for i in ids)

    # ---- people and customers
    people = [(ALICE, "Alice Ng", "Software Engineer"), (BOB, "Bob Roy", "Software Engineer"),
              (CARA, "Cara Diaz", "Engineering Lead"), (DAN, "Dan Lee", "QA Specialist"),
              (EVE, "Eve Park", "Product Manager"), (FINN, "Finn Ho", "Technical Architect"),
              (SAM1, "Sam Kim", "UX Researcher"), (SAM2, "Sam Kim", "Marketing Manager"),
              (GUS, "Gus Moe", "Marketing Research Analyst")]
    nodes += [N("Person", e, eid=e, name=n, role=r, in_directory=True) for e, n, r in people]
    for cid, comp in (("CUST-0001", "TechCorp"), ("CUST-0002", "TechCorp"), ("CUST-0003", "BlueWave"),
                      ("CUST-0004", "CloudSync")):
        nodes.append(N("Customer", cid, id=cid, company=comp, name=f"Contact {cid}", role="CTO"))

    # ---- bugs: (key, title, component, customer, [(type, ts, actor, assignees, fix_prs)], flags)
    A, R, D, O = "ASSIGNED", "RESOLVED", "DISMISSED", "REOPENED"
    bugs = [
        ("b1", "Login crash", "Authentication", "CUST-0001", [
            ("REPORTED", "2026-06-10T09:00:00", ALICE, [], []), (A, "2026-06-11T09:00:00", CARA, [ALICE], []),
            (R, "2026-06-20T09:00:00", ALICE, [], [pr(1), pr(17)])]),
        ("b2", "Slow export", "Backend Processing & Task Queue Management", "CUST-0002", [
            ("REPORTED", "2026-06-11T09:00:00", CARA, [], []), (A, "2026-06-12T09:00:00", CARA, [BOB], []),
            (R, "2026-06-20T10:00:00", BOB, [], [pr(2), pr(3)]), (O, "2026-07-01T09:00:00", CARA, [], []),
            (A, "2026-07-02T09:00:00", CARA, [ALICE], [])]),
        ("b3", "Avatar upload fails", "Authentication", "CUST-0003", [
            ("REPORTED", "2026-06-12T09:00:00", CARA, [], []), (A, "2026-06-13T09:00:00", CARA, [BOB], []),
            (D, "2026-06-15T09:00:00", CARA, [], [])]),
        ("b4", "Dark mode flicker", "User Interface", "CUST-0003", [
            ("REPORTED", "2026-06-13T09:00:00", CARA, [], []), (A, "2026-06-14T09:00:00", CARA, [ALICE], [])]),
        ("b5", "Report email bounce", "User Interface", "CUST-0004", [
            ("REPORTED", "2026-06-14T09:00:00", CARA, [], []), (A, "2026-06-15T09:00:00", CARA, [ALICE], []),
            (D, "2026-06-16T09:00:00", CARA, [], []), (O, "2026-06-30T09:00:00", CARA, [], []),
            (A, "2026-06-30T10:00:00", CARA, [ALICE], []), (R, "2026-07-05T09:00:00", ALICE, [], [pr(4)])]),
        ("b6", "Search lag", "Backend Processing & Task Queue Management", "CUST-0001", [
            ("REPORTED", "2026-06-15T09:00:00", CARA, [], []), (A, "2026-06-16T09:00:00", CARA, [ALICE], []),
            (R, "2026-06-25T09:00:00", ALICE, [], [pr(5)]), (O, "2026-07-10T09:00:00", CARA, [], [])]),
    ]
    for key, title, comp, cust, events in bugs:
        bid = f"Bug:{key}"
        types = [e[0] for e in events]
        fixes = sorted({p for e in events for p in e[4]})
        assignees = [e for e in events if e[0] == A][-1][3] if A in types else []
        company = {"CUST-0001": "TechCorp", "CUST-0002": "TechCorp", "CUST-0003": "BlueWave",
                   "CUST-0004": "CloudSync"}[cust]
        nodes.append(N("Bug", key, id=key, title=title, component=comp, impact_text=f"{title} hurts {company}.",
                       status=types[-1], unresolved=not fixes, resolved=bool(fixes),
                       active=types[-1] in ("REPORTED", A, O), reopened=O in types, no_fix_needed=D in types,
                       companies=[company], customer_ids=[cust], current_assignees=assignees,
                       event_types=types))
        has(bid)
        edges.append(_e("REPORTED_BY", bid, f"Customer:{cust}", [f"rec-{key}-0"]))
        for i, (typ, ts, actor, assigned, fix_links) in enumerate(events):
            eid = f"{key}#{i}"
            nodes.append(N("BugEvent", eid, id=eid, type=typ, ts=ts, seq=0, message_id=f"msg-{key}-{i}",
                           fix_pr_links=fix_links, assignee_eids=assigned))
            edges.append(_e("HAS_EVENT", bid, f"BugEvent:{eid}", [f"msg-{key}-{i}"]))
            edges.append(_e("ACTED_BY", f"BugEvent:{eid}", f"Person:{actor}", [f"msg-{key}-{i}"]))
            edges.extend(_e("ASSIGNED_TO", f"BugEvent:{eid}", f"Person:{a}", [f"msg-{key}-{i}"]) for a in assigned)
        for link in fixes:
            edges.append(_e("FIXES", f"PR:{link}", bid, [f"msg-{key}-fix"]))

    # ---- PRs: (n, title, author, approval, merged, open_unmerged, scope, pending_action, features, fixes)
    prs = [
        (1, "Fix login crash", ALICE, "approved", True, False, "product", None, [], "b1"),
        (2, "Speed up export", BOB, "approved", True, False, "product", None, [], "b2"),
        (3, "Speed up export again", BOB, "approved", True, False, "product", None, [], "b2"),
        (4, "Fix report bounce", ALICE, "approved", True, False, "product", None, [], "b5"),
        (5, "Fix search lag", ALICE, "approved", True, False, "product", None, [], "b6"),
        (10, "Predictive analytics model", ALICE, "approved", False, True, "product", "need_merge", ["fA", "fB"], None),
        (11, "Predictive analytics UI", ALICE, "approved", False, True, "product", "need_close", ["fA"], None),
        (12, "Predictive analytics draft", BOB, "unapproved", False, True, "product", None, ["fA"], None),
        (13, "Dark mode theme", BOB, "approved", True, False, "product", None, ["fB"], None),
        (14, "Revert dark mode theme", BOB, "approved", True, False, "product", None, [], None),
        (15, "Export v2 pipeline", ALICE, "approved", True, False, "product", None, ["fC"], None),
        (16, "Revert export v2 pipeline", CARA, "approved", True, False, "product", None, [], None),
        (17, "Dark mode login fix", BOB, "approved", True, False, "product", None, ["fB"], "b1"),
        (18, "Dark mode tweaks", CARA, "unapproved", False, False, "product", None, ["fB"], None),
        (20, "Predictive analytics (open source)", DAN, "approved", True, False, "external", None, ["fA"], None),
        (21, "SSO login", CARA, "approved", True, False, "product", None, ["fE"], None),
        (22, "Third-party scanner", DAN, "approved", True, False, "product", None, ["fF"], None),
        (23, "Revert third-party scanner", EVE, "approved", True, False, "product", None, [], None),
        (24, "Gamification", EVE, "approved", True, False, "product", None, ["fH"], None),
        (25, "Revert gamification", EVE, "approved", True, False, "product", None, [], None),
    ]
    for n, title, author, appr, merged, open_um, scope, pending, feats, _fix in prs:
        link = pr(n)
        props = dict(link=link, title=title, number=str(n), merged=merged, state="open" if open_um else "closed",
                     repo=f"salesforce/{P}", repo_scope=scope, approval=appr, open_unmerged=open_um,
                     created_at="2026-06-01T00:00:00")
        if pending:
            props["pending_action"] = pending
        nodes.append(N("PR", link, **props))
        has(f"PR:{link}")
        edges.append(_e("OPENED", f"Person:{author}", f"PR:{link}", [f"rec-pr{n}"]))
        edges.extend(_e("ABOUT", f"PR:{link}", f"Feature:{f}", [f"rec-pr{n}-{f}"], kind="development") for f in feats)
    # (revert pr, target pr, reason, applied)
    for rv, tgt, reason, applied in ((14, 13, "product_direction", True), (16, 15, "introduced_bugs", True),
                                      (23, 22, "security", True), (25, 24, "security", False)):
        edges.append(_e("REVERTS", f"PR:{pr(rv)}", f"PR:{pr(tgt)}", [f"rec-revert{rv}"], reason=reason,
                        applied=applied, reason_text=f"revert because {reason}"))

    # ---- features: key -> (title, aliases, details, discussed_not_implemented, [(kind, date, reason)], impl PRs).
    # discussed_not_implemented is None where the semantic layer's manifest is incomplete (fA, fG), as it writes it.
    # The fate properties are derived exactly as the semantic layer derives them (semantic.py, fate/v1).
    feats = {
        "fA": ("Predictive analytics", ["predictive analytics model"], ["Retrains models nightly"], None,
               [("discussed", "2026-05-01", None)], []),
        "fB": ("Dark mode", [], ["Theme switch in settings"], False,
               [("discussed", "2026-05-02", None), ("implemented", "2026-05-20", None),
                ("reverted", "2026-06-05", "product_direction")], [13]),
        "fC": ("Export v2", [], ["Streaming CSV export"], False,
               [("discussed", "2026-05-03", None), ("implemented", "2026-05-25", None),
                ("reverted", "2026-06-06", "introduced_bugs")], [15]),
        "fD": ("Voice assistant", [], ["Hands-free report queries"], True, [("discussed", "2026-05-04", None)], []),
        "fE": ("SSO login", ["single sign-on"], ["SAML sign-in"], False,
               [("held", "2026-05-01", "product_direction"), ("resumed", "2026-05-20", None),
                ("implemented", "2026-06-01", None)], [21]),
        "fF": ("Third-party scanner", [], ["Scans uploads"], False, [("implemented", "2026-05-10", None), ("reverted", "2026-06-07", "security")], [22]),
        "fG": ("Blockchain audit", [], ["Immutable audit log"], None, [("held", "2026-05-06", "product_direction")], []),
        # held, then resumed, never implemented: the semantic layer derives held and delayed_then_added for it
        "fZ": ("Offline mode", [], ["Works without a network"], None,
               [("held", "2026-05-02", "product_direction"), ("resumed", "2026-05-21", None)], []),
        "fH": ("Gamification", [], ["Badges for reports"], False, [("implemented", "2026-05-11", None)], [24]),
    }
    for k, (title, aliases, details, dni, events, impl) in feats.items():
        fid = f"Feature:{k}"
        kinds = [e[0] for e in events]
        held_dates = sorted(d for kd, d, _ in events if kd == "held")
        later = [d for kd, d, _ in events if held_dates and kd in ("resumed", "implemented") and d > held_dates[0]]
        nodes.append(N("Feature", k, id=k, title=title, topic=title.lower(), aliases=aliases, details=details,
                       product=P, implemented="implemented" in kinds or bool(impl), reverted="reverted" in kinds,
                       held="held" in kinds, resumed="resumed" in kinds, delayed_then_added=bool(held_dates and later),
                       discussed=True, discussed_not_implemented=dni, derived_by="fate/v1"))
        has(fid)
        for i, (kind, date, reason) in enumerate(events):
            ev = f"{k}:ev{i}"
            nodes.append(N("FeatureEvent", ev, id=ev, kind=kind, reason=reason, reason_text=reason, date=date,
                           record_id=f"rec-{k}-ev{i}"))
            edges.append(_e("HAS_EVENT", fid, f"FeatureEvent:{ev}", [f"rec-{k}-ev{i}"]))
        edges.extend(_e("IMPLEMENTED_BY", fid, f"PR:{pr(n)}", [f"rec-impl{n}"]) for n in impl)
    edges += [_e("CONTRIBUTED", f"Person:{DAN}", "Feature:fD", ["rec-c1"], kind="lead", ambiguous=False),
              _e("CONTRIBUTED", f"Person:{EVE}", "Feature:fD", ["rec-c2"], kind="support", ambiguous=False),
              _e("CONTRIBUTED", f"Person:{CARA}", "Feature:fE", ["rec-c3"], kind="lead", ambiguous=False)]

    # ---- competitor, observations, urls
    nodes.append(N("Competitor", "Rival Corp", name="Rival Corp", aliases=["Rival"]))
    has("Competitor:Rival Corp")
    obs = [  # key, kind, text, scope, about, stated_by, reported_by customer, reported_by person
        ("o1", "competitor_feature", "Rival has live dashboards", "unknown", "Competitor:Rival Corp", ALICE, None, None),
        ("o2", "competitor_weakness", "Rival is slow on export", "unknown", "Competitor:Rival Corp", BOB, None, None),
        ("o3", "customer_issue", "Reports failed to export", "previous", PROD, EVE, "CUST-0003", None),
        ("o4", "customer_strength", "Loved the dashboard", "previous", PROD, EVE, "CUST-0001", None),
        ("o5", "qa_issue", "Memory leak in sync", "previous", PROD, EVE, None, DAN),
        ("o6", "customer_issue", "Login is slow", "current", PROD, ALICE, "CUST-0004", None),
    ]
    for k, kind, text, scope, about, stated, cust, qa in obs:
        oid = f"Observation:{k}"
        nodes.append(N("Observation", k, id=k, kind=kind, text=text, release_scope=scope, subject="Rival Corp" if about.startswith("Competitor") else P, record_id=f"rec-{k}"))
        edges.append(_e("ABOUT", oid, about, [f"rec-{k}"]))
        edges.append(_e("STATED", f"Person:{stated}", oid, [f"rec-{k}"]))
        if cust:
            edges.append(_e("REPORTED_BY", oid, f"Customer:{cust}", [f"rec-{k}"]))
        if qa:
            edges.append(_e("REPORTED_BY", oid, f"Person:{qa}", [f"rec-{k}"]))
    urls = [("https://demo.example.com/rival", "demo", "unknown", "Competitor:Rival Corp", [ALICE, BOB]),
            ("https://demo.example.com/v1", "demo", "previous", PROD, [EVE]),
            ("https://demo.example.com/current", "demo", "unknown", PROD, [CARA]),
            ("https://docs.example.com/notes", None, "unknown", None, [DAN])]
    for link, kind, scope, about, sharers in urls:
        props = {"link": link, "release_scope": scope}
        if kind:
            props["kind"] = kind
        nodes.append(N("Url", link, **props))
        if about:
            edges.append(_e("ABOUT", f"Url:{link}", about, [f"rec-url-{link[-6:]}"], kind="demo"))
        edges.extend(_e("SHARED", f"Person:{p}", f"Url:{link}", [f"rec-share-{link[-6:]}-{p[-3:]}"]) for p in sharers)

    # ---- documents and meeting
    docs = [  # key, type, scope, author
        ("latest_mrr", "Market Research Report", "current", GUS),
        ("prox_mrr", "Market Research Report", "previous", FINN),
        ("latest_pvd", "Product Vision Document", "current", EVE),
        ("prox_pvd", "Product Vision Document", "previous", EVE),
        ("odd_mrr", "Market Research Report", "unknown", GUS),
    ]
    for k, typ, scope, author in docs:
        props = dict(id=k, type=typ, release_scope=scope, content=f"{typ} body", date="2026-04-01T00:00:00")
        if k == "latest_mrr":
            props["feedback_items"] = ["Make it shorter"]
        nodes.append(N("Document", k, **props))
        has(f"Document:{k}")
        edges.append(_e("AUTHORED", f"Person:{author}", f"Document:{k}", [f"rec-auth-{k}"]))
    sug = [(CARA, "Pricing", "Add pricing tiers", "suggestion", False, "rec-s1"),
           (SAM1, "Market", "Add churn data", "suggestion", True, "rec-s2"),
           (SAM2, "Market", "Add churn data", "suggestion", True, "rec-s2"),
           (DAN, "Scope", "What about the EU?", "review_question", False, "m1#3")]
    for who, section, text, kind, amb, rec in sug:
        edges.append(_e("SUGGESTED", f"Person:{who}", "Document:latest_mrr", [rec], key=rec, section=section,
                        text=text, kind=kind, ambiguous=amb))
    # a review question raised in Slack counts (Bob); Dan's question above came from a transcript line and does not
    edges.append(_e("SUGGESTED", f"Person:{BOB}", "Document:latest_mrr", ["20260401-0-abcde"], key="20260401-0-abcde",
                    section="Pricing", text="Is pricing final?", kind="review_question", ambiguous=False))
    edges.append(_e("SUGGESTED", f"Person:{ALICE}", "Document:latest_pvd", ["rec-s4"], key="rec-s4",
                    section="Goals", text="State the goals", kind="suggestion", ambiguous=False))
    nodes.append(N("Meeting", "m1", id="m1", document_type="product-planning", date="2026-05-01T10:00:00",
                   release_scope="unknown"))
    has("Meeting:m1")
    for who in (ALICE, BOB, SAM1, SAM2):
        edges.append(_e("ATTENDED", f"Person:{who}", "Meeting:m1", ["rec-m1"]))
    edges.append(_e("DISCUSSES", "Meeting:m1", "Document:latest_mrr", ["m1#3"]))
    for who, name, amb, n in ((ALICE, "Alice Ng", False, 1), (BOB, "Bob Roy", False, 1), (SAM1, "Sam Kim", True, 2),
                              (SAM2, "Sam Kim", True, 2)):
        edges.append(_e("SPOKE_IN", f"Person:{who}", "Meeting:m1", ["m1#1"], speaker_name=name, ambiguous=amb,
                        n_candidates=n, score=0))

    # ---- collaboration a1 -> b2 -> c3 -> {d4, e5 -> ghost -> f6} (stored once per pair, src < dst)
    nodes.append(N("Person", GHOST, eid=GHOST, in_directory=False))
    for a, b in ((ALICE, BOB), (BOB, CARA), (CARA, DAN), (CARA, EVE), (EVE, GHOST), (GHOST, FINN)):
        edges.append(_e("COLLABORATED", f"Person:{a}", f"Person:{b}", ["rec-collab"], weight=1,
                        counts={"co_attend": 1, "mention": 0, "pr_review": 0, "session": 0}))
    return Facts(nodes, edges)


# ---------------------------------------------------------------- servers
def load(db, facts=None):
    loadmod.load_product(db, P, facts or tool_facts(), embed=fake_embed)
    return db.select_graph(f"herb:{P}")


def agent_graph(db, tmp_path):
    pw = acl_mod.setup_acl(db, [P], path=tmp_path / "acl.json", label="t")[f"agent_{P}"]
    kw = db.connection.connection_pool.connection_kwargs
    return read_only_graph(kw["host"], kw["port"], f"agent_{P}", pw, f"herb:{P}")


def clean(db) -> None:
    for g in db.list_graphs():
        if g == f"herb:{P}" or g.startswith(f"whatif:{P}:"):
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
    db.version_id = ver
    return db


@pytest.fixture(params=["4.22", "6.0"])
def tsrv(request):
    db = _connect(request.param)
    clean(db)
    yield db
    clean(db)


@pytest.fixture
def tsrv_both():
    dbs = {v: _connect(v) for v in load_settings().falkor_urls}
    for db in dbs.values():
        clean(db)
    yield dbs
    for db in dbs.values():
        clean(db)


@pytest.fixture
def tg(tsrv, tmp_path):
    """The fixture product as the read-only agent user sees it, on the parametrised server."""
    load(tsrv)
    g = agent_graph(tsrv, tmp_path)
    g.server_version = tsrv.version_id
    return g
