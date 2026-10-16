"""A small product graph with Task 4-shaped facts for the Task 12 tests. Not HERB data.

Expected groups when P leaves (default dependency rules), asserted in tests/test_whatif.py:
* newly unowned: bs, bnew, pr_orph, pr_orph2, pr_chain
* already unowned: bnone, pr_na
* affected: bfix (bug fixed by an orphaned PR), f_impl, f_dev, c1
* affected also: f_pend_missing, f_pend_false (not implemented, a development PR still open)
* nothing: f_pend_impl (implemented), f_pend_closed (its development PR is closed), f_nondev (ABOUT is not development),
  bsh (P and Q), bmulti (P, Rob, Zed), bold and btie (re-assigned to Q), bclosed, pr_sh, pr_cu (closed, unmerged), pr_m (merged),
  f_unrec (fate not reconciled), f_via_na (its PR was already unowned), f_alone, f_other (ABOUT but not development)
"""

from __future__ import annotations

import redis
from falkordb import FalkorDB

from conftest import fake_embed, fx_edge, fx_node
from inherit.config import load_settings
from inherit.facts import Edge, Facts
from inherit.graph import acl as acl_mod
from inherit.graph import load as loadmod

P = "wmini"
PROD = f"Product:{P}"
LEAVER, QUEEN, ROB, ZED = "eid_a0000001", "eid_b0000002", "eid_c0000003", "eid_d0000004"
SAM1, SAM2 = "eid_5a000005", "eid_5b000006"  # two employees named Sam Kim


def pr(n: int) -> str:
    return f"https://github.com/salesforce/{P}/pull/{n}"


def slug(n: int) -> str:
    """The record id of a PR's own record (the link without scheme, punctuation as underscores)."""
    return pr(n).split("://", 1)[1].replace("/", "_").replace(".", "_")


def _e(typ, src, dst, rec, key="", **props):
    return Edge(typ, src, dst, props, tuple(rec), "fixture/v1", key)


def whatif_facts(chain: int = 5, reverse_chain: bool = False) -> Facts:
    """`chain` features c1..cN: c1 is implemented by pr_chain, c(i+1) is linked to c(i) by a TEST_NEEDS edge
    (the two real dependency kinds reach one hop only; the test edge needs an injected rule)."""
    N = fx_node
    nodes = [N("Product", P, name=P)]
    edges: list[Edge] = []

    def has(i):
        edges.append(_e("HAS", PROD, i, ["root"]))

    for eid, name, role in ((LEAVER, "Pat Lee", "Engineering Lead"), (QUEEN, "Quinn Ray", "Engineering Lead"),
                            (ROB, "Rob Cole", "Software Engineer"), (ZED, "Zed Park", "Product Manager"),
                            (SAM1, "Sam Kim", "Software Engineer"), (SAM2, "Sam Kim", "Marketing Manager")):
        nodes.append(N("Person", eid, eid=eid, name=name, role=role, in_directory=True))

    # ---- bugs: key -> (active, title, [(ts, message_id, seq, type, [assignees])])
    R, A = "REPORTED", "ASSIGNED"
    bugs = {
        "bs": (True, "Solo bug", [("2026-06-01T09:00:00", "20260601-0-aaaa1", 0, R, []),
                                  ("2026-06-02T09:00:00", "20260602-0-aaaa2", 0, A, [LEAVER])]),
        "bsh": (True, "Shared bug", [("2026-06-01T09:00:00", "20260601-1-aaaa3", 0, A, [LEAVER, QUEEN])]),
        "bold": (True, "Reassigned away", [("2026-06-01T09:00:00", "20260601-2-aaaa4", 0, A, [LEAVER]),
                                           ("2026-06-05T09:00:00", "20260605-0-aaaa5", 0, A, [QUEEN])]),
        "bnew": (True, "Reassigned to leaver", [("2026-06-01T09:00:00", "20260601-3-aaaa6", 0, A, [QUEEN]),
                                                ("2026-06-05T09:00:00", "20260605-1-aaaa7", 0, A, [LEAVER])]),
        # same timestamp: record position 10 is after position 2 (a string compare would say the opposite)
        "btie": (True, "Tie on timestamp", [("2026-06-07T09:00:00", "20260607-2-aaaa8", 0, A, [LEAVER]),
                                            ("2026-06-07T09:00:00", "20260607-10-aaaa9", 0, A, [QUEEN])]),
        "bmulti": (True, "Three assignees", [("2026-06-08T09:00:00", "20260608-0-aaab1", 0, A, [LEAVER, ROB, ZED])]),
        "bnone": (True, "Nobody took it", [("2026-06-01T09:00:00", "20260601-4-aaaa0", 0, R, [])]),
        "bclosed": (False, "Closed bug", [("2026-06-01T09:00:00", "20260601-5-aaab2", 0, A, [LEAVER])]),
        "bfix": (True, "Bug fixed by an orphan PR", [("2026-06-01T09:00:00", "20260601-6-aaab3", 0, A, [QUEEN])]),
    }
    for key, (active, title, events) in bugs.items():
        bid = f"Bug:{key}"
        nodes.append(N("Bug", key, id=key, title=title, status="ASSIGNED" if active else "RESOLVED", active=active,
                       channel_id="ch1", aliases=[title.lower()]))
        has(bid)
        for i, (ts, mid, seq, typ, who) in enumerate(events):
            ev = f"{key}#{i}"
            nodes.append(N("BugEvent", ev, id=ev, ts=ts, message_id=mid, seq=seq, type=typ))
            edges.append(_e("HAS_EVENT", bid, f"BugEvent:{ev}", [mid]))
            for w in dict.fromkeys(who):
                edges.append(_e("ASSIGNED_TO", f"BugEvent:{ev}", f"Person:{w}", [mid]))

    # ---- PRs: n -> (state, merged, [authors])
    prs = {
        1: ("open", False, [LEAVER]),    # pr_orph
        2: ("open", False, [LEAVER, QUEEN]),  # pr_sh
        3: ("closed", False, [LEAVER]),  # pr_cu
        4: ("closed", True, [LEAVER]),   # pr_m
        5: ("open", False, []),          # pr_na
        6: ("open", False, [LEAVER]),    # pr_orph2
        7: ("open", False, [LEAVER]),    # pr_chain
    }
    for n, (state, merged, authors) in prs.items():
        link = pr(n)
        nodes.append(N("PR", link, link=link, title=f"PR {n}", summary=f"Summary of PR {n}", state=state, merged=merged,
                       open_unmerged=state == "open" and not merged))
        has(f"PR:{link}")
        edges.extend(_e("OPENED", f"Person:{a}", f"PR:{link}", [f"rec-pr{n}"]) for a in authors)
    edges.append(_e("FIXES", f"PR:{pr(1)}", "Bug:bfix", ["20260610-0-fff01"]))

    # ---- features: key -> dni (discussed_not_implemented)
    feats = {"f_impl": True, "f_dev": True, "f_unrec": None, "f_via_na": True, "f_alone": True, "f_other": True,
             "f_done": False}
    for k, dni in feats.items():
        nodes.append(N("Feature", k, id=k, title=k.replace("_", " "), discussed_not_implemented=dni))
        has(f"Feature:{k}")
    edges += [
        _e("IMPLEMENTED_BY", "Feature:f_impl", f"PR:{pr(1)}", [slug(1)], basis="merged"),
        _e("ABOUT", f"PR:{pr(6)}", "Feature:f_dev", [slug(6)], kind="development", basis="batch"),
        _e("IMPLEMENTED_BY", "Feature:f_unrec", f"PR:{pr(1)}", ["rec-impl-u"]),
        _e("IMPLEMENTED_BY", "Feature:f_via_na", f"PR:{pr(5)}", ["rec-impl5"]),
        _e("ABOUT", f"PR:{pr(1)}", "Feature:f_other", ["rec-about"], kind="mention"),
        _e("IMPLEMENTED_BY", "Feature:f_done", f"PR:{pr(4)}", ["rec-impl4"]),
    ]
    pend = {"f_pend_missing": {}, "f_pend_false": {"implemented": False}, "f_pend_impl": {"implemented": True},
            "f_pend_closed": {"implemented": False}, "f_nondev": {"implemented": False}}
    for k, extra in pend.items():
        nodes.append(N("Feature", k, id=k, title=k.replace("_", " "), discussed_not_implemented=False, **extra))
        has(f"Feature:{k}")
    edges += [
        _e("ABOUT", f"PR:{pr(1)}", "Feature:f_pend_missing", ["rec-pm"], kind="development"),
        _e("ABOUT", f"PR:{pr(6)}", "Feature:f_pend_false", ["rec-pf"], kind="development"),
        _e("ABOUT", f"PR:{pr(1)}", "Feature:f_pend_impl", ["rec-pi"], kind="development"),
        _e("ABOUT", f"PR:{pr(3)}", "Feature:f_pend_closed", ["rec-pc"], kind="development"),  # PR 3 is closed
        _e("IMPLEMENTED_BY", "Feature:f_pend_closed", f"PR:{pr(1)}", ["rec-pc2"]),  # depends on an orphan, not open
        _e("IMPLEMENTED_BY", "Feature:f_nondev", f"PR:{pr(1)}", ["rec-nd"]),
        _e("ABOUT", f"PR:{pr(1)}", "Feature:f_nondev", ["rec-nd2"], kind="mention"),
    ]
    for i in range(1, chain + 1):
        nodes.append(N("Feature", f"c{i}", id=f"c{i}", title=f"chain {i}", discussed_not_implemented=True))
        has(f"Feature:c{i}")
    if chain:
        edges.append(_e("IMPLEMENTED_BY", "Feature:c1", f"PR:{pr(7)}", ["rec-chain"]))
    for i in range(1, chain):
        a, b = (f"Feature:c{i + 1}", f"Feature:c{i}")
        edges.append(_e("TEST_NEEDS", *((b, a) if reverse_chain else (a, b)), [f"rec-need{i}"]))

    # ---- messages for the audit hook (channel ch1 holds the bug threads)
    msgs = [("20260602-0-aaaa2", "2026-06-02T09:00:00", "ch1", "@eid_a0000001 please take Solo bug"),
            ("20260603-0-m0001", "2026-06-03T10:00:00", "ch1", "Solo bug: Quinn will look at it instead"),
            ("20260603-1-m0002", "2026-06-03T11:00:00", "ch2", "unrelated chatter"),
            ("20260604-0-m0003", "2026-06-04T11:00:00", "ch2", "any news on the solo bug?"),
            ("20260502-0-m0004", "2026-05-02T11:00:00", "ch1", "an earlier remark about Solo bug"),
            ("20260609-0-p0001", "2026-06-09T09:00:00", "ch3", f"Review needed: {pr(1)}"),
            ("20260609-1-p0002", "2026-06-09T10:00:00", "ch3", "Quinn will take this one over"),
            ("20260609-2-p0003", "2026-06-09T11:00:00", "ch4", "nothing to do with it")]
    for mid, ts, ch, text in msgs:
        nodes.append(N("Message", mid, id=mid, ts=ts, channel_id=ch, text=text, user_id=ZED))
    return Facts(nodes, edges)


# ---------------------------------------------------------------- servers
def load(db, facts=None):
    loadmod.load_product(db, P, facts or whatif_facts(), embed=fake_embed)
    return db.select_graph(f"herb:{P}")


def whatif_conn(db, tmp_path, user: str = "whatif"):
    """A bare connection as whatif_<P> (or agent_<P>); the users hold no INFO right, so no FalkorDB client."""
    pw = acl_mod.setup_acl(db, [P], path=tmp_path / "acl.json", label="t")
    kw = db.connection.connection_pool.connection_kwargs
    return redis.Redis(host=kw["host"], port=kw["port"], username=f"{user}_{P}", password=pw[f"{user}_{P}"],
                       decode_responses=True)


def clean(db) -> None:
    for g in db.list_graphs():
        if g == f"herb:{P}" or g.startswith(f"whatif:{P}:"):
            db.select_graph(g).delete()
    for u in (f"agent_{P}", f"whatif_{P}", f"build_{P}"):
        db.connection.execute_command("ACL", "DELUSER", u)


def connect(ver):
    db = FalkorDB.from_url(load_settings().falkor_urls[ver])
    db.connection.ping()
    db.version_id = ver
    return db
