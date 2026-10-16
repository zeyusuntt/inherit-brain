"""Hand-checked dev sessions (message ids read in the bundle). Rules only: no model call is made here."""

from pathlib import Path

import pytest

import bugfx as fx
from inherit.data.bundle import load_bundle, load_metadata
from inherit.data.records import build_records
from inherit.extract import bug_rules, bugs
from inherit.extract.catalog import build_catalog
from inherit.extract.sessions import sessions

BUNDLE = Path(__file__).resolve().parents[1] / "data" / "bundle"
pytestmark = pytest.mark.skipif(not (BUNDLE / "products" / "LeadForce.json").exists(), reason="dev bundle not built")


@pytest.fixture(scope="module")
def dev():
    emp, cust = load_metadata(BUNDLE)
    cache = {}

    def get(product):
        if product not in cache:
            rec = build_records(load_bundle(BUNDLE, product), emp, cust, product)
            cache[product] = (rec, build_catalog(rec))
        return cache[product]

    return get


def parse_containing(dev, product, message_id):
    rec, cat = dev(product)
    chan = [m for m in rec.messages if m.bug_channel_owner]
    for s in sessions(chan):
        pass
    by_channel = {}
    for m in chan:
        by_channel.setdefault(m.channel_id, []).append(m)
    for msgs in by_channel.values():
        for s in sessions(msgs):
            if any(m.id == message_id for m in s):
                return bug_rules.parse_session(s, cat, rec.customers)
    raise KeyError(message_id)


def test_actiongenie_first_intake_is_fully_parsed_by_rules(dev):
    p = parse_containing(dev, "ActionGenie", "20261117-1-a77c3")
    assert [b["title"] for b in p.bugs] == ["Duplicate Log Entries for Manual Actions",
                                            "Inefficient Database Query Optimization",
                                            "Missing Log Entries for Automated Actions"]
    assert [b["customer_ids"] for b in p.bugs] == [["CUST-0047"], ["CUST-0108"], ["CUST-0064"]]
    assert [b["component"] for b in p.bugs] == ["Database Management & Logging System",
                                                "Backend Processing & Task Queue Management",
                                                "Database Management & Logging System"]
    assert [e["assignee_eids"] for e in p.events if e["type"] == "ASSIGNED"] == [
        ["eid_0ac476e4"], ["eid_4bcfb482"], ["eid_ae1d94d2"]]
    assert p.residual == []


def test_thanks_mention_does_not_become_an_assignee(dev):
    # AnomalyForce 20261026-3-7ca93: "Thanks, @eid_8986ddc3! Next, ... @eid_63ea0ec4, can you take this one?"
    p = parse_containing(dev, "AnomalyForce", "20261026-3-7ca93")
    [a] = [e for e in p.events if e["type"] == "ASSIGNED" and e["message_id"] == "20261026-3-7ca93"]
    assert a["assignee_eids"] == ["eid_63ea0ec4"]


def test_bold_titles_with_component_without_ampersand(dev):
    p = parse_containing(dev, "AnomalyForce", "20261014-1-aa399")
    assert [b["title"] for b in p.bugs] == ["Weak Encryption Protocols for Data Transmission",
                                            "Inefficient Query Execution Plans", "Data Preprocessing Inconsistencies"]
    assert [b["component"] for b in p.bugs] == ["API Security", "Database Performance", "Anomaly Detection Algorithm"]
    assert [b["customer_ids"] for b in p.bugs] == [["CUST-0041"], ["CUST-0097"], ["CUST-0091"]]
    assert p.residual == []


def test_assignment_in_a_follow_up_message(dev):
    # ContentForce 20260915: the lead names the bug, then assigns in the next message.
    p = parse_containing(dev, "ContentForce", "20260915-1-0dc6c")
    a = [(e["message_id"], e["assignee_eids"]) for e in p.events if e["type"] == "ASSIGNED"]
    assert a == [("20260915-2-44cf0", ["eid_07cc9100"]), ("20260915-5-48845", ["eid_07cc9100"]),
                 ("20260915-8-5b3e2", ["eid_c39dc658"])]


def test_dismissal_precedes_reopening_by_time_not_by_message_id(tmp_path, dev):
    rec, _ = dev("TrendForce")
    # 20260927-1-8d6b4 (dismissed, 09-27 14:12) has a later id than 20260927-0-8c21b (reopened, 09-28 00:22).
    out = bugs.run_bugs(rec, chat=fx.ScriptedChat(), cache_root=tmp_path)
    b = next(n for n in out.facts.nodes if n.label == "Bug" and n.props["title"] == "Session Fixation in Export Sessions")
    types = [t for t in b.props["event_types"] if t != "ASSIGNED"]
    assert types == ["REPORTED", "DISMISSED", "REOPENED"]
    assert b.props["reopened"] and b.props["no_fix_needed"] and b.props["active"] and b.props["unresolved"]
