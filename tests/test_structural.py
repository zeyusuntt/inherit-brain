import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from inherit.data.bundle import load_bundle, load_metadata
from inherit.extract.structural import structural_facts
from inherit.ontology import check_edge

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def facts(mini_bundle):
    emp, cust = load_metadata(mini_bundle)
    return structural_facts(load_bundle(mini_bundle, "mini"), emp, cust, "mini")


def edges(f, type_):
    return [e for e in f.edges if e.type == type_]


def pairs(f, type_):
    return {(e.src_id, e.dst_id) for e in edges(f, type_)}


def node(f, id_):
    return next(x for x in f.nodes if x.id == id_)


def test_node_ids_follow_label_key(facts):
    key_prop = {"Person": "eid", "Customer": "id", "Channel": "id", "Message": "id", "Document": "id",
                "Meeting": "id", "PR": "link", "Url": "link"}
    assert {x.label for x in facts.nodes} == set(key_prop)
    for x in facts.nodes:
        assert x.id == f"{x.label}:{x.props[key_prop[x.label]]}"
    assert len({x.id for x in facts.nodes}) == len(facts.nodes)


def test_every_edge_passes_check_and_has_src_and_extractor(facts):
    ids = {x.id for x in facts.nodes}
    for e in facts.edges:
        check_edge(e)
        assert e.src_id in ids and e.dst_id in ids
        assert e.src and e.extractor == "structural/v1"


def test_authored(facts):
    assert pairs(facts, "AUTHORED") == {
        ("Person:eid_e5000005", "Document:mini_design_spec"),
        ("Person:eid_a1000001", "Document:mini_release_notes"),
    }
    assert edges(facts, "AUTHORED")[0].src == ("mini_design_spec",) or edges(facts, "AUTHORED")[0].src == ("mini_release_notes",)


def test_attended(facts):
    got = pairs(facts, "ATTENDED")
    assert len(got) == 6
    assert ("Person:eid_d4000004", "Meeting:mini_design_1") in got
    assert ("Person:eid_a1000001", "Meeting:mini_review_1") in got
    assert all(e.src == (e.dst_id.split(":", 1)[1],) for e in edges(facts, "ATTENDED"))


def test_posted_skips_bots_and_bots_are_not_people(facts):
    posted = edges(facts, "POSTED")
    assert len(posted) == 18 - 2
    assert not any("slack_admin_bot" in x.id for x in facts.nodes if x.label == "Person")
    # slack_deploy_bot is not a known bot and not in the directory: a non-directory person
    assert node(facts, "Person:slack_deploy_bot").props["in_directory"] is False
    assert ("Person:eid_e5000005", "Message:20260610-0-a0002") in pairs(facts, "POSTED")
    assert node(facts, "Message:20260609-0-a0001").props["kind"] == "slack"


def test_in_channel_and_channel_props(facts):
    assert len(edges(facts, "IN_CHANNEL")) == 18
    assert ("Message:20260617-0-c0001", "Channel:ch-mini-bug") in pairs(facts, "IN_CHANNEL")
    assert node(facts, "Channel:ch-mini-bug").props["bug_owner"] == "alicechen"
    assert node(facts, "Channel:ch-mini-plan").props["bug_owner"] is None


def test_mentions(facts):
    got = pairs(facts, "MENTIONS")
    assert ("Message:20260610-0-a0002", "Person:eid_a1000001") in got
    assert ("Message:20260610-0-a0002", "Person:eid_f6000006") in got
    assert ("Message:20260609-0-a0001", "Person:eid_e5000005") in got


def test_pr_opened_reviewed_and_not_in_directory(facts):
    assert ("Person:eid_b2000002", "PR:https://github.com/mini/mini-server/pull/101") in pairs(facts, "OPENED")
    p = node(facts, "Person:EMP_555000111")
    assert p.props["in_directory"] is False and p.props["name"] is None
    assert node(facts, "Person:eid_a1000001").props["in_directory"] is True
    rev = {(e.src_id, e.dst_id, e.key): e for e in edges(facts, "REVIEWED")}
    assert len(rev) == 3
    e = rev[("Person:eid_a1000001", "PR:https://github.com/mini/mini-server/pull/101", "0")]
    assert e.props["state"] == "APPROVED" and e.props["comment"] == "Looks good."
    assert e.src == ("github_com_mini_mini-server_pull_101",)
    assert ("Person:EMP_555000222", "PR:https://github.com/mini/mini-server/pull/103", "0") in rev


def test_pr_merged_parsed_not_bool_of_string(facts):
    base = "PR:https://github.com/mini/mini-server/pull/"
    assert node(facts, base + "101").props["merged"] is True
    assert node(facts, base + "102").props["merged"] is False
    assert node(facts, base + "102").props["merged_raw"] == "False"
    assert node(facts, base + "102").props["state"] == "closed"


def test_two_reviews_by_same_person_give_two_edges(mini_raw, tmp_path):
    d = json.loads((mini_raw / "products" / "mini.json").read_text())
    d["prs"][0]["reviews"].append(dict(d["prs"][0]["reviews"][0], comment="again", submitted_at="2026-06-17T09:00:00"))
    emp, cust = load_metadata(mini_raw)
    f = structural_facts(d, emp, cust, "mini")
    mine = [e for e in edges(f, "REVIEWED")
            if e.src_id == "Person:eid_a1000001" and e.dst_id.endswith("pull/101")]
    assert sorted(e.key for e in mine) == ["0", "2"]


def test_duplicate_url_ids_merge(facts):
    u = node(facts, "Url:https://docs.example.com/export")
    assert u.props["ids"] == ["docs_example_com_export"]
    assert u.props["descriptions"] == ["Guide to streaming exports.", "Vendor reference for chunked CSV export."]
    assert sum(1 for x in facts.nodes if x.label == "Url") == 2


def test_shared(facts):
    got = {(e.src_id, e.dst_id): e for e in edges(facts, "SHARED")}
    assert got[("Person:eid_e5000005", "Document:mini_design_spec")].src == ("20260610-0-a0002",)
    assert ("Person:eid_d4000004", "Document:mini_design_spec") in got
    assert ("Person:eid_a1000001", "PR:https://github.com/mini/mini-server/pull/102") in got
    # pull/10 style prefix must not match; the tensorflow link is not a known record
    assert not any("tensorflow" in k[1] for k in got)
    assert len(edges(facts, "SHARED")) == 6


def test_customers(facts):
    c = node(facts, "Customer:CUST-0001")
    assert c.props["company"] == "TechCorp" and c.props["name"] == "Tom Hill" and c.props["role"] == "CTO"
    assert {x.id for x in facts.nodes if x.label == "Customer"} == {"Customer:CUST-0001", "Customer:CUST-0002"}


def test_collaborated(facts):
    col = {(e.src_id, e.dst_id): e for e in edges(facts, "COLLABORATED")}
    assert all(a < b for a, b in col)
    e = col[("Person:eid_a1000001", "Person:eid_e5000005")]
    assert e.props["counts"] == {"co_attend": 1, "pr_review": 0, "mention": 2, "session": 1}
    assert e.props["weight"] == 4
    assert set(e.src) >= {"mini_review_1", "20260610-0-a0002", "20260611-0-a0003"}
    assert col[("Person:eid_c3000003", "Person:eid_d4000004")].props["weight"] == 2
    ab = col[("Person:eid_a1000001", "Person:eid_b2000002")]
    assert ab.props["counts"] == {"co_attend": 0, "pr_review": 1, "mention": 2, "session": 0}
    pr = col[("Person:EMP_555000111", "Person:EMP_555000222")]
    assert pr.props["counts"]["pr_review"] == 1
    assert not any("slack_admin_bot" in a or "slack_admin_bot" in b for a, b in col)
    assert col[("Person:eid_a1000001", "Person:slack_deploy_bot")].props["counts"]["session"] == 1


def test_unknown_mentioned_and_slack_users_become_non_directory_people(mini_raw):
    d = json.loads((mini_raw / "products" / "mini.json").read_text())
    s = json.loads(json.dumps(d["slack"][1]))
    s["Message"]["User"].update(userId="eid_zzzzzzzz", utterranceID="u-new", text="hi @eid_yyyyyyyy")
    s["id"] = "u-new"
    d["slack"].append(s)
    emp, cust = load_metadata(mini_raw)
    f = structural_facts(d, emp, cust, "mini")
    assert node(f, "Person:eid_zzzzzzzz").props["in_directory"] is False
    assert node(f, "Person:eid_yyyyyyyy").props["in_directory"] is False
    assert ("Person:eid_zzzzzzzz", "Message:u-new") in pairs(f, "POSTED")


def _runner_root(tmp_path, mini_bundle):
    import shutil

    root = tmp_path / "root"
    shutil.copytree(mini_bundle, root / "data" / "bundle")
    (root / "src").mkdir()
    return root


def test_runner_writes_facts_and_access_log(tmp_path, mini_bundle):
    root = _runner_root(tmp_path, mini_bundle)
    r = subprocess.run([sys.executable, "-m", "inherit.extract.run", "--layer", "structural", "--products", "mini",
                        "--repo-root", str(root)], capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    out = root / "data" / "facts" / "mini" / "structural.jsonl"
    assert out.exists() and out.read_text().count("\n") > 50
    assert "Person" in r.stdout and "COLLABORATED" in r.stdout
    logs = list((root / "runs" / "access").glob("extract-*.jsonl"))
    assert len(logs) == 1
    lines = [json.loads(x) for x in logs[0].read_text().splitlines()]
    assert any(x["path"].endswith("products/mini.json") and x["allowed"] for x in lines)


def test_runner_refuses_raw(tmp_path, mini_bundle):
    root = _runner_root(tmp_path, mini_bundle)
    (root / "data" / "raw").mkdir()
    (root / "data" / "raw" / "products").mkdir()
    (root / "data" / "raw" / "products" / "mini.json").write_text("{}")
    r = subprocess.run([sys.executable, "-m", "inherit.extract.run", "--layer", "structural", "--products", "mini",
                        "--repo-root", str(root), "--bundle", str(root / "data" / "raw")],
                       capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode != 0 and "ioguard" in r.stderr
    assert not (root / "data" / "facts" / "mini").exists()
