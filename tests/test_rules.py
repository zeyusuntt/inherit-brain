import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import FIXTURES
from inherit.data.bundle import load_metadata, make_bundle
from inherit.data.records import build_records
from inherit.extract import rules
from inherit.extract.catalog import build_catalog
from inherit.facts import Facts
from inherit.ontology import check_edge

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "bundle"
needs_dev = pytest.mark.skipif(not (BUNDLE / "products" / "LeadForce.json").exists(), reason="dev bundle not built")
DEV = ("ActionGenie", "AnomalyForce", "ContentForce", "KnowledgeForce", "LeadForce", "TrendForce")


def extended_data():
    data = json.loads((FIXTURES / "mini_product.json").read_text())
    for k, v in json.loads((FIXTURES / "mini_rules_extra.json").read_text()).items():
        data[k] = data[k] + v
    return data


@pytest.fixture(scope="module")
def mini_ext(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ext")
    raw = tmp / "raw"
    (raw / "products").mkdir(parents=True)
    (raw / "metadata").mkdir()
    (raw / "products" / "mini.json").write_text(json.dumps(extended_data()))
    for f in ("employee.json", "customers_data.json"):
        shutil.copy(FIXTURES / "mini_metadata" / f, raw / "metadata" / f)
    (raw / "metadata" / "salesforce_team.json").write_text("{}")
    make_bundle(raw, tmp / "bundle")
    return tmp / "bundle"


@pytest.fixture(scope="module")
def mini_rec(mini_ext):
    emp, cust = load_metadata(mini_ext)
    return build_records(copy.deepcopy(extended_data()), emp, cust, "mini")


@pytest.fixture(scope="module")
def out(mini_rec):
    return rules.run_rules(mini_rec)


def node(f, id_):
    return next((x for x in f.nodes if x.id == id_), None)


def props(f, id_):
    return node(f, id_).props


def edges(f, type_):
    return [e for e in f.edges if e.type == type_]


PR = "PR:https://github.com/mini/mini-server/pull/"


# ---------------------------------------------------------------- pure helpers

def rv(state, ts, comment="", pos=0, login="eid_a1000001"):
    from inherit.data.records import Review
    return Review(pos, state, login, comment, ts, True)


def test_approval_predicate_is_versioned():
    assert rules.APPROVAL_VERSION == "pr_approval/v1"


def test_approval_predicate_cases():
    f = rules.pr_approval
    assert f(()) == "unreviewed"
    assert f((rv("APPROVED", "2026-01-01T00:00:00"),)) == "approved"
    assert f((rv("APPROVED", "2026-01-01T00:00:00"), rv("PENDING", "2026-01-02T00:00:00", pos=1))) == "approved"
    assert f((rv("CHANGES_REQUESTED", "2026-01-01T00:00:00"), rv("COMMENTED", "2026-01-02T00:00:00", pos=1))) == "unapproved"
    assert f((rv("PENDING", "2026-01-01T00:00:00"),)) == "unapproved"
    # changes requested after the last approval -> unapproved; before it -> approved
    assert f((rv("APPROVED", "2026-01-01T00:00:00"), rv("CHANGES_REQUESTED", "2026-01-02T00:00:00", pos=1))) == "unapproved"
    assert f((rv("CHANGES_REQUESTED", "2026-01-01T00:00:00"), rv("APPROVED", "2026-01-02T00:00:00", pos=1))) == "approved"
    # order is by submitted_at, not list position
    assert f((rv("CHANGES_REQUESTED", "2026-01-02T00:00:00", pos=0), rv("APPROVED", "2026-01-01T00:00:00", pos=1))) == "unapproved"


@pytest.mark.parametrize("summary,reason", [
    ("This revert is necessary due to a change in product direction. Security notes were also discussed.", "product_direction"),
    ("This revert is necessary due to identified security risks associated with the changes in PR# 13.", "security"),
    ("This reverts PR# 13 due to critical bugs introduced in the application, including authentication failures.", "introduced_bugs"),
    ("The deployment has caused application crashes and significant performance degradation.", "introduced_bugs"),
    ("This revert is necessary due to critical issues introduced by PR# 35, which integrated a new AI model. The deployment has caused crashes.", "introduced_bugs"),
    ("This revert is necessary due to a change in product direction, which now focuses on REST APIs. Additionally, integration issues with third-party apps.", "product_direction"),
    ("Reverting to keep the history tidy.", "other"),
    ("", "other"),
])
def test_revert_reason_classifies_the_reason_clause(summary, reason):
    assert rules.classify_revert_reason(summary)[0] == reason


def test_revert_reason_ignores_later_sentences():
    r, clause = rules.classify_revert_reason("Cleanup of the history. This introduced security problems later.")
    assert r == "other" and clause == "Cleanup of the history."


def test_parse_revert_title_variants():
    p = rules.parse_revert_title
    assert p("Revert PR# 4: Address Security Risks") == 4
    assert p("Revert PR#2: x") == 2
    assert p("Revert PR #17: x") == 17
    assert p("revert pr# 9 foo") == 9
    assert p("Reverted the thing") is None
    assert p("Add revert button") is None


def test_named_qa_eids_templates():
    d = {"eid_4de5cd7f": {}, "eid_ddbe5577": {}}
    f = rules.named_qa_eids
    assert f("The QA specialist, eid_4de5cd7f, reported that x.", d) == (["eid_4de5cd7f"], [], [])
    assert f("The report came from QA Specialist eid_ddbe5577.", d)[0] == ["eid_ddbe5577"]
    assert f("QA team found issues reported by QA Specialist eid_4de5cd7f. Let's discuss.", d)[0] == ["eid_4de5cd7f"]
    valid, invalid, other = f("The report came from QA Specialist eid_99999999.", d)
    assert (valid, invalid) == ([], ["eid_99999999"])
    valid, invalid, other = f("Ask eid_4de5cd7f about the retry change.", d)
    assert (valid, invalid, other) == ([], [], ["eid_4de5cd7f"])
    assert f("Thanks @eid_4de5cd7f, the QA team agrees.", d) == ([], [], [])  # a mention is not a bare id
    assert f("QA specialist @eid_4de5cd7f said so", d) == ([], [], [])


# ---------------------------------------------------------------- mini layer

def test_every_edge_passes_check_and_carries_provenance(out):
    f, _ = out
    ids = {n.id for n in f.nodes}
    assert f.edges
    for e in f.edges:
        check_edge(e)
        assert e.src and e.extractor == rules.EXTRACTOR
    assert len(ids) == len(f.nodes)
    assert f.conflicts == 0


def test_product_root_aliases_and_previous_names(out):
    f, rep = out
    p = props(f, "Product:mini")
    assert p["name"] == "mini" and p["aliases"] == ["minipro", "oldmini"] and p["previous_names"] == ["oldmini"]
    assert "document_id:oldmini_product_vision_document_final" in p["alias_evidence"]["oldmini"]
    assert p["previous_evidence"]["oldmini"] and "20260702-0-p0002" in p["previous_evidence"]["oldmini"]
    assert "minipro" not in p["previous_evidence"]  # also on the unknown-scope new_minipro document
    assert rep["previous_names"] == ["oldmini"]


def test_has_edges_to_every_document_meeting_and_pr(out, mini_rec):
    f, _ = out
    has = {e.dst_id for e in edges(f, "HAS")}
    want = ({f"Document:{d.id}" for d in mini_rec.documents} | {f"Meeting:{m.id}" for m in mini_rec.meetings}
            | {f"PR:{p.link}" for p in mini_rec.prs})
    assert has == want
    assert {e.src_id for e in edges(f, "HAS")} == {"Product:mini"}


def test_document_scope_is_record_specific(out):
    f, rep = out
    prev = props(f, "Document:oldmini_product_vision_document_final")
    assert prev["release_scope"] == "previous" and prev["scope_basis"] == "anchor_link"
    assert prev["scope_evidence"] == ["20260702-0-p0002"] and prev["source_product_name"] == "oldmini"
    assert props(f, "Document:latest_minipro_product_vision_document")["release_scope"] == "previous"
    assert props(f, "Document:latest_minipro_product_vision_document")["source_product_name"] == "minipro"
    same = props(f, "Document:oldmini_product_requirements_document_final")
    assert same["release_scope"] == "previous" and same["scope_basis"] == "anchor_session"
    new = props(f, "Document:new_minipro_product_vision_document")
    assert new["release_scope"] == "unknown" and new["source_product_name"] == "minipro"  # alias never sets scope
    assert props(f, "Document:mini_design_spec")["release_scope"] == "unknown"
    assert rep["scope_counts"]["Document"] == {"previous": 3, "unknown": 3}
    assert props(f, "Meeting:mini_design_1")["release_scope"] == "unknown"


def test_message_scope_only_from_the_phrase(out):
    f, _ = out
    a = props(f, "Message:20260702-0-p0002")
    assert a["release_scope"] == "previous" and a["scope_basis"] == "phrase"
    qa = props(f, "Message:20260712-0-p0005")
    assert qa["release_scope"] == "previous" and qa["source_product_name"] == "minipro"
    assert qa["named_eids"] == ["eid_d4000004"]
    assert node(f, "Message:20260702-1-p0003") is None  # same session, no phrase: left to 4c
    assert node(f, "Message:20260710-0-p0004") is None


def test_named_eids_validated_against_directory(out):
    f, rep = out
    assert rep["named_qa_eids"] == [{"message": "20260712-0-p0005", "eid": "eid_d4000004"}]
    bad = node(f, "Message:20260713-1-p0007")
    assert bad is None or "named_eids" not in bad.props
    left = {(x["message"], x["eid"], x["why"]) for x in rep["bare_eids_left"]}
    assert ("20260713-1-p0007", "eid_99999999", "not_in_directory") in left
    assert ("20260713-0-p0006", "eid_b2000002", "not_qa_template") in left


def test_demo_links(out):
    f, rep = out
    old = props(f, "Url:https://sf-internal.slack.com/archives/oldmini/demo_1")
    assert old["kind"] == "demo" and old["subject"] == "Product:mini" and old["release_scope"] == "previous"
    assert old["source_product_name"] == "oldmini"
    cur = props(f, "Url:https://sf-internal.slack.com/archives/mini/demo_1")
    assert cur["kind"] == "demo" and cur["subject"] == "Product:mini" and cur["release_scope"] == "unknown"
    assert cur["source_product_name"] == "mini"
    rival = props(f, "Url:https://www.rivalco.com/demo")
    assert rival["kind"] == "demo" and "subject" not in rival and rival["release_scope"] == "unknown"
    about = {(e.src_id, e.dst_id) for e in edges(f, "ABOUT")}
    assert about == {("Url:https://sf-internal.slack.com/archives/oldmini/demo_1", "Product:mini"),
                     ("Url:https://sf-internal.slack.com/archives/mini/demo_1", "Product:mini")}
    assert rep["demos_without_subject"] == ["https://www.rivalco.com/demo"]
    assert node(f, "Url:https://docs.example.com/export") is None  # not a demo


def test_pr_props_repo_scope_and_approval(out):
    f, rep = out
    p = props(f, PR + "101")
    assert p["repo_scope"] == "product" and p["approval"] == "approved" and p["approval_rule"] == "pr_approval/v1"
    assert props(f, PR + "102")["approval"] == "unreviewed"
    assert props(f, PR + "103")["approval"] == "unapproved"
    assert props(f, PR + "109")["approval"] == "unapproved" and props(f, PR + "110")["approval"] == "approved"
    assert props(f, PR + "111")["approval"] == "unapproved" and props(f, PR + "112")["approval"] == "approved"
    assert props(f, "PR:https://github.com/tensorflow/tensorflow/pull/777")["repo_scope"] == "external"
    assert props(f, PR + "112")["open_unmerged"] is True and props(f, PR + "101")["open_unmerged"] is False
    assert rep["pr_counts"]["external"] == {"approved": 1}
    assert rep["pr_counts"]["product"] == {"approved": 7, "unapproved": 4, "unreviewed": 3}


def test_duplicate_title_prs_stay_separate_in_facts(out, mini_rec):
    f, _ = out
    a, b = props(f, PR + "109"), props(f, PR + "110")
    assert a["link"] != b["link"] and a["approval"] != b["approval"]
    cat = build_catalog(mini_rec)
    assert [x.number for x in cat.prs_by_title("Retry settings", scope="product")] == ["109", "110"]


def test_approval_contradictions_are_listed(out):
    _, rep = out
    got = {c["pr"].rsplit("/", 1)[1]: c for c in rep["approval_contradictions"]}
    assert set(got) == {"110", "113"}
    assert got["113"]["approval"] == "unapproved"
    assert got["113"]["reviews"][0]["why"] == "comment_says_approved_state_is_not"
    assert got["110"]["approval"] == "approved" and got["110"]["reviews"][0]["state"] == "PENDING"
    assert rep["approval_flips"] == ["https://github.com/mini/mini-server/pull/113"]


def test_reverts(out):
    f, rep = out
    byk = {(e.src_id[-3:], e.dst_id[-3:]): e for e in edges(f, "REVERTS")}
    assert set(byk) == {("105", "104"), ("107", "106")}
    e = byk[("105", "104")]
    assert e.props["reason"] == "product_direction" and e.props["applied"] is True
    assert e.props["reason_text"].startswith("This revert is necessary due to a change in product direction")
    e = byk[("107", "106")]
    assert e.props["reason"] == "introduced_bugs" and e.props["applied"] is False
    missing = [r for r in rep["reverts"] if r["original"] is None]
    assert len(missing) == 1 and missing[0]["revert"].endswith("/108") and missing[0]["target_number"] == 999
    assert rep["reverts_by_reason"] == {"product_direction": {"applied": 1, "not_applied": 0},
                                        "introduced_bugs": {"applied": 0, "not_applied": 1},
                                        "security": {"applied": 1, "not_applied": 0}}
    assert props(f, PR + "105")["reverts_number"] == 104


def test_output_is_deterministic_and_round_trips(mini_rec, tmp_path):
    f1, r1 = rules.run_rules(mini_rec)
    f2, r2 = rules.run_rules(mini_rec)
    f1.to_jsonl(tmp_path / "a.jsonl")
    f2.to_jsonl(tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert Facts.from_jsonl(tmp_path / "a.jsonl").nodes
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_merges_cleanly_with_structural_and_names(mini_rec):
    from inherit.extract.names import resolve_meeting_names
    from inherit.extract.structural import structural_facts_from_records
    f, _ = rules.run_rules(mini_rec)
    base = structural_facts_from_records(mini_rec).merge(resolve_meeting_names(mini_rec)[0])
    merged = base.merge(f)
    assert merged.conflicts == base.conflicts
    ids = {n.id for n in merged.nodes}
    for e in f.edges:
        assert e.src_id in ids and e.dst_id in ids


def test_runner_writes_rules_layer_and_report(tmp_path, mini_ext):
    root = tmp_path / "root"
    shutil.copytree(mini_ext, root / "data" / "bundle")
    (root / "src").mkdir()
    r = subprocess.run([sys.executable, "-m", "inherit.extract.run", "--layer", "rules", "--products", "mini",
                        "--repo-root", str(root)], capture_output=True, text=True, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    d = root / "data" / "facts" / "mini"
    assert (d / "rules.jsonl").exists() and (d / "rules_report.json").exists()
    rep = json.loads((d / "rules_report.json").read_text())
    assert rep["previous_names"] == ["oldmini"]
    assert "mini:" in r.stdout and "previous" in r.stdout
    log = next((root / "runs" / "access").glob("extract-*.jsonl")).read_text()
    assert '"allowed": false' not in log


# ---------------------------------------------------------------- dev records

@pytest.fixture(scope="module")
def dev():
    from inherit.data.bundle import load_bundle
    emp, cust = load_metadata(BUNDLE)
    res = {}
    for p in DEV:
        rec = build_records(load_bundle(BUNDLE, p), emp, cust, p)
        res[p] = (rec, *rules.run_rules(rec))
    return res


@needs_dev
def test_dev_previous_names_only_where_records_say_so(dev):
    prev = {p: dev[p][2]["previous_names"] for p in DEV}
    assert prev["LeadForce"] == ["ProX"] and prev["TrendForce"] == ["dProX"]
    for p in ("ActionGenie", "AnomalyForce", "ContentForce", "KnowledgeForce"):
        assert prev[p] == []
        assert dev[p][2]["scope_counts"]["Document"].get("previous", 0) == 0


@needs_dev
def test_dev_aliases(dev):
    a = lambda p: set(dev[p][2]["aliases"])  # noqa: E731
    assert {"onForceX"} <= a("ActionGenie") and {"alyAIX"} <= a("AnomalyForce")
    assert {"entAIX"} <= a("ContentForce") and {"ledgeAIX"} <= a("KnowledgeForce")
    assert {"AIX", "ProX"} <= a("LeadForce") and {"dAIX", "dProX"} <= a("TrendForce")
    assert "LeadForce" not in a("LeadForce")


@needs_dev
def test_dev_leadforce_scope_1281f_and_c2dfc(dev):
    f = dev["LeadForce"][1]
    assert props(f, "Message:20260325-0-1281f")["release_scope"] == "previous"
    for d in ("prox_market_research_report_final", "latest_aix_market_research_report",
              "prox_system_design_document_final", "latest_aix_system_design_document"):
        assert props(f, f"Document:{d}")["release_scope"] == "previous", d
    for d in ("new_aix_market_research_report", "new_aix_system_design_document"):
        assert props(f, f"Document:{d}")["release_scope"] == "unknown", d
    c2 = node(f, "Message:20260412-0-c2dfc")
    assert c2 is None or c2.props.get("release_scope") != "previous"
    assert props(f, "Document:latest_aix_market_research_report")["source_product_name"] == "AIX"
    assert props(f, "Document:prox_market_research_report_final")["source_product_name"] == "ProX"


@needs_dev
def test_dev_leadforce_demos(dev):
    f = dev["LeadForce"][1]
    for n in (1, 2, 3, 4):
        u = props(f, f"Url:https://sf-internal.slack.com/archives/ProX/demo_{n}")
        assert u["kind"] == "demo" and u["release_scope"] == "previous" and u["subject"] == "Product:LeadForce"
    cur = props(f, "Url:https://sf-internal.slack.com/archives/LeadForce/demo_1")
    assert cur["kind"] == "demo" and cur["release_scope"] == "unknown"


@needs_dev
def test_dev_competitor_demo_has_no_subject_in_4a(dev):
    f = dev["AnomalyForce"][1]
    u = props(f, "Url:https://www.insightguard.com/demo")
    assert u["kind"] == "demo" and "subject" not in u
    assert "https://www.insightguard.com/demo" in dev["AnomalyForce"][2]["demos_without_subject"]


@needs_dev
def test_dev_named_qa_eids(dev):
    assert props(dev["LeadForce"][1], "Message:20260327-0-bc38c")["named_eids"] == ["eid_4de5cd7f"]
    assert props(dev["LeadForce"][1], "Message:20260418-0-f333a")["named_eids"] == ["eid_ddbe5577"]
    assert props(dev["TrendForce"][1], "Message:20260404-0-349cb")["named_eids"] == ["eid_4f731d34"]
    assert props(dev["TrendForce"][1], "Message:20260314-0-10e98")["named_eids"] == ["eid_f944b0ee"]
    assert all(not dev[p][2]["bare_eids_left"] for p in DEV)


@needs_dev
def test_dev_approval_contradictions_never_flip_a_state(dev):
    # 100 dev PRs carry a PENDING or COMMENTED review whose text says "approved"; every one also has an APPROVED review
    assert sum(len(dev[p][2]["approval_contradictions"]) for p in DEV) == 100
    assert all(not dev[p][2]["approval_flips"] for p in DEV)
    f = dev["ActionGenie"][1]
    for n in (8, 24):  # APPROVED, then CHANGES_REQUESTED: unapproved
        assert props(f, f"PR:https://github.com/salesforce/onForceX/pull/{n}")["approval"] == "unapproved"


@needs_dev
def test_dev_revert_reasons(dev):
    def rev(p, repo, n):
        f = dev[p][1]
        return next(e for e in edges(f, "REVERTS") if e.src_id.endswith(f"/{repo}/pull/{n}"))
    assert rev("ContentForce", "ContentForce", 22).props["reason"] == "product_direction"  # title says Enhance Security
    assert rev("LeadForce", "AIX", 36).props["reason"] == "introduced_bugs"
    assert rev("LeadForce", "AIX", 36).dst_id.endswith("/AIX/pull/35") and rev("LeadForce", "AIX", 36).props["applied"] is True
    assert rev("LeadForce", "AIX", 9).props["reason"] == "product_direction"
    assert rev("LeadForce", "AIX", 15).props["reason"] == "introduced_bugs"
    assert rev("LeadForce", "AIX", 20).props["reason"] == "introduced_bugs"
    assert rev("ActionGenie", "onForceX", 7).props["reason"] == "security"
    assert rev("ActionGenie", "onForceX", 19).props["reason"] == "product_direction"
    assert sum(len(edges(dev[p][1], "REVERTS")) for p in DEV) == 22
    for p in DEV:
        assert not dev[p][2]["revert_titles_unparsed"] and all(r["original"] for r in dev[p][2]["reverts"])


@needs_dev
def test_dev_anomalyforce_duplicate_title_prs(dev):
    rec, f, rep = dev["AnomalyForce"]
    base = "PR:https://github.com/salesforce/AnomalyForce/pull/"
    a, b = props(f, base + "2"), props(f, base + "5")
    assert (a["approval"], b["approval"]) == ("unapproved", "approved")
    cat = build_catalog(rec)
    title = next(p.title for p in rec.prs if p.number == "2" and "salesforce" in p.link)
    got = cat.prs_by_title(title, scope="product")
    assert [x.number for x in got] == ["2", "5"]
    assert [x.author for x in got][0] != [x.author for x in got][1]


@needs_dev
def test_dev_pr_repo_scope_counts(dev):
    c = {p: dev[p][2]["pr_counts"] for p in DEV}
    assert sum(c["AnomalyForce"]["product"].values()) == 37 and sum(c["AnomalyForce"]["external"].values()) == 66
    assert sum(c["LeadForce"]["product"].values()) == 36 + 19
    assert sum(c["ActionGenie"]["product"].values()) == 26 + 22
    # the external flink Avro PR is never a product PR
    f = dev["AnomalyForce"][1]
    assert props(f, "PR:https://github.com/apache/flink/pull/2407")["repo_scope"] == "external"
