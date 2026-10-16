import copy
import json

import pytest

from conftest import FIXTURES
from inherit.data.bundle import load_metadata
from inherit.data.records import build_records
from inherit.extract.catalog import build_catalog, derive_aliases, doc_stem, repo_of


@pytest.fixture
def rec(mini_raw, mini_bundle):
    data = json.loads((FIXTURES / "mini_product.json").read_text())
    extra = json.loads((FIXTURES / "mini_rules_extra.json").read_text())
    for k, v in extra.items():
        data[k] = data[k] + v
    emp, cust = load_metadata(mini_bundle)
    return build_records(copy.deepcopy(data), emp, cust, "mini")


def test_doc_stem_parses_versioned_ids():
    assert doc_stem("prox_market_research_report_final") == "prox"
    assert doc_stem("latest_aix_product_vision_document") == "aix"
    assert doc_stem("new_daix_technical_specifications_document") == "daix"
    assert doc_stem("final_onforcex_system_design_document") == "onforcex"
    assert doc_stem("anomalyforce_product_requirements_document") == "anomalyforce"
    assert doc_stem("mini_design_spec") is None


def test_repo_of():
    assert repo_of("https://github.com/salesforce/AIX/pull/9") == ("salesforce", "AIX")
    assert repo_of("https://example.com/x") is None


def test_aliases_with_evidence(rec):
    al = derive_aliases(rec)
    assert set(al) == {"minipro", "oldmini"}
    assert any(e.startswith("channel_name:") for e in al["minipro"])
    assert "channel_creation:20260701-0-p0001" in al["minipro"]
    assert "document_id:latest_minipro_product_vision_document" in al["minipro"]
    assert "document_id:oldmini_product_vision_document_final" in al["oldmini"]
    assert any(e.startswith("url:") for e in al["oldmini"])


def test_catalog_documents_by_id_type_link(rec):
    cat = build_catalog(rec)
    assert cat.product == "mini" and cat.aliases == ("minipro", "oldmini")
    d = cat.doc("mini_design_spec")
    assert d.type == "Design Spec" and d.link.endswith("/mini_design_spec")
    assert cat.doc_by_link(d.link) is d
    assert cat.doc("https://sf-internal.slack.com/archives/docs/mini_design_spec") is d  # link accepted as ref
    assert cat.doc("nope") is None
    assert [x.id for x in cat.docs_of_type("Product Vision Document")] == [
        "oldmini_product_vision_document_final", "latest_minipro_product_vision_document",
        "new_minipro_product_vision_document"]
    assert cat.doc("oldmini_product_vision_document_final").stem == "oldmini"


def test_catalog_prs_by_link_and_exact_title_keep_duplicates_apart(rec):
    cat = build_catalog(rec)
    p = cat.pr("https://github.com/mini/mini-server/pull/101")
    assert p.number == "101" and p.repo_scope == "product"
    dups = cat.prs_by_title("Retry settings")
    assert [x.number for x in dups] == ["109", "110", "777"]
    assert [x.number for x in cat.prs_by_title("Retry settings", scope="product")] == ["109", "110"]
    assert [x.repo_scope for x in dups] == ["product", "product", "external"]
    assert cat.prs_by_title("retry settings") == []  # exact, not fuzzy
    assert cat.pr("https://github.com/x/y/pull/1") is None


def test_repo_scope_uses_bundle_evidence_not_just_the_org(rec):
    cat = build_catalog(rec)
    assert cat.product_repos == {"mini/mini-server"}
    assert cat.pr("https://github.com/tensorflow/tensorflow/pull/777").repo_scope == "external"
    assert "mini-server" in cat.product_repos or "mini/mini-server" in cat.product_repos


PAIRS = [("salesforce", "LeadForce"), ("salesforce", "AIX"), ("salesforce", "flink"), ("apache", "AIX"),
         ("apache", "kafka"), ("acme", "leadforce-api")]


def test_product_repo_set_is_explicit_exact_membership():
    from inherit.extract.catalog import product_repo_set, repo_scope
    repos = product_repo_set(PAIRS, "LeadForce", ("AIX", "ProX"))
    assert repos == {"salesforce/LeadForce", "salesforce/AIX"}
    link = lambda o, r: f"https://github.com/{o}/{r}/pull/9"  # noqa: E731
    assert repo_scope(link("salesforce", "AIX"), repos) == "product"
    for o, r in (("salesforce", "flink"), ("apache", "AIX"), ("apache", "kafka"), ("acme", "leadforce-api")):
        assert repo_scope(link(o, r), repos) == "external", (o, r)
    assert repo_scope("https://example.com/x", repos) == "external"


def test_product_repo_set_without_a_product_named_repo_accepts_alias_repos():
    from inherit.extract.catalog import product_repo_set
    assert product_repo_set([("salesforce", "AIX"), ("apache", "kafka")], "LeadForce", ("AIX",)) == {"salesforce/AIX"}


def test_feature_registry_canonical_ids_and_resolution(rec):
    cat = build_catalog(rec)
    fid = cat.register_feature("Real-Time  Alerts", aliases=["WebSockets endpoint"],
                               citations=[{"record_id": "m1", "quote": "alerts"}])
    assert fid == "Feature:mini:real time alerts"
    assert cat.resolve_feature("real-time alerts") == fid and cat.resolve_feature("websockets ENDPOINT") == fid
    assert cat.resolve_feature("unknown") is None
    # an alias that already resolves joins the same feature; citations and aliases are unioned
    assert cat.register_feature("Alerting", aliases=["Real time alerts"], citations=[{"record_id": "m2", "quote": "x"}]) == fid
    e = cat.features.entries[fid]
    assert len(e["citations"]) == 2 and "Alerting" in e["aliases"]
    assert cat.resolve_feature("alerting") == fid
    other = cat.register_feature("Email digest")
    assert other == "Feature:mini:email digest" and other != fid
    with pytest.raises(ValueError):
        cat.register_feature("  ")


def test_competitor_registry(rec):
    cat = build_catalog(rec)
    cid = cat.register_competitor("InsightGuard", citations=[{"record_id": "20260803-0-e89ae", "quote": "InsightGuard"}])
    assert cid == "Competitor:InsightGuard"
    assert cat.resolve_competitor("insight guard") == cid and cat.resolve_competitor("INSIGHTGUARD") == cid
    assert cat.register_competitor("Insight-Guard", citations=[{"record_id": "x", "quote": "y"}]) == cid
    assert len(cat.competitors.entries[cid]["citations"]) == 2
    assert cat.resolve_competitor("PatternPulse") is None
    assert build_catalog(rec).resolve_competitor("InsightGuard") is None  # registries are per catalogue


def test_with_previous_names(rec):
    cat = build_catalog(rec).with_previous(("oldmini",))
    assert cat.previous_names == ("oldmini",) and cat.aliases == ("minipro", "oldmini")
