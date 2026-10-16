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
    assert cat.pr("https://github.com/tensorflow/tensorflow/pull/777").repo_scope == "external"
    assert "mini-server" in cat.product_repos or "mini/mini-server" in cat.product_repos


def test_salesforce_org_repos_are_product_repos():
    from inherit.extract.catalog import repo_scope
    assert repo_scope("https://github.com/salesforce/AIX/pull/9", "LeadForce", ()) == "product"
    assert repo_scope("https://github.com/apache/kafka/pull/9", "LeadForce", ()) == "external"
    assert repo_scope("https://github.com/acme/leadforce-api/pull/9", "LeadForce", ()) == "external"
    assert repo_scope("https://github.com/acme/AIX/pull/9", "LeadForce", ("AIX",)) == "product"


def test_with_previous_names(rec):
    cat = build_catalog(rec).with_previous(("oldmini",))
    assert cat.previous_names == ("oldmini",) and cat.aliases == ("minipro", "oldmini")
