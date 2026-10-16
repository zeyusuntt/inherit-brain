from inherit.data.records import build_records
from inherit.extract.cite import (
    Citation, SourceRecord, normalise, source_records, validate_citation, validate_items,
)

MSG = SourceRecord("m1", "message", "text", "Hi team, “QA team” found   issues in the LAST release.")
DOC = SourceRecord("d1", "document", "content", "Export: stream rows in 10k chunks.")
DOCFB = SourceRecord("d1", "document", "feedback", "- Add numbers for the export section.")
URL0 = SourceRecord("u1", "url", "description", "A demo video of ProX.", position=0)
URL1 = SourceRecord("u1", "url", "description", "A demo video of LeadForce.", position=1)
ALLOWED = {("message", "text"), ("document", "content"), ("url", "description")}


def test_normalise_straightens_quotes_collapses_space_and_casefolds():
    assert normalise("It’s  a “Test”\n\tHERE") == "it's a \"test\" here"
    assert normalise("‘x’ „y‟") == "'x' \"y\""


def test_valid_quote_is_accepted_with_span():
    r = validate_citation({"record_id": "m1", "quote": 'qa TEAM" found issues'}, [MSG], ALLOWED)
    assert r.ok and r.reason == "ok"
    # the supporting span is a slice of the normalised record text
    assert r.span is not None and normalise(MSG.text)[r.span[0]:r.span[1]] == normalise('qa TEAM" found issues')
    assert r.position is None


def test_empty_and_whitespace_quotes_rejected():
    for q in ("", "   ", "\n\t", None):
        r = validate_citation({"record_id": "m1", "quote": q}, [MSG], ALLOWED)
        assert not r.ok and r.reason == "empty_quote"


def test_record_must_be_supplied_to_the_call():
    r = validate_citation({"record_id": "m2", "quote": "hi"}, [MSG], ALLOWED)
    assert not r.ok and r.reason == "record_not_supplied"


def test_type_and_field_must_be_allowed():
    r = validate_citation({"record_id": "d1", "quote": "numbers", "field": "feedback"}, [DOC, DOCFB], ALLOWED)
    assert not r.ok and r.reason == "field_not_allowed"
    r2 = validate_citation({"record_id": "d1", "quote": "numbers"}, [DOCFB], ALLOWED)
    assert not r2.ok and r2.reason == "field_not_allowed"
    ok = validate_citation({"record_id": "d1", "quote": "stream rows", "field": "content"}, [DOC, DOCFB], ALLOWED)
    assert ok.ok
    pr = SourceRecord("p1", "pr", "summary", "Adds retry.")
    assert validate_citation({"record_id": "p1", "quote": "retry"}, [pr], ALLOWED).reason == "field_not_allowed"


def test_quote_must_be_substring():
    r = validate_citation({"record_id": "m1", "quote": "totally absent"}, [MSG], ALLOWED)
    assert not r.ok and r.reason == "quote_not_found"


def test_url_descriptions_are_a_valid_source_and_repeated_ids_use_position():
    r = validate_citation({"record_id": "u1", "quote": "demo video of leadforce"}, [URL0, URL1], ALLOWED)
    assert r.ok and r.position == 1
    pinned = validate_citation({"record_id": "u1", "quote": "demo video of prox", "position": 0}, [URL0, URL1], ALLOWED)
    assert pinned.ok and pinned.position == 0
    wrong = validate_citation({"record_id": "u1", "quote": "demo video of prox", "position": 1}, [URL0, URL1], ALLOWED)
    assert not wrong.ok and wrong.reason == "quote_not_found"


def test_citation_object_is_accepted():
    r = validate_citation(Citation("m1", "last release"), [MSG], ALLOWED)
    assert r.ok


def test_validate_items_keeps_valid_citations_and_counts_drops():
    items = [
        {"title": "a", "citations": [{"record_id": "m1", "quote": "last release"},
                                     {"record_id": "m1", "quote": "nope"}]},
        {"title": "b", "citations": [{"record_id": "m1", "quote": ""}]},
        {"title": "c", "citations": []},
        {"title": "d"},
    ]
    kept, dropped = validate_items(items, [MSG], ALLOWED)
    assert [k["title"] for k in kept] == ["a"]
    assert len(kept[0]["citations"]) == 1 and kept[0]["citations"][0]["quote"] == "last release"
    assert dropped == {"empty_quote": 1, "no_citation": 2}


def test_source_records_from_mini_fixture(mini_bundle):
    from inherit.data.bundle import load_bundle, load_metadata
    emp, cust = load_metadata(mini_bundle)
    rec = build_records(load_bundle(mini_bundle, "mini"), emp, cust, "mini")
    srcs = source_records(rec)
    kinds = {(s.source_type, s.field) for s in srcs}
    assert {("message", "text"), ("segment", "text"), ("chat", "text"), ("document", "content"),
            ("document", "feedback"), ("pr", "title"), ("pr", "summary"), ("pr", "review"),
            ("url", "description")} <= kinds
    seg = next(s for s in srcs if s.source_type == "segment")
    assert "#" in seg.id
    dup = [s for s in srcs if s.id == "docs_example_com_export"]
    assert [s.position for s in dup] == [0, 1]  # repeated URL ids keep their stable position
    rev = [s for s in srcs if s.source_type == "pr" and s.field == "review"]
    assert any(s.id == "github_com_mini_mini-server_pull_101" and s.position == 0 for s in rev)
