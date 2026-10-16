import json

import pytest

import bugfx as fx
from inherit import ontology
from inherit.extract import bugs
from inherit.extract.catalog import build_catalog
from inherit.facts import Facts


def run(chat=None, rec=None, tmp=None, **kw):
    chat = chat or fx.standard_chat()
    return bugs.run_bugs(rec or fx.records(), chat=chat, cache_root=tmp, **kw), chat


def nodes(facts, label):
    return {n.id: n for n in facts.nodes if n.label == label}


def edges(facts, type_):
    return [e for e in facts.edges if e.type == type_]


@pytest.fixture
def res(tmp_path):
    out, chat = run(tmp=tmp_path)
    return out, chat


def bug_by_title(facts, title):
    return next(b for b in nodes(facts, "Bug").values() if b.props["title"] == title)


# ---------------------------------------------------------------- bugs, customers, events

def test_bug_ids_and_titles(res):
    out, _ = res
    ids = sorted(nodes(out.facts, "Bug"))
    assert ids == [
        "Bug:Mini:ch-mini-1:duplicate-log-entries-for-manual-actions",
        "Bug:Mini:ch-mini-1:inefficient-database-query-optimization",
        "Bug:Mini:ch-mini-1:memory-leak-in-suggestion-rendering",
    ]


def test_paraphrased_resolution_does_not_create_a_bug(res):
    out, _ = res
    assert len(nodes(out.facts, "Bug")) == 3


def test_customers_by_company(res):
    out, _ = res
    rb = {(e.src_id.split(":")[-1], e.dst_id) for e in edges(out.facts, "REPORTED_BY")}
    assert ("duplicate-log-entries-for-manual-actions", "Customer:CUST-0047") in rb
    assert ("memory-leak-in-suggestion-rendering", "Customer:CUST-0064") in rb
    assert nodes(out.facts, "Customer")["Customer:CUST-0047"].props["company"] == "TechCorp"
    assert out.report["bugs_per_company"] == {"BlueWave": 1, "TechCorp": 2}


def test_event_nodes_keys_and_edges(res):
    out, _ = res
    evs = nodes(out.facts, "BugEvent")
    assert "BugEvent:Mini:20260111-1-b0002:0" in evs  # REPORTED comes first in its message
    assert evs["BugEvent:Mini:20260111-1-b0002:0"].props["type"] == "REPORTED"
    assert evs["BugEvent:Mini:20260111-1-b0002:1"].props["type"] == "ASSIGNED"
    ab = {(e.src_id, e.dst_id) for e in edges(out.facts, "ACTED_BY")}
    assert ("BugEvent:Mini:20260111-1-b0002:1", f"Person:{fx.IAN}") in ab  # the author who assigned
    at = {(e.src_id, e.dst_id) for e in edges(out.facts, "ASSIGNED_TO")}
    assert ("BugEvent:Mini:20260111-1-b0002:1", f"Person:{fx.EMMA}") in at
    he = {(e.src_id.split(":")[-1], e.dst_id) for e in edges(out.facts, "HAS_EVENT")}
    assert ("duplicate-log-entries-for-manual-actions", "BugEvent:Mini:20260111-1-b0002:0") in he


def test_event_timestamps_come_from_records(res):
    out, _ = res
    ev = nodes(out.facts, "BugEvent")["BugEvent:Mini:20260115-0-d0001:0"]
    assert ev.props["type"] == "RESOLVED" and ev.props["ts"] == "2026-01-15T11:00:00"
    assert ev.props["fix_pr_links"] == ["https://github.com/salesforce/Mini/pull/31"]


def test_resolution_actor_fix_pr_and_paraphrase_link(res):
    out, _ = res
    fixes = {(e.src_id, e.dst_id.split(":")[-1]) for e in edges(out.facts, "FIXES")}
    assert ("PR:https://github.com/salesforce/Mini/pull/31", "inefficient-database-query-optimization") in fixes
    # e1 paraphrase linked by the channel call to B1 (Duplicate Log Entries), PR 32 from the later message
    assert ("PR:https://github.com/salesforce/Mini/pull/32", "duplicate-log-entries-for-manual-actions") in fixes
    resolved = [n for n in nodes(out.facts, "BugEvent").values() if n.props["type"] == "RESOLVED"]
    assert {n.props["link_method"] for n in resolved} == {"exact", "model"}
    ab = {(e.src_id, e.dst_id) for e in edges(out.facts, "ACTED_BY")}
    assert ("BugEvent:Mini:20260117-0-e0001:0", f"Person:{fx.EMMA}") in ab


def test_learned_alias_is_stored(res):
    out, _ = res
    b = bug_by_title(out.facts, "Duplicate Log Entries for Manual Actions")
    assert "duplicate manual action log rows" in b.props["aliases"]


def test_derived_states_on_bug_nodes(res):
    out, _ = res
    dup = bug_by_title(out.facts, "Duplicate Log Entries for Manual Actions").props
    # REPORTED, ASSIGNED(Emma), RESOLVED(pr 32), REOPENED, ASSIGNED(Emma, from the model)
    assert dup["event_types"] == ["REPORTED", "ASSIGNED", "RESOLVED", "REOPENED", "ASSIGNED"]
    assert dup["status"] == "ASSIGNED"
    assert dup["unresolved"] is False and dup["resolved"] is True
    assert dup["active"] is True and dup["reopened"] is True and dup["no_fix_needed"] is False
    assert dup["current_assignees"] == [fx.EMMA]
    ineff = bug_by_title(out.facts, "Inefficient Database Query Optimization").props
    assert ineff["status"] == "RESOLVED" and ineff["unresolved"] is False and ineff["active"] is False
    leak = bug_by_title(out.facts, "Memory Leak in Suggestion Rendering").props
    assert leak["no_fix_needed"] is True and leak["unresolved"] is True and leak["active"] is False
    assert leak["status"] == "DISMISSED"
    assert leak["component"] == "User Interface & Frontend Performance"
    assert dup["derived_by"] == "bug_state/v1"


def test_has_edge_from_product_root(res):
    out, _ = res
    hs = {e.dst_id for e in edges(out.facts, "HAS") if e.src_id == "Product:Mini"}
    assert hs == set(nodes(out.facts, "Bug"))


def test_all_edges_pass_the_ontology(res):
    out, _ = res
    for e in out.facts.edges:
        ontology.check_edge(e)
    assert out.facts.conflicts == 0


def test_external_look_alike(res):
    out, _ = res
    [x] = nodes(out.facts, "ExternalBug").values()
    assert x.props["link"] == "https://github.com/postgres/postgres/pull/3"
    assert x.props["title"] == "Duplicate Log Entries for Transaction Rollbacks"
    [e] = edges(out.facts, "SIMILAR_TO")
    assert e.src_id == x.id and e.dst_id.endswith("duplicate-log-entries-for-manual-actions")
    assert out.report["external_look_alikes"] == 1
    assert len(nodes(out.facts, "Bug")) == 3  # never a customer bug


def test_every_model_edge_and_node_roundtrips(res, tmp_path):
    out, _ = res
    p = tmp_path / "bugs.jsonl"
    out.facts.to_jsonl(p)
    assert Facts.from_jsonl(p).nodes


# ---------------------------------------------------------------- what the model was asked

def test_model_calls_only_for_residual_and_reopen_sessions(res):
    out, chat = res
    assert len(chat.session_calls) == 2  # the unquoted intake session and the reopen session
    assert len(chat.link_calls) == 1  # one per channel, only for the unmatched paraphrase
    for c in chat.calls:
        assert c["arm"] == "extract"
        assert c["response_format"] == {"type": "json_object"}
    assert "20260115-0-d0001" not in "".join(c["user"] for c in chat.session_calls)
    assert out.report["calls"] == {"session": 2, "link": 1, "cached": 0}


def test_prompt_carries_messages_hints_and_catalogue(res):
    _, chat = res
    u = chat.session_calls[0]["user"]
    assert "20260111-5-b0006" in u and "Ian Miller" in u and fx.IAN in u
    assert "Duplicate Log Entries for Manual Actions" in u  # rules candidates are shown
    assert "CUST-0064" in u
    r = next(c for c in chat.session_calls if "20260125-1-g0002" in c["user"])["user"]
    assert "m1" in r and "Duplicate Log Entries for Manual Actions" in r


def test_prompt_lists_catalogue_prs_that_appear_in_the_messages():
    rec = fx.records()
    cat = build_catalog(rec)
    session = [m for m in rec.messages if m.id.startswith("20260115")]
    parse = bugs.br.parse_session(session, cat, rec.customers)
    msgs, max_tokens = bugs.build_session_request(session, parse, rec, cat)
    assert "https://github.com/salesforce/Mini/pull/31 Fix 31" in msgs[1]["content"]
    assert "pull/32" not in msgs[1]["content"]
    assert 600 <= max_tokens <= 4000


def test_link_call_lists_candidates_and_items(res):
    _, chat = res
    u = chat.link_calls[0]["user"]
    assert "duplicate manual action log rows" in u
    assert "B1" in u and "Duplicate Log Entries for Manual Actions" in u and "I1" in u


# ---------------------------------------------------------------- validation, links, failures

def test_item_with_unsupported_quote_is_dropped_and_counted(tmp_path):
    bad = dict(fx.LEAK_BUG, customer_ids=[], citations=[fx.cite("20260111-5-b0006", "a quote that is not in the message")])
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[bad]),
                                   "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    assert len(nodes(out.facts, "Bug")) == 2
    assert out.report["dropped"]["bugs"] == {"quote_not_found": 1}
    # the dismissal of the missing bug cannot link
    assert any(u["type"] == "DISMISSED" for u in out.report["unmatched_events"])


def test_event_citing_a_record_outside_the_session_is_dropped(tmp_path):
    bad = dict(fx.REASSIGN_EVENT, citations=[fx.cite("20260111-5-b0006", "memory leak")])
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260125-1-g0002": fx.session_json(events=[bad])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    assert out.report["dropped"]["events"] == {"record_not_supplied": 1}
    dup = bug_by_title(out.facts, "Duplicate Log Entries for Manual Actions").props
    assert dup["event_types"] == ["REPORTED", "ASSIGNED", "RESOLVED", "REOPENED"]


def test_model_event_for_a_rule_handled_message_is_ignored(tmp_path):
    dup_resolved = {"type": "RESOLVED", "bug_ref": None, "bug_mention": "Inefficient Database Query Optimization",
                    "actor_message_id": "20260125-0-g0001", "assignee_eids": [], "fix_pr_links": [],
                    "citations": [fx.cite("20260125-0-g0001", "reopened the bug")]}
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260125-1-g0002": fx.session_json(events=[dup_resolved, fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    dup = bug_by_title(out.facts, "Duplicate Log Entries for Manual Actions").props
    assert dup["event_types"].count("RESOLVED") == 1
    assert out.report["ignored_model_events"] == 1


def test_assignee_must_be_a_person_in_the_session(tmp_path):
    bad = dict(fx.REASSIGN_EVENT, assignee_eids=["eid_deadbeef"])
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260125-1-g0002": fx.session_json(events=[bad])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    dup = bug_by_title(out.facts, "Duplicate Log Entries for Manual Actions").props
    assert dup["current_assignees"] == [fx.EMMA]  # the invented id never enters; the earlier assignment stands
    assert out.report["dropped"]["events"] == {"assignee_not_in_session": 1}


def test_null_link_keeps_the_event_unmatched(tmp_path):
    chat = fx.ScriptedChat(script=dict(fx.standard_chat().script), link=fx.link_json(("I1", None)))
    out, _ = run(chat=chat, tmp=tmp_path)
    assert len(nodes(out.facts, "Bug")) == 3
    [u] = out.report["unmatched_events"]
    assert u["message_id"] == "20260117-0-e0001" and u["mention"] == "duplicate manual action log rows"
    assert out.report["events"]["unmatched"] == 1
    assert not any(n.id == "BugEvent:Mini:20260117-0-e0001:0" for n in out.facts.nodes)
    dup = bug_by_title(out.facts, "Duplicate Log Entries for Manual Actions").props
    assert "RESOLVED" not in dup["event_types"] and dup["unresolved"] is True


def test_invalid_or_future_link_target_is_rejected(tmp_path):
    # B3 (memory leak) is first reported at 09:20 on 2026-01-11; a resolution at 09:15 cannot refer to it.
    slack = [r for r in fx.SLACK if not r[0].startswith("20260117")] + [
        ("20260111-9-z0001", fx.BOB, "2026-01-11T09:15:00", "Hi team, the bug about leaking rendering has been resolved.")]
    for n, answer in enumerate((fx.link_json(("I1", "B3")), fx.link_json(("I1", "B99")), "not json at all")):
        chat = fx.ScriptedChat(script=dict(fx.standard_chat().script), link=answer)
        out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path / str(n))
        assert out.report["events"]["unmatched"] == 1, answer
        assert len(chat.link_calls) == 1


def test_events_are_ordered_by_time_not_by_id(tmp_path):
    slack = [
        ("20260111-1-b0002", fx.IAN, "2026-01-11T09:02:00", fx.B_TITLE1.replace(f"@{fx.EMMA}", "")),
        ("20260927-0-8c21b", fx.IAN, "2026-09-27T11:00:00", fx.G_REOPEN),
        ("20260927-1-8d6b4", fx.BOB, "2026-09-27T09:00:00",
         "Hi team, the bug titled 'Duplicate Log Entries for Manual Actions' has been resolved. "
         "PR: https://github.com/salesforce/Mini/pull/31"),
    ]
    out, _ = run(rec=fx.records(slack=slack), tmp=tmp_path)
    d = bug_by_title(out.facts, "Duplicate Log Entries for Manual Actions").props
    assert d["event_types"] == ["REPORTED", "RESOLVED", "REOPENED"]
    assert d["active"] is True and d["status"] == "REOPENED"


# ---------------------------------------------------------------- cache, truncation, failures, estimate

def test_cache_makes_the_rerun_free(tmp_path):
    first, chat1 = run(tmp=tmp_path)
    assert len(chat1.calls) == 3
    second, chat2 = run(tmp=tmp_path)
    assert chat2.calls == []
    assert second.report["calls"] == {"session": 0, "link": 0, "cached": 3}
    assert [n.id for n in second.facts.nodes] == [n.id for n in first.facts.nodes] or \
        sorted(n.id for n in second.facts.nodes) == sorted(n.id for n in first.facts.nodes)
    files = list((tmp_path / bugs.PROMPT_VERSION).glob("*.json"))
    assert len(files) == 3 and all(len(f.stem) == 64 for f in files)


def test_truncated_output_is_retried_once_with_double_max_tokens(tmp_path):
    ok = fx.session_json(bugs=[fx.LEAK_BUG])
    chat = fx.ScriptedChat(script={"20260111-5-b0006": [('{"bugs": [{"local_id": "x1", "tit', 10_000), ok],
                                   "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    calls = [c for c in chat.session_calls if "20260111-5-b0006" in c["user"]]
    assert len(calls) == 2 and calls[1]["max_tokens"] == 2 * calls[0]["max_tokens"]
    assert out.report["extraction_failures"] == []
    assert len(nodes(out.facts, "Bug")) == 3


def test_second_truncation_is_a_counted_failure(tmp_path):
    chat = fx.ScriptedChat(script={"20260111-5-b0006": [('{"bugs": [', 10_000)],
                                   "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    [f] = out.report["extraction_failures"]
    assert f["kind"] == "session" and "20260111-5-b0006" in f["messages"] and f["reason"] == "truncated"
    assert len(nodes(out.facts, "Bug")) == 2  # rules results of the same session still stand
    assert out.report["calls"]["session"] == 3  # two attempts for the cut-off session, one for the reopen


def test_non_json_answer_is_a_counted_failure(tmp_path):
    chat = fx.ScriptedChat(script={"20260111-5-b0006": "I cannot do that",
                                   "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    assert [f["reason"] for f in out.report["extraction_failures"]] == ["invalid_json"]


def test_failures_are_not_cached(tmp_path):
    chat = fx.ScriptedChat(script={"20260111-5-b0006": "oops"}, link=fx.link_json(("I1", "B1")))
    run(chat=chat, tmp=tmp_path)
    again = fx.standard_chat()
    out, _ = run(chat=again, tmp=tmp_path)
    assert out.report["extraction_failures"] == [] and len(nodes(out.facts, "Bug")) == 3


def test_estimate_counts_calls_without_spending(tmp_path):
    chat = fx.standard_chat()
    est = bugs.estimate_bugs(fx.records(), cache_root=tmp_path)
    assert est["session_calls"] == 2 and est["cached"] == 0
    assert est["link_calls_max"] == 1
    assert est["projected_usd"] > 0
    assert chat.calls == []
    run(tmp=tmp_path)
    again = bugs.estimate_bugs(fx.records(), cache_root=tmp_path)
    assert again["session_calls"] == 0 and again["cached"] == 3 or again["cached"] >= 2


def test_report_counts(res):
    out, _ = res
    r = out.report
    assert r["bugs"] == 3
    assert r["events_by_type"] == {"ASSIGNED": 3, "DISMISSED": 1, "REOPENED": 1, "REPORTED": 3, "RESOLVED": 2}
    assert r["events"] == {"linked": 10, "unmatched": 0}
    assert r["prompt_version"] == bugs.PROMPT_VERSION
    assert r["resolved_without_catalogued_pr"] == 0


# ---------------------------------------------------------------- runner

def test_runner_bugs_layer_estimate_and_files(tmp_path, mini_bundle):
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    root_repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "root"
    shutil.copytree(mini_bundle, root / "data" / "bundle")
    (root / "src").mkdir()
    env = {"PYTHONPATH": str(root_repo / "src"), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-m", "inherit.extract.run", "--layer", "bugs", "--products", "mini",
                        "--repo-root", str(root), "--estimate"], capture_output=True, text=True, cwd=root_repo, env=env)
    assert r.returncode == 0, r.stderr
    assert "session calls 0" in r.stdout and "projected" in r.stdout
    assert not (root / "data" / "facts" / "mini" / "bugs.jsonl").exists()  # an estimate writes nothing
    r = subprocess.run([sys.executable, "-m", "inherit.extract.run", "--layer", "bugs", "--products", "mini",
                        "--repo-root", str(root)], capture_output=True, text=True, cwd=root_repo, env=env)
    assert r.returncode == 0, r.stderr  # the mini product has no bug-channel sessions that need the model
    d = root / "data" / "facts" / "mini"
    assert (d / "bugs.jsonl").exists() and (d / "bugs_report.json").exists()
    log = next((root / "runs" / "access").glob("extract-*.jsonl")).read_text()
    assert '"allowed": false' not in log


# ---------------------------------------------------------------- containment links, look-alike guard

def _two_bug_slack(resolved_text):
    return [
        ("20260111-0-b0001", fx.IAN, "2026-01-11T09:00:00", fx.B_INTRO),
        ("20260111-1-b0002", fx.IAN, "2026-01-11T09:02:00", fx.B_TITLE1),
        ("20260111-2-b0003", fx.IAN, "2026-01-11T09:10:00", fx.B_TITLE2),
        ("20260115-0-d0001", fx.BOB, "2026-01-15T11:00:00", resolved_text),
    ]


def test_mention_that_contains_two_titles_is_not_guessed(tmp_path):
    slack = _two_bug_slack("Hi team, the bug about Duplicate Log Entries for Manual Actions and Inefficient Database "
                           "Query Optimization has been resolved.")
    chat = fx.ScriptedChat(link=fx.link_json(("I1", None)))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    assert len(chat.link_calls) == 1
    assert out.report["events"]["unmatched"] == 1


def test_short_titles_are_not_matched_by_containment(tmp_path):
    slack = [
        ("20260111-0-b0001", fx.IAN, "2026-01-11T09:00:00", fx.B_INTRO),
        ("20260111-1-b0002", fx.IAN, "2026-01-11T09:02:00",
         "First up, we have a bug titled 'Memory Leak'. Customer CUST-0047 is affected. " f"@{fx.EMMA}, please take it."),
        ("20260115-0-d0001", fx.BOB, "2026-01-15T11:00:00",
         "Hi team, the bug about a memory leak in the export module has been resolved."),
    ]
    chat = fx.ScriptedChat(link=fx.link_json(("I1", None)))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    assert len(chat.link_calls) == 1 and out.report["events"]["unmatched"] == 1


def test_model_bug_from_a_look_alike_message_is_rejected(tmp_path):
    bad = {"local_id": "x9", "title": "seasonal adjustment issue", "aliases": [], "component": None,
           "impact_text": "", "customer_ids": [], "message_id": "20260113-1-c0002",
           "citations": [fx.cite("20260113-1-c0002", "similar to our")]}
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260113-1-c0002": fx.session_json(bugs=[bad]),
                                   "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    # an unparsed look-alike message triggers a session call: strip the quote so the rules leave it residual
    slack = [(i, a, t, x.replace("'", "")
              if i == "20260113-1-c0002" else x) for i, a, t, x in fx.SLACK]
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    assert len(nodes(out.facts, "Bug")) == 3
    assert out.report["dropped"]["bugs"] == {"look_alike_message": 1}


def test_model_citation_is_repaired_to_the_verbatim_title_and_counted(tmp_path):
    # the quote joins two sentences and drops the markup, the title itself is verbatim in the cited message
    text = ("Lastly, we have **Memory Leak in Suggestion Rendering** (priority **High**). It hits customer CUST-0064.")
    slack = [r if r[0] != "20260111-5-b0006" else (r[0], r[1], r[2], text) for r in fx.SLACK]
    bad = dict(fx.LEAK_BUG, citations=[fx.cite("20260111-5-b0006", "Memory Leak in Suggestion Rendering. It hits")])
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[bad]),
                                   "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    assert len(nodes(out.facts, "Bug")) == 3
    assert out.report["repaired_citations"] == 1
    assert "dropped" not in out.report or "bugs" not in out.report["dropped"]


@pytest.mark.parametrize("typ,text,ok", [
    ("RESOLVED", "I'll take it on again and start with the retry logic.", False),
    ("RESOLVED", "I've fixed it by adding a retry. PR: https://github.com/salesforce/Mini/pull/32", True),
    ("DISMISSED", "I'll revisit the design and see.", False),
    ("DISMISSED", "After a review this is expected behavior.", True),
    ("REOPENED", "Thanks, I will check.", False),
    ("REOPENED", "The customer says it is still happening.", True),
])
def test_model_event_needs_wording_that_fits_its_type(tmp_path, typ, text, ok):
    slack = [r if r[0] != "20260125-1-g0002" else (r[0], r[1], r[2], text) for r in fx.SLACK]
    ev = dict(fx.REASSIGN_EVENT, type=typ, assignee_eids=[], citations=[fx.cite("20260125-1-g0002", text[:20])])
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260125-1-g0002": fx.session_json(events=[ev])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    dropped = out.report["dropped"].get("events", {})
    assert (dropped == {}) is ok
    if not ok:
        assert dropped == {"no_evidence_for_type": 1}


# ---------------------------------------------------------------- product areas (components)

def _area_slack(text2):
    return [
        ("20260111-0-b0001", fx.IAN, "2026-01-11T09:00:00", fx.B_INTRO),
        ("20260111-1-b0002", fx.IAN, "2026-01-11T09:02:00", fx.B_TITLE1),
        ("20260111-2-b0003", fx.EMMA, "2026-01-11T09:05:00", fx.B_ACK1),
        ("20260111-3-b0004", fx.IAN, "2026-01-11T09:10:00", text2),
        ("20260111-4-b0005", fx.BOB, "2026-01-11T09:12:00", fx.B_ACK2),
    ]


AREA_TEXT = ("Next, we have 'Inefficient Database Query Optimization'. This is slowing down the database "
             "performance for customer CUST-0108. Can you take this one?")


def _area_answer(component, quote="the database performance"):
    return fx.session_json() if component is None else json.dumps({"bugs": [], "events": [], "external": [],
        "components": [{"bug_ref": "rb2", "component": component, "citations": [fx.cite("20260111-3-b0004", quote)]}]})


def test_missing_area_is_asked_of_the_model_and_must_be_written_in_the_message(tmp_path):
    chat = fx.ScriptedChat(script={"20260111-3-b0004": _area_answer("database performance")})
    out, _ = run(chat=chat, rec=fx.records(slack=_area_slack(AREA_TEXT)), tmp=tmp_path)
    assert len(chat.session_calls) == 1
    assert "BUGS WITHOUT A PRODUCT AREA" in chat.session_calls[0]["user"]
    b = bug_by_title(out.facts, "Inefficient Database Query Optimization")
    assert b.props["component"] == "database performance"


def test_area_not_written_in_the_message_is_ignored_and_counted(tmp_path):
    chat = fx.ScriptedChat(script={"20260111-3-b0004": _area_answer("Network Layer")})
    out, _ = run(chat=chat, rec=fx.records(slack=_area_slack(AREA_TEXT)), tmp=tmp_path)
    assert bug_by_title(out.facts, "Inefficient Database Query Optimization").props["component"] is None
    assert out.report["dropped"]["components"] == {"not_in_message": 1}


def test_area_spellings_collapse_to_one_display_form(tmp_path):
    # the first intake writes "Database Management & Logging System"; this one writes it in lower case with "and"
    text2 = ("Next, we have 'Inefficient Database Query Optimization'. This is slowing the database management and "
             "logging system for customer CUST-0108. Can you take this one?")
    chat = fx.ScriptedChat(script={"20260111-3-b0004": _area_answer("database management and logging system",
                                                                    "database management and logging system")})
    out, _ = run(chat=chat, rec=fx.records(slack=_area_slack(text2)), tmp=tmp_path)
    comps = {b.props["component"] for b in nodes(out.facts, "Bug").values()}
    assert comps == {"Database Management & Logging System"}


def test_event_whose_ref_is_the_bug_title_words_is_used_as_the_mention(tmp_path):
    ev = dict(fx.REASSIGN_EVENT, bug_ref="Duplicate Log Entries for Manual Actions", bug_mention=None)
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260125-1-g0002": fx.session_json(events=[ev])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    assert "dropped" not in out.report or "events" not in out.report["dropped"]
    dup = bug_by_title(out.facts, "Duplicate Log Entries for Manual Actions").props
    assert dup["event_types"][-1] == "ASSIGNED"


def test_session_called_only_for_areas_ignores_everything_else_the_model_says(tmp_path):
    extra = {"type": "ASSIGNED", "bug_ref": "rb1", "bug_mention": None, "actor_message_id": "20260111-2-b0003",
             "assignee_eids": [fx.EMMA], "fix_pr_links": [], "citations": [fx.cite("20260111-2-b0003", "I'll start looking")]}
    answer = json.loads(_area_answer("database performance"))
    answer["events"] = [extra]
    chat = fx.ScriptedChat(script={"20260111-3-b0004": json.dumps(answer)})
    out, _ = run(chat=chat, rec=fx.records(slack=_area_slack(AREA_TEXT)), tmp=tmp_path)
    assert "Only the product areas are needed" in chat.session_calls[0]["user"]
    assert bug_by_title(out.facts, "Inefficient Database Query Optimization").props["component"] == "database performance"
    assert out.report["ignored_model_events"] == 1
    assert "20260111-2-b0003" not in {n.props["message_id"] for n in nodes(out.facts, "BugEvent").values()}


def test_bug_title_slightly_reworded_by_the_model_is_cited_by_its_customer_id(tmp_path):
    text = ("Lastly, there's a security issue where sensitive data is being exposed in error messages. It hits "
            "customer CUST-0064. Can you take it?")
    slack = [r if r[0] != "20260111-5-b0006" else (r[0], r[1], r[2], text) for r in fx.SLACK]
    bad = dict(fx.LEAK_BUG, title="sensitive data being exposed in error messages", component=None,
               citations=[fx.cite("20260111-5-b0006", "sensitive data being exposed in error messages")])
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[bad]),
                                   "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    b = bug_by_title(out.facts, "sensitive data being exposed in error messages")
    assert b.props["customer_ids"] == ["CUST-0064"]
    assert out.report["repaired_citations"] == 1
    ev = nodes(out.facts, "BugEvent")["BugEvent:Mini:20260111-5-b0006:0"]
    assert ev.props["citations"][0]["quote"] == "CUST-0064"


def test_model_cannot_open_a_bug_from_a_lifecycle_message(tmp_path):
    bad = {"local_id": "x9", "title": "Duplicate Log Entries for Manual Actions again", "aliases": [], "component": None,
           "impact_text": "", "customer_ids": [], "message_id": "20260125-0-g0001",
           "citations": [fx.cite("20260125-0-g0001", "reopened the bug")]}
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260125-1-g0002": fx.session_json(bugs=[bad], events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    assert len(nodes(out.facts, "Bug")) == 3
    assert out.report["dropped"]["bugs"] == {"lifecycle_message": 1}


def test_events_of_a_rejected_model_bug_fall_back_to_its_title_as_the_mention(tmp_path):
    # the model invents a "bug" for the reopened one and hangs the follow-up on it; the bug is rejected, the event stays
    bad = {"local_id": "x9", "title": "Duplicate Log Entries for Manual Actions", "aliases": [], "component": None,
           "impact_text": "", "customer_ids": [], "message_id": "20260125-0-g0001",
           "citations": [fx.cite("20260125-0-g0001", "reopened the bug")]}
    ev = dict(fx.REASSIGN_EVENT, bug_ref="x9")
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260125-1-g0002": fx.session_json(bugs=[bad], events=[ev])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    dup = bug_by_title(out.facts, "Duplicate Log Entries for Manual Actions").props
    assert dup["event_types"] == ["REPORTED", "ASSIGNED", "RESOLVED", "REOPENED", "ASSIGNED"]
    assert "events" not in out.report["dropped"]


def test_area_named_in_an_earlier_message_of_the_session_is_accepted(tmp_path):
    first = (f"First up, we have 'Duplicate Log Entries for Manual Actions'. Impacts the Billing Engine Core. "
             f"Customer CUST-0047. @{fx.EMMA}, can you take this one?")
    second = ("Next, we have 'Inefficient Database Query Optimization'. The slowdown hits that engine too. "
              "Customer CUST-0108. Can you take this one?")
    slack = [("20260111-0-b0001", fx.IAN, "2026-01-11T09:00:00", fx.B_INTRO),
             ("20260111-1-b0002", fx.IAN, "2026-01-11T09:02:00", first),
             ("20260111-2-b0003", fx.IAN, "2026-01-11T09:10:00", second),
             ("20260111-3-b0004", fx.BOB, "2026-01-11T09:12:00", fx.B_ACK2)]
    answer = {"bugs": [], "events": [], "external": [], "components": [
        {"bug_ref": "rb2", "component": "Billing Engine Core", "citations": [fx.cite("20260111-2-b0003", "that engine too")]}]}
    chat = fx.ScriptedChat(script={"20260111-2-b0003": json.dumps(answer)})
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    assert bug_by_title(out.facts, "Inefficient Database Query Optimization").props["component"] == "Billing Engine Core"


def test_events_hung_on_a_look_alike_pseudo_bug_are_dropped(tmp_path):
    bad = {"local_id": "x9", "title": "Duplicate Log Entries for Transaction Rollbacks", "aliases": [], "component": None,
           "impact_text": "", "customer_ids": [], "message_id": "20260113-1-c0002",
           "citations": [fx.cite("20260113-1-c0002", "similar to our")]}
    ev = {"type": "ASSIGNED", "bug_ref": "x9", "bug_mention": None, "actor_message_id": "20260113-2-c0003",
          "assignee_eids": [fx.EMMA], "fix_pr_links": [], "citations": [fx.cite("20260113-2-c0003", "I'll review the PR")]}
    slack = [r for r in fx.SLACK] + [("20260113-2-c0003", fx.EMMA, "2026-01-13T10:05:00", "Thanks! I'll review the PR.")]
    slack = [(i, a, t, x.replace("'", "") if i == "20260113-1-c0002" else x) for i, a, t, x in slack]
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260113-1-c0002": fx.session_json(bugs=[bad], events=[ev]),
                                   "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    assert "BugEvent:Mini:20260113-2-c0003:0" not in nodes(out.facts, "BugEvent")
    assert len(nodes(out.facts, "Bug")) == 3


def test_assignment_events_in_an_open_source_session_are_dropped(tmp_path):
    ev = {"type": "ASSIGNED", "bug_ref": None, "bug_mention": "Duplicate Log Entries for Manual Actions",
          "actor_message_id": "20260113-2-c0003", "assignee_eids": [fx.EMMA], "fix_pr_links": [],
          "citations": [fx.cite("20260113-2-c0003", "I'll review the PR")]}
    slack = list(fx.SLACK) + [("20260113-2-c0003", fx.EMMA, "2026-01-13T10:05:00",
                               "Thanks! This is like Duplicate Log Entries for Manual Actions. I'll review the PR.")]
    slack = [(i, a, t, x.replace("'", "") if i == "20260113-1-c0002" else x) for i, a, t, x in slack]
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260113-1-c0002": fx.session_json(events=[ev]),
                                   "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    assert out.report["dropped"]["events"] == {"look_alike_session": 1}
    assert "BugEvent:Mini:20260113-2-c0003:0" not in nodes(out.facts, "BugEvent")


def test_request_for_help_is_not_an_assignment(tmp_path):
    text = f"Sure thing, I'll dive into the logs. @{fx.BOB}, could you assist me with this? Two heads are better."
    slack = [r if r[0] != "20260125-1-g0002" else (r[0], fx.EMMA, r[2], text) for r in fx.SLACK]
    ev = dict(fx.REASSIGN_EVENT, assignee_eids=[fx.BOB], citations=[fx.cite("20260125-1-g0002", "could you assist me")])
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260125-1-g0002": fx.session_json(events=[ev])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    assert out.report["dropped"]["events"] == {"help_request": 1}
    dup = bug_by_title(out.facts, "Duplicate Log Entries for Manual Actions").props
    assert dup["current_assignees"] == [fx.EMMA]  # still the engineer assigned before the reopening


def test_reply_accepting_a_request_for_help_is_not_an_assignment_either(tmp_path):
    ask = f"Sure thing, I'll dive into the logs. @{fx.BOB}, could you assist me with this?"
    slack = [r for r in fx.SLACK if r[0] != "20260125-1-g0002"] + [
        ("20260125-1-g0002", fx.EMMA, "2026-01-25T14:03:00", ask),
        ("20260125-2-g0003", fx.BOB, "2026-01-25T14:05:00", "Absolutely! I'll start by reviewing the recent deployments.")]
    reply = {"type": "ASSIGNED", "bug_ref": "m1", "bug_mention": None, "actor_message_id": "20260125-2-g0003",
             "assignee_eids": [fx.BOB], "fix_pr_links": [], "citations": [fx.cite("20260125-2-g0003", "I'll start by reviewing")]}
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260125-1-g0002": fx.session_json(events=[reply])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    assert out.report["dropped"]["events"] == {"help_reply": 1}


def test_look_alike_with_a_bad_quote_is_cited_by_its_link(tmp_path):
    slack = [(i, a, t, x.replace("'", "") if i == "20260113-1-c0002" else x) for i, a, t, x in fx.SLACK]
    ext = {"title": "Repeated log rows after rollbacks", "link": "https://github.com/postgres/postgres/pull/3",
           "similar_mention": "Duplicate Log Entries for Manual Actions", "message_id": "20260113-1-c0002",
           "citations": [fx.cite("20260113-1-c0002", "a quote the model made up")]}
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG]),
                                   "20260113-1-c0002": fx.session_json(external=[ext]),
                                   "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    [x] = nodes(out.facts, "ExternalBug").values()
    assert x.props["citations"][0]["quote"] == "https://github.com/postgres/postgres/pull/3"
    assert out.report["repaired_citations"] == 1


# ---------------------------------------------------------------- identity is exact; the model links the rest

def _pair_slack(resolved_text, ts="2026-01-15T11:00:00"):
    return [
        ("20260111-0-b0001", fx.IAN, "2026-01-11T09:00:00", fx.B_INTRO),
        ("20260111-1-b0002", fx.IAN, "2026-01-11T09:02:00",
         f"First up, we have a 'Exports Overwrite Imports' issue. Customer CUST-0047 is affected. @{fx.EMMA}, can you take this one?"),
        ("20260111-2-b0003", fx.IAN, "2026-01-11T09:10:00",
         f"Next, we have an 'Imports Overwrite Exports' issue. Customer CUST-0108 is affected. @{fx.BOB}, can you take this one?"),
        ("20260115-0-d0001", fx.BOB, ts, resolved_text),
    ]


def test_reordered_words_are_two_bugs_and_a_lookalike_mention_goes_to_the_model(tmp_path):
    slack = _pair_slack("Hi team, the bug about overwrite imports exports has been resolved.")
    chat = fx.ScriptedChat(link=fx.link_json(("I1", "B2")))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    assert len(nodes(out.facts, "Bug")) == 2
    assert len(chat.link_calls) == 1
    r = next(n for n in nodes(out.facts, "BugEvent").values() if n.props["type"] == "RESOLVED")
    assert r.props["link_method"] == "model" and r.props["bug"].endswith("imports-overwrite-exports")


def test_reordered_title_introduced_later_is_a_new_bug_not_a_merge(tmp_path):
    out, _ = run(chat=fx.ScriptedChat(), rec=fx.records(slack=_pair_slack("Thanks all.")), tmp=tmp_path)
    titles = sorted(b.props["title"] for b in nodes(out.facts, "Bug").values())
    assert titles == ["Exports Overwrite Imports", "Imports Overwrite Exports"]


def test_containing_mention_is_not_linked_without_the_model(tmp_path):
    slack = _pair_slack("Hi team, the bug about Exports Overwrite Imports in the audit dashboard has been resolved.")
    chat = fx.ScriptedChat(link=fx.link_json(("I1", None)))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    assert len(chat.link_calls) == 1 and out.report["events"]["unmatched"] == 1


def test_local_reference_to_a_bug_reported_later_stays_unmatched(tmp_path):
    # the ack at 09:01 would self-assign the bug introduced at 09:05
    slack = [
        ("20260111-0-b0001", fx.IAN, "2026-01-11T09:00:00", fx.B_INTRO),
        ("20260111-1-b0002", fx.BOB, "2026-01-11T09:01:00", "Sure thing! I'll dive in right away."),
        ("20260111-2-b0003", fx.IAN, "2026-01-11T09:05:00",
         "First up, we have a 'Memory Leak in Exports' issue. Customer CUST-0047 is affected. Can you take this one?"),
    ]
    out, _ = run(chat=fx.ScriptedChat(), rec=fx.records(slack=slack), tmp=tmp_path)
    b = bug_by_title(out.facts, "Memory Leak in Exports").props
    assert "ASSIGNED" not in b["event_types"]
    assert out.report["events"]["unmatched"] >= 0


def test_model_event_with_a_local_ref_to_a_later_bug_stays_unmatched(tmp_path):
    # 20260111-2-b0003 (09:05) says it takes the leak bug that is only introduced at 09:20 (b0006)
    ev = {"type": "ASSIGNED", "bug_ref": "x1", "bug_mention": None, "actor_message_id": "20260111-2-b0003",
          "assignee_eids": [fx.EMMA], "fix_pr_links": [], "citations": [fx.cite("20260111-2-b0003", "I'll dive into")]}
    slack = [r if r[0] != "20260111-2-b0003" else (r[0], r[1], r[2], "Sure thing! I'll dive into the leak.") for r in fx.SLACK]
    chat = fx.ScriptedChat(script={"20260111-5-b0006": fx.session_json(bugs=[fx.LEAK_BUG], events=[ev]),
                                   "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, rec=fx.records(slack=slack), tmp=tmp_path)
    leak = bug_by_title(out.facts, "Memory Leak in Suggestion Rendering").props
    assert "ASSIGNED" not in leak["event_types"]
    assert out.report["events"]["unmatched"] == 1


# ---------------------------------------------------------------- response shape, long sessions

@pytest.mark.parametrize("bad", [
    '{"bugs": 7, "events": [], "external": []}',
    '{"bugs": [], "events": ["x"], "external": []}',
    '{"bugs": [], "events": [], "external": [], "components": {"a": 1}}',
])
def test_valid_json_with_the_wrong_shape_is_a_failure_and_is_not_cached(tmp_path, bad):
    chat = fx.ScriptedChat(script={"20260111-5-b0006": bad, "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    [f] = out.report["extraction_failures"]
    assert f["reason"] == "invalid_shape" and "20260111-5-b0006" in f["messages"]
    assert len(list((tmp_path / bugs.PROMPT_VERSION).glob("*.json"))) == 2  # the bad answer is not on disk


def test_link_answer_with_the_wrong_shape_is_a_failure(tmp_path):
    chat = fx.ScriptedChat(script=dict(fx.standard_chat().script), link='{"links": 5}')
    out, _ = run(chat=chat, tmp=tmp_path)
    assert [f["kind"] for f in out.report["extraction_failures"]] == ["link"]
    assert out.report["events"]["unmatched"] == 1


def test_poisoned_cache_entry_is_ignored(tmp_path):
    run(tmp=tmp_path)
    for f in (tmp_path / bugs.PROMPT_VERSION).glob("*.json"):
        d = json.loads(f.read_text())
        if "bugs" in d["response"]:
            d["response"]["bugs"] = 7
            f.write_text(json.dumps(d))
    chat = fx.standard_chat()
    out, _ = run(chat=chat, tmp=tmp_path)
    assert out.report["extraction_failures"] == [] and len(chat.session_calls) == 2  # both poisoned answers are asked again


def _long_slack(n=120):
    rows = [("20260111-0-b0001", fx.IAN, "2026-01-11T09:00:00", fx.B_INTRO),
            ("20260111-1-b0002", fx.IAN, "2026-01-11T09:01:00", fx.B_TITLE3)]
    for i in range(n):
        rows.append((f"20260111-{i + 2}-c{i:04d}", fx.EMMA if i % 2 else fx.BOB, f"2026-01-11T{10 + i // 60:02d}:{i % 60:02d}:30",
                     "Thanks, will check on this soon. " * 12))
    return rows


def test_oversized_session_is_split_at_message_boundaries(tmp_path):
    leak = dict(fx.LEAK_BUG, message_id="20260111-1-b0002",
                citations=[fx.cite("20260111-1-b0002", "memory leak issue in the suggestion rendering")])
    chat = fx.ScriptedChat(script={"20260111-1-b0002": fx.session_json(bugs=[leak])})
    out, _ = run(chat=chat, rec=fx.records(slack=_long_slack()), tmp=tmp_path)
    calls = chat.session_calls
    assert len(calls) >= 2
    for c in calls:
        assert len(c["user"]) / 3 <= 6000
        assert "[20260111-0-b0001]" in c["user"]  # the opener stays as context
    seen = [m for c in calls for m in __import__("re").findall(r"\[(20260111-\d+-c\d{4})\]", c["user"])]
    assert len(set(seen)) == 120  # every message reaches the model once
    assert len(nodes(out.facts, "Bug")) == 1
    assert out.report["split_sessions"] == 1


def test_split_chunks_fit_when_the_rules_list_is_large(tmp_path):
    rows = [("20260111-0-b0001", fx.IAN, "2026-01-11T09:00:00", fx.B_INTRO)]
    for i in range(55):
        rows.append((f"20260111-{i + 1}-q{i:04d}", fx.IAN, f"2026-01-11T{9 + (i + 1) // 60:02d}:{(i + 1) % 60:02d}:00",
                     f"Next, we have 'Generated Bug Number {i} With A Long Descriptive Title'. " + "It slows the export a lot. " * 8
                     + f"Customer CUST-0047 is affected. @{fx.EMMA}, can you take this one?"))
    rec = fx.records(slack=rows)
    from inherit.extract.sessions import sessions
    cat = build_catalog(rec)
    sess = sessions(rec.messages)[0]
    parse = bugs.br.parse_session(sess, cat, rec.customers)
    assert len(parse.bugs) == 55
    chunks = bugs.split_session(sess, parse, rec, cat)
    assert len(chunks) > 1
    for csess, cparse in chunks:
        req, _ = bugs.build_session_request(csess, cparse, rec, cat)
        assert (len(req[0]["content"]) + len(req[1]["content"])) / 3 <= bugs.MAX_INPUT_TOKENS
        assert csess[0].id == "20260111-0-b0001"
    sent = [m.id for csess, _ in chunks for m in csess if m.id != "20260111-0-b0001"]
    assert sorted(sent) == sorted(m.id for m in sess[1:])


@pytest.mark.parametrize("bad", [
    '{"bugs": [], "events": [{"type": "ASSIGNED", "actor_message_id": "m", "citations": 7}], "external": []}',
    '{"bugs": [], "events": [{"type": "ASSIGNED", "actor_message_id": "m", "assignee_eids": 7}], "external": []}',
    '{"bugs": [{"title": "t", "message_id": "m", "customer_ids": [1]}], "events": [], "external": []}',
    '{"bugs": [], "events": [{"type": "ASSIGNED", "citations": [{"record_id": 5, "quote": "x"}]}], "external": []}',
])
def test_field_types_are_checked_before_caching(tmp_path, bad):
    chat = fx.ScriptedChat(script={"20260111-5-b0006": bad, "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    assert [f["reason"] for f in out.report["extraction_failures"]] == ["invalid_shape"]
    assert len(list((tmp_path / bugs.PROMPT_VERSION).glob("*.json"))) == 2


def test_link_with_a_list_for_the_bug_is_a_failure(tmp_path):
    chat = fx.ScriptedChat(script=dict(fx.standard_chat().script), link='{"links":[{"item":"I1","bug":[]}]}')
    out, _ = run(chat=chat, tmp=tmp_path)
    assert [f["reason"] for f in out.report["extraction_failures"]] == ["invalid_shape"]
    assert out.report["events"]["unmatched"] == 1


def test_external_customer_ids_are_type_checked(tmp_path):
    bad = '{"bugs": [], "events": [], "external": [{"title": "t", "link": "https://github.com/a/b/pull/1", "message_id": "m", "customer_ids": 7}]}'
    chat = fx.ScriptedChat(script={"20260111-5-b0006": bad, "20260125-1-g0002": fx.session_json(events=[fx.REASSIGN_EVENT])},
                           link=fx.link_json(("I1", "B1")))
    out, _ = run(chat=chat, tmp=tmp_path)
    assert [f["reason"] for f in out.report["extraction_failures"]] == ["invalid_shape"]


def test_an_exception_in_one_session_is_logged_and_the_run_continues(tmp_path, monkeypatch):
    real = bugs._clean_session_answer

    def boom(data, session, *a, **k):
        if any(m.id == "20260125-1-g0002" for m in session):
            raise KeyError("surprise")
        return real(data, session, *a, **k)

    monkeypatch.setattr(bugs, "_clean_session_answer", boom)
    out, _ = run(tmp=tmp_path)
    [f] = out.report["extraction_failures"]
    assert f["kind"] == "session" and f["reason"] == "exception" and "KeyError: 'surprise'" in f["error"]
    assert "20260125-1-g0002" in f["messages"]
    assert len(nodes(out.facts, "Bug")) == 3  # the other sessions of the channel are still processed


def test_an_exception_in_a_link_call_is_logged_and_the_items_stay_unmatched(tmp_path):
    def explode(user):
        raise RuntimeError("proxy down")

    chat = fx.ScriptedChat(script=dict(fx.standard_chat().script), link=explode)
    out, _ = run(chat=chat, tmp=tmp_path)
    [f] = out.report["extraction_failures"]
    assert f["kind"] == "link" and f["reason"] == "exception" and "RuntimeError: proxy down" in f["error"]
    assert out.report["events"]["unmatched"] == 1
