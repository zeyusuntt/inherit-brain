import builtins
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from eval import audit_extraction as ae
from inherit.data.records import build_records

EMP = {"eid_a": {"employee_id": "eid_a", "name": "Ann Lee", "role": "Engineer", "org": "x"},
       "eid_b": {"employee_id": "eid_b", "name": "Bob Ray", "role": "PM", "org": "x"},
       "eid_c": {"employee_id": "eid_c", "name": "Bob Ray", "role": "QA", "org": "x"}}
CUST = [{"id": "CUST-0001", "name": "Cy Dow", "role": "CTO", "company": "Acme"}]
DOC_LINK = "https://docs/design"


def slack(i, text, ts, ch="ch-bug", name="bug-ann-{P}", user="eid_b", P="P1"):
    return {"Channel": {"channelID": ch, "name": name.format(P=P)},
            "Message": {"User": {"userId": user, "utterranceID": i, "timestamp": ts, "text": text}}, "id": i}


def data(P):
    return {
        "slack": [slack("m1", 'Customer CUST-0001 reported "Crash on save". @eid_a can you take it?', "2026-01-01T10:00:00", P=P),
                  slack("m2", "On it.", "2026-01-01T10:05:00", user="eid_a", P=P),
                  slack("m3", "The crash has been resolved.", "2026-01-01T11:00:00", user="eid_a", P=P),
                  slack("m4", f"See {DOC_LINK}. Ann Lee will merge Fix crash", "2026-02-01T11:00:00", ch="ch-dev", name="develop-x-{P}", P=P)],
        "documents": [{"id": "design_v1", "author": "eid_a", "type": "Design Doc", "document_link": DOC_LINK,
                       "date": "2026-01-01T00:00:00", "content": "x", "feedback": "- add a diagram"},
                      {"id": "design_v2", "author": "eid_a", "type": "Design Doc", "document_link": DOC_LINK + "2",
                       "date": "2026-01-02T00:00:00", "content": "y"}],
        "meeting_transcripts": [{"id": "mtg1", "date": "2026-01-03T00:00:00", "document_type": "Design Doc",
                                 "participants": ["eid_a", "eid_b"],
                                 "transcript": "Attendees\nAnn Lee, Bob Ray\nTranscript\nAnn Lee: Ship it.\nBob Ray: Add examples."}],
        "meeting_chats": [], "urls": [],
        "prs": [{"id": "pr1", "title": "Fix crash", "summary": "s", "link": "https://github.com/o/r/pull/1",
                 "mergeable": "True", "merged": "False", "number": "1", "state": "open",
                 "user": {"login": "eid_a"}, "created_at": "2026-01-01T12:00:00", "reviews": []}],
    }


def ev(P, i, typ, mid, ts, assignees=()):
    return {"id": f"BugEvent:{P}:{i}", "kind": "node", "label": "BugEvent",
            "props": {"assignee_eids": list(assignees), "bug": f"Bug:{P}:ch-bug:crash", "channel_id": "ch-bug",
                      "message_id": mid, "ts": ts, "type": typ, "fix_pr_links": []}}


def facts(P, drop=()):
    bug = {"id": f"Bug:{P}:ch-bug:crash", "kind": "node", "label": "Bug",
           "props": {"title": "Crash on save", "channel_name": f"bug-ann-{P}", "component": "Core", "active": False,
                     "status": "RESOLVED", "customer_ids": ["CUST-0001"], "current_assignees": ["eid_a"],
                     "resolved": True, "reopened": False, "no_fix_needed": False}}
    events = [ev(P, 0, "REPORTED", "m1", "2026-01-01T10:00:00"), ev(P, 1, "ASSIGNED", "m1", "2026-01-01T10:00:00", ["eid_a"]),
              ev(P, 2, "RESOLVED", "m3", "2026-01-01T11:00:00")]
    acted = [{"kind": "edge", "type": "ACTED_BY", "src_id": e["id"], "dst_id": "Person:eid_b", "src": ["m1"], "props": {}}
             for e in events]
    sug = [{"kind": "edge", "type": "SUGGESTED", "src_id": "Person:eid_b", "dst_id": f"Document:{d}", "src": ["mtg1#5"],
            "key": "mtg1#5", "props": {"kind": "suggestion", "section": "s", "text": "add examples",
                                       "version_ambiguous": True}} for d in ("design_v1", "design_v2")]
    names = [{"name": "Bob Ray", "meeting_id": "mtg1", "status": "ambiguous", "candidates": ["eid_b", "eid_c"],
              "ids": ["eid_b", "eid_c"]},
             {"name": "Ann Lee", "meeting_id": "mtg1", "status": "resolved", "candidates": ["eid_a"], "ids": ["eid_a"]}]
    return {"bugs": [bug] + [e for e in events if e["props"]["type"] not in drop] + acted, "semantic": sug,
            "names_facts": [], "structural": [],
            "rules": [{"id": "PR:https://github.com/o/r/pull/1", "kind": "node", "label": "PR",
                       "props": {"title": "Fix crash"}}]}, names


def source(P, drop=()):
    f, names = facts(P, drop)
    return ae.Source(P, build_records(data(P), EMP, CUST, P), f, names)


@pytest.fixture
def sources():
    return {"P1": source("P1"), "P2": source("P2")}


def wi_fn(P):
    pr = "PR:https://github.com/o/r/pull/1"
    bug = f"Bug:{P}:ch-bug:crash"
    return {"newly_unowned": [{"person": "eid_a", "item": pr, "label": "PR", "title": "Fix crash",
                               "assignment": {"kind": "opened", "records": ["pr1"], "message_id": None, "text": None},
                               "later_context": [{"id": "m4"}]}],
            "dependency_edges": [{"dependent": bug, "prerequisite": pr, "rule": "FIXES", "citations": ["pr1", "m3"], "texts": []}],
            "affected_paths": [{"person": "eid_a", "affected": bug, "path": [bug, pr],
                                "edges": [{"dependent": bug, "prerequisite": pr, "rule": "FIXES", "citations": ["pr1"]}]}],
            "shortages": {}}


SMALL = {"bug_event": 4, "review_relation": 3, "name_resolution": 3, "whatif_unowned": 30, "whatif_edge": 30,
         "whatif_path": 20, "recall_bug": 2, "recall_reviewer": 2}


def test_sampling_is_deterministic_per_seed_and_round(sources):
    a, _ = ae.build_packets(sources, 1, wi_fn, SMALL)
    b, _ = ae.build_packets(sources, 1, wi_fn, SMALL)
    assert [p["id"] for p in a] == [p["id"] for p in b]
    c, _ = ae.build_packets(sources, 2, wi_fn, SMALL)
    assert ae.rng_for(1, "bug_event").random() != ae.rng_for(2, "bug_event").random()
    assert {p["id"] for p in c}  # a fresh round still builds


def test_water_filling_quota_and_census():
    pool = [(f"a{i}", "A", i) for i in range(10)] + [(f"b{i}", "B", i) for i in range(2)] + [("c0", "C", 0)]
    picked, info = ae.sample_strata(pool, 7, random.Random(1))
    assert Counter(x[1] for x in picked) == {"A": 4, "B": 2, "C": 1}  # small strata capped, remainder to the big one
    allp, info = ae.sample_strata(pool, 50, random.Random(1))
    assert len(allp) == len(pool) and info == {"requested": 50, "available": 13}  # census, never padded
    assert len({x[0] for x in allp}) == len(allp)


def test_quotas_census_and_actual_sizes(sources):
    packets, sizes = ae.build_packets(sources, 1, wi_fn, SMALL)
    assert sizes["bug_event"]["actual"] == 4 and sizes["bug_event"]["available"] == 6
    assert sizes["whatif_unowned"] == {"requested": 30, "available": 2, "actual": 2}
    assert sizes["recall_bug"]["actual"] == 2 and sizes["recall_reviewer"]["actual"] == 2
    assert {p["product"] for p in packets if p["kind"] == "bug_event"} == {"P1", "P2"}


def test_frozen_quotas():
    assert ae.PRECISION_QUOTAS == {"bug_event": 50, "review_relation": 50, "name_resolution": 50,
                                   "whatif_unowned": 30, "whatif_edge": 30, "whatif_path": 20}
    assert ae.RECALL_QUOTAS == {"recall_bug": 3, "recall_reviewer": 3}


def test_recall_packets_show_no_extraction(sources):
    packets, _ = ae.build_packets(sources, 1, wi_fn, SMALL)
    rec = [p for p in packets if ae.is_recall(p["kind"])]
    assert rec and all("claim" not in p and "task" in p for p in rec)
    for p in rec:
        blob = json.dumps(p)
        assert "Crash on save" not in p["task"] and "SUGGESTED" not in blob and "ACTED_BY" not in blob
        assert "version_ambiguous" not in blob and "Bug:P" not in blob and "Document:" not in blob
    bug = next(p for p in rec if p["kind"] == "recall_bug")
    assert {r["id"] for r in bug["records"]} == {"m1", "m2", "m3"}  # the complete channel
    revw = next(p for p in rec if p["kind"] == "recall_reviewer")
    ids = {r["id"] for r in revw["records"]}
    assert {"mtg1#5", "m4"} <= ids and any(i.startswith("design_v") for i in ids)


def test_precision_packets_carry_claim_and_raw_ids(sources):
    packets, _ = ae.build_packets(sources, 1, wi_fn, SMALL)
    kinds = {p["kind"] for p in packets}
    assert kinds == set(ae.KINDS)
    for p in packets:
        assert p["text"].startswith("PACKET ") and p["records"]
        if not ae.is_recall(p["kind"]):
            assert p["claim"]
    ev_ = next(p for p in packets if p["kind"] == "bug_event")
    assert "joint" not in ev_["text"] and "customer(s) of the bug: CUST-0001" in ev_["text"]
    rev = next(p for p in packets if p["kind"] == "review_relation")
    assert "document TYPE" in rev["text"] and "Bob Ray" in rev["text"]
    un = next(p for p in packets if p["kind"] == "whatif_unowned")
    assert "state open" in un["text"] and "[m4]" in un["text"]  # PR hydrated from the bundle, later context too


def test_packets_never_read_questions_or_gold(sources, monkeypatch):
    seen = []
    real_open, real_read = builtins.open, Path.read_text
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (seen.append(str(f)), real_open(f, *a, **k))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (seen.append(str(self)), real_read(self, *a, **k))[1])
    packets, _ = ae.build_packets(sources, 1, wi_fn, SMALL)
    assert not [s for s in seen if "question" in s.lower() or "gold" in s.lower()]
    assert not any(w in p["text"].lower() for p in packets for w in ("answerable_questions", "gold"))
    assert "eval.questions" not in ae.__dict__.get("__doc__", "") and not hasattr(ae, "load_questions")


AUD = {w: {"model": "m", "version": "1", "settings": "default", "prompt_hash": "h"} for w in ("codex", "claude", "third")}


def write_round(tmp_path, sources, rnd=1, auditors=True):
    packets, sizes = ae.build_packets(sources, rnd, wi_fn, SMALL)
    ae.write_round(packets, sizes, rnd, "bundlesha", tmp_path)
    if auditors:
        (tmp_path / f"round-{rnd}" / "auditors.json").write_text(json.dumps(AUD))
    return packets


def test_manifest_hashes_and_batches(tmp_path, sources):
    packets = write_round(tmp_path, sources)
    m = json.loads((tmp_path / "round-1" / "manifest.json").read_text())
    assert len(m["packets"]) == len(packets) and m["seed"] == ae.BASE_SEED + 1
    f = tmp_path / "round-1" / "packets" / f"{m['packets'][0]['id']}.json"
    assert ae.sha256_text(f.read_text()) == m["packets"][0]["sha256"]
    assert m["prompt_hashes"]["version"] == ae.PROMPT_VERSION and m["shortages"]
    names = ae.write_batches(1, 3, tmp_path)
    plan = json.loads((tmp_path / "round-1" / "batches" / "index.json").read_text())
    assert sorted(names) == sorted(plan) and sum(len(v) for v in plan.values()) == len(packets)
    assert max(len(v) for v in plan.values()) <= 3
    first = (tmp_path / "round-1" / "batches" / f"{names[0]}.txt").read_text()
    assert "Output only" in first and "==== " in first


@pytest.mark.parametrize("a,b,t,final,route", [
    ("correct", "correct", None, "correct", "agree"),
    ("incorrect", "incorrect", None, "incorrect", "agree"),
    ("correct", "incorrect", "correct", "correct", "third"),
    ("correct", "incorrect", "incorrect", "incorrect", "third"),
    ("uncertain", "correct", "incorrect", "incorrect", "third"),
    (None, "correct", "correct", "correct", "third"),
    ("correct", None, None, None, "pending"),
    ("correct", "incorrect", "uncertain", "unresolved", "third_uncertain"),
])
def test_truth_table(a, b, t, final, route):
    r = ae.resolve(a, b, t)
    assert (r["final"], r["route"]) == (final, route) and r["needs_third"] == (route == "pending")


def put(tmp_path, who, name, rows, rnd=1):
    d = tmp_path / f"round-{rnd}" / "verdicts" / who
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(rows))


def v(pid, verdict, fields=None):
    out = {"packet_id": pid, "verdict": verdict, "reason": "see m1"}
    if fields is None and pid.startswith("bug_event-"):
        fields = dict.fromkeys(ae.BUG_FIELDS, "correct") | ({"identity": "incorrect"} if verdict == "incorrect" else {})
    return out | ({"fields": fields} if fields else {})


def test_validation_rejects_malformed(sources, tmp_path):
    packets = write_round(tmp_path, sources)
    bug = next(p for p in packets if p["kind"] == "bug_event")
    assert ae.validate_verdict(v(bug["id"], "correct"), bug) is None
    assert ae.validate_verdict(v(bug["id"], "maybe"), bug)
    assert ae.validate_verdict({"packet_id": bug["id"], "verdict": "correct"}, bug)  # no reason
    assert ae.validate_verdict(v(bug["id"], "correct", {**dict.fromkeys(ae.BUG_FIELDS, "correct"), "time": "incorrect"}), bug)
    assert ae.validate_verdict(v("other", "correct"), bug)
    rb = next(p for p in packets if p["kind"] == "recall_bug")
    assert ae.validate_verdict({"packet_id": rb["id"], "items": [{"bug": "m1", "type": "NOPE", "actor": "a", "time": "t", "message_id": "m"}]}, rb)


def fill(tmp_path, packets, third=True):
    """Auditors agree on everything except one disagreement; recall lists are exact."""
    prec = [p for p in packets if not ae.is_recall(p["kind"])]
    put(tmp_path, "codex", "a", [v(p["id"], "correct") for p in prec])
    put(tmp_path, "claude", "a", [v(p["id"], "correct") for p in prec[1:]] + [v(prec[0]["id"], "incorrect")])
    if third:
        put(tmp_path, "third", "a", [v(prec[0]["id"], "correct")])
    return prec


def recall_items(packet, src):
    if packet["kind"] == "recall_bug":
        return [{"bug": "m1", "type": t, "actor": "eid_b", "time": "x", "message_id": m} for t, m in
                [("REPORTED", "m1"), ("ASSIGNED", "m1"), ("RESOLVED", "m3")]]
    return [{"reviewer": "Bob Ray", "source": "mtg1#5"}]


def test_collect_third_pass_and_missing(tmp_path, sources):
    packets = write_round(tmp_path, sources)
    prec = fill(tmp_path, packets, third=False)
    need = ae.pending_third(1, tmp_path)
    assert prec[0]["id"] in need
    names = ae.write_third(1, 10, tmp_path)
    t = "\n".join((tmp_path / "round-1" / "third_batches" / f"{n}.txt").read_text() for n in names)
    assert "auditor 1: correct" in t and "auditor 2: incorrect" in t
    s = ae.collect(1, sources, tmp_path)
    assert not s["complete"] and s["pending_third"] == len(need)
    put(tmp_path, "third", "a", [v(prec[0]["id"], "correct")])
    for p in packets:
        if ae.is_recall(p["kind"]):
            for w in ("codex", "claude", "third"):
                put(tmp_path, w, "r" + p["id"], [{"packet_id": p["id"], "items": recall_items(p, None)}])
    assert ae.pending_third(1, tmp_path) == []
    s = ae.collect(1, sources, tmp_path)
    assert s["complete"]
    cat = s["precision"][prec[0]["kind"]]
    assert cat["precision"] == 1.0 and cat["to_third"] >= 1 and cat["third_decided"] >= 1
    assert cat["double_positive"] == cat["n"] - 1 and cat["agreement_before_adjudication"] < 1
    report = (tmp_path / "round-1" / "report.md").read_text()
    assert "Crash on save" not in report and "sample hash" in report and "rubric:bug_event" in report


def test_unusable_answers_count_as_missing(tmp_path, sources):
    packets = write_round(tmp_path, sources)
    p = next(p for p in packets if p["kind"] == "name_resolution")
    put(tmp_path, "codex", "a", [v(p["id"], "correct")])
    put(tmp_path, "claude", "a", [{"packet_id": p["id"], "verdict": "correct"}])  # malformed: no reason
    assert p["id"] in ae.pending_third(1, tmp_path)
    put(tmp_path, "third", "a", [v(p["id"], "uncertain")])
    s = ae.collect(1, sources, tmp_path)
    assert s["precision"]["name_resolution"]["unresolved"] == 1
    assert s["verdict_file_stats"]["claude"]["malformed"] == 1


def test_recall_reconciliation_and_comparison(tmp_path):
    srcs = {"P1": source("P1", drop=("RESOLVED",)), "P2": source("P2")}
    packets, sizes = ae.build_packets(srcs, 1, wi_fn, {**SMALL, "recall_bug": 1, "recall_reviewer": 1})
    ae.write_round(packets, sizes, 1, None, tmp_path)
    (tmp_path / "round-1" / "auditors.json").write_text(json.dumps(AUD))
    rb = next(p for p in packets if p["kind"] == "recall_bug")
    rr = next(p for p in packets if p["kind"] == "recall_reviewer")
    truth = recall_items(rb, None)
    # auditors disagree (claude misses one); the third pass reconciles to the full list; always needs the third pass
    put(tmp_path, "codex", "r", [{"packet_id": rb["id"], "items": truth}, {"packet_id": rr["id"], "items": recall_items(rr, None)}])
    put(tmp_path, "claude", "r", [{"packet_id": rb["id"], "items": truth[:2]}, {"packet_id": rr["id"], "items": recall_items(rr, None)}])
    assert set(ae.pending_third(1, tmp_path)) >= {rb["id"], rr["id"]}
    third = ae.write_third(1, 10, tmp_path)
    assert any("recall" in n for n in third)
    put(tmp_path, "third", "r", [{"packet_id": rb["id"], "items": truth}, {"packet_id": rr["id"], "items": recall_items(rr, None)}])
    s = ae.collect(1, srcs, tmp_path)
    r = s["recall"]["recall_bug"]
    miss_p1 = rb["product"] == "P1"
    assert r["truth"] == 3 and r["matched"] == (2 if miss_p1 else 3)
    assert r["lists_equal"] == 0
    assert s["recall"]["recall_reviewer"]["recall"] == 1.0
    assert ("m1|RESOLVED|m3" in r["detail"][0]["missed"]) == miss_p1


def test_recall_keys_normalise_names(sources):
    src = sources["P1"]
    assert ae.truth_keys("recall_reviewer", [{"reviewer": "bob ray ", "source": "x"}, {"reviewer": "eid_b", "source": "x"}], src) \
        == Counter({("bob ray", "x"): 1})


def test_gates(tmp_path):
    def summ(prec=0.92, recall=0.9, complete=True):
        return {"complete": complete, "pending_third": 0 if complete else 2,
                "precision": {k: {"n": 50, "correct": int(50 * prec), "precision": prec, "pending": 0}
                              for k in ae.PRECISION_QUOTAS},
                "recall": {k: {"truth": 20, "matched": int(20 * recall), "recall": recall, "pending": 0}
                           for k in ae.RECALL_QUOTAS}}
    assert all(ok for _, ok, _ in ae.gates(summ()))
    bad = dict((g, ok) for g, ok, _ in ae.gates(summ(prec=0.89, recall=0.84)))
    assert not any(bad[g] for g in bad if g != "complete")
    assert not dict((g, ok) for g, ok, _ in ae.gates(summ(complete=False)))["complete"]
    edge = summ()
    edge["precision"]["bug_event"]["precision"] = 0.9  # exactly at the line passes
    assert dict((g, ok) for g, ok, _ in ae.gates(edge))["precision:bug_event"]
    # CLI exit codes
    d = tmp_path / "round-3"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(summ(prec=0.5)))
    assert ae.main(["gates", "--round", "3"], root=tmp_path) == 1
    (d / "summary.json").write_text(json.dumps(summ()))
    assert ae.main(["gates", "--round", "3"], root=tmp_path) == 1  # no manifest or verdicts on disk: not complete
    assert ae.main(["gates", "--round", "4"], root=tmp_path) == 1


def test_cli_end_to_end(tmp_path, sources, capsys):
    assert ae.main(["packets", "--round", "1"], root=tmp_path, whatif_fn=wi_fn, sources=sources) == 0
    assert ae.main(["batches", "--round", "1", "--size", "10"], root=tmp_path) == 0
    assert ae.main(["third", "--round", "1"], root=tmp_path) == 0
    assert ae.main(["collect", "--round", "1"], root=tmp_path, sources=sources) == 0
    out = capsys.readouterr().out
    assert "complete: False" in out


# ---------------------------------------------------------------- fix round 1
def source_extra(P, n_extra=0):
    d = data(P)
    d["slack"] += [slack(f"x{i:03d}", f"more about Fix crash {i}", f"2026-03-01T10:{i % 60:02d}:00", ch="ch-dev",
                         name="develop-x-{P}", P=P) for i in range(n_extra)]
    d["slack"].append(slack("late1", "Bob Ray: the Crash on save for Acme keeps coming back, see "
                            "https://github.com/o/r/pull/1", "2026-04-01T10:00:00", P=P))
    f, names = facts(P)
    f["bugs"][0]["props"].update(channel_id="ch-bug", companies=["Acme"], aliases=[])
    f["bugs"][2]["props"]["fix_pr_links"] = ["https://github.com/o/r/pull/1"]
    return ae.Source(P, build_records(d, EMP, CUST, P), f, names)


def test_later_context_is_complete_and_uncapped():
    src = source_extra("P1", 150)
    pr = "PR:https://github.com/o/r/pull/1"
    x = wi_fn("P1")["newly_unowned"][0]
    p = ae._whatif_unowned_packet("k", src, x)
    ids = {r["id"] for r in p["records"]}
    assert {f"x{i:03d}" for i in range(150)} <= ids and "m4" in ids  # no 60 message cut
    assert p["meta"]["later_messages"] >= 151 and p["meta"]["chars"] == len(p["text"]) and "later_context_cut" not in p["meta"]
    assert not hasattr(ae, "MAX_LATER") and pr in p["text"]
    times = [r["text"].split()[0] for r in p["records"] if r["id"].startswith("x")]
    assert times == sorted(times)


def test_bug_context_is_independent_of_extraction():
    src = source_extra("P1")
    ids = {r["id"] for r in ae.bug_context_records(src, "Bug:P1:ch-bug:crash", "2026-01-01T10:00:00")}
    assert {"m1", "m3"} <= ids
    # a PR linked on the RESOLVED event is hydrated for every event of the bug, not only the sampled one
    assert "pr1" in ids


def test_bug_fields_are_all_required_and_joint():
    src = sources_for_validation()
    bug = next(p for p in src if p["kind"] == "bug_event")
    ok = dict.fromkeys(ae.BUG_FIELDS, "correct")
    chk = lambda verdict, f: ae.validate_verdict(  # noqa: E731
        {"packet_id": bug["id"], "verdict": verdict, "reason": "r", **({"fields": f} if f is not None else {})}, bug)
    assert chk("correct", ok) is None
    assert chk("correct", None)  # missing fields are malformed
    assert chk("correct", {k: "correct" for k in ae.BUG_FIELDS[:4]})  # one field missing
    assert chk("correct", {**ok, "time": "na"})  # na where the claim has a value
    assert chk("correct", dict.fromkeys(ae.BUG_FIELDS, "na"))  # the vacuous pass
    assert chk("correct", {**ok, "status": "incorrect"})
    assert chk("incorrect", {**ok, "status": "incorrect"}) is None
    assert chk("incorrect", ok)  # incorrect with no failing field is inconsistent


def test_na_allowed_only_where_claim_is_empty():
    src = source("P1")
    f, names = facts("P1")
    f["bugs"][0]["props"]["customer_ids"] = []
    s2 = ae.Source("P1", src.records, f, names)
    packets, _ = ae.bug_event_packets({"P1": s2}, 1, random.Random(1))
    p = packets[0]
    assert p["claim"]["customer(s) of the bug"] == ae.NOT_EXTRACTED
    f_ = dict.fromkeys(ae.BUG_FIELDS, "correct") | {"customer": "na"}
    assert ae.validate_verdict({"packet_id": p["id"], "verdict": "correct", "reason": "r", "fields": f_}, p) is None
    assert ae.validate_verdict({"packet_id": p["id"], "verdict": "correct", "reason": "r",
                                "fields": f_ | {"identity": "na"}}, p)


def sources_for_validation():
    packets, _ = ae.build_packets({"P1": source("P1"), "P2": source("P2")}, 1, wi_fn, SMALL)
    return packets


def test_recall_bug_identity_and_multiplicity(sources):
    packets, _ = ae.build_packets(sources, 1, wi_fn, SMALL)
    rb = next(p for p in packets if p["kind"] == "recall_bug")
    src = sources[rb["product"]]
    ex = ae.extracted_keys(rb, src)
    assert ex == Counter({("m1", "REPORTED", "m1"): 1, ("m1", "ASSIGNED", "m1"): 1, ("m1", "RESOLVED", "m3"): 1})
    wrong = ae.truth_keys("recall_bug", [{"bug": "m2", "type": "RESOLVED", "actor": "a", "time": "t", "message_id": "m3"}], src)
    assert ae.compare_recall(wrong, ex)["matched"] == 0  # an event of another bug is not recalled
    twice = Counter({("m1", "ASSIGNED", "m1"): 2})
    assert ae.compare_recall(twice, ex)["matched"] == 1  # one message may carry several events
    assert ae.validate_verdict({"packet_id": rb["id"], "items": [{"type": "REPORTED", "actor": "a", "time": "t",
                                                                  "message_id": "m1"}]}, rb)  # bug is required


def test_collect_validates_the_exact_manifest_set(tmp_path, sources):
    packets = write_round(tmp_path, sources)
    assert ae.integrity_problems(1, tmp_path) == []
    victim = tmp_path / "round-1" / "packets" / f"{packets[0]['id']}.json"
    other = tmp_path / "round-1" / "packets" / f"{packets[1]['id']}.json"
    other.write_text(other.read_text() + " ")
    victim.unlink()
    probs = ae.integrity_problems(1, tmp_path)
    assert any("missing" in p for p in probs) and any("changed" in p for p in probs)
    s = ae.collect(1, sources, tmp_path)
    assert not s["complete"] and s["integrity_problems"]
    assert not dict((g, ok) for g, ok, _ in ae.gates(s))["complete"]


def test_round_is_never_overwritten(tmp_path, sources):
    write_round(tmp_path, sources)
    with pytest.raises(FileExistsError):
        write_round(tmp_path, sources)
    assert ae.main(["packets", "--round", "1"], root=tmp_path, whatif_fn=wi_fn, sources=sources) == 2


def test_name_packet_publishes_rule_and_candidate_evidence(sources):
    packets, _ = ae.build_packets(sources, 1, wi_fn, {**SMALL, "name_resolution": 6})
    amb = next(p for p in packets if p["kind"] == "name_resolution" and p["meta"]["status"] == "ambiguous")
    ids = {r["id"] for r in amb["records"]}
    assert {"candidate:eid_b", "candidate:eid_c"} <= ids
    assert "returned" in amb["text"] and "eligible candidates the extractor found" in amb["text"]
    assert "3 if the speaker" in ae.RUBRICS["name_resolution"] and "tie" in ae.RUBRICS["name_resolution"]
    cand = next(r for r in amb["records"] if r["id"] == "candidate:eid_b")["text"]
    assert "directory role PM" in cand and "In the meeting's participants: yes" in cand


def test_segment_records_carry_meeting_header_and_neighbours(sources):
    t = ae.record_text(sources["P1"], "mtg1#5")["text"]
    assert "document type Design Doc" in t and "date 2026-01-03" in t and "line 4" in t and "Ship it." in t


def test_rubrics_state_the_exact_predicates():
    for k in ("whatif_unowned", "whatif_path"):
        r = ae.RUBRICS[k]
        assert "REPORTED, ASSIGNED or REOPENED" in r and "reopened bug is active" in r
        assert 'state is "open" and merged is "False"' in r and "not implemented" in r and "development" in r
    assert "use rarely" not in ae.COMMON_PRECISION and "contradict the claim" in ae.COMMON_PRECISION


@pytest.mark.parametrize("raw", [{"verdicts": None}, {"packet_id": ["x"], "verdict": "correct"}, "text", 5,
                                 [{"packet_id": 3}], {"verdicts": {"a": 1}}, [None, "x"]])
def test_malformed_envelopes_do_not_crash(tmp_path, sources, raw):
    packets = write_round(tmp_path, sources)
    put(tmp_path, "codex", "bad", raw)
    good, stats = ae.load_verdicts(tmp_path / "round-1" / "verdicts" / "codex", ae.load_packets(1, tmp_path))
    assert good == {} and stats.get("malformed", 0) >= 1


def test_batches_are_bounded_by_packets_and_characters():
    mk = lambda i, n: {"id": f"bug_event-{i}", "kind": "bug_event", "text": "x" * n}  # noqa: E731
    pk = {p["id"]: p for p in [mk(i, 40_000) for i in range(5)] + [mk(9, 500_000)]}
    plan = ae._batch_plan(pk, 10)
    assert all(len(v) <= 3 or len(v) == 1 for v in plan.values())
    assert ["bug_event-9"] in plan.values()  # an oversized packet goes alone
    assert sum(len(v) for v in plan.values()) == 6
    assert all(sum(len(pk[i]["text"]) for i in v) <= ae.BATCH_MAX_CHARS or len(v) == 1 for v in plan.values())


def test_auditor_metadata_is_required_and_reported(tmp_path, sources):
    packets = write_round(tmp_path, sources, auditors=False)
    assert sum("auditor metadata" in p for p in ae.integrity_problems(1, tmp_path)) == 3
    (tmp_path / "round-1" / "auditors.json").write_text(json.dumps(AUD))
    assert ae.integrity_problems(1, tmp_path) == []
    s = ae.collect(1, sources, tmp_path)
    rep = (tmp_path / "round-1" / "report.md").read_text()
    assert "third pass shares a model family with auditor 1" in rep and "codex: model m" in rep
    assert "batch holds only packets of one kind" in rep


def test_gates_output_names_its_scope(tmp_path, capsys):
    d = tmp_path / "round-1"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"complete": True, "pending_third": 0, "precision": {}, "recall": {}}))
    ae.main(["gates", "--round", "1"], root=tmp_path)
    assert "audit gates only; the dev score gate is in the acceptance report" in capsys.readouterr().out


def test_gates_report_is_the_acceptance_interface(tmp_path, sources):
    packets = write_round(tmp_path, sources)
    prec = [p for p in packets if not ae.is_recall(p["kind"])]
    put(tmp_path, "codex", "a", [v(p["id"], "correct") for p in prec])
    put(tmp_path, "claude", "a", [v(p["id"], "correct") for p in prec])
    for p in packets:
        if ae.is_recall(p["kind"]):
            for w in ("codex", "claude", "third"):
                put(tmp_path, w, "r" + p["id"], [{"packet_id": p["id"], "items": recall_items(p, None)}])
    s = ae.collect(1, sources, tmp_path)
    rd = tmp_path / "round-1"
    rep = ae.gates_report(rd)
    assert set(rep) == {"complete", "reasons", "gates", "pass", "auditors", "prompt_hashes", "sample_hash", "summary_sha256", "report_md"}
    assert rep["complete"] and rep["pass"] and len(rep["gates"]) == 8
    assert set(rep["gates"][0]) == {"name", "numerator", "denominator", "threshold", "pass"}
    assert rep["auditors"] == AUD and rep["sample_hash"] == s["sample_hash"]
    assert rep["summary_sha256"] == ae.sha256_text((rd / "summary.json").read_text()) and Path(rep["report_md"]).exists()
    assert ae.main(["gates", "--round", "1"], root=tmp_path) == 0
    assert ae.gates_report(tmp_path / "round-9") ["pass"] is False
    (rd / "packets" / f"{packets[0]['id']}.json").unlink()
    ae.collect(1, sources, tmp_path)
    assert ae.gates_report(rd)["complete"] is False and ae.gates_report(rd)["pass"] is False


# ---------------------------------------------------------------- fix round 2
def passing_round(tmp_path, sources):
    packets = write_round(tmp_path, sources)
    prec = [p for p in packets if not ae.is_recall(p["kind"])]
    put(tmp_path, "codex", "a", [v(p["id"], "correct") for p in prec])
    put(tmp_path, "claude", "a", [v(p["id"], "correct") for p in prec])
    for p in packets:
        if ae.is_recall(p["kind"]):
            for w in ("codex", "claude", "third"):
                put(tmp_path, w, "r" + p["id"], [{"packet_id": p["id"], "items": recall_items(p, None)}])
    # one disagreement settled by the third pass, so a third verdict file is required
    put(tmp_path, "claude", "a", [v(prec[0]["id"], "incorrect")] + [v(p["id"], "correct") for p in prec[1:]])
    put(tmp_path, "third", "a", [v(prec[0]["id"], "correct")])
    ae.collect(1, sources, tmp_path)
    return packets, tmp_path / "round-1"


def test_gates_report_recomputes_completeness(tmp_path, sources):
    packets, rd = passing_round(tmp_path, sources)
    assert ae.gates_report(rd)["pass"] is True

    def run(mutate):
        import shutil
        d = tmp_path / "copy"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(tmp_path / "round-1", d / "round-1")
        mutate(d / "round-1")
        return ae.gates_report(d / "round-1")

    pdir = lambda r: r / "packets"  # noqa: E731
    first = f"{packets[0]['id']}.json"

    def delete_packet(r): (pdir(r) / first).unlink()

    def change_packet(r): (pdir(r) / first).write_text((pdir(r) / first).read_text() + " ")

    def extra_packet(r): (pdir(r) / "stray-1.json").write_text("{}")

    def delete_primary(r): next((r / "verdicts" / "claude").glob("*.json")).unlink()

    def delete_third(r): next((r / "verdicts" / "third").glob("a.json")).unlink()

    def delete_auditors(r): (r / "auditors.json").unlink()

    def edit_verdict(r):
        f = r / "verdicts" / "codex" / "a.json"
        f.write_text(f.read_text() + " ")

    for mut in (delete_packet, change_packet, extra_packet, delete_primary, delete_third, delete_auditors, edit_verdict):
        rep = run(mut)
        assert rep["complete"] is False and rep["pass"] is False and rep["reasons"], mut.__name__
        assert any("re-run collect" in x for x in rep["reasons"])


def test_recall_prompt_example_parses_and_matches(sources):
    import re
    packets, _ = ae.build_packets(sources, 1, wi_fn, SMALL)
    rb = next(p for p in packets if p["kind"] == "recall_bug")
    ex = re.search(r"Example item: (\{.*\})", ae.RECALL_RULES["recall_bug"]).group(1)
    item = json.loads(ex)
    assert "bug_title" in item and ae.validate_verdict({"packet_id": rb["id"], "items": [item]}, rb) is None
    src = sources[rb["product"]]
    key = next(iter(ae.truth_keys("recall_bug", [item], src)))
    assert key == (item["bug"], item["type"], item["message_id"])
    assert "never a title" in ae.RECALL_RULES["recall_bug"]


def test_report_discloses_shared_report_messages_and_anaphora(tmp_path, sources):
    src = sources["P1"]
    extra = dict(ev("P1", 9, "REPORTED", "m1", "2026-01-01T10:00:00"))
    extra["props"] = {**extra["props"], "bug": "Bug:P1:ch-bug:other"}
    src.facts["bugs"].append(extra)
    assert ae.shared_report_messages(src, "ch-bug") == 1
    assert ae.shared_report_messages(sources["P2"], "ch-bug") == 0
    packets, rd = passing_round(tmp_path, sources)
    s = json.loads((rd / "summary.json").read_text())
    rep = (rd / "report.md").read_text()
    assert "first-report messages carrying more than one bug" in rep and "anaphora" in rep
    assert "shared_report_messages" in s["recall"]["recall_bug"]


def test_name_candidates_carry_sessions_and_related_evidence():
    d = data("P1")
    d["slack"] += [slack("w1", f"Reviewing the Design Doc today", "2026-01-03T09:00:00", ch="ch-dev", name="develop-x-{P}",
                         user="eid_b"),
                   slack("w2", "ok, will do", "2026-01-03T09:10:00", ch="ch-dev", name="develop-x-{P}", user="eid_a")]
    f, names = facts("P1")
    src = ae.Source("P1", build_records(d, EMP, CUST, "P1"), f, names)
    recs = ae.candidate_records(src, names[0])
    ids = {r["id"] for r in recs}
    assert {"candidate:eid_b", "candidate:eid_c", "w1", "w2"} <= ids  # the full session of the window post
    b = next(r for r in recs if r["id"] == "candidate:eid_b")["text"]
    assert "Related post under the rule: yes" in b and "design doc" in b.lower()
    c = next(r for r in recs if r["id"] == "candidate:eid_c")["text"]
    assert "Related post under the rule: no" in c
    assert "document type 'Design Doc'" in b


# ---------------------------------------------------------------- round 2 rubric (audit-prompts/v2)
def test_v2_version_recorded_and_v1_frozen(tmp_path, sources):
    assert ae.PROMPT_VERSION == "audit-prompts/v2"
    packets = write_round(tmp_path, sources, rnd=2)
    m = json.loads((tmp_path / "round-2" / "manifest.json").read_text())
    assert m["prompt_hashes"]["version"] == "audit-prompts/v2" and m["seed"] == ae.BASE_SEED + 2 == 20261017
    v1 = ae.prompt_hashes("audit-prompts/v1")
    assert v1["rubric:bug_event"] == "77f9e3f494192ddccaeff7d3a337bf4c13bb0c0da7b05972e511357ee1141c5c"  # as in round 1
    assert v1 != ae.prompt_hashes()
    assert ae.batch_header("bug_event", version="audit-prompts/v1") != ae.batch_header("bug_event")
    # a round recorded as v1 keeps its own prompts for batches
    d1 = tmp_path / "round-1"
    write_round(tmp_path, sources, rnd=1)
    mf = json.loads((d1 / "manifest.json").read_text())
    mf["prompt_hashes"] = v1
    (d1 / "manifest.json").write_text(json.dumps(mf))
    ae.write_batches(1, 10, tmp_path)
    assert "use rarely" not in (d1 / "batches" / "bug_event-01.txt").read_text()
    assert "Event model (frozen)" not in (d1 / "batches" / "bug_event-01.txt").read_text()
    ae.write_batches(2, 10, tmp_path)
    assert "Event model (frozen)" in (tmp_path / "round-2" / "batches" / "bug_event-01.txt").read_text()


def test_v2_rubric_texts():
    r = ae.RUBRICS
    assert "carries no assignees by definition" in r["bug_event"]
    assert "separate ASSIGNED event, placed on the message" in r["bug_event"]
    assert "no linked fix PR" in r["bug_event"] and "no fix needed = at least one DISMISSED event ever" in r["bug_event"]
    assert "not extracted" in r["bug_event"] and "judged separately" in r["bug_event"]
    assert "Placement of ASSIGNED events" in ae.RECALL_RULES["recall_bug"]
    assert "OF THIS PRODUCT" in r["review_relation"] and "candidate set" in r["review_relation"]
    assert "Topical association alone" in r["whatif_edge"] and "test and documentation PRs" in r["whatif_edge"]
    assert "necessity" in r["whatif_path"]
    assert ae.RECALL_RULES["recall_bug"].count('"bug": ') == 2  # the format line and the example item; no second definition


def test_reported_event_renders_not_applicable_and_missing_values(sources):
    f, names = facts("P1")
    f["bugs"][0]["props"].update(customer_ids=[], current_assignees=[], component=None)
    src = ae.Source("P1", sources["P1"].records, f, names)
    pk, _ = ae.bug_event_packets({"P1": src}, 6, random.Random(1))
    rep = next(p for p in pk if p["claim"]["event"].startswith("REPORTED"))
    asg = next(p for p in pk if p["claim"]["event"].startswith("ASSIGNED"))
    assert rep["claim"]["assignee(s) of this event"] == ae.NOT_APPLICABLE
    assert asg["claim"]["assignee(s) of this event"] != ae.NOT_APPLICABLE
    assert rep["claim"]["customer(s) of the bug"] == ae.NOT_EXTRACTED and "component not extracted" in rep["claim"]["bug identity"]
    assert "no linked fix PR" in rep["claim"]["bug status (frozen historical flags)"]
    f_ = dict.fromkeys(ae.BUG_FIELDS, "correct") | {"customer": "na", "assignee": "na"}
    ok = lambda p: ae.validate_verdict({"packet_id": p["id"], "verdict": "correct", "reason": "r", "fields": f_}, p)  # noqa: E731
    assert ok(rep) is None  # REPORTED, no customer, no current assignee: both may be na
    assert ok(asg)  # an ASSIGNED event has assignees: na is not allowed


def test_ambiguous_review_candidates_are_rendered(sources):
    src = sources["P1"]
    for eid in ("eid_b", "eid_c"):
        src.facts["semantic"].append({"kind": "edge", "type": "SUGGESTED", "src_id": f"Person:{eid}",
                                      "dst_id": "Document:design_v1", "src": ["mtg1#5"], "key": "amb",
                                      "props": {"kind": "suggestion", "section": "s", "text": "t", "ambiguous": True,
                                                "speaker_name": "Bob Ray", "version_ambiguous": False}})
    pk, info = ae.review_packets({"P1": src}, 100, random.Random(1))
    amb = [p for p in pk if p["claim"].get("ambiguous")]
    assert len(amb) == 1  # one claim for the pair, not one per candidate
    t = amb[0]["text"]
    assert "candidate speakers" in t and "eid_b" in t and "eid_c" in t and "candidate set" in t
    assert amb[0]["meta"]["candidates"] == ["eid_b", "eid_c"]
    assert not any(p["claim"].get("ambiguous") for p in pk if p is not amb[0])


def test_report_lists_prompt_changes(tmp_path, sources):
    packets, rd = passing_round(tmp_path, sources)
    rep = (rd / "report.md").read_text()
    assert "audit-prompts/v2" in rep and "round 1 used audit-prompts/v1" in rep and "Placement" not in rep.split("Prompt changes")[0]
    assert all(c in rep for c in ae.PROMPT_CHANGES)
