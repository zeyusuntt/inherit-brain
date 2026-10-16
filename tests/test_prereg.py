import json
import subprocess

import pytest

from eval import families as F
from eval import grade as G
from eval import prereg as P
from eval import run as R
from eval import stats
from eval.arms import ARMS
from eval.questions import template
from inherit.config import MODEL
from inherit.contracts import QuestionKey

from evalfx import FACTS, make_arm, mk_questions

BUNDLE = "b" * 64
OK = lambda root: []  # noqa: E731
GOOD_REL = {"approved_by": "Zeyu", "approved_utc": "2026-10-20T10:00:00Z", "prereg_sha256": "",
            "post_url": "https://example.org/post", "wayback_url": "https://web.archive.org/web/20261020101010/https://example.org/post"}


def jdump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))


def a5_sha():
    return ARMS["A5"].config_sha()


def mk_manifest(root, **over):
    from inherit import freeze

    arms = {"A5": {"products": ["P1", "P2", "P3"], "sub_cap_usd": 3.0}, "A2": {"products": ["P3"], "sub_cap_usd": None},
            "A0": {"products": ["P1", "P2", "P3"], "sub_cap_usd": None}}
    m = {"code_commit": "c" * 40, "upstream_herb_commit": "u" * 40, "split_hash": "s" * 64, "bundle_sha256": BUNDLE,
         "facts_hash": FACTS, "config_hash": "h" * 64, "products": ["P1", "P2", "P3"], "held_products": ["P3"],
         "arms": arms, "config": freeze.effective_config(arms)}  # the freeze's own schema
    m.update(over)
    jdump(P.manifest_path(root), m)
    return m


def mk_families(root, multi=("how many bugs",), excluded=True, diagnostics=True):
    fams = {t: {"source": "dev", "verdict": "multi"} for t in multi}
    fams["single one"] = {"source": "dev", "verdict": "single"}
    if excluded:
        fams["unclear one"] = {"source": "held-unseen", "codex": "multi", "claude": "single", "verdict": "uncertain"}
    diag = {"voting_instances": 10, "visible_instances": 8, "needs_more_instances": 2, "values_total": 20,
            "values_paging_only": 3, "large_group_or_missing_families": 2, "large_group_or_missing_questions": 5}
    jdump(P.families_path(root), F.build_families_doc(fams, {}, diag if diagnostics else None))


def gates_stub(ok=True, complete=True):
    def f(round_dir):
        return {"complete": complete, "pass": ok and complete,
                "gates": [{"name": "precision:bug", "numerator": 95 if ok else 50, "denominator": 100, "threshold": 0.9,
                           "pass": ok}],
                "auditors": {"auditor1": "codex-x", "auditor2": "claude-y"}, "prompt_hashes": {"a": "1" * 64},
                "sample_hash": "9" * 64, "summary_sha256": "8" * 64, "report_md": "report"}
    return f


def mk_dev_run(root, monkeypatch, *, facts=FACTS, bundle=BUNDLE, cfg=None, arm="A5", score=90.0, products=("P1", "P2"),
               model=None, pv=None, commit="c" * 40):
    qs = mk_questions(products)
    d = root / "runs" / "eval" / "A5-dev"
    jdump(d / "manifest.json", {"run_id": "A5-dev", "arm": arm, "split": "dev", "limit": None, "facts_hash": facts,
                                 "bundle_sha": bundle, "arm_config_sha": cfg or a5_sha(),
                                 "model": model or MODEL, "prompt_versions": pv or ARMS["A5"].prompt_versions,
                                 "code_commit": commit, "slots": [str(q.key) for q in qs]})
    (d / "records.jsonl").write_text("x\n")
    graded = [G.Graded(q.key, arm, q.type, score if q.type in ("person", "company") else 10.0, False) for q in qs]
    monkeypatch.setattr(G, "load_graded", lambda run_dir: graded)
    return d, qs


@pytest.fixture
def world(tmp_path, monkeypatch):
    mk_manifest(tmp_path)
    mk_families(tmp_path)
    jdump(tmp_path / "results" / "freeze" / "whatif_check.json", {"6.0": {"P1": {"mismatches": 0, "identity": "i"}}})
    d, qs = mk_dev_run(tmp_path, monkeypatch)
    return tmp_path, d, qs


def accept(root, d, qs, **kw):
    kw.setdefault("freeze_check", OK)
    kw.setdefault("gates_report", gates_stub())
    kw.setdefault("code_same", lambda root, a, b: True)
    return P.write_acceptance(root, 1, d, questions=qs, dev_products=("P1", "P2"), **kw)


def gate(doc, prefix):
    return next(g for g in doc["gates"] if g["name"].startswith(prefix))


# ---------------------------------------------------------------- acceptance
def test_acceptance_passes_and_records_inputs(world):
    root, d, qs = world
    doc = accept(root, d, qs)
    assert doc["pass"] is True
    on_disk = json.loads(P.acceptance_path(root).read_text())
    assert on_disk["pass"] and on_disk["inputs"]["manifest_sha256"] == P._sha(P.manifest_path(root))
    assert on_disk["inputs"]["families_sha256"] == P._sha(P.families_path(root))
    assert on_disk["inputs"]["dev_records_sha256"] and on_disk["inputs"]["audit_summary_sha256"] == "8" * 64
    assert gate(doc, "dev score")["numerator"] == 90.0 and gate(doc, "audit:precision")["denominator"] == 100


def test_audit_round_report_is_copied(world):
    root, d, qs = world
    rd = root / "runs" / "audit" / "round-1"
    rd.mkdir(parents=True)
    (rd / "report.md").write_text("source free report\n")
    accept(root, d, qs)
    assert (root / "results" / "acceptance" / "audit-round-1.md").read_text() == "source free report\n"


def test_failing_audit_gate_and_incomplete_round_fail(world):
    root, d, qs = world
    assert accept(root, d, qs, gates_report=gates_stub(ok=False))["pass"] is False
    doc = accept(root, d, qs, gates_report=gates_stub(complete=False))
    assert doc["pass"] is False and not gate(doc, "audit:complete")["pass"]


def test_dev_score_below_threshold_fails(world, monkeypatch):
    root, _, _ = world
    d, qs = mk_dev_run(root, monkeypatch, score=60.0)
    doc = accept(root, d, qs)
    assert not doc["pass"] and not gate(doc, "dev score")["pass"] and gate(doc, "dev score")["numerator"] == 60.0


@pytest.mark.parametrize("over", [{"facts": "0" * 64}, {"bundle": "0" * 64}, {"cfg": "0" * 64}, {"arm": "A4"},
                                  {"model": "other/model"}, {"pv": {"agent": "old"}}])
def test_dev_run_not_on_the_frozen_system_fails(world, monkeypatch, over):
    root, _, _ = world
    d, qs = mk_dev_run(root, monkeypatch, **over)
    doc = accept(root, d, qs)
    assert not doc["pass"] and "not on the frozen system" in doc["dev_run"]["why"]


def test_dev_run_from_different_code_or_unknown_commit_fails(world, monkeypatch):
    root, _, _ = world
    d, qs = mk_dev_run(root, monkeypatch)
    doc = accept(root, d, qs, code_same=lambda r, run, frozen: False)
    assert not doc["pass"] and "behaviour code differs" in doc["dev_run"]["why"]
    seen = []
    accept(root, d, qs, code_same=lambda r, run, frozen: seen.append((run, frozen)) or True)
    assert seen == [("c" * 40, "c" * 40)]


def test_code_unchanged_uses_git_diff_and_rejects_dirty_or_unknown(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "1")
    c1 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    (tmp_path / "notes.txt").write_text("n\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "2")
    c2 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    git(tmp_path, "commit", "-q", "-am", "3")
    c3 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    assert P.code_unchanged(tmp_path, c1, c2) is True  # only a non-code file changed
    assert P.code_unchanged(tmp_path, c1, c3) is False
    assert P.code_unchanged(tmp_path, c1 + "-dirty-abc", c2) is False
    assert P.code_unchanged(tmp_path, "unknown", c2) is False
    assert P.code_unchanged(tmp_path, "0" * 40, c2) is False


def test_dev_run_must_cover_every_dev_question_and_product(world, monkeypatch):
    root, _, _ = world
    d, qs = mk_dev_run(root, monkeypatch)
    graded = [G.Graded(q.key, "A5", q.type, 99.0, False) for q in qs if q.key.product == "P1"]  # P2 missing
    monkeypatch.setattr(G, "load_graded", lambda run_dir: graded)
    doc = accept(root, d, qs)
    assert not doc["pass"] and "exactly the dev question set" in doc["dev_run"]["why"]


def test_freeze_check_difference_and_whatif_mismatch_fail(world):
    root, d, qs = world
    doc = accept(root, d, qs, freeze_check=lambda r: ["facts differ"])
    assert not doc["pass"] and gate(doc, "freeze check")["numerator"] == 1
    jdump(root / "results" / "freeze" / "whatif_check.json", {"6.0": {"P1": {"mismatches": 2, "identity": "i"}}})
    doc = accept(root, d, qs)
    assert not gate(doc, "what-if")["pass"] and gate(doc, "what-if")["numerator"] == 2


def test_missing_manifest_fails_not_crashes(world):
    root, d, qs = world
    P.manifest_path(root).unlink()
    assert accept(root, d, qs)["pass"] is False


def test_stale_detection(world):
    root, d, qs = world
    accept(root, d, qs)
    assert P.acceptance_problems(root, OK) == []
    assert any("freeze check fails" in m for m in P.acceptance_problems(root, lambda r: ["d"]))
    mk_families(root, multi=("how many bugs", "another"))
    assert any("families.json" in m for m in P.acceptance_problems(root, OK))
    accept(root, d, qs)
    mk_manifest(root, config_hash="z" * 64)
    assert any("manifest.json" in m for m in P.acceptance_problems(root, OK))


# ---------------------------------------------------------------- write
def test_write_refuses_missing_failing_and_stale_acceptance(world):
    root, d, qs = world
    with pytest.raises(P.PreregError, match="missing"):
        P.write_prereg(root, freeze_check=OK)
    accept(root, d, qs, gates_report=gates_stub(ok=False))
    with pytest.raises(P.PreregError, match="failing"):
        P.write_prereg(root, freeze_check=OK)
    accept(root, d, qs)
    mk_families(root, multi=("changed",))
    with pytest.raises(P.PreregError, match="stale"):
        P.write_prereg(root, freeze_check=OK)
    accept(root, d, qs)
    with pytest.raises(P.PreregError, match="freeze check fails"):
        P.write_prereg(root, freeze_check=lambda r: ["x"])
    assert not P.prereg_path(root).exists()


def test_write_renders_everything_once(world):
    root, d, qs = world
    accept(root, d, qs)
    out = P.write_prereg(root, freeze_check=OK)
    text = out.read_text()
    assert out == root / "PREREG.md" and "{{" not in text
    assert R.PREREG_FACTS_RE.findall(text) == [FACTS] and len(R.PREREG_FACTS_RE.findall(text)) == 1
    assert f"| Acceptance report sha256 | {P._sha(P.acceptance_path(root))} |" in text
    assert "| A2 | P3 | default |" in text and "| A5 | all 24 | 3.0 |" in text
    assert "arm A2 baseline" not in text and "| arm A5 agent |" in text and "| arm A0 baseline |" in text
    assert "codex-x" in text and "precision:bug" in text and "1 products: P3" in text
    assert "- unclear one (Codex: multi, Claude: single, result: uncertain)" in text
    assert stats.read_multihop_families(text) == {"how many bugs"}
    assert "`messages`" in text or "### " in text  # the card field list is in


def test_write_empty_family_block_round_trips(world):
    root, d, qs = world
    mk_families(root, multi=(), excluded=False)
    accept(root, d, qs)
    text = P.write_prereg(root, freeze_check=OK).read_text()
    assert "multi-hop families: none" in text and stats.read_multihop_families(text) == set()


def test_render_rejects_unknown_placeholder_and_extra_facts_hash(world):
    with pytest.raises(P.PreregError, match="nope"):
        P.render("a {{nope}}", {"x": "1"})
    assert P.render("a {{x}}", {"x": "1"}) == "a 1"
    root, d, qs = world
    tpl = root / "t.md"
    tpl.write_text("facts hash: {{facts_hash}}\nfacts hash: {{facts_hash}}\n")
    accept(root, d, qs)
    with pytest.raises(P.PreregError, match="exactly once"):
        P.write_prereg(root, freeze_check=OK, template=tpl)


def test_shipped_template_placeholders_are_all_known():
    names = set(P.PLACEHOLDER_RE.findall(P.TEMPLATE.read_text()))
    assert names == {"generated_utc", "code_commit", "upstream_herb_commit", "split_hash", "bundle_sha", "manifest_sha",
                     "config_hash", "acceptance_sha", "facts_hash", "versions_table", "card_fields", "arms_table",
                     "a2_sample", "dev_visibility_summary", "family_counts", "family_block", "excluded_unseen", "auditors", "acceptance_table"}


def test_digest(world):
    root, d, qs = world
    accept(root, d, qs)
    P.write_prereg(root, freeze_check=OK)
    h, line = P.digest(root)
    assert h == P._sha(P.prereg_path(root)) and h in line


def test_release_template_never_overwrites(tmp_path):
    p = P.write_release_template(tmp_path)
    assert json.loads(p.read_text()) == {k: "" for k in P.RELEASE_FIELDS}
    p.write_text(json.dumps({"approved_by": "Zeyu"}))
    assert P.write_release_template(tmp_path) is None and json.loads(p.read_text()) == {"approved_by": "Zeyu"}


# ---------------------------------------------------------------- preflight
def git(cwd, *a):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def ready(world):
    """A fully valid launch state in a real git repo."""
    root, d, qs = world
    accept(root, d, qs)
    P.write_prereg(root, freeze_check=OK)
    rel = {**GOOD_REL, "prereg_sha256": P._sha(P.prereg_path(root))}
    jdump(P.release_path(root), rel)
    (root / "src").mkdir(exist_ok=True)
    (root / "src" / "a.py").write_text("x = 1\n")
    (root / ".gitignore").write_text("runs/\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "c")
    return root


def check(root, **kw):
    from eval.receipt import require_clean_tree
    return P.launch_check(root, clean_check=kw.pop("clean_check", require_clean_tree),
                          freeze_check=kw.pop("freeze_check", OK), **kw)


def test_preflight_passes_on_a_complete_fixture(ready):
    res = check(ready)
    assert res["pass"] is True and res["failures"] == [] and set(res["checks"]) == {
        "clean tree", "freeze check", "acceptance report", "PREREG.md", "release record", "committed"}


def test_preflight_dirty_tree_fails_alone(ready):
    (ready / "src" / "new.py").write_text("y\n")
    res = check(ready)
    assert res["failures"] and res["checks"]["clean tree"] and all(
        v is None for k, v in res["checks"].items() if k != "clean tree")


def test_preflight_freeze_check_fails_alone_with_clear_message(ready):
    res = check(ready, freeze_check=lambda r: ["facts differ"])
    assert res["checks"]["freeze check"].startswith("freeze check: facts differ")


def test_preflight_acceptance_failures(ready):
    doc = json.loads(P.acceptance_path(ready).read_text())
    jdump(P.acceptance_path(ready), {**doc, "pass": False})
    res = check(ready, clean_check=lambda r: None, committed=lambda r, rel: None)
    assert any("failing" in m for m in res["failures"]) and any("differs from the one written" in m for m in res["failures"])


def test_preflight_acceptance_hash_must_match_prereg(ready):
    doc = json.loads(P.acceptance_path(ready).read_text())
    doc["created_utc"] = "2030-01-01T00:00:00Z"  # still passing, not stale, but not the file PREREG recorded
    jdump(P.acceptance_path(ready), doc)
    res = check(ready, clean_check=lambda r: None, committed=lambda r, rel: None)
    assert res["failures"] == ["acceptance.json sha256 differs from the one written in PREREG.md"]


def test_preflight_prereg_checks(ready):
    nocheck = dict(clean_check=lambda r: None, committed=lambda r, rel: None)
    rel = json.loads(P.release_path(ready).read_text())
    jdump(P.release_path(ready), {**rel, "prereg_sha256": "0" * 64})
    assert check(ready, **nocheck)["failures"] == ["PREREG.md sha256 differs from release.json prereg_sha256"]
    jdump(P.release_path(ready), rel)
    mk_manifest(ready, facts_hash="e" * 64)  # manifest facts hash moves away from PREREG's
    res = check(ready, **nocheck)
    assert any("facts hash" in m for m in res["failures"])
    P.prereg_path(ready).unlink()
    assert "PREREG.md is missing" in check(ready, **nocheck)["failures"]


@pytest.mark.parametrize("field,value,msg", [
    ("approved_by", "someone", "approved_by"), ("approved_utc", "yesterday", "approved_utc"),
    ("approved_utc", "2026-10-20T10:00:00+02:00", "approved_utc"), ("post_url", "http://x.org", "post_url"),
    ("post_url", "", "post_url"), ("post_url", "https:///post", "post_url"), ("post_url", "https://", "post_url"),
    ("wayback_url", "https://web.archive.org.evil.com/web/123/x", "wayback_url"),
    ("wayback_url", "https://web.archive.org/web/123", "wayback_url"),
    ("wayback_url", "http://web.archive.org/web/123/x", "wayback_url"), ("wayback_url", "https://web.archive.org/web/latest/x", "wayback_url"),
    ("wayback_url", "https://example.org/web/123/x", "wayback_url")])
def test_preflight_release_record_fields(ready, field, value, msg):
    rel = json.loads(P.release_path(ready).read_text())
    jdump(P.release_path(ready), {**rel, field: value})
    res = check(ready, clean_check=lambda r: None, committed=lambda r, rel_: None)
    assert [m for m in res["failures"] if msg in m] and len(res["failures"]) >= 1
    assert res["checks"]["release record"]


def test_preflight_missing_release_record(ready):
    P.release_path(ready).unlink()
    res = check(ready, clean_check=lambda r: None, committed=lambda r, rel: None)
    assert "results/freeze/release.json is missing" in res["failures"]


def test_preflight_uncommitted_files_fail_alone(ready):
    # release.json and PREREG.md must be tracked and unchanged at HEAD (clean tree check disabled to isolate)
    P.prereg_path(ready).write_text(P.prereg_path(ready).read_text() + "\n")
    msgs = check(ready, clean_check=lambda r: None)["failures"]
    assert "PREREG.md differs from HEAD" in msgs
    git(ready, "rm", "-q", "--cached", "results/freeze/release.json")
    assert "results/freeze/release.json is not tracked by git" in check(ready, clean_check=lambda r: None)["failures"]


def test_preflight_cli_exit_code(ready, capsys):
    assert P.main(["--repo-root", str(ready), "preflight"]) == 1  # real freeze check cannot pass in a fixture
    assert "FAIL" in capsys.readouterr().out


# ---------------------------------------------------------------- run_arm(held) integration
def held_kw(tmp_path, **kw):
    from evalfx import env
    kw.setdefault("env", env())
    kw.setdefault("cap_setter", lambda a, usd: None)
    kw.setdefault("clean_check", lambda root: None)
    kw.setdefault("spend_reader", lambda arm: 0.0)
    kw.setdefault("expected_facts_hash", FACTS)
    return dict(root=tmp_path, **kw)


def test_held_run_never_loads_questions_when_preflight_fails(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(R, "load_questions", lambda split: calls.append(split) or [])
    monkeypatch.setitem(ARMS, "fake", make_arm())

    def refuse(root):
        raise R.RunStateError("held-out launch refused: no release record")

    with pytest.raises(R.RunStateError, match="release record"):
        R.run_arm("fake", "held", launch_check=refuse, arm_spec=lambda r, a: {"products": None, "sub_cap_usd": None},
                  **held_kw(tmp_path))
    assert calls == [] and not (tmp_path / "runs").exists()


def test_default_launch_check_refuses_in_an_empty_root(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(R, "load_questions", lambda split: calls.append(split) or [])
    monkeypatch.setitem(ARMS, "fake", make_arm())
    with pytest.raises(R.RunStateError, match="held-out launch refused"):
        R.run_arm("fake", "held", **held_kw(tmp_path))
    assert calls == []


def write_arms(root, arms):
    jdump(root / "results" / "freeze" / "arms.json", {"arms": arms})


def held_run(tmp_path, monkeypatch, arms, caps, arm="fake", **kw):
    monkeypatch.setitem(ARMS, arm, make_arm(name=arm, sub_cap_usd=1.0))
    write_arms(tmp_path, arms)
    return R.run_arm(arm, "held", questions=mk_questions(("P1", "P2", "P3")), launch_check=lambda r: None,
                     **{**held_kw(tmp_path), "cap_setter": lambda a, usd: caps.append((a, usd)), **kw})


def test_held_arm_not_in_arms_json_is_refused(tmp_path, monkeypatch):
    with pytest.raises(R.RunStateError, match="not listed"):
        held_run(tmp_path, monkeypatch, {"other": {"products": "all"}}, [])
    assert not (tmp_path / "runs").exists()


def test_held_products_filter_manifest_and_cap_from_arms_json(tmp_path, monkeypatch):
    caps = []
    d = held_run(tmp_path, monkeypatch, {"fake": {"products": ["P2", "P3"], "sub_cap_usd": 2.5}}, caps)
    m = R.read_manifest(d)
    assert m["products"] == ["P2", "P3"] and all(s.startswith("P2") or s.startswith("P3") for s in m["slots"])
    assert len(m["slots"]) == 14 and caps == [("fake", 2.5)]


def test_held_all_products_and_registered_cap_default(tmp_path, monkeypatch):
    caps = []
    d = held_run(tmp_path, monkeypatch, {"fake": {"products": "all"}}, caps)
    assert R.read_manifest(d)["products"] == ["P1", "P2", "P3"] and caps == [("fake", 1.0)]


def test_dev_run_unchanged_by_arms_json(tmp_path, monkeypatch):
    monkeypatch.setitem(ARMS, "fake", make_arm(sub_cap_usd=1.0))
    caps = []
    d = R.run_arm("fake", "dev", questions=mk_questions(), root=tmp_path, env=held_kw(tmp_path)["env"],
                  cap_setter=lambda a, u: caps.append(u), clean_check=lambda r: None)
    assert R.read_manifest(d)["products"] is None and caps == [1.0]


def test_cap_is_the_committed_spend_plus_the_sub_cap_and_resume_keeps_the_base(tmp_path, monkeypatch):
    caps, reads = [], []

    def reader(arm):
        reads.append(arm)
        return 1.96

    d = held_run(tmp_path, monkeypatch, {"fake": {"products": "all", "sub_cap_usd": 3.0}}, caps, spend_reader=reader)
    assert R.read_manifest(d)["cap_base_usd"] == 1.96 and caps == [("fake", pytest.approx(4.96))] and reads == ["fake"]
    # resume: the recorded base is reused, the proxy is not asked again (and its answer would be ignored)
    (d / "records.jsonl").write_text("")
    (d / "receipt.json").unlink()
    caps2 = []
    d2 = held_run(tmp_path, monkeypatch, {"fake": {"products": "all", "sub_cap_usd": 3.0}}, caps2,
                  spend_reader=lambda arm: pytest.fail("must not re-read the spend"))
    assert d2 == d and R.read_manifest(d)["cap_base_usd"] == 1.96


def test_arm_absent_from_spend_has_a_zero_base(tmp_path, monkeypatch):
    import io
    import urllib.request

    body = json.dumps({"per_arm": {"other": {"committed_usd": 2.0}}}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=10: io.BytesIO(body))
    monkeypatch.setattr("inherit.llm.proxy_url", lambda: "http://127.0.0.1:1/v1")
    assert R._read_arm_spend("fake") == 0.0
    body2 = json.dumps({"per_arm": {"fake": {"committed_usd": 1.5}}}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=10: io.BytesIO(body2))
    assert R._read_arm_spend("fake") == 1.5
    caps = []
    d = held_run(tmp_path, monkeypatch, {"fake": {"products": "all", "sub_cap_usd": 3.0}}, caps, spend_reader=lambda a: 0.0)
    assert R.read_manifest(d)["cap_base_usd"] == 0.0 and caps == [("fake", 3.0)]


def test_dev_visibility_summary_sentence(tmp_path):
    root = tmp_path
    mk_families(root, multi=("a", "b"))
    text = P.dev_visibility_summary(json.loads(P.families_path(root).read_text()))
    assert text == (
        "Under the unchanged visibility rule, the development set yielded 2 multi-hop, 1 single-hop and 0 content "
        "families; 8 of 10 voting instances were visible. A paging diagnostic found 3 of 20 answer values visible "
        "only inside neighbour groups larger than one page (200 cards or 12,000 tokens), and 2 families covering 5 "
        "development questions with every instance in that state or missing. This does not establish one-call "
        "inaccessibility, because A4 can filter, count and group before paging, so it defines no subgroup.")


def test_missing_dev_diagnostics_are_refused(world):
    root, d, qs = world
    mk_families(root, diagnostics=False)
    with pytest.raises(P.PreregError, match="dev_diagnostics"):
        P.dev_visibility_summary(json.loads(P.families_path(root).read_text()))
    accept(root, d, qs)
    mk_families(root, diagnostics=False)
    with pytest.raises(P.PreregError):
        P.write_prereg(root, freeze_check=OK)


def test_empty_pool_gate_is_shown_as_not_estimable_not_as_a_pass(world):
    root, d, qs = world
    base = gates_stub()

    def with_empty(round_dir):
        rep = base(round_dir)
        rep["gates"].append({"name": "precision:whatif_path", "numerator": 0, "denominator": 0, "threshold": 0.9,
                             "pass": True, "status": "not estimable (empty dev pool)"})
        return rep

    doc = accept(root, d, qs, gates_report=with_empty)
    assert doc["pass"] is True
    g = gate(doc, "audit:precision:whatif_path")
    assert g["status"] == "not estimable (empty dev pool)"
    table = P.acceptance_table(doc)
    row = next(line for line in table.splitlines() if "whatif_path" in line)
    assert "not estimable (empty dev pool)" in row and "pass" not in row.split("|")[-2]
    assert "| pass |" in next(line for line in table.splitlines() if "precision:bug" in line)
