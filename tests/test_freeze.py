import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

import whatiffx as fx
from conftest import fake_embed
from inherit import embed as embed_mod
from inherit import freeze
from inherit.graph import load as loadmod

P = "mini"


@pytest.fixture
def frozen(mini_bundle, load_facts, falkor_both, tmp_path):
    """A one-product freeze root: layer files (all but structural empty), loaded graphs on both servers, embedding
    cache, access log, what-if check record. Returns the Env."""
    root = tmp_path / "repo"
    shutil.copytree(mini_bundle, root / "data" / "bundle")
    fd = root / "data" / "facts" / P
    fd.mkdir(parents=True)
    load_facts.to_jsonl(fd / "structural.jsonl")
    for f in loadmod.LAYER_FILES[1:]:
        (fd / f).write_text("")
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "COMMIT").write_text("abc123\n")
    (root / "uv.lock").write_text("lock\n")
    (root / "runs" / "access").mkdir(parents=True)
    (root / "runs" / "access" / "extract-1.jsonl").write_text('{"path": "x", "allowed": true}\n')
    facts = loadmod.load_layers(P, root / "data" / "facts")
    cache = root / "data" / "cache" / "embed"
    for _, _, text in loadmod.embed_targets(facts):
        f = embed_mod._cache_file(cache, text)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, np.array(fake_embed([text])[0], dtype=np.float32))
    for db in falkor_both.values():
        loadmod.load_product(db, P, facts, embed=fake_embed)
    env = freeze.Env(root=root, servers=("6.0", "4.22"), expected=1, split=lambda inv: (list(inv), []),
                     graph=lambda p, v: falkor_both[v].select_graph(f"herb:{p}"),
                     commit=lambda r: "c0ffee", drift=lambda r, c: [])
    env.out.mkdir(parents=True)
    res = {v: {P: {"leads": 1, "mismatches": 0, "identity": freeze.graph_identity(env.graph(P, v))["identity"],
                   "diffs": []}} for v in env.servers}
    (env.out / "whatif_check.json").write_text(json.dumps(res))
    env.extra["dbs"] = falkor_both
    return env


def test_manifest_passes_with_empty_layer_files_and_records_the_pieces(frozen):
    m = freeze.build_manifest(frozen)
    assert m["code_commit"] == "c0ffee" and m["products"] == [P]
    assert set(m["layers"][P]) == set(loadmod.LAYER_FILES)
    assert m["layers"][P]["rules.jsonl"] == freeze.sha256_bytes(b"")
    from eval.receipt import facts_hash

    assert m["facts_hash"] == facts_hash(frozen.facts)
    assert m["upstream_herb_commit"] == "abc123" and m["uv_lock_sha256"]
    assert set(m["graphs"][P]) == {"6.0", "4.22"} and m["card_corpus_sha256"][P]
    assert m["config"]["versions"]["card"] and m["config_hash"]
    assert (frozen.out / "manifest.json").exists()
    assert freeze.check(frozen) == []


def test_manifest_fails_on_a_missing_layer_file(frozen):
    (frozen.facts / P / "bugs.jsonl").unlink()
    with pytest.raises(freeze.FreezeError, match="bugs.jsonl is absent"):
        freeze.build_manifest(frozen)
    assert not (frozen.out / "manifest.json").exists()


def test_manifest_fails_on_a_missing_product(frozen):
    shutil.rmtree(frozen.facts / P)
    with pytest.raises(freeze.FreezeError, match="mini: no fact directory"):
        freeze.build_manifest(frozen)


def test_manifest_fails_without_the_expected_product_count(frozen):
    frozen.expected = 30
    with pytest.raises(freeze.FreezeError, match="expected 30"):
        freeze.build_manifest(frozen)


def test_manifest_fails_when_a_graph_differs_from_the_facts(frozen):
    frozen.extra["dbs"]["4.22"].select_graph(f"herb:{P}").query("MATCH (n:Bug {id: 'Bug:b1'}) SET n.title = 'changed'")
    with pytest.raises(freeze.FreezeError, match=r"mini \[4.22\]: graph content identity differs"):
        freeze.build_manifest(frozen)


def test_manifest_fails_without_a_clean_whatif_check(frozen):
    p = frozen.out / "whatif_check.json"
    res = json.loads(p.read_text())
    res["6.0"][P]["mismatches"] = 1
    p.write_text(json.dumps(res))
    with pytest.raises(freeze.FreezeError, match="what-if check has 1 mismatches"):
        freeze.build_manifest(frozen)
    p.unlink()
    with pytest.raises(freeze.FreezeError, match="whatif_check.json missing"):
        freeze.build_manifest(frozen)


def test_check_detects_a_changed_layer_file(frozen):
    freeze.build_manifest(frozen)
    with (frozen.facts / P / "rules.jsonl").open("a") as f:
        f.write("\n")
    out = freeze.check(frozen)
    assert any("rules.jsonl" in d for d in out) and any("/facts_hash" in d for d in out)


def test_check_detects_a_changed_graph(frozen):
    freeze.build_manifest(frozen)
    frozen.extra["dbs"]["6.0"].select_graph(f"herb:{P}").query("MATCH (n:Bug {id: 'Bug:b1'}) SET n.title = 'changed'")
    out = freeze.check(frozen)
    assert any("mini [6.0]: graph content identity differs" in d for d in out)
    assert any("/graphs/mini/6.0/identity" in d for d in out)


def test_check_detects_code_drift_and_a_changed_bundle(frozen):
    freeze.build_manifest(frozen)
    frozen.drift = lambda r, c: ["code changed since c0ffee: src/x.py"]
    assert "code changed since c0ffee: src/x.py" in freeze.check(frozen)
    (frozen.bundle / "products" / "mini.json").write_text("{}")
    assert any("/bundle_sha256" in d for d in freeze.check(frozen))


def test_check_ignores_indexes_added_after_the_manifest_but_flags_changed_ones(frozen):
    freeze.build_manifest(frozen)
    meta = frozen.root / "data" / "cache" / "baselines" / "raw" / P / "meta.json"
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({"fingerprint": "f1"}))
    assert freeze.check(frozen) == []
    m = json.loads((frozen.out / "manifest.json").read_text())
    m["a0_indexes"] = {"raw": {P: {"fingerprint": "f0", "meta_sha256": "x"}}}
    (frozen.out / "manifest.json").write_text(json.dumps(m))
    assert any("/a0_indexes/raw/mini" in d for d in freeze.check(frozen))


def test_the_manifest_command_exit_codes(frozen, capsys):
    assert freeze.main(["check"], env=frozen) == 1  # no manifest yet
    assert freeze.main(["manifest"], env=frozen) == 0
    assert freeze.main(["check"], env=frozen) == 0
    (frozen.facts / P / "semantic.jsonl").write_text("x\n")
    assert freeze.main(["check"], env=frozen) == 1
    assert "DIFF" in capsys.readouterr().out


def test_no_subcommand_opens_a_question_file(frozen, tmp_path):
    (frozen.root / "answerable_questions.json").write_text("[]")
    (frozen.root / "data" / "raw" / "questions.json").write_text("[]")
    seen: list[str] = []
    active = [True]

    def hook(event, args):
        if active[0] and event == "open" and args and isinstance(args[0], (str, bytes)):
            seen.append(str(args[0]))

    sys.addaudithook(hook)
    try:
        freeze.build_manifest(frozen)
        freeze.check(frozen)
        freeze.extract(frozen, [P], estimate=True, runner=lambda *a: {"projected_usd": 0.1}, records_for=lambda p: [])
        freeze.extract(frozen, [P], ("structural",), runner=lambda *a: None, records_for=lambda p: [])
    finally:
        active[0] = False
    assert seen
    assert not [s for s in seen if "question" in s.lower() or "ground_truth" in s.lower()]


# ---------------------------------------------------------------- extract and R1
def two_products(env):
    p = env.bundle / "manifest.json"
    m = json.loads(p.read_text())
    m["products"] = [P, "other"]
    p.write_text(json.dumps(m))
    (env.facts / "other").mkdir()


def test_extract_refuses_a_layer_whose_version_differs_from_frozen_products(frozen):
    two_products(frozen)
    (frozen.facts / "other" / "bugs_report.json").write_text(json.dumps({"prompt_version": "bugs-v0",
                                                                         "rules_version": "bug_rules/v1"}))
    calls = []
    runner = lambda root, layer, p, rec, est: calls.append((layer, p))  # noqa: E731
    with pytest.raises(freeze.FreezeError, match="layer 'bugs' changed version.*all 2 products"):
        freeze.extract(frozen, [P], ("structural", "bugs"), runner=runner, records_for=lambda p: [])
    assert calls == []  # refused before any layer ran
    freeze.extract(frozen, [P, "other"], ("structural", "bugs"), runner=runner, records_for=lambda p: [])
    assert calls == [("structural", P), ("structural", "other"), ("bugs", P), ("bugs", "other")]


def test_extract_records_versions_and_refuses_a_later_change(frozen, monkeypatch):
    two_products(frozen)
    runner = lambda root, layer, p, rec, est: {"extraction_failures": [1, 2], "manifest": {"complete": False}}  # noqa: E731
    out = freeze.extract(frozen, [P, "other"], ("semantic",), runner=runner, records_for=lambda p: [])
    assert out[P]["semantic"] == {"failures": 2, "complete": False}
    assert json.loads((frozen.out / "extraction_summary.json").read_text())["other"]["semantic"]["failures"] == 2
    freeze.extract(frozen, [P], ("semantic",), runner=runner, records_for=lambda p: [])  # same version: a subset is fine
    monkeypatch.setattr("inherit.extract.semantic_prompts.SEMANTIC_PROMPT_VERSION", "semantic-v3")
    with pytest.raises(freeze.FreezeError, match="semantic"):
        freeze.extract(frozen, [P], ("semantic",), runner=runner, records_for=lambda p: [])


def test_a_fatal_error_stops_the_run(frozen):
    two_products(frozen)
    seen = []

    def runner(root, layer, p, rec, est):
        seen.append((layer, p))
        if layer == "names":
            raise RuntimeError("budget exhausted")

    with pytest.raises(RuntimeError, match="budget"):
        freeze.extract(frozen, [P, "other"], ("structural", "names", "rules"), runner=runner, records_for=lambda p: [])
    assert seen == [("structural", P), ("structural", "other"), ("names", P)]


# ---------------------------------------------------------------- git
def test_the_manifest_refuses_a_dirty_tree(tmp_path):
    def git(*a):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path, check=True,
                       capture_output=True)

    git("init")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-m", "c")
    head = freeze.git_commit(tmp_path)
    assert freeze.git_drift(tmp_path, head) == []
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="not clean"):
        freeze.git_commit(tmp_path)
    assert "not clean" in freeze.git_drift(tmp_path, head)[0]


# ---------------------------------------------------------------- what-if check
def test_verify_whatif_compares_every_lead_on_both_servers(falkor_both, tmp_path):
    conns = {}
    for ver, db in falkor_both.items():
        fx.clean(db)
        fx.load(db)
        conns[ver] = fx.whatif_conn(db, tmp_path)
    env = freeze.Env(root=tmp_path, servers=("6.0", "4.22"),
                     graph=lambda p, v: falkor_both[v].select_graph(f"herb:{p}"), conn=lambda p, v: conns[v])
    try:
        assert freeze.verify_whatif(env, [fx.P]) == 0
        res = json.loads((env.out / "whatif_check.json").read_text())
        assert all(res[v][fx.P]["leads"] >= 2 and res[v][fx.P]["mismatches"] == 0 for v in env.servers)
    finally:
        for db in falkor_both.values():
            fx.clean(db)
