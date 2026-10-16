import json
from pathlib import Path

import shutil

from inherit.audit import audit, check_access_log, check_bundle, check_question_leakage, check_receipts, main
from inherit.data.bundle import make_bundle

H = "a" * 64


def write_prereg(path, h=H):
    path.write_text(f"# Pre-registration\n\nfacts hash: {h}\n")


# check 1: bundle
def test_check_bundle_passes_on_clean_bundle(mini_bundle):
    r = check_bundle(mini_bundle)
    assert r.ok, r.details


def test_audit_flags_forbidden_key(mini_bundle):
    p = mini_bundle / "products" / "mini.json"
    d = json.loads(p.read_text())
    d["prs"][0]["nested"] = [{"team": ["eid_x"]}]
    p.write_text(json.dumps(d))
    r = check_bundle(mini_bundle)
    assert not r.ok
    assert any("team" in x and "mini.json" in x for x in r.details)


def test_check_bundle_each_forbidden_key(mini_bundle):
    for key in ["question", "team", "customers", "answerable_questions", "unanswerable_questions", "ground_truth", "citations"]:
        p = mini_bundle / "products" / "mini.json"
        orig = p.read_text()
        d = json.loads(orig)
        d["slack"][0][key] = 1
        p.write_text(json.dumps(d))
        assert not check_bundle(mini_bundle).ok, key
        p.write_text(orig)
    assert check_bundle(mini_bundle).ok


def test_check_bundle_missing_manifest_fails(mini_bundle):
    (mini_bundle / "manifest.json").unlink()
    r = check_bundle(mini_bundle)
    assert not r.ok and any("manifest" in x for x in r.details)


def test_check_bundle_missing_dir_fails(tmp_path):
    r = check_bundle(tmp_path / "nope")
    assert not r.ok


def test_check_bundle_hash_mismatch_fails(mini_bundle):
    p = mini_bundle / "metadata" / "employee.json"
    p.write_text(p.read_text() + " ")
    r = check_bundle(mini_bundle)
    assert not r.ok and any("sha256" in x for x in r.details)


def test_check_bundle_unexpected_file_fails(mini_bundle):
    (mini_bundle / "metadata" / "salesforce_team.json").write_text("{}")
    assert not check_bundle(mini_bundle).ok


def test_check_bundle_manifest_only_fails(tmp_path):
    import hashlib

    b = tmp_path / "bundle"
    b.mkdir()
    (b / "manifest.json").write_text(json.dumps(
        {"sha256": hashlib.sha256(b"").hexdigest(), "dropped_keys": {}, "products": []}))
    assert not check_bundle(b).ok


def test_check_bundle_missing_product_fails(mini_raw, tmp_path):
    (mini_raw / "products" / "second.json").write_text((mini_raw / "products" / "mini.json").read_text())
    b = tmp_path / "bundle"
    make_bundle(mini_raw, b)
    assert check_bundle(b).ok
    (b / "products" / "second.json").unlink()
    r = check_bundle(b)
    assert not r.ok and any("second" in x for x in r.details)


def test_check_bundle_missing_metadata_fails(mini_bundle):
    (mini_bundle / "metadata" / "customers_data.json").unlink()
    r = check_bundle(mini_bundle)
    assert not r.ok and any("customers_data" in x for x in r.details)


def test_check_bundle_bad_manifest_schema_fails(mini_bundle):
    (mini_bundle / "manifest.json").write_text(json.dumps({"sha256": "zz", "products": ["mini"], "dropped_keys": {}}))
    assert not check_bundle(mini_bundle).ok


def test_check_bundle_expected_product_count(mini_bundle):
    assert check_bundle(mini_bundle, expected_products=1).ok
    assert not check_bundle(mini_bundle, expected_products=30).ok


# check 2: access log
def good_line(repo, sub="data/bundle/products/mini.json", allowed=True):
    return json.dumps({"path": str(repo / sub), "mode": "r", "allowed": allowed})


def test_check_access_log_passes(tmp_path):
    repo = tmp_path / "repo"
    acc = repo / "runs" / "access"
    acc.mkdir(parents=True)
    (acc / "a.jsonl").write_text(good_line(repo) + "\n" + good_line(repo, "data/cache/x") + "\n" + json.dumps({"path": "/usr/lib/x", "mode": "rb", "allowed": True}) + "\n")
    r = check_access_log(acc, repo)
    assert r.ok, r.details


def test_check_access_log_missing_dir_fails(tmp_path):
    r = check_access_log(tmp_path / "repo" / "runs" / "access", tmp_path / "repo")
    assert not r.ok and any("no access log" in x for x in r.details)


def test_check_access_log_empty_dir_or_file_fails(tmp_path):
    repo = tmp_path / "repo"
    acc = repo / "runs" / "access"
    acc.mkdir(parents=True)
    assert not check_access_log(acc, repo).ok
    (acc / "a.jsonl").write_text("")
    assert not check_access_log(acc, repo).ok


def test_check_access_log_refused_event_fails(tmp_path):
    repo = tmp_path / "repo"
    acc = repo / "runs" / "access"
    acc.mkdir(parents=True)
    (acc / "a.jsonl").write_text(good_line(repo, "data/raw/x.json", allowed=False) + "\n")
    r = check_access_log(acc, repo)
    assert not r.ok and any("refused" in x for x in r.details)


def test_check_access_log_in_repo_path_outside_allowed_dirs_fails(tmp_path):
    repo = tmp_path / "repo"
    acc = repo / "runs" / "access"
    acc.mkdir(parents=True)
    (acc / "a.jsonl").write_text(good_line(repo, "data/raw/x.json", allowed=True) + "\n")
    r = check_access_log(acc, repo)
    assert not r.ok and any("outside allowed" in x for x in r.details)


def test_check_access_log_bad_json_fails(tmp_path):
    repo = tmp_path / "repo"
    acc = repo / "runs" / "access"
    acc.mkdir(parents=True)
    (acc / "a.jsonl").write_text("not json\n")
    assert not check_access_log(acc, repo).ok


# check 3: receipts
def write_receipt(d, name, h):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps({"facts_hash": h, "arm": "A1"}))


def test_check_receipts_pass(tmp_path):
    prereg = tmp_path / "PREREG.md"
    write_prereg(prereg)
    a, b = tmp_path / "results" / "receipts", tmp_path / "runs" / "receipts"
    write_receipt(a, "1.json", H)
    write_receipt(b, "2.json", H)
    r = check_receipts([a, b], prereg)
    assert r.ok, r.details


def test_check_receipts_no_receipts_fails(tmp_path):
    prereg = tmp_path / "PREREG.md"
    write_prereg(prereg)
    r = check_receipts([tmp_path / "results" / "receipts"], prereg)
    assert not r.ok and any("no receipts" in x for x in r.details)


def test_check_receipts_no_prereg_fails(tmp_path):
    a = tmp_path / "r"
    write_receipt(a, "1.json", H)
    r = check_receipts([a], tmp_path / "PREREG.md")
    assert not r.ok and any("PREREG" in x for x in r.details)


def test_check_receipts_prereg_without_hash_fails(tmp_path):
    prereg = tmp_path / "PREREG.md"
    prereg.write_text("nothing here")
    a = tmp_path / "r"
    write_receipt(a, "1.json", H)
    assert not check_receipts([a], prereg).ok


def test_check_receipts_mixed_hashes_fail(tmp_path):
    prereg = tmp_path / "PREREG.md"
    write_prereg(prereg)
    a = tmp_path / "r"
    write_receipt(a, "1.json", H)
    write_receipt(a, "2.json", "b" * 64)
    r = check_receipts([a], prereg)
    assert not r.ok and any("differ" in x or "mismatch" in x for x in r.details)


def test_check_receipts_not_equal_to_prereg_fails(tmp_path):
    prereg = tmp_path / "PREREG.md"
    write_prereg(prereg, "c" * 64)
    a = tmp_path / "r"
    write_receipt(a, "1.json", H)
    assert not check_receipts([a], prereg).ok


def test_check_receipts_missing_field_fails(tmp_path):
    prereg = tmp_path / "PREREG.md"
    write_prereg(prereg)
    a = tmp_path / "r"
    a.mkdir()
    (a / "1.json").write_text("{}")
    assert not check_receipts([a], prereg).ok


# whole audit
def test_audit_fails_when_inputs_missing_and_main_exits_nonzero(mini_raw, tmp_path, capsys):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    make_bundle(mini_raw, root / "data" / "bundle")
    results = audit(root)
    assert [r.ok for r in results] == [True, False, False, False]
    assert main(["--root", str(root)]) == 1  # default expects 30 products
    assert main(["--root", str(root), "--expect-products", "1"]) == 1
    out = capsys.readouterr().out
    assert "PASS" in out and out.count("FAIL") >= 2


def test_audit_all_pass(mini_raw, tmp_path):
    root = tmp_path / "root"
    (root / "data").mkdir(parents=True)
    make_bundle(mini_raw, root / "data" / "bundle")
    acc = root / "runs" / "access"
    acc.mkdir(parents=True)
    (acc / "a.jsonl").write_text(good_line(root) + "\n")
    write_prereg(root / "PREREG.md")
    write_receipt(root / "results" / "receipts", "1.json", H)
    shutil.copytree(mini_raw, root / "data" / "raw")
    assert len(audit(root)) == 4 and all(r.ok for r in audit(root))
    assert main(["--root", str(root), "--expect-products", "1"]) == 0


def test_check_access_log_directory_scan_is_a_read(tmp_path):
    repo = tmp_path / "repo"
    acc = repo / "runs" / "access"
    acc.mkdir(parents=True)
    scan = lambda mode, extra: json.dumps({"path": str(repo / "data/bundle/products"), "mode": mode, "allowed": True, **extra})  # noqa: E731
    for mode in ("scandir", "listdir"):
        (acc / "a.jsonl").write_text(scan(mode, {}) + "\n" + scan(mode, {"write": False}) + "\n")
        r = check_access_log(acc, repo)
        assert r.ok, r.details


def test_check_access_log_write_under_bundle_fails(tmp_path):
    repo = tmp_path / "repo"
    acc = repo / "runs" / "access"
    acc.mkdir(parents=True)
    p = str(repo / "data/bundle/products/x.json")
    for e in ({"mode": "w"}, {"mode": "r", "write": True}):
        (acc / "a.jsonl").write_text(json.dumps({"path": p, "allowed": True, **e}) + "\n")
        r = check_access_log(acc, repo)
        assert not r.ok and any("outside allowed" in x for x in r.details)


# check 4: question strings
def questions_of(mini_raw):
    from eval.questions import load_all_questions

    return load_all_questions(mini_raw)


def test_leakage_clean_bundle_passes(mini_raw, mini_bundle):
    r = check_question_leakage(mini_bundle, mini_raw)
    assert r.ok, r.details
    assert "questions" in r.details[0]


def test_leakage_flags_a_question_in_any_bundle_file_after_normalising(mini_raw, mini_bundle):
    qs = questions_of(mini_raw)
    ans = next(q for q in qs if q.key.kind == "ans")
    unans = next(q for q in qs if q.key.kind == "unans")
    p = mini_bundle / "products" / "mini.json"
    d = json.loads(p.read_text())
    d["slack"][0]["text"] = "Hey team, " + ans.text.upper().replace("?", "") + " -- thanks"
    p.write_text(json.dumps(d))
    r = check_question_leakage(mini_bundle, mini_raw)
    assert not r.ok and any(str(ans.key) in x and "mini.json" in x for x in r.details)
    d["slack"][0]["text"] = "ok"
    d["documents"][0]["content"] = unans.text
    p.write_text(json.dumps(d))
    r = check_question_leakage(mini_bundle, mini_raw)
    assert not r.ok and any(str(unans.key) in x for x in r.details)
    assert all(ans.text not in x and unans.text not in x for x in r.details)  # keys only, never the text


def test_leakage_in_a_metadata_file_is_flagged(mini_raw, mini_bundle):
    q = questions_of(mini_raw)[0]
    m = mini_bundle / "metadata" / "employee.json"
    d = json.loads(m.read_text())
    d["zz"] = {"note": q.text}
    m.write_text(json.dumps(d))
    r = check_question_leakage(mini_bundle, mini_raw)
    assert not r.ok and any("employee.json" in x for x in r.details)


def test_leakage_check_fails_when_inputs_are_missing(mini_raw, mini_bundle, tmp_path):
    assert not check_question_leakage(mini_bundle, tmp_path / "no-raw").ok
    assert not check_question_leakage(tmp_path / "no-bundle", mini_raw).ok
