import json
import os
import stat
from pathlib import Path

import pytest

from inherit.data.bundle import (
    ALLOWED_PRODUCT_KEYS,
    bundle_hash,
    load_bundle,
    make_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ["team", "customers", "answerable_questions", "unanswerable_questions"]


def test_bundle_drops_forbidden_keys(mini_raw, tmp_path):
    out = tmp_path / "bundle"
    m = make_bundle(mini_raw, out)
    assert m.products == ["mini"]
    assert m.dropped_keys == {"mini": FORBIDDEN}
    data = json.loads((out / "products" / "mini.json").read_text())
    assert list(data) == list(ALLOWED_PRODUCT_KEYS)
    assert (out / "metadata" / "employee.json").exists()
    assert (out / "metadata" / "customers_data.json").exists()
    assert not (out / "metadata" / "salesforce_team.json").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["sha256"] == m.sha256
    assert manifest["dropped_keys"] == {"mini": FORBIDDEN}
    assert manifest["products"] == ["mini"]


def test_bundle_keeps_allowed_records_unchanged(mini_raw, tmp_path):
    out = tmp_path / "bundle"
    make_bundle(mini_raw, out)
    src = json.loads((mini_raw / "products" / "mini.json").read_text())
    got = load_bundle(out, "mini")
    for k in ALLOWED_PRODUCT_KEYS:
        assert got[k] == src[k]


def test_bundle_hash_stable(mini_raw, tmp_path):
    a = make_bundle(mini_raw, tmp_path / "a")
    b = make_bundle(mini_raw, tmp_path / "b")
    assert a.sha256 == b.sha256
    assert len(a.sha256) == 64
    assert bundle_hash(tmp_path / "a") == a.sha256


def test_bundle_hash_changes_with_content(mini_raw, tmp_path):
    a = make_bundle(mini_raw, tmp_path / "a")
    p = mini_raw / "products" / "mini.json"
    d = json.loads(p.read_text())
    d["urls"][0]["description"] = "changed"
    p.write_text(json.dumps(d))
    b = make_bundle(mini_raw, tmp_path / "b")
    assert a.sha256 != b.sha256


def test_bundle_hash_ignores_dropped_keys(mini_raw, tmp_path):
    a = make_bundle(mini_raw, tmp_path / "a")
    p = mini_raw / "products" / "mini.json"
    d = json.loads(p.read_text())
    d["team"] = ["eid_other"]
    p.write_text(json.dumps(d))
    b = make_bundle(mini_raw, tmp_path / "b")
    assert a.sha256 == b.sha256


@pytest.mark.skipif(os.geteuid() == 0, reason="chmod 000 does not block root")
def test_bundle_never_opens_salesforce_team(mini_raw, tmp_path):
    team = mini_raw / "metadata" / "salesforce_team.json"
    team.chmod(0)
    try:
        make_bundle(mini_raw, tmp_path / "bundle")
    finally:
        team.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_bundle_missing_allowed_key_raises(mini_raw, tmp_path):
    p = mini_raw / "products" / "mini.json"
    d = json.loads(p.read_text())
    del d["prs"]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="prs"):
        make_bundle(mini_raw, tmp_path / "bundle")


def test_bundle_refuses_to_wipe_unrelated_dir(mini_raw, tmp_path):
    out = tmp_path / "precious"
    out.mkdir()
    (out / "notes.txt").write_text("keep")
    with pytest.raises(ValueError, match="not a bundle"):
        make_bundle(mini_raw, out)
    assert (out / "notes.txt").exists()


def test_bundle_rebuild_replaces_old_bundle(mini_raw, tmp_path):
    out = tmp_path / "bundle"
    first = make_bundle(mini_raw, out)
    (out / "stale.json").write_text("{}")
    second = make_bundle(mini_raw, out)
    assert not (out / "stale.json").exists()
    assert first.sha256 == second.sha256


RAW = ROOT / "data" / "raw"


@pytest.mark.skipif(not (RAW / "products").exists(), reason="data/raw absent")
def test_real_bundle_has_30_products_and_no_forbidden_keys(tmp_path):
    m = make_bundle(RAW, tmp_path / "bundle")
    assert len(m.products) == 30
    assert all("ground_truth" not in json.dumps(load_bundle(tmp_path / "bundle", p)) for p in m.products[:3])
