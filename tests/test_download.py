import pytest

from inherit.data.download import PINNED_SHA, PRODUCT_NAMES, download

SHA = "db3bf9b3f911745726c579c9dbf9f7f6b2c05b36"


def test_pinned_sha():
    assert PINNED_SHA == SHA


def test_download_layout_and_pinning(tmp_path):
    urls = []

    def fake(url):
        urls.append(url)
        return url.encode()

    download(tmp_path / "raw", SHA, fetch=fake)
    raw = tmp_path / "raw"
    assert all(f"/HERB/{SHA}/" in u for u in urls)
    assert sorted(p.stem for p in (raw / "products").glob("*.json")) == sorted(PRODUCT_NAMES)
    for f in ["employee.json", "customers_data.json", "salesforce_team.json"]:
        assert (raw / "metadata" / f).exists()
    for f in ["evaluate.py", "react.py", "metadata_search_functions.py", "long_context_eval.py", "utils.py", "product_rag.py"]:
        assert (raw / "code" / f).exists()
    assert (raw / "LICENSE.txt").exists() and (raw / "AI_ETHICS.md").exists()
    assert (raw / "products" / "ActionGenie.json").read_bytes().endswith(b"data/products/ActionGenie.json")
    assert (raw / "COMMIT").read_text().strip() == SHA


def test_download_failure_leaves_no_partial_file(tmp_path):
    calls = {"n": 0}

    def flaky(url):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("boom")
        return b"x"

    with pytest.raises(OSError):
        download(tmp_path / "raw", SHA, fetch=flaky)
    assert not list((tmp_path / "raw").rglob("*.part"))
