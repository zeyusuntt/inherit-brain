"""scripts/fetch_release.py: checksum check, unsafe archive members, local extraction. No network."""

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("fetch_release", Path(__file__).parents[1] / "scripts" / "fetch_release.py")
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)

FILES = {
    "inherit-facts.tgz": {"data/facts/P/structural.jsonl": b"{}\n", "data/bundle/manifest.json": b"{}"},
    "inherit-embed-cache.tgz": {"data/cache/embed/m/1024/a.npy": b"npy"},
    "inherit-a5-held-run.tgz": {"runs/eval/A5-held/records.jsonl": b"{}\n"},
}


def _tgz(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _release(d: Path, files=FILES) -> Path:
    d.mkdir()
    for name, members in files.items():
        _tgz(d / name, members)
    (d / "SHA256SUMS").write_text("".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n" for n in files))
    return d


def test_from_dir_extracts_into_the_root(tmp_path):
    src, root = _release(tmp_path / "rel"), tmp_path / "root"
    assert fr.main(["--from", str(src), "--root", str(root)]) == 0
    assert (root / "data/facts/P/structural.jsonl").read_bytes() == b"{}\n"
    assert (root / "data/bundle/manifest.json").is_file()
    assert (root / "data/cache/embed/m/1024/a.npy").read_bytes() == b"npy"
    assert (root / "runs/eval/A5-held/records.jsonl").is_file()


def test_checksum_mismatch_fails_and_extracts_nothing(tmp_path):
    src, root = _release(tmp_path / "rel"), tmp_path / "root"
    _tgz(src / "inherit-embed-cache.tgz", {"data/cache/embed/m/1024/b.npy": b"other"})
    with pytest.raises(SystemExit, match="checksum mismatch for inherit-embed-cache.tgz"):
        fr.main(["--from", str(src), "--root", str(root)])
    assert not root.exists()


@pytest.mark.parametrize("name", ["../evil.txt", "/abs/evil.txt", "data/../../evil.txt"])
def test_unsafe_member_is_rejected(tmp_path, name):
    files = {**FILES, "inherit-facts.tgz": {name: b"x"}}
    src, root = _release(tmp_path / "rel", files), tmp_path / "root"
    with pytest.raises(SystemExit, match="unsafe path"):
        fr.main(["--from", str(src), "--root", str(root)])
    assert not (tmp_path / "evil.txt").exists()


def test_existing_data_needs_force(tmp_path):
    src, root = _release(tmp_path / "rel"), tmp_path / "root"
    (root / "data/facts").mkdir(parents=True)
    (root / "data/facts/keep.txt").write_text("mine")
    with pytest.raises(SystemExit, match="already present: data/facts"):
        fr.main(["--from", str(src), "--root", str(root)])
    assert (root / "data/facts/keep.txt").read_text() == "mine"
    assert fr.main(["--from", str(src), "--root", str(root), "--force"]) == 0
    assert (root / "data/facts/P/structural.jsonl").is_file()
