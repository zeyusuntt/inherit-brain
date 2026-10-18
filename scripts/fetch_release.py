"""Fetch the release assets and unpack them at the repo root. Standard library only.

    python3 scripts/fetch_release.py [--tag v1.0] [--force] [--from DIR] [--root DIR]

Downloads inherit-facts.tgz, inherit-embed-cache.tgz and inherit-a5-held-run.tgz plus SHA256SUMS from the GitHub
release, checks every sha256, then extracts. It refuses to touch a target that already holds data unless --force.
`--from DIR` reads the files from DIR instead of downloading.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

URL = "https://github.com/zeyusuntt/inherit-brain/releases/download/{tag}/{name}"
ASSETS = ("inherit-facts.tgz", "inherit-embed-cache.tgz", "inherit-a5-held-run.tgz")
# What each asset puts in place; non-empty means "already fetched".
TARGETS = {
    "inherit-facts.tgz": ("data/facts", "data/bundle/manifest.json"),
    "inherit-embed-cache.tgz": ("data/cache/embed",),
    "inherit-a5-held-run.tgz": ("runs/eval/A5-held",),
}
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_sums(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = digest.lower()
    return out


def verify(directory: Path) -> None:
    sums = parse_sums((directory / "SHA256SUMS").read_text())
    for name in ASSETS:
        if name not in sums:
            raise SystemExit(f"SHA256SUMS has no entry for {name}")
        got = sha256(directory / name)
        if got != sums[name]:
            raise SystemExit(f"checksum mismatch for {name}: expected {sums[name]}, got {got}")
        print(f"ok {name} {got}")


def check_members(tar: tarfile.TarFile) -> None:
    for m in tar.getmembers():
        p = PurePosixPath(m.name)
        if p.is_absolute() or ".." in p.parts:
            raise SystemExit(f"unsafe path in archive: {m.name!r}")
        if not (m.isfile() or m.isdir()):
            raise SystemExit(f"unsupported member type in archive: {m.name!r}")


def extract(archive: Path, root: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        check_members(tar)
        # check_members already allows only regular files and directories under the root; the "data" filter
        # (Python 3.12+, and 3.8-3.11 security releases) adds its own checks on top.
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(root, **kw)


def occupied(root: Path, rel: str) -> bool:
    p = root / rel
    return p.is_file() or (p.is_dir() and any(p.iterdir()))


def download(tag: str, dest: Path) -> None:
    for name in (*ASSETS, "SHA256SUMS"):
        url = URL.format(tag=tag, name=name)
        print(f"downloading {url}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(dest / name, "wb") as f:
                shutil.copyfileobj(r, f)
        except OSError as e:
            raise SystemExit(f"download failed for {url}: {e}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v1.0")
    ap.add_argument("--force", action="store_true", help="extract over existing data")
    ap.add_argument("--from", dest="src", metavar="DIR", help="use local files instead of downloading")
    ap.add_argument("--root", default=str(ROOT), help="where to extract (default: the repo root)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if not a.force:
        busy = [t for ts in TARGETS.values() for t in ts if occupied(root, t)]
        if busy:
            raise SystemExit(f"already present: {', '.join(busy)}; use --force to overwrite")
    with tempfile.TemporaryDirectory() as tmp:
        if a.src:
            directory = Path(a.src)
        else:
            directory = Path(tmp)
            download(a.tag, directory)
        verify(directory)
        for name in ASSETS:
            print(f"extracting {name}", flush=True)
            extract(directory / name, root)
    print(f"done: data in {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
