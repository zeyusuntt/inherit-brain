"""Build the release assets a judge fetches (no API key needed to load the graphs).

    uv run python scripts/make_release_assets.py [--out dist/release]

Writes inherit-facts.tgz (data/facts and data/bundle/manifest.json, the product inventory the loader reads),
inherit-embed-cache.tgz (only the cached vectors for the texts the loader embeds), inherit-a5-held-run.tgz (only
the files `inherit.ask --replay` reads) and SHA256SUMS. Archive paths are relative to the repo root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

from inherit import embed
from inherit.config import EMBED_MODEL, ROOT
from inherit.graph.load import embed_targets, load_layers

RUN = Path("runs/eval/A5-held")
FACTS = Path("data/facts")
MANIFEST = Path("data/bundle/manifest.json")


def _add(tar: tarfile.TarFile, root: Path, rel: Path) -> None:
    info = tar.gettarinfo(str(root / rel), arcname=rel.as_posix())
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if info.isdir() else 0o644
    if info.isdir():
        tar.addfile(info)
    else:
        with open(root / rel, "rb") as f:
            tar.addfile(info, f)


def _write(path: Path, root: Path, rels: list[Path]) -> None:
    with tarfile.open(path, "w:gz", compresslevel=9) as tar:
        for rel in sorted(rels):
            _add(tar, root, rel)


def _tree(root: Path, rel: Path) -> list[Path]:
    return [p.relative_to(root) for p in sorted((root / rel).rglob("*")) if p.is_file() and p.name != ".DS_Store"]


def cache_files(root: Path) -> tuple[list[Path], int]:
    """(cache files of every embed target of every product, number of distinct texts). Fails if any is missing."""
    products = json.loads((root / MANIFEST).read_text())["products"]
    texts = dict.fromkeys(t for p in products for _, _, t in embed_targets(load_layers(p, root / FACTS)))
    cache = root / "data" / "cache" / "embed"
    files = {t: embed._cache_file(cache, t) for t in texts}  # noqa: SLF001 - the loader's own cache key
    missing = [f for f in files.values() if not f.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(texts)} embed targets have no cache file, e.g. {missing[0]}")
    rels = [f.relative_to(root) for f in files.values()]
    if len(rels) != len(texts):
        raise SystemExit(f"cache file count {len(rels)} != distinct embed targets {len(texts)}")
    return rels, len(texts)


def replay_files(root: Path) -> list[Path]:
    return [RUN / "records.jsonl", *_tree(root, RUN / "tools")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "dist" / "release"))
    out = Path(ap.parse_args(argv).out)
    out.mkdir(parents=True, exist_ok=True)
    cache, n = cache_files(ROOT)
    assets = {
        "inherit-facts.tgz": [*_tree(ROOT, FACTS), MANIFEST],
        "inherit-embed-cache.tgz": cache,
        "inherit-a5-held-run.tgz": replay_files(ROOT),
    }
    sums = []
    for name, rels in assets.items():
        missing = [r for r in rels if not (ROOT / r).is_file()]
        if missing:
            raise SystemExit(f"{name}: missing {missing[0]}")
        _write(out / name, ROOT, rels)
        digest = hashlib.sha256((out / name).read_bytes()).hexdigest()
        sums.append(f"{digest}  {name}")
        print(f"{name}: {len(rels)} files, {(out / name).stat().st_size / 1e6:.1f} MB, sha256 {digest}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"embed cache: {len(cache)} files for {n} distinct texts of {EMBED_MODEL}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
