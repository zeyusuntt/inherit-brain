"""Build the allowlisted input bundle from raw HERB files.

Only the product keys in ALLOWED_PRODUCT_KEYS and the two allowed metadata files are copied.
salesforce_team.json is downloaded but never opened here. Every dropped key is recorded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from inherit.config import ROOT
from inherit.data.scan import find_forbidden_keys

RAW_DIR = ROOT / "data" / "raw"
BUNDLE_DIR = ROOT / "data" / "bundle"

ALLOWED_PRODUCT_KEYS = ("slack", "documents", "meeting_transcripts", "meeting_chats", "urls", "prs")
ALLOWED_METADATA = ("employee.json", "customers_data.json")
MANIFEST = "manifest.json"


@dataclass(frozen=True)
class BundleManifest:
    sha256: str
    dropped_keys: dict[str, list[str]]
    products: list[str]

    def to_dict(self) -> dict:
        return {"sha256": self.sha256, "dropped_keys": self.dropped_keys, "products": self.products}


def _dump(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def bundle_hash(out: Path) -> str:
    """sha256 over sorted relative paths, each followed by its byte length and its bytes. manifest.json is excluded."""
    out = Path(out)
    h = hashlib.sha256()
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.relative_to(out).as_posix() != MANIFEST)
    for p in files:
        data = p.read_bytes()
        h.update(p.relative_to(out).as_posix().encode("utf-8") + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()


def make_bundle(raw: Path, out: Path) -> BundleManifest:
    raw, out = Path(raw), Path(out)
    # Load and validate everything before touching the output directory.
    names = sorted(p.stem for p in (raw / "products").glob("*.json"))
    products: dict[str, dict] = {}
    dropped: dict[str, list[str]] = {}
    problems: list[str] = []
    for name in names:
        data = json.loads((raw / "products" / f"{name}.json").read_text(encoding="utf-8"))
        missing = [k for k in ALLOWED_PRODUCT_KEYS if k not in data]
        if missing:
            raise ValueError(f"product {name} is missing allowed keys {missing}")
        dropped[name] = [k for k in data if k not in ALLOWED_PRODUCT_KEYS]
        products[name] = {k: data[k] for k in ALLOWED_PRODUCT_KEYS}
        problems += find_forbidden_keys(products[name], f"products/{name}.json")
    metadata = {}
    for f in ALLOWED_METADATA:
        metadata[f] = json.loads((raw / "metadata" / f).read_text(encoding="utf-8"))
        problems += find_forbidden_keys(metadata[f], f"metadata/{f}")
    if problems:
        raise ValueError("refusing to build the bundle, forbidden keys in retained data: " + "; ".join(problems[:10]))

    if out.exists() and any(out.iterdir()):
        if not (out / MANIFEST).exists():
            raise ValueError(f"{out} exists and is not a bundle (no {MANIFEST}); refusing to wipe it")
        shutil.rmtree(out)
    (out / "products").mkdir(parents=True, exist_ok=True)
    (out / "metadata").mkdir(exist_ok=True)
    for name, data in products.items():
        (out / "products" / f"{name}.json").write_bytes(_dump(data))
    for f, data in metadata.items():
        (out / "metadata" / f).write_bytes(_dump(data))

    manifest = BundleManifest(sha256=bundle_hash(out), dropped_keys=dropped, products=names)
    (out / MANIFEST).write_text(json.dumps(manifest.to_dict(), indent=1) + "\n", encoding="utf-8")
    return manifest


def load_bundle(out: Path, product: str) -> dict:
    return json.loads((Path(out) / "products" / f"{product}.json").read_text(encoding="utf-8"))


def load_metadata(out: Path) -> tuple[dict, list]:
    """(employees dict keyed by id, customers list)."""
    out = Path(out)
    emp = json.loads((out / "metadata" / "employee.json").read_text(encoding="utf-8"))
    cust = json.loads((out / "metadata" / "customers_data.json").read_text(encoding="utf-8"))
    return emp, cust


def main() -> int:
    from inherit.data.split import split, split_hash

    m = make_bundle(RAW_DIR, BUNDLE_DIR)
    counts: dict[str, int] = {}
    for keys in m.dropped_keys.values():
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
    print(f"bundle: {len(m.products)} products in {BUNDLE_DIR}")
    print(f"dropped keys (products each): {counts}")
    dev, held = split(m.products)
    print(f"split: {len(dev)} dev, {len(held)} held, sha256 {split_hash(dev, held)}")
    print(f"manifest sha256 {m.sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
