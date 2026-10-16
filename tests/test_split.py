from pathlib import Path

import pytest

from inherit.data.download import PRODUCT_NAMES
from inherit.data.split import split, split_hash

FROZEN = "70ec2f994d94aeb2854ddbbcb204c3a286aa7e393b583ff9c99c6fa9c2846e60"
DEV = ["ActionGenie", "AnomalyForce", "ContentForce", "KnowledgeForce", "LeadForce", "TrendForce"]
ROOT = Path(__file__).resolve().parents[1]


def test_split_matches_frozen_hash():
    dev, held = split(list(PRODUCT_NAMES))
    assert dev == DEV
    assert len(held) == 24
    assert split_hash(dev, held) == FROZEN


def test_split_ignores_input_order():
    names = list(PRODUCT_NAMES)
    assert split(names[::-1]) == split(names)


def test_split_hash_payload_format():
    import hashlib

    assert split_hash(["a", "b"], ["c"]) == hashlib.sha256(b"a,b|c").hexdigest()


def test_product_names_count():
    assert len(PRODUCT_NAMES) == 30 and len(set(PRODUCT_NAMES)) == 30


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "products").exists(), reason="data/raw absent")
def test_split_matches_frozen_hash_on_real_files():
    names = sorted(p.stem for p in (ROOT / "data" / "raw" / "products").glob("*.json"))
    dev, held = split(names)
    assert split_hash(dev, held) == FROZEN
