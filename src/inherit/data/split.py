"""Frozen dev and held-out split. The payload and hash are fixed in the pre-registration."""

from __future__ import annotations

import hashlib
import random

ANCHOR = "ActionGenie"
SEED = 20261015
N_RANDOM_DEV = 5


def split(products: list[str]) -> tuple[list[str], list[str]]:
    """Return (dev, held). dev is the anchor plus 5 seeded draws; held is every other product."""
    names = sorted(products)
    rest = [n for n in names if n != ANCHOR]
    dev = [ANCHOR] + sorted(random.Random(SEED).sample(rest, N_RANDOM_DEV))
    held = [n for n in names if n not in dev]
    return dev, held


def split_hash(dev: list[str], held: list[str]) -> str:
    """sha256 of dev names joined by ',' then '|' then held names joined by ',' (UTF-8, no newline)."""
    payload = (",".join(dev) + "|" + ",".join(held)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
