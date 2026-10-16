"""Print the frozen dev and held-out split for a directory of HERB product files.

Usage: python analysis/herb_split.py data/raw/products
The logic lives in src/inherit/data/split.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from inherit.data.split import split, split_hash  # noqa: E402

names = sorted(f[:-5] for f in os.listdir(sys.argv[1]) if f.endswith(".json"))
dev, held = split(names)
print("dev", dev)
print("held", held)
print("bytes: dev names joined by ',' then '|' then held names joined by ',' (UTF-8, no newline)")
print("sha256", split_hash(dev, held))
