"""Reproducible dev and held-out split. Prints the split and sha256 over the exact bytes below."""
import os, random, sys, hashlib
names = sorted(f[:-5] for f in os.listdir(sys.argv[1]) if f.endswith(".json"))
rest = [n for n in names if n != "ActionGenie"]
dev = ["ActionGenie"] + sorted(random.Random(20261015).sample(rest, 5))
held = [n for n in names if n not in dev]
payload = (",".join(dev) + "|" + ",".join(held)).encode("utf-8")
print("dev", dev); print("held", held)
print("bytes: dev names joined by ',' then '|' then held names joined by ',' (UTF-8, no newline)")
print("sha256", hashlib.sha256(payload).hexdigest())
