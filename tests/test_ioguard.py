import json
import subprocess
import sys
from pathlib import Path

import pytest

from inherit.ioguard import is_allowed

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BETA_COPY = Path(
    "/private/tmp/claude-501/-Users-sunzeyu-dev-beta-fund/b339f194-22fe-4ce5-a098-b1fde0a7c625/scratchpad/herb-data/products/ActionGenie.json"
)

SCRIPT = """
import json, os, sys
sys.path.insert(0, {src!r})
from inherit import ioguard
ioguard.install({repo!r}, {log!r})
targets = json.loads(sys.argv[1])
out = {{}}
for name, (kind, p) in targets.items():
    try:
        if kind == "open":
            open(p).read()
        elif kind == "listdir":
            os.listdir(p)
        elif kind == "scandir":
            list(os.scandir(p))
        out[name] = "ok"
    except PermissionError:
        out[name] = "denied"
    except OSError:
        out[name] = "oserror"
print(json.dumps(out))
"""


def make_repo(tmp_path):
    repo = tmp_path / "repo"
    for d in ["src", "data/bundle/products", "data/raw/products", "data/cache", "data/facts", "runs/access", "tests"]:
        (repo / d).mkdir(parents=True)
    (repo / "data/bundle/products/mini.json").write_text('{"ok": 1}')
    (repo / "data/raw/products/mini.json").write_text('{"secret": 1}')
    (repo / "data/cache/c.json").write_text("{}")
    (repo / "tests/t.py").write_text("x")
    (repo / "pyproject.toml").write_text("x")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "answerable_questions.json").write_text("[]")
    (outside / "questions.json").write_text("[]")
    (outside / "employee.json").write_text("{}")
    (outside / "notes.txt").write_text("fine")
    (tmp_path / "herb-data" / "products").mkdir(parents=True)
    (tmp_path / "herb-data" / "products" / "mini.json").write_text("{}")
    return repo, outside


def run_guarded(repo, log, targets):
    code = SCRIPT.format(src=str(SRC), repo=str(repo), log=str(log))
    p = subprocess.run([sys.executable, "-c", code, json.dumps(targets)], capture_output=True, text=True, cwd=str(repo))
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_is_allowed_inside_repo(tmp_path):
    repo, outside = make_repo(tmp_path)
    ok = lambda p: is_allowed(str(p), repo)  # noqa: E731
    assert ok(repo / "data/bundle/products/mini.json")
    assert ok(repo / "data/cache/c.json")
    assert ok(repo / "data/facts/f.jsonl")
    assert ok(repo / "runs/access/x.jsonl")
    assert ok(repo / "src/inherit/x.py")
    assert ok(repo / ".venv/lib/python3.11/site.py")
    assert not ok(repo / "data/raw/products/mini.json")
    assert not ok(repo / "data/raw")
    assert not ok(repo / "tests/t.py")
    assert not ok(repo / "pyproject.toml")
    assert not ok(repo / "data/bundlex/y")
    assert not ok(repo / "src/../data/raw/products/mini.json")
    assert not ok(repo / "data/bundle/../raw/products/mini.json")


def test_is_allowed_outside_repo(tmp_path):
    repo, outside = make_repo(tmp_path)
    ok = lambda p: is_allowed(str(p), repo)  # noqa: E731
    assert ok(Path(sys.prefix) / "lib" / "x.py")
    assert ok("/usr/lib/x")
    assert ok("/dev/null")
    assert ok("/etc/hosts")
    assert ok(outside / "notes.txt")  # temp dir
    assert not ok(outside / "answerable_questions.json")
    assert not ok(outside / "questions.json")
    assert not ok(outside / "employee.json")
    assert not ok(tmp_path / "herb-data" / "products" / "mini.json")
    assert not ok("/Users/someone/Documents/x.txt")
    assert not ok("/private/tmp/herb-data/products/ActionGenie.json")


def test_name_rules_do_not_block_bundle_metadata(tmp_path):
    repo, _ = make_repo(tmp_path)
    (repo / "data/bundle/metadata").mkdir(parents=True)
    assert is_allowed(str(repo / "data/bundle/metadata/employee.json"), repo)


def test_guarded_process_reads_bundle_not_raw(tmp_path):
    repo, outside = make_repo(tmp_path)
    log = repo / "runs/access/t.jsonl"
    targets = {
        "bundle": ["open", str(repo / "data/bundle/products/mini.json")],
        "cache": ["open", str(repo / "data/cache/c.json")],
        "raw": ["open", str(repo / "data/raw/products/mini.json")],
        "question": ["open", str(outside / "answerable_questions.json")],
        "copy": ["open", str(tmp_path / "herb-data/products/mini.json")],
        "tests": ["open", str(repo / "tests/t.py")],
        "list_raw": ["listdir", str(repo / "data/raw")],
        "scan_raw": ["scandir", str(repo / "data/raw")],
        "list_bundle": ["listdir", str(repo / "data/bundle")],
        "notes": ["open", str(outside / "notes.txt")],
    }
    if BETA_COPY.exists():
        targets["beta"] = ["open", str(BETA_COPY)]
    got = run_guarded(repo, log, targets)
    assert got["bundle"] == "ok" and got["cache"] == "ok" and got["list_bundle"] == "ok" and got["notes"] == "ok"
    for k in ["raw", "question", "copy", "tests", "list_raw", "scan_raw"] + (["beta"] if "beta" in targets else []):
        assert got[k] == "denied", k
    events = [json.loads(l) for l in log.read_text().splitlines()]
    assert all(set(e) >= {"path", "mode", "allowed"} for e in events)
    by_path = {}
    for e in events:
        by_path.setdefault(e["path"], []).append(e["allowed"])
    real = lambda p: str(Path(p).resolve())  # noqa: E731
    assert by_path[real(repo / "data/bundle/products/mini.json")] == [True]
    assert by_path[real(repo / "data/raw/products/mini.json")] == [False]
    assert by_path[real(outside / "answerable_questions.json")] == [False]
    assert by_path[real(tmp_path / "herb-data/products/mini.json")] == [False]
    assert real(repo / "data/raw") in by_path
    if BETA_COPY.exists():
        assert False in by_path[real(BETA_COPY)]
