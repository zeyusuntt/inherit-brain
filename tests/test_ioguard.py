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


def test_forbidden_names_beat_directory_allowances(tmp_path):
    repo, _ = make_repo(tmp_path)
    for rel in ["runs/answerable_questions.json", "src/salesforce_team.json", "data/cache/ground_truth.json",
                "data/bundle/citations.json", "data/facts/unanswerable_questions.json", "runs/other.json"]:
        assert not is_allowed(str(repo / rel), repo), rel
    assert is_allowed(str(repo / "src/inherit/graph/udf/herb.js"), repo)
    assert not is_allowed(str(repo / "runs/eval/out.json"), repo)
    assert is_allowed(str(repo / "runs/access/a.jsonl"), repo)
    assert not is_allowed("/usr/lib/python3/answerable_questions.py", repo)


def test_write_rules(tmp_path):
    repo, _ = make_repo(tmp_path)
    w = lambda rel: is_allowed(str(repo / rel), repo, write=True)  # noqa: E731
    assert w("runs/access/a.jsonl") and w("data/cache/x") and w("data/facts/x")
    assert not w("src/x.py") and not w("data/bundle/products/mini.json") and not w("runs/out.json")
    assert w("src/inherit/__pycache__/x.cpython-311.pyc")


def test_guarded_process_cannot_start_child(tmp_path):
    repo, _ = make_repo(tmp_path)
    log = repo / "runs/access/p.jsonl"
    code = """
import json, os, sys, subprocess
sys.path.insert(0, {src!r})
from inherit import ioguard
ioguard.install({repo!r}, {log!r})
out = {{}}
for name, fn in {{
    "popen": lambda: subprocess.Popen(["/bin/echo", "hi"]),
    "run": lambda: subprocess.run(["/bin/echo", "hi"]),
    "system": lambda: os.system("echo hi"),
    "fork": lambda: os.fork(),
    "spawn": lambda: os.spawnv(os.P_WAIT, "/bin/echo", ["echo"]),
    "execv": lambda: os.execv("/bin/echo", ["echo"]),
    "write_src": lambda: open({srcfile!r}, "w"),
    "write_log": lambda: open({log!r}, "a").close(),
}}.items():
    try:
        fn(); out[name] = "ok"
    except PermissionError:
        out[name] = "denied"
print(json.dumps(out))
""".format(src=str(SRC), repo=str(repo), log=str(log), srcfile=str(repo / "src/new.py"))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(repo))
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout.strip().splitlines()[-1])
    assert {k: got[k] for k in ["popen", "run", "system", "fork", "spawn", "execv", "write_src"]} == {
        k: "denied" for k in ["popen", "run", "system", "fork", "spawn", "execv", "write_src"]}
    assert got["write_log"] == "ok"
    events = [json.loads(l) for l in log.read_text().splitlines()]
    refused = {e["mode"] for e in events if not e["allowed"]}
    assert {"subprocess.Popen", "os.system", "os.fork", "os.exec"} <= refused


def test_guarded_client_construction_loads_settings_before_hook(tmp_path):
    repo, _ = make_repo(tmp_path)
    log = repo / "runs/access/c.jsonl"
    code = """
import json, sys
sys.path.insert(0, {src!r})
from inherit import ioguard, llm
ioguard.install({repo!r}, {log!r})
c = llm._client("A1", None)
try:
    open({env!r}).read(); env = "readable"
except PermissionError:
    env = "denied"
print(json.dumps({{"client": type(c).__name__, "env": env}}))
""".format(src=str(SRC), repo=str(repo), log=str(log), env=str(ROOT / ".env"))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(repo))
    assert p.returncode == 0, p.stderr
    got = json.loads(p.stdout.strip().splitlines()[-1])
    assert got == {"client": "OpenAI", "env": "denied"}


def test_only_the_extraction_summary_is_open_under_results_freeze(tmp_path):
    repo, _ = make_repo(tmp_path)
    (repo / "results/freeze/questions").mkdir(parents=True)
    ok = repo / "results/freeze/extraction_summary.json"
    ok.write_text("{}")
    assert is_allowed(ok, repo) and is_allowed(ok, repo, write=True)
    for name in ("answers.json", "questions/held.json", "manifest.json"):
        f = repo / "results/freeze" / name
        assert not is_allowed(f, repo) and not is_allowed(f, repo, write=True), name


def test_the_log_starts_with_a_line_naming_the_run(tmp_path):
    repo, _ = make_repo(tmp_path)
    log = repo / "runs/access/extract-20260101T000000Z.jsonl"
    code = "import sys\nsys.path.insert(0, %r)\nfrom inherit import ioguard\nioguard.install(%r, %r)\n" % (
        str(SRC), str(repo), str(log))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(repo))
    assert p.returncode == 0, p.stderr
    first = json.loads(log.read_text().splitlines()[0])
    assert first["run_id"] == "20260101T000000Z" and first["allowed"] is True
