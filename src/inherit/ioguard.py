"""File-access guard for extraction and graph building.

`install(repo_root, log_path)` registers an audit hook (audit hooks cannot be removed) that logs every
open, os.listdir and os.scandir as a JSON line {path, mode, allowed} and raises PermissionError for
anything that is not allowed.

Inside the repo only src/, .venv/, data/bundle/, data/cache/, data/facts/ and runs/ are allowed.
Raw HERB data, question files and metadata files outside the bundle are refused wherever they live.
Outside the repo, only the Python install, the temp dir and a short list of system dirs are allowed.
"""

from __future__ import annotations

import json
import os
import site
import sys
import sysconfig
import tempfile
import threading
from pathlib import Path

READ_DIRS = ("src", ".venv", "data/bundle", "data/cache", "data/facts")
WRITE_DIRS = ("runs/access", "data/cache", "data/facts")
PROCESS_EVENTS = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty")
# Never readable, wherever they live. Checked before any directory allowance.
FORBIDDEN_NAMES = ("answerable_questions", "unanswerable_questions", "ground_truth", "citations", "salesforce_team")
SYSTEM_PREFIXES = ("/dev", "/etc", "/private/etc", "/System", "/usr", "/Library", "/opt/homebrew")
HERB_FILE_NAMES = {"employee.json", "customers_data.json", "salesforce_team.json"}
QUESTION_WORDS = ("question", "ground_truth", "citations", "answerable")


def _real(p: str | os.PathLike | bytes) -> str:
    return os.path.realpath(os.fsdecode(p))


def _under(path: str, base: str) -> bool:
    return path == base or path.startswith(base.rstrip(os.sep) + os.sep)


def _python_prefixes() -> list[str]:
    cands = [sys.prefix, sys.base_prefix, sys.exec_prefix, tempfile.gettempdir(),
             os.path.expanduser("~/.local/share/uv"), *SYSTEM_PREFIXES]
    try:
        cands += site.getsitepackages()
    except Exception:
        pass
    cands += [sysconfig.get_paths().get(k, "") for k in ("stdlib", "platstdlib", "purelib", "platlib")]
    return sorted({_real(c) for c in cands if c})


def _python_only_prefixes() -> list[str]:
    """Trees that hold Python itself and its libraries; name rules do not apply there."""
    cands = [sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.expanduser("~/.local/share/uv")]
    try:
        cands += site.getsitepackages()
    except Exception:
        pass
    return sorted({_real(c) for c in cands if c})


def _name_refused(path: str) -> bool:
    parts = [p.lower() for p in Path(path).parts]
    base = parts[-1] if parts else ""
    if base in HERB_FILE_NAMES:
        return True
    if any(w in base for w in QUESTION_WORDS):
        return True
    if any("herb" in p for p in parts):
        return True
    # a products/<name>.json file anywhere outside the bundle
    if base.endswith(".json") and len(parts) >= 2 and parts[-2] == "products":
        return True
    return False


def _is_write_mode(mode) -> bool:
    return isinstance(mode, str) and any(c in mode for c in "wax+")


def _flags_write(flags) -> bool:
    return isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))


def repo_dir_allowed(path: str | os.PathLike | bytes, repo_root: str | os.PathLike, write: bool = False) -> bool:
    """True if a path inside the repo lies under an allowed directory (no name rules)."""
    p, root = _real(path), _real(repo_root)
    dirs = WRITE_DIRS if write else READ_DIRS + ("runs/access",)
    if any(_under(p, os.path.join(root, d)) for d in dirs):
        return True
    # bytecode caches are written next to the code
    return write and "__pycache__" in p.split(os.sep) and any(_under(p, os.path.join(root, d)) for d in ("src", ".venv"))


def _forbidden(p: str) -> bool:
    if "/data/raw/" in p + "/":
        return True
    base = os.path.basename(p).lower()
    return any(w in base for w in FORBIDDEN_NAMES)


def is_allowed(path: str | os.PathLike | bytes, repo_root: str | os.PathLike, write: bool = False) -> bool:
    p = _real(path)
    root = _real(repo_root)
    if _forbidden(p):
        return False
    if _under(p, root):
        rel = os.path.relpath(p, root)
        # other question and HERB metadata files are refused by name, except in the bundle and in code dirs
        exempt = ("src", ".venv", "data/bundle")
        if not any(_under(p, os.path.join(root, d)) for d in exempt) and _name_refused(rel):
            return False
        return repo_dir_allowed(p, root, write)
    if write:
        if "__pycache__" in p.split(os.sep) and any(_under(p, b) for b in _python_only_prefixes()):
            return True
        return any(_under(p, b) for b in (_real(tempfile.gettempdir()), "/dev"))
    if any(_under(p, b) for b in _python_only_prefixes()):
        return True
    if _name_refused(p):
        return False
    return any(_under(p, b) for b in _python_prefixes())


_state = threading.local()


def install(repo_root: str | os.PathLike, log_path: str | os.PathLike) -> None:
    # Public settings come from .env, which the guard will refuse to open. Load them first.
    try:
        from inherit.config import load_settings

        load_settings()
    except Exception:
        pass
    root = _real(repo_root)
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    log_real = _real(log)

    def emit(path: str, mode: str, allowed: bool) -> None:
        line = json.dumps({"path": path, "mode": mode, "allowed": allowed}, ensure_ascii=False) + "\n"
        os.write(fd, line.encode("utf-8"))

    def hook(event: str, args: tuple) -> None:
        if event not in ("open", "os.listdir", "os.scandir") and event not in PROCESS_EVENTS:
            return
        if getattr(_state, "busy", False):
            return
        _state.busy = True
        try:
            if event in PROCESS_EVENTS:
                target = args[1] if event == "os.spawn" and len(args) > 1 else (args[0] if args else "")
                emit(os.fsdecode(target) if isinstance(target, (str, bytes, os.PathLike)) else "", event, False)
                refusal = f"ioguard: {event} refused"
            else:
                raw = args[0] if args else None
                if event == "open":
                    if isinstance(raw, int) or raw is None:
                        return
                    mode = args[1] if len(args) > 1 and isinstance(args[1], str) else None
                    flags = args[2] if len(args) > 2 else None
                    write = _is_write_mode(mode) or _flags_write(flags)
                    if mode is None:
                        mode = "w" if write else "r"
                else:
                    mode, write = event.split(".")[1], False
                    if raw is None:
                        raw = "."
                    elif isinstance(raw, int):
                        return
                p = _real(raw)
                allowed = p == log_real or is_allowed(p, root, write)
                emit(p, mode, allowed)
                if allowed:
                    return
                refusal = f"ioguard: {event} refused for {p}"
        finally:
            _state.busy = False
        raise PermissionError(refusal)

    sys.addaudithook(hook)
