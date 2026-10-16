"""Acceptance report, PREREG.md generator and held-out launch preflight (task 13, step E).

    uv run python -m eval.prereg acceptance --audit-round N --dev-run runs/eval/<run id>
    uv run python -m eval.prereg write
    uv run python -m eval.prereg digest
    uv run python -m eval.prereg release-template
    uv run python -m eval.prereg preflight

`acceptance` writes results/acceptance/acceptance.json (every gate with numerator, denominator, threshold and
pass, plus the sha256 of every input) and copies the audit's source-free report. `write` renders PREREG.md from
src/eval/prereg_template.md and refuses unless that report passes and is not stale. `preflight` is the one check
a held-out run must pass (run.py calls `launch_check`): clean tree, freeze check, acceptance report, PREREG.md,
Zeyu's release record, and both files committed. Nothing here fills the release record; Zeyu does. No model
calls, no network, no question file is opened except the dev questions the dev score gate grades against.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from inherit.config import ROOT
from inherit.freeze import sha256_file

TEMPLATE = Path(__file__).with_name("prereg_template.md")
DEV_SCORE_THRESHOLD = 75.0
RELEASE_FIELDS = ("approved_by", "approved_utc", "prereg_sha256", "post_url", "wayback_url")
UTC_RE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?(Z|\+00:00)")
WAYBACK_PATH_RE = re.compile(r"/web/\d+/\S+")
ACCEPT_SHA_RE = re.compile(r"Acceptance report sha256 \| ([0-9a-f]{64}) \|")
PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


class PreregError(RuntimeError):
    pass


def acceptance_path(root: Path) -> Path:
    return Path(root) / "results" / "acceptance" / "acceptance.json"


def manifest_path(root: Path) -> Path:
    return Path(root) / "results" / "freeze" / "manifest.json"


def families_path(root: Path) -> Path:
    return Path(root) / "results" / "families" / "families.json"


def release_path(root: Path) -> Path:
    return Path(root) / "results" / "freeze" / "release.json"


def prereg_path(root: Path) -> Path:
    return Path(root) / "PREREG.md"


def _sha(p: Path) -> str | None:
    return sha256_file(p) if Path(p).is_file() else None


def _json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).is_file() else None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_freeze_check(root: Path) -> list[str]:
    """The differences `freeze check` finds now (an empty list means the freeze holds)."""
    from inherit import freeze

    return freeze.check(freeze.Env(root=Path(root)))


def _gate(name, numerator, denominator, threshold, ok, status: str = "evaluated") -> dict:
    """status is "evaluated", or the audit's not-estimable text for an empty dev pool (it neither passes nor fails a
    gate and is shown as such, never as a passing 0/0)."""
    return {"name": name, "numerator": numerator, "denominator": denominator, "threshold": threshold, "pass": bool(ok),
            "status": status}


# ---------------------------------------------------------------- acceptance
def audit_gates(root: Path, audit_round: int, gates_report: Callable | None = None) -> tuple[list[dict], dict]:
    """Every audit gate, read only through audit_extraction.gates_report(round_dir)."""
    if gates_report is None:
        from eval import audit_extraction

        gates_report = audit_extraction.gates_report
    rd = Path(root) / "runs" / "audit" / f"round-{audit_round}"
    rep = gates_report(rd)
    gates = [_gate(f"audit:{g['name']}", g["numerator"], g["denominator"], g["threshold"], g["pass"],
                   g.get("status", "evaluated"))
             for g in rep["gates"]]
    gates.append(_gate("audit:complete", int(bool(rep["complete"])), 1, 1, rep["complete"]))
    summary = _json(rd / "summary.json") or {}
    info = {"round": audit_round, "complete": bool(rep["complete"]), "auditors": rep.get("auditors"),
            "prompt_hashes": rep.get("prompt_hashes"), "sample_hash": rep.get("sample_hash"),
            "summary_sha256": rep.get("summary_sha256"), "verdict_hashes": summary.get("verdict_hashes")}
    return gates, info


def code_unchanged(root: Path, run_commit: str, freeze_commit: str) -> bool:
    """True when src, pyproject.toml and uv.lock are identical at the two commits (False if either is unknown)."""
    if not re.fullmatch(r"[0-9a-f]{40}", str(run_commit or "")):
        return False  # a dirty or unknown commit cannot be compared
    r = subprocess.run(["git", "diff", "--quiet", run_commit, freeze_commit, "--", "src", "pyproject.toml", "uv.lock"],
                       cwd=root, capture_output=True)
    return r.returncode == 0


def dev_score_gate(dev_run: Path, manifest: dict | None, questions=None, dev_products=None, *,
                   root: Path = ROOT, code_same: Callable | None = None) -> tuple[dict, dict]:
    """A5 person and company macro on the dev products, graded with the Task 10 grader and stats, from a complete
    A5 dev run made on the frozen system. The score is pooled over the dev products; each product's own macro is
    recorded next to it."""
    from eval import grade as G
    from eval import run as R
    from eval import stats
    from eval.arms import ARMS
    from eval.questions import DEV_PRODUCTS, load_questions

    name = "dev score: A5 person and company macro"
    fail = lambda why: (_gate(name, None, 100, DEV_SCORE_THRESHOLD, False), {"why": why})  # noqa: E731
    dev_run = Path(dev_run)
    if manifest is None:
        return fail("no freeze manifest")
    try:
        rm = R.read_manifest(dev_run)
    except (OSError, ValueError) as e:
        return fail(f"dev run manifest unreadable: {e}")
    wrong = []
    if rm.get("arm") != "A5" or rm.get("split") != "dev" or rm.get("limit") is not None:
        wrong.append("not a complete A5 dev run")
    for k, want in (("facts_hash", manifest.get("facts_hash")), ("bundle_sha", manifest.get("bundle_sha256")),
                    ("arm_config_sha", ARMS["A5"].config_sha())):
        if rm.get(k) != want:
            wrong.append(f"{k} differs")
    cfg = manifest.get("config", {})
    if rm.get("model") != cfg.get("models", {}).get("chat"):
        wrong.append("model differs")
    if rm.get("prompt_versions") != cfg.get("arms", {}).get("A5", {}).get("prompt_versions"):
        wrong.append("prompt versions differ")
    if not (code_same or code_unchanged)(root, rm.get("code_commit"), manifest.get("code_commit")):
        wrong.append("behaviour code differs between the run's commit and the freeze commit, or the commit is unknown")
    if wrong:
        return fail("not on the frozen system: " + "; ".join(wrong))
    try:
        graded = G.load_graded(dev_run)
    except Exception as e:  # noqa: BLE001
        return fail(f"dev run is not completely graded: {type(e).__name__}: {e}")
    qs = questions if questions is not None else load_questions("dev")
    if {str(g.key) for g in graded} != {str(q.key) for q in qs} or len(graded) != len(qs):
        return fail("the graded slots are not exactly the dev question set")
    prods = set(dev_products or DEV_PRODUCTS)
    if {g.key.product for g in graded} != prods:
        return fail("the dev run does not cover all six dev products")

    def macro(rows):
        groups = stats.summarize(rows, pilot=True)["groups"]
        vals = [groups["person"], groups["company"]]
        return None if None in vals else sum(vals) / 2

    score = macro(graded)
    per = {p: macro([g for g in graded if g.key.product == p]) for p in sorted(prods)}
    ok = score is not None and score >= DEV_SCORE_THRESHOLD
    return (_gate(name, None if score is None else round(score, 2), 100, DEV_SCORE_THRESHOLD, ok),
            {"per_product": {p: None if v is None else round(v, 2) for p, v in per.items()},
             "records_sha256": _sha(dev_run / "records.jsonl"), "run_id": rm.get("run_id")})


def freeze_gates(root: Path, manifest: dict | None, freeze_check: Callable) -> list[dict]:
    if manifest is None:
        return [_gate("freeze check", None, None, "0 differences", False)]
    try:
        diffs = freeze_check(root)
    except Exception as e:  # noqa: BLE001
        diffs = [f"freeze check raised {type(e).__name__}: {e}"]
    wi = _json(Path(root) / "results" / "freeze" / "whatif_check.json")
    rows = [w for ver in (wi or {}).values() for w in ver.values()]
    bad = sum(w.get("mismatches", 0) for w in rows)
    return [_gate("freeze check", len(diffs), None, "0 differences", not diffs),
            _gate("what-if verification mismatches", bad, len(rows), "0 mismatches", bool(rows) and bad == 0)]


def build_acceptance(root: Path, audit_round: int, dev_run: Path, *, freeze_check: Callable | None = None,
                     gates_report: Callable | None = None, questions=None, dev_products=None,
                     code_same: Callable | None = None) -> dict:
    root = Path(root)
    freeze_check = freeze_check or default_freeze_check
    manifest = _json(manifest_path(root))
    gates, audit = audit_gates(root, audit_round, gates_report)
    dev_gate, dev_info = dev_score_gate(dev_run, manifest, questions, dev_products, root=root, code_same=code_same)
    gates.append(dev_gate)
    gates += freeze_gates(root, manifest, freeze_check)
    gates.append(_gate("families.json present", int(families_path(root).is_file()), 1, 1, families_path(root).is_file()))
    return {
        "pass": all(g["pass"] for g in gates), "created_utc": _now(), "gates": gates, "audit": audit, "dev_run": dev_info,
        "inputs": {"manifest_sha256": _sha(manifest_path(root)), "families_sha256": _sha(families_path(root)),
                   "audit_summary_sha256": audit["summary_sha256"], "audit_verdict_hashes": audit["verdict_hashes"],
                   "dev_records_sha256": dev_info.get("records_sha256")},
    }


def write_acceptance(root: Path, audit_round: int, dev_run: Path, *, gates_report: Callable | None = None,
                     **kw) -> dict:
    root = Path(root)
    doc = build_acceptance(root, audit_round, dev_run, gates_report=gates_report, **kw)
    out = acceptance_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    rd = root / "runs" / "audit" / f"round-{audit_round}"
    if (rd / "report.md").is_file():
        shutil.copyfile(rd / "report.md", out.parent / f"audit-round-{audit_round}.md")
    return doc


# ---------------------------------------------------------------- stale and refusal checks
def acceptance_problems(root: Path, freeze_check: Callable | None = None) -> list[str]:
    """Why acceptance.json cannot be used: missing, failing or stale."""
    root = Path(root)
    doc = _json(acceptance_path(root))
    if doc is None:
        return ["results/acceptance/acceptance.json is missing"]
    out = []
    if not doc.get("pass"):
        out.append("the acceptance report is failing: " + ", ".join(g["name"] for g in doc.get("gates", [])
                                                                    if not g["pass"]))
    for key, path in (("manifest_sha256", manifest_path(root)), ("families_sha256", families_path(root))):
        if doc.get("inputs", {}).get(key) != _sha(path) or _sha(path) is None:
            out.append(f"the acceptance report is stale: {path.name} differs from the one it recorded")
    try:
        diffs = (freeze_check or default_freeze_check)(root)
    except Exception as e:  # noqa: BLE001
        diffs = [f"freeze check raised {type(e).__name__}: {e}"]
    if diffs:
        out.append(f"the acceptance report is stale: freeze check fails now ({len(diffs)} differences)")
    return out


# ---------------------------------------------------------------- PREREG
def _flat(prefix: str, v) -> list[tuple[str, str]]:
    if isinstance(v, dict):
        return [r for k, x in sorted(v.items()) for r in _flat(f"{prefix}.{k}" if prefix else k, x)]
    return [(prefix, str(v))]


def versions_table(manifest: dict) -> str:
    rows = _flat("", manifest["config"]["versions"])
    for arm, a in sorted(manifest["config"]["arms"].items()):
        rows += [(f"arm {arm} {k}", v) for k, v in sorted((a.get("prompt_versions") or {}).items())]
        rows.append((f"arm {arm} config sha256", a["config_sha"]))
    return "| Item | Version |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)


def arms_table(manifest: dict) -> str:
    allp = sorted(manifest["products"])
    lines = ["| Arm | Held-out products | Sub-cap USD |", "|---|---|---|"]
    for arm, a in sorted(manifest["arms"].items()):
        prods = "all 24" if sorted(a["products"]) == allp else ", ".join(a["products"])
        if a.get("include_families") == "multihop":
            prods += " plus every held question in the multi-hop families"
        cap = a.get("sub_cap_usd")
        lines.append(f"| {arm} | {prods} | {'default' if cap is None else cap} |")
    return "\n".join(lines)


def a2_sample(manifest: dict) -> str:
    a2 = (manifest.get("arms") or {}).get("A2")
    if not a2:
        return "none (A2 is not in the arm list)"
    return f"{len(a2['products'])} products: " + ", ".join(a2["products"])


def _cell(v) -> str:
    return "n/a" if v is None else str(v)


def _result(g: dict) -> str:
    status = g.get("status", "evaluated")
    return status if status != "evaluated" else ("pass" if g["pass"] else "FAIL")


def acceptance_table(acc: dict) -> str:
    lines = ["| Gate | Numerator | Denominator | Threshold | Result |", "|---|---|---|---|---|"]
    for g in acc["gates"]:
        lines.append(f"| {g['name']} | {_cell(g['numerator'])} | {_cell(g['denominator'])} | {_cell(g['threshold'])} "
                     f"| {_result(g)} |")
    return "\n".join(lines)


def auditors_block(acc: dict) -> str:
    a = acc["audit"]
    lines = [f"Audit round {a['round']}; sample hash {a['sample_hash']}; summary sha256 {a['summary_sha256']}.", ""]
    for title, val in (("Auditors (exact model ids and settings)", a.get("auditors")),
                       ("Prompt hashes", a.get("prompt_hashes"))):
        lines.append(f"{title}:")
        if isinstance(val, dict):
            lines += [f"- {k}: {json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v}"
                      for k, v in sorted(val.items())]
        elif isinstance(val, list):
            lines += [f"- {json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v}" for v in val]
        else:
            lines.append(f"- {val}")
        lines.append("")
    return "\n".join(lines).rstrip()


def family_texts(fam_doc: dict) -> tuple[str, str, str]:
    from eval import families as F

    fams = fam_doc["families"]
    multi = F.multi_set(fams)
    counts = ", ".join(f"{k} {v}" for k, v in sorted(fam_doc["counts"].items()))
    summary = f"{len(fams)} question families classified ({counts}); {len(multi)} are in the multi-hop set."
    excl = sorted((t, f) for t, f in fams.items() if f.get("source") == "held-unseen" and f["verdict"] != "multi")
    excluded = "\n".join(f"- {t} (Codex: {f.get('codex')}, Claude: {f.get('claude')}, result: {f['verdict']})"
                         for t, f in excl) or "none"
    return summary, F.render_block(multi), excluded


DIAG_KEYS = ("voting_instances", "visible_instances", "values_total", "values_paging_only",
             "large_group_or_missing_families", "large_group_or_missing_questions")


def dev_visibility_summary(fam_doc: dict) -> str:
    """The dev visibility sentences, from the families.json counts and `dev_diagnostics` (descriptive only)."""
    d = fam_doc.get("dev_diagnostics")
    if not isinstance(d, dict) or any(not isinstance(d.get(k), int) for k in DIAG_KEYS):
        raise PreregError("families.json has no complete dev_diagnostics; rerun `python -m eval.families dev`")
    dev = Counter(f["verdict"] for f in fam_doc["families"].values() if f.get("source") == "dev")
    return (f"Under the unchanged visibility rule, the development set yielded {dev['multi']} multi-hop, "
            f"{dev['single']} single-hop and {dev['content (not classified)']} content families; "
            f"{d['visible_instances']} of {d['voting_instances']} voting instances were visible. "
            f"A paging diagnostic found {d['values_paging_only']} of {d['values_total']} answer values visible only "
            f"inside neighbour groups larger than one page (200 cards or 12,000 tokens), and "
            f"{d['large_group_or_missing_families']} families covering {d['large_group_or_missing_questions']} "
            "development questions with every instance in that state or missing. This does not establish one-call "
            "inaccessibility, because A4 can filter, count and group before paging, so it defines no subgroup.")


def render(template: str, values: dict[str, str]) -> str:
    names = set(PLACEHOLDER_RE.findall(template))
    unknown = sorted(names - set(values))
    if unknown:
        raise PreregError(f"template placeholders with no value: {unknown}")
    text = PLACEHOLDER_RE.sub(lambda m: str(values[m.group(1)]), template)
    if PLACEHOLDER_RE.search(text) or "{{" in text:
        raise PreregError("an unfilled placeholder remains")
    return text


def write_prereg(root: Path = ROOT, *, freeze_check: Callable | None = None, template: Path = TEMPLATE,
                 now: str | None = None) -> Path:
    from eval import run as R
    from eval import stats
    from inherit.graph import cards

    root = Path(root)
    problems = acceptance_problems(root, freeze_check)
    if problems:
        raise PreregError("PREREG refused:\n  " + "\n  ".join(problems))
    manifest = _json(manifest_path(root))
    acc = _json(acceptance_path(root))
    fam_doc = _json(families_path(root))
    fam_summary, fam_block, excluded = family_texts(fam_doc)
    values = {
        "generated_utc": now or _now(), "code_commit": manifest["code_commit"],
        "upstream_herb_commit": manifest["upstream_herb_commit"], "split_hash": manifest["split_hash"],
        "bundle_sha": manifest["bundle_sha256"], "manifest_sha": sha256_file(manifest_path(root)),
        "config_hash": manifest["config_hash"], "acceptance_sha": sha256_file(acceptance_path(root)),
        "facts_hash": manifest["facts_hash"], "versions_table": versions_table(manifest),
        "card_fields": cards.render_fields(), "arms_table": arms_table(manifest), "a2_sample": a2_sample(manifest),
        "dev_visibility_summary": dev_visibility_summary(fam_doc), "family_counts": fam_summary, "family_block": fam_block, "excluded_unseen": excluded,
        "auditors": auditors_block(acc), "acceptance_table": acceptance_table(acc),
    }
    text = render(Path(template).read_text(encoding="utf-8"), values)
    found = R.PREREG_FACTS_RE.findall(text)
    if found != [manifest["facts_hash"]]:
        raise PreregError(f"the rendered file must carry the facts hash exactly once, found {len(found)} matches")
    from eval import families as F

    if stats.read_multihop_families(text) != F.multi_set(fam_doc["families"]):
        raise PreregError("the family block does not read back to the family set")
    out = prereg_path(root)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)
    return out


def digest(root: Path = ROOT) -> tuple[str, str]:
    p = prereg_path(root)
    if not p.is_file():
        raise PreregError("PREREG.md does not exist")
    h = sha256_file(p)
    return h, f"inherit-brain PREREG.md sha256: {h}"


# ---------------------------------------------------------------- release record and preflight
def write_release_template(root: Path = ROOT) -> Path | None:
    """Write results/freeze/release.json with empty fields if it is absent; never fills a field."""
    p = release_path(root)
    if p.exists():
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: "" for k in RELEASE_FIELDS}, indent=1) + "\n", encoding="utf-8")
    return p


def _committed(root: Path, rel: str) -> str | None:
    def git(*a):
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True)

    if git("ls-files", "--error-unmatch", "--", rel).returncode != 0:
        return f"{rel} is not tracked by git"
    if git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        return f"{rel} differs from HEAD"
    return None


def _https_host(url) -> tuple[str, str] | None:
    """(hostname, path) of a well-formed https URL with a host, else None. Never fetched."""
    if not isinstance(url, str) or re.search(r"\s", url):
        return None
    try:
        u = urlparse(url)
        host = u.hostname
    except ValueError:
        return None
    return (host, u.path) if u.scheme == "https" and host else None


def _release_problems(rel: dict | None) -> list[str]:
    if rel is None:
        return ["results/freeze/release.json is missing"]
    out = []
    if rel.get("approved_by") != "Zeyu":
        out.append("release.json: approved_by is not 'Zeyu'")
    ts = rel.get("approved_utc") or ""
    try:
        datetime.fromisoformat(ts.replace("Z", "+00:00"))
        ok = bool(UTC_RE.fullmatch(ts))
    except ValueError:
        ok = False
    if not ok:
        out.append("release.json: approved_utc is not an ISO UTC timestamp")
    if not _https_host(rel.get("post_url")):
        out.append("release.json: post_url is not an https URL with a host")
    w = _https_host(rel.get("wayback_url"))
    if not (w and w[0] == "web.archive.org" and WAYBACK_PATH_RE.fullmatch(w[1])):
        out.append("release.json: wayback_url is not https://web.archive.org/web/<digits>/...")
    return out


def launch_check(root: Path = ROOT, *, clean_check: Callable | None = None, freeze_check: Callable | None = None,
                 committed: Callable | None = None) -> dict:
    """All launch checks, failing closed: {"checks": {name: None | message}, "failures": [...], "pass": bool}."""
    from eval import receipt as rc
    from eval import run as R

    root = Path(root)
    clean_check = clean_check or rc.require_clean_tree
    committed = committed or _committed
    checks: dict[str, list[str]] = {}

    def run(name, fn):
        try:
            checks[name] = fn()
        except Exception as e:  # noqa: BLE001
            checks[name] = [f"{name}: {type(e).__name__}: {e}"]

    def clean():
        clean_check(root)
        return []

    def freeze():
        d = (freeze_check or default_freeze_check)(root)
        return [f"freeze check: {x}" for x in d[:5]] + ([f"freeze check: {len(d)} differences in all"] if d else [])

    def acceptance():
        out = acceptance_problems(root, freeze_check)
        p = prereg_path(root)
        m = ACCEPT_SHA_RE.search(p.read_text(encoding="utf-8")) if p.is_file() else None
        if m is None or m.group(1) != _sha(acceptance_path(root)):
            out.append("acceptance.json sha256 differs from the one written in PREREG.md")
        return out

    def prereg():
        p = prereg_path(root)
        if not p.is_file():
            return ["PREREG.md is missing"]
        text = p.read_text(encoding="utf-8")
        found = R.PREREG_FACTS_RE.findall(text)
        man = _json(manifest_path(root)) or {}
        out = []
        if len(found) != 1 or found[0] != man.get("facts_hash"):
            out.append("PREREG.md facts hash is missing, repeated, or differs from the manifest's")
        rel = _json(release_path(root)) or {}
        if sha256_file(p) != rel.get("prereg_sha256"):
            out.append("PREREG.md sha256 differs from release.json prereg_sha256")
        return out

    def release():
        return _release_problems(_json(release_path(root)))

    def tracked():
        return [m for m in (committed(root, "results/freeze/release.json"), committed(root, "PREREG.md")) if m]

    for name, fn in (("clean tree", clean), ("freeze check", freeze), ("acceptance report", acceptance),
                     ("PREREG.md", prereg), ("release record", release), ("committed", tracked)):
        run(name, fn)
    failures = [m for ms in checks.values() for m in ms]
    return {"checks": {k: (ms[0] if ms else None) for k, ms in checks.items()}, "details": checks,
            "failures": failures, "pass": not failures}


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval.prereg")
    ap.add_argument("--repo-root", default=None, help="tests only")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acceptance")
    a.add_argument("--audit-round", type=int, required=True)
    a.add_argument("--dev-run", type=Path, required=True)
    for n in ("write", "digest", "release-template", "preflight"):
        sub.add_parser(n)
    args = ap.parse_args(argv)
    root = Path(args.repo_root) if args.repo_root else ROOT
    try:
        if args.cmd == "acceptance":
            doc = write_acceptance(root, args.audit_round, args.dev_run)
            for g in doc["gates"]:
                print(f"{'pass' if g['pass'] else 'FAIL'}  {g['name']}: {_cell(g['numerator'])}/{_cell(g['denominator'])} "
                      f"(threshold {g['threshold']})")
            print(f"acceptance {'PASSES' if doc['pass'] else 'FAILS'}; wrote {acceptance_path(root)}")
            return 0 if doc["pass"] else 1
        if args.cmd == "write":
            print(f"wrote {write_prereg(root)}")
        elif args.cmd == "digest":
            h, line = digest(root)
            print(h)
            print(line)
        elif args.cmd == "release-template":
            p = write_release_template(root)
            print(f"wrote {p}" if p else f"{release_path(root)} exists; left unchanged")
        else:
            res = launch_check(root)
            for name, msgs in res["details"].items():
                print(f"{'FAIL' if msgs else 'pass'}  {name}")
                for m in msgs:
                    print(f"      {m}")
            return 0 if res["pass"] else 1
    except PreregError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
