"""Fetch the pinned HERB files into data/raw/. Raw data is git-ignored and never committed."""

from __future__ import annotations

import ssl
import sys
import urllib.request
from pathlib import Path
from typing import Callable

from inherit.config import ROOT

REPO = "SalesforceAIResearch/HERB"
PINNED_SHA = "db3bf9b3f911745726c579c9dbf9f7f6b2c05b36"
RAW_DIR = ROOT / "data" / "raw"

PRODUCT_NAMES = (
    "ActionGenie", "AnomalyForce", "AutoTuneForce", "CoachForce", "CollaborateForce", "CollaborationForce",
    "ConnectForce", "ContentForce", "ContextForce", "EdgeForce", "ExplainabilityForce", "FeedbackForce",
    "FlowForce", "ForecastForce", "InsightForce", "KnowledgeForce", "LeadForce", "MonitorForce",
    "PersonalizeForce", "PitchForce", "ProposalForce", "SearchFlow", "SearchForce", "SecurityForce",
    "SentimentForce", "SummarizeForce", "SupportForce", "TrendForce", "VizForce", "WorkFlowGenie",
)
METADATA_FILES = ("employee.json", "customers_data.json", "salesforce_team.json")
CODE_FILES = (
    "evaluate.py", "react.py", "metadata_search_functions.py", "long_context_eval.py", "utils.py", "product_rag.py",
)
TOP_FILES = ("LICENSE.txt", "AI_ETHICS.md")


def _context() -> ssl.SSLContext:
    # python.org builds on macOS ship without a CA bundle; certifi arrives with httpx.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60, context=_context()) as r:
        return r.read()


def plan() -> list[tuple[str, str]]:
    """(path in the HERB repo, path under data/raw) for every file we fetch."""
    items = [(f"data/products/{n}.json", f"products/{n}.json") for n in PRODUCT_NAMES]
    items += [(f"data/metadata/{f}", f"metadata/{f}") for f in METADATA_FILES]
    items += [(f"code/{f}", f"code/{f}") for f in CODE_FILES]
    items += [(f, f) for f in TOP_FILES]
    return items


def download(dest: Path, sha: str = PINNED_SHA, fetch: Callable[[str], bytes] = _fetch) -> None:
    dest = Path(dest)
    for repo_path, rel in plan():
        url = f"https://raw.githubusercontent.com/{REPO}/{sha}/{repo_path}"
        body = fetch(url)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(body)
        tmp.replace(target)
    (dest / "COMMIT").write_text(sha + "\n")


def main() -> int:
    download(RAW_DIR, PINNED_SHA)
    print(f"downloaded {len(plan())} files at {PINNED_SHA} into {RAW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
