"""The only module that reads HERB question files (rule 2).

Gold answers and citations live on `Question` and stay on the eval side. Arms are handed
`ExecInput(product, text)` built by `Question.exec_input()` and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from inherit.config import ROOT
from inherit.contracts import ExecInput, QuestionKey
from inherit.data.split import split

RAW_DIR = ROOT / "data" / "raw"

# Frozen inventory (pre-registration). load_questions refuses anything else.
DEV_PRODUCTS = ("ActionGenie", "AnomalyForce", "ContentForce", "KnowledgeForce", "LeadForce", "TrendForce")
FROZEN_COUNTS = {"dev": (156, 155), "held": (659, 544)}  # answerable, unanswerable
FROZEN_SEEN = (416, 243)  # held answerable questions whose template is seen / unseen


@dataclass(frozen=True)
class Question:
    key: QuestionKey
    text: str
    type: str  # person, company, pr, url, content, or unanswerable
    gold: Any  # list[str] for ID types, str (or list) for content, None for unanswerable
    citations: list[str] = field(default_factory=list)
    seen: bool | None = None  # held split only: the normalised string also occurs among dev answerable questions

    @property
    def product(self) -> str:
        return self.key.product

    def exec_input(self) -> ExecInput:
        return ExecInput(self.product, self.text)


def template(text: str, product: str) -> str:
    """The seen/unseen normaliser. Copied unchanged from analysis/herb_split_templates.py (`norm`)."""
    q = text.replace(product, "<P>")
    q = re.sub(r"eid_[0-9a-f]+|EMP_\d+", "<E>", q); q = re.sub(r"https?://\S+", "<URL>", q)
    q = re.sub(r"\b[A-Za-z]*Force[A-Za-z]*\b", "<X>", q); q = re.sub(r"\d+", "<N>", q)
    q = re.sub(r"\b(TechCorp|FusionTech|BlueWave|DataSolutions|NextGenTech|CloudSync|ComputeWorks|InnovateAI|AITransformers|SmartData)\b", "<C>", q)
    return q.strip().rstrip("?.").lower()


def snapshot_hash(questions: list["Question"]) -> str:
    """sha256 of the canonical eval-side content of questions: key, text, type, gold, citations, in the given order."""
    rows = [[str(q.key), q.text, q.type, q.gold, list(q.citations)] for q in questions]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _validate_frozen(split_name: str, dev: list[str], out: list["Question"]) -> None:
    if set(dev) != set(DEV_PRODUCTS):
        raise ValueError(f"frozen dev split changed: {dev}")
    got = (sum(q.key.kind == "ans" for q in out), sum(q.key.kind == "unans" for q in out))
    if got != FROZEN_COUNTS[split_name]:
        raise ValueError(f"{split_name} split has {got[0]} answerable and {got[1]} unanswerable questions, "
                         f"expected {FROZEN_COUNTS[split_name]}")
    if split_name == "held":
        seen = (sum(q.seen is True for q in out if q.key.kind == "ans"), sum(q.seen is False for q in out if q.key.kind == "ans"))
        if seen != FROZEN_SEEN:
            raise ValueError(f"held seen/unseen counts {seen}, expected {FROZEN_SEEN}")


def _read(raw_dir: Path, product: str) -> dict:
    return json.loads((Path(raw_dir) / "products" / f"{product}.json").read_text(encoding="utf-8"))


def _product_questions(product: str, d: dict) -> list[Question]:
    out = [
        Question(QuestionKey(product, "ans", i), q["question"], q["type"], q["ground_truth"], list(q["citations"]))
        for i, q in enumerate(d["answerable_questions"])
    ]
    out += [
        Question(QuestionKey(product, "unans", i), text, "unanswerable", None, [])
        for i, text in enumerate(d["unanswerable_questions"])
    ]
    return out


def _names(raw_dir: Path) -> list[str]:
    return sorted(p.stem for p in (Path(raw_dir) / "products").glob("*.json"))


def load_all_questions(raw_dir: Path = RAW_DIR) -> list[Question]:
    """Every question of every product in sorted product order, with no split applied (seen is None)."""
    out: list[Question] = []
    for name in _names(raw_dir):
        out += _product_questions(name, _read(raw_dir, name))
    return out


def load_questions(split_name: Literal["dev", "held"], raw_dir: Path = RAW_DIR) -> list[Question]:
    """The questions of one split in the frozen split order. Held questions carry `seen`."""
    if split_name not in ("dev", "held"):
        raise ValueError(f"split must be 'dev' or 'held', got {split_name!r}")
    names = _names(raw_dir)
    dev, held = split(names)
    need = dev if split_name == "dev" else dev + held  # held needs the dev templates; dev never opens a held file
    data = {n: _read(raw_dir, n) for n in need}
    out: list[Question] = []
    if split_name == "dev":
        for n in dev:
            out += _product_questions(n, data[n])
        _validate_frozen("dev", dev, out)
        return out
    dev_templates = {template(q["question"], n) for n in dev for q in data[n]["answerable_questions"]}
    for n in held:
        for q in _product_questions(n, data[n]):
            out.append(Question(q.key, q.text, q.type, q.gold, q.citations, template(q.text, n) in dev_templates))
    _validate_frozen("held", dev, out)
    return out
