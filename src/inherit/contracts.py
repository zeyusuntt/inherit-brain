"""Shared contracts. Every extraction, agent, arm and grader module builds on these."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

OUTCOMES = frozenset({"path", "memory", "not_in_graph", "unknown_name", "ambiguous_name", "partial", "run_failed"})

# Which FinalAnswer field is graded for each question type. Evidence is never graded.
GRADED_FIELD = {
    "person": "answer_ids",
    "company": "answer_ids",
    "pr": "answer_ids",
    "url": "answer_ids",
    "content": "answer_text",
    "unanswerable": "answer_text",
}

KINDS = ("ans", "unans")


@dataclass(frozen=True)
class ExecInput:
    """All an arm ever sees: the product name and the question text."""

    product: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"product": self.product, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExecInput":
        return cls(product=d["product"], text=d["text"])


@dataclass(frozen=True)
class TraceOp:
    tool: str
    command: str
    query: str
    params: dict[str, Any]
    row_count: int
    ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool, "command": self.command, "query": self.query,
            "params": dict(self.params), "row_count": self.row_count, "ms": self.ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TraceOp":
        return cls(d["tool"], d["command"], d["query"], dict(d["params"]), d["row_count"], d["ms"])


@dataclass(frozen=True)
class CallUsage:
    arm: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm, "model": self.model, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "cost_usd": self.cost_usd, "latency_s": self.latency_s,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CallUsage":
        return cls(d["arm"], d["model"], d["prompt_tokens"], d["completion_tokens"], d["cost_usd"], d["latency_s"])


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    n_calls: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd, "n_calls": self.n_calls,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Usage":
        return cls(d["prompt_tokens"], d["completion_tokens"], d["cost_usd"], d["n_calls"])


@dataclass(frozen=True)
class FinalAnswer:
    outcome: str
    answer_text: str
    answer_ids: list[str]
    evidence_ids: list[str]
    trace: list[TraceOp]
    usage: Usage

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.outcome!r}; allowed: {sorted(OUTCOMES)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome, "answer_text": self.answer_text,
            "answer_ids": list(self.answer_ids), "evidence_ids": list(self.evidence_ids),
            "trace": [t.to_dict() for t in self.trace], "usage": self.usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FinalAnswer":
        return cls(
            outcome=d["outcome"], answer_text=d["answer_text"], answer_ids=list(d["answer_ids"]),
            evidence_ids=list(d["evidence_ids"]), trace=[TraceOp.from_dict(t) for t in d["trace"]],
            usage=Usage.from_dict(d["usage"]),
        )


@dataclass(frozen=True)
class QuestionKey:
    product: str
    kind: Literal["ans", "unans"]
    index: int

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be 'ans' or 'unans', got {self.kind!r}")
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise ValueError(f"index must be a non-negative int, got {self.index!r}")
        if not self.product or "/" in self.product:
            raise ValueError(f"bad product name {self.product!r}")

    def __str__(self) -> str:
        return f"{self.product}/{self.kind}/{self.index}"

    @classmethod
    def parse(cls, s: str) -> "QuestionKey":
        parts = s.split("/")
        if len(parts) != 3:
            raise ValueError(f"bad question key {s!r}: expected <product>/<kind>/<index>")
        product, kind, idx = parts
        if not (idx.isascii() and idx.isdigit()):
            raise ValueError(f"bad question key {s!r}: index must be a non-negative integer")
        return cls(product, kind, int(idx))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {"product": self.product, "kind": self.kind, "index": self.index}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuestionKey":
        return cls(d["product"], d["kind"], d["index"])


@dataclass(frozen=True)
class RunRecord:
    key: QuestionKey
    arm: str
    final: FinalAnswer
    calls: list[CallUsage] = field(default_factory=list)
    tool_ops: list[TraceOp] = field(default_factory=list)
    latency_s: float = 0.0
    failure: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(), "arm": self.arm, "final": self.final.to_dict(),
            "calls": [c.to_dict() for c in self.calls], "tool_ops": [t.to_dict() for t in self.tool_ops],
            "latency_s": self.latency_s, "failure": self.failure,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunRecord":
        return cls(
            key=QuestionKey.from_dict(d["key"]), arm=d["arm"], final=FinalAnswer.from_dict(d["final"]),
            calls=[CallUsage.from_dict(c) for c in d["calls"]], tool_ops=[TraceOp.from_dict(t) for t in d["tool_ops"]],
            latency_s=d["latency_s"], failure=d["failure"],
        )
