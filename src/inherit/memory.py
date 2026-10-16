"""In-session memory (SPEC 5.6): a user confirms an answer, and the same question in the same session is answered from it.

A `Finding` is bound to a product, the confirmed question's normalised text, the resolved root ids and a predicate
name, and keeps the original path (the trace and the evidence ids). A `Session` lives in process memory only: nothing
is written to disk or to the graph, and a new session knows nothing. Not used in any evaluation run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from inherit.contracts import ExecInput, FinalAnswer, TraceOp
from inherit.tools.depth1 import fold


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Finding:
    product: str
    question: str  # as asked; matched on its normalised form
    text: str
    answer_ids: list[str]
    evidence_ids: list[str]
    trace: list[TraceOp]  # the original path
    confirmed_at: str
    confirmed_by: str
    roots: tuple[str, ...] = ()  # resolved root ids the answer was about
    predicate: str | None = None  # a name for the relation asked; with the roots it identifies the same request

    @classmethod
    def from_final(cls, inp: ExecInput, final: FinalAnswer, *, confirmed_by: str, confirmed_at: str | None = None,
                   roots: tuple[str, ...] = (), predicate: str | None = None) -> "Finding":
        if final.outcome == "run_failed":
            raise ValueError("a failed run cannot be confirmed")
        return cls(inp.product, inp.text, final.answer_text, list(final.answer_ids), list(final.evidence_ids),
                   list(final.trace), confirmed_at or now_utc(), confirmed_by, tuple(roots), predicate)


@dataclass
class Session:
    findings: list[Finding] = field(default_factory=list)

    def confirm(self, finding: Finding) -> None:
        self.findings.append(finding)

    def find(self, product: str, question: str, *, predicate: str | None = None,
             roots: tuple[str, ...] | None = None) -> Finding | None:
        """The latest finding for the same product and the same normalised question text, or for the same predicate
        over the same roots. Anything else is a different request and gets no memory."""
        q = fold(question)
        for f in reversed(self.findings):
            if f.product != product:
                continue
            if fold(f.question) == q:
                return f
            if predicate is not None and roots is not None and f.predicate == predicate \
                    and set(f.roots) == set(roots):
                return f
        return None
