"""The arm registry. An arm turns an ExecInput into an ArmResult; the runner does everything else."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from inherit.contracts import CallUsage, ExecInput, FinalAnswer


@dataclass(frozen=True)
class ArmResult:
    """One question's outcome as the arm reports it.

    `calls` is every model call with its usage, `envelopes` the tool response envelopes (kept in a gzip file
    beside the run), `failure` the arm's own failure reason, for example its cooperative 300 s deadline. An arm
    that fails still returns its single terminal result, with outcome run_failed.
    """

    final: FinalAnswer
    calls: list[CallUsage] = field(default_factory=list)
    envelopes: list[Any] = field(default_factory=list)
    failure: str | None = None


@dataclass(frozen=True)
class Arm:
    name: str
    run: Callable[[ExecInput], ArmResult]
    sub_cap_usd: float | None = None  # pushed to the proxy's /admin/arm_cap before a run
    config: dict[str, Any] = field(default_factory=dict)  # every knob that changes behaviour; hashed into the manifest
    prompt_versions: dict[str, str] = field(default_factory=dict)

    def config_sha(self) -> str:
        blob = json.dumps({"name": self.name, "config": self.config, "sub_cap_usd": self.sub_cap_usd},
                          sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


ARMS: dict[str, Arm] = {}


def register(arm: Arm) -> Arm:
    if arm.name in ARMS:
        raise ValueError(f"arm {arm.name!r} is already registered")
    ARMS[arm.name] = arm
    return arm


def _register_agent_arms() -> None:
    """A5, A4 and A4c (and their dev twins for pilots) differ only in the toolset; budgets, prompt and answer
    handling are the same agent. Imported late: the agent builds its results from this module's ArmResult."""
    from inherit import agent

    for name, toolset in (("A5", "full"), ("A4", "depth1"), ("A4c", "depth1_chain")):
        for arm_name in (name, f"dev-{name.lower()}"):
            register(Arm(arm_name, lambda inp, _n=arm_name, _t=toolset: agent.answer(inp, toolset=_t, arm=_n),
                         config=agent.arm_config(toolset), prompt_versions={"agent": agent.PROMPT_VERSION}))


_register_agent_arms()


from baselines import graphrag_sdk_arm as _a2  # noqa: E402,F401  registers A2
