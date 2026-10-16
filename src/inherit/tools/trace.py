"""The trace of one question: every tool call as ordered TraceOps (tool, command, query, params, rows, ms).

`Trace.run` is the only way the A5 agent calls a tool. It counts every call against the shared budget (a refused call,
a page of an earlier result and a what-if call count too) and records what actually ran: the actual bound parameters,
one op per executed query. Read-only tools and the what-if tool are separate classes of operation (`Tool.kind`).
"""

from __future__ import annotations

import time

from inherit.contracts import TraceOp
from inherit.tools.catalog import TOOLS, ToolError, ToolResult
from inherit.tools.depth1 import CALL_CAP


class Trace:
    def __init__(self, max_calls: int = CALL_CAP):
        self.max_calls = max_calls
        self.calls = 0
        self.ops: list[TraceOp] = []

    def run(self, conn, product: str, name: str, /, **params) -> ToolResult:
        if self.calls >= self.max_calls:
            raise ToolError(f"call limit reached: at most {self.max_calls} calls per question")
        self.calls += 1
        t0 = time.perf_counter()
        try:
            tool = TOOLS.get(name)
            if tool is None:
                raise ToolError(f"unknown tool {name!r}; tools: {', '.join(sorted(TOOLS))}")
            res = tool.run(conn, product, **params)
        except (ToolError, NotImplementedError) as e:
            self.ops.append(TraceOp(name, "refused", "", {**params, "error": str(e) or type(e).__name__}, 0,
                                    round((time.perf_counter() - t0) * 1000, 3)))
            raise
        self.ops.extend(res.ops)
        return res

    def to_dicts(self) -> list[dict]:
        return [op.to_dict() for op in self.ops]
