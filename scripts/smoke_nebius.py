"""One live metered chat call with a tool definition, plus one tiny embed call."""

from inherit.llm import chat, embed

BUG_BOARD = {
    "type": "function",
    "function": {
        "name": "bug_board",
        "description": "Count bugs grouped by customer or engineer, filtered by status.",
        "parameters": {
            "type": "object",
            "properties": {
                "group_by": {"type": "string", "enum": ["customer", "engineer"]},
                "status": {
                    "type": "string",
                    "enum": ["unresolved", "active", "reopened", "no_fix_needed"],
                },
            },
            "required": ["group_by", "status"],
        },
    },
}

res = chat(
    [{"role": "user", "content": "Which customer has the most unresolved bugs?"}],
    arm="smoke",
    tools=[BUG_BOARD],
)
calls = [(c.function.name, c.function.arguments) for c in (res.message.tool_calls or [])]
print("tool_calls:", calls)
print("tokens:", res.prompt_tokens, "in,", res.completion_tokens, "out")
print("cost_usd:", res.cost_usd)
print("latency_s:", round(res.latency_s, 2))

e = embed(["unresolved bugs"], arm="smoke")
print("embed dims:", len(e.vectors[0]), "tokens:", e.prompt_tokens, "cost_usd:", e.cost_usd)
