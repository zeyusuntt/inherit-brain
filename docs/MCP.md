# MCP server

`inherit.mcp_server` serves the nine A5 graph tools over MCP (stdio). The tool names, descriptions and JSON schemas are
the ones the A5 agent gets (`agent._Full.specs`), and a call runs through the same code, so the result is the envelope the
model sees: `{"rows", "truncated", "cursor"}` plus `"note"` when the tool has a hint. A bad call returns an error result
`{"error": "..."}`. The agent's 12-call per-question cap does not apply to a session.

The server connects as the read-only `agent_<product>` user on the FalkorDB 6.0 server (port 6380) and never uses the build
or admin user. It needs `docker compose up -d` and a loaded graph (`data/acl.json`).

## Product selection

One server process is bound to one product with `--product`. The tool schemas stay identical to the agent's (no product
argument), and the agent's graph is scoped to one product the same way. To work across products, register one server per
product. The resource `inherit://products` lists the 30 loaded products and the bound one;
`python -m inherit.mcp_server --list-products` prints the same list.

```
uv run python -m inherit.mcp_server --product ActionGenie
```

## Client config

Claude Code:

```
claude mcp add inherit-actiongenie -- uv run --project /path/to/inherit-brain python -m inherit.mcp_server --product ActionGenie
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "inherit-actiongenie": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/inherit-brain", "python", "-m", "inherit.mcp_server",
               "--product", "ActionGenie"]
    }
  }
}
```

## Tools

| Tool | Use |
|---|---|
| `resolve_person` | name text to canonical person ids |
| `resolve_entity` | text to a customer, document, PR, bug, feature or other node |
| `doc_people` | who authored, reviewed or is named in a document |
| `bug_board` | bugs by status, component, company, actor, with ranking |
| `fix_trail` | the fix PRs and events of a bug |
| `feature_fate` | what happened to a feature, and why |
| `who_said` | who reported, demoed or worked on a subject |
| `pr_board` | pull requests by state, author, reviewer, feature |
| `bridges` | people who connect groups (betweenness on collaboration edges) |

Each tool's full description and parameter schema come from `tools/list`.

## Example session

```
> tools/call who_said {"subject": "previous_release", "kind": "worked_on", "role": "Technical Architect", "return": "people"}
{"rows":[{"answer":"eid_a253c65a","name":"Hannah Brown","role":"Technical Architect","via":["author"],
  "refs":["Document:tuneprox_technical_specifications_document_final","Person:eid_a253c65a"],
  "evidence":["tuneprox_technical_specifications_document_final"]}, ...],"truncated":false,"cursor":null}
```

(started with `--product AutoTuneForce`). Ids in `answer` and `evidence` are what you quote back as the source.

## Related CLI

`uv run python -m inherit.ask --product ActionGenie "Who reviewed the market research report?"` runs the A5 agent live and
prints the answer, each tool call with its Cypher, row count and milliseconds, and the evidence ids. It makes model calls, so
start the metering proxy first (`uv run python -m inherit.proxy`). `--replay runs/eval/A5-held --slot AutoTuneForce/ans/2`
re-executes the recorded tool calls of a held-out slot against the live graph with no model call and prints fresh timings next
to the recorded ones.
