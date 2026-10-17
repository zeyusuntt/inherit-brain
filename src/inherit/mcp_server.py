"""An MCP server (stdio) over the A5 graph tools, exactly as the A5 agent sees them.

One server process is bound to one product (`--product`), the same way one A5 question is: the tool schemas have no
product argument, and the connection is the read-only `agent_<product>` user on the 6.0 server (never the build or
admin user). Tool names, descriptions and JSON schemas are the A5 catalogue's (`agent._Full.specs`), and a call runs
through `agent._Full.call`, so the result is the same envelope the model gets: {"rows", "truncated", "cursor"[, "note"]}.
A bad call comes back as an MCP error result {"error": ...}. The loaded products are listed by the resource
`inherit://products`.

    uv run python -m inherit.mcp_server --product ActionGenie
"""

from __future__ import annotations

import argparse
import asyncio
import json

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from inherit import agent
from inherit.config import ROOT
from inherit.tools.catalog import ToolError

PRODUCTS_URI = "inherit://products"


def list_products() -> list[str]:
    """The products with a read-only agent user on the 6.0 server (names only, from data/acl.json)."""
    users = json.loads((ROOT / "data" / "acl.json").read_text())[agent.SERVER]
    return sorted(u.removeprefix("agent_") for u in users if u.startswith("agent_"))


def specs() -> list[dict]:
    """The A5 tool specs, as the agent's function-calling layer builds them."""
    return agent._Full(None, "").specs()  # noqa: SLF001 - the very list the agent offers


def call(conn, product: str, name: str, args: dict) -> tuple[dict, bool]:
    """Run one tool as the agent does. Returns (envelope, is_error). A fresh call budget per call."""
    try:
        env, _rows = agent._Full(conn, product).call(name, args)  # noqa: SLF001
        return env, False
    except (ToolError, TypeError, ValueError) as e:
        return {"error": str(e)}, True


def build_server(product: str, graph=None) -> Server:
    """`graph` is for tests; by default the read-only agent graph of `product` is opened lazily on the first call."""
    state = {"conn": graph}

    def conn():
        if state["conn"] is None:
            state["conn"] = agent.open_graph(product)
        return state["conn"]

    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        tools = [types.Tool(name=s["function"]["name"], description=s["function"]["description"],
                            inputSchema=s["function"]["parameters"]) for s in specs()]
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx, params) -> types.CallToolResult:
        env, is_error = await asyncio.to_thread(lambda: call(conn(), product, params.name, params.arguments or {}))
        text = json.dumps(env, ensure_ascii=False, separators=(",", ":"))
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=is_error)

    async def on_list_resources(ctx, params) -> types.ListResourcesResult:
        return types.ListResourcesResult(resources=[types.Resource(
            name="products", uri=PRODUCTS_URI, mime_type="application/json",
            description=f"The loaded products. This server is bound to {product}.")])

    async def on_read_resource(ctx, params) -> types.ReadResourceResult:
        if str(params.uri) != PRODUCTS_URI:
            raise ValueError(f"unknown resource {params.uri}")
        body = json.dumps({"bound": product, "products": list_products()})
        return types.ReadResourceResult(contents=[types.TextResourceContents(
            uri=PRODUCTS_URI, mime_type="application/json", text=body)])

    return Server(f"inherit-brain:{product}", on_list_tools=on_list_tools, on_call_tool=on_call_tool,
                  on_list_resources=on_list_resources, on_read_resource=on_read_resource)


async def serve(product: str) -> None:
    server = build_server(product)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--product", help="the product graph to serve, e.g. ActionGenie")
    ap.add_argument("--list-products", action="store_true", help="print the loaded products and exit")
    a = ap.parse_args(argv)
    if a.list_products:
        print("\n".join(list_products()))
        return
    if a.product not in list_products():  # also catches a missing --product
        ap.error(f"unknown product {a.product!r}; see --list-products")
    asyncio.run(serve(a.product))


if __name__ == "__main__":
    main()
