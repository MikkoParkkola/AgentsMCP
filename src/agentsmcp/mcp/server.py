from __future__ import annotations

"""AgentsMCP MCP Gateway Server.

This module exposes the tool registry and the generic mcp_call tool over MCP,
advertising the latest protocol version while providing downgrade compatibility
for older MCP client versions. It enables older MCP clients to access new tools.

Implementation notes
- Uses the optional `mcp` Python SDK if available; otherwise this module
  provides a stub entry and guidance.
- Version negotiation is handled via a simple min/max handshake and
  schema down-conversion for older versions.
- Transport: stdio by default; can be run as SSE or WebSocket where supported.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..tools import tool_registry
from ..tools.mcp_tool import MCPCallTool
from ..config import Config

try:
    # Hypothetical MCP server interface; actual signatures may differ.
    from mcp.server import Server, stdio_serve  # type: ignore
    MCP_SERVER_AVAILABLE = True
except Exception:  # pragma: no cover
    Server = object  # type: ignore
    stdio_serve = None  # type: ignore
    MCP_SERVER_AVAILABLE = False

LATEST_VERSION = "1.0"  # placeholder for latest supported MCP protocol version
BACKCOMPAT_VERSIONS = ["1.0", "0.5", "0.4", "0.3"]

log = logging.getLogger(__name__)


def _negotiate_version(client_versions: List[str]) -> Tuple[str, List[str]]:
    """Pick a protocol version supported by both server and client.

    Returns the selected version and the full list of server-supported versions.
    """
    for v in BACKCOMPAT_VERSIONS:
        if v in client_versions:
            return v, BACKCOMPAT_VERSIONS
    # Default to latest if client didn't send list (assume modern)
    return LATEST_VERSION, BACKCOMPAT_VERSIONS


def _downgrade_tool_schema(schema: Dict[str, Any], to_version: str) -> Dict[str, Any]:
    """Best-effort schema down-conversion for older clients.

    Strategy: strip unknown top-level fields and keep JSONSchema core (type,
    properties, required). This stays compatible with many earlier MCP versions
    that consumed simple JSON function schemas.
    """
    allowed = {"type", "properties", "required", "description", "enum", "default"}
    return {k: v for k, v in schema.items() if k in allowed}


def _tools_for_version(to_version: str) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    for t in tool_registry.get_all_tools():
        fn = t.to_openai_function()["function"]
        schema = fn["parameters"]
        if to_version != LATEST_VERSION:
            schema = _downgrade_tool_schema(schema, to_version)
        tools.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": schema,
        })
    return tools


async def serve_stdio(config: Optional[Config] = None) -> None:
    """Run the MCP gateway over stdio."""
    if not MCP_SERVER_AVAILABLE:  # pragma: no cover
        log.error("MCP server library not installed. Install a Python MCP server SDK.")
        return

    cfg = config or Config()
    server = Server("agentsmcp-gateway")  # type: ignore[call-arg]

    # Negotiate protocol version with client
    @server.on_handshake  # type: ignore[attr-defined]
    async def _handshake(info: Dict[str, Any]) -> Dict[str, Any]:
        client_versions = info.get("versions", []) if isinstance(info, dict) else []
        selected, supported = _negotiate_version(client_versions)
        return {"selected_version": selected, "supported_versions": supported}

    # List tools
    @server.method("tools/list")  # type: ignore[attr-defined]
    async def list_tools(params: Dict[str, Any]) -> Dict[str, Any]:
        to_version = params.get("protocol_version", LATEST_VERSION)
        return {"protocol_version": to_version, "tools": _tools_for_version(to_version)}

    # Execute a tool call
    @server.method("tools/execute")  # type: ignore[attr-defined]
    async def execute_tool(params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not name:
            return {"error": "Missing tool name"}
        tool = tool_registry.get_tool(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = tool.execute(**(arguments or {}))
            return {"ok": True, "result": result}
        except Exception as e:  # pragma: no cover
            log.exception("Tool execution failed")
            return {"error": str(e)}

    # Expose mcp_call gateway tool if any MCP servers configured
    if getattr(cfg, "mcp", None):
        # Allow all configured servers by default
        try:
            from ..tools.mcp_tool import MCPCallTool
            gateway_tool = MCPCallTool(cfg)
            tool_registry.register(gateway_tool)
        except Exception:
            pass

    await stdio_serve(server)  # type: ignore[func-returns-value]


def run_stdio_blocking(config: Optional[Config] = None) -> None:
    asyncio.run(serve_stdio(config))
