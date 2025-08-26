from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from ..config import Config
from ..mcp.manager import MCPManager, MCPServer, get_global_manager
from .base_tools import BaseTool


class MCPCallTool(BaseTool):
    """Generic MCP call tool.

    Allows the agent to invoke a tool exposed by any configured MCP server.
    You can restrict which servers are callable via `allowed_servers`.
    """

    def __init__(self, config: Config, allowed_servers: Optional[List[str]] = None):
        super().__init__(
            name="mcp_call",
            description=(
                "Invoke a tool exposed by a configured MCP server. "
                "Parameters: server name, tool name, and tool-specific params."
            ),
        )
        self._config = config
        self._allowed_servers = set(allowed_servers) if allowed_servers else None

    def _build_manager(self) -> MCPManager:
        servers: List[MCPServer] = []
        for s in getattr(self._config, "mcp", []) or []:
            servers.append(
                MCPServer(
                    name=s.name,
                    command=s.command,
                    transport=s.transport,
                    url=s.url,
                    env=s.env or {},
                    cwd=s.cwd,
                    enabled=s.enabled,
                )
            )
        # Pass transport flags from config
        allow_stdio = bool(getattr(self._config, "mcp_stdio_enabled", True))
        allow_ws = bool(getattr(self._config, "mcp_ws_enabled", False))
        allow_sse = bool(getattr(self._config, "mcp_sse_enabled", False))
        return get_global_manager(servers, allow_stdio=allow_stdio, allow_ws=allow_ws, allow_sse=allow_sse)

    async def aexecute(self, server: str, tool: str, params: Optional[Dict[str, Any]] = None) -> str:
        # Validate server access
        if self._allowed_servers is not None and server not in self._allowed_servers:
            return (
                f"Error: MCP server '{server}' is not allowed for this agent. "
                f"Allowed: {sorted(self._allowed_servers)}"
            )
        params = params or {}

        mgr = self._build_manager()
        return await mgr.call_tool(server, tool, params)

    def execute(self, server: str, tool: str, params: Optional[Dict[str, Any]] = None) -> str:
        # Validate server access
        if self._allowed_servers is not None and server not in self._allowed_servers:
            return (
                f"Error: MCP server '{server}' is not allowed for this agent. "
                f"Allowed: {sorted(self._allowed_servers)}"
            )
        params = params or {}

        mgr = self._build_manager()
        try:
            return asyncio.run(mgr.call_tool(server, tool, params))
        except RuntimeError as e:
            # Likely called inside an event loop; fall back to a user-friendly message
            return (
                "Unable to execute MCP call synchronously within a running event loop.\n"
                f"Requested call -> server: {server}, tool: {tool}, params: {params}.\n"
                "Consider calling .aexecute(...) from async code for proper async-first behavior."
            )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "Name of the configured MCP server to use",
                },
                "tool": {
                    "type": "string",
                    "description": "Tool name exposed by the MCP server",
                },
                "params": {
                    "type": "object",
                    "description": "Tool-specific parameters (object)",
                    "additionalProperties": True,
                    "default": {},
                },
            },
            "required": ["server", "tool"],
        }
