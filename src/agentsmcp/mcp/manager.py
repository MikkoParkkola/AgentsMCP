from __future__ import annotations

"""Lightweight MCP manager with optional runtime integration.

This module exposes a minimal interface to:
- register configured MCP servers (from Config)
- toggle servers on/off
- perform a generic `call_tool(server, tool, params)` operation

It does not require the MCP client library to be installed. If unavailable,
`call_tool` returns an informative message so the rest of the system keeps working.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

try:
    # Optional dependency; not required for basic CLI wiring
    # Reference: https://github.com/modelcontextprotocol/python-sdk (if available)
    from mcp.client import Client  # type: ignore
    MCP_AVAILABLE = True
except Exception:  # pragma: no cover
    Client = object  # type: ignore
    MCP_AVAILABLE = False


@dataclass
class MCPServer:
    name: str
    command: Optional[List[str]] = None
    transport: Optional[str] = None  # e.g., "stdio", "sse", "websocket"
    url: Optional[str] = None        # for sse/websocket
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    enabled: bool = True


class MCPManager:
    """Manages configured MCP servers and provides a generic call interface."""

    def __init__(self, servers: List[MCPServer]):
        self.log = logging.getLogger(self.__class__.__name__)
        self._servers: Dict[str, MCPServer] = {s.name: s for s in servers}
        self._clients: Dict[str, Any] = {}

    def list_servers(self) -> List[MCPServer]:
        return list(self._servers.values())

    def get_server(self, name: str) -> Optional[MCPServer]:
        return self._servers.get(name)

    def enable(self, name: str) -> bool:
        s = self._servers.get(name)
        if not s:
            return False
        s.enabled = True
        return True

    def disable(self, name: str) -> bool:
        s = self._servers.get(name)
        if not s:
            return False
        s.enabled = False
        return True

    async def call_tool(self, server: str, tool: str, params: Dict[str, Any]) -> str:
        """Call a tool on a configured MCP server.

        If the MCP Python client is not available, return an informative message.
        """
        srv = self._servers.get(server)
        if not srv:
            return f"Error: MCP server not found: {server}"
        if not srv.enabled:
            return f"Error: MCP server '{server}' is disabled"

        if not MCP_AVAILABLE:
            return (
                "MCP client not installed; install a Python MCP client to enable runtime calls.\n"
                f"Requested call -> server: {server}, tool: {tool}, params: {params}"
            )

        # Example flow; actual connection details depend on the concrete MCP client API.
        # This is intentionally defensive and minimal to avoid hard binding.
        try:  # pragma: no cover
            client = await self._get_client(srv)
            # Hypothetical: client.call_tool(tool, params)
            if hasattr(client, "call_tool"):
                result = await client.call_tool(tool, params)  # type: ignore[attr-defined]
                return str(result)
            return (
                "MCP client connected, but generic call_tool() is not available.\n"
                f"Attempted -> server: {server}, tool: {tool}, params: {params}"
            )
        except Exception as e:  # pragma: no cover
            self.log.exception("MCP call failed")
            return f"Error calling MCP tool: {e}"

    async def _get_client(self, srv: MCPServer) -> Any:  # pragma: no cover
        if srv.name in self._clients:
            return self._clients[srv.name]

        # Very rough sketch; real clients differ by transport
        if srv.transport in {None, "stdio"} and srv.command:
            client = await Client.create_with_stdio(srv.command, env=srv.env, cwd=srv.cwd)  # type: ignore[attr-defined]
        elif srv.transport == "sse" and srv.url:
            client = await Client.create_with_sse(srv.url, env=srv.env)  # type: ignore[attr-defined]
        elif srv.transport == "websocket" and srv.url:
            client = await Client.create_with_websocket(srv.url, env=srv.env)  # type: ignore[attr-defined]
        else:
            raise ValueError(
                f"Unsupported MCP transport for server '{srv.name}': transport={srv.transport}, command={srv.command}, url={srv.url}"
            )
        self._clients[srv.name] = client
        return client
