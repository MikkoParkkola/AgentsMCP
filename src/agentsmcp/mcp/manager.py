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
from typing import Any, Dict, List, Optional, Tuple
import logging
import asyncio
import time

from pydantic import BaseModel, Field, ValidationError


@dataclass
class MCPServer:
    name: str
    command: Optional[List[str]] = None
    transport: Optional[str] = None  # e.g., "stdio", "sse", "websocket"
    url: Optional[str] = None        # for sse/websocket
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    enabled: bool = True


class CallToolRequest(BaseModel):
    server: str = Field(..., description="MCP server name")
    tool: str = Field(..., description="Tool to invoke")
    params: Dict[str, Any] = Field(default_factory=dict)


class MCPManager:
    """Async-only manager for MCP servers with TTL cache and simple backoff."""

    _DEFAULT_TTL = 300  # seconds
    _MAX_RETRIES = 3
    _BASE_DELAY = 0.5

    def __init__(
        self,
        servers: List[MCPServer],
        ttl_seconds: int = _DEFAULT_TTL,
        *,
        allow_stdio: bool = True,
        allow_ws: bool = False,
        allow_sse: bool = False,
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self._servers: Dict[str, MCPServer] = {s.name: s for s in servers}
        # name -> (client, created_at)
        self._clients: Dict[str, Tuple[Any, float]] = {}
        self._last_error: Dict[str, str] = {}
        self._ttl = int(ttl_seconds)
        self._allow_stdio = allow_stdio
        self._allow_ws = allow_ws
        self._allow_sse = allow_sse
        # stats
        self._stats: Dict[str, Dict[str, Any]] = {
            s.name: {
                "connect_attempts": 0,
                "connect_failures": 0,
                "last_connect_ts": None,
                "created_at": None,
                "expires_at": None,
            }
            for s in servers
        }

    async def list_servers(self) -> List[MCPServer]:
        return list(self._servers.values())

    async def get_server(self, name: str) -> Optional[MCPServer]:
        return self._servers.get(name)

    async def enable(self, name: str) -> bool:
        s = self._servers.get(name)
        if not s:
            return False
        s.enabled = True
        return True

    async def disable(self, name: str) -> bool:
        s = self._servers.get(name)
        if not s:
            return False
        s.enabled = False
        await self.close(name)
        return True

    async def call_tool(self, server: str, tool: str, params: Dict[str, Any]) -> str:
        """Validate, obtain a client (TTL/backoff) and call the remote tool."""
        try:
            req = CallToolRequest(server=server, tool=tool, params=params)
        except ValidationError as ve:
            return f"Invalid request: {ve}"

        srv = self._servers.get(req.server)
        if not srv:
            return f"Error: MCP server not found: {req.server}"
        if not srv.enabled:
            return f"Error: MCP server '{req.server}' is disabled"

        try:
            client = await self._get_client(srv)
        except Exception as e:  # pragma: no cover
            self._last_error[req.server] = str(e)
            self.log.exception("Failed to obtain MCP client")
            return f"Error connecting to server '{req.server}': {e}"

        try:
            if hasattr(client, "call_tool"):
                result = await client.call_tool(req.tool, req.params)  # type: ignore[attr-defined]
                return str(result)
            return (
                "MCP client connected, but generic call_tool() is not available.\n"
                f"Attempted -> server: {req.server}, tool: {req.tool}, params: {req.params}"
            )
        except Exception as e:  # pragma: no cover
            self._last_error[req.server] = str(e)
            self.log.exception("MCP call failed")
            return f"Error calling MCP tool: {e}"

    async def _get_client(self, srv: MCPServer) -> Any:  # pragma: no cover
        entry = self._clients.get(srv.name)
        if entry:
            client, created = entry
            if (time.time() - created) < self._ttl:
                return client
            # expired – close and drop
            await self._close_client(srv.name, client)

        client = await self._create_client_with_backoff(srv)
        self._clients[srv.name] = (client, time.time())
        return client

    async def _create_client_with_backoff(self, srv: MCPServer) -> Any:
        delay = self._BASE_DELAY
        for attempt in range(1, self._MAX_RETRIES + 1):
            self._stats[srv.name]["connect_attempts"] += 1
            try:
                client = await self._create_client(srv)
                now = time.time()
                self._stats[srv.name]["last_connect_ts"] = now
                self._stats[srv.name]["created_at"] = now
                self._stats[srv.name]["expires_at"] = now + self._ttl if self._ttl else None
                return client
            except Exception as e:  # pragma: no cover
                self._stats[srv.name]["connect_failures"] += 1
                self._last_error[srv.name] = str(e)
                if attempt >= self._MAX_RETRIES:
                    raise
                self.log.warning(
                    "Client creation failed for %s (attempt %s/%s): %s; retrying in %.2fs",
                    srv.name,
                    attempt,
                    self._MAX_RETRIES,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)
                delay *= 2

    async def _create_client(self, srv: MCPServer) -> Any:
        # Lazy import only when needed
        from mcp.client import Client  # type: ignore

        if srv.transport in {None, "stdio"} and srv.command:
            if not self._allow_stdio:
                raise ValueError("transport disabled: stdio")
            return await Client.create_with_stdio(srv.command, env=srv.env, cwd=srv.cwd)  # type: ignore[attr-defined]
        if srv.transport == "sse" and srv.url:
            if not self._allow_sse:
                raise ValueError("transport disabled: sse")
            return await Client.create_with_sse(srv.url, env=srv.env)  # type: ignore[attr-defined]
        if srv.transport == "websocket" and srv.url:
            if not self._allow_ws:
                raise ValueError("transport disabled: websocket")
            return await Client.create_with_websocket(srv.url, env=srv.env)  # type: ignore[attr-defined]
        raise ValueError(
            f"Unsupported MCP transport for server '{srv.name}': transport={srv.transport}, command={srv.command}, url={srv.url}"
        )

    async def close(self, name: str) -> None:
        entry = self._clients.pop(name, None)
        if entry:
            client, _ = entry
            await self._close_client(name, client)

    async def close_all(self) -> None:
        for name in list(self._clients.keys()):
            await self.close(name)

    async def _close_client(self, name: str, client: Any) -> None:
        try:
            close_coro = getattr(client, "close", None)
            if close_coro:
                await close_coro()
        except Exception:  # pragma: no cover
            self.log.exception("Error closing client for %s", name)

    async def get_status(self) -> Dict[str, Dict[str, Any]]:
        now = time.time()
        status: Dict[str, Dict[str, Any]] = {}
        for name, srv in self._servers.items():
            entry = self._clients.get(name)
            connected = bool(entry and (now - entry[1] < self._ttl))
            stat = self._stats.get(name, {})
            info: Dict[str, Any] = {
                "enabled": srv.enabled,
                "connected": connected,
                "created_at": stat.get("created_at"),
                "expires_at": stat.get("expires_at"),
                "connect_attempts": stat.get("connect_attempts", 0),
                "connect_failures": stat.get("connect_failures", 0),
                "last_connect_ts": stat.get("last_connect_ts"),
            }
            if name in self._last_error:
                info["last_error"] = self._last_error[name]
            status[name] = info
        return status

    def get_config(self) -> Dict[str, Any]:
        """Return manager configuration values for diagnostics."""
        return {
            "ttl_seconds": self._ttl,
            "max_retries": self._MAX_RETRIES,
            "base_delay": self._BASE_DELAY,
        }


# Module-level singleton for reuse across app components
_GLOBAL_MGR: Optional[MCPManager] = None
_GLOBAL_FLAGS: Tuple[bool, bool, bool] = (True, False, False)


def get_global_manager(
    servers: List[MCPServer],
    *,
    allow_stdio: bool = True,
    allow_ws: bool = False,
    allow_sse: bool = False,
) -> MCPManager:
    global _GLOBAL_MGR, _GLOBAL_FLAGS
    new_flags = (allow_stdio, allow_ws, allow_sse)
    if _GLOBAL_MGR is None or new_flags != _GLOBAL_FLAGS:
        _GLOBAL_MGR = MCPManager(
            servers,
            allow_stdio=allow_stdio,
            allow_ws=allow_ws,
            allow_sse=allow_sse,
        )
        _GLOBAL_FLAGS = new_flags
    return _GLOBAL_MGR
