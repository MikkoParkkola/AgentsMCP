# MCP Gateway & Version Compatibility

AgentsMCP can run as an MCP server to expose its tools (filesystem, web, analysis, and `mcp_call`) to external MCP clients. It supports the latest protocol and negotiates down to older versions when necessary.

## Start gateway (stdio)
```bash
agentsmcp mcp serve --stdio
```

## Version negotiation
- The server advertises a list of supported protocol versions.
- During handshake, it chooses the highest version also supported by the client.
- If a downgrade is required, the server down-converts tool schemas conservatively (keeps core JSONSchema fields) to remain compatible with older clients.

## Acting as a bridge
Use `mcp_call` to bridge external MCP servers with newer tools while serving older clients:
- Client (older MCP) -> AgentsMCP gateway -> `mcp_call` -> External MCP server (latest)

This lets older clients consume newer tools via the gateway.

## Notes
- Actual protocol coverage depends on the Python MCP server SDK available in your environment.
- If none is installed, the CLI will guide you to install one; the rest of the system still works without it.
