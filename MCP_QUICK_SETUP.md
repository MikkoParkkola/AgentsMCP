# 🔧 MCP Servers Quick Setup

## One-Command Setup

```bash
# Run automated MCP server setup
./setup_mcp_servers.sh
```

## Manual Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "git": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-git", "--repository", "."]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-sequential-thinking"]
    },
    "memory": {
      "command": "npx", 
      "args": ["@modelcontextprotocol/server-memory"]
    }
  }
}
```

## Verify Setup

```bash
python test_agent_role_system.py
```

**📖 Full Guide:** See [MCP_SERVERS_SETUP.md](MCP_SERVERS_SETUP.md) for complete installation instructions.

---

**✅ Result:** All 22 specialist agents will have access to their optimized MCP tool sets for maximum effectiveness.