# Configuration

AgentsMCP reads configuration from `agentsmcp.yaml` (if present) or environment variables. The CLI supports `--config path.yaml` on most commands.

## Minimal example

```yaml
server:
  host: localhost
  port: 8000

agents:
  codex:
    type: codex
    provider: openai
    model: gpt-4o
    tools: [filesystem, bash]
```

## Providers and models

Agent fields:
- `provider`: `openai` | `openrouter` | `ollama` | `custom`
- `api_base`: override API base (OpenRouter: `https://openrouter.ai/api/v1`)
- `api_key_env`: env var name for CUSTOM provider
- `model`: exact model
- `model_priority`: list of models to try (first is preferred when `model` not set)

Environment:
- `OPENAI_API_KEY`
- `OPENROUTER_API_KEY`
- `AGENTSMCP_<AGENT_TYPE>_API_KEY` overrides per agent (e.g., `AGENTSMCP_CODEX_API_KEY`)

## MCP servers

```yaml
mcp:
  - name: git-mcp
    enabled: true
    transport: stdio
    command: ["npx", "-y", "@modelcontextprotocol/server-git"]

agents:
  codex:
    mcp: [git-mcp]
```

Use CLI to manage:
- `agentsmcp mcp list`
- `agentsmcp mcp enable <name>`
- `agentsmcp mcp disable <name>`
- `agentsmcp mcp add <name> --transport stdio --command npx --command -y --command @modelcontextprotocol/server-git`
- `agentsmcp mcp remove <name>`

