# Usage

## CLI

List agents
```bash
agentsmcp agent list
```

Spawn a job (with overrides)
```bash
agentsmcp agent spawn codex "Summarize docs" --timeout 300 --model gpt-4o-mini
agentsmcp agent spawn codex "Implement LRU cache" --provider openrouter --api-base https://openrouter.ai/api/v1 --model meta-llama/llama-3.1-8b-instruct
```

Check/cancel
```bash
agentsmcp agent status <job-id>
agentsmcp agent cancel <job-id>
```

Run API server
```bash
agentsmcp server start --host 0.0.0.0 --port 8000
```

MCP gateway/server
```bash
# Serve tools over MCP (stdio)
agentsmcp mcp serve --stdio
```

## Agents at a glance
- `codex`: coding-focused assistant
- `claude`: deep reasoning assistant (can still route via OpenAI-compatible APIs)
- `ollama`: local-friendly assistant

## MCP tool
Agents include `mcp_call` when MCP servers are configured. Parameters:
- `server`: MCP server name from config
- `tool`: tool name at that server
- `params`: JSON object of tool params

If no MCP client is installed, `mcp_call` returns a helpful message and continues.
