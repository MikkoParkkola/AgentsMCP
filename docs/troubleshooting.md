# Troubleshooting

## Common issues

- Missing API key
  - Error: `No API key found for provider=...`
  - Fix: set `OPENAI_API_KEY` (OpenAI) or `OPENROUTER_API_KEY` (OpenRouter) or use `api_key_env`.

- MCP client not installed
  - Symptom: `mcp_call` returns guidance instead of calling remote servers.
  - Fix: install a Python MCP client SDK (`pip install modelcontextprotocol`) or run without MCP.

- MCP server command fails
  - Check the command in `agentsmcp.yaml` under `mcp:`; verify `npx` and server package availability.

- CLI cannot find config
  - Pass `--config path/to/agentsmcp.yaml` or set `AGENTSMCP_CONFIG`.

- OpenRouter rate limits
  - Ensure you have a valid key and retry later; reduce concurrency.

## Debug logs
Set log format/level:
```bash
agentsmcp --log-level DEBUG --log-format text agent list
```

## Network settings
- For custom endpoints, set `api_base` in agent config.
- For proxies, export `HTTP_PROXY`/`HTTPS_PROXY`.
