# Standalone CLI Client

This project includes an interactive command-line AI client with a friendly UI.

## Start the client
```bash
agentsmcp chat --agent codex
```

Options:
- `--agent`: agent type (`codex`, `claude`, `ollama`)
- `--model`: session model override
- `--provider`: `openai|openrouter|ollama|custom`
- `--api-base`: override API base URL
- `--system`: session system prompt
- `--save-on-exit`: write settings back to `agentsmcp.yaml` on exit
- `--config`: specify a config file path

In-chat commands:
- `/help` – show commands
- `/model <name>` – set model
- `/provider <name>` – set provider
- `/api_base <url>` – set base URL
- `/system` – edit system prompt
- `/temp <0..2>` – set temperature
- `/new` – start new session (clears chat history)
- `/mcp` – list/enable/disable MCP servers interactively
- `/save` – save settings immediately
- `/quit` – exit

## Save settings back to file
- Start with `--save-on-exit` to auto-save
- Or run `/save` inside the session

## Running tasks
Each prompt executes as a short job under the current agent’s settings. This keeps things robust and simple.

## Tips
- Use `--provider openrouter --api-base https://openrouter.ai/api/v1` with `OPENROUTER_API_KEY` to switch to OpenRouter quickly.
- Configure MCP servers in `agentsmcp.yaml` and toggle them via `/mcp`.
