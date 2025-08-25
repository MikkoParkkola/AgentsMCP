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
- `/models [provider]` – list models for current or given provider
- After listing, you can type text to filter or a number to select and set the current model.
- `/model <name>` – set model
- `/provider <name>` – set provider
- If run without arguments, `/provider` shows an interactive list to choose from configured and known providers.
- `/api_base <url>` – set base URL
- `/apikey [provider]` – enter and persist an API key (masked)
- `/context <percent|off>` – include recent chat context in prompts (simple trimming)
- `/stream on|off` – toggle incremental output rendering. For native OpenAI streaming, set `AGENTSMCP_NATIVE_STREAM=1` and ensure `OPENAI_API_KEY` is configured; otherwise the CLI chunks the final output as a fallback.
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
- Provider validation runs opportunistically when you use `/provider` or `/models`. If validation fails (e.g., missing API key), you’ll see a one-line banner with a suggested fix, like `run /apikey openai`.
