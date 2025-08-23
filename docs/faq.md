# FAQs

- Can I change models at runtime?
  - Yes. Use `--model` on `agent spawn` to override just for that run. You can also set `model_priority` in config to guide default pick.

- Does this support OpenRouter?
  - Yes. Set `provider: openrouter` and ensure `OPENROUTER_API_KEY` is set. You can also pass `--provider openrouter` and `--api-base https://openrouter.ai/api/v1` on the CLI.

- Do I need an MCP client installed?
  - Only if you want `mcp_call` to execute against external MCP servers. Without it, the tool returns a helpful message instead of failing.

- Can this act as an MCP server for other clients?
  - Yes. Run `agentsmcp mcp serve`. It advertises the latest supported protocol and negotiates a compatible version for older clients.

- Is Ollama supported?
  - Yes. Use `provider: ollama`. Many local setups require no API key.

- How do I enable/disable MCP servers?
  - `agentsmcp mcp enable <name>` or `agentsmcp mcp disable <name>`.
