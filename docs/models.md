# Models and Providers

AgentsMCP supports multiple model providers via the OpenAI Python client and OpenAI-compatible APIs.

## Providers
- `openai` (default): uses `OPENAI_API_KEY` and OpenAI endpoints.
- `openrouter`: uses `OPENROUTER_API_KEY` and `https://openrouter.ai/api/v1` (OpenAI-compatible).
- `ollama`: local; typically no key required. Use small context tasks for speed.
- `custom`: supply `api_base` and `api_key_env`.

## Selection strategy
- Explicit `model` wins.
- Otherwise, first `model_priority` entry is used.

You can override per run:
```bash
agentsmcp agent spawn codex "task" --model gpt-4o-mini
agentsmcp agent spawn codex "task" --provider openrouter --api-base https://openrouter.ai/api/v1 --model openrouter/model-name
```

## Guidance by task type (examples)
- Coding small/medium: `gpt-4o-mini`, `llama-3.1-8b-instruct` (OpenRouter)
- Large context reasoning: `gpt-4o`, `o4-mini`, `sonnet`-class
- Fast drafting: `gpt-4o-mini`, `qwen2.5:7b` (local)

Tune `model_priority` per agent to reflect desired defaults.
