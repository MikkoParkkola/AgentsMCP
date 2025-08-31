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

## Provider Guidance

### Claude (large-context analysis and refactors)
- Prefer for very large-context analysis and documentation refactors.
- Use structured, explicit instructions; provide file paths and concrete goals.
- When editing code, request minimal diffs with rationale; validate with lint/tests when possible.
- Safeguards: redact secrets, prefer deterministic, idempotent scripts, and stage incremental migrations with rollback.

### Gemini (broad reasoning and content generation)
- Good for content generation, multi-step reasoning, and cross‑modal tasks.
- Structure complex tasks into numbered steps and checkpoints; include schemas/contracts when generating code.
- Quality gates: include quick validation steps and safe defaults with clear flags.

### Qwen (multilingual and cost-sensitive)
- Useful for multilingual work, summarization, and low‑cost runs.
- Keep prompts concise; specify output formats, language versions, and style guides.
- For translations: specify tone, audience, and domain terminology.
- Safeguards: avoid speculative fixes without tests; flag ambiguities and propose clarifying questions.
