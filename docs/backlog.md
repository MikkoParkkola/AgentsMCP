# Backlog (Decomposed to ≤500 LOC Tasks)

Purpose: break work into parallelizable units, each expected to be 500 lines of code (LOC) or less, with tight boundaries and clear acceptance criteria. Reference the high-level goals in `docs/work-plan.md`.

Legend: [Size] S ≤200 LOC, M ≤500 LOC, D = docs-only.

## Providers & Models

B1. Providers module skeleton [S]
- Scope: Add `src/agentsmcp/providers.py` with types (`ProviderType`, `ProviderConfig`, `Model`) and error classes; no HTTP calls yet.
- Files: new providers.py only.
- Acceptance: module imports; types available; no runtime behavior.

B2. OpenAI list_models adapter [M]
- Scope: Implement `openai_list_models(config)` + normalization; simple bearer auth; handle `api_base`.
- Files: providers.py only.
- Acceptance: returns non-empty list with valid key; errors map to ProviderAuth/Network/Protocol.

B3. OpenRouter list_models adapter [M]
- Scope: Implement `openrouter_list_models(config)`; bearer auth; `api_base` support.
- Files: providers.py only.
- Acceptance: returns list with valid key; proper error mapping.

B4. Ollama list_models adapter [S]
- Scope: Implement `ollama_list_models(config)` using `/api/tags`; no key for localhost.
- Files: providers.py only.
- Acceptance: returns list if daemon running; Network error otherwise.

B5. Facade `list_models(provider, config)` [S]
- Scope: Route to per-provider functions; unify errors; add minimal logging hooks.
- Files: providers.py only.
- Acceptance: switching provider yields expected calls; consistent exceptions.

B6. Agent hook `discover_models()` [S]
- Scope: In Agent base, add `discover_models(provider)` using `Config.providers` and facade.
- Files: `src/agentsmcp/agents/base.py` only.
- Acceptance: method returns models or structured error; no side effects.

## Chat CLI: Models & Provider UX

C1. Command plumbing: `/models` [S]
- Scope: Register command, parse arg `[provider?]`, call providers facade.
- Files: `src/agentsmcp/commands/chat.py` only.
- Acceptance: `/models` triggers fetch and shows raw list in console.

C2. Model list UI (search + select) [M]
- Scope: Add filterable list and selection callback; no persistence.
- Files: chat.py only.
- Acceptance: filter by substring; selecting sets current model in session.

C3. Provider autocomplete and `/provider` setter [S]
- Scope: Show configured providers; set session provider.
- Files: chat.py only.
- Acceptance: `/provider openai` switches provider; validation deferred.

C4. Apply selection to runtime [S]
- Scope: Ensure next message uses selected provider/model.
- Files: chat.py only.
- Acceptance: inspect outgoing request shows updated provider/model.

## API Keys: Validation & Persistence

K1. Validation helpers [S]
- Scope: Implement `validate_provider_config` probing endpoints; no prompts.
- Files: providers.py or `src/agentsmcp/providers_validate.py` (choose one file only).
- Acceptance: returns ValidationResult; never raises.

K2. Prompt + persist [S]
- Scope: Implement `prompt_for_api_key`, `persist_provider_api_key`.
- Files: chat.py (prompt); small helper in new `src/agentsmcp/config_write.py` for YAML merge.
- Acceptance: user can enter and persist key safely.

K3. Wire validation into `/provider` and `/models` [S]
- Scope: On demand, run K1; show actionable banner on missing/invalid; allow continue.
- Files: chat.py only.
- Acceptance: UX degrades gracefully without blocking.

## MCP Gateway: Version Negotiation

M1. `negotiate_version()` [S]
- Scope: Implement version selection with safe defaults.
- Files: `src/agentsmcp/mcp/server.py` only.
- Acceptance: logs negotiated version; function unit tested.

M2. `downconvert_tools()` [S]
- Scope: Strip unknown fields to legacy shape; pure function.
- Files: server.py only.
- Acceptance: unit tests demonstrate field filtering.

M3. Wire negotiation + downconversion [S]
- Scope: Apply to tool registration path for non-latest clients.
- Files: server.py only.
- Acceptance: manual test with mocked client version path passes.

## Context Window Management

X1. Token estimation + Trim function [S]
- Scope: Implement `estimate_tokens`, `trim_history` as pure helpers.
- Files: new `src/agentsmcp/context.py` only.
- Acceptance: deterministic trimming; unit tests on sample conversations.

X2. Integrate `/context` command [S]
- Scope: Add command to set percent/off and apply on send.
- Files: chat.py only.
- Acceptance: long threads get trimmed; setting applies immediately.

## Streaming

S1. Unified `generate_stream()` interface [S]
- Scope: Introduce provider-agnostic streaming function and `Chunk` type.
- Files: new `src/agentsmcp/stream.py` only.
- Acceptance: interface compiles; shim returns single final chunk for non-stream providers.

S2. OpenAI/OpenRouter streaming adapters [M]
- Scope: Add per-provider stream implementations behind the interface.
- Files: stream.py only.
- Acceptance: incremental chunks received; final finish_reason set.

S3. Chat UI rendering [S]
- Scope: Buffer and coalesce partials; `/stream on|off` command.
- Files: chat.py only.
- Acceptance: toggling works; partial tokens render.

## Packaging & E2E

P1. PyInstaller script [S]
- Scope: Add `scripts/build_binary.sh`; minimal options; prints output path.
- Files: new script only.
- Acceptance: binary builds locally in CI-like env.

P2. E2E smoke workflow [S]
- Scope: Add `.github/workflows/e2e.yml` + tiny Python smoke script.
- Files: new workflow + `scripts/e2e_smoke.py` only.
- Acceptance: lists tools; returns 0; on failure uploads logs.

## Delegation (Docs-first)

D1. Delegation spec docs [D]
- Scope: Fill `docs/delegation.md` with sequence diagrams and states.
- Files: docs only.
- Acceptance: reviewers can implement without ambiguity.

---

Guidelines:
- Keep each task touching ≤2 files where possible; avoid cross-cutting changes.
- Prefer pure functions and local wiring per task; integration tasks are separate items.
- If implementation exceeds 500 LOC, split by moving adapters/UI/integration to a new backlog item.

