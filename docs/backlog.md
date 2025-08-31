# Backlog (Decomposed to ≤500 LOC Tasks)

This is the canonical source of truth for upcoming work. Keep it current. If you find contradictions in other docs, update this file and add a note in the older document.

Purpose: break work into parallelizable units, each expected to be 500 lines of code (LOC) or less, with tight boundaries and clear acceptance criteria. Reference the high-level goals in `docs/work-plan.md`.

Legend: [Size] S ≤200 LOC, M ≤500 LOC, D = docs-only.

## Now / Next / Later
- Now (P0)
  - TUI v2 input reliability: single input path, echo/ICANON control, fallback reader; ensure chat input renders typed characters; exit via /quit and Ctrl+C. [S]
  - CLI standardization: single entry path, consistent command schema, Problem+Solution error format. [S]
  - Setup wizard: interactive first‑run for API keys/config + test connection. [M]
- Next (P1)
  - Providers/model discovery facade and `/models` UX (B1–B6, C1–C4). [S/M]
  - MCP version negotiation + downconvert (M1–M3). [S]
  - Context window management + `/context` (X1–X2). [S]
  - Streaming interface + adapters + `/stream` (S1–S3). [S/M]
- Later (P2)
  - SSE event bus + minimal web dashboard (WUI1–WUI3). [S/M]
  - Discovery announcer/client/handshake (AD1–AD4). [S/M]
  - Packaging + E2E smoke (P1–P2). [S]

## Providers & Models

Delight & Automation: Auto-detect available providers from env; on failure, show a one-line fix. Cache model lists with gentle refresh; highlight recommended defaults. Zero-config works for Ollama localhost.

B1. Providers module skeleton [S] [Done]
- Scope: Add `src/agentsmcp/providers.py` with types (`ProviderType`, `ProviderConfig`, `Model`) and error classes; no HTTP calls yet.
- Files: new providers.py only.
- Acceptance: module imports; types available; no runtime behavior.

B2. OpenAI list_models adapter [M] [Done]
- Scope: Implement `openai_list_models(config)` + normalization; simple bearer auth; handle `api_base`.
- Files: providers.py only.
- Acceptance: returns non-empty list with valid key; errors map to ProviderAuth/Network/Protocol.

B3. OpenRouter list_models adapter [M] [Done]
- Scope: Implement `openrouter_list_models(config)`; bearer auth; `api_base` support.
- Files: providers.py only.
- Acceptance: returns list with valid key; proper error mapping.

B4. Ollama list_models adapter [S] [Done]
- Scope: Implement `ollama_list_models(config)` using `/api/tags`; no key for localhost.
- Files: providers.py only.
- Acceptance: returns list if daemon running; Network error otherwise.

B5. Facade `list_models(provider, config)` [S] [Done]
- Scope: Route to per-provider functions; unify errors; add minimal logging hooks.
- Files: providers.py only.
- Acceptance: switching provider yields expected calls; consistent exceptions.

B6. Agent hook `discover_models()` [S] [Todo]
- Scope: In Agent base, add `discover_models(provider)` using `Config.providers` and facade.
- Files: `src/agentsmcp/agents/base.py` only.
- Acceptance: method returns models or structured error; no side effects.

## Chat CLI: Models & Provider UX

Delight & Automation: `/models` is fast, filter-as-you-type, and marks “best fit” models for your provider. Smart defaults; remembers last choice. Minimal prompts; clear, consistent feedback.

C1. Command plumbing: `/models` [S] [Done]
- Scope: Register command, parse arg `[provider?]`, call providers facade.
- Files: `src/agentsmcp/commands/chat.py` only.
- Acceptance: `/models` triggers fetch and shows raw list in console.

C2. Model list UI (search + select) [M] [Done]
- Scope: Add filterable list and selection callback; no persistence.
- Files: chat.py only.
- Acceptance: filter by substring; selecting sets current model in session.

C3. Provider autocomplete and `/provider` setter [S] [Done]
- Scope: Show configured providers; set session provider.
- Files: chat.py only.
- Acceptance: `/provider openai` switches provider; validation deferred.

C4. Apply selection to runtime [S] [Done]
- Scope: Ensure next message uses selected provider/model.
- Files: chat.py only.
- Acceptance: inspect outgoing request shows updated provider/model.

## API Keys: Validation & Persistence

Delight & Automation: Wizard-grade prompts; masked input; tests keys immediately with friendly remediation. Idempotent writes; no surprises.

K1. Validation helpers [S] [Done]
- Scope: Implement `validate_provider_config` probing endpoints; no prompts.
- Files: providers.py or `src/agentsmcp/providers_validate.py` (choose one file only).
- Acceptance: returns ValidationResult; never raises.

K2. Prompt + persist [S] [Done]
- Scope: Implement `prompt_for_api_key`, `persist_provider_api_key`.
- Files: chat.py (prompt); small helper in new `src/agentsmcp/config_write.py` for YAML merge.
- Acceptance: user can enter and persist key safely.

K3. Wire validation into `/provider` and `/models` [S] [Done]
- Scope: On demand, run K1; show actionable banner on missing/invalid; allow continue.
- Files: chat.py only.
- Acceptance: UX degrades gracefully without blocking.

## MCP Gateway: Version Negotiation

Delight & Automation: Logs a single concise negotiation line. Automatically down-converts schemas; warns only when truly incompatible.

M1. `negotiate_version()` [S] [Todo]
- Scope: Implement version selection with safe defaults.
- Files: `src/agentsmcp/mcp/server.py` only.
- Acceptance: logs negotiated version; function unit tested.

M2. `downconvert_tools()` [S] [Todo]
- Scope: Strip unknown fields to legacy shape; pure function.
- Files: server.py only.
- Acceptance: unit tests demonstrate field filtering.

M3. Wire negotiation + downconversion [S] [Todo]
- Scope: Apply to tool registration path for non-latest clients.
- Files: server.py only.
- Acceptance: manual test with mocked client version path passes.

## Context Window Management

Delight & Automation: Intelligent trimming with context awareness (keep recent conversation and system), optional pinning of key messages. Predictable and explained in UI.

X1. Token estimation + Trim function [S] [Done]
- Scope: Implement `estimate_tokens`, `trim_history` as pure helpers.
- Files: new `src/agentsmcp/context.py` only.
- Acceptance: deterministic trimming; unit tests on sample conversations.

X2. Integrate `/context` command [S] [Done]
- Scope: Add command to set percent/off and apply on send.
- Files: chat.py only.
- Acceptance: long threads get trimmed; setting applies immediately.

## Streaming

Delight & Automation: Smooth streaming by default when provider supports it; automatic fallback to non-stream; progress indicator in chat UI.

S1. Unified `generate_stream()` interface [S] [Done]
- Scope: Introduce provider-agnostic streaming function and `Chunk` type.
- Files: new `src/agentsmcp/stream.py` only.
- Acceptance: interface compiles; shim returns single final chunk for non-stream providers.

S2. OpenAI/OpenRouter streaming adapters [M] [Partial] (OpenAI native stub + CLI wiring; full provider wiring pending)
- Scope: Add per-provider stream implementations behind the interface.
- Files: stream.py only.
- Acceptance: incremental chunks received; final finish_reason set.

S3. Chat UI rendering [S] [Done] (CLI coalescing; TUI v2 wiring pending)
- Scope: Buffer and coalesce partials; `/stream on|off` command.
- Files: chat.py only.
- Acceptance: toggling works; partial tokens render.

## Packaging & E2E

Delight & Automation: One-command build scripts; prebuilt binaries in releases; smoke tests verify the most common paths and give clear guidance on failures.

P1. PyInstaller script [S]
- Scope: Add `scripts/build_binary.sh`; minimal options; prints output path.
- Files: new script only.
- Acceptance: binary builds locally in CI-like env.

P2. E2E smoke workflow [S]
- Scope: Add `.github/workflows/e2e.yml` + tiny Python smoke script.
- Files: new workflow + `scripts/e2e_smoke.py` only.
- Acceptance: lists tools; returns 0; on failure uploads logs.

## Delegation (Docs-first)

Delight & Automation: Clean command pattern to spawn short-lived workers. Guardrails built-in; confirmation flows are one-liners; audit logs human-readable.

D1. Delegation spec docs [D]
- Scope: Fill `docs/delegation.md` with sequence diagrams and states.
- Files: docs only.
- Acceptance: reviewers can implement without ambiguity.

---

Guidelines:
- Keep each task touching ≤2 files where possible; avoid cross-cutting changes.
- Prefer pure functions and local wiring per task; integration tasks are separate items.
- If implementation exceeds 500 LOC, split by moving adapters/UI/integration to a new backlog item.

## Agent Discovery & Coordination (New)

AD1. Discovery protocol spec [D]
- Scope: Author `docs/interfaces/agent-discovery.md` describing discovery/announce protocol, identifiers (agent id, name, capabilities, transport), and security model (allowlist, tokens).
- Files: docs only.
- Acceptance: clear, implementable spec with compatibility notes (Zeroconf/mDNS, broadcast, or registry fallback).

AD2. Announcer/registry (daemon) [M]
- Scope: Implement a lightweight announcer that advertises this agent’s presence and capabilities.
- Option A: Zeroconf/mDNS service record with TXT for capabilities.
- Option B: Fallback to a local registry file or Unix domain socket broker.
- Files: new `src/agentsmcp/discovery/announcer.py` (and optional `registry.py`).
- Acceptance: running agent appears in `agentsmcp discovery list` on the same host; respects enable/disable flag.

AD3. Discovery client [S]
- Scope: Implement `agentsmcp discovery list` to enumerate other agents with id/name/capabilities and endpoint.
- Files: `src/agentsmcp/commands/discovery.py` + `src/agentsmcp/discovery/client.py`.
- Acceptance: lists at least the local agent when announcer is enabled; handles unreachable entries gracefully.

AD4. Coordination handshake [M]
- Scope: Define and implement a minimal handshake to exchange capabilities and a control channel URL (MCP or REST).
- Files: `discovery/client.py`, `discovery/announcer.py`.
- Acceptance: two local agents can discover each other and exchange a test message (e.g., ping/capabilities).

AD5. Security & config [S]
- Scope: Config flags (enable/disable), allowlists, optional shared secret/token; docs in `docs/usage.md`.
- Files: `src/agentsmcp/config.py` (flags), `docs/usage.md`.
- Acceptance: discovery disabled by default; enabling requires explicit opt-in; allowlist enforced when set.

## Local Web Interface + SSE (New)

WUI1. SSE event bus [S]
- Scope: Add SSE endpoint (e.g., `/events`) that streams job lifecycle events, logs, and status updates.
- Files: `src/agentsmcp/server.py` (FastAPI route), small event publisher util.
- Acceptance: `curl` can subscribe and receive events when jobs start/finish; documented retry policy.

WUI2. REST control & status [S]
- Scope: Endpoints for current status, list jobs, spawn, cancel; basic JSON payloads.
- Files: `src/agentsmcp/server.py`.
- Acceptance: Can spawn and cancel via REST; status returns JSON reflecting AgentManager state.

WUI3. Web UI scaffold [M]
- Scope: Static assets (no build step preferred): HTML + minimal JS to render dashboard and subscribe to SSE.
- Files: `src/agentsmcp/web/static/` and template in `server.py`.
- Acceptance: Opening `/ui` shows status tiles and updates live via SSE.

WUI4. Metrics & charts [M]
- Scope: Add metrics endpoint or reuse existing; UI charts for jobs per hour, success/failure, latency.
- Files: `server.py` (metrics JSON), `web/static` (charts with small lib or vanilla JS).
- Acceptance: Charts render with real data; no blocking calls on server path.

WUI5. Controls in UI [S]
- Scope: Buttons to spawn predefined agent tasks and cancel running jobs; confirmation prompts.
- Files: `web/static` JS + REST calls.
- Acceptance: Actions reflect immediately in UI; errors surfaced as notifications.

WUI6. Docs & config [D]
- Scope: Document how to enable/disable UI, set bind host/port, and use SSE; troubleshooting.
- Files: `docs/usage.md`, `docs/deployment.md`.
- Acceptance: A fresh user can enable UI and see live dashboard in under 2 minutes.
