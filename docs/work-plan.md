# Work Plan: Providers, CLI, MCP Gateway

Last updated: 2025-08-23

This document captures the prioritized, delegable plan to get the system ready fast without refactors. Each task includes specs and acceptance criteria for straightforward delegation and verification.

Note: The high-level tasks below are decomposed into sub-500 LOC backlog items in `docs/backlog.md` for parallel execution. See references like B1–B6, C1–C4, etc.

## 1) Provider Model Discovery (OpenAI, OpenRouter, Ollama)
Backlog: B1–B6

- Specs:
  - Add `src/agentsmcp/providers.py` with adapters:
    - `OpenAIProvider.list_models(client, api_base) -> [Model]` via `GET /v1/models`.
    - `OpenRouterProvider.list_models(client, api_base) -> [Model]` via `GET /models`.
    - `OllamaProvider.list_models(api_base) -> [Model]` via `GET /api/tags` (no API key for localhost).
  - `Model` dataclass: `id`, `provider`, `context_window?`, `display_name?`, `aliases?`.
  - Wire into `Agent` base: `discover_models(provider)` using `Config.providers[provider]`.
- Acceptance criteria:
  - OpenAI/OpenRouter with valid keys return non-empty lists including current model if available.
  - Local Ollama returns models or a clear “daemon not running” message.
  - Missing/invalid key and network errors produce actionable messages.
- Dependencies: None. Effort: S.

## 2) CLI `/models` Command in Chat
Backlog: C1–C4

- Specs:
  - In `src/agentsmcp/commands/chat.py`, add `/models [provider?]`:
    - Lists models for current or specified provider; searchable UI.
    - Selecting a model sets session model and optionally switches provider.
  - Add `/provider` auto-complete from configured providers.
- Acceptance criteria:
  - `/models` shows a filterable list and updates session model/provider immediately.
  - Invalid provider shows guidance + configured providers.
- Dependencies: Task 1. Effort: S.

## 3) Provider API Key Validation + UX
Backlog: K1–K3

- Specs:
  - Lazy validation on first use (`/models` or first chat run) with a lightweight probe per provider.
  - Friendly error prompts for missing/invalid keys with remediation and docs links.
  - Allow interactive key entry via `/provider` and optional persistence to `agentsmcp.yaml`.
- Acceptance criteria:
  - Clear guidance for missing/invalid keys; other providers remain usable.
  - Interactive key entry works and persists when user confirms.
- Dependencies: Task 2. Effort: S.

## 4) MCP Version Negotiation (Minimal Downgrade)
Backlog: M1–M3

- Specs:
  - In `src/agentsmcp/mcp/server.py`, add version negotiation:
    - Accept client `version`; advertise `NEGOTIATED_VERSION`.
    - `downconvert_tools(tools, client_version)` strips unknown fields for older clients.
  - Document minimum supported version with examples in `docs/mcp-gateway.md`.
- Acceptance criteria:
  - New clients get full schemas; older clients get stripped schemas without errors.
  - Gateway logs negotiation outcome once at startup.
- Dependencies: None. Effort: S.

## 5) Docs: Provider Setup, Models, Quickstart

- Specs:
  - Update `docs/models.md`: providers, keys, `/models`, provider switching.
  - Update `docs/usage.md`: `/model`, `/provider`, `/api_base`, `/mcp` commands.
  - Update `docs/troubleshooting.md`: missing keys, network, Ollama daemon.
  - README “MCP + Providers Quickstart” linking these docs.
- Acceptance criteria:
  - A new user can configure a provider, list models, and switch models in one pass.
- Dependencies: Tasks 1–3. Effort: XS.

## 6) Basic Context Window Management
Backlog: X1–X2

- Specs:
  - Add session parameters: `max_tokens_out`, `context_budget` (% of model context).
  - On send, trim history oldest-first to fit `context_budget`.
  - Add `/context [percent|off]` to control and persist via session save.
- Acceptance criteria:
  - Large histories do not overflow; trimming occurs automatically and predictably.
- Dependencies: None. Effort: S.

## 7) Single-File Binary Packaging (PyInstaller)

- Specs:
  - Add `scripts/build_binary.sh` using PyInstaller:
    - Entry `src/agentsmcp/cli.py` → `agentsmcp` (`--onefile`).
    - Version injected from package metadata.
  - Document steps in `docs/standalone-binary.md` for macOS/Linux.
- Acceptance criteria:
  - Built binary runs chat and MCP server on a clean environment.
- Dependencies: None. Effort: S.

## 8) Backlog, Roadmap, and Logs

- Specs:
  - Ensure concise docs exist and are current:
    - `docs/backlog.md` (prioritized items), `docs/roadmap.md` (near/next/later),
      `docs/decision-log.md` (key choices), `docs/changelog.md` (current release).
- Acceptance criteria:
  - Documents reflect this plan and current status.
- Dependencies: None. Effort: XS.

## 9) Streaming Responses (Optional)
Backlog: S1–S3

- Specs:
  - If provider supports streaming, add `--stream` and `/stream on|off`.
  - Render partial tokens in chat UI, coalesce on completion.
- Acceptance criteria:
  - With streaming on, responses render incrementally; off returns full responses.
- Dependencies: None. Effort: S–M. Priority: optional.

## 10) Delegation Workflows (Spec-Only)

- Specs:
  - Author `docs/delegation.md` and `src/agentsmcp/tools/delegate_spec.md`:
    - Command pattern for spawning short-lived workers and MCP-based delegation via `mcp_call`.
    - Guardrails: allowlists, confirmations, audit logging.
  - Include sequence diagrams and state transitions.
- Acceptance criteria:
  - Clear, implementable spec suitable for delegation.
- Dependencies: None. Effort: XS.

---

Execution preference: Start with Tasks 1–2 for immediate UX value, then Task 5 (docs) and Task 3 (key UX), followed by Task 6 (context), Task 7 (binary), and the remainder.
