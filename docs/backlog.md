# Backlog (Decomposed to ≤500 LOC Tasks)

This is the canonical source of truth for upcoming work. Keep it current. If you find contradictions in other docs, update this file and add a note in the older document.

Purpose: break work into parallelizable units, each expected to be 500 lines of code (LOC) or less, with tight boundaries and clear acceptance criteria.

Legend: [Size] S ≤200 LOC, M ≤500 LOC, D = docs-only.

## Now / Next / Later
- Now (P0)
  - TUI: terminal resize support — handle SIGWINCH and recompute layout; ensure immediate reflow without artifacts. [S]
  - TUI: guarantee terminal state restoration on crash/TERM — install signal handlers and always restore cooked mode. [S]
  - CLI standardization: single entry path, consistent command schema, Problem+Solution error format. [S]
- Next (P1)
  - TUI: Unicode grapheme-aware backspace/delete behavior. [S]
  - TUI: mouse click mapping to UI actions (selection, focus). [S]
  - MCP version negotiation + downconvert (M1–M3). [S]
  - Streaming adapters + `/stream` (S2) — complete provider wiring. [M]
- Later (P2)
  - TUI: theme consistency — remove raw ANSI in components and route via ThemeManager. [S]
  - TUI: Smart Suggestions panel (context-aware quick actions). [M]
  - TUI: Zen/Dashboard/Command Center layout modes (adaptive layouts). [M]
  - Intelligent model routing and cost optimization with OpenRouter (task-aware selection, telemetry). [M]
  - SSE event bus + minimal web dashboard (WUI1–WUI3). [S/M]
  - Discovery announcer/client/handshake (AD1–AD4). [S/M]
  - Packaging + E2E smoke (P1–P2). [S]

## Providers & Models

All core provider/model discovery work is complete in code (`src/agentsmcp/providers.py`, CLI commands in `src/agentsmcp/commands/chat.py`). Future improvements should be tracked as new backlog items if needed.

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

Validation helpers and prompting/persistence are implemented (`providers_validate.py`, CLI). Track any provider-specific additions as new items if needed.

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

Implemented (`src/agentsmcp/context.py`, CLI `/context`). Add improvements as new items if discovered.

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

## Delegation

Functional; document improvements no longer tracked outside this backlog.

---

Guidelines:
- Keep each task touching ≤2 files where possible; avoid cross-cutting changes.
- Prefer pure functions and local wiring per task; integration tasks are separate items.
- If implementation exceeds 500 LOC, split by moving adapters/UI/integration to a new backlog item.

## Agent Discovery & Coordination (New)

AD1. Discovery protocol spec — track as implementation tasks only (docs are derived from code as needed).

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
- Files: `docs/usage.md`.
- Acceptance: A fresh user can enable UI and see live dashboard in under 2 minutes.

---

### P0 Triage (Small, testable tasks)

P0-A. SIGWINCH resize support [S]
- Scope: Install `SIGWINCH` handler; recompute layout on resize.
- Files: `src/agentsmcp/ui/*`.
- Acceptance: Resizing terminal triggers reflow without visual artifacts.

P0-B. Terminal state restoration [S]
- Scope: Install TERM/INT signal handlers; always restore cooked mode.
- Files: `src/agentsmcp/ui/*` (terminal state manager).
- Acceptance: After forced TERM, terminal echo/line mode restored (manual reproduction).

P0-C. CLI standardization [S]
- Scope: Single entry path; consistent command schema; Problem+Solution error format.
- Files: `src/agentsmcp/cli.py`, `src/agentsmcp/commands/*`.
- Acceptance: Commands show consistent help and error messages.

Completed: Setup wizard implemented (`src/agentsmcp/commands/setup.py`).

### P1 Triage (Small, testable tasks)

P1-A. Unicode grapheme-aware backspace [S]
- Scope: Use grapheme clusters for delete/backspace.
- Files: `src/agentsmcp/ui/*` (line buffer).
- Acceptance: Backspace removes entire grapheme (emoji/accented) correctly.

P1-B. Mouse click mapping [S]
- Scope: Map clicks to selection/focus actions in scrollable lists.
- Files: `src/agentsmcp/ui/*`.
- Acceptance: Clicking on items focuses/selects accordingly.

P1-C. MCP version negotiation (M1–M3) [S]
- Scope: Implement `negotiate_version()`, `downconvert_tools()`, and wire on registration.
- Files: `src/agentsmcp/mcp/server.py`.
- Acceptance: Logs negotiated version; unit tests cover field filtering; manual mock path passes.

P1-D. Streaming adapters (S2) [M]
- Scope: Implement provider streaming wiring behind `stream.generate_stream()`.
- Files: `src/agentsmcp/stream.py`.
- Acceptance: Incremental chunks arrive; `finish_reason` set; toggling `/stream on|off` works.
