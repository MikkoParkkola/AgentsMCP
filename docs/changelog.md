# Changelog

This file records notable changes to the project. Keep entries in reverse chronological order.

## [Unreleased]
- TUI v2: CRLF-based line resets; markdown with colors via ansi_markdown_processor in fixed TUI.
- Providers: `providers_enabled` + `AGENTS_PROVIDERS_ENABLED` to allowlist; default only `ollama-turbo`.
- Roles: Added human roles (BA, BE, Web FE, API, TUI FE, Backend/Web/TUI QA, Chief QA, IT lawyer, marketing manager, CI/CD, dev tooling, data analyst/scientist, ML scientist/engineer) and routing.
- Continuous Improvement: Retrospectives -> JSONL + improvements.md; auto-update versioned role/team docs under docs/roles/.
- TUI v2: Fixed progressive indentation by writing CRLF at line starts; moved connection status to its own line; integrated ANSI Markdown renderer with colors for assistant responses in fixed working TUI.
- Providers: Added provider allowlist via config `providers_enabled` and env `AGENTS_PROVIDERS_ENABLED`. Default enables only `ollama-turbo`.
- Roles: Added human-oriented roles (business analyst, backend engineer, web frontend engineer, API engineer, TUI frontend engineer; backend/web/TUI QA engineers; chief QA engineer) and hooked into role registry and routing.
- Continuous Improvement: Added post-task retrospective (individual + joint), persisted to `build/retrospectives/` with a rolling `improvements.md` fed into the system prompt.
- TUI: Fix chat rendering preserving line breaks and remove odd indentation in assistant/user messages by wrapping each physical line independently.
- TUI: Handle multi-line bracketed paste as a single input chunk (no auto-submit); pasted newlines are kept in the input buffer and rendered as multi-line until user presses Enter.
- P6.2 Design: Add user-friendly configuration proposal (role preferences, auto-detection, smart defaults, first-run wizard, task-to-team mapping) in docs/P6.2-config-design.md.
- Agents: add environment-tailored guide `AGENTS_LOCAL.md` and link from `AGENTS.md`.
- Repo-Mapper: scaffold `ownership_map.json` and `path_locks.json` with no-overlap and lock policies.
- Interfaces: add `interfaces/` with example ICD `auth.validate_token.json` and README.
- Golden tests: add `golden_tests/auth.validate_token.json` and a skipped pytest placeholder in `tests/golden/`.
- Automation: add `scripts/pre_pr_check.sh` to run ruff and pytest locally before opening PRs.
- UI/UX: Added comprehensive UI/UX review and adopted phased uplift plan (TUI via Textual shell; Web UI parity; readiness gating; SSE with backoff; standardized CLI outputs).
- Architecture: add architecture analysis synthesis and adopt prioritized roadmap (AgentManager singleton in API; job cleanup fix; EventBus consolidation; queue/worker pool + 429 overload; async providers with retry/backoff; tool sandboxing; JWT auth, rate limits; config consolidation).
- Provider config validation (K1): add `validate_provider_config` returning non-raising results.
- API key persistence (K2): new `/apikey [provider]` command with masked input; keys saved to YAML providers map.
- Validation wiring (K3): `/provider` and `/models` show friendly, non-blocking validation banners.
- Chat CLI model selection (C2): `/models` supports interactive filter-and-select to set the session model.
- Provider selection UX (C3): `/provider` without args shows an interactive list to pick a provider.
- Context management (X1–X2): add simple token estimation + trimming helper and `/context <percent|off>` in chat to include recent conversation.
- SSE events (WUI1): `/events` streams job lifecycle events.
- Web UI scaffold (WUI3): `/ui` serves a minimal dashboard with live events and spawn form.
- E2E smoke (P2): GitHub Action `.github/workflows/e2e.yml` with `scripts/e2e_smoke.py`.
- Delegation spec (D1): `docs/delegation.md` drafted.
- Agent discovery spec (AD1): `docs/interfaces/agent-discovery.md` authored.
- Discovery announcer + client (AD2–AD3): registry-based announcer, `agentsmcp discovery list` command.
- Handshake capabilities endpoint (AD4): `/capabilities` returns basic info.
- Discovery config flags (AD5): `discovery_enabled`, `discovery_allowlist`, `discovery_token` in config.
- Coordination endpoints (AD4): `/coord/ping` and `/coord/handshake` with allowlist/token checks; `agentsmcp discovery handshake` command.
- Web UI additions (WUI4–WUI5): `/metrics` endpoint and simple canvas chart; jobs list + cancel control.
- Web UI enable/disable (WUI6): new `ui_enabled` config flag to mount `/ui`.
- Streaming adapters (S2): added optional OpenAI native streaming function (not wired by default).
- Add provider model discovery facade and adapters (OpenAI, OpenRouter, Ollama); expose discover_models() on BaseAgent; add /models command in chat CLI to list models.
- Document roadmap and backlog and add default MCP client configuration.
- Run Danger CI non-interactively to avoid npx install prompts.
- Correct CI lint command and supply token for Danger to avoid spurious failures.
- Establish automated CI pipeline with linting, testing, security scans, and automerge.
- Reorganize documentation into generic, AI-specific, and product-specific sections with status and details files.
- Document AI-agent project best practices.
- Add BRoA-based guidelines for human and AI agent collaboration.
- Initial placeholder.
- Added Business Rules of Acquisition guidelines and linked them from AGENTS.md.
- Added engineering handbook outlining principles and practices for AI-augmented teams.
 - Governance: Set repository CODEOWNERS to @MikkoParkkola to enable review ownership and enforcement.

<!--
## [vX.Y.Z] - YYYY-MM-DD
### Added
- ...

### Changed
- ...

### Fixed
- ...
-->
## 2025-08-26

- Default interactive agent set to `ollama-turbo-coding` using `gpt-oss:120b`.
- CLI `--provider` now accepts `ollama-turbo`.
- README updated with macOS binary path and usage; documented defaults and MCP tools.
- Verified web UI serves from `/ui` and health endpoint responds.
## [Unreleased]

- Security: Introduced Human-In-The-Loop (HITL) Approval System.
  - Decorator `hitl_required()` for critical operations.
  - Priority approval queue with timeout + default action.
  - Web UI at `/hitl/` for approve/reject with RBAC.
  - Cryptographically signed, one-time approval tokens (JWT) with replay protection.
  - Audit trail persisted to `build/hitl_audit.log` (JSONL).
  - Rate limiting on decision endpoint and security headers.
  - Zero-impact fast path for non-critical ops.
## 2025-08-31 — Documentation consolidation

- Consolidated Markdown docs to one source of truth per topic.
- Canonical CLI doc is now `docs/cli-client.md` (removed `CLI_README.md`).
- Removed archived/superseded docs: `docs/ARCHITECTURE_ANALYSIS_2025.md`, `docs/ui-ux-improvement-plan.md`.
- Merged provider guidance (Claude, Gemini, Qwen) into `docs/models.md`; removed `CLAUDE.md`, `GEMINI.md`, `QWEN.md`.
- Merged ICD policy into `docs/interfaces/README.md`; removed duplicate `interfaces/README.md`.
- Moved TUI issue items into `docs/backlog.md` (top P0/P1); removed `docs/tui_issues.md`, `IMMEDIATE_TUI_FIX.md`, `CRITICAL_TUI_FIXES_SUMMARY.md`.
- Promoted viable ideation items to backlog (Later/P2) and removed the separate ideation docs.
