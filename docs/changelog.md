# Changelog

This file records notable changes to the project. Keep entries in reverse chronological order.

## [Unreleased]
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
