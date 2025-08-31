# Open Issues Log

Track significant open questions, decision needs, and risks. Move entries to `decision-log.md` once resolved.

Note: For actionable work items, the single source of truth is `docs/backlog.md`. Use this file to capture decision needs and risks; link the relevant backlog tasks.

<!--
Template:
## [ID] - Title
- **Date Added:** YYYY-MM-DD
- **Version:** vX.Y.Z
- **Branch:** https://example.com/branch
- **Submitter:** name
- **Decision Maker:** name
- **Needed By:** YYYY-MM-DD
- **Status:** open
- **Problem:** description
- **Options:**
  - Option A - pros / cons
  - Option B - pros / cons
-->

## [0009] - Implement UI/UX Overhaul Phase A
- **Date Added:** 2025-08-26
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Needed By:** 2025-09-12
- **Status:** open
- **Problem:** Current interactive CLI and Web UI lack professional polish, feature parity, and robustness. Web UI is broken due to JS error; no readiness gating; limited error surfacing.
- **Options:**
  - Implement Phase A plan: Textual-based TUI layout (Sidebar/Topbar/Pages), Web UI readiness gate + error banners + SSE with backoff, and parity for Jobs/Agents/Models/Providers/MCP/Discovery/Costs/Settings.
  - Defer full TUI refactor; patch Web UI only (faster, less cohesive experience).
- **Tasks:**
  - Fix Web UI JS syntax error and add `/health/ready` gating.
  - Standardize CLI outputs and add `--json` + topical help.
  - Build TUI scaffold with pages and async spinners/error banners.
  - Implement SSE with retry/backoff and per-job filters (TUI/Web).
  - Implement Providers/Models management UIs with validation feedback.
  - Add global error banner and toasts; quiet professional tone by default.
  - Ensure feature parity list is satisfied (see docs/ui-ux-review.md).

## [0008] - Implement Architecture Synthesis Phase 1
- **Date Added:** 2025-08-26
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Needed By:** 2025-09-05
- **Status:** open
- **Problem:** Core correctness and lifecycle issues (API uses multiple AgentManager instances; cleanup bug; duplicate EventBus) and lack of base metrics.
- **Options:**
  - Implement Phase 1 as proposed: single AgentManager in API, fix cleanup time, consolidate EventBus, add metrics.
  - Defer EventBus consolidation and only patch API and cleanup bug (faster, less robust).

## [0010] - P6.2 Configuration Integration (Implementation)
- **Date Added:** 2025-08-27
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Needed By:** 2025-09-06
- **Status:** open
- **Problem:** Current config lacks user-friendly role preferences, first-run wizard, environment auto-detection, and automatic task-to-team composition.
- **Solution Proposal:** See `docs/P6.2-config-design.md`.
- **Tasks:**
  - Add Pydantic models: RolePreference, TeamStage, RoutingPolicy, EnvironmentCapabilities, UXQuickPrefs.
  - Implement detection module and `agentsmcp doctor`.
  - Implement `agentsmcp setup` wizard.
  - Integrate `role_preferences` and `routing.strategy` into DelegationEngine.
  - Add task classifier + team templates in coordinator.
  - Docs and tests per design.
