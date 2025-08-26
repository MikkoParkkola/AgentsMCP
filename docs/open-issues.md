# Open Issues Log

Track significant open questions, decision needs, and risks. Move entries to `decision-log.md` once resolved.

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
## [0002] - Define CODEOWNERS and maintainer list
- **Date Added:** 2025-08-21
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Needed By:** 2025-09-01
- **Status:** open
- **Problem:** Placeholder CODEOWNERS uses @REPO_OWNER; real maintainers need to be specified.
- **Options:**
  - Provide GitHub usernames of maintainers.
  - Use a GitHub team handle once defined.

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
