# Decision & Info Log

Archive of resolved questions and decisions moved from `open-issues.md`.

<!--
Template:
## [ID] - Title
- **Date Added:** YYYY-MM-DD
- **Version:** vX.Y.Z
- **Branch:** https://example.com/branch
- **Submitter:** name
- **Decision Maker:** name
- **Decision Date:** YYYY-MM-DD
- **Outcome:** what was decided
- **Rationale:** why
- **Details:** links to commits/PRs/docs
-->
## [0001] - Adopt BRoA Guidelines
- **Date Added:** 2025-08-21
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Decision Date:** 2025-08-21
- **Outcome:** Adopted BRoA-based guidelines for human and AI agent collaboration.
- **Rationale:** Provide shared principles for working with AI agents and mixed human–AI teams.
- **Details:** See [docs/broa-agent-guidelines.md](broa-agent-guidelines.md).

## [0002] - Adopt GitHub Actions CI with automerge
- **Outcome:** Implemented GitHub Actions workflows for CI, security scanning, and automerge with branch cleanup.
- **Rationale:** Provide baseline automation and safety nets before product code exists.
- **Details:** See `.github/workflows` directory.

## [0003] - Adopt AI-Augmented Engineering Handbook
- **Date Added:** 2025-08-21
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Decision Date:** 2025-08-21
- **Outcome:** Adopted comprehensive engineering handbook for humans and AI agents.
- **Rationale:** Provide evidence-based principles and practices to guide collaboration and delivery.
- **Details:** See [docs/engineering-handbook.md](engineering-handbook.md).


## [0004] - Adopt AI-Agent Project Best Practices
- **Date Added:** 2025-08-22
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Decision Date:** 2025-08-22
- **Outcome:** Updated lint job to use `ruff check` and passed `GITHUB_TOKEN` to Danger.
- **Rationale:** Previous configuration caused CI failures despite no code changes.
- **Details:** See `.github/workflows/ci.yml`.

## [0005] - Auto-install Danger via npx
- **Date Added:** 2025-08-22
- **Outcome:** Documented baseline principles for AI-agent team software projects.
- **Rationale:** Standardize quality gates and collaboration practices for multi-agent development.
- **Details:** See [docs/ai-agent-project-principles.md](ai-agent-project-principles.md).

## [0006] - Reorganize Documentation Structure
- **Date Added:** 2025-08-23
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Decision Date:** 2025-08-22
- **Outcome:** Added `--yes` flag to `npx danger@11 ci` to avoid interactive install prompts.
- **Rationale:** Prevent unknown errors caused by cancelled npx installations.
- **Details:** See `.github/workflows/ci.yml`.
- **Decision Date:** 2025-08-23
- **Outcome:** Split documentation into generic, AI-specific, and product-specific best practices plus status and details files.
- **Rationale:** Provide a clearer blueprint template for future projects.
- Note: Consolidated; refer to `docs/backlog.md` (improvements), `docs/changelog.md` (changes), and `docs/AGENTIC_ARCHITECTURE.md` (architecture).

## [0007] - Fix CI lint and Danger configuration
## 2025-08-26 — Default Model/Agent Selection

- Decision: Default to cloud `ollama-turbo` with `gpt-oss:120b` for interactive sessions.
- Rationale: Best implementable balance of capability, speed, and cost for everyday coding tasks per project guidelines.
- Changes: CLI defaults updated (`ollama-turbo-coding`), provider option extended, documentation updated.

## [0008] - Architecture Review Synthesis and Roadmap
- **Date Added:** 2025-08-26
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Decision Date:** 2025-08-26
- **Outcome:** Accepted the architecture synthesis focusing on: single AgentManager instance in API, job cleanup fix, EventBus consolidation, queue/worker pool with 429 overload, async provider calls with retry/backoff, tool sandboxing, JWT auth (prod), rate limits, and config consolidation via AppSettings.
- **Rationale:** Addresses correctness, robustness, and maintainability with minimal disruption and clear phased rollout.
- **Details:** See [docs/architecture-analysis-synthesis.md](architecture-analysis-synthesis.md).
## [0009] - UI/UX Overhaul Plan
- **Date Added:** 2025-08-26
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Decision Date:** 2025-08-26
- **Outcome:** Adopt two-phase UI/UX uplift: Phase A (stabilize + professional polish) and Phase B (delight + advanced), delivering feature parity across TUI and Web UI with professional tone and performance.
- **Rationale:** Improve usability, completeness, performance, and visual consistency across entry points; address breakages and missing parity.
- Note: Superseded by consolidated backlog items.

## [0002] - Define CODEOWNERS and maintainer list
- **Date Added:** 2025-08-21
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Decision Date:** 2025-08-27
- **Outcome:** Set repository CODEOWNERS to @MikkoParkkola as the current maintainer.
- **Rationale:** Replace placeholder @REPO_OWNER with an actual maintainer to enable reviews, ownership, and CODEOWNERS enforcement.
- **Details:** See update to `.github/CODEOWNERS` in this commit.
### HITL Approval System (2025-08-27)

- Decision: Implemented HITL with async in-memory queue and JWT-signed approval tokens. Default timeout denies actions. Web UI added under `/hitl/` with RBAC via env-based user lists. Minimal integration by decorating internal high-risk operations.
- Rationale: Immediate security hardening without heavy infra dependencies; preserves performance for non-critical operations.
- Alternatives: External queue (Redis), DB-backed audit, cookie sessions. Deferred to future phases to keep scope tight.
- Impact: Critical ops now blocked pending human approval; audit trail created; non-critical ops unchanged.
## 2025-08-31 — Docs consolidation decisions

Context: Reduce duplication and ensure a single source of truth for each documentation area.

Decisions:
- CLI canon: Keep `docs/cli-client.md`; delete `CLI_README.md`.
- Architecture canon: Keep `docs/AGENTIC_ARCHITECTURE.md`; delete `docs/ARCHITECTURE_ANALYSIS_2025.md` (already archived).
- Interfaces doc: Merge ICD policy into `docs/interfaces/README.md`; delete duplicate `interfaces/README.md`.
- Provider docs: Merge `CLAUDE.md`, `GEMINI.md`, `QWEN.md` guidance into `docs/models.md`; delete the separate files.
- TUI issues: Move valid items into `docs/backlog.md` (P0/P1) and remove `docs/tui_issues.md`, `IMMEDIATE_TUI_FIX.md`, `CRITICAL_TUI_FIXES_SUMMARY.md`.
- Ideation/strategy: Promote viable items into `docs/backlog.md` (Later/P2) and remove the separate ideation docs.

Rationale: Single source per topic improves maintainability and reduces drift. Operational logs record changes.
