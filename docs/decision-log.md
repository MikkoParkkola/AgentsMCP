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
## [0004] - Fix CI lint and Danger configuration
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
- **Version:** Unreleased
- **Branch:** main
- **Submitter:** automated agent
- **Decision Maker:** project maintainers
- **Decision Date:** 2025-08-22
- **Outcome:** Added `--yes` flag to `npx danger@11 ci` to avoid interactive install prompts.
- **Rationale:** Prevent unknown errors caused by cancelled npx installations.
- **Details:** See `.github/workflows/ci.yml`.
