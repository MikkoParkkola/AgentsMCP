# CI Workflows: Interfaces, Outputs, and Acceptance Criteria

This directory documents how our GitHub Actions workflows should operate, what they produce, and the conditions under which they pass or fail. It enables contributors to reason about CI behavior and add jobs without surprises.

Workflows documented here:
- lint.md — Ruff linting for Python (and optional JS/Docs lint hooks)
- tests.md — Pytest with coverage across a Python matrix
- security.md — Bandit, pip-audit, CodeQL, Semgrep, Gitleaks, SBOM
- container.md — CI docker build sanity
- release.md — Tagged releases to GitHub Releases and PyPI
- ai-review.md — Lightweight AI review comment on PRs (opt-in)
- automerge.md — Label-driven automerge and branch cleanup
- e2e.md — (Planned) e2e smoke test for MCP gateway

