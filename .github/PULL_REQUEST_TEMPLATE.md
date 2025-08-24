## What
<short description of the change>

## Why
<business or technical reason for the change>

## How
- Architecture & design decisions (ADR ref if any)
- API/schema changes
- Feature flags & rollout plan

## Tests
- Unit:
- Integration:
- E2E/Contract:
- Coverage: __%

## Security & Perf
- SAST/SCA/Secrets/IaC/Container: OK
- Performance budgets: OK

## Risks & Rollback
- Risks identified:
- Rollback plan:

## Docs
- AGENTS.md/ADR/diagrams updated

## Checklist
- [ ] Conventional Commit used
- [ ] Changelog auto-updates correctly
- [ ] Preview environment link(s) included
 - [ ] CI acceptance criteria met (see docs/ci/*):
   - Lint passes (ruff)
   - Tests pass (matrix) and coverage artifact uploaded
   - Security scans pass (bandit, pip-audit, semgrep, gitleaks, sbom)
 - [ ] Interfaces adhered to (see docs/interfaces/*); deviations documented
 - [ ] Backlog item(s) reference (see docs/backlog.md) and scope ≤500 LOC per task
