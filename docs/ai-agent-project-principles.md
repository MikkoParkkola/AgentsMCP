# Principles for AI-Agent Team Software Projects

## 1. Project & Environment Foundations
- Maintain clear product scope and avoid fake or unimplemented features in UI or documentation
- Document development flows (e.g., Docker setup) so any agent can bootstrap quickly
- Use environment files with runtime checks that fail fast when required variables are missing

## 2. Coding Conventions
- Standardize tech stack, component style, naming, and module organization across the codebase
- Commit lockfiles and use deterministic installs (npm ci) for reproducible builds

## 3. Testing & Quality Gates
- Follow TDD; every feature requires unit, integration, and E2E coverage with ≥80% thresholds (target 95%)
- Lint, type-check, build, audit dependencies, and run accessibility scans before committing
- Danger rules fail PRs without matching tests or changelog updates and warn on PRs >400 lines
- Use commit hooks and lint-staged to auto-format, lint, and run related tests on changed files

## 4. Continuous Integration & Deployment
- Cache dependencies, shard tests, and parallelize CI jobs for speed while collecting coverage artifacts
- Centralize deployment coordination in a shared document with standardized request/response templates, status tracking, and rollback plans

## 5. Pull Request Workflow
- Use a consistent PR template that captures change summary, testing evidence, security/performance checks, documentation updates, and rollback considerations
- Enforce Conventional Commits and keep PRs small and scoped to reduce merge conflicts

## 6. Versioning, Changelog & Releases
- Bump semantic versions and update CHANGELOG.md for any functional change; verify in CI/Danger checks
- Automate release tagging and changelog generation via scripts (e.g., standard-version)

## 7. Security, Error Handling & UX
- Validate all security reviews, dependency audits, and legal checks before merging
- Provide global error boundaries and log/recover gracefully without exposing sensitive data
- Conduct accessibility audits; ensure components are keyboard-accessible and meet contrast standards

These principles form a baseline for multi-agent software development: strict quality gates, deterministic tooling, transparent deployment protocols, and comprehensive documentation ensure that AI agents (and humans) collaborate effectively and deliver secure, maintainable software.
