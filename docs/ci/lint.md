# CI: Lint & Style

Workflows: `.github/workflows/ci.yml` (job `lint`), `.github/workflows/python-ci.yml` (part of `lint-test`).

## Purpose
- Enforce Python style and basic static checks with `ruff`.

## Inputs
- Python 3.12 (ci.yml) and 3.11 (python-ci.yml).
- Dependencies: `pip install .[dev]` or `ruff` installed explicitly.

## Steps (Reference)
- Checkout code.
- Setup Python.
- Install dev deps.
- Run `ruff check src tests`.

## Outputs
- GitHub Annotations for lint findings.
- Job status (pass/fail).

## Acceptance Criteria
- No `ruff` errors reported (warnings allowed per config).
- Job exits 0.

## Notes
- Optional future hooks: ESLint/Prettier for JS, `markdownlint`/`yamllint` for docs/config.

