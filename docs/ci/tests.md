# CI: Tests & Coverage

Workflows: `.github/workflows/ci.yml` (job `test`), `.github/workflows/python-ci.yml` (job `lint-test`).

## Purpose
- Run unit tests across a Python matrix and collect coverage artifacts.

## Inputs
- Matrix: Python `3.10`, `3.11`, `3.12` (ci.yml); single `3.11` run in python-ci.yml.
- Dependencies: `pip install .[dev]`.

## Steps (Reference)
- Checkout code.
- Setup matrix Python.
- Install dev deps.
- Run `pytest --cov-report=xml --cov-report=term-missing` (ci.yml) or `pytest -q` (python-ci.yml).
- Upload `coverage.xml` (ci.yml) or `.coverage` (python-ci.yml) as artifact.

## Outputs
- Test logs in job output.
- Coverage report artifact per Python version.

## Acceptance Criteria
- All test jobs pass (exit code 0).
- Coverage artifact uploaded for each matrix entry.
- Optional: establish a minimum coverage threshold (future).

## Notes
- Fail-fast disabled in matrix to surface issues across versions.

