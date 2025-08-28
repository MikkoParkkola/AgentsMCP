#!/usr/bin/env bash
set -euo pipefail

echo "[pre-pr] Running lint (ruff)"
if ! command -v ruff >/dev/null 2>&1; then
  echo "[pre-pr] ruff not found. Install with: pip install ruff" >&2
else
  ruff check .
fi

echo "[pre-pr] Running tests (pytest -q)"
if ! command -v pytest >/dev/null 2>&1; then
  echo "[pre-pr] pytest not found. Install with: pip install -r requirements-test.txt" >&2
  exit 1
fi

pytest -q

echo "\n[pre-pr] Reminder: ensure PR checklist is satisfied (see .github/PULL_REQUEST_TEMPLATE.md)"
echo "- Conventional Commit, ICD adherence or CR approval, golden tests, changelog, scans"

echo "[pre-pr] OK"

