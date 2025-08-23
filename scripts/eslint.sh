#!/usr/bin/env bash
set -euo pipefail

# Delegate to locally installed ESLint (or fetch via npx if not installed)
exec npx --yes eslint "$@"
