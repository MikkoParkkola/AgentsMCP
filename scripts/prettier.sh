#!/usr/bin/env bash
set -euo pipefail

# Delegate to locally installed Prettier (or fetch via npx if not installed)
exec npx --yes prettier "$@"
