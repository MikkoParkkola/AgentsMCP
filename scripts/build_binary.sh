#!/usr/bin/env bash
set -euo pipefail

echo "[agentsmcp] Building standalone binary via PyInstaller (P1)"
if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "PyInstaller not found. Install with: pip install pyinstaller" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

OUTDIR="dist/agentsmcp"
pyinstaller -F -n agentsmcp --paths src --hidden-import=agentsmcp.cli -i none \
  --collect-all agentsmcp \
  -p src \
  src/agentsmcp/cli.py

echo "Binary output (if successful): dist/agentsmcp"
ls -la dist || true

