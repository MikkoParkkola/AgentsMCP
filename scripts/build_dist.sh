#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")"/.. && pwd)
cd "$ROOT_DIR"

echo "[build] Cleaning previous artifacts"
rm -rf build dist .build || true

echo "[build] Building with PyInstaller (agentsmcp.spec)"
# Ensure PyInstaller uses workspace-local cache/work dirs to satisfy sandbox
export PYINSTALLER_CONFIG_DIR="$ROOT_DIR/.build/pyinstaller_cfg"
mkdir -p "$PYINSTALLER_CONFIG_DIR" .build/pyi_work dist
python -m PyInstaller \
  --clean --noconfirm \
  --workpath .build/pyi_work \
  --distpath dist \
  agentsmcp.spec 2>&1 | tee build.log

echo "[build] Build complete: dist/agentsmcp"
if command -v strings >/dev/null 2>&1; then
  if strings -n 4 dist/agentsmcp | rg -q "TUIShell|tui_shell"; then
    echo "[build] Verified: TUI shell present in binary"
  else
    echo "[build] Warning: TUI shell not detected in binary (hidden import missing?)" >&2
  fi
fi

echo "[build] Try: ./dist/agentsmcp interactive --ui tui --no-welcome"
