#!/usr/bin/env bash
set -euo pipefail

# Build a single-file agentsmcp binary using PyInstaller.
# Produces platform-native binary (arm64 when run on Apple Silicon runners).

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENTRYPOINT="src/agentsmcp/cli.py"
NAME="agentsmcp"

echo "[build] Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "[build] Installing PyInstaller..."
  python -m pip install --upgrade pip
  python -m pip install pyinstaller
fi

echo "[build] Building $NAME binary..."
pyinstaller -F -n "$NAME" -p src "$ENTRYPOINT"

echo "[build] Built binary at: $(pwd)/dist/$NAME"
