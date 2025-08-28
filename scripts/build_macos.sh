#!/usr/bin/env bash
set -euo pipefail

# Build Apple Silicon (arm64) standalone binary via PyInstaller
# Prereqs: python3, pip, pyinstaller installed in the environment

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> Installing PyInstaller if missing"
python3 -m pip show pyinstaller >/dev/null 2>&1 || python3 -m pip install pyinstaller

echo "==> Building agentsmcp (arm64)"
# Disable discovery components during build to avoid noisy side‑effects
export AGENTSMCP_DISABLE_DISCOVERY=1
# Ensure we build in single-file mode and output to dist/
pyinstaller --clean agentsmcp.spec

echo "==> Done. Binary at: dist/agentsmcp"
