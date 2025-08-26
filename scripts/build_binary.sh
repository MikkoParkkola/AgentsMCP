#!/usr/bin/env bash
# =============================================================================
# Build script for AgentsMCP binaries (minimal & full) using PyInstaller.
#
#   * Isolated virtual‑envs are created under .build/venv‑minimal and
#     .build/venv‑full.
#   * The project is installed with the appropriate extras.
#   * PyInstaller 6.6.* (or $PYINSTALLER_VERSION) is used to produce a single‑file
#     executable.
#   * The resulting binaries are placed in ./dist/ as
#       agentsmcp‑minimal   and   agentsmcp‑full
#   * After each build the script reports:
#       – un‑stripped size
#       – stripped size (if `strip` is available)
#       – SHA‑256 checksum
#   * Helpful macOS code‑signing / Gatekeeper hints are printed.
#
#   Flags:
#       --reuse-venvs          Do not delete existing virtual environments.
#       --full-only            Build only the full binary.
#       --minimal-only         Build only the minimal binary.
#       --extras "a,b,c"       Extras for the full build (default: metrics,discovery,rag,security)
#
#   The script aborts on the first error (set -euo pipefail) and prints
#   verbose step banners.
#
#   It is written for POSIX‑compatible shells (bash, dash, ksh, …) and does
#   not rely on Homebrew, sudo or any external package manager.
# =============================================================================

set -euo pipefail

log_banner() { printf "\n===== %s =====\n" "$*"; }
log_info()   { printf "[INFO] %s\n" "$*"; }
log_error()  { printf "[ERROR] %s\n" "$*" 1>&2; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --reuse-venvs          Keep existing virtual environments.
  --full-only            Build only the full binary.
  --minimal-only         Build only the minimal binary.
  --extras "a,b,c"       Extras for the full build (default: metrics,discovery,rag,security)
  -h, --help             Show this help message.
EOF
  exit 1
}

# Defaults
REUSE_VENVS=0
BUILD_MINIMAL=1
BUILD_FULL=1
FULL_EXTRAS="metrics,discovery,rag,security"

PYTHON_BIN=""

while [ $# -gt 0 ]; do
  case "$1" in
    --reuse-venvs)   REUSE_VENVS=1; shift ;;
    --full-only)     BUILD_MINIMAL=0; shift ;;
    --minimal-only)  BUILD_FULL=0; shift ;;
    --extras)
      if [ -z "${2-}" ]; then log_error "--extras requires a value"; usage; fi
      FULL_EXTRAS="$2"; shift 2 ;;
    --python)
      if [ -z "${2-}" ]; then log_error "--python requires a path (e.g., /usr/bin/python3.10)"; usage; fi
      PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) log_error "Unknown option: $1"; usage ;;
  esac
done

# Resolve repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR=".build"
PYINSTALLER_VERSION="${PYINSTALLER_VERSION:-6.6.*}"
EXCLUDE_SUBPACKAGES="ui web discovery rag benchmarking"

pick_python() {
  if [ -n "$PYTHON_BIN" ]; then
    echo "$PYTHON_BIN"
    return
  fi
  for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      ver=$("$cand" -c 'import sys; print(sys.version_info[:2])' 2>/dev/null | tr -d '() ,')
      major=$("$cand" -c 'import sys; print(sys.version_info[0])')
      minor=$("$cand" -c 'import sys; print(sys.version_info[1])')
      if [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; }; then
        echo "$cand"
        return
      fi
    fi
  done
  log_error "No suitable Python (>=3.10) found. Specify with --python /path/to/python3.10"
  exit 1
}

create_venv() {
  name="$1"
  path="${VENV_DIR}/venv-${name}"
  if [ -d "$path" ]; then
    if [ "$REUSE_VENVS" -eq 1 ]; then
      log_info "Re-using existing venv at $path"
    else
      log_info "Removing existing venv at $path"
      rm -rf "$path"
    fi
  fi
  if [ ! -d "$path" ]; then
    log_banner "Creating virtual environment for ${name}"
    PY=$(pick_python)
    "$PY" -m venv "$path"
  fi
  # shellcheck disable=SC1091
  . "${path}/bin/activate"
  python -m pip install --quiet --upgrade pip setuptools wheel
  python -m pip install --quiet "pyinstaller==${PYINSTALLER_VERSION}"
}

install_project() {
  extras="$1"
  if [ -n "$extras" ]; then
    log_info "Installing project with extras: $extras"
    python -m pip install --quiet ".[${extras}]"
  else
    log_info "Installing project without extras"
    python -m pip install --quiet .
  fi
}

finalize_binary() {
  name="$1"
  src_path="dist/${name}"
  if [ ! -f "$src_path" ]; then
    log_error "Expected binary $src_path not found"
    exit 1
  fi
  log_banner "Size report for $src_path"
  size_unstripped=$(stat -c%s "$src_path" 2>/dev/null || stat -f%z "$src_path")
  printf "  Unstripped size: %'d bytes\n" "$size_unstripped"
  if command -v strip >/dev/null 2>&1; then
    tmp_stripped="${src_path}.stripped"
    cp "$src_path" "$tmp_stripped"
    strip "$tmp_stripped" || true
    size_stripped=$(stat -c%s "$tmp_stripped" 2>/dev/null || stat -f%z "$tmp_stripped")
    printf "  Stripped size:   %'d bytes\n" "$size_stripped"
    rm -f "$tmp_stripped"
  else
    printf "  (strip not available – skipping stripped size)\n"
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$src_path"
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$src_path"
  else
    printf "  (no SHA-256 tool found – skipping checksum)\n"
  fi
}

build_minimal() {
  log_banner "Building minimal binary"
  create_venv minimal
  install_project ""
  SPEC_FILE="agentsmcp.spec"
  ENTRY_POINT="src/agentsmcp/cli.py"
  BUILD_NAME="agentsmcp-minimal"
  CMD="pyinstaller"
  for pkg in $EXCLUDE_SUBPACKAGES; do
    CMD="$CMD --exclude-module agentsmcp.$pkg"
  done
  if [ -f "$SPEC_FILE" ]; then
    log_info "Using spec file $SPEC_FILE (with excludes)"
    CMD="$CMD $SPEC_FILE"
  else
    log_info "No spec file – building directly from $ENTRY_POINT"
    CMD="$CMD -F --name \"$BUILD_NAME\" $ENTRY_POINT"
  fi
  # shellcheck disable=SC2086
  eval $CMD
  # When using a spec, PyInstaller will emit the name from the spec (likely 'agentsmcp').
  if [ -f "dist/$BUILD_NAME" ]; then
    finalize_binary "$BUILD_NAME"
  elif [ -f "dist/agentsmcp" ]; then
    mv "dist/agentsmcp" "dist/$BUILD_NAME"
    finalize_binary "$BUILD_NAME"
  else
    log_error "Expected binary not found in dist/. Check build logs."
    exit 1
  fi
}

build_full() {
  log_banner "Building full binary"
  create_venv full
  install_project "$FULL_EXTRAS"
  ENTRY_POINT="src/agentsmcp/cli.py"
  BUILD_NAME="agentsmcp-full"
  pyinstaller -F --name "$BUILD_NAME" "$ENTRY_POINT"
  finalize_binary "$BUILD_NAME"
}

print_macos_hints() {
  cat <<'EOF'

--- macOS troubleshooting hints ------------------------------------------------
If the binary fails to launch on macOS (e.g. “cannot be opened because the
developer cannot be verified”), consider the following steps:

1. Code signing (requires a valid Apple developer certificate):
   codesign --sign "Developer ID Application: Your Name (TEAMID)" \
            --timestamp --options runtime \
            path/to/agentsmcp-minimal

2. Notarisation (optional but recommended for Gatekeeper compliance):
   xcrun altool --notarize-app --primary-bundle-id com.example.agentsmcp \
                --username "apple-id@example.com" \
                --password "@keychain:AC_PASSWORD" \
                --file path/to/agentsmcp-minimal.zip

3. After notarisation, staple the ticket:
   xcrun stapler staple path/to/agentsmcp-minimal

4. If you just want to bypass Gatekeeper for testing:
   sudo spctl --add --label "AgentsMCP" path/to/agentsmcp-minimal
   sudo spctl --enable --label "AgentsMCP"

For more details see Apple’s “Notarizing Your Software” guide.

--- End of macOS hints ---------------------------------------------------------
EOF
}

log_banner "AgentsMCP PyInstaller build script"
log_info "Repository root: $REPO_ROOT"
log_info "PyInstaller version: $PYINSTALLER_VERSION"
log_info "Reuse venvs: $([ "$REUSE_VENVS" -eq 1 ] && echo yes || echo no)"
log_info "Build minimal: $([ "$BUILD_MINIMAL" -eq 1 ] && echo yes || echo no)"
log_info "Build full:    $([ "$BUILD_FULL" -eq 1 ] && echo yes || echo no)"
if [ "$BUILD_FULL" -eq 1 ]; then
  log_info "Full build extras: $FULL_EXTRAS"
fi

mkdir -p dist "$VENV_DIR"

[ "$BUILD_MINIMAL" -eq 1 ] && build_minimal || true
[ "$BUILD_FULL" -eq 1 ] && build_full || true

print_macos_hints

log_banner "All done."
