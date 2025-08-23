# Interfaces: Single-File Binary Build

Source: `scripts/build_binary.sh` (to be added)

## Inputs

- Environment variables:
  - `AGENTSMCP_VERSION` (optional): overrides version embedded in binary.
- Command-line flags:
  - `--onefile` (default): produce a single binary.
  - `--name agentsmcp` (default binary name).

## Behavior

- Entry: `src/agentsmcp/cli.py`.
- Uses PyInstaller with a minimal spec; bundles required resources.
- Outputs binary to `dist/agentsmcp`.

## Exit Criteria

- Exit code 0 on success; non-zero otherwise.
- Prints location of the produced binary.

