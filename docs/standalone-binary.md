# Single-file Executable (Standalone)

You can package AgentsMCP into a single executable that bundles Python + deps using PyInstaller or Nuitka.

## PyInstaller (simple)

Prereqs: `pip install pyinstaller`

Build:
```bash
# From repo root
pyinstaller -F -n agentsmcp -p src src/agentsmcp/cli.py
```

- This creates `dist/agentsmcp` (macOS/Linux) or `dist/agentsmcp.exe` (Windows)
- The binary includes all required modules; no extra files needed

Run:
```bash
./dist/agentsmcp --help
./dist/agentsmcp chat
```

Notes:
- If you have dynamic optional deps (e.g., MCP SDK), include with `--hidden-import` if the auto-discovery misses them.
- You can add icons or metadata via additional PyInstaller flags.

## Nuitka (optimized)

Prereqs: `pip install nuitka`

Build:
```bash
python -m nuitka --onefile --include-package=agentsmcp --output-filename=agentsmcp src/agentsmcp/cli.py
```

This produces a single optimized binary. Nuitka can yield smaller/faster binaries at the cost of longer builds.

## Minimizing footprint
- Exclude dev/test tooling (`-[dev]`) from your install
- For a tiny user-only package, publish a separate PyPI dist (see below) that vendors only runtime files

## Minimal user package (optional)
- Create a second package name (e.g., `agentsmcp-client`) that includes:
  - Core modules: `src/agentsmcp/{cli.py,commands,agents,tools,settings.py,config.py}`
  - Exclude: `tests`, CI, heavy optional modules
- Provide the same console script entry point `agentsmcp`
- Publish to PyPI for easy install: `pip install agentsmcp-client`

## Prebuilt macOS (Apple Silicon, arm64)

- Our GitHub Actions release pipeline builds a native macOS arm64 binary on `macos-14` runners and attaches it to each tagged release (see Releases).
- File name: `agentsmcp` (no extension). Download, `chmod +x agentsmcp`, and run.
- For local builds on M-series Macs, simply run the PyInstaller command on your machine to get an arm64-native binary.
