# AgentsMCP

![CI](https://github.com/MikkoParkkola/AgentsMCP/actions/workflows/ci.yml/badge.svg)
![Tests](https://github.com/MikkoParkkola/AgentsMCP/actions/workflows/python-ci.yml/badge.svg)
![CodeQL](https://github.com/MikkoParkkola/AgentsMCP/actions/workflows/codeql-python.yml/badge.svg)
![Semgrep](https://github.com/MikkoParkkola/AgentsMCP/actions/workflows/semgrep.yml/badge.svg)

Production-ready MCP server for managing AI agents (Claude, Codex, Ollama) with extensible RAG pipeline and comprehensive CLI/API interfaces.

## Features

- **Multi-Agent Support**: Claude, Codex, and Ollama agents with configurable capabilities
- **CLI & API**: Complete command-line interface and REST API
- **Production Ready**: Comprehensive CI/CD, security scanning, containerization, health checks
- **Flexible Storage**: Memory, Redis, and PostgreSQL backends
- **RAG Pipeline**: Extensible retrieval-augmented generation system
- **Observability**: Structured logging, metrics, and health monitoring

## Quick Start

### Docker Compose (Recommended)

```bash
# Clone and configure
git clone <repository-url>
cd AgentsMCP
cp .env.example .env

# Start services
docker-compose up -d

# Test the API
curl http://localhost:8000/health
```

### CLI Installation

```bash
# Install
pip install -e ".[dev,rag]"

# Start server
agentsmcp server start

# Spawn an agent
agentsmcp agent spawn codex "Analyze this code structure"

### macOS Binary

- Binary path: `dist/agentsmcp` (Mach-O arm64)
- Interactive UI: `./dist/agentsmcp interactive` (defaults to `ollama-turbo-coding` with `gpt-oss:120b`)
- Web UI server: `uvicorn agentsmcp.server:create_app --factory --host 127.0.0.1 --port 8000` then open http://127.0.0.1:8000/ui

Defaults:
- Provider: `ollama-turbo`
- Model: `gpt-oss:120b`
- MCP tools pre-wired for coding: GitHub, filesystem, git, bash, web search. Additional MCP servers can be added via `agentsmcp mcp add ...`.

Startup guidance loading:
- On startup, AgentsMCP reads local `AGENTS.md` and folds key guidance into the system context. Provider-specific guidance lives in `docs/models.md`.
```

## MCP Integration

- Define MCP servers in `agentsmcp.yaml` under `mcp:` with `transport`, `command` (for `stdio`) or `url` (for `sse`/`websocket`), and `enabled` flag.
- Allow agents to use MCP by listing server names under each agent's `mcp:` field. Agents gain a generic `mcp_call` tool that routes calls to the selected server.

Example:

```yaml
mcp:
  - name: git-mcp
    enabled: true
    transport: stdio
    command: ["npx", "-y", "@modelcontextprotocol/server-git"]

agents:
  codex:
    mcp: [git-mcp]
```

CLI to manage MCP servers:

```bash
# List configured MCP servers
agentsmcp mcp list

# Enable/disable by name
agentsmcp mcp enable git-mcp
agentsmcp mcp disable git-mcp

# Add or remove servers in config
agentsmcp mcp add git-mcp --transport stdio --command npx --command -y --command @modelcontextprotocol/server-git
agentsmcp mcp remove git-mcp
```

At runtime, agents have a `mcp_call` tool with parameters: `server`, `tool`, and `params`.
If a Python MCP client isn’t installed, calls return an informative message rather than failing.

## CLI Usage

### Agent Management
```bash
# Spawn agents
agentsmcp agent spawn codex "Write a Python function"
agentsmcp agent spawn claude "Review this large codebase"  
agentsmcp agent spawn ollama "Simple code formatting"

# Monitor jobs
agentsmcp agent list
agentsmcp agent status <job-id>
agentsmcp agent cancel <job-id>
```

### Server Management
```bash
# Start/stop server
agentsmcp server start --host 0.0.0.0 --port 8000
agentsmcp server stop
```

## API Usage

### Spawn Agent
```bash
curl -X POST http://localhost:8000/spawn \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "codex", "task": "Write a hello world function", "timeout": 300}'
```

### Check Status
```bash
curl http://localhost:8000/status/<job-id>
```

### Health Checks
```bash
curl http://localhost:8000/health        # Basic health
curl http://localhost:8000/health/ready  # Readiness probe
curl http://localhost:8000/health/live   # Liveness probe
curl http://localhost:8000/metrics       # Metrics
```

## Configuration

AgentsMCP uses environment-based configuration. See `.env.example` for all options. For quick starts, use the presets under `examples/`:

- Minimal: `examples/config-minimal.yaml` (memory storage, core agents, UI and discovery off)
- Full: `examples/config-full.yaml` (adds UI mount, some MCP servers)

Install extras for optional features:

- Discovery: `pip install -e .[discovery]`
- Metrics: `pip install -e .[metrics]`
- RAG: `pip install -e .[rag]`
- Security: `pip install -e .[security]`
- UI/Cost/Bench placeholders: `.[ui]`, `.[cost]`, `.[bench]`

Environment flags of interest:

- `AGENTS_PROMETHEUS_ENABLED=false` (default) to disable metrics overhead
- `AGENTSMCP_CONFIG=/path/to/config.yaml` to pick a preset

```bash
# Server
AGENTSMCP_HOST=localhost
AGENTSMCP_PORT=8000
AGENTSMCP_LOG_LEVEL=info

# Storage (memory/redis/postgresql)
AGENTSMCP_STORAGE_TYPE=memory
AGENTSMCP_STORAGE_DATABASE_URL=postgresql://user:pass@localhost/agentsmcp

# Agent API Keys
AGENTSMCP_CODEX_API_KEY=your_key_here
AGENTSMCP_CLAUDE_API_KEY=your_key_here
AGENTSMCP_OLLAMA_HOST=http://localhost:11434
```

## Development

### Setup
```bash
# Install development dependencies
pip install -e ".[dev,rag]"

# Run tests
pytest

# Lint code
ruff check src tests
ruff format src tests

# Security scan
bandit -r src
pip-audit
```

### Testing
```bash
# Run all tests with coverage
pytest --cov-report=xml --cov-report=term-missing

# Run specific tests
pytest tests/test_agent_manager.py
```

## Architecture

- **Agent Manager**: Orchestrates agent lifecycle and job execution
- **Storage Layer**: Pluggable persistence (Memory/Redis/PostgreSQL)
- **CLI Interface**: Complete command-line management
- **REST API**: FastAPI server with OpenAPI documentation
- **RAG Pipeline**: Extensible document retrieval and processing

## Production Deployment

## Performance Notes

- Use uvicorn factory mode to avoid import-time app creation:
  - `uvicorn agentsmcp.server:create_app --factory --host 127.0.0.1 --port 8000`
- CLI lazy-loads heavy modules; startup is fastest when using simple commands:
  - `agentsmcp --help` or `agentsmcp models`
- Interactive mode does not start the Web UI by default; opt-in with:
  - `agentsmcp interactive --webui`
- MCP package warmup is now explicit to avoid default network calls:
  - `agentsmcp mcp warmup` (optional; caches common `npx` MCP servers)

MCP API and UI
- `/mcp` REST endpoints are disabled by default. Enable by config:
  - Set `mcp_api_enabled: true` in your config (or env override via `AGENTSMCP_CONFIG`).
- The static Web UI (`/ui`) is mounted only when `ui_enabled: true` in config.

MCP Transport Flags
- Control which transports the manager is allowed to use (defaults favor stdio only):
  - `mcp_stdio_enabled: true`
  - `mcp_ws_enabled: false`
  - `mcp_sse_enabled: false`
- These flags are merged from environment via AppSettings; set `AGENTSMCP_MCP_STDIO_ENABLED`, `AGENTSMCP_MCP_WS_ENABLED`, `AGENTSMCP_MCP_SSE_ENABLED` as needed.

CLI
- Show MCP status: `agentsmcp mcp status` (also available via REST at `/mcp/status` when enabled)


See [docs/deployment.md](docs/deployment.md) for comprehensive deployment guidance including:

- Docker and Kubernetes manifests
- Load balancing and scaling
- Monitoring and observability
- Security considerations
- Troubleshooting guide

## Documentation

- Backlog: [docs/backlog.md](docs/backlog.md) — prioritized tasks
- Changelog: [docs/changelog.md](docs/changelog.md) — notable changes
- Architecture: [docs/AGENTIC_ARCHITECTURE.md](docs/AGENTIC_ARCHITECTURE.md)
- Interfaces: [docs/interfaces/README.md](docs/interfaces/README.md)

  

## License

See [LICENSE](LICENSE) file for details.
# Recommended for TUI input handling on iTerm2/Terminal.app
pip install prompt_toolkit
