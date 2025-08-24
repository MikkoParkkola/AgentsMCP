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

AgentsMCP uses environment-based configuration. See `.env.example` for all options:

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

See [docs/deployment.md](docs/deployment.md) for comprehensive deployment guidance including:

- Docker and Kubernetes manifests
- Load balancing and scaling
- Monitoring and observability
- Security considerations
- Troubleshooting guide

## Documentation

### Core Documentation
- [docs/deployment.md](docs/deployment.md) - Production deployment guide
- [docs/product-status.md](docs/product-status.md) - Current status and roadmap
 - [docs/product-details.md](docs/product-details.md) - Detailed product information
 - [docs/work-plan.md](docs/work-plan.md) - Prioritized development plan (specs + acceptance criteria)
  - [docs/backlog.md](docs/backlog.md) - Decomposed backlog (≤500 LOC tasks)
 - CI workflows (expected outputs and acceptance criteria):
   - [docs/ci/README.md](docs/ci/README.md)
   - [docs/ci/lint.md](docs/ci/lint.md)
   - [docs/ci/tests.md](docs/ci/tests.md)
   - [docs/ci/security.md](docs/ci/security.md)
   - [docs/ci/container.md](docs/ci/container.md)
   - [docs/ci/release.md](docs/ci/release.md)
   - [docs/ci/ai-review.md](docs/ci/ai-review.md)
   - [docs/ci/automerge.md](docs/ci/automerge.md)
   - [docs/ci/e2e.md](docs/ci/e2e.md)
 - Interfaces (for parallel work):
   - [docs/interfaces/README.md](docs/interfaces/README.md)
   - [docs/interfaces/providers.md](docs/interfaces/providers.md)
   - [docs/interfaces/chat.md](docs/interfaces/chat.md)
   - [docs/interfaces/api-keys.md](docs/interfaces/api-keys.md)
   - [docs/interfaces/mcp-versioning.md](docs/interfaces/mcp-versioning.md)
   - [docs/interfaces/context.md](docs/interfaces/context.md)
   - [docs/interfaces/streaming.md](docs/interfaces/streaming.md)
 - [docs/interfaces/build.md](docs/interfaces/build.md)
  - [docs/interfaces/delegation.md](docs/interfaces/delegation.md)

### Best Practices
- [docs/generic-best-practices.md](docs/generic-best-practices.md)
- [docs/ai-agent-best-practices.md](docs/ai-agent-best-practices.md)

## Binaries

- macOS (Apple Silicon, arm64): prebuilt binary attached to Releases.
- Build locally: see [docs/standalone-binary.md](docs/standalone-binary.md).
- [docs/product-best-practices.md](docs/product-best-practices.md)

## License

See [LICENSE](LICENSE) file for details.
