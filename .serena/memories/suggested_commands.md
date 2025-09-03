# AgentsMCP Development Commands

## Installation & Setup
```bash
# Install development dependencies
pip install -e ".[dev,rag]"

# Install with all features
pip install -e ".[dev,rag,security,metrics,discovery]"

# Copy example configuration
cp .env.example .env
```

## Running the Application
```bash
# Main TUI interface (Revolutionary TUI)
./agentsmcp tui

# Alternative TUI commands
agentsmcp tui-alias                    # Launch Revolutionary TUI with auto-detection
agentsmcp tui-v2-dev                   # Launch v2 TUI directly for development
agentsmcp tui-v2-raw                   # Ultra-minimal raw TTY input tester

# Interactive CLI mode
agentsmcp interactive

# Start web server
agentsmcp server start --host 0.0.0.0 --port 8000

# Development shortcuts via Makefile
make dev-run                           # Run interactive mode
make dev-run-tui                       # Run TUI mode
```

## Testing Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agentsmcp --cov-fail-under=80

# Run specific test markers
pytest -m "ui"                         # UI tests
pytest -m "integration"                # Integration tests
pytest -m "unit"                       # Unit tests
pytest -m "async"                      # Async tests

# Run tests with timeout
pytest --timeout=30

# Run flaky tests with retries
pytest -m flaky --flaky-reruns=2 --flaky-reruns-delay=1
```

## Code Quality
```bash
# Lint and format with ruff
ruff check .
ruff format .

# Security scanning with bandit
bandit -r src/

# Dependency audit
pip-audit
```

## Build and Distribution
```bash
# Build distribution
make dist

# Clean build artifacts
make clean
```

## Development Environment
- **Platform**: Darwin (macOS)
- **Python**: 3.10+
- **Terminal**: TTY-capable terminal for TUI features
- **Dependencies**: See pyproject.toml for complete list