# Tech Stack and Code Conventions

## Tech Stack
- **Language**: Python 3.10+
- **CLI Framework**: Click for command-line interface
- **Web Framework**: FastAPI + Uvicorn for API server
- **TUI Framework**: Rich for terminal user interface
- **Input Handling**: prompt_toolkit for advanced terminal input
- **Configuration**: Pydantic for settings and validation
- **Async**: asyncio for concurrent operations
- **Logging**: structlog for structured logging
- **HTTP Client**: httpx for API calls
- **AI Integration**: OpenAI API, with support for multiple providers

## Code Style and Conventions
- **Formatting**: Uses ruff for linting and formatting
- **Line Length**: 88 characters (Black-style)
- **Type Hints**: Extensive use of typing annotations
- **Docstrings**: Comprehensive docstrings for classes and functions
- **Error Handling**: Custom exception hierarchy with AgentsMCPError base class
- **Async Patterns**: Proper async/await usage throughout

## Project Structure
- `src/agentsmcp/` - Main package directory
- `src/agentsmcp/ui/` - User interface components
- `src/agentsmcp/ui/v2/` - Revolutionary TUI interface (v2)
- `src/agentsmcp/agents/` - AI agent implementations
- `src/agentsmcp/orchestration/` - Multi-agent coordination
- `src/agentsmcp/commands/` - CLI command implementations
- `src/agentsmcp/api/` - REST API endpoints
- `tests/` - Test suite

## Key Dependencies
- `rich` - Terminal UI framework
- `click` - CLI framework
- `fastapi` - Web API framework
- `pydantic` - Data validation and settings
- `structlog` - Structured logging
- `prompt_toolkit` - Advanced terminal input handling
- `httpx` - HTTP client
- `openai` - AI API integration