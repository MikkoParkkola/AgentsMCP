# Installation

This project ships two common install paths:

1) Full repo (developers)
- Requirements: Python 3.10+, pip
- Steps:
  - `git clone <repo-url>`
  - `cd AgentsMCP`
  - `pip install -e .[dev,rag]`

2) Minimal user install (recommended for users)
- Requirements: Python 3.10+, pip
- Steps:
  - `pip install -e .`
  - Or install only core deps: `pip install agentsmcp` (when published)

Optional MCP server support
- For MCP gateway/server features, also install a Python MCP SDK.
- Example (placeholder): `pip install modelcontextprotocol`

Quick sanity check
- `agentsmcp --version`
- `agentsmcp agent list`

