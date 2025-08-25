# Current Product Status

AgentsMCP now ships a basic CLI, an async `AgentManager`, a FastAPI server, and an experimental RAG pipeline. The repository includes tests and CI workflows but lacks production-ready orchestration, pluggable retrieval, and end-to-end examples. Configuration via `agentsmcp.yaml` allows MCP clients to access tools such as `filesystem`, `git`, `bash`, and `web_search`.

## User Experience
- CLI-only interface for invoking agents.
- Documentation outlines expected behaviors and configuration points.
- Chat CLI adds `/models` for model discovery and `/apikey` for persisting keys.
- Provider validation runs non-blocking with actionable hints.

## Developer Experience
- Repository provides best-practice documentation and CI scaffolding.
- Sample configuration file enables tool access for agents.

## Integrations
- No external cloud services are integrated today.
- Future plans include pluggable data sources for retrieval-augmented generation.

## Opportunities
- Define standard agent roles and contracts.
- Showcase coordinated CI tasks using specialized agents.
 - Expand discovery (announcer, list, handshake) into richer coordination.
 - Grow web UI from scaffold to charts and controls.

## Currently Working Use Cases
- Acts as a template repository for spinning up new MCP-based agent systems.
 - Local discovery via registry + CLI listing when enabled.
 - Minimal web dashboard with SSE events, spawn, and cancel controls.
