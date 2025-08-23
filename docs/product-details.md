# Product Details

This repository documents a CLI-driven MCP agent system with an **extensible retrieval-augmented generation (RAG) pipeline**, a **configurable runtime**, and a **multi-agent DevOps workflow**. It serves as a reference implementation for building agents that interact through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

## Architecture Overview
- **CLI-first interface** for invoking agents.
- **Extensible RAG layer** to plug in various data sources for context.
- **Configurable runtime** to adapt transports, storage, and tools via configuration files.
- **Multi-agent workflow** enabling specialized agents for linting, testing, deployment, and other lifecycle tasks.

## Roadmap
### Phase 0 – Repository foundation *(completed)*
- Baseline docs, CI scaffolding, and QA guardrails.
### Phase 1 – Core runtime & CLI *(in progress)*
- Introduce a CLI skeleton to orchestrate agents.
- Ship configuration examples for transports and tools.
### Phase 2 – Retrieval-augmented generation (RAG)
- Implement a pluggable RAG module with at least one data-source plugin.
- Add unit and integration tests to meet ≥80% coverage.
### Phase 3 – Multi-agent workflow
- Demonstrate a multi-agent CI pipeline driven via the new CLI.
- Define standard agent roles and interaction contracts.
### Phase 4 – Integrations & polish
- Document deployment options and external data-source integrations.
- Expand docs, diagrams, and ADRs for maintainability.

## Backlog
- Replace placeholder CODEOWNERS with real maintainers.
- Complete decision-log entry 0007.
- Scaffold CLI for agent orchestration.
- Implement minimal pluggable RAG module.
- Provide configuration examples.
- Demonstrate multi-agent CI pipeline.
- Define standard agent roles and contracts.
- Expand test coverage to ≥80%.
- Document external data-source integration.
- Create runtime setup guide.

## MCP Client Configuration

AgentsMCP uses `src/agentsmcp/config.py` to define agents and the tools they can access. A sample configuration file is provided at the repository root (`agentsmcp.yaml`):

```yaml
server:
  host: localhost
  port: 8000
transport:
  type: http
storage:
  type: memory
rag:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  chunk_size: 512
  chunk_overlap: 50
  max_results: 10
  similarity_threshold: 0.7
agents:
  codex:
    type: codex
    model: gpt-4
    system_prompt: "You are a code generation and analysis expert."
    tools: ["filesystem", "git", "bash"]
  claude:
    type: claude
    model: claude-3-sonnet
    system_prompt: "You are a helpful AI assistant with deep reasoning capabilities."
    tools: ["filesystem", "web_search"]
  ollama:
    type: ollama
    model: llama2
    system_prompt: "You are a cost-effective AI assistant for general tasks."
    tools: ["filesystem"]
tools:
  - name: filesystem
    type: filesystem
    config:
      allowed_paths:
        - /tmp
        - .
  - name: git
    type: git
    config: {}
  - name: bash
    type: bash
    config:
      timeout: 60
  - name: web_search
    type: web_search
    config: {}
```

This configuration enables MCP clients to use additional tools out of the box.

For foundational practices on running these projects, see [generic-best-practices.md](generic-best-practices.md), [ai-agent-best-practices.md](ai-agent-best-practices.md), and [product-best-practices.md](product-best-practices.md).

For additional background and context, see the original project description:

<https://chatgpt.com/s/dr_68a797b295e08191aee3dc2b2523646b>
