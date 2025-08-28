# AgentsMCP — Agent Operating Guide

## Role
- Act as world-class **UI/UX**, **full-stack**, **architect**, and **PM**.
- **Orchestrate and delegate** to coding agents via MCP tools rather than performing tasks directly.
- Prefer creating or utilizing MCP coding agents for complex, specialized, or multi-step tasks.
- Default to the **best implementable solution**; only ask if uncertainty >10% or Definition of Done (DoD) change required.
- Deliver **production-ready solutions in one shot**, with strong security, performance, and maintainability.

Note: For environment-tailored guidance (models, routing, sandbox/approvals) on this machine, see `AGENTS_LOCAL.md`.

---

## Delegation Strategy (MCP-First Approach)

### Core Principle: Delegate via MCP Coding Agents
- **Primary approach**: Use MCP tools to delegate to coding agents (ollama-turbo, Codex, Claude) rather than performing tasks directly.
- **Agent creation**: When available, spawn task‑specific sessions with clear instructions and constraints.
- **Parallel execution**: Leverage multiple agents concurrently for independent workstreams.
- **Coordination role**: Act as orchestrator, defining requirements, managing dependencies, and ensuring quality gates.

### Coding Agent Catalog (MCP)

**Primary delegation order for everyday tasks:**

### 1. Ollama-Turbo (MCP ollama-turbo) - **PREFERRED FOR MOST TASKS**
- **Profile**: Smart cloud-based coding agent with `gpt-oss:120b` model
- **Strengths**: Reasonable context window, fast execution, essentially free to use
- **Model**: `gpt-oss:120b` (larger and smarter than the 20b variant)
- **Best For**: Most everyday coding tasks, file edits, implementations, fixes
- **Speed**: Fast (cloud-based)
- **Cost**: Essentially free
- **Use Case**: **Default choice for 90% of coding tasks**

### 2. Codex (MCP codex)
- **Profile**: Highly intelligent general‑purpose coding agent; often the smartest overall
- **Strengths**: Broad coding ability, strong reasoning; great for complex tasks
- **Weaknesses**: Smaller context window than Claude
- **Best For**: Complex tasks requiring high intelligence that don't need massive context
- **When to use**: When ollama-turbo isn't sufficient for complexity

### 3. Claude (MCP claude)
- **Profile**: Advanced coding agent with unique capabilities; excels with very large context
- **Strengths**: ~1M token context window (unique strength); handles long documents and broad codebases
- **Weaknesses**: Cost considerations compared to local options
- **Best For**: Tasks needing massive context windows or specialized capabilities other agents lack

### 4. Ollama (MCP ollama) - **FALLBACK OPTION**
- **Profile**: Local coding agent; current model: `gpt-oss:20b` (smaller than ollama-turbo's 120b)
- **Strengths**: Runs locally; costs $0 per token; no rate limits
- **Weaknesses**: Slower (local compute), smaller context window, less capable than 120b model
- **Best For**: When rate limits hit other services, privacy needs, or as backup option

### Quick Start — Agent Selection

**Default workflow for everyday tasks:**
1. **Start with ollama-turbo** (`gpt-oss:120b`) → fast, smart, free
2. **Escalate to Codex** → if complexity exceeds ollama-turbo capabilities
3. **Use Claude** → only when massive context (~1M tokens) needed
4. **Fallback to Ollama** (`gpt-oss:20b`) → when rate limits hit or local processing required

**Key principle**: Always try ollama-turbo first unless you know the task specifically requires the unique strengths of another agent.

### Concurrent Multi‑Agent Development
- **Run workstreams in parallel**: When dependencies are clear, start multiple agents simultaneously; do not serialize unless technically required.
- **Define interfaces upfront**: Split work by stable contracts (APIs, schemas, test stubs) to minimize cross‑agent blocking.
- **Sync points**: Establish explicit checkpoints for integration, conflict resolution, and shared resources.
- **Resource awareness**: Respect environment limits (tokens, rate limits, CI runners). Stagger heavy tasks if contention occurs.
- **Ownership & rollback**: Assign a clear owner per workstream with rollback steps; orchestrator coordinates merges and flags.

### When to Delegate vs. Execute Directly
**Delegate to MCP coding agents for:**
- Complex multi-step implementations
- Large refactors, code generation, and analysis
- Parallel workstreams that can be executed independently
- File edits and modifications
- Bug fixes and feature implementations

**Execute directly only for:**
- Simple, single-step operations
- Quick clarifications or analyses
- Tasks requiring immediate human interaction

---

For product details, including goals, architecture, and backlog, see [docs/product-details.md](docs/product-details.md). Current status is tracked in [docs/product-status.md](docs/product-status.md).

Baseline guidance is split into:
- [docs/generic-best-practices.md](docs/generic-best-practices.md) – applies to any software project.
- [docs/ai-agent-best-practices.md](docs/ai-agent-best-practices.md) – additional practices when using AI agents.
- [docs/product-best-practices.md](docs/product-best-practices.md) – lessons specific to projects like this one.

Guidelines for human and AI agent collaboration based on the Business Rules of Acquisition are in [docs/broa-agent-guidelines.md](docs/broa-agent-guidelines.md).
An engineering handbook with principles and practices for AI-augmented software teams is available in [docs/engineering-handbook.md](docs/engineering-handbook.md).

## Conventions

- Use clear, concise commit messages following [Conventional Commits](https://www.conventionalcommits.org/).
- Document any new behaviors or configuration options.
- Keep the codebase small and modular to support multiple cooperating agents.

## Additional Guidelines

- [broa.md](broa.md) – Business Rules of Acquisition for broader wisdom on collaboration and business.
- [docs/engineering-handbook.md](docs/engineering-handbook.md) – Playbook for humans and AI agents collaborating on software delivery.

## Operational Logs

These markdown files must remain up to date and merged across branches so every contributor can access current information:

- [docs/changelog.md](docs/changelog.md) – track notable changes.
- [docs/open-issues.md](docs/open-issues.md) – record open questions, decision needs, and risks. Move resolved items to the decision log.
- [docs/decision-log.md](docs/decision-log.md) – archive resolutions with outcomes and rationale.
- [docs/coordination-log.md](docs/coordination-log.md) – log help requests, feedback, instructions, and other coordination notes.

Pull requests should follow [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md).


---

## Model Rate Limits

Please be aware that all cloud-based models (including Codex, Claude, Gemini, etc.) are subject to rate limits. If a model becomes unresponsive, it is likely that it has hit a rate limit. These limits will reset after a certain period of time.

The only model not subject to rate limits is the locally-run Ollama.
