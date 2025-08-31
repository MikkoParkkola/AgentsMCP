# Current Product Status

AgentsMCP provides a CLI, an async `AgentManager`, a FastAPI server, and early UI surfaces (legacy TUI v1; TUI v2 in active development). The repository includes tests and CI scaffolding. Production-grade orchestration, pluggable retrieval, and full E2E examples are in progress.

## User Experience
- CLI: core commands; chat CLI includes `/models` and `/provider` workflows (see `docs/usage.md`).
- TUI: v1 (Rich-based) is stable but limited; v2 is the target and under active development (see `docs/ui-ux-improvement-plan-revised.md`).
- Provider validation: non-blocking with actionable hints; setup wizard planned (see backlog).

## Developer Experience
- Clear backlog (`docs/backlog.md`) with ≤500‑LOC tasks and acceptance criteria.
- Best-practice docs and CI scaffolding for contributions.

## Integrations
- MCP tools via configuration; provider model discovery and streaming tracked in the backlog.

## Opportunities (near-term)
- Finish TUI v2 input + command palette and logs (Phase 1 in UX plan).
- Provider/model discovery facade and `/models` UX.
- SSE event bus + basic web dashboard.

## Currently Working Use Cases
- Template for MCP-based agent systems.
- Local discovery (registry) + CLI listing when enabled.
- Minimal web dashboard with SSE (scaffold).

See `docs/backlog.md` for what’s next and `docs/changelog.md` for recent changes.

