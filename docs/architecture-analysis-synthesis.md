# AgentsMCP Architecture Review and Synthesis (Archived)

Reference only. For current architecture priorities, see `docs/backlog.md`, `docs/work-plan.md`, and `docs/decision-log.md`.

---

This document synthesizes a fresh architectural/code analysis of AgentsMCP with prior inputs from Claude’s reports (notably docs/RECENT_CHANGES_ANALYSIS.md and PERFORMANCE_ANALYSIS_REPORT.md). It consolidates practical, production-focused recommendations with concrete next steps.

## Scope
- Assess core runtime (CLI, API, AgentManager), providers, tools, MCP gateway, discovery, and config surfaces.
- Reconcile Claude’s proposals with current code realities.
- Prioritize changes for robustness, error tolerance, performance, and maintainability.

## Current Architecture (Summary)
- API: FastAPI server with health, spawn, status, events, UI static, metrics.
- Orchestration: `AgentManager` with async job spawning, status, cancel.
- Agents: OpenAI Agents SDK-based wrappers (codex/claude/ollama); tools via registry; optional generic `mcp_call` tool.
- Config: Pydantic models + AppSettings; YAML loading and env merging.
- Storage: Memory default; interfaces defined for SQLite/Redis/Postgres.
- MCP: Minimal manager + gateway (stdio) with version negotiation, tool export.
- Discovery: Rich subsystem (announcer/registry/raft/etc.) behind flags.

## Key Issues Found (Fresh Analysis)
1. API uses new `AgentManager` instances per request for spawn/status/cancel. With memory storage, jobs aren’t visible across requests and in-flight jobs are lost. Use the single server instance (`self.agent_manager`).
2. `cleanup_completed_jobs` uses `datetime.replace(hour=hour - max_age_hours)` leading to invalid times. Use `datetime.utcnow() - timedelta(hours=...)`.
3. EventBus duplication: `events.py` (SSE dict bus) vs `orchestration.py` (typed/backpressure). Consolidate to one bus with backpressure + SSE adapter.
4. Providers are sync and lack standardized retry/backoff/circuit-breaking; risk of blocking I/O in async flows.
5. Tool parameter validation is permissive; file/web tools need stricter sandboxing and schema validation.
6. Config layering duplicated between `Config.from_env/load` and `AppSettings.to_runtime_config`, increasing drift risk.
7. API lacks AuthN/AuthZ and rate limiting; DoS and misuse possible in prod.

## Claude’s Inputs (Synthesized)
- Resource limits and centralized resource manager to prevent exhaustion (agents/memory/tokens).
- Configuration consolidation to a single source of truth with precedence.
- Emphasis on orchestration patterns, persistent memory, sandboxed execution, and richer coordination models.
- Performance reports suggest high potential; highlight orchestrator misconfig risk.

These align broadly with our findings; we focus on immediate, implementable steps that improve correctness and operability without over-complexity.

## Prioritized Recommendations

1) Correctness and Lifecycle
- Single `AgentManager` instance for all API routes.
- Fix job cleanup time math; add periodic job GC.
- Unify EventBus; typed events, bounded queues, SSE adapter; remove prints in libraries.

2) Robustness and Limits
- Add queue + worker pool (semaphore) in `AgentManager` with `max_concurrent_jobs` and `queue_capacity`.
- Return 429 with `Retry-After` when saturated; expose queue depth metric.
- Per-agent concurrency/rate limits to protect external APIs.

3) Providers and Networking
- Convert provider calls to async (`httpx.AsyncClient`), with timeouts, retry/backoff (exponential + jitter), and simple circuit breaker.
- Cache model lists with TTL per provider; explicit refresh endpoint.

4) Security and Safety
- JWT bearer auth middleware for spawn/cancel (dev bypass via flag); roles for admin/read.
- Optional TLS via `require_tls`, `tls_cert_path`, `tls_key_path`; fail fast if required assets missing.
- Rate limiting per IP (e.g., slowapi) on spawn/status.
- Harden tools: strict schema validation, canonical realpath checks for files, deny traversal/symlinks; web tool timeouts and domain allowlists.

5) Config and Maintainability
- Make `AppSettings.to_runtime_config` the single merge path; `Config` as schema-only. Deprecate `Config.from_env` path.
- Split API into routers (agents, health, metrics, coord, ui) with DI for shared instances.
- Standardize error envelope (code, type, message) across API; map provider errors consistently.
- Add correlation IDs (request + job_id) to logs; remove prints.

6) Observability
- Metrics: counters/histograms for job lifecycle per agent (spawned/running/completed/failed/timeout/duration/queue depth).
- Readiness: storage ping, queue headroom, optional provider probe (short timeout).
- Optional OpenTelemetry tracing for request→agent→provider spans.

## Rollout Plan
- Phase 1 (Now): Fix API AgentManager usage; cleanup math; consolidate EventBus; add basic metrics.
- Phase 2: Add queue/worker pool and 429 overload; async providers with retries; file/web tool hardening.
- Phase 3: JWT auth (prod), rate limiting, TLS; config consolidation; error envelope; readiness probes.
- Phase 4: Caching, tracing, model discovery endpoint; router refactor.

## Definition of Done (DoD) per Phase
- P1 DoD: Jobs persist across calls; cleanup works; events stable under load; unit/integration tests cover spawn/status/cancel/events; metrics visible.
- P2 DoD: Overload returns 429; no blocking I/O in async; file/web tools pass security tests.
- P3 DoD: Auth enabled in prod; rate limits enforce; TLS honored; config merge path unified; readiness is meaningful.
- P4 DoD: Caches with TTL; standardized API errors; tracing togglable.

## Test Strategy
- Add integration tests for: single AgentManager lifecycle; 429 saturation; cancel behavior; event stream under slow consumer; provider retry/backoff; file path sandbox.
- Fuzz tests for tool parameter validation and path traversal.
- Load test: sustained spawn/status with bounded concurrency and verify stability.

## Open Risks and Mitigations
- Complexity creep from discovery subsystem: keep behind flags; isolate imports.
- Auth/TLS adoption friction: clear env flags and dev defaults; strong prod checks.
- Provider variance: strict error mapping; opt-in probes in readiness.

## Summary
This synthesis balances Claude’s ambitious proposals with immediate, high-value fixes. The first two phases focus on correctness, safety, and scalability without architectural upheaval. Later phases add security hardening, standardization, and observability to reach production readiness.

