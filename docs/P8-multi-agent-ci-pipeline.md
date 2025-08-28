**P8: Multi-Agent CI Pipeline — Architecture & Implementation Guide**

Goal: Provide a production-ready, multi-agent CI pipeline that integrates cleanly with AGENTS.md v2 orchestration, supports parallel execution, rich UX, configuration presets, status monitoring, and robust error handling.

—

**1) Pipeline Architecture**
- **Core Engine:** `PipelineExecutor` executes a `PipelineSpec` (Pydantic schema in `src/agentsmcp/pipeline/schema.py`). Each `StageSpec` contains one or more `AgentAssignment`s.
- **Agent Orchestration:** Reuse AGENTS.md v2 components:
  - Tier 1 coordinator patterns (single main loop principles) via `MainCoordinator` concepts for queueing and quality gates.
  - Tier 2 stateless agent execution via `DelegationEngine` or direct `AgentManager` calls.
- **Parallelism:** Within a stage, if `parallel=True`, execute agent assignments concurrently with `asyncio.gather`. Cross-stage execution is ordered by `stages[]` (future: `depends_on`).
- **Agent Mapping:** Schema agent types → internal agents:
  - `ollama-turbo` → internal `ollama` (compat mapping; transparent to users)
  - `codex` → `codex`
  - `claude` → `claude`
- **Events:** Publish typed events via `agentsmcp.orchestration.EventBus`:
  - `PipelineStarted/Completed/Failed`
  - `StageStarted/Completed/Failed`
  - Reuse existing `JobStarted/JobCompleted/JobFailed` for per-assignment tracking.
- **Observability:** Instrument with `observability.instrument_agent_operation` and emit metrics:
  - `agentsmcp_ci_pipeline_runs_total{status}`
  - `agentsmcp_ci_stage_duration_seconds{stage}`
  - Correlate runs using `x-correlation-id` middleware.
- **HITL Security:** For sensitive stages (e.g., deploy), gate execution with `@hitl_required(...)` and surface approval UI via the existing HITL endpoints.

—

**2) Configuration Schema & Management**
- **Schema:** Use `PipelineSpec` plus `StageSpec` and `AgentAssignment` from `src/agentsmcp/pipeline/schema.py`.
- **Loader:** `PipelineConfig` in `src/agentsmcp/config/pipeline_config.py` loads and validates `pipeline:` blocks from YAML.
- **Presets & Templates:** Provide curated YAML presets per project type in `templates/pipelines/` (Python, Node, Docker). Each preset wires typical stages: build → test → analysis → package → deploy.
- **Integration with AgentsMCP Config:** Allow provider/model overrides and per-role preferences from `Config` to influence agent selection where `model` is omitted.
- **Project Detection:** On `agentsmcp ci init`, auto-detect project type (presence of `pyproject.toml`, `package.json`, `Dockerfile`) and suggest a preset.

—

**3) CLI Command Structure**
- Root group: `agentsmcp ci`
  - `init` — scaffold a pipeline file from preset: `agentsmcp ci init --preset python --out .agentsmcp/pipeline.yml`
  - `run` — execute a pipeline: `agentsmcp ci run --file .agentsmcp/pipeline.yml [--stage test] [--from build --until deploy] [--parallelism N] [--json]`
  - `status` — show latest or specific run: `agentsmcp ci status [--run-id RUN] [--json]`
  - `watch` — live TUI with progress bars and status panels (Claude Code–style polish)
  - `list-templates` — list available presets
  - `explain` — describe what the pipeline will do and which agents/models will run
  - Common flags: `--resume`, `--dry-run`, `--max-retries`, `--timeout`, `--on-failure (abort|skip|retry)`

—

**4) Status Monitoring & Progress Tracking**
- **Event Sources:** Subscribe to `JobStarted/JobCompleted/JobFailed` and new `Stage*/Pipeline*` events to aggregate status.
- **TUI UX:** New `src/agentsmcp/ui/pipeline_view.py` that reuses `ui_components` for:
  - Header with pipeline name, run ID, elapsed time
  - Stage panels with status chips, per-assignment progress bars, and logs tail
  - Footer with key bindings (pause, cancel, retry failed)
- **Web UI:** Add `/ci/runs`, `/ci/events` (SSE) and `/ci/run/{id}` endpoints; extend existing dashboard with a Pipeline tab.
- **Metrics:** Expose Prometheus counters/histograms; document label cardinality discipline.

—

**5) File Organization & Modules**
- `src/agentsmcp/pipeline/`
  - `schema.py` (already present)
  - `executor.py` — orchestrates a single run of `PipelineSpec` with retry/backoff and events
  - `types.py` — `PipelineRun`, `StageRun`, `AssignmentRun` dataclasses for state tracking
  - `events.py` — `PipelineStarted/Completed/Failed`, `StageStarted/Completed/Failed`
  - `presets/` — programmatic builders (optional; YAML lives in templates/)
- `src/agentsmcp/commands/ci.py` — CLI group and subcommands
- `src/agentsmcp/ui/pipeline_view.py` — live terminal renderer for runs
- `src/agentsmcp/web/routes_ci.py` — web/API endpoints for CI runs and SSE

—

**6) Execution Semantics**
- Per-stage:
  - If `parallel=True` (default), run all `AgentAssignment`s concurrently; otherwise sequentially.
  - Apply inherited defaults: `timeout_seconds`, `retries`, `on_failure` from stage/pipeline.
  - Map schema agent types to internal `AgentManager` types; build a structured prompt from `AgentAssignment.task` + `payload`.
- Failure policies:
  - `retry` — exponential backoff (jitter), respect `retries`
  - `skip` — mark assignment skipped and continue; stage success if any assignment succeeded (configurable rule: all/any)
  - `abort` — fail stage and short-circuit pipeline
- Recovery:
  - `--resume` picks up from the first failed stage/assignment using persisted run state in `storage` backends
  - Idempotency: compute an assignment key from `(stage, task, payload_hash)` to support safe retries

—

**7) Integration Points**
- **AgentManager:** Primary execution surface; reuse `spawn_agent` and `wait_for_completion` (benefits: unified lifecycle, events, storage, cleanup).
- **DelegationEngine:** For role-aware routing, convert `AgentAssignment` to `TaskEnvelopeV1` and leverage roles for richer prompts where beneficial.
- **HITL:** Decorate deploy-like operations (`on_before_stage('deploy')`) with `@hitl_required(operation='deploy', risk_level='high')`.
- **Observability:** Wrap stage/assignment execution in `@instrument_agent_operation('ci.stage')` and emit `instrument_agent_event('ci.assignment', ...)`.

—

**8) Templates & Presets (YAML)**
- `templates/pipelines/python-ci.yaml` — build (ruff), test (pytest), analysis (coverage, bandit)
- `templates/pipelines/node-ci.yaml` — build (npm ci + build), test (jest), analysis (eslint)
- `templates/pipelines/docker-ci.yaml` — build image, scan (trivy), optional deploy

Each uses `ollama-turbo` as the default workhorse, with `codex` or `claude` for complex analysis where appropriate.

—

**9) Error Handling & UX**
- Clear, consistent status chips: success, running, pending, skipped, error
- On failure: show cause, next action hint (retry/skip/abort); provide `agentsmcp ci run --resume --from <stage>` convenience
- Persist run logs and metadata under `~/.agentsmcp/runs/<run-id>.jsonl`

—

**10) Implementation Checklist**
- Add executor, events, types modules under `src/agentsmcp/pipeline/`
- Wire `commands/ci.py` with subcommands (init/run/status/watch)
- Add `ui/pipeline_view.py` using existing UI components
- Add web routes for CI runs and SSE
- Provide YAML presets under `templates/pipelines/`
- Document in `docs/ci/` and link from `AGENTS.md` and `README.md`
- Update `docs/changelog.md` and track open issues/decisions

—

**Definition of Done (DoD)**
- Can run `agentsmcp ci run --file templates/pipelines/python-ci.yaml` locally
- Live TUI shows per-stage progress and status updates
- Metrics exposed at `/metrics` include pipeline counters and stage durations
- HITL approval gates protect deploy stages when enabled
- Resume works after failure; configuration validated with helpful errors

