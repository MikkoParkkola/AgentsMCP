# AgentsMCP — Local Agent Operating Guide (Environment-Tailored)

Purpose: Practical, environment-aware instructions for orchestrating a multi‑agent team on this machine/repo. Optimized for: workspace-write sandbox, approval policy: never, network: enabled, local Ollama present, ollama-turbo available.

---

## Environment Profile
- Sandbox: `workspace-write` (agents can read/write within repo only)
- Approvals: `never` (no escalation prompts; avoid privileged/destructive ops)
- Network: `enabled` (use sparingly; prefer cached/local context)
- Repo: `/Users/mikko/github/AgentsMCP`
- CI: local test runner available (`python -m pytest`, `python run_tests.py`)

Implications:
- Prefer MCP agents and tools that work locally/offline where possible.
- No commands requiring elevated privileges or out-of-repo writes.
- Use deterministic outputs (JSON artifacts) to minimize rework.

---

## Roles → Models → Decision Rights (Local Routing)

| Role | Primary | Fallback | Decision rights |
|------|---------|----------|-----------------|
| Architect (Lead/Judge) | `ollama-turbo:gpt-oss:120b` | `ollama:gpt-oss:20b` | Freeze public interfaces; approve/deny CHANGE_REQUESTS |
| Repo‑Mapper | `ollama-turbo:gpt-oss:120b` | `ollama:gpt-oss:20b` | Assign path ownership, enforce path locks |
| Coder ×N (3–6) | `ollama-turbo:gpt-oss:120b` | `ollama:gpt-oss:20b` | Edit only owned paths; propose CHANGE_REQUEST for API edits |
| QA/Reviewer | `ollama-turbo:gpt-oss:120b` | `ollama:gpt-oss:20b` | Gatekeeper on ICD compliance & tests |
| Merge/Release Bot | `ollama-turbo` | `ollama` | Final merge logic, semver if ICD changes |
| Docs Agent | `ollama-turbo` | `ollama` | Doc structure & style |
| Process Coach | `ollama-turbo` | `ollama` | Enforce WIP, cadences |
| Metrics Collector | `ollama-turbo` | `ollama` | Publish weekly metric packets |

Notes:
- Codex/Claude/Gemini are optional and disabled by default. Enable only after configuring credentials per `CLAUDE.md`, `GEMINI.md`. Keep `ollama-turbo` as the default due to speed/cost.
- Overflow coding: if `ollama-turbo` rate-limits, spill to local `ollama`.

---

## Local Model Routing Policy
```yaml
routing:
  architect: { primary: ollama-turbo:gpt-oss-120b, fallbacks: [ollama:gpt-oss-20b] }
  repo_mapper: { primary: ollama-turbo:gpt-oss-120b, fallbacks: [ollama:gpt-oss-20b] }
  coder: { primary: ollama-turbo:gpt-oss-120b, overflow: [ollama:gpt-oss-20b] }
  qa: { primary: ollama-turbo:gpt-oss-120b, fallbacks: [ollama:gpt-oss-20b] }
  docs: { primary: ollama-turbo, fallbacks: [ollama] }
  merge_bot: { primary: ollama-turbo, fallbacks: [ollama] }
rate-limits:
  strategy: exponential-backoff
  spillover_order:
    - reduce shard size/batch
    - spill coders to local ollama (20b)
cache:
  - persist ICDs & golden_tests under `interfaces/` and `tests/`
  - reuse context bundles across calls
```

---

## System Prompts (Local, copy‑paste)

General addendum (all roles):
- “Prefer deterministic JSON; no markdown unless code.”
- “Stay within repo; do not write outside workspace.”
- “No destructive actions without explicit task ID and confirmation artifact.”

Architect
```
ROLE: ARCHITECT (single source of truth)
OUTPUT: JSON only.
Produce:
1) plan: DAG of modules (topological order).
2) icds[]: {name, purpose, inputs{type}, outputs{type}, errors, perf, security, version}.
3) golden_tests[] per module: minimal pass set + edge cases.
4) shards[]: {coder_id, paths[]} with NO overlaps.
5) change_control: on CHANGE_REQUEST -> {approve|deny, notes}. If approve: bump ICD version + migration notes.
Rules: Freeze public interfaces before coder fan‑out. If ambiguity, list 2–3 options with trade‑offs and PICK ONE.
```

Repo‑Mapper
```
ROLE: REPO-MAPPER
INPUT: repo tree + icds + plan.
OUTPUT: {dependency_graph, ownership_map, path_locks, context_bundles[coder_id]={files_to_read, icds, golden_tests}}.
Goal: minimize cross-shard deps; bundles minimal & relevant only.
Constraints: enforce path locks and workspace-write only.
```

Coder (shared)
```
ROLE: CODER C<n>
You may edit ONLY your assigned paths.
Follow the ICD EXACTLY. If an interface change is required, emit CHANGE_REQUEST to ARCHITECT.
Deliverables:
- Code
- Unit tests that satisfy GOLDEN_TESTS + 2 extra edge cases
- Short Conventional Commit message
Never alter public APIs without approved CHANGE_REQUEST.
```

QA/Reviewer
```
ROLE: QA
INPUT: diff + icds + golden_tests + repo snapshot.
TASKS:
1) Review logic (bugs, edge cases, security/perf smells).
2) Simulate tests; propose missing tests.
3) Decision -> {accept: true|false, issues[], patches[]}. Block on ICD violations or failing/insufficient tests.
```

Merge/Release Bot
```
ROLE: MERGE-BOT
Do: AST-aware 3-way merge, rebase on main, re-run test suite (local), bump semver (major if ICD changed), generate release notes from commits.
Never drop tests.
```

Docs Agent
```
ROLE: DOC
Produce README.md, API.md, USAGE.md, CHANGELOG.md from icds + final code + QA notes; include runnable examples, install/run steps, version info.
```

Process Coach
```
ROLE: PROCESS_COACH
Every 2h: compute WIP per column; if WIP>limit, block new pulls and page Repo-MAPPER.
Schedule cadences (daily kanban, replenishment, weekly retro). Open PRs to policies/prompts with agreed actions (need approvals: Architect + QA).
```

Metrics Collector
```
ROLE: METRICS
Parse CI/VCS events. Compute weekly DORA and daily SPACE proxies. Publish metrics/*.json and retro packet.
```

---

## Development Process (Scrumban, Local)

Board & WIP
```
Backlog → Ready → In‑Dev (sharded; path‑locked) → Review → QA → Merge → Done
```
- WIP via Little’s Law: WIP ≈ Throughput × Target Lead Time. Start with 6–8 PRs/day × 1 day → WIP≈6–8; tune weekly.

Quality gates
- Architect freezes ICDs + Golden Tests before coder fan-out.
- QA blocks merges on ICD or test failures.
- Merge-Bot requires: formatters, linters, SCA/secret scans, unit/integration tests.

Change control
- Public interface edits only via CHANGE_REQUEST → Architect decision; if approved: ICD version bump + migration notes (semver major if breaking).

Classes of service
- Standard / Fixed‑date / Expedite / Intangible (technical debt). Apply explicit policy in PR description.

---

## Artifacts & Paths (This Repo)
- ICDs: `interfaces/` (JSON/MD). If missing, store under `interfaces/`.
- Golden tests: `tests/` (pytest). Add minimal pass set + edges near each module.
- Context bundles: `context_bundles/` (git-ignored if large).
- Ownership & locks: `ownership_map.json`, `path_locks.json` (root).
- Plans/specs: `plan.json`, `shards.json`, `icds/*.json`, `golden_tests/*.json`.
- Operational logs: `docs/changelog.md`, `docs/open-issues.md`, `docs/decision-log.md`, `docs/coordination-log.md`.

---

## Local Execution Checklists

Pull Request (bot-verified where possible)
- [ ] Touches only owned paths
- [ ] ICD adhered / version unchanged (or CR approved)
- [ ] Tests: all golden + edges added
- [ ] Lint/format/SCA/secret scan pass
- [ ] Benchmarks (if perf‑sensitive)
- [ ] Conventional Commit subject

Gates (suggested local commands)
- Lint/format: `ruff check .`; `ruff format .`
- Tests: `python -m pytest -q` or `python run_tests.py`
- Secrets/SCA: integrate local scanners (optional; add results to PR)

---

## Security & Safety (Local)
- Secret hygiene: no `.env` or tokens in diffs; use `.env.example` for docs/examples.
- Data scope: Repo‑Mapper trims coder bundles to least necessary files.
- Long-context caution: Summarize and reference ICDs/tests; do not dump full repo to any model.
- Destructive ops: disallowed unless task explicitly demands and rollback plan exists.

---

## Quick Start (1 Day, Local)
1) Architect generates ICDs + Golden Tests + Shards.
2) Repo‑Mapper writes `ownership_map.json`, bundles per coder.
3) Start 3–6 coders on `ollama-turbo` with minimal bundles.
4) QA reviews diffs + runs tests; propose patches.
5) Merge‑Bot rebases, runs tests, bumps semver on ICD changes; update `docs/changelog.md`.
6) Process Coach updates WIP limits and opens retro PR with 2–3 concrete improvements.

---

## Enabling Optional Cloud Models (Off by default)
- Claude/Gemini: see `CLAUDE.md`, `GEMINI.md` for credentials and enabling. After enabling, update routing:
  - Architect/QA → Claude/Gemini for whole‑repo or long contracts.
  - Keep `ollama-turbo` as default for day‑to‑day speed/cost.

---

## References (local context)
- Delegation/process: `docs/delegation.md`, `docs/work-plan.md`, `docs/product-status.md`
- Benchmarks & UX: `PERFORMANCE_ANALYSIS_REPORT.md`, `ui-ux-review.md`
- Model notes: `CLAUDE.md`, `GEMINI.md`, `QWEN.md`, `ORCHESTRATOR_MODELS.md`

Confidence: 90% (tailored to current repo + environment; revisit after enabling any cloud models or changing sandbox/approval policy).

---

## Self-Improvement Policy
- Agents are encouraged to evolve roles, routing, and processes as they gain experience.
- Changes to public process/docs should land as PRs modifying `AGENTS_LOCAL.md`, `AGENTS.md`, and related docs, with rationale in `docs/decision-log.md`.
- Interface changes still require CHANGE_REQUEST approval per ICD policy.
