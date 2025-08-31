# AgentsMCP Documentation Index

This folder is the canonical source of truth for project status, plans, and backlog. To keep contributors aligned and avoid stale guidance, the structure below defines a single place for each kind of information.

Core sources of truth
- Backlog: `backlog.md` (authoritative, prioritized, ≤500‑LOC tasks). Start here.
- Status logs: `product-status.md` (current state), `changelog.md` (what changed), `open-issues.md` (decisions/risks), `decision-log.md` (resolved decisions), `coordination-log.md` (requests/notes).
- UX plan: `ui-ux-improvement-plan-revised.md` (canonical UX roadmap). Any older UX docs are reference only.
- Docs maintenance: `CONTRIBUTING-DOCS.md` (how to keep docs tidy and current).

How to use these docs
- New contributors: Read `backlog.md` first, then `product-status.md`. If something looks off, propose an update rather than assuming another doc is correct.
- Planning/triage: Update `backlog.md` and link entries from `open-issues.md`. Once decided, move to `decision-log.md` and update the backlog.
- Avoid duplication: Prefer linking to an existing doc over copying content. If you find contradictions, fix the canonical file and add a short note (“superseded by…”) to the older doc.

Document roles
- `backlog.md`: Prioritized, bite‑sized tasks with acceptance criteria. This is the single source of truth for what’s next.
- `work-plan.md`: High‑level organization of the backlog. Use it as overview only; update the backlog, not this plan, when priorities change.
- `implementation-roadmap.md`: Time‑phased roadmap. Should reference backlog items; keep it light.
- `ui-ux-improvement-plan-revised.md`: Canonical UX plan. Older `ui-ux-improvement-plan.md` is kept as historical context and now points here.
- `IMPLEMENTATION_BACKLOG.md`: Archived long‑form backlog. Do not implement from it; see `backlog.md`.

Contributing checklist
- Before starting work: Confirm the task exists in `backlog.md` and reflects current code.
- After landing changes: Update `changelog.md`; if a decision was made, add an entry to `decision-log.md`; if plans changed, update `backlog.md`.
- If docs seem outdated: Fix the canonical doc and add a short “Superseded by …” banner to any legacy pages.

Questions or contradictions
- Open an item in `open-issues.md` with “Docs mismatch” and link the files. Propose the fix in a PR updating the canonical file.
