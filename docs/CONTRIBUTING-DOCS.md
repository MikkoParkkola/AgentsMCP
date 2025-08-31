# Contributing to Documentation

Our goal is one source of truth and zero duplication. Please follow these rules:

Canonical sources
- Backlog: `docs/backlog.md` — authoritative for upcoming work. Keep it current.
- UX roadmap: `docs/ui-ux-improvement-plan-revised.md` — canonical UX plan.
- Status logs: `docs/product-status.md`, `docs/changelog.md`, `docs/open-issues.md`, `docs/decision-log.md`, `docs/coordination-log.md`.

When updating docs
- If you change priorities or scope, update `backlog.md`. Link decisions in `open-issues.md` and resolve them in `decision-log.md`.
- Do not duplicate content. If another doc contains overlapping guidance, add a one‑line banner pointing to the canonical doc and trim the duplicate section.
- Keep summaries in high‑level plans (`work-plan.md`, `implementation-roadmap.md`) brief and link to `backlog.md` for details.

Maintenance checklist (PRs)
- [ ] Backlog updated (if work items changed).
- [ ] Changelog updated (if user‑visible change landed).
- [ ] Decision logged (if a decision was made/changed).
- [ ] Status updated (if overall state changed).
- [ ] Older/duplicate docs tagged with a short banner (Archived/Superseded/Reference) when applicable.

Disagreements or contradictions
- Open an item in `docs/open-issues.md` with “Docs mismatch”, link the files, and propose the fix in the PR.
