# BRoA Guidelines for Human and AI Agent Teams

This document synthesizes the "Business Rules of Acquisition" (BRoA) hosted at [broa.biz](https://broa.biz) with additional principles for working with AI agents and mixed human–AI teams.

## Core Principles
- **Customers & cash first.** Clarify the offer, demand, and willingness to pay before building. Lean pull logic ties work to real demand.
- **Measure, but design metrics carefully.** Use multi-perspective metrics and experimentation, and watch for Goodhart's Law and other metric traps.
- **Quality is a profit weapon.** Well-engineered products reduce cost and increase trust.
- **Teams over heroes.** Seek expertise, pair brains, and prioritize results over ego. Psychological safety fuels performance.
- **Plan for slack; avoid fake productivity.** Utilization above ~80% or chronic overtime increases cycle time and harms health.
- **Lean pull and change agility.** Operate pull systems (e.g., Kanban/JIT) and adapt quickly to change.
- **Meetings and focus.** Keep processes simple and avoid meetings without purpose.

## Conflicts & Guardrails
- **Exploration vs. exploitation.** "Only engage in battles you have already won" is useful for expected value but must not kill exploration. Isolate exploratory bets.
- **Details vs. quality.** Balance "To hell with the details" with "Quality is a weapon" by focusing on vital customer-facing details first.
- **Work people pay for vs. long-term investment.** Operations should deliver immediate value, but reserve budget for R&D, platform work, and brand building.
- **Efficiency targets.** Aim for ~75–85% resource utilization to avoid queue explosions.

## Gaps for Modern Teams
- Incorporate AI, data, and security governance.
- Address ethics, sustainability, and ESG commitments.
- Support remote/distributed practices and explicit DEI and accessibility principles.

## Suggested Refinements
- Fix typos and clarify attributions (e.g., "imcompetence" → "incompetence").
- Clarify heuristics like efficiency targets and exploration bets.
- New heuristics: build psychological safety before speed; experimentation beats opinion.

## Rules for AI Agents
- **[AI-AGENT-NEW-1]** Let the AI agent write boilerplate; you handle the cohesion.
- **[AI-AGENT-NEW-2]** If the AI agent suggests more code than you asked, thank it—then trim the fat.
- **[AI-AGENT-NEW-3]** The AI agent is only as smart as its last prompt. Treat prompts like cargo-pants pockets—messy, but loaded.

## Shared Rules for Human + AI Teams
- **[HYBRID-AGENT-NEW-4]** If you're debugging, loop in the AI agent—two sets of eyes are better than one.
- **[HYBRID-AGENT-NEW-5]** Always review AI agent–written code—even if it compiles.
- **[HYBRID-AGENT-NEW-6]** Let the AI agent generate test cases—and you break them.
- **[HYBRID-AGENT-NEW-7]** Celebrate when the AI agent autofixes—but still document the fix.

## "Use Now" Checklist
1. Make it easy to pay/ask/see: instrument funnels, publish service catalogs, and close feedback loops.
2. Measure what matters: write metric intents and audit for Goodhart risk quarterly.
3. Try new tactics: A/B test backlog items and kill under-performers quickly.
4. Target ~80% utilization: staff for slack and track WIP/queue time.
5. Quality as a weapon: add automated tests and SLOs to definitions of done.
6. Ask experts: maintain decision logs with SME sign-off.
7. Solve, don’t blame: run incident reviews with psychological safety.
8. Meeting hygiene: mandate agenda/owner/decision and track meeting cost.
9. Find/generate pull: use Kanban for inbound work and limit WIP.
10. ROI with balance: pair ROI with customer, process, and learning metrics.
11. Overtime is expensive: cap hours and design for sustainability.
12. Keep it simple: ship the smallest valuable change and iterate.

## Theme Mapping
- **Customer & value:** #2–#4, #16, #70, #73, #77.
- **Metrics & learning:** #9, #30, #38, #41, #43, #101, #108.
- **People & org:** #11, #14, #26, #29, #46–#49, #83, #94, #111.
- **Flow & ops:** #31, #32, #50, #71, #73, #106, #109.

## Nuance
- Over-indexing on ROI can starve discovery; balance with learning metrics.
- Quality without cost discipline risks gold-plating; use cost-of-quality and SLO error budgets.
- Slack may look "inefficient"; defend it with queueing math and incident data.

## Verdict
Most BRoA rules remain evergreen. Refine them with modern governance, clarify exploration vs. exploitation, and apply the new AI-focused rules above.
