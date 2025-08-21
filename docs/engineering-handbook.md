# Engineering Handbook for AI-Augmented Teams

This playbook outlines principles and practices for humans and AI agents collaborating on software delivery. It is adapted from evidence-based industry research and should guide day-to-day work.

## Principles (What to Optimize For)
- **Fast, safe flow of value**: work in small batches with frequent integration and delivery. Measure using the four key outcomes (Deployment Frequency, Lead Time, Change Failure Rate, Mean Time to Restore).
- **Work on outcomes, not output**: follow Agile values and 12 principles alongside Lean software development principles (eliminate waste, build quality in, defer commitment, deliver fast, empower people, build integrity in, optimize the whole).
- **Reliability as a feature**: use SLOs and error budgets to govern the pace of change and prioritization.
- **Cross-functional, stream-aligned teams**: reduce handoffs, leverage platform/enabling teams to lower cognitive load, and maintain psychological safety.
- **XP mindset**: apply simple design, TDD, refactoring, pair/mob programming, collective ownership, and continuous integration.
- **Continuous discovery & early validation**: test assumptions before building through dual-track discovery/delivery, RATs, and build-measure-learn loops.
- **AI-agent augmentation, not autopilot**: use instrumented, least-privilege tools via open protocols, secure against LLM-specific risks, and evaluate agents on real software tasks.

## Practices (How to Implement)

### 1. Planning, Prioritization & Feedback
- Set one Product Goal leading to Sprint Goals; maintain a clear Definition of Done.
- Run dual-track: weekly discovery loops (interviews, assumption tests, rapid prototypes) in parallel with delivery.
- Prioritize by value & speed using WSJF (Cost of Delay / Duration) and RICE (Reach × Impact × Confidence / Effort).
- Hold retrospectives each iteration with explicit actions.

### 2. Architecture & Code Quality
- Use trunk-based development (≤3 active branches; daily merges; no code freezes). Ship safely with feature flags and branch-by-abstraction.
- Review code for health (design, tests, simplicity, readability) with small PRs.
- Practice TDD and refactoring, aiming for a healthy test pyramid (many fast unit tests, fewer service/UI tests).

### 3. Testing & Risk Reduction
- Test the riskiest assumptions first (feasibility, value, compliance). Prefer cheap experiments over building.
- Keep tests fast, reliable, and readable; classify by size (small/medium/large) to keep pipelines quick.

### 4. CI/CD, DevOps & SRE
- Automate the pipeline: build → test → security scan → artifact → deploy → verify → rollback on failure. Ship on demand.
- Use progressive delivery (canary/blue-green) behind flags; tie observability and auto-rollback to SLO alerts.
- Track DORA metrics as the north star for software delivery performance.

### 5. AI Coding Agents — Team Setup, Rules & Guardrails
- **Tooling architecture**: standardize agent tool access via MCP; version prompts, tools, and policies as code; capture full telemetry of agent steps.
- **Safety & compliance**: apply OWASP LLM Top 10 and NIST AI RMF; sandbox agents, allow-list tools, secret-scan all diffs, and require human approval for destructive operations.
- **Evaluation & QA**: benchmark agent changes on SWE-bench (Verified)/SWE-Gym before rollout; track cost, success rate, and regressions.
- **Operational limits**: set budgets (tokens/time), provide a kill switch and escalation routes, and ensure reproducible runs with logged seeds and configs. Treat agent tools as production-grade integrations.
- **Suggested roles**:
  - *Humans*: Product lead, Tech lead/architect, Platform/DevEx, SRE, Security, QA/SET, Designer.
  - *Agents*: Planner/orchestrator, Coder, Reviewer/Linter, Test-writer, Security scanner, Docs/Changelog bot, CI operator. Map agents to tools with least privilege via MCP.

### 6. Team Topology, Skills & Soft Skills
- Organize around stream-aligned teams; use platform/enabling teams to reduce cognitive load and accelerate flow.
- Invest in psychological safety; encourage speaking up and sharing mistakes.
- Core skills include outcome-driven product thinking, readable code, test design, observability, incident management, secure-by-default practices, data literacy, and prompt/tool design for agents.

## Minimal Golden Rules (One-Pager)
1. Ship small to trunk daily; gate with CI; hide behind flags.
2. Keep SLOs and error budgets; freeze features when budgets are burned.
3. Practice TDD on critical paths; maintain a steep test pyramid.
4. Run discovery every week; test riskiest assumptions first.
5. Standardize agent tools via MCP; enforce LLM security controls; benchmark agents before rollout.
6. Measure with DORA metrics; improve with retrospectives.

## Selected Sources
- Agile Manifesto & Principles; Lean software development principles.
- DORA research on CI/CD and trunk-based development; *Continuous Delivery*.
- Google Engineering Practices: code review; test sizing; Fowler's Test Pyramid.
- Site Reliability Engineering: SLOs, error budgets, alerting on SLOs.
- AI agents: MCP specification; OWASP LLM Top 10; SWE-bench/SWE-Gym.
- Team Topologies; psychological safety (Project Aristotle).
- Community reflections on CI/TBD and dual-track agile debates.

## Assumptions
- Modern cloud-hosted service, Git-based workflow, automated CI available, MCP-capable agent stack planned.

## Confidence
- 88% — breadth of guidance is high, but specific organizational constraints may require tailoring. Adjust this handbook as the team learns and evolves.

