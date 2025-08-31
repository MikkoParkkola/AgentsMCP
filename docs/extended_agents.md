# Extended Agents and Self‑Improving Loop

## New World‑Class Agent Roles

| Role | Focus | Key Responsibilities |
|------|-------|-----------------------|
| **Product Manager** | Vision, roadmap, stakeholder alignment | • Define product strategy and OKRs<br>• Prioritise backlog items<br>• Communicate requirements to design & engineering<br>• Measure success metrics and iterate |
| **UX/UI Designer** | User experience, visual design | • Conduct user research & usability testing<br>• Produce wireframes, mock‑ups, design system components<br>• Ensure accessibility and design consistency |
| **Backend Developer** | Server‑side architecture & data services | • Design scalable APIs, databases, and background jobs<br>• Implement security, performance, and reliability best practices |
| **Frontend Developer** | Client‑side implementation | • Translate UI designs into responsive, performant code (React, Vue, etc.)<br>• Implement state management, component libraries, and accessibility |
| **API Developer** | Contract‑first API design | • Define OpenAPI/GraphQL schemas<br>• Generate SDKs, documentation, and versioning strategy |
| **Full‑Stack Developer** | End‑to‑end feature delivery | • Own feature from design through deployment, bridging frontend & backend concerns |
| **TUI Specialist Developer** | Terminal‑User‑Interface experiences | • Build rich, interactive CLI/TUI tools (e.g., using `rich`, `blessed`, `curses`)
• Ensure ergonomics for power‑users |
| **Business Analyst** | Requirements elicitation & analysis | • Gather stakeholder needs, model business processes, create functional specifications |

## Self‑Improving Retrospective Loop

1. **Task Completion** – After any task (LLM‑generated output or an agent’s work) a *Retrospective* is triggered.
2. **Retrospective Structure**
   - **What Went Well** – successes, correct assumptions, efficient patterns.
   - **What Could Be Better** – mis‑steps, missing context, performance issues.
   - **Experiment Ideas** – new prompts, tooling, architectural alternatives.
3. **Logging** – The results are appended to a timestamped log file `logs/retrospective-<ISO8601>.md`.
4. **Feedback Application** –
   - Update the **Orchestrator’s internal instructions** (the system prompt) to incorporate learnings.
   - Update each **Agent’s instruction set** (e.g., `agents/agent‑<role>.md`) with refined guidance.
   - Optionally adjust CI pipelines, test suites, or coding standards.
5. **Automation** – The orchestrator will:
   - Parse the latest retrospective file.
   - Merge relevant sections into the target instruction files.
   - Commit the changes with a conventional commit message `chore: apply retrospective insights <timestamp>`.

## Integration Steps

1. **Create Agent Instruction Templates** – Add a directory `agents/` with a markdown file per role (e.g., `agents/product_manager.md`). Populate each with the role description above and a placeholder for dynamic guidance.
2. **Add Retrospective Logger** – Implement a small utility script `scripts/retrospect.sh` that accepts a JSON payload, formats it, and writes to the log directory.
3. **Orchestrator Hook** – Extend the orchestrator (the current LLM client) to call the logger after each delegated task and then run a *sync* step that reads the latest log and patches the instruction files.
4. **Version Control** – Ensure all generated files are tracked. Use conventional commit messages for each automated update so the history remains clear.
5. **Monitoring** – Add a simple dashboard (e.g., `dashboard/retrospective.html`) that lists recent retrospectives and the cumulative changes to instructions.

## Benefits

- Continuous learning loop drives higher quality code and clearer communication.
- Explicit role separation enables parallel development streams, reducing latency.
- Retrospective‑driven instruction updates keep the system aligned with evolving best practices without manual intervention.

---

*This document provides the blueprint for extending the existing AgentsMCP environment with world‑class specialist agents and an automated self‑improving feedback cycle.*