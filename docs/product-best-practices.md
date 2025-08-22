# Product-Specific Best Practices

These learnings apply to projects similar to this CLI-driven MCP agent system.

## Self-Evaluation Questions
1. Does your product interact through a command-line interface?
2. Will multiple specialized agents collaborate on tasks?
3. Do you integrate external data sources or APIs?
4. Are there domain or compliance constraints (e.g., PII, licensing)?

### How to Use
- **Mostly "yes" answers:** adopt the practices below.
- **Mostly "no" answers:** treat them as optional; tailor to your context.

## Practices
- Provide a unified configuration format for transports, tools, and storage.
- Keep agent roles explicit (planner, coder, reviewer, tester, etc.).
- Record telemetry for agent decisions and outcomes.
- Maintain a backlog of agent capabilities and required integrations.

## Guidance for AGENTS.md
Each project should include in its AGENTS.md:
- A concise project overview and scope.
- Coding standards and language preferences.
- Required programmatic checks (tests, linters, security scans).
- Domain-specific constraints such as data handling rules or integration endpoints.
- Contacts or escalation paths for unresolved questions.
