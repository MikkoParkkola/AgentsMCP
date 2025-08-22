# AI-Specific Best Practices

These practices complement the generic guidance when a project includes one or more AI agents.

## Opportunities
- Automate boilerplate coding, testing, and documentation tasks.
- Use agents to generate test cases and propose refactors.
- Leverage retrieval-augmented generation (RAG) for context-aware responses.

## Guardrails
- Sandbox agents and grant tools least-privilege access.
- Log all agent actions with reproducible seeds and configuration.
- Require human review for any destructive operation.

## Limitations
- Agents cannot guarantee factual accuracy or business viability.
- They struggle with long-term planning and non-deterministic systems.
- Agents lack accountability; humans remain responsible for decisions.

## Collaboration Patterns
- Pair agents with humans for code review and test design.
- Version prompts, tools, and policies as code.
- Evaluate agents on real tasks and track cost and success metrics.

## Security
- Apply OWASP LLM Top 10 and NIST AI Risk Management Framework.
- Secret-scan all diffs and restrict outbound network access.
