# Coordination Log

Record help requests, offers, feedback, instructions, orders, and escalations among contributors.

<!--
Template:
## YYYY-MM-DD HH:MM UTC
- **From:** name
- **To:** name or team
- **Type:** request | offer | feedback | instruction | escalation
- **Message:** summary
- **Follow-up:** next steps
-->

## 2025-08-21 23:11 UTC
- **From:** user
- **To:** contributors
- **Type:** instruction
- **Message:** Include Business Rules of Acquisition as broa.md and link from AGENTS.md.
- **Follow-up:** Document added and linked.

## 2025-08-21 23:34 UTC
- **From:** automated agent
- **To:** contributors
- **Type:** instruction
- **Message:** Set up CI, automerge, security scanning, and dependency automation.
- **Follow-up:** Define CODEOWNERS and expand tests as code evolves.

## 2025-08-21 23:54 UTC
- **From:** user
- **To:** contributors
- **Type:** instruction
- **Message:** Implement engineering handbook and guidelines for humans and AI agents.
- **Follow-up:** Handbook added and AGENTS.md updated.

## 2025-08-22 00:15 UTC
- **From:** user
- **To:** contributors
- **Type:** instruction
- **Message:** Rebase repository and fix failing Danger and lint workflows.
- **Follow-up:** Updated CI to use `ruff check` and provide `GITHUB_TOKEN` to Danger.

## 2025-08-22 00:30 UTC
- **From:** user
- **To:** contributors
- **Type:** instruction
- **Message:** Address unknown error by ensuring Danger installs non-interactively.
- **Follow-up:** Updated CI to invoke `npx --yes danger@11 ci`.

## 2025-08-22 00:58 UTC
- **From:** user
- **To:** contributors
- **Type:** instruction
- **Message:** Incorporate AI-agent team project best practices into product documentation.
- **Follow-up:** Best practices documented and referenced.

## 2025-08-23 01:00 UTC
- **From:** user
- **To:** contributors
- **Type:** instruction
- **Message:** Separate documentation into generic, AI-specific, product-specific, status, and product details sections.
- **Follow-up:** Docs reorganized and references updated.

## 2025-08-23 09:50 UTC
- **From:** user
- **To:** contributors
- **Type:** instruction
- **Message:** Document roadmap and backlog and provide MCP client configuration for tool access.
- **Follow-up:** Product docs updated and default configuration file added.

## 2025-08-26 10:15 UTC
- **From:** automated agent
- **To:** contributors
- **Type:** instruction
- **Message:** Completed fresh architecture/code analysis using Claude’s prior analysis as input; created synthesis and updated logs.
- **Follow-up:** Implement Phase 1 items: use single AgentManager in API, fix job cleanup time math, consolidate EventBus, add basic metrics. Prepare tests and PRs accordingly.
