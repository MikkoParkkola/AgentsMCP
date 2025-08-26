# CLAUDE AGENT GUIDELINES

This document provides operating guidance for Claude-based coding and analysis workflows.

- Prefer Claude for large-context analysis and documentation refactors.
- Use structured, explicit instructions; provide file paths and goals.
- When editing code, propose minimal diffs and clear rationale.
- Validate changes logically (lint/tests) before proposing risky refactors.

Security and quality:
- Never exfiltrate secrets or tokens. Redact sensitive content in outputs.
- Favor deterministic, idempotent scripts and commands.
- Prefer incremental migrations with clear rollback steps.

