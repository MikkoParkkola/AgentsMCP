# Developer Interfaces

Concise interface specifications to enable parallel implementation across modules. These are stable contracts (subject to minor adjustments) and should be adhered to by contributors.

- providers.md: Model discovery adapters and shared types
- chat.md: Chat command handlers and session runtime
- api-keys.md: Provider key validation, prompting, persistence
- mcp-versioning.md: MCP gateway version negotiation and tool schema down-conversion
- context.md: Deterministic context window trimming
- streaming.md: Streaming response interfaces
- build.md: Single-file binary build script inputs/outputs
- delegation.md: Delegation spec (command + MCP workflows)

For priorities and acceptance criteria, see `../backlog.md`.


## ICD Policy (source of truth)

This repository treats interface control documents (ICDs) as the single source of truth for public module contracts used by agents.

- Files: one JSON (or Markdown when necessary) per module, named by dotted module id (dots may be kept or replaced with dashes).
- Required keys: name, purpose, inputs, outputs, errors, perf, security, version.
- Versioning: semantic versioning. Any breaking change requires a CHANGE_REQUEST approved by the Architect.
- Self‑improvement: Agents may propose ICD updates via CHANGE_REQUEST. Upon approval, bump the version and add migration notes to `docs/changelog.md`.
