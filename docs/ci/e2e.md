# CI: E2E Smoke (Planned)

Workflow: (to be added) `.github/workflows/e2e.yml`.

## Purpose
- Launch minimal MCP gateway/server and run a smoke test to enumerate tools and perform a no-op call.

## Proposed Steps
- Checkout; setup Python; install .[dev].
- Start MCP server in background (stdio).
- Run a small Python script to connect and list tools; assert non-empty.

## Outputs
- Smoke logs; artifact with server logs on failure.

## Acceptance Criteria
- Tools list is non-empty; a sample tool call returns a 2xx/OK-like result.

## Notes
- Keep runtime under ~2 minutes; avoid network calls.

