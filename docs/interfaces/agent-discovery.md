# Interfaces: Agent Discovery & Coordination

Goal: effortless, privacy-respecting discovery of other AI agents on the same host/network, with a minimal handshake to exchange capabilities and an optional control channel for coordination. Opt-in, with Apple-like simplicity for users.

## Principles
- Opt‑in and safe by default; nothing announces without explicit enable.
- Minimal mental model: agents “appear” with name and status; click to connect.
- Pluggable transports: Zeroconf/mDNS (preferred), local registry/socket fallback.
- No surprises: clear allowlists and optional shared token for trust.

## Identifiers & Capability Schema

```json
{
  "id": "uuid-v4",
  "name": "codex",
  "version": "1.0.0",
  "provider": "openai|openrouter|ollama|custom",
  "models": ["o4-mini", "gpt-4o", "llama3:8b"],
  "roles": ["chat", "code", "rag"],
  "endpoints": {
    "mcp": {"stdio": true, "sse": "http://127.0.0.1:8000/mcp", "ws": null},
    "rest": {"base": "http://127.0.0.1:8000"},
    "sse": "http://127.0.0.1:8000/events"
  },
  "auth": {"token_required": false},
  "meta": {"uptime_s": 12345, "load": 0.12}
}
```

## Zeroconf/mDNS (Preferred)
- Service type: `_agentsmcp._tcp.local.`
- SRV: host:port of REST base.
- TXT keys (short, limited):
  - `id`, `name`, `ver`, `prov`, `roles`, `sse=1|0`, `mcp=stdio|sse|ws`, `auth=token|none`.
- Payload details resolved by fetching `/.well-known/agentsmcp.json` from host.

## Local Registry Fallback
- Path: `~/.agentsmcp/registry.json` (owned by user, 0600).
- Each agent appends (creates if missing) its descriptor; stale entries pruned by heartbeat timestamp.

## Handshake (Coordination)
- Client fetches descriptor (mDNS -> well-known; or reads registry).
- Optional token exchange: if `auth.token_required=true`, include `Authorization: Bearer <token>` in subsequent requests.
- Capability verification: GET `/capabilities` returns supported tools and limits.
- Optional control channel agreed: MCP (stdio/sse/ws) or REST.

## REST Endpoints (Minimum)
- `GET /.well-known/agentsmcp.json` → descriptor (above schema)
- `GET /capabilities` → list of tools, limits, model set
- `POST /coord/ping` → `{ok:true, ts}` sanity check

## Security
- Off by default; require `discovery.enabled=true`.
- Optional `discovery.allowlist=[agent_id|name|cidr]`.
- Optional `discovery.token` required to accept coordination calls.
- Never broadcast secrets.

## CLI
- `agentsmcp discovery enable|disable`
- `agentsmcp discovery list`
- `agentsmcp discovery info <id|name>`
- `agentsmcp discovery trust add <id|name>`

## Events
- When enabled, emit SSE `discovery/seen` and `discovery/lost` with `{id,name}` for UI feedback.

```text
id: 123
event: discovery/seen
data: {"id":"...","name":"codex"}
```

## Notes
- Keep payloads small in TXT; prefer well-known HTTP for details.
- Registry fallback should prune entries with a TTL (e.g., 10 min).
- Avoid invasive network scans; rely on passive/standard mechanisms.

