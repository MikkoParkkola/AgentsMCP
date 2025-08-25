# Agent Discovery Protocol Specification  
*(docs/interfaces/agent-discovery.md)*

> **Audience** – Engineers building the Agents MCP orchestrator and its
>  constituent agents.  
> **Purpose** – A single, stable discovery protocol that allows any
>  compatible agent to:  
> • announce its presence,  
> • query for peers,  
> • receive information about peers,  
> • do so securely and with graceful degradation when only a registry is
>   available.  
> **Scope** – The specification covers the *over-wire* interface, transport
>  binding, JSON payloads, compatibility, and the security model. It does
>  **not** prescribe a particular registry back-end implementation or an
>  internal service mesh.

---

## 1. Protocol Overview

| Concept | Description |
|---------|-------------|
| **Discovery Service** | An agent-managed, optional *local* service that advertises & resolves agents in its local network. |
| **Registry Fallback** | A *central* registry (e.g. Consul, ETCD, or a custom MCP endpoint) required when mDNS is not viable or when agents span heterogeneous networks. |
| **Transport Abstraction** | The same JSON schema is used over multiple transports: UDP/mDNS, HTTP/REST, WebSocket, or the custom MCP bus. |
| **Version Negotiation** | Every message contains `protocol_version`. Agents ignore unknown versions and optionally downgrade. |

**High-level flow**

```
+-----------+          +-----------+          +-----------+
| Agent A   | <-----   | Discovery | -------- | Agent B   |
| (Announce)                 | (Registry)          | (Response)
+-----------+          +-----------+          +-----------+
```

1. Agent A sends *Announce* on the local network (mDNS) and optionally to the registry.  
2. Agent B listens for `announcement` messages or queries the registry.  
3. Optional `query` requests may be sent; responses contain the full metadata.

---

## 2. Agent Identity Model

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_id` | `string` | *Yes* | Globally-unique UUIDv4. Must be RFC 4122 UUID. |
| `agent_name` | `string` | No | Human-readable, DNS-label compliant. |
| `capabilities` | `{ [string]string }` | *Yes* | Key/value map of feature name → version or description. |
| `public_key` | `string` | *Yes* (TLS-only) | PEM-encoded public key used for mutual-auth if required. |
| `transport` | `{ type:string, endpoint:string }` | *Yes* | Describes how the agent can be reached (TCP, Unix-sock, WebSocket). |
| `metadata` | `{ [string]any }` | No | Arbitrary key/value blob. Reserved keys: `__mcp_version`, `__os`, `__arch`. |

> **Example**  
> ```json
> {
>   "agent_id": "3cde1b52-1fc2-4f7b-8d8c-5e3e6f5c9c1d",
>   "agent_name": "data-processor",
>   "capabilities": {
>     "task-manager": "1.4",
>     "metrics": "2.0",
>     "store": "0.9"
>   },
>   "public_key": "MIIBIjANBgkqh…",
>   "transport": { "type": "tcp", "endpoint": "10.0.0.12:9000" },
>   "metadata": { "__os": "linux", "__arch": "amd64" }
> }
> ```

**Acceptance Criteria**

1. Every running instance must publish a `agent_id` that is unique across the cluster.  
2. `capabilities` must include at least an empty object; missing this field is rejected.  
3. The `public_key` must be PEM-encoded X.509; signature verification must succeed when provided.  

---

## 3. Discovery Mechanisms

| Mechanism | Use-case | Notes |
|-----------|---------|-------|
| **mDNS / Zeroconf** | On-prem, same-subnet discovery. | Default "primary". Uses UDP broadcast on port 5353. |
| **UDP Broadcast (Custom)** | Legacy or non-DNS networks. | Agents broadcast on a configurable network interface/port. |
| **Registry Fallback** | Multi-subnet, external networks, highly reliable. | Registry writes/reads *announcements*, *queries*, and *responses* through HTTP. |
| **Event Bus (MCP)** | High-throughput or message-heavy environments. | Agents publish to a *discovery topic* on the MCP bus. |

### 3.1 mDNS

* Service type: `_agentsc._tcp.local.`  
* Payload: JSON string as defined below.  
* TTL: 10 seconds (configurable).  
* Discovery client: listens for responses.

### 3.2 Registry Fallback

* Endpoint: `<registry>/agents`  
* Methods: `POST /agents` (announce), `GET /agents/{id}` (query), `GET /agents` (list).  
* Authentication: Mutual TLS or HTTP Bearer token.  

> **Rollback Path** – If mDNS fails (e.g., DNS disabled), agents immediately register with the central registry. When the mDNS interface becomes available again, they *suppress* registry updates; both mechanisms can coexist.

---

## 4. Message Formats

All messages are UTF-8 encoded JSON. The top-level object contains a mandatory `protocol_version`.

| Message | Mandatory | Optional |
|---------|-----------|----------|
| **Announce** | `agent` (see 2) | `protocol_version`, `signature` |
| **Query** | `protocol_version`, `query` | `agent_id` (optional) |
| **Response** | `protocol_version`, `agent` | `matched` (bool) |

### 4.1 JSON Schemas

#### 4.1.1 Announce

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "AnnounceMessage",
  "type": "object",
  "required": ["protocol_version", "agent"],
  "properties": {
    "protocol_version": { "type": "string", "pattern": "^v\\d+\\.\\d+$" },
    "agent": { "$ref": "#/definitions/Agent" },
    "signature": {
      "type": "string",
      "description": "Base64 of RSA or ECDSA signature over the canonical JSON of the agent object."
    }
  },
  "definitions": {
    "Agent": { "$ref": "#/definitions/AgentObject" }
  }
}
```

#### 4.1.2 Query

```json
{
  "title": "QueryMessage",
  "type": "object",
  "required": ["protocol_version", "query"],
  "properties": {
    "protocol_version": { "type": "string", "pattern": "^v\\d+\\.\\d+$" },
    "query": {
      "type": "object",
      "properties": {
        "capability": { "type": "string" },
        "agent_id": { "type": "string" }
      },
      "required": ["capability"]
    }
  }
}
```

#### 4.1.3 Response

```json
{
  "title": "ResponseMessage",
  "type": "object",
  "required": ["protocol_version", "agent", "matched"],
  "properties": {
    "protocol_version": { "type": "string", "pattern": "^v\\d+\\.\\d+$" },
    "agent": { "$ref": "#/definitions/Agent" },
    "matched": { "type": "boolean" }
  }
}
```

> **Canonical JSON** – The signature is computed over a sorted, whitespace-free JSON of the `agent` object (RFC 8785).  
> **Timestamp** – Optional `timestamp` field is allowed but *ignored* by receivers.

### 4.2 Signed Announcements

Agents MUST sign the canonical JSON of the `agent` field. The resulting Base64 signature is included as `signature`. If the sign-/verify chain is broken, the message is dropped.

---

## 5. Transport Layer

| Transport | Binding | Features |
|-----------|---------|----------|
| **UDP/mDNS** | `udp://255.255.255.255:5353` | No connection, low overhead, TTL |
| **HTTP/REST** | `https://registry.example.com/agents` | Full reliability, filtering, authentication |
| **WebSocket** | `wss://discovery.example.com/ws` | Streaming queries, long-lived session |
| **MCP Bus** | `/topic/discovery` | Event-driven, high throughput (optional) |

> **Transport Handlers** – Each implementation must expose the same high-level contract: `Announce(agent)`, `Query(query) -> [Response]`, `Subscribe(cb)`.

### 5.1 mDNS UDP

* Port: 5353 (RFC 6762).  
* Payload length: <= 512 bytes.  
* If the payload exceeds 512 bytes, the agent must split into multiple UDP packets following RFC 6762's fragmentation rules (optional; rare).

### 5.2 HTTP POST

* Content-Type: `application/json`.  
* Response: `200 OK` + optional JSON echo.  
* Idempotent: Multiple identical `Announcements` are coalesced by the registry.

### 5.3 WebSocket

* Path: `/ws`.  
* Subprotocol: `agent-discovery`.  
* Messages are the JSON types defined above, prefixed with a 1-byte type ID (`0x01 = ANNOUNCE`, `0x02 = QUERY`, `0x03 = RESPONSE`).  
* Heartbeat: every 30 s; on miss → close.

---

## 6. Security Model

| Layer | Mechanism | Notes |
|-------|-----------|-------|
| **Transport** | TLS 1.3 | Mutual authentication for HTTP/WS; mDNS is open (no auth). |
| **Message** | Digital Signature | Guarantees authenticity & integrity of the `agent` object. |
| **Allow-list** | Global ACL or per-service ACL in the registry | Only *whitelisted* `agent_id` values may register. |
| **Authentication Token** | JWT-bearer (scoped to `discover:write`, `discover:read`) | Applied on HTTP/WS endpoints. |
| **Trust Boundary** | `discovery` namespace | All non-discovery traffic is blocked by default. |

### 6.1 Signing Keys

* Each agent holds a long-term key pair (`private_key.pem`, `public_key.pem`).  
* The `public_key` is published in the `agent` object.  
* Registry and other agents MUST maintain a *trust store* of accepted keys (e.g., by `agent_id`).  
* Key rotation is performed by issuing a new key pair and re-announcing; old keys are retired after a grace period (default 24 h).  

**Acceptance Criteria**

1. All announcements with invalid signatures are discarded.  
2. Messages from unknown `agent_id`s are ignored unless they match an allow-list rule.  
3. Registry endpoints reject any request without a valid JWT bearer token or TLS client cert.

---

## 7. Implementation Guidance

Below are code snippets to illustrate typical integration patterns.

### 7.1 Python – Registry Registration (HTTP)

```python
import requests, json, jwt, time
from cryptography.hazmat.primitives import serialization

def load_key(path: str):
    with open(path, "rb") as f:
        key = serialization.load_pem_private_key(f.read(), password=None)
    return key

agent = {
    "agent_id": "3cde1b52-1fc2-4f7b-8d8c-5e3e6f5c9c1d",
    "capabilities": {"metrics":"2.0"},
    "public_key": "-----BEGIN PUBLIC KEY-----\n...",
    "transport": {"type":"tcp","endpoint":"10.0.0.12:9000"},
}

# Canonical JSON
payload = json.dumps(agent, separators=(",", ":"), sort_keys=True)

# Sign
key = load_key("/path/key.pem")
signature = base64.b64encode(key.sign(payload.encode(), padding=..., hash_alg=...)).decode()

announce_msg = {"protocol_version":"v1.0","agent":agent,"signature":signature}
jwt_token = jwt.encode({"scope":"discover:write"}, "secret", algorithm="HS256")

headers = {"Authorization":"Bearer "+jwt_token,
           "Content-Type":"application/json"}

resp = requests.post("https://registry.example.com/agents",
                     data=json.dumps(announce_msg),
                     headers=headers)
resp.raise_for_status()
```

### 7.2 Integration Pattern

1. **bootstrap**
   * Load identity (keys, `agent_id`).
   * Read config for registry endpoint, broadcast interface, heartbeat.
2. **announce_loop**
   * Emit `Announce` every `N` seconds over mDNS + registry.
3. **query_loop**
   * When a new capability is required (e.g., to send a task), send a `Query` over WebSocket or HTTP to get the target agent(s).  
   * Cache responses for 5 min; invalidate after TTL.
4. **subscribe**
   * Register a callback with the discovery module to react to announcements (e.g., load-balance, health-checks).

---

## 8. Compatibility & Version Negotiation

### 8.1 Protocol Version

* `protocol_version` is a string `v{major}.{minor}`.  
* Major = schema *structure*; minor = *semantic* changes.  
* Agents MUST ignore messages with an unsupported major version.  
* Agents MUST send back at least the major version they support if they *receive* a message with a higher major.  

### 8.2 Backward Compatibility Checklist

| Feature | 1.0 | 1.1 | 2.0 |
|---------|-----|-----|-----|
| `transport.type` | string | string | enum (`tcp\|ws\|unix`) |
| `signature` | required | optional | required |
| `capabilities` | simple map | map+versions | hierarchical map |
| `allow_list` | N/A | optional header | mandatory header |

> **Upgrade Path** – To move from 1.0 to 2.0, deploy a new registry that accepts both `v1.0` and `v2.0`. Agents with `v1.0` skip the `signature` verification but still advertise all capabilities.

### 8.3 Error Handling

| Error | Code | Message | Retry |
|-------|------|---------|-------|
| `400 Bad Request` | `ErrMalformedPayload` | JSON schema validation failed | Try again after 5 s |
| `401 Unauthorized` | `ErrAuth` | JWT or TLS cert mis-match | Try again after 30 s |
| `403 Forbidden` | `ErrForbidden` | Not in allow-list | Wait for ACL update |
| `409 Conflict` | `ErrDuplicateID` | `agent_id` already registered | Resolve ID collision |

---

## Acceptance Criteria (Cumulative)

| # | Criterion | Description | Validation |
|---|-----------|-------------|------------|
| 1 | **Announce message** | JSON schema matches; signature verifies; TTL <= 512 bytes | Unit tests + integration test on UDP and HTTP |
| 2 | **Query/Response** | Query finds existing agent(s); response contains accurate metadata | Functional test with two agents |
| 3 | **Transport abstraction** | One agent can be discovered using mDNS, another only via registry **and** both must interoperate | Cross-transport test |
| 4 | **Security** | Agents with revoked certificates cannot register; signed messages only from trusted agents are accepted | Security scan, mock revoked cert |
| 5 | **Version negotiation** | Agent v1.0 receives v1.1 message → logs warning and continues | Mock version mismatch |
| 6 | **Allow-list enforcement** | Non-whitelisted `agent_id` receives 403 | ACL test |
| 7 | **Heartbeat & stale cleanup** | After TTL without announcement, agent disappears from registry | Timing test |

All unit tests must pass 100% on the CI pipeline. The reference implementation must be production-ready with proper error handling.

---

## References

* RFC 6762 – Dynamic Host Name Configuration (mDNS)  
* RFC 8785 – Canonical JSON  
* RFC 7519 – JSON Web Token (JWT)  
* OPC UA (for transport security guidelines, TLS 1.3)

