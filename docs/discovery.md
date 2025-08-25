# AgentsMCP – Discovery Protocol Specification  
*(Version 1.0 – 2025‑08‑25)*  

> **Purpose** – Define a production‑ready, extensible discovery and coordination layer for the **AgentsMCP** multi‑agent orchestration platform. The protocol enables agents (running in Docker, Firecracker, or native runtimes) to **register**, **advertise capabilities**, **discover peers**, **balance load**, **monitor health**, and **co‑ordinate** in the presence of network partitions, while enforcing a strict security model.  

> **Scope** – This document covers:
1. **Agent registration & service discovery**  
2. **Capability advertisement & matching**  
3. **Load‑balancing & health monitoring**  
4. **Peer‑to‑peer coordination**  
5. **Network partition handling & consensus**  
6. **Security (auth‑z/auth‑n)**  
7. **Public JSON‑based API** (with JSON‑Schema)  
8. **Implementation guidance** (reference architecture, data structures, Docker/Firecracker integration, delegation subsystem hooks)  

---  

## 1. Architecture Overview  

```mermaid
graph LR
    subgraph "Cluster"
        A[Load‑Balancer (LB)] -->|REST/WS| DS[Discovery Service]
        DS <-->|gRPC| DS2[Discovery Service (replica)]
        subgraph "Agents"
            agent1[Agent #1] 
            agent2[Agent #2] 
            agent3[Agent #3] 
        end
        agent1 <-->|HTTP/WS| DS
        agent2 <-->|HTTP/WS| DS
        agent3 <-->|HTTP/WS| DS
    end
    subgraph "External"
        client[User / Delegation Subsystem] -->|API| LB
    end
```

* **Discovery Service (DS)** – A stateless front‑end that forwards requests to a **Raft‑based replicated state machine** (the "Discovery Store").  
* **Discovery Store** – Holds agent registrations, capability indexes, health metrics, and load‑balancing tokens. Replicated via **Raft** for strong consistency.  
* **Load‑Balancer** – Terminates TLS, performs client authentication (mTLS) and forwards to any DS replica.  
* **Agents** – Self‑contained containers (Docker or Firecracker). On start‑up they **register** themselves via the **Agent Registration API** and subsequently interact with the DS for discovery, health pings, and peer coordination.  

---  

## 2. Core Concepts & Terminology  

| Term | Definition |
|------|------------|
| **AgentID** | Globally unique identifier (`uuidv4`). Assigned by the DS at registration. |
| **ServiceName** | Logical name of the functionality an agent provides (e.g., `data‑ingest`, `model‑trainer`). |
| **Capability** | Structured description of an agent's functional and non‑functional attributes (CPU, GPU, tags, version, etc.). |
| **Endpoint** | Network address (URL) on which the agent can be reached (`host:port`). |
| **HealthCheck** | Periodic heartbeat (`/healthz`) sent by an agent; includes load metrics & status. |
| **LoadToken** | A lightweight token (JWT) issued by DS used for **client‑side request routing** and **rate‑limiting**. |
| **Delegation Token** | Opaque token created by the **delegation subsystem** that authorises a request on behalf of a user or parent service. |
| **Consensus Group** | The set of DS replicas participating in Raft. |
| **Partition** | Temporary loss of connectivity between a subset of agents or between agents and DS. |

---  

## 3. Agent Registration & Service Discovery  

### 3.1. Registration Workflow  

1. **Bootstrap** – Agent obtains a **bootstrap token** from the delegation subsystem (pre‑shared secret or signed JWT).  
2. **TLS Handshake** – Mutual TLS (mTLS) is established with the DS front‑end (certificate‑based authentication).  
3. **POST `/v1/agents/register`** – Agent sends a `RegistrationRequest`. The DS validates the bootstrap token, assigns an `AgentID`, stores the record in the Raft log, and responds with a `RegistrationResponse`.  

#### 3.1.1. RegistrationRequest JSON Schema  

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://agentsmcp.io/schemas/registration-request.json",
  "title": "Agent Registration Request",
  "type": "object",
  "required": ["bootstrapToken", "serviceName", "endpoint", "capabilities"],
  "properties": {
    "bootstrapToken": {
      "type": "string",
      "description": "Signed JWT issued by delegation subsystem"
    },
    "serviceName": {
      "type": "string",
      "pattern": "^[a-z0-9\\-]{3,64}$",
      "description": "Logical service name"
    },
    "endpoint": {
      "type": "string",
      "format": "uri",
      "description": "Agent reachable URL (e.g., https://10.0.1.22:8443)"
    },
    "capabilities": {
      "$ref": "capability.json"
    },
    "metadata": {
      "type": "object",
      "additionalProperties": { "type": "string" },
      "description": "Optional free‑form key/value pairs"
    }
  },
  "additionalProperties": false
}
```

> **Capability schema** (`capability.json`) is defined in § 3.2.

#### 3.1.2. RegistrationResponse  

```json
{
  "agentId": "550e8400-e29b-41d4-a716-446655440000",
  "assignedToken": "<JWT>",   // short‑lived (5 min) for subsequent health pings
  "ttlSeconds": 86400,
  "message": "registration successful"
}
```

### 3.2. Capability Advertisement & Matching  

Agents describe their **capabilities** using a **typed, versioned schema**. This supports rich matching (e.g., "GPU ≥ 2, CUDA = 11.4, region = us‑east‑1").  

#### 3.2.1. Capability JSON Schema (`capability.json`)  

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://agentsmcp.io/schemas/capability.json",
  "title": "Capability Description",
  "type": "object",
  "required": ["version", "resources", "tags"],
  "properties": {
    "version": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$",
      "description": "Semantic version of the capability model"
    },
    "resources": {
      "type": "object",
      "required": ["cpu", "memory"],
      "properties": {
        "cpu": {
          "type": "number",
          "minimum": 0.1,
          "description": "Number of virtual CPUs"
        },
        "memory": {
          "type": "integer",
          "minimum": 64,
          "description": "MiB of RAM"
        },
        "gpu": {
          "type": "integer",
          "minimum": 0,
          "description": "Number of GPUs"
        },
        "disk": {
          "type": "integer",
          "minimum": 0,
          "description": "MiB of local storage"
        }
      },
      "additionalProperties": false
    },
    "tags": {
      "type": "array",
      "items": { "type": "string", "pattern": "^[a-z0-9\\-]{2,32}$" },
      "description": "Free‑form labels for categorical matching"
    },
    "properties": {
      "type": "object",
      "additionalProperties": {
        "type": ["string", "number", "boolean"]
      },
      "description": "Arbitrary key/value pairs (e.g., cudaVersion, region)"
    }
  },
  "additionalProperties": false
}
```

### 3.3. Service Discovery API  

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/agents/{serviceName}` | Returns **all live agents** that announced `serviceName`. Supports optional query filters (`?tag=GPU&minCpu=4`). |
| `POST` | `/v1/agents/{serviceName}/match` | Submit a **CapabilityMatcher** (see schema) and receive the **best‑fit** agent(s). |
| `GET` | `/v1/agents/{agentId}` | Retrieve a single agent's registration record. |
| `DELETE` | `/v1/agents/{agentId}` | Deregister (used by graceful shutdown). |

#### 3.3.1. CapabilityMatcher schema  

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://agentsmcp.io/schemas/matcher.json",
  "title": "Capability Matcher",
  "type": "object",
  "properties": {
    "minResources": {
      "$ref": "capability.json#/properties/resources"
    },
    "requiredTags": {
      "type": "array",
      "items": { "type": "string" }
    },
    "properties": {
      "type": "object",
      "additionalProperties": { "type": ["string","number","boolean"] }
    },
    "strategy": {
      "type": "string",
      "enum": ["least‑loaded","most‑available","round‑robin"],
      "default": "least‑loaded"
    }
  },
  "required": ["minResources"]
}
```

**Response** – List of `AgentRecord` objects (see § 5).  

---  

## 4. Load Balancing & Health Monitoring  

### 4.1. Health Ping  

*Agents* send a **heartbeat** every **`heartbeatInterval` seconds** (default 15 s) to **POST `/v1/agents/{agentId}/heartbeat`**.  

Payload (`HealthPing`):

```json
{
  "loadToken": "<LoadToken JWT>",
  "status": "healthy",
  "metrics": {
    "cpuUtil": 0.62,
    "memUtil": 0.71,
    "activeRequests": 12,
    "queueDepth": 3
  },
  "timestamp": "2025-08-25T12:34:56.789Z"
}
```

* The DS validates the `loadToken` (issued at registration) and stores the metrics in an in‑memory time‑series table.  
* Load‑balancing algorithms use the **latest metric snapshot** (TTL 2 × `heartbeatInterval`).  

### 4.2. Load Token  

A **short‑lived JWT** (`exp` ≤ 5 min) containing:
```json
{
  "sub": "agentId",
  "iss": "agentsmcp-discovery",
  "capabilitiesHash": "<sha256 of capability JSON>",
  "lbStrategy": "least-loaded"
}
```
Clients (or other agents) can request a token for a specific service via **GET `/v1/load-token?serviceName=…`**. The DS signs the token with the platform's private key, allowing downstream services to **verify freshness** without a round‑trip to the DS.

### 4.3. Automatic Deregistration  

If **no heartbeat** is received within **`heartbeatTTL = 3 × heartbeatInterval`**, the DS marks the agent **`unreachable`** and removes it from the service index after **`gracePeriod=30s`** to allow for transient network glitches.

---  

## 5. Peer‑to‑Peer Coordination Protocol  

While the DS provides **centralised discovery**, many use‑cases require **direct agent‑to‑agent interaction** (e.g., distributed training). The protocol defines a **lightweight, bidirectional WebSocket channel** with a tiny message envelope.  

### 5.1. Connection Establishment  

1. **Agent A** discovers **Agent B** via `/v1/agents/{serviceName}`.  
2. Agent A opens a **mutual‑TLS WebSocket** to `Agent B.endpoint/ws/coord`.  
3. Both sides exchange a **CoordinationHandshake**:

```json
{
  "srcAgentId": "550e8400‑e29b‑41d4‑a716‑446655440000",
  "dstAgentId": "e2c0b6d1‑f2a9‑4d9f‑ae63‑7e6e0d5bc9f2",
  "capabilitiesHash": "<sha256>",
  "sessionId": "<uuidv4>",
  "timestamp": "2025-08-25T12:35:10Z"
}
```

Server validates the handshake (capability hash must match stored record).  

### 5.2. Message Envelope  

All coordination messages are wrapped in:

```json
{
  "sessionId": "<uuidv4>",
  "seq": 123,
  "type": "command|event|ack",
  "payload": { … }
}
```

* **`seq`** – Monotonically increasing per‑session identifier (used for ordering & retransmission).  
* **`type`** – Determines payload schema (`command.json`, `event.json`).  

### 5.3. Reliable Delivery  

* **ACK** messages (`type: "ack"`) confirm receipt of a given `seq`.  
* Retransmission logic (exponential back‑off) is implemented in the agent SDK.  

### 5.4. Coordination Use‑Cases  

| Use‑Case | Message Flow |
|----------|--------------|
| **Task delegation** | A → B (`command` = `executeTask`), B replies with `event` = `taskStarted`, later sends `event` = `taskCompleted`. |
| **State sync** | Periodic `event` messages containing diff‑patches of local state. |
| **Leader election (local)** | Agents broadcast `command` = `nominateLeader`; winner replies with `event` = `leaderElected`. |

---  

## 6. Network Partition Handling & Consensus  

### 6.1. Partition Detection  

* Each DS replica monitors **heartbeat** from the others (Raft's own leader election).  
* Agents that lose connectivity to any DS node will **fallback** to the last known **peer list** and continue operating in **degraded mode** (no new registration, but can still discover peers via cached data).  

### 6.2. Consensus Guarantees  

| Property | Guarantees |
|----------|------------|
| **Safety** | Raft ensures at most one leader; all committed registrations are immutable. |
| **Liveness** | As long as a majority of DS replicas are reachable, the cluster makes progress. |
| **Read‑Your‑Writes** | Agents receive **linearizable** reads via the **`/v1/agents/{serviceName}`** endpoint that forwards to the leader (or any replica with **Read‑Index**). |

### 6.3. Split‑Brain Mitigation  

* **Quorum‑based writes** – Any write (registration, deregistration, capability update) requires **≥ ⌈N/2⌉ + 1** acknowledgments.  
* **Automatic Re‑join** – When a partition heals, the DS replicates missing log entries using Raft's `AppendEntries`. Agents that were deregistered due to missed heartbeats are **re‑registered** automatically if they are still alive (they will send a fresh registration after reconnection).  

---  

## 7. Security Model  

### 7.1. Threat Model  

| Threat | Mitigation |
|--------|------------|
| **Impersonation** | Mutual TLS with per‑node client certificates. |
| **Replay attacks** | All tokens (`bootstrapToken`, `loadToken`, `sessionId`) contain short TTL and are signed with platform private key. |
| **Unauthorized discovery** | Service‑level ACLs enforced on GET `/v1/agents/{serviceName}`; only principals with `discover:<service>` permission can list agents. |
| **Privilege escalation** | Capability hashes are bound to the agent's certificate – any modification invalidates the token. |
| **DoS via bogus registrations** | Rate‑limiting per IP + registration throttling (max 5 regs/min per certificate). |

### 7.2. Authentication  

1. **Bootstrap Token** – JWT signed by **Delegation Subsystem** (contains `sub` = requester, `aud` = `agentsmcp-discovery`). Valid for **5 min**.  
2. **mTLS** – Both client (agent) and server present X.509 certificates signed by the **AgentsMCP PKI**. Certificate CN encodes the **agent's principal** (`agent:<serviceName>`).  

### 7.3. Authorization  

* **RBAC** model stored in the **Policy Service** (outside the scope of this spec).  
* Each request includes the **Authorization** header (`Bearer <JWT>`). The DS validates against the Policy Service via a **gRPC** call.  

### 7.4. Confidentiality & Integrity  

* All traffic is **TLS 1.3** (with forward secrecy).  
* Payloads are signed (JWT) and optionally **encrypted** using **AEAD** (if `enc` claim is present).  

### 7.5. Auditing  

* Every registration, deregistration, capability update, and health ping is **logged** with immutable hash chain (log entry includes `prevHash`).  
* Log entries are replicated via Raft and can be exported to an external **SIEM**.  

---  

## 8. API Specification  

All endpoints are **versioned** under `/v1`. The API uses **JSON** over **HTTPS**.  

### 8.1. Common Response Envelope  

```json
{
  "requestId": "<uuidv4>",
  "timestamp": "2025-08-25T12:40:00Z",
  "status": "ok|error",
  "payload": { … },
  "error": {
    "code": "string",
    "message": "string",
    "details": { … }
  }
}
```

### 8.2. Endpoint Summary  

| Method | Path | Request Schema | Response Payload |
|--------|------|----------------|------------------|
| `POST` | `/v1/agents/register` | `registration-request.json` | `{ agentId, assignedToken, ttlSeconds }` |
| `DELETE` | `/v1/agents/{agentId}` | *none* | `{ message }` |
| `GET` | `/v1/agents/{serviceName}` | *query params* (`tag`, `minCpu`, …) | `AgentRecord[]` |
| `POST` | `/v1/agents/{serviceName}/match` | `matcher.json` | `AgentRecord[]` |
| `POST` | `/v1/agents/{agentId}/heartbeat` | `health-ping.json` | `{ acknowledged:true }` |
| `GET` | `/v1/load-token?serviceName=…` | *none* | `{ token }` |
| `GET` | `/v1/policy/evaluate` | *internal* (gRPC) | *policy verdict* |

#### 8.2.1. `AgentRecord` schema  

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://agentsmcp.io/schemas/agent-record.json",
  "title": "Agent Record",
  "type": "object",
  "required": ["agentId", "serviceName", "endpoint", "capabilities", "status"],
  "properties": {
    "agentId": { "type": "string", "format": "uuid" },
    "serviceName": { "type": "string" },
    "endpoint": { "type": "string", "format": "uri" },
    "capabilities": { "$ref": "capability.json" },
    "status": { "type": "string", "enum": ["healthy","degraded","unreachable"] },
    "lastHeartbeat": { "type": "string", "format": "date-time" },
    "metrics": { "$ref": "health-ping.json#/properties/metrics" },
    "metadata": { "type": "object", "additionalProperties": { "type": "string" } }
  },
  "additionalProperties": false
}
```

#### 8.2.2. `health-ping.json`  

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://agentsmcp.io/schemas/health-ping.json",
  "title": "Health Ping",
  "type": "object",
  "required": ["loadToken", "status", "metrics", "timestamp"],
  "properties": {
    "loadToken": { "type": "string" },
    "status": { "type": "string", "enum": ["healthy","degraded","failed"] },
    "metrics": {
      "type": "object",
      "properties": {
        "cpuUtil": { "type": "number", "minimum": 0, "maximum": 1 },
        "memUtil": { "type": "number", "minimum": 0, "maximum": 1 },
        "activeRequests": { "type": "integer", "minimum": 0 },
        "queueDepth": { "type": "integer", "minimum": 0 }
      },
      "required": ["cpuUtil","memUtil"],
      "additionalProperties": false
    },
    "timestamp": { "type": "string", "format": "date-time" }
  },
  "additionalProperties": false
}
```

---  

## 9. Implementation Guidance  

### 9.1. Reference Architecture  

| Component | Language/Framework | Recommended Libraries |
|-----------|--------------------|-----------------------|
| **Discovery Service (API)** | Go 1.22 | `gin` (HTTP), `grpc-go` (Raft), `go-jwt`, `tls-config` |
| **Discovery Store (Raft)** | Rust | `raft-rs`, `sled` for log persistence |
| **Agent SDK** | Python 3.12 / Go / Rust | `requests` (HTTPS), `websockets`, `pyjwt`, `docker-py` |
| **Policy Service** | Java 21 | `grpc-java`, `OPA` (Open Policy Agent) as policy engine |
| **Health Metrics Store** | Prometheus (remote‑write) & in‑memory cache | `prometheus-client` |

### 9.2. Docker / Firecracker Integration  

* **Docker** – Agents embed a **side‑car** that runs the **Agent SDK** and performs registration on container start (`ENTRYPOINT`).  
* **Firecracker** – The micro‑VM boots a minimal rootfs containing the SDK. The `init` process calls the registration endpoint via a **VSOCK** tunnel to the host's DS front‑end (exposed on 127.0.0.1:8443).  

Both runtimes should mount a **read‑only copy** of the platform's **PKI certificates** (`/etc/agentsmcp/certs/`).  

### 9.3. Delegation Subsystem Hook  

The **Delegation Service** issues bootstrap JWTs using the platform's **Authorization Server** (OAuth 2.0).  

```go
// Pseudo‑code for token issuance
func IssueBootstrapToken(principal string, ttl time.Duration) (string, error) {
    claims := jwt.MapClaims{
        "sub": principal,
        "aud": "agentsmcp-discovery",
        "exp": time.Now().Add(ttl).Unix(),
        "iat": time.Now().Unix(),
        "jti": uuid.New().String(),
    }
    return signer.Sign(claims)
}
```

Agents retrieve the token via the **delegation API** (`GET /v1/delegation/bootstrap?service=…`) before starting.  

### 9.4. Scaling Considerations  

| Dimension | Scaling Strategy |
|-----------|------------------|
| **Number of agents** | Shard the **service index** by `serviceName` using **consistent hashing** across DS replicas. |
| **Health‑ping load** | Use **gossip** among DS replicas to propagate metrics, reducing read‑only RPCs from the leader. |
| **Discovery queries** | Cache read‑only service lists in **Redis** or **memcached** with TTL = 2 × heartbeat interval; cache is invalidated on Raft commit events. |
| **TLS termination** | Deploy a **sidecar Envoy** per DS node to offload TLS, enabling HTTP/2 multiplexing for WebSocket connections. |

### 9.5. Testing & Validation  

1. **Unit tests** – JSON‑schema validation, JWT verification, Raft state machine transitions.  
2. **Integration tests** – Spin up a 3‑node DS cluster (Docker‑compose) with simulated agents; verify registration, heartbeat loss, and partition recovery.  
3. **Chaos engineering** – Use **Chaos Mesh** to drop network packets between a subset of agents and DS, confirm graceful degradation and automatic re‑join.  
4. **Security audits** – Run **OWASP ZAP** against the API; perform **certificate pinning** checks.  

### 9.6. Operational Monitoring  

| Metric | Description | Exporter |
|--------|-------------|----------|
| `ds_registration_total` | Cumulative count of successful registrations | Prometheus |
| `ds_heartbeat_lost_total` | Number of missed heartbeats per service | Prometheus |
| `ds_raft_leader_changes_total` | Raft leader transitions (indicative of partitions) | Prometheus |
| `agent_active_sessions` | Number of open P2P WebSocket sessions | Agent SDK -> Prometheus |
| `load_token_issued_total` | Count of load‑token issuances | Prometheus |
| `auth_failed_total` | Authentication failures (mTLS or JWT) | Prometheus |

Dashboards should show **service health heatmaps**, **load distribution**, and **partition alerts** (Raft leader change rate > 1/min).  

---  

## 10. Revision History  

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025‑08‑25 | **AgentsMCP Architecture Team** | Initial full specification (this document). |

---  

*Prepared by the AgentsMCP Architecture Team.*  

*All components described herein are subject to the AgentsMCP Open Source License (Apache 2.0) and must be used in conjunction with the platform's PKI and delegation services.*