# Multi-Agent System Configuration

## Overview
AgentsMCP now implements comprehensive multi-agent workflow best practices with three core systems:

1. **JSON Envelope System** - Stateless request/response normalization
2. **HITL Security System** - Human-In-The-Loop approval workflow with cryptographic security
3. **OpenTelemetry Observability** - Distributed tracing, metrics, and correlation tracking

## Installation

```bash
# Install with all multi-agent features
pip install -e .[security,metrics]

# Or install specific features
pip install -e .[security]  # HITL + JWT auth
pip install -e .[metrics]   # OpenTelemetry + Prometheus
```

## Environment Variables

### OpenTelemetry Observability
```bash
# Service identification
OTEL_SERVICE_NAME=agentsmcp                    # Service name in traces
OTEL_SERVICE_VERSION=1.0.0                     # Service version

# Jaeger tracing (choose one)
OTEL_EXPORTER_JAEGER_ENDPOINT=http://localhost:14268/api/traces
# OR OTLP tracing
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_INSECURE=true              # Allow insecure OTLP
```

### HITL Security System
```bash
# Core settings
AGENTSMCP_HITL_ENABLED=true                    # Enable HITL approvals
AGENTSMCP_HITL_REQUIRED_OPS=delete_project,modify_critical  # Comma-separated operations
AGENTSMCP_HITL_DEFAULT_TIMEOUT_MINS=30         # Approval timeout
AGENTSMCP_HITL_DEFAULT_ACTION=reject           # reject|approve on timeout

# JWT security
AGENTSMCP_JWT_SECRET=your-secure-secret-key    # JWT signing secret
AGENTSMCP_JWT_ALGORITHM=HS256                  # JWT algorithm
AGENTSMCP_JWT_EXPIRY_MINS=60                  # JWT token expiry

# User roles (comma-separated usernames)
AGENTSMCP_HITL_ADMIN_USERS=admin,alice         # Admin users
AGENTSMCP_HITL_OPERATOR_USERS=admin,alice,bob  # Operator users

# Rate limiting
AGENTSMCP_HITL_DECISION_RATELIMIT_PER_MIN=30   # Decisions per minute per user
```

### JSON Envelope System
```bash
# No configuration required - automatically detects client preferences
# Clients can request envelopes via:
# - Header: X-Envelope: 1
# - Query param: ?envelope=1  
# - Accept: application/vnd.agentsmcp.envelope+json
```

## Usage Examples

### 1. JSON Envelope System
```python
from agentsmcp.models import EnvelopeParser, EnvelopeStatus

# Server-side: Handle both raw and enveloped requests
def handle_request(request_body: dict):
    payload, meta = EnvelopeParser.parse_body(request_body)
    
    # Process payload...
    result = process_job(payload)
    
    # Return envelope if client wants it
    if EnvelopeParser.wants_envelope(headers=request.headers):
        return EnvelopeParser.build_envelope(
            payload=result,
            status=EnvelopeStatus.SUCCESS,
            request_id=meta.request_id
        )
    return result  # Legacy raw response
```

### 2. HITL Security Decorator
```python
from agentsmcp.security.hitl import hitl_required

@hitl_required(operation="delete_project", risk_level="high")
async def delete_project(project_id: str, _current_user: str = "system"):
    """Requires human approval before execution."""
    # This code only runs after approval
    return await actually_delete_project(project_id)
```

### 3. OpenTelemetry Instrumentation
```python
from agentsmcp.observability import instrument_agent_operation, instrument_agent_event

@instrument_agent_operation("process_request")
async def process_request(data):
    instrument_agent_event("request_received", user_id="123", operation="spawn")
    # Automatically traced with correlation ID
    return await do_work(data)
```

## Architecture Benefits

✅ **Stateless Design** - No shared state between agents
✅ **Backward Compatible** - Existing clients continue to work  
✅ **Security First** - Cryptographic approval tokens with replay protection
✅ **Observable** - End-to-end tracing with correlation IDs
✅ **Production Ready** - Rate limiting, graceful fallbacks, structured errors
✅ **Standards Based** - OpenTelemetry, JWT, RFC3339 timestamps

## Monitoring Endpoints

- `GET /metrics` - Prometheus metrics
- `GET /hitl/` - HITL approval UI (requires auth)
- `GET /hitl/queue` - Approval queue API

## Development

```bash
# Run tests
pytest tests/

# Install dev dependencies  
pip install -e .[dev]

# Lint and format
ruff check src/
ruff format src/
```

## Next Steps

1. Deploy with monitoring stack (Prometheus + Jaeger)
2. Configure user roles and approval workflows
3. Set up alerting on approval queue depth
4. Implement custom instrumentation for domain-specific events