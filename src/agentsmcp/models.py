"""Core data models used by AgentsMCP.

This module implements a production-ready, stateless JSON envelope standard
to normalize API request and response shapes across the service.

It now supports AGENTS.md v2 structured task decomposition via versioned envelopes:
- TaskEnvelope v1: structured task descriptions for agent coordination
- ResultEnvelope v1: structured task results with artifacts and metrics

Target envelope format:

    {
      "status": "success|error|pending", 
      "meta": {
        "timestamp": "2025-01-27T10:30:00Z",
        "request_id": "uuid",
        "version": "1.0",
        "source": "agent_id",
        "type": "generic|task|result"
      },
      "payload": {},
      "errors": []
    }

The envelope is optional on inputs for backwards compatibility: handlers should
gracefully accept either a raw payload or a full envelope. For outputs, callers
can opt-in via request metadata (header or query param) — see
``EnvelopeParser.wants_envelope``.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import BaseModel, Field, ValidationError
import uuid


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class JobStatus:
    job_id: str
    state: JobState
    output: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at


#
# Stateless JSON Envelope models and utilities
#


class EnvelopeStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


class EnvelopeMeta(BaseModel):
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Server-side timestamp (UTC)",
    )
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    version: str = Field(
        default="1.0",
        description="Envelope schema version for forwards compatibility",
    )
    source: Optional[str] = Field(
        default=None, description="Originating agent or service identifier"
    )
    type: Optional[str] = Field(
        default="generic", description="Envelope type (generic|task|result)"
    )


class EnvelopeError(BaseModel):
    code: Optional[str] = Field(default=None, description="Machine readable code")
    message: str = Field(description="Human readable error message")
    details: Optional[dict] = Field(
        default=None, description="Optional structured error details"
    )


class Envelope(BaseModel):
    status: EnvelopeStatus
    meta: EnvelopeMeta
    payload: Any | None = None
    errors: list[EnvelopeError] = Field(default_factory=list)


class EnvelopeParser:
    """Parser and builder for the standardized stateless JSON Envelope.

    This helper provides:
    - tolerant request parsing (accepts both enveloped and raw payloads)
    - envelope construction for responses
    - minimal convenience validation against Pydantic payload models
    """

    DEFAULT_SOURCE = "agentsmcp-server"

    @staticmethod
    def build_envelope(
        payload: Any | None,
        *,
        status: EnvelopeStatus = EnvelopeStatus.SUCCESS,
        meta: EnvelopeMeta | None = None,
        errors: list[EnvelopeError] | None = None,
        request_id: Optional[str] = None,
        source: Optional[str] = None,
        version: str = "1.0",
    ) -> Envelope:
        m = meta or EnvelopeMeta(
            request_id=request_id or str(uuid.uuid4()),
            source=source or EnvelopeParser.DEFAULT_SOURCE,
            version=version,
        )
        return Envelope(
            status=status,
            meta=m,
            payload=payload,
            errors=errors or [],
        )

    @staticmethod
    def parse_body(body: Any) -> tuple[Any, EnvelopeMeta]:
        """Parse a JSON body into (payload, meta).

        Accepts either a full envelope or a raw payload dict/list. If raw, a
        synthetic meta is generated.
        """
        if isinstance(body, dict) and {"status", "meta", "payload"}.issubset(
            body.keys()
        ):
            env = Envelope.model_validate(body)
            return env.payload, env.meta
        # Raw payload; construct synthetic meta
        return body, EnvelopeMeta()

    @staticmethod
    def validate_payload(payload: Any, model: type[BaseModel] | None) -> Any:
        if model is None:
            return payload
        return model.model_validate(payload)

    @staticmethod
    def wants_envelope(
        headers: Mapping[str, str] | None = None,
        query_params: Mapping[str, str] | None = None,
    ) -> bool:
        """Determine whether the caller requested an enveloped response.

        Heuristics (any one enables envelope):
        - Header ``X-Envelope: 1|true|yes``
        - Query param ``envelope=1|true|yes``
        - Accept header includes vendor type ``application/vnd.agentsmcp.envelope+json``

        Returns False when not present to maintain backward compatibility.
        """
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        q = {k.lower(): v for k, v in (query_params or {}).items()}

        def _truthy(v: Optional[str]) -> bool:
            return str(v).strip().lower() in {"1", "true", "yes", "on"}

        if _truthy(headers.get("x-envelope")):
            return True
        if _truthy(q.get("envelope")):
            return True
        accept = headers.get("accept", "")
        if "application/vnd.agentsmcp.envelope+json" in accept:
            return True
        return False

    @staticmethod
    def coerce_timestamp_to_z(dt: datetime) -> str:
        """Return an RFC3339/ISO8601 timestamp with trailing 'Z' for UTC.

        Pydantic serializes timezone-aware UTC as ``+00:00``; this helper emits
        the canonical ``Z`` suffix many clients expect. Safe for serialization
        and logs. Not used automatically to avoid surprising consumers.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------
# AGENTS.md v2: Versioned Task/Result Envelopes for Multi-Agent Coordination
# ----------------------------------------------------------------------


class TaskEnvelopeV1(BaseModel):
    """
    Structured task description following AGENTS.md v2 specification.
    Used for decomposing complex tasks across multiple agents.
    """

    objective: str = Field(..., description="High-level goal of the task")
    bounded_context: Optional[str] = Field(
        None, description="Context that limits the task scope"
    )
    inputs: Optional[Dict[str, Any]] = Field(None, description="Input data for the task")
    output_schema: Optional[Dict[str, Any]] = Field(
        None, description="Expected output shape (JSON Schema format)"
    )
    constraints: Optional[List[str]] = Field(
        None, description="Business/logical constraints"
    )
    routing: Optional[Dict[str, Any]] = Field(
        None, description="Agent assignment and model routing hints"
    )
    telemetry: Optional[Dict[str, Any]] = Field(
        None, description="Monitoring and tracing information"
    )


class ResultEnvelopeV1(BaseModel):
    """
    Structured task result following AGENTS.md v2 specification.
    Captures artifacts, metrics, and execution details.
    """

    status: EnvelopeStatus = Field(..., description="Task execution status")
    artifacts: Optional[Dict[str, Any]] = Field(
        None, description="Generated files, code, or data artifacts"
    )
    diffs: Optional[Dict[str, Any]] = Field(
        None, description="Changes from previous artifact versions"
    )
    metrics: Optional[Dict[str, Any]] = Field(
        None, description="Performance and quality metrics"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Result confidence score (0-1)"
    )
    notes: Optional[str] = Field(None, description="Human-readable annotations")


# ----------------------------------------------------------------------
# Enhanced EnvelopeParser with Versioned Envelope Support
# ----------------------------------------------------------------------


class EnvelopeParserError(Exception):
    """Raised when envelope parsing or validation fails."""


def to_task_envelope(envelope: Envelope) -> TaskEnvelopeV1:
    """Convert generic envelope to structured TaskEnvelope v1."""
    if envelope.meta.type != "task":
        raise EnvelopeParserError(f"Expected task envelope, got: {envelope.meta.type}")
    
    try:
        return TaskEnvelopeV1.model_validate(envelope.payload)
    except ValidationError as e:
        raise EnvelopeParserError(f"TaskEnvelope validation failed: {e}") from e


def to_result_envelope(envelope: Envelope) -> ResultEnvelopeV1:
    """Convert generic envelope to structured ResultEnvelope v1.""" 
    if envelope.meta.type != "result":
        raise EnvelopeParserError(f"Expected result envelope, got: {envelope.meta.type}")
    
    try:
        return ResultEnvelopeV1.model_validate(envelope.payload)
    except ValidationError as e:
        raise EnvelopeParserError(f"ResultEnvelope validation failed: {e}") from e


def build_task_envelope(
    task: TaskEnvelopeV1,
    *,
    request_id: Optional[str] = None,
    source: Optional[str] = None
) -> Envelope:
    """Build a complete envelope containing a TaskEnvelope v1."""
    meta = EnvelopeMeta(
        request_id=request_id or str(uuid.uuid4()),
        source=source or EnvelopeParser.DEFAULT_SOURCE,
        type="task",
        version="1.0"
    )
    
    return Envelope(
        status=EnvelopeStatus.PENDING,
        meta=meta,
        payload=task.model_dump(),
        errors=[]
    )


def build_result_envelope(
    result: ResultEnvelopeV1,
    *,
    request_id: Optional[str] = None,  
    source: Optional[str] = None
) -> Envelope:
    """Build a complete envelope containing a ResultEnvelope v1."""
    meta = EnvelopeMeta(
        request_id=request_id or str(uuid.uuid4()),
        source=source or EnvelopeParser.DEFAULT_SOURCE,
        type="result",
        version="1.0"
    )
    
    return Envelope(
        status=result.status,
        meta=meta,
        payload=result.model_dump(),
        errors=[]
    )
