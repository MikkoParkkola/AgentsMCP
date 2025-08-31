"""Base role interfaces and envelope models.

Roles are designed as stateless pure functions over Pydantic models. Each role
must implement ``apply(task: TaskEnvelope) -> ResultEnvelope``. No IO, global
state, or network calls are performed here; integration with AgentsMCP's
orchestrator can choose an underlying agent/model to execute follow-up work.

This module defines:
- RoleName: canonical role identifiers
- TaskEnvelope: normalized task input
- ResultEnvelope: normalized role output
- Role: base class providing helpers and composition hooks
- Behavior mixins for reuse across roles
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..models import EnvelopeMeta, EnvelopeStatus, TaskEnvelopeV1, ResultEnvelopeV1

# For type hints only; avoid heavy imports at module import time
if True:  # type: ignore[truthy-bool]
    from ..agent_manager import AgentManager  # noqa: F401


class RoleName(str, Enum):
    ARCHITECT = "architect"
    CODER = "coder"
    QA = "qa"
    MERGE_BOT = "merge_bot"
    DOCS = "docs"
    PROCESS_COACH = "process_coach"
    METRICS_COLLECTOR = "metrics_collector"
    BUSINESS_ANALYST = "business_analyst"
    BACKEND_ENGINEER = "backend_engineer"
    WEB_FRONTEND_ENGINEER = "web_frontend_engineer"
    API_ENGINEER = "api_engineer"
    TUI_FRONTEND_ENGINEER = "tui_frontend_engineer"
    BACKEND_QA_ENGINEER = "backend_qa_engineer"
    WEB_FRONTEND_QA_ENGINEER = "web_frontend_qa_engineer"
    TUI_FRONTEND_QA_ENGINEER = "tui_frontend_qa_engineer"
    CHIEF_QA_ENGINEER = "chief_qa_engineer"
    IT_LAWYER = "it_lawyer"
    MARKETING_MANAGER = "marketing_manager"
    CI_CD_ENGINEER = "ci_cd_engineer"
    DEV_TOOLING_ENGINEER = "dev_tooling_engineer"
    DATA_ANALYST = "data_analyst"
    DATA_SCIENTIST = "data_scientist"
    ML_SCIENTIST = "ml_scientist"
    ML_ENGINEER = "ml_engineer"


class ModelAssignment(BaseModel):
    """Model/agent selection hint for a role.

    This is advisory metadata used by orchestrators. It maps cleanly to
    AgentsMCP's configured agent types: "claude", "codex", and "ollama".
    """

    agent_type: str = Field(description="Agent type key (e.g., 'codex', 'claude', 'ollama')")
    reason: Optional[str] = Field(default=None, description="Rationale for selection")


class TaskEnvelope(BaseModel):
    """Normalized, stateless task structure consumed by roles.

    The ``payload`` field should contain role-relevant inputs (e.g., specs,
    code snippets, diffs). ``meta`` carries tracing metadata compatible with
    the global envelope model.
    """

    id: str = Field(description="Stable task identifier")
    title: str = Field(description="Short task title")
    description: str = Field(description="Detailed task description")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Structured inputs for the role")
    context: Dict[str, Any] = Field(default_factory=dict, description="Ambient context (repo, paths, configs)")
    constraints: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    priority: Optional[str] = None
    role_hint: Optional[RoleName] = None
    requested_agent_type: Optional[str] = Field(default=None, description="Override agent type (e.g., 'codex')")
    meta: EnvelopeMeta = Field(default_factory=EnvelopeMeta)


class ResultEnvelope(BaseModel):
    """Role output structure.

    ``outputs`` is role-specific, but should remain structured and
    deterministic. Timestamps live in ``meta``; callers comparing results for
    idempotency can ignore ``meta``.
    """

    id: str
    role: RoleName
    status: EnvelopeStatus = EnvelopeStatus.SUCCESS
    model_assigned: Optional[str] = Field(default=None, description="Agent type hint used for downstream work")
    decisions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    followups: List[str] = Field(default_factory=list)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    meta: EnvelopeMeta = Field(default_factory=EnvelopeMeta)


class PlanningBehavior:
    """Reusable planning behavior for roles that produce plans/designs."""

    @staticmethod
    def plan_components(task: TaskEnvelope) -> Dict[str, Any]:
        reqs = task.acceptance_criteria or []
        constraints = task.constraints or []
        components = [
            {"name": "api", "responsibilities": ["expose endpoints"], "depends_on": ["core"]},
            {"name": "core", "responsibilities": ["business logic"], "depends_on": []},
            {"name": "storage", "responsibilities": ["persistence"], "depends_on": ["core"]},
        ]
        interfaces = {
            "api->core": {"contract": "pure funcs", "schema": "pydantic models"},
            "core->storage": {"contract": "repo iface", "schema": "dataclasses/pydantic"},
        }
        return {
            "components": components,
            "interfaces": interfaces,
            "constraints": constraints,
            "acceptance_criteria": reqs,
        }


class ImplementationBehavior:
    """Reusable implementation behavior for code-centric roles."""

    @staticmethod
    def propose_changes(task: TaskEnvelope) -> Dict[str, Any]:
        spec = task.payload.get("spec", task.description)
        return {
            "change_plan": [
                {"path": "src/", "action": "modify", "summary": f"Implement: {task.title}"}
            ],
            "rationale": f"Implements spec: {str(spec)[:120]}",
        }


class ReviewBehavior:
    """Reusable review behavior for QA-like roles."""

    @staticmethod
    def review_findings(task: TaskEnvelope) -> Dict[str, Any]:
        notes = []
        if task.acceptance_criteria:
            notes.append("Validated acceptance criteria presence")
        if task.constraints:
            notes.append("Checked stated constraints")
        return {
            "summary": "Performed static review with policy checks",
            "notes": notes,
            "issues": [],
        }


class Role:
    """Base role. Stateless; subclasses should implement ``apply``.

    The instance carries only advisory metadata (e.g., ``model_assignment``)
    that does not affect functional determinism of ``apply``.
    """

    # Optional advisory model assignment set by the factory/registry
    model_assignment: Optional[ModelAssignment] = None

    @classmethod
    def name(cls) -> RoleName:
        raise NotImplementedError

    @classmethod
    def responsibilities(cls) -> List[str]:
        """Human-readable responsibilities for documentation and tests."""
        return []

    @classmethod
    def decision_rights(cls) -> List[str]:
        """Human-readable decision rights for documentation and tests."""
        return []

    @classmethod
    def apply(cls, task: TaskEnvelope) -> ResultEnvelope:  # pragma: no cover - interface
        """Pure function: TaskEnvelope -> ResultEnvelope.

        Subclasses must implement deterministic logic based only on input.
        """
        raise NotImplementedError


class BaseRole:
    """Async role interface using AGENTS.md v2 Task/Result envelopes.

    Implementations may call into AgentManager to route work to a concrete
    model/agent. Instances are stateless aside from advisory model assignment.
    """

    # Optional advisory model assignment set by the factory/registry
    model_assignment: Optional[ModelAssignment] = None

    @classmethod
    def name(cls) -> RoleName:
        raise NotImplementedError

    @classmethod
    def responsibilities(cls) -> List[str]:
        return []

    @classmethod
    def decision_rights(cls) -> List[str]:
        return []

    @classmethod
    def preferred_agent_type(cls) -> str:
        return (getattr(cls, "_preferred_agent_type", None) or "ollama")  # type: ignore[attr-defined]

    async def execute(
        self,
        task: TaskEnvelopeV1,
        agent_manager: "AgentManager",
        *,
        timeout: Optional[int] = None,
        max_retries: int = 1,
    ) -> ResultEnvelopeV1:
        agent_type = (
            getattr(self.model_assignment, "agent_type", None)
            or self.preferred_agent_type()
        )
        prompt = self._build_prompt(task)

        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                job_id = await agent_manager.spawn_agent(agent_type, prompt, timeout=timeout or 300)
                status = await agent_manager.wait_for_completion(job_id)
                if status.state == status.state.COMPLETED:
                    return ResultEnvelopeV1(
                        status=EnvelopeStatus.SUCCESS,
                        artifacts={
                            "output": status.output or "",
                            "agent_type": agent_type,
                            "job_id": job_id,
                        },
                        metrics={"retries": attempt - 1},
                        confidence=0.6,
                        notes=f"Executed by {self.name().value}",
                    )
                # Otherwise treat as retryable except cancelled
                if status.state.name == "CANCELLED":
                    last_err = RuntimeError("Job cancelled")
                    break
                last_err = RuntimeError(status.error or "Unknown error")
            except Exception as e:  # pragma: no cover - network/timeout
                last_err = e

        return ResultEnvelopeV1(
            status=EnvelopeStatus.ERROR,
            artifacts={"agent_type": agent_type},
            metrics={"retries": max_retries},
            confidence=0.0,
            notes=f"{self.name().value} failed: {last_err}",
        )

    def _build_prompt(self, task: TaskEnvelopeV1) -> str:
        parts = [f"Role: {self.name().value}", f"Objective: {task.objective}"]
        if task.bounded_context:
            parts.append(f"Context: {task.bounded_context}")
        if task.constraints:
            parts.append("Constraints: " + "; ".join(task.constraints))
        if task.inputs:
            parts.append(f"Inputs: {task.inputs}")
        if task.output_schema:
            parts.append(f"Output schema (hint): {task.output_schema}")
        parts.append("Return succinct, production‑ready results.")
        return "\n".join(parts)


    
