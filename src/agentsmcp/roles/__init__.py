"""Role registry and factory for AgentsMCP.

This package provides a small, extensible role system that integrates with the
existing AgentsMCP components while keeping interfaces clean. Roles expose an
async ``execute(TaskEnvelopeV1) -> ResultEnvelopeV1`` method and include model
routing preferences. A lightweight registry maps role names to implementations
and default model hints.

Design goals:
- Clear responsibilities and decision rights per role
- Deterministic core logic; async integration for execution
- Composition-friendly base behaviors for reuse
- Minimal, dependency-free integration touch points

Typical usage:
    from agentsmcp.roles import get_role, RoleName
    role = get_role(RoleName.ARCHITECT)
    result = await role.execute(task_envelope, agent_manager)

Model selection can be overridden by orchestrators or callers if desired.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Role, BaseRole, RoleName, ModelAssignment
from .architect import ArchitectRole
from .coder import CoderRole
from .qa import QARole
from .merge_bot import MergeBotRole
from .docs import DocsRole
from .agile_coach import AgileCoachRole
from .metrics import MetricsCollectorRole


# Registry of role name -> role class
ROLE_REGISTRY: Dict[RoleName, Type[BaseRole]] = {
    RoleName.ARCHITECT: ArchitectRole,
    RoleName.CODER: CoderRole,
    RoleName.QA: QARole,
    RoleName.MERGE_BOT: MergeBotRole,
    RoleName.DOCS: DocsRole,
    RoleName.PROCESS_COACH: AgileCoachRole,
    RoleName.METRICS_COLLECTOR: MetricsCollectorRole,
}


# Default model assignments aligned to existing AgentsMCP agent types
DEFAULT_MODEL_ASSIGNMENTS: Dict[RoleName, ModelAssignment] = {
    # Complex reasoning and planning – prefer specialist
    RoleName.ARCHITECT: ModelAssignment(agent_type="codex", reason="complex reasoning & planning"),
    # High-volume implementation – prefer workhorse
    RoleName.CODER: ModelAssignment(agent_type="ollama", reason="implementation work, cost effective"),
    # Verification & analysis – prefer specialist
    RoleName.QA: ModelAssignment(agent_type="codex", reason="analysis and test synthesis"),
    # Light automation, policy checks – workhorse
    RoleName.MERGE_BOT: ModelAssignment(agent_type="ollama", reason="automation and policy checks"),
    # Summarization and docs – workhorse by default
    RoleName.DOCS: ModelAssignment(agent_type="ollama", reason="summaries and documentation"),
    # Orchestration & guidance – orchestrator
    RoleName.PROCESS_COACH: ModelAssignment(agent_type="claude", reason="process guidance & long context"),
    # Lightweight analytics – workhorse
    RoleName.METRICS_COLLECTOR: ModelAssignment(agent_type="ollama", reason="lightweight metrics aggregation"),
}


def get_role(name: RoleName) -> BaseRole:
    """Return a role instance with default model assignment.

    While roles are implemented as stateless classes exposing a pure
    ``apply(...)`` function, returning an instance allows future extension to
    provide metadata/config without carrying mutable state.
    """
    cls = ROLE_REGISTRY[name]
    role = cls()
    role.model_assignment = DEFAULT_MODEL_ASSIGNMENTS.get(name)
    return role


__all__ = [
    "RoleName",
    "BaseRole",
    "get_role",
    "ROLE_REGISTRY",
    "DEFAULT_MODEL_ASSIGNMENTS",
]
