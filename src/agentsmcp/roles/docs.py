from __future__ import annotations

from typing import List

from .base import BaseRole, RoleName, TaskEnvelope, ResultEnvelope, EnvelopeStatus


class DocsRole(BaseRole):
    """Documentation role: produces summaries and docs updates."""

    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.DOCS

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Summaries", "Docs updates"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Docs structure", "Clarity"]

    @classmethod
    def apply(cls, task: TaskEnvelope) -> ResultEnvelope:  # type: ignore[override]
        payload = task.payload or {}
        summary = f"Docs summary for: {task.title}"
        return ResultEnvelope(
            id=task.id,
            role=cls.name(),
            status=EnvelopeStatus.SUCCESS,
            model_assigned=(task.requested_agent_type or cls._preferred_agent_type),
            decisions=["Create/Update docs pages"],
            outputs={"summary": summary, "inputs": payload},
        )

