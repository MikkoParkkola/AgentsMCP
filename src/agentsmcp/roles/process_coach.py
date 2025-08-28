from __future__ import annotations

from typing import List

from .base import BaseRole, RoleName, TaskEnvelope, ResultEnvelope, EnvelopeStatus


class ProcessCoachRole(BaseRole):
    """Process coach role: provides workflow guidance and coordination notes."""

    _preferred_agent_type = "claude"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.PROCESS_COACH

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Process guidance", "Coordination"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Workflow adjustments"]

    @classmethod
    def apply(cls, task: TaskEnvelope) -> ResultEnvelope:  # type: ignore[override]
        steps = ["Plan", "Implement", "Review", "Merge"]
        return ResultEnvelope(
            id=task.id,
            role=cls.name(),
            status=EnvelopeStatus.SUCCESS,
            model_assigned=(task.requested_agent_type or cls._preferred_agent_type),
            decisions=["Define checkpoints"],
            outputs={"guidance": steps},
        )

