from __future__ import annotations

from typing import List

from .base import BaseRole, RoleName, TaskEnvelope, ResultEnvelope, EnvelopeStatus, PlanningBehavior


class ArchitectRole(BaseRole):
    """Architecture and design role.

    Produces high-level designs, component breakdowns, and interface contracts.
    Prefers a reasoning-strong model for planning tasks.
    """

    _preferred_agent_type = "codex"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.ARCHITECT

    @classmethod
    def responsibilities(cls) -> List[str]:
        return [
            "System design",
            "Component decomposition",
            "Interface contracts",
        ]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Architecture decisions", "Design trade-offs"]

    def _build_prompt(self, task):  # type: ignore[override]
        base = super()._build_prompt(task)
        guidance = (
            "Produce a concise architecture: components, data flow, contracts,"
            " risks. Optimize for security, performance, maintainability."
        )
        return f"{base}\n{guidance}"

    # Backward-compatible deterministic apply
    @classmethod
    def apply(cls, task: TaskEnvelope) -> ResultEnvelope:  # type: ignore[override]
        design = PlanningBehavior.plan_components(task)
        return ResultEnvelope(
            id=task.id,
            role=cls.name(),
            status=EnvelopeStatus.SUCCESS,
            model_assigned=(task.requested_agent_type or (getattr(cls, "_preferred_agent_type", None) or "ollama")),
            decisions=["Adopt layered architecture", "Define clear interfaces"],
            risks=["Interface drift", "Under-specified contracts"],
            followups=["Create sequence diagrams", "Draft API schemas"],
            outputs={"design": design},
        )
