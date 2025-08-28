from __future__ import annotations

from typing import List

from .base import BaseRole, RoleName, TaskEnvelope, ResultEnvelope, EnvelopeStatus, ImplementationBehavior


class CoderRole(BaseRole):
    """Code implementation role.

    Focuses on writing and modifying code according to a spec, following
    project conventions and producing production-ready patches.
    """

    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.CODER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return [
            "Code implementation",
            "Small refactors",
            "Bug fixes",
        ]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Implementation details", "Local refactors"]

    def _build_prompt(self, task):  # type: ignore[override]
        base = super()._build_prompt(task)
        guidance = (
            "Implement changes with minimal diffs, follow existing style, add"
            " type hints and docstrings. Prefer safe, maintainable solutions."
        )
        return f"{base}\n{guidance}"

    # Backward-compatible deterministic apply
    @classmethod
    def apply(cls, task: TaskEnvelope) -> ResultEnvelope:  # type: ignore[override]
        plan = ImplementationBehavior.propose_changes(task)
        return ResultEnvelope(
            id=task.id,
            role=cls.name(),
            status=EnvelopeStatus.SUCCESS,
            model_assigned=(task.requested_agent_type or (getattr(cls, "_preferred_agent_type", None) or "ollama")),
            decisions=["Limit scope of changes", "Adhere to style"],
            risks=["Regression risk"],
            followups=["Write unit tests"],
            outputs=plan,
        )
