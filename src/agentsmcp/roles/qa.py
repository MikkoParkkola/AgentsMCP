from __future__ import annotations

from typing import List

from .base import BaseRole, RoleName, TaskEnvelope, ResultEnvelope, EnvelopeStatus, ReviewBehavior


class QARole(BaseRole):
    """Quality assurance and testing role.

    Reviews changes, proposes tests, and checks acceptance criteria. Prefers a
    reasoning-strong model.
    """

    _preferred_agent_type = "codex"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.QA

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Test planning", "Static review", "Risk assessment"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Test adequacy", "Release readiness"]

    def _build_prompt(self, task):  # type: ignore[override]
        base = super()._build_prompt(task)
        guidance = (
            "Identify risks, propose unit tests, verify acceptance criteria."
            " Output findings and prioritized follow-ups."
        )
        return f"{base}\n{guidance}"

    # Backward-compatible deterministic apply
    @classmethod
    def apply(cls, task: TaskEnvelope) -> ResultEnvelope:  # type: ignore[override]
        findings = ReviewBehavior.review_findings(task)
        return ResultEnvelope(
            id=task.id,
            role=cls.name(),
            status=EnvelopeStatus.SUCCESS,
            model_assigned=(task.requested_agent_type or (getattr(cls, "_preferred_agent_type", None) or "ollama")),
            decisions=["Block release until tests added"],
            risks=["Insufficient coverage"],
            followups=["Add unit tests", "Run lints"],
            outputs=findings,
        )
