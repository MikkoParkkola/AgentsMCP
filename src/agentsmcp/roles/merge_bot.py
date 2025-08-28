from __future__ import annotations

from typing import List

from .base import BaseRole, RoleName, TaskEnvelope, ResultEnvelope, EnvelopeStatus


class MergeBotRole(BaseRole):
    """Merge coordination role.

    Performs policy checks and summarizes PR status for merge readiness.
    Prefers a cost-effective model.
    """

    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.MERGE_BOT

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Policy checks", "Merge summary", "Conflict hints"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Merge readiness gate"]

    def _build_prompt(self, task):  # type: ignore[override]
        base = super()._build_prompt(task)
        guidance = (
            "Evaluate PR readiness: CI status, reviews, conflicts, labels."
            " Provide a clear merge recommendation and next steps."
        )
        return f"{base}\n{guidance}"

    # Backward-compatible deterministic apply
    @classmethod
    def apply(cls, task: TaskEnvelope) -> ResultEnvelope:  # type: ignore[override]
        payload = task.payload or {}
        ci = payload.get("ci_status", "unknown")
        reviews = payload.get("reviews", 0)
        coverage = payload.get("coverage", 0)

        ok = ci == "passing" and int(reviews) >= 1 and int(coverage) >= 80
        recommendation = "merge" if ok else "block"
        reasons = []
        if ci != "passing":
            reasons.append("CI failing")
        if int(reviews) < 1:
            reasons.append("Needs review")
        if int(coverage) < 80:
            reasons.append("Low coverage")

        return ResultEnvelope(
            id=task.id,
            role=cls.name(),
            status=EnvelopeStatus.SUCCESS,
            model_assigned=(task.requested_agent_type or (getattr(cls, "_preferred_agent_type", None) or "ollama")),
            decisions=[f"Recommendation: {recommendation}"],
            risks=["Merge conflicts"],
            followups=["Fix CI", "Request review", "Increase tests"],
            outputs={
                "recommendation": recommendation,
                "reasons": reasons,
            },
        )
