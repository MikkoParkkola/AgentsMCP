from __future__ import annotations

from typing import List

from .base import BaseRole, RoleName, TaskEnvelope, ResultEnvelope, EnvelopeStatus


class MetricsCollectorRole(BaseRole):
    """Metrics collector role: aggregates simple metrics for visibility."""

    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.METRICS_COLLECTOR

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Metrics aggregation", "Reporting"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Alert thresholds"]

    @classmethod
    def apply(cls, task: TaskEnvelope) -> ResultEnvelope:  # type: ignore[override]
        payload = task.payload or {}
        metrics = {"items": len(payload) if hasattr(payload, "__len__") else 1}
        return ResultEnvelope(
            id=task.id,
            role=cls.name(),
            status=EnvelopeStatus.SUCCESS,
            model_assigned=(task.requested_agent_type or cls._preferred_agent_type),
            outputs={"metrics": metrics},
        )

