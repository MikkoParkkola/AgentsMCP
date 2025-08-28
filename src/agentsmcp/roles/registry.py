from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Type

from .base import BaseRole, RoleName, ModelAssignment
from .architect import ArchitectRole
from .coder import CoderRole
from .qa import QARole
from .merge_bot import MergeBotRole


@dataclass(frozen=True)
class RoutingDecision:
    role: RoleName
    agent_type: str
    reason: str


class RoleRegistry:
    """Dynamic role assignment and model routing.

    Keeps simple heuristics for now:
    - Map by task keywords when provided
    - Use objective length and presence of design keywords for complexity
    - Prefer 'codex' for complex reasoning; 'ollama' for basic/automation
    """

    ROLE_CLASSES: dict[RoleName, Type[BaseRole]] = {
        RoleName.ARCHITECT: ArchitectRole,
        RoleName.CODER: CoderRole,
        RoleName.QA: QARole,
        RoleName.MERGE_BOT: MergeBotRole,
    }

    KEYWORD_TO_ROLE = {
        "design": RoleName.ARCHITECT,
        "architecture": RoleName.ARCHITECT,
        "refactor": RoleName.CODER,
        "implement": RoleName.CODER,
        "fix": RoleName.CODER,
        "test": RoleName.QA,
        "qa": RoleName.QA,
        "merge": RoleName.MERGE_BOT,
        "rebase": RoleName.MERGE_BOT,
    }

    def resolve_role(self, objective: str, role_hint: Optional[RoleName] = None) -> RoleName:
        if role_hint:
            return role_hint
        lo = objective.lower()
        for k, r in self.KEYWORD_TO_ROLE.items():
            if k in lo:
                return r
        # Default to coder for general tasks
        return RoleName.CODER

    def choose_agent_type(self, role: RoleName, objective: str) -> str:
        lo = objective.lower()
        complex_markers = ["design", "architecture", "plan", "strategy"]
        is_complex = any(m in lo for m in complex_markers) or len(objective) > 200

        if role in {RoleName.ARCHITECT, RoleName.QA}:
            return "codex"
        if role is RoleName.MERGE_BOT:
            return "ollama"
        # role == CODER
        return "codex" if is_complex else "ollama"

    def instantiate(self, role: RoleName, *, agent_type: Optional[str] = None) -> BaseRole:
        cls = self.ROLE_CLASSES[role]
        inst = cls()
        inst.model_assignment = ModelAssignment(
            agent_type=agent_type or cls.preferred_agent_type(),
            reason="registry default",
        )
        return inst

    def route(self, task) -> Tuple[BaseRole, RoutingDecision]:  # TaskEnvelopeV1 duck-typed
        role = self.resolve_role(task.objective, getattr(task, "role_hint", None))
        agent_type = self.choose_agent_type(role, task.objective)
        reason = "complex reasoning" if agent_type == "codex" else "basic automation"
        inst = self.instantiate(role, agent_type=agent_type)
        return inst, RoutingDecision(role=role, agent_type=agent_type, reason=reason)

