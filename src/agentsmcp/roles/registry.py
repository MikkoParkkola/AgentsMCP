from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Type

from .base import BaseRole, RoleName, ModelAssignment
from .architect import ArchitectRole
from .coder import CoderRole
from .qa import QARole
from .merge_bot import MergeBotRole
from .human_specialists import (
    BusinessAnalystRole,
    BackendEngineerRole,
    WebFrontendEngineerRole,
    APIEngineerRole,
    TUIFrontendEngineerRole,
    BackendQARole,
    WebFrontendQARole,
    TUIFrontendQARole,
    ChiefQARole,
    ITLawyerRole,
    MarketingManagerRole,
    CICDEngineerRole,
    DevToolingEngineerRole,
    DataAnalystRole,
    DataScientistRole,
    MLSicentistRole,
    MLEngineerRole,
)


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
        RoleName.BUSINESS_ANALYST: BusinessAnalystRole,
        RoleName.BACKEND_ENGINEER: BackendEngineerRole,
        RoleName.WEB_FRONTEND_ENGINEER: WebFrontendEngineerRole,
        RoleName.API_ENGINEER: APIEngineerRole,
        RoleName.TUI_FRONTEND_ENGINEER: TUIFrontendEngineerRole,
        RoleName.BACKEND_QA_ENGINEER: BackendQARole,
        RoleName.WEB_FRONTEND_QA_ENGINEER: WebFrontendQARole,
        RoleName.TUI_FRONTEND_QA_ENGINEER: TUIFrontendQARole,
        RoleName.CHIEF_QA_ENGINEER: ChiefQARole,
        RoleName.IT_LAWYER: ITLawyerRole,
        RoleName.MARKETING_MANAGER: MarketingManagerRole,
        RoleName.CI_CD_ENGINEER: CICDEngineerRole,
        RoleName.DEV_TOOLING_ENGINEER: DevToolingEngineerRole,
        RoleName.DATA_ANALYST: DataAnalystRole,
        RoleName.DATA_SCIENTIST: DataScientistRole,
        RoleName.ML_SCIENTIST: MLSicentistRole,
        RoleName.ML_ENGINEER: MLEngineerRole,
    }

    KEYWORD_TO_ROLE = {
        "design": RoleName.ARCHITECT,
        "architecture": RoleName.ARCHITECT,
        "analysis": RoleName.BUSINESS_ANALYST,
        "requirements": RoleName.BUSINESS_ANALYST,
        "refactor": RoleName.CODER,
        "implement": RoleName.CODER,
        "fix": RoleName.CODER,
        "test": RoleName.QA,
        "qa": RoleName.QA,
        "backend": RoleName.BACKEND_ENGINEER,
        "database": RoleName.BACKEND_ENGINEER,
        "api": RoleName.API_ENGINEER,
        "rest": RoleName.API_ENGINEER,
        "openapi": RoleName.API_ENGINEER,
        "frontend": RoleName.WEB_FRONTEND_ENGINEER,
        "web": RoleName.WEB_FRONTEND_ENGINEER,
        "react": RoleName.WEB_FRONTEND_ENGINEER,
        "tui": RoleName.TUI_FRONTEND_ENGINEER,
        "terminal": RoleName.TUI_FRONTEND_ENGINEER,
        "backend qa": RoleName.BACKEND_QA_ENGINEER,
        "web qa": RoleName.WEB_FRONTEND_QA_ENGINEER,
        "tui qa": RoleName.TUI_FRONTEND_QA_ENGINEER,
        "chief qa": RoleName.CHIEF_QA_ENGINEER,
        "merge": RoleName.MERGE_BOT,
        "rebase": RoleName.MERGE_BOT,
        "legal": RoleName.IT_LAWYER,
        "license": RoleName.IT_LAWYER,
        "privacy": RoleName.IT_LAWYER,
        "gdpr": RoleName.IT_LAWYER,
        "marketing": RoleName.MARKETING_MANAGER,
        "seo": RoleName.MARKETING_MANAGER,
        "ci ": RoleName.CI_CD_ENGINEER,
        "cd ": RoleName.CI_CD_ENGINEER,
        "pipeline": RoleName.CI_CD_ENGINEER,
        "deploy": RoleName.CI_CD_ENGINEER,
        "devtools": RoleName.DEV_TOOLING_ENGINEER,
        "tooling": RoleName.DEV_TOOLING_ENGINEER,
        "analytics": RoleName.DATA_ANALYST,
        "sql": RoleName.DATA_ANALYST,
        "data science": RoleName.DATA_SCIENTIST,
        "statistics": RoleName.DATA_SCIENTIST,
        "ml research": RoleName.ML_SCIENTIST,
        "ml scientist": RoleName.ML_SCIENTIST,
        "training": RoleName.ML_ENGINEER,
        "dataset": RoleName.ML_ENGINEER,
        "inference": RoleName.ML_ENGINEER,
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
