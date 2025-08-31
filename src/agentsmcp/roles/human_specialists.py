from __future__ import annotations

from typing import List

from .base import BaseRole, RoleName


class BusinessAnalystRole(BaseRole):
    @classmethod
    def name(cls) -> RoleName:
        return RoleName.BUSINESS_ANALYST

    @classmethod
    def responsibilities(cls) -> List[str]:
        return [
            "Elicit requirements and acceptance criteria",
            "Clarify scope and value",
            "Translate business needs into engineering-ready tasks",
        ]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Define acceptance criteria", "Prioritize user value"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a business analyst. Elicit requirements, define acceptance criteria, "
            "clarify scope, and translate needs into engineering-ready tasks."
        )


class BackendEngineerRole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.BACKEND_ENGINEER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Design/implement services", "Data models", "Persistence and APIs"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Choose backend patterns", "Optimize performance"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a backend engineer. Design and implement robust services, data models, "
            "persistence layers, and APIs with performance and security in mind."
        )


class WebFrontendEngineerRole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.WEB_FRONTEND_ENGINEER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Implement web UI", "Accessibility", "Responsive design"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["UI composition", "Component architecture"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a web frontend engineer. Build accessible, responsive, and maintainable UI "
            "components with great UX."
        )


class APIEngineerRole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.API_ENGINEER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["API design", "Contracts/ICD", "Versioning"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Endpoint shape", "Error semantics"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are an API engineer. Define contracts/ICDs, versioning, and error semantics; "
            "ensure clarity, stability, and testability."
        )


class TUIFrontendEngineerRole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.TUI_FRONTEND_ENGINEER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Design/implement TUI", "Keybindings", "Terminal compatibility"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["TUI layout", "Interaction model"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a TUI frontend engineer. Design/implement terminal UIs with clean layout, "
            "great keyboard interaction, and broad terminal compatibility."
        )


class BackendQARole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.BACKEND_QA_ENGINEER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Test backend services", "Contract tests", "Load/error cases"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Test coverage", "Quality gates"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a backend QA engineer. Design and execute tests for services and data layers, "
            "covering contracts, errors, and load scenarios."
        )


class WebFrontendQARole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.WEB_FRONTEND_QA_ENGINEER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Test web UI", "Accessibility checks", "Cross-browser"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["UI quality gates", "UX regressions"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a web frontend QA engineer. Validate accessibility, rendering, and interaction "
            "across browsers; prevent UX regressions."
        )


class TUIFrontendQARole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.TUI_FRONTEND_QA_ENGINEER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Test TUI", "Terminal coverage", "Input edge cases"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["TUI test strategy", "Compatibility matrix"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a TUI frontend QA engineer. Test TUI across terminals, inputs, and edge cases; "
            "ensure reliable behavior."
        )


class ChiefQARole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.CHIEF_QA_ENGINEER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Define QA strategy", "Approve releases", "Ensure quality system"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Release gate", "Quality KPIs"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a chief QA engineer. Define QA strategy, quality gates, and approve releases; "
            "drive continuous quality improvements."
        )

class ITLawyerRole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.IT_LAWYER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Review licenses", "Privacy/GDPR", "Contracts and compliance"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["License compatibility", "Compliance risk"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are an IT lawyer. Advise on licensing, privacy/GDPR, and compliance for the project; "
            "flag legal risks and propose mitigations."
        )

class MarketingManagerRole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.MARKETING_MANAGER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Positioning", "Messaging", "Content/SEO"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Messaging strategy", "Campaign priorities"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a marketing manager. Craft positioning, messaging, and content/SEO plans aligned "
            "to audience and product goals."
        )

class CICDEngineerRole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.CI_CD_ENGINEER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Build pipelines", "Deploy automation", "Release flows"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Pipeline design", "Env promotion policy"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a CI/CD engineer. Design reliable build/test/deploy pipelines and safe release flows."
        )

class DevToolingEngineerRole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.DEV_TOOLING_ENGINEER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["DX tooling", "Automation", "Linters/formatters"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Tool choices", "DX policies"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a developer tooling engineer. Improve developer experience with effective tooling and automation."
        )

class DataAnalystRole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.DATA_ANALYST

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Exploratory analysis", "Dashboards", "SQL"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Metrics selection", "Reporting cadence"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a data analyst. Perform exploratory analysis, build metrics and dashboards, and communicate insights."
        )

class DataScientistRole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.DATA_SCIENTIST

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Hypothesis testing", "Modeling", "Experiment design"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Model choice", "Experimentation roadmap"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a data scientist. Design experiments and models to answer questions and validate hypotheses."
        )

class MLSicentistRole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.ML_SCIENTIST

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["ML research", "Paper reproduction", "Novel approaches"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Research direction", "Technique selection"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a machine learning scientist. Explore and evaluate novel ML approaches and research directions."
        )

class MLEngineerRole(BaseRole):
    _preferred_agent_type = "ollama"

    @classmethod
    def name(cls) -> RoleName:
        return RoleName.ML_ENGINEER

    @classmethod
    def responsibilities(cls) -> List[str]:
        return ["Model training", "Datasets", "Inference systems"]

    @classmethod
    def decision_rights(cls) -> List[str]:
        return ["Training strategy", "Deployment design"]

    @classmethod
    def default_prompt(cls) -> str:
        return (
            "You are a machine learning engineer. Build reliable training, data, and inference systems for ML models."
        )
