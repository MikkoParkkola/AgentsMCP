import os
import asyncio

import pytest

from src.agentsmcp.config import Config
from src.agentsmcp.agent_manager import AgentManager
from src.agentsmcp.roles.registry import RoleRegistry
from src.agentsmcp.roles.base import RoleName
from src.agentsmcp.models import TaskEnvelopeV1, EnvelopeStatus


@pytest.mark.asyncio
async def test_role_registry_routing_basic():
    reg = RoleRegistry()
    role = reg.resolve_role("Design a modular architecture for the service")
    assert role == RoleName.ARCHITECT
    agent = reg.choose_agent_type(role, "Design a modular architecture for the service")
    assert agent == "codex"


@pytest.mark.asyncio
async def test_execute_role_task_coder_success(monkeypatch):
    # Simulate environment to avoid network
    monkeypatch.setenv("AGENTSMCP_TEST_MODE", "1")
    cfg = Config.from_env()
    mgr = AgentManager(cfg)

    task = TaskEnvelopeV1(objective="Implement new feature flag parser", inputs={"lang": "python"})
    result = await mgr.execute_role_task(task)

    assert result.status == EnvelopeStatus.SUCCESS
    assert result.artifacts is not None
    assert result.artifacts.get("agent_type") in {"ollama", "codex"}


@pytest.mark.asyncio
async def test_execute_role_task_qa_routes_to_codex(monkeypatch):
    monkeypatch.setenv("AGENTSMCP_TEST_MODE", "1")
    cfg = Config.from_env()
    mgr = AgentManager(cfg)

    task = TaskEnvelopeV1(objective="Test coverage analysis and risks", inputs={})
    result = await mgr.execute_role_task(task)
    assert result.status == EnvelopeStatus.SUCCESS
    assert result.artifacts is not None
    # QA prefers codex
    assert result.artifacts.get("agent_type") == "codex"

