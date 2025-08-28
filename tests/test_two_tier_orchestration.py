import asyncio
import os

import pytest

from src.agentsmcp.config import Config
from src.agentsmcp.agent_manager import AgentManager
from src.agentsmcp.models import EnvelopeStatus
from src.agentsmcp.orchestration import MainCoordinator, EventBus, JobStarted, JobCompleted, JobFailed


@pytest.mark.asyncio
async def test_two_tier_coordinator_happy_path(monkeypatch):
    # Avoid real network/model calls
    monkeypatch.setenv("AGENTSMCP_TEST_MODE", "1")

    cfg = Config.from_env()

    # Typed event bus for orchestration events
    typed_bus = EventBus()

    # Collect events for assertions
    seen = {"started": 0, "completed": 0, "failed": 0}

    async def on_started(ev: JobStarted):
        seen["started"] += 1

    async def on_completed(ev: JobCompleted):
        seen["completed"] += 1

    async def on_failed(ev: JobFailed):
        seen["failed"] += 1

    await typed_bus.subscribe(JobStarted, on_started)
    await typed_bus.subscribe(JobCompleted, on_completed)
    await typed_bus.subscribe(JobFailed, on_failed)

    # AgentManager with orchestrator bus to publish typed events as well
    mgr = AgentManager(cfg, orchestrator_bus=typed_bus)

    coord = MainCoordinator(agent_manager=mgr, event_bus=typed_bus)
    await coord.start()

    task_id = await coord.submit_task(
        objective="Implement a simple utility function and summarize result",
        bounded_context="unit-test",
        inputs={"lang": "python"},
        constraints=["follow project conventions"],
    )

    # Wait for result with a timeout
    result = None
    for _ in range(100):  # up to ~10s
        result = await coord.get_task_result(task_id)
        if result is not None:
            break
        await asyncio.sleep(0.1)

    assert result is not None, "Coordinator did not produce a result"
    assert result.status == EnvelopeStatus.SUCCESS
    assert (result.artifacts or {}).get("output", "").startswith("Ollama") or \
           (result.artifacts or {}).get("output", "").startswith("Codex") or \
           (result.artifacts or {}).get("output", "").startswith("Claude") or \
           "Simulation" in (result.artifacts or {}).get("output", "")

    # Quality gates should pass for successful result
    assert coord.quality_gates.all_gates_passed(task_id)

    # At least one started and one completed event should be seen
    assert seen["started"] >= 1
    assert seen["completed"] >= 1
    assert seen["failed"] == 0

    await typed_bus.shutdown()
