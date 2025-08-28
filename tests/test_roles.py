import copy

import pytest

from agentsmcp.roles import get_role, RoleName, DEFAULT_MODEL_ASSIGNMENTS
from agentsmcp.roles.base import TaskEnvelope, ResultEnvelope


def _make_task(role: RoleName) -> TaskEnvelope:
    return TaskEnvelope(
        id=f"task-{role.value}",
        title=f"Test {role.value.title()} Role",
        description="Implement P1: Role-Based Agent System",
        payload={
            "files": ["src/a.py", "tests/b_test.py"],
            "changes": [{"path": "src/a.py", "action": "modify"}],
            "ci_status": "passing",
            "reviews": 1,
            "coverage": 85,
        },
        context={"repo": "AgentsMCP"},
        constraints=["Stateless", "Pure functions"],
        acceptance_criteria=["All roles implemented", "Unit tests present"],
        priority="P1",
        role_hint=role,
    )


@pytest.mark.parametrize(
    "role_name",
    [
        RoleName.ARCHITECT,
        RoleName.CODER,
        RoleName.QA,
        RoleName.MERGE_BOT,
        RoleName.DOCS,
        RoleName.PROCESS_COACH,
        RoleName.METRICS_COLLECTOR,
    ],
)
def test_role_registry_and_apply(role_name: RoleName):
    role = get_role(role_name)

    # Ensure responsibilities/decision rights declared
    assert isinstance(role.responsibilities(), list)
    assert isinstance(role.decision_rights(), list)
    assert len(role.responsibilities()) > 0
    assert len(role.decision_rights()) > 0

    task = _make_task(role_name)

    result = role.apply(task)
    assert isinstance(result, ResultEnvelope)
    assert result.id == task.id
    assert result.role == role_name
    assert result.status.value in {"success", "pending", "error"}
    assert isinstance(result.outputs, dict)

    # Default model assignment should be present (or honored from task)
    if task.requested_agent_type:
        assert result.model_assigned == task.requested_agent_type
    else:
        assert result.model_assigned == DEFAULT_MODEL_ASSIGNMENTS[role_name].agent_type


@pytest.mark.parametrize(
    "role_name",
    [
        RoleName.ARCHITECT,
        RoleName.CODER,
        RoleName.QA,
        RoleName.MERGE_BOT,
        RoleName.DOCS,
        RoleName.PROCESS_COACH,
        RoleName.METRICS_COLLECTOR,
    ],
)
def test_role_pure_function_determinism(role_name: RoleName):
    role = get_role(role_name)
    task = _make_task(role_name)
    r1 = role.apply(task)
    r2 = role.apply(copy.deepcopy(task))

    # Compare excluding meta timestamps
    d1 = r1.model_dump()
    d2 = r2.model_dump()
    d1.pop("meta", None)
    d2.pop("meta", None)
    assert d1 == d2


def test_merge_bot_policy_checks():
    role = get_role(RoleName.MERGE_BOT)
    t_ok = _make_task(RoleName.MERGE_BOT)
    res_ok = role.apply(t_ok)
    assert res_ok.outputs["recommendation"] == "merge"

    t_bad = _make_task(RoleName.MERGE_BOT)
    t_bad.payload["ci_status"] = "failing"
    res_bad = role.apply(t_bad)
    assert res_bad.outputs["recommendation"] == "block"

