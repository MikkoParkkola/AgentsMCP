"""Tests for AgileCoachRole integration with BaseRole architecture."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from agentsmcp.roles.agile_coach import AgileCoachRole
from agentsmcp.roles.base import RoleName, TaskEnvelope, EnvelopeStatus
from agentsmcp.models import EnvelopeMeta, TaskEnvelopeV1, ResultEnvelopeV1
from agentsmcp.agent_manager import AgentManager


class TestAgileCoachRoleIntegration:
    """Test AgileCoachRole integration with BaseRole architecture."""

    @pytest.fixture
    def agile_coach_role(self):
        """Create AgileCoachRole instance."""
        return AgileCoachRole()

    @pytest.fixture
    def mock_agent_manager(self):
        """Create mock agent manager."""
        return Mock(spec=AgentManager)

    @pytest.fixture
    def task_envelope_v1(self):
        """Create TaskEnvelopeV1 for async execution tests."""
        return TaskEnvelopeV1(
            objective="Provide agile coaching for sprint planning",
            bounded_context="Development team of 4 members working on API implementation",
            inputs={
                "coaching_type": "planning",
                "task_classification": {
                    "task_type": "implementation",
                    "complexity": "high",
                    "risk_level": "medium"
                },
                "team_composition": {
                    "primary_team": [
                        {"role": "architect", "model_assignment": "claude"},
                        {"role": "coder", "model_assignment": "codex"}
                    ]
                }
            },
            constraints=["Complete within 1 hour", "Use existing team structure"],
            routing={"model": "claude", "effort": "medium"}
        )

    def test_inheritance_and_interfaces(self, agile_coach_role):
        """Test that AgileCoachRole properly inherits from BaseRole."""
        from agentsmcp.roles.base import BaseRole
        
        # Test inheritance
        assert isinstance(agile_coach_role, BaseRole)
        
        # Test required class methods
        assert hasattr(AgileCoachRole, 'name')
        assert hasattr(AgileCoachRole, 'responsibilities')
        assert hasattr(AgileCoachRole, 'decision_rights')
        assert hasattr(AgileCoachRole, 'preferred_agent_type')
        assert hasattr(AgileCoachRole, 'apply')
        
        # Test async execution method from BaseRole
        assert hasattr(agile_coach_role, 'execute')

    def test_role_metadata(self, agile_coach_role):
        """Test role metadata is properly defined."""
        # Test name
        assert AgileCoachRole.name() == RoleName.PROCESS_COACH
        
        # Test responsibilities are comprehensive
        responsibilities = AgileCoachRole.responsibilities()
        assert len(responsibilities) >= 6
        assert any('retrospective' in r.lower() for r in responsibilities)
        assert any('performance' in r.lower() for r in responsibilities)
        assert any('improvement' in r.lower() for r in responsibilities)
        
        # Test decision rights are appropriate
        decision_rights = AgileCoachRole.decision_rights()
        assert len(decision_rights) >= 4
        assert any('process' in d.lower() for d in decision_rights)
        assert any('team' in d.lower() for d in decision_rights)
        
        # Test preferred agent type
        assert AgileCoachRole.preferred_agent_type() == "claude"

    @pytest.mark.asyncio
    async def test_async_execution_success(self, agile_coach_role, mock_agent_manager, task_envelope_v1):
        """Test successful async execution through BaseRole.execute."""
        # Mock successful agent execution  
        mock_completed_state = Mock()
        mock_completed_state.name = "completed"
        
        mock_job_status = Mock()
        mock_job_status.state = mock_completed_state
        mock_job_status.state.COMPLETED = mock_completed_state  # Make state comparison work
        mock_job_status.output = "Coaching recommendations: Use incremental approach with frequent checkpoints"
        mock_job_status.error = None
        
        mock_agent_manager.spawn_agent = AsyncMock(return_value="job_123")
        mock_agent_manager.wait_for_completion = AsyncMock(return_value=mock_job_status)
        
        # Execute the role
        result = await agile_coach_role.execute(task_envelope_v1, mock_agent_manager)
        
        # Verify execution
        assert isinstance(result, ResultEnvelopeV1)
        assert result.status == EnvelopeStatus.SUCCESS
        assert "output" in result.artifacts
        assert "claude" in result.artifacts["agent_type"]
        assert result.artifacts["job_id"] == "job_123"
        
        # Verify agent manager was called correctly
        mock_agent_manager.spawn_agent.assert_called_once()
        call_args = mock_agent_manager.spawn_agent.call_args
        assert call_args[0][0] == "claude"  # agent_type
        assert "agile coaching" in call_args[0][1].lower() or "process_coach" in call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_async_execution_failure(self, agile_coach_role, mock_agent_manager, task_envelope_v1):
        """Test async execution failure handling."""
        # Mock failed agent execution
        mock_job_status = Mock()
        mock_job_status.state = Mock()
        mock_job_status.state.COMPLETED = "completed"
        mock_job_status.state.name = "failed"
        mock_job_status.output = None
        mock_job_status.error = "Agent execution failed"
        
        mock_agent_manager.spawn_agent = AsyncMock(return_value="job_456")
        mock_agent_manager.wait_for_completion = AsyncMock(return_value=mock_job_status)
        
        # Execute the role
        result = await agile_coach_role.execute(task_envelope_v1, mock_agent_manager, max_retries=1)
        
        # Verify error handling
        assert isinstance(result, ResultEnvelopeV1)
        assert result.status == EnvelopeStatus.ERROR
        assert result.confidence == 0.0
        assert "failed" in result.notes.lower()

    @pytest.mark.asyncio
    async def test_async_execution_timeout(self, agile_coach_role, mock_agent_manager, task_envelope_v1):
        """Test async execution with timeout."""
        # Mock timeout scenario
        mock_agent_manager.spawn_agent = AsyncMock(side_effect=TimeoutError("Agent spawn timeout"))
        
        # Execute with short timeout
        result = await agile_coach_role.execute(
            task_envelope_v1, 
            mock_agent_manager, 
            timeout=30, 
            max_retries=1
        )
        
        # Verify timeout handling
        assert isinstance(result, ResultEnvelopeV1)
        assert result.status == EnvelopeStatus.ERROR
        assert result.metrics["retries"] == 1

    @pytest.mark.asyncio
    async def test_async_execution_with_retries(self, agile_coach_role, mock_agent_manager, task_envelope_v1):
        """Test async execution with retries."""
        # Mock first call fails, second succeeds
        mock_job_status_fail = Mock()
        mock_job_status_fail.state = Mock()
        mock_job_status_fail.state.name = "failed"
        mock_job_status_fail.error = "Temporary failure"
        
        mock_completed_state = Mock()
        mock_completed_state.name = "completed"
        
        mock_job_status_success = Mock()
        mock_job_status_success.state = mock_completed_state
        mock_job_status_success.state.COMPLETED = mock_completed_state  # Make state comparison work
        mock_job_status_success.output = "Successful coaching after retry"
        mock_job_status_success.error = None
        
        mock_agent_manager.spawn_agent = AsyncMock(side_effect=["job_fail", "job_success"])
        mock_agent_manager.wait_for_completion = AsyncMock(side_effect=[
            mock_job_status_fail, 
            mock_job_status_success
        ])
        
        # Execute with retries
        result = await agile_coach_role.execute(
            task_envelope_v1, 
            mock_agent_manager, 
            max_retries=2
        )
        
        # Verify successful retry
        assert result.status == EnvelopeStatus.SUCCESS
        assert result.metrics["retries"] == 1  # One retry
        assert "Successful coaching after retry" in result.artifacts["output"]

    def test_prompt_building(self, agile_coach_role, task_envelope_v1):
        """Test that prompts are built correctly."""
        prompt = agile_coach_role._build_prompt(task_envelope_v1)
        
        # Verify prompt structure
        assert "Role: process_coach" in prompt
        assert task_envelope_v1.objective in prompt
        
        if task_envelope_v1.bounded_context:
            assert task_envelope_v1.bounded_context in prompt
        
        if task_envelope_v1.constraints:
            for constraint in task_envelope_v1.constraints:
                assert constraint in prompt
        
        # Should contain coaching-specific context
        assert any(keyword in prompt.lower() for keyword in [
            'coaching', 'agile', 'process', 'team', 'retrospective', 'improvement'
        ])

    def test_model_assignment_preference(self, agile_coach_role):
        """Test model assignment preferences."""
        # Test default preferred agent type
        assert agile_coach_role.preferred_agent_type() == "claude"
        
        # Test model assignment override
        from agentsmcp.roles.base import ModelAssignment
        custom_assignment = ModelAssignment(
            agent_type="codex",
            reason="Testing custom assignment"
        )
        agile_coach_role.model_assignment = custom_assignment
        
        # When model assignment is set, it should be used in execution
        task = TaskEnvelopeV1(
            objective="Test coaching with custom model",
            inputs={"coaching_type": "general"}
        )
        
        prompt = agile_coach_role._build_prompt(task)
        # The model assignment affects execution, not prompt building directly
        assert isinstance(prompt, str)

    def test_legacy_apply_method_compatibility(self, agile_coach_role):
        """Test compatibility with legacy apply method."""
        # Test with legacy TaskEnvelope
        from agentsmcp.roles.base import TaskEnvelope
        
        legacy_task = TaskEnvelope(
            id="legacy_test",
            title="Legacy Coaching Test",
            description="Test legacy interface",
            payload={
                "coaching_type": "health_assessment",
                "team_metrics": {
                    "quality_score": 0.8,
                    "collaboration_score": 0.75,
                    "team_satisfaction": 0.85
                }
            },
            meta=EnvelopeMeta()
        )
        
        # Should work with legacy apply method
        result = AgileCoachRole.apply(legacy_task)
        
        assert result.role == RoleName.PROCESS_COACH
        assert result.status == EnvelopeStatus.SUCCESS
        assert result.outputs["coaching_type"] == "health_assessment"

    def test_error_resilience(self, agile_coach_role):
        """Test error resilience in various scenarios."""
        from agentsmcp.roles.base import TaskEnvelope
        
        # Test with minimal payload
        minimal_task = TaskEnvelope(
            id="minimal_test",
            title="Minimal Test",
            description="Test with minimal data",
            payload={},
            meta=EnvelopeMeta()
        )
        
        result = AgileCoachRole.apply(minimal_task)
        assert result.status == EnvelopeStatus.SUCCESS
        assert result.outputs["coaching_type"] == "general"
        
        # Test with malformed coaching type
        malformed_task = TaskEnvelope(
            id="malformed_test",
            title="Malformed Test", 
            description="Test with invalid coaching type",
            payload={"coaching_type": None},
            meta=EnvelopeMeta()
        )
        
        result = AgileCoachRole.apply(malformed_task)
        # Should fall back to general coaching
        assert result.status == EnvelopeStatus.SUCCESS

    def test_output_schema_compliance(self, agile_coach_role):
        """Test that outputs comply with expected schemas."""
        from agentsmcp.roles.base import TaskEnvelope
        
        task = TaskEnvelope(
            id="schema_test",
            title="Schema Compliance Test",
            description="Test output schema compliance",
            payload={
                "coaching_type": "planning",
                "task_classification": {
                    "task_type": "implementation",
                    "complexity": "medium",
                    "risk_level": "low"
                },
                "team_composition": {
                    "primary_team": [{"role": "coder", "model_assignment": "ollama"}]
                }
            },
            meta=EnvelopeMeta()
        )
        
        result = AgileCoachRole.apply(task)
        
        # Verify result envelope structure
        assert hasattr(result, 'id')
        assert hasattr(result, 'role')
        assert hasattr(result, 'status')
        assert hasattr(result, 'decisions')
        assert hasattr(result, 'risks')
        assert hasattr(result, 'followups')
        assert hasattr(result, 'outputs')
        assert hasattr(result, 'errors')
        
        # Verify coaching-specific output structure
        assert "coaching_type" in result.outputs
        
        if result.outputs["coaching_type"] == "planning":
            assert "coach_actions" in result.outputs
            coach_actions = result.outputs["coach_actions"]
            assert "recommended_approach" in coach_actions
            assert "risk_mitigations" in coach_actions
            assert "estimated_velocity" in coach_actions

    @pytest.mark.asyncio
    async def test_integration_with_orchestration(self, agile_coach_role):
        """Test integration with orchestration system."""
        # This test verifies that the role can be used in orchestration scenarios
        
        # Test that the role has the required integration point
        assert hasattr(agile_coach_role, '_coach_integration')
        
        from agentsmcp.orchestration.agile_coach import AgileCoachIntegration
        assert isinstance(agile_coach_role._coach_integration, AgileCoachIntegration)
        
        # Test that coaching history is maintained
        initial_history_length = len(agile_coach_role._coach_integration.coaching_history)
        
        # Simulate coaching interaction (would normally be async)
        from agentsmcp.orchestration.models import TaskClassification, TeamComposition, AgentSpec
        
        task_classification = TaskClassification(
            task_type="implementation",
            complexity="medium", 
            required_roles=["coder"],
            estimated_effort=50,
            risk_level="low",
            confidence=0.8
        )
        
        team_composition = TeamComposition(
            primary_team=[AgentSpec(role="coder", model_assignment="ollama", priority=1)],
            load_order=["coder"],
            coordination_strategy="collaborative",
            confidence_score=0.8
        )
        
        # This would normally be called during role execution
        # Here we test the integration exists and works
        coach_actions = await agile_coach_role._coach_integration.coach_planning(
            task_classification,
            team_composition
        )
        
        # Verify coaching history was updated
        assert len(agile_coach_role._coach_integration.coaching_history) > initial_history_length
        
        # Verify coach actions structure
        assert hasattr(coach_actions, 'recommended_approach')
        assert hasattr(coach_actions, 'risk_mitigations')
        assert hasattr(coach_actions, 'estimated_velocity')
        assert hasattr(coach_actions, 'confidence_score')