"""Unit tests for the DynamicOrchestrator component."""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from src.agentsmcp.orchestration.dynamic_orchestrator import (
    DynamicOrchestrator,
    OrchestrationMetrics,
    FallbackConfig,
    OrchestrationError,
    TeamLoadError,
    ExecutionTimeoutError,
    InsufficientResourcesError,
)
from src.agentsmcp.orchestration.models import TeamComposition, CoordinationStrategy, AgentSpec, ResourceConstraints
from src.agentsmcp.orchestration.execution_engine import TeamExecution, ExecutionStatus, ExecutionProgress
from src.agentsmcp.roles.base import RoleName, ModelAssignment
from src.agentsmcp.models import TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus


# Mock classes for testing
class MockAgentManager:
    """Mock AgentManager for testing."""
    
    async def spawn_agent(self, agent_type, prompt, timeout=300):
        return f"job_{agent_type}"


@pytest.fixture
def mock_agent_manager():
    """Create mock agent manager."""
    return MockAgentManager()


@pytest.fixture
def fallback_config():
    """Create fallback configuration for testing."""
    return FallbackConfig(
        enable_fallbacks=True,
        max_retry_attempts=2,
        fallback_delay_seconds=0.1,  # Fast for testing
        use_fallback_agents=True,
        graceful_degradation=True,
    )


@pytest.fixture
def orchestrator(mock_agent_manager, fallback_config):
    """Create dynamic orchestrator for testing."""
    return DynamicOrchestrator(
        agent_manager=mock_agent_manager,
        resource_limits={
            "memory_mb": 2000,
            "max_agents": 10,
            "max_executions": 5,
        },
        fallback_config=fallback_config,
        performance_tracking=True,
    )


@pytest.fixture
def team_composition():
    """Create team composition for testing."""
    return TeamComposition(
        primary_team=[
            AgentSpec(role="architect", model_assignment="claude", priority=1),
            AgentSpec(role="coder", model_assignment="codex", priority=2),
        ],
        fallback_agents=[
            AgentSpec(role="architect", model_assignment="ollama", priority=1),
            AgentSpec(role="coder", model_assignment="ollama", priority=2),
        ],
        load_order=["architect", "coder"],
        coordination_strategy=CoordinationStrategy.SEQUENTIAL,
        confidence_score=0.9,
        rationale="Test team for orchestration",
    )


@pytest.fixture
def resource_constraints():
    """Create resource constraints for testing."""
    return ResourceConstraints(
        max_agents=3,
        memory_limit=500,
        time_budget=60,
        cost_budget=5.0,
    )


class TestOrchestrationMetrics:
    """Test OrchestrationMetrics functionality."""
    
    def test_metrics_initialization(self):
        """Test OrchestrationMetrics initialization."""
        metrics = OrchestrationMetrics()
        assert metrics.teams_orchestrated == 0
        assert metrics.successful_executions == 0
        assert metrics.failed_executions == 0
        assert metrics.average_execution_time == 0.0
        assert metrics.total_agents_loaded == 0
        assert isinstance(metrics.resource_utilization, dict)


class TestFallbackConfig:
    """Test FallbackConfig functionality."""
    
    def test_fallback_config_initialization(self):
        """Test FallbackConfig initialization."""
        config = FallbackConfig(
            enable_fallbacks=True,
            max_retry_attempts=3,
            fallback_delay_seconds=1.5,
            use_fallback_agents=True,
            graceful_degradation=False,
        )
        
        assert config.enable_fallbacks is True
        assert config.max_retry_attempts == 3
        assert config.fallback_delay_seconds == 1.5
        assert config.use_fallback_agents is True
        assert config.graceful_degradation is False


class TestDynamicOrchestrator:
    """Test DynamicOrchestrator functionality."""
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test DynamicOrchestrator initialization."""
        assert orchestrator.agent_manager is not None
        assert orchestrator.resource_manager is not None
        assert orchestrator.agent_loader is not None
        assert orchestrator.execution_engine is not None
        assert orchestrator.fallback_config.enable_fallbacks is True
        assert orchestrator.performance_tracking is True
    
    @pytest.mark.asyncio
    async def test_start_orchestrator(self, orchestrator):
        """Test starting the orchestrator."""
        await orchestrator.start()
        
        # Should have started background tasks
        assert orchestrator._maintenance_task is not None
        assert not orchestrator._maintenance_task.done()
        
        # Cleanup
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_validate_team_composition_success(self, orchestrator, team_composition):
        """Test successful team composition validation."""
        with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_role_class.return_value = Mock()  # Mock role class exists
            mock_registry_class.return_value = mock_registry
            
            # Should not raise exception
            await orchestrator._validate_team_composition(team_composition, None)
    
    @pytest.mark.asyncio
    async def test_validate_team_composition_empty_team(self, orchestrator):
        """Test validation failure with empty team."""
        empty_team = TeamComposition(
            primary_team=[],  # Empty team
            load_order=[],
            coordination_strategy=CoordinationStrategy.PARALLEL,
            confidence_score=0.0,
        )
        
        with pytest.raises(TeamLoadError, match="at least one primary agent"):
            await orchestrator._validate_team_composition(empty_team, None)
    
    @pytest.mark.asyncio
    async def test_validate_team_composition_unknown_role(self, orchestrator, team_composition):
        """Test validation failure with unknown role."""
        with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_role_class.return_value = None  # Role doesn't exist
            mock_registry_class.return_value = mock_registry
            
            with pytest.raises(TeamLoadError, match="Unknown role"):
                await orchestrator._validate_team_composition(team_composition, None)
    
    @pytest.mark.asyncio
    async def test_validate_team_composition_resource_constraints(self, orchestrator, team_composition):
        """Test validation with resource constraints."""
        tight_constraints = ResourceConstraints(max_agents=1)  # Less than team size
        
        with pytest.raises(InsufficientResourcesError):
            await orchestrator._validate_team_composition(team_composition, tight_constraints)
    
    @pytest.mark.asyncio
    async def test_generate_tasks_from_objective(self, orchestrator, team_composition):
        """Test task generation from objective."""
        objective = "Build a web application"
        
        tasks = await orchestrator._generate_tasks_from_objective(objective, team_composition)
        
        assert len(tasks) == len(team_composition.primary_team)
        assert all(isinstance(task, TaskEnvelopeV1) for task in tasks)
        assert all(objective in task.objective for task in tasks)
    
    @pytest.mark.asyncio
    async def test_estimate_team_resource_requirements(self, orchestrator, team_composition):
        """Test team resource requirement estimation."""
        requirements = await orchestrator._estimate_team_resource_requirements(team_composition)
        
        assert "memory" in [rt.value for rt in requirements.keys()]
        assert "agent_slots" in [rt.value for rt in requirements.keys()]
        
        # Should estimate resources for each agent
        from src.agentsmcp.orchestration.resource_manager import ResourceType
        memory_req = requirements.get(ResourceType.MEMORY, 0)
        agent_slots_req = requirements.get(ResourceType.AGENT_SLOTS, 0)
        
        assert memory_req > 0
        assert agent_slots_req == len(team_composition.primary_team)
    
    @pytest.mark.asyncio
    async def test_ensure_sufficient_resources_success(self, orchestrator, team_composition):
        """Test successful resource availability check."""
        # Should not raise exception with default limits
        await orchestrator._ensure_sufficient_resources(team_composition, None)
    
    @pytest.mark.asyncio
    async def test_ensure_sufficient_resources_failure(self, orchestrator, team_composition):
        """Test resource availability check failure."""
        # Exhaust resources first
        for resource_type, quota in orchestrator.resource_manager.quotas.items():
            quota.current_usage = quota.limit  # Max out all resources
        
        with pytest.raises(InsufficientResourcesError):
            await orchestrator._ensure_sufficient_resources(team_composition, None)
    
    @pytest.mark.asyncio
    async def test_apply_fallback_agents(self, orchestrator, team_composition):
        """Test applying fallback agents."""
        fallback_team = await orchestrator._apply_fallback_agents(team_composition)
        
        assert fallback_team.primary_team == team_composition.fallback_agents
        assert len(fallback_team.fallback_agents) == 0
        assert fallback_team.confidence_score < team_composition.confidence_score
    
    @pytest.mark.asyncio
    async def test_apply_fallback_agents_no_fallbacks(self, orchestrator):
        """Test applying fallback agents when none available."""
        team_without_fallbacks = TeamComposition(
            primary_team=[AgentSpec(role="coder", model_assignment="codex")],
            fallback_agents=[],  # No fallbacks
            load_order=["coder"],
            coordination_strategy=CoordinationStrategy.PARALLEL,
            confidence_score=0.9,
        )
        
        result = await orchestrator._apply_fallback_agents(team_without_fallbacks)
        
        # Should return the same team
        assert result is team_without_fallbacks
    
    def test_is_execution_successful(self, orchestrator):
        """Test execution success evaluation."""
        # Completed execution is successful
        completed_execution = TeamExecution(
            execution_id="test",
            team_composition=TeamComposition(
                primary_team=[AgentSpec(role="coder", model_assignment="codex")],
                load_order=["coder"],
                coordination_strategy=CoordinationStrategy.PARALLEL,
                confidence_score=0.9,
            ),
            objective="Test",
            status=ExecutionStatus.COMPLETED,
            progress=ExecutionProgress(total_tasks=1),
        )
        
        assert orchestrator._is_execution_successful(completed_execution) is True
        
        # Failed execution with graceful degradation
        failed_execution = TeamExecution(
            execution_id="test",
            team_composition=TeamComposition(
                primary_team=[AgentSpec(role="coder", model_assignment="codex")],
                load_order=["coder"],
                coordination_strategy=CoordinationStrategy.PARALLEL,
                confidence_score=0.9,
            ),
            objective="Test",
            status=ExecutionStatus.FAILED,
            progress=ExecutionProgress(total_tasks=10),
        )
        failed_execution.progress.completed_tasks = 6  # 60% success
        
        assert orchestrator._is_execution_successful(failed_execution) is True
        
        # Completely failed execution
        failed_execution.progress.completed_tasks = 2  # 20% success
        assert orchestrator._is_execution_successful(failed_execution) is False
    
    def test_update_average_execution_time(self, orchestrator):
        """Test updating average execution time."""
        orchestrator.metrics.teams_orchestrated = 2
        orchestrator.metrics.average_execution_time = 5.0
        
        # Add new execution time
        orchestrator._update_average_execution_time(7.0)
        
        # New average should be (5*2 + 7) / 3 = 5.67
        assert abs(orchestrator.metrics.average_execution_time - 5.67) < 0.01
    
    @patch('src.agentsmcp.orchestration.dynamic_orchestrator.DynamicOrchestrator._execute_with_fallbacks')
    @pytest.mark.asyncio
    async def test_orchestrate_team_success(self, mock_execute, orchestrator, team_composition):
        """Test successful team orchestration."""
        # Setup mock to return successful execution
        successful_execution = TeamExecution(
            execution_id="success-test",
            team_composition=team_composition,
            objective="Test objective",
            status=ExecutionStatus.COMPLETED,
            progress=ExecutionProgress(total_tasks=2),
        )
        mock_execute.return_value = successful_execution
        
        with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_role_class.return_value = Mock()
            mock_registry_class.return_value = mock_registry
            
            result = await orchestrator.orchestrate_team(
                team_spec=team_composition,
                objective="Test objective",
            )
            
            assert result.status == ExecutionStatus.COMPLETED
            assert orchestrator.metrics.successful_executions == 1
            assert len(orchestrator.orchestration_history) == 1
    
    @patch('src.agentsmcp.orchestration.dynamic_orchestrator.DynamicOrchestrator._execute_with_fallbacks')
    @pytest.mark.asyncio
    async def test_orchestrate_team_failure(self, mock_execute, orchestrator, team_composition):
        """Test team orchestration failure."""
        # Setup mock to raise exception
        mock_execute.side_effect = Exception("Execution failed")
        
        with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_role_class.return_value = Mock()
            mock_registry_class.return_value = mock_registry
            
            with pytest.raises(OrchestrationError):
                await orchestrator.orchestrate_team(
                    team_spec=team_composition,
                    objective="Test objective",
                )
            
            assert orchestrator.metrics.failed_executions == 1
    
    @patch('src.agentsmcp.orchestration.dynamic_orchestrator.DynamicOrchestrator.execution_engine')
    @pytest.mark.asyncio
    async def test_execute_with_fallbacks_retry_mechanism(self, mock_execution_engine, orchestrator, team_composition):
        """Test fallback retry mechanism."""
        # Setup mock to fail first time, succeed second time
        failed_execution = TeamExecution(
            execution_id="retry-test",
            team_composition=team_composition,
            objective="Retry test",
            status=ExecutionStatus.FAILED,
            progress=ExecutionProgress(total_tasks=1),
        )
        failed_execution.progress.failed_tasks = 1
        
        successful_execution = TeamExecution(
            execution_id="retry-test",
            team_composition=team_composition,
            objective="Retry test",
            status=ExecutionStatus.COMPLETED,
            progress=ExecutionProgress(total_tasks=1),
        )
        successful_execution.progress.completed_tasks = 1
        
        mock_execution_engine.execute_team.side_effect = [failed_execution, successful_execution]
        
        with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_role_class.return_value = Mock()
            mock_registry_class.return_value = mock_registry
            
            result = await orchestrator._execute_with_fallbacks(
                orchestration_id="retry-test",
                team_spec=team_composition,
                objective="Retry test",
                tasks=[TaskEnvelopeV1(objective="Test task")],
                progress_callback=None,
                timeout_seconds=None,
            )
            
            assert result.status == ExecutionStatus.COMPLETED
            assert mock_execution_engine.execute_team.call_count == 2
    
    def test_get_orchestration_status_active(self, orchestrator):
        """Test getting status of active orchestration."""
        execution = TeamExecution(
            execution_id="active-test",
            team_composition=TeamComposition(
                primary_team=[AgentSpec(role="coder", model_assignment="codex")],
                load_order=["coder"],
                coordination_strategy=CoordinationStrategy.PARALLEL,
                confidence_score=0.8,
            ),
            objective="Active test",
            status=ExecutionStatus.RUNNING,
            progress=ExecutionProgress(total_tasks=1),
        )
        
        orchestrator.active_orchestrations["active-test"] = execution
        
        retrieved = orchestrator.get_orchestration_status("active-test")
        assert retrieved is not None
        assert retrieved.execution_id == "active-test"
    
    def test_get_orchestration_status_history(self, orchestrator):
        """Test getting status from orchestration history."""
        execution = TeamExecution(
            execution_id="history-test",
            team_composition=TeamComposition(
                primary_team=[AgentSpec(role="coder", model_assignment="codex")],
                load_order=["coder"],
                coordination_strategy=CoordinationStrategy.PARALLEL,
                confidence_score=0.8,
            ),
            objective="History test",
            status=ExecutionStatus.COMPLETED,
            progress=ExecutionProgress(total_tasks=1),
        )
        
        orchestrator.orchestration_history.append(execution)
        
        retrieved = orchestrator.get_orchestration_status("history-test")
        assert retrieved is not None
        assert retrieved.execution_id == "history-test"
    
    def test_get_orchestration_metrics(self, orchestrator):
        """Test getting orchestration metrics."""
        orchestrator.metrics.teams_orchestrated = 5
        orchestrator.metrics.successful_executions = 4
        orchestrator.metrics.failed_executions = 1
        
        metrics = orchestrator.get_orchestration_metrics()
        
        assert "orchestrator_metrics" in metrics
        assert "resource_manager_status" in metrics
        assert "agent_loader_stats" in metrics
        assert "execution_engine_stats" in metrics
        
        orch_metrics = metrics["orchestrator_metrics"]
        assert orch_metrics["teams_orchestrated"] == 5
        assert orch_metrics["success_rate"] == 0.8  # 4/5
    
    @pytest.mark.asyncio
    async def test_cancel_orchestration(self, orchestrator):
        """Test canceling an orchestration."""
        execution = TeamExecution(
            execution_id="cancel-test",
            team_composition=TeamComposition(
                primary_team=[AgentSpec(role="coder", model_assignment="codex")],
                load_order=["coder"],
                coordination_strategy=CoordinationStrategy.PARALLEL,
                confidence_score=0.8,
            ),
            objective="Cancel test",
            status=ExecutionStatus.RUNNING,
            progress=ExecutionProgress(total_tasks=1),
        )
        
        orchestrator.active_orchestrations["cancel-test"] = execution
        
        with patch.object(orchestrator.execution_engine, 'cancel_execution') as mock_cancel:
            mock_cancel.return_value = True
            
            cancelled = await orchestrator.cancel_orchestration("cancel-test")
            
            assert cancelled is True
            assert "cancel-test" not in orchestrator.active_orchestrations
    
    @pytest.mark.asyncio
    async def test_health_check(self, orchestrator):
        """Test health check functionality."""
        health = await orchestrator.health_check()
        
        assert "status" in health
        assert "timestamp" in health
        assert "components" in health
        
        components = health["components"]
        assert "resource_manager" in components
        assert "agent_loader" in components
        assert "execution_engine" in components
    
    @pytest.mark.asyncio
    async def test_shutdown(self, orchestrator):
        """Test orchestrator shutdown."""
        await orchestrator.start()
        
        # Add some active orchestrations
        orchestrator.active_orchestrations["test1"] = Mock()
        orchestrator.active_orchestrations["test2"] = Mock()
        
        await orchestrator.shutdown()
        
        assert orchestrator._shutdown is True
        assert len(orchestrator.active_orchestrations) == 0


# Golden tests as specified in ICD
class TestDynamicOrchestratorGoldenTests:
    """Golden tests for DynamicOrchestrator as specified in ICD."""
    
    @pytest.mark.asyncio
    async def test_golden_concurrent_team_executions(self, mock_agent_manager):
        """Golden test: Support 50 concurrent team executions."""
        orchestrator = DynamicOrchestrator(
            agent_manager=mock_agent_manager,
            resource_limits={
                "memory_mb": 20000,  # High limits for concurrent test
                "max_agents": 200,
                "max_executions": 60,
            },
        )
        
        await orchestrator.start()
        
        try:
            with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
                mock_registry = Mock()
                mock_registry.get_role_class.return_value = Mock()
                mock_registry_class.return_value = mock_registry
                
                # Mock execution engine to return quick successful results
                with patch.object(orchestrator.execution_engine, 'execute_team') as mock_execute:
                    async def quick_execution(*args, **kwargs):
                        return TeamExecution(
                            execution_id=f"concurrent-{len(orchestrator.orchestration_history)}",
                            team_composition=kwargs.get('team_composition'),
                            objective=kwargs.get('objective', 'Test'),
                            status=ExecutionStatus.COMPLETED,
                            progress=ExecutionProgress(total_tasks=1),
                        )
                    
                    mock_execute.side_effect = quick_execution
                    
                    # Create concurrent orchestration tasks
                    async def orchestrate_team_task(i):
                        team_comp = TeamComposition(
                            primary_team=[AgentSpec(role="coder", model_assignment="codex")],
                            load_order=["coder"],
                            coordination_strategy=CoordinationStrategy.PARALLEL,
                            confidence_score=0.8,
                        )
                        
                        return await orchestrator.orchestrate_team(
                            team_spec=team_comp,
                            objective=f"Concurrent test {i}",
                        )
                    
                    # Run 50 concurrent orchestrations
                    tasks = [orchestrate_team_task(i) for i in range(50)]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # All should succeed
                    successful = [r for r in results if not isinstance(r, Exception)]
                    assert len(successful) == 50
                    assert orchestrator.metrics.successful_executions == 50
        
        finally:
            await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_golden_agent_loading_performance(self, mock_agent_manager):
        """Golden test: Agent loading within 2s per agent."""
        orchestrator = DynamicOrchestrator(
            agent_manager=mock_agent_manager,
            resource_limits={"memory_mb": 10000, "max_agents": 20},
        )
        
        await orchestrator.start()
        
        try:
            # Create large team to test loading performance
            large_team = TeamComposition(
                primary_team=[
                    AgentSpec(role="architect", model_assignment="claude"),
                    AgentSpec(role="coder", model_assignment="codex"),
                    AgentSpec(role="qa", model_assignment="claude"),
                    AgentSpec(role="docs", model_assignment="ollama"),
                ],
                load_order=["architect", "coder", "qa", "docs"],
                coordination_strategy=CoordinationStrategy.PARALLEL,
                confidence_score=0.9,
            )
            
            with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
                mock_registry = Mock()
                mock_registry.get_role_class.return_value = Mock()
                mock_registry_class.return_value = mock_registry
                
                # Mock execution engine with timing
                with patch.object(orchestrator.execution_engine, 'execute_team') as mock_execute:
                    async def timed_execution(*args, **kwargs):
                        # Simulate agent loading time
                        await asyncio.sleep(0.1)  # Fast for testing
                        return TeamExecution(
                            execution_id="perf-test",
                            team_composition=kwargs.get('team_composition'),
                            objective="Performance test",
                            status=ExecutionStatus.COMPLETED,
                            progress=ExecutionProgress(total_tasks=4),
                        )
                    
                    mock_execute.side_effect = timed_execution
                    
                    # Measure orchestration time
                    import time
                    start_time = time.time()
                    
                    result = await orchestrator.orchestrate_team(
                        team_spec=large_team,
                        objective="Performance test",
                    )
                    
                    total_time = time.time() - start_time
                    
                    assert result.status == ExecutionStatus.COMPLETED
                    # Should be well under 2s per agent (8s total for 4 agents)
                    assert total_time < 2.0  # Fast mock execution
        
        finally:
            await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_golden_fallback_mechanisms(self, mock_agent_manager):
        """Golden test: Comprehensive fallback and recovery mechanisms."""
        fallback_config = FallbackConfig(
            enable_fallbacks=True,
            max_retry_attempts=3,
            fallback_delay_seconds=0.1,
            use_fallback_agents=True,
            graceful_degradation=True,
        )
        
        orchestrator = DynamicOrchestrator(
            agent_manager=mock_agent_manager,
            fallback_config=fallback_config,
        )
        
        await orchestrator.start()
        
        try:
            team_with_fallbacks = TeamComposition(
                primary_team=[AgentSpec(role="coder", model_assignment="claude")],
                fallback_agents=[AgentSpec(role="coder", model_assignment="ollama")],
                load_order=["coder"],
                coordination_strategy=CoordinationStrategy.PARALLEL,
                confidence_score=0.9,
            )
            
            with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
                mock_registry = Mock()
                mock_registry.get_role_class.return_value = Mock()
                mock_registry_class.return_value = mock_registry
                
                # Mock execution to fail first attempts, succeed with fallback
                call_count = 0
                with patch.object(orchestrator.execution_engine, 'execute_team') as mock_execute:
                    async def fallback_execution(*args, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        
                        if call_count == 1:
                            # First attempt fails
                            execution = TeamExecution(
                                execution_id="fallback-test",
                                team_composition=kwargs.get('team_composition'),
                                objective="Fallback test",
                                status=ExecutionStatus.FAILED,
                                progress=ExecutionProgress(total_tasks=1),
                            )
                            execution.progress.failed_tasks = 1
                            return execution
                        else:
                            # Fallback succeeds
                            execution = TeamExecution(
                                execution_id="fallback-test",
                                team_composition=kwargs.get('team_composition'),
                                objective="Fallback test",
                                status=ExecutionStatus.COMPLETED,
                                progress=ExecutionProgress(total_tasks=1),
                            )
                            execution.progress.completed_tasks = 1
                            return execution
                    
                    mock_execute.side_effect = fallback_execution
                    
                    result = await orchestrator.orchestrate_team(
                        team_spec=team_with_fallbacks,
                        objective="Fallback test",
                    )
                    
                    # Should succeed with fallback
                    assert result.status == ExecutionStatus.COMPLETED
                    assert mock_execute.call_count == 2  # Initial attempt + fallback
        
        finally:
            await orchestrator.shutdown()


# Edge case tests (2 additional as specified)
class TestDynamicOrchestratorEdgeCases:
    """Additional edge case tests for DynamicOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_edge_case_resource_exhaustion_during_orchestration(self, mock_agent_manager):
        """Edge case: Handle resource exhaustion during orchestration."""
        # Create orchestrator with very limited resources
        orchestrator = DynamicOrchestrator(
            agent_manager=mock_agent_manager,
            resource_limits={
                "memory_mb": 100,  # Very limited
                "max_agents": 1,
                "max_executions": 1,
            },
        )
        
        await orchestrator.start()
        
        try:
            # Create team that exceeds resource limits
            large_team = TeamComposition(
                primary_team=[
                    AgentSpec(role="architect", model_assignment="claude"),
                    AgentSpec(role="coder", model_assignment="codex"),
                    AgentSpec(role="qa", model_assignment="claude"),
                ],  # 3 agents but limit is 1
                load_order=["architect", "coder", "qa"],
                coordination_strategy=CoordinationStrategy.PARALLEL,
                confidence_score=0.9,
            )
            
            with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
                mock_registry = Mock()
                mock_registry.get_role_class.return_value = Mock()
                mock_registry_class.return_value = mock_registry
                
                # Should fail with resource exhaustion
                with pytest.raises(InsufficientResourcesError):
                    await orchestrator.orchestrate_team(
                        team_spec=large_team,
                        objective="Resource exhaustion test",
                    )
        
        finally:
            await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_edge_case_maintenance_loop_error_handling(self, mock_agent_manager):
        """Edge case: Maintenance loop error handling doesn't crash orchestrator."""
        orchestrator = DynamicOrchestrator(agent_manager=mock_agent_manager)
        
        # Mock resource manager to raise exception during maintenance
        with patch.object(orchestrator.resource_manager, 'cleanup_expired_allocations') as mock_cleanup:
            mock_cleanup.side_effect = Exception("Maintenance error")
            
            await orchestrator.start()
            
            # Wait for maintenance loop to run and handle error
            await asyncio.sleep(0.1)
            
            # Orchestrator should still be functional
            assert orchestrator._maintenance_task is not None
            assert not orchestrator._shutdown
            
            # Should be able to get metrics despite maintenance error
            metrics = orchestrator.get_orchestration_metrics()
            assert "orchestrator_metrics" in metrics
            
            await orchestrator.shutdown()