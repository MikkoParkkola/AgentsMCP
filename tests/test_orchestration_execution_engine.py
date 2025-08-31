"""Unit tests for the ExecutionEngine component."""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from src.agentsmcp.orchestration.execution_engine import (
    ExecutionEngine,
    ExecutionStatus,
    ExecutionTask,
    ExecutionProgress,
    TeamExecution,
    TaskPriority,
    ParallelExecution,
    SequentialExecution,
    HierarchicalExecution,
)
from src.agentsmcp.orchestration.models import TeamComposition, CoordinationStrategy, AgentSpec
from src.agentsmcp.orchestration.agent_loader import AgentLoader
from src.agentsmcp.orchestration.resource_manager import ResourceManager
from src.agentsmcp.roles.base import RoleName, ModelAssignment, BaseRole
from src.agentsmcp.models import TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus


# Mock classes for testing
class MockAgentManager:
    """Mock AgentManager for testing."""
    
    async def spawn_agent(self, agent_type, prompt, timeout=300):
        return f"job_{agent_type}"


class MockRole(BaseRole):
    """Mock role implementation for testing."""
    
    @classmethod
    def name(cls):
        return RoleName.CODER
    
    async def execute(self, task_envelope, agent_manager, timeout=None, max_retries=1):
        # Simulate successful execution
        return ResultEnvelopeV1(
            status=EnvelopeStatus.SUCCESS,
            artifacts={"output": f"Completed: {task_envelope.objective}"},
            confidence=0.8,
            notes="Mock execution successful",
        )


@pytest.fixture
def mock_agent_manager():
    """Create mock agent manager."""
    return MockAgentManager()


@pytest.fixture
def resource_manager():
    """Create resource manager for testing."""
    return ResourceManager(
        memory_limit_mb=2000,
        max_concurrent_agents=10,
    )


@pytest.fixture
def agent_loader(mock_agent_manager, resource_manager):
    """Create agent loader for testing."""
    loader = AgentLoader(
        agent_manager=mock_agent_manager,
        resource_manager=resource_manager,
    )
    return loader


@pytest.fixture
def execution_engine(agent_loader, resource_manager, mock_agent_manager):
    """Create execution engine for testing."""
    return ExecutionEngine(
        agent_loader=agent_loader,
        resource_manager=resource_manager,
        agent_manager=mock_agent_manager,
    )


@pytest.fixture
def team_composition():
    """Create team composition for testing."""
    return TeamComposition(
        primary_team=[
            AgentSpec(role="coder", model_assignment="codex", priority=1),
            AgentSpec(role="qa", model_assignment="claude", priority=2),
        ],
        load_order=["coder", "qa"],
        coordination_strategy=CoordinationStrategy.PARALLEL,
        confidence_score=0.9,
        rationale="Test team composition",
    )


@pytest.fixture
def sample_tasks():
    """Create sample tasks for testing."""
    return [
        TaskEnvelopeV1(
            objective="Implement feature A",
            inputs={"feature": "A"},
        ),
        TaskEnvelopeV1(
            objective="Test feature A", 
            inputs={"feature": "A"},
        ),
    ]


class TestExecutionStatus:
    """Test ExecutionStatus enum."""
    
    def test_execution_statuses(self):
        """Test all execution status values."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.CANCELLED.value == "cancelled"
        assert ExecutionStatus.TIMEOUT.value == "timeout"


class TestTaskPriority:
    """Test TaskPriority enum."""
    
    def test_task_priorities(self):
        """Test task priority values."""
        assert TaskPriority.CRITICAL.value == 1
        assert TaskPriority.HIGH.value == 2
        assert TaskPriority.MEDIUM.value == 3
        assert TaskPriority.LOW.value == 4
        assert TaskPriority.BACKGROUND.value == 5


class TestExecutionTask:
    """Test ExecutionTask functionality."""
    
    def test_execution_task_initialization(self, sample_tasks):
        """Test ExecutionTask initialization."""
        task = ExecutionTask(
            task_id="test-task",
            agent_spec=AgentSpec(role="coder", model_assignment="codex"),
            envelope=sample_tasks[0],
            priority=TaskPriority.HIGH,
        )
        
        assert task.task_id == "test-task"
        assert task.priority == TaskPriority.HIGH
        assert task.timeout_seconds == 300
        assert task.retries_remaining == 2
        assert task.started_at is None
        assert task.completed_at is None


class TestExecutionProgress:
    """Test ExecutionProgress functionality."""
    
    def test_progress_initialization(self):
        """Test ExecutionProgress initialization."""
        progress = ExecutionProgress(total_tasks=10)
        assert progress.total_tasks == 10
        assert progress.completed_tasks == 0
        assert progress.failed_tasks == 0
        assert progress.running_tasks == 0
    
    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        progress = ExecutionProgress(total_tasks=10)
        progress.completed_tasks = 3
        progress.failed_tasks = 2
        
        assert progress.completion_percentage == 30.0  # 3/10 * 100
    
    def test_is_complete(self):
        """Test completion check."""
        progress = ExecutionProgress(total_tasks=5)
        assert progress.is_complete is False
        
        progress.completed_tasks = 3
        progress.failed_tasks = 2
        assert progress.is_complete is True


class TestTeamExecution:
    """Test TeamExecution functionality."""
    
    def test_team_execution_initialization(self, team_composition):
        """Test TeamExecution initialization."""
        execution = TeamExecution(
            execution_id="test-exec",
            team_composition=team_composition,
            objective="Test objective",
            status=ExecutionStatus.PENDING,
            progress=ExecutionProgress(total_tasks=2),
        )
        
        assert execution.execution_id == "test-exec"
        assert execution.objective == "Test objective"
        assert execution.status == ExecutionStatus.PENDING
        assert execution.progress.total_tasks == 2


class TestParallelExecution:
    """Test ParallelExecution strategy."""
    
    @pytest.mark.asyncio
    async def test_parallel_execution_initialization(self, agent_loader, resource_manager, mock_agent_manager):
        """Test ParallelExecution initialization."""
        strategy = ParallelExecution(
            agent_loader=agent_loader,
            resource_manager=resource_manager,
            agent_manager=mock_agent_manager,
            max_concurrent_tasks=5,
        )
        
        assert strategy.max_concurrent_tasks == 5
    
    @patch('src.agentsmcp.orchestration.execution_engine.ExecutionStrategy._run_task_with_agent')
    @pytest.mark.asyncio
    async def test_parallel_execution_success(self, mock_run_task, agent_loader, resource_manager, mock_agent_manager, team_composition, sample_tasks):
        """Test successful parallel execution."""
        # Setup mock
        mock_run_task.return_value = None  # Simulate successful task execution
        
        strategy = ParallelExecution(
            agent_loader=agent_loader,
            resource_manager=resource_manager,
            agent_manager=mock_agent_manager,
        )
        
        # Create execution
        execution = TeamExecution(
            execution_id="parallel-test",
            team_composition=team_composition,
            objective="Parallel test",
            status=ExecutionStatus.PENDING,
            progress=ExecutionProgress(total_tasks=2),
        )
        
        # Add tasks
        execution.tasks = {
            "task1": ExecutionTask(
                task_id="task1",
                agent_spec=team_composition.primary_team[0],
                envelope=sample_tasks[0],
            ),
            "task2": ExecutionTask(
                task_id="task2",
                agent_spec=team_composition.primary_team[1],
                envelope=sample_tasks[1],
            ),
        }
        
        # Execute
        result = await strategy.execute(execution)
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.started_at is not None
        assert result.completed_at is not None
        assert mock_run_task.call_count == 2
    
    @patch('src.agentsmcp.orchestration.execution_engine.ExecutionStrategy._run_task_with_agent')
    @pytest.mark.asyncio
    async def test_parallel_execution_with_failures(self, mock_run_task, agent_loader, resource_manager, mock_agent_manager, team_composition, sample_tasks):
        """Test parallel execution with some task failures."""
        # Setup mock to fail on second call
        mock_run_task.side_effect = [None, Exception("Task failed")]
        
        strategy = ParallelExecution(
            agent_loader=agent_loader,
            resource_manager=resource_manager,
            agent_manager=mock_agent_manager,
        )
        
        # Create execution
        execution = TeamExecution(
            execution_id="parallel-fail-test",
            team_composition=team_composition,
            objective="Parallel fail test",
            status=ExecutionStatus.PENDING,
            progress=ExecutionProgress(total_tasks=2),
        )
        
        # Add tasks
        execution.tasks = {
            "task1": ExecutionTask(
                task_id="task1",
                agent_spec=team_composition.primary_team[0],
                envelope=sample_tasks[0],
            ),
            "task2": ExecutionTask(
                task_id="task2",
                agent_spec=team_composition.primary_team[1],
                envelope=sample_tasks[1],
            ),
        }
        
        # Execute
        result = await strategy.execute(execution)
        
        assert result.status == ExecutionStatus.FAILED
        assert result.progress.failed_tasks > 0
        assert len(result.errors) > 0


class TestSequentialExecution:
    """Test SequentialExecution strategy."""
    
    @patch('src.agentsmcp.orchestration.execution_engine.ExecutionStrategy._run_task_with_agent')
    @pytest.mark.asyncio
    async def test_sequential_execution_success(self, mock_run_task, agent_loader, resource_manager, mock_agent_manager, team_composition, sample_tasks):
        """Test successful sequential execution."""
        mock_run_task.return_value = None
        
        strategy = SequentialExecution(
            agent_loader=agent_loader,
            resource_manager=resource_manager,
            agent_manager=mock_agent_manager,
        )
        
        # Create execution
        execution = TeamExecution(
            execution_id="sequential-test",
            team_composition=team_composition,
            objective="Sequential test",
            status=ExecutionStatus.PENDING,
            progress=ExecutionProgress(total_tasks=2),
        )
        
        # Add tasks
        execution.tasks = {
            "task1": ExecutionTask(
                task_id="task1",
                agent_spec=team_composition.primary_team[0],
                envelope=sample_tasks[0],
                priority=TaskPriority.HIGH,
            ),
            "task2": ExecutionTask(
                task_id="task2",
                agent_spec=team_composition.primary_team[1],
                envelope=sample_tasks[1],
                priority=TaskPriority.MEDIUM,
            ),
        }
        
        # Execute
        result = await strategy.execute(execution)
        
        assert result.status == ExecutionStatus.COMPLETED
        assert mock_run_task.call_count == 2
    
    def test_sort_tasks_for_sequential_execution(self, agent_loader, resource_manager, mock_agent_manager):
        """Test task sorting for sequential execution."""
        strategy = SequentialExecution(
            agent_loader=agent_loader,
            resource_manager=resource_manager,
            agent_manager=mock_agent_manager,
        )
        
        tasks = {
            "low": ExecutionTask(
                task_id="low",
                agent_spec=AgentSpec(role="coder", model_assignment="codex"),
                envelope=TaskEnvelopeV1(objective="Low priority task"),
                priority=TaskPriority.LOW,
            ),
            "high": ExecutionTask(
                task_id="high",
                agent_spec=AgentSpec(role="coder", model_assignment="codex"),
                envelope=TaskEnvelopeV1(objective="High priority task"),
                priority=TaskPriority.HIGH,
            ),
            "critical": ExecutionTask(
                task_id="critical",
                agent_spec=AgentSpec(role="coder", model_assignment="codex"),
                envelope=TaskEnvelopeV1(objective="Critical task"),
                priority=TaskPriority.CRITICAL,
            ),
        }
        
        sorted_tasks = strategy._sort_tasks_for_sequential_execution(tasks)
        
        # Should be sorted by priority (critical first)
        assert sorted_tasks[0].task_id == "critical"
        assert sorted_tasks[1].task_id == "high" 
        assert sorted_tasks[2].task_id == "low"


class TestHierarchicalExecution:
    """Test HierarchicalExecution strategy."""
    
    @patch('src.agentsmcp.orchestration.execution_engine.HierarchicalExecution._load_coordination_agent')
    @patch('src.agentsmcp.orchestration.execution_engine.HierarchicalExecution._create_coordination_plan')
    @patch('src.agentsmcp.orchestration.execution_engine.HierarchicalExecution._execute_phase')
    @pytest.mark.asyncio
    async def test_hierarchical_execution_success(self, mock_execute_phase, mock_create_plan, mock_load_coordinator, 
                                                 agent_loader, resource_manager, mock_agent_manager, team_composition, sample_tasks):
        """Test successful hierarchical execution."""
        # Setup mocks
        mock_coordinator = MockRole()
        mock_load_coordinator.return_value = mock_coordinator
        
        mock_plan = {
            "phases": [
                {"name": "phase1", "tasks": ["task1", "task2"], "strategy": "parallel"}
            ]
        }
        mock_create_plan.return_value = mock_plan
        mock_execute_phase.return_value = None
        
        strategy = HierarchicalExecution(
            agent_loader=agent_loader,
            resource_manager=resource_manager,
            agent_manager=mock_agent_manager,
        )
        
        # Create execution
        execution = TeamExecution(
            execution_id="hierarchical-test",
            team_composition=team_composition,
            objective="Hierarchical test",
            status=ExecutionStatus.PENDING,
            progress=ExecutionProgress(total_tasks=2),
        )
        
        # Add tasks
        execution.tasks = {
            "task1": ExecutionTask(
                task_id="task1",
                agent_spec=team_composition.primary_team[0],
                envelope=sample_tasks[0],
            ),
            "task2": ExecutionTask(
                task_id="task2",
                agent_spec=team_composition.primary_team[1],
                envelope=sample_tasks[1],
            ),
        }
        
        # Execute
        result = await strategy.execute(execution)
        
        assert result.status == ExecutionStatus.COMPLETED
        assert mock_load_coordinator.called
        assert mock_create_plan.called
        assert mock_execute_phase.called


class TestExecutionEngine:
    """Test ExecutionEngine functionality."""
    
    def test_execution_engine_initialization(self, execution_engine):
        """Test ExecutionEngine initialization."""
        assert CoordinationStrategy.PARALLEL in execution_engine.strategies
        assert CoordinationStrategy.SEQUENTIAL in execution_engine.strategies
        assert CoordinationStrategy.HIERARCHICAL in execution_engine.strategies
        assert len(execution_engine.active_executions) == 0
    
    @patch('src.agentsmcp.orchestration.execution_engine.ExecutionEngine._run_task_with_agent')
    @pytest.mark.asyncio
    async def test_execute_team_success(self, mock_run_task, execution_engine, team_composition, sample_tasks):
        """Test successful team execution."""
        mock_run_task.return_value = None
        
        result = await execution_engine.execute_team(
            team_composition=team_composition,
            objective="Test team execution",
            tasks=sample_tasks,
        )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.objective == "Test team execution"
        assert len(result.tasks) == 2
    
    @pytest.mark.asyncio
    async def test_execute_team_with_timeout(self, execution_engine, team_composition, sample_tasks):
        """Test team execution with timeout."""
        # Create a team composition that will take too long
        with patch('src.agentsmcp.orchestration.execution_engine.ParallelExecution.execute') as mock_execute:
            async def slow_execution(*args, **kwargs):
                await asyncio.sleep(2)  # Longer than timeout
                return TeamExecution(
                    execution_id="timeout-test",
                    team_composition=team_composition,
                    objective="Timeout test",
                    status=ExecutionStatus.COMPLETED,
                    progress=ExecutionProgress(total_tasks=2),
                )
            
            mock_execute.side_effect = slow_execution
            
            result = await execution_engine.execute_team(
                team_composition=team_composition,
                objective="Timeout test",
                tasks=sample_tasks,
                timeout_seconds=1,  # Very short timeout
            )
            
            assert result.status == ExecutionStatus.TIMEOUT
    
    def test_get_execution_status(self, execution_engine):
        """Test getting execution status."""
        # Add a mock execution
        execution = TeamExecution(
            execution_id="status-test",
            team_composition=TeamComposition(
                primary_team=[AgentSpec(role="coder", model_assignment="codex")],
                load_order=["coder"],
                coordination_strategy=CoordinationStrategy.PARALLEL,
                confidence_score=0.8,
            ),
            objective="Status test",
            status=ExecutionStatus.RUNNING,
            progress=ExecutionProgress(total_tasks=1),
        )
        
        execution_engine.active_executions["status-test"] = execution
        
        retrieved = execution_engine.get_execution_status("status-test")
        assert retrieved is not None
        assert retrieved.execution_id == "status-test"
        assert retrieved.status == ExecutionStatus.RUNNING
    
    def test_get_active_executions(self, execution_engine):
        """Test getting all active executions."""
        # Add mock executions
        for i in range(3):
            execution = TeamExecution(
                execution_id=f"active-{i}",
                team_composition=TeamComposition(
                    primary_team=[AgentSpec(role="coder", model_assignment="codex")],
                    load_order=["coder"],
                    coordination_strategy=CoordinationStrategy.PARALLEL,
                    confidence_score=0.8,
                ),
                objective=f"Active test {i}",
                status=ExecutionStatus.RUNNING,
                progress=ExecutionProgress(total_tasks=1),
            )
            execution_engine.active_executions[f"active-{i}"] = execution
        
        active = execution_engine.get_active_executions()
        assert len(active) == 3
        assert "active-0" in active
        assert "active-2" in active
    
    @pytest.mark.asyncio
    async def test_cancel_execution(self, execution_engine):
        """Test canceling an execution."""
        # Add a running execution
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
        
        execution_engine.active_executions["cancel-test"] = execution
        
        # Cancel execution
        cancelled = await execution_engine.cancel_execution("cancel-test")
        
        assert cancelled is True
        assert execution.status == ExecutionStatus.CANCELLED


# Golden tests as specified in ICD
class TestExecutionEngineGoldenTests:
    """Golden tests for ExecutionEngine as specified in ICD."""
    
    @patch('src.agentsmcp.orchestration.execution_engine.ExecutionEngine._run_task_with_agent')
    @pytest.mark.asyncio
    async def test_golden_parallel_execution_performance(self, mock_run_task, execution_engine):
        """Golden test: Parallel execution of multiple agents."""
        mock_run_task.return_value = None
        
        # Create large team composition
        large_team = TeamComposition(
            primary_team=[
                AgentSpec(role="coder", model_assignment="codex", priority=i)
                for i in range(1, 11)  # 10 agents
            ],
            load_order=[f"coder" for _ in range(10)],
            coordination_strategy=CoordinationStrategy.PARALLEL,
            confidence_score=0.9,
        )
        
        # Create tasks for all agents
        tasks = [
            TaskEnvelopeV1(objective=f"Task {i}", inputs={"task_id": i})
            for i in range(10)
        ]
        
        # Execute with timing
        import time
        start_time = time.time()
        
        result = await execution_engine.execute_team(
            team_composition=large_team,
            objective="Large team parallel test",
            tasks=tasks,
        )
        
        execution_time = time.time() - start_time
        
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.tasks) == 10
        assert execution_time < 10.0  # Should complete within reasonable time
    
    @patch('src.agentsmcp.orchestration.execution_engine.ExecutionEngine._run_task_with_agent')
    @pytest.mark.asyncio
    async def test_golden_sequential_execution_order(self, mock_run_task, execution_engine):
        """Golden test: Sequential execution maintains proper order."""
        execution_order = []
        
        async def track_execution(task, execution):
            execution_order.append(task.task_id)
            
        mock_run_task.side_effect = track_execution
        
        # Create team with sequential strategy
        sequential_team = TeamComposition(
            primary_team=[
                AgentSpec(role="architect", model_assignment="claude", priority=1),
                AgentSpec(role="coder", model_assignment="codex", priority=2),
                AgentSpec(role="qa", model_assignment="claude", priority=3),
            ],
            load_order=["architect", "coder", "qa"],
            coordination_strategy=CoordinationStrategy.SEQUENTIAL,
            confidence_score=0.9,
        )
        
        tasks = [
            TaskEnvelopeV1(objective="Design system", inputs={}),
            TaskEnvelopeV1(objective="Implement code", inputs={}),
            TaskEnvelopeV1(objective="Test implementation", inputs={}),
        ]
        
        result = await execution_engine.execute_team(
            team_composition=sequential_team,
            objective="Sequential workflow test",
            tasks=tasks,
        )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert len(execution_order) == 3
        # Tasks should be executed in priority order
        assert execution_order == ["task_0", "task_1", "task_2"]
    
    @patch('src.agentsmcp.orchestration.execution_engine.ExecutionEngine._run_task_with_agent')
    @pytest.mark.asyncio
    async def test_golden_error_recovery_mechanisms(self, mock_run_task, execution_engine):
        """Golden test: Proper error handling and recovery."""
        # Setup mock to fail on specific tasks
        def failing_execution(task, execution):
            if task.task_id == "task_1":  # Fail middle task
                raise Exception("Simulated task failure")
        
        mock_run_task.side_effect = failing_execution
        
        team_composition = TeamComposition(
            primary_team=[
                AgentSpec(role="coder", model_assignment="codex", priority=1),
                AgentSpec(role="qa", model_assignment="claude", priority=2),
                AgentSpec(role="docs", model_assignment="ollama", priority=3),
            ],
            load_order=["coder", "qa", "docs"],
            coordination_strategy=CoordinationStrategy.PARALLEL,
            confidence_score=0.8,
        )
        
        tasks = [
            TaskEnvelopeV1(objective="Task 0", inputs={}),
            TaskEnvelopeV1(objective="Task 1 (will fail)", inputs={}),
            TaskEnvelopeV1(objective="Task 2", inputs={}),
        ]
        
        result = await execution_engine.execute_team(
            team_composition=team_composition,
            objective="Error recovery test",
            tasks=tasks,
        )
        
        # Should handle partial failures gracefully
        assert result.status == ExecutionStatus.FAILED
        assert result.progress.failed_tasks > 0
        assert len(result.errors) > 0


# Edge case tests (2 additional as specified)
class TestExecutionEngineEdgeCases:
    """Additional edge case tests for ExecutionEngine."""
    
    @pytest.mark.asyncio
    async def test_edge_case_empty_team_composition(self, execution_engine):
        """Edge case: Handle empty team composition gracefully."""
        empty_team = TeamComposition(
            primary_team=[],  # Empty team
            load_order=[],
            coordination_strategy=CoordinationStrategy.PARALLEL,
            confidence_score=0.0,
        )
        
        # Should handle empty team without crashing
        result = await execution_engine.execute_team(
            team_composition=empty_team,
            objective="Empty team test",
            tasks=[],
        )
        
        assert result.status == ExecutionStatus.COMPLETED  # No tasks to fail
        assert len(result.tasks) == 0
    
    @pytest.mark.asyncio
    async def test_edge_case_concurrent_execution_cancellation(self, execution_engine, team_composition, sample_tasks):
        """Edge case: Cancel execution while tasks are running."""
        
        # Mock a slow execution
        with patch('src.agentsmcp.orchestration.execution_engine.ParallelExecution.execute') as mock_execute:
            async def slow_execution(execution, progress_callback=None):
                execution.status = ExecutionStatus.RUNNING
                # Simulate long-running execution with cancellation check
                for i in range(100):
                    if execution.status == ExecutionStatus.CANCELLED:
                        break
                    await asyncio.sleep(0.01)
                return execution
            
            mock_execute.side_effect = slow_execution
            
            # Start execution
            exec_task = asyncio.create_task(
                execution_engine.execute_team(
                    team_composition=team_composition,
                    objective="Cancellation test",
                    tasks=sample_tasks,
                )
            )
            
            # Wait a bit then cancel
            await asyncio.sleep(0.1)
            execution_id = list(execution_engine.active_executions.keys())[0] if execution_engine.active_executions else "test"
            cancelled = await execution_engine.cancel_execution(execution_id)
            
            # Wait for execution to complete
            result = await exec_task
            
            # Execution should handle cancellation gracefully
            assert isinstance(result, TeamExecution)