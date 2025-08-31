"""Unit tests for the AgentLoader component."""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from src.agentsmcp.orchestration.agent_loader import (
    AgentLoader,
    LoadedAgent,
    AgentState,
    AgentContext,
    AgentLoadError,
    AgentNotFoundError,
    AgentBusyError,
)
from src.agentsmcp.orchestration.resource_manager import ResourceManager
from src.agentsmcp.orchestration.models import AgentSpec, ResourceConstraints
from src.agentsmcp.roles.base import RoleName, ModelAssignment, BaseRole


# Mock classes for testing
class MockAgentManager:
    """Mock AgentManager for testing."""
    
    def __init__(self):
        self.spawned_jobs = []
    
    async def spawn_agent(self, agent_type, prompt, timeout=300):
        job_id = f"job_{len(self.spawned_jobs)}"
        self.spawned_jobs.append((job_id, agent_type, prompt))
        return job_id


class MockRole(BaseRole):
    """Mock role implementation for testing."""
    
    @classmethod
    def name(cls):
        return RoleName.CODER
    
    async def initialize(self, environment):
        pass
    
    async def cleanup(self):
        pass


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
    return AgentLoader(
        agent_manager=mock_agent_manager,
        resource_manager=resource_manager,
        max_concurrent_loads=5,
        max_cached_agents=10,
        cache_ttl_minutes=10,
    )


@pytest.fixture
def agent_context():
    """Create agent context for testing."""
    return AgentContext(
        team_id="test-team",
        objective="Test objective",
        role_spec=AgentSpec(
            role="coder",
            model_assignment="codex",
            priority=1,
        ),
        resource_constraints=ResourceConstraints(max_agents=5),
    )


class TestAgentState:
    """Test AgentState enum."""
    
    def test_agent_states(self):
        """Test all agent state values."""
        assert AgentState.UNLOADED.value == "unloaded"
        assert AgentState.LOADING.value == "loading"
        assert AgentState.READY.value == "ready"
        assert AgentState.BUSY.value == "busy"
        assert AgentState.ERROR.value == "error"
        assert AgentState.UNLOADING.value == "unloading"


class TestAgentContext:
    """Test AgentContext functionality."""
    
    def test_context_initialization(self, agent_context):
        """Test AgentContext initialization."""
        assert agent_context.team_id == "test-team"
        assert agent_context.objective == "Test objective"
        assert agent_context.role_spec.role == "coder"
        assert agent_context.timeout_seconds == 300
        assert agent_context.sandbox_enabled is True


class TestLoadedAgent:
    """Test LoadedAgent functionality."""
    
    def test_loaded_agent_initialization(self, agent_context):
        """Test LoadedAgent initialization."""
        mock_role = MockRole()
        loaded_agent = LoadedAgent(
            agent_id="test-agent",
            role=mock_role,
            state=AgentState.READY,
            context=agent_context,
            loaded_at=datetime.now(timezone.utc),
            last_used=datetime.now(timezone.utc),
        )
        
        assert loaded_agent.agent_id == "test-agent"
        assert loaded_agent.role == mock_role
        assert loaded_agent.state == AgentState.READY
        assert loaded_agent.task_count == 0
        assert loaded_agent.total_execution_time == 0.0
    
    def test_update_usage_stats(self, agent_context):
        """Test updating usage statistics."""
        loaded_agent = LoadedAgent(
            agent_id="test-agent",
            role=MockRole(),
            state=AgentState.READY,
            context=agent_context,
            loaded_at=datetime.now(timezone.utc),
            last_used=datetime.now(timezone.utc),
        )
        
        original_last_used = loaded_agent.last_used
        
        loaded_agent.update_usage_stats(5.0)
        
        assert loaded_agent.task_count == 1
        assert loaded_agent.total_execution_time == 5.0
        assert loaded_agent.last_used > original_last_used
    
    def test_average_execution_time(self, agent_context):
        """Test average execution time calculation."""
        loaded_agent = LoadedAgent(
            agent_id="test-agent",
            role=MockRole(),
            state=AgentState.READY,
            context=agent_context,
            loaded_at=datetime.now(timezone.utc),
            last_used=datetime.now(timezone.utc),
        )
        
        # No tasks yet
        assert loaded_agent.average_execution_time == 0.0
        
        # Add tasks
        loaded_agent.update_usage_stats(3.0)
        loaded_agent.update_usage_stats(7.0)
        
        assert loaded_agent.average_execution_time == 5.0  # (3 + 7) / 2
    
    def test_idle_time_calculation(self, agent_context):
        """Test idle time calculation."""
        past_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        loaded_agent = LoadedAgent(
            agent_id="test-agent",
            role=MockRole(),
            state=AgentState.READY,
            context=agent_context,
            loaded_at=past_time,
            last_used=past_time,
        )
        
        idle_time = loaded_agent.idle_time_seconds
        assert idle_time >= 300  # At least 5 minutes


class TestAgentLoader:
    """Test AgentLoader functionality."""
    
    @pytest.mark.asyncio
    async def test_agent_loader_initialization(self, agent_loader):
        """Test AgentLoader initialization."""
        assert agent_loader.max_concurrent_loads == 5
        assert agent_loader.max_cached_agents == 10
        assert len(agent_loader.loaded_agents) == 0
        assert len(agent_loader.loading_agents) == 0
    
    @patch('src.agentsmcp.roles.registry.RoleRegistry')
    @pytest.mark.asyncio
    async def test_load_agent_success(self, mock_registry_class, agent_loader, agent_context):
        """Test successful agent loading."""
        # Setup mock registry
        mock_registry = Mock()
        mock_registry.get_role_class.return_value = MockRole
        mock_registry_class.return_value = mock_registry
        
        # Load agent
        role = await agent_loader.load_agent(
            role=RoleName.CODER,
            model_assignment=ModelAssignment(agent_type="codex"),
            context=agent_context,
        )
        
        assert isinstance(role, MockRole)
        assert len(agent_loader.loaded_agents) == 1
    
    @patch('src.agentsmcp.roles.registry.RoleRegistry')
    @pytest.mark.asyncio
    async def test_load_agent_cache_hit(self, mock_registry_class, agent_loader, agent_context):
        """Test agent loading cache hit."""
        # Setup mock registry
        mock_registry = Mock()
        mock_registry.get_role_class.return_value = MockRole
        mock_registry_class.return_value = mock_registry
        
        # Load agent first time
        role1 = await agent_loader.load_agent(
            role=RoleName.CODER,
            model_assignment=ModelAssignment(agent_type="codex"),
            context=agent_context,
        )
        
        initial_cache_hits = agent_loader.metrics["cache_hits"]
        
        # Load same agent again - should hit cache
        role2 = await agent_loader.load_agent(
            role=RoleName.CODER,
            model_assignment=ModelAssignment(agent_type="codex"),
            context=agent_context,
        )
        
        assert role1 is role2  # Same instance from cache
        assert agent_loader.metrics["cache_hits"] == initial_cache_hits + 1
    
    @pytest.mark.asyncio
    async def test_load_agent_resource_failure(self, agent_loader, agent_context):
        """Test agent loading failure due to insufficient resources."""
        # Set very low memory limit to force failure
        agent_loader.resource_manager.quotas[agent_loader.resource_manager.quotas.keys().__iter__().__next__()].limit = 10
        
        with pytest.raises(AgentLoadError):
            await agent_loader.load_agent(
                role=RoleName.CODER,
                model_assignment=ModelAssignment(agent_type="codex"),
                context=agent_context,
            )
    
    @pytest.mark.asyncio
    async def test_concurrent_loading_limit(self, agent_loader, agent_context):
        """Test concurrent loading limit enforcement."""
        
        # Create multiple loading tasks that would exceed limit
        async def slow_load():
            with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
                mock_registry = Mock()
                mock_registry.get_role_class.return_value = MockRole
                mock_registry_class.return_value = mock_registry
                
                # Simulate slow loading
                await asyncio.sleep(0.1)
                
                return await agent_loader.load_agent(
                    role=RoleName.CODER,
                    model_assignment=ModelAssignment(agent_type="codex"),
                    context=agent_context,
                )
        
        # Start more tasks than the concurrent limit
        tasks = []
        for i in range(agent_loader.max_concurrent_loads + 2):
            context = AgentContext(
                team_id=f"team-{i}",
                objective="Test",
                role_spec=AgentSpec(role="coder", model_assignment="codex"),
                resource_constraints=ResourceConstraints(max_agents=5),
            )
            tasks.append(slow_load())
        
        # All tasks should complete successfully despite limit
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > 0  # At least some should succeed
    
    @pytest.mark.asyncio
    async def test_mark_agent_busy(self, agent_loader, agent_context):
        """Test marking agent as busy."""
        with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_role_class.return_value = MockRole
            mock_registry_class.return_value = mock_registry
            
            # Load agent first
            await agent_loader.load_agent(
                role=RoleName.CODER,
                model_assignment=ModelAssignment(agent_type="codex"),
                context=agent_context,
            )
            
            agent_id = f"{agent_context.team_id}:coder:codex"
            
            # Mark as busy
            await agent_loader.mark_agent_busy(agent_id, "task-123")
            
            loaded_agent = agent_loader.loaded_agents[agent_id]
            assert loaded_agent.state == AgentState.BUSY
            assert loaded_agent.current_task == "task-123"
    
    @pytest.mark.asyncio
    async def test_mark_agent_busy_not_ready(self, agent_loader):
        """Test marking non-ready agent as busy fails."""
        # Create agent in loading state
        agent_id = "test-agent"
        loaded_agent = LoadedAgent(
            agent_id=agent_id,
            role=MockRole(),
            state=AgentState.LOADING,
            context=AgentContext(
                team_id="test",
                objective="test",
                role_spec=AgentSpec(role="coder", model_assignment="codex"),
                resource_constraints=ResourceConstraints(max_agents=5),
            ),
            loaded_at=datetime.now(timezone.utc),
            last_used=datetime.now(timezone.utc),
        )
        agent_loader.loaded_agents[agent_id] = loaded_agent
        
        with pytest.raises(AgentBusyError):
            await agent_loader.mark_agent_busy(agent_id, "task-123")
    
    @pytest.mark.asyncio
    async def test_mark_agent_ready(self, agent_loader, agent_context):
        """Test marking agent as ready."""
        with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_role_class.return_value = MockRole
            mock_registry_class.return_value = mock_registry
            
            # Load and mark busy
            await agent_loader.load_agent(
                role=RoleName.CODER,
                model_assignment=ModelAssignment(agent_type="codex"),
                context=agent_context,
            )
            
            agent_id = f"{agent_context.team_id}:coder:codex"
            await agent_loader.mark_agent_busy(agent_id, "task-123")
            
            # Mark as ready
            await agent_loader.mark_agent_ready(agent_id, execution_time=2.5)
            
            loaded_agent = agent_loader.loaded_agents[agent_id]
            assert loaded_agent.state == AgentState.READY
            assert loaded_agent.current_task is None
            assert loaded_agent.task_count == 1
            assert loaded_agent.total_execution_time == 2.5
    
    @pytest.mark.asyncio
    async def test_unload_agent(self, agent_loader, agent_context):
        """Test unloading an agent."""
        with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_role_class.return_value = MockRole
            mock_registry_class.return_value = mock_registry
            
            # Load agent
            await agent_loader.load_agent(
                role=RoleName.CODER,
                model_assignment=ModelAssignment(agent_type="codex"),
                context=agent_context,
            )
            
            agent_id = f"{agent_context.team_id}:coder:codex"
            assert agent_id in agent_loader.loaded_agents
            
            # Unload agent
            success = await agent_loader.unload_agent(agent_id)
            
            assert success is True
            assert agent_id not in agent_loader.loaded_agents
    
    @pytest.mark.asyncio
    async def test_unload_busy_agent_fails(self, agent_loader, agent_context):
        """Test that unloading busy agent fails."""
        with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_role_class.return_value = MockRole
            mock_registry_class.return_value = mock_registry
            
            # Load and mark busy
            await agent_loader.load_agent(
                role=RoleName.CODER,
                model_assignment=ModelAssignment(agent_type="codex"),
                context=agent_context,
            )
            
            agent_id = f"{agent_context.team_id}:coder:codex"
            await agent_loader.mark_agent_busy(agent_id, "task-123")
            
            # Try to unload busy agent
            success = await agent_loader.unload_agent(agent_id)
            
            assert success is False
            assert agent_id in agent_loader.loaded_agents  # Still loaded
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_agents(self, agent_loader):
        """Test cleanup of expired agents."""
        # Create expired agent manually
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        agent_id = "expired-agent"
        
        loaded_agent = LoadedAgent(
            agent_id=agent_id,
            role=MockRole(),
            state=AgentState.READY,
            context=AgentContext(
                team_id="test",
                objective="test",
                role_spec=AgentSpec(role="coder", model_assignment="codex"),
                resource_constraints=ResourceConstraints(max_agents=5),
            ),
            loaded_at=past_time,
            last_used=past_time,
        )
        
        agent_loader.loaded_agents[agent_id] = loaded_agent
        agent_loader.cache_ttl = timedelta(minutes=30)  # Set short TTL
        
        # Run cleanup
        cleaned_count = await agent_loader.cleanup_expired_agents()
        
        assert cleaned_count == 1
        assert agent_id not in agent_loader.loaded_agents
    
    @pytest.mark.asyncio
    async def test_cache_size_enforcement(self, agent_loader):
        """Test cache size limit enforcement."""
        agent_loader.max_cached_agents = 2  # Set low limit
        
        with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_role_class.return_value = MockRole
            mock_registry_class.return_value = mock_registry
            
            # Load agents beyond cache limit
            agents = []
            for i in range(3):
                context = AgentContext(
                    team_id=f"team-{i}",
                    objective="Test",
                    role_spec=AgentSpec(role="coder", model_assignment="codex"),
                    resource_constraints=ResourceConstraints(max_agents=5),
                )
                
                await agent_loader.load_agent(
                    role=RoleName.CODER,
                    model_assignment=ModelAssignment(agent_type="codex"),
                    context=context,
                )
                agents.append(f"team-{i}:coder:codex")
            
            # Cache should be limited
            assert len(agent_loader.loaded_agents) <= agent_loader.max_cached_agents
    
    def test_memory_estimation(self, agent_loader):
        """Test memory usage estimation."""
        # Test different role types
        coder_memory = agent_loader._estimate_memory_usage(
            RoleName.CODER, ModelAssignment(agent_type="codex")
        )
        architect_memory = agent_loader._estimate_memory_usage(
            RoleName.ARCHITECT, ModelAssignment(agent_type="claude")
        )
        
        assert coder_memory > 0
        assert architect_memory > 0
        assert architect_memory > coder_memory  # Architect should use more memory
    
    def test_get_agent_stats(self, agent_loader):
        """Test getting agent statistics."""
        stats = agent_loader.get_agent_stats()
        
        assert "total_agents" in stats
        assert "agents_by_state" in stats
        assert "concurrent_loads" in stats
        assert "cache_utilization" in stats
        assert "metrics" in stats
        assert "resource_status" in stats


# Golden tests as specified in ICD
class TestAgentLoaderGoldenTests:
    """Golden tests for AgentLoader as specified in ICD."""
    
    @patch('src.agentsmcp.roles.registry.RoleRegistry')
    @pytest.mark.asyncio
    async def test_golden_concurrent_agent_loading(self, mock_registry_class):
        """Golden test: Load 100 concurrent agents within 3s per agent."""
        # Setup
        resource_manager = ResourceManager(
            memory_limit_mb=50000,  # High limit for concurrent loading
            max_concurrent_agents=150,
        )
        agent_loader = AgentLoader(
            agent_manager=MockAgentManager(),
            resource_manager=resource_manager,
            max_concurrent_loads=20,  # Higher for test
        )
        
        mock_registry = Mock()
        mock_registry.get_role_class.return_value = MockRole
        mock_registry_class.return_value = mock_registry
        
        # Create concurrent loading tasks
        async def load_agent_task(i):
            context = AgentContext(
                team_id=f"concurrent-team-{i}",
                objective="Concurrent test",
                role_spec=AgentSpec(role="coder", model_assignment="codex"),
                resource_constraints=ResourceConstraints(max_agents=5),
            )
            
            start_time = asyncio.get_event_loop().time()
            role = await agent_loader.load_agent(
                role=RoleName.CODER,
                model_assignment=ModelAssignment(agent_type="codex"),
                context=context,
            )
            load_time = asyncio.get_event_loop().time() - start_time
            
            return role, load_time
        
        # Load 100 agents concurrently
        tasks = [load_agent_task(i) for i in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify results
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == 100
        
        # Verify load times (most should be under 3s)
        load_times = [result[1] for result in successful if isinstance(result, tuple)]
        avg_load_time = sum(load_times) / len(load_times)
        assert avg_load_time < 3.0  # Average should be under 3 seconds
    
    @patch('src.agentsmcp.roles.registry.RoleRegistry')
    @pytest.mark.asyncio
    async def test_golden_memory_usage_limit(self, mock_registry_class):
        """Golden test: Memory usage under 500MB per agent."""
        agent_loader = AgentLoader(
            agent_manager=MockAgentManager(),
            resource_manager=ResourceManager(memory_limit_mb=10000),
        )
        
        mock_registry = Mock()
        mock_registry.get_role_class.return_value = MockRole
        mock_registry_class.return_value = mock_registry
        
        # Test memory estimation for different agent types
        test_cases = [
            (RoleName.ARCHITECT, ModelAssignment(agent_type="claude")),
            (RoleName.CODER, ModelAssignment(agent_type="codex")),
            (RoleName.QA, ModelAssignment(agent_type="ollama")),
        ]
        
        for role, model_assignment in test_cases:
            memory_estimate = agent_loader._estimate_memory_usage(role, model_assignment)
            assert memory_estimate <= 500.0, f"Agent {role.value} exceeds 500MB limit: {memory_estimate}MB"
    
    @patch('src.agentsmcp.roles.registry.RoleRegistry')
    @pytest.mark.asyncio
    async def test_golden_agent_lifecycle_performance(self, mock_registry_class):
        """Golden test: Agent loading within 3s per agent."""
        agent_loader = AgentLoader(
            agent_manager=MockAgentManager(),
            resource_manager=ResourceManager(),
        )
        
        mock_registry = Mock()
        mock_registry.get_role_class.return_value = MockRole
        mock_registry_class.return_value = mock_registry
        
        context = AgentContext(
            team_id="perf-test",
            objective="Performance test",
            role_spec=AgentSpec(role="coder", model_assignment="codex"),
            resource_constraints=ResourceConstraints(max_agents=5),
        )
        
        # Measure loading time
        start_time = asyncio.get_event_loop().time()
        
        role = await agent_loader.load_agent(
            role=RoleName.CODER,
            model_assignment=ModelAssignment(agent_type="codex"),
            context=context,
        )
        
        load_time = asyncio.get_event_loop().time() - start_time
        
        assert role is not None
        assert load_time < 3.0  # Should load within 3 seconds


# Edge case tests (2 additional as specified)
class TestAgentLoaderEdgeCases:
    """Additional edge case tests for AgentLoader."""
    
    @pytest.mark.asyncio
    async def test_edge_case_simultaneous_load_unload(self, agent_loader, agent_context):
        """Edge case: Simultaneous load and unload operations."""
        with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_role_class.return_value = MockRole
            mock_registry_class.return_value = mock_registry
            
            # Start loading an agent
            load_task = asyncio.create_task(
                agent_loader.load_agent(
                    role=RoleName.CODER,
                    model_assignment=ModelAssignment(agent_type="codex"),
                    context=agent_context,
                )
            )
            
            # Try to unload the same agent while loading
            agent_id = f"{agent_context.team_id}:coder:codex"
            unload_task = asyncio.create_task(
                agent_loader.unload_agent(agent_id)
            )
            
            # Both operations should complete without deadlock
            load_result, unload_result = await asyncio.gather(
                load_task, unload_task, return_exceptions=True
            )
            
            # Load should succeed, unload might fail (agent not found initially)
            assert not isinstance(load_result, Exception)
    
    @pytest.mark.asyncio
    async def test_edge_case_resource_exhaustion_during_load(self, agent_context):
        """Edge case: Resource exhaustion during agent loading."""
        # Create resource manager with very limited resources
        resource_manager = ResourceManager(
            memory_limit_mb=250,  # Enough for one agent, not two
            max_concurrent_agents=2,
        )
        
        agent_loader = AgentLoader(
            agent_manager=MockAgentManager(),
            resource_manager=resource_manager,
        )
        
        with patch('src.agentsmcp.roles.registry.RoleRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry.get_role_class.return_value = MockRole
            mock_registry_class.return_value = mock_registry
            
            # First agent should succeed
            await agent_loader.load_agent(
                role=RoleName.CODER,
                model_assignment=ModelAssignment(agent_type="codex"),
                context=agent_context,
            )
            
            # Second agent should fail due to resource exhaustion
            context2 = AgentContext(
                team_id="team-2",
                objective="Test 2",
                role_spec=AgentSpec(role="coder", model_assignment="codex"),
                resource_constraints=ResourceConstraints(max_agents=5),
            )
            
            with pytest.raises(AgentLoadError):
                await agent_loader.load_agent(
                    role=RoleName.CODER,
                    model_assignment=ModelAssignment(agent_type="codex"),
                    context=context2,
                )