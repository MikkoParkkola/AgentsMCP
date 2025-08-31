"""Unit tests for the ResourceManager component."""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, Mock, patch

from src.agentsmcp.orchestration.resource_manager import (
    ResourceManager,
    ResourceType,
    ResourceQuota,
    ResourceAllocation,
    CircuitBreaker,
    CircuitBreakerState,
    ResourceExhaustedError,
)


@pytest.fixture
def resource_manager():
    """Create a ResourceManager instance for testing."""
    return ResourceManager(
        memory_limit_mb=1000,
        cpu_limit_percent=50.0,
        cost_limit_eur=10.0,
        max_concurrent_agents=5,
        max_concurrent_executions=3,
    )


@pytest.fixture
def sample_requirements():
    """Sample resource requirements for testing."""
    return {
        ResourceType.MEMORY: 200.0,
        ResourceType.CPU: 10.0,
        ResourceType.AGENT_SLOTS: 1,
    }


class TestResourceQuota:
    """Test ResourceQuota functionality."""
    
    def test_quota_initialization(self):
        """Test ResourceQuota initialization."""
        quota = ResourceQuota(ResourceType.MEMORY, 1000, unit="MB")
        assert quota.resource_type == ResourceType.MEMORY
        assert quota.limit == 1000
        assert quota.current_usage == 0
        assert quota.reserved == 0
        assert quota.unit == "MB"
    
    def test_quota_available_calculation(self):
        """Test available resource calculation."""
        quota = ResourceQuota(ResourceType.MEMORY, 1000)
        quota.current_usage = 300
        quota.reserved = 200
        
        assert quota.available == 500
        assert quota.utilization == 0.5
    
    def test_quota_utilization_calculation(self):
        """Test utilization percentage calculation."""
        quota = ResourceQuota(ResourceType.MEMORY, 1000)
        quota.current_usage = 400
        quota.reserved = 100
        
        assert quota.utilization == 0.5  # 50%
    
    def test_quota_over_limit_utilization(self):
        """Test utilization calculation when over limit."""
        quota = ResourceQuota(ResourceType.MEMORY, 1000)
        quota.current_usage = 800
        quota.reserved = 400
        
        assert quota.utilization == 1.0  # Capped at 100%


class TestCircuitBreaker:
    """Test CircuitBreaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test CircuitBreaker initialization."""
        breaker = CircuitBreaker("test", failure_threshold=3, recovery_timeout=30)
        assert breaker.name == "test"
        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30
        assert breaker.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in CLOSED state."""
        breaker = CircuitBreaker("test")
        assert breaker.can_execute() is True
    
    def test_circuit_breaker_success_recording(self):
        """Test recording successful executions."""
        breaker = CircuitBreaker("test", failure_threshold=3)
        
        # Record some failures
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.failure_count == 2
        
        # Record success should reduce failure count
        breaker.record_success()
        assert breaker.failure_count == 1
    
    def test_circuit_breaker_failure_recording(self):
        """Test recording failed executions."""
        breaker = CircuitBreaker("test", failure_threshold=2)
        
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 1
        
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.failure_count == 2
    
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker in OPEN state."""
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60)
        
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.can_execute() is False
    
    def test_circuit_breaker_half_open_transition(self):
        """Test transition to HALF_OPEN state."""
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0)
        
        # Trip the breaker
        breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery (immediate with timeout=0)
        assert breaker.can_execute() is True
        assert breaker.state == CircuitBreakerState.HALF_OPEN
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test recovery from HALF_OPEN state."""
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0, test_requests=2)
        
        # Trip and transition to half-open
        breaker.record_failure()
        breaker.can_execute()  # Transitions to HALF_OPEN
        
        # Record successful test requests
        breaker.record_success()
        breaker.record_success()
        
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0


class TestResourceAllocation:
    """Test ResourceAllocation functionality."""
    
    def test_allocation_initialization(self):
        """Test ResourceAllocation initialization."""
        allocation = ResourceAllocation(
            allocation_id="test-123",
            agent_id="agent-456",
            team_id="team-789",
            allocations={ResourceType.MEMORY: 100.0},
        )
        
        assert allocation.allocation_id == "test-123"
        assert allocation.agent_id == "agent-456"
        assert allocation.team_id == "team-789"
        assert allocation.allocations[ResourceType.MEMORY] == 100.0
        assert isinstance(allocation.allocated_at, datetime)
    
    def test_allocation_expiration(self):
        """Test allocation expiration checking."""
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        allocation = ResourceAllocation(
            allocation_id="test-expired",
            expires_at=past_time,
        )
        
        assert allocation.is_expired() is True
        
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        allocation.expires_at = future_time
        assert allocation.is_expired() is False


class TestResourceManager:
    """Test ResourceManager functionality."""
    
    @pytest.mark.asyncio
    async def test_resource_manager_initialization(self, resource_manager):
        """Test ResourceManager initialization."""
        assert ResourceType.MEMORY in resource_manager.quotas
        assert ResourceType.CPU in resource_manager.quotas
        assert ResourceType.COST in resource_manager.quotas
        
        memory_quota = resource_manager.quotas[ResourceType.MEMORY]
        assert memory_quota.limit == 1000
        assert memory_quota.unit == "MB"
    
    @pytest.mark.asyncio
    async def test_check_resource_availability_success(self, resource_manager, sample_requirements):
        """Test successful resource availability check."""
        available = await resource_manager.check_resource_availability(sample_requirements)
        assert available is True
    
    @pytest.mark.asyncio
    async def test_check_resource_availability_insufficient(self, resource_manager):
        """Test resource availability check with insufficient resources."""
        large_requirements = {ResourceType.MEMORY: 2000.0}  # Exceeds limit
        available = await resource_manager.check_resource_availability(large_requirements)
        assert available is False
    
    @pytest.mark.asyncio
    async def test_allocate_resources_success(self, resource_manager, sample_requirements):
        """Test successful resource allocation."""
        allocation = await resource_manager.allocate_resources(
            allocation_id="test-alloc",
            requirements=sample_requirements,
            agent_id="agent-123",
        )
        
        assert allocation.allocation_id == "test-alloc"
        assert allocation.agent_id == "agent-123"
        assert allocation.allocations == sample_requirements
        
        # Check that resources are reserved
        memory_quota = resource_manager.quotas[ResourceType.MEMORY]
        assert memory_quota.reserved == 200.0
    
    @pytest.mark.asyncio
    async def test_allocate_resources_insufficient(self, resource_manager):
        """Test resource allocation failure due to insufficient resources."""
        large_requirements = {ResourceType.MEMORY: 2000.0}
        
        with pytest.raises(ResourceExhaustedError):
            await resource_manager.allocate_resources(
                allocation_id="test-fail",
                requirements=large_requirements,
            )
    
    @pytest.mark.asyncio
    async def test_allocate_resources_duplicate_id(self, resource_manager, sample_requirements):
        """Test resource allocation failure with duplicate ID."""
        # First allocation should succeed
        await resource_manager.allocate_resources(
            allocation_id="duplicate-id",
            requirements=sample_requirements,
        )
        
        # Second allocation with same ID should fail
        with pytest.raises(ValueError, match="already exists"):
            await resource_manager.allocate_resources(
                allocation_id="duplicate-id",
                requirements=sample_requirements,
            )
    
    @pytest.mark.asyncio
    async def test_commit_allocation(self, resource_manager, sample_requirements):
        """Test committing a resource allocation."""
        allocation = await resource_manager.allocate_resources(
            allocation_id="commit-test",
            requirements=sample_requirements,
        )
        
        # Initially resources are reserved
        memory_quota = resource_manager.quotas[ResourceType.MEMORY]
        assert memory_quota.reserved == 200.0
        assert memory_quota.current_usage == 0
        
        # Commit allocation
        await resource_manager.commit_allocation("commit-test")
        
        # Resources should move from reserved to current usage
        assert memory_quota.reserved == 0
        assert memory_quota.current_usage == 200.0
    
    @pytest.mark.asyncio
    async def test_free_resources(self, resource_manager, sample_requirements):
        """Test freeing allocated resources."""
        allocation = await resource_manager.allocate_resources(
            allocation_id="free-test",
            requirements=sample_requirements,
        )
        
        await resource_manager.commit_allocation("free-test")
        
        # Verify resources are in use
        memory_quota = resource_manager.quotas[ResourceType.MEMORY]
        assert memory_quota.current_usage == 200.0
        
        # Free resources
        await resource_manager.free_resources("free-test")
        
        # Resources should be freed
        assert memory_quota.current_usage == 0
        assert "free-test" not in resource_manager.allocations
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_allocations(self, resource_manager):
        """Test cleanup of expired allocations."""
        # Create allocation with immediate expiration
        past_time = datetime.now(timezone.utc) - timedelta(seconds=1)
        requirements = {ResourceType.MEMORY: 100.0}
        
        allocation = await resource_manager.allocate_resources(
            allocation_id="expired-test",
            requirements=requirements,
            timeout_seconds=0,  # Immediate expiration
        )
        
        # Manually set expiration to past
        allocation.expires_at = past_time
        
        # Run cleanup
        cleaned_count = await resource_manager.cleanup_expired_allocations()
        
        assert cleaned_count == 1
        assert "expired-test" not in resource_manager.allocations
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, resource_manager):
        """Test circuit breaker integration."""
        breaker = resource_manager.get_circuit_breaker("memory")
        assert breaker is not None
        assert breaker.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker_success(self, resource_manager):
        """Test successful execution with circuit breaker."""
        
        async def mock_operation():
            return "success"
        
        result = await resource_manager.execute_with_circuit_breaker(
            "memory", mock_operation
        )
        
        assert result == "success"
        
        breaker = resource_manager.get_circuit_breaker("memory")
        assert breaker.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker_failure(self, resource_manager):
        """Test failed execution with circuit breaker."""
        
        async def failing_operation():
            raise ValueError("Test failure")
        
        # Trip the circuit breaker with multiple failures
        breaker = resource_manager.get_circuit_breaker("memory")
        for _ in range(breaker.failure_threshold):
            with pytest.raises(ValueError):
                await resource_manager.execute_with_circuit_breaker(
                    "memory", failing_operation
                )
        
        # Circuit should now be open
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Further attempts should be rejected immediately
        with pytest.raises(ResourceExhaustedError):
            await resource_manager.execute_with_circuit_breaker(
                "memory", failing_operation
            )
    
    def test_get_resource_status(self, resource_manager):
        """Test getting resource status."""
        status = resource_manager.get_resource_status()
        
        assert "quotas" in status
        assert "allocations" in status
        assert "circuit_breakers" in status
        assert "metrics" in status
        
        # Check quota structure
        memory_quota = status["quotas"]["memory"]
        assert "limit" in memory_quota
        assert "current_usage" in memory_quota
        assert "utilization" in memory_quota
    
    def test_update_quota(self, resource_manager):
        """Test updating resource quota."""
        original_limit = resource_manager.quotas[ResourceType.MEMORY].limit
        new_limit = original_limit + 500
        
        resource_manager.update_quota(ResourceType.MEMORY, new_limit)
        
        assert resource_manager.quotas[ResourceType.MEMORY].limit == new_limit
    
    @pytest.mark.asyncio
    async def test_callback_registration_and_trigger(self, resource_manager):
        """Test callback registration and triggering."""
        callback_called = False
        callback_args = {}
        
        async def test_callback(**kwargs):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = kwargs
        
        resource_manager.register_callback("test_event", test_callback)
        
        await resource_manager._trigger_callback("test_event", test_arg="test_value")
        
        assert callback_called is True
        assert callback_args["test_arg"] == "test_value"


# Golden tests as specified in ICD
class TestResourceManagerGoldenTests:
    """Golden tests for ResourceManager as specified in ICD."""
    
    @pytest.mark.asyncio
    async def test_golden_concurrent_allocations(self):
        """Golden test: Support 100 concurrent resource allocations."""
        resource_manager = ResourceManager(
            memory_limit_mb=10000,
            max_concurrent_agents=150,
        )
        
        # Create 100 concurrent allocation tasks
        async def allocate_resource(i):
            allocation_id = f"concurrent-{i}"
            requirements = {ResourceType.MEMORY: 50.0, ResourceType.AGENT_SLOTS: 1}
            return await resource_manager.allocate_resources(
                allocation_id=allocation_id,
                requirements=requirements,
            )
        
        # Run 100 concurrent allocations
        tasks = [allocate_resource(i) for i in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed (no exceptions)
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == 100
    
    @pytest.mark.asyncio
    async def test_golden_resource_quota_enforcement(self):
        """Golden test: Enforce resource quotas correctly."""
        resource_manager = ResourceManager(memory_limit_mb=500)
        
        # Should succeed within limit
        allocation1 = await resource_manager.allocate_resources(
            allocation_id="quota-test-1",
            requirements={ResourceType.MEMORY: 300.0},
        )
        assert allocation1 is not None
        
        # Should fail when exceeding quota
        with pytest.raises(ResourceExhaustedError):
            await resource_manager.allocate_resources(
                allocation_id="quota-test-2",
                requirements={ResourceType.MEMORY: 300.0},  # Would exceed 500MB limit
            )
    
    @pytest.mark.asyncio
    async def test_golden_circuit_breaker_protection(self):
        """Golden test: Circuit breaker provides proper protection."""
        resource_manager = ResourceManager()
        
        # Get circuit breaker with low failure threshold
        breaker = resource_manager.circuit_breakers["execution"]
        breaker.failure_threshold = 2
        
        async def failing_operation():
            raise RuntimeError("Simulated failure")
        
        # Trip the circuit breaker
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await resource_manager.execute_with_circuit_breaker(
                    "execution", failing_operation
                )
        
        # Circuit should be open and reject further requests
        assert breaker.state == CircuitBreakerState.OPEN
        
        with pytest.raises(ResourceExhaustedError):
            await resource_manager.execute_with_circuit_breaker(
                "execution", failing_operation
            )


# Edge case tests (2 additional as specified)
class TestResourceManagerEdgeCases:
    """Additional edge case tests for ResourceManager."""
    
    @pytest.mark.asyncio
    async def test_edge_case_zero_resource_limits(self):
        """Edge case: Handle zero resource limits gracefully."""
        resource_manager = ResourceManager(
            memory_limit_mb=0,
            max_concurrent_agents=0,
        )
        
        # Should reject any allocation with zero limits
        with pytest.raises(ResourceExhaustedError):
            await resource_manager.allocate_resources(
                allocation_id="zero-limit-test",
                requirements={ResourceType.MEMORY: 1.0},
            )
    
    @pytest.mark.asyncio
    async def test_edge_case_resource_cleanup_during_allocation(self):
        """Edge case: Resource cleanup during active allocation."""
        resource_manager = ResourceManager(memory_limit_mb=100)
        
        # Allocate resources
        allocation = await resource_manager.allocate_resources(
            allocation_id="cleanup-race-test",
            requirements={ResourceType.MEMORY: 50.0},
        )
        
        # Simulate concurrent cleanup and commit
        cleanup_task = asyncio.create_task(
            resource_manager.cleanup_expired_allocations()
        )
        commit_task = asyncio.create_task(
            resource_manager.commit_allocation("cleanup-race-test")
        )
        
        # Both operations should complete without error
        await asyncio.gather(cleanup_task, commit_task, return_exceptions=True)
        
        # Allocation should still exist and be committed
        assert "cleanup-race-test" in resource_manager.allocations
        memory_quota = resource_manager.quotas[ResourceType.MEMORY]
        assert memory_quota.current_usage == 50.0