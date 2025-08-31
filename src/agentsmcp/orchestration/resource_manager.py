"""Resource management system for dynamic orchestration.

This module provides comprehensive resource monitoring, quota enforcement,
and allocation management for agent teams. It ensures efficient resource
utilization while preventing system overload through circuit breaker patterns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Set, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading

from ..models import EnvelopeStatus


class ResourceType(str, Enum):
    """Types of resources managed by the system."""
    MEMORY = "memory"
    CPU = "cpu"
    COST = "cost"
    AGENT_SLOTS = "agent_slots"
    CONCURRENT_EXECUTIONS = "concurrent_executions"


class CircuitBreakerState(str, Enum):
    """Circuit breaker states for resource protection."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ResourceQuota:
    """Resource quota definition with limits and current usage."""
    resource_type: ResourceType
    limit: Union[int, float]
    current_usage: Union[int, float] = 0
    reserved: Union[int, float] = 0
    unit: str = "units"
    
    @property
    def available(self) -> Union[int, float]:
        """Calculate available resource capacity."""
        return max(0, self.limit - self.current_usage - self.reserved)
    
    @property
    def utilization(self) -> float:
        """Calculate current utilization percentage."""
        if self.limit <= 0:
            return 0.0
        return min(1.0, (self.current_usage + self.reserved) / self.limit)


@dataclass
class ResourceAllocation:
    """Resource allocation for a specific agent or team."""
    allocation_id: str
    agent_id: Optional[str] = None
    team_id: Optional[str] = None
    allocations: Dict[ResourceType, Union[int, float]] = field(default_factory=dict)
    allocated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if allocation has expired."""
        return self.expires_at is not None and datetime.now(timezone.utc) > self.expires_at


@dataclass
class CircuitBreaker:
    """Circuit breaker for resource protection."""
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60
    test_requests: int = 3
    
    # State tracking
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    test_count: int = 0
    
    def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit breaker state."""
        now = datetime.now(timezone.utc)
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if (self.last_failure_time and 
                (now - self.last_failure_time).total_seconds() > self.recovery_timeout):
                self.state = CircuitBreakerState.HALF_OPEN
                self.test_count = 0
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return self.test_count < self.test_requests
        
        return False
    
    def record_success(self) -> None:
        """Record successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.test_count += 1
            if self.test_count >= self.test_requests:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class ResourceExhaustedError(Exception):
    """Raised when resource allocation fails due to insufficient resources."""
    pass


class ResourceManager:
    """Manages resource allocation, monitoring, and quota enforcement.
    
    Provides comprehensive resource management including:
    - Memory, CPU, and cost tracking
    - Resource allocation and cleanup
    - Circuit breaker patterns for protection
    - Quota enforcement and monitoring
    """
    
    def __init__(
        self,
        memory_limit_mb: int = 5000,
        cpu_limit_percent: float = 80.0,
        cost_limit_eur: float = 100.0,
        max_concurrent_agents: int = 50,
        max_concurrent_executions: int = 20,
    ):
        self.log = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Initialize resource quotas
        self.quotas: Dict[ResourceType, ResourceQuota] = {
            ResourceType.MEMORY: ResourceQuota(
                ResourceType.MEMORY, memory_limit_mb, unit="MB"
            ),
            ResourceType.CPU: ResourceQuota(
                ResourceType.CPU, cpu_limit_percent, unit="%"
            ),
            ResourceType.COST: ResourceQuota(
                ResourceType.COST, cost_limit_eur, unit="EUR"
            ),
            ResourceType.AGENT_SLOTS: ResourceQuota(
                ResourceType.AGENT_SLOTS, max_concurrent_agents, unit="agents"
            ),
            ResourceType.CONCURRENT_EXECUTIONS: ResourceQuota(
                ResourceType.CONCURRENT_EXECUTIONS, max_concurrent_executions, unit="executions"
            ),
        }
        
        # Active allocations
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # Circuit breakers for different resource types
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            "memory": CircuitBreaker("memory", failure_threshold=3, recovery_timeout=30),
            "agent_loading": CircuitBreaker("agent_loading", failure_threshold=5, recovery_timeout=60),
            "execution": CircuitBreaker("execution", failure_threshold=10, recovery_timeout=120),
        }
        
        # Metrics and monitoring
        self.metrics = {
            "allocations_created": 0,
            "allocations_failed": 0,
            "allocations_freed": 0,
            "circuit_breaker_opens": 0,
            "quota_breaches": 0,
        }
        
        # Callbacks for resource events
        self.callbacks: Dict[str, Callable] = {}
        
        self.log.info("ResourceManager initialized with limits: memory=%dMB, cost=%.2f EUR, agents=%d",
                     memory_limit_mb, cost_limit_eur, max_concurrent_agents)
    
    async def check_resource_availability(
        self,
        requirements: Dict[ResourceType, Union[int, float]]
    ) -> bool:
        """Check if requested resources are available."""
        with self._lock:
            for resource_type, amount in requirements.items():
                quota = self.quotas.get(resource_type)
                if not quota:
                    self.log.warning("Unknown resource type: %s", resource_type)
                    continue
                
                if quota.available < amount:
                    self.log.debug("Insufficient %s: need %.2f, available %.2f",
                                 resource_type.value, amount, quota.available)
                    return False
            
            return True
    
    async def allocate_resources(
        self,
        allocation_id: str,
        requirements: Dict[ResourceType, Union[int, float]],
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> ResourceAllocation:
        """Allocate resources with quota enforcement.
        
        Args:
            allocation_id: Unique identifier for this allocation
            requirements: Resource requirements by type
            agent_id: Optional agent identifier
            team_id: Optional team identifier
            timeout_seconds: Optional expiration timeout
            
        Returns:
            ResourceAllocation object
            
        Raises:
            ResourceExhaustedError: If resources cannot be allocated
        """
        with self._lock:
            # Check if allocation already exists
            if allocation_id in self.allocations:
                raise ValueError(f"Allocation {allocation_id} already exists")
            
            # Validate requirements against quotas
            for resource_type, amount in requirements.items():
                quota = self.quotas.get(resource_type)
                if not quota:
                    raise ValueError(f"Unknown resource type: {resource_type}")
                
                if quota.available < amount:
                    self.metrics["allocations_failed"] += 1
                    self.metrics["quota_breaches"] += 1
                    raise ResourceExhaustedError(
                        f"Insufficient {resource_type.value}: need {amount}, available {quota.available}"
                    )
            
            # Create allocation
            expires_at = None
            if timeout_seconds:
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)
            
            allocation = ResourceAllocation(
                allocation_id=allocation_id,
                agent_id=agent_id,
                team_id=team_id,
                allocations=dict(requirements),
                expires_at=expires_at,
            )
            
            # Reserve resources
            for resource_type, amount in requirements.items():
                self.quotas[resource_type].reserved += amount
            
            self.allocations[allocation_id] = allocation
            self.metrics["allocations_created"] += 1
            
            self.log.debug("Allocated resources for %s: %s", allocation_id, requirements)
            return allocation
    
    async def commit_allocation(self, allocation_id: str) -> None:
        """Commit reserved resources to active usage."""
        with self._lock:
            allocation = self.allocations.get(allocation_id)
            if not allocation:
                raise ValueError(f"Allocation {allocation_id} not found")
            
            # Move from reserved to current usage
            for resource_type, amount in allocation.allocations.items():
                quota = self.quotas[resource_type]
                quota.reserved -= amount
                quota.current_usage += amount
            
            self.log.debug("Committed allocation %s", allocation_id)
    
    async def free_resources(self, allocation_id: str) -> None:
        """Free allocated resources."""
        with self._lock:
            allocation = self.allocations.pop(allocation_id, None)
            if not allocation:
                self.log.warning("Allocation %s not found", allocation_id)
                return
            
            # Free resources
            for resource_type, amount in allocation.allocations.items():
                quota = self.quotas[resource_type]
                if quota.current_usage >= amount:
                    quota.current_usage -= amount
                elif quota.reserved >= amount:
                    quota.reserved -= amount
                else:
                    # Handle partial free
                    remaining = amount
                    if quota.current_usage > 0:
                        freed = min(quota.current_usage, remaining)
                        quota.current_usage -= freed
                        remaining -= freed
                    if remaining > 0 and quota.reserved > 0:
                        freed = min(quota.reserved, remaining)
                        quota.reserved -= freed
            
            self.metrics["allocations_freed"] += 1
            self.log.debug("Freed resources for %s", allocation_id)
    
    async def cleanup_expired_allocations(self) -> int:
        """Clean up expired resource allocations."""
        expired_count = 0
        
        with self._lock:
            expired_ids = [
                alloc_id for alloc_id, allocation in self.allocations.items()
                if allocation.is_expired()
            ]
        
        for alloc_id in expired_ids:
            await self.free_resources(alloc_id)
            expired_count += 1
        
        if expired_count > 0:
            self.log.info("Cleaned up %d expired allocations", expired_count)
        
        return expired_count
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    async def execute_with_circuit_breaker(
        self,
        circuit_breaker_name: str,
        operation: Callable[[], Any],
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with circuit breaker protection."""
        breaker = self.circuit_breakers.get(circuit_breaker_name)
        if not breaker:
            raise ValueError(f"Circuit breaker {circuit_breaker_name} not found")
        
        if not breaker.can_execute():
            raise ResourceExhaustedError(f"Circuit breaker {circuit_breaker_name} is {breaker.state.value}")
        
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            breaker.record_success()
            return result
        
        except Exception as e:
            breaker.record_failure()
            if breaker.state == CircuitBreakerState.OPEN:
                self.metrics["circuit_breaker_opens"] += 1
            raise e
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource utilization status."""
        with self._lock:
            status = {
                "quotas": {
                    resource_type.value: {
                        "limit": quota.limit,
                        "current_usage": quota.current_usage,
                        "reserved": quota.reserved,
                        "available": quota.available,
                        "utilization": quota.utilization,
                        "unit": quota.unit,
                    }
                    for resource_type, quota in self.quotas.items()
                },
                "allocations": {
                    "active": len(self.allocations),
                    "by_team": len(set(a.team_id for a in self.allocations.values() if a.team_id)),
                    "by_agent": len(set(a.agent_id for a in self.allocations.values() if a.agent_id)),
                },
                "circuit_breakers": {
                    name: {
                        "state": breaker.state.value,
                        "failure_count": breaker.failure_count,
                        "last_failure": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None,
                    }
                    for name, breaker in self.circuit_breakers.items()
                },
                "metrics": self.metrics.copy(),
            }
            
            return status
    
    def update_quota(self, resource_type: ResourceType, new_limit: Union[int, float]) -> None:
        """Update resource quota limit."""
        with self._lock:
            if resource_type in self.quotas:
                old_limit = self.quotas[resource_type].limit
                self.quotas[resource_type].limit = new_limit
                self.log.info("Updated %s quota: %s -> %s %s",
                            resource_type.value, old_limit, new_limit,
                            self.quotas[resource_type].unit)
    
    def register_callback(self, event_name: str, callback: Callable) -> None:
        """Register callback for resource events."""
        self.callbacks[event_name] = callback
    
    async def _trigger_callback(self, event_name: str, **kwargs) -> None:
        """Trigger registered callback."""
        callback = self.callbacks.get(event_name)
        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(**kwargs)
                else:
                    callback(**kwargs)
            except Exception as e:
                self.log.error("Callback %s failed: %s", event_name, e)