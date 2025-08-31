"""Agent loader system for dynamic team orchestration.

This module provides lazy loading, caching, and lifecycle management for agents
in dynamic team compositions. It supports concurrent loading, resource optimization,
and sandboxed execution environments.
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Set, Any, Callable, List, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import threading

from .models import AgentSpec, ResourceConstraints
from .resource_manager import ResourceManager, ResourceType
from ..roles.base import BaseRole, RoleName, ModelAssignment
from ..models import TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus

if TYPE_CHECKING:
    from ..agent_manager import AgentManager


class AgentState(str, Enum):
    """Agent lifecycle states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNLOADING = "unloading"
    CLEANUP = "cleanup"


@dataclass
class AgentContext:
    """Context information for agent initialization."""
    team_id: str
    objective: str
    role_spec: AgentSpec
    resource_constraints: ResourceConstraints
    environment: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    sandbox_enabled: bool = True


@dataclass
class LoadedAgent:
    """Container for a loaded agent with metadata."""
    agent_id: str
    role: BaseRole
    state: AgentState
    context: AgentContext
    loaded_at: datetime
    last_used: datetime
    resource_allocation_id: Optional[str] = None
    current_task: Optional[str] = None
    error: Optional[str] = None
    
    # Performance metrics
    load_time_seconds: float = 0.0
    task_count: int = 0
    total_execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    def update_usage_stats(self, execution_time: float) -> None:
        """Update agent usage statistics."""
        self.last_used = datetime.now(timezone.utc)
        self.task_count += 1
        self.total_execution_time += execution_time
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time per task."""
        return self.total_execution_time / self.task_count if self.task_count > 0 else 0.0
    
    @property
    def idle_time_seconds(self) -> float:
        """Calculate time since last usage."""
        return (datetime.now(timezone.utc) - self.last_used).total_seconds()


class AgentLoadError(Exception):
    """Raised when agent loading fails."""
    pass


class AgentNotFoundError(Exception):
    """Raised when requested agent is not found."""
    pass


class AgentBusyError(Exception):
    """Raised when agent is busy and cannot accept new tasks."""
    pass


class AgentLoader:
    """Manages agent loading, caching, and lifecycle.
    
    Provides lazy loading system with resource optimization and concurrent
    support for dynamic team compositions.
    
    Features:
    - Lazy loading with caching
    - Concurrent agent loading (up to 100 agents)
    - Resource-aware loading with memory limits
    - Agent lifecycle management
    - Performance monitoring and metrics
    """
    
    def __init__(
        self,
        agent_manager: "AgentManager",
        resource_manager: ResourceManager,
        max_concurrent_loads: int = 10,
        max_cached_agents: int = 50,
        cache_ttl_minutes: int = 30,
        cleanup_interval_minutes: int = 5,
    ):
        self.agent_manager = agent_manager
        self.resource_manager = resource_manager
        self.log = logging.getLogger(__name__)
        
        # Configuration
        self.max_concurrent_loads = max_concurrent_loads
        self.max_cached_agents = max_cached_agents
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.cleanup_interval = timedelta(minutes=cleanup_interval_minutes)
        
        # Agent cache and state tracking
        self.loaded_agents: Dict[str, LoadedAgent] = {}
        self.loading_agents: Set[str] = set()
        self._lock = threading.RLock()
        
        # Concurrency control
        self._loading_semaphore = asyncio.Semaphore(max_concurrent_loads)
        self._loading_tasks: Dict[str, asyncio.Task] = {}
        
        # Role registry cache
        self._role_registry: Optional[Any] = None
        
        # Metrics
        self.metrics = {
            "agents_loaded": 0,
            "agents_cached": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "load_failures": 0,
            "concurrent_loads": 0,
            "cleanup_runs": 0,
        }
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        self.log.info("AgentLoader initialized: max_concurrent=%d, max_cached=%d, ttl=%d min",
                     max_concurrent_loads, max_cached_agents, cache_ttl_minutes)
    
    async def start_background_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                await self.cleanup_expired_agents()
                await self.resource_manager.cleanup_expired_allocations()
                self.metrics["cleanup_runs"] += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.error("Cleanup loop error: %s", e)
    
    def _get_role_registry(self):
        """Lazy load role registry."""
        if self._role_registry is None:
            from ..roles.registry import RoleRegistry
            self._role_registry = RoleRegistry()
        return self._role_registry
    
    async def load_agent(
        self,
        role: RoleName,
        model_assignment: ModelAssignment,
        context: AgentContext,
    ) -> BaseRole:
        """Load an agent with the specified role and configuration.
        
        Args:
            role: Role identifier for the agent
            model_assignment: Model/agent type assignment
            context: Agent context and environment
            
        Returns:
            Loaded BaseRole instance
            
        Raises:
            AgentLoadError: If agent loading fails
            ResourceExhaustedError: If insufficient resources
        """
        agent_id = f"{context.team_id}:{role.value}:{model_assignment.agent_type}"
        
        # Check cache first
        with self._lock:
            if agent_id in self.loaded_agents:
                loaded_agent = self.loaded_agents[agent_id]
                if loaded_agent.state == AgentState.READY:
                    self.metrics["cache_hits"] += 1
                    loaded_agent.last_used = datetime.now(timezone.utc)
                    self.log.debug("Cache hit for agent %s", agent_id)
                    return loaded_agent.role
                elif loaded_agent.state == AgentState.LOADING:
                    # Wait for loading to complete
                    loading_task = self._loading_tasks.get(agent_id)
                    if loading_task:
                        try:
                            await loading_task
                            return self.loaded_agents[agent_id].role
                        except Exception as e:
                            raise AgentLoadError(f"Agent loading failed: {e}") from e
        
        self.metrics["cache_misses"] += 1
        
        # Load agent asynchronously
        return await self._load_agent_async(agent_id, role, model_assignment, context)
    
    async def _load_agent_async(
        self,
        agent_id: str,
        role: RoleName,
        model_assignment: ModelAssignment,
        context: AgentContext,
    ) -> BaseRole:
        """Asynchronously load an agent with resource management."""
        
        # Check if already loading
        with self._lock:
            if agent_id in self.loading_agents:
                loading_task = self._loading_tasks.get(agent_id)
                if loading_task:
                    await loading_task
                    return self.loaded_agents[agent_id].role
        
        async with self._loading_semaphore:
            with self._lock:
                self.loading_agents.add(agent_id)
                self.metrics["concurrent_loads"] = len(self.loading_agents)
            
            try:
                # Start loading task
                loading_task = asyncio.create_task(
                    self._do_load_agent(agent_id, role, model_assignment, context)
                )
                self._loading_tasks[agent_id] = loading_task
                
                loaded_agent = await loading_task
                return loaded_agent.role
            
            finally:
                with self._lock:
                    self.loading_agents.discard(agent_id)
                    self._loading_tasks.pop(agent_id, None)
                    self.metrics["concurrent_loads"] = len(self.loading_agents)
    
    async def _do_load_agent(
        self,
        agent_id: str,
        role: RoleName,
        model_assignment: ModelAssignment,
        context: AgentContext,
    ) -> LoadedAgent:
        """Internal agent loading implementation."""
        start_time = time.time()
        
        # Estimate resource requirements
        estimated_memory = self._estimate_memory_usage(role, model_assignment)
        resource_requirements = {
            ResourceType.MEMORY: estimated_memory,
            ResourceType.AGENT_SLOTS: 1,
        }
        
        # Allocate resources
        allocation_id = f"agent:{agent_id}:{int(start_time)}"
        try:
            allocation = await self.resource_manager.allocate_resources(
                allocation_id=allocation_id,
                requirements=resource_requirements,
                agent_id=agent_id,
                team_id=context.team_id,
                timeout_seconds=context.timeout_seconds,
            )
        except Exception as e:
            raise AgentLoadError(f"Resource allocation failed: {e}") from e
        
        try:
            # Create loaded agent entry
            loaded_agent = LoadedAgent(
                agent_id=agent_id,
                role=None,  # Will be set after successful loading
                state=AgentState.LOADING,
                context=context,
                loaded_at=datetime.now(timezone.utc),
                last_used=datetime.now(timezone.utc),
                resource_allocation_id=allocation_id,
            )
            
            with self._lock:
                self.loaded_agents[agent_id] = loaded_agent
            
            # Load the actual role instance
            role_instance = await self._create_role_instance(role, model_assignment, context)
            
            # Commit resources and finalize
            await self.resource_manager.commit_allocation(allocation_id)
            
            loaded_agent.role = role_instance
            loaded_agent.state = AgentState.READY
            loaded_agent.load_time_seconds = time.time() - start_time
            loaded_agent.memory_usage_mb = estimated_memory
            
            self.metrics["agents_loaded"] += 1
            self.metrics["agents_cached"] += 1
            
            # Cleanup cache if it's too large
            await self._enforce_cache_size()
            
            self.log.info("Loaded agent %s in %.2fs", agent_id, loaded_agent.load_time_seconds)
            return loaded_agent
        
        except Exception as e:
            # Clean up on failure
            await self.resource_manager.free_resources(allocation_id)
            with self._lock:
                if agent_id in self.loaded_agents:
                    self.loaded_agents[agent_id].state = AgentState.ERROR
                    self.loaded_agents[agent_id].error = str(e)
            
            self.metrics["load_failures"] += 1
            self.log.error("Failed to load agent %s: %s", agent_id, e)
            raise AgentLoadError(f"Agent loading failed: {e}") from e
    
    async def _create_role_instance(
        self,
        role: RoleName,
        model_assignment: ModelAssignment,
        context: AgentContext,
    ) -> BaseRole:
        """Create and initialize a role instance."""
        registry = self._get_role_registry()
        
        # Get role class from registry
        role_class = registry.get_role_class(role)
        if not role_class:
            raise AgentLoadError(f"Role {role.value} not found in registry")
        
        # Create instance
        role_instance = role_class()
        
        # Set model assignment
        role_instance.model_assignment = model_assignment
        
        # Additional initialization based on context
        if hasattr(role_instance, 'initialize'):
            await role_instance.initialize(context.environment)
        
        return role_instance
    
    def _estimate_memory_usage(self, role: RoleName, model_assignment: ModelAssignment) -> float:
        """Estimate memory usage for an agent."""
        # Base memory requirements by role type
        base_memory = {
            RoleName.ARCHITECT: 200.0,
            RoleName.CODER: 150.0,
            RoleName.QA: 100.0,
            RoleName.BACKEND_ENGINEER: 180.0,
            RoleName.WEB_FRONTEND_ENGINEER: 160.0,
            RoleName.TUI_FRONTEND_ENGINEER: 140.0,
        }.get(role, 100.0)  # Default 100MB
        
        # Model-specific overhead
        model_overhead = {
            "claude": 50.0,
            "codex": 40.0,
            "ollama": 200.0,  # Local model requires more memory
        }.get(model_assignment.agent_type, 30.0)
        
        return base_memory + model_overhead
    
    async def _enforce_cache_size(self) -> None:
        """Enforce maximum cache size by evicting least recently used agents."""
        with self._lock:
            if len(self.loaded_agents) <= self.max_cached_agents:
                return
            
            # Sort by last used time (oldest first)
            agents_by_usage = sorted(
                self.loaded_agents.values(),
                key=lambda a: a.last_used
            )
            
            # Evict oldest agents that are not busy
            evicted = 0
            for agent in agents_by_usage:
                if len(self.loaded_agents) <= self.max_cached_agents:
                    break
                
                if agent.state == AgentState.READY and not agent.current_task:
                    await self._unload_agent(agent.agent_id)
                    evicted += 1
            
            if evicted > 0:
                self.log.info("Evicted %d agents from cache", evicted)
    
    async def get_agent(self, agent_id: str) -> Optional[LoadedAgent]:
        """Get loaded agent by ID."""
        with self._lock:
            return self.loaded_agents.get(agent_id)
    
    async def mark_agent_busy(self, agent_id: str, task_id: str) -> None:
        """Mark agent as busy with a specific task."""
        with self._lock:
            if agent_id in self.loaded_agents:
                agent = self.loaded_agents[agent_id]
                if agent.state == AgentState.READY:
                    agent.state = AgentState.BUSY
                    agent.current_task = task_id
                    self.log.debug("Marked agent %s as busy with task %s", agent_id, task_id)
                else:
                    raise AgentBusyError(f"Agent {agent_id} is not ready (state: {agent.state})")
            else:
                raise AgentNotFoundError(f"Agent {agent_id} not found")
    
    async def mark_agent_ready(self, agent_id: str, execution_time: float = 0.0) -> None:
        """Mark agent as ready after task completion."""
        with self._lock:
            if agent_id in self.loaded_agents:
                agent = self.loaded_agents[agent_id]
                agent.state = AgentState.READY
                agent.current_task = None
                agent.update_usage_stats(execution_time)
                self.log.debug("Marked agent %s as ready", agent_id)
    
    async def unload_agent(self, agent_id: str) -> bool:
        """Unload a specific agent."""
        return await self._unload_agent(agent_id)
    
    async def _unload_agent(self, agent_id: str) -> bool:
        """Internal agent unloading implementation."""
        with self._lock:
            agent = self.loaded_agents.get(agent_id)
            if not agent:
                return False
            
            if agent.state == AgentState.BUSY:
                self.log.warning("Cannot unload busy agent %s", agent_id)
                return False
            
            agent.state = AgentState.UNLOADING
        
        try:
            # Free resources
            if agent.resource_allocation_id:
                await self.resource_manager.free_resources(agent.resource_allocation_id)
            
            # Cleanup agent
            if hasattr(agent.role, 'cleanup'):
                await agent.role.cleanup()
            
            # Remove from cache
            with self._lock:
                self.loaded_agents.pop(agent_id, None)
                self.metrics["agents_cached"] = len(self.loaded_agents)
            
            self.log.debug("Unloaded agent %s", agent_id)
            return True
        
        except Exception as e:
            with self._lock:
                if agent_id in self.loaded_agents:
                    self.loaded_agents[agent_id].state = AgentState.ERROR
                    self.loaded_agents[agent_id].error = str(e)
            
            self.log.error("Failed to unload agent %s: %s", agent_id, e)
            return False
    
    async def cleanup_expired_agents(self) -> int:
        """Clean up expired and idle agents."""
        cleanup_count = 0
        now = datetime.now(timezone.utc)
        
        with self._lock:
            expired_agents = [
                agent_id for agent_id, agent in self.loaded_agents.items()
                if (agent.state == AgentState.READY and 
                    now - agent.last_used > self.cache_ttl and
                    not agent.current_task)
            ]
        
        for agent_id in expired_agents:
            if await self._unload_agent(agent_id):
                cleanup_count += 1
        
        if cleanup_count > 0:
            self.log.info("Cleaned up %d expired agents", cleanup_count)
        
        return cleanup_count
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent loading and cache statistics."""
        with self._lock:
            agents_by_state = {}
            for state in AgentState:
                agents_by_state[state.value] = len([
                    a for a in self.loaded_agents.values() if a.state == state
                ])
            
            return {
                "total_agents": len(self.loaded_agents),
                "agents_by_state": agents_by_state,
                "concurrent_loads": len(self.loading_agents),
                "cache_utilization": len(self.loaded_agents) / self.max_cached_agents,
                "metrics": self.metrics.copy(),
                "resource_status": self.resource_manager.get_resource_status(),
            }
    
    async def shutdown(self) -> None:
        """Shutdown agent loader and clean up resources."""
        self._shutdown = True
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all loading tasks
        for task in self._loading_tasks.values():
            task.cancel()
        
        if self._loading_tasks:
            await asyncio.gather(*self._loading_tasks.values(), return_exceptions=True)
        
        # Unload all agents
        agent_ids = list(self.loaded_agents.keys())
        for agent_id in agent_ids:
            await self._unload_agent(agent_id)
        
        self.log.info("AgentLoader shutdown complete")