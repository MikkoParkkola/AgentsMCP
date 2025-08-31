"""Dynamic orchestrator for agent team management and execution.

This module provides the main orchestration system for dynamic agent teams.
It coordinates agent lifecycle management, supports multiple execution strategies,
handles failures with fallback mechanisms, and integrates with the broader
AgentsMCP ecosystem.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union, Set, TYPE_CHECKING
from dataclasses import dataclass
import uuid

from .models import TeamComposition, CoordinationStrategy, ResourceConstraints, TaskResult
from .agent_loader import AgentLoader, AgentContext
from .resource_manager import ResourceManager, ResourceType
from .execution_engine import ExecutionEngine, TeamExecution, ExecutionStatus, ExecutionProgress
from ..roles.base import RoleName, ModelAssignment
from ..models import TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus

if TYPE_CHECKING:
    from ..agent_manager import AgentManager


@dataclass
class OrchestrationMetrics:
    """Metrics for orchestration performance tracking."""
    teams_orchestrated: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    total_agents_loaded: int = 0
    resource_utilization: Dict[str, float] = None
    
    def __post_init__(self):
        if self.resource_utilization is None:
            self.resource_utilization = {}


@dataclass
class FallbackConfig:
    """Configuration for fallback mechanisms."""
    enable_fallbacks: bool = True
    max_retry_attempts: int = 3
    fallback_delay_seconds: float = 2.0
    use_fallback_agents: bool = True
    graceful_degradation: bool = True


class OrchestrationError(Exception):
    """Base exception for orchestration errors."""
    pass


class TeamLoadError(OrchestrationError):
    """Raised when team loading fails."""
    pass


class ExecutionTimeoutError(OrchestrationError):
    """Raised when execution times out."""
    pass


class InsufficientResourcesError(OrchestrationError):
    """Raised when insufficient resources are available."""
    pass


class DynamicOrchestrator:
    """Main orchestrator for dynamic agent teams.
    
    Provides comprehensive orchestration capabilities including:
    - Agent lifecycle management
    - Multiple execution strategies
    - Resource management and quota enforcement
    - Fallback mechanisms and error recovery
    - Performance monitoring and metrics
    - Integration with AgentsMCP ecosystem
    """
    
    def __init__(
        self,
        agent_manager: "AgentManager",
        resource_limits: Optional[Dict[str, Union[int, float]]] = None,
        fallback_config: Optional[FallbackConfig] = None,
        performance_tracking: bool = True,
    ):
        self.agent_manager = agent_manager
        self.log = logging.getLogger(__name__)
        
        # Initialize resource manager
        limits = resource_limits or {}
        self.resource_manager = ResourceManager(
            memory_limit_mb=limits.get("memory_mb", 5000),
            cpu_limit_percent=limits.get("cpu_percent", 80.0),
            cost_limit_eur=limits.get("cost_eur", 100.0),
            max_concurrent_agents=limits.get("max_agents", 50),
            max_concurrent_executions=limits.get("max_executions", 20),
        )
        
        # Initialize agent loader
        self.agent_loader = AgentLoader(
            agent_manager=agent_manager,
            resource_manager=self.resource_manager,
            max_concurrent_loads=limits.get("max_concurrent_loads", 10),
            max_cached_agents=limits.get("max_cached_agents", 50),
        )
        
        # Initialize execution engine
        self.execution_engine = ExecutionEngine(
            agent_loader=self.agent_loader,
            resource_manager=self.resource_manager,
            agent_manager=agent_manager,
        )
        
        # Configuration
        self.fallback_config = fallback_config or FallbackConfig()
        self.performance_tracking = performance_tracking
        
        # State tracking
        self.active_orchestrations: Dict[str, TeamExecution] = {}
        self.orchestration_history: List[TeamExecution] = []
        self.metrics = OrchestrationMetrics()
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        self.log.info("DynamicOrchestrator initialized with resource limits: %s", limits)
    
    async def start(self) -> None:
        """Start the orchestrator and background services."""
        # Start agent loader background cleanup
        await self.agent_loader.start_background_cleanup()
        
        # Start maintenance task
        if self._maintenance_task is None or self._maintenance_task.done():
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        self.log.info("DynamicOrchestrator started")
    
    async def _maintenance_loop(self) -> None:
        """Background maintenance loop for resource cleanup and metrics."""
        while not self._shutdown:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Cleanup expired resources
                await self.resource_manager.cleanup_expired_allocations()
                
                # Update metrics
                if self.performance_tracking:
                    await self._update_performance_metrics()
                
                # Cleanup old orchestration history
                await self._cleanup_orchestration_history()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.error("Maintenance loop error: %s", e)
    
    async def orchestrate_team(
        self,
        team_spec: TeamComposition,
        objective: str,
        tasks: Optional[List[TaskEnvelopeV1]] = None,
        resource_constraints: Optional[ResourceConstraints] = None,
        progress_callback: Optional[Callable[[ExecutionProgress], None]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> TeamExecution:
        """Orchestrate a team to execute the given objective.
        
        Args:
            team_spec: Team composition specification
            objective: High-level objective for the team
            tasks: Optional pre-defined tasks; if None, will be generated
            resource_constraints: Optional resource limitations
            progress_callback: Optional callback for progress updates
            timeout_seconds: Optional execution timeout
            
        Returns:
            TeamExecution result with status and outputs
            
        Raises:
            OrchestrationError: For various orchestration failures
        """
        start_time = time.time()
        orchestration_id = f"orch_{int(start_time)}_{str(uuid.uuid4())[:8]}"
        
        self.log.info("Starting orchestration %s for objective: %s", 
                     orchestration_id, objective[:100])
        
        # Validate team composition
        await self._validate_team_composition(team_spec, resource_constraints)
        
        # Generate tasks if not provided
        if tasks is None:
            tasks = await self._generate_tasks_from_objective(objective, team_spec)
        
        # Pre-flight resource check
        await self._ensure_sufficient_resources(team_spec, resource_constraints)
        
        execution: Optional[TeamExecution] = None
        
        try:
            # Execute with fallbacks if enabled
            execution = await self._execute_with_fallbacks(
                orchestration_id=orchestration_id,
                team_spec=team_spec,
                objective=objective,
                tasks=tasks,
                progress_callback=progress_callback,
                timeout_seconds=timeout_seconds,
            )
            
            # Track success
            self.metrics.successful_executions += 1
            
        except Exception as e:
            self.log.error("Orchestration %s failed: %s", orchestration_id, e)
            self.metrics.failed_executions += 1
            
            # Create failure execution record
            execution = TeamExecution(
                execution_id=orchestration_id,
                team_composition=team_spec,
                objective=objective,
                status=ExecutionStatus.FAILED,
                progress=ExecutionProgress(total_tasks=len(tasks)),
                errors=[str(e)],
            )
            
            raise OrchestrationError(f"Orchestration failed: {e}") from e
        
        finally:
            # Update metrics
            self.metrics.teams_orchestrated += 1
            
            if execution:
                # Store in history
                self.orchestration_history.append(execution)
                
                # Update average execution time
                total_time = time.time() - start_time
                self._update_average_execution_time(total_time)
                
                self.log.info("Orchestration %s completed in %.2fs with status: %s",
                             orchestration_id, total_time, execution.status.value)
        
        return execution
    
    async def _execute_with_fallbacks(
        self,
        orchestration_id: str,
        team_spec: TeamComposition,
        objective: str,
        tasks: List[TaskEnvelopeV1],
        progress_callback: Optional[Callable[[ExecutionProgress], None]],
        timeout_seconds: Optional[int],
    ) -> TeamExecution:
        """Execute with fallback mechanisms."""
        
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self.fallback_config.max_retry_attempts + 1):
            try:
                self.log.info("Orchestration attempt %d/%d for %s", 
                             attempt, self.fallback_config.max_retry_attempts, orchestration_id)
                
                # Use fallback agents on retry if enabled
                if attempt > 1 and self.fallback_config.use_fallback_agents:
                    team_spec = await self._apply_fallback_agents(team_spec)
                
                # Execute team
                execution = await self.execution_engine.execute_team(
                    team_composition=team_spec,
                    objective=objective,
                    tasks=tasks,
                    progress_callback=progress_callback,
                    timeout_seconds=timeout_seconds,
                )
                
                # Check if execution succeeded sufficiently
                if await self._is_execution_successful(execution):
                    return execution
                
                # If not successful enough, treat as failure for retry
                last_error = OrchestrationError(
                    f"Execution completed but with {execution.progress.failed_tasks} failed tasks"
                )
                
            except Exception as e:
                last_error = e
                self.log.warning("Orchestration attempt %d failed: %s", attempt, e)
                
                if not self.fallback_config.enable_fallbacks:
                    raise e
                
                # Wait before retry
                if attempt < self.fallback_config.max_retry_attempts:
                    await asyncio.sleep(self.fallback_config.fallback_delay_seconds)
        
        # All attempts failed
        raise OrchestrationError(
            f"All {self.fallback_config.max_retry_attempts} orchestration attempts failed. "
            f"Last error: {last_error}"
        )
    
    async def _validate_team_composition(
        self,
        team_spec: TeamComposition,
        resource_constraints: Optional[ResourceConstraints],
    ) -> None:
        """Validate team composition and constraints."""
        
        if not team_spec.primary_team:
            raise TeamLoadError("Team composition must have at least one primary agent")
        
        # Validate agent roles exist
        from ..roles.registry import RoleRegistry
        registry = RoleRegistry()
        
        for agent_spec in team_spec.primary_team:
            try:
                role = RoleName(agent_spec.role)
                if not registry.get_role_class(role):
                    raise TeamLoadError(f"Unknown role: {agent_spec.role}")
            except ValueError:
                raise TeamLoadError(f"Invalid role name: {agent_spec.role}")
        
        # Validate resource constraints if provided
        if resource_constraints:
            if resource_constraints.max_agents < len(team_spec.primary_team):
                raise InsufficientResourcesError(
                    f"Team requires {len(team_spec.primary_team)} agents but limit is {resource_constraints.max_agents}"
                )
    
    async def _generate_tasks_from_objective(
        self,
        objective: str,
        team_spec: TeamComposition,
    ) -> List[TaskEnvelopeV1]:
        """Generate tasks from high-level objective."""
        
        # For now, create a single task for each agent
        # In a more sophisticated implementation, this would use AI to decompose the objective
        tasks = []
        
        for i, agent_spec in enumerate(team_spec.primary_team):
            task = TaskEnvelopeV1(
                objective=f"Execute {agent_spec.role} responsibilities for: {objective}",
                inputs={"role": agent_spec.role, "specializations": agent_spec.specializations},
                constraints=["Follow role responsibilities", "Collaborate with team"],
                routing={"model": agent_spec.model_assignment, "effort": "medium"},
            )
            tasks.append(task)
        
        return tasks
    
    async def _ensure_sufficient_resources(
        self,
        team_spec: TeamComposition,
        resource_constraints: Optional[ResourceConstraints],
    ) -> None:
        """Ensure sufficient resources are available for the team."""
        
        # Estimate resource requirements
        estimated_requirements = await self._estimate_team_resource_requirements(team_spec)
        
        # Check availability
        available = await self.resource_manager.check_resource_availability(estimated_requirements)
        
        if not available:
            resource_status = self.resource_manager.get_resource_status()
            raise InsufficientResourcesError(
                f"Insufficient resources for team execution. Current status: {resource_status}"
            )
    
    async def _estimate_team_resource_requirements(
        self,
        team_spec: TeamComposition,
    ) -> Dict[ResourceType, Union[int, float]]:
        """Estimate resource requirements for the team."""
        
        # Estimate memory per agent type
        total_memory = 0.0
        for agent_spec in team_spec.primary_team:
            role = RoleName(agent_spec.role)
            model_assignment = ModelAssignment(agent_type=agent_spec.model_assignment)
            
            # Use agent loader's estimation
            memory_estimate = self.agent_loader._estimate_memory_usage(role, model_assignment)
            total_memory += memory_estimate
        
        return {
            ResourceType.MEMORY: total_memory,
            ResourceType.AGENT_SLOTS: len(team_spec.primary_team),
            ResourceType.CONCURRENT_EXECUTIONS: 1,
        }
    
    async def _apply_fallback_agents(self, team_spec: TeamComposition) -> TeamComposition:
        """Apply fallback agents from the team composition."""
        
        if not team_spec.fallback_agents:
            return team_spec
        
        # Create new team spec with fallback agents
        fallback_team = TeamComposition(
            primary_team=team_spec.fallback_agents,
            fallback_agents=[],
            load_order=[agent.role for agent in team_spec.fallback_agents],
            coordination_strategy=team_spec.coordination_strategy,
            confidence_score=team_spec.confidence_score * 0.8,  # Lower confidence for fallbacks
            rationale=f"Fallback team: {team_spec.rationale}",
        )
        
        self.log.info("Applied fallback agents: %s", 
                     [agent.role for agent in fallback_team.primary_team])
        
        return fallback_team
    
    async def _is_execution_successful(self, execution: TeamExecution) -> bool:
        """Check if execution is considered successful."""
        
        if execution.status == ExecutionStatus.COMPLETED:
            return True
        
        if execution.status == ExecutionStatus.FAILED:
            # Check if graceful degradation allows partial success
            if self.fallback_config.graceful_degradation:
                success_rate = (
                    execution.progress.completed_tasks / execution.progress.total_tasks
                    if execution.progress.total_tasks > 0 else 0.0
                )
                return success_rate >= 0.5  # At least 50% success
        
        return False
    
    async def _update_performance_metrics(self) -> None:
        """Update performance metrics from various components."""
        
        # Get resource utilization
        resource_status = self.resource_manager.get_resource_status()
        for resource_type, quota_info in resource_status["quotas"].items():
            self.metrics.resource_utilization[resource_type] = quota_info["utilization"]
        
        # Get agent stats
        agent_stats = self.agent_loader.get_agent_stats()
        self.metrics.total_agents_loaded = agent_stats["total_agents"]
    
    def _update_average_execution_time(self, execution_time: float) -> None:
        """Update running average of execution times."""
        current_avg = self.metrics.average_execution_time
        total_executions = self.metrics.teams_orchestrated
        
        # Running average calculation
        self.metrics.average_execution_time = (
            (current_avg * (total_executions - 1) + execution_time) / total_executions
        )
    
    async def _cleanup_orchestration_history(self, max_history: int = 100) -> None:
        """Clean up old orchestration history."""
        if len(self.orchestration_history) > max_history:
            # Keep only the most recent entries
            self.orchestration_history = self.orchestration_history[-max_history:]
    
    def get_orchestration_status(self, orchestration_id: str) -> Optional[TeamExecution]:
        """Get status of a specific orchestration."""
        
        # Check active orchestrations
        execution = self.active_orchestrations.get(orchestration_id)
        if execution:
            return execution
        
        # Check history
        for execution in self.orchestration_history:
            if execution.execution_id == orchestration_id:
                return execution
        
        return None
    
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration metrics."""
        
        return {
            "orchestrator_metrics": {
                "teams_orchestrated": self.metrics.teams_orchestrated,
                "successful_executions": self.metrics.successful_executions,
                "failed_executions": self.metrics.failed_executions,
                "success_rate": (
                    self.metrics.successful_executions / self.metrics.teams_orchestrated
                    if self.metrics.teams_orchestrated > 0 else 0.0
                ),
                "average_execution_time": self.metrics.average_execution_time,
                "total_agents_loaded": self.metrics.total_agents_loaded,
                "resource_utilization": self.metrics.resource_utilization,
            },
            "resource_manager_status": self.resource_manager.get_resource_status(),
            "agent_loader_stats": self.agent_loader.get_agent_stats(),
            "execution_engine_stats": {
                "active_executions": len(self.execution_engine.active_executions),
                "execution_strategies": list(self.execution_engine.strategies.keys()),
            },
            "active_orchestrations": len(self.active_orchestrations),
            "orchestration_history_size": len(self.orchestration_history),
        }
    
    async def cancel_orchestration(self, orchestration_id: str) -> bool:
        """Cancel a running orchestration."""
        
        # Try to cancel in execution engine
        cancelled = await self.execution_engine.cancel_execution(orchestration_id)
        
        if cancelled:
            # Remove from active orchestrations
            self.active_orchestrations.pop(orchestration_id, None)
            self.log.info("Cancelled orchestration %s", orchestration_id)
        
        return cancelled
    
    async def shutdown(self) -> None:
        """Shutdown orchestrator and clean up resources."""
        self.log.info("Shutting down DynamicOrchestrator")
        self._shutdown = True
        
        # Cancel maintenance task
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all active orchestrations
        for orchestration_id in list(self.active_orchestrations.keys()):
            await self.cancel_orchestration(orchestration_id)
        
        # Shutdown components
        await self.agent_loader.shutdown()
        
        self.log.info("DynamicOrchestrator shutdown complete")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {},
        }
        
        try:
            # Check resource manager
            resource_status = self.resource_manager.get_resource_status()
            health_status["components"]["resource_manager"] = {
                "status": "healthy",
                "circuit_breakers": resource_status["circuit_breakers"],
            }
            
            # Check agent loader
            agent_stats = self.agent_loader.get_agent_stats()
            health_status["components"]["agent_loader"] = {
                "status": "healthy",
                "cached_agents": agent_stats["total_agents"],
                "cache_utilization": agent_stats["cache_utilization"],
            }
            
            # Check execution engine
            health_status["components"]["execution_engine"] = {
                "status": "healthy",
                "active_executions": len(self.execution_engine.active_executions),
            }
            
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["error"] = str(e)
        
        return health_status