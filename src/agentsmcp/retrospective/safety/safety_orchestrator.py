"""Safety orchestrator for coordinating the complete safety workflow."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable

from ..data_models import ActionPoint, SystemicImprovement
from .safety_config import SafetyConfig, RollbackTrigger, SafetyLevel
from .safety_validator import SafetyValidator, ValidationResult
from .rollback_manager import RollbackManager, RollbackPoint
from .health_monitor import HealthMonitor, HealthMetrics, HealthBaseline

logger = logging.getLogger(__name__)


class SafetyWorkflowState(str, Enum):
    """States of the safety workflow."""
    IDLE = "idle"
    VALIDATING = "validating"
    CREATING_ROLLBACK_POINT = "creating_rollback_point"
    COLLECTING_BASELINE = "collecting_baseline"
    APPLYING_CHANGES = "applying_changes"
    MONITORING_HEALTH = "monitoring_health"
    ROLLBACK_TRIGGERED = "rollback_triggered"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    FAILED = "failed"


class SafetyWorkflowResult(str, Enum):
    """Results of safety workflow execution."""
    SUCCESS = "success"
    VALIDATION_FAILED = "validation_failed"
    ROLLBACK_TRIGGERED = "rollback_triggered"
    ROLLBACK_FAILED = "rollback_failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class SafetyWorkflowContext:
    """Context for safety workflow execution."""
    workflow_id: str
    improvements: List[Union[ActionPoint, SystemicImprovement]]
    state: SafetyWorkflowState = SafetyWorkflowState.IDLE
    result: Optional[SafetyWorkflowResult] = None
    
    # Workflow artifacts
    validation_results: Dict[str, ValidationResult] = field(default_factory=dict)
    rollback_point: Optional[RollbackPoint] = None
    baseline: Optional[HealthBaseline] = None
    
    # Timing and metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafetyOrchestrator:
    """Orchestrates the complete safety validation and monitoring workflow."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.validator = SafetyValidator(config)
        self.rollback_manager = RollbackManager(config)
        self.health_monitor = HealthMonitor(config)
        
        # Workflow state
        self._active_workflows: Dict[str, SafetyWorkflowContext] = {}
        self._workflow_lock = asyncio.Lock()
        
        # Callbacks and hooks
        self._pre_validation_hooks: List[Callable[[SafetyWorkflowContext], Any]] = []
        self._post_validation_hooks: List[Callable[[SafetyWorkflowContext, ValidationResult], Any]] = []
        self._pre_application_hooks: List[Callable[[SafetyWorkflowContext], Any]] = []
        self._post_application_hooks: List[Callable[[SafetyWorkflowContext], Any]] = []
        self._rollback_hooks: List[Callable[[SafetyWorkflowContext, RollbackTrigger], Any]] = []
        
        # Initialize health monitoring callbacks
        self.health_monitor.add_health_change_callback(self._on_health_change)
    
    async def initialize(self):
        """Initialize the safety orchestrator."""
        try:
            await self.rollback_manager.initialize()
            
            if self.config.health_monitoring_enabled:
                await self.health_monitor.start_monitoring()
            
            self.logger.info("SafetyOrchestrator initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize SafetyOrchestrator: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the safety orchestrator."""
        try:
            await self.health_monitor.stop_monitoring()
            self.logger.info("SafetyOrchestrator shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during SafetyOrchestrator shutdown: {e}")
    
    async def execute_safe_improvement_workflow(
        self,
        improvements: List[Union[ActionPoint, SystemicImprovement]],
        improvement_implementer: Optional[Callable[[List[Union[ActionPoint, SystemicImprovement]]], Any]] = None,
        workflow_id: Optional[str] = None,
        timeout_seconds: Optional[int] = None
    ) -> SafetyWorkflowContext:
        """Execute the complete safe improvement workflow."""
        # Generate workflow ID if not provided
        if workflow_id is None:
            workflow_id = f"safety_workflow_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Create workflow context
        context = SafetyWorkflowContext(
            workflow_id=workflow_id,
            improvements=improvements,
            started_at=datetime.now(timezone.utc)
        )
        
        async with self._workflow_lock:
            self._active_workflows[workflow_id] = context
        
        try:
            # Set timeout if specified
            if timeout_seconds:
                return await asyncio.wait_for(
                    self._execute_workflow_internal(context, improvement_implementer),
                    timeout=timeout_seconds
                )
            else:
                return await self._execute_workflow_internal(context, improvement_implementer)
                
        except asyncio.TimeoutError:
            context.state = SafetyWorkflowState.FAILED
            context.result = SafetyWorkflowResult.TIMEOUT
            context.error_message = "Workflow timed out"
            context.completed_at = datetime.now(timezone.utc)
            
            self.logger.error(f"Safety workflow {workflow_id} timed out")
            return context
        
        except Exception as e:
            context.state = SafetyWorkflowState.FAILED
            context.result = SafetyWorkflowResult.ERROR
            context.error_message = str(e)
            context.completed_at = datetime.now(timezone.utc)
            
            self.logger.error(f"Safety workflow {workflow_id} failed: {e}")
            return context
        
        finally:
            async with self._workflow_lock:
                if workflow_id in self._active_workflows:
                    del self._active_workflows[workflow_id]
    
    async def _execute_workflow_internal(
        self,
        context: SafetyWorkflowContext,
        improvement_implementer: Optional[Callable]
    ) -> SafetyWorkflowContext:
        """Internal workflow execution."""
        
        try:
            # Phase 1: Validation
            context.state = SafetyWorkflowState.VALIDATING
            self.logger.info(f"Starting validation phase for workflow {context.workflow_id}")
            
            # Run pre-validation hooks
            await self._run_hooks(self._pre_validation_hooks, context)
            
            # Validate all improvements
            validation_results = await self.validator.validate_improvements_batch(
                context.improvements
            )
            context.validation_results = validation_results
            
            # Check for validation failures
            failed_validations = [
                result for result in validation_results.values()
                if not result.passed or result.has_blocking_issues()
            ]
            
            if failed_validations:
                context.state = SafetyWorkflowState.FAILED
                context.result = SafetyWorkflowResult.VALIDATION_FAILED
                context.error_message = f"Validation failed for {len(failed_validations)} improvements"
                context.completed_at = datetime.now(timezone.utc)
                
                self.logger.error(f"Validation failed for workflow {context.workflow_id}")
                return context
            
            # Run post-validation hooks
            for result in validation_results.values():
                await self._run_hooks(self._post_validation_hooks, context, result)
            
            # Phase 2: Create rollback point
            if self.config.enable_configuration_backup:
                context.state = SafetyWorkflowState.CREATING_ROLLBACK_POINT
                self.logger.info(f"Creating rollback point for workflow {context.workflow_id}")
                
                rollback_point = await self.rollback_manager.create_rollback_point(
                    name=f"Pre-improvement rollback - {context.workflow_id}",
                    description=f"Rollback point before applying {len(context.improvements)} improvements",
                    created_by="safety_orchestrator",
                    capture_configuration=True,
                    expires_in_hours=24
                )
                context.rollback_point = rollback_point
            
            # Phase 3: Collect health baseline
            if self.config.health_monitoring_enabled:
                context.state = SafetyWorkflowState.COLLECTING_BASELINE
                self.logger.info(f"Collecting health baseline for workflow {context.workflow_id}")
                
                baseline = await self.health_monitor.collect_baseline()
                context.baseline = baseline
            
            # Phase 4: Apply improvements
            context.state = SafetyWorkflowState.APPLYING_CHANGES
            self.logger.info(f"Applying improvements for workflow {context.workflow_id}")
            
            # Run pre-application hooks
            await self._run_hooks(self._pre_application_hooks, context)
            
            # Apply improvements using provided implementer
            if improvement_implementer:
                try:
                    # If in dry run mode, skip actual implementation
                    if self.config.enable_dry_run_mode:
                        self.logger.info("Dry run mode: Skipping actual improvement implementation")
                    else:
                        await self._execute_improvement_implementer(improvement_implementer, context.improvements)
                except Exception as e:
                    # Implementation failed - trigger rollback
                    self.logger.error(f"Improvement implementation failed: {e}")
                    await self._trigger_rollback(context, RollbackTrigger.MANUAL_TRIGGER)
                    return context
            
            # Run post-application hooks
            await self._run_hooks(self._post_application_hooks, context)
            
            # Phase 5: Monitor health
            if self.config.health_monitoring_enabled:
                context.state = SafetyWorkflowState.MONITORING_HEALTH
                self.logger.info(f"Monitoring health for workflow {context.workflow_id}")
                
                rollback_triggered = await self._monitor_health_with_timeout(
                    context,
                    self.config.post_change_monitoring_duration_seconds
                )
                
                if rollback_triggered:
                    return context  # Rollback was triggered
            
            # Phase 6: Success
            context.state = SafetyWorkflowState.COMPLETED
            context.result = SafetyWorkflowResult.SUCCESS
            context.completed_at = datetime.now(timezone.utc)
            
            self.logger.info(f"Safety workflow {context.workflow_id} completed successfully")
            return context
            
        except Exception as e:
            context.state = SafetyWorkflowState.FAILED
            context.result = SafetyWorkflowResult.ERROR
            context.error_message = str(e)
            context.completed_at = datetime.now(timezone.utc)
            
            self.logger.error(f"Safety workflow {context.workflow_id} failed: {e}")
            raise
    
    async def _monitor_health_with_timeout(
        self,
        context: SafetyWorkflowContext,
        duration_seconds: int
    ) -> bool:
        """Monitor health for specified duration and trigger rollback if needed."""
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(seconds=duration_seconds)
        
        while datetime.now(timezone.utc) < end_time:
            try:
                # Check for health degradation
                degradation_triggers = await self.health_monitor.check_health_degradation()
                
                if degradation_triggers:
                    self.logger.warning(f"Health degradation detected: {degradation_triggers}")
                    
                    # Trigger rollback for the first detected issue
                    await self._trigger_rollback(context, degradation_triggers[0])
                    return True  # Rollback was triggered
                
                # Wait before next check
                await asyncio.sleep(min(30, self.config.health_check_interval_seconds))
                
            except Exception as e:
                self.logger.error(f"Error during health monitoring: {e}")
        
        self.logger.info(f"Health monitoring completed without issues for {duration_seconds}s")
        return False  # No rollback triggered
    
    async def _trigger_rollback(self, context: SafetyWorkflowContext, trigger: RollbackTrigger):
        """Trigger rollback for the workflow."""
        context.state = SafetyWorkflowState.ROLLBACK_TRIGGERED
        
        self.logger.warning(f"Triggering rollback for workflow {context.workflow_id}, trigger: {trigger}")
        
        # Run rollback hooks
        await self._run_hooks(self._rollback_hooks, context, trigger)
        
        if context.rollback_point:
            context.state = SafetyWorkflowState.ROLLING_BACK
            
            success = await self.rollback_manager.execute_rollback(
                context.rollback_point.rollback_id,
                trigger
            )
            
            if success:
                context.result = SafetyWorkflowResult.ROLLBACK_TRIGGERED
                self.logger.info(f"Rollback successful for workflow {context.workflow_id}")
            else:
                context.result = SafetyWorkflowResult.ROLLBACK_FAILED
                context.error_message = "Rollback execution failed"
                self.logger.error(f"Rollback failed for workflow {context.workflow_id}")
        else:
            context.result = SafetyWorkflowResult.ROLLBACK_FAILED
            context.error_message = "No rollback point available"
            self.logger.error(f"No rollback point available for workflow {context.workflow_id}")
        
        context.completed_at = datetime.now(timezone.utc)
    
    async def _execute_improvement_implementer(
        self,
        improvement_implementer: Callable,
        improvements: List[Union[ActionPoint, SystemicImprovement]]
    ):
        """Execute the improvement implementer function."""
        if asyncio.iscoroutinefunction(improvement_implementer):
            await improvement_implementer(improvements)
        else:
            # Run synchronous function in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, improvement_implementer, improvements)
    
    async def _on_health_change(self, previous: HealthMetrics, current: HealthMetrics):
        """Handle health changes during monitoring."""
        # This is called by the health monitor when health changes
        # We can use this for additional monitoring logic if needed
        self.logger.debug(f"Health change detected: {previous.overall_status} -> {current.overall_status}")
    
    async def _run_hooks(self, hooks: List[Callable], *args, **kwargs):
        """Run a list of hooks with error handling."""
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(*args, **kwargs)
                else:
                    hook(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Hook execution failed: {e}")
    
    async def trigger_manual_rollback(self, workflow_id: str) -> bool:
        """Manually trigger rollback for a workflow."""
        context = self._active_workflows.get(workflow_id)
        if not context:
            self.logger.error(f"No active workflow found: {workflow_id}")
            return False
        
        await self._trigger_rollback(context, RollbackTrigger.MANUAL_TRIGGER)
        return True
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[SafetyWorkflowContext]:
        """Get the status of a workflow."""
        return self._active_workflows.get(workflow_id)
    
    async def list_active_workflows(self) -> List[SafetyWorkflowContext]:
        """List all active workflows."""
        return list(self._active_workflows.values())
    
    # Hook registration methods
    def add_pre_validation_hook(self, hook: Callable[[SafetyWorkflowContext], Any]):
        """Add a pre-validation hook."""
        self._pre_validation_hooks.append(hook)
    
    def add_post_validation_hook(self, hook: Callable[[SafetyWorkflowContext, ValidationResult], Any]):
        """Add a post-validation hook."""
        self._post_validation_hooks.append(hook)
    
    def add_pre_application_hook(self, hook: Callable[[SafetyWorkflowContext], Any]):
        """Add a pre-application hook."""
        self._pre_application_hooks.append(hook)
    
    def add_post_application_hook(self, hook: Callable[[SafetyWorkflowContext], Any]):
        """Add a post-application hook."""
        self._post_application_hooks.append(hook)
    
    def add_rollback_hook(self, hook: Callable[[SafetyWorkflowContext, RollbackTrigger], Any]):
        """Add a rollback hook."""
        self._rollback_hooks.append(hook)
    
    # Integration methods for improvement system
    async def safe_apply_improvements(
        self,
        improvements: List[Union[ActionPoint, SystemicImprovement]],
        implementer_function: Callable,
        timeout_seconds: int = 300
    ) -> SafetyWorkflowResult:
        """High-level method for safely applying improvements with full safety workflow."""
        context = await self.execute_safe_improvement_workflow(
            improvements=improvements,
            improvement_implementer=implementer_function,
            timeout_seconds=timeout_seconds
        )
        
        return context.result or SafetyWorkflowResult.ERROR
    
    async def validate_improvements_only(
        self,
        improvements: List[Union[ActionPoint, SystemicImprovement]]
    ) -> Dict[str, ValidationResult]:
        """Validate improvements without applying them."""
        return await self.validator.validate_improvements_batch(improvements)
    
    async def create_safety_checkpoint(self, name: str, description: str) -> RollbackPoint:
        """Create a safety checkpoint for manual rollback."""
        return await self.rollback_manager.create_rollback_point(
            name=name,
            description=description,
            created_by="manual",
            capture_configuration=True,
            expires_in_hours=48
        )