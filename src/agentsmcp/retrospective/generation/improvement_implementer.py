"""
ImprovementImplementer - Safe execution of approved improvements.

This component provides a framework for safely implementing approved improvements
with proper validation, rollback mechanisms, and monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Any, AsyncContextManager
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from .improvement_generator import ImprovementOpportunity, ImprovementType, RiskLevel


class ImplementationStatus(Enum):
    """Status of improvement implementation."""
    PENDING = "pending"
    PRE_CHECKS_RUNNING = "pre_checks_running"
    PRE_CHECKS_FAILED = "pre_checks_failed"
    IMPLEMENTING = "implementing"
    POST_CHECKS_RUNNING = "post_checks_running"
    POST_CHECKS_FAILED = "post_checks_failed"
    VALIDATION_RUNNING = "validation_running"
    VALIDATION_FAILED = "validation_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"


@dataclass
class ImplementationStep:
    """A single implementation step with safety checks."""
    
    step_id: str
    description: str
    implementation_func: Callable[[], Any]
    validation_func: Optional[Callable[[], bool]] = None
    rollback_func: Optional[Callable[[], Any]] = None
    timeout_seconds: int = 300  # 5 minutes default
    retries: int = 0
    required_approvals: List[str] = field(default_factory=list)


@dataclass
class SafetyCheck:
    """Safety check configuration."""
    
    check_id: str
    description: str
    check_func: Callable[[], bool]
    severity: str = "warning"  # warning, error, critical
    timeout_seconds: int = 60
    required_for_implementation: bool = True


@dataclass
class ImplementationResult:
    """Result of improvement implementation."""
    
    improvement_id: str
    status: ImplementationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Execution details
    steps_completed: List[str] = field(default_factory=list)
    steps_failed: List[str] = field(default_factory=list)
    
    # Safety check results
    pre_checks_passed: List[str] = field(default_factory=list)
    pre_checks_failed: List[str] = field(default_factory=list)
    post_checks_passed: List[str] = field(default_factory=list)
    post_checks_failed: List[str] = field(default_factory=list)
    
    # Validation results
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Error details
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # Rollback details
    rollback_performed: bool = False
    rollback_success: bool = False
    rollback_details: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    total_duration: Optional[timedelta] = None
    implementation_duration: Optional[timedelta] = None
    
    # Success metrics
    success_metrics_met: Dict[str, bool] = field(default_factory=dict)


class ImprovementImplementer:
    """
    Safe implementation framework for approved improvements.
    
    Provides structured implementation with safety checks, rollback mechanisms,
    and comprehensive monitoring to ensure improvements are applied safely.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Implementation tracking
        self._active_implementations: Dict[str, ImplementationResult] = {}
        self._implementation_history: List[ImplementationResult] = []
        
        # Safety configurations
        self._global_safety_checks: List[SafetyCheck] = []
        self._type_specific_checks: Dict[ImprovementType, List[SafetyCheck]] = {}
        
        # Implementation strategies by type
        self._implementation_strategies = {
            ImprovementType.ALGORITHM_OPTIMIZATION: self._implement_algorithm_optimization,
            ImprovementType.RESOURCE_SCALING: self._implement_resource_scaling,
            ImprovementType.ERROR_HANDLING: self._implement_error_handling,
            ImprovementType.USER_INTERFACE: self._implement_ui_improvement,
            ImprovementType.INTEGRATION: self._implement_integration_improvement,
            ImprovementType.INFRASTRUCTURE: self._implement_infrastructure_improvement
        }
        
        # Default safety checks
        self._setup_default_safety_checks()
    
    async def implement_improvement(
        self,
        improvement: ImprovementOpportunity,
        approval_token: Optional[str] = None,
        dry_run: bool = False
    ) -> ImplementationResult:
        """
        Safely implement an approved improvement.
        
        Args:
            improvement: The improvement to implement
            approval_token: Token proving approval (if required)
            dry_run: If True, simulate implementation without making changes
            
        Returns:
            Implementation result with detailed status and metrics
        """
        try:
            self.logger.info(f"Starting implementation of improvement {improvement.opportunity_id}")
            
            # Create implementation result
            result = ImplementationResult(
                improvement_id=improvement.opportunity_id,
                status=ImplementationStatus.PENDING,
                started_at=datetime.utcnow()
            )
            
            self._active_implementations[improvement.opportunity_id] = result
            
            # Validate approval if required
            if not await self._validate_approval(improvement, approval_token):
                result.status = ImplementationStatus.FAILED
                result.error_message = "Implementation approval required but not provided"
                return result
            
            # Execute implementation pipeline
            async with self._implementation_context(improvement, dry_run):
                success = await self._execute_implementation_pipeline(
                    improvement, result, dry_run
                )
                
                if success:
                    result.status = ImplementationStatus.COMPLETED
                    result.completed_at = datetime.utcnow()
                    result.total_duration = result.completed_at - result.started_at
                    
                    self.logger.info(f"Successfully implemented improvement {improvement.opportunity_id}")
                else:
                    result.status = ImplementationStatus.FAILED
                    self.logger.warning(f"Failed to implement improvement {improvement.opportunity_id}")
            
            # Move to history
            self._implementation_history.append(result)
            if improvement.opportunity_id in self._active_implementations:
                del self._active_implementations[improvement.opportunity_id]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Implementation error for {improvement.opportunity_id}: {e}")
            
            result.status = ImplementationStatus.FAILED
            result.error_message = str(e)
            result.error_details = {"exception_type": type(e).__name__}
            
            self._implementation_history.append(result)
            if improvement.opportunity_id in self._active_implementations:
                del self._active_implementations[improvement.opportunity_id]
            
            return result
    
    async def rollback_improvement(
        self,
        improvement_id: str,
        reason: str = "Manual rollback requested"
    ) -> bool:
        """
        Roll back a previously implemented improvement.
        
        Args:
            improvement_id: ID of improvement to roll back
            reason: Reason for rollback
            
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            self.logger.info(f"Starting rollback of improvement {improvement_id}: {reason}")
            
            # Find implementation result
            implementation_result = None
            for result in self._implementation_history:
                if result.improvement_id == improvement_id:
                    implementation_result = result
                    break
            
            if not implementation_result:
                self.logger.warning(f"No implementation found for {improvement_id}")
                return False
            
            if implementation_result.status != ImplementationStatus.COMPLETED:
                self.logger.warning(f"Cannot rollback non-completed implementation {improvement_id}")
                return False
            
            # Perform rollback
            implementation_result.status = ImplementationStatus.ROLLING_BACK
            implementation_result.rollback_details = {"reason": reason, "started_at": datetime.utcnow()}
            
            # Execute rollback steps in reverse order
            rollback_success = True
            for step_id in reversed(implementation_result.steps_completed):
                try:
                    await self._execute_rollback_step(step_id)
                except Exception as e:
                    self.logger.error(f"Rollback step {step_id} failed: {e}")
                    rollback_success = False
            
            # Update status
            if rollback_success:
                implementation_result.status = ImplementationStatus.ROLLED_BACK
                implementation_result.rollback_success = True
                self.logger.info(f"Successfully rolled back improvement {improvement_id}")
            else:
                implementation_result.status = ImplementationStatus.ROLLBACK_FAILED
                implementation_result.rollback_success = False
                self.logger.error(f"Rollback failed for improvement {improvement_id}")
            
            implementation_result.rollback_performed = True
            implementation_result.rollback_details["completed_at"] = datetime.utcnow()
            
            return rollback_success
            
        except Exception as e:
            self.logger.error(f"Rollback error for {improvement_id}: {e}")
            return False
    
    async def _execute_implementation_pipeline(
        self,
        improvement: ImprovementOpportunity,
        result: ImplementationResult,
        dry_run: bool
    ) -> bool:
        """Execute the full implementation pipeline with safety checks."""
        
        try:
            # 1. Pre-implementation safety checks
            result.status = ImplementationStatus.PRE_CHECKS_RUNNING
            if not await self._run_safety_checks(improvement, result, "pre"):
                result.status = ImplementationStatus.PRE_CHECKS_FAILED
                return False
            
            # 2. Execute implementation
            result.status = ImplementationStatus.IMPLEMENTING
            impl_start = datetime.utcnow()
            
            if not await self._execute_implementation_steps(improvement, result, dry_run):
                result.status = ImplementationStatus.FAILED
                return False
            
            result.implementation_duration = datetime.utcnow() - impl_start
            
            # 3. Post-implementation safety checks
            result.status = ImplementationStatus.POST_CHECKS_RUNNING
            if not await self._run_safety_checks(improvement, result, "post"):
                result.status = ImplementationStatus.POST_CHECKS_FAILED
                # Trigger automatic rollback for high-risk failures
                if improvement.risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    await self._perform_emergency_rollback(improvement, result)
                return False
            
            # 4. Validation checks
            result.status = ImplementationStatus.VALIDATION_RUNNING
            if not await self._validate_implementation(improvement, result):
                result.status = ImplementationStatus.VALIDATION_FAILED
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Implementation pipeline error: {e}")
            result.error_message = str(e)
            return False
    
    async def _execute_implementation_steps(
        self,
        improvement: ImprovementOpportunity,
        result: ImplementationResult,
        dry_run: bool
    ) -> bool:
        """Execute the actual implementation steps."""
        
        # Get implementation strategy for this improvement type
        strategy = self._implementation_strategies.get(improvement.improvement_type)
        if not strategy:
            self.logger.warning(f"No implementation strategy for type {improvement.improvement_type}")
            # Fall back to generic implementation
            return await self._generic_implementation(improvement, result, dry_run)
        
        return await strategy(improvement, result, dry_run)
    
    async def _implement_algorithm_optimization(
        self,
        improvement: ImprovementOpportunity,
        result: ImplementationResult,
        dry_run: bool
    ) -> bool:
        """Implement algorithm optimization improvements."""
        steps = [
            "backup_current_algorithm",
            "deploy_optimized_algorithm", 
            "verify_performance_improvement",
            "update_monitoring_thresholds"
        ]
        
        for step in steps:
            try:
                if dry_run:
                    self.logger.info(f"[DRY RUN] Would execute: {step}")
                else:
                    await self._execute_algorithm_step(step, improvement)
                
                result.steps_completed.append(step)
                
            except Exception as e:
                self.logger.error(f"Algorithm step {step} failed: {e}")
                result.steps_failed.append(step)
                return False
        
        return True
    
    async def _implement_resource_scaling(
        self,
        improvement: ImprovementOpportunity,
        result: ImplementationResult,
        dry_run: bool
    ) -> bool:
        """Implement resource scaling improvements."""
        steps = [
            "analyze_current_resource_usage",
            "calculate_scaling_requirements",
            "apply_resource_changes",
            "verify_resource_availability",
            "update_resource_monitoring"
        ]
        
        for step in steps:
            try:
                if dry_run:
                    self.logger.info(f"[DRY RUN] Would execute: {step}")
                else:
                    await self._execute_scaling_step(step, improvement)
                
                result.steps_completed.append(step)
                
            except Exception as e:
                self.logger.error(f"Scaling step {step} failed: {e}")
                result.steps_failed.append(step)
                return False
        
        return True
    
    async def _implement_error_handling(
        self,
        improvement: ImprovementOpportunity,
        result: ImplementationResult,
        dry_run: bool
    ) -> bool:
        """Implement error handling improvements."""
        steps = [
            "backup_current_error_handlers",
            "deploy_improved_error_handling",
            "test_error_scenarios",
            "verify_error_rate_reduction",
            "update_error_monitoring"
        ]
        
        for step in steps:
            try:
                if dry_run:
                    self.logger.info(f"[DRY RUN] Would execute: {step}")
                else:
                    await self._execute_error_handling_step(step, improvement)
                
                result.steps_completed.append(step)
                
            except Exception as e:
                self.logger.error(f"Error handling step {step} failed: {e}")
                result.steps_failed.append(step)
                return False
        
        return True
    
    async def _generic_implementation(
        self,
        improvement: ImprovementOpportunity,
        result: ImplementationResult,
        dry_run: bool
    ) -> bool:
        """Generic implementation for unknown types."""
        # Execute the implementation steps from the improvement
        for i, step_desc in enumerate(improvement.implementation_steps):
            step_id = f"generic_step_{i}"
            
            try:
                if dry_run:
                    self.logger.info(f"[DRY RUN] Would execute: {step_desc}")
                else:
                    # Simulate implementation
                    await asyncio.sleep(0.1)  # Brief delay to simulate work
                    self.logger.info(f"Executed: {step_desc}")
                
                result.steps_completed.append(step_id)
                
            except Exception as e:
                self.logger.error(f"Generic step '{step_desc}' failed: {e}")
                result.steps_failed.append(step_id)
                return False
        
        return True
    
    async def _run_safety_checks(
        self,
        improvement: ImprovementOpportunity,
        result: ImplementationResult,
        phase: str  # "pre" or "post"
    ) -> bool:
        """Run safety checks for the specified phase."""
        
        # Get relevant safety checks
        checks = self._global_safety_checks.copy()
        type_checks = self._type_specific_checks.get(improvement.improvement_type, [])
        checks.extend(type_checks)
        
        passed_checks = []
        failed_checks = []
        
        for check in checks:
            try:
                # Run check with timeout
                check_result = await asyncio.wait_for(
                    self._execute_safety_check(check, improvement),
                    timeout=check.timeout_seconds
                )
                
                if check_result:
                    passed_checks.append(check.check_id)
                else:
                    failed_checks.append(check.check_id)
                    
                    # Check if this is a critical failure
                    if check.required_for_implementation and check.severity == "critical":
                        self.logger.critical(f"Critical safety check {check.check_id} failed")
                        
            except asyncio.TimeoutError:
                self.logger.warning(f"Safety check {check.check_id} timed out")
                failed_checks.append(check.check_id)
                
            except Exception as e:
                self.logger.error(f"Safety check {check.check_id} error: {e}")
                failed_checks.append(check.check_id)
        
        # Update result
        if phase == "pre":
            result.pre_checks_passed = passed_checks
            result.pre_checks_failed = failed_checks
        else:
            result.post_checks_passed = passed_checks  
            result.post_checks_failed = failed_checks
        
        # Determine if checks passed overall
        critical_failures = [
            check.check_id for check in checks
            if check.check_id in failed_checks and 
               check.required_for_implementation and 
               check.severity in ["error", "critical"]
        ]
        
        return len(critical_failures) == 0
    
    async def _validate_implementation(
        self,
        improvement: ImprovementOpportunity,
        result: ImplementationResult
    ) -> bool:
        """Validate that implementation achieved expected results."""
        
        validation_results = {}
        
        # Check success metrics
        for metric in improvement.success_metrics:
            try:
                # This would normally check actual metrics
                # For now, simulate metric validation
                metric_met = await self._check_success_metric(metric, improvement)
                validation_results[metric] = metric_met
                result.success_metrics_met[metric] = metric_met
                
            except Exception as e:
                self.logger.error(f"Validation error for metric '{metric}': {e}")
                validation_results[metric] = False
                result.success_metrics_met[metric] = False
        
        result.validation_results = validation_results
        
        # Consider validation successful if majority of metrics are met
        met_count = sum(1 for met in validation_results.values() if met)
        total_count = len(validation_results)
        
        success_rate = met_count / max(total_count, 1)
        return success_rate >= 0.7  # 70% threshold
    
    @asynccontextmanager
    async def _implementation_context(self, improvement: ImprovementOpportunity, dry_run: bool):
        """Context manager for safe implementation with cleanup."""
        self.logger.info(f"Entering implementation context for {improvement.opportunity_id}")
        
        try:
            # Setup implementation environment
            if not dry_run:
                await self._setup_implementation_environment(improvement)
            
            yield
            
        finally:
            # Cleanup implementation environment
            self.logger.info(f"Exiting implementation context for {improvement.opportunity_id}")
            if not dry_run:
                await self._cleanup_implementation_environment(improvement)
    
    def _setup_default_safety_checks(self):
        """Setup default safety checks."""
        
        # Global checks
        self._global_safety_checks = [
            SafetyCheck(
                check_id="system_health_check",
                description="Verify system is healthy before implementation",
                check_func=self._check_system_health,
                severity="critical",
                required_for_implementation=True
            ),
            SafetyCheck(
                check_id="backup_verification",
                description="Verify backups are available",
                check_func=self._verify_backups,
                severity="error",
                required_for_implementation=True
            )
        ]
    
    # Placeholder methods for specific implementation steps
    async def _execute_algorithm_step(self, step: str, improvement: ImprovementOpportunity):
        """Execute algorithm optimization step."""
        await asyncio.sleep(0.1)  # Simulate work
    
    async def _execute_scaling_step(self, step: str, improvement: ImprovementOpportunity):
        """Execute resource scaling step."""
        await asyncio.sleep(0.1)  # Simulate work
    
    async def _execute_error_handling_step(self, step: str, improvement: ImprovementOpportunity):
        """Execute error handling step."""
        await asyncio.sleep(0.1)  # Simulate work
    
    async def _execute_rollback_step(self, step_id: str):
        """Execute rollback for a specific step."""
        await asyncio.sleep(0.1)  # Simulate rollback
    
    async def _execute_safety_check(self, check: SafetyCheck, improvement: ImprovementOpportunity) -> bool:
        """Execute a safety check."""
        return await check.check_func()
    
    async def _check_success_metric(self, metric: str, improvement: ImprovementOpportunity) -> bool:
        """Check if a success metric is met."""
        # Simulate metric checking
        await asyncio.sleep(0.1)
        return True  # Optimistic for demo
    
    async def _validate_approval(self, improvement: ImprovementOpportunity, approval_token: Optional[str]) -> bool:
        """Validate implementation approval."""
        # High-risk improvements require approval
        if improvement.risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return approval_token is not None
        return True
    
    async def _setup_implementation_environment(self, improvement: ImprovementOpportunity):
        """Setup environment for implementation."""
        pass
    
    async def _cleanup_implementation_environment(self, improvement: ImprovementOpportunity):
        """Cleanup implementation environment."""
        pass
    
    async def _perform_emergency_rollback(self, improvement: ImprovementOpportunity, result: ImplementationResult):
        """Perform emergency rollback after critical failure."""
        self.logger.warning(f"Performing emergency rollback for {improvement.opportunity_id}")
        # This would trigger automatic rollback procedures
    
    async def _check_system_health(self) -> bool:
        """Check overall system health."""
        # Simulate health check
        return True
    
    async def _verify_backups(self) -> bool:
        """Verify backups are available."""
        # Simulate backup verification
        return True
    
    def get_active_implementations(self) -> Dict[str, ImplementationResult]:
        """Get currently active implementations."""
        return self._active_implementations.copy()
    
    def get_implementation_history(self, limit: int = 50) -> List[ImplementationResult]:
        """Get implementation history."""
        return self._implementation_history[-limit:]
    
    def get_implementation_stats(self) -> Dict[str, Any]:
        """Get implementation statistics."""
        if not self._implementation_history:
            return {}
        
        total_implementations = len(self._implementation_history)
        successful = sum(1 for r in self._implementation_history if r.status == ImplementationStatus.COMPLETED)
        
        avg_duration = None
        if self._implementation_history:
            durations = [
                r.total_duration.total_seconds() 
                for r in self._implementation_history 
                if r.total_duration
            ]
            if durations:
                avg_duration = sum(durations) / len(durations)
        
        return {
            "total_implementations": total_implementations,
            "successful_implementations": successful,
            "success_rate": successful / total_implementations if total_implementations > 0 else 0,
            "active_implementations": len(self._active_implementations),
            "average_duration_seconds": avg_duration
        }