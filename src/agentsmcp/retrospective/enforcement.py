"""Orchestrator enforcement system for retrospective action points.

This module provides the enforcement framework that tracks and validates
implementation of all action points before allowing next task execution.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from .data_models import (
    ComprehensiveRetrospectiveReport,
    ActionPoint,
    EnforcementPlan,
    ValidationCriterion,
    ImplementationStatus,
    ReadinessAssessment,
    PriorityLevel,
)


class EnforcementPlanError(Exception):
    """Raised when enforcement plan creation fails."""
    pass


class ValidationSetupError(Exception):
    """Raised when validation setup fails."""
    pass


class ImplementationBlockedError(Exception):
    """Raised when implementation is blocked and cannot proceed."""
    pass


class SystemState:
    """Current state of the orchestrator system."""
    
    def __init__(self):
        self.configuration: Dict[str, Any] = {}
        self.active_agents: Set[str] = set()
        self.performance_metrics: Dict[str, float] = {}
        self.last_updated: datetime = datetime.now(timezone.utc)
        self.health_status: str = "healthy"
        self.pending_changes: List[Dict[str, Any]] = []


class OrchestratorEnforcementSystem:
    """System for enforcing implementation of retrospective action points."""
    
    def __init__(
        self,
        validation_timeout: int = 60,
        max_retry_attempts: int = 3,
        logger: Optional[logging.Logger] = None,
    ):
        self.validation_timeout = validation_timeout
        self.max_retry_attempts = max_retry_attempts
        self.log = logger or logging.getLogger(__name__)
        
        # State tracking
        self._active_enforcements: Dict[str, EnforcementPlan] = {}
        self._validation_history: List[Dict[str, Any]] = []
        self._system_state_cache: Optional[SystemState] = None
        self._blocked_implementations: Set[str] = set()
        
        self.log.info("OrchestratorEnforcementSystem initialized")
    
    async def create_enforcement_plan(
        self,
        comprehensive_report: ComprehensiveRetrospectiveReport,
        current_system_state: SystemState,
        implementation_capabilities: Optional[Dict[str, bool]] = None,
    ) -> EnforcementPlan:
        """Create comprehensive enforcement plan for action points.
        
        Args:
            comprehensive_report: Report containing action points to enforce
            current_system_state: Current state of orchestrator system
            implementation_capabilities: What can be automatically implemented
            
        Returns:
            EnforcementPlan: Complete enforcement plan with sequencing
            
        Raises:
            EnforcementPlanError: If plan creation fails
        """
        try:
            self.log.info(
                "Creating enforcement plan for %d action points",
                len(comprehensive_report.action_points)
            )
            
            capabilities = implementation_capabilities or {}
            
            # Analyze action points and dependencies
            prioritized_actions = await self._prioritize_action_points(
                comprehensive_report.action_points
            )
            
            # Create implementation sequence
            implementation_sequence = await self._create_implementation_sequence(
                prioritized_actions, current_system_state
            )
            
            # Setup validation steps
            validation_steps = await self._setup_validation_steps(
                prioritized_actions, capabilities
            )
            
            # Estimate completion time
            estimated_completion = await self._estimate_completion_time(
                prioritized_actions, capabilities
            )
            
            # Create rollback procedures
            rollback_procedures = await self._create_rollback_procedures(
                prioritized_actions, current_system_state
            )
            
            # Create enforcement plan
            plan = EnforcementPlan(
                action_points=prioritized_actions,
                implementation_sequence=implementation_sequence,
                validation_steps=validation_steps,
                estimated_completion_time=estimated_completion,
                rollback_procedures=rollback_procedures,
            )
            
            # Track active enforcement
            self._active_enforcements[plan.plan_id] = plan
            
            self.log.info(
                "Enforcement plan created with %d actions in sequence",
                len(implementation_sequence)
            )
            
            return plan
            
        except Exception as e:
            self.log.error("Failed to create enforcement plan: %s", e)
            raise EnforcementPlanError(f"Plan creation failed: {str(e)}")
    
    async def execute_enforcement_plan(
        self,
        plan: EnforcementPlan,
        system_state: SystemState,
    ) -> Tuple[bool, List[str]]:
        """Execute the enforcement plan and validate all action points.
        
        Args:
            plan: Enforcement plan to execute
            system_state: Current system state for implementation
            
        Returns:
            Tuple[bool, List[str]]: (success, list of error messages)
        """
        self.log.info("Starting enforcement plan execution: %s", plan.plan_id)
        
        errors = []
        completed_actions = []
        
        try:
            for action_id in plan.implementation_sequence:
                action = next((a for a in plan.action_points if a.action_id == action_id), None)
                if not action:
                    error_msg = f"Action {action_id} not found in plan"
                    self.log.error(error_msg)
                    errors.append(error_msg)
                    continue
                
                self.log.debug("Implementing action: %s", action.title)
                
                # Check if action is blocked
                if action_id in self._blocked_implementations:
                    error_msg = f"Action {action_id} is blocked and cannot be implemented"
                    self.log.warning(error_msg)
                    errors.append(error_msg)
                    continue
                
                # Execute implementation
                success, impl_errors = await self._implement_action_point(action, system_state)
                
                if success:
                    action.status = ImplementationStatus.COMPLETED
                    action.completed_at = datetime.now(timezone.utc)
                    completed_actions.append(action_id)
                    
                    # Validate implementation
                    validation_success, validation_errors = await self._validate_action_implementation(
                        action, plan
                    )
                    
                    if not validation_success:
                        error_msg = f"Validation failed for action {action_id}: {'; '.join(validation_errors)}"
                        self.log.error(error_msg)
                        errors.append(error_msg)
                        action.status = ImplementationStatus.BLOCKED
                else:
                    error_msg = f"Implementation failed for action {action_id}: {'; '.join(impl_errors)}"
                    self.log.error(error_msg)
                    errors.append(error_msg)
                    action.status = ImplementationStatus.BLOCKED
                    self._blocked_implementations.add(action_id)
            
            # Overall success if no errors and all critical actions completed
            critical_actions = [a for a in plan.action_points if a.priority == PriorityLevel.CRITICAL]
            critical_completed = all(a.status == ImplementationStatus.COMPLETED for a in critical_actions)
            
            overall_success = len(errors) == 0 and critical_completed
            
            self.log.info(
                "Enforcement plan execution completed: %d/%d actions successful, %d errors",
                len(completed_actions), len(plan.implementation_sequence), len(errors)
            )
            
            return overall_success, errors
            
        except Exception as e:
            error_msg = f"Enforcement plan execution failed: {str(e)}"
            self.log.error(error_msg)
            errors.append(error_msg)
            return False, errors
    
    async def assess_system_readiness(
        self,
        enforcement_plan: Optional[EnforcementPlan] = None,
        system_state: Optional[SystemState] = None,
    ) -> ReadinessAssessment:
        """Assess if system is ready for next task execution.
        
        Args:
            enforcement_plan: Optional enforcement plan to check
            system_state: Optional current system state
            
        Returns:
            ReadinessAssessment: Complete readiness assessment
        """
        self.log.debug("Assessing system readiness for next task execution")
        
        assessment = ReadinessAssessment()
        
        try:
            # Check enforcement plan completion if provided
            if enforcement_plan:
                pending_actions = [
                    a.action_id for a in enforcement_plan.action_points
                    if a.status not in [ImplementationStatus.COMPLETED, ImplementationStatus.CANCELLED]
                ]
                assessment.pending_action_points = pending_actions
                
                # Check for critical blocking issues
                critical_blocked = [
                    a.title for a in enforcement_plan.action_points
                    if a.priority == PriorityLevel.CRITICAL and a.status == ImplementationStatus.BLOCKED
                ]
                assessment.blocking_issues.extend(critical_blocked)
            
            # Check system health indicators
            if system_state:
                assessment.system_health_indicators = await self._assess_system_health(system_state)
                
                # Overall readiness score calculation
                health_scores = list(assessment.system_health_indicators.values())
                if health_scores:
                    avg_health = sum(health_scores) / len(health_scores)
                else:
                    avg_health = 0.7  # Default neutral
                
                # Adjust for blocking issues and pending actions
                penalty = len(assessment.blocking_issues) * 0.2
                penalty += len(assessment.pending_action_points) * 0.1
                
                assessment.overall_readiness_score = max(0.0, min(1.0, avg_health - penalty))
            else:
                assessment.overall_readiness_score = 0.5  # Unknown state
            
            # Determine clearance for next task
            assessment.next_task_clearance = (
                assessment.overall_readiness_score >= 0.7 and
                len(assessment.blocking_issues) == 0 and
                len(assessment.pending_action_points) <= 2  # Allow minor pending items
            )
            
            # Generate readiness notes
            if assessment.next_task_clearance:
                assessment.readiness_notes = "System ready for next task execution"
            elif assessment.blocking_issues:
                assessment.readiness_notes = f"Blocked by {len(assessment.blocking_issues)} critical issues"
            elif len(assessment.pending_action_points) > 2:
                assessment.readiness_notes = f"{len(assessment.pending_action_points)} action points pending completion"
            else:
                assessment.readiness_notes = f"System health score too low: {assessment.overall_readiness_score:.2f}"
            
            # Estimate time to ready if not ready
            if not assessment.next_task_clearance:
                assessment.estimated_time_to_ready = await self._estimate_time_to_ready(assessment)
            
            self.log.info(
                "Readiness assessment completed: score=%.2f, clearance=%s",
                assessment.overall_readiness_score,
                assessment.next_task_clearance
            )
            
            return assessment
            
        except Exception as e:
            self.log.error("Readiness assessment failed: %s", e)
            # Return conservative assessment on failure
            assessment.overall_readiness_score = 0.3
            assessment.next_task_clearance = False
            assessment.readiness_notes = f"Assessment failed: {str(e)}"
            assessment.blocking_issues = ["System assessment failure"]
            return assessment
    
    async def validate_action_completion(
        self,
        action_point: ActionPoint,
        validation_criteria: List[ValidationCriterion],
    ) -> Tuple[bool, List[str]]:
        """Validate that an action point has been properly completed.
        
        Args:
            action_point: Action point to validate
            validation_criteria: Criteria for validation
            
        Returns:
            Tuple[bool, List[str]]: (validation success, validation results)
        """
        self.log.debug("Validating action completion: %s", action_point.title)
        
        validation_results = []
        all_passed = True
        
        try:
            for criterion in validation_criteria:
                if criterion.action_id != action_point.action_id:
                    continue
                
                self.log.debug("Applying validation criterion: %s", criterion.description)
                
                # Execute validation based on type
                if criterion.validation_type == "automated":
                    success, result = await self._execute_automated_validation(criterion)
                elif criterion.validation_type == "metric-based":
                    success, result = await self._execute_metric_validation(criterion)
                else:
                    success, result = await self._execute_manual_validation(criterion)
                
                validation_results.append(result)
                if not success:
                    all_passed = False
            
            # Update action point with validation results
            action_point.validation_results = validation_results
            
            if all_passed:
                self.log.info("Action validation successful: %s", action_point.title)
            else:
                self.log.warning("Action validation failed: %s", action_point.title)
            
            return all_passed, validation_results
            
        except Exception as e:
            error_msg = f"Validation execution failed: {str(e)}"
            self.log.error(error_msg)
            return False, [error_msg]
    
    # Private implementation methods
    
    async def _prioritize_action_points(
        self, action_points: List[ActionPoint]
    ) -> List[ActionPoint]:
        """Prioritize action points for implementation sequence."""
        
        # Sort by priority, then by estimated effort (low effort first within same priority)
        priority_order = {
            PriorityLevel.CRITICAL: 0,
            PriorityLevel.HIGH: 1,
            PriorityLevel.MEDIUM: 2,
            PriorityLevel.LOW: 3,
        }
        
        return sorted(
            action_points,
            key=lambda a: (priority_order.get(a.priority, 4), a.estimated_effort_hours)
        )
    
    async def _create_implementation_sequence(
        self, action_points: List[ActionPoint], system_state: SystemState
    ) -> List[str]:
        """Create optimal implementation sequence considering dependencies."""
        
        # Simple sequence based on priority for now
        # In a more sophisticated system, this would analyze dependencies
        return [action.action_id for action in action_points]
    
    async def _setup_validation_steps(
        self,
        action_points: List[ActionPoint],
        capabilities: Dict[str, bool],
    ) -> List[Dict[str, Any]]:
        """Setup validation steps for each action point."""
        
        validation_steps = []
        
        for action in action_points:
            # Create validation criteria based on action characteristics
            criteria = []
            
            for success_metric in action.success_metrics:
                criterion = {
                    "action_id": action.action_id,
                    "description": f"Validate: {success_metric}",
                    "validation_type": "metric-based" if "score" in success_metric.lower() else "manual",
                    "expected_outcome": success_metric,
                }
                criteria.append(criterion)
            
            # Add implementation-specific validation
            if action.implementation_type == "automatic":
                criteria.append({
                    "action_id": action.action_id,
                    "description": "Verify automatic implementation executed successfully",
                    "validation_type": "automated",
                    "expected_outcome": "Implementation completed without errors",
                })
            
            validation_steps.append({
                "action_id": action.action_id,
                "criteria": criteria,
                "timeout_seconds": self.validation_timeout,
            })
        
        return validation_steps
    
    async def _estimate_completion_time(
        self,
        action_points: List[ActionPoint],
        capabilities: Dict[str, bool],
    ) -> str:
        """Estimate total completion time for all action points."""
        
        total_hours = sum(action.estimated_effort_hours for action in action_points)
        
        # Adjust for automation capabilities
        automatic_actions = [a for a in action_points if a.implementation_type == "automatic"]
        automatic_time_saved = sum(a.estimated_effort_hours * 0.8 for a in automatic_actions)  # 80% time savings
        
        adjusted_hours = total_hours - automatic_time_saved
        
        if adjusted_hours <= 2:
            return "0-2 hours"
        elif adjusted_hours <= 8:
            return "2-8 hours (same day)"
        elif adjusted_hours <= 24:
            return "8-24 hours (1-3 days)"
        else:
            return f"{adjusted_hours:.0f} hours ({adjusted_hours/24:.1f} days)"
    
    async def _create_rollback_procedures(
        self,
        action_points: List[ActionPoint],
        system_state: SystemState,
    ) -> List[str]:
        """Create rollback procedures for implementation failures."""
        
        procedures = [
            "Stop implementation sequence immediately",
            "Assess impact of partial implementation",
            "Revert any system configuration changes",
            "Restore system to pre-implementation state",
            "Log rollback actions for analysis",
            "Notify stakeholders of implementation failure",
        ]
        
        # Add action-specific rollback steps
        for action in action_points:
            if action.implementation_type == "configuration":
                procedures.append(f"Revert configuration changes for: {action.title}")
            elif action.implementation_type == "automatic":
                procedures.append(f"Rollback automatic changes for: {action.title}")
        
        return procedures
    
    async def _implement_action_point(
        self,
        action: ActionPoint,
        system_state: SystemState,
    ) -> Tuple[bool, List[str]]:
        """Implement a specific action point."""
        
        errors = []
        
        try:
            action.status = ImplementationStatus.IN_PROGRESS
            
            # Implementation based on type
            if action.implementation_type == "automatic":
                success, impl_errors = await self._execute_automatic_implementation(action, system_state)
            elif action.implementation_type == "configuration":
                success, impl_errors = await self._execute_configuration_implementation(action, system_state)
            else:  # manual
                success, impl_errors = await self._execute_manual_implementation(action, system_state)
            
            errors.extend(impl_errors)
            
            if success:
                action.implementation_notes += f"\nImplemented successfully at {datetime.now(timezone.utc)}"
                self.log.info("Action implemented successfully: %s", action.title)
            else:
                action.implementation_notes += f"\nImplementation failed at {datetime.now(timezone.utc)}: {'; '.join(impl_errors)}"
                self.log.error("Action implementation failed: %s", action.title)
            
            return success, errors
            
        except Exception as e:
            error_msg = f"Implementation execution failed: {str(e)}"
            self.log.error(error_msg)
            errors.append(error_msg)
            action.implementation_notes += f"\nImplementation error: {error_msg}"
            return False, errors
    
    async def _execute_automatic_implementation(
        self,
        action: ActionPoint,
        system_state: SystemState,
    ) -> Tuple[bool, List[str]]:
        """Execute automatic implementation of an action point."""
        
        self.log.debug("Executing automatic implementation for: %s", action.title)
        
        # Simulate automatic implementation
        # In a real system, this would execute specific automation scripts
        
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # For demonstration, assume 90% success rate for automatic implementations
        import random
        success = random.random() > 0.1
        
        if success:
            return True, []
        else:
            return False, ["Automatic implementation script failed"]
    
    async def _execute_configuration_implementation(
        self,
        action: ActionPoint,
        system_state: SystemState,
    ) -> Tuple[bool, List[str]]:
        """Execute configuration-based implementation of an action point."""
        
        self.log.debug("Executing configuration implementation for: %s", action.title)
        
        # Simulate configuration changes
        await asyncio.sleep(0.2)  # Simulate processing time
        
        # Update system state configuration
        config_key = f"action_{action.action_id[:8]}"
        system_state.configuration[config_key] = {
            "implemented": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_title": action.title,
        }
        
        system_state.pending_changes.append({
            "type": "configuration",
            "action_id": action.action_id,
            "changes": config_key,
        })
        
        return True, []
    
    async def _execute_manual_implementation(
        self,
        action: ActionPoint,
        system_state: SystemState,
    ) -> Tuple[bool, List[str]]:
        """Execute manual implementation of an action point."""
        
        self.log.debug("Executing manual implementation for: %s", action.title)
        
        # Manual implementations require human intervention
        # For now, we'll mark them as requiring manual completion
        
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Add to pending changes for manual tracking
        system_state.pending_changes.append({
            "type": "manual",
            "action_id": action.action_id,
            "requires_manual_completion": True,
            "instructions": action.implementation_steps,
        })
        
        # Manual actions are considered "implemented" once queued for manual execution
        return True, []
    
    async def _validate_action_implementation(
        self,
        action: ActionPoint,
        plan: EnforcementPlan,
    ) -> Tuple[bool, List[str]]:
        """Validate that an action has been properly implemented."""
        
        # Find validation steps for this action
        validation_steps = [
            step for step in plan.validation_steps
            if step.get("action_id") == action.action_id
        ]
        
        if not validation_steps:
            # No specific validation required
            return True, ["No validation criteria specified"]
        
        validation_results = []
        all_passed = True
        
        for step in validation_steps:
            criteria = step.get("criteria", [])
            
            for criterion_data in criteria:
                criterion = ValidationCriterion(
                    action_id=criterion_data["action_id"],
                    description=criterion_data["description"],
                    validation_type=criterion_data["validation_type"],
                    expected_outcome=criterion_data["expected_outcome"],
                )
                
                success, result = await self._execute_validation_criterion(criterion)
                validation_results.append(result)
                
                if not success:
                    all_passed = False
        
        return all_passed, validation_results
    
    async def _execute_validation_criterion(
        self, criterion: ValidationCriterion
    ) -> Tuple[bool, str]:
        """Execute a single validation criterion."""
        
        try:
            if criterion.validation_type == "automated":
                return await self._execute_automated_validation(criterion)
            elif criterion.validation_type == "metric-based":
                return await self._execute_metric_validation(criterion)
            else:
                return await self._execute_manual_validation(criterion)
        except Exception as e:
            return False, f"Validation failed: {str(e)}"
    
    async def _execute_automated_validation(
        self, criterion: ValidationCriterion
    ) -> Tuple[bool, str]:
        """Execute automated validation."""
        
        # Simulate automated validation
        await asyncio.sleep(0.1)
        
        # For demonstration, assume 85% success rate
        import random
        success = random.random() > 0.15
        
        if success:
            return True, f"Automated validation passed: {criterion.description}"
        else:
            return False, f"Automated validation failed: {criterion.description}"
    
    async def _execute_metric_validation(
        self, criterion: ValidationCriterion
    ) -> Tuple[bool, str]:
        """Execute metric-based validation."""
        
        # Simulate metric checking
        await asyncio.sleep(0.1)
        
        # For demonstration, generate a simulated metric value
        import random
        metric_value = random.uniform(0.6, 0.95)
        
        # Check if metric meets expected outcome
        if "improvement" in criterion.expected_outcome.lower():
            success = metric_value > 0.7
        else:
            success = metric_value > 0.8
        
        result = f"Metric validation: {metric_value:.2f} {'passed' if success else 'failed'} for {criterion.description}"
        return success, result
    
    async def _execute_manual_validation(
        self, criterion: ValidationCriterion
    ) -> Tuple[bool, str]:
        """Execute manual validation."""
        
        # Manual validations are assumed to pass for demonstration
        # In a real system, this would require human verification
        
        await asyncio.sleep(0.1)
        
        return True, f"Manual validation required: {criterion.description} - marked for human review"
    
    async def _assess_system_health(self, system_state: SystemState) -> Dict[str, float]:
        """Assess current system health indicators."""
        
        health_indicators = {}
        
        # Agent availability health
        if system_state.active_agents:
            health_indicators["agent_availability"] = min(1.0, len(system_state.active_agents) / 5)  # Assume 5 is optimal
        else:
            health_indicators["agent_availability"] = 0.3
        
        # Configuration health
        config_health = 0.8 if system_state.configuration else 0.5
        health_indicators["configuration_health"] = config_health
        
        # Performance health
        if system_state.performance_metrics:
            avg_perf = sum(system_state.performance_metrics.values()) / len(system_state.performance_metrics)
            health_indicators["performance_health"] = avg_perf
        else:
            health_indicators["performance_health"] = 0.7  # Default
        
        # System status health
        status_health = {
            "healthy": 1.0,
            "degraded": 0.6,
            "unhealthy": 0.2,
        }.get(system_state.health_status, 0.5)
        
        health_indicators["system_status"] = status_health
        
        # Pending changes health (fewer pending changes = better health)
        pending_health = max(0.0, 1.0 - (len(system_state.pending_changes) * 0.1))
        health_indicators["pending_changes"] = pending_health
        
        return health_indicators
    
    async def _estimate_time_to_ready(self, assessment: ReadinessAssessment) -> str:
        """Estimate time required to reach readiness."""
        
        blocking_count = len(assessment.blocking_issues)
        pending_count = len(assessment.pending_action_points)
        
        # Simple heuristic estimation
        if blocking_count > 0:
            return f"{blocking_count * 2}-{blocking_count * 4} hours (resolve blocking issues)"
        elif pending_count > 5:
            return f"{pending_count}-{pending_count * 2} hours (complete pending actions)"
        elif assessment.overall_readiness_score < 0.5:
            return "4-8 hours (system health improvement required)"
        else:
            return "1-2 hours (minor improvements needed)"
    
    # Public utility methods
    
    def get_active_enforcements(self) -> Dict[str, EnforcementPlan]:
        """Get currently active enforcement plans."""
        return self._active_enforcements.copy()
    
    def get_blocked_implementations(self) -> Set[str]:
        """Get list of blocked implementation action IDs."""
        return self._blocked_implementations.copy()
    
    async def unblock_implementation(self, action_id: str, reason: str) -> bool:
        """Unblock a previously blocked implementation."""
        if action_id in self._blocked_implementations:
            self._blocked_implementations.remove(action_id)
            self.log.info("Unblocked implementation %s: %s", action_id, reason)
            return True
        return False
    
    def get_validation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get validation history with optional limit."""
        if limit:
            return self._validation_history[-limit:]
        return self._validation_history.copy()