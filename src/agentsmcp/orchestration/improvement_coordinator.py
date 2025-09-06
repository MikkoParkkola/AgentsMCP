"""
ImprovementCoordinator - Improvement Lifecycle Management

This component manages the complete lifecycle of improvements from identification
to implementation and validation. It coordinates with the ProcessCoach and
RetrospectiveOrchestrator to ensure improvements are properly tracked,
prioritized, and executed.

Key responsibilities:
- Improvement lifecycle management (creation → implementation → validation)
- Progress tracking and measurement
- Impact assessment and ROI calculation
- Rollback coordination for failed improvements
- Cross-improvement dependency management
- Improvement backlog management and prioritization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from ..retrospective import (
    ActionPoint,
    ImplementationStatus,
    PriorityLevel,
    ImprovementCategory
)
from .models import TaskResult


logger = logging.getLogger(__name__)


class ImprovementLifecycleStage(Enum):
    """Stages in the improvement lifecycle."""
    IDENTIFIED = "identified"
    ANALYZED = "analyzed"
    PRIORITIZED = "prioritized"
    APPROVED = "approved"
    PLANNED = "planned"
    IMPLEMENTING = "implementing"
    IMPLEMENTED = "implemented"
    VALIDATING = "validating"
    VALIDATED = "validated"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class ImprovementImpactLevel(Enum):
    """Impact levels for improvements."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DependencyType(Enum):
    """Types of dependencies between improvements."""
    PREREQUISITE = "prerequisite"  # Must complete before this improvement
    BLOCKING = "blocking"  # This improvement blocks others
    CONFLICTING = "conflicting"  # Cannot be implemented together
    SYNERGISTIC = "synergistic"  # Better when implemented together


@dataclass
class ImprovementDependency:
    """Represents a dependency between improvements."""
    source_improvement_id: str
    target_improvement_id: str
    dependency_type: DependencyType
    description: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementMetrics:
    """Metrics tracking for an improvement."""
    implementation_start: Optional[datetime] = None
    implementation_end: Optional[datetime] = None
    validation_start: Optional[datetime] = None
    validation_end: Optional[datetime] = None
    
    # Impact measurements
    performance_before: Optional[float] = None
    performance_after: Optional[float] = None
    efficiency_gain: Optional[float] = None
    cost_impact: Optional[float] = None
    
    # Success measurements
    success_rate: float = 0.0
    user_satisfaction_score: Optional[float] = None
    rollback_count: int = 0
    retry_count: int = 0


@dataclass
class ManagedImprovement:
    """An improvement under lifecycle management."""
    improvement: ActionPoint
    lifecycle_stage: ImprovementLifecycleStage
    created_at: datetime
    updated_at: datetime
    
    # Lifecycle tracking
    stage_history: List[Tuple[ImprovementLifecycleStage, datetime]] = field(default_factory=list)
    
    # Management metadata
    assigned_to: Optional[str] = None
    priority_score: float = 0.0
    impact_level: ImprovementImpactLevel = ImprovementImpactLevel.MEDIUM
    estimated_effort: Optional[int] = None  # In hours
    estimated_roi: Optional[float] = None
    
    # Dependencies
    dependencies: List[ImprovementDependency] = field(default_factory=list)
    blocked_by: Set[str] = field(default_factory=set)
    blocking: Set[str] = field(default_factory=set)
    
    # Metrics and tracking
    metrics: ImprovementMetrics = field(default_factory=ImprovementMetrics)
    implementation_attempts: int = 0
    last_error: Optional[str] = None
    
    # Validation and rollback
    validation_criteria: List[str] = field(default_factory=list)
    rollback_plan: Optional[Dict[str, Any]] = None
    rollback_triggers: List[str] = field(default_factory=list)


@dataclass
class CoordinatorConfig:
    """Configuration for the ImprovementCoordinator."""
    # Lifecycle management
    max_concurrent_implementations: int = 5
    max_implementation_time_hours: int = 24
    max_validation_time_hours: int = 6
    
    # Priority and scoring
    enable_priority_scoring: bool = True
    priority_refresh_interval_minutes: int = 60
    roi_calculation_enabled: bool = True
    
    # Dependency management
    enable_dependency_checking: bool = True
    auto_resolve_conflicts: bool = True
    max_dependency_depth: int = 5
    
    # Validation and rollback
    enable_automatic_validation: bool = True
    enable_automatic_rollback: bool = True
    rollback_threshold_error_rate: float = 0.3
    validation_timeout_minutes: int = 30
    
    # Metrics and tracking
    track_performance_impact: bool = True
    track_cost_impact: bool = True
    enable_success_rate_tracking: bool = True
    
    # Backlog management
    max_backlog_size: int = 100
    auto_archive_completed_days: int = 30
    priority_decay_rate: float = 0.95  # Daily decay for stale improvements


class ImprovementCoordinator:
    """
    Improvement Lifecycle Management and Coordination.
    
    Manages the complete lifecycle of improvements from identification through
    validation and completion. Provides comprehensive tracking, dependency
    management, and coordination with other system components.
    """
    
    def __init__(self, config: Optional[CoordinatorConfig] = None):
        """Initialize the improvement coordinator."""
        self.config = config or CoordinatorConfig()
        
        # Core data structures
        self.managed_improvements: Dict[str, ManagedImprovement] = {}
        self.improvement_backlog: List[str] = []  # Ordered by priority
        self.active_implementations: Set[str] = set()
        self.completed_improvements: Dict[str, ManagedImprovement] = {}
        self.failed_improvements: Dict[str, ManagedImprovement] = {}
        
        # Dependency management
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependency_graph: Dict[str, Set[str]] = {}
        
        # Metrics and tracking
        self.coordinator_metrics: Dict[str, float] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        # Initialize coordinator
        self._start_background_tasks()
        
        logger.info("ImprovementCoordinator initialized")

    def _start_background_tasks(self):
        """Start background monitoring and management tasks."""
        self._background_tasks = [
            asyncio.create_task(self._priority_refresh_loop()),
            asyncio.create_task(self._validation_monitor_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._dependency_resolver_loop()),
        ]

    async def add_improvement(
        self, 
        improvement: ActionPoint,
        priority_override: Optional[float] = None,
        assigned_to: Optional[str] = None
    ) -> str:
        """Add a new improvement to lifecycle management."""
        improvement_id = getattr(improvement, 'id', f"improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if improvement_id in self.managed_improvements:
            logger.warning(f"Improvement {improvement_id} already exists")
            return improvement_id
        
        # Create managed improvement
        managed = ManagedImprovement(
            improvement=improvement,
            lifecycle_stage=ImprovementLifecycleStage.IDENTIFIED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            assigned_to=assigned_to
        )
        
        # Set initial priority
        if priority_override is not None:
            managed.priority_score = priority_override
        else:
            managed.priority_score = await self._calculate_priority_score(improvement)
        
        # Set impact level
        managed.impact_level = await self._assess_impact_level(improvement)
        
        # Add to management
        self.managed_improvements[improvement_id] = managed
        self._update_stage(improvement_id, ImprovementLifecycleStage.IDENTIFIED)
        
        # Add to backlog
        self._add_to_backlog(improvement_id)
        
        logger.info(f"Added improvement {improvement_id} to lifecycle management")
        return improvement_id

    async def advance_improvement_lifecycle(self, improvement_id: str) -> bool:
        """Advance an improvement to the next lifecycle stage."""
        if improvement_id not in self.managed_improvements:
            logger.error(f"Improvement {improvement_id} not found")
            return False
        
        managed = self.managed_improvements[improvement_id]
        current_stage = managed.lifecycle_stage
        
        # Determine next stage based on current stage
        next_stage = self._get_next_lifecycle_stage(current_stage)
        if not next_stage:
            logger.info(f"Improvement {improvement_id} is at final stage {current_stage}")
            return True
        
        # Check if advancement is possible
        can_advance = await self._can_advance_to_stage(improvement_id, next_stage)
        if not can_advance:
            logger.warning(f"Cannot advance improvement {improvement_id} to {next_stage}")
            return False
        
        # Advance to next stage
        return await self._transition_to_stage(improvement_id, next_stage)

    async def implement_improvement(self, improvement_id: str) -> bool:
        """Implement a specific improvement."""
        if improvement_id not in self.managed_improvements:
            logger.error(f"Improvement {improvement_id} not found")
            return False
        
        managed = self.managed_improvements[improvement_id]
        
        # Check if ready for implementation
        if not await self._is_ready_for_implementation(improvement_id):
            logger.warning(f"Improvement {improvement_id} not ready for implementation")
            return False
        
        # Check concurrent implementation limit
        if len(self.active_implementations) >= self.config.max_concurrent_implementations:
            logger.info(f"Implementation limit reached, queueing improvement {improvement_id}")
            return False
        
        try:
            # Transition to implementing stage
            await self._transition_to_stage(improvement_id, ImprovementLifecycleStage.IMPLEMENTING)
            self.active_implementations.add(improvement_id)
            
            # Record implementation start
            managed.metrics.implementation_start = datetime.now()
            managed.implementation_attempts += 1
            
            # Execute implementation
            success = await self._execute_improvement_implementation(improvement_id)
            
            if success:
                # Record implementation end
                managed.metrics.implementation_end = datetime.now()
                
                # Transition to implemented stage
                await self._transition_to_stage(improvement_id, ImprovementLifecycleStage.IMPLEMENTED)
                
                # Start validation if enabled
                if self.config.enable_automatic_validation:
                    await self._start_validation(improvement_id)
                
                logger.info(f"Successfully implemented improvement {improvement_id}")
                return True
            else:
                # Handle implementation failure
                await self._handle_implementation_failure(improvement_id)
                return False
        
        except Exception as e:
            logger.error(f"Implementation failed for improvement {improvement_id}: {e}")
            managed.last_error = str(e)
            await self._handle_implementation_failure(improvement_id)
            return False
        
        finally:
            self.active_implementations.discard(improvement_id)

    async def validate_improvement(self, improvement_id: str) -> bool:
        """Validate an implemented improvement."""
        if improvement_id not in self.managed_improvements:
            logger.error(f"Improvement {improvement_id} not found")
            return False
        
        managed = self.managed_improvements[improvement_id]
        
        if managed.lifecycle_stage != ImprovementLifecycleStage.IMPLEMENTED:
            logger.error(f"Improvement {improvement_id} not in implemented stage")
            return False
        
        try:
            # Transition to validating stage
            await self._transition_to_stage(improvement_id, ImprovementLifecycleStage.VALIDATING)
            
            # Record validation start
            managed.metrics.validation_start = datetime.now()
            
            # Execute validation
            validation_results = await self._execute_improvement_validation(improvement_id)
            
            # Record validation end
            managed.metrics.validation_end = datetime.now()
            
            if validation_results['success']:
                # Update metrics
                await self._update_improvement_metrics(improvement_id, validation_results)
                
                # Transition to validated stage
                await self._transition_to_stage(improvement_id, ImprovementLifecycleStage.VALIDATED)
                
                # Mark as completed
                await self._transition_to_stage(improvement_id, ImprovementLifecycleStage.COMPLETED)
                
                logger.info(f"Successfully validated improvement {improvement_id}")
                return True
            else:
                # Validation failed - check if rollback is needed
                if self._should_trigger_rollback(improvement_id, validation_results):
                    await self.rollback_improvement(improvement_id, "Validation failed")
                else:
                    await self._transition_to_stage(improvement_id, ImprovementLifecycleStage.FAILED)
                
                return False
        
        except Exception as e:
            logger.error(f"Validation failed for improvement {improvement_id}: {e}")
            managed.last_error = str(e)
            await self._transition_to_stage(improvement_id, ImprovementLifecycleStage.FAILED)
            return False

    async def rollback_improvement(self, improvement_id: str, reason: str) -> bool:
        """Rollback an improvement implementation."""
        if improvement_id not in self.managed_improvements:
            logger.error(f"Improvement {improvement_id} not found")
            return False
        
        managed = self.managed_improvements[improvement_id]
        
        try:
            # Execute rollback
            rollback_success = await self._execute_improvement_rollback(improvement_id, reason)
            
            if rollback_success:
                # Update metrics
                managed.metrics.rollback_count += 1
                
                # Transition to rolled back stage
                await self._transition_to_stage(improvement_id, ImprovementLifecycleStage.ROLLED_BACK)
                
                # Remove from active implementations if present
                self.active_implementations.discard(improvement_id)
                
                logger.info(f"Successfully rolled back improvement {improvement_id}: {reason}")
                return True
            else:
                logger.error(f"Rollback failed for improvement {improvement_id}")
                return False
        
        except Exception as e:
            logger.error(f"Rollback error for improvement {improvement_id}: {e}")
            managed.last_error = str(e)
            return False

    async def get_improvement_status(self, improvement_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status of an improvement."""
        if improvement_id not in self.managed_improvements:
            return None
        
        managed = self.managed_improvements[improvement_id]
        
        return {
            'improvement_id': improvement_id,
            'lifecycle_stage': managed.lifecycle_stage.value,
            'priority_score': managed.priority_score,
            'impact_level': managed.impact_level.value,
            'assigned_to': managed.assigned_to,
            'created_at': managed.created_at.isoformat(),
            'updated_at': managed.updated_at.isoformat(),
            
            # Progress tracking
            'implementation_attempts': managed.implementation_attempts,
            'estimated_effort': managed.estimated_effort,
            'estimated_roi': managed.estimated_roi,
            
            # Dependencies
            'blocked_by_count': len(managed.blocked_by),
            'blocking_count': len(managed.blocking),
            
            # Metrics
            'metrics': {
                'implementation_duration': self._calculate_duration(
                    managed.metrics.implementation_start,
                    managed.metrics.implementation_end
                ),
                'validation_duration': self._calculate_duration(
                    managed.metrics.validation_start, 
                    managed.metrics.validation_end
                ),
                'success_rate': managed.metrics.success_rate,
                'rollback_count': managed.metrics.rollback_count,
                'performance_gain': managed.metrics.efficiency_gain,
                'cost_impact': managed.metrics.cost_impact
            },
            
            # Status
            'last_error': managed.last_error,
            'is_active_implementation': improvement_id in self.active_implementations
        }

    async def get_coordinator_status(self) -> Dict[str, Any]:
        """Get comprehensive coordinator status."""
        return {
            'managed_improvements': len(self.managed_improvements),
            'backlog_size': len(self.improvement_backlog),
            'active_implementations': len(self.active_implementations),
            'completed_improvements': len(self.completed_improvements),
            'failed_improvements': len(self.failed_improvements),
            
            # Stage distribution
            'stage_distribution': self._get_stage_distribution(),
            
            # Priority distribution
            'priority_distribution': self._get_priority_distribution(),
            
            # Impact distribution
            'impact_distribution': self._get_impact_distribution(),
            
            # Performance metrics
            'coordinator_metrics': self.coordinator_metrics.copy(),
            
            # Configuration
            'config': {
                'max_concurrent_implementations': self.config.max_concurrent_implementations,
                'automatic_validation': self.config.enable_automatic_validation,
                'automatic_rollback': self.config.enable_automatic_rollback,
                'dependency_checking': self.config.enable_dependency_checking
            }
        }

    def get_improvement_backlog(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the prioritized improvement backlog."""
        backlog_items = []
        
        items_to_process = self.improvement_backlog[:limit] if limit else self.improvement_backlog
        
        for improvement_id in items_to_process:
            if improvement_id in self.managed_improvements:
                managed = self.managed_improvements[improvement_id]
                backlog_items.append({
                    'improvement_id': improvement_id,
                    'title': managed.improvement.title,
                    'priority_score': managed.priority_score,
                    'impact_level': managed.impact_level.value,
                    'lifecycle_stage': managed.lifecycle_stage.value,
                    'estimated_effort': managed.estimated_effort,
                    'estimated_roi': managed.estimated_roi,
                    'created_at': managed.created_at.isoformat(),
                    'blocked_by_count': len(managed.blocked_by)
                })
        
        return backlog_items

    # Internal Implementation Methods

    def _update_stage(self, improvement_id: str, new_stage: ImprovementLifecycleStage):
        """Update the lifecycle stage of an improvement."""
        if improvement_id not in self.managed_improvements:
            return
        
        managed = self.managed_improvements[improvement_id]
        old_stage = managed.lifecycle_stage
        
        managed.lifecycle_stage = new_stage
        managed.updated_at = datetime.now()
        managed.stage_history.append((new_stage, datetime.now()))
        
        logger.debug(f"Improvement {improvement_id} transitioned from {old_stage} to {new_stage}")

    def _get_next_lifecycle_stage(self, current_stage: ImprovementLifecycleStage) -> Optional[ImprovementLifecycleStage]:
        """Get the next lifecycle stage for a given current stage."""
        stage_transitions = {
            ImprovementLifecycleStage.IDENTIFIED: ImprovementLifecycleStage.ANALYZED,
            ImprovementLifecycleStage.ANALYZED: ImprovementLifecycleStage.PRIORITIZED,
            ImprovementLifecycleStage.PRIORITIZED: ImprovementLifecycleStage.APPROVED,
            ImprovementLifecycleStage.APPROVED: ImprovementLifecycleStage.PLANNED,
            ImprovementLifecycleStage.PLANNED: ImprovementLifecycleStage.IMPLEMENTING,
            ImprovementLifecycleStage.IMPLEMENTING: ImprovementLifecycleStage.IMPLEMENTED,
            ImprovementLifecycleStage.IMPLEMENTED: ImprovementLifecycleStage.VALIDATING,
            ImprovementLifecycleStage.VALIDATING: ImprovementLifecycleStage.VALIDATED,
            ImprovementLifecycleStage.VALIDATED: ImprovementLifecycleStage.COMPLETED,
        }
        
        return stage_transitions.get(current_stage)

    async def _can_advance_to_stage(self, improvement_id: str, target_stage: ImprovementLifecycleStage) -> bool:
        """Check if an improvement can advance to a target stage."""
        if improvement_id not in self.managed_improvements:
            return False
        
        managed = self.managed_improvements[improvement_id]
        
        # Check dependencies if dependency checking is enabled
        if self.config.enable_dependency_checking and managed.blocked_by:
            # Check if blocking improvements are resolved
            for blocking_id in managed.blocked_by:
                if blocking_id in self.managed_improvements:
                    blocking_managed = self.managed_improvements[blocking_id]
                    if blocking_managed.lifecycle_stage not in [
                        ImprovementLifecycleStage.COMPLETED,
                        ImprovementLifecycleStage.CANCELLED
                    ]:
                        return False
        
        # Stage-specific checks
        if target_stage == ImprovementLifecycleStage.IMPLEMENTING:
            # Check concurrent implementation limit
            if len(self.active_implementations) >= self.config.max_concurrent_implementations:
                return False
        
        return True

    async def _transition_to_stage(self, improvement_id: str, target_stage: ImprovementLifecycleStage) -> bool:
        """Transition an improvement to a target stage."""
        if not await self._can_advance_to_stage(improvement_id, target_stage):
            return False
        
        self._update_stage(improvement_id, target_stage)
        
        # Handle stage-specific actions
        if target_stage == ImprovementLifecycleStage.COMPLETED:
            await self._handle_improvement_completion(improvement_id)
        elif target_stage == ImprovementLifecycleStage.FAILED:
            await self._handle_improvement_failure(improvement_id)
        
        return True

    async def _calculate_priority_score(self, improvement: ActionPoint) -> float:
        """Calculate priority score for an improvement."""
        if not self.config.enable_priority_scoring:
            return 0.5  # Default medium priority
        
        # Base score from improvement priority
        priority_mapping = {
            PriorityLevel.LOW: 0.25,
            PriorityLevel.MEDIUM: 0.5, 
            PriorityLevel.HIGH: 0.75,
            PriorityLevel.CRITICAL: 1.0
        }
        
        base_score = priority_mapping.get(improvement.priority, 0.5)
        
        # Adjust based on category
        category_multipliers = {
            ImprovementCategory.PERFORMANCE: 0.9,
            ImprovementCategory.RELIABILITY: 0.95,
            ImprovementCategory.SCALABILITY: 0.85,
            ImprovementCategory.MAINTAINABILITY: 0.7,
            ImprovementCategory.USER_EXPERIENCE: 0.8,
        }
        
        category_multiplier = category_multipliers.get(improvement.category, 0.75)
        
        return min(base_score * category_multiplier, 1.0)

    async def _assess_impact_level(self, improvement: ActionPoint) -> ImprovementImpactLevel:
        """Assess the impact level of an improvement."""
        # Simple assessment based on priority and category
        if improvement.priority == PriorityLevel.CRITICAL:
            return ImprovementImpactLevel.CRITICAL
        elif improvement.priority == PriorityLevel.HIGH:
            return ImprovementImpactLevel.HIGH
        elif improvement.priority == PriorityLevel.MEDIUM:
            return ImprovementImpactLevel.MEDIUM
        else:
            return ImprovementImpactLevel.LOW

    def _add_to_backlog(self, improvement_id: str):
        """Add an improvement to the prioritized backlog."""
        if improvement_id in self.improvement_backlog:
            return
        
        managed = self.managed_improvements[improvement_id]
        
        # Insert in priority order
        inserted = False
        for i, existing_id in enumerate(self.improvement_backlog):
            if existing_id in self.managed_improvements:
                existing_managed = self.managed_improvements[existing_id]
                if managed.priority_score > existing_managed.priority_score:
                    self.improvement_backlog.insert(i, improvement_id)
                    inserted = True
                    break
        
        if not inserted:
            self.improvement_backlog.append(improvement_id)
        
        # Trim backlog if too large
        if len(self.improvement_backlog) > self.config.max_backlog_size:
            self.improvement_backlog = self.improvement_backlog[:self.config.max_backlog_size]

    async def _is_ready_for_implementation(self, improvement_id: str) -> bool:
        """Check if an improvement is ready for implementation."""
        if improvement_id not in self.managed_improvements:
            return False
        
        managed = self.managed_improvements[improvement_id]
        
        # Must be in approved or planned stage
        if managed.lifecycle_stage not in [
            ImprovementLifecycleStage.APPROVED,
            ImprovementLifecycleStage.PLANNED
        ]:
            return False
        
        # Check dependencies
        if managed.blocked_by:
            return False
        
        return True

    async def _execute_improvement_implementation(self, improvement_id: str) -> bool:
        """Execute the implementation of an improvement with git-aware verification."""
        logger.info(f"Executing implementation for improvement {improvement_id}")
        
        # Get improvement details
        if improvement_id not in self.managed_improvements:
            return False
        
        managed = self.managed_improvements[improvement_id]
        improvement = managed.improvement
        
        # Pre-implementation verification: capture current state
        from ..verification import GitAwareVerifier
        verifier = GitAwareVerifier()
        pre_implementation_status = verifier.get_git_status_summary()
        
        try:
            # Execute the actual improvement (this would be replaced with real implementation)
            # For documentation improvements, this might involve file operations
            success = await self._perform_actual_implementation(improvement)
            
            if not success:
                logger.warning(f"Implementation of {improvement_id} failed at execution stage")
                return False
            
            # Post-implementation verification: verify claimed changes actually occurred
            post_implementation_status = verifier.get_git_status_summary()
            
            # Verify any file operations that were claimed
            claimed_files = getattr(improvement, 'files_modified', [])
            if claimed_files:
                verification_result = verifier.verify_documentation_updates_complete(claimed_files)
                
                if not verification_result.success:
                    logger.error(f"Verification failed for improvement {improvement_id}")
                    logger.error(f"False claims: {verification_result.false_claims}")
                    logger.error(f"Missing operations: {verification_result.missing_operations}")
                    
                    # Save verification report for debugging
                    report_path = verifier.save_verification_report(f"failed_verification_{improvement_id}.json")
                    logger.error(f"Detailed verification report saved to: {report_path}")
                    
                    return False
                
                logger.info(f"Verification passed for improvement {improvement_id}")
                logger.info(f"Successfully verified files: {claimed_files}")
            
            # Check for commits if this was supposed to create commits
            if hasattr(improvement, 'should_commit') and improvement.should_commit:
                # Verify that new commits exist
                last_commit_before = pre_implementation_status.get('last_commit', {})
                last_commit_after = post_implementation_status.get('last_commit', {})
                
                if last_commit_before.get('hash') == last_commit_after.get('hash'):
                    logger.error(f"No new commit created for improvement {improvement_id} despite claiming to commit")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during implementation of {improvement_id}: {e}")
            return False
    
    async def _perform_actual_implementation(self, improvement) -> bool:
        """Perform the actual implementation based on improvement type."""
        # This is where the real implementation logic would go
        # For now, simulate with delay and mostly successful outcomes
        await asyncio.sleep(1)
        
        # Simulate different success rates based on improvement category
        import random
        success_rate = 0.9  # Default 90% success rate
        
        if hasattr(improvement, 'category'):
            category = getattr(improvement, 'category', 'general')
            if 'documentation' in category.lower():
                success_rate = 0.95  # Higher success for docs
            elif 'performance' in category.lower():
                success_rate = 0.8   # Lower success for complex performance changes
        
        return random.random() < success_rate

    async def _execute_improvement_validation(self, improvement_id: str) -> Dict[str, Any]:
        """Execute validation for an implemented improvement."""
        # This is a placeholder - actual validation would depend on the improvement type
        logger.info(f"Executing validation for improvement {improvement_id}")
        
        # Simulate validation delay
        await asyncio.sleep(0.5)
        
        # For now, return success (95% success rate simulation)
        import random
        success = random.random() > 0.05
        
        return {
            'success': success,
            'validation_results': {
                'performance_improvement': random.uniform(0.05, 0.30) if success else 0,
                'error_rate_reduction': random.uniform(0.01, 0.15) if success else 0,
                'user_satisfaction_gain': random.uniform(0.02, 0.20) if success else 0
            },
            'validation_timestamp': datetime.now().isoformat()
        }

    async def _execute_improvement_rollback(self, improvement_id: str, reason: str) -> bool:
        """Execute rollback of an improvement."""
        # This is a placeholder - actual rollback would depend on the improvement type
        logger.info(f"Executing rollback for improvement {improvement_id}: {reason}")
        
        # Simulate rollback delay
        await asyncio.sleep(0.5)
        
        # For now, return success (98% success rate simulation)
        import random
        return random.random() > 0.02

    async def _update_improvement_metrics(self, improvement_id: str, validation_results: Dict[str, Any]):
        """Update metrics for an improvement based on validation results."""
        if improvement_id not in self.managed_improvements:
            return
        
        managed = self.managed_improvements[improvement_id]
        
        if validation_results['success']:
            results = validation_results['validation_results']
            managed.metrics.efficiency_gain = results.get('performance_improvement')
            managed.metrics.success_rate = 1.0
            managed.metrics.user_satisfaction_score = results.get('user_satisfaction_gain')

    def _should_trigger_rollback(self, improvement_id: str, validation_results: Dict[str, Any]) -> bool:
        """Determine if an improvement should be rolled back based on validation results."""
        if not self.config.enable_automatic_rollback:
            return False
        
        if improvement_id not in self.managed_improvements:
            return False
        
        managed = self.managed_improvements[improvement_id]
        
        # Check rollback triggers
        if managed.rollback_triggers:
            for trigger in managed.rollback_triggers:
                if trigger in validation_results.get('errors', []):
                    return True
        
        # Check error rate threshold
        error_rate = validation_results.get('error_rate', 0.0)
        if error_rate > self.config.rollback_threshold_error_rate:
            return True
        
        return False

    async def _handle_improvement_completion(self, improvement_id: str):
        """Handle completion of an improvement."""
        if improvement_id not in self.managed_improvements:
            return
        
        managed = self.managed_improvements[improvement_id]
        
        # Move to completed improvements
        self.completed_improvements[improvement_id] = managed
        
        # Remove from active management
        del self.managed_improvements[improvement_id]
        
        # Remove from backlog
        if improvement_id in self.improvement_backlog:
            self.improvement_backlog.remove(improvement_id)
        
        # Update dependencies - unblock dependent improvements
        if improvement_id in self.reverse_dependency_graph:
            for dependent_id in self.reverse_dependency_graph[improvement_id]:
                if dependent_id in self.managed_improvements:
                    self.managed_improvements[dependent_id].blocked_by.discard(improvement_id)

    async def _handle_improvement_failure(self, improvement_id: str):
        """Handle failure of an improvement."""
        if improvement_id not in self.managed_improvements:
            return
        
        managed = self.managed_improvements[improvement_id]
        
        # Move to failed improvements
        self.failed_improvements[improvement_id] = managed
        
        # Remove from active implementations
        self.active_implementations.discard(improvement_id)

    def _calculate_duration(self, start_time: Optional[datetime], end_time: Optional[datetime]) -> Optional[float]:
        """Calculate duration in seconds between two timestamps."""
        if not start_time or not end_time:
            return None
        return (end_time - start_time).total_seconds()

    def _get_stage_distribution(self) -> Dict[str, int]:
        """Get distribution of improvements across lifecycle stages."""
        distribution = {}
        for managed in self.managed_improvements.values():
            stage = managed.lifecycle_stage.value
            distribution[stage] = distribution.get(stage, 0) + 1
        return distribution

    def _get_priority_distribution(self) -> Dict[str, int]:
        """Get distribution of improvements across priority levels."""
        distribution = {}
        for managed in self.managed_improvements.values():
            # Convert priority score to range
            if managed.priority_score >= 0.75:
                priority = "high"
            elif managed.priority_score >= 0.5:
                priority = "medium"
            else:
                priority = "low"
            distribution[priority] = distribution.get(priority, 0) + 1
        return distribution

    def _get_impact_distribution(self) -> Dict[str, int]:
        """Get distribution of improvements across impact levels."""
        distribution = {}
        for managed in self.managed_improvements.values():
            impact = managed.impact_level.value
            distribution[impact] = distribution.get(impact, 0) + 1
        return distribution

    # Background Task Loops

    async def _priority_refresh_loop(self):
        """Background loop to refresh improvement priorities."""
        while True:
            try:
                await asyncio.sleep(self.config.priority_refresh_interval_minutes * 60)
                await self._refresh_priorities()
            except Exception as e:
                logger.error(f"Priority refresh loop error: {e}")

    async def _validation_monitor_loop(self):
        """Background loop to monitor validation timeouts."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._check_validation_timeouts()
            except Exception as e:
                logger.error(f"Validation monitor loop error: {e}")

    async def _metrics_collection_loop(self):
        """Background loop to collect and update metrics."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._collect_coordinator_metrics()
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")

    async def _dependency_resolver_loop(self):
        """Background loop to resolve improvement dependencies."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._resolve_dependencies()
            except Exception as e:
                logger.error(f"Dependency resolver loop error: {e}")

    async def _refresh_priorities(self):
        """Refresh priorities for all managed improvements."""
        for improvement_id, managed in self.managed_improvements.items():
            new_priority = await self._calculate_priority_score(managed.improvement)
            
            # Apply decay for stale improvements
            days_old = (datetime.now() - managed.created_at).days
            decay_factor = self.config.priority_decay_rate ** days_old
            
            managed.priority_score = new_priority * decay_factor
        
        # Re-sort backlog
        self._resort_backlog()

    def _resort_backlog(self):
        """Re-sort the improvement backlog by priority."""
        self.improvement_backlog.sort(
            key=lambda imp_id: self.managed_improvements.get(imp_id, ManagedImprovement(
                improvement=None, lifecycle_stage=ImprovementLifecycleStage.IDENTIFIED,
                created_at=datetime.now(), updated_at=datetime.now()
            )).priority_score,
            reverse=True
        )

    async def _check_validation_timeouts(self):
        """Check for validation timeouts and handle them."""
        timeout_threshold = datetime.now() - timedelta(minutes=self.config.validation_timeout_minutes)
        
        for improvement_id, managed in self.managed_improvements.items():
            if (managed.lifecycle_stage == ImprovementLifecycleStage.VALIDATING and
                managed.metrics.validation_start and
                managed.metrics.validation_start < timeout_threshold):
                
                logger.warning(f"Validation timeout for improvement {improvement_id}")
                await self._handle_validation_timeout(improvement_id)

    async def _handle_validation_timeout(self, improvement_id: str):
        """Handle a validation timeout."""
        if self.config.enable_automatic_rollback:
            await self.rollback_improvement(improvement_id, "Validation timeout")
        else:
            await self._transition_to_stage(improvement_id, ImprovementLifecycleStage.FAILED)

    async def _collect_coordinator_metrics(self):
        """Collect coordinator-level metrics."""
        total_managed = len(self.managed_improvements)
        total_completed = len(self.completed_improvements)
        total_failed = len(self.failed_improvements)
        
        self.coordinator_metrics.update({
            'total_improvements_managed': total_managed + total_completed + total_failed,
            'active_improvements': total_managed,
            'completed_improvements': total_completed,
            'failed_improvements': total_failed,
            'success_rate': total_completed / max(total_completed + total_failed, 1),
            'active_implementations': len(self.active_implementations),
            'backlog_size': len(self.improvement_backlog)
        })

    async def _resolve_dependencies(self):
        """Resolve and update improvement dependencies."""
        if not self.config.enable_dependency_checking:
            return
        
        # Check for newly unblocked improvements
        for improvement_id, managed in self.managed_improvements.items():
            if managed.blocked_by:
                # Check if blocking improvements are resolved
                resolved_blockers = set()
                for blocking_id in managed.blocked_by:
                    if blocking_id in self.completed_improvements:
                        resolved_blockers.add(blocking_id)
                
                # Remove resolved blockers
                managed.blocked_by -= resolved_blockers
                
                if not managed.blocked_by:
                    logger.info(f"Improvement {improvement_id} is no longer blocked")

    def cleanup(self):
        """Cleanup coordinator resources."""
        for task in self._background_tasks:
            task.cancel()
        
        logger.info("ImprovementCoordinator cleanup completed")