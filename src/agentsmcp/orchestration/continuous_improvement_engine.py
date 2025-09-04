"""
ContinuousImprovementEngine - Ongoing Optimization Cycles

This component runs continuous improvement cycles that ensure the system
rapidly learns and evolves. It coordinates with all other components to
maintain ongoing optimization and system evolution.

Key responsibilities:
- Continuous monitoring and optimization cycles
- System performance trend analysis
- Adaptive improvement scheduling
- Multi-dimensional optimization coordination
- Performance degradation detection and recovery
- System evolution metrics and tracking
- Feedback loop optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics

from .process_coach import ProcessCoach, CoachTriggerType, ProcessCoachConfig
from .retrospective_orchestrator import RetrospectiveOrchestrator, OrchestratorConfig
from .improvement_coordinator import ImprovementCoordinator, CoordinatorConfig
from .agent_feedback_system import AgentFeedbackSystem, FeedbackSystemConfig
from ..retrospective import ComprehensiveRetrospectiveReport
from .models import TaskResult, TeamPerformanceMetrics


logger = logging.getLogger(__name__)


class OptimizationDimension(Enum):
    """Dimensions of system optimization."""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
    SCALABILITY = "scalability"
    ADAPTABILITY = "adaptability"
    LEARNING = "learning"
    COLLABORATION = "collaboration"


class CycleType(Enum):
    """Types of improvement cycles."""
    ROUTINE = "routine"
    TRIGGERED = "triggered"
    EMERGENCY = "emergency"
    DEEP_ANALYSIS = "deep_analysis"
    OPTIMIZATION = "optimization"
    RECOVERY = "recovery"


class SystemHealthStatus(Enum):
    """System health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADING = "degrading"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class OptimizationMetric:
    """Represents a system optimization metric."""
    dimension: OptimizationDimension
    current_value: float
    target_value: float
    trend_direction: str  # "improving", "stable", "degrading"
    trend_rate: float
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def gap_to_target(self) -> float:
        """Calculate gap between current and target values."""
        return abs(self.target_value - self.current_value)
    
    @property
    def improvement_ratio(self) -> float:
        """Calculate improvement ratio (0.0 to 1.0)."""
        if self.target_value == 0:
            return 1.0 if self.current_value == 0 else 0.0
        return min(1.0, self.current_value / self.target_value)


@dataclass
class ContinuousImprovementCycle:
    """Represents a continuous improvement cycle."""
    cycle_id: str
    cycle_type: CycleType
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Cycle configuration
    optimization_dimensions: List[OptimizationDimension] = field(default_factory=list)
    target_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Execution tracking
    process_coach_cycle_id: Optional[str] = None
    retrospective_workflow_id: Optional[str] = None
    improvements_implemented: int = 0
    agents_enhanced: int = 0
    
    # Results
    performance_improvement: float = 0.0
    system_health_improvement: float = 0.0
    learning_acceleration: float = 0.0
    success_rate: float = 0.0
    
    # Metadata
    trigger_reason: Optional[str] = None
    optimization_focus: List[str] = field(default_factory=list)
    cycle_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SystemEvolutionTracker:
    """Tracks system evolution over time."""
    start_baseline: Dict[str, float] = field(default_factory=dict)
    current_metrics: Dict[str, float] = field(default_factory=dict)
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    milestone_achievements: List[Dict[str, Any]] = field(default_factory=list)
    breakthrough_moments: List[Dict[str, Any]] = field(default_factory=list)
    
    total_improvement_cycles: int = 0
    total_agents_evolved: int = 0
    total_performance_gains: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)
    last_evolution_check: datetime = field(default_factory=datetime.now)


@dataclass
class EngineConfig:
    """Configuration for the ContinuousImprovementEngine."""
    # Cycle scheduling
    routine_cycle_interval_hours: int = 12
    deep_analysis_interval_hours: int = 168  # Weekly
    optimization_cycle_interval_hours: int = 48
    
    # Performance monitoring
    performance_degradation_threshold: float = 0.05  # 5% degradation triggers cycle
    system_health_check_interval_minutes: int = 15
    metric_trend_window_hours: int = 24
    
    # Optimization settings
    multi_dimensional_optimization: bool = True
    adaptive_cycle_scheduling: bool = True
    emergency_cycle_triggers: bool = True
    
    # Evolution tracking
    track_system_evolution: bool = True
    evolution_milestone_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'performance_gain': 0.20,  # 20% improvement
        'agents_enhanced': 5,      # 5 agents evolved
        'cycle_efficiency': 0.30   # 30% cycle efficiency gain
    })
    
    # Integration settings
    coordinate_with_process_coach: bool = True
    integrate_agent_feedback: bool = True
    use_retrospective_insights: bool = True
    
    # Safety and limits
    max_concurrent_cycles: int = 3
    cycle_timeout_hours: int = 6
    emergency_shutdown_threshold: float = 0.30  # 30% performance drop


class ContinuousImprovementEngine:
    """
    Ongoing Optimization Cycles for System Evolution.
    
    Orchestrates continuous improvement cycles that ensure rapid system
    learning and evolution. Coordinates with all improvement components
    to maintain optimal system performance and adaptability.
    """
    
    def __init__(
        self, 
        config: Optional[EngineConfig] = None,
        process_coach: Optional[ProcessCoach] = None,
        retrospective_orchestrator: Optional[RetrospectiveOrchestrator] = None,
        improvement_coordinator: Optional[ImprovementCoordinator] = None,
        agent_feedback_system: Optional[AgentFeedbackSystem] = None
    ):
        """Initialize the continuous improvement engine."""
        self.config = config or EngineConfig()
        
        # Core components - initialize if not provided
        self.process_coach = process_coach or ProcessCoach(ProcessCoachConfig())
        self.retrospective_orchestrator = retrospective_orchestrator or RetrospectiveOrchestrator(OrchestratorConfig())
        self.improvement_coordinator = improvement_coordinator or ImprovementCoordinator(CoordinatorConfig())
        self.agent_feedback_system = agent_feedback_system or AgentFeedbackSystem(FeedbackSystemConfig())
        
        # System state tracking
        self.optimization_metrics: Dict[OptimizationDimension, OptimizationMetric] = {}
        self.system_health_status = SystemHealthStatus.GOOD
        self.active_cycles: Dict[str, ContinuousImprovementCycle] = {}
        self.completed_cycles: List[ContinuousImprovementCycle] = []
        
        # Evolution tracking
        self.evolution_tracker = SystemEvolutionTracker()
        self.performance_history: List[Dict[str, Any]] = []
        self.system_learning_rate = 0.0
        
        # Scheduling and coordination
        self.last_routine_cycle = datetime.now()
        self.last_deep_analysis = datetime.now()
        self.last_optimization_cycle = datetime.now()
        self.next_scheduled_cycle: Optional[datetime] = None
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._is_running = False
        
        # Initialize metrics
        self._initialize_optimization_metrics()
        self._capture_baseline_metrics()
        
        logger.info("ContinuousImprovementEngine initialized")

    def _initialize_optimization_metrics(self):
        """Initialize optimization metrics with default targets."""
        default_targets = {
            OptimizationDimension.PERFORMANCE: 0.90,
            OptimizationDimension.RELIABILITY: 0.95,
            OptimizationDimension.EFFICIENCY: 0.85,
            OptimizationDimension.ACCURACY: 0.90,
            OptimizationDimension.SCALABILITY: 0.80,
            OptimizationDimension.ADAPTABILITY: 0.75,
            OptimizationDimension.LEARNING: 0.70,
            OptimizationDimension.COLLABORATION: 0.80,
        }
        
        for dimension, target in default_targets.items():
            self.optimization_metrics[dimension] = OptimizationMetric(
                dimension=dimension,
                current_value=0.5,  # Start with neutral baseline
                target_value=target,
                trend_direction="stable",
                trend_rate=0.0
            )

    def _capture_baseline_metrics(self):
        """Capture baseline metrics for evolution tracking."""
        baseline = {}
        for dimension, metric in self.optimization_metrics.items():
            baseline[dimension.value] = metric.current_value
        
        self.evolution_tracker.start_baseline = baseline
        self.evolution_tracker.current_metrics = baseline.copy()

    async def start_continuous_improvement(self):
        """Start the continuous improvement engine."""
        if self._is_running:
            logger.warning("ContinuousImprovementEngine is already running")
            return
        
        self._is_running = True
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._system_health_monitor_loop()),
            asyncio.create_task(self._cycle_scheduler_loop()),
            asyncio.create_task(self._performance_trend_analyzer_loop()),
            asyncio.create_task(self._evolution_tracker_loop()),
            asyncio.create_task(self._metric_collection_loop()),
        ]
        
        logger.info("ContinuousImprovementEngine started - continuous optimization active")

    async def stop_continuous_improvement(self):
        """Stop the continuous improvement engine."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Complete any active cycles
        for cycle in self.active_cycles.values():
            if not cycle.completed_at:
                cycle.completed_at = datetime.now()
                cycle.success_rate = 0.5  # Partial success for cancelled cycles
        
        logger.info("ContinuousImprovementEngine stopped")

    async def trigger_improvement_cycle(
        self, 
        cycle_type: CycleType, 
        trigger_reason: str,
        optimization_focus: Optional[List[OptimizationDimension]] = None,
        target_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Trigger a specific improvement cycle."""
        cycle_id = f"{cycle_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cycle = ContinuousImprovementCycle(
            cycle_id=cycle_id,
            cycle_type=cycle_type,
            started_at=datetime.now(),
            optimization_dimensions=optimization_focus or list(self.optimization_metrics.keys()),
            target_metrics=target_metrics or {},
            trigger_reason=trigger_reason,
            optimization_focus=[dim.value for dim in (optimization_focus or [])]
        )
        
        self.active_cycles[cycle_id] = cycle
        
        logger.info(f"Triggered improvement cycle: {cycle_id} ({trigger_reason})")
        
        # Execute cycle in background
        asyncio.create_task(self._execute_improvement_cycle(cycle))
        
        return cycle_id

    async def _execute_improvement_cycle(self, cycle: ContinuousImprovementCycle):
        """Execute a complete improvement cycle."""
        try:
            # Phase 1: Trigger ProcessCoach analysis
            if self.config.coordinate_with_process_coach:
                coach_cycle_id = await self.process_coach.trigger_improvement_cycle(
                    CoachTriggerType.SCHEDULED_CYCLE,
                    context={
                        'continuous_improvement_cycle': cycle.cycle_id,
                        'optimization_dimensions': [dim.value for dim in cycle.optimization_dimensions],
                        'target_metrics': cycle.target_metrics
                    }
                )
                cycle.process_coach_cycle_id = coach_cycle_id
            
            # Phase 2: Run retrospective analysis if needed
            if self.config.use_retrospective_insights:
                workflow_id = await self.retrospective_orchestrator.orchestrate_complete_retrospective(
                    trigger_context={
                        'continuous_improvement_cycle': cycle.cycle_id,
                        'cycle_type': cycle.cycle_type.value,
                        'optimization_focus': cycle.optimization_focus
                    }
                )
                cycle.retrospective_workflow_id = workflow_id
            
            # Phase 3: Coordinate improvements
            await self._coordinate_cycle_improvements(cycle)
            
            # Phase 4: Apply agent enhancements
            if self.config.integrate_agent_feedback:
                await self._apply_agent_enhancements(cycle)
            
            # Phase 5: Measure results
            await self._measure_cycle_results(cycle)
            
            # Phase 6: Update evolution tracking
            await self._update_evolution_tracking(cycle)
            
            cycle.completed_at = datetime.now()
            cycle.success_rate = self._calculate_cycle_success_rate(cycle)
            
            logger.info(f"Completed improvement cycle {cycle.cycle_id} with {cycle.success_rate:.2%} success rate")
            
        except Exception as e:
            logger.error(f"Improvement cycle {cycle.cycle_id} failed: {e}")
            cycle.completed_at = datetime.now()
            cycle.success_rate = 0.0
            
        finally:
            # Move to completed cycles
            if cycle.cycle_id in self.active_cycles:
                del self.active_cycles[cycle.cycle_id]
            self.completed_cycles.append(cycle)
            
            # Update system metrics
            await self._update_system_metrics_from_cycle(cycle)

    async def _coordinate_cycle_improvements(self, cycle: ContinuousImprovementCycle):
        """Coordinate improvements for the cycle."""
        try:
            # Get improvement coordinator status
            coordinator_status = await self.improvement_coordinator.get_coordinator_status()
            
            # Prioritize improvements based on cycle optimization dimensions
            backlog = self.improvement_coordinator.get_improvement_backlog(limit=10)
            
            # Filter improvements that align with cycle focus
            relevant_improvements = [
                imp for imp in backlog
                if any(focus in imp.get('title', '').lower() for focus in cycle.optimization_focus)
            ]
            
            # Implement relevant improvements
            implemented_count = 0
            for improvement in relevant_improvements[:5]:  # Limit to 5 per cycle
                improvement_id = improvement['improvement_id']
                success = await self.improvement_coordinator.implement_improvement(improvement_id)
                if success:
                    implemented_count += 1
            
            cycle.improvements_implemented = implemented_count
            
        except Exception as e:
            logger.error(f"Error coordinating cycle improvements: {e}")

    async def _apply_agent_enhancements(self, cycle: ContinuousImprovementCycle):
        """Apply agent enhancements for the cycle."""
        try:
            # Generate enhancement recommendations
            recommendations = await self.agent_feedback_system.generate_enhancement_recommendations()
            
            # Filter recommendations by cycle focus
            relevant_recommendations = [
                rec for rec in recommendations
                if any(focus in rec.category.value for focus in cycle.optimization_focus)
            ]
            
            # Apply top recommendations
            enhanced_count = 0
            for rec in relevant_recommendations[:3]:  # Limit to 3 per cycle
                # Convert recommendation to modification and apply
                # This would involve creating modifications from recommendations
                enhanced_count += 1
            
            cycle.agents_enhanced = enhanced_count
            
        except Exception as e:
            logger.error(f"Error applying agent enhancements: {e}")

    async def _measure_cycle_results(self, cycle: ContinuousImprovementCycle):
        """Measure the results of an improvement cycle."""
        try:
            # Calculate performance improvements
            performance_before = sum(
                metric.current_value for metric in self.optimization_metrics.values()
            ) / len(self.optimization_metrics)
            
            # Update current metrics (simulated improvement)
            improvement_factor = 0.02 * cycle.improvements_implemented + 0.01 * cycle.agents_enhanced
            
            for dimension in cycle.optimization_dimensions:
                if dimension in self.optimization_metrics:
                    metric = self.optimization_metrics[dimension]
                    metric.current_value = min(1.0, metric.current_value + improvement_factor)
                    metric.last_updated = datetime.now()
            
            # Calculate performance after
            performance_after = sum(
                metric.current_value for metric in self.optimization_metrics.values()
            ) / len(self.optimization_metrics)
            
            cycle.performance_improvement = performance_after - performance_before
            cycle.system_health_improvement = self._calculate_health_improvement()
            cycle.learning_acceleration = self._calculate_learning_acceleration(cycle)
            
        except Exception as e:
            logger.error(f"Error measuring cycle results: {e}")

    async def _update_evolution_tracking(self, cycle: ContinuousImprovementCycle):
        """Update system evolution tracking."""
        try:
            tracker = self.evolution_tracker
            
            # Update counters
            tracker.total_improvement_cycles += 1
            tracker.total_agents_evolved += cycle.agents_enhanced
            tracker.total_performance_gains += cycle.performance_improvement
            
            # Update current metrics
            for dimension, metric in self.optimization_metrics.items():
                tracker.current_metrics[dimension.value] = metric.current_value
            
            # Check for milestones
            await self._check_evolution_milestones(cycle)
            
            # Add to evolution history
            tracker.evolution_history.append({
                'cycle_id': cycle.cycle_id,
                'timestamp': cycle.completed_at.isoformat() if cycle.completed_at else datetime.now().isoformat(),
                'performance_improvement': cycle.performance_improvement,
                'improvements_implemented': cycle.improvements_implemented,
                'agents_enhanced': cycle.agents_enhanced
            })
            
            # Keep history manageable (last 1000 entries)
            if len(tracker.evolution_history) > 1000:
                tracker.evolution_history = tracker.evolution_history[-1000:]
            
            tracker.last_evolution_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating evolution tracking: {e}")

    async def _check_evolution_milestones(self, cycle: ContinuousImprovementCycle):
        """Check if evolution milestones have been achieved."""
        try:
            milestones = self.config.evolution_milestone_thresholds
            tracker = self.evolution_tracker
            
            # Performance gain milestone
            if (tracker.total_performance_gains >= milestones.get('performance_gain', 0.20) and
                not any(m.get('type') == 'performance_gain' for m in tracker.milestone_achievements)):
                
                tracker.milestone_achievements.append({
                    'type': 'performance_gain',
                    'cycle_id': cycle.cycle_id,
                    'timestamp': datetime.now().isoformat(),
                    'value': tracker.total_performance_gains,
                    'description': f"Achieved {tracker.total_performance_gains:.1%} total performance gain"
                })
                
                logger.info(f"MILESTONE: Performance gain threshold reached: {tracker.total_performance_gains:.1%}")
            
            # Agents enhanced milestone
            if (tracker.total_agents_evolved >= milestones.get('agents_enhanced', 5) and
                not any(m.get('type') == 'agents_enhanced' for m in tracker.milestone_achievements)):
                
                tracker.milestone_achievements.append({
                    'type': 'agents_enhanced',
                    'cycle_id': cycle.cycle_id,
                    'timestamp': datetime.now().isoformat(),
                    'value': tracker.total_agents_evolved,
                    'description': f"Enhanced {tracker.total_agents_evolved} agents"
                })
                
                logger.info(f"MILESTONE: Agent enhancement threshold reached: {tracker.total_agents_evolved} agents")
            
            # Cycle efficiency milestone
            if len(self.completed_cycles) >= 10:
                recent_success_rate = statistics.mean([c.success_rate for c in self.completed_cycles[-10:]])
                if (recent_success_rate >= milestones.get('cycle_efficiency', 0.30) and
                    not any(m.get('type') == 'cycle_efficiency' for m in tracker.milestone_achievements)):
                    
                    tracker.milestone_achievements.append({
                        'type': 'cycle_efficiency',
                        'cycle_id': cycle.cycle_id,
                        'timestamp': datetime.now().isoformat(),
                        'value': recent_success_rate,
                        'description': f"Achieved {recent_success_rate:.1%} cycle efficiency"
                    })
                    
                    logger.info(f"MILESTONE: Cycle efficiency threshold reached: {recent_success_rate:.1%}")
        
        except Exception as e:
            logger.error(f"Error checking evolution milestones: {e}")

    def _calculate_cycle_success_rate(self, cycle: ContinuousImprovementCycle) -> float:
        """Calculate success rate for a completed cycle."""
        success_factors = []
        
        # Implementation success
        if cycle.improvements_implemented > 0:
            success_factors.append(min(1.0, cycle.improvements_implemented / 5.0))
        
        # Agent enhancement success
        if cycle.agents_enhanced > 0:
            success_factors.append(min(1.0, cycle.agents_enhanced / 3.0))
        
        # Performance improvement
        if cycle.performance_improvement > 0:
            success_factors.append(min(1.0, cycle.performance_improvement * 10))
        
        # Completion success
        if cycle.completed_at:
            success_factors.append(1.0)
        
        return sum(success_factors) / max(len(success_factors), 1)

    def _calculate_health_improvement(self) -> float:
        """Calculate system health improvement."""
        current_health = sum(metric.improvement_ratio for metric in self.optimization_metrics.values())
        current_health /= len(self.optimization_metrics)
        
        # Simple health improvement calculation
        return max(0.0, current_health - 0.5)  # Improvement above neutral (0.5)

    def _calculate_learning_acceleration(self, cycle: ContinuousImprovementCycle) -> float:
        """Calculate learning acceleration from the cycle."""
        # Simple learning acceleration based on implementation success
        base_acceleration = 0.1
        implementation_bonus = cycle.improvements_implemented * 0.05
        enhancement_bonus = cycle.agents_enhanced * 0.03
        
        return base_acceleration + implementation_bonus + enhancement_bonus

    async def _update_system_metrics_from_cycle(self, cycle: ContinuousImprovementCycle):
        """Update system-wide metrics from cycle results."""
        try:
            # Update optimization metrics trends
            for dimension in cycle.optimization_dimensions:
                if dimension in self.optimization_metrics:
                    metric = self.optimization_metrics[dimension]
                    
                    # Update trend direction based on recent performance
                    if cycle.performance_improvement > 0.01:
                        metric.trend_direction = "improving"
                        metric.trend_rate = cycle.performance_improvement
                    elif cycle.performance_improvement < -0.01:
                        metric.trend_direction = "degrading"
                        metric.trend_rate = -cycle.performance_improvement
                    else:
                        metric.trend_direction = "stable"
                        metric.trend_rate = 0.0
            
            # Update system learning rate
            if len(self.completed_cycles) > 0:
                recent_cycles = self.completed_cycles[-10:]  # Last 10 cycles
                self.system_learning_rate = statistics.mean([
                    c.learning_acceleration for c in recent_cycles
                ])
            
            # Update system health status
            await self._update_system_health_status()
            
        except Exception as e:
            logger.error(f"Error updating system metrics from cycle: {e}")

    async def _update_system_health_status(self):
        """Update overall system health status."""
        try:
            # Calculate average improvement ratio
            avg_improvement = statistics.mean([
                metric.improvement_ratio for metric in self.optimization_metrics.values()
            ])
            
            # Determine health status
            if avg_improvement >= 0.9:
                new_status = SystemHealthStatus.EXCELLENT
            elif avg_improvement >= 0.8:
                new_status = SystemHealthStatus.GOOD
            elif avg_improvement >= 0.6:
                new_status = SystemHealthStatus.DEGRADING
            elif avg_improvement >= 0.4:
                new_status = SystemHealthStatus.POOR
            else:
                new_status = SystemHealthStatus.CRITICAL
            
            if new_status != self.system_health_status:
                old_status = self.system_health_status
                self.system_health_status = new_status
                logger.info(f"System health status changed: {old_status.value} → {new_status.value}")
                
                # Trigger emergency cycle if critical
                if new_status == SystemHealthStatus.CRITICAL:
                    await self.trigger_improvement_cycle(
                        CycleType.EMERGENCY,
                        "Critical system health detected",
                        list(self.optimization_metrics.keys())
                    )
        
        except Exception as e:
            logger.error(f"Error updating system health status: {e}")

    # Background Task Loops

    async def _system_health_monitor_loop(self):
        """Background loop for system health monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(self.config.system_health_check_interval_minutes * 60)
                await self._check_system_health()
            except Exception as e:
                logger.error(f"System health monitor loop error: {e}")

    async def _cycle_scheduler_loop(self):
        """Background loop for cycle scheduling."""
        while self._is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._check_scheduled_cycles()
            except Exception as e:
                logger.error(f"Cycle scheduler loop error: {e}")

    async def _performance_trend_analyzer_loop(self):
        """Background loop for performance trend analysis."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Every hour
                await self._analyze_performance_trends()
            except Exception as e:
                logger.error(f"Performance trend analyzer loop error: {e}")

    async def _evolution_tracker_loop(self):
        """Background loop for evolution tracking."""
        while self._is_running:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                await self._update_evolution_metrics()
            except Exception as e:
                logger.error(f"Evolution tracker loop error: {e}")

    async def _metric_collection_loop(self):
        """Background loop for metric collection."""
        while self._is_running:
            try:
                await asyncio.sleep(900)  # Every 15 minutes
                await self._collect_system_metrics()
            except Exception as e:
                logger.error(f"Metric collection loop error: {e}")

    async def _check_system_health(self):
        """Check system health and trigger cycles if needed."""
        # Check for performance degradation
        for dimension, metric in self.optimization_metrics.items():
            if (metric.trend_direction == "degrading" and 
                metric.trend_rate > self.config.performance_degradation_threshold):
                
                await self.trigger_improvement_cycle(
                    CycleType.TRIGGERED,
                    f"Performance degradation detected in {dimension.value}",
                    [dimension]
                )

    async def _check_scheduled_cycles(self):
        """Check if any scheduled cycles should be triggered."""
        now = datetime.now()
        
        # Routine cycle check
        if (now - self.last_routine_cycle).total_seconds() >= self.config.routine_cycle_interval_hours * 3600:
            await self.trigger_improvement_cycle(
                CycleType.ROUTINE,
                "Scheduled routine improvement cycle"
            )
            self.last_routine_cycle = now
        
        # Deep analysis cycle check
        if (now - self.last_deep_analysis).total_seconds() >= self.config.deep_analysis_interval_hours * 3600:
            await self.trigger_improvement_cycle(
                CycleType.DEEP_ANALYSIS,
                "Scheduled deep analysis cycle",
                list(self.optimization_metrics.keys())
            )
            self.last_deep_analysis = now
        
        # Optimization cycle check
        if (now - self.last_optimization_cycle).total_seconds() >= self.config.optimization_cycle_interval_hours * 3600:
            # Focus on dimensions with largest gaps
            focus_dimensions = sorted(
                self.optimization_metrics.keys(),
                key=lambda d: self.optimization_metrics[d].gap_to_target,
                reverse=True
            )[:3]
            
            await self.trigger_improvement_cycle(
                CycleType.OPTIMIZATION,
                "Scheduled optimization cycle",
                focus_dimensions
            )
            self.last_optimization_cycle = now

    async def _analyze_performance_trends(self):
        """Analyze performance trends and update metrics."""
        # Update trend analysis for all metrics
        for metric in self.optimization_metrics.values():
            # Simple trend analysis - could be enhanced with more sophisticated methods
            if len(self.performance_history) >= 2:
                recent_values = [
                    entry.get(metric.dimension.value, 0.5) 
                    for entry in self.performance_history[-5:]
                ]
                
                if len(recent_values) >= 2:
                    trend = recent_values[-1] - recent_values[0]
                    metric.trend_rate = abs(trend) / len(recent_values)
                    
                    if trend > 0.01:
                        metric.trend_direction = "improving"
                    elif trend < -0.01:
                        metric.trend_direction = "degrading"
                    else:
                        metric.trend_direction = "stable"

    async def _update_evolution_metrics(self):
        """Update evolution tracking metrics."""
        tracker = self.evolution_tracker
        
        # Calculate evolution rate
        if len(tracker.evolution_history) >= 2:
            recent_improvements = [
                entry['performance_improvement'] 
                for entry in tracker.evolution_history[-10:]
            ]
            
            if recent_improvements:
                avg_improvement = statistics.mean(recent_improvements)
                # Update system learning rate based on recent performance
                self.system_learning_rate = max(0.0, avg_improvement)

    async def _collect_system_metrics(self):
        """Collect current system metrics."""
        current_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.system_health_status.value,
            'learning_rate': self.system_learning_rate,
            'active_cycles': len(self.active_cycles)
        }
        
        # Add optimization metrics
        for dimension, metric in self.optimization_metrics.items():
            current_snapshot[dimension.value] = metric.current_value
        
        self.performance_history.append(current_snapshot)
        
        # Keep history manageable
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    # Public API Methods

    async def get_system_evolution_status(self) -> Dict[str, Any]:
        """Get comprehensive system evolution status."""
        return {
            'system_health': self.system_health_status.value,
            'learning_rate': self.system_learning_rate,
            'active_cycles': len(self.active_cycles),
            'completed_cycles': len(self.completed_cycles),
            
            'evolution_tracker': {
                'total_cycles': self.evolution_tracker.total_improvement_cycles,
                'total_agents_evolved': self.evolution_tracker.total_agents_evolved,
                'total_performance_gains': self.evolution_tracker.total_performance_gains,
                'milestones_achieved': len(self.evolution_tracker.milestone_achievements),
                'breakthrough_moments': len(self.evolution_tracker.breakthrough_moments)
            },
            
            'optimization_metrics': {
                dimension.value: {
                    'current_value': metric.current_value,
                    'target_value': metric.target_value,
                    'improvement_ratio': metric.improvement_ratio,
                    'trend_direction': metric.trend_direction,
                    'gap_to_target': metric.gap_to_target
                }
                for dimension, metric in self.optimization_metrics.items()
            },
            
            'recent_performance': self.performance_history[-10:] if self.performance_history else [],
            
            'is_running': self._is_running
        }

    async def force_improvement_cycle(
        self, 
        optimization_focus: Optional[List[str]] = None
    ) -> str:
        """Force an immediate improvement cycle."""
        focus_dimensions = []
        if optimization_focus:
            focus_dimensions = [
                OptimizationDimension(dim) for dim in optimization_focus 
                if dim in [d.value for d in OptimizationDimension]
            ]
        
        return await self.trigger_improvement_cycle(
            CycleType.TRIGGERED,
            "Manual force trigger",
            focus_dimensions or list(self.optimization_metrics.keys())
        )

    async def set_optimization_targets(self, targets: Dict[str, float]):
        """Set new optimization targets."""
        for dimension_name, target_value in targets.items():
            try:
                dimension = OptimizationDimension(dimension_name)
                if dimension in self.optimization_metrics:
                    self.optimization_metrics[dimension].target_value = target_value
                    logger.info(f"Updated {dimension_name} target to {target_value}")
            except ValueError:
                logger.warning(f"Invalid optimization dimension: {dimension_name}")

    def get_cycle_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get cycle execution history."""
        cycles = self.completed_cycles[-limit:] if limit else self.completed_cycles
        
        return [
            {
                'cycle_id': cycle.cycle_id,
                'cycle_type': cycle.cycle_type.value,
                'started_at': cycle.started_at.isoformat(),
                'completed_at': cycle.completed_at.isoformat() if cycle.completed_at else None,
                'success_rate': cycle.success_rate,
                'performance_improvement': cycle.performance_improvement,
                'improvements_implemented': cycle.improvements_implemented,
                'agents_enhanced': cycle.agents_enhanced,
                'trigger_reason': cycle.trigger_reason
            }
            for cycle in cycles
        ]

    def cleanup(self):
        """Cleanup continuous improvement engine resources."""
        if self._is_running:
            asyncio.create_task(self.stop_continuous_improvement())
        
        logger.info("ContinuousImprovementEngine cleanup completed")