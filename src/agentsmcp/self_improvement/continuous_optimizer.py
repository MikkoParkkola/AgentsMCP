"""Continuous Optimizer - Main Orchestration and Scheduling

This module provides the main orchestration and scheduling system for the
AgentsMCP self-improvement loops, coordinating all components to deliver
continuous system optimization.

SECURITY: Secure orchestration with comprehensive validation
PERFORMANCE: Efficient scheduling with adaptive optimization - <200ms overhead per task
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager
import signal
import threading

from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from .improvement_detector import ImprovementDetector, ImprovementOpportunity
from .improvement_implementer import ImprovementImplementer, ImplementationResult
from .metrics_collector import MetricsCollector
from .improvement_history import ImprovementHistory
from .user_feedback_integrator import UserFeedbackIntegrator

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """Optimization operation modes."""
    DISABLED = "disabled"          # No optimization
    PASSIVE = "passive"            # Collect data only
    ANALYSIS_ONLY = "analysis_only"  # Analyze but don't implement
    CONSERVATIVE = "conservative"   # Implement low-risk improvements only
    ACTIVE = "active"              # Full optimization with safety checks
    AGGRESSIVE = "aggressive"      # Maximum optimization (use with caution)


class OptimizationPhase(Enum):
    """Current phase of optimization cycle."""
    IDLE = "idle"
    COLLECTING_METRICS = "collecting_metrics"
    ANALYZING_PERFORMANCE = "analyzing_performance" 
    DETECTING_IMPROVEMENTS = "detecting_improvements"
    IMPLEMENTING_IMPROVEMENTS = "implementing_improvements"
    VALIDATING_RESULTS = "validating_results"
    UPDATING_HISTORY = "updating_history"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class OptimizationConfig:
    """Configuration for continuous optimization."""
    
    # Operation mode
    mode: OptimizationMode = OptimizationMode.ACTIVE
    
    # Scheduling
    optimization_interval_seconds: int = 300  # 5 minutes between optimizations
    analysis_interval_seconds: int = 60      # 1 minute between analyses
    metrics_collection_interval_seconds: int = 10  # 10 seconds between metric collections
    
    # Safety limits
    max_implementations_per_cycle: int = 3
    max_parallel_implementations: int = 2
    implementation_timeout_seconds: int = 300  # 5 minutes per implementation
    
    # Quality gates
    min_confidence_threshold: float = 0.7
    max_acceptable_regression: float = 0.1  # 10%
    require_user_feedback_integration: bool = True
    
    # Resource limits
    max_cpu_usage_during_optimization: float = 50.0  # 50%
    max_memory_usage_mb: int = 512
    
    # Feature flags
    enable_automatic_rollback: bool = True
    enable_a_b_testing: bool = False
    enable_predictive_optimization: bool = True
    enable_user_notification: bool = True
    
    # Debugging and observability
    detailed_logging: bool = False
    export_optimization_reports: bool = True
    telemetry_enabled: bool = True


@dataclass
class OptimizationCycle:
    """Represents one complete optimization cycle."""
    
    # Identification
    cycle_id: str
    start_time: datetime = field(default_factory=datetime.now)
    
    # Phase tracking
    current_phase: OptimizationPhase = OptimizationPhase.IDLE
    completed_phases: List[OptimizationPhase] = field(default_factory=list)
    
    # Results
    metrics_collected: Optional[PerformanceMetrics] = None
    opportunities_detected: List[ImprovementOpportunity] = field(default_factory=list)
    implementations_attempted: List[ImplementationResult] = field(default_factory=list)
    
    # Performance tracking
    cycle_duration_seconds: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    
    # Impact measurement
    before_metrics: Optional[PerformanceMetrics] = None
    after_metrics: Optional[PerformanceMetrics] = None
    measured_improvement: Dict[str, float] = field(default_factory=dict)


class OptimizationScheduler:
    """Manages scheduling and coordination of optimization cycles."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._scheduler_active = False
        self._current_cycle: Optional[OptimizationCycle] = None
        self._cycle_history: List[OptimizationCycle] = []
        
        # Async tasks
        self._scheduler_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Resource monitoring
        self._resource_monitor_active = False
        
    async def start(self) -> None:
        """Start the optimization scheduler."""
        if self._scheduler_active:
            logger.warning("Optimization scheduler already running")
            return
        
        self._scheduler_active = True
        
        # Start background tasks
        self._scheduler_task = asyncio.create_task(self._optimization_loop())
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        self._metrics_task = asyncio.create_task(self._metrics_collection_loop())
        
        logger.info(f"Optimization scheduler started in {self.config.mode.value} mode")
    
    async def stop(self) -> None:
        """Stop the optimization scheduler gracefully."""
        self._scheduler_active = False
        
        # Cancel background tasks
        for task in [self._scheduler_task, self._analysis_task, self._metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Optimization scheduler stopped")
    
    async def _optimization_loop(self) -> None:
        """Main optimization scheduling loop."""
        try:
            while self._scheduler_active:
                if self.config.mode != OptimizationMode.DISABLED:
                    await self._maybe_trigger_optimization_cycle()
                
                await asyncio.sleep(self.config.optimization_interval_seconds)
                
        except asyncio.CancelledError:
            logger.debug("Optimization loop cancelled")
        except Exception as e:
            logger.error(f"Optimization loop error: {e}")
    
    async def _analysis_loop(self) -> None:
        """Continuous analysis loop."""
        try:
            while self._scheduler_active:
                if self.config.mode not in [OptimizationMode.DISABLED, OptimizationMode.PASSIVE]:
                    # Perform lightweight analysis
                    pass
                
                await asyncio.sleep(self.config.analysis_interval_seconds)
                
        except asyncio.CancelledError:
            logger.debug("Analysis loop cancelled")
        except Exception as e:
            logger.error(f"Analysis loop error: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Continuous metrics collection loop."""
        try:
            while self._scheduler_active:
                # Metrics collection happens automatically through other components
                await asyncio.sleep(self.config.metrics_collection_interval_seconds)
                
        except asyncio.CancelledError:
            logger.debug("Metrics collection loop cancelled")
        except Exception as e:
            logger.error(f"Metrics collection loop error: {e}")
    
    async def _maybe_trigger_optimization_cycle(self) -> None:
        """Determine if optimization cycle should be triggered."""
        # Check if already running
        if self._current_cycle and self._current_cycle.current_phase != OptimizationPhase.IDLE:
            return
        
        # Check resource constraints
        if not await self._check_resource_availability():
            logger.debug("Skipping optimization cycle due to resource constraints")
            return
        
        # Trigger optimization cycle
        await self._run_optimization_cycle()
    
    async def _check_resource_availability(self) -> bool:
        """Check if system resources are available for optimization."""
        try:
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config.max_cpu_usage_during_optimization:
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            available_mb = memory.available / 1024 / 1024
            if available_mb < self.config.max_memory_usage_mb:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
            return True  # Proceed if check fails
    
    async def _run_optimization_cycle(self) -> OptimizationCycle:
        """Run complete optimization cycle."""
        cycle_id = f"opt_cycle_{int(time.time())}"
        cycle = OptimizationCycle(cycle_id=cycle_id)
        self._current_cycle = cycle
        
        logger.info(f"Starting optimization cycle: {cycle_id}")
        
        try:
            # The actual optimization work is delegated to ContinuousOptimizer
            # This scheduler just manages timing and coordination
            cycle.success = True
            
        except Exception as e:
            logger.error(f"Optimization cycle failed: {e}")
            cycle.error_message = str(e)
            cycle.success = False
        
        finally:
            cycle.cycle_duration_seconds = (datetime.now() - cycle.start_time).total_seconds()
            self._cycle_history.append(cycle)
            self._current_cycle = None
        
        return cycle


class ContinuousOptimizer:
    """
    Main orchestration and scheduling system for AgentsMCP self-improvement.
    
    Coordinates all improvement components to deliver continuous system
    optimization with comprehensive safety mechanisms and user visibility.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # Initialize configuration
        self.config = OptimizationConfig(**(config or {}))
        
        # Initialize core components
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.metrics_collector = MetricsCollector(config)
        self.improvement_detector = ImprovementDetector(self.performance_analyzer, config)
        self.improvement_implementer = ImprovementImplementer(self.performance_analyzer, config)
        self.improvement_history = ImprovementHistory(config)
        self.user_feedback_integrator = UserFeedbackIntegrator(self.metrics_collector, config)
        
        # Initialize scheduler
        self.scheduler = OptimizationScheduler(self.config)
        
        # System state
        self._optimizer_active = False
        self._current_task_id: Optional[str] = None
        self._optimization_stats = {
            'cycles_completed': 0,
            'improvements_implemented': 0,
            'rollbacks_performed': 0,
            'total_optimization_time': 0.0
        }
        
        # Integration callbacks
        self._task_completion_callbacks: List[Callable] = []
        self._improvement_notification_callbacks: List[Callable] = []
        
        logger.info("ContinuousOptimizer initialized")
    
    async def start(self) -> None:
        """Start the continuous optimization system."""
        if self._optimizer_active:
            logger.warning("ContinuousOptimizer already active")
            return
        
        self._optimizer_active = True
        
        # Start scheduler
        await self.scheduler.start()
        
        logger.info(f"ContinuousOptimizer started in {self.config.mode.value} mode")
    
    async def stop(self) -> None:
        """Stop the continuous optimization system gracefully."""
        self._optimizer_active = False
        
        # Stop scheduler
        await self.scheduler.stop()
        
        # Shutdown components
        self.metrics_collector.shutdown()
        
        logger.info("ContinuousOptimizer stopped")
    
    async def on_task_start(self, task_id: str, context: Dict[str, Any] = None) -> None:
        """
        Hook called when a user task starts.
        
        SECURITY: Input validation for task tracking
        PERFORMANCE: <5ms overhead for task start tracking
        """
        if not self._optimizer_active:
            return
        
        # THREAT: Task ID injection
        # MITIGATION: Input validation
        if not isinstance(task_id, str) or len(task_id) > 100:
            logger.warning(f"Invalid task_id: {task_id}")
            return
        
        self._current_task_id = task_id
        
        # Start performance measurement
        await self.performance_analyzer.start_task_measurement(task_id, context)
        
        # Start metrics collection context
        self.metrics_collector.start_task_context(task_id, context)
        
        logger.debug(f"Task started: {task_id}")
    
    async def on_task_complete(self, 
                             task_id: str, 
                             success: bool = True, 
                             error: Optional[str] = None,
                             user_feedback: Optional[Dict[str, Any]] = None) -> None:
        """
        Hook called when a user task completes.
        
        This is the main trigger for the self-improvement cycle.
        """
        if not self._optimizer_active or task_id != self._current_task_id:
            return
        
        logger.info(f"Task completed: {task_id}, success: {success}")
        
        try:
            # End performance measurement
            performance_metrics = await self.performance_analyzer.end_task_measurement(task_id)
            
            # End metrics collection context
            self.metrics_collector.end_task_context(task_id, success, error)
            
            # Collect user feedback if provided
            if user_feedback:
                await self._process_user_feedback(task_id, user_feedback)
            
            # Trigger self-improvement cycle if in active mode
            if self.config.mode in [OptimizationMode.ACTIVE, OptimizationMode.AGGRESSIVE]:
                await self._run_improvement_cycle(task_id, performance_metrics)
            elif self.config.mode == OptimizationMode.ANALYSIS_ONLY:
                await self._run_analysis_only(task_id, performance_metrics)
            
            # Notify callbacks
            for callback in self._task_completion_callbacks:
                try:
                    await callback(task_id, success, performance_metrics)
                except Exception as e:
                    logger.error(f"Task completion callback failed: {e}")
        
        except Exception as e:
            logger.error(f"Task completion processing failed: {e}")
        
        finally:
            self._current_task_id = None
    
    async def _process_user_feedback(self, task_id: str, user_feedback: Dict[str, Any]) -> None:
        """Process user feedback from task completion."""
        feedback_type = user_feedback.get('type', 'explicit')
        
        if feedback_type == 'explicit':
            # Explicit rating and comment
            rating = user_feedback.get('rating')
            comment = user_feedback.get('comment', '')
            if rating:
                self.user_feedback_integrator.collect_explicit_feedback(
                    user_session_id=self.metrics_collector._current_session_id,
                    rating=rating,
                    comment=comment,
                    task_id=task_id
                )
        
        elif feedback_type == 'implicit':
            # Implicit behavior data
            behavior_data = user_feedback.get('behavior_data', {})
            self.user_feedback_integrator.collect_implicit_feedback(
                user_session_id=self.metrics_collector._current_session_id,
                behavior_data=behavior_data,
                task_id=task_id
            )
    
    async def _run_improvement_cycle(self, task_id: str, metrics: PerformanceMetrics) -> None:
        """Run complete improvement cycle after task completion."""
        cycle_start_time = time.time()
        
        try:
            logger.info(f"Running improvement cycle for task: {task_id}")
            
            # 1. Detect improvement opportunities
            opportunities = await self.improvement_detector.detect_improvements(force_analysis=True)
            if not opportunities:
                logger.debug("No improvement opportunities detected")
                return
            
            logger.info(f"Detected {len(opportunities)} improvement opportunities")
            
            # 2. Filter and prioritize opportunities
            filtered_opportunities = self._filter_opportunities(opportunities)
            
            # 3. Implement improvements
            implementation_results = []
            for opportunity in filtered_opportunities[:self.config.max_implementations_per_cycle]:
                try:
                    result = await self.improvement_implementer.implement_improvement(
                        opportunity, 
                        dry_run=(self.config.mode == OptimizationMode.CONSERVATIVE and opportunity.risk_level == "high")
                    )
                    implementation_results.append(result)
                    
                    # Record in history
                    if result.success:
                        self.improvement_history.record_implementation(
                            opportunity, result,
                            before_state={'metrics': asdict(metrics)},
                            after_state={'metrics': asdict(result.after_metrics) if result.after_metrics else {}}
                        )
                        
                        # Update stats
                        self._optimization_stats['improvements_implemented'] += 1
                        
                        # Notify about successful improvement
                        await self._notify_improvement_implemented(opportunity, result)
                    
                except Exception as e:
                    logger.error(f"Failed to implement improvement {opportunity.opportunity_id}: {e}")
            
            # 4. Analyze feedback patterns (background task)
            asyncio.create_task(self._analyze_user_feedback())
            
            # Update stats
            self._optimization_stats['cycles_completed'] += 1
            self._optimization_stats['total_optimization_time'] += time.time() - cycle_start_time
            
            logger.info(f"Improvement cycle completed: {len(implementation_results)} implementations attempted")
            
        except Exception as e:
            logger.error(f"Improvement cycle failed: {e}")
    
    async def _run_analysis_only(self, task_id: str, metrics: PerformanceMetrics) -> None:
        """Run analysis without implementing improvements."""
        try:
            # Detect opportunities
            opportunities = await self.improvement_detector.detect_improvements()
            
            # Analyze feedback
            feedback_insights = await self.user_feedback_integrator.analyze_feedback_patterns()
            
            logger.info(f"Analysis completed: {len(opportunities)} opportunities, {len(feedback_insights)} feedback insights")
            
        except Exception as e:
            logger.error(f"Analysis cycle failed: {e}")
    
    def _filter_opportunities(self, opportunities: List[ImprovementOpportunity]) -> List[ImprovementOpportunity]:
        """Filter and prioritize improvement opportunities."""
        # Filter by confidence threshold
        filtered = [
            opp for opp in opportunities 
            if opp.confidence_score >= self.config.min_confidence_threshold
        ]
        
        # Filter by mode-specific rules
        if self.config.mode == OptimizationMode.CONSERVATIVE:
            filtered = [opp for opp in filtered if opp.risk_level in ["low", "medium"]]
        elif self.config.mode == OptimizationMode.AGGRESSIVE:
            pass  # Allow all opportunities
        
        # Sort by priority and impact
        filtered.sort(key=lambda opp: (
            opp.priority.value,  # Critical first
            -opp.estimated_roi,  # Higher ROI first
            -opp.confidence_score  # Higher confidence first
        ))
        
        return filtered
    
    async def _analyze_user_feedback(self) -> None:
        """Analyze user feedback patterns (background task)."""
        try:
            insights = await self.user_feedback_integrator.analyze_feedback_patterns()
            if insights:
                logger.info(f"Generated {len(insights)} user feedback insights")
                
                # Convert high-impact insights to improvement opportunities
                # This would be implemented based on specific insight categories
                
        except Exception as e:
            logger.error(f"User feedback analysis failed: {e}")
    
    async def _notify_improvement_implemented(self, 
                                           opportunity: ImprovementOpportunity,
                                           result: ImplementationResult) -> None:
        """Notify about successful improvement implementation."""
        # Notify registered callbacks
        for callback in self._improvement_notification_callbacks:
            try:
                await callback(opportunity, result)
            except Exception as e:
                logger.error(f"Improvement notification callback failed: {e}")
    
    def add_task_completion_callback(self, callback: Callable) -> None:
        """Add callback for task completion events."""
        self._task_completion_callbacks.append(callback)
    
    def add_improvement_notification_callback(self, callback: Callable) -> None:
        """Add callback for improvement implementation events."""
        self._improvement_notification_callbacks.append(callback)
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization system status."""
        # Get component statuses
        performance_trends = await self.performance_analyzer.analyze_performance_trends()
        improvement_summary = await self.improvement_detector.get_improvement_summary()
        implementation_status = await self.improvement_implementer.get_implementation_status()
        history_summary = self.improvement_history.get_history_summary()
        feedback_summary = self.user_feedback_integrator.get_feedback_summary()
        metrics_summary = await self.metrics_collector.get_metrics_summary()
        
        return {
            'optimizer_active': self._optimizer_active,
            'current_mode': self.config.mode.value,
            'current_task': self._current_task_id,
            'optimization_stats': self._optimization_stats.copy(),
            'performance_trends': performance_trends,
            'improvement_summary': improvement_summary,
            'implementation_status': implementation_status,
            'history_summary': history_summary,
            'feedback_summary': feedback_summary,
            'metrics_summary': metrics_summary,
            'last_updated': datetime.now().isoformat()
        }
    
    async def manual_optimization_cycle(self) -> Dict[str, Any]:
        """Manually trigger an optimization cycle."""
        if not self._optimizer_active:
            return {'error': 'Optimizer not active'}
        
        logger.info("Manual optimization cycle triggered")
        
        # Create synthetic task for manual optimization
        manual_task_id = f"manual_{int(time.time())}"
        
        # Start measurement
        await self.performance_analyzer.start_task_measurement(manual_task_id)
        await asyncio.sleep(1)  # Brief measurement period
        metrics = await self.performance_analyzer.end_task_measurement(manual_task_id)
        
        # Run improvement cycle
        await self._run_improvement_cycle(manual_task_id, metrics)
        
        return {
            'success': True,
            'task_id': manual_task_id,
            'timestamp': datetime.now().isoformat(),
            'status': await self.get_optimization_status()
        }
    
    async def rollback_improvement(self, entry_id: str) -> Dict[str, Any]:
        """Rollback a specific improvement."""
        try:
            success = await self.improvement_history.execute_rollback(entry_id, "manual_request")
            
            if success:
                self._optimization_stats['rollbacks_performed'] += 1
                
            return {
                'success': success,
                'entry_id': entry_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Rollback failed for entry {entry_id}: {e}")
            return {
                'success': False,
                'entry_id': entry_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def export_optimization_report(self, filepath: Optional[str] = None) -> str:
        """Export comprehensive optimization report."""
        if not filepath:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'/tmp/agentsmcp_optimization_report_{timestamp}.json'
        
        # Gather comprehensive data
        status = await self.get_optimization_status()
        
        report_data = {
            'report_timestamp': datetime.now().isoformat(),
            'system_status': status,
            'config': asdict(self.config),
            'recommendations': await self.performance_analyzer.get_optimization_recommendations(),
            'rollback_candidates': self.improvement_history.get_rollback_candidates(),
        }
        
        # Export metrics and feedback analysis
        metrics_export = await self.metrics_collector.export_metrics()
        feedback_export = await self.user_feedback_integrator.export_feedback_analysis()
        
        report_data['exported_files'] = {
            'metrics': metrics_export,
            'feedback': feedback_export
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Optimization report exported to: {filepath}")
        return filepath
    
    @asynccontextmanager
    async def optimization_context(self):
        """Context manager for optimization lifecycle."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()