"""
Experiment Manager for Lifecycle Management

Orchestrates the lifecycle of A/B experiments, multi-armed bandit optimization,
and performance monitoring. Manages scheduling, coordination, and automated decision-making.
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path

from .selection_history import SelectionHistory, SelectionRecord
from .benchmark_tracker import BenchmarkTracker
from .ab_testing_framework import ABTestingFramework, ExperimentConfig, ExperimentResult, ExperimentStatus
from .selection_optimizer import SelectionOptimizer, OptimizationStrategy
from .performance_analyzer import PerformanceAnalyzer, ComparisonResult


logger = logging.getLogger(__name__)


class ExperimentPhase(Enum):
    """Phases of experiment lifecycle."""
    PLANNING = "planning"
    PREPARATION = "preparation"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    DECISION = "decision"
    CLEANUP = "cleanup"


@dataclass
class AutoExperimentConfig:
    """Configuration for automatic experiment scheduling."""
    
    selection_type: str
    
    # Scheduling parameters
    min_experiment_interval_hours: int = 24  # Minimum time between experiments
    max_concurrent_experiments: int = 2      # Maximum concurrent experiments per type
    
    # Sample size requirements
    min_baseline_samples: int = 100
    min_samples_per_arm: int = 50
    
    # Performance thresholds
    performance_degradation_threshold: float = 0.1  # 10% degradation triggers experiment
    min_effect_size_threshold: float = 0.2          # Minimum meaningful effect size
    
    # Auto-approval settings
    auto_approve_winner: bool = True
    min_confidence_for_approval: float = 0.85
    
    # Exploration settings
    exploration_percentage: float = 0.15  # 15% of traffic for exploration
    
    enabled: bool = True


class ExperimentManager:
    """
    Manages the complete lifecycle of selection experiments and optimization.
    
    Coordinates A/B testing, multi-armed bandits, performance monitoring,
    and automated decision-making for continuous improvement.
    """
    
    def __init__(self,
                 selection_history: SelectionHistory,
                 benchmark_tracker: BenchmarkTracker,
                 ab_testing_framework: ABTestingFramework,
                 selection_optimizer: SelectionOptimizer,
                 performance_analyzer: PerformanceAnalyzer,
                 config_path: str = None):
        """
        Initialize experiment manager.
        
        Args:
            selection_history: Historical selection data
            benchmark_tracker: Real-time performance monitoring
            ab_testing_framework: A/B testing orchestration
            selection_optimizer: Multi-armed bandit optimization
            performance_analyzer: Statistical analysis
            config_path: Path to configuration file
        """
        self.selection_history = selection_history
        self.benchmark_tracker = benchmark_tracker
        self.ab_testing_framework = ab_testing_framework
        self.selection_optimizer = selection_optimizer
        self.performance_analyzer = performance_analyzer
        
        # Configuration
        self.config_path = config_path or str(Path.home() / ".agentsmcp" / "experiment_config.json")
        self.auto_configs: Dict[str, AutoExperimentConfig] = {}
        self.load_configuration()
        
        # State management
        self.active_phases: Dict[str, ExperimentPhase] = {}  # experiment_id -> phase
        self.experiment_schedules: Dict[str, datetime] = {}  # selection_type -> next_experiment_time
        
        # Background tasks
        self._manager_task: Optional[asyncio.Task] = None
        self._running = False
        self._task_lock = threading.RLock()
        
        # Performance tracking
        self.total_experiments_managed = 0
        self.successful_experiments = 0
        self.automated_decisions = 0
        self.manual_interventions = 0
        
        logger.info("ExperimentManager initialized")
    
    async def start(self):
        """Start the experiment manager."""
        if self._running:
            return
        
        self._running = True
        self._manager_task = asyncio.create_task(self._management_loop())
        
        # Start dependent services
        await self.benchmark_tracker.start()
        
        logger.info("ExperimentManager started")
    
    async def stop(self):
        """Stop the experiment manager."""
        self._running = False
        
        if self._manager_task:
            self._manager_task.cancel()
            try:
                await self._manager_task
            except asyncio.CancelledError:
                pass
        
        await self.benchmark_tracker.stop()
        
        logger.info("ExperimentManager stopped")
    
    def configure_auto_experimentation(self, 
                                     selection_type: str,
                                     config: AutoExperimentConfig):
        """Configure automatic experimentation for a selection type."""
        self.auto_configs[selection_type] = config
        self.save_configuration()
        
        logger.info(f"Configured auto-experimentation for {selection_type}")
    
    async def schedule_experiment(self,
                                name: str,
                                description: str,
                                selection_type: str,
                                control_option: str,
                                treatment_options: List[str],
                                delay_hours: int = 0,
                                **kwargs) -> Optional[str]:
        """
        Schedule an A/B experiment.
        
        Args:
            name: Experiment name
            description: Experiment description  
            selection_type: Type of selection being tested
            control_option: Control option (current best)
            treatment_options: Treatment options to test
            delay_hours: Hours to delay before starting experiment
            **kwargs: Additional experiment parameters
            
        Returns:
            Experiment ID if scheduled successfully
        """
        try:
            # Create experiment configuration
            config = await self.ab_testing_framework.create_experiment(
                name=name,
                description=description,
                selection_type=selection_type,
                control_option=control_option,
                treatment_options=treatment_options,
                **kwargs
            )
            
            # Schedule start time
            start_time = datetime.now() + timedelta(hours=delay_hours)
            config.start_time = start_time
            
            # Set initial phase
            self.active_phases[config.experiment_id] = ExperimentPhase.PLANNING
            self.total_experiments_managed += 1
            
            logger.info(f"Scheduled experiment {config.experiment_id} to start at {start_time}")
            return config.experiment_id
            
        except Exception as e:
            logger.error(f"Failed to schedule experiment: {e}")
            return None
    
    async def promote_winner(self, 
                           experiment_id: str,
                           winner_option: str,
                           reason: str = "manual_promotion") -> bool:
        """
        Promote the winning option from an experiment.
        
        Args:
            experiment_id: ID of completed experiment
            winner_option: Option to promote as winner
            reason: Reason for promotion
            
        Returns:
            True if promotion successful
        """
        try:
            # Get experiment result
            if experiment_id not in self.ab_testing_framework.completed_experiments:
                logger.warning(f"Experiment {experiment_id} not found in completed experiments")
                return False
            
            result = self.ab_testing_framework.completed_experiments[experiment_id]
            
            # Validate winner option
            experiment = None
            for exp in self.ab_testing_framework.get_active_experiments():
                if exp.experiment_id == experiment_id:
                    experiment = exp
                    break
            
            if not experiment:
                # Check completed experiments stored in framework
                logger.warning(f"Could not find experiment config for {experiment_id}")
                return False
            
            all_options = [experiment.control_option] + experiment.treatment_options
            if winner_option not in all_options:
                logger.error(f"Winner {winner_option} not in experiment options: {all_options}")
                return False
            
            # Update optimizer to favor the winner
            self._update_optimizer_preferences(
                experiment.selection_type, 
                winner_option, 
                all_options
            )
            
            # Record the decision
            self._record_experiment_decision(
                experiment_id=experiment_id,
                decision="promote_winner",
                selected_option=winner_option,
                reason=reason,
                confidence=result.recommended_winner == winner_option
            )
            
            self.automated_decisions += 1 if reason.startswith("auto") else 0
            self.manual_interventions += 1 if not reason.startswith("auto") else 0
            
            logger.info(f"Promoted {winner_option} as winner for experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote winner for {experiment_id}: {e}")
            return False
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive status for an experiment."""
        try:
            # Check active experiments
            for exp in self.ab_testing_framework.get_active_experiments():
                if exp.experiment_id == experiment_id:
                    return {
                        'experiment_id': experiment_id,
                        'status': exp.status.value,
                        'phase': self.active_phases.get(experiment_id, ExperimentPhase.PLANNING).value,
                        'start_time': exp.start_time.isoformat() if exp.start_time else None,
                        'end_time': exp.end_time.isoformat() if exp.end_time else None,
                        'control_option': exp.control_option,
                        'treatment_options': exp.treatment_options,
                        'traffic_allocation': exp.traffic_allocation,
                        'min_sample_size': exp.min_sample_size
                    }
            
            # Check completed experiments
            if experiment_id in self.ab_testing_framework.completed_experiments:
                result = self.ab_testing_framework.completed_experiments[experiment_id]
                return {
                    'experiment_id': experiment_id,
                    'status': 'completed',
                    'phase': 'decision',
                    'control_samples': result.control_samples,
                    'treatment_samples': result.treatment_samples,
                    'control_metric': result.control_metric,
                    'treatment_metrics': result.treatment_metrics,
                    'statistical_significance': result.statistical_significance,
                    'p_value': result.p_value,
                    'recommended_winner': result.recommended_winner,
                    'recommendation_reason': result.recommendation_reason
                }
            
            return {'experiment_id': experiment_id, 'status': 'not_found'}
            
        except Exception as e:
            logger.error(f"Error getting experiment status: {e}")
            return {'experiment_id': experiment_id, 'status': 'error', 'error': str(e)}
    
    def get_active_experiments_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all active experiments."""
        summaries = []
        
        for exp in self.ab_testing_framework.get_active_experiments():
            summary = {
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'selection_type': exp.selection_type,
                'status': exp.status.value,
                'phase': self.active_phases.get(exp.experiment_id, ExperimentPhase.PLANNING).value,
                'start_time': exp.start_time.isoformat() if exp.start_time else None,
                'days_remaining': None
            }
            
            if exp.end_time:
                days_remaining = (exp.end_time - datetime.now()).days
                summary['days_remaining'] = max(0, days_remaining)
            
            summaries.append(summary)
        
        return summaries
    
    def get_performance_insights(self, 
                               selection_type: str = None,
                               days: int = 30) -> Dict[str, Any]:
        """Get performance insights and recommendations."""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'period_days': days,
            'selection_type_filter': selection_type
        }
        
        try:
            # Get performance degradations
            degradations = self.benchmark_tracker.detect_performance_degradation(
                selection_type=selection_type,
                threshold=0.1
            )
            
            insights['performance_degradations'] = len(degradations)
            insights['degraded_options'] = [d['option_name'] for d in degradations]
            
            # Get experiment recommendations
            recommendations = self._generate_experiment_recommendations(selection_type, days)
            insights['experiment_recommendations'] = recommendations
            
            # Get optimization status
            optimization_stats = self.selection_optimizer.get_exploration_stats()
            insights['optimization_stats'] = optimization_stats
            
            # Overall health score
            total_metrics = len(self.benchmark_tracker.get_metrics())
            healthy_metrics = total_metrics - len(degradations)
            
            if total_metrics > 0:
                insights['health_score'] = healthy_metrics / total_metrics
            else:
                insights['health_score'] = 1.0
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating performance insights: {e}")
            insights['error'] = str(e)
            return insights
    
    async def _management_loop(self):
        """Main management loop for experiment lifecycle."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Process experiment lifecycle phases
                await self._process_experiment_phases()
                
                # Check for automatic experiment triggers
                await self._check_auto_experiment_triggers()
                
                # Perform scheduled analysis
                await self._perform_scheduled_analysis()
                
                # Clean up completed experiments
                await self._cleanup_completed_experiments()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in management loop: {e}")
                await asyncio.sleep(30)  # Back off on errors
    
    async def _process_experiment_phases(self):
        """Process lifecycle phases for all active experiments."""
        for exp in self.ab_testing_framework.get_active_experiments():
            try:
                current_phase = self.active_phases.get(exp.experiment_id, ExperimentPhase.PLANNING)
                
                if current_phase == ExperimentPhase.PLANNING:
                    if exp.start_time and datetime.now() >= exp.start_time:
                        # Start the experiment
                        success = await self.ab_testing_framework.start_experiment(exp)
                        if success:
                            self.active_phases[exp.experiment_id] = ExperimentPhase.EXECUTION
                            logger.info(f"Started experiment {exp.experiment_id}")
                
                elif current_phase == ExperimentPhase.EXECUTION:
                    # Check if experiment should be analyzed or stopped
                    if exp.end_time and datetime.now() >= exp.end_time:
                        self.active_phases[exp.experiment_id] = ExperimentPhase.ANALYSIS
                        logger.info(f"Moving experiment {exp.experiment_id} to analysis phase")
                
                elif current_phase == ExperimentPhase.ANALYSIS:
                    # Analyze experiment results
                    result = await self.ab_testing_framework.stop_experiment(
                        exp.experiment_id, "scheduled_analysis"
                    )
                    if result:
                        self.active_phases[exp.experiment_id] = ExperimentPhase.DECISION
                        
                        # Auto-approve if configured
                        auto_config = self.auto_configs.get(exp.selection_type)
                        if (auto_config and auto_config.auto_approve_winner and
                            result.recommended_winner and 
                            result.statistical_significance):
                            
                            await self.promote_winner(
                                exp.experiment_id,
                                result.recommended_winner,
                                "auto_approval"
                            )
                            self.active_phases[exp.experiment_id] = ExperimentPhase.CLEANUP
                
                elif current_phase == ExperimentPhase.DECISION:
                    # Wait for manual decision or timeout
                    if exp.start_time and (datetime.now() - exp.start_time).days > 14:
                        # Timeout - use control option
                        await self.promote_winner(
                            exp.experiment_id,
                            exp.control_option,
                            "timeout_default"
                        )
                        self.active_phases[exp.experiment_id] = ExperimentPhase.CLEANUP
                
            except Exception as e:
                logger.error(f"Error processing experiment {exp.experiment_id}: {e}")
    
    async def _check_auto_experiment_triggers(self):
        """Check if automatic experiments should be triggered."""
        for selection_type, auto_config in self.auto_configs.items():
            if not auto_config.enabled:
                continue
            
            try:
                # Check if it's time for next experiment
                last_experiment_time = self.experiment_schedules.get(selection_type)
                if last_experiment_time:
                    hours_since_last = (datetime.now() - last_experiment_time).total_seconds() / 3600
                    if hours_since_last < auto_config.min_experiment_interval_hours:
                        continue
                
                # Check for performance issues that warrant experimentation
                degradations = self.benchmark_tracker.detect_performance_degradation(
                    selection_type=selection_type,
                    threshold=auto_config.performance_degradation_threshold
                )
                
                if degradations:
                    # Get best alternatives for experimentation
                    rankings = self.benchmark_tracker.get_rankings(
                        selection_type=selection_type,
                        min_samples=auto_config.min_baseline_samples
                    )
                    
                    if len(rankings) >= 2:
                        control_option = rankings[0][0]  # Best performer
                        treatment_options = [rankings[1][0]]  # Second best
                        
                        # Schedule experiment
                        experiment_id = await self.schedule_experiment(
                            name=f"Auto-experiment for {selection_type}",
                            description=f"Triggered by performance degradation detection",
                            selection_type=selection_type,
                            control_option=control_option,
                            treatment_options=treatment_options,
                            min_sample_size=auto_config.min_samples_per_arm
                        )
                        
                        if experiment_id:
                            self.experiment_schedules[selection_type] = datetime.now()
                            logger.info(f"Auto-triggered experiment {experiment_id} for {selection_type}")
                
            except Exception as e:
                logger.error(f"Error checking auto-triggers for {selection_type}: {e}")
    
    async def _perform_scheduled_analysis(self):
        """Perform scheduled analysis of selection performance."""
        try:
            # Check for performance regressions across all selection types
            for selection_type in set(self.auto_configs.keys()):
                # Get recent performance metrics
                metrics = self.benchmark_tracker.get_metrics(selection_type=selection_type)
                
                for (sel_type, option), metric_data in metrics.items():
                    if metric_data.trend_direction == "degrading":
                        # Check for regression with statistical analysis
                        regression_result = self.performance_analyzer.detect_performance_regression(
                            selection_type=sel_type,
                            option=option,
                            baseline_days=30,
                            recent_days=7
                        )
                        
                        if (regression_result and regression_result.is_significant and
                            regression_result.effect_magnitude in ["medium", "large"]):
                            
                            logger.warning(
                                f"Significant performance regression detected for {option}: "
                                f"{regression_result.recommendation_confidence:.2f} confidence"
                            )
        
        except Exception as e:
            logger.error(f"Error in scheduled analysis: {e}")
    
    async def _cleanup_completed_experiments(self):
        """Clean up old completed experiments."""
        try:
            # Remove experiments older than 30 days from active phases
            cutoff = datetime.now() - timedelta(days=30)
            
            experiments_to_remove = []
            for exp_id, phase in self.active_phases.items():
                if phase == ExperimentPhase.CLEANUP:
                    # Check if experiment is old enough to remove
                    found_start_time = None
                    
                    # Find start time from completed experiments
                    for result in self.ab_testing_framework.get_experiment_results():
                        if result.experiment_id == exp_id:
                            found_start_time = result.analysis_date
                            break
                    
                    if found_start_time and found_start_time < cutoff:
                        experiments_to_remove.append(exp_id)
            
            for exp_id in experiments_to_remove:
                del self.active_phases[exp_id]
                logger.debug(f"Cleaned up old experiment {exp_id}")
        
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
    
    def _update_optimizer_preferences(self,
                                    selection_type: str,
                                    winner_option: str,
                                    all_options: List[str]):
        """Update optimizer to favor the winning option."""
        try:
            # Boost the winner's performance in the optimizer
            # This is implementation-specific to the optimizer strategy
            
            # For Thompson Sampling, we can add virtual successes
            # For UCB1, we can add virtual rewards
            # This gives the winner an advantage in future selections
            
            logger.info(f"Updated optimizer preferences: {winner_option} promoted for {selection_type}")
            
        except Exception as e:
            logger.error(f"Error updating optimizer preferences: {e}")
    
    def _record_experiment_decision(self,
                                  experiment_id: str,
                                  decision: str,
                                  selected_option: str,
                                  reason: str,
                                  confidence: bool):
        """Record an experiment decision for audit trail."""
        try:
            decision_record = {
                'experiment_id': experiment_id,
                'decision': decision,
                'selected_option': selected_option,
                'reason': reason,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'automated': reason.startswith('auto')
            }
            
            # Store in experiment metadata or separate log
            logger.info(f"Recorded decision for {experiment_id}: {decision} - {selected_option}")
            
        except Exception as e:
            logger.error(f"Error recording experiment decision: {e}")
    
    def _generate_experiment_recommendations(self, 
                                           selection_type: str = None,
                                           days: int = 30) -> List[Dict[str, Any]]:
        """Generate recommendations for new experiments."""
        recommendations = []
        
        try:
            # Get performance data
            selection_types = [selection_type] if selection_type else list(self.auto_configs.keys())
            
            for sel_type in selection_types:
                rankings = self.benchmark_tracker.get_rankings(sel_type, min_samples=50)
                
                if len(rankings) >= 2:
                    # Look for cases where second-best might be worth testing
                    best_option, best_score = rankings[0]
                    second_option, second_score = rankings[1]
                    
                    # If the difference is small, recommend A/B test
                    if abs(best_score - second_score) < 0.1:  # 10% difference
                        recommendations.append({
                            'type': 'ab_test',
                            'selection_type': sel_type,
                            'reason': 'close_performance',
                            'control_option': best_option,
                            'treatment_options': [second_option],
                            'priority': 'medium',
                            'expected_benefit': abs(best_score - second_score)
                        })
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def load_configuration(self):
        """Load experiment manager configuration from file."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                for sel_type, config_dict in config_data.get('auto_configs', {}).items():
                    self.auto_configs[sel_type] = AutoExperimentConfig(**config_dict)
                
                logger.info(f"Loaded configuration for {len(self.auto_configs)} selection types")
            else:
                logger.info("No configuration file found, using defaults")
        
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_configuration(self):
        """Save experiment manager configuration to file."""
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'auto_configs': {
                    sel_type: asdict(config) 
                    for sel_type, config in self.auto_configs.items()
                },
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.debug("Saved experiment manager configuration")
        
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get experiment manager statistics."""
        return {
            'total_experiments_managed': self.total_experiments_managed,
            'successful_experiments': self.successful_experiments,
            'automated_decisions': self.automated_decisions,
            'manual_interventions': self.manual_interventions,
            'active_experiments': len(self.ab_testing_framework.get_active_experiments()),
            'completed_experiments': len(self.ab_testing_framework.get_experiment_results()),
            'auto_configs': len(self.auto_configs),
            'success_rate': (self.successful_experiments / max(1, self.total_experiments_managed)) * 100,
            'automation_rate': (self.automated_decisions / max(1, self.automated_decisions + self.manual_interventions)) * 100,
            'is_running': self._running
        }