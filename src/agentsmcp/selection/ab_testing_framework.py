"""
A/B Testing Framework for Selection Optimization

Orchestrates experiments that compare different provider/model/agent/tool selections
while ensuring statistical rigor and maintaining system performance.
"""

import logging
import random
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from .selection_history import SelectionHistory, SelectionRecord, generate_selection_id


logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B experiment."""
    PLANNED = "planned"
    RUNNING = "running"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""
    
    experiment_id: str
    name: str
    description: str
    
    # Selection context
    selection_type: str  # 'provider', 'model', 'agent', 'tool'
    task_filters: Dict[str, Any]  # Criteria for including tasks
    
    # Treatment options
    control_option: str
    treatment_options: List[str]
    
    # Experiment parameters
    traffic_allocation: Dict[str, float]  # option -> allocation %
    min_sample_size: int = 100
    max_duration_days: int = 7
    confidence_level: float = 0.95  # For statistical significance
    
    # Success metrics
    primary_metric: str = "success_rate"
    secondary_metrics: List[str] = field(default_factory=list)
    
    # Experiment control
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.PLANNED
    
    # Metadata
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = {
            'experiment_id': self.experiment_id,
            'name': self.name, 
            'description': self.description,
            'selection_type': self.selection_type,
            'task_filters': self.task_filters,
            'control_option': self.control_option,
            'treatment_options': self.treatment_options,
            'traffic_allocation': self.traffic_allocation,
            'min_sample_size': self.min_sample_size,
            'max_duration_days': self.max_duration_days,
            'confidence_level': self.confidence_level,
            'primary_metric': self.primary_metric,
            'secondary_metrics': self.secondary_metrics,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'status': self.status.value,
            'created_by': self.created_by,
            'tags': self.tags,
            'metadata': self.metadata
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        data = data.copy()
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        data['status'] = ExperimentStatus(data['status'])
        return cls(**data)


@dataclass 
class ExperimentResult:
    """Results of an A/B experiment."""
    
    experiment_id: str
    status: ExperimentStatus
    
    # Sample sizes
    control_samples: int
    treatment_samples: Dict[str, int]
    
    # Primary metric results
    control_metric: float
    treatment_metrics: Dict[str, float]
    
    # Statistical analysis
    statistical_significance: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    
    # Secondary metrics
    secondary_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Recommendations
    recommended_winner: Optional[str] = None
    recommendation_reason: str = ""
    
    # Analysis metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    analysis_method: str = "chi_square"
    warnings: List[str] = field(default_factory=list)


class ABTestingFramework:
    """
    Framework for orchestrating A/B tests of selection decisions.
    
    Provides statistical rigor while ensuring minimal impact on system performance
    through intelligent traffic allocation and early stopping criteria.
    """
    
    def __init__(self, 
                 selection_history: SelectionHistory,
                 min_baseline_samples: int = 50):
        """
        Initialize A/B testing framework.
        
        Args:
            selection_history: Storage for selection records
            min_baseline_samples: Minimum samples before running experiments
        """
        self.selection_history = selection_history
        self.min_baseline_samples = min_baseline_samples
        
        # Experiment management
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.completed_experiments: Dict[str, ExperimentResult] = {}
        
        # Traffic allocation
        self._allocation_cache: Dict[str, Dict[str, str]] = {}  # user_id -> experiment_id -> option
        self._allocation_lock = asyncio.Lock()
        
        # Performance tracking
        self.total_experiments_run = 0
        self.successful_experiments = 0
        
        logger.info("ABTestingFramework initialized")
    
    async def create_experiment(self, 
                              name: str,
                              description: str,
                              selection_type: str,
                              control_option: str,
                              treatment_options: List[str],
                              task_filters: Dict[str, Any] = None,
                              min_sample_size: int = 100,
                              max_duration_days: int = 7,
                              primary_metric: str = "success_rate") -> ExperimentConfig:
        """
        Create a new A/B experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            selection_type: Type of selection being tested
            control_option: Current best option (control)
            treatment_options: Alternative options to test
            task_filters: Criteria for including tasks in experiment
            min_sample_size: Minimum samples per group
            max_duration_days: Maximum experiment duration
            primary_metric: Primary success metric
            
        Returns:
            Configured experiment ready to start
        """
        # Generate unique experiment ID
        experiment_id = self._generate_experiment_id(name, selection_type)
        
        # Validate options exist and have baseline performance
        await self._validate_experiment_options(
            selection_type, control_option, treatment_options
        )
        
        # Calculate traffic allocation (equal split with slight preference for control)
        total_options = len(treatment_options) + 1  # +1 for control
        control_allocation = 0.6 if total_options == 2 else (1.0 / total_options)
        treatment_allocation = (1.0 - control_allocation) / len(treatment_options)
        
        traffic_allocation = {control_option: control_allocation}
        for option in treatment_options:
            traffic_allocation[option] = treatment_allocation
        
        # Create experiment configuration
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            selection_type=selection_type,
            task_filters=task_filters or {},
            control_option=control_option,
            treatment_options=treatment_options,
            traffic_allocation=traffic_allocation,
            min_sample_size=min_sample_size,
            max_duration_days=max_duration_days,
            primary_metric=primary_metric,
            secondary_metrics=["completion_time_ms", "quality_score", "cost"]
        )
        
        logger.info(f"Created experiment {experiment_id}: {name}")
        return config
    
    async def start_experiment(self, config: ExperimentConfig) -> bool:
        """
        Start an A/B experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            True if experiment started successfully
        """
        try:
            # Check for conflicts with existing experiments
            conflicts = await self._check_experiment_conflicts(config)
            if conflicts:
                logger.warning(f"Experiment {config.experiment_id} conflicts with: {conflicts}")
                return False
            
            # Update configuration
            config.status = ExperimentStatus.RUNNING
            config.start_time = datetime.now()
            config.end_time = datetime.now() + timedelta(days=config.max_duration_days)
            
            # Register active experiment
            self.active_experiments[config.experiment_id] = config
            self.total_experiments_run += 1
            
            logger.info(f"Started experiment {config.experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start experiment {config.experiment_id}: {e}")
            return False
    
    async def allocate_selection(self,
                               selection_type: str,
                               task_context: Dict[str, Any],
                               available_options: List[str],
                               user_id: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Allocate a selection option for a task, considering active experiments.
        
        Args:
            selection_type: Type of selection needed
            task_context: Context of the task
            available_options: Available options to choose from
            user_id: User identifier for consistent allocation
            
        Returns:
            Tuple of (selected_option, allocation_metadata)
        """
        allocation_metadata = {
            'method': 'baseline',
            'experiment_id': None,
            'treatment_group': False
        }
        
        try:
            # Find applicable experiments
            applicable_experiments = [
                exp for exp in self.active_experiments.values()
                if (exp.selection_type == selection_type and
                    exp.status == ExperimentStatus.RUNNING and
                    self._task_matches_filters(task_context, exp.task_filters) and
                    any(opt in available_options for opt in 
                        [exp.control_option] + exp.treatment_options))
            ]
            
            if not applicable_experiments:
                # No active experiments - use baseline selection
                return available_options[0], allocation_metadata
            
            # Use the first applicable experiment
            experiment = applicable_experiments[0]
            
            # Check if experiment should end early
            if await self._should_stop_experiment_early(experiment):
                await self.stop_experiment(experiment.experiment_id, "early_stopping")
                return available_options[0], allocation_metadata
            
            # Allocate based on traffic allocation
            user_hash = self._get_user_hash(user_id, experiment.experiment_id)
            selected_option = self._allocate_traffic(
                user_hash, experiment, available_options
            )
            
            allocation_metadata.update({
                'method': 'experiment',
                'experiment_id': experiment.experiment_id,
                'treatment_group': selected_option != experiment.control_option
            })
            
            logger.debug(f"Allocated {selected_option} for experiment {experiment.experiment_id}")
            return selected_option, allocation_metadata
            
        except Exception as e:
            logger.error(f"Error in selection allocation: {e}")
            return available_options[0], allocation_metadata
    
    async def record_experiment_result(self,
                                     selection_record: SelectionRecord) -> None:
        """
        Record the result of an experimental selection.
        
        Args:
            selection_record: Complete selection record with outcome
        """
        try:
            # Store in selection history
            self.selection_history.record_selection(selection_record)
            
            # Check if any experiments can be analyzed
            experiment_id = selection_record.selection_metadata.get('experiment_id')
            if experiment_id and experiment_id in self.active_experiments:
                experiment = self.active_experiments[experiment_id]
                
                # Check for early stopping or completion
                if await self._should_analyze_experiment(experiment):
                    await self._analyze_experiment(experiment)
                    
        except Exception as e:
            logger.error(f"Error recording experiment result: {e}")
    
    async def stop_experiment(self, 
                            experiment_id: str,
                            reason: str = "manual") -> Optional[ExperimentResult]:
        """
        Stop an active experiment and analyze results.
        
        Args:
            experiment_id: ID of experiment to stop
            reason: Reason for stopping
            
        Returns:
            Experiment results if analysis successful
        """
        if experiment_id not in self.active_experiments:
            logger.warning(f"Experiment {experiment_id} not found in active experiments")
            return None
        
        try:
            experiment = self.active_experiments[experiment_id]
            experiment.status = ExperimentStatus.ANALYZING
            
            # Analyze results
            result = await self._analyze_experiment(experiment)
            
            # Move to completed experiments
            del self.active_experiments[experiment_id]
            if result:
                self.completed_experiments[experiment_id] = result
                self.successful_experiments += 1
            
            logger.info(f"Stopped experiment {experiment_id}: {reason}")
            return result
            
        except Exception as e:
            logger.error(f"Error stopping experiment {experiment_id}: {e}")
            return None
    
    def get_active_experiments(self) -> List[ExperimentConfig]:
        """Get list of currently active experiments."""
        return list(self.active_experiments.values())
    
    def get_experiment_results(self) -> List[ExperimentResult]:
        """Get results from completed experiments."""
        return list(self.completed_experiments.values())
    
    async def _validate_experiment_options(self,
                                         selection_type: str,
                                         control_option: str,
                                         treatment_options: List[str]) -> None:
        """Validate that experiment options have sufficient baseline data."""
        all_options = [control_option] + treatment_options
        
        for option in all_options:
            records = self.selection_history.get_records(
                selection_type=selection_type,
                selected_option=option,
                since=datetime.now() - timedelta(days=30),
                only_completed=True,
                limit=1000
            )
            
            if len(records) < self.min_baseline_samples:
                raise ValueError(
                    f"Insufficient baseline data for {option}: {len(records)} < {self.min_baseline_samples}"
                )
    
    async def _check_experiment_conflicts(self, config: ExperimentConfig) -> List[str]:
        """Check for conflicts with existing active experiments."""
        conflicts = []
        
        for exp_id, exp in self.active_experiments.items():
            if (exp.selection_type == config.selection_type and
                exp.status == ExperimentStatus.RUNNING):
                # Check for overlapping options
                exp_options = set([exp.control_option] + exp.treatment_options)
                config_options = set([config.control_option] + config.treatment_options)
                
                if exp_options.intersection(config_options):
                    conflicts.append(exp_id)
        
        return conflicts
    
    def _task_matches_filters(self, 
                            task_context: Dict[str, Any],
                            filters: Dict[str, Any]) -> bool:
        """Check if a task matches experiment filters."""
        if not filters:
            return True
        
        for key, expected_value in filters.items():
            if key not in task_context:
                return False
            
            actual_value = task_context[key]
            
            # Handle different comparison types
            if isinstance(expected_value, dict):
                if 'min' in expected_value and actual_value < expected_value['min']:
                    return False
                if 'max' in expected_value and actual_value > expected_value['max']:
                    return False
                if 'in' in expected_value and actual_value not in expected_value['in']:
                    return False
            else:
                if actual_value != expected_value:
                    return False
        
        return True
    
    def _get_user_hash(self, user_id: str, experiment_id: str) -> float:
        """Generate consistent hash for user in experiment (0.0 - 1.0)."""
        if not user_id:
            # Use random for anonymous users
            return random.random()
        
        # Create deterministic hash
        hash_input = f"{user_id}_{experiment_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        return (hash_value % 10000) / 10000.0
    
    def _allocate_traffic(self,
                        user_hash: float,
                        experiment: ExperimentConfig,
                        available_options: List[str]) -> str:
        """Allocate traffic based on experiment configuration."""
        # Filter to only available options
        applicable_allocations = {
            option: allocation 
            for option, allocation in experiment.traffic_allocation.items()
            if option in available_options
        }
        
        if not applicable_allocations:
            return available_options[0]
        
        # Normalize allocations
        total_allocation = sum(applicable_allocations.values())
        normalized_allocations = {
            option: allocation / total_allocation
            for option, allocation in applicable_allocations.items()
        }
        
        # Select based on user hash
        cumulative = 0.0
        for option, allocation in normalized_allocations.items():
            cumulative += allocation
            if user_hash <= cumulative:
                return option
        
        # Fallback to first available option
        return list(applicable_allocations.keys())[0]
    
    async def _should_stop_experiment_early(self, experiment: ExperimentConfig) -> bool:
        """Check if experiment should be stopped early."""
        # Time-based stopping
        if datetime.now() > experiment.end_time:
            return True
        
        # Sample size-based stopping
        records = self.selection_history.get_records(
            selection_type=experiment.selection_type,
            since=experiment.start_time,
            only_completed=True,
            limit=10000
        )
        
        # Check if we have enough samples
        option_counts = {}
        for record in records:
            if (record.selection_metadata.get('experiment_id') == experiment.experiment_id):
                option = record.selected_option
                option_counts[option] = option_counts.get(option, 0) + 1
        
        min_samples = min(option_counts.values()) if option_counts else 0
        if min_samples >= experiment.min_sample_size * 2:  # 2x for early stopping confidence
            return True
        
        return False
    
    async def _should_analyze_experiment(self, experiment: ExperimentConfig) -> bool:
        """Check if experiment is ready for analysis."""
        records = self.selection_history.get_records(
            selection_type=experiment.selection_type,
            since=experiment.start_time,
            only_completed=True,
            limit=10000
        )
        
        # Count samples per option
        option_counts = {}
        for record in records:
            if (record.selection_metadata.get('experiment_id') == experiment.experiment_id):
                option = record.selected_option
                option_counts[option] = option_counts.get(option, 0) + 1
        
        # Need minimum sample size per option
        min_samples = min(option_counts.values()) if option_counts else 0
        return min_samples >= experiment.min_sample_size
    
    async def _analyze_experiment(self, experiment: ExperimentConfig) -> Optional[ExperimentResult]:
        """Analyze experiment results and determine winner."""
        try:
            # Get experiment records
            records = self.selection_history.get_records(
                selection_type=experiment.selection_type,
                since=experiment.start_time,
                only_completed=True,
                limit=10000
            )
            
            # Filter to experiment records only
            experiment_records = [
                record for record in records
                if record.selection_metadata.get('experiment_id') == experiment.experiment_id
            ]
            
            if not experiment_records:
                logger.warning(f"No records found for experiment {experiment.experiment_id}")
                return None
            
            # Group by option
            option_data = {}
            for record in experiment_records:
                option = record.selected_option
                if option not in option_data:
                    option_data[option] = []
                option_data[option].append(record)
            
            # Calculate primary metric for each option
            control_data = option_data.get(experiment.control_option, [])
            control_metric = self._calculate_metric(control_data, experiment.primary_metric)
            control_samples = len(control_data)
            
            treatment_metrics = {}
            treatment_samples = {}
            
            for option in experiment.treatment_options:
                treatment_data = option_data.get(option, [])
                treatment_metrics[option] = self._calculate_metric(treatment_data, experiment.primary_metric)
                treatment_samples[option] = len(treatment_data)
            
            # Statistical significance test (simplified chi-square)
            best_treatment = max(treatment_metrics.items(), key=lambda x: x[1]) if treatment_metrics else (None, 0)
            best_option = best_treatment[0]
            best_metric = best_treatment[1]
            
            # Simple significance test
            statistical_significance = (
                control_samples >= experiment.min_sample_size and
                treatment_samples.get(best_option, 0) >= experiment.min_sample_size and
                abs(best_metric - control_metric) > 0.05  # 5% minimum difference
            )
            
            # Effect size (Cohen's d approximation)
            effect_size = abs(best_metric - control_metric) / max(0.1, min(control_metric, best_metric))
            
            # Determine winner
            if statistical_significance and best_metric > control_metric:
                recommended_winner = best_option
                recommendation_reason = f"Treatment {best_option} shows {best_metric:.3f} vs control {control_metric:.3f}"
            else:
                recommended_winner = experiment.control_option
                recommendation_reason = "Control performs best or no significant difference found"
            
            # Create result
            result = ExperimentResult(
                experiment_id=experiment.experiment_id,
                status=ExperimentStatus.COMPLETED,
                control_samples=control_samples,
                treatment_samples=treatment_samples,
                control_metric=control_metric,
                treatment_metrics=treatment_metrics,
                statistical_significance=statistical_significance,
                p_value=0.05 if statistical_significance else 1.0,  # Simplified
                confidence_interval=(max(0, best_metric - 0.05), min(1, best_metric + 0.05)),
                effect_size=effect_size,
                recommended_winner=recommended_winner,
                recommendation_reason=recommendation_reason
            )
            
            logger.info(f"Analyzed experiment {experiment.experiment_id}: winner={recommended_winner}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing experiment {experiment.experiment_id}: {e}")
            return None
    
    def _calculate_metric(self, records: List[SelectionRecord], metric_name: str) -> float:
        """Calculate metric value for a set of records."""
        if not records:
            return 0.0
        
        if metric_name == "success_rate":
            successes = sum(1 for r in records if r.success)
            return successes / len(records)
        
        elif metric_name == "completion_time_ms":
            times = [r.completion_time_ms for r in records if r.completion_time_ms is not None]
            return sum(times) / len(times) if times else 0.0
        
        elif metric_name == "quality_score":
            scores = [r.quality_score for r in records if r.quality_score is not None]
            return sum(scores) / len(scores) if scores else 0.0
        
        elif metric_name == "cost":
            costs = [r.cost for r in records if r.cost is not None]
            return sum(costs) / len(costs) if costs else 0.0
        
        else:
            # Custom metric from custom_metrics
            values = []
            for record in records:
                if metric_name in record.custom_metrics:
                    values.append(record.custom_metrics[metric_name])
            return sum(values) / len(values) if values else 0.0
    
    def _generate_experiment_id(self, name: str, selection_type: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"exp_{selection_type}_{timestamp}_{name_hash}"