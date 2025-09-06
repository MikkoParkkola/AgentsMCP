"""
Adaptive Selector - Main Intelligent Selection Interface

Provides the unified API for making optimized selection decisions by coordinating
A/B testing, multi-armed bandits, performance monitoring, and statistical analysis.
"""

import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import threading

from .selection_history import SelectionHistory, SelectionRecord, generate_selection_id
from .benchmark_tracker import BenchmarkTracker
from .ab_testing_framework import ABTestingFramework
from .selection_optimizer import SelectionOptimizer, OptimizationStrategy
from .performance_analyzer import PerformanceAnalyzer
from .experiment_manager import ExperimentManager, AutoExperimentConfig


logger = logging.getLogger(__name__)


class SelectionMode(Enum):
    """Selection modes available."""
    EXPLOITATION = "exploitation"  # Always pick the best known option
    EXPLORATION = "exploration"    # Always explore alternatives
    ADAPTIVE = "adaptive"          # Balance exploitation and exploration
    AB_TEST = "ab_test"           # Participate in A/B tests when available
    BANDIT = "bandit"             # Use multi-armed bandit algorithms


@dataclass
class SelectionRequest:
    """Request for intelligent selection."""
    
    selection_type: str  # 'provider', 'model', 'agent', 'tool'
    available_options: List[str]
    task_context: Dict[str, Any]
    
    # Request preferences
    mode: SelectionMode = SelectionMode.ADAPTIVE
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Constraints
    max_cost: Optional[float] = None
    min_performance: Optional[float] = None
    required_features: List[str] = None
    
    # Metadata
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.request_id is None:
            self.request_id = self._generate_request_id()
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.required_features is None:
            self.required_features = []
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        content = f"{self.selection_type}_{self.available_options}_{datetime.now().timestamp()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class SelectionResponse:
    """Response from intelligent selection."""
    
    selected_option: str
    confidence: float  # 0.0 - 1.0
    
    # Selection metadata
    selection_method: str
    exploration: bool
    experiment_id: Optional[str] = None
    
    # Performance context
    expected_success_rate: float = 0.0
    expected_completion_time_ms: float = 0.0
    expected_quality_score: float = 0.0
    expected_cost: float = 0.0
    
    # Decision rationale
    alternatives_considered: List[str] = None
    decision_factors: Dict[str, float] = None
    warnings: List[str] = None
    
    # Tracking
    request_id: str = ""
    selection_id: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.alternatives_considered is None:
            self.alternatives_considered = []
        if self.decision_factors is None:
            self.decision_factors = {}
        if self.warnings is None:
            self.warnings = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AdaptiveSelector:
    """
    Main intelligent selection interface for AgentsMCP.
    
    Coordinates all selection optimization components to provide the best
    possible selection decisions while continuously learning and improving.
    """
    
    def __init__(self,
                 selection_history: Optional[SelectionHistory] = None,
                 benchmark_tracker: Optional[BenchmarkTracker] = None,
                 ab_testing_framework: Optional[ABTestingFramework] = None,
                 selection_optimizer: Optional[SelectionOptimizer] = None,
                 performance_analyzer: Optional[PerformanceAnalyzer] = None,
                 experiment_manager: Optional[ExperimentManager] = None,
                 default_mode: SelectionMode = SelectionMode.ADAPTIVE):
        """
        Initialize adaptive selector.
        
        Args:
            selection_history: Historical selection data (will create if None)
            benchmark_tracker: Performance monitoring (will create if None)
            ab_testing_framework: A/B testing (will create if None)
            selection_optimizer: Bandit optimization (will create if None)
            performance_analyzer: Statistical analysis (will create if None)
            experiment_manager: Experiment coordination (will create if None)
            default_mode: Default selection mode
        """
        # Initialize components (create if not provided)
        self.selection_history = selection_history or SelectionHistory()
        self.benchmark_tracker = benchmark_tracker or BenchmarkTracker(self.selection_history)
        self.ab_testing_framework = ab_testing_framework or ABTestingFramework(self.selection_history)
        self.selection_optimizer = selection_optimizer or SelectionOptimizer(
            self.selection_history, self.benchmark_tracker
        )
        self.performance_analyzer = performance_analyzer or PerformanceAnalyzer(
            self.selection_history, self.benchmark_tracker
        )
        self.experiment_manager = experiment_manager or ExperimentManager(
            self.selection_history,
            self.benchmark_tracker,
            self.ab_testing_framework,
            self.selection_optimizer,
            self.performance_analyzer
        )
        
        self.default_mode = default_mode
        
        # State management
        self._initialized = False
        self._running = False
        self._init_lock = threading.RLock()
        
        # Performance tracking
        self.total_selections = 0
        self.successful_selections = 0
        self.exploration_selections = 0
        self.experiment_selections = 0
        
        # Caching
        self._prediction_cache: Dict[str, Tuple[datetime, Dict[str, float]]] = {}
        self._cache_ttl_seconds = 300  # 5 minute cache
        
        logger.info("AdaptiveSelector initialized")
    
    async def initialize(self):
        """Initialize the adaptive selector and all components."""
        if self._initialized:
            return
        
        with self._init_lock:
            if self._initialized:
                return
            
            try:
                # Start all components
                await self.benchmark_tracker.start()
                await self.experiment_manager.start()
                
                # Configure default auto-experimentation
                self._configure_default_auto_experiments()
                
                self._initialized = True
                self._running = True
                
                logger.info("AdaptiveSelector initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize AdaptiveSelector: {e}")
                raise
    
    async def shutdown(self):
        """Shutdown the adaptive selector and all components."""
        if not self._running:
            return
        
        self._running = False
        
        try:
            await self.experiment_manager.stop()
            await self.benchmark_tracker.stop()
            
            logger.info("AdaptiveSelector shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def select(self, request: Union[SelectionRequest, Dict[str, Any]]) -> SelectionResponse:
        """
        Make an intelligent selection decision.
        
        Args:
            request: Selection request (can be dict or SelectionRequest object)
            
        Returns:
            Selection response with chosen option and metadata
        """
        # Ensure initialization
        if not self._initialized:
            await self.initialize()
        
        # Convert dict to SelectionRequest if needed
        if isinstance(request, dict):
            request = SelectionRequest(**request)
        
        if not request.available_options:
            raise ValueError("No available options provided")
        
        try:
            # Update tracking
            self.total_selections += 1
            
            # Generate selection ID for tracking
            selection_id = generate_selection_id(
                request.selection_type, 
                request.task_context,
                request.timestamp
            )
            
            # Determine selection strategy
            strategy = self._determine_selection_strategy(request)
            
            # Make selection based on strategy
            if strategy == "ab_test":
                selected_option, metadata = await self._ab_test_selection(request)
                self.experiment_selections += 1
            elif strategy == "bandit":
                selected_option, metadata = await self._bandit_selection(request)
            elif strategy == "exploration":
                selected_option, metadata = await self._exploration_selection(request)
                self.exploration_selections += 1
            elif strategy == "baseline":
                selected_option, metadata = await self._baseline_selection(request)
            else:
                # Default to adaptive selection
                selected_option, metadata = await self._adaptive_selection(request)
            
            # Get performance predictions for the selected option
            predictions = await self._get_performance_predictions(
                request.selection_type, selected_option, request.task_context
            )
            
            # Build response
            response = SelectionResponse(
                selected_option=selected_option,
                confidence=metadata.get('confidence', 0.5),
                selection_method=metadata.get('method', strategy),
                exploration=metadata.get('exploration', False),
                experiment_id=metadata.get('experiment_id'),
                expected_success_rate=predictions.get('success_rate', 0.0),
                expected_completion_time_ms=predictions.get('completion_time_ms', 0.0),
                expected_quality_score=predictions.get('quality_score', 0.0),
                expected_cost=predictions.get('cost', 0.0),
                alternatives_considered=request.available_options.copy(),
                decision_factors=metadata.get('decision_factors', {}),
                warnings=metadata.get('warnings', []),
                request_id=request.request_id,
                selection_id=selection_id
            )
            
            # Record the selection (outcome will be updated later)
            await self._record_selection(request, response, metadata)
            
            logger.debug(f"Selected {selected_option} for {request.selection_type} using {strategy}")
            return response
            
        except Exception as e:
            logger.error(f"Error making selection: {e}")
            
            # Fallback to first available option
            fallback_response = SelectionResponse(
                selected_option=request.available_options[0],
                confidence=0.1,
                selection_method="fallback_error",
                exploration=False,
                warnings=[f"Selection error: {str(e)}"],
                request_id=request.request_id,
                selection_id=generate_selection_id(request.selection_type, request.task_context)
            )
            
            return fallback_response
    
    async def report_outcome(self,
                           selection_id: str,
                           success: bool,
                           completion_time_ms: Optional[int] = None,
                           quality_score: Optional[float] = None,
                           cost: Optional[float] = None,
                           error_message: Optional[str] = None,
                           user_feedback: Optional[int] = None,
                           custom_metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Report the outcome of a selection for learning.
        
        Args:
            selection_id: ID of the selection to update
            success: Whether the selection was successful
            completion_time_ms: Time taken to complete
            quality_score: Quality score (0.0 - 1.0)
            cost: Cost of the selection
            error_message: Error message if failed
            user_feedback: User feedback (-1, 0, 1)
            custom_metrics: Additional custom metrics
            
        Returns:
            True if outcome recorded successfully
        """
        try:
            # Update selection history
            updated = self.selection_history.update_outcome(
                selection_id=selection_id,
                success=success,
                completion_time_ms=completion_time_ms,
                quality_score=quality_score,
                cost=cost,
                error_message=error_message,
                user_feedback=user_feedback,
                custom_metrics=custom_metrics
            )
            
            if updated:
                # Update tracking
                if success:
                    self.successful_selections += 1
                
                # Get the full record for component updates
                records = self.selection_history.get_records(limit=1)
                if records and records[0].selection_id == selection_id:
                    record = records[0]
                    
                    # Update benchmark tracker
                    self.benchmark_tracker.record_selection_outcome(record)
                    
                    # Update optimizer
                    self.selection_optimizer.update_outcome(
                        record.selection_type,
                        record.selected_option,
                        record
                    )
                    
                    # Report to experiment framework if part of experiment
                    if record.selection_metadata.get('experiment_id'):
                        await self.ab_testing_framework.record_experiment_result(record)
                
                logger.debug(f"Recorded outcome for selection {selection_id}: success={success}")
                return True
            else:
                logger.warning(f"Failed to update outcome for selection {selection_id}")
                return False
        
        except Exception as e:
            logger.error(f"Error reporting outcome for {selection_id}: {e}")
            return False
    
    def get_performance_insights(self, 
                               selection_type: str = None,
                               days: int = 7) -> Dict[str, Any]:
        """Get performance insights and recommendations."""
        try:
            insights = {
                'timestamp': datetime.now().isoformat(),
                'period_days': days,
                'selection_type': selection_type
            }
            
            # Get benchmark insights
            if selection_type:
                metrics = self.benchmark_tracker.get_metrics(selection_type=selection_type)
                rankings = self.benchmark_tracker.get_rankings(selection_type)
                
                insights['options_tracked'] = len(metrics)
                insights['top_performer'] = rankings[0] if rankings else None
                insights['performance_degradations'] = len(
                    self.benchmark_tracker.detect_performance_degradation(selection_type)
                )
            
            # Get experiment insights
            experiment_insights = self.experiment_manager.get_performance_insights(
                selection_type=selection_type, days=days
            )
            insights.update(experiment_insights)
            
            # Get optimization stats
            optimization_stats = self.selection_optimizer.get_exploration_stats()
            insights['optimization'] = optimization_stats
            
            # Overall selector performance
            success_rate = (self.successful_selections / max(1, self.total_selections)) * 100
            exploration_rate = (self.exploration_selections / max(1, self.total_selections)) * 100
            experiment_rate = (self.experiment_selections / max(1, self.total_selections)) * 100
            
            insights['selector_performance'] = {
                'total_selections': self.total_selections,
                'success_rate': success_rate,
                'exploration_rate': exploration_rate,
                'experiment_participation_rate': experiment_rate
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting performance insights: {e}")
            return {'error': str(e)}
    
    def configure_selection_type(self,
                               selection_type: str,
                               optimization_strategy: OptimizationStrategy = None,
                               auto_experiment_config: AutoExperimentConfig = None):
        """Configure settings for a specific selection type."""
        try:
            # Configure optimization strategy
            if optimization_strategy:
                # This would require extending the optimizer to support per-type strategies
                logger.info(f"Configured optimization strategy for {selection_type}: {optimization_strategy.value}")
            
            # Configure auto-experimentation
            if auto_experiment_config:
                self.experiment_manager.configure_auto_experimentation(
                    selection_type, auto_experiment_config
                )
                logger.info(f"Configured auto-experimentation for {selection_type}")
        
        except Exception as e:
            logger.error(f"Error configuring {selection_type}: {e}")
    
    async def _determine_selection_strategy(self, request: SelectionRequest) -> str:
        """Determine the best selection strategy for this request."""
        # Check for active A/B experiments first
        active_experiments = self.ab_testing_framework.get_active_experiments()
        for exp in active_experiments:
            if (exp.selection_type == request.selection_type and
                any(opt in request.available_options for opt in 
                    [exp.control_option] + exp.treatment_options)):
                return "ab_test"
        
        # Apply request mode
        if request.mode == SelectionMode.EXPLOITATION:
            return "baseline"
        elif request.mode == SelectionMode.EXPLORATION:
            return "exploration"
        elif request.mode == SelectionMode.AB_TEST:
            return "ab_test"  # Will fallback to adaptive if no experiments
        elif request.mode == SelectionMode.BANDIT:
            return "bandit"
        else:
            # Adaptive mode - choose based on context
            return "adaptive"
    
    async def _ab_test_selection(self, request: SelectionRequest) -> Tuple[str, Dict[str, Any]]:
        """Make selection considering active A/B tests."""
        selected_option, allocation_metadata = await self.ab_testing_framework.allocate_selection(
            selection_type=request.selection_type,
            task_context=request.task_context,
            available_options=request.available_options,
            user_id=request.user_id
        )
        
        return selected_option, {
            'method': 'ab_test',
            'experiment_id': allocation_metadata.get('experiment_id'),
            'exploration': allocation_metadata.get('treatment_group', False),
            'confidence': 0.8 if allocation_metadata.get('experiment_id') else 0.5
        }
    
    async def _bandit_selection(self, request: SelectionRequest) -> Tuple[str, Dict[str, Any]]:
        """Make selection using multi-armed bandit algorithms."""
        selected_option, selection_metadata = self.selection_optimizer.select_option(
            selection_type=request.selection_type,
            available_options=request.available_options,
            task_context=request.task_context,
            user_id=request.user_id
        )
        
        return selected_option, {
            'method': 'bandit',
            'confidence': 0.7,
            'exploration': selection_metadata.get('exploration', False),
            'decision_factors': selection_metadata
        }
    
    async def _exploration_selection(self, request: SelectionRequest) -> Tuple[str, Dict[str, Any]]:
        """Make exploratory selection to try alternatives."""
        # Get current performance rankings
        rankings = self.benchmark_tracker.get_rankings(request.selection_type)
        
        # Filter to available options
        available_rankings = [
            (option, score) for option, score in rankings 
            if option in request.available_options
        ]
        
        if len(available_rankings) > 1:
            # Choose second-best or random from lower-ranked options
            import random
            exploration_options = [opt for opt, _ in available_rankings[1:]]
            selected_option = random.choice(exploration_options)
        else:
            # Random selection if no ranking data
            import random
            selected_option = random.choice(request.available_options)
        
        return selected_option, {
            'method': 'exploration',
            'confidence': 0.3,
            'exploration': True
        }
    
    async def _baseline_selection(self, request: SelectionRequest) -> Tuple[str, Dict[str, Any]]:
        """Make baseline selection (best known option)."""
        # Get best performer
        rankings = self.benchmark_tracker.get_rankings(request.selection_type)
        
        # Find best available option
        for option, score in rankings:
            if option in request.available_options:
                return option, {
                    'method': 'baseline',
                    'confidence': 0.9,
                    'exploration': False,
                    'decision_factors': {'performance_score': score}
                }
        
        # Fallback to first option
        return request.available_options[0], {
            'method': 'baseline_fallback',
            'confidence': 0.5,
            'exploration': False
        }
    
    async def _adaptive_selection(self, request: SelectionRequest) -> Tuple[str, Dict[str, Any]]:
        """Make adaptive selection balancing exploration and exploitation."""
        # Use the bandit optimizer in adaptive mode
        selected_option, selection_metadata = self.selection_optimizer.select_option(
            selection_type=request.selection_type,
            available_options=request.available_options,
            task_context=request.task_context,
            user_id=request.user_id
        )
        
        return selected_option, {
            'method': 'adaptive',
            'confidence': 0.75,
            'exploration': selection_metadata.get('exploration', False),
            'decision_factors': selection_metadata
        }
    
    async def _get_performance_predictions(self,
                                         selection_type: str,
                                         option: str,
                                         task_context: Dict[str, Any]) -> Dict[str, float]:
        """Get performance predictions for an option."""
        cache_key = f"{selection_type}:{option}:{hash(str(task_context))}"
        
        # Check cache
        if cache_key in self._prediction_cache:
            cache_time, predictions = self._prediction_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self._cache_ttl_seconds:
                return predictions
        
        try:
            # Get metrics from benchmark tracker
            metrics = self.benchmark_tracker.get_metrics(selection_type, option)
            key = (selection_type, option)
            
            if key in metrics:
                metric = metrics[key]
                predictions = {
                    'success_rate': metric.success_rate,
                    'completion_time_ms': metric.avg_completion_time_ms,
                    'quality_score': metric.avg_quality_score,
                    'cost': metric.avg_cost
                }
            else:
                # Default predictions for unknown options
                predictions = {
                    'success_rate': 0.5,
                    'completion_time_ms': 30000.0,  # 30 seconds
                    'quality_score': 0.5,
                    'cost': 0.01
                }
            
            # Cache predictions
            self._prediction_cache[cache_key] = (datetime.now(), predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return {
                'success_rate': 0.5,
                'completion_time_ms': 30000.0,
                'quality_score': 0.5,
                'cost': 0.01
            }
    
    async def _record_selection(self,
                              request: SelectionRequest,
                              response: SelectionResponse,
                              metadata: Dict[str, Any]):
        """Record the selection decision."""
        try:
            record = SelectionRecord(
                selection_id=response.selection_id,
                timestamp=response.timestamp,
                selection_type=request.selection_type,
                task_context=request.task_context,
                available_options=request.available_options,
                selected_option=response.selected_option,
                selection_method=response.selection_method,
                confidence_score=response.confidence,
                selection_metadata={
                    'request_id': request.request_id,
                    'experiment_id': response.experiment_id,
                    'exploration': response.exploration,
                    'user_id': request.user_id,
                    'session_id': request.session_id,
                    **metadata
                }
            )
            
            self.selection_history.record_selection(record)
            
        except Exception as e:
            logger.error(f"Error recording selection: {e}")
    
    def _configure_default_auto_experiments(self):
        """Configure default auto-experimentation for common selection types."""
        default_types = ['provider', 'model', 'agent', 'tool']
        
        for selection_type in default_types:
            config = AutoExperimentConfig(
                selection_type=selection_type,
                min_experiment_interval_hours=72,  # 3 days
                max_concurrent_experiments=1,
                min_baseline_samples=100,
                min_samples_per_arm=30,
                performance_degradation_threshold=0.15,  # 15%
                auto_approve_winner=True,
                min_confidence_for_approval=0.80,
                exploration_percentage=0.10,  # 10%
                enabled=True
            )
            
            self.experiment_manager.configure_auto_experimentation(selection_type, config)
        
        logger.info("Configured default auto-experimentation for standard selection types")