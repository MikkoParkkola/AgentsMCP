"""
Adaptive Model Selector Integration

Integrates the sophisticated A/B testing and continuous evaluation system
with the existing ModelSelector to provide intelligent model selection.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .selector import ModelSelector, TaskSpec, SelectionResult
from .models import ModelDB, Model
from ..selection.adaptive_selector import AdaptiveSelector, SelectionRequest, SelectionMode
from ..selection.selection_history import SelectionHistory
from ..selection.benchmark_tracker import BenchmarkTracker
from ..selection.ab_testing_framework import ABTestingFramework
from ..selection.selection_optimizer import SelectionOptimizer, OptimizationStrategy
from ..selection.performance_analyzer import PerformanceAnalyzer
from ..selection.experiment_manager import ExperimentManager


logger = logging.getLogger(__name__)


@dataclass
class AdaptiveSelectionResult:
    """Enhanced selection result with learning capabilities."""
    
    # Original selection result
    model: Model
    explanation: str
    score: float
    
    # Adaptive selection metadata
    selection_id: str
    confidence: float
    exploration: bool
    experiment_id: Optional[str]
    expected_performance: Dict[str, float]
    alternatives_considered: List[str]
    
    # Outcome tracking
    _outcome_reported: bool = False
    
    async def report_outcome(self,
                           success: bool,
                           completion_time_ms: Optional[int] = None,
                           quality_score: Optional[float] = None,
                           cost: Optional[float] = None,
                           error_message: Optional[str] = None,
                           user_feedback: Optional[int] = None) -> bool:
        """Report the outcome of using this model selection."""
        if self._outcome_reported:
            logger.warning(f"Outcome already reported for selection {self.selection_id}")
            return False
        
        try:
            # Get the adaptive selector instance (this requires access to the parent selector)
            # For now, we'll log the outcome - integration will handle the actual reporting
            logger.info(f"Model outcome: {self.model.name} success={success} time={completion_time_ms}ms")
            self._outcome_reported = True
            return True
        except Exception as e:
            logger.error(f"Error reporting outcome: {e}")
            return False


class AdaptiveModelSelector:
    """
    Enhanced model selector that integrates A/B testing, multi-armed bandits,
    and continuous learning for optimal model selection decisions.
    """
    
    def __init__(self, 
                 model_db: ModelDB,
                 enable_learning: bool = True,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.THOMPSON_SAMPLING,
                 exploration_rate: float = 0.1):
        """
        Initialize adaptive model selector.
        
        Args:
            model_db: Database of available models
            enable_learning: Whether to enable learning and optimization
            optimization_strategy: Multi-armed bandit strategy to use
            exploration_rate: Rate of exploration vs exploitation
        """
        self.model_db = model_db
        self.enable_learning = enable_learning
        
        # Initialize the original model selector
        self.base_selector = ModelSelector(model_db)
        
        # Initialize adaptive selection system if learning is enabled
        self.adaptive_selector: Optional[AdaptiveSelector] = None
        if enable_learning:
            self._initialize_adaptive_system(optimization_strategy, exploration_rate)
        
        # Performance tracking
        self.total_selections = 0
        self.learning_selections = 0
        self.baseline_selections = 0
        
        logger.info(f"AdaptiveModelSelector initialized (learning={'enabled' if enable_learning else 'disabled'})")
    
    def _initialize_adaptive_system(self, 
                                   optimization_strategy: OptimizationStrategy,
                                   exploration_rate: float):
        """Initialize the adaptive selection system."""
        try:
            # Initialize components
            selection_history = SelectionHistory()
            benchmark_tracker = BenchmarkTracker(selection_history)
            ab_testing_framework = ABTestingFramework(selection_history)
            selection_optimizer = SelectionOptimizer(
                selection_history, benchmark_tracker, 
                strategy=optimization_strategy,
                exploration_rate=exploration_rate
            )
            performance_analyzer = PerformanceAnalyzer(selection_history, benchmark_tracker)
            experiment_manager = ExperimentManager(
                selection_history, benchmark_tracker, ab_testing_framework,
                selection_optimizer, performance_analyzer
            )
            
            self.adaptive_selector = AdaptiveSelector(
                selection_history=selection_history,
                benchmark_tracker=benchmark_tracker,
                ab_testing_framework=ab_testing_framework,
                selection_optimizer=selection_optimizer,
                performance_analyzer=performance_analyzer,
                experiment_manager=experiment_manager,
                default_mode=SelectionMode.ADAPTIVE
            )
            
            logger.info("Adaptive selection system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize adaptive system: {e}")
            self.enable_learning = False
    
    async def select_model(self, 
                          spec: TaskSpec,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None) -> AdaptiveSelectionResult:
        """
        Select the optimal model for the given task specification.
        
        Args:
            spec: Task specification with requirements
            user_id: User identifier for consistent allocation  
            session_id: Session identifier
            
        Returns:
            Enhanced selection result with learning capabilities
        """
        self.total_selections += 1
        
        try:
            if self.enable_learning and self.adaptive_selector:
                return await self._adaptive_model_selection(spec, user_id, session_id)
            else:
                return await self._baseline_model_selection(spec)
                
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            # Fallback to baseline selection
            return await self._baseline_model_selection(spec)
    
    async def _adaptive_model_selection(self, 
                                      spec: TaskSpec,
                                      user_id: Optional[str],
                                      session_id: Optional[str]) -> AdaptiveSelectionResult:
        """Use adaptive selection system for model selection."""
        try:
            # Ensure adaptive selector is initialized
            if not self.adaptive_selector._initialized:
                await self.adaptive_selector.initialize()
            
            # Get available models that meet the specification constraints
            available_models = self._get_candidate_models(spec)
            if not available_models:
                raise ValueError("No models meet the specified requirements")
            
            # Create selection request
            selection_request = SelectionRequest(
                selection_type="model",
                available_options=[model.name for model in available_models],
                task_context={
                    "task_type": spec.task_type,
                    "max_cost_per_1k_tokens": spec.max_cost_per_1k_tokens,
                    "min_performance_tier": spec.min_performance_tier,
                    "required_context_length": spec.required_context_length,
                    "preferences": spec.preferences
                },
                mode=SelectionMode.ADAPTIVE,
                user_id=user_id,
                session_id=session_id,
                max_cost=spec.max_cost_per_1k_tokens,
                min_performance=spec.min_performance_tier
            )
            
            # Make adaptive selection
            selection_response = await self.adaptive_selector.select(selection_request)
            
            # Find the selected model
            selected_model = next(
                model for model in available_models 
                if model.name == selection_response.selected_option
            )
            
            # Generate explanation
            explanation = self._generate_explanation(selected_model, selection_response, spec)
            
            # Calculate composite score (for compatibility)
            score = self._calculate_composite_score(selected_model, spec, selection_response.confidence)
            
            self.learning_selections += 1
            
            return AdaptiveSelectionResult(
                model=selected_model,
                explanation=explanation,
                score=score,
                selection_id=selection_response.selection_id,
                confidence=selection_response.confidence,
                exploration=selection_response.exploration,
                experiment_id=selection_response.experiment_id,
                expected_performance={
                    "success_rate": selection_response.expected_success_rate,
                    "completion_time_ms": selection_response.expected_completion_time_ms,
                    "quality_score": selection_response.expected_quality_score,
                    "cost": selection_response.expected_cost
                },
                alternatives_considered=selection_response.alternatives_considered
            )
            
        except Exception as e:
            logger.error(f"Error in adaptive selection: {e}")
            # Fallback to baseline
            return await self._baseline_model_selection(spec)
    
    async def _baseline_model_selection(self, spec: TaskSpec) -> AdaptiveSelectionResult:
        """Fallback to baseline model selection."""
        try:
            # Use the original selector
            result = self.base_selector.select_model(spec)
            
            self.baseline_selections += 1
            
            # Convert to adaptive result format
            return AdaptiveSelectionResult(
                model=result.model,
                explanation=result.explanation,
                score=result.score,
                selection_id=f"baseline_{datetime.now().timestamp()}",
                confidence=0.7,  # Moderate confidence for baseline selection
                exploration=False,
                experiment_id=None,
                expected_performance={
                    "success_rate": 0.8,  # Default expectations
                    "completion_time_ms": 30000.0,
                    "quality_score": 0.7,
                    "cost": result.model.cost_per_input_token
                },
                alternatives_considered=[]
            )
            
        except Exception as e:
            logger.error(f"Error in baseline selection: {e}")
            raise
    
    def _get_candidate_models(self, spec: TaskSpec) -> List[Model]:
        """Get models that meet the basic specification requirements."""
        all_models = self.model_db.all_models()
        candidates = []
        
        for model in all_models:
            # Check basic constraints
            if spec.max_cost_per_1k_tokens and model.cost_per_input_token > spec.max_cost_per_1k_tokens:
                continue
            
            if model.performance_tier < spec.min_performance_tier:
                continue
            
            if (spec.required_context_length and 
                model.context_length and 
                model.context_length < spec.required_context_length):
                continue
            
            # Check capabilities
            model_capabilities = {c.lower() for c in model.categories}
            if spec.task_type.lower() not in model_capabilities and "general" not in model_capabilities:
                continue
            
            candidates.append(model)
        
        return candidates
    
    def _generate_explanation(self, 
                            model: Model,
                            selection_response,
                            spec: TaskSpec) -> str:
        """Generate explanation for the adaptive selection."""
        base_explanation = f"Selected {model.name} by {model.provider}"
        
        method_explanations = {
            "ab_test": "participating in A/B experiment",
            "bandit": "using multi-armed bandit optimization",
            "exploration": "exploring alternative options",
            "adaptive": "using adaptive learning algorithm",
            "baseline": "using performance-based ranking"
        }
        
        method_explanation = method_explanations.get(
            selection_response.selection_method, "using intelligent selection"
        )
        
        confidence_text = f"(confidence: {selection_response.confidence:.0%})"
        
        exploration_text = ""
        if selection_response.exploration:
            exploration_text = " [exploration]"
        
        return f"{base_explanation} {method_explanation} {confidence_text}{exploration_text}"
    
    def _calculate_composite_score(self, 
                                 model: Model, 
                                 spec: TaskSpec, 
                                 confidence: float) -> float:
        """Calculate composite score for compatibility."""
        # Use the base selector's scoring method as a starting point
        base_result = self.base_selector.select_model(spec)
        base_score = base_result.score if base_result.model.name == model.name else 0.0
        
        # Adjust based on confidence from adaptive system
        adaptive_bonus = confidence * 2.0  # Up to 2 point bonus
        
        return base_score + adaptive_bonus
    
    async def report_outcome(self,
                           selection_id: str,
                           success: bool,
                           completion_time_ms: Optional[int] = None,
                           quality_score: Optional[float] = None,
                           cost: Optional[float] = None,
                           error_message: Optional[str] = None,
                           user_feedback: Optional[int] = None) -> bool:
        """
        Report the outcome of a model selection for learning.
        
        Args:
            selection_id: ID from the AdaptiveSelectionResult
            success: Whether the model selection was successful
            completion_time_ms: Time taken to complete the task
            quality_score: Quality score (0.0 - 1.0)
            cost: Actual cost incurred
            error_message: Error message if failed
            user_feedback: User feedback (-1, 0, 1)
            
        Returns:
            True if outcome was recorded successfully
        """
        if not self.enable_learning or not self.adaptive_selector:
            logger.debug(f"Learning disabled, ignoring outcome for {selection_id}")
            return True
        
        try:
            return await self.adaptive_selector.report_outcome(
                selection_id=selection_id,
                success=success,
                completion_time_ms=completion_time_ms,
                quality_score=quality_score,
                cost=cost,
                error_message=error_message,
                user_feedback=user_feedback
            )
        except Exception as e:
            logger.error(f"Error reporting outcome: {e}")
            return False
    
    def get_performance_insights(self, days: int = 7) -> Dict[str, Any]:
        """Get performance insights and recommendations."""
        insights = {
            'selector_stats': {
                'total_selections': self.total_selections,
                'learning_selections': self.learning_selections,
                'baseline_selections': self.baseline_selections,
                'learning_enabled': self.enable_learning
            }
        }
        
        if self.enable_learning and self.adaptive_selector:
            try:
                adaptive_insights = self.adaptive_selector.get_performance_insights(
                    selection_type="model", days=days
                )
                insights.update(adaptive_insights)
            except Exception as e:
                logger.error(f"Error getting adaptive insights: {e}")
                insights['error'] = str(e)
        
        return insights
    
    def configure_experimentation(self,
                                auto_experiment: bool = True,
                                experiment_interval_hours: int = 48,
                                min_samples_for_experiment: int = 100):
        """Configure automatic experimentation settings."""
        if not self.enable_learning or not self.adaptive_selector:
            logger.warning("Cannot configure experimentation - learning is disabled")
            return
        
        try:
            from ..selection.experiment_manager import AutoExperimentConfig
            
            config = AutoExperimentConfig(
                selection_type="model",
                min_experiment_interval_hours=experiment_interval_hours,
                min_baseline_samples=min_samples_for_experiment,
                auto_approve_winner=auto_experiment,
                enabled=auto_experiment
            )
            
            self.adaptive_selector.configure_selection_type("model", auto_experiment_config=config)
            logger.info(f"Configured model experimentation: auto={auto_experiment}, interval={experiment_interval_hours}h")
            
        except Exception as e:
            logger.error(f"Error configuring experimentation: {e}")
    
    async def shutdown(self):
        """Shutdown the adaptive selector and clean up resources."""
        if self.adaptive_selector:
            try:
                await self.adaptive_selector.shutdown()
                logger.info("AdaptiveModelSelector shutdown complete")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")


# Convenience functions for easy integration
def create_adaptive_model_selector(model_db: ModelDB, 
                                 enable_learning: bool = True) -> AdaptiveModelSelector:
    """
    Create an adaptive model selector with sensible defaults.
    
    Args:
        model_db: Model database
        enable_learning: Whether to enable learning and optimization
        
    Returns:
        Configured adaptive model selector
    """
    return AdaptiveModelSelector(
        model_db=model_db,
        enable_learning=enable_learning,
        optimization_strategy=OptimizationStrategy.THOMPSON_SAMPLING,
        exploration_rate=0.1
    )


async def demo_adaptive_selection():
    """Demo of adaptive model selection capabilities."""
    from .models import ModelDB
    
    # Initialize with default model database
    model_db = ModelDB()
    selector = create_adaptive_model_selector(model_db, enable_learning=True)
    
    try:
        print("=== Adaptive Model Selection Demo ===\n")
        
        # Configure experimentation
        selector.configure_experimentation(
            auto_experiment=True,
            experiment_interval_hours=24,
            min_samples_for_experiment=50
        )
        
        # Example selections
        tasks = [
            TaskSpec(task_type="coding", max_cost_per_1k_tokens=2.0, min_performance_tier=3),
            TaskSpec(task_type="reasoning", max_cost_per_1k_tokens=5.0, min_performance_tier=4),
            TaskSpec(task_type="general", min_performance_tier=2)
        ]
        
        for i, task in enumerate(tasks):
            print(f"Task {i+1}: {task.task_type} (tier {task.min_performance_tier}+)")
            
            result = await selector.select_model(task, user_id=f"user_{i}")
            
            print(f"  Selected: {result.model.name}")
            print(f"  Explanation: {result.explanation}")
            print(f"  Confidence: {result.confidence:.0%}")
            print(f"  Expected success rate: {result.expected_performance['success_rate']:.0%}")
            print(f"  Exploration: {'Yes' if result.exploration else 'No'}")
            if result.experiment_id:
                print(f"  Experiment: {result.experiment_id}")
            print()
            
            # Simulate outcome reporting
            import random
            success = random.choice([True, True, True, False])  # 75% success rate
            completion_time = random.randint(5000, 45000)  # 5-45 seconds
            quality = random.uniform(0.6, 1.0) if success else random.uniform(0.0, 0.5)
            
            await result.report_outcome(
                success=success,
                completion_time_ms=completion_time,
                quality_score=quality
            )
        
        # Get performance insights
        insights = selector.get_performance_insights(days=1)
        print("Performance Insights:")
        print(f"  Total selections: {insights['selector_stats']['total_selections']}")
        print(f"  Learning selections: {insights['selector_stats']['learning_selections']}")
        print(f"  Learning enabled: {insights['selector_stats']['learning_enabled']}")
        
    finally:
        await selector.shutdown()


if __name__ == "__main__":
    asyncio.run(demo_adaptive_selection())