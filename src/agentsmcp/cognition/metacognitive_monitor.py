"""
Self-reflection and adjustment system for monitoring thinking quality and adapting strategies.

This module implements metacognitive monitoring that evaluates thinking effectiveness,
identifies improvement opportunities, and adapts strategies based on performance feedback.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .models import (
    ThinkingStep, ThinkingPhase, QualityAssessment, StrategyAdjustment,
    ExecutionResult
)
from .config import MetacognitiveConfig, DEFAULT_METACOGNITIVE_CONFIG

logger = logging.getLogger(__name__)


class InsufficientData(Exception):
    """Raised when insufficient data is available for analysis."""
    pass


class AnalysisFailure(Exception):
    """Raised when metacognitive analysis fails."""
    pass


class CalibrationError(Exception):
    """Raised when confidence calibration fails."""
    pass


@dataclass
class ThinkingAnalysisResult:
    """Result of analyzing a thinking process."""
    phase_quality_scores: Dict[ThinkingPhase, float]
    overall_effectiveness: float
    identified_issues: List[str]
    improvement_suggestions: List[str]
    confidence_accuracy: Optional[float] = None


@dataclass
class PerformanceHistory:
    """Historical performance data for learning."""
    thinking_outcomes: List[Dict[str, Any]] = field(default_factory=list)
    strategy_effectiveness: Dict[str, List[float]] = field(default_factory=dict)
    confidence_calibration_data: List[Tuple[float, bool]] = field(default_factory=list)
    adaptation_history: List[StrategyAdjustment] = field(default_factory=list)


class MetacognitiveMonitor:
    """
    Self-reflection and adjustment system for thinking quality monitoring.
    
    This monitor evaluates the effectiveness of thinking processes, identifies
    areas for improvement, and suggests adaptive changes to thinking strategies.
    """
    
    def __init__(self, config: Optional[MetacognitiveConfig] = None):
        """Initialize the metacognitive monitor."""
        self.config = config or DEFAULT_METACOGNITIVE_CONFIG
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.performance_history = PerformanceHistory()
        
        # Learning and adaptation state
        self._adaptation_counter = 0
        self._quality_threshold = 0.7
        self._confidence_calibration_samples = []
        
        # Monitoring metrics
        self._total_assessments = 0
        self._adaptations_made = 0
        self._calibration_improvements = 0
        
        self.logger.info("MetacognitiveMonitor initialized")
    
    async def assess_thinking_quality(
        self,
        thinking_trace: List[ThinkingStep],
        claimed_confidence: float
    ) -> QualityAssessment:
        """
        Assess the quality of a thinking process.
        
        Args:
            thinking_trace: Sequence of thinking steps to analyze
            claimed_confidence: The confidence claimed by the thinking process
            
        Returns:
            QualityAssessment with detailed quality evaluation
            
        Raises:
            InsufficientData: If thinking trace is too short for analysis
            AnalysisFailure: If quality assessment fails
        """
        if len(thinking_trace) < 2:
            raise InsufficientData("Thinking trace too short for quality assessment")
        
        try:
            self._total_assessments += 1
            
            # Analyze each phase of thinking
            phase_scores = await self._analyze_phase_quality(thinking_trace)
            
            # Calculate overall quality
            overall_quality = await self._calculate_overall_quality(
                phase_scores, thinking_trace
            )
            
            # Identify strengths and weaknesses
            strengths, weaknesses = await self._identify_strengths_weaknesses(
                phase_scores, thinking_trace
            )
            
            # Identify improvement areas
            improvement_areas = await self._identify_improvement_areas(
                phase_scores, thinking_trace, overall_quality
            )
            
            # Assess confidence calibration if execution data is available
            confidence_accuracy = await self._assess_confidence_calibration(
                claimed_confidence, overall_quality
            )
            
            assessment = QualityAssessment(
                overall_quality=overall_quality,
                phase_scores=phase_scores,
                strengths=strengths,
                weaknesses=weaknesses,
                improvement_areas=improvement_areas,
                confidence_accuracy=confidence_accuracy
            )
            
            # Record for learning
            await self._record_assessment(assessment, thinking_trace, claimed_confidence)
            
            self.logger.debug(
                f"Quality assessment complete: {overall_quality:.2f} overall, "
                f"{len(strengths)} strengths, {len(weaknesses)} weaknesses"
            )
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing thinking quality: {e}", exc_info=True)
            raise AnalysisFailure(f"Quality assessment failed: {e}")
    
    async def suggest_strategy_adjustments(
        self,
        thinking_trace: List[ThinkingStep],
        quality_assessment: QualityAssessment
    ) -> List[StrategyAdjustment]:
        """
        Suggest strategy adjustments based on thinking analysis.
        
        Args:
            thinking_trace: The thinking process to analyze
            quality_assessment: Quality assessment results
            
        Returns:
            List of suggested strategy adjustments
        """
        if not self.config.enable_strategy_adaptation:
            return []
        
        adjustments = []
        
        # Analyze phase-specific issues
        for phase, score in quality_assessment.phase_scores.items():
            if score < self.config.adaptation_sensitivity:
                phase_adjustments = await self._suggest_phase_adjustments(
                    phase, score, thinking_trace
                )
                adjustments.extend(phase_adjustments)
        
        # Global strategy adjustments
        if quality_assessment.overall_quality < self._quality_threshold:
            global_adjustments = await self._suggest_global_adjustments(
                quality_assessment, thinking_trace
            )
            adjustments.extend(global_adjustments)
        
        # Confidence calibration adjustments
        if (quality_assessment.confidence_accuracy and 
            abs(quality_assessment.confidence_accuracy) > 0.3):
            calibration_adjustments = await self._suggest_calibration_adjustments(
                quality_assessment.confidence_accuracy
            )
            adjustments.extend(calibration_adjustments)
        
        # Filter and prioritize adjustments
        prioritized_adjustments = await self._prioritize_adjustments(adjustments)
        
        # Record adjustments for learning
        for adjustment in prioritized_adjustments:
            self.performance_history.adaptation_history.append(adjustment)
        
        self._adaptations_made += len(prioritized_adjustments)
        
        return prioritized_adjustments
    
    async def update_performance_feedback(
        self,
        thinking_trace: List[ThinkingStep],
        execution_results: List[ExecutionResult],
        user_satisfaction: Optional[float] = None
    ):
        """
        Update performance tracking with execution feedback.
        
        Args:
            thinking_trace: The thinking process that was executed
            execution_results: Results from executing the planned tasks
            user_satisfaction: Optional user satisfaction rating (0.0-1.0)
        """
        if not self.config.collect_execution_feedback:
            return
        
        # Calculate execution success rate
        total_tasks = len(execution_results)
        successful_tasks = sum(1 for result in execution_results if result.success)
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # Calculate average execution time vs. estimates
        time_accuracy = await self._calculate_time_accuracy(execution_results)
        
        # Record thinking outcome
        outcome = {
            'timestamp': datetime.now(),
            'thinking_phases': [step.phase.value for step in thinking_trace],
            'execution_success_rate': success_rate,
            'time_accuracy': time_accuracy,
            'user_satisfaction': user_satisfaction,
            'thinking_duration_ms': sum(step.duration_ms or 0 for step in thinking_trace)
        }
        
        self.performance_history.thinking_outcomes.append(outcome)
        
        # Update confidence calibration data
        if len(thinking_trace) > 0:
            final_confidence = thinking_trace[-1].confidence
            was_successful = success_rate > 0.7  # Success threshold
            self.performance_history.confidence_calibration_data.append(
                (final_confidence, was_successful)
            )
        
        # Trigger adaptive learning if we have enough data
        if (len(self.performance_history.thinking_outcomes) >= 
            self.config.min_samples_for_adaptation):
            await self._trigger_adaptive_learning()
    
    async def _analyze_phase_quality(
        self,
        thinking_trace: List[ThinkingStep]
    ) -> Dict[ThinkingPhase, float]:
        """Analyze the quality of each thinking phase."""
        phase_scores = {}
        
        # Group steps by phase
        phases = {}
        for step in thinking_trace:
            if step.phase not in phases:
                phases[step.phase] = []
            phases[step.phase].append(step)
        
        # Analyze each phase
        for phase, steps in phases.items():
            if phase == ThinkingPhase.ANALYZE_REQUEST:
                score = await self._assess_analysis_quality(steps)
            elif phase == ThinkingPhase.EXPLORE_OPTIONS:
                score = await self._assess_exploration_quality(steps)
            elif phase == ThinkingPhase.EVALUATE_APPROACHES:
                score = await self._assess_evaluation_quality(steps)
            elif phase == ThinkingPhase.SELECT_STRATEGY:
                score = await self._assess_selection_quality(steps)
            elif phase == ThinkingPhase.DECOMPOSE_TASKS:
                score = await self._assess_decomposition_quality(steps)
            elif phase == ThinkingPhase.PLAN_EXECUTION:
                score = await self._assess_planning_quality(steps)
            elif phase == ThinkingPhase.REFLECT_ADJUST:
                score = await self._assess_reflection_quality(steps)
            else:
                score = 0.5  # Default score for unknown phases
            
            phase_scores[phase] = score
        
        return phase_scores
    
    async def _assess_analysis_quality(self, steps: List[ThinkingStep]) -> float:
        """Assess quality of request analysis phase."""
        if not steps:
            return 0.0
        
        score = 0.5  # Base score
        
        for step in steps:
            # Check for thoroughness indicators
            content_lower = step.content.lower()
            
            thoroughness_indicators = [
                'complexity', 'requirements', 'context', 'constraints',
                'stakeholders', 'objectives', 'scope'
            ]
            
            thoroughness_count = sum(
                1 for indicator in thoroughness_indicators 
                if indicator in content_lower
            )
            
            # Boost score for thoroughness
            score += min(0.3, thoroughness_count * 0.05)
            
            # Check content length (longer analysis often more thorough)
            if len(step.content) > 200:
                score += 0.1
            elif len(step.content) < 50:
                score -= 0.1
            
            # Consider confidence
            score += (step.confidence - 0.5) * 0.2
        
        return min(1.0, max(0.0, score / len(steps)))
    
    async def _assess_exploration_quality(self, steps: List[ThinkingStep]) -> float:
        """Assess quality of option exploration phase."""
        if not steps:
            return 0.0
        
        score = 0.5
        
        for step in steps:
            # Check for creativity and diversity indicators
            content = step.content.lower()
            
            creativity_indicators = [
                'alternative', 'different', 'various', 'multiple',
                'creative', 'innovative', 'novel', 'unique'
            ]
            
            creativity_count = sum(
                1 for indicator in creativity_indicators
                if indicator in content
            )
            
            score += min(0.3, creativity_count * 0.08)
            
            # Check for mention of specific number of options
            if any(num in content for num in ['2', '3', '4', '5', 'two', 'three', 'several']):
                score += 0.2
            
            # Confidence adjustment
            score += (step.confidence - 0.5) * 0.1
        
        return min(1.0, max(0.0, score / len(steps)))
    
    async def _assess_evaluation_quality(self, steps: List[ThinkingStep]) -> float:
        """Assess quality of approach evaluation phase."""
        if not steps:
            return 0.0
        
        score = 0.5
        
        for step in steps:
            content = step.content.lower()
            
            # Check for evaluation criteria
            evaluation_indicators = [
                'criteria', 'compare', 'evaluate', 'pros', 'cons',
                'trade-off', 'advantage', 'disadvantage', 'score', 'rank'
            ]
            
            evaluation_count = sum(
                1 for indicator in evaluation_indicators
                if indicator in content
            )
            
            score += min(0.4, evaluation_count * 0.06)
            
            # Check for systematic evaluation
            if 'systematic' in content or 'methodical' in content:
                score += 0.2
            
            # Confidence alignment
            score += (step.confidence - 0.5) * 0.15
        
        return min(1.0, max(0.0, score / len(steps)))
    
    async def _assess_selection_quality(self, steps: List[ThinkingStep]) -> float:
        """Assess quality of strategy selection phase."""
        if not steps:
            return 0.0
        
        score = 0.6  # Higher base score as selection should be more straightforward
        
        for step in steps:
            content = step.content.lower()
            
            # Check for clear selection rationale
            rationale_indicators = [
                'because', 'due to', 'since', 'reason', 'therefore',
                'selected', 'chosen', 'best', 'optimal'
            ]
            
            rationale_count = sum(
                1 for indicator in rationale_indicators
                if indicator in content
            )
            
            score += min(0.3, rationale_count * 0.1)
            
            # High confidence in selection is good
            if step.confidence > 0.8:
                score += 0.1
            elif step.confidence < 0.5:
                score -= 0.2
        
        return min(1.0, max(0.0, score / len(steps)))
    
    async def _assess_decomposition_quality(self, steps: List[ThinkingStep]) -> float:
        """Assess quality of task decomposition phase."""
        if not steps:
            return 0.0
        
        score = 0.5
        
        for step in steps:
            content = step.content.lower()
            
            # Check for decomposition indicators
            decomposition_indicators = [
                'subtask', 'breakdown', 'steps', 'phase', 'stage',
                'component', 'module', 'dependency', 'parallel'
            ]
            
            decomposition_count = sum(
                1 for indicator in decomposition_indicators
                if indicator in content
            )
            
            score += min(0.4, decomposition_count * 0.08)
            
            # Check for dependency analysis
            if 'dependency' in content or 'depends' in content:
                score += 0.15
            
            # Check for parallel execution consideration
            if 'parallel' in content or 'concurrent' in content:
                score += 0.1
            
            # Confidence adjustment
            score += (step.confidence - 0.5) * 0.1
        
        return min(1.0, max(0.0, score / len(steps)))
    
    async def _assess_planning_quality(self, steps: List[ThinkingStep]) -> float:
        """Assess quality of execution planning phase."""
        if not steps:
            return 0.0
        
        score = 0.5
        
        for step in steps:
            content = step.content.lower()
            
            # Check for planning indicators
            planning_indicators = [
                'schedule', 'timeline', 'resource', 'allocation',
                'order', 'sequence', 'priority', 'checkpoint'
            ]
            
            planning_count = sum(
                1 for indicator in planning_indicators
                if indicator in content
            )
            
            score += min(0.4, planning_count * 0.1)
            
            # Resource consideration is important
            if 'resource' in content:
                score += 0.15
            
            # Timeline/schedule consideration
            if any(word in content for word in ['schedule', 'timeline', 'duration']):
                score += 0.1
            
            # Confidence adjustment
            score += (step.confidence - 0.5) * 0.1
        
        return min(1.0, max(0.0, score / len(steps)))
    
    async def _assess_reflection_quality(self, steps: List[ThinkingStep]) -> float:
        """Assess quality of reflection and adjustment phase."""
        if not steps:
            return 0.0
        
        score = 0.6  # Reflection is inherently valuable
        
        for step in steps:
            content = step.content.lower()
            
            # Check for reflection indicators
            reflection_indicators = [
                'reflect', 'consider', 'review', 'assess', 'evaluate',
                'improve', 'adjust', 'refine', 'optimize'
            ]
            
            reflection_count = sum(
                1 for indicator in reflection_indicators
                if indicator in content
            )
            
            score += min(0.3, reflection_count * 0.08)
            
            # Self-critical analysis is valuable
            critical_indicators = ['weakness', 'problem', 'issue', 'concern', 'risk']
            if any(indicator in content for indicator in critical_indicators):
                score += 0.1
            
            # Improvement suggestions
            if 'improve' in content or 'better' in content:
                score += 0.1
        
        return min(1.0, max(0.0, score / len(steps)))
    
    async def _calculate_overall_quality(
        self,
        phase_scores: Dict[ThinkingPhase, float],
        thinking_trace: List[ThinkingStep]
    ) -> float:
        """Calculate overall thinking quality from phase scores."""
        if not phase_scores:
            return 0.5
        
        # Weight different phases by importance
        phase_weights = {
            ThinkingPhase.ANALYZE_REQUEST: 0.20,
            ThinkingPhase.EXPLORE_OPTIONS: 0.20,
            ThinkingPhase.EVALUATE_APPROACHES: 0.25,
            ThinkingPhase.SELECT_STRATEGY: 0.15,
            ThinkingPhase.DECOMPOSE_TASKS: 0.10,
            ThinkingPhase.PLAN_EXECUTION: 0.08,
            ThinkingPhase.REFLECT_ADJUST: 0.02
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for phase, score in phase_scores.items():
            weight = phase_weights.get(phase, 0.1)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        base_quality = weighted_score / total_weight
        
        # Adjust for thinking trace characteristics
        total_duration = sum(step.duration_ms or 0 for step in thinking_trace)
        
        # Bonus for reasonable thinking time
        if 1000 <= total_duration <= 10000:  # 1-10 seconds
            base_quality += 0.05
        elif total_duration > 30000:  # > 30 seconds might indicate inefficiency
            base_quality -= 0.05
        
        # Bonus for consistent confidence
        confidences = [step.confidence for step in thinking_trace if step.confidence > 0]
        if confidences:
            confidence_variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
            if confidence_variance < 0.1:  # Consistent confidence
                base_quality += 0.03
        
        return min(1.0, max(0.0, base_quality))
    
    async def _identify_strengths_weaknesses(
        self,
        phase_scores: Dict[ThinkingPhase, float],
        thinking_trace: List[ThinkingStep]
    ) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses in thinking process."""
        strengths = []
        weaknesses = []
        
        # Analyze phase performance
        for phase, score in phase_scores.items():
            phase_name = phase.value.replace('_', ' ').title()
            
            if score >= 0.8:
                strengths.append(f"Excellent {phase_name}")
            elif score >= 0.65:
                strengths.append(f"Good {phase_name}")
            elif score < 0.4:
                weaknesses.append(f"Poor {phase_name}")
            elif score < 0.55:
                weaknesses.append(f"Weak {phase_name}")
        
        # Analyze thinking trace patterns
        if len(thinking_trace) > 5:
            strengths.append("Thorough thinking process")
        elif len(thinking_trace) < 3:
            weaknesses.append("Limited thinking depth")
        
        # Confidence analysis
        confidences = [step.confidence for step in thinking_trace if step.confidence > 0]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence > 0.8:
                strengths.append("High confidence in decisions")
            elif avg_confidence < 0.5:
                weaknesses.append("Low confidence in decisions")
        
        return strengths, weaknesses
    
    async def _identify_improvement_areas(
        self,
        phase_scores: Dict[ThinkingPhase, float],
        thinking_trace: List[ThinkingStep],
        overall_quality: float
    ) -> List[str]:
        """Identify specific areas for improvement."""
        improvements = []
        
        # Phase-specific improvements
        for phase, score in phase_scores.items():
            if score < 0.6:
                phase_name = phase.value.replace('_', ' ')
                improvements.append(f"Improve {phase_name} thoroughness")
        
        # General improvements based on overall quality
        if overall_quality < 0.7:
            improvements.append("Increase overall thinking depth")
            improvements.append("Consider more alternative approaches")
        
        # Confidence calibration
        confidences = [step.confidence for step in thinking_trace]
        if confidences and max(confidences) - min(confidences) > 0.5:
            improvements.append("Improve confidence calibration consistency")
        
        # Time efficiency
        total_duration = sum(step.duration_ms or 0 for step in thinking_trace)
        if total_duration > 20000:  # > 20 seconds
            improvements.append("Optimize thinking efficiency")
        
        return improvements
    
    async def _assess_confidence_calibration(
        self,
        claimed_confidence: float,
        actual_quality: float
    ) -> Optional[float]:
        """Assess how well-calibrated the confidence estimate was."""
        if not self.config.enable_confidence_calibration:
            return None
        
        # Simple calibration: difference between claimed confidence and actual quality
        calibration_error = claimed_confidence - actual_quality
        
        # Store for learning
        self.performance_history.confidence_calibration_data.append(
            (claimed_confidence, actual_quality > 0.7)
        )
        
        return calibration_error
    
    async def _suggest_phase_adjustments(
        self,
        phase: ThinkingPhase,
        score: float,
        thinking_trace: List[ThinkingStep]
    ) -> List[StrategyAdjustment]:
        """Suggest adjustments for a specific thinking phase."""
        adjustments = []
        
        phase_name = phase.value.replace('_', ' ')
        
        if phase == ThinkingPhase.ANALYZE_REQUEST:
            adjustments.append(StrategyAdjustment(
                adjustment_type="analysis_depth",
                description=f"Increase depth of request analysis",
                impact_level="medium",
                implementation_steps=[
                    "Spend more time understanding requirements",
                    "Consider broader context and constraints",
                    "Identify key stakeholders and objectives"
                ],
                expected_improvement="Better foundation for subsequent thinking phases"
            ))
        
        elif phase == ThinkingPhase.EXPLORE_OPTIONS:
            adjustments.append(StrategyAdjustment(
                adjustment_type="option_generation",
                description="Generate more diverse approaches",
                impact_level="high",
                implementation_steps=[
                    "Use structured brainstorming techniques",
                    "Consider unconventional approaches",
                    "Ensure minimum number of options"
                ],
                expected_improvement="Better final solutions through wider exploration"
            ))
        
        # Add more phase-specific adjustments as needed...
        
        return adjustments
    
    async def _suggest_global_adjustments(
        self,
        assessment: QualityAssessment,
        thinking_trace: List[ThinkingStep]
    ) -> List[StrategyAdjustment]:
        """Suggest global strategy adjustments."""
        adjustments = []
        
        if assessment.overall_quality < 0.5:
            adjustments.append(StrategyAdjustment(
                adjustment_type="thinking_depth",
                description="Increase overall thinking thoroughness",
                impact_level="high",
                implementation_steps=[
                    "Allocate more time to each thinking phase",
                    "Require minimum quality thresholds",
                    "Add validation steps between phases"
                ],
                expected_improvement="Higher quality decisions and better outcomes"
            ))
        
        return adjustments
    
    async def _suggest_calibration_adjustments(
        self,
        calibration_error: float
    ) -> List[StrategyAdjustment]:
        """Suggest confidence calibration adjustments."""
        adjustments = []
        
        if calibration_error > 0.3:  # Over-confident
            adjustments.append(StrategyAdjustment(
                adjustment_type="confidence_calibration",
                description="Reduce confidence overestimation",
                impact_level="medium",
                implementation_steps=[
                    "Consider more potential failure modes",
                    "Increase uncertainty estimates",
                    "Seek disconfirming evidence"
                ],
                expected_improvement="Better calibrated confidence estimates"
            ))
        elif calibration_error < -0.3:  # Under-confident
            adjustments.append(StrategyAdjustment(
                adjustment_type="confidence_calibration",
                description="Increase confidence in good solutions",
                impact_level="medium",
                implementation_steps=[
                    "Recognize when solutions are strong",
                    "Trust thorough analysis results",
                    "Avoid excessive second-guessing"
                ],
                expected_improvement="More appropriate confidence in decisions"
            ))
        
        return adjustments
    
    async def _prioritize_adjustments(
        self,
        adjustments: List[StrategyAdjustment]
    ) -> List[StrategyAdjustment]:
        """Prioritize and filter adjustment suggestions."""
        # Sort by impact level
        impact_order = {"high": 3, "medium": 2, "low": 1}
        
        sorted_adjustments = sorted(
            adjustments,
            key=lambda adj: impact_order.get(adj.impact_level, 0),
            reverse=True
        )
        
        # Limit to most important adjustments
        return sorted_adjustments[:5]
    
    async def _record_assessment(
        self,
        assessment: QualityAssessment,
        thinking_trace: List[ThinkingStep],
        claimed_confidence: float
    ):
        """Record assessment for learning and adaptation."""
        outcome = {
            'timestamp': datetime.now(),
            'overall_quality': assessment.overall_quality,
            'claimed_confidence': claimed_confidence,
            'phase_scores': {phase.value: score for phase, score in assessment.phase_scores.items()},
            'num_strengths': len(assessment.strengths),
            'num_weaknesses': len(assessment.weaknesses),
            'thinking_steps': len(thinking_trace)
        }
        
        self.performance_history.thinking_outcomes.append(outcome)
        
        # Keep history bounded
        max_history = 1000
        if len(self.performance_history.thinking_outcomes) > max_history:
            self.performance_history.thinking_outcomes = \
                self.performance_history.thinking_outcomes[-max_history:]
    
    async def _calculate_time_accuracy(
        self,
        execution_results: List[ExecutionResult]
    ) -> float:
        """Calculate how accurate time estimates were."""
        if not execution_results:
            return 1.0
        
        # This would need actual time estimates from the planning phase
        # For now, return a placeholder
        return 0.8
    
    async def _trigger_adaptive_learning(self):
        """Trigger adaptive learning based on accumulated performance data."""
        if not self.config.enable_strategy_adaptation:
            return
        
        # Analyze performance trends
        recent_outcomes = self.performance_history.thinking_outcomes[-20:]
        
        if len(recent_outcomes) >= 10:
            recent_quality = sum(outcome['overall_quality'] for outcome in recent_outcomes) / len(recent_outcomes)
            
            # Adapt quality threshold based on performance
            if recent_quality > 0.8:
                self._quality_threshold = min(0.8, self._quality_threshold + 0.05)
            elif recent_quality < 0.6:
                self._quality_threshold = max(0.6, self._quality_threshold - 0.05)
        
        # Update confidence calibration
        if len(self.performance_history.confidence_calibration_data) >= 20:
            calibration_data = self.performance_history.confidence_calibration_data[-20:]
            await self._update_confidence_calibration(calibration_data)
    
    async def _update_confidence_calibration(
        self,
        calibration_data: List[Tuple[float, bool]]
    ):
        """Update confidence calibration based on feedback."""
        if not calibration_data:
            return
        
        # Simple calibration analysis
        over_confident_count = sum(
            1 for confidence, success in calibration_data
            if confidence > 0.8 and not success
        )
        
        under_confident_count = sum(
            1 for confidence, success in calibration_data
            if confidence < 0.6 and success
        )
        
        # Record calibration improvements
        if over_confident_count > len(calibration_data) * 0.3:
            self._calibration_improvements += 1
        elif under_confident_count > len(calibration_data) * 0.3:
            self._calibration_improvements += 1
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get metacognitive monitoring statistics."""
        return {
            "total_assessments": self._total_assessments,
            "adaptations_made": self._adaptations_made,
            "calibration_improvements": self._calibration_improvements,
            "current_quality_threshold": self._quality_threshold,
            "performance_history_size": len(self.performance_history.thinking_outcomes),
            "calibration_data_size": len(self.performance_history.confidence_calibration_data),
            "config": {
                "enable_phase_scoring": self.config.enable_phase_scoring,
                "enable_confidence_calibration": self.config.enable_confidence_calibration,
                "enable_strategy_adaptation": self.config.enable_strategy_adaptation,
                "learning_rate": self.config.learning_rate
            }
        }