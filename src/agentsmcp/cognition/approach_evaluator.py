"""
Multi-option evaluation and selection system for ranking different approaches.

This module implements comprehensive evaluation of approaches based on multiple
criteria with weighted scoring and detailed rationale generation.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from .models import (
    Approach, RankedApproach, EvaluationCriteria, EvaluationScore, 
    EvaluationCriterion
)
from .config import ApproachEvaluatorConfig, DEFAULT_EVALUATOR_CONFIG

logger = logging.getLogger(__name__)


class NoApproachesProvided(Exception):
    """Raised when no approaches are provided for evaluation."""
    pass


class InvalidCriteria(Exception):
    """Raised when evaluation criteria are invalid."""
    pass


class EvaluationFailure(Exception):
    """Raised when approach evaluation fails."""
    pass


@dataclass
class EvaluationContext:
    """Context for approach evaluation."""
    request: str = ""
    user_preferences: Dict[str, float] = None
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.user_preferences is None:
            self.user_preferences = {}
        if self.constraints is None:
            self.constraints = {}


class ApproachEvaluator:
    """
    Multi-option evaluation and selection system for ranking approaches.
    
    This evaluator uses weighted scoring across multiple criteria to rank
    approaches and provide detailed rationale for selections.
    """
    
    def __init__(self, config: Optional[ApproachEvaluatorConfig] = None):
        """Initialize the approach evaluator."""
        self.config = config or DEFAULT_EVALUATOR_CONFIG
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Evaluation metrics
        self._total_evaluations = 0
        self._evaluation_time_sum = 0
        
        # Scoring functions for each criterion
        self._scoring_functions = {
            EvaluationCriterion.FEASIBILITY: self._score_feasibility,
            EvaluationCriterion.PERFORMANCE: self._score_performance,
            EvaluationCriterion.RELIABILITY: self._score_reliability,
            EvaluationCriterion.MAINTAINABILITY: self._score_maintainability,
            EvaluationCriterion.SECURITY: self._score_security,
            EvaluationCriterion.COST: self._score_cost,
            EvaluationCriterion.TIME_TO_COMPLETION: self._score_time_to_completion,
            EvaluationCriterion.RISK_LEVEL: self._score_risk_level
        }
        
        self.logger.info("ApproachEvaluator initialized")
    
    async def evaluate_approaches(
        self,
        approaches: List[Approach],
        criteria: EvaluationCriteria,
        context: Optional[EvaluationContext] = None
    ) -> List[RankedApproach]:
        """
        Evaluate and rank approaches based on criteria.
        
        Args:
            approaches: List of approaches to evaluate
            criteria: Evaluation criteria with weights
            context: Optional evaluation context
            
        Returns:
            List of ranked approaches sorted by score (highest first)
            
        Raises:
            NoApproachesProvided: If approaches list is empty
            InvalidCriteria: If criteria are invalid
            EvaluationFailure: If evaluation process fails
        """
        if not approaches:
            raise NoApproachesProvided("No approaches provided for evaluation")
        
        if not criteria.criteria and not criteria.custom_criteria:
            raise InvalidCriteria("No evaluation criteria provided")
        
        context = context or EvaluationContext()
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Validate and normalize criteria
            await self._validate_criteria(criteria)
            criteria.normalize_weights()
            
            # Evaluate all approaches
            if self.config.parallel_evaluation and len(approaches) > 1:
                evaluation_tasks = [
                    self._evaluate_single_approach(approach, criteria, context)
                    for approach in approaches
                ]
                evaluation_results = await asyncio.gather(*evaluation_tasks)
            else:
                evaluation_results = []
                for approach in approaches:
                    result = await self._evaluate_single_approach(approach, criteria, context)
                    evaluation_results.append(result)
            
            # Create ranked approaches
            ranked_approaches = []
            for i, (approach, scores, total_score) in enumerate(evaluation_results):
                selection_rationale = await self._generate_selection_rationale(
                    approach, scores, total_score, criteria
                )
                
                ranked_approach = RankedApproach(
                    approach=approach,
                    total_score=total_score,
                    individual_scores=scores,
                    rank=i + 1,  # Will be reordered after sorting
                    selection_rationale=selection_rationale
                )
                ranked_approaches.append(ranked_approach)
            
            # Sort by total score (highest first)
            ranked_approaches.sort(key=lambda ra: ra.total_score, reverse=True)
            
            # Update ranks
            for i, ranked_approach in enumerate(ranked_approaches):
                ranked_approach.rank = i + 1
            
            # Record metrics
            evaluation_time = asyncio.get_event_loop().time() - start_time
            self._total_evaluations += 1
            self._evaluation_time_sum += evaluation_time
            
            self.logger.info(
                f"Evaluated {len(approaches)} approaches in {evaluation_time*1000:.1f}ms. "
                f"Top approach: {ranked_approaches[0].approach.name} "
                f"(score: {ranked_approaches[0].total_score:.3f})"
            )
            
            return ranked_approaches
            
        except Exception as e:
            self.logger.error(f"Error evaluating approaches: {e}", exc_info=True)
            raise EvaluationFailure(f"Approach evaluation failed: {e}")
    
    async def _evaluate_single_approach(
        self,
        approach: Approach,
        criteria: EvaluationCriteria,
        context: EvaluationContext
    ) -> tuple[Approach, Dict[EvaluationCriterion, EvaluationScore], float]:
        """Evaluate a single approach against all criteria."""
        individual_scores = {}
        
        # Evaluate against standard criteria
        for criterion, weight in criteria.criteria.items():
            if criterion in self._scoring_functions:
                score = await self._scoring_functions[criterion](approach, context)
                individual_scores[criterion] = EvaluationScore(
                    criterion=criterion,
                    score=score,
                    weight=weight,
                    rationale=self._get_scoring_rationale(criterion, score, approach)
                )
        
        # Calculate total weighted score
        total_score = sum(
            eval_score.weighted_score 
            for eval_score in individual_scores.values()
        )
        
        # Normalize total score if using weighted scoring
        if self.config.use_weighted_scoring and self.config.normalize_scores:
            total_weight = sum(criteria.criteria.values())
            if total_weight > 0:
                total_score = total_score / total_weight
        
        return approach, individual_scores, total_score
    
    async def _score_feasibility(self, approach: Approach, context: EvaluationContext) -> float:
        """Score approach feasibility (0.0 to 1.0)."""
        score = 0.7  # Base feasibility score
        
        # Adjust based on approach characteristics
        if approach.steps:
            # More detailed approaches are often more feasible
            if len(approach.steps) >= 3:
                score += 0.1
            if len(approach.steps) > 10:
                score -= 0.1  # Too many steps may be complex
        
        # Consider risks
        if approach.risks:
            risk_penalty = len(approach.risks) * 0.05
            score = max(0.1, score - risk_penalty)
        
        # Consider estimated effort
        if approach.estimated_effort:
            if approach.estimated_effort > 10:  # Arbitrary high effort threshold
                score -= 0.2
            elif approach.estimated_effort < 2:
                score += 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _score_performance(self, approach: Approach, context: EvaluationContext) -> float:
        """Score expected performance (0.0 to 1.0)."""
        score = 0.6  # Base performance score
        
        # Look for performance indicators in approach
        performance_keywords = ['optimize', 'fast', 'efficient', 'cache', 'parallel', 'async']
        approach_text = f"{approach.name} {approach.description} {' '.join(approach.steps)}".lower()
        
        performance_mentions = sum(1 for keyword in performance_keywords if keyword in approach_text)
        score += performance_mentions * 0.1
        
        # Penalty for approaches that might be slower
        slow_keywords = ['sequential', 'synchronous', 'blocking', 'single-thread']
        slow_mentions = sum(1 for keyword in slow_keywords if keyword in approach_text)
        score -= slow_mentions * 0.15
        
        return min(1.0, max(0.0, score))
    
    async def _score_reliability(self, approach: Approach, context: EvaluationContext) -> float:
        """Score approach reliability (0.0 to 1.0)."""
        score = 0.7  # Base reliability score
        
        # Look for reliability indicators
        reliable_keywords = ['test', 'validate', 'verify', 'backup', 'rollback', 'monitor', 'recover']
        approach_text = f"{approach.name} {approach.description} {' '.join(approach.steps)}".lower()
        
        reliability_mentions = sum(1 for keyword in reliable_keywords if keyword in approach_text)
        score += reliability_mentions * 0.08
        
        # Consider number of steps (more steps = more failure points)
        if len(approach.steps) > 7:
            score -= 0.1
        elif len(approach.steps) < 3:
            score -= 0.05  # Too simple might miss important steps
        
        # Risk assessment
        if approach.risks:
            high_risk_keywords = ['data loss', 'security', 'crash', 'failure', 'corruption']
            high_risk_count = sum(
                1 for risk in approach.risks 
                for keyword in high_risk_keywords 
                if keyword in risk.lower()
            )
            score -= high_risk_count * 0.2
        
        return min(1.0, max(0.0, score))
    
    async def _score_maintainability(self, approach: Approach, context: EvaluationContext) -> float:
        """Score approach maintainability (0.0 to 1.0)."""
        score = 0.6  # Base maintainability score
        
        # Look for maintainability indicators
        maintainable_keywords = ['modular', 'clean', 'documented', 'standard', 'pattern', 'reusable']
        approach_text = f"{approach.name} {approach.description} {' '.join(approach.steps)}".lower()
        
        maintainability_mentions = sum(1 for keyword in maintainable_keywords if keyword in approach_text)
        score += maintainability_mentions * 0.1
        
        # Well-structured approaches are more maintainable
        if approach.steps and len(approach.steps) >= 3:
            score += 0.1
        
        # Consider complexity
        if approach.estimated_effort and approach.estimated_effort > 8:
            score -= 0.1  # Complex approaches may be harder to maintain
        
        return min(1.0, max(0.0, score))
    
    async def _score_security(self, approach: Approach, context: EvaluationContext) -> float:
        """Score approach security (0.0 to 1.0)."""
        score = 0.6  # Base security score
        
        # Look for security indicators
        security_keywords = ['secure', 'encrypt', 'authenticate', 'authorize', 'validate', 'sanitize']
        approach_text = f"{approach.name} {approach.description} {' '.join(approach.steps)}".lower()
        
        security_mentions = sum(1 for keyword in security_keywords if keyword in approach_text)
        score += security_mentions * 0.12
        
        # Check for security risks
        if approach.risks:
            security_risk_keywords = ['vulnerability', 'exploit', 'breach', 'leak', 'attack']
            security_risk_count = sum(
                1 for risk in approach.risks
                for keyword in security_risk_keywords
                if keyword in risk.lower()
            )
            score -= security_risk_count * 0.3
        
        return min(1.0, max(0.0, score))
    
    async def _score_cost(self, approach: Approach, context: EvaluationContext) -> float:
        """Score approach cost-effectiveness (0.0 to 1.0, higher = lower cost)."""
        score = 0.7  # Base cost score
        
        # Consider estimated effort as a cost proxy
        if approach.estimated_effort:
            if approach.estimated_effort <= 2:
                score = 0.9
            elif approach.estimated_effort <= 5:
                score = 0.7
            elif approach.estimated_effort <= 10:
                score = 0.5
            else:
                score = 0.3
        
        # Look for cost indicators
        cost_keywords = ['expensive', 'resource-intensive', 'complex', 'time-consuming']
        low_cost_keywords = ['simple', 'quick', 'minimal', 'lightweight', 'efficient']
        
        approach_text = f"{approach.name} {approach.description} {' '.join(approach.steps)}".lower()
        
        high_cost_mentions = sum(1 for keyword in cost_keywords if keyword in approach_text)
        low_cost_mentions = sum(1 for keyword in low_cost_keywords if keyword in approach_text)
        
        score -= high_cost_mentions * 0.15
        score += low_cost_mentions * 0.1
        
        return min(1.0, max(0.0, score))
    
    async def _score_time_to_completion(self, approach: Approach, context: EvaluationContext) -> float:
        """Score time to completion (0.0 to 1.0, higher = faster completion)."""
        score = 0.6  # Base time score
        
        # Consider number of steps as time proxy
        if approach.steps:
            if len(approach.steps) <= 3:
                score = 0.9
            elif len(approach.steps) <= 6:
                score = 0.7
            elif len(approach.steps) <= 10:
                score = 0.5
            else:
                score = 0.3
        
        # Look for time indicators
        fast_keywords = ['quick', 'rapid', 'immediate', 'fast', 'parallel']
        slow_keywords = ['thorough', 'comprehensive', 'detailed', 'complete']
        
        approach_text = f"{approach.name} {approach.description} {' '.join(approach.steps)}".lower()
        
        fast_mentions = sum(1 for keyword in fast_keywords if keyword in approach_text)
        slow_mentions = sum(1 for keyword in slow_keywords if keyword in approach_text)
        
        score += fast_mentions * 0.1
        score -= slow_mentions * 0.05  # Thorough is good, but slower
        
        return min(1.0, max(0.0, score))
    
    async def _score_risk_level(self, approach: Approach, context: EvaluationContext) -> float:
        """Score risk level (0.0 to 1.0, higher = lower risk)."""
        score = 0.8  # Base low-risk score
        
        # Assess risks directly
        if approach.risks:
            risk_penalty = len(approach.risks) * 0.1
            
            # Weight risks by severity keywords
            high_severity_keywords = ['critical', 'severe', 'major', 'catastrophic', 'data loss']
            medium_severity_keywords = ['moderate', 'significant', 'important']
            
            high_severity_count = sum(
                1 for risk in approach.risks
                for keyword in high_severity_keywords
                if keyword in risk.lower()
            )
            medium_severity_count = sum(
                1 for risk in approach.risks
                for keyword in medium_severity_keywords  
                if keyword in risk.lower()
            )
            
            score -= high_severity_count * 0.3
            score -= medium_severity_count * 0.15
            score -= risk_penalty
        
        # Look for risk mitigation in approach
        mitigation_keywords = ['backup', 'rollback', 'recovery', 'fallback', 'safety', 'test']
        approach_text = f"{approach.name} {approach.description} {' '.join(approach.steps)}".lower()
        
        mitigation_mentions = sum(1 for keyword in mitigation_keywords if keyword in approach_text)
        score += mitigation_mentions * 0.05
        
        return min(1.0, max(0.0, score))
    
    def _get_scoring_rationale(
        self,
        criterion: EvaluationCriterion,
        score: float,
        approach: Approach
    ) -> str:
        """Generate rationale for a specific criterion score."""
        rationales = {
            EvaluationCriterion.FEASIBILITY: {
                (0.8, 1.0): "Highly feasible with clear implementation path",
                (0.6, 0.8): "Feasible with reasonable complexity",
                (0.4, 0.6): "Moderately feasible, may require additional planning",
                (0.0, 0.4): "Low feasibility, significant challenges expected"
            },
            EvaluationCriterion.PERFORMANCE: {
                (0.8, 1.0): "Expected excellent performance characteristics", 
                (0.6, 0.8): "Good performance expected",
                (0.4, 0.6): "Adequate performance, some optimization may be needed",
                (0.0, 0.4): "Performance concerns, optimization required"
            },
            EvaluationCriterion.RELIABILITY: {
                (0.8, 1.0): "High reliability with robust error handling",
                (0.6, 0.8): "Good reliability with standard safeguards", 
                (0.4, 0.6): "Moderate reliability, additional validation needed",
                (0.0, 0.4): "Reliability concerns, significant testing required"
            },
            EvaluationCriterion.RISK_LEVEL: {
                (0.8, 1.0): "Low risk approach with good safeguards",
                (0.6, 0.8): "Acceptable risk level with mitigation",
                (0.4, 0.6): "Moderate risk, careful execution required",
                (0.0, 0.4): "High risk approach, consider alternatives"
            }
        }
        
        criterion_rationales = rationales.get(criterion, {})
        
        for (min_score, max_score), rationale in criterion_rationales.items():
            if min_score <= score < max_score:
                return rationale
        
        return f"Score: {score:.2f}"
    
    async def _generate_selection_rationale(
        self,
        approach: Approach,
        scores: Dict[EvaluationCriterion, EvaluationScore],
        total_score: float,
        criteria: EvaluationCriteria
    ) -> str:
        """Generate comprehensive rationale for approach selection."""
        rationale_parts = []
        
        # Overall assessment
        if total_score >= 0.8:
            rationale_parts.append("Excellent overall approach")
        elif total_score >= 0.6:
            rationale_parts.append("Strong approach with good characteristics")
        elif total_score >= 0.4:
            rationale_parts.append("Viable approach with some trade-offs")
        else:
            rationale_parts.append("Approach has significant limitations")
        
        # Highlight top strengths
        sorted_scores = sorted(scores.items(), key=lambda x: x[1].score, reverse=True)
        top_strengths = sorted_scores[:2]
        
        if top_strengths:
            strength_names = [score[0].value.replace('_', ' ') for score in top_strengths]
            rationale_parts.append(f"Strengths: {', '.join(strength_names)}")
        
        # Note significant weaknesses
        weak_scores = [score for score in sorted_scores if score[1].score < 0.5]
        if weak_scores:
            weakness_names = [score[0].value.replace('_', ' ') for score in weak_scores[:2]]
            rationale_parts.append(f"Areas of concern: {', '.join(weakness_names)}")
        
        # Benefits and risks summary
        if approach.benefits:
            rationale_parts.append(f"Key benefits: {', '.join(approach.benefits[:2])}")
        
        if approach.risks:
            rationale_parts.append(f"Key risks: {', '.join(approach.risks[:2])}")
        
        return ". ".join(rationale_parts)
    
    async def _validate_criteria(self, criteria: EvaluationCriteria):
        """Validate evaluation criteria."""
        # Check that required criteria are present
        for required_criterion in self.config.required_criteria:
            if required_criterion not in criteria.criteria:
                raise InvalidCriteria(f"Required criterion missing: {required_criterion.value}")
        
        # Validate weights are positive
        for criterion, weight in criteria.criteria.items():
            if weight < 0:
                raise InvalidCriteria(f"Negative weight for {criterion.value}: {weight}")
        
        # Check total weight is reasonable
        total_weight = sum(criteria.criteria.values())
        if total_weight == 0:
            raise InvalidCriteria("Total criteria weight is zero")
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation performance statistics."""
        avg_time = (
            self._evaluation_time_sum / self._total_evaluations
            if self._total_evaluations > 0 else 0
        )
        
        return {
            "total_evaluations": self._total_evaluations,
            "average_time_ms": round(avg_time * 1000, 2),
            "config": {
                "use_weighted_scoring": self.config.use_weighted_scoring,
                "parallel_evaluation": self.config.parallel_evaluation,
                "confidence_threshold": self.config.confidence_threshold
            }
        }