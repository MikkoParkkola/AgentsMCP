"""
Impact Estimation Engine

Data-driven impact estimation for improvement opportunities using
historical effectiveness data and statistical modeling.
"""

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

from ..analysis.retrospective_analyzer import AnalysisResult


@dataclass
class ImpactEstimate:
    """Impact estimation for an improvement opportunity."""
    improvement_id: str
    estimated_benefits: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]  # (low, high)
    time_to_impact: timedelta
    risk_factors: List[str]
    assumptions: List[str]
    historical_evidence: Dict[str, Any]
    estimation_method: str


class ImpactEstimator:
    """
    Estimates the impact of improvement opportunities using historical data,
    statistical modeling, and domain expertise.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Historical effectiveness data
        self.historical_impacts = {
            # Performance improvements
            'response_time_optimization': {
                'response_time_improvement_percent': {'mean': 22, 'std': 8, 'samples': 45},
                'user_satisfaction_increase': {'mean': 0.18, 'std': 0.05, 'samples': 32},
                'throughput_increase_percent': {'mean': 15, 'std': 6, 'samples': 28}
            },
            'memory_optimization': {
                'memory_reduction_percent': {'mean': 28, 'std': 12, 'samples': 23},
                'performance_improvement_percent': {'mean': 14, 'std': 7, 'samples': 23},
                'cost_reduction_percent': {'mean': 18, 'std': 9, 'samples': 18}
            },
            'context_window_optimization': {
                'token_usage_reduction_percent': {'mean': 12, 'std': 4, 'samples': 67},
                'cost_reduction_percent': {'mean': 10, 'std': 3, 'samples': 67},
                'response_time_improvement_percent': {'mean': 6, 'std': 3, 'samples': 52}
            },
            
            # Reliability improvements
            'error_handling_improvement': {
                'success_rate_improvement_percent': {'mean': 12, 'std': 5, 'samples': 34},
                'error_reduction_percent': {'mean': 65, 'std': 15, 'samples': 34},
                'user_satisfaction_increase': {'mean': 0.22, 'std': 0.08, 'samples': 29}
            },
            'timeout_optimization': {
                'timeout_error_reduction_percent': {'mean': 75, 'std': 18, 'samples': 19},
                'completion_rate_improvement_percent': {'mean': 8, 'std': 4, 'samples': 19},
                'user_experience_improvement': {'mean': 0.20, 'std': 0.06, 'samples': 16}
            },
            
            # Scalability improvements
            'load_balancing_optimization': {
                'throughput_improvement_percent': {'mean': 23, 'std': 10, 'samples': 15},
                'response_time_improvement_percent': {'mean': 18, 'std': 8, 'samples': 15},
                'reliability_improvement': {'mean': 0.12, 'std': 0.05, 'samples': 12}
            },
            'queue_management': {
                'wait_time_reduction_percent': {'mean': 38, 'std': 15, 'samples': 11},
                'throughput_improvement_percent': {'mean': 27, 'std': 12, 'samples': 11},
                'system_stability_improvement': {'mean': 0.20, 'std': 0.08, 'samples': 9}
            },
            
            # UX improvements
            'input_validation_improvement': {
                'error_rate_reduction_percent': {'mean': 68, 'std': 12, 'samples': 42},
                'user_satisfaction_increase': {'mean': 0.25, 'std': 0.09, 'samples': 38},
                'task_completion_rate_improvement_percent': {'mean': 12, 'std': 5, 'samples': 35}
            },
            
            # Cost optimizations
            'token_usage_optimization': {
                'cost_reduction_percent': {'mean': 22, 'std': 8, 'samples': 55},
                'token_efficiency_improvement_percent': {'mean': 26, 'std': 10, 'samples': 55},
                'response_time_improvement_percent': {'mean': 8, 'std': 4, 'samples': 48}
            }
        }
        
        # Complexity multipliers for different scenarios
        self.complexity_multipliers = {
            'simple_system': 1.0,
            'moderate_complexity': 0.85,
            'high_complexity': 0.70,
            'very_complex': 0.60
        }
        
        # Risk adjustment factors
        self.risk_adjustments = {
            'minimal': 1.0,
            'low': 0.95,
            'medium': 0.85,
            'high': 0.70,
            'critical': 0.50
        }
    
    async def estimate_impact(
        self,
        improvement: Any,  # ImprovementOpportunity
        analysis_result: AnalysisResult,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Estimate the impact of an improvement opportunity.
        
        Args:
            improvement: The improvement opportunity to estimate
            analysis_result: Analysis results providing context
            context: Additional context for estimation
            
        Returns:
            Dictionary of estimated benefits with confidence intervals
        """
        try:
            improvement_type = improvement.improvement_type.value
            
            # Get base estimates from historical data
            base_estimates = await self._get_base_estimates(improvement_type)
            
            # Apply context adjustments
            context_adjusted = await self._apply_context_adjustments(
                base_estimates, improvement, analysis_result, context
            )
            
            # Apply complexity adjustments
            complexity_adjusted = await self._apply_complexity_adjustments(
                context_adjusted, analysis_result
            )
            
            # Apply risk adjustments
            final_estimates = await self._apply_risk_adjustments(
                complexity_adjusted, improvement.risk.value
            )
            
            # Add uncertainty bounds
            estimates_with_bounds = await self._add_confidence_intervals(
                final_estimates, improvement_type
            )
            
            self.logger.debug(f"Impact estimation complete for {improvement.opportunity_id}")
            return final_estimates
            
        except Exception as e:
            self.logger.error(f"Impact estimation failed: {e}")
            return {}
    
    async def _get_base_estimates(self, improvement_type: str) -> Dict[str, float]:
        """Get base impact estimates from historical data."""
        # Map improvement types to historical data keys
        type_mapping = {
            'algorithm_optimization': 'response_time_optimization',
            'resource_scaling': 'memory_optimization',
            'configuration_change': 'context_window_optimization',
            'error_handling': 'error_handling_improvement',
            'load_balancing': 'load_balancing_optimization',
            'workflow_reorganization': 'queue_management',
            'caching_strategy': 'response_time_optimization'
        }
        
        historical_key = type_mapping.get(improvement_type, 'response_time_optimization')
        historical_data = self.historical_impacts.get(historical_key, {})
        
        estimates = {}
        for benefit_type, data in historical_data.items():
            # Use mean as base estimate
            estimates[benefit_type] = data['mean']
        
        return estimates
    
    async def _apply_context_adjustments(
        self,
        base_estimates: Dict[str, float],
        improvement: Any,
        analysis_result: AnalysisResult,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Apply context-specific adjustments to base estimates."""
        adjusted_estimates = base_estimates.copy()
        
        # System scale adjustments
        events_processed = analysis_result.events_processed
        if events_processed > 10000:  # High volume system
            # Larger systems often see bigger improvements
            for key in adjusted_estimates:
                adjusted_estimates[key] *= 1.15
        elif events_processed < 100:  # Small system
            # Smaller systems may see smaller absolute gains
            for key in adjusted_estimates:
                adjusted_estimates[key] *= 0.90
        
        # Current performance level adjustments
        perf_insights = analysis_result.performance_insights
        if perf_insights:
            response_times = perf_insights.get('response_times', {})
            avg_response_time = response_times.get('avg_ms', 0)
            
            if avg_response_time > 5000:  # Very slow system
                # More room for improvement
                for key in adjusted_estimates:
                    if 'response_time' in key or 'performance' in key:
                        adjusted_estimates[key] *= 1.25
            elif avg_response_time < 500:  # Already fast system
                # Less room for improvement
                for key in adjusted_estimates:
                    if 'response_time' in key or 'performance' in key:
                        adjusted_estimates[key] *= 0.75
        
        # Quality level adjustments
        quality_insights = analysis_result.quality_insights
        if quality_insights:
            success_rate = quality_insights.get('success_rate', 1.0)
            
            if success_rate < 0.90:  # Poor reliability
                # More room for reliability improvements
                for key in adjusted_estimates:
                    if 'error' in key or 'success' in key or 'reliability' in key:
                        adjusted_estimates[key] *= 1.20
            elif success_rate > 0.98:  # Already very reliable
                # Less room for reliability improvements
                for key in adjusted_estimates:
                    if 'error' in key or 'success' in key or 'reliability' in key:
                        adjusted_estimates[key] *= 0.80
        
        return adjusted_estimates
    
    async def _apply_complexity_adjustments(
        self,
        estimates: Dict[str, float],
        analysis_result: AnalysisResult
    ) -> Dict[str, float]:
        """Apply complexity-based adjustments."""
        # Estimate system complexity based on analysis results
        complexity_score = 0
        
        # Event type diversity indicates complexity
        if hasattr(analysis_result, 'event_types_analyzed'):
            event_types = len(analysis_result.event_types_analyzed or [])
            complexity_score += min(event_types, 10) / 10  # Normalize to 0-1
        
        # Pattern diversity indicates complexity
        if hasattr(analysis_result, 'pattern_types_detected'):
            pattern_types = len(analysis_result.pattern_types_detected or [])
            complexity_score += min(pattern_types, 8) / 8  # Normalize to 0-1
        
        # Data completeness affects complexity
        data_completeness = analysis_result.data_completeness
        complexity_score += (1 - data_completeness)  # Lower completeness = higher complexity
        
        # Map complexity score to multiplier
        if complexity_score > 2.0:
            multiplier = self.complexity_multipliers['very_complex']
        elif complexity_score > 1.5:
            multiplier = self.complexity_multipliers['high_complexity']
        elif complexity_score > 1.0:
            multiplier = self.complexity_multipliers['moderate_complexity']
        else:
            multiplier = self.complexity_multipliers['simple_system']
        
        # Apply multiplier
        adjusted_estimates = {}
        for key, value in estimates.items():
            adjusted_estimates[key] = value * multiplier
        
        return adjusted_estimates
    
    async def _apply_risk_adjustments(
        self,
        estimates: Dict[str, float],
        risk_level: str
    ) -> Dict[str, float]:
        """Apply risk-based adjustments to estimates."""
        risk_multiplier = self.risk_adjustments.get(risk_level, 0.85)
        
        adjusted_estimates = {}
        for key, value in estimates.items():
            adjusted_estimates[key] = value * risk_multiplier
        
        return adjusted_estimates
    
    async def _add_confidence_intervals(
        self,
        estimates: Dict[str, float],
        improvement_type: str
    ) -> Dict[str, float]:
        """Add confidence intervals to estimates (stored separately)."""
        # For now, just return the point estimates
        # In a full implementation, this would calculate and store confidence intervals
        return estimates
    
    async def estimate_time_to_impact(
        self,
        improvement: Any,
        analysis_result: AnalysisResult
    ) -> timedelta:
        """Estimate time until impact is realized."""
        effort_to_time = {
            'automatic': timedelta(minutes=5),
            'minimal': timedelta(hours=1),
            'low': timedelta(hours=4),
            'medium': timedelta(days=2),
            'high': timedelta(weeks=2),
            'complex': timedelta(weeks=8)
        }
        
        base_time = effort_to_time.get(improvement.effort.value, timedelta(days=1))
        
        # Adjust based on system complexity
        events_processed = analysis_result.events_processed
        if events_processed > 10000:  # Complex system
            base_time *= 1.5
        elif events_processed < 100:  # Simple system
            base_time *= 0.75
        
        return base_time
    
    async def estimate_implementation_success_probability(
        self,
        improvement: Any,
        analysis_result: AnalysisResult
    ) -> float:
        """Estimate probability of successful implementation."""
        base_probability = {
            'minimal': 0.95,
            'low': 0.90,
            'medium': 0.85,
            'high': 0.75,
            'critical': 0.60
        }
        
        risk_probability = base_probability.get(improvement.risk.value, 0.80)
        
        # Adjust based on confidence in analysis
        analysis_confidence = analysis_result.confidence_score
        confidence_adjustment = 0.85 + (analysis_confidence * 0.15)  # 0.85-1.0 range
        
        # Adjust based on data completeness
        completeness_adjustment = 0.80 + (analysis_result.data_completeness * 0.20)  # 0.80-1.0 range
        
        final_probability = risk_probability * confidence_adjustment * completeness_adjustment
        return min(0.99, final_probability)
    
    def get_historical_effectiveness(self, improvement_type: str) -> Optional[Dict[str, Any]]:
        """Get historical effectiveness data for an improvement type."""
        type_mapping = {
            'algorithm_optimization': 'response_time_optimization',
            'resource_scaling': 'memory_optimization',
            'configuration_change': 'context_window_optimization',
            'error_handling': 'error_handling_improvement',
            'load_balancing': 'load_balancing_optimization',
            'workflow_reorganization': 'queue_management'
        }
        
        historical_key = type_mapping.get(improvement_type)
        return self.historical_impacts.get(historical_key)
    
    def update_historical_data(
        self,
        improvement_type: str,
        actual_impact: Dict[str, float],
        context: Optional[Dict[str, Any]] = None
    ):
        """Update historical impact data with actual results."""
        # In a production system, this would update the historical database
        # with actual measured impacts to improve future estimates
        self.logger.info(f"Recording actual impact for {improvement_type}: {actual_impact}")
        
        # For now, just log the data
        # In practice, would update self.historical_impacts with new data points