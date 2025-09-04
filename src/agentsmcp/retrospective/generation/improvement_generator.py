"""
Improvement Generator Engine

Generates specific, actionable improvement opportunities based on analysis results
with data-driven impact estimation and priority ranking.
"""

import asyncio
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import json

from ..analysis.pattern_detection import DetectedPattern, PatternType
from ..analysis.bottleneck_identification import Bottleneck, BottleneckType
from ..analysis.retrospective_analyzer import AnalysisResult
from .suggestion_templates import SuggestionTemplates
from .impact_estimation import ImpactEstimator


class ImprovementType(Enum):
    """Types of improvements that can be generated."""
    CONFIGURATION_CHANGE = "configuration_change"
    ALGORITHM_OPTIMIZATION = "algorithm_optimization"
    RESOURCE_SCALING = "resource_scaling"
    WORKFLOW_REORGANIZATION = "workflow_reorganization"
    CACHING_STRATEGY = "caching_strategy"
    LOAD_BALANCING = "load_balancing"
    ERROR_HANDLING = "error_handling"
    MONITORING_ENHANCEMENT = "monitoring_enhancement"


class ImplementationEffort(Enum):
    """Effort levels for implementing improvements."""
    AUTOMATIC = "automatic"        # Can be applied automatically
    MINIMAL = "minimal"           # Simple config changes
    LOW = "low"                   # 1-2 hours of work
    MEDIUM = "medium"             # 1-2 days of work  
    HIGH = "high"                 # 1+ weeks of work
    COMPLEX = "complex"           # Major refactoring required


class RiskLevel(Enum):
    """Risk levels for implementing improvements."""
    MINIMAL = "minimal"           # Reversible, no side effects
    LOW = "low"                   # Minor impact if issues occur
    MEDIUM = "medium"             # Moderate impact possible
    HIGH = "high"                 # Significant impact possible
    CRITICAL = "critical"         # Could break system if wrong


@dataclass
class ImprovementOpportunity:
    """An improvement opportunity with impact estimation."""
    opportunity_id: str
    title: str
    description: str
    improvement_type: ImprovementType
    effort: ImplementationEffort
    risk: RiskLevel
    
    # Impact estimates
    expected_benefits: Dict[str, float]
    confidence: float
    priority_score: float
    
    # Implementation details
    implementation_steps: List[str]
    rollback_plan: str
    testing_strategy: str
    
    # Evidence and context
    supporting_evidence: List[str]
    affected_components: List[str]
    related_patterns: List[str] = field(default_factory=list)
    related_bottlenecks: List[str] = field(default_factory=list)
    
    # Metadata
    created_timestamp: datetime = field(default_factory=datetime.utcnow)
    estimated_completion_time: Optional[timedelta] = None
    success_metrics: List[str] = field(default_factory=list)


class ImprovementGenerator:
    """
    Generates improvement opportunities from analysis results using
    template-based suggestions and data-driven impact estimation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = SuggestionTemplates()
        self.impact_estimator = ImpactEstimator()
        
        # Generation state
        self._improvement_history = {}
        self._template_effectiveness = defaultdict(float)
    
    async def generate_improvements(
        self,
        analysis_result: AnalysisResult,
        patterns: List[DetectedPattern] = None,
        bottlenecks: List[Bottleneck] = None,
        max_suggestions: int = 10
    ) -> List[ImprovementOpportunity]:
        """
        Generate improvement opportunities from analysis results.
        
        Args:
            analysis_result: Results from retrospective analysis
            patterns: Detected patterns (optional, will extract from analysis)
            bottlenecks: Identified bottlenecks (optional, will extract from analysis)
            max_suggestions: Maximum number of suggestions to generate
            
        Returns:
            List of prioritized improvement opportunities
        """
        try:
            self.logger.info("Generating improvement opportunities")
            
            # Extract patterns and bottlenecks if not provided
            if patterns is None:
                patterns = self._extract_patterns_from_analysis(analysis_result)
            
            if bottlenecks is None:
                bottlenecks = self._extract_bottlenecks_from_analysis(analysis_result)
            
            # Generate improvements from different sources
            improvements = []
            
            # 1. Pattern-based improvements
            pattern_improvements = await self._generate_pattern_improvements(patterns, analysis_result)
            improvements.extend(pattern_improvements)
            
            # 2. Bottleneck-based improvements
            bottleneck_improvements = await self._generate_bottleneck_improvements(bottlenecks, analysis_result)
            improvements.extend(bottleneck_improvements)
            
            # 3. Performance metric improvements
            metric_improvements = await self._generate_metric_improvements(analysis_result)
            improvements.extend(metric_improvements)
            
            # 4. Quality improvements
            quality_improvements = await self._generate_quality_improvements(analysis_result)
            improvements.extend(quality_improvements)
            
            # 5. User experience improvements
            ux_improvements = await self._generate_ux_improvements(analysis_result)
            improvements.extend(ux_improvements)
            
            # Remove duplicates and merge similar improvements
            unique_improvements = await self._deduplicate_improvements(improvements)
            
            # Calculate priority scores and rank
            ranked_improvements = await self._rank_improvements(unique_improvements, analysis_result)
            
            # Apply impact estimation
            final_improvements = []
            for improvement in ranked_improvements[:max_suggestions]:
                # Estimate impact
                impact_estimates = await self.impact_estimator.estimate_impact(
                    improvement, analysis_result
                )
                improvement.expected_benefits = impact_estimates
                
                # Calculate confidence based on evidence strength
                improvement.confidence = self._calculate_confidence(improvement, analysis_result)
                
                final_improvements.append(improvement)
            
            self.logger.info(f"Generated {len(final_improvements)} improvement opportunities")
            return final_improvements
            
        except Exception as e:
            self.logger.error(f"Improvement generation failed: {e}")
            return []
    
    async def _generate_pattern_improvements(
        self,
        patterns: List[DetectedPattern],
        analysis_result: AnalysisResult
    ) -> List[ImprovementOpportunity]:
        """Generate improvements based on detected patterns."""
        improvements = []
        
        for pattern in patterns:
            if pattern.pattern_type == PatternType.PERFORMANCE_ANOMALY:
                improvement = await self._create_performance_improvement(pattern, analysis_result)
                if improvement:
                    improvements.append(improvement)
            
            elif pattern.pattern_type == PatternType.WORKFLOW_INEFFICIENCY:
                improvement = await self._create_workflow_improvement(pattern, analysis_result)
                if improvement:
                    improvements.append(improvement)
            
            elif pattern.pattern_type == PatternType.RESOURCE_USAGE_PATTERN:
                improvement = await self._create_resource_improvement(pattern, analysis_result)
                if improvement:
                    improvements.append(improvement)
            
            elif pattern.pattern_type == PatternType.TEMPORAL_PATTERN:
                improvement = await self._create_temporal_improvement(pattern, analysis_result)
                if improvement:
                    improvements.append(improvement)
        
        return improvements
    
    async def _generate_bottleneck_improvements(
        self,
        bottlenecks: List[Bottleneck],
        analysis_result: AnalysisResult
    ) -> List[ImprovementOpportunity]:
        """Generate improvements based on identified bottlenecks."""
        improvements = []
        
        for bottleneck in bottlenecks:
            if bottleneck.bottleneck_type == BottleneckType.PROCESSING_DELAY:
                improvement = await self._create_processing_optimization(bottleneck, analysis_result)
                if improvement:
                    improvements.append(improvement)
            
            elif bottleneck.bottleneck_type == BottleneckType.MEMORY_PRESSURE:
                improvement = await self._create_memory_optimization(bottleneck, analysis_result)
                if improvement:
                    improvements.append(improvement)
            
            elif bottleneck.bottleneck_type == BottleneckType.QUEUE_BUILDUP:
                improvement = await self._create_queue_optimization(bottleneck, analysis_result)
                if improvement:
                    improvements.append(improvement)
            
            elif bottleneck.bottleneck_type == BottleneckType.AGENT_OVERLOAD:
                improvement = await self._create_load_balancing_improvement(bottleneck, analysis_result)
                if improvement:
                    improvements.append(improvement)
        
        return improvements
    
    async def _generate_metric_improvements(
        self,
        analysis_result: AnalysisResult
    ) -> List[ImprovementOpportunity]:
        """Generate improvements based on performance metrics."""
        improvements = []
        
        perf_insights = analysis_result.performance_insights
        if perf_insights and perf_insights.get('response_times'):
            response_times = perf_insights['response_times']
            avg_time = response_times.get('avg_ms', 0)
            p95_time = response_times.get('p95_ms', 0)
            
            # Slow response time improvement
            if avg_time > 2000 or p95_time > 5000:  # >2s avg or >5s p95
                improvement = ImprovementOpportunity(
                    opportunity_id=f"response_time_opt_{analysis_result.session_id}",
                    title="Optimize Response Times",
                    description=f"Average response time is {avg_time:.0f}ms with p95 at {p95_time:.0f}ms. Optimization can reduce latency significantly.",
                    improvement_type=ImprovementType.ALGORITHM_OPTIMIZATION,
                    effort=ImplementationEffort.MEDIUM,
                    risk=RiskLevel.LOW,
                    expected_benefits={
                        'response_time_improvement_percent': 25,
                        'user_satisfaction_improvement': 0.15
                    },
                    confidence=0.80,
                    priority_score=8.5,
                    implementation_steps=[
                        "Profile slow operations to identify bottlenecks",
                        "Implement caching for frequently accessed data",
                        "Optimize database queries and API calls",
                        "Add response time monitoring"
                    ],
                    rollback_plan="Revert to previous algorithm implementation",
                    testing_strategy="A/B test with 10% traffic, monitor response times",
                    supporting_evidence=[
                        f"Average response time: {avg_time:.0f}ms",
                        f"95th percentile: {p95_time:.0f}ms"
                    ],
                    affected_components=["response_processing"],
                    success_metrics=[
                        "Average response time < 1500ms",
                        "95th percentile response time < 3000ms"
                    ]
                )
                improvements.append(improvement)
        
        return improvements
    
    async def _generate_quality_improvements(
        self,
        analysis_result: AnalysisResult
    ) -> List[ImprovementOpportunity]:
        """Generate improvements for quality metrics."""
        improvements = []
        
        quality_insights = analysis_result.quality_insights
        if quality_insights:
            success_rate = quality_insights.get('success_rate', 1.0)
            
            # Low success rate improvement
            if success_rate < 0.95:  # Less than 95% success
                improvement = ImprovementOpportunity(
                    opportunity_id=f"quality_improvement_{analysis_result.session_id}",
                    title="Improve Task Success Rate",
                    description=f"Current success rate is {success_rate:.1%}. Implementing better error handling and validation can improve reliability.",
                    improvement_type=ImprovementType.ERROR_HANDLING,
                    effort=ImplementationEffort.MEDIUM,
                    risk=RiskLevel.LOW,
                    expected_benefits={
                        'success_rate_improvement_percent': (0.98 - success_rate) * 100,
                        'error_reduction_percent': 50
                    },
                    confidence=0.75,
                    priority_score=9.0,
                    implementation_steps=[
                        "Analyze failure patterns and root causes",
                        "Implement comprehensive input validation",
                        "Add retry mechanisms for transient failures",
                        "Improve error handling and recovery"
                    ],
                    rollback_plan="Disable new error handling, revert to previous logic",
                    testing_strategy="Test error scenarios, monitor success rates",
                    supporting_evidence=[
                        f"Current success rate: {success_rate:.1%}",
                        f"Target improvement to 98%+"
                    ],
                    affected_components=["error_handling", "validation"],
                    success_metrics=[
                        "Task success rate > 98%",
                        "Error rate < 2%"
                    ]
                )
                improvements.append(improvement)
        
        return improvements
    
    async def _generate_ux_improvements(
        self,
        analysis_result: AnalysisResult
    ) -> List[ImprovementOpportunity]:
        """Generate user experience improvements."""
        improvements = []
        
        ux_insights = analysis_result.user_experience_insights
        if ux_insights:
            error_rate = ux_insights.get('error_rate', 0.0)
            
            # High error rate improvement
            if error_rate > 0.05:  # More than 5% error rate
                improvement = ImprovementOpportunity(
                    opportunity_id=f"ux_error_reduction_{analysis_result.session_id}",
                    title="Reduce User-Facing Errors",
                    description=f"User error rate is {error_rate:.1%}. Better input validation and user guidance can significantly improve experience.",
                    improvement_type=ImprovementType.ERROR_HANDLING,
                    effort=ImplementationEffort.LOW,
                    risk=RiskLevel.MINIMAL,
                    expected_benefits={
                        'error_rate_reduction_percent': 60,
                        'user_satisfaction_improvement': 0.20
                    },
                    confidence=0.85,
                    priority_score=7.5,
                    implementation_steps=[
                        "Add input validation with helpful error messages",
                        "Implement guided input flows",
                        "Add examples and tooltips",
                        "Improve error message clarity"
                    ],
                    rollback_plan="Revert to previous validation logic",
                    testing_strategy="User testing with error scenarios",
                    supporting_evidence=[
                        f"Current error rate: {error_rate:.1%}",
                        "User feedback indicates confusion"
                    ],
                    affected_components=["user_interface", "validation"],
                    success_metrics=[
                        "User error rate < 2%",
                        "Positive user feedback > 90%"
                    ]
                )
                improvements.append(improvement)
        
        return improvements
    
    async def _create_performance_improvement(
        self,
        pattern: DetectedPattern,
        analysis_result: AnalysisResult
    ) -> Optional[ImprovementOpportunity]:
        """Create performance improvement from pattern."""
        evidence = pattern.evidence
        
        if 'outlier_values' in evidence:
            outliers = evidence['outlier_values']
            mean_time = evidence.get('mean_response_time', 0)
            
            return ImprovementOpportunity(
                opportunity_id=f"perf_pattern_{pattern.pattern_id}",
                title="Optimize Performance Anomalies",
                description=f"Detected {len(outliers)} performance outliers. Optimizing these cases can improve overall performance.",
                improvement_type=ImprovementType.ALGORITHM_OPTIMIZATION,
                effort=ImplementationEffort.MEDIUM,
                risk=RiskLevel.LOW,
                expected_benefits={
                    'response_time_improvement_percent': 20,
                    'consistency_improvement': 0.30
                },
                confidence=pattern.confidence,
                priority_score=8.0,
                implementation_steps=[
                    "Identify common characteristics of outlier cases",
                    "Optimize algorithms for these specific patterns",
                    "Add performance monitoring for early detection"
                ],
                rollback_plan="Revert algorithm changes",
                testing_strategy="Performance testing with outlier scenarios",
                supporting_evidence=[f"Pattern confidence: {pattern.confidence:.2f}"],
                affected_components=["performance_critical_path"],
                related_patterns=[pattern.pattern_id]
            )
        
        return None
    
    async def _create_memory_optimization(
        self,
        bottleneck: Bottleneck,
        analysis_result: AnalysisResult
    ) -> Optional[ImprovementOpportunity]:
        """Create memory optimization improvement."""
        impact = bottleneck.impact_metrics
        avg_memory = impact.get('avg_memory_mb', 0)
        peak_memory = impact.get('peak_memory_mb', 0)
        
        return ImprovementOpportunity(
            opportunity_id=f"memory_opt_{bottleneck.bottleneck_id}",
            title="Optimize Memory Usage",
            description=f"Memory usage averaging {avg_memory:.0f}MB with peaks at {peak_memory:.0f}MB. Memory optimization can improve performance and reduce costs.",
            improvement_type=ImprovementType.RESOURCE_SCALING,
            effort=ImplementationEffort.MEDIUM,
            risk=RiskLevel.MEDIUM,
            expected_benefits={
                'memory_reduction_percent': 30,
                'performance_improvement_percent': 15,
                'cost_reduction_percent': 20
            },
            confidence=bottleneck.confidence,
            priority_score=7.5,
            implementation_steps=[
                "Implement memory pooling and object reuse",
                "Optimize data structures for memory efficiency",
                "Add memory usage monitoring and alerts",
                "Implement garbage collection tuning"
            ],
            rollback_plan="Revert to previous memory management",
            testing_strategy="Load testing with memory monitoring",
            supporting_evidence=[
                f"Average memory: {avg_memory:.0f}MB",
                f"Peak memory: {peak_memory:.0f}MB"
            ],
            affected_components=[bottleneck.location],
            related_bottlenecks=[bottleneck.bottleneck_id]
        )
    
    async def _deduplicate_improvements(
        self,
        improvements: List[ImprovementOpportunity]
    ) -> List[ImprovementOpportunity]:
        """Remove duplicate and merge similar improvements."""
        unique_improvements = {}
        
        for improvement in improvements:
            # Create similarity key based on type and affected components
            similarity_key = (
                improvement.improvement_type,
                frozenset(improvement.affected_components)
            )
            
            if similarity_key not in unique_improvements:
                unique_improvements[similarity_key] = improvement
            else:
                # Merge similar improvements (keep higher priority)
                existing = unique_improvements[similarity_key]
                if improvement.priority_score > existing.priority_score:
                    unique_improvements[similarity_key] = improvement
        
        return list(unique_improvements.values())
    
    async def _rank_improvements(
        self,
        improvements: List[ImprovementOpportunity],
        analysis_result: AnalysisResult
    ) -> List[ImprovementOpportunity]:
        """Rank improvements by priority score."""
        for improvement in improvements:
            # Calculate priority score based on multiple factors
            priority_factors = []
            
            # Impact factor (estimated benefits)
            impact_factor = sum(improvement.expected_benefits.values()) / len(improvement.expected_benefits) if improvement.expected_benefits else 5.0
            priority_factors.append(min(10.0, impact_factor))
            
            # Confidence factor
            confidence_factor = improvement.confidence * 10
            priority_factors.append(confidence_factor)
            
            # Effort factor (inverse - easier implementation gets higher priority)
            effort_weights = {
                ImplementationEffort.AUTOMATIC: 10,
                ImplementationEffort.MINIMAL: 9,
                ImplementationEffort.LOW: 8,
                ImplementationEffort.MEDIUM: 6,
                ImplementationEffort.HIGH: 4,
                ImplementationEffort.COMPLEX: 2
            }
            effort_factor = effort_weights.get(improvement.effort, 5)
            priority_factors.append(effort_factor)
            
            # Risk factor (inverse - lower risk gets higher priority)
            risk_weights = {
                RiskLevel.MINIMAL: 10,
                RiskLevel.LOW: 8,
                RiskLevel.MEDIUM: 6,
                RiskLevel.HIGH: 4,
                RiskLevel.CRITICAL: 2
            }
            risk_factor = risk_weights.get(improvement.risk, 5)
            priority_factors.append(risk_factor)
            
            # Calculate weighted average
            improvement.priority_score = statistics.mean(priority_factors)
        
        # Sort by priority score (descending)
        return sorted(improvements, key=lambda i: i.priority_score, reverse=True)
    
    def _calculate_confidence(
        self,
        improvement: ImprovementOpportunity,
        analysis_result: AnalysisResult
    ) -> float:
        """Calculate confidence score for improvement."""
        confidence_factors = []
        
        # Data completeness factor
        completeness_factor = analysis_result.data_completeness
        confidence_factors.append(completeness_factor)
        
        # Evidence strength factor
        evidence_count = len(improvement.supporting_evidence)
        evidence_factor = min(1.0, evidence_count / 3)  # Normalize to 3 pieces of evidence
        confidence_factors.append(evidence_factor)
        
        # Pattern/bottleneck relation factor
        relation_factor = 0.8 if (improvement.related_patterns or improvement.related_bottlenecks) else 0.5
        confidence_factors.append(relation_factor)
        
        return statistics.mean(confidence_factors)
    
    def _extract_patterns_from_analysis(self, analysis_result: AnalysisResult) -> List[DetectedPattern]:
        """Extract patterns from analysis result."""
        # This would normally extract from analysis_result.improvement_opportunities
        # For now, return empty list since patterns are passed separately
        return []
    
    def _extract_bottlenecks_from_analysis(self, analysis_result: AnalysisResult) -> List[Bottleneck]:
        """Extract bottlenecks from analysis result."""
        # Convert bottleneck dictionaries back to objects if needed
        bottlenecks = []
        for bottleneck_data in analysis_result.bottlenecks_identified:
            # Would need to reconstruct Bottleneck objects from dictionary data
            pass
        return bottlenecks