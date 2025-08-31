"""Pattern analysis engine for identifying team effectiveness patterns and anti-patterns.

This module provides comprehensive pattern recognition capabilities including:
- Success pattern identification from historical execution data
- Anti-pattern detection and warning systems
- Role effectiveness scoring and trend analysis
- Statistical significance testing for pattern reliability
- Temporal pattern analysis and evolution tracking
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import uuid
import math

from .models import (
    TeamComposition,
    TeamPerformanceMetrics,
    AgentSpec,
    CoordinationStrategy,
    ComplexityLevel,
    RiskLevel,
    TaskType,
)
from .execution_engine import TeamExecution, ExecutionStatus


class PatternType(str, Enum):
    """Types of patterns that can be identified."""
    SUCCESS_PATTERN = "success_pattern"
    FAILURE_PATTERN = "failure_pattern"
    ANTI_PATTERN = "anti_pattern"
    EMERGING_PATTERN = "emerging_pattern"
    DECLINING_PATTERN = "declining_pattern"


class PatternCategory(str, Enum):
    """Categories of patterns for classification."""
    TEAM_COMPOSITION = "team_composition"
    COORDINATION_STRATEGY = "coordination_strategy"
    ROLE_EFFECTIVENESS = "role_effectiveness"
    RESOURCE_USAGE = "resource_usage"
    TASK_EXECUTION = "task_execution"
    TEMPORAL = "temporal"


class PatternScope(str, Enum):
    """Scope of pattern analysis."""
    SINGLE_EXECUTION = "single_execution"
    TEAM_SPECIFIC = "team_specific"
    ROLE_SPECIFIC = "role_specific"
    ORGANIZATION_WIDE = "organization_wide"
    TEMPORAL_TRENDS = "temporal_trends"


class StatisticalSignificance(str, Enum):
    """Statistical significance levels."""
    NOT_SIGNIFICANT = "not_significant"
    MARGINALLY_SIGNIFICANT = "marginally_significant"  # p < 0.1
    SIGNIFICANT = "significant"  # p < 0.05
    HIGHLY_SIGNIFICANT = "highly_significant"  # p < 0.01
    VERY_HIGHLY_SIGNIFICANT = "very_highly_significant"  # p < 0.001


@dataclass
class PatternEvidence:
    """Evidence supporting a pattern identification."""
    evidence_type: str
    description: str
    data_points: List[Any] = field(default_factory=list)
    statistical_measure: Optional[float] = None
    confidence_level: float = 0.0
    supporting_executions: List[str] = field(default_factory=list)
    temporal_range: Optional[Tuple[datetime, datetime]] = None


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results for a pattern."""
    sample_size: int
    mean_performance: float
    standard_deviation: float
    confidence_interval_95: Tuple[float, float]
    p_value: Optional[float] = None
    significance_level: StatisticalSignificance = StatisticalSignificance.NOT_SIGNIFICANT
    effect_size: Optional[float] = None
    power_analysis: Optional[float] = None


@dataclass
class TemporalAnalysis:
    """Temporal analysis of pattern evolution."""
    trend_direction: str  # "increasing", "decreasing", "stable", "cyclical"
    trend_strength: float  # 0.0 to 1.0
    trend_significance: StatisticalSignificance
    seasonal_components: Dict[str, float] = field(default_factory=dict)
    change_points: List[datetime] = field(default_factory=list)
    forecast_confidence: float = 0.0


@dataclass
class TeamPattern:
    """Comprehensive pattern representation."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: PatternType = PatternType.SUCCESS_PATTERN
    pattern_category: PatternCategory = PatternCategory.TEAM_COMPOSITION
    scope: PatternScope = PatternScope.TEAM_SPECIFIC
    
    # Core pattern definition
    pattern_name: str = ""
    description: str = ""
    context_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Pattern components
    roles_involved: Set[str] = field(default_factory=set)
    coordination_strategy: Optional[CoordinationStrategy] = None
    team_size_range: Tuple[int, int] = (1, 10)
    
    # Performance characteristics
    success_rate: float = 0.0
    average_performance: float = 0.0
    performance_variance: float = 0.0
    cost_efficiency: float = 0.0
    quality_score: float = 0.0
    
    # Statistical validation
    statistical_analysis: Optional[StatisticalAnalysis] = None
    temporal_analysis: Optional[TemporalAnalysis] = None
    
    # Evidence and support
    evidence: List[PatternEvidence] = field(default_factory=list)
    supporting_executions: List[str] = field(default_factory=list)
    contradicting_executions: List[str] = field(default_factory=list)
    
    # Context effectiveness
    effective_for_task_types: Set[TaskType] = field(default_factory=set)
    effective_for_complexity: Set[ComplexityLevel] = field(default_factory=set)
    effective_for_risk_levels: Set[RiskLevel] = field(default_factory=set)
    
    # Temporal characteristics
    first_observed: Optional[datetime] = None
    last_observed: Optional[datetime] = None
    observation_count: int = 0
    recent_performance_trend: str = "stable"
    
    # Pattern relationships
    related_patterns: List[str] = field(default_factory=list)  # Pattern IDs
    conflicting_patterns: List[str] = field(default_factory=list)  # Pattern IDs
    
    # Metadata
    confidence_score: float = 0.0
    reliability_score: float = 0.0
    actionability_score: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def overall_strength(self) -> float:
        """Calculate overall pattern strength."""
        weights = {
            'statistical': 0.3,
            'confidence': 0.25,
            'reliability': 0.25,
            'actionability': 0.2,
        }
        
        statistical_strength = 0.0
        if self.statistical_analysis:
            significance_map = {
                StatisticalSignificance.NOT_SIGNIFICANT: 0.0,
                StatisticalSignificance.MARGINALLY_SIGNIFICANT: 0.4,
                StatisticalSignificance.SIGNIFICANT: 0.7,
                StatisticalSignificance.HIGHLY_SIGNIFICANT: 0.9,
                StatisticalSignificance.VERY_HIGHLY_SIGNIFICANT: 1.0,
            }
            statistical_strength = significance_map.get(
                self.statistical_analysis.significance_level, 0.0
            )
        
        return (
            weights['statistical'] * statistical_strength +
            weights['confidence'] * self.confidence_score +
            weights['reliability'] * self.reliability_score +
            weights['actionability'] * self.actionability_score
        )


@dataclass
class AntiPattern:
    """Anti-pattern representation with warning characteristics."""
    anti_pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    warning_indicators: List[str] = field(default_factory=list)
    
    # Risk characteristics
    risk_level: RiskLevel = RiskLevel.MEDIUM
    failure_probability: float = 0.0
    impact_severity: float = 0.0
    
    # Conditions that trigger this anti-pattern
    trigger_conditions: Dict[str, Any] = field(default_factory=dict)
    affected_roles: Set[str] = field(default_factory=set)
    problematic_combinations: List[Tuple[str, ...]] = field(default_factory=list)
    
    # Historical evidence
    observed_failures: List[str] = field(default_factory=list)  # Execution IDs
    failure_modes: List[str] = field(default_factory=list)
    cost_impact: float = 0.0
    
    # Prevention and mitigation
    prevention_strategies: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)
    alternative_patterns: List[str] = field(default_factory=list)  # Pattern IDs
    
    # Validation
    statistical_support: Optional[StatisticalAnalysis] = None
    confidence_level: float = 0.0
    
    @property
    def severity_score(self) -> float:
        """Calculate overall severity score."""
        risk_weights = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 1.0,
        }
        
        risk_component = risk_weights.get(self.risk_level, 0.5)
        return (risk_component * 0.4 + 
                self.failure_probability * 0.3 + 
                self.impact_severity * 0.3)


@dataclass
class RoleEffectivenessAnalysis:
    """Detailed analysis of role effectiveness."""
    role_name: str
    
    # Performance metrics
    overall_effectiveness: float = 0.0
    success_rate: float = 0.0
    average_performance: float = 0.0
    consistency_score: float = 0.0
    
    # Contextual effectiveness
    effectiveness_by_task_type: Dict[TaskType, float] = field(default_factory=dict)
    effectiveness_by_complexity: Dict[ComplexityLevel, float] = field(default_factory=dict)
    effectiveness_by_team_size: Dict[int, float] = field(default_factory=dict)
    
    # Collaboration patterns
    best_collaborations: List[Tuple[str, float]] = field(default_factory=list)  # (role, synergy_score)
    problematic_collaborations: List[Tuple[str, float]] = field(default_factory=list)
    
    # Temporal trends
    performance_trend: str = "stable"
    trend_strength: float = 0.0
    recent_performance_change: float = 0.0
    
    # Recommendations
    improvement_areas: List[str] = field(default_factory=list)
    optimal_contexts: List[str] = field(default_factory=list)
    development_suggestions: List[str] = field(default_factory=list)
    
    # Statistical validation
    statistical_confidence: StatisticalSignificance = StatisticalSignificance.NOT_SIGNIFICANT
    sample_size: int = 0


class PatternAnalyzer:
    """Main engine for pattern analysis and recognition."""
    
    def __init__(
        self,
        min_pattern_occurrences: int = 5,
        significance_threshold: float = 0.05,
        confidence_threshold: float = 0.7,
        temporal_window_days: int = 90,
    ):
        self.log = logging.getLogger(__name__)
        self.min_pattern_occurrences = min_pattern_occurrences
        self.significance_threshold = significance_threshold
        self.confidence_threshold = confidence_threshold
        self.temporal_window_days = temporal_window_days
        
        # Pattern storage
        self.identified_patterns: Dict[str, TeamPattern] = {}
        self.identified_anti_patterns: Dict[str, AntiPattern] = {}
        self.role_effectiveness_analysis: Dict[str, RoleEffectivenessAnalysis] = {}
        
        # Analysis history
        self.analysis_history: List[Dict[str, Any]] = []
        self.pattern_evolution_tracking: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        self.log.info("PatternAnalyzer initialized")
    
    async def analyze_patterns(
        self, execution_history: List[TeamExecution]
    ) -> List[TeamPattern]:
        """Analyze execution history to identify patterns.
        
        Args:
            execution_history: List of team executions to analyze
            
        Returns:
            List of identified patterns sorted by strength
        """
        start_time = datetime.now(timezone.utc)
        self.log.info("Starting pattern analysis on %d executions", len(execution_history))
        
        if not execution_history:
            return []
        
        try:
            # Phase 1: Preprocessing and data validation
            validated_executions = await self._preprocess_executions(execution_history)
            
            # Phase 2: Success pattern identification
            success_patterns = await self._identify_success_patterns(validated_executions)
            
            # Phase 3: Anti-pattern identification
            anti_patterns = await self._identify_anti_patterns(validated_executions)
            
            # Phase 4: Role effectiveness analysis
            role_analysis = await self._analyze_role_effectiveness(validated_executions)
            
            # Phase 5: Temporal pattern analysis
            temporal_patterns = await self._analyze_temporal_patterns(validated_executions)
            
            # Phase 6: Statistical validation
            validated_patterns = await self._validate_patterns_statistically(
                success_patterns + temporal_patterns
            )
            
            # Phase 7: Pattern relationship analysis
            await self._analyze_pattern_relationships(validated_patterns, anti_patterns)
            
            # Phase 8: Generate actionable insights
            insights = await self._generate_actionable_insights(
                validated_patterns, anti_patterns, role_analysis
            )
            
            # Store results
            for pattern in validated_patterns:
                self.identified_patterns[pattern.pattern_id] = pattern
            
            for anti_pattern in anti_patterns:
                self.identified_anti_patterns[anti_pattern.anti_pattern_id] = anti_pattern
            
            self.role_effectiveness_analysis = role_analysis
            
            # Record analysis
            analysis_record = {
                'timestamp': start_time,
                'executions_analyzed': len(validated_executions),
                'patterns_identified': len(validated_patterns),
                'anti_patterns_identified': len(anti_patterns),
                'insights_generated': len(insights),
                'duration_seconds': (datetime.now(timezone.utc) - start_time).total_seconds(),
            }
            self.analysis_history.append(analysis_record)
            
            self.log.info("Pattern analysis completed: %d patterns, %d anti-patterns in %.2fs",
                         len(validated_patterns), len(anti_patterns),
                         analysis_record['duration_seconds'])
            
            return sorted(validated_patterns, key=lambda p: p.overall_strength, reverse=True)
            
        except Exception as e:
            self.log.error("Pattern analysis failed: %s", e)
            raise
    
    async def _preprocess_executions(
        self, execution_history: List[TeamExecution]
    ) -> List[TeamExecution]:
        """Preprocess and validate execution data."""
        
        validated_executions = []
        
        for execution in execution_history:
            # Basic validation
            if not execution.team_composition or not execution.team_composition.primary_team:
                continue
            
            if not execution.started_at:
                continue
            
            # Skip executions that are too old
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.temporal_window_days * 2)
            if execution.started_at < cutoff_date:
                continue
            
            validated_executions.append(execution)
        
        self.log.debug("Validated %d out of %d executions", 
                      len(validated_executions), len(execution_history))
        return validated_executions
    
    async def _identify_success_patterns(
        self, executions: List[TeamExecution]
    ) -> List[TeamPattern]:
        """Identify successful team patterns."""
        
        patterns = []
        
        # Group executions by team characteristics
        pattern_groups = defaultdict(list)
        
        for execution in executions:
            if execution.status != ExecutionStatus.COMPLETED:
                continue
            
            # Create pattern signatures
            roles = tuple(sorted(agent.role for agent in execution.team_composition.primary_team))
            team_size = len(execution.team_composition.primary_team)
            coordination = execution.team_composition.coordination_strategy
            
            # Basic pattern signature
            basic_signature = f"roles:{'-'.join(roles)}_coord:{coordination.value}_size:{team_size}"
            pattern_groups[basic_signature].append(execution)
            
            # Role combination patterns (ignoring order)
            role_combo_signature = f"combo:{'-'.join(sorted(roles))}"
            pattern_groups[role_combo_signature].append(execution)
            
            # Coordination strategy patterns
            coord_signature = f"coordination:{coordination.value}_size:{team_size}"
            pattern_groups[coord_signature].append(execution)
        
        # Analyze each pattern group
        for signature, group_executions in pattern_groups.items():
            if len(group_executions) < self.min_pattern_occurrences:
                continue
            
            pattern = await self._create_pattern_from_executions(
                signature, group_executions, PatternType.SUCCESS_PATTERN
            )
            
            if pattern and pattern.success_rate >= 0.7:  # Minimum success threshold
                patterns.append(pattern)
        
        return patterns
    
    async def _identify_anti_patterns(
        self, executions: List[TeamExecution]
    ) -> List[AntiPattern]:
        """Identify problematic patterns that lead to failures."""
        
        anti_patterns = []
        
        # Group failed executions
        failure_groups = defaultdict(list)
        
        for execution in executions:
            if execution.status == ExecutionStatus.FAILED:
                roles = tuple(sorted(agent.role for agent in execution.team_composition.primary_team))
                team_size = len(execution.team_composition.primary_team)
                coordination = execution.team_composition.coordination_strategy
                
                # Look for patterns in failures
                failure_signature = f"roles:{'-'.join(roles)}_coord:{coordination.value}_size:{team_size}"
                failure_groups[failure_signature].append(execution)
        
        # Identify anti-patterns
        for signature, failed_executions in failure_groups.items():
            if len(failed_executions) < self.min_pattern_occurrences:
                continue
            
            # Calculate failure characteristics
            total_similar = self._count_similar_executions(signature, executions)
            failure_rate = len(failed_executions) / total_similar if total_similar > 0 else 0
            
            if failure_rate > 0.5:  # High failure rate threshold
                anti_pattern = await self._create_anti_pattern_from_failures(
                    signature, failed_executions, failure_rate
                )
                anti_patterns.append(anti_pattern)
        
        return anti_patterns
    
    async def _analyze_role_effectiveness(
        self, executions: List[TeamExecution]
    ) -> Dict[str, RoleEffectivenessAnalysis]:
        """Analyze effectiveness of individual roles."""
        
        role_data = defaultdict(lambda: {
            'executions': [],
            'successes': 0,
            'total': 0,
            'performance_scores': [],
            'task_types': defaultdict(list),
            'complexities': defaultdict(list),
            'team_sizes': defaultdict(list),
            'collaborations': defaultdict(list),
        })
        
        # Collect role data
        for execution in executions:
            performance_score = self._calculate_execution_performance_score(execution)
            success = execution.status == ExecutionStatus.COMPLETED
            
            for agent in execution.team_composition.primary_team:
                role = agent.role
                data = role_data[role]
                
                data['executions'].append(execution)
                data['total'] += 1
                if success:
                    data['successes'] += 1
                data['performance_scores'].append(performance_score)
                
                # Context tracking
                if hasattr(execution, 'task_classification') and execution.task_classification:
                    data['task_types'][execution.task_classification.task_type].append(performance_score)
                    data['complexities'][execution.task_classification.complexity].append(performance_score)
                
                team_size = len(execution.team_composition.primary_team)
                data['team_sizes'][team_size].append(performance_score)
                
                # Collaboration tracking
                other_roles = [a.role for a in execution.team_composition.primary_team if a.role != role]
                for other_role in other_roles:
                    data['collaborations'][other_role].append(performance_score)
        
        # Create role effectiveness analyses
        role_analyses = {}
        for role, data in role_data.items():
            if data['total'] < 3:  # Minimum data requirement
                continue
            
            analysis = RoleEffectivenessAnalysis(role_name=role)
            analysis.sample_size = data['total']
            analysis.success_rate = data['successes'] / data['total']
            analysis.average_performance = statistics.mean(data['performance_scores'])
            analysis.consistency_score = 1.0 / (1.0 + statistics.stdev(data['performance_scores']))
            analysis.overall_effectiveness = (
                analysis.success_rate * 0.4 +
                analysis.average_performance * 0.4 +
                analysis.consistency_score * 0.2
            )
            
            # Context effectiveness
            for task_type, scores in data['task_types'].items():
                if len(scores) >= 2:
                    analysis.effectiveness_by_task_type[task_type] = statistics.mean(scores)
            
            for complexity, scores in data['complexities'].items():
                if len(scores) >= 2:
                    analysis.effectiveness_by_complexity[complexity] = statistics.mean(scores)
            
            for team_size, scores in data['team_sizes'].items():
                if len(scores) >= 2:
                    analysis.effectiveness_by_team_size[team_size] = statistics.mean(scores)
            
            # Collaboration analysis
            for collab_role, scores in data['collaborations'].items():
                if len(scores) >= 3:
                    synergy_score = statistics.mean(scores)
                    if synergy_score > analysis.average_performance * 1.1:
                        analysis.best_collaborations.append((collab_role, synergy_score))
                    elif synergy_score < analysis.average_performance * 0.9:
                        analysis.problematic_collaborations.append((collab_role, synergy_score))
            
            # Statistical confidence
            if data['total'] >= 10:
                analysis.statistical_confidence = StatisticalSignificance.SIGNIFICANT
            elif data['total'] >= 5:
                analysis.statistical_confidence = StatisticalSignificance.MARGINALLY_SIGNIFICANT
            
            role_analyses[role] = analysis
        
        return role_analyses
    
    async def _analyze_temporal_patterns(
        self, executions: List[TeamExecution]
    ) -> List[TeamPattern]:
        """Analyze temporal patterns and trends."""
        
        temporal_patterns = []
        
        # Sort executions by time
        sorted_executions = sorted(executions, key=lambda e: e.started_at or datetime.min.replace(tzinfo=timezone.utc))
        
        # Time-based performance analysis
        monthly_performance = defaultdict(list)
        weekly_performance = defaultdict(list)
        
        for execution in sorted_executions:
            if not execution.started_at:
                continue
                
            performance = self._calculate_execution_performance_score(execution)
            
            month_key = execution.started_at.strftime('%Y-%m')
            week_key = execution.started_at.strftime('%Y-W%U')
            
            monthly_performance[month_key].append(performance)
            weekly_performance[week_key].append(performance)
        
        # Identify trends
        if len(monthly_performance) >= 3:
            pattern = await self._create_temporal_trend_pattern(monthly_performance, 'monthly')
            if pattern:
                temporal_patterns.append(pattern)
        
        return temporal_patterns
    
    async def _validate_patterns_statistically(
        self, patterns: List[TeamPattern]
    ) -> List[TeamPattern]:
        """Validate patterns using statistical tests."""
        
        validated_patterns = []
        
        for pattern in patterns:
            if len(pattern.supporting_executions) < self.min_pattern_occurrences:
                continue
            
            # Perform statistical analysis
            statistical_analysis = await self._perform_statistical_tests(pattern)
            pattern.statistical_analysis = statistical_analysis
            
            # Update confidence based on statistical significance
            if statistical_analysis.significance_level in [
                StatisticalSignificance.SIGNIFICANT,
                StatisticalSignificance.HIGHLY_SIGNIFICANT,
                StatisticalSignificance.VERY_HIGHLY_SIGNIFICANT
            ]:
                pattern.confidence_score = min(1.0, pattern.confidence_score * 1.2)
                validated_patterns.append(pattern)
            elif statistical_analysis.significance_level == StatisticalSignificance.MARGINALLY_SIGNIFICANT:
                pattern.confidence_score = pattern.confidence_score * 0.9
                if pattern.confidence_score >= self.confidence_threshold:
                    validated_patterns.append(pattern)
        
        return validated_patterns
    
    async def _analyze_pattern_relationships(
        self, patterns: List[TeamPattern], anti_patterns: List[AntiPattern]
    ) -> None:
        """Analyze relationships between patterns and anti-patterns."""
        
        # Find complementary patterns
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                similarity = self._calculate_pattern_similarity(pattern1, pattern2)
                if 0.3 < similarity < 0.7:  # Moderate similarity suggests complementarity
                    pattern1.related_patterns.append(pattern2.pattern_id)
                    pattern2.related_patterns.append(pattern1.pattern_id)
                elif similarity > 0.9:  # Very high similarity suggests redundancy
                    # Keep the pattern with higher strength
                    if pattern1.overall_strength > pattern2.overall_strength:
                        pattern2.conflicting_patterns.append(pattern1.pattern_id)
                    else:
                        pattern1.conflicting_patterns.append(pattern2.pattern_id)
        
        # Link anti-patterns to alternative patterns
        for anti_pattern in anti_patterns:
            for pattern in patterns:
                if self._patterns_are_alternatives(anti_pattern, pattern):
                    anti_pattern.alternative_patterns.append(pattern.pattern_id)
    
    async def _generate_actionable_insights(
        self,
        patterns: List[TeamPattern],
        anti_patterns: List[AntiPattern],
        role_analyses: Dict[str, RoleEffectivenessAnalysis],
    ) -> List[Dict[str, Any]]:
        """Generate actionable insights from pattern analysis."""
        
        insights = []
        
        # Top patterns insights
        if patterns:
            top_pattern = max(patterns, key=lambda p: p.overall_strength)
            insights.append({
                'type': 'top_success_pattern',
                'title': f"Most Effective Pattern: {top_pattern.pattern_name}",
                'description': top_pattern.description,
                'action': f"Prioritize teams with {', '.join(top_pattern.roles_involved)} roles",
                'impact': f"Expected success rate: {top_pattern.success_rate:.1%}",
                'confidence': top_pattern.confidence_score,
            })
        
        # Anti-pattern warnings
        for anti_pattern in anti_patterns:
            if anti_pattern.severity_score > 0.7:
                insights.append({
                    'type': 'anti_pattern_warning',
                    'title': f"Avoid: {anti_pattern.name}",
                    'description': anti_pattern.description,
                    'action': f"Prevention: {'; '.join(anti_pattern.prevention_strategies)}",
                    'risk_level': anti_pattern.risk_level.value,
                    'severity': anti_pattern.severity_score,
                })
        
        # Role effectiveness insights
        top_roles = sorted(
            role_analyses.values(),
            key=lambda r: r.overall_effectiveness,
            reverse=True
        )[:3]
        
        if top_roles:
            insights.append({
                'type': 'top_performing_roles',
                'title': "Most Effective Roles",
                'roles': [(r.role_name, r.overall_effectiveness) for r in top_roles],
                'action': "Prioritize these roles in future team compositions",
            })
        
        return insights
    
    async def _create_pattern_from_executions(
        self,
        signature: str,
        executions: List[TeamExecution],
        pattern_type: PatternType,
    ) -> Optional[TeamPattern]:
        """Create a pattern from a group of executions."""
        
        if not executions:
            return None
        
        pattern = TeamPattern(pattern_type=pattern_type)
        
        # Extract pattern characteristics
        first_execution = executions[0]
        pattern.roles_involved = set(
            agent.role for agent in first_execution.team_composition.primary_team
        )
        pattern.coordination_strategy = first_execution.team_composition.coordination_strategy
        pattern.team_size_range = (
            min(len(e.team_composition.primary_team) for e in executions),
            max(len(e.team_composition.primary_team) for e in executions),
        )
        
        # Calculate performance metrics
        successes = sum(1 for e in executions if e.status == ExecutionStatus.COMPLETED)
        pattern.success_rate = successes / len(executions)
        
        performance_scores = [
            self._calculate_execution_performance_score(e) for e in executions
        ]
        pattern.average_performance = statistics.mean(performance_scores)
        pattern.performance_variance = statistics.variance(performance_scores) if len(performance_scores) > 1 else 0.0
        
        # Context effectiveness
        for execution in executions:
            if hasattr(execution, 'task_classification') and execution.task_classification:
                pattern.effective_for_task_types.add(execution.task_classification.task_type)
                pattern.effective_for_complexity.add(execution.task_classification.complexity)
                pattern.effective_for_risk_levels.add(execution.task_classification.risk_level)
        
        # Temporal characteristics
        pattern.first_observed = min(e.started_at for e in executions if e.started_at)
        pattern.last_observed = max(e.started_at for e in executions if e.started_at)
        pattern.observation_count = len(executions)
        pattern.supporting_executions = [e.execution_id for e in executions]
        
        # Generate name and description
        pattern.pattern_name = f"{'-'.join(sorted(pattern.roles_involved))} Team Pattern"
        pattern.description = (
            f"Team with {', '.join(sorted(pattern.roles_involved))} roles using "
            f"{pattern.coordination_strategy.value} coordination achieves "
            f"{pattern.success_rate:.1%} success rate"
        )
        
        # Initial confidence score
        pattern.confidence_score = min(1.0, pattern.success_rate * (len(executions) / 10.0))
        pattern.reliability_score = max(0.0, 1.0 - pattern.performance_variance)
        pattern.actionability_score = 0.8  # Most patterns are actionable
        
        return pattern
    
    async def _create_anti_pattern_from_failures(
        self,
        signature: str,
        failed_executions: List[TeamExecution],
        failure_rate: float,
    ) -> AntiPattern:
        """Create an anti-pattern from failed executions."""
        
        anti_pattern = AntiPattern()
        
        # Extract characteristics from failures
        first_failure = failed_executions[0]
        roles = set(agent.role for agent in first_failure.team_composition.primary_team)
        
        anti_pattern.affected_roles = roles
        anti_pattern.failure_probability = failure_rate
        anti_pattern.observed_failures = [e.execution_id for e in failed_executions]
        
        # Analyze failure modes
        failure_modes = []
        for execution in failed_executions:
            if execution.errors:
                failure_modes.extend([error[:50] for error in execution.errors])
        
        anti_pattern.failure_modes = list(set(failure_modes))
        
        # Generate name and description
        anti_pattern.name = f"High-Risk {'-'.join(sorted(roles))} Configuration"
        anti_pattern.description = (
            f"Team configuration with {', '.join(sorted(roles))} roles shows "
            f"{failure_rate:.1%} failure rate with common issues in coordination and execution"
        )
        
        # Risk assessment
        if failure_rate > 0.8:
            anti_pattern.risk_level = RiskLevel.CRITICAL
        elif failure_rate > 0.6:
            anti_pattern.risk_level = RiskLevel.HIGH
        else:
            anti_pattern.risk_level = RiskLevel.MEDIUM
        
        anti_pattern.impact_severity = min(1.0, failure_rate + len(failed_executions) / 20.0)
        
        # Generate prevention strategies
        anti_pattern.prevention_strategies = [
            f"Avoid combining {', '.join(sorted(roles))} without senior oversight",
            "Implement additional coordination checkpoints",
            "Consider breaking down tasks into smaller components",
        ]
        
        anti_pattern.confidence_level = min(1.0, len(failed_executions) / 10.0)
        
        return anti_pattern
    
    async def _create_temporal_trend_pattern(
        self,
        time_performance: Dict[str, List[float]],
        timeframe: str,
    ) -> Optional[TeamPattern]:
        """Create a temporal trend pattern."""
        
        if len(time_performance) < 3:
            return None
        
        # Calculate trend
        time_keys = sorted(time_performance.keys())
        avg_performances = [statistics.mean(time_performance[key]) for key in time_keys]
        
        if len(avg_performances) < 3:
            return None
        
        # Simple linear trend calculation
        n = len(avg_performances)
        x_vals = list(range(n))
        
        # Calculate correlation coefficient as trend strength
        mean_x = statistics.mean(x_vals)
        mean_y = statistics.mean(avg_performances)
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, avg_performances))
        denominator = math.sqrt(
            sum((x - mean_x) ** 2 for x in x_vals) *
            sum((y - mean_y) ** 2 for y in avg_performances)
        )
        
        correlation = numerator / denominator if denominator != 0 else 0
        
        # Create pattern if trend is significant
        if abs(correlation) > 0.6:
            pattern = TeamPattern(
                pattern_type=PatternType.EMERGING_PATTERN if correlation > 0 else PatternType.DECLINING_PATTERN,
                pattern_category=PatternCategory.TEMPORAL,
            )
            
            direction = "improving" if correlation > 0 else "declining"
            pattern.pattern_name = f"{timeframe.title()} Performance Trend"
            pattern.description = f"Team performance shows {direction} trend over {timeframe} periods"
            
            pattern.temporal_analysis = TemporalAnalysis(
                trend_direction=direction,
                trend_strength=abs(correlation),
                trend_significance=StatisticalSignificance.SIGNIFICANT if abs(correlation) > 0.8 else StatisticalSignificance.MARGINALLY_SIGNIFICANT,
            )
            
            pattern.confidence_score = abs(correlation)
            pattern.reliability_score = min(1.0, len(time_performance) / 12.0)  # More data = higher reliability
            pattern.actionability_score = 0.6  # Temporal patterns are moderately actionable
            
            return pattern
        
        return None
    
    async def _perform_statistical_tests(self, pattern: TeamPattern) -> StatisticalAnalysis:
        """Perform statistical tests on a pattern."""
        
        # This is a simplified implementation
        # In a real system, you'd perform proper statistical tests
        
        sample_size = len(pattern.supporting_executions)
        mean_performance = pattern.average_performance
        std_dev = math.sqrt(pattern.performance_variance)
        
        # Simple confidence interval calculation
        margin_of_error = 1.96 * (std_dev / math.sqrt(sample_size)) if sample_size > 0 else 0
        confidence_interval = (
            max(0.0, mean_performance - margin_of_error),
            min(1.0, mean_performance + margin_of_error)
        )
        
        # Determine significance based on sample size and performance
        if sample_size >= 30 and pattern.success_rate > 0.8:
            significance = StatisticalSignificance.HIGHLY_SIGNIFICANT
            p_value = 0.001
        elif sample_size >= 20 and pattern.success_rate > 0.7:
            significance = StatisticalSignificance.SIGNIFICANT
            p_value = 0.03
        elif sample_size >= 10 and pattern.success_rate > 0.6:
            significance = StatisticalSignificance.MARGINALLY_SIGNIFICANT
            p_value = 0.08
        else:
            significance = StatisticalSignificance.NOT_SIGNIFICANT
            p_value = 0.15
        
        return StatisticalAnalysis(
            sample_size=sample_size,
            mean_performance=mean_performance,
            standard_deviation=std_dev,
            confidence_interval_95=confidence_interval,
            p_value=p_value,
            significance_level=significance,
        )
    
    def _calculate_execution_performance_score(self, execution: TeamExecution) -> float:
        """Calculate a performance score for an execution."""
        
        base_score = 1.0 if execution.status == ExecutionStatus.COMPLETED else 0.0
        
        # Adjust for errors
        error_penalty = len(execution.errors) * 0.1
        score = max(0.0, base_score - error_penalty)
        
        # Adjust for duration if available
        if execution.total_duration_seconds and execution.total_duration_seconds > 0:
            # Assume 300 seconds (5 minutes) is optimal
            duration_factor = min(1.0, 300.0 / execution.total_duration_seconds)
            score *= (0.8 + 0.2 * duration_factor)
        
        return min(1.0, score)
    
    def _count_similar_executions(self, signature: str, all_executions: List[TeamExecution]) -> int:
        """Count executions matching a given signature."""
        
        count = 0
        for execution in all_executions:
            exec_roles = tuple(sorted(agent.role for agent in execution.team_composition.primary_team))
            exec_coord = execution.team_composition.coordination_strategy
            exec_size = len(execution.team_composition.primary_team)
            
            exec_signature = f"roles:{'-'.join(exec_roles)}_coord:{exec_coord.value}_size:{exec_size}"
            
            if exec_signature == signature:
                count += 1
        
        return count
    
    def _calculate_pattern_similarity(self, pattern1: TeamPattern, pattern2: TeamPattern) -> float:
        """Calculate similarity between two patterns."""
        
        # Role overlap
        role_overlap = len(pattern1.roles_involved.intersection(pattern2.roles_involved))
        role_union = len(pattern1.roles_involved.union(pattern2.roles_involved))
        role_similarity = role_overlap / role_union if role_union > 0 else 0.0
        
        # Coordination strategy similarity
        coord_similarity = 1.0 if pattern1.coordination_strategy == pattern2.coordination_strategy else 0.0
        
        # Performance similarity
        perf_diff = abs(pattern1.average_performance - pattern2.average_performance)
        perf_similarity = max(0.0, 1.0 - perf_diff)
        
        # Weighted similarity
        return (role_similarity * 0.5 + coord_similarity * 0.2 + perf_similarity * 0.3)
    
    def _patterns_are_alternatives(self, anti_pattern: AntiPattern, pattern: TeamPattern) -> bool:
        """Check if a pattern is a good alternative to an anti-pattern."""
        
        # Check if they involve similar roles but with different outcomes
        role_overlap = len(anti_pattern.affected_roles.intersection(pattern.roles_involved))
        if role_overlap > 0 and pattern.success_rate > 0.8:
            return True
        
        return False
    
    # Public API methods
    
    def get_identified_patterns(self) -> List[TeamPattern]:
        """Get all identified patterns."""
        return list(self.identified_patterns.values())
    
    def get_anti_patterns(self) -> List[AntiPattern]:
        """Get all identified anti-patterns."""
        return list(self.identified_anti_patterns.values())
    
    def get_role_effectiveness(self) -> Dict[str, RoleEffectivenessAnalysis]:
        """Get role effectiveness analysis."""
        return self.role_effectiveness_analysis.copy()
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[TeamPattern]:
        """Get pattern by ID."""
        return self.identified_patterns.get(pattern_id)
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history."""
        return self.analysis_history.copy()
    
    def get_pattern_evolution(self, pattern_id: str) -> List[Tuple[datetime, float]]:
        """Get evolution tracking for a pattern."""
        return self.pattern_evolution_tracking[pattern_id].copy()
    
    async def check_for_anti_patterns(
        self, team_composition: TeamComposition
    ) -> List[AntiPattern]:
        """Check if a team composition matches any known anti-patterns."""
        
        matching_anti_patterns = []
        
        team_roles = set(agent.role for agent in team_composition.primary_team)
        
        for anti_pattern in self.identified_anti_patterns.values():
            # Check role overlap
            if anti_pattern.affected_roles.issubset(team_roles):
                matching_anti_patterns.append(anti_pattern)
            
            # Check problematic combinations
            team_role_tuple = tuple(sorted(team_roles))
            for problematic_combo in anti_pattern.problematic_combinations:
                if set(problematic_combo).issubset(team_roles):
                    matching_anti_patterns.append(anti_pattern)
                    break
        
        return matching_anti_patterns