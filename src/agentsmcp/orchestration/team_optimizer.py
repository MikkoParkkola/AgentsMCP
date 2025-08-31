"""Team composition optimizer for continuous improvement and efficiency.

This module provides optimization capabilities including:
- Cost-efficiency optimization without quality loss
- Identification of most effective role combinations
- Historical data analysis and recommendation generation
- Performance-based team composition suggestions
- Resource allocation optimization
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, Counter

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


class OptimizationObjective(str, Enum):
    """Primary optimization objectives."""
    COST_EFFICIENCY = "cost_efficiency"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    SPEED = "speed"
    RELIABILITY = "reliability"
    BALANCED = "balanced"


class OptimizationStrategy(str, Enum):
    """Optimization strategies for team composition."""
    GREEDY = "greedy"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    HILL_CLIMBING = "hill_climbing"
    ENSEMBLE = "ensemble"


@dataclass
class OptimizationConstraints:
    """Constraints for team optimization."""
    max_team_size: int = 10
    min_team_size: int = 1
    max_cost_per_execution: Optional[float] = None
    min_success_rate: float = 0.8
    max_execution_time: Optional[float] = None
    required_roles: Set[str] = field(default_factory=set)
    forbidden_roles: Set[str] = field(default_factory=set)
    resource_limits: Dict[str, Union[int, float]] = field(default_factory=dict)


@dataclass
class RoleEffectiveness:
    """Effectiveness metrics for a specific role."""
    role_name: str
    success_rate: float = 0.0
    average_performance_score: float = 0.0
    cost_efficiency: float = 0.0
    collaboration_score: float = 0.0
    total_executions: int = 0
    recent_trend: float = 0.0  # Positive = improving, negative = declining
    specialized_tasks: List[str] = field(default_factory=list)
    best_team_combinations: List[Tuple[str, ...]] = field(default_factory=list)
    
    @property
    def overall_effectiveness(self) -> float:
        """Calculate overall effectiveness score."""
        if self.total_executions == 0:
            return 0.0
        
        weights = {
            'success_rate': 0.3,
            'performance': 0.25,
            'cost_efficiency': 0.2,
            'collaboration': 0.15,
            'trend': 0.1
        }
        
        return (
            weights['success_rate'] * self.success_rate +
            weights['performance'] * self.average_performance_score +
            weights['cost_efficiency'] * self.cost_efficiency +
            weights['collaboration'] * self.collaboration_score +
            weights['trend'] * max(0, self.recent_trend)  # Only positive trends
        )


@dataclass
class TeamPattern:
    """Pattern representing successful team compositions."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    roles: Tuple[str, ...] = field(default_factory=tuple)
    coordination_strategy: CoordinationStrategy = CoordinationStrategy.SEQUENTIAL
    team_size: int = 0
    
    # Performance metrics
    success_rate: float = 0.0
    average_cost: float = 0.0
    average_duration: float = 0.0
    quality_score: float = 0.0
    
    # Usage statistics
    usage_count: int = 0
    last_used: Optional[datetime] = None
    
    # Context effectiveness
    effective_for_task_types: List[TaskType] = field(default_factory=list)
    effective_for_complexity: List[ComplexityLevel] = field(default_factory=list)
    effective_for_risk: List[RiskLevel] = field(default_factory=list)
    
    # Confidence and reliability
    confidence_score: float = 0.0
    statistical_significance: float = 0.0
    
    @property
    def pattern_key(self) -> str:
        """Generate unique key for this pattern."""
        sorted_roles = tuple(sorted(self.roles))
        return f"{'-'.join(sorted_roles)}_{self.coordination_strategy.value}"
    
    @property
    def effectiveness_score(self) -> float:
        """Calculate overall effectiveness score for this pattern."""
        if self.usage_count == 0:
            return 0.0
        
        # Weight factors based on importance
        weights = {
            'success': 0.35,
            'cost': 0.25,
            'quality': 0.2,
            'confidence': 0.1,
            'usage': 0.1
        }
        
        # Normalize cost (lower is better)
        normalized_cost = max(0, 1.0 - (self.average_cost / 100.0))  # Assuming max reasonable cost is 100
        
        # Normalize usage count (higher usage indicates proven effectiveness)
        normalized_usage = min(1.0, self.usage_count / 50.0)  # Cap at 50 uses
        
        return (
            weights['success'] * self.success_rate +
            weights['cost'] * normalized_cost +
            weights['quality'] * self.quality_score +
            weights['confidence'] * self.confidence_score +
            weights['usage'] * normalized_usage
        )


@dataclass
class OptimizationResults:
    """Results from team optimization process."""
    optimization_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    optimization_objective: OptimizationObjective = OptimizationObjective.BALANCED
    strategy_used: OptimizationStrategy = OptimizationStrategy.GREEDY
    
    # Optimization results
    recommended_teams: List[TeamComposition] = field(default_factory=list)
    role_effectiveness: Dict[str, RoleEffectiveness] = field(default_factory=dict)
    identified_patterns: List[TeamPattern] = field(default_factory=list)
    
    # Performance analysis
    current_performance_baseline: Dict[str, float] = field(default_factory=dict)
    projected_improvements: Dict[str, float] = field(default_factory=dict)
    cost_savings_estimate: float = 0.0
    quality_improvement_estimate: float = 0.0
    
    # Recommendations
    optimization_recommendations: List[str] = field(default_factory=list)
    role_addition_suggestions: List[str] = field(default_factory=list)
    role_removal_suggestions: List[str] = field(default_factory=list)
    coordination_improvements: List[str] = field(default_factory=list)
    
    # Metadata
    optimization_duration_seconds: float = 0.0
    data_points_analyzed: int = 0
    confidence_score: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TeamOptimizer:
    """Main optimizer for team compositions and patterns."""
    
    def __init__(
        self,
        optimization_history_limit: int = 1000,
        pattern_confidence_threshold: float = 0.7,
        statistical_significance_threshold: float = 0.05,
    ):
        self.log = logging.getLogger(__name__)
        self.optimization_history_limit = optimization_history_limit
        self.pattern_confidence_threshold = pattern_confidence_threshold
        self.statistical_significance_threshold = statistical_significance_threshold
        
        # State tracking
        self.optimization_history: List[OptimizationResults] = []
        self.known_patterns: Dict[str, TeamPattern] = {}
        self.role_effectiveness_cache: Dict[str, RoleEffectiveness] = {}
        self.last_optimization_time: Optional[datetime] = None
        
        self.log.info("TeamOptimizer initialized")
    
    async def optimize_patterns(
        self,
        historical_results: List[TeamExecution],
        performance_metrics: Dict[str, TeamPerformanceMetrics],
        optimization_objective: OptimizationObjective = OptimizationObjective.BALANCED,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.GREEDY,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> OptimizationResults:
        """Optimize team patterns based on historical data.
        
        Args:
            historical_results: List of past team executions
            performance_metrics: Performance data for teams
            optimization_objective: Primary objective to optimize for
            optimization_strategy: Strategy to use for optimization
            constraints: Optional constraints to respect
            
        Returns:
            OptimizationResults with recommendations and analysis
        """
        start_time = datetime.now(timezone.utc)
        self.log.info("Starting team optimization with objective: %s", optimization_objective.value)
        
        # Initialize results
        results = OptimizationResults(
            optimization_objective=optimization_objective,
            strategy_used=optimization_strategy,
            data_points_analyzed=len(historical_results),
        )
        
        constraints = constraints or OptimizationConstraints()
        
        try:
            # Phase 1: Analyze role effectiveness
            results.role_effectiveness = await self._analyze_role_effectiveness(
                historical_results, performance_metrics
            )
            
            # Phase 2: Extract and analyze patterns
            results.identified_patterns = await self._extract_team_patterns(
                historical_results, performance_metrics
            )
            
            # Phase 3: Calculate baseline performance
            results.current_performance_baseline = await self._calculate_performance_baseline(
                historical_results, performance_metrics
            )
            
            # Phase 4: Generate optimized team recommendations
            results.recommended_teams = await self._generate_optimized_teams(
                results.role_effectiveness,
                results.identified_patterns,
                optimization_objective,
                constraints,
            )
            
            # Phase 5: Project improvements and savings
            results.projected_improvements = await self._project_improvements(
                results.recommended_teams,
                results.current_performance_baseline,
                optimization_objective,
            )
            
            # Phase 6: Generate specific recommendations
            results.optimization_recommendations = await self._generate_optimization_recommendations(
                results.role_effectiveness,
                results.identified_patterns,
                results.projected_improvements,
            )
            
            # Calculate confidence score
            results.confidence_score = await self._calculate_optimization_confidence(
                results, len(historical_results)
            )
            
        except Exception as e:
            self.log.error("Team optimization failed: %s", e)
            results.optimization_recommendations.append(f"Optimization failed: {str(e)}")
            raise
        
        finally:
            # Finalize results
            end_time = datetime.now(timezone.utc)
            results.optimization_duration_seconds = (end_time - start_time).total_seconds()
            
            # Store in history
            self.optimization_history.append(results)
            if len(self.optimization_history) > self.optimization_history_limit:
                self.optimization_history = self.optimization_history[-self.optimization_history_limit:]
            
            self.last_optimization_time = end_time
            
            self.log.info("Team optimization completed in %.2fs with confidence %.2f",
                         results.optimization_duration_seconds, results.confidence_score)
        
        return results
    
    async def _analyze_role_effectiveness(
        self,
        historical_results: List[TeamExecution],
        performance_metrics: Dict[str, TeamPerformanceMetrics],
    ) -> Dict[str, RoleEffectiveness]:
        """Analyze effectiveness of individual roles."""
        
        role_stats = defaultdict(lambda: {
            'successes': 0,
            'total': 0,
            'performance_scores': [],
            'costs': [],
            'collaboration_scores': [],
            'recent_performances': [],
            'specialized_tasks': set(),
            'team_combinations': []
        })
        
        # Analyze historical executions
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)  # Recent trend analysis
        
        for execution in historical_results:
            if not execution.team_composition or not execution.team_composition.primary_team:
                continue
                
            # Team-level metrics
            team_success = execution.status == ExecutionStatus.COMPLETED
            team_cost = execution.resource_usage.get('cost', 0.0)
            team_performance = self._calculate_team_performance_score(execution)
            
            team_roles = tuple(agent.role for agent in execution.team_composition.primary_team)
            
            # Per-role analysis
            for agent in execution.team_composition.primary_team:
                role = agent.role
                stats = role_stats[role]
                
                stats['total'] += 1
                if team_success:
                    stats['successes'] += 1
                
                stats['performance_scores'].append(team_performance)
                stats['costs'].append(team_cost / len(execution.team_composition.primary_team))
                stats['collaboration_scores'].append(
                    self._estimate_role_collaboration_score(role, execution)
                )
                
                # Track recent performance for trend analysis
                if execution.started_at and execution.started_at >= cutoff_date:
                    stats['recent_performances'].append(team_performance)
                
                # Track specialized tasks
                if hasattr(execution, 'task_classification'):
                    stats['specialized_tasks'].add(execution.task_classification.task_type.value)
                
                # Track team combinations
                stats['team_combinations'].append(team_roles)
        
        # Convert to RoleEffectiveness objects
        role_effectiveness = {}
        for role, stats in role_stats.items():
            if stats['total'] == 0:
                continue
                
            effectiveness = RoleEffectiveness(role_name=role)
            effectiveness.success_rate = stats['successes'] / stats['total']
            effectiveness.total_executions = stats['total']
            
            # Calculate averages
            if stats['performance_scores']:
                effectiveness.average_performance_score = statistics.mean(stats['performance_scores'])
            
            if stats['costs']:
                avg_cost = statistics.mean(stats['costs'])
                # Convert to efficiency (lower cost = higher efficiency)
                effectiveness.cost_efficiency = max(0, 1.0 - (avg_cost / 100.0))
            
            if stats['collaboration_scores']:
                effectiveness.collaboration_score = statistics.mean(stats['collaboration_scores'])
            
            # Calculate recent trend
            if len(stats['recent_performances']) >= 3:
                # Simple trend: compare recent vs historical average
                recent_avg = statistics.mean(stats['recent_performances'])
                historical_avg = statistics.mean(stats['performance_scores'])
                effectiveness.recent_trend = (recent_avg - historical_avg) / historical_avg
            
            # Set specialized tasks
            effectiveness.specialized_tasks = list(stats['specialized_tasks'])
            
            # Find best team combinations
            combination_counts = Counter(stats['team_combinations'])
            effectiveness.best_team_combinations = [
                combo for combo, count in combination_counts.most_common(5)
            ]
            
            role_effectiveness[role] = effectiveness
        
        return role_effectiveness
    
    async def _extract_team_patterns(
        self,
        historical_results: List[TeamExecution],
        performance_metrics: Dict[str, TeamPerformanceMetrics],
    ) -> List[TeamPattern]:
        """Extract successful team patterns from historical data."""
        
        pattern_stats = defaultdict(lambda: {
            'executions': [],
            'success_count': 0,
            'total_count': 0,
            'costs': [],
            'durations': [],
            'quality_scores': [],
            'task_types': set(),
            'complexities': set(),
            'risks': set(),
        })
        
        # Group executions by team pattern
        for execution in historical_results:
            if not execution.team_composition or not execution.team_composition.primary_team:
                continue
                
            # Create pattern key
            roles = tuple(sorted(agent.role for agent in execution.team_composition.primary_team))
            coordination = execution.team_composition.coordination_strategy
            pattern_key = f"{'-'.join(roles)}_{coordination.value}"
            
            stats = pattern_stats[pattern_key]
            stats['executions'].append(execution)
            stats['total_count'] += 1
            
            if execution.status == ExecutionStatus.COMPLETED:
                stats['success_count'] += 1
            
            # Collect metrics
            if execution.resource_usage:
                cost = execution.resource_usage.get('cost', 0.0)
                stats['costs'].append(cost)
            
            if execution.total_duration_seconds:
                stats['durations'].append(execution.total_duration_seconds)
            
            # Quality score based on errors and completion
            error_count = len(execution.errors)
            quality = max(0.0, 1.0 - (error_count * 0.1))  # Penalize errors
            stats['quality_scores'].append(quality)
            
            # Context tracking (if available)
            if hasattr(execution, 'task_classification') and execution.task_classification:
                stats['task_types'].add(execution.task_classification.task_type)
                stats['complexities'].add(execution.task_classification.complexity)
                stats['risks'].add(execution.task_classification.risk_level)
        
        # Convert to TeamPattern objects
        patterns = []
        for pattern_key, stats in pattern_stats.items():
            if stats['total_count'] < 3:  # Require minimum usage for statistical significance
                continue
                
            # Parse pattern key
            parts = pattern_key.rsplit('_', 1)
            if len(parts) != 2:
                continue
                
            role_string, coordination_str = parts
            roles = tuple(role_string.split('-'))
            
            try:
                coordination = CoordinationStrategy(coordination_str)
            except ValueError:
                continue
            
            # Create pattern
            pattern = TeamPattern(
                roles=roles,
                coordination_strategy=coordination,
                team_size=len(roles),
                usage_count=stats['total_count'],
            )
            
            # Calculate metrics
            pattern.success_rate = stats['success_count'] / stats['total_count']
            
            if stats['costs']:
                pattern.average_cost = statistics.mean(stats['costs'])
            
            if stats['durations']:
                pattern.average_duration = statistics.mean(stats['durations'])
            
            if stats['quality_scores']:
                pattern.quality_score = statistics.mean(stats['quality_scores'])
            
            # Context effectiveness
            pattern.effective_for_task_types = list(stats['task_types'])
            pattern.effective_for_complexity = list(stats['complexities'])
            pattern.effective_for_risk = list(stats['risks'])
            
            # Calculate confidence based on usage and consistency
            if stats['total_count'] >= 10:
                pattern.confidence_score = min(1.0, pattern.success_rate * 
                                             (1.0 + stats['total_count'] / 100.0))
            else:
                pattern.confidence_score = pattern.success_rate * (stats['total_count'] / 10.0)
            
            # Statistical significance (simplified chi-square test approximation)
            expected_success = 0.5  # Assume 50% baseline success rate
            observed_success = pattern.success_rate
            if stats['total_count'] > 5:
                chi_square = ((observed_success - expected_success) ** 2) / expected_success
                pattern.statistical_significance = min(1.0, chi_square / 10.0)  # Normalized
            
            patterns.append(pattern)
        
        # Sort by effectiveness score
        patterns.sort(key=lambda p: p.effectiveness_score, reverse=True)
        
        # Update known patterns
        for pattern in patterns:
            self.known_patterns[pattern.pattern_key] = pattern
        
        return patterns
    
    async def _calculate_performance_baseline(
        self,
        historical_results: List[TeamExecution],
        performance_metrics: Dict[str, TeamPerformanceMetrics],
    ) -> Dict[str, float]:
        """Calculate current performance baseline metrics."""
        
        if not historical_results:
            return {}
        
        # Collect metrics from recent executions
        recent_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        recent_executions = [
            e for e in historical_results
            if e.started_at and e.started_at >= recent_cutoff
        ]
        
        if not recent_executions:
            recent_executions = historical_results[-10:]  # Fall back to last 10
        
        baseline = {}
        
        # Success rate
        successful = sum(1 for e in recent_executions if e.status == ExecutionStatus.COMPLETED)
        baseline['success_rate'] = successful / len(recent_executions) if recent_executions else 0.0
        
        # Average cost
        costs = [
            e.resource_usage.get('cost', 0.0) for e in recent_executions
            if e.resource_usage
        ]
        baseline['average_cost'] = statistics.mean(costs) if costs else 0.0
        
        # Average duration
        durations = [
            e.total_duration_seconds for e in recent_executions
            if e.total_duration_seconds and e.total_duration_seconds > 0
        ]
        baseline['average_duration'] = statistics.mean(durations) if durations else 0.0
        
        # Average quality (based on error rates)
        quality_scores = []
        for execution in recent_executions:
            error_count = len(execution.errors)
            quality = max(0.0, 1.0 - (error_count * 0.1))
            quality_scores.append(quality)
        baseline['average_quality'] = statistics.mean(quality_scores) if quality_scores else 0.0
        
        return baseline
    
    async def _generate_optimized_teams(
        self,
        role_effectiveness: Dict[str, RoleEffectiveness],
        patterns: List[TeamPattern],
        objective: OptimizationObjective,
        constraints: OptimizationConstraints,
    ) -> List[TeamComposition]:
        """Generate optimized team compositions."""
        
        recommendations = []
        
        # Strategy 1: Use proven high-performing patterns
        proven_patterns = [
            p for p in patterns
            if p.confidence_score >= self.pattern_confidence_threshold
            and p.usage_count >= 5
        ]
        
        # Sort patterns by objective
        if objective == OptimizationObjective.COST_EFFICIENCY:
            proven_patterns.sort(key=lambda p: p.average_cost)
        elif objective == OptimizationObjective.PERFORMANCE:
            proven_patterns.sort(key=lambda p: p.success_rate, reverse=True)
        elif objective == OptimizationObjective.QUALITY:
            proven_patterns.sort(key=lambda p: p.quality_score, reverse=True)
        elif objective == OptimizationObjective.SPEED:
            proven_patterns.sort(key=lambda p: p.average_duration)
        else:  # BALANCED or other
            proven_patterns.sort(key=lambda p: p.effectiveness_score, reverse=True)
        
        # Convert top patterns to team compositions
        for pattern in proven_patterns[:3]:  # Top 3 patterns
            team_comp = await self._pattern_to_team_composition(pattern, role_effectiveness)
            if team_comp and self._meets_constraints(team_comp, constraints):
                recommendations.append(team_comp)
        
        # Strategy 2: Generate new combinations based on top-performing roles
        top_roles = sorted(
            role_effectiveness.values(),
            key=lambda r: r.overall_effectiveness,
            reverse=True
        )[:8]  # Top 8 roles
        
        new_combinations = await self._generate_role_combinations(
            top_roles, objective, constraints
        )
        
        for combination in new_combinations[:2]:  # Top 2 new combinations
            recommendations.append(combination)
        
        return recommendations
    
    async def _project_improvements(
        self,
        recommended_teams: List[TeamComposition],
        baseline: Dict[str, float],
        objective: OptimizationObjective,
    ) -> Dict[str, float]:
        """Project improvements from recommended team compositions."""
        
        improvements = {}
        
        if not recommended_teams or not baseline:
            return improvements
        
        # Estimate improvements based on historical performance of similar teams
        for i, team in enumerate(recommended_teams):
            team_key = f"team_{i + 1}"
            
            # Find matching pattern for projection
            roles = tuple(sorted(agent.role for agent in team.primary_team))
            coordination = team.coordination_strategy
            pattern_key = f"{'-'.join(roles)}_{coordination.value}"
            
            if pattern_key in self.known_patterns:
                pattern = self.known_patterns[pattern_key]
                
                # Project improvements
                if objective == OptimizationObjective.COST_EFFICIENCY:
                    cost_improvement = max(0, baseline.get('average_cost', 0) - pattern.average_cost)
                    improvements[f"{team_key}_cost_savings"] = cost_improvement
                
                success_improvement = pattern.success_rate - baseline.get('success_rate', 0)
                improvements[f"{team_key}_success_rate_improvement"] = success_improvement
                
                quality_improvement = pattern.quality_score - baseline.get('average_quality', 0)
                improvements[f"{team_key}_quality_improvement"] = quality_improvement
        
        return improvements
    
    async def _generate_optimization_recommendations(
        self,
        role_effectiveness: Dict[str, RoleEffectiveness],
        patterns: List[TeamPattern],
        projected_improvements: Dict[str, float],
    ) -> List[str]:
        """Generate specific optimization recommendations."""
        
        recommendations = []
        
        # Role-based recommendations
        low_performing_roles = [
            name for name, effectiveness in role_effectiveness.items()
            if effectiveness.overall_effectiveness < 0.6
        ]
        
        if low_performing_roles:
            recommendations.append(
                f"Consider reviewing or replacing underperforming roles: {', '.join(low_performing_roles)}"
            )
        
        high_performing_roles = [
            name for name, effectiveness in role_effectiveness.items()
            if effectiveness.overall_effectiveness > 0.8
        ]
        
        if high_performing_roles:
            recommendations.append(
                f"Prioritize high-performing roles in future teams: {', '.join(high_performing_roles[:3])}"
            )
        
        # Pattern-based recommendations
        if patterns:
            best_pattern = max(patterns, key=lambda p: p.effectiveness_score)
            recommendations.append(
                f"Most effective team pattern: {'-'.join(best_pattern.roles)} "
                f"with {best_pattern.coordination_strategy.value} coordination"
            )
        
        # Size-based recommendations
        if patterns:
            size_performance = defaultdict(list)
            for pattern in patterns:
                size_performance[pattern.team_size].append(pattern.effectiveness_score)
            
            best_size = max(
                size_performance.items(),
                key=lambda x: statistics.mean(x[1])
            )[0]
            
            recommendations.append(f"Optimal team size appears to be {best_size} members")
        
        # Coordination strategy recommendations
        if patterns:
            strategy_performance = defaultdict(list)
            for pattern in patterns:
                strategy_performance[pattern.coordination_strategy].append(pattern.effectiveness_score)
            
            if strategy_performance:
                best_strategy = max(
                    strategy_performance.items(),
                    key=lambda x: statistics.mean(x[1])
                )[0]
                
                recommendations.append(
                    f"Most effective coordination strategy: {best_strategy.value}"
                )
        
        return recommendations
    
    async def _pattern_to_team_composition(
        self,
        pattern: TeamPattern,
        role_effectiveness: Dict[str, RoleEffectiveness],
    ) -> Optional[TeamComposition]:
        """Convert a team pattern to a team composition."""
        
        primary_team = []
        for role in pattern.roles:
            # Get model assignment based on role effectiveness
            effectiveness = role_effectiveness.get(role)
            if effectiveness:
                model_assignment = self._determine_best_model_for_role(role, effectiveness)
            else:
                model_assignment = "default"
            
            agent_spec = AgentSpec(
                role=role,
                model_assignment=model_assignment,
                priority=self._calculate_role_priority(role, role_effectiveness),
            )
            primary_team.append(agent_spec)
        
        return TeamComposition(
            primary_team=primary_team,
            load_order=list(pattern.roles),
            coordination_strategy=pattern.coordination_strategy,
            confidence_score=pattern.confidence_score,
            rationale=f"Based on pattern with {pattern.usage_count} successful executions "
                     f"and {pattern.success_rate:.1%} success rate",
        )
    
    async def _generate_role_combinations(
        self,
        top_roles: List[RoleEffectiveness],
        objective: OptimizationObjective,
        constraints: OptimizationConstraints,
    ) -> List[TeamComposition]:
        """Generate new role combinations based on top-performing roles."""
        
        combinations = []
        
        # Try different team sizes within constraints
        min_size = max(1, constraints.min_team_size)
        max_size = min(len(top_roles), constraints.max_team_size)
        
        for team_size in range(min_size, min(max_size + 1, 6)):  # Limit to reasonable sizes
            # Select top roles for this size
            selected_roles = top_roles[:team_size]
            
            # Determine best coordination strategy
            if team_size <= 2:
                coordination = CoordinationStrategy.SEQUENTIAL
            elif objective == OptimizationObjective.SPEED:
                coordination = CoordinationStrategy.PARALLEL
            else:
                coordination = CoordinationStrategy.COLLABORATIVE
            
            # Create team composition
            primary_team = []
            for i, role_eff in enumerate(selected_roles):
                model_assignment = self._determine_best_model_for_role(
                    role_eff.role_name, role_eff
                )
                
                agent_spec = AgentSpec(
                    role=role_eff.role_name,
                    model_assignment=model_assignment,
                    priority=i + 1,
                )
                primary_team.append(agent_spec)
            
            team_comp = TeamComposition(
                primary_team=primary_team,
                load_order=[role.role_name for role in selected_roles],
                coordination_strategy=coordination,
                confidence_score=0.7,  # Lower confidence for new combinations
                rationale=f"Generated combination of top {team_size} performing roles",
            )
            
            combinations.append(team_comp)
        
        return combinations
    
    def _meets_constraints(
        self, team_comp: TeamComposition, constraints: OptimizationConstraints
    ) -> bool:
        """Check if team composition meets the given constraints."""
        
        team_size = len(team_comp.primary_team)
        
        # Size constraints
        if team_size < constraints.min_team_size or team_size > constraints.max_team_size:
            return False
        
        # Role constraints
        team_roles = {agent.role for agent in team_comp.primary_team}
        
        # Required roles
        if constraints.required_roles and not constraints.required_roles.issubset(team_roles):
            return False
        
        # Forbidden roles
        if constraints.forbidden_roles and team_roles.intersection(constraints.forbidden_roles):
            return False
        
        return True
    
    def _calculate_team_performance_score(self, execution: TeamExecution) -> float:
        """Calculate performance score for a team execution."""
        
        if execution.status == ExecutionStatus.COMPLETED:
            base_score = 1.0
        elif execution.status == ExecutionStatus.FAILED:
            base_score = 0.0
        else:
            base_score = 0.5
        
        # Adjust for quality (error count)
        error_penalty = len(execution.errors) * 0.1
        quality_score = max(0.0, base_score - error_penalty)
        
        # Adjust for efficiency (if duration available)
        if execution.total_duration_seconds and execution.total_duration_seconds > 0:
            # Assume 300 seconds (5 minutes) is optimal
            efficiency_factor = min(1.0, 300.0 / execution.total_duration_seconds)
            quality_score *= (0.8 + 0.2 * efficiency_factor)  # Weight efficiency 20%
        
        return min(1.0, quality_score)
    
    def _estimate_role_collaboration_score(self, role: str, execution: TeamExecution) -> float:
        """Estimate collaboration score for a role in an execution."""
        
        # Simple heuristic based on team size and coordination strategy
        team_size = len(execution.team_composition.primary_team)
        coordination = execution.team_composition.coordination_strategy
        
        base_score = 0.7  # Default neutral score
        
        # Adjust based on coordination strategy
        if coordination == CoordinationStrategy.COLLABORATIVE:
            base_score += 0.2
        elif coordination == CoordinationStrategy.PARALLEL:
            base_score += 0.1
        
        # Adjust based on team success
        if execution.status == ExecutionStatus.COMPLETED:
            base_score += 0.1
        
        # Penalize very large teams (coordination difficulty)
        if team_size > 6:
            base_score -= 0.1
        
        return min(1.0, max(0.0, base_score))
    
    def _determine_best_model_for_role(self, role: str, effectiveness: RoleEffectiveness) -> str:
        """Determine the best model assignment for a role."""
        
        # Simple logic based on role type and effectiveness
        if effectiveness.overall_effectiveness > 0.8:
            return "premium"  # Use best model for high-performing roles
        elif effectiveness.overall_effectiveness > 0.6:
            return "standard"
        else:
            return "basic"
    
    def _calculate_role_priority(
        self, role: str, role_effectiveness: Dict[str, RoleEffectiveness]
    ) -> int:
        """Calculate priority for a role (1 = highest priority)."""
        
        effectiveness = role_effectiveness.get(role)
        if not effectiveness:
            return 5
        
        # Higher effectiveness = higher priority (lower number)
        if effectiveness.overall_effectiveness > 0.8:
            return 1
        elif effectiveness.overall_effectiveness > 0.6:
            return 2
        elif effectiveness.overall_effectiveness > 0.4:
            return 3
        else:
            return 4
    
    async def _calculate_optimization_confidence(
        self, results: OptimizationResults, data_points: int
    ) -> float:
        """Calculate confidence score for optimization results."""
        
        confidence_factors = []
        
        # Data sufficiency
        if data_points >= 50:
            confidence_factors.append(1.0)
        elif data_points >= 20:
            confidence_factors.append(0.8)
        elif data_points >= 10:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        # Pattern strength
        if results.identified_patterns:
            avg_pattern_confidence = statistics.mean(
                p.confidence_score for p in results.identified_patterns
            )
            confidence_factors.append(avg_pattern_confidence)
        
        # Role effectiveness data quality
        if results.role_effectiveness:
            avg_role_confidence = statistics.mean(
                min(1.0, eff.total_executions / 10.0)
                for eff in results.role_effectiveness.values()
            )
            confidence_factors.append(avg_role_confidence)
        
        # Recommendation consistency
        if len(results.recommended_teams) >= 3:
            confidence_factors.append(0.9)
        elif len(results.recommended_teams) >= 1:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.5
    
    # Public API methods
    
    def get_optimization_history(self, limit: Optional[int] = None) -> List[OptimizationResults]:
        """Get optimization history with optional limit."""
        if limit:
            return self.optimization_history[-limit:]
        return self.optimization_history.copy()
    
    def get_known_patterns(self) -> Dict[str, TeamPattern]:
        """Get all known team patterns."""
        return self.known_patterns.copy()
    
    def get_role_effectiveness_cache(self) -> Dict[str, RoleEffectiveness]:
        """Get cached role effectiveness data."""
        return self.role_effectiveness_cache.copy()
    
    async def add_execution_result(self, execution: TeamExecution) -> None:
        """Add new execution result for continuous learning."""
        # This would typically trigger incremental pattern updates
        # For now, just log the addition
        self.log.debug("Added execution result: %s", execution.execution_id)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "total_optimizations": len(self.optimization_history),
            "known_patterns": len(self.known_patterns),
            "cached_role_effectiveness": len(self.role_effectiveness_cache),
            "last_optimization": self.last_optimization_time.isoformat() if self.last_optimization_time else None,
            "pattern_confidence_threshold": self.pattern_confidence_threshold,
            "statistical_significance_threshold": self.statistical_significance_threshold,
        }