"""Intelligent team composition for optimal agent team creation.

This module provides the TeamComposer class that creates optimal agent teams
based on task classification, available resources, and historical performance data.
It supports multiple composition strategies and provides fallback mechanisms.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional
import logging

from .models import (
    AgentSpec,
    ResourceConstraints,
    TaskClassification,
    TeamComposition,
    TeamPerformanceMetrics,
    ComplexityLevel,
    RiskLevel,
    TeamCompositionError,
    InsufficientResources,
    NoSuitableAgents,
)
from .composition_strategies import (
    CompositionStrategy,
    MinimalTeamStrategy,
    FullStackStrategy,
    CostOptimizedStrategy,
    PerformanceOptimizedStrategy,
)

logger = logging.getLogger(__name__)


class TeamComposer:
    """Intelligent team composition engine.
    
    Creates optimal agent teams based on task classification, resource constraints,
    and historical performance data. Supports multiple composition strategies and
    provides fallback mechanisms when primary strategies fail.
    """
    
    def __init__(self):
        """Initialize the team composer with available strategies."""
        self._strategies = {
            'minimal': MinimalTeamStrategy(),
            'full_stack': FullStackStrategy(), 
            'cost_optimized': CostOptimizedStrategy(),
            'performance_optimized': PerformanceOptimizedStrategy(),
        }
        self._default_strategy = 'performance_optimized'
    
    def compose_team(
        self,
        classification: TaskClassification,
        available_roles: List[str],
        resource_limits: ResourceConstraints,
        historical_performance: Optional[Dict[str, TeamPerformanceMetrics]] = None
    ) -> TeamComposition:
        """Compose an optimal team based on task classification and constraints.
        
        Args:
            classification: Task classification results
            available_roles: List of available role names
            resource_limits: Resource constraints for team composition
            historical_performance: Historical performance metrics by team composition
            
        Returns:
            TeamComposition with primary team, fallback agents, and metadata
            
        Raises:
            TeamCompositionError: If team composition fails
            InsufficientResources: If resource constraints cannot be met
            NoSuitableAgents: If no suitable agents are available
        """
        start_time = time.time()
        
        if not available_roles:
            raise NoSuitableAgents("No roles available for team composition")
        
        if historical_performance is None:
            historical_performance = {}
        
        # Validate resource constraints
        self._validate_constraints(resource_limits)
        
        # Choose primary strategy based on task characteristics
        strategy_name = self._select_strategy(classification, resource_limits)
        strategy = self._strategies[strategy_name]
        
        logger.info(f"Using {strategy_name} strategy for task: {classification.task_type}")
        
        try:
            # Attempt primary composition
            composition = strategy.compose_team(
                classification=classification,
                available_roles=available_roles,
                resource_limits=resource_limits,
                historical_performance=historical_performance
            )
            
            # Validate composition
            self._validate_composition(composition, resource_limits)
            
            # Check if we need fallback due to constraints
            if self._needs_fallback(composition, resource_limits):
                logger.warning("Primary team exceeds constraints, creating fallback")
                composition = self._create_fallback_team(
                    classification=classification,
                    available_roles=available_roles,
                    resource_limits=resource_limits,
                    historical_performance=historical_performance,
                    primary_composition=composition
                )
            
            # Add performance metadata
            composition_time = time.time() - start_time
            composition.rationale += f" (composed in {composition_time:.3f}s using {strategy_name})"
            
            logger.info(f"Successfully composed team with {len(composition.primary_team)} agents")
            return composition
            
        except Exception as e:
            logger.error(f"Primary strategy {strategy_name} failed: {e}")
            # Try fallback strategies
            return self._attempt_fallback_strategies(
                classification=classification,
                available_roles=available_roles,
                resource_limits=resource_limits,
                historical_performance=historical_performance,
                failed_strategy=strategy_name,
                start_time=start_time
            )
    
    def _validate_constraints(self, resource_limits: ResourceConstraints) -> None:
        """Validate that resource constraints are reasonable."""
        if resource_limits.max_agents < 1:
            raise InsufficientResources("max_agents must be at least 1")
        
        if resource_limits.parallel_limit < 1:
            raise InsufficientResources("parallel_limit must be at least 1")
        
        if (resource_limits.cost_budget is not None and 
            resource_limits.cost_budget < 0.01):  # Minimum viable budget
            raise InsufficientResources("cost_budget too low for any agent execution")
    
    def _select_strategy(
        self,
        classification: TaskClassification,
        resource_limits: ResourceConstraints
    ) -> str:
        """Select the most appropriate composition strategy."""
        # Cost-constrained tasks
        if (resource_limits.cost_budget is not None and 
            resource_limits.cost_budget < 0.5):
            return 'cost_optimized'
        
        # High-risk or critical tasks need full coverage
        if (classification.complexity == ComplexityLevel.CRITICAL or
            classification.risk_level == RiskLevel.CRITICAL or
            len(classification.required_roles) > 3):
            return 'full_stack'
        
        # Simple tasks with tight resource limits
        if (classification.complexity in [ComplexityLevel.TRIVIAL, ComplexityLevel.LOW] and
            resource_limits.max_agents <= 2):
            return 'minimal'
        
        # Default to performance-optimized for most cases
        return 'performance_optimized'
    
    def _validate_composition(
        self,
        composition: TeamComposition,
        resource_limits: ResourceConstraints
    ) -> None:
        """Validate that composition meets basic requirements."""
        if not composition.primary_team:
            raise TeamCompositionError("Primary team cannot be empty")
        
        if len(composition.primary_team) > resource_limits.max_agents:
            raise InsufficientResources(
                f"Team size {len(composition.primary_team)} exceeds max_agents {resource_limits.max_agents}"
            )
        
        # Validate load order contains all primary team roles
        team_roles = {agent.role for agent in composition.primary_team}
        load_order_roles = set(composition.load_order)
        
        if team_roles != load_order_roles:
            raise TeamCompositionError(
                f"Load order roles {load_order_roles} don't match team roles {team_roles}"
            )
    
    def _needs_fallback(
        self,
        composition: TeamComposition,
        resource_limits: ResourceConstraints
    ) -> bool:
        """Check if composition exceeds resource constraints requiring fallback."""
        # Check cost constraints
        if (resource_limits.cost_budget is not None and
            composition.estimated_cost is not None and
            composition.estimated_cost > resource_limits.cost_budget):
            return True
        
        # Check parallel execution limits
        parallel_agents = min(len(composition.primary_team), resource_limits.parallel_limit)
        if parallel_agents > resource_limits.parallel_limit:
            return True
        
        # Check time constraints (if we had estimated duration)
        # This could be implemented with duration estimation in the future
        
        return False
    
    def _create_fallback_team(
        self,
        classification: TaskClassification,
        available_roles: List[str],
        resource_limits: ResourceConstraints,
        historical_performance: Dict[str, TeamPerformanceMetrics],
        primary_composition: TeamComposition
    ) -> TeamComposition:
        """Create a fallback team that meets constraints."""
        # Try minimal strategy first for resource-constrained fallback
        minimal_strategy = self._strategies['minimal']
        
        # Reduce constraints for fallback
        fallback_limits = ResourceConstraints(
            max_agents=min(resource_limits.max_agents, 3),
            memory_limit=resource_limits.memory_limit,
            time_budget=resource_limits.time_budget,
            cost_budget=resource_limits.cost_budget,
            parallel_limit=min(resource_limits.parallel_limit, 2)
        )
        
        try:
            fallback_composition = minimal_strategy.compose_team(
                classification=classification,
                available_roles=available_roles,
                resource_limits=fallback_limits,
                historical_performance=historical_performance
            )
            
            # Use primary team's unused agents as fallback agents
            primary_roles = {agent.role for agent in fallback_composition.primary_team}
            fallback_agents = [
                agent for agent in primary_composition.primary_team 
                if agent.role not in primary_roles
            ]
            
            fallback_composition.fallback_agents = fallback_agents
            fallback_composition.rationale = (
                "Fallback team composition due to resource constraints. " +
                fallback_composition.rationale
            )
            
            return fallback_composition
            
        except Exception as e:
            logger.error(f"Fallback team creation failed: {e}")
            # Last resort: single agent team
            return self._create_emergency_team(available_roles, resource_limits)
    
    def _attempt_fallback_strategies(
        self,
        classification: TaskClassification,
        available_roles: List[str],
        resource_limits: ResourceConstraints,
        historical_performance: Dict[str, TeamPerformanceMetrics],
        failed_strategy: str,
        start_time: float
    ) -> TeamComposition:
        """Attempt fallback strategies when primary strategy fails."""
        
        # Try strategies in order of preference (excluding failed one)
        strategy_order = ['minimal', 'cost_optimized', 'performance_optimized', 'full_stack']
        strategy_order = [s for s in strategy_order if s != failed_strategy]
        
        for strategy_name in strategy_order:
            try:
                logger.info(f"Attempting fallback strategy: {strategy_name}")
                strategy = self._strategies[strategy_name]
                
                composition = strategy.compose_team(
                    classification=classification,
                    available_roles=available_roles,
                    resource_limits=resource_limits,
                    historical_performance=historical_performance
                )
                
                self._validate_composition(composition, resource_limits)
                
                # Add fallback metadata
                composition_time = time.time() - start_time
                composition.rationale += f" (fallback strategy after {failed_strategy} failed, composed in {composition_time:.3f}s)"
                composition.confidence_score *= 0.8  # Reduce confidence for fallback
                
                logger.info(f"Fallback strategy {strategy_name} succeeded")
                return composition
                
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy_name} failed: {e}")
                continue
        
        # All strategies failed, create emergency team
        logger.error("All composition strategies failed, creating emergency team")
        return self._create_emergency_team(available_roles, resource_limits)
    
    def _create_emergency_team(
        self,
        available_roles: List[str],
        resource_limits: ResourceConstraints
    ) -> TeamComposition:
        """Create a minimal emergency team when all strategies fail."""
        # Find the most basic role available
        priority_roles = ['coder', 'architect', 'qa']
        selected_role = None
        
        for role in priority_roles:
            if role in available_roles:
                selected_role = role
                break
        
        if not selected_role:
            # Take the first available role
            selected_role = available_roles[0] if available_roles else 'coder'
        
        emergency_agent = AgentSpec(
            role=selected_role,
            model_assignment='ollama',  # Most reliable/cost-effective
            priority=1,
            specializations=[]
        )
        
        return TeamComposition(
            primary_team=[emergency_agent],
            fallback_agents=[],
            load_order=[selected_role],
            coordination_strategy='sequential',
            estimated_cost=0.0,  # Assuming ollama is free
            confidence_score=0.3,  # Low confidence for emergency team
            rationale="Emergency team composition - all strategic approaches failed"
        )
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available composition strategies."""
        return list(self._strategies.keys())
    
    def set_default_strategy(self, strategy_name: str) -> None:
        """Set the default composition strategy."""
        if strategy_name not in self._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        self._default_strategy = strategy_name
    
    def add_custom_strategy(self, name: str, strategy: CompositionStrategy) -> None:
        """Add a custom composition strategy."""
        self._strategies[name] = strategy
    
    def estimate_team_cost(
        self,
        classification: TaskClassification,
        team_roles: List[str],
        complexity_override: Optional[ComplexityLevel] = None
    ) -> float:
        """Estimate the cost of a specific team composition.
        
        This is a utility method that can be used for cost planning.
        """
        complexity = complexity_override or classification.complexity
        
        # Use the cost estimation from strategies
        strategy = self._strategies['cost_optimized']  # Has good cost estimation
        
        # Create dummy team specs
        team = [
            AgentSpec(
                role=role,
                model_assignment='codex',  # Default for estimation
                priority=1,
                specializations=[]
            )
            for role in team_roles
        ]
        
        return strategy._estimate_cost(team, complexity)