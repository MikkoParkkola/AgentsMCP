"""Team composition strategies for optimal agent team creation.

This module implements different strategies for composing agent teams based on
task classification, resource constraints, and historical performance data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set
import logging

from .models import (
    AgentSpec,
    CoordinationStrategy,
    ResourceConstraints,
    TaskClassification,
    TeamComposition,
    TeamPerformanceMetrics,
    ComplexityLevel,
    RiskLevel,
    TaskType,
    TechnologyStack,
)
from ..roles.base import RoleName

logger = logging.getLogger(__name__)


class CompositionStrategy(ABC):
    """Base class for team composition strategies."""
    
    @abstractmethod
    def compose_team(
        self,
        classification: TaskClassification,
        available_roles: List[str],
        resource_limits: ResourceConstraints,
        historical_performance: Dict[str, TeamPerformanceMetrics]
    ) -> TeamComposition:
        """Compose a team based on the given parameters."""
        pass
    
    def _calculate_confidence(
        self,
        classification: TaskClassification,
        team: List[AgentSpec],
        historical_performance: Dict[str, TeamPerformanceMetrics]
    ) -> float:
        """Calculate confidence score for team composition."""
        base_confidence = classification.confidence
        
        # Adjust based on role coverage
        required_roles = set(classification.required_roles)
        team_roles = {agent.role for agent in team}
        coverage = len(required_roles & team_roles) / max(len(required_roles), 1)
        
        # Adjust based on historical performance
        perf_boost = 0.0
        team_key = "-".join(sorted(team_roles))
        if team_key in historical_performance:
            metrics = historical_performance[team_key]
            if metrics.total_executions >= 3:  # Require some history
                perf_boost = (metrics.success_rate - 0.5) * 0.2  # -0.1 to +0.1
        
        confidence = min(1.0, base_confidence * coverage + perf_boost)
        return max(0.1, confidence)  # Minimum confidence
    
    def _estimate_cost(self, team: List[AgentSpec], task_complexity: ComplexityLevel) -> float:
        """Estimate execution cost based on team composition and complexity."""
        base_costs = {
            "codex": 0.15,      # EUR per moderate task
            "claude": 0.25,     # More expensive but capable
            "ollama": 0.0,      # Local execution
        }
        
        complexity_multipliers = {
            ComplexityLevel.TRIVIAL: 0.3,
            ComplexityLevel.LOW: 0.6,
            ComplexityLevel.MEDIUM: 1.0,
            ComplexityLevel.HIGH: 2.0,
            ComplexityLevel.CRITICAL: 3.5,
        }
        
        total_cost = 0.0
        multiplier = complexity_multipliers.get(task_complexity, 1.0)
        
        for agent in team:
            agent_cost = base_costs.get(agent.model_assignment, 0.1)
            total_cost += agent_cost * multiplier
        
        return total_cost
    
    def _determine_load_order(
        self,
        team: List[AgentSpec],
        classification: TaskClassification
    ) -> List[str]:
        """Determine optimal loading order based on dependencies."""
        # Define role dependencies (who depends on whom)
        dependencies = {
            RoleName.ARCHITECT.value: [],  # Independent
            RoleName.BUSINESS_ANALYST.value: [],  # Independent
            RoleName.CODER.value: [RoleName.ARCHITECT.value],  # Needs architecture
            RoleName.BACKEND_ENGINEER.value: [RoleName.ARCHITECT.value],
            RoleName.WEB_FRONTEND_ENGINEER.value: [RoleName.ARCHITECT.value],
            RoleName.API_ENGINEER.value: [RoleName.ARCHITECT.value],
            RoleName.TUI_FRONTEND_ENGINEER.value: [RoleName.ARCHITECT.value],
            RoleName.QA.value: [RoleName.CODER.value, RoleName.BACKEND_ENGINEER.value],
            RoleName.BACKEND_QA_ENGINEER.value: [RoleName.BACKEND_ENGINEER.value],
            RoleName.WEB_FRONTEND_QA_ENGINEER.value: [RoleName.WEB_FRONTEND_ENGINEER.value],
            RoleName.TUI_FRONTEND_QA_ENGINEER.value: [RoleName.TUI_FRONTEND_ENGINEER.value],
            RoleName.CHIEF_QA_ENGINEER.value: [RoleName.QA.value],
            RoleName.MERGE_BOT.value: [RoleName.QA.value],  # Needs QA approval
            RoleName.DOCS.value: [RoleName.CODER.value],  # Needs implemented code
            RoleName.CI_CD_ENGINEER.value: [RoleName.CODER.value],
            RoleName.DEV_TOOLING_ENGINEER.value: [],  # Independent
            RoleName.DATA_ANALYST.value: [],  # Independent
            RoleName.DATA_SCIENTIST.value: [RoleName.DATA_ANALYST.value],
            RoleName.ML_SCIENTIST.value: [RoleName.DATA_SCIENTIST.value],
            RoleName.ML_ENGINEER.value: [RoleName.ML_SCIENTIST.value],
            RoleName.IT_LAWYER.value: [],  # Independent
            RoleName.MARKETING_MANAGER.value: [RoleName.CODER.value],  # Needs product
        }
        
        team_roles = {agent.role for agent in team}
        ordered_roles = []
        processed = set()
        
        def process_role(role: str):
            if role in processed or role not in team_roles:
                return
            
            # Process dependencies first
            for dep in dependencies.get(role, []):
                if dep in team_roles:
                    process_role(dep)
            
            ordered_roles.append(role)
            processed.add(role)
        
        # Process all team roles
        for agent in sorted(team, key=lambda x: x.priority):
            process_role(agent.role)
        
        return ordered_roles
    
    def _determine_coordination_strategy(
        self,
        classification: TaskClassification,
        team: List[AgentSpec]
    ) -> CoordinationStrategy:
        """Determine the best coordination strategy for the team."""
        if len(team) == 1:
            return CoordinationStrategy.SEQUENTIAL
        
        # For high complexity/risk, use hierarchical coordination
        if (classification.complexity in [ComplexityLevel.HIGH, ComplexityLevel.CRITICAL] or
            classification.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]):
            return CoordinationStrategy.HIERARCHICAL
        
        # For design/architecture tasks, use collaborative
        if classification.task_type in [TaskType.DESIGN, TaskType.ANALYSIS]:
            return CoordinationStrategy.COLLABORATIVE
        
        # For implementation with multiple engineers, use pipeline
        engineer_count = sum(1 for agent in team if 'engineer' in agent.role.lower())
        if engineer_count > 1:
            return CoordinationStrategy.PIPELINE
        
        # Default to parallel for simple tasks
        return CoordinationStrategy.PARALLEL


class MinimalTeamStrategy(CompositionStrategy):
    """Compose the bare minimum team needed for task completion."""
    
    def compose_team(
        self,
        classification: TaskClassification,
        available_roles: List[str],
        resource_limits: ResourceConstraints,
        historical_performance: Dict[str, TeamPerformanceMetrics]
    ) -> TeamComposition:
        """Create minimal viable team."""
        team = []
        available_set = set(available_roles)
        
        # Always include required roles if available
        for role in classification.required_roles:
            if role in available_set and len(team) < resource_limits.max_agents:
                model_assignment = self._choose_model(role, classification)
                team.append(AgentSpec(
                    role=role,
                    model_assignment=model_assignment,
                    priority=1,
                    specializations=self._get_specializations(role, classification.technologies)
                ))
        
        # If no required roles, add a coder as default
        if not team and RoleName.CODER.value in available_set:
            team.append(AgentSpec(
                role=RoleName.CODER.value,
                model_assignment="ollama",  # Cost-effective for minimal team
                priority=1,
                specializations=[]
            ))
        
        load_order = self._determine_load_order(team, classification)
        coordination = self._determine_coordination_strategy(classification, team)
        
        return TeamComposition(
            primary_team=team,
            fallback_agents=[],
            load_order=load_order,
            coordination_strategy=coordination,
            estimated_cost=self._estimate_cost(team, classification.complexity),
            confidence_score=self._calculate_confidence(classification, team, historical_performance),
            rationale="Minimal team composition focusing on essential roles only"
        )
    
    def _choose_model(self, role: str, classification: TaskClassification) -> str:
        """Choose model assignment for minimal cost."""
        # Use local model by default for cost efficiency
        if classification.complexity in [ComplexityLevel.HIGH, ComplexityLevel.CRITICAL]:
            return "codex"  # Need capability for complex tasks
        return "ollama"  # Cost-effective default
    
    def _get_specializations(self, role: str, technologies: List[TechnologyStack]) -> List[str]:
        """Get relevant specializations for a role."""
        return [tech.value for tech in technologies]


class FullStackStrategy(CompositionStrategy):
    """Compose comprehensive team with complete coverage for complex tasks."""
    
    def compose_team(
        self,
        classification: TaskClassification,
        available_roles: List[str],
        resource_limits: ResourceConstraints,
        historical_performance: Dict[str, TeamPerformanceMetrics]
    ) -> TeamComposition:
        """Create comprehensive team with full coverage."""
        team = []
        fallback_agents = []
        available_set = set(available_roles)
        
        # Core team roles
        core_roles = [
            RoleName.ARCHITECT.value,
            RoleName.CODER.value,
            RoleName.QA.value,
        ]
        
        # Technology-specific roles
        tech_roles = self._get_tech_specific_roles(classification.technologies)
        
        # Priority order: required -> core -> tech-specific -> optional
        role_priority = (
            classification.required_roles +
            [r for r in core_roles if r not in classification.required_roles] +
            [r for r in tech_roles if r not in classification.required_roles and r not in core_roles] +
            classification.optional_roles
        )
        
        added_roles = set()
        for role in role_priority:
            if (role in available_set and 
                role not in added_roles and 
                len(team) < resource_limits.max_agents):
                
                model_assignment = self._choose_model(role, classification)
                priority = self._calculate_priority(role, classification)
                
                team.append(AgentSpec(
                    role=role,
                    model_assignment=model_assignment,
                    priority=priority,
                    specializations=self._get_specializations(role, classification.technologies)
                ))
                added_roles.add(role)
        
        # Add fallback agents for high-priority roles
        for role in classification.required_roles:
            if role not in added_roles and role in available_set:
                fallback_agents.append(AgentSpec(
                    role=role,
                    model_assignment="ollama",  # Cost-effective fallback
                    priority=5,
                    specializations=[]
                ))
        
        load_order = self._determine_load_order(team, classification)
        coordination = self._determine_coordination_strategy(classification, team)
        
        return TeamComposition(
            primary_team=team,
            fallback_agents=fallback_agents,
            load_order=load_order,
            coordination_strategy=coordination,
            estimated_cost=self._estimate_cost(team, classification.complexity),
            confidence_score=self._calculate_confidence(classification, team, historical_performance),
            rationale="Full-stack team composition with comprehensive coverage and fallback options"
        )
    
    def _get_tech_specific_roles(self, technologies: List[TechnologyStack]) -> List[str]:
        """Get technology-specific roles."""
        tech_roles = []
        
        for tech in technologies:
            if tech in [TechnologyStack.REACT, TechnologyStack.JAVASCRIPT, TechnologyStack.TYPESCRIPT]:
                tech_roles.append(RoleName.WEB_FRONTEND_ENGINEER.value)
            elif tech == TechnologyStack.TUI:
                tech_roles.append(RoleName.TUI_FRONTEND_ENGINEER.value)
            elif tech in [TechnologyStack.API, TechnologyStack.NODEJS]:
                tech_roles.append(RoleName.API_ENGINEER.value)
            elif tech in [TechnologyStack.PYTHON, TechnologyStack.DATABASE]:
                tech_roles.append(RoleName.BACKEND_ENGINEER.value)
            elif tech == TechnologyStack.DEVOPS:
                tech_roles.append(RoleName.CI_CD_ENGINEER.value)
            elif tech in [TechnologyStack.MACHINE_LEARNING, TechnologyStack.DATA_ANALYSIS]:
                tech_roles.extend([RoleName.DATA_SCIENTIST.value, RoleName.ML_ENGINEER.value])
        
        return list(set(tech_roles))  # Remove duplicates
    
    def _choose_model(self, role: str, classification: TaskClassification) -> str:
        """Choose optimal model for full-stack deployment."""
        # Use best available models for comprehensive coverage
        if role in [RoleName.ARCHITECT.value, RoleName.CHIEF_QA_ENGINEER.value]:
            return "codex"  # Complex reasoning required
        elif classification.complexity == ComplexityLevel.CRITICAL:
            return "codex"  # High-stakes tasks need best model
        elif role in [RoleName.MERGE_BOT.value, RoleName.DEV_TOOLING_ENGINEER.value]:
            return "ollama"  # Automation tasks
        else:
            return "codex"  # Default to capable model
    
    def _calculate_priority(self, role: str, classification: TaskClassification) -> int:
        """Calculate priority based on role importance."""
        if role in classification.required_roles:
            return 1
        elif role == RoleName.ARCHITECT.value:
            return 2
        elif role == RoleName.QA.value:
            return 3
        else:
            return 4
    
    def _get_specializations(self, role: str, technologies: List[TechnologyStack]) -> List[str]:
        """Get comprehensive specializations."""
        all_specs = [tech.value for tech in technologies]
        
        # Add role-specific specializations
        if "frontend" in role.lower():
            all_specs.extend(["ui_ux", "accessibility", "responsive_design"])
        elif "backend" in role.lower():
            all_specs.extend(["scalability", "security", "performance"])
        elif "qa" in role.lower():
            all_specs.extend(["automated_testing", "security_testing", "performance_testing"])
        
        return list(set(all_specs))


class CostOptimizedStrategy(CompositionStrategy):
    """Compose team optimized for cost efficiency within budget constraints."""
    
    def compose_team(
        self,
        classification: TaskClassification,
        available_roles: List[str],
        resource_limits: ResourceConstraints,
        historical_performance: Dict[str, TeamPerformanceMetrics]
    ) -> TeamComposition:
        """Create cost-optimized team."""
        team = []
        available_set = set(available_roles)
        
        # Sort roles by cost-effectiveness (required roles first)
        role_candidates = []
        
        # Add required roles with high priority
        for role in classification.required_roles:
            if role in available_set:
                cost = self._estimate_role_cost(role, classification)
                role_candidates.append((role, cost, 1))  # Priority 1
        
        # Add optional roles with lower priority
        for role in classification.optional_roles:
            if role in available_set and role not in classification.required_roles:
                cost = self._estimate_role_cost(role, classification)
                role_candidates.append((role, cost, 3))  # Priority 3
        
        # Sort by priority first, then cost
        role_candidates.sort(key=lambda x: (x[2], x[1]))
        
        # Add roles within budget
        total_cost = 0.0
        for role, role_cost, priority in role_candidates:
            if (len(team) < resource_limits.max_agents and 
                (resource_limits.cost_budget is None or 
                 total_cost + role_cost <= resource_limits.cost_budget)):
                
                model_assignment = self._choose_cost_effective_model(role, classification)
                team.append(AgentSpec(
                    role=role,
                    model_assignment=model_assignment,
                    priority=priority,
                    specializations=self._get_essential_specializations(role, classification.technologies)
                ))
                total_cost += role_cost
        
        # If no team formed due to budget, at least add a basic coder with ollama
        if not team and RoleName.CODER.value in available_set:
            team.append(AgentSpec(
                role=RoleName.CODER.value,
                model_assignment="ollama",
                priority=1,
                specializations=[]
            ))
        
        load_order = self._determine_load_order(team, classification)
        coordination = self._determine_coordination_strategy(classification, team)
        
        return TeamComposition(
            primary_team=team,
            fallback_agents=[],
            load_order=load_order,
            coordination_strategy=coordination,
            estimated_cost=self._estimate_cost(team, classification.complexity),
            confidence_score=self._calculate_confidence(classification, team, historical_performance),
            rationale=f"Cost-optimized team composition within budget constraints (estimated: €{total_cost:.2f})"
        )
    
    def _estimate_role_cost(self, role: str, classification: TaskClassification) -> float:
        """Estimate cost for a specific role."""
        base_cost = 0.1  # Base cost
        
        # Role-specific cost adjustments
        if role in [RoleName.ARCHITECT.value, RoleName.CHIEF_QA_ENGINEER.value]:
            base_cost = 0.25  # Complex reasoning roles
        elif role == RoleName.MERGE_BOT.value:
            base_cost = 0.0   # Automated tasks
        else:
            base_cost = 0.15  # Standard roles
        
        # Complexity multiplier
        complexity_multipliers = {
            ComplexityLevel.TRIVIAL: 0.3,
            ComplexityLevel.LOW: 0.6,
            ComplexityLevel.MEDIUM: 1.0,
            ComplexityLevel.HIGH: 2.0,
            ComplexityLevel.CRITICAL: 3.5,
        }
        
        multiplier = complexity_multipliers.get(classification.complexity, 1.0)
        return base_cost * multiplier
    
    def _choose_cost_effective_model(self, role: str, classification: TaskClassification) -> str:
        """Choose most cost-effective model that meets requirements."""
        # Use free local model whenever possible
        if classification.complexity in [ComplexityLevel.TRIVIAL, ComplexityLevel.LOW]:
            return "ollama"
        
        # For critical tasks, invest in better models
        if classification.complexity == ComplexityLevel.CRITICAL:
            return "codex"
        
        # Balance cost and capability
        if role in [RoleName.ARCHITECT.value, RoleName.QA.value]:
            return "codex"  # Need reasoning capability
        
        return "ollama"  # Default to cost-effective
    
    def _get_essential_specializations(self, role: str, technologies: List[TechnologyStack]) -> List[str]:
        """Get only essential specializations to minimize complexity."""
        # Limit to most relevant technologies only
        relevant_techs = technologies[:2]  # Top 2 most relevant
        return [tech.value for tech in relevant_techs]


class PerformanceOptimizedStrategy(CompositionStrategy):
    """Compose team based on historical performance data and success rates."""
    
    def compose_team(
        self,
        classification: TaskClassification,
        available_roles: List[str],
        resource_limits: ResourceConstraints,
        historical_performance: Dict[str, TeamPerformanceMetrics]
    ) -> TeamComposition:
        """Create performance-optimized team based on historical data."""
        team = []
        available_set = set(available_roles)
        
        # Analyze historical performance to find best combinations
        best_combinations = self._analyze_historical_combinations(
            classification.task_type,
            available_roles,
            historical_performance
        )
        
        # Start with best performing combination if available
        if best_combinations and resource_limits.max_agents >= len(best_combinations[0]):
            for role in best_combinations[0]:
                if role in available_set:
                    model_assignment = self._choose_performance_model(role, classification, historical_performance)
                    team.append(AgentSpec(
                        role=role,
                        model_assignment=model_assignment,
                        priority=1,
                        specializations=self._get_performance_specializations(role, classification.technologies)
                    ))
        
        # Fill gaps with required roles
        team_roles = {agent.role for agent in team}
        for role in classification.required_roles:
            if role not in team_roles and role in available_set and len(team) < resource_limits.max_agents:
                model_assignment = self._choose_performance_model(role, classification, historical_performance)
                team.append(AgentSpec(
                    role=role,
                    model_assignment=model_assignment,
                    priority=2,
                    specializations=self._get_performance_specializations(role, classification.technologies)
                ))
        
        # If still no team, fall back to proven performers
        if not team:
            proven_roles = self._get_proven_performers(classification.task_type, available_roles, historical_performance)
            for role in proven_roles[:min(3, resource_limits.max_agents)]:
                if role in available_set:
                    team.append(AgentSpec(
                        role=role,
                        model_assignment="codex",  # Proven performer
                        priority=1,
                        specializations=[]
                    ))
                    break
        
        load_order = self._determine_load_order(team, classification)
        coordination = self._determine_coordination_strategy(classification, team)
        
        return TeamComposition(
            primary_team=team,
            fallback_agents=[],
            load_order=load_order,
            coordination_strategy=coordination,
            estimated_cost=self._estimate_cost(team, classification.complexity),
            confidence_score=self._calculate_confidence(classification, team, historical_performance),
            rationale="Performance-optimized team based on historical success rates and proven combinations"
        )
    
    def _analyze_historical_combinations(
        self,
        task_type: TaskType,
        available_roles: List[str],
        historical_performance: Dict[str, TeamPerformanceMetrics]
    ) -> List[List[str]]:
        """Analyze historical performance to find best role combinations."""
        combinations = []
        available_set = set(available_roles)
        
        for team_key, metrics in historical_performance.items():
            if (metrics.task_type == task_type and 
                metrics.total_executions >= 2 and  # Need some history
                metrics.success_rate >= 0.7):      # Require good success rate
                
                roles = team_key.split("-")
                # Check if all roles are available
                if all(role in available_set for role in roles):
                    combinations.append((roles, metrics.success_rate, metrics.total_executions))
        
        # Sort by success rate, then by number of executions
        combinations.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        return [combo[0] for combo in combinations[:3]]  # Top 3 combinations
    
    def _choose_performance_model(
        self,
        role: str,
        classification: TaskClassification,
        historical_performance: Dict[str, TeamPerformanceMetrics]
    ) -> str:
        """Choose model based on historical performance."""
        # Look for role-specific performance data
        role_performance = {k: v for k, v in historical_performance.items() 
                          if role in k.split("-") and v.total_executions >= 2}
        
        if role_performance:
            # Find best performing model assignments for this role
            best_success_rate = max(v.success_rate for v in role_performance.values())
            if best_success_rate >= 0.8:
                return "codex"  # High performer
        
        # Default based on role requirements
        if role in [RoleName.ARCHITECT.value, RoleName.QA.value]:
            return "codex"
        elif classification.complexity in [ComplexityLevel.HIGH, ComplexityLevel.CRITICAL]:
            return "codex"
        else:
            return "ollama"
    
    def _get_proven_performers(
        self,
        task_type: TaskType,
        available_roles: List[str],
        historical_performance: Dict[str, TeamPerformanceMetrics]
    ) -> List[str]:
        """Get roles that have historically performed well for this task type."""
        role_performance = {}
        
        for team_key, metrics in historical_performance.items():
            if metrics.task_type == task_type and metrics.total_executions >= 2:
                for role in team_key.split("-"):
                    if role in available_roles:
                        if role not in role_performance:
                            role_performance[role] = []
                        role_performance[role].append(metrics.success_rate)
        
        # Calculate average success rate per role
        role_averages = {}
        for role, rates in role_performance.items():
            role_averages[role] = sum(rates) / len(rates)
        
        # Sort by performance
        return sorted(role_averages.keys(), key=lambda x: role_averages[x], reverse=True)
    
    def _get_performance_specializations(self, role: str, technologies: List[TechnologyStack]) -> List[str]:
        """Get specializations that historically improve performance."""
        # Include all relevant technologies for optimal performance
        specializations = [tech.value for tech in technologies]
        
        # Add performance-focused specializations
        if "qa" in role.lower():
            specializations.extend(["performance_testing", "automated_testing"])
        elif "engineer" in role.lower():
            specializations.extend(["best_practices", "code_review"])
        
        return list(set(specializations))