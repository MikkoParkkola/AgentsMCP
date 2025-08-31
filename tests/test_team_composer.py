"""Unit tests for team composer functionality.

Tests cover all composition strategies, fallback mechanisms, edge cases,
and integration with the main TeamComposer class.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, List

from src.agentsmcp.orchestration.team_composer import TeamComposer
from src.agentsmcp.orchestration.composition_strategies import (
    MinimalTeamStrategy,
    FullStackStrategy, 
    CostOptimizedStrategy,
    PerformanceOptimizedStrategy,
)
from src.agentsmcp.orchestration.models import (
    TaskClassification,
    TeamComposition,
    TeamPerformanceMetrics,
    ResourceConstraints,
    AgentSpec,
    TaskType,
    ComplexityLevel,
    RiskLevel,
    TechnologyStack,
    CoordinationStrategy,
    TeamCompositionError,
    InsufficientResources,
    NoSuitableAgents,
)
from src.agentsmcp.roles.base import RoleName


class TestTeamComposer:
    """Test cases for the main TeamComposer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.composer = TeamComposer()
        self.basic_classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.MEDIUM,
            required_roles=[RoleName.CODER.value],
            optional_roles=[RoleName.QA.value],
            technologies=[TechnologyStack.PYTHON],
            estimated_effort=50,
            risk_level=RiskLevel.MEDIUM,
            keywords=["implement", "feature"],
            confidence=0.85
        )
        self.basic_constraints = ResourceConstraints(
            max_agents=5,
            cost_budget=1.0,
            parallel_limit=3
        )
        self.available_roles = [
            RoleName.ARCHITECT.value,
            RoleName.CODER.value,
            RoleName.QA.value,
            RoleName.BACKEND_ENGINEER.value,
        ]
    
    def test_compose_basic_team_success(self):
        """Test successful team composition with basic parameters."""
        result = self.composer.compose_team(
            classification=self.basic_classification,
            available_roles=self.available_roles,
            resource_limits=self.basic_constraints
        )
        
        assert isinstance(result, TeamComposition)
        assert len(result.primary_team) > 0
        assert len(result.load_order) == len(result.primary_team)
        assert result.confidence_score > 0
        assert result.estimated_cost is not None
        assert RoleName.CODER.value in [agent.role for agent in result.primary_team]
    
    def test_compose_team_no_available_roles(self):
        """Test team composition fails with no available roles."""
        with pytest.raises(NoSuitableAgents):
            self.composer.compose_team(
                classification=self.basic_classification,
                available_roles=[],
                resource_limits=self.basic_constraints
            )
    
    def test_compose_team_invalid_constraints(self):
        """Test team composition fails with invalid constraints."""
        # Test with cost budget too low for any agent execution
        invalid_constraints = ResourceConstraints(max_agents=1, cost_budget=0.001)  # Very low but positive
        
        with pytest.raises(InsufficientResources):
            self.composer.compose_team(
                classification=self.basic_classification,
                available_roles=self.available_roles,
                resource_limits=invalid_constraints
            )
    
    def test_strategy_selection_cost_optimized(self):
        """Test automatic selection of cost-optimized strategy."""
        low_budget_constraints = ResourceConstraints(
            max_agents=3,
            cost_budget=0.3,  # Very low budget
            parallel_limit=2
        )
        
        result = self.composer.compose_team(
            classification=self.basic_classification,
            available_roles=self.available_roles,
            resource_limits=low_budget_constraints
        )
        
        assert "cost-optimized" in result.rationale.lower()
        assert result.estimated_cost <= low_budget_constraints.cost_budget
    
    def test_strategy_selection_full_stack(self):
        """Test automatic selection of full-stack strategy for critical tasks."""
        critical_classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.CRITICAL,
            required_roles=[RoleName.ARCHITECT.value, RoleName.CODER.value, RoleName.QA.value],
            optional_roles=[RoleName.BACKEND_ENGINEER.value],
            technologies=[TechnologyStack.PYTHON, TechnologyStack.API],
            estimated_effort=90,
            risk_level=RiskLevel.CRITICAL,
            keywords=["critical", "system"],
            confidence=0.9
        )
        
        result = self.composer.compose_team(
            classification=critical_classification,
            available_roles=self.available_roles,
            resource_limits=self.basic_constraints
        )
        
        assert "full_stack" in result.rationale.lower() or "full-stack" in result.rationale.lower()
        assert len(result.primary_team) >= 3  # Should have comprehensive team
    
    def test_strategy_selection_minimal(self):
        """Test automatic selection of minimal strategy for simple tasks."""
        simple_classification = TaskClassification(
            task_type=TaskType.BUG_FIX,
            complexity=ComplexityLevel.LOW,
            required_roles=[RoleName.CODER.value],
            optional_roles=[],
            technologies=[TechnologyStack.PYTHON],
            estimated_effort=20,
            risk_level=RiskLevel.LOW,
            keywords=["fix", "bug"],
            confidence=0.8
        )
        
        tight_constraints = ResourceConstraints(max_agents=2, parallel_limit=1)
        
        result = self.composer.compose_team(
            classification=simple_classification,
            available_roles=self.available_roles,
            resource_limits=tight_constraints
        )
        
        assert "minimal" in result.rationale.lower()
        assert len(result.primary_team) <= 2
    
    def test_fallback_team_creation(self):
        """Test fallback team creation when primary exceeds constraints."""
        expensive_constraints = ResourceConstraints(
            max_agents=1,  # Very tight limit
            cost_budget=0.05,  # Very low budget
            parallel_limit=1
        )
        
        result = self.composer.compose_team(
            classification=self.basic_classification,
            available_roles=self.available_roles,
            resource_limits=expensive_constraints
        )
        
        assert len(result.primary_team) <= expensive_constraints.max_agents
        # With ollama model assignment, cost should be 0 and within budget
        # The cost_optimized strategy should handle the constraints gracefully
        assert result.estimated_cost <= expensive_constraints.cost_budget or result.estimated_cost == 0.0
    
    def test_historical_performance_integration(self):
        """Test integration with historical performance data."""
        historical_performance = {
            "coder-qa": TeamPerformanceMetrics(
                team_id="coder-qa",
                task_type=TaskType.IMPLEMENTATION,
                success_rate=0.95,
                average_duration=300.0,
                average_cost=0.25,
                total_executions=10
            ),
            "architect-coder": TeamPerformanceMetrics(
                team_id="architect-coder",
                task_type=TaskType.IMPLEMENTATION,
                success_rate=0.8,
                average_duration=400.0,
                average_cost=0.35,
                total_executions=5
            )
        }
        
        result = self.composer.compose_team(
            classification=self.basic_classification,
            available_roles=self.available_roles,
            resource_limits=self.basic_constraints,
            historical_performance=historical_performance
        )
        
        # Should prefer the high-performing coder-qa combination
        team_roles = {agent.role for agent in result.primary_team}
        assert RoleName.CODER.value in team_roles
        # QA should be included given the high success rate
        assert result.confidence_score > 0.8
    
    def test_load_order_dependencies(self):
        """Test that load order respects role dependencies."""
        full_team_roles = [
            RoleName.ARCHITECT.value,
            RoleName.CODER.value,
            RoleName.QA.value,
            RoleName.MERGE_BOT.value
        ]
        
        result = self.composer.compose_team(
            classification=self.basic_classification,
            available_roles=full_team_roles,
            resource_limits=ResourceConstraints(max_agents=4)
        )
        
        load_order = result.load_order
        
        # Architect should come before coder
        if RoleName.ARCHITECT.value in load_order and RoleName.CODER.value in load_order:
            architect_pos = load_order.index(RoleName.ARCHITECT.value)
            coder_pos = load_order.index(RoleName.CODER.value)
            assert architect_pos < coder_pos
        
        # QA should come after coder
        if RoleName.QA.value in load_order and RoleName.CODER.value in load_order:
            qa_pos = load_order.index(RoleName.QA.value)
            coder_pos = load_order.index(RoleName.CODER.value)
            assert qa_pos > coder_pos
        
        # Merge bot should come after QA
        if RoleName.MERGE_BOT.value in load_order and RoleName.QA.value in load_order:
            merge_pos = load_order.index(RoleName.MERGE_BOT.value)
            qa_pos = load_order.index(RoleName.QA.value)
            assert merge_pos > qa_pos
    
    def test_emergency_team_creation(self):
        """Test emergency team creation when all strategies fail."""
        # Mock all strategies to fail
        with patch.object(self.composer._strategies['minimal'], 'compose_team', side_effect=Exception("Strategy failed")):
            with patch.object(self.composer._strategies['cost_optimized'], 'compose_team', side_effect=Exception("Strategy failed")):
                with patch.object(self.composer._strategies['performance_optimized'], 'compose_team', side_effect=Exception("Strategy failed")):
                    with patch.object(self.composer._strategies['full_stack'], 'compose_team', side_effect=Exception("Strategy failed")):
                        
                        result = self.composer.compose_team(
                            classification=self.basic_classification,
                            available_roles=self.available_roles,
                            resource_limits=self.basic_constraints
                        )
                        
                        assert len(result.primary_team) == 1
                        assert "emergency" in result.rationale.lower()
                        assert result.confidence_score <= 0.5
    
    def test_cost_estimation_utility(self):
        """Test the cost estimation utility method."""
        team_roles = [RoleName.ARCHITECT.value, RoleName.CODER.value]
        
        cost = self.composer.estimate_team_cost(
            classification=self.basic_classification,
            team_roles=team_roles
        )
        
        assert cost > 0
        assert isinstance(cost, float)
        
        # Test with complexity override
        high_cost = self.composer.estimate_team_cost(
            classification=self.basic_classification,
            team_roles=team_roles,
            complexity_override=ComplexityLevel.CRITICAL
        )
        
        assert high_cost > cost
    
    def test_custom_strategy_addition(self):
        """Test adding and using custom composition strategies."""
        class CustomStrategy(Mock):
            def compose_team(self, *args, **kwargs):
                return TeamComposition(
                    primary_team=[AgentSpec(role="custom_role", model_assignment="codex", priority=1)],
                    fallback_agents=[],
                    load_order=["custom_role"],
                    coordination_strategy=CoordinationStrategy.SEQUENTIAL,
                    estimated_cost=0.1,
                    confidence_score=0.7,
                    rationale="Custom strategy test"
                )
        
        custom_strategy = CustomStrategy()
        self.composer.add_custom_strategy("custom", custom_strategy)
        
        assert "custom" in self.composer.get_available_strategies()
    
    def test_performance_within_time_limit(self):
        """Test that team composition completes within 500ms."""
        import time
        
        start_time = time.time()
        
        result = self.composer.compose_team(
            classification=self.basic_classification,
            available_roles=self.available_roles,
            resource_limits=self.basic_constraints
        )
        
        elapsed_time = time.time() - start_time
        assert elapsed_time < 0.5  # Must complete within 500ms
        assert isinstance(result, TeamComposition)


class TestCompositionStrategies:
    """Test cases for individual composition strategies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.MEDIUM,
            required_roles=[RoleName.CODER.value],
            optional_roles=[RoleName.QA.value, RoleName.ARCHITECT.value],
            technologies=[TechnologyStack.PYTHON, TechnologyStack.API],
            estimated_effort=60,
            risk_level=RiskLevel.MEDIUM,
            keywords=["api", "backend"],
            confidence=0.8
        )
        
        self.available_roles = [
            RoleName.ARCHITECT.value,
            RoleName.CODER.value,
            RoleName.QA.value,
            RoleName.BACKEND_ENGINEER.value,
            RoleName.API_ENGINEER.value,
        ]
        
        self.constraints = ResourceConstraints(max_agents=5, cost_budget=2.0)
        self.historical_performance = {}
    
    def test_minimal_strategy(self):
        """Test minimal team strategy creates lean teams."""
        strategy = MinimalTeamStrategy()
        
        result = strategy.compose_team(
            self.classification,
            self.available_roles,
            self.constraints,
            self.historical_performance
        )
        
        assert len(result.primary_team) <= 2  # Should be minimal
        assert RoleName.CODER.value in [agent.role for agent in result.primary_team]
        assert result.estimated_cost < 1.0  # Should be cost-effective
        assert "minimal" in result.rationale.lower()
    
    def test_full_stack_strategy(self):
        """Test full-stack strategy creates comprehensive teams."""
        strategy = FullStackStrategy()
        
        result = strategy.compose_team(
            self.classification,
            self.available_roles,
            self.constraints,
            self.historical_performance
        )
        
        assert len(result.primary_team) >= 3  # Should be comprehensive
        team_roles = {agent.role for agent in result.primary_team}
        
        # Should include core roles
        assert RoleName.CODER.value in team_roles
        assert RoleName.QA.value in team_roles or RoleName.ARCHITECT.value in team_roles
        
        # Should include technology-specific roles for API
        assert RoleName.API_ENGINEER.value in team_roles or RoleName.BACKEND_ENGINEER.value in team_roles
        
        assert "full-stack" in result.rationale.lower()
    
    def test_cost_optimized_strategy(self):
        """Test cost-optimized strategy respects budget constraints."""
        strategy = CostOptimizedStrategy()
        
        tight_budget = ResourceConstraints(max_agents=5, cost_budget=0.5)  # Tight budget
        
        result = strategy.compose_team(
            self.classification,
            self.available_roles,
            tight_budget,
            self.historical_performance
        )
        
        assert result.estimated_cost <= tight_budget.cost_budget
        
        # Should prefer ollama (free) models for cost efficiency
        ollama_count = sum(1 for agent in result.primary_team if agent.model_assignment == "ollama")
        assert ollama_count > 0
        
        assert "cost-optimized" in result.rationale.lower()
    
    def test_performance_optimized_strategy(self):
        """Test performance-optimized strategy uses historical data."""
        strategy = PerformanceOptimizedStrategy()
        
        # Add historical performance data
        historical_performance = {
            "coder-backend_engineer": TeamPerformanceMetrics(
                team_id="coder-backend_engineer",
                task_type=TaskType.IMPLEMENTATION,
                success_rate=0.9,
                total_executions=5
            ),
            "architect-qa": TeamPerformanceMetrics(
                team_id="architect-qa", 
                task_type=TaskType.IMPLEMENTATION,
                success_rate=0.7,
                total_executions=3
            )
        }
        
        result = strategy.compose_team(
            self.classification,
            self.available_roles,
            self.constraints,
            historical_performance
        )
        
        team_roles = {agent.role for agent in result.primary_team}
        
        # Should prefer the high-performing combination
        assert RoleName.CODER.value in team_roles
        assert RoleName.BACKEND_ENGINEER.value in team_roles
        
        assert "performance-optimized" in result.rationale.lower()
    
    def test_coordination_strategy_selection(self):
        """Test that appropriate coordination strategies are selected."""
        strategy = FullStackStrategy()
        
        # Test hierarchical for high-risk tasks
        high_risk_classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.CRITICAL,
            required_roles=[RoleName.ARCHITECT.value, RoleName.CODER.value],
            optional_roles=[],
            technologies=[TechnologyStack.PYTHON],
            estimated_effort=90,
            risk_level=RiskLevel.CRITICAL,
            keywords=["critical"],
            confidence=0.9
        )
        
        result = strategy.compose_team(
            high_risk_classification,
            self.available_roles,
            self.constraints,
            self.historical_performance
        )
        
        assert result.coordination_strategy == CoordinationStrategy.HIERARCHICAL
    
    def test_technology_specific_role_selection(self):
        """Test that technology-specific roles are selected appropriately."""
        strategy = FullStackStrategy()
        
        # Test with TUI technology
        tui_classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.MEDIUM,
            required_roles=[RoleName.CODER.value],
            optional_roles=[],
            technologies=[TechnologyStack.TUI],
            estimated_effort=50,
            risk_level=RiskLevel.MEDIUM,
            keywords=["tui", "terminal"],
            confidence=0.8
        )
        
        tui_roles = self.available_roles + [RoleName.TUI_FRONTEND_ENGINEER.value]
        
        result = strategy.compose_team(
            tui_classification,
            tui_roles,
            self.constraints,
            self.historical_performance
        )
        
        team_roles = {agent.role for agent in result.primary_team}
        assert RoleName.TUI_FRONTEND_ENGINEER.value in team_roles


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_required_roles(self):
        """Test handling of tasks with no required roles."""
        classification = TaskClassification(
            task_type=TaskType.ANALYSIS,
            complexity=ComplexityLevel.LOW,
            required_roles=[],  # Empty required roles
            optional_roles=[RoleName.BUSINESS_ANALYST.value],
            technologies=[],
            estimated_effort=30,
            risk_level=RiskLevel.LOW,
            keywords=["analysis"],
            confidence=0.7
        )
        
        composer = TeamComposer()
        result = composer.compose_team(
            classification=classification,
            available_roles=[RoleName.CODER.value, RoleName.BUSINESS_ANALYST.value],
            resource_limits=ResourceConstraints(max_agents=3)
        )
        
        assert len(result.primary_team) > 0  # Should still create a team
    
    def test_single_agent_limit(self):
        """Test team composition with max_agents=1."""
        classification = TaskClassification(
            task_type=TaskType.BUG_FIX,
            complexity=ComplexityLevel.LOW,
            required_roles=[RoleName.CODER.value],
            optional_roles=[RoleName.QA.value],
            technologies=[TechnologyStack.PYTHON],
            estimated_effort=20,
            risk_level=RiskLevel.LOW,
            keywords=["fix"],
            confidence=0.8
        )
        
        composer = TeamComposer()
        result = composer.compose_team(
            classification=classification,
            available_roles=[RoleName.CODER.value, RoleName.QA.value],
            resource_limits=ResourceConstraints(max_agents=1)
        )
        
        assert len(result.primary_team) == 1
        assert result.primary_team[0].role == RoleName.CODER.value
    
    def test_no_matching_roles(self):
        """Test handling when required roles are not available."""
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.MEDIUM,
            required_roles=[RoleName.ML_ENGINEER.value],  # Not in available roles
            optional_roles=[],
            technologies=[TechnologyStack.MACHINE_LEARNING],
            estimated_effort=50,
            risk_level=RiskLevel.MEDIUM,
            keywords=["ml"],
            confidence=0.8
        )
        
        composer = TeamComposer()
        result = composer.compose_team(
            classification=classification,
            available_roles=[RoleName.CODER.value, RoleName.QA.value],  # No ML engineer
            resource_limits=ResourceConstraints(max_agents=3)
        )
        
        # Should still create a team with available roles
        assert len(result.primary_team) > 0
        # Should have lower confidence due to missing required role
        assert result.confidence_score < 0.8