"""Golden test scenarios for team composer functionality.

These tests cover the specific scenarios outlined in the ICD requirements
plus additional edge cases for comprehensive coverage.
"""

import pytest
import time
from typing import Dict, List

from src.agentsmcp.orchestration.team_composer import TeamComposer
from src.agentsmcp.orchestration.models import (
    TaskClassification,
    TeamComposition,
    TeamPerformanceMetrics,
    ResourceConstraints,
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


class TestGoldenScenarios:
    """Golden test scenarios as specified in the ICD."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.composer = TeamComposer()
    
    def test_golden_scenario_1_simple_implementation(self):
        """Golden Test 1: Simple implementation task with basic requirements."""
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.LOW,
            required_roles=[RoleName.CODER.value],
            optional_roles=[],
            technologies=[TechnologyStack.PYTHON],
            estimated_effort=30,
            risk_level=RiskLevel.LOW,
            keywords=["implement", "simple"],
            confidence=0.9
        )
        
        constraints = ResourceConstraints(
            max_agents=3,
            cost_budget=0.5,
            parallel_limit=2
        )
        
        available_roles = [RoleName.CODER.value, RoleName.QA.value]
        
        result = self.composer.compose_team(
            classification=classification,
            available_roles=available_roles,
            resource_limits=constraints
        )
        
        # Assertions
        assert len(result.primary_team) >= 1
        assert len(result.primary_team) <= constraints.max_agents
        assert RoleName.CODER.value in [agent.role for agent in result.primary_team]
        assert result.estimated_cost <= constraints.cost_budget
        assert result.confidence_score > 0.7
        assert len(result.load_order) == len(result.primary_team)
    
    def test_golden_scenario_2_complex_multi_role_project(self):
        """Golden Test 2: Complex project requiring multiple specialized roles."""
        classification = TaskClassification(
            task_type=TaskType.DESIGN,
            complexity=ComplexityLevel.HIGH,
            required_roles=[RoleName.ARCHITECT.value, RoleName.BACKEND_ENGINEER.value, RoleName.QA.value],
            optional_roles=[RoleName.API_ENGINEER.value, RoleName.DEV_TOOLING_ENGINEER.value],
            technologies=[TechnologyStack.PYTHON, TechnologyStack.API, TechnologyStack.DATABASE],
            estimated_effort=85,
            risk_level=RiskLevel.HIGH,
            keywords=["architecture", "design", "scalable"],
            confidence=0.8
        )
        
        constraints = ResourceConstraints(
            max_agents=6,
            cost_budget=3.0,
            parallel_limit=4
        )
        
        available_roles = [
            RoleName.ARCHITECT.value,
            RoleName.BACKEND_ENGINEER.value, 
            RoleName.API_ENGINEER.value,
            RoleName.QA.value,
            RoleName.DEV_TOOLING_ENGINEER.value
        ]
        
        result = self.composer.compose_team(
            classification=classification,
            available_roles=available_roles,
            resource_limits=constraints
        )
        
        # Assertions
        assert len(result.primary_team) >= 3  # At least required roles
        assert len(result.primary_team) <= constraints.max_agents
        
        team_roles = {agent.role for agent in result.primary_team}
        required_roles = set(classification.required_roles)
        assert required_roles.issubset(team_roles)  # All required roles present
        
        assert result.estimated_cost <= constraints.cost_budget
        assert result.coordination_strategy == CoordinationStrategy.HIERARCHICAL  # Complex project
        assert result.confidence_score > 0.6
    
    def test_golden_scenario_3_cost_constrained_optimization(self):
        """Golden Test 3: Cost-constrained team optimization."""
        classification = TaskClassification(
            task_type=TaskType.BUG_FIX,
            complexity=ComplexityLevel.MEDIUM,
            required_roles=[RoleName.CODER.value, RoleName.QA.value],
            optional_roles=[RoleName.ARCHITECT.value],
            technologies=[TechnologyStack.JAVASCRIPT],
            estimated_effort=45,
            risk_level=RiskLevel.MEDIUM,
            keywords=["fix", "bug", "frontend"],
            confidence=0.85
        )
        
        tight_constraints = ResourceConstraints(
            max_agents=2,
            cost_budget=0.3,  # Very tight budget
            parallel_limit=2
        )
        
        available_roles = [
            RoleName.ARCHITECT.value,
            RoleName.CODER.value,
            RoleName.QA.value,
            RoleName.WEB_FRONTEND_ENGINEER.value
        ]
        
        result = self.composer.compose_team(
            classification=classification,
            available_roles=available_roles,
            resource_limits=tight_constraints
        )
        
        # Assertions
        assert len(result.primary_team) <= tight_constraints.max_agents
        assert result.estimated_cost <= tight_constraints.cost_budget
        
        # Should use cost-effective models
        ollama_count = sum(1 for agent in result.primary_team if agent.model_assignment == "ollama")
        assert ollama_count > 0  # At least one ollama agent for cost efficiency
        
        # Must include at least one required role
        team_roles = {agent.role for agent in result.primary_team}
        assert len(set(classification.required_roles) & team_roles) > 0
    
    def test_golden_scenario_4_historical_performance_optimization(self):
        """Golden Test 4: Team optimization based on historical performance."""
        classification = TaskClassification(
            task_type=TaskType.TESTING,
            complexity=ComplexityLevel.MEDIUM,
            required_roles=[RoleName.QA.value],
            optional_roles=[RoleName.CODER.value, RoleName.BACKEND_QA_ENGINEER.value],
            technologies=[TechnologyStack.TESTING, TechnologyStack.API],
            estimated_effort=55,
            risk_level=RiskLevel.MEDIUM,
            keywords=["test", "qa", "automation"],
            confidence=0.8
        )
        
        constraints = ResourceConstraints(max_agents=4, cost_budget=1.5)
        
        available_roles = [
            RoleName.QA.value,
            RoleName.CODER.value,
            RoleName.BACKEND_QA_ENGINEER.value,
            RoleName.CHIEF_QA_ENGINEER.value
        ]
        
        # Historical performance data showing QA + Backend QA is very effective
        historical_performance = {
            "qa-backend_qa_engineer": TeamPerformanceMetrics(
                team_id="qa-backend_qa_engineer",
                task_type=TaskType.TESTING,
                success_rate=0.95,
                average_duration=200.0,
                average_cost=0.8,
                total_executions=8
            ),
            "qa-coder": TeamPerformanceMetrics(
                team_id="qa-coder",
                task_type=TaskType.TESTING,
                success_rate=0.75,
                average_duration=300.0,
                average_cost=0.9,
                total_executions=4
            )
        }
        
        result = self.composer.compose_team(
            classification=classification,
            available_roles=available_roles,
            resource_limits=constraints,
            historical_performance=historical_performance
        )
        
        # Assertions
        team_roles = {agent.role for agent in result.primary_team}
        
        # Should prefer the high-performing QA + Backend QA combination
        assert RoleName.QA.value in team_roles
        assert RoleName.BACKEND_QA_ENGINEER.value in team_roles
        
        assert result.confidence_score >= 0.8  # High confidence due to good historical data
        assert result.estimated_cost <= constraints.cost_budget
    
    def test_golden_scenario_5_emergency_fallback_creation(self):
        """Golden Test 5: Emergency team when all strategies fail."""
        classification = TaskClassification(
            task_type=TaskType.MAINTENANCE,
            complexity=ComplexityLevel.LOW,
            required_roles=[RoleName.DEV_TOOLING_ENGINEER.value],  # Specialized role
            optional_roles=[],
            technologies=[TechnologyStack.DEVOPS],
            estimated_effort=25,
            risk_level=RiskLevel.LOW,
            keywords=["maintenance", "tooling"],
            confidence=0.7
        )
        
        constraints = ResourceConstraints(max_agents=1, cost_budget=0.05)  # Extreme constraints
        
        # Available roles don't include the required specialized role
        available_roles = [RoleName.CODER.value, RoleName.QA.value]
        
        result = self.composer.compose_team(
            classification=classification,
            available_roles=available_roles,
            resource_limits=constraints
        )
        
        # Assertions
        assert len(result.primary_team) == 1  # Single agent due to constraints
        assert result.primary_team[0].role in available_roles  # Uses available role
        assert result.estimated_cost <= constraints.cost_budget
        
        # Should have reduced confidence due to missing required role
        assert result.confidence_score <= 0.7


class TestAdditionalEdgeCases:
    """Additional edge case tests for comprehensive coverage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.composer = TeamComposer()
    
    def test_edge_case_1_zero_cost_budget_with_ollama_only(self):
        """Edge Case 1: Zero cost budget should still work with ollama agents."""
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.LOW,
            required_roles=[RoleName.CODER.value],
            optional_roles=[],
            technologies=[TechnologyStack.PYTHON],
            estimated_effort=20,
            risk_level=RiskLevel.LOW,
            keywords=["simple", "implement"],
            confidence=0.8
        )
        
        zero_cost_constraints = ResourceConstraints(
            max_agents=2,
            cost_budget=None,  # No budget constraint (will use free ollama)
            parallel_limit=1
        )
        
        available_roles = [RoleName.CODER.value, RoleName.QA.value]
        
        result = self.composer.compose_team(
            classification=classification,
            available_roles=available_roles,
            resource_limits=zero_cost_constraints
        )
        
        # Assertions
        assert result.estimated_cost == 0.0  # Should use only ollama agents (free)
        # Should prefer cost-effective model assignments for low complexity
        ollama_count = sum(1 for agent in result.primary_team if agent.model_assignment == "ollama")
        assert ollama_count > 0  # At least some ollama agents for cost efficiency
        assert RoleName.CODER.value in [agent.role for agent in result.primary_team]
    
    def test_edge_case_2_performance_degradation_with_many_historical_metrics(self):
        """Edge Case 2: Ensure performance doesn't degrade with large historical datasets."""
        classification = TaskClassification(
            task_type=TaskType.ANALYSIS,
            complexity=ComplexityLevel.MEDIUM,
            required_roles=[RoleName.BUSINESS_ANALYST.value],
            optional_roles=[RoleName.DATA_ANALYST.value],
            technologies=[TechnologyStack.DATA_ANALYSIS],
            estimated_effort=40,
            risk_level=RiskLevel.LOW,
            keywords=["analysis", "data"],
            confidence=0.8
        )
        
        constraints = ResourceConstraints(max_agents=3)
        available_roles = [RoleName.BUSINESS_ANALYST.value, RoleName.DATA_ANALYST.value, RoleName.CODER.value]
        
        # Create large historical performance dataset (100 entries)
        historical_performance = {}
        for i in range(100):
            team_key = f"team_{i}"
            historical_performance[team_key] = TeamPerformanceMetrics(
                team_id=team_key,
                task_type=TaskType.ANALYSIS,
                success_rate=0.7 + (i % 3) * 0.1,  # Vary success rates
                total_executions=i % 10 + 1
            )
        
        start_time = time.time()
        
        result = self.composer.compose_team(
            classification=classification,
            available_roles=available_roles,
            resource_limits=constraints,
            historical_performance=historical_performance
        )
        
        elapsed_time = time.time() - start_time
        
        # Assertions
        assert elapsed_time < 0.5  # Must still complete within 500ms
        assert RoleName.BUSINESS_ANALYST.value in [agent.role for agent in result.primary_team]
        assert isinstance(result, TeamComposition)
    
    def test_edge_case_3_all_roles_unavailable_except_one(self):
        """Edge Case 3: Handle scenario where only one role type is available."""
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.HIGH,
            required_roles=[RoleName.ARCHITECT.value, RoleName.BACKEND_ENGINEER.value, RoleName.QA.value],
            optional_roles=[RoleName.API_ENGINEER.value],
            technologies=[TechnologyStack.API, TechnologyStack.DATABASE],
            estimated_effort=75,
            risk_level=RiskLevel.HIGH,
            keywords=["complex", "architecture"],
            confidence=0.9
        )
        
        constraints = ResourceConstraints(max_agents=5)
        
        # Only one role available despite multiple required roles
        available_roles = [RoleName.CODER.value]
        
        result = self.composer.compose_team(
            classification=classification,
            available_roles=available_roles,
            resource_limits=constraints
        )
        
        # Assertions
        assert len(result.primary_team) == 1
        assert result.primary_team[0].role == RoleName.CODER.value
        
        # Confidence should be lower due to missing required roles
        assert result.confidence_score < classification.confidence
    
    def test_edge_case_4_complex_load_order_with_deep_dependencies(self):
        """Edge Case 4: Test load order with complex dependency chains."""
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.HIGH,
            required_roles=[
                RoleName.ARCHITECT.value,
                RoleName.BACKEND_ENGINEER.value,
                RoleName.BACKEND_QA_ENGINEER.value,
                RoleName.CHIEF_QA_ENGINEER.value,
                RoleName.MERGE_BOT.value
            ],
            optional_roles=[],
            technologies=[TechnologyStack.API, TechnologyStack.DATABASE],
            estimated_effort=90,
            risk_level=RiskLevel.HIGH,
            keywords=["complex", "integration"],
            confidence=0.85
        )
        
        constraints = ResourceConstraints(max_agents=6)
        
        available_roles = [
            RoleName.ARCHITECT.value,
            RoleName.BACKEND_ENGINEER.value,
            RoleName.BACKEND_QA_ENGINEER.value,
            RoleName.CHIEF_QA_ENGINEER.value,
            RoleName.MERGE_BOT.value
        ]
        
        result = self.composer.compose_team(
            classification=classification,
            available_roles=available_roles,
            resource_limits=constraints
        )
        
        load_order = result.load_order
        
        # Validate dependency chain: Architect -> Backend Engineer -> QA -> Chief QA -> Merge Bot
        architect_pos = load_order.index(RoleName.ARCHITECT.value)
        backend_pos = load_order.index(RoleName.BACKEND_ENGINEER.value)
        qa_pos = load_order.index(RoleName.BACKEND_QA_ENGINEER.value)
        
        assert architect_pos < backend_pos  # Architect before Backend Engineer
        assert backend_pos < qa_pos  # Backend Engineer before QA
        
        if RoleName.CHIEF_QA_ENGINEER.value in load_order:
            chief_qa_pos = load_order.index(RoleName.CHIEF_QA_ENGINEER.value)
            assert qa_pos < chief_qa_pos  # QA before Chief QA
            
            if RoleName.MERGE_BOT.value in load_order:
                merge_pos = load_order.index(RoleName.MERGE_BOT.value)
                assert chief_qa_pos < merge_pos  # Chief QA before Merge Bot
    
    def test_edge_case_5_mixed_technology_specialization_assignment(self):
        """Edge Case 5: Ensure proper specialization assignment for mixed technology stacks."""
        classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.MEDIUM,
            required_roles=[RoleName.WEB_FRONTEND_ENGINEER.value, RoleName.BACKEND_ENGINEER.value],
            optional_roles=[RoleName.API_ENGINEER.value],
            technologies=[
                TechnologyStack.REACT,
                TechnologyStack.TYPESCRIPT,
                TechnologyStack.PYTHON,
                TechnologyStack.API,
                TechnologyStack.DATABASE
            ],
            estimated_effort=65,
            risk_level=RiskLevel.MEDIUM,
            keywords=["fullstack", "web", "api"],
            confidence=0.8
        )
        
        constraints = ResourceConstraints(max_agents=4)
        
        available_roles = [
            RoleName.WEB_FRONTEND_ENGINEER.value,
            RoleName.BACKEND_ENGINEER.value,
            RoleName.API_ENGINEER.value
        ]
        
        result = self.composer.compose_team(
            classification=classification,
            available_roles=available_roles,
            resource_limits=constraints
        )
        
        # Check that specializations are properly assigned
        for agent in result.primary_team:
            if "frontend" in agent.role.lower():
                # Frontend engineer should have frontend-related specializations
                expected_frontend_techs = {"react", "typescript"}
                agent_tech_specs = set(agent.specializations) & expected_frontend_techs
                assert len(agent_tech_specs) > 0, f"Frontend engineer missing frontend specializations: {agent.specializations}"
                
            elif "backend" in agent.role.lower():
                # Backend engineer should have backend-related specializations  
                expected_backend_techs = {"python", "database"}
                agent_tech_specs = set(agent.specializations) & expected_backend_techs
                assert len(agent_tech_specs) > 0, f"Backend engineer missing backend specializations: {agent.specializations}"
                
            elif "api" in agent.role.lower():
                # API engineer should have API-related specializations
                expected_api_techs = {"api"}
                agent_tech_specs = set(agent.specializations) & expected_api_techs
                assert len(agent_tech_specs) > 0, f"API engineer missing API specializations: {agent.specializations}"
    
    def test_edge_case_6_coordination_strategy_edge_cases(self):
        """Edge Case 6: Test coordination strategy selection for edge cases."""
        composer = TeamComposer()
        
        # Single agent scenario
        single_agent_classification = TaskClassification(
            task_type=TaskType.BUG_FIX,
            complexity=ComplexityLevel.LOW,
            required_roles=[RoleName.CODER.value],
            optional_roles=[],
            technologies=[TechnologyStack.PYTHON],
            estimated_effort=15,
            risk_level=RiskLevel.LOW,
            keywords=["fix"],
            confidence=0.9
        )
        
        result = composer.compose_team(
            classification=single_agent_classification,
            available_roles=[RoleName.CODER.value],
            resource_limits=ResourceConstraints(max_agents=1)
        )
        
        assert result.coordination_strategy == CoordinationStrategy.SEQUENTIAL
        
        # Multiple engineers scenario  
        multi_engineer_classification = TaskClassification(
            task_type=TaskType.IMPLEMENTATION,
            complexity=ComplexityLevel.MEDIUM,
            required_roles=[RoleName.BACKEND_ENGINEER.value, RoleName.WEB_FRONTEND_ENGINEER.value],
            optional_roles=[RoleName.API_ENGINEER.value],
            technologies=[TechnologyStack.API, TechnologyStack.REACT],
            estimated_effort=60,
            risk_level=RiskLevel.MEDIUM,
            keywords=["implementation"],
            confidence=0.8
        )
        
        result = composer.compose_team(
            classification=multi_engineer_classification,
            available_roles=[RoleName.BACKEND_ENGINEER.value, RoleName.WEB_FRONTEND_ENGINEER.value, RoleName.API_ENGINEER.value],
            resource_limits=ResourceConstraints(max_agents=4)
        )
        
        # Should use pipeline for multiple engineers
        assert result.coordination_strategy == CoordinationStrategy.PIPELINE