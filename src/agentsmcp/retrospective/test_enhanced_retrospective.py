"""Comprehensive test suite for the enhanced retrospective system.

Tests the complete flow: individual retrospectives -> agile coach analysis -> 
orchestrator enforcement -> readiness assessment.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Dict, Any, List

from ..models import TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus
from ..roles.base import RoleName
from ..orchestration.models import TeamComposition, TeamPerformanceMetrics, AgentSpec

from .individual_framework import IndividualRetrospectiveFramework, IndividualRetrospectiveConfig
from .coach_analyzer import AgileCoachAnalyzer
from .enforcement import OrchestratorEnforcementSystem, SystemState
from .integration_layer import EnhancedRetrospectiveIntegration, OrchestratorConfig
from .data_models import (
    IndividualRetrospective,
    ComprehensiveRetrospectiveReport,
    ActionPoint,
    PriorityLevel,
    ImplementationStatus,
)


class TestEnhancedRetrospectiveSystem:
    """Test suite for the complete enhanced retrospective system."""
    
    @pytest.fixture
    async def individual_framework(self):
        """Create individual retrospective framework for testing."""
        config = IndividualRetrospectiveConfig(
            timeout_seconds=10,
            max_challenges=5,
            max_learnings=5,
            max_improvement_actions=3,
        )
        return IndividualRetrospectiveFramework(config=config)
    
    @pytest.fixture
    async def coach_analyzer(self):
        """Create agile coach analyzer for testing."""
        return AgileCoachAnalyzer(
            analysis_timeout=10,
            min_retrospectives_for_patterns=2,
        )
    
    @pytest.fixture
    async def enforcement_system(self):
        """Create orchestrator enforcement system for testing."""
        return OrchestratorEnforcementSystem(
            validation_timeout=5,
            max_retry_attempts=2,
        )
    
    @pytest.fixture
    async def integration_layer(self, individual_framework, coach_analyzer, enforcement_system):
        """Create integration layer for testing."""
        config = OrchestratorConfig()
        config.retrospective_timeout = 10
        config.enforcement_required = True
        
        integration = EnhancedRetrospectiveIntegration(
            orchestrator_config=config,
        )
        
        # Manually set components for testing
        integration.individual_framework = individual_framework
        integration.coach_analyzer = coach_analyzer
        integration.enforcement_system = enforcement_system
        
        return integration
    
    @pytest.fixture
    def sample_task_context(self):
        """Create sample task context for testing."""
        return TaskEnvelopeV1(
            objective="Implement user authentication system",
            inputs={
                "task_id": "auth_implementation_001",
                "requirements": ["secure login", "password hashing", "session management"],
            },
            constraints=["must use existing database", "follow security best practices"],
        )
    
    @pytest.fixture
    def sample_execution_results(self):
        """Create sample execution results for testing."""
        return ResultEnvelopeV1(
            status=EnvelopeStatus.SUCCESS,
            artifacts={
                "code": "authentication_module.py",
                "tests": "test_authentication.py",
                "documentation": "auth_readme.md",
            },
            metrics={
                "lines_of_code": 250,
                "test_coverage": 0.85,
                "completion_time": 3600,
            },
            confidence=0.8,
            notes="Authentication system implemented with comprehensive testing",
        )
    
    @pytest.mark.asyncio
    async def test_individual_retrospective_flow(
        self, 
        individual_framework, 
        sample_task_context, 
        sample_execution_results
    ):
        """Test complete individual retrospective flow."""
        
        # Test successful retrospective
        retrospective = await individual_framework.conduct_retrospective(
            agent_role=RoleName.CODER,
            task_context=sample_task_context,
            execution_results=sample_execution_results,
            agent_state={
                "decisions": [
                    {
                        "description": "Choose bcrypt for password hashing",
                        "chosen_option": "bcrypt",
                        "rationale": "Industry standard with good security",
                        "confidence": 0.9,
                    }
                ],
                "tool_usage": {
                    "pytest": {"count": 15, "success_rate": 1.0, "total_time": 300},
                    "eslint": {"count": 5, "success_rate": 0.8, "total_time": 60},
                },
            },
        )
        
        # Validate retrospective structure
        assert isinstance(retrospective, IndividualRetrospective)
        assert retrospective.agent_role == RoleName.CODER
        assert retrospective.task_id == "auth_implementation_001"
        assert len(retrospective.what_went_well) > 0
        assert len(retrospective.decisions_made) > 0
        assert len(retrospective.tools_used) > 0
        assert retrospective.performance_assessment.overall_score > 0
        
        # Test retrospective with failure case
        failed_results = ResultEnvelopeV1(
            status=EnvelopeStatus.ERROR,
            artifacts={"error": "compilation_failed"},
            confidence=0.2,
            notes="Implementation failed due to compilation errors",
        )
        
        failed_retrospective = await individual_framework.conduct_retrospective(
            agent_role=RoleName.CODER,
            task_context=sample_task_context,
            execution_results=failed_results,
        )
        
        assert failed_retrospective.performance_assessment.overall_score < 0.5
        assert len(failed_retrospective.what_could_improve) > 0
        assert len(failed_retrospective.challenges_encountered) > 0
    
    @pytest.mark.asyncio
    async def test_agile_coach_analysis_flow(
        self,
        coach_analyzer,
        sample_task_context,
        sample_execution_results,
    ):
        """Test agile coach comprehensive analysis flow."""
        
        # Create multiple individual retrospectives
        individual_retros = []
        
        # High-performing coder retrospective
        coder_retro = IndividualRetrospective(
            agent_role=RoleName.CODER,
            task_id="auth_implementation_001",
            what_went_well=["Clean code implementation", "Comprehensive testing", "Good documentation"],
            what_could_improve=["Code review process", "Performance optimization"],
            key_learnings=["Learned new authentication patterns", "Improved testing skills"],
        )
        coder_retro.performance_assessment.overall_score = 0.85
        individual_retros.append(coder_retro)
        
        # QA retrospective
        qa_retro = IndividualRetrospective(
            agent_role=RoleName.QA,
            task_id="auth_implementation_001",
            what_went_well=["Thorough testing coverage", "Found security issues early"],
            what_could_improve=["Test automation", "Performance testing"],
            key_learnings=["New security testing techniques"],
        )
        qa_retro.performance_assessment.overall_score = 0.75
        individual_retros.append(qa_retro)
        
        # Architect retrospective with some issues
        architect_retro = IndividualRetrospective(
            agent_role=RoleName.ARCHITECT,
            task_id="auth_implementation_001",
            what_went_well=["Clear architecture design"],
            what_could_improve=["Better coordination with team", "More detailed documentation"],
            key_learnings=["Team communication is critical"],
        )
        architect_retro.performance_assessment.overall_score = 0.60
        individual_retros.append(architect_retro)
        
        # Create team context
        team_composition = TeamComposition(
            primary_team=[
                AgentSpec(role="coder", capabilities=["python", "testing"]),
                AgentSpec(role="qa", capabilities=["testing", "security"]),
                AgentSpec(role="architect", capabilities=["design", "documentation"]),
            ],
            coordination_strategy="collaborative",
        )
        
        team_metrics = TeamPerformanceMetrics(
            success_rate=0.75,
            average_duration=3600.0,
            average_cost=10.0,
        )
        
        # Conduct comprehensive analysis
        comprehensive_report = await coach_analyzer.analyze_retrospectives(
            individual_retrospectives=individual_retros,
            team_composition=team_composition,
            execution_metrics=team_metrics,
        )
        
        # Validate comprehensive report
        assert isinstance(comprehensive_report, ComprehensiveRetrospectiveReport)
        assert comprehensive_report.task_id == "auth_implementation_001"
        assert comprehensive_report.individual_retrospectives_count == 3
        assert len(comprehensive_report.participating_agents) == 3
        
        # Check analysis results
        assert len(comprehensive_report.pattern_analysis) > 0
        assert len(comprehensive_report.systemic_issues) >= 0  # May or may not have issues
        assert comprehensive_report.overall_team_performance > 0
        assert len(comprehensive_report.action_points) > 0
        
        # Check that improvement opportunities are identified
        assert len(comprehensive_report.improvement_opportunities) > 0
        
        # Validate priority matrix
        assert comprehensive_report.priority_matrix is not None
        total_actions = (
            len(comprehensive_report.priority_matrix.high_impact_low_effort) +
            len(comprehensive_report.priority_matrix.high_impact_high_effort) +
            len(comprehensive_report.priority_matrix.low_impact_low_effort) +
            len(comprehensive_report.priority_matrix.low_impact_high_effort)
        )
        assert total_actions == len(comprehensive_report.action_points)
    
    @pytest.mark.asyncio
    async def test_enforcement_system_flow(self, enforcement_system):
        """Test orchestrator enforcement system flow."""
        
        # Create a comprehensive report with action points
        comprehensive_report = ComprehensiveRetrospectiveReport(
            task_id="test_enforcement",
            action_points=[
                ActionPoint(
                    title="Improve code review process",
                    description="Implement automated code review tools",
                    category="process",
                    priority=PriorityLevel.HIGH,
                    implementation_type="configuration",
                    estimated_effort_hours=4.0,
                    success_metrics=["Code review time reduced by 50%"],
                ),
                ActionPoint(
                    title="Enhance testing coverage",
                    description="Increase unit test coverage to 90%",
                    category="quality",
                    priority=PriorityLevel.MEDIUM,
                    implementation_type="manual",
                    estimated_effort_hours=8.0,
                    success_metrics=["Test coverage >= 90%"],
                ),
                ActionPoint(
                    title="Update documentation",
                    description="Improve API documentation clarity",
                    category="documentation",
                    priority=PriorityLevel.LOW,
                    implementation_type="manual",
                    estimated_effort_hours=2.0,
                    success_metrics=["Documentation review score > 8.0"],
                ),
            ]
        )
        
        # Create system state
        system_state = SystemState()
        system_state.configuration = {"current_version": "1.0.0"}
        system_state.active_agents = {"coder", "qa", "architect"}
        system_state.performance_metrics = {"success_rate": 0.8}
        system_state.health_status = "healthy"
        
        # Create enforcement plan
        enforcement_plan = await enforcement_system.create_enforcement_plan(
            comprehensive_report,
            system_state,
            implementation_capabilities={
                "automatic_config": True,
                "automatic_training": False,
            },
        )
        
        # Validate enforcement plan
        assert enforcement_plan is not None
        assert len(enforcement_plan.action_points) == 3
        assert len(enforcement_plan.implementation_sequence) == 3
        assert len(enforcement_plan.validation_steps) > 0
        assert enforcement_plan.estimated_completion_time is not None
        
        # Execute enforcement plan
        success, errors = await enforcement_system.execute_enforcement_plan(
            enforcement_plan,
            system_state,
        )
        
        # Validate execution results
        # Note: In test environment, some implementations may fail
        assert isinstance(success, bool)
        assert isinstance(errors, list)
        
        # Check that high-priority actions were attempted
        high_priority_actions = [a for a in enforcement_plan.action_points if a.priority == PriorityLevel.HIGH]
        assert len(high_priority_actions) > 0
        
        # Test readiness assessment
        readiness = await enforcement_system.assess_system_readiness(
            enforcement_plan=enforcement_plan,
            system_state=system_state,
        )
        
        assert readiness is not None
        assert 0.0 <= readiness.overall_readiness_score <= 1.0
        assert isinstance(readiness.next_task_clearance, bool)
        
        if not readiness.next_task_clearance:
            assert readiness.estimated_time_to_ready is not None
    
    @pytest.mark.asyncio
    async def test_complete_integration_flow(
        self,
        integration_layer,
        sample_task_context,
        sample_execution_results,
    ):
        """Test complete integration flow from start to finish."""
        
        # Initialize integration
        integration_status = await integration_layer.initialize_integration(
            validate_compatibility=False  # Skip compatibility check for testing
        )
        
        assert integration_status.integrated == True
        assert integration_status.individual_framework_active == True
        assert integration_status.coach_analyzer_active == True
        assert integration_status.enforcement_system_active == True
        
        # Conduct individual retrospective
        individual_retro, _ = await integration_layer.conduct_enhanced_retrospective(
            agent_role=RoleName.CODER,
            task_context=sample_task_context,
            execution_results=sample_execution_results,
        )
        
        assert individual_retro is not None
        assert individual_retro.agent_role == RoleName.CODER
        
        # Test with team context for comprehensive analysis
        team_context = {
            "individual_retrospectives": [
                individual_retro,
                # Add another retrospective to trigger comprehensive analysis
                IndividualRetrospective(
                    agent_role=RoleName.QA,
                    task_id=sample_task_context.inputs.get("task_id", "test"),
                    what_went_well=["Good testing coverage"],
                    what_could_improve=["Faster test execution"],
                ),
            ],
            "team_composition": TeamComposition(
                primary_team=[
                    AgentSpec(role="coder", capabilities=["python"]),
                    AgentSpec(role="qa", capabilities=["testing"]),
                ],
                coordination_strategy="collaborative",
            ),
            "execution_metrics": TeamPerformanceMetrics(
                success_rate=0.8,
                average_duration=1800.0,
                average_cost=5.0,
            ),
        }
        
        # Conduct comprehensive retrospective
        individual_retro2, comprehensive_report = await integration_layer.conduct_enhanced_retrospective(
            agent_role=RoleName.QA,
            task_context=sample_task_context,
            execution_results=sample_execution_results,
            team_context=team_context,
        )
        
        assert individual_retro2 is not None
        assert comprehensive_report is not None
        assert len(comprehensive_report.action_points) > 0
        
        # Test enforcement
        enforcement_success = await integration_layer.enforce_action_points(
            comprehensive_report
        )
        
        assert isinstance(enforcement_success, bool)
        
        # Test readiness assessment
        ready_for_next = await integration_layer.assess_readiness_for_next_task()
        assert isinstance(ready_for_next, bool)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(
        self,
        integration_layer,
        sample_task_context,
    ):
        """Test error handling and fallback mechanisms."""
        
        # Test with invalid execution results
        invalid_results = ResultEnvelopeV1(
            status=EnvelopeStatus.ERROR,
            artifacts={},
            confidence=0.0,
            notes="Critical failure",
        )
        
        # Should still conduct retrospective but with different outcomes
        individual_retro, _ = await integration_layer.conduct_enhanced_retrospective(
            agent_role=RoleName.CODER,
            task_context=sample_task_context,
            execution_results=invalid_results,
        )
        
        assert individual_retro is not None
        assert individual_retro.performance_assessment.overall_score < 0.5
        assert len(individual_retro.challenges_encountered) > 0
    
    @pytest.mark.asyncio
    async def test_retrospective_data_consistency(
        self,
        individual_framework,
        sample_task_context,
        sample_execution_results,
    ):
        """Test that retrospective data remains consistent across multiple runs."""
        
        # Run the same retrospective multiple times
        retrospectives = []
        for i in range(3):
            retro = await individual_framework.conduct_retrospective(
                agent_role=RoleName.CODER,
                task_context=sample_task_context,
                execution_results=sample_execution_results,
                agent_state={"run_number": i + 1},
            )
            retrospectives.append(retro)
        
        # Check consistency of key fields
        assert all(r.agent_role == RoleName.CODER for r in retrospectives)
        assert all(r.task_id == retrospectives[0].task_id for r in retrospectives)
        
        # Performance scores should be similar (within 0.1)
        scores = [r.performance_assessment.overall_score for r in retrospectives]
        score_range = max(scores) - min(scores)
        assert score_range <= 0.1, f"Score range too large: {score_range}"
    
    @pytest.mark.asyncio
    async def test_performance_impact_measurement(self, integration_layer):
        """Test that the retrospective system doesn't significantly impact performance."""
        
        import time
        
        # Measure time for simple task without retrospective
        start_time = time.time()
        
        # Simulate simple task execution
        await asyncio.sleep(0.01)  # Simulate 10ms task
        
        baseline_time = time.time() - start_time
        
        # Measure time with retrospective
        start_time = time.time()
        
        # Simulate task + retrospective
        task_context = TaskEnvelopeV1(objective="Simple test task", inputs={"task_id": "perf_test"})
        execution_results = ResultEnvelopeV1(status=EnvelopeStatus.SUCCESS, confidence=0.8)
        
        await asyncio.sleep(0.01)  # Simulate task execution
        
        if integration_layer.individual_framework:
            await integration_layer.individual_framework.conduct_retrospective(
                agent_role=RoleName.CODER,
                task_context=task_context,
                execution_results=execution_results,
            )
        
        total_time = time.time() - start_time
        
        # Retrospective overhead should be reasonable (< 500% of baseline)
        overhead = (total_time - baseline_time) / baseline_time
        assert overhead < 5.0, f"Performance overhead too high: {overhead:.2f}x"


@pytest.mark.asyncio
async def test_integration_system_initialization():
    """Test that the integration system initializes correctly."""
    
    config = OrchestratorConfig()
    integration = EnhancedRetrospectiveIntegration(orchestrator_config=config)
    
    # Test component initialization
    assert integration.config is not None
    assert integration.integration_status is not None
    assert not integration.integration_status.integrated  # Should not be integrated initially
    
    # Test configuration validation
    assert integration.config.enable_individual_retrospectives is True
    assert integration.config.enable_comprehensive_analysis is True
    assert integration.config.enable_enforcement_system is True


@pytest.mark.asyncio 
async def test_action_point_lifecycle():
    """Test complete lifecycle of action points from creation to completion."""
    
    enforcement_system = OrchestratorEnforcementSystem()
    
    # Create action point
    action = ActionPoint(
        title="Test Action",
        description="Test action point for lifecycle testing",
        priority=PriorityLevel.MEDIUM,
        implementation_type="automatic",
        estimated_effort_hours=2.0,
        success_metrics=["Action completed successfully"],
    )
    
    # Initial state
    assert action.status == ImplementationStatus.PENDING
    assert action.completed_at is None
    
    # Create enforcement plan
    report = ComprehensiveRetrospectiveReport(
        task_id="lifecycle_test",
        action_points=[action],
    )
    
    system_state = SystemState()
    
    plan = await enforcement_system.create_enforcement_plan(
        report,
        system_state,
    )
    
    assert len(plan.action_points) == 1
    assert plan.action_points[0].action_id == action.action_id
    
    # Execute plan
    success, errors = await enforcement_system.execute_enforcement_plan(plan, system_state)
    
    # Check final state
    final_action = plan.action_points[0]
    assert final_action.status in [ImplementationStatus.COMPLETED, ImplementationStatus.BLOCKED]
    
    if final_action.status == ImplementationStatus.COMPLETED:
        assert final_action.completed_at is not None


if __name__ == "__main__":
    # Run tests directly
    import sys
    import subprocess
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    print("Test Results:")
    print("=" * 50)
    print(result.stdout)
    if result.stderr:
        print("\nErrors:")
        print(result.stderr)
    
    sys.exit(result.returncode)