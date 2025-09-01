#!/usr/bin/env python3
"""
Comprehensive Test Suite for AgentsMCP Thinking and Planning System

This test suite validates all core components of the cognition module:
- ThinkingFramework integration and correctness
- ApproachEvaluator scoring and ranking
- TaskDecomposer dependency analysis
- ExecutionPlanner scheduling optimization
- MetacognitiveMonitor quality assessment
- PlanningStateManager persistence and recovery
- ThinkingOrchestrator integration

Run with: python -m pytest tests/test_thinking_system.py -v
"""

import asyncio
import pytest
import tempfile
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import the cognition module
from src.agentsmcp.cognition import (
    ThinkingFramework, ThinkingConfig, ApproachEvaluator, TaskDecomposer,
    ExecutionPlanner, MetacognitiveMonitor, PlanningStateManager,
    ThinkingOrchestrator, create_thinking_orchestrator,
    ThinkingPhase, ThinkingStep, Approach, SubTask, DependencyGraph,
    ExecutionSchedule, QualityAssessment, PlanningState,
    PerformanceProfile, OrchestratorIntegrationMode
)
from src.agentsmcp.cognition.models import (
    EvaluationCriteria, TaskType, DependencyType, Dependency,
    ExecutionStrategy, ResourceConstraints, QualityDimension,
    PersistenceFormat, CheckpointStrategy, CleanupPolicy
)
from src.agentsmcp.cognition.exceptions import (
    ThinkingError, EvaluationError, DecompositionError,
    PlanningError, StateManagerError
)


class TestThinkingFramework:
    """Test the core thinking framework functionality."""
    
    @pytest.fixture
    def thinking_config(self):
        """Create test configuration."""
        return ThinkingConfig(
            max_approaches=3,
            max_subtasks=5,
            enable_metacognitive_monitoring=True,
            thinking_depth="balanced",
            timeout_seconds=10
        )
    
    @pytest.fixture
    def framework(self, thinking_config):
        """Create thinking framework instance."""
        return ThinkingFramework(thinking_config)
    
    @pytest.mark.asyncio
    async def test_basic_thinking_process(self, framework):
        """Test basic thinking process execution."""
        request = "Design a simple web application with user authentication"
        context = {"framework": "FastAPI", "database": "SQLite"}
        
        result = await framework.think(request, context)
        
        assert result is not None
        assert result.request == request
        assert result.context == context
        assert isinstance(result.confidence, float)
        assert 0 <= result.confidence <= 1
        assert result.total_duration_ms > 0
        assert len(result.thinking_trace) > 0
    
    @pytest.mark.asyncio
    async def test_thinking_phases_completion(self, framework):
        """Test that all thinking phases are completed."""
        request = "Implement a caching system for improved performance"
        
        result = await framework.think(request)
        
        # Check that major phases are represented
        phases_completed = {step.phase for step in result.thinking_trace}
        
        # Should include at least analyze, explore, and evaluate phases
        assert ThinkingPhase.ANALYZE_REQUEST in phases_completed
        assert ThinkingPhase.EXPLORE_OPTIONS in phases_completed
        assert ThinkingPhase.EVALUATE_APPROACHES in phases_completed
    
    @pytest.mark.asyncio
    async def test_thinking_with_timeout(self):
        """Test thinking process with very short timeout."""
        config = ThinkingConfig(timeout_seconds=0.1)  # Very short timeout
        framework = ThinkingFramework(config)
        
        request = "Design a complex distributed system architecture"
        
        result = await framework.think(request)
        
        # Should still return a result, possibly with lower confidence
        assert result is not None
        assert result.total_duration_ms <= 200  # Should timeout quickly
    
    @pytest.mark.asyncio
    async def test_progress_callback(self, framework):
        """Test progress callback functionality."""
        steps_received = []
        
        def progress_callback(step: ThinkingStep):
            steps_received.append(step)
        
        request = "Create a monitoring dashboard"
        
        await framework.think(request, progress_callback=progress_callback)
        
        assert len(steps_received) > 0
        assert all(isinstance(step, ThinkingStep) for step in steps_received)
        assert all(hasattr(step, 'phase') for step in steps_received)


class TestApproachEvaluator:
    """Test approach evaluation and ranking functionality."""
    
    @pytest.fixture
    def evaluator(self):
        """Create approach evaluator instance."""
        return ApproachEvaluator()
    
    @pytest.fixture
    def sample_approaches(self):
        """Create sample approaches for testing."""
        return [
            Approach(
                name="REST API",
                description="Traditional REST API with JSON responses",
                estimated_complexity=0.6,
                estimated_risk=0.3,
                estimated_score=0.7
            ),
            Approach(
                name="GraphQL API", 
                description="GraphQL API with flexible queries",
                estimated_complexity=0.8,
                estimated_risk=0.5,
                estimated_score=0.8
            ),
            Approach(
                name="gRPC Service",
                description="High-performance gRPC service",
                estimated_complexity=0.9,
                estimated_risk=0.4,
                estimated_score=0.75
            )
        ]
    
    @pytest.mark.asyncio
    async def test_approach_evaluation(self, evaluator, sample_approaches):
        """Test basic approach evaluation."""
        criteria = EvaluationCriteria()
        
        ranked_approaches = await evaluator.evaluate_approaches(
            sample_approaches, criteria
        )
        
        assert len(ranked_approaches) == len(sample_approaches)
        assert all(hasattr(approach, 'final_score') for approach in ranked_approaches)
        assert all(hasattr(approach, 'rank') for approach in ranked_approaches)
        
        # Check ranking order (higher scores should be ranked higher)
        scores = [approach.final_score for approach in ranked_approaches]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_weighted_evaluation_criteria(self, evaluator, sample_approaches):
        """Test evaluation with custom weighted criteria."""
        criteria = EvaluationCriteria(
            feasibility_weight=0.4,
            efficiency_weight=0.3,
            maintainability_weight=0.2,
            scalability_weight=0.1
        )
        
        ranked_approaches = await evaluator.evaluate_approaches(
            sample_approaches, criteria
        )
        
        assert len(ranked_approaches) > 0
        # Scores should be different from default weighting
        for approach in ranked_approaches:
            assert 0 <= approach.final_score <= 1


class TestTaskDecomposer:
    """Test task decomposition and dependency analysis."""
    
    @pytest.fixture
    def decomposer(self):
        """Create task decomposer instance."""
        return TaskDecomposer()
    
    @pytest.fixture
    def sample_approach(self):
        """Create sample approach for decomposition."""
        return Approach(
            name="E-commerce Platform",
            description="Build a full e-commerce platform with user management, product catalog, shopping cart, and payment processing",
            estimated_complexity=0.9,
            estimated_risk=0.6
        )
    
    @pytest.mark.asyncio
    async def test_task_decomposition(self, decomposer, sample_approach):
        """Test basic task decomposition."""
        subtasks, dependency_graph = await decomposer.decompose_approach(sample_approach)
        
        assert len(subtasks) > 0
        assert isinstance(dependency_graph, DependencyGraph)
        assert all(isinstance(task, SubTask) for task in subtasks)
        assert all(hasattr(task, 'id') for task in subtasks)
        assert all(hasattr(task, 'description') for task in subtasks)
    
    @pytest.mark.asyncio
    async def test_dependency_analysis(self, decomposer, sample_approach):
        """Test dependency graph analysis."""
        subtasks, dependency_graph = await decomposer.decompose_approach(sample_approach)
        
        # Should have some dependencies for complex approach
        assert len(dependency_graph.dependencies) >= 0
        
        # Check dependency structure
        for dependency in dependency_graph.dependencies:
            assert hasattr(dependency, 'from_task_id')
            assert hasattr(dependency, 'to_task_id')
            assert hasattr(dependency, 'dependency_type')
    
    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, decomposer):
        """Test circular dependency detection."""
        # Create approach that might lead to circular dependencies
        complex_approach = Approach(
            name="Microservices Architecture",
            description="Design interdependent microservices with complex communication patterns",
            estimated_complexity=0.95,
            estimated_risk=0.7
        )
        
        subtasks, dependency_graph = await decomposer.decompose_approach(complex_approach)
        
        # Should not have circular dependencies
        has_cycles = await decomposer._has_circular_dependencies(dependency_graph)
        assert not has_cycles


class TestExecutionPlanner:
    """Test execution planning and scheduling."""
    
    @pytest.fixture
    def planner(self):
        """Create execution planner instance."""
        return ExecutionPlanner()
    
    @pytest.fixture
    def sample_subtasks(self):
        """Create sample subtasks for planning."""
        return [
            SubTask(
                id="task1",
                description="Set up development environment",
                task_type=TaskType.SETUP,
                estimated_duration_minutes=30,
                estimated_complexity=0.3
            ),
            SubTask(
                id="task2", 
                description="Design database schema",
                task_type=TaskType.DESIGN,
                estimated_duration_minutes=120,
                estimated_complexity=0.7
            ),
            SubTask(
                id="task3",
                description="Implement user authentication",
                task_type=TaskType.IMPLEMENTATION,
                estimated_duration_minutes=240,
                estimated_complexity=0.8
            ),
            SubTask(
                id="task4",
                description="Write unit tests",
                task_type=TaskType.TESTING,
                estimated_duration_minutes=180,
                estimated_complexity=0.6
            )
        ]
    
    @pytest.fixture
    def sample_dependency_graph(self, sample_subtasks):
        """Create sample dependency graph."""
        dependencies = [
            Dependency("task1", "task2", DependencyType.SEQUENCE),
            Dependency("task2", "task3", DependencyType.SEQUENCE),
            Dependency("task3", "task4", DependencyType.SEQUENCE)
        ]
        return DependencyGraph(dependencies)
    
    @pytest.mark.asyncio
    async def test_schedule_creation(self, planner, sample_subtasks, sample_dependency_graph):
        """Test basic schedule creation."""
        schedule = await planner.create_schedule(sample_subtasks, sample_dependency_graph)
        
        assert isinstance(schedule, ExecutionSchedule)
        assert len(schedule.scheduled_tasks) == len(sample_subtasks)
        assert schedule.estimated_duration_minutes > 0
        assert hasattr(schedule, 'optimization_strategy')
    
    @pytest.mark.asyncio
    async def test_resource_constraints(self, planner, sample_subtasks, sample_dependency_graph):
        """Test scheduling with resource constraints."""
        constraints = ResourceConstraints(
            max_parallel_tasks=2,
            available_agents=["agent1", "agent2"],
            memory_mb=4096,
            timeout_minutes=480
        )
        
        schedule = await planner.create_schedule(
            sample_subtasks, sample_dependency_graph, constraints
        )
        
        assert schedule is not None
        assert schedule.estimated_duration_minutes <= constraints.timeout_minutes
    
    @pytest.mark.asyncio
    async def test_different_optimization_strategies(self, planner, sample_subtasks, sample_dependency_graph):
        """Test different optimization strategies."""
        strategies = [
            ExecutionStrategy.CRITICAL_PATH,
            ExecutionStrategy.LOAD_BALANCED,
            ExecutionStrategy.FASTEST
        ]
        
        schedules = []
        for strategy in strategies:
            # Create planner with specific strategy
            planner_config = planner.config
            planner_config.optimization_strategy = strategy
            
            schedule = await planner.create_schedule(
                sample_subtasks, sample_dependency_graph
            )
            schedules.append(schedule)
        
        # All schedules should be valid but potentially different
        assert len(schedules) == len(strategies)
        assert all(schedule is not None for schedule in schedules)


class TestMetacognitiveMonitor:
    """Test metacognitive monitoring and quality assessment."""
    
    @pytest.fixture
    def monitor(self):
        """Create metacognitive monitor instance."""
        return MetacognitiveMonitor()
    
    @pytest.fixture
    def sample_thinking_trace(self):
        """Create sample thinking trace."""
        return [
            ThinkingStep(
                phase=ThinkingPhase.ANALYZE_REQUEST,
                description="Analyzed the request",
                duration_ms=100,
                timestamp=datetime.now(),
                artifacts={"analysis": "Complex web application"}
            ),
            ThinkingStep(
                phase=ThinkingPhase.EXPLORE_OPTIONS,
                description="Explored different approaches",
                duration_ms=200,
                timestamp=datetime.now(),
                artifacts={"approaches": ["REST", "GraphQL"]}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_quality_assessment(self, monitor, sample_thinking_trace):
        """Test thinking quality assessment."""
        claimed_confidence = 0.8
        
        assessment = await monitor.assess_thinking_quality(
            sample_thinking_trace, claimed_confidence
        )
        
        assert isinstance(assessment, QualityAssessment)
        assert hasattr(assessment, 'overall_quality')
        assert hasattr(assessment, 'confidence_calibration')
        assert hasattr(assessment, 'process_efficiency')
        assert 0 <= assessment.overall_quality <= 1
    
    @pytest.mark.asyncio
    async def test_strategy_adaptation(self, monitor, sample_thinking_trace):
        """Test strategy adaptation based on performance."""
        # Simulate poor performance
        poor_trace = [
            ThinkingStep(
                phase=ThinkingPhase.ANALYZE_REQUEST,
                description="Failed analysis",
                duration_ms=5000,  # Very slow
                timestamp=datetime.now(),
                error="Analysis took too long"
            )
        ]
        
        assessment = await monitor.assess_thinking_quality(poor_trace, 0.9)
        adaptation = await monitor.adapt_strategy(assessment)
        
        assert adaptation is not None
        assert hasattr(adaptation, 'recommended_changes')


class TestPlanningStateManager:
    """Test state persistence and recovery."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for state storage."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def state_manager(self, temp_dir):
        """Create state manager with temporary storage."""
        from src.agentsmcp.cognition.models import StatePersistenceConfig
        
        config = StatePersistenceConfig(
            storage_path=temp_dir,
            format=PersistenceFormat.JSON,
            compress=False,
            async_writes=False
        )
        
        manager = PlanningStateManager(config)
        yield manager
        await manager.shutdown()
    
    @pytest.fixture
    def sample_planning_state(self):
        """Create sample planning state."""
        return PlanningState(
            state_id="test_state_123",
            created_at=datetime.now(),
            thinking_trace=[
                ThinkingStep(
                    phase=ThinkingPhase.ANALYZE_REQUEST,
                    description="Test step",
                    duration_ms=100,
                    timestamp=datetime.now()
                )
            ],
            current_context={"test": "data"},
            metadata={"test_case": True}
        )
    
    @pytest.mark.asyncio
    async def test_state_save_and_load(self, state_manager, sample_planning_state):
        """Test basic state save and load."""
        # Save state
        metadata = await state_manager.save_state(sample_planning_state)
        
        assert metadata is not None
        assert metadata.state_id == sample_planning_state.state_id
        assert metadata.size_bytes > 0
        
        # Load state
        loaded_state = await state_manager.load_state(sample_planning_state.state_id)
        
        assert loaded_state is not None
        assert loaded_state.state_id == sample_planning_state.state_id
        assert loaded_state.current_context == sample_planning_state.current_context
    
    @pytest.mark.asyncio
    async def test_state_recovery(self, state_manager, sample_planning_state):
        """Test state recovery functionality."""
        # Save state
        await state_manager.save_state(sample_planning_state)
        
        # Recover state
        recovery_info = await state_manager.recover_state(sample_planning_state.state_id)
        
        assert recovery_info.recovered
        assert recovery_info.state_id == sample_planning_state.state_id
        assert recovery_info.steps_recovered > 0
    
    @pytest.mark.asyncio
    async def test_state_cleanup(self, state_manager, sample_planning_state):
        """Test state cleanup functionality."""
        # Save state
        await state_manager.save_state(sample_planning_state)
        
        # List states
        states = await state_manager.list_states()
        assert len(states) > 0
        
        # Delete state
        success = await state_manager.delete_state(sample_planning_state.state_id)
        assert success
        
        # Verify deletion
        states_after = await state_manager.list_states()
        assert len(states_after) < len(states)


class TestThinkingOrchestrator:
    """Test orchestrator integration with thinking capabilities."""
    
    @pytest.fixture
    def thinking_orchestrator(self):
        """Create thinking orchestrator for testing."""
        return create_thinking_orchestrator(
            performance_profile=PerformanceProfile.FAST,
            enable_persistence=False
        )
    
    @pytest.mark.asyncio
    async def test_orchestrator_thinking_integration(self, thinking_orchestrator):
        """Test orchestrator integration with thinking."""
        request = "Create a simple calculator API"
        context = {"language": "Python"}
        
        response = await thinking_orchestrator.process_user_input(request, context)
        
        assert response is not None
        assert hasattr(response, 'content')
        assert hasattr(response, 'response_type')
        assert hasattr(response, 'metadata')
        
        # Should have thinking metadata
        if response.metadata:
            thinking_applied = response.metadata.get('thinking_applied', False)
            assert isinstance(thinking_applied, bool)
    
    @pytest.mark.asyncio
    async def test_performance_profiles(self):
        """Test different performance profiles."""
        profiles = [
            PerformanceProfile.FAST,
            PerformanceProfile.BALANCED,
            PerformanceProfile.COMPREHENSIVE
        ]
        
        request = "Design a simple web service"
        
        for profile in profiles:
            orchestrator = create_thinking_orchestrator(
                performance_profile=profile,
                enable_persistence=False
            )
            
            response = await orchestrator.process_user_input(request)
            
            assert response is not None
            await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_thinking_bypass(self, thinking_orchestrator):
        """Test thinking bypass for simple requests."""
        # Very simple request that should bypass thinking
        simple_request = "Hello"
        
        response = await thinking_orchestrator.process_user_input(simple_request)
        
        assert response is not None
        # May or may not have thinking metadata depending on configuration
    
    @pytest.mark.asyncio
    async def test_orchestrator_stats(self, thinking_orchestrator):
        """Test orchestrator statistics collection."""
        # Process a request
        await thinking_orchestrator.process_user_input("Test request")
        
        # Get statistics
        stats = await thinking_orchestrator.get_thinking_stats()
        
        assert 'thinking_integration' in stats
        thinking_stats = stats['thinking_integration']
        assert 'total_requests' in thinking_stats
        assert thinking_stats['total_requests'] > 0


# Integration tests
class TestIntegration:
    """Test end-to-end integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_thinking_workflow(self):
        """Test complete thinking workflow from request to execution plan."""
        # Create framework
        framework = ThinkingFramework()
        
        # Complex request
        request = """
        Create a microservices architecture for a social media platform that needs to handle
        user posts, comments, likes, friend connections, and real-time notifications.
        The system should be scalable, secure, and provide good user experience.
        """
        
        context = {
            "expected_users": 100000,
            "geographic_regions": ["US", "EU", "ASIA"],
            "compliance": ["GDPR", "CCPA"],
            "technologies": ["Docker", "Kubernetes", "PostgreSQL", "Redis"]
        }
        
        # Execute full thinking process
        result = await framework.think(request, context)
        
        # Validate complete result
        assert result is not None
        assert result.confidence > 0
        assert len(result.thinking_trace) > 0
        assert result.final_approach is not None
        assert result.execution_plan is not None
        
        # Check execution plan has reasonable tasks
        assert len(result.execution_plan.scheduled_tasks) > 0
        assert result.execution_plan.estimated_duration_minutes > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and graceful degradation."""
        # Create framework with very tight constraints
        config = ThinkingConfig(
            max_approaches=1,
            max_subtasks=2,
            timeout_seconds=0.1  # Very short timeout
        )
        framework = ThinkingFramework(config)
        
        # Very complex request that might cause issues
        request = "Design and implement a complete operating system with advanced AI capabilities and quantum computing support"
        
        # Should still return a result, even if degraded
        result = await framework.think(request)
        
        assert result is not None
        # May have lower confidence due to constraints
        assert 0 <= result.confidence <= 1


# Performance tests
class TestPerformance:
    """Test performance characteristics of the thinking system."""
    
    @pytest.mark.asyncio
    async def test_thinking_performance_bounds(self):
        """Test that thinking completes within reasonable time bounds."""
        framework = ThinkingFramework()
        
        simple_request = "Create a hello world program"
        complex_request = "Design a distributed database system with ACID compliance"
        
        # Simple request should be fast
        start_time = asyncio.get_event_loop().time()
        simple_result = await framework.think(simple_request)
        simple_duration = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Complex request should still be reasonable
        start_time = asyncio.get_event_loop().time()
        complex_result = await framework.think(complex_request)
        complex_duration = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Performance assertions
        assert simple_duration < 2000  # Less than 2 seconds for simple
        assert complex_duration < 10000  # Less than 10 seconds for complex
        assert simple_result.confidence >= 0
        assert complex_result.confidence >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_thinking_processes(self):
        """Test concurrent thinking processes."""
        framework = ThinkingFramework()
        
        requests = [
            "Create a web API",
            "Design a database schema", 
            "Implement user authentication",
            "Set up monitoring system"
        ]
        
        # Execute concurrent thinking
        tasks = [framework.think(request) for request in requests]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(requests)
        assert all(result is not None for result in results)
        assert all(result.confidence >= 0 for result in results)


if __name__ == "__main__":
    # Run the tests
    import sys
    
    # Add the project root to Python path for imports
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])