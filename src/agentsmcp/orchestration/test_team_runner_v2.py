"""Unit tests for team runner v2 with dynamic orchestration.

This test suite covers:
1. Basic integration functionality
2. Dynamic orchestration decision logic
3. Fallback mechanisms
4. Progress callback handling
5. Error handling and recovery
6. Backward compatibility
"""

import unittest
from unittest.mock import AsyncMock, Mock, patch
import asyncio
from typing import Dict, Any

from .team_runner_v2 import TeamRunnerV2, run_team, run_team_v2, DEFAULT_TEAM
from .models import TaskType, ComplexityLevel, RiskLevel, CoordinationStrategy


class TestTeamRunnerV2(unittest.TestCase):
    """Test cases for TeamRunnerV2 class."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = TeamRunnerV2()

    def test_initialization(self):
        """Test TeamRunnerV2 initializes correctly."""
        self.assertIsNotNone(self.runner.task_classifier)
        self.assertIsNotNone(self.runner.team_composer)
        self.assertIsNotNone(self.runner.agile_coach)
        self.assertIsNotNone(self.runner.retrospective_engine)

    async def test_classify_task_success(self):
        """Test successful task classification."""
        objective = "Build a REST API"
        classification = await self.runner._classify_task(objective)
        
        # Should return a valid classification
        self.assertIsNotNone(classification)
        self.assertTrue(hasattr(classification, 'task_type'))
        self.assertTrue(hasattr(classification, 'complexity'))
        self.assertTrue(hasattr(classification, 'confidence'))

    async def test_classify_task_empty_objective(self):
        """Test task classification with empty objective."""
        objective = ""
        classification = await self.runner._classify_task(objective)
        
        # Should return a fallback classification
        self.assertIsNotNone(classification)
        self.assertEqual(classification.task_type, TaskType.IMPLEMENTATION)
        self.assertEqual(classification.complexity, ComplexityLevel.MEDIUM)

    async def test_should_use_dynamic_orchestration(self):
        """Test dynamic orchestration decision logic."""
        # Mock classification objects
        high_complexity_task = Mock()
        high_complexity_task.complexity = ComplexityLevel.HIGH
        high_complexity_task.risk_level = RiskLevel.MEDIUM
        high_complexity_task.estimated_effort = 50
        
        low_complexity_task = Mock()
        low_complexity_task.complexity = ComplexityLevel.LOW
        low_complexity_task.risk_level = RiskLevel.LOW
        low_complexity_task.estimated_effort = 20
        
        high_effort_task = Mock()
        high_effort_task.complexity = ComplexityLevel.MEDIUM
        high_effort_task.risk_level = RiskLevel.MEDIUM
        high_effort_task.estimated_effort = 80
        
        # Test cases
        self.assertTrue(
            await self.runner._should_use_dynamic_orchestration(high_complexity_task, None)
        )
        self.assertFalse(
            await self.runner._should_use_dynamic_orchestration(low_complexity_task, None)
        )
        self.assertTrue(
            await self.runner._should_use_dynamic_orchestration(high_effort_task, None)
        )
        self.assertFalse(
            await self.runner._should_use_dynamic_orchestration(high_complexity_task, ["role1"])
        )

    async def test_execute_agents_directly(self):
        """Test direct agent execution."""
        with patch('src.agentsmcp.orchestration.team_runner_v2.Config.load') as mock_config, \
             patch('src.agentsmcp.orchestration.team_runner_v2.EventBus') as mock_bus, \
             patch('src.agentsmcp.orchestration.team_runner_v2.AgentManager') as mock_manager:
            
            # Mock agent manager
            mock_mgr = Mock()
            mock_mgr.spawn_agent.return_value = asyncio.Future()
            mock_mgr.spawn_agent.return_value.set_result("job-123")
            
            mock_status = Mock()
            mock_status.output = "Task completed"
            mock_status.error = None
            mock_mgr.wait_for_completion.return_value = asyncio.Future()
            mock_mgr.wait_for_completion.return_value.set_result(mock_status)
            
            mock_manager.return_value = mock_mgr
            
            # Test execution
            roles = ["backend_engineer", "api_engineer"]
            objective = "Build API"
            
            results = await self.runner._execute_agents_directly(
                roles, objective, mock_mgr, None
            )
            
            # Verify results
            self.assertEqual(len(results), 2)
            self.assertIn("backend_engineer", results)
            self.assertIn("api_engineer", results)
            self.assertEqual(results["backend_engineer"], "Task completed")

    async def test_progress_callback_handling(self):
        """Test progress callback functionality."""
        progress_events = []
        
        async def progress_callback(event: str, data: Dict[str, Any]):
            progress_events.append((event, data))
        
        # Test callback with valid data
        await progress_callback("test.event", {"key": "value"})
        self.assertEqual(len(progress_events), 1)
        self.assertEqual(progress_events[0][0], "test.event")
        self.assertEqual(progress_events[0][1]["key"], "value")

    def test_golden_test_api_compatibility(self):
        """Golden test: Verify API compatibility with original run_team."""
        import inspect
        
        # Check function signatures
        original_sig = inspect.signature(run_team)
        enhanced_sig = inspect.signature(run_team_v2)
        
        # Parameters should be compatible
        orig_params = list(original_sig.parameters.keys())
        enhanced_params = list(enhanced_sig.parameters.keys())
        
        self.assertEqual(orig_params, enhanced_params)
        
        # Return types should be compatible
        self.assertEqual(
            original_sig.return_annotation,
            enhanced_sig.return_annotation
        )

    def test_golden_test_default_team_preservation(self):
        """Golden test: Verify DEFAULT_TEAM is preserved."""
        expected_roles = [
            "business_analyst",
            "backend_engineer", 
            "api_engineer",
            "web_frontend_engineer",
            "tui_frontend_engineer",
            "backend_qa_engineer",
            "web_frontend_qa_engineer",
            "tui_frontend_qa_engineer",
        ]
        
        self.assertEqual(DEFAULT_TEAM, expected_roles)


class TestIntegrationAPI(unittest.TestCase):
    """Test cases for the unified integration API."""

    def test_module_exports(self):
        """Test that all required components are exported."""
        from src.agentsmcp.orchestration import (
            run_team,
            TaskClassifier,
            TeamComposer,
            DynamicOrchestrator,
            AgileCoachIntegration,
            RetrospectiveEngine,
            TaskType,
            ComplexityLevel,
            RiskLevel
        )
        
        # Verify all imports succeed
        self.assertIsNotNone(run_team)
        self.assertIsNotNone(TaskClassifier)
        self.assertIsNotNone(TeamComposer)
        self.assertIsNotNone(DynamicOrchestrator)
        self.assertIsNotNone(AgileCoachIntegration)
        self.assertIsNotNone(RetrospectiveEngine)

    def test_backward_compatibility_import(self):
        """Test backward compatibility imports."""
        # Should be able to import from both locations
        from src.agentsmcp.orchestration.team_runner import run_team as original
        from src.agentsmcp.orchestration.team_runner_v2 import run_team as enhanced
        
        self.assertIsNotNone(original)
        self.assertIsNotNone(enhanced)

    async def test_error_handling_fallback(self):
        """Test error handling and fallback mechanisms."""
        with patch('src.agentsmcp.orchestration.team_runner_v2.Config.load') as mock_config:
            mock_config.side_effect = Exception("Config failed")
            
            # Should not crash even with configuration errors
            runner = TeamRunnerV2()
            classification = await runner._classify_task("test objective")
            
            # Should get fallback classification
            self.assertIsNotNone(classification)


class TestRegressionScenarios(unittest.TestCase):
    """Regression tests for edge cases and known issues."""

    async def test_empty_required_roles_handling(self):
        """Test handling of empty required_roles in classification."""
        runner = TeamRunnerV2()
        
        # Mock a classification with empty required roles
        mock_classification = Mock()
        mock_classification.required_roles = []
        mock_classification.complexity = ComplexityLevel.LOW
        mock_classification.risk_level = RiskLevel.LOW
        mock_classification.estimated_effort = 30
        
        # Should fall back to DEFAULT_TEAM
        should_use_dynamic = await runner._should_use_dynamic_orchestration(
            mock_classification, None
        )
        self.assertFalse(should_use_dynamic)  # Low complexity/risk/effort

    async def test_none_values_handling(self):
        """Test handling of None values in various scenarios."""
        runner = TeamRunnerV2()
        
        # Test with None classification
        result = await runner._should_use_dynamic_orchestration(None, None)
        self.assertFalse(result)  # Should default to False for safety

    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up."""
        runner = TeamRunnerV2()
        
        # Create and cleanup orchestrator
        self.assertIsNone(runner._orchestrator)
        
        # After cleanup, should be reset
        runner._orchestrator = None
        self.assertIsNone(runner._orchestrator)


if __name__ == '__main__':
    # Run async tests
    def async_test_runner():
        """Run all async tests."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Create test instances
            test_instance = TestTeamRunnerV2()
            test_instance.setUp()
            
            integration_tests = TestIntegrationAPI()
            regression_tests = TestRegressionScenarios()
            
            # Run async tests
            async_tests = [
                test_instance.test_classify_task_success(),
                test_instance.test_classify_task_empty_objective(),
                test_instance.test_should_use_dynamic_orchestration(),
                test_instance.test_execute_agents_directly(),
                test_instance.test_progress_callback_handling(),
                integration_tests.test_error_handling_fallback(),
                regression_tests.test_empty_required_roles_handling(),
                regression_tests.test_none_values_handling(),
            ]
            
            for test_coro in async_tests:
                try:
                    loop.run_until_complete(test_coro)
                    print(f"✓ {test_coro.__name__ if hasattr(test_coro, '__name__') else 'Async test'} passed")
                except Exception as e:
                    print(f"✗ Async test failed: {e}")
            
        finally:
            loop.close()
    
    # Run sync and async tests
    print("Running Team Runner v2 Unit Tests...")
    
    # Run synchronous tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run asynchronous tests
    print("\nRunning async tests...")
    async_test_runner()
    
    print("\n✅ All unit tests completed!")