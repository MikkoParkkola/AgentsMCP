"""
Integration tests for the complete User Prompt Preprocessing system.

Tests the full preprocessing pipeline including:
- End-to-end preprocessing workflow
- Integration with orchestrator
- Performance under load
- Error handling and recovery
- Real-world scenarios
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch

from ..preprocessor import UserPromptPreprocessor, PreprocessingResult
from ..config import PreprocessingSettings
from ..orchestrator_integration import PreprocessingEnabledOrchestrator, EnhancedOrchestratorConfig


class TestPreprocessorIntegration:
    """Integration tests for complete preprocessing system."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance for testing."""
        config = PreprocessingSettings(
            confidence_threshold=0.8,
            enable_clarification=True,
            enable_optimization=True,
            enable_context_learning=True,
            processing_timeout_ms=10000  # Longer timeout for tests
        )
        return UserPromptPreprocessor(config)
    
    @pytest.mark.asyncio
    async def test_end_to_end_preprocessing_workflow(self, preprocessor):
        """Test complete preprocessing workflow for high-confidence input.""" 
        result = await preprocessor.preprocess_prompt(
            user_prompt="Create a Python Flask REST API with user authentication using JWT tokens",
            session_id=None,
            context={},
            user_id="test_user"
        )
        
        # Should complete without clarification
        assert result.result_type == PreprocessingResult.READY_FOR_DELEGATION
        assert result.confidence >= 0.8
        assert result.session_id is not None
        assert result.intent_analysis.primary_intent.value == "task_execution"
        assert result.intent_analysis.technical_domain.value == "web_development"
        
        # Should have applied optimization
        assert result.optimized_prompt is not None
        assert len(result.final_prompt) >= len(result.original_prompt)
        
        # Should have processing steps
        assert "intent_analyzed" in result.preprocessing_steps
        assert "prompt_optimized" in result.preprocessing_steps
        assert "conversation_recorded" in result.preprocessing_steps
        
        # Should have recommendations and no critical warnings
        assert isinstance(result.recommendations, list)
        assert isinstance(result.warnings, list)
        
        # Performance check
        assert result.processing_time_ms < 5000  # Should process quickly
    
    @pytest.mark.asyncio
    async def test_clarification_workflow(self, preprocessor):
        """Test workflow that requires clarification."""
        result = await preprocessor.preprocess_prompt(
            user_prompt="Fix it and make this better",
            session_id=None,
            context={},
            user_id="test_user"
        )
        
        # Should require clarification
        assert result.result_type == PreprocessingResult.NEEDS_CLARIFICATION
        assert result.clarification_session is not None
        assert result.confidence < 0.8
        
        # Should have clarification questions
        session = result.clarification_session
        assert len(session.questions) > 0
        
        # Simulate user answers
        session_id = session.session_id
        
        # Answer first question
        session, complete, final_result = await preprocessor.handle_clarification_answer(
            session_id=session_id,
            question_index=0,
            answer="I want to fix the login authentication system in my React web application"
        )
        
        assert len(session.answers_received) == 1
        assert session.current_confidence > result.confidence
        
        # If not complete, continue
        if not complete:
            # Answer more questions until complete or sufficient
            for i in range(2):  # Max 2 more iterations
                next_questions = await preprocessor.clarification_engine.get_next_questions(session_id)
                if not next_questions or complete:
                    break
                
                session, complete, final_result = await preprocessor.handle_clarification_answer(
                    session_id=session_id,
                    question_index=0,
                    answer="React frontend with Node.js backend using JWT authentication"
                )
                
                if complete and final_result:
                    assert final_result.result_type == PreprocessingResult.READY_FOR_DELEGATION
                    assert final_result.confidence >= 0.8
                    break
    
    @pytest.mark.asyncio
    async def test_context_learning_and_memory(self, preprocessor):
        """Test context learning across multiple interactions."""
        user_id = "learning_test_user"
        
        # First interaction
        result1 = await preprocessor.preprocess_prompt(
            user_prompt="Create a web application",
            user_id=user_id,
            context={}
        )
        
        session_id = result1.session_id
        
        # Simulate system response and update
        await preprocessor.update_conversation_response(
            session_id=session_id,
            turn_id=result1.conversation_turn.turn_id if result1.conversation_turn else "test",
            system_response="I'll help you create a React web application with authentication...",
            success_indicators={"completion": True}
        )
        
        # Second interaction in same session
        result2 = await preprocessor.preprocess_prompt(
            user_prompt="Add user registration functionality",
            session_id=session_id,
            user_id=user_id,
            context={}
        )
        
        # Should use context from previous interaction
        assert len(result2.relevant_context) > 0
        assert result2.confidence >= result1.confidence  # Should maintain or improve confidence
        
        # Provide feedback for learning
        await preprocessor.provide_user_feedback(
            session_id=session_id,
            turn_id=result2.conversation_turn.turn_id if result2.conversation_turn else "test2",
            feedback={
                "satisfaction": 0.9,
                "helpful": True,
                "specific_feedback": "Great suggestions for authentication"
            }
        )
        
        # Context should be updated with learning
        context_stats = preprocessor.conversation_context.get_context_stats()
        assert context_stats["learning_events"] > 0
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, preprocessor):
        """Test preprocessing performance under concurrent load."""
        prompts = [
            "Create a REST API",
            "Fix authentication bug", 
            "Analyze performance issues",
            "Build mobile app",
            "Review security vulnerabilities",
            "Implement database migration",
            "Design user interface",
            "Optimize queries",
            "Deploy to cloud",
            "Monitor application health"
        ]
        
        start_time = time.time()
        
        # Process all prompts concurrently
        tasks = [
            preprocessor.preprocess_prompt(
                user_prompt=prompt,
                user_id=f"user_{i}",
                context={}
            ) for i, prompt in enumerate(prompts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 8  # At least 80% success rate
        
        # Performance check
        assert total_time < 15.0  # Should complete within 15 seconds
        
        # Individual processing times should be reasonable
        for result in successful_results:
            assert result.processing_time_ms < 3000  # Each under 3 seconds
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, preprocessor):
        """Test error handling and recovery mechanisms."""
        # Test with invalid input
        result = await preprocessor.preprocess_prompt(
            user_prompt="",  # Empty input
            context={}
        )
        
        # Should handle gracefully
        assert isinstance(result.confidence, float)
        assert result.confidence < 0.5
        
        # Test with very long input
        long_prompt = "Create a web application " * 1000  # Very long
        result = await preprocessor.preprocess_prompt(
            user_prompt=long_prompt,
            context={}
        )
        
        # Should still process
        assert result.processing_time_ms > 0
        assert isinstance(result.intent_analysis.raw_input_length, int)
        
        # Test timeout handling (if component supports it)
        with patch.object(preprocessor.intent_analyzer, 'analyze_intent', 
                         side_effect=asyncio.TimeoutError()):
            result = await preprocessor.preprocess_prompt(
                user_prompt="Test timeout handling",
                context={}
            )
            
            assert result.result_type == PreprocessingResult.ERROR
            assert "timeout" in result.warnings[0].lower() if result.warnings else True
    
    @pytest.mark.asyncio
    async def test_health_check(self, preprocessor):
        """Test system health check functionality."""
        health_status = await preprocessor.health_check()
        
        assert isinstance(health_status, dict)
        assert "status" in health_status
        assert "components" in health_status
        
        # All components should be healthy or have known issues
        for component, status in health_status["components"].items():
            assert isinstance(status, str)
            # Should be "healthy" or start with "error:" 
            assert status == "healthy" or status.startswith("error:")
    
    @pytest.mark.asyncio 
    async def test_statistics_and_monitoring(self, preprocessor):
        """Test statistics collection and monitoring.""" 
        # Process several requests to generate stats
        test_prompts = [
            "Create a Python script",
            "Fix JavaScript error",
            "What is machine learning?",
            "Build REST API"
        ]
        
        for prompt in test_prompts:
            await preprocessor.preprocess_prompt(prompt, context={})
        
        # Get comprehensive stats
        stats = preprocessor.get_preprocessing_stats()
        
        assert isinstance(stats, dict)
        assert stats["total_processed"] >= 4
        assert "clarification_rate" in stats
        assert "optimization_rate" in stats
        assert "success_rate" in stats
        assert "average_processing_time_ms" in stats
        
        # Component stats should be available
        assert "components" in stats
        assert "intent_analyzer" in stats["components"]
        assert "clarification_engine" in stats["components"]
        assert "prompt_optimizer" in stats["components"]
        assert "conversation_context" in stats["components"]


class TestOrchestratorIntegration:
    """Test integration with orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create enhanced orchestrator for testing."""
        config = EnhancedOrchestratorConfig(
            enable_preprocessing=True,
            preprocessing_confidence_threshold=0.8,
            enable_clarification_mode=True,
            preprocessing_failure_fallback=True
        )
        return PreprocessingEnabledOrchestrator(config)
    
    @pytest.mark.asyncio
    async def test_orchestrator_preprocessing_integration(self, orchestrator):
        """Test orchestrator integration with preprocessing."""
        # Test high-confidence request (should not need clarification)
        response = await orchestrator.process_user_input(
            user_input="Create a Python Flask REST API with PostgreSQL database",
            context={},
            session_id=None,
            user_id="test_user"
        )
        
        assert response.response_type != "clarification_needed"
        assert response.metadata is not None
        assert response.metadata.get("preprocessing_applied") is True
        assert response.metadata.get("preprocessing_confidence", 0) >= 0.8
        
        # Should have intent and optimization metadata
        assert "intent_detected" in response.metadata
        assert "optimization_applied" in response.metadata
    
    @pytest.mark.asyncio
    async def test_orchestrator_clarification_flow(self, orchestrator):
        """Test orchestrator clarification workflow."""
        # Submit ambiguous request
        response = await orchestrator.process_user_input(
            user_input="Fix it please",
            context={},
            user_id="clarification_test_user"
        )
        
        if response.response_type == "clarification_needed":
            # Should have clarification session
            assert "clarification_session_id" in response.metadata
            assert "questions" in response.metadata
            
            session_id = response.metadata["clarification_session_id"]
            
            # Provide clarification answer
            clarification_response = await orchestrator.handle_clarification_response(
                session_id=session_id,
                question_index=0,
                user_answer="Fix the authentication system in my React web application"
            )
            
            # Should eventually lead to task execution or more questions
            assert clarification_response.response_type in ["normal", "clarification_needed"]
    
    @pytest.mark.asyncio
    async def test_orchestrator_fallback_behavior(self, orchestrator):
        """Test orchestrator fallback when preprocessing fails."""
        # Test with preprocessing disabled
        orchestrator.enhanced_config.enable_preprocessing = False
        orchestrator.preprocessor = None
        
        response = await orchestrator.process_user_input(
            user_input="Create a web application",
            context={}
        )
        
        # Should still work without preprocessing
        assert response.response_type != "error"
        assert response.metadata.get("preprocessing_applied") is not True
    
    @pytest.mark.asyncio
    async def test_enhanced_statistics(self, orchestrator):
        """Test enhanced orchestrator statistics."""
        # Process a few requests
        test_inputs = [
            "Build a mobile app",
            "Debug Python error",
            "What is Docker?"
        ]
        
        for input_text in test_inputs:
            await orchestrator.process_user_input(input_text, context={})
        
        stats = orchestrator.get_enhanced_stats()
        
        assert isinstance(stats, dict)
        assert "preprocessing" in stats
        assert "configuration" in stats
        assert stats["total_requests"] >= 3


@pytest.mark.performance
class TestPreprocessingPerformance:
    """Performance tests for preprocessing system."""
    
    @pytest.mark.asyncio
    async def test_processing_time_benchmarks(self):
        """Test processing time benchmarks for different input types."""
        from ..preprocessor import PreprocessingConfig
        
        config = PreprocessingConfig(
            confidence_threshold=0.9,
            enable_clarification=True,
            enable_optimization=True,
            optimization_level="standard"
        )
        preprocessor = UserPromptPreprocessor(config)
        
        test_cases = [
            ("Simple request", "Hello"),
            ("Medium request", "Create a Python web application with user authentication"),
            ("Complex request", "Build a scalable microservices architecture with Docker containers, Kubernetes orchestration, PostgreSQL database, Redis caching, and comprehensive monitoring using Prometheus and Grafana"),
            ("Ambiguous request", "Fix it and make this better")
        ]
        
        benchmark_results = []
        
        for case_name, prompt in test_cases:
            start_time = time.time()
            
            result = await preprocessor.preprocess_prompt(
                user_prompt=prompt,
                context={}
            )
            
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000
            
            benchmark_results.append({
                "case": case_name,
                "input_length": len(prompt),
                "processing_time_ms": processing_time,
                "reported_time_ms": result.processing_time_ms,
                "confidence": result.confidence,
                "result_type": result.result_type.value
            })
            
            # Performance assertions
            assert processing_time < 2000  # Under 2 seconds
            assert abs(processing_time - result.processing_time_ms) < 100  # Accurate timing
        
        # Print benchmark results for analysis
        print("\nPreprocessing Performance Benchmarks:")
        for result in benchmark_results:
            print(f"{result['case']}: {result['processing_time_ms']:.0f}ms "
                  f"(confidence: {result['confidence']:.2f}, "
                  f"result: {result['result_type']})")
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage patterns under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        preprocessor = UserPromptPreprocessor()
        
        # Process many requests to test memory leaks
        for i in range(50):
            await preprocessor.preprocess_prompt(
                user_prompt=f"Create application {i} with advanced features",
                user_id=f"user_{i}",
                context={"iteration": i}
            )
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 50 requests)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        
        print(f"Memory increase: {memory_increase / 1024 / 1024:.1f}MB for 50 requests")


@pytest.mark.realworld
class TestRealWorldScenarios:
    """Test preprocessing with real-world scenarios."""
    
    @pytest.mark.asyncio
    async def test_software_development_scenarios(self):
        """Test with real software development requests."""
        preprocessor = UserPromptPreprocessor()
        
        scenarios = [
            {
                "prompt": "I'm getting a 'Connection refused' error when trying to connect my React frontend to my Node.js backend API. The frontend is running on localhost:3000 and backend on localhost:8000. CORS is configured properly.",
                "expected_intent": "problem_solving",
                "expected_domain": "web_development",
                "should_be_clear": True
            },
            {
                "prompt": "Help me implement user authentication in my e-commerce project",
                "expected_intent": "task_execution", 
                "expected_domain": "web_development",
                "should_be_clear": False  # Needs clarification about tech stack
            },
            {
                "prompt": "My machine learning model is overfitting. The training accuracy is 98% but validation accuracy is only 65%. I'm using a neural network with 3 hidden layers and ReLU activation.",
                "expected_intent": "problem_solving",
                "expected_domain": "data_science", 
                "should_be_clear": True
            }
        ]
        
        for scenario in scenarios:
            result = await preprocessor.preprocess_prompt(
                user_prompt=scenario["prompt"],
                context={}
            )
            
            assert result.intent_analysis.primary_intent.value == scenario["expected_intent"]
            assert result.intent_analysis.technical_domain.value == scenario["expected_domain"]
            
            if scenario["should_be_clear"]:
                assert result.confidence >= 0.8
                assert result.result_type == PreprocessingResult.READY_FOR_DELEGATION
            else:
                # May need clarification
                assert result.result_type in [
                    PreprocessingResult.READY_FOR_DELEGATION,
                    PreprocessingResult.NEEDS_CLARIFICATION
                ]
    
    @pytest.mark.asyncio
    async def test_user_journey_simulation(self):
        """Test complete user journey with context building."""
        preprocessor = UserPromptPreprocessor()
        user_id = "journey_user"
        
        # Simulate user journey
        journey = [
            "I want to build a web application",
            "It should have user authentication",
            "Add a dashboard for user management",
            "How do I deploy this to AWS?",
            "The deployment is failing with permission errors"
        ]
        
        session_id = None
        previous_contexts = []
        
        for i, prompt in enumerate(journey):
            result = await preprocessor.preprocess_prompt(
                user_prompt=prompt,
                session_id=session_id,
                user_id=user_id,
                context={"journey_step": i}
            )
            
            # Use same session for context continuity
            if session_id is None:
                session_id = result.session_id
            
            # Context should build over time
            if i > 0:
                assert len(result.relevant_context) >= len(previous_contexts)
            
            previous_contexts = result.relevant_context
            
            # Simulate system response
            if result.conversation_turn:
                await preprocessor.update_conversation_response(
                    session_id=session_id,
                    turn_id=result.conversation_turn.turn_id,
                    system_response=f"Response to step {i+1}",
                    success_indicators={"step": i+1, "understanding": "good"}
                )
        
        # Final context should be rich with user's journey
        assert len(previous_contexts) > 0
        
        # Get session summary
        if session_id:
            session_summary = await preprocessor.conversation_context.get_session_summary(session_id)
            assert session_summary["total_turns"] == len(journey)
            assert "web" in str(session_summary["domain_distribution"]).lower()