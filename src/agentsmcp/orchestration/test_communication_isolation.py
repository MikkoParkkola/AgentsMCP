#!/usr/bin/env python3
"""
Test Script for Communication Isolation

This script tests the strict orchestrator-only communication architecture to ensure:
1. Simple greetings don't spawn agents
2. Complex tasks delegate to agents but show only orchestrator responses
3. Agent outputs are completely intercepted and never shown to user
4. Response synthesis works correctly
5. Communication isolation is maintained

Usage: python test_communication_isolation.py
"""

import asyncio
import logging
import sys
from datetime import datetime

# Setup logging to see what's happening internally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import orchestrator components
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from agentsmcp.orchestration.orchestrator import Orchestrator, OrchestratorConfig, OrchestratorMode
    from agentsmcp.orchestration.task_classifier import TaskClassification
    print("✅ Successfully imported orchestrator components")
except ImportError as e:
    print(f"❌ Failed to import orchestrator components: {e}")
    sys.exit(1)


async def test_simple_greeting():
    """Test that simple greetings don't spawn agents."""
    print("\n🧪 TEST 1: Simple Greeting (should NOT spawn agents)")
    print("-" * 50)
    
    orchestrator = Orchestrator()
    
    test_inputs = [
        "hello",
        "hi there", 
        "good morning",
        "hey",
        "how are you"
    ]
    
    for user_input in test_inputs:
        print(f"\n👤 User: {user_input}")
        
        response = await orchestrator.process_user_input(user_input, {})
        
        print(f"🤖 Orchestrator: {response.content}")
        print(f"📊 Type: {response.response_type}, Agents: {response.agents_consulted}, Time: {response.processing_time_ms}ms")
        
        # Verify no agents were spawned for simple greetings
        assert response.response_type == "simple", f"Expected simple response, got {response.response_type}"
        assert len(response.agents_consulted) == 0, f"Expected no agents, but got {response.agents_consulted}"
        
        print("✅ PASSED - No agents spawned for simple greeting")


async def test_complex_task():
    """Test that complex tasks delegate to agents but show unified response."""
    print("\n🧪 TEST 2: Complex Task (should delegate to agents but show unified response)")
    print("-" * 70)
    
    orchestrator = Orchestrator()
    
    test_inputs = [
        "write a python function to calculate fibonacci numbers",
        "analyze the performance of this code and suggest improvements", 
        "create a web application with user authentication"
    ]
    
    for user_input in test_inputs:
        print(f"\n👤 User: {user_input}")
        
        response = await orchestrator.process_user_input(user_input, {})
        
        print(f"🤖 Orchestrator: {response.content[:200]}...")
        print(f"📊 Type: {response.response_type}, Agents: {response.agents_consulted}, Time: {response.processing_time_ms}ms")
        
        # Verify agents were consulted but response is unified
        assert response.response_type in ["agent_delegated", "multi_agent", "fallback"], f"Expected agent involvement, got {response.response_type}"
        
        # Verify response doesn't contain agent identifiers
        assert "🧩" not in response.content, "Response contains agent identifier emoji"
        assert "Agent " not in response.content, "Response contains 'Agent ' identifier"
        assert not response.content.lower().startswith("i am"), "Response starts with agent self-identification"
        
        print("✅ PASSED - Agents consulted but response unified")


async def test_communication_interception():
    """Test that agent communications are properly intercepted."""
    print("\n🧪 TEST 3: Communication Interception")
    print("-" * 50)
    
    orchestrator = Orchestrator()
    
    # Test the communication interceptor directly
    test_agent_responses = [
        "🧩 codex:\nI am a coding agent and I can help you with Python programming.",
        "Agent Claude: As an analysis agent, I'll examine this code for you.",
        "Hello! How can I assist you today?",  # Common agent greeting
    ]
    
    for agent_response in test_agent_responses:
        print(f"\n🔍 Testing agent response: {agent_response[:50]}...")
        
        intercepted = orchestrator.communication_interceptor.intercept_response(
            "test_agent", agent_response, {}
        )
        
        sanitized_content = intercepted.processed_response.get("sanitized_content", "")
        
        print(f"🚫 Intercepted: {sanitized_content}")
        print(f"📊 Sanitization applied: {intercepted.sanitization_applied}")
        
        # Verify agent identifiers are removed
        assert "🧩" not in sanitized_content, "Agent emoji not removed"
        assert "Agent " not in sanitized_content, "Agent prefix not removed"
        assert not sanitized_content.lower().startswith("i am"), "Agent self-identification not removed"
        
        print("✅ PASSED - Agent identifiers properly removed")


async def test_task_classification():
    """Test that task classification works correctly."""
    print("\n🧪 TEST 4: Task Classification")
    print("-" * 50)
    
    orchestrator = Orchestrator()
    classifier = orchestrator.task_classifier
    
    test_cases = [
        ("hello", TaskClassification.SIMPLE_RESPONSE),
        ("help", TaskClassification.SIMPLE_RESPONSE),
        ("status", TaskClassification.SIMPLE_RESPONSE),
        ("write a python function", TaskClassification.SINGLE_AGENT_NEEDED),
        ("analyze this code", TaskClassification.SINGLE_AGENT_NEEDED),
        ("create a full web application with database and API", TaskClassification.MULTI_AGENT_NEEDED),
        ("design and implement a microservices architecture", TaskClassification.MULTI_AGENT_NEEDED),
    ]
    
    for user_input, expected_classification in test_cases:
        print(f"\n🔍 Classifying: {user_input}")
        
        result = await classifier.classify_task(user_input, {})
        
        print(f"📊 Classification: {result.classification.value}, Confidence: {result.confidence:.2f}")
        print(f"📊 Required agents: {result.required_agents}")
        print(f"📊 Reasoning: {result.reasoning}")
        
        assert result.classification == expected_classification, f"Expected {expected_classification.value}, got {result.classification.value}"
        
        print("✅ PASSED - Correct classification")


async def test_orchestrator_stats():
    """Test that orchestrator statistics are tracked correctly."""
    print("\n🧪 TEST 5: Orchestrator Statistics")
    print("-" * 50)
    
    orchestrator = Orchestrator()
    
    # Process several different types of requests
    await orchestrator.process_user_input("hello", {})
    await orchestrator.process_user_input("write a function", {})
    await orchestrator.process_user_input("status", {})
    
    stats = orchestrator.get_orchestrator_stats()
    
    print(f"📊 Total requests: {stats['total_requests']}")
    print(f"📊 Simple responses: {stats['simple_responses']}")
    print(f"📊 Agent delegations: {stats['agent_delegations']}")
    print(f"📊 Multi-agent tasks: {stats['multi_agent_tasks']}")
    
    assert stats['total_requests'] == 3, f"Expected 3 total requests, got {stats['total_requests']}"
    assert stats['simple_responses'] >= 1, f"Expected at least 1 simple response, got {stats['simple_responses']}"
    
    print("✅ PASSED - Statistics tracked correctly")


async def run_all_tests():
    """Run all communication isolation tests."""
    print("🚀 STARTING COMMUNICATION ISOLATION TESTS")
    print("=" * 60)
    
    try:
        await test_simple_greeting()
        await test_complex_task()
        await test_communication_interception() 
        await test_task_classification()
        await test_orchestrator_stats()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Communication isolation is working correctly")
        print("✅ Users will only see orchestrator responses")
        print("✅ Agent outputs are properly intercepted and synthesized")
        print("✅ Simple tasks don't spawn unnecessary agents")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        logger.error(f"Test error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    print("🧪 AgentsMCP Communication Isolation Test Suite")
    print(f"🕒 Started at: {datetime.now()}")
    
    success = asyncio.run(run_all_tests())
    
    if success:
        print(f"\n✅ Test suite completed successfully at {datetime.now()}")
        sys.exit(0)
    else:
        print(f"\n❌ Test suite failed at {datetime.now()}")
        sys.exit(1)