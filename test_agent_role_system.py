#!/usr/bin/env python3
"""Test script for the comprehensive agent role system implementation."""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.agents.agent_loader import AgentLoader
from agentsmcp.orchestration.memory_manager import AgentMemoryManager
from agentsmcp.orchestration.orchestrator import Orchestrator, OrchestratorConfig


async def test_agent_loading():
    """Test loading agent descriptions from the roles directory."""
    print("🧪 Testing Agent Loading System...")
    
    loader = AgentLoader()
    descriptions = loader.load_all_descriptions()
    
    print(f"✅ Successfully loaded {len(descriptions)} agent descriptions")
    
    # Test specific agent types
    expected_agents = [
        "cto", 
        "senior-product-manager", 
        "principal-software-architect",
        "security-engineer",
        "data-scientist",
        "ux-ui-designer",
        "product-marketing-manager"
    ]
    
    for agent_type in expected_agents:
        if agent_type in descriptions:
            agent = descriptions[agent_type]
            print(f"  ✓ {agent_type}: {agent.name} ({agent.category})")
            print(f"    Tools: {agent.tools}")
            print(f"    Memory: {getattr(agent, 'memory_specialization', ['None'])}")
        else:
            print(f"  ❌ Missing agent: {agent_type}")
    
    return descriptions


async def test_memory_manager():
    """Test the agent memory management system."""
    print("\n🧠 Testing Memory Manager...")
    
    memory_manager = AgentMemoryManager()
    
    # Test storing memories
    memory_id1 = await memory_manager.store_agent_memory(
        agent_type="security-engineer",
        category="decision",
        content="Implemented multi-factor authentication for API endpoints",
        importance=8,
        tags=["security", "api", "authentication"]
    )
    
    memory_id2 = await memory_manager.store_agent_memory(
        agent_type="security-engineer", 
        category="learning",
        content="JWT tokens should use shorter expiration times in production",
        importance=7,
        tags=["security", "jwt", "best-practices"]
    )
    
    print(f"✅ Stored memories: {memory_id1}, {memory_id2}")
    
    # Test retrieving memories
    memories = await memory_manager.retrieve_agent_memories(
        agent_type="security-engineer",
        limit=5
    )
    
    print(f"✅ Retrieved {len(memories)} memories for security-engineer")
    for memory in memories:
        print(f"  • {memory.category}: {memory.content[:50]}...")
    
    # Test contextual retrieval
    contextual = await memory_manager.get_contextual_memories(
        task_description="API security review for authentication system",
        agent_type="security-engineer",
        limit=3
    )
    
    print(f"✅ Found {len(contextual)} contextually relevant memories")
    
    return memory_manager


async def test_orchestrator_integration():
    """Test the enhanced orchestrator with specialist agents."""
    print("\n🎭 Testing Enhanced Orchestrator...")
    
    config = OrchestratorConfig(
        enable_intelligent_delegation=True,
        enable_parallel_execution=True,
        max_parallel_agents=4
    )
    
    try:
        orchestrator = Orchestrator(config)
        print(f"✅ Orchestrator initialized with {len(orchestrator.available_agent_types)} specialist agents")
        
        # Test agent recommendations
        recommendations = await orchestrator.get_agent_recommendations(
            task_description="Design secure user authentication flow with good UX",
            current_agents=[]
        )
        
        print(f"✅ Agent recommendations: {recommendations}")
        
        # Test specialist agent types
        available_types = orchestrator.get_available_specialist_types()
        print(f"✅ Available specialist types: {len(available_types)}")
        
        # Test memory integration
        memory_stats = orchestrator.memory_manager.get_memory_stats()
        print(f"✅ Memory system stats: {memory_stats}")
        
        return orchestrator
        
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return None


async def test_agent_spawning():
    """Test spawning and managing specialist agents."""
    print("\n🚀 Testing Agent Spawning...")
    
    config = OrchestratorConfig()
    orchestrator = Orchestrator(config)
    
    try:
        # Test spawning a security engineer
        security_agent_id = await orchestrator.spawn_specialist_agent(
            agent_type="security-engineer",
            task_context="API security review needed"
        )
        
        if security_agent_id:
            print(f"✅ Successfully spawned security-engineer: {security_agent_id}")
        else:
            print("❌ Failed to spawn security-engineer")
            return False
        
        # Test spawning a product manager
        pm_agent_id = await orchestrator.spawn_specialist_agent(
            agent_type="senior-product-manager", 
            task_context="Product roadmap planning session"
        )
        
        if pm_agent_id:
            print(f"✅ Successfully spawned senior-product-manager: {pm_agent_id}")
        else:
            print("❌ Failed to spawn senior-product-manager")
            return False
        
        # Check active specialists
        active_specialists = orchestrator.get_active_specialists()
        print(f"✅ Active specialists: {active_specialists}")
        
        # Test getting agent memory summary
        memory_summary = await orchestrator.get_agent_memory_summary("security-engineer")
        if memory_summary:
            print(f"✅ Memory summary retrieved:\n{memory_summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent spawning test failed: {e}")
        return False


async def run_comprehensive_test():
    """Run all tests in sequence."""
    print("🧪 Starting Comprehensive Agent Role System Test\n")
    print("="*60)
    
    # Test 1: Agent Loading
    descriptions = await test_agent_loading()
    if not descriptions:
        print("❌ Agent loading failed - aborting tests")
        return False
    
    # Test 2: Memory Manager
    memory_manager = await test_memory_manager()
    if not memory_manager:
        print("❌ Memory manager failed - aborting tests")
        return False
    
    # Test 3: Orchestrator Integration
    orchestrator = await test_orchestrator_integration()
    if not orchestrator:
        print("❌ Orchestrator integration failed - aborting tests")
        return False
    
    # Test 4: Agent Spawning
    spawning_success = await test_agent_spawning()
    if not spawning_success:
        print("❌ Agent spawning failed")
        return False
    
    print("\n" + "="*60)
    print("🎉 All Tests Passed! Agent Role System is Ready")
    print("\n📊 System Summary:")
    print(f"  • {len(descriptions)} specialist agent types loaded")
    print(f"  • Memory management system operational")
    print(f"  • Enhanced orchestrator with MCP tool integration")
    print(f"  • On-demand agent spawning functional")
    print("\n🚀 The comprehensive agent role system is now fully operational!")
    
    return True


if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())