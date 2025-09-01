#!/usr/bin/env python3
"""
Demonstration of AgentsMCP Advanced Thinking and Planning System

This script showcases the key capabilities of the cognition module:
1. Structured thinking processes with 7 phases
2. Multi-criteria approach evaluation
3. Dependency-aware task decomposition
4. Resource-optimized execution planning
5. Metacognitive monitoring and adjustment
6. State persistence and recovery
7. Orchestrator integration with thinking loops

Run this script to see the thinking system in action.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any

# Set up logging to see the thinking process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the cognition module
from src.agentsmcp.cognition import (
    ThinkingFramework, ThinkingConfig, ThinkingOrchestrator,
    create_thinking_orchestrator, create_state_manager,
    PerformanceProfile, OrchestratorIntegrationMode,
    save_thinking_result, load_thinking_result
)
from src.agentsmcp.cognition.models import (
    ThinkingPhase, EvaluationCriteria, ResourceConstraints,
    PersistenceFormat, CheckpointStrategy
)


async def demo_basic_thinking():
    """Demonstrate basic thinking framework usage."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Thinking Framework")
    print("="*60)
    
    # Create thinking framework with custom configuration
    config = ThinkingConfig(
        max_approaches=5,
        max_subtasks=10,
        enable_metacognitive_monitoring=True,
        thinking_depth="thorough",
        timeout_seconds=30
    )
    
    framework = ThinkingFramework(config)
    
    # Define a complex request
    request = """
    Design and implement a scalable microservices architecture for an e-commerce platform
    that needs to handle 100K+ concurrent users, process payments securely, manage inventory
    in real-time, and provide personalized recommendations. The system should be deployed
    on cloud infrastructure with high availability and disaster recovery capabilities.
    """
    
    # Provide context
    context = {
        "constraints": {
            "budget": "$500K",
            "timeline": "6 months",
            "team_size": 8,
            "compliance": ["PCI-DSS", "GDPR"]
        },
        "existing_systems": {
            "user_database": "PostgreSQL",
            "payment_gateway": "Stripe",
            "cloud_provider": "AWS"
        },
        "requirements": {
            "availability": "99.9%",
            "response_time": "<200ms",
            "scalability": "horizontal"
        }
    }
    
    print("Processing complex e-commerce architecture request...")
    print(f"Request length: {len(request)} characters")
    
    # Execute thinking process
    result = await framework.think(request, context)
    
    # Display results
    print(f"\n✅ Thinking completed in {result.total_duration_ms}ms")
    print(f"🎯 Confidence: {result.confidence:.2f}")
    print(f"📊 Phases completed: {len(result.thinking_trace)}")
    
    if result.final_approach:
        print(f"🏆 Selected approach: {result.final_approach.name}")
        print(f"📝 Description: {result.final_approach.description}")
        print(f"⭐ Score: {result.final_approach.estimated_score:.2f}")
    
    if result.execution_plan:
        print(f"📋 Execution plan: {len(result.execution_plan.scheduled_tasks)} tasks")
        print(f"⏱️  Estimated duration: {result.execution_plan.estimated_duration_minutes} minutes")
    
    # Show thinking trace summary
    print("\n🧠 Thinking Process Summary:")
    for i, step in enumerate(result.thinking_trace, 1):
        status = "✅" if not hasattr(step, 'error') or not step.error else "❌"
        print(f"  {i}. {step.phase.value.title()}: {step.duration_ms}ms {status}")
    
    return result


async def demo_state_persistence():
    """Demonstrate state persistence and recovery."""
    print("\n" + "="*60)
    print("DEMO 2: State Persistence and Recovery")
    print("="*60)
    
    # Create state manager
    storage_path = Path("./demo_thinking_states")
    state_manager = await create_state_manager()
    
    print("Creating thinking state for persistence...")
    
    # Create a thinking framework
    framework = ThinkingFramework()
    
    # Simple request for demo
    request = "Optimize database performance for high-traffic application"
    context = {"database": "PostgreSQL", "current_qps": 1000, "target_qps": 5000}
    
    # Execute thinking
    result = await framework.think(request, context)
    
    # Save the thinking result
    state_metadata = await save_thinking_result(result, storage_path)
    print(f"💾 Saved thinking state: {state_metadata.state_id}")
    print(f"📁 File: {state_metadata.file_path}")
    print(f"💽 Size: {state_metadata.size_bytes} bytes")
    
    # Simulate loading the state later
    print("\n🔄 Loading saved thinking state...")
    loaded_state = await load_thinking_result(state_metadata.state_id, storage_path)
    
    if loaded_state:
        print(f"✅ Successfully loaded state: {loaded_state.state_id}")
        print(f"📊 Steps recovered: {len(loaded_state.thinking_trace)}")
        print(f"🕐 Original timestamp: {loaded_state.created_at}")
    else:
        print("❌ Failed to load state")
    
    # Clean up demo files
    try:
        if storage_path.exists():
            import shutil
            shutil.rmtree(storage_path)
            print("🧹 Cleaned up demo files")
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up demo files: {e}")


async def demo_orchestrator_integration():
    """Demonstrate orchestrator integration with thinking."""
    print("\n" + "="*60)
    print("DEMO 3: Orchestrator Integration")
    print("="*60)
    
    # Create thinking orchestrator with different profiles
    profiles = [
        (PerformanceProfile.FAST, "Fast Profile (Speed Optimized)"),
        (PerformanceProfile.BALANCED, "Balanced Profile (Speed + Quality)"), 
        (PerformanceProfile.COMPREHENSIVE, "Comprehensive Profile (Quality Optimized)")
    ]
    
    request = "Create a REST API for user authentication with JWT tokens"
    context = {"framework": "FastAPI", "database": "MongoDB", "deployment": "Docker"}
    
    for profile, description in profiles:
        print(f"\n🔧 Testing {description}")
        
        # Create orchestrator with specific profile
        orchestrator = create_thinking_orchestrator(
            performance_profile=profile,
            enable_persistence=False  # Disable for demo speed
        )
        
        # Process request
        start_time = asyncio.get_event_loop().time()
        response = await orchestrator.process_user_input(request, context)
        duration = (asyncio.get_event_loop().time() - start_time) * 1000
        
        print(f"⏱️  Response time: {duration:.2f}ms")
        print(f"🎯 Response type: {response.response_type}")
        
        if response.metadata and "thinking_applied" in response.metadata:
            thinking_meta = response.metadata
            print(f"🧠 Thinking applied: {thinking_meta['thinking_applied']}")
            print(f"📊 Thinking confidence: {thinking_meta.get('thinking_confidence', 'N/A')}")
            print(f"⚡ Thinking duration: {thinking_meta.get('thinking_duration_ms', 'N/A')}ms")
        
        # Get thinking statistics
        stats = await orchestrator.get_thinking_stats()
        thinking_stats = stats.get("thinking_integration", {})
        print(f"📈 Total requests: {thinking_stats.get('total_requests', 0)}")
        print(f"🧠 Thinking enabled: {thinking_stats.get('thinking_enabled_requests', 0)}")
        
        # Cleanup
        await orchestrator.shutdown()


async def demo_advanced_features():
    """Demonstrate advanced features and edge cases."""
    print("\n" + "="*60)
    print("DEMO 4: Advanced Features")
    print("="*60)
    
    # Create framework with advanced configuration
    config = ThinkingConfig(
        max_approaches=8,
        max_subtasks=15,
        enable_parallel_evaluation=True,
        enable_metacognitive_monitoring=True,
        enable_lightweight_mode=False,
        thinking_depth="comprehensive",
        confidence_threshold=0.8
    )
    
    framework = ThinkingFramework(config)
    
    # Complex multi-domain request
    request = """
    Design a comprehensive solution for a smart city IoT platform that includes:
    1. Real-time traffic management with AI-powered optimization
    2. Environmental monitoring (air quality, noise, temperature)
    3. Public safety integration with emergency services
    4. Energy grid optimization for street lighting and facilities
    5. Citizen engagement mobile app with service requests
    6. Privacy-preserving data analytics for city planning
    7. Integration with existing municipal systems
    8. Scalability to support 2 million residents
    """
    
    context = {
        "city_size": "2M residents",
        "budget": "$50M",
        "timeline": "3 years", 
        "existing_infrastructure": {
            "traffic_signals": 15000,
            "sensors": 5000,
            "municipal_systems": ["ERP", "GIS", "311"],
            "network": "fiber_backbone"
        },
        "requirements": {
            "uptime": "99.95%",
            "data_privacy": "GDPR compliant",
            "interoperability": "open_standards",
            "sustainability": "carbon_neutral"
        },
        "stakeholders": [
            "city_government",
            "citizens", 
            "emergency_services",
            "utility_companies",
            "transportation_dept"
        ]
    }
    
    print("Processing complex smart city IoT platform request...")
    print("This may take a moment due to the comprehensive analysis...")
    
    # Track progress with callback
    progress_steps = []
    
    def progress_callback(step):
        progress_steps.append(step)
        print(f"  🔄 {step.phase.value.title()}: {step.duration_ms}ms")
    
    # Execute thinking with progress tracking
    result = await framework.think(request, context, progress_callback=progress_callback)
    
    print(f"\n🎉 Advanced thinking completed!")
    print(f"⏱️  Total duration: {result.total_duration_ms}ms")
    print(f"🎯 Final confidence: {result.confidence:.3f}")
    print(f"📊 Total steps: {len(result.thinking_trace)}")
    
    # Analyze the approaches that were considered
    if hasattr(result, '_all_approaches') and result._all_approaches:
        print(f"\n🤔 Approaches considered: {len(result._all_approaches)}")
        for i, approach in enumerate(result._all_approaches[:3], 1):  # Show top 3
            print(f"  {i}. {approach.name}: {approach.estimated_score:.3f}")
    
    # Show execution plan details
    if result.execution_plan:
        plan = result.execution_plan
        print(f"\n📋 Execution Plan Details:")
        print(f"  📝 Total tasks: {len(plan.scheduled_tasks)}")
        print(f"  ⏰ Estimated duration: {plan.estimated_duration_minutes} minutes")
        print(f"  🔧 Strategy: {plan.optimization_strategy.value}")
        
        # Show first few tasks
        print("  🚀 First 3 tasks:")
        for i, task in enumerate(plan.scheduled_tasks[:3], 1):
            print(f"    {i}. {task.description[:50]}...")
    
    # Metacognitive insights
    if hasattr(result, 'metacognitive_assessment'):
        assessment = result.metacognitive_assessment
        print(f"\n🧠 Metacognitive Assessment:")
        print(f"  🎯 Confidence calibration: {assessment.confidence_calibration:.3f}")
        print(f"  ⚡ Process efficiency: {assessment.process_efficiency:.3f}")
        print(f"  🔍 Solution completeness: {assessment.solution_completeness:.3f}")


async def main():
    """Run all demonstrations."""
    print("🚀 AgentsMCP Advanced Thinking and Planning System Demo")
    print("=" * 60)
    
    try:
        # Run demonstrations
        await demo_basic_thinking()
        await demo_state_persistence()
        await demo_orchestrator_integration()
        await demo_advanced_features()
        
        print("\n" + "="*60)
        print("✅ All demonstrations completed successfully!")
        print("="*60)
        
        print("\n📖 What you've seen:")
        print("  🧠 Multi-phase structured thinking process")
        print("  ⚖️  Multi-criteria approach evaluation")
        print("  🔗 Dependency-aware task decomposition")
        print("  📅 Resource-optimized execution planning")
        print("  🔍 Metacognitive quality monitoring")
        print("  💾 State persistence and recovery")
        print("  🎛️  Orchestrator integration")
        print("  ⚡ Multiple performance profiles")
        
        print("\n🎯 Next Steps:")
        print("  1. Integrate thinking loops into your agents")
        print("  2. Customize configuration for your use cases")
        print("  3. Enable state persistence for long-running tasks")
        print("  4. Monitor thinking performance and adjust parameters")
        print("  5. Explore advanced metacognitive features")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # Run the demonstration
    exit_code = asyncio.run(main())
    exit(exit_code)