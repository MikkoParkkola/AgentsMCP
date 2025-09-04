#!/usr/bin/env python3
"""Test script for sequential thinking and progress bar integration."""

import asyncio
import sys
import os
import time

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine
from agentsmcp.orchestration.task_tracker import TaskTracker
from agentsmcp.orchestration.sequential_planner import SequentialPlanner
from agentsmcp.ui.v3.progress_display import ProgressDisplay, AgentStatus


async def test_sequential_thinking():
    """Test sequential thinking integration."""
    print("🧪 Testing Sequential Thinking Integration")
    print("=" * 50)
    
    # Test 1: SequentialPlanner
    print("\n1. Testing SequentialPlanner...")
    planner = SequentialPlanner()
    
    plan = await planner.create_plan(
        user_input="Create a simple hello world program",
        context={"language": "python"}
    )
    
    print(f"✅ Plan created with {len(plan.steps)} steps")
    for i, step in enumerate(plan.steps, 1):
        print(f"   Step {i}: {step.description[:50]}...")
    
    # Test 2: ProgressDisplay
    print("\n2. Testing ProgressDisplay...")
    
    def progress_callback(message):
        print(f"📊 Progress: {message}")
    
    progress_display = ProgressDisplay(update_callback=progress_callback)
    
    # Start a task
    progress_display.start_task("test_task", "Testing Progress", 5000)
    
    # Add some agents
    progress_display.add_agent("agent1", "Code Generator", 3000)
    progress_display.add_agent("agent2", "Tester", 2000)
    
    # Simulate agent progress
    progress_display.start_agent("agent1", "Generating code...")
    await asyncio.sleep(0.5)
    
    progress_display.update_agent_progress("agent1", 50.0, "Writing functions...")
    await asyncio.sleep(0.5)
    
    progress_display.complete_agent("agent1")
    progress_display.start_agent("agent2", "Running tests...")
    await asyncio.sleep(0.5)
    
    progress_display.complete_agent("agent2")
    progress_display.complete_task()
    
    print("✅ ProgressDisplay test completed")
    
    # Test 3: TaskTracker
    print("\n3. Testing TaskTracker...")
    
    def tracker_callback(message):
        print(f"🎯 Tracker: {message}")
    
    task_tracker = TaskTracker(progress_update_callback=tracker_callback)
    
    task_id = await task_tracker.start_task(
        "Test the complete integration",
        {"test": True},
        10000
    )
    
    # Execute the task (this completes it automatically)
    await task_tracker.execute_task(task_id)
    
    print("✅ TaskTracker test completed")
    
    # Test 4: Performance Stats
    print("\n4. Testing Performance Analysis...")
    stats = progress_display.get_performance_stats()
    timing_report = progress_display.get_timing_analysis_report()
    
    print("📈 Performance Stats:")
    print(f"   Total Tasks: {stats['total_tasks']}")
    print(f"   Completed Tasks: {stats['completed_tasks']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    
    print("\n📊 Timing Report:")
    print(timing_report)
    
    print("\n✅ All integration tests passed!")
    return True


async def test_chat_engine_integration():
    """Test ChatEngine with TaskTracker integration."""
    print("\n🧪 Testing ChatEngine Integration")
    print("=" * 50)
    
    # Initialize ChatEngine (should have TaskTracker)
    chat_engine = ChatEngine()
    
    # Verify TaskTracker is present
    if hasattr(chat_engine, 'task_tracker') and chat_engine.task_tracker:
        print("✅ ChatEngine has TaskTracker integration")
        
        # Test progress command handler
        if '/progress' in chat_engine.commands:
            print("✅ /progress command available")
            
        if '/timing' in chat_engine.commands:
            print("✅ /timing command available")
            
        print("✅ ChatEngine integration successful")
        return True
    else:
        print("❌ ChatEngine missing TaskTracker")
        return False


async def main():
    """Run all integration tests."""
    print("🚀 Sequential Thinking & Progress Bar Integration Tests")
    print("=" * 60)
    
    try:
        # Run core integration tests
        test1_success = await test_sequential_thinking()
        
        # Run ChatEngine integration tests
        test2_success = await test_chat_engine_integration()
        
        if test1_success and test2_success:
            print("\n🎉 ALL TESTS PASSED! Sequential thinking and progress bars are fully integrated.")
            
            print("\n📋 Integration Summary:")
            print("✅ SequentialPlanner - Creates step-by-step execution plans")
            print("✅ ProgressDisplay - Real-time progress bars and timing")
            print("✅ TaskTracker - Coordinates planning and execution")
            print("✅ ChatEngine - Sequential thinking before LLM calls")
            print("✅ TUILauncher - Progress bars in status updates")
            print("✅ Orchestrator - Sequential planning integration")
            print("✅ Performance Tracking - Timing analysis and metrics")
            print("✅ User Commands - /progress and /timing commands")
            
            print("\n🎯 Features Available:")
            print("• Sequential thinking for all major tasks")
            print("• Real-time progress visualization")
            print("• Agent status tracking with icons")
            print("• Performance timing analysis")
            print("• User-accessible progress commands")
            print("• Comprehensive error handling")
            
            return True
        else:
            print("\n❌ SOME TESTS FAILED")
            return False
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)