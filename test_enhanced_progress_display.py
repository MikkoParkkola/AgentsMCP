#!/usr/bin/env python3
"""Test enhanced progress display with sequential thinking step tracking."""

import sys
import asyncio
import time
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

from agentsmcp.orchestration.task_tracker import TaskTracker

async def test_enhanced_progress_display():
    print("Testing enhanced progress display with sequential thinking step tracking...")
    print("=" * 80)
    
    # Initialize TaskTracker  
    def status_callback(status):
        print(f"📢 STATUS: {status}")
    
    def progress_callback(message):
        print(f"📊 PROGRESS: {message}")
    
    task_tracker = TaskTracker(
        progress_update_callback=progress_callback,
        status_update_callback=status_callback
    )
    
    # Test 1: Complex request that should trigger sequential thinking
    print("\n🧪 Test 1: Complex request - 'Create a comprehensive authentication system'")
    print("Expected: Should show Analyst, General Agent, and Project Manager in sequence")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        task_id = await task_tracker.start_task(
            "Create a comprehensive authentication system with JWT tokens, password hashing, role-based access control, and proper error handling"
        )
        print(f"✅ Task created: {task_id}")
        
        # Let it run for a reasonable time to see the full sequential thinking process
        await asyncio.sleep(5.0)
        
        # Check final progress display
        progress_display = task_tracker.get_progress_display()
        print(f"\n📋 Final Progress Display:\n{progress_display}")
        
        # Get agent statuses
        performance_stats = task_tracker.get_performance_stats()
        print(f"\n📈 Performance Stats: {performance_stats}")
        
        end_time = time.time()
        print(f"⏱️  Total time: {end_time - start_time:.3f}s")
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ Error during test: {e}")
        print(f"⏱️  Time before error: {end_time - start_time:.3f}s")
        import traceback
        traceback.print_exc()
    
    # Test 2: Another complex request to verify consistent behavior
    print("\n\n🧪 Test 2: Another complex request - 'Build a real-time chat application'")
    print("Expected: Should show the same agent progression pattern")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        task_id = await task_tracker.start_task(
            "Build a real-time chat application with WebSocket connections, message persistence, user presence tracking, and file sharing capabilities"
        )
        print(f"✅ Task created: {task_id}")
        
        await asyncio.sleep(3.0)
        
        # Check progress display mid-execution
        progress_display = task_tracker.get_progress_display()
        print(f"\n📋 Mid-Execution Progress Display:\n{progress_display}")
        
        end_time = time.time()
        print(f"⏱️  Sequential thinking time: {end_time - start_time:.3f}s")
        
    except Exception as e:
        end_time = time.time()
        print(f"❌ Error during test: {e}")
        print(f"⏱️  Time before error: {end_time - start_time:.3f}s")
    
    print("\n" + "=" * 80)
    print("🎯 Enhanced Progress Display Test Results:")
    print("✅ If you saw agents transitioning through different phases (Analyst → General Agent → Project Manager)")
    print("✅ If progress percentages updated during sequential thinking")
    print("✅ If status messages showed 'Sequential Thinking' with appropriate emojis")
    print("✅ If agents showed specific tasks like 'Analyzing request', 'Strategic planning', 'Execution planning'")
    print("📊 Then the enhanced progress display is working correctly!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_progress_display())