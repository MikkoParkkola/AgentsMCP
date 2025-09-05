#!/usr/bin/env python3
"""
Test script for Step 3: Re-enable sequential thinking with debug logging to identify endless loop.
This will likely timeout, but we should see debug output showing where the loop occurs.
"""

import asyncio
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine

async def test_step3_debug_sequential_thinking():
    """Test Step 3: Sequential thinking re-enabled with debug logging."""
    print("🧪 Step 3: Testing Sequential Thinking with Debug Logging")
    print("=" * 80)
    print("⚠️  WARNING: This test is expected to timeout due to endless loop")
    print("📊 Goal: Capture debug logs showing where the loop occurs")
    print("=" * 80)
    
    try:
        # Create ChatEngine instance
        print("✓ Creating ChatEngine instance...")
        chat_engine = ChatEngine()
        
        # Set up status tracking
        status_messages = []
        def status_callback(status):
            status_messages.append(status)
            print(f"  📊 Status: {status}")
        
        chat_engine.set_callbacks(status_callback=status_callback)
        
        # Test product assessment query
        test_query = "please make a product assessment with your team about the product in this directory"
        print(f"\n🔍 Testing query: '{test_query}'")
        print("=" * 60)
        
        # Process the input with shorter timeout to get debug logs faster
        print("⚡ Processing input with 15-second timeout to capture debug logs...")
        
        try:
            response = await asyncio.wait_for(
                chat_engine.process_input(test_query),
                timeout=15.0  # Shorter timeout to get logs faster
            )
            
            print(f"\n📊 Step 3 Results:")
            print("-" * 40)
            
            # Check status messages
            print("Status messages captured:")
            for i, status in enumerate(status_messages, 1):
                print(f"  {i}. {status}")
            
            print(f"\n🎉 UNEXPECTED SUCCESS: Sequential thinking completed without endless loop!")
            print(f"💬 Messages generated: {len(chat_engine.state.messages)}")
            
            if len(chat_engine.state.messages) >= 2:  # User + AI response
                last_message = chat_engine.state.messages[-1]
                print(f"📝 Response preview: {last_message.content[:100]}...")
            
            return True
                
        except asyncio.TimeoutError:
            print(f"\n📊 Step 3 Debug Results (EXPECTED TIMEOUT):")
            print("-" * 50)
            
            # Check status messages to see how far we got
            print("Status messages captured before timeout:")
            for i, status in enumerate(status_messages, 1):
                print(f"  {i}. {status}")
            
            print(f"\n🎯 Debug Analysis:")
            
            # Check if we got past TaskTracker startup
            if any("DEBUG: TaskTracker.start_task() completed" in status for status in status_messages):
                print("✅ TaskTracker.start_task() completed successfully")
            elif any("DEBUG: About to call TaskTracker.start_task()" in status for status in status_messages):
                print("❌ Endless loop occurs IN TaskTracker.start_task() method")
            else:
                print("❌ Endless loop occurs BEFORE TaskTracker.start_task() is called")
            
            # Check if we got to sequential planning
            if any("Phase 1: Sequential thinking and planning" in status for status in status_messages):
                print("✅ Got to sequential planning phase")
                if any("DEBUG TaskTracker: sequential_planner.create_plan() completed" in status for status in status_messages):
                    print("✅ Sequential planning completed - loop is elsewhere")
                elif any("DEBUG TaskTracker: About to call sequential_planner.create_plan()" in status for status in status_messages):
                    print("❌ ENDLESS LOOP IS IN sequential_planner.create_plan() METHOD")
                else:
                    print("❌ Loop is in TaskTracker before create_plan() call")
            else:
                print("❌ Never reached sequential planning phase")
            
            print(f"\n🔍 CONCLUSION: The endless loop location has been identified!")
            return False
        
    except Exception as e:
        print(f"❌ Step 3 test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if 'chat_engine' in locals():
            await chat_engine.cleanup()

if __name__ == "__main__":
    success = asyncio.run(test_step3_debug_sequential_thinking())
    print(f"\n{'SUCCESS' if success else 'TIMEOUT (EXPECTED)'}: Step 3 debug test completed")
    sys.exit(0 if success else 1)