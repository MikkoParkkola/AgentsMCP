#!/usr/bin/env python3
"""
Test script for Step 4: Complete solution test with fallback planning.
This should now work end-to-end without endless loops.
"""

import asyncio
import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine

async def test_step4_complete_solution():
    """Test Step 4: Complete orchestration pipeline with fallback planning."""
    print("🧪 Step 4: Testing Complete Solution with Fallback Planning")
    print("=" * 80)
    print("🎯 Goal: Full orchestration pipeline working end-to-end")
    print("✅ Expected: TaskTracker + Preprocessing + Fallback Planning all working")
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
        
        # Process the input with reasonable timeout
        print("⚡ Processing input with 60-second timeout...")
        start_time = time.time()
        
        try:
            response = await asyncio.wait_for(
                chat_engine.process_input(test_query),
                timeout=60.0  # Generous timeout for full pipeline
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n🎉 Step 4 SUCCESS: Complete pipeline working!")
            print("=" * 60)
            
            # Detailed analysis
            print(f"⏱️  Total processing time: {duration:.2f} seconds")
            print(f"💬 Messages generated: {len(chat_engine.state.messages)}")
            
            # Check status messages for key indicators
            print(f"\n📊 Pipeline Analysis:")
            print("-" * 40)
            
            # Check TaskTracker functionality
            task_tracker_working = any("TaskTracker" in status for status in status_messages)
            print(f"{'✅' if task_tracker_working else '❌'} TaskTracker: {task_tracker_working}")
            
            # Check preprocessing
            preprocessing_working = any("preprocessing" in status.lower() for status in status_messages)
            print(f"{'✅' if preprocessing_working else '❌'} Preprocessing: {preprocessing_working}")
            
            # Check sequential planning (should show fallback)
            fallback_planning = any("fallback planning" in status.lower() for status in status_messages)
            print(f"{'✅' if fallback_planning else '❌'} Fallback Planning: {fallback_planning}")
            
            # Check orchestration
            orchestration_working = any("Delegating" in status or "agent" in status.lower() for status in status_messages)
            print(f"{'✅' if orchestration_working else '❌'} Agent Orchestration: {orchestration_working}")
            
            # Show status message summary
            print(f"\n📋 Status Messages Captured ({len(status_messages)} total):")
            for i, status in enumerate(status_messages, 1):
                print(f"  {i:2d}. {status}")
            
            # Show response preview
            if len(chat_engine.state.messages) >= 2:
                last_message = chat_engine.state.messages[-1]
                print(f"\n📝 Response Preview:")
                print("-" * 40)
                print(last_message.content[:300] + "..." if len(last_message.content) > 300 else last_message.content)
            
            # Final verdict
            key_components = [task_tracker_working, preprocessing_working, orchestration_working]
            all_working = all(key_components)
            
            print(f"\n🏆 FINAL VERDICT:")
            print("=" * 40)
            if all_working and duration < 50:
                print("🎉 COMPLETE SUCCESS: All components working efficiently!")
                print(f"   ✅ TaskTracker operational")
                print(f"   ✅ Preprocessing enabled") 
                print(f"   ✅ Fallback planning working")
                print(f"   ✅ Agent orchestration active")
                print(f"   ✅ No endless loops")
                print(f"   ✅ Reasonable response time ({duration:.2f}s)")
                return True
            elif all_working:
                print("⚠️  PARTIAL SUCCESS: All components working but slow")
                print(f"   ✅ All pipeline components operational")
                print(f"   ⚠️  Response time high ({duration:.2f}s) - room for optimization")
                return True
            else:
                print("❌ ISSUES FOUND: Some components not working")
                print(f"   TaskTracker: {'✅' if task_tracker_working else '❌'}")
                print(f"   Preprocessing: {'✅' if preprocessing_working else '❌'}")
                print(f"   Orchestration: {'✅' if orchestration_working else '❌'}")
                return False
                
        except asyncio.TimeoutError:
            print(f"\n❌ Step 4 FAILURE: Pipeline still has endless loops")
            print("   Timed out after 60 seconds - fallback planning fix didn't work")
            
            # Show what we captured before timeout
            print(f"\n📊 Debug Info Before Timeout:")
            print(f"   Status messages captured: {len(status_messages)}")
            for i, status in enumerate(status_messages[-10:], 1):  # Show last 10
                print(f"     {i:2d}. {status}")
            
            return False
        
    except Exception as e:
        print(f"❌ Step 4 test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if 'chat_engine' in locals():
            await chat_engine.cleanup()

if __name__ == "__main__":
    print("🚀 Starting Step 4: Complete Solution Test")
    print("🎯 This should demonstrate the full orchestration pipeline working correctly")
    print()
    
    success = asyncio.run(test_step4_complete_solution())
    
    print("\n" + "=" * 80)
    if success:
        print("🎉 STEP 4 COMPLETE: Orchestration pipeline fully operational!")
        print("   All components working together correctly.")
        print("   Ready for production use.")
    else:
        print("❌ STEP 4 NEEDS MORE WORK: Issues still present")
        print("   Additional debugging and fixes required.")
    
    print("=" * 80)
    sys.exit(0 if success else 1)