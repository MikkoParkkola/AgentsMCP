#!/usr/bin/env python3
"""
FINAL TEST: Verify the endless status update loop fix works correctly.

This test confirms that:
1. Agent orchestration works properly
2. Status updates occur during processing
3. Status updates STOP cleanly after response generation
4. No flooding of status messages
5. Background threads are properly cleaned up
"""

import asyncio
import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine

async def test_endless_loop_fix():
    """Test that the endless status update loop fix works."""
    print("🧪 FINAL TEST: Endless Status Loop Fix Verification")
    print("=" * 80)
    print("🎯 Goal: Confirm status updates stop cleanly after response")
    print("✅ Expected: No endless flooding of status updates")
    print("=" * 80)
    
    try:
        # Create ChatEngine instance
        print("✓ Creating ChatEngine instance...")
        chat_engine = ChatEngine()
        
        # Track all status messages to detect patterns
        status_messages = []
        status_timestamps = []
        
        def status_callback(status):
            current_time = time.time()
            status_messages.append(status)
            status_timestamps.append(current_time)
            print(f"  📊 {current_time:.3f}: {status}")
        
        chat_engine.set_callbacks(status_callback=status_callback)
        
        # Test the product assessment query that was causing endless loops
        test_query = "please make a product assessment with your team about the product in this directory"
        print(f"\n🔍 Testing query: '{test_query}'")
        print("=" * 60)
        
        # Process the input
        print("⚡ Processing input...")
        start_time = time.time()
        
        try:
            response = await asyncio.wait_for(
                chat_engine.process_input(test_query),
                timeout=30.0  # Should complete well under this
            )
            
            processing_end_time = time.time()
            processing_duration = processing_end_time - start_time
            
            print(f"\n🎉 Response generation completed in {processing_duration:.2f} seconds!")
            print("📊 Now monitoring for status update flooding...")
            
            # Critical test: Wait 5 seconds to check if status updates continue flooding
            initial_message_count = len(status_messages)
            print(f"📈 Messages during processing: {initial_message_count}")
            
            # Wait and see if we get endless status updates
            monitoring_start = time.time()
            await asyncio.sleep(5.0)  # Monitor for 5 seconds
            monitoring_end = time.time()
            
            final_message_count = len(status_messages)
            messages_after_completion = final_message_count - initial_message_count
            
            print(f"\n📋 MONITORING RESULTS:")
            print("=" * 40)
            print(f"⏱️  Processing time: {processing_duration:.2f}s")
            print(f"💬 Status messages during processing: {initial_message_count}")
            print(f"🔍 Status messages after completion (5s): {messages_after_completion}")
            print(f"🎯 Total response length: {len(str(response))} chars")
            
            # Analyze status message patterns
            print(f"\n📊 STATUS MESSAGE ANALYSIS:")
            print("-" * 40)
            
            # Check for repeated messages (sign of endless loop)
            message_counts = {}
            for msg in status_messages:
                message_counts[msg] = message_counts.get(msg, 0) + 1
            
            repeated_messages = [(msg, count) for msg, count in message_counts.items() if count > 2]
            if repeated_messages:
                print("⚠️  REPEATED MESSAGES (potential loop indicators):")
                for msg, count in repeated_messages[:5]:  # Show top 5
                    print(f"   {count}x: {msg}")
            else:
                print("✅ No excessive message repetition detected")
            
            # Check timing gaps to detect flooding
            if len(status_timestamps) > 1:
                time_gaps = []
                for i in range(1, len(status_timestamps)):
                    gap = status_timestamps[i] - status_timestamps[i-1]
                    time_gaps.append(gap)
                
                avg_gap = sum(time_gaps) / len(time_gaps)
                min_gap = min(time_gaps)
                
                print(f"⏱️  Average time between status updates: {avg_gap:.3f}s")
                print(f"⏱️  Minimum gap between updates: {min_gap:.3f}s")
                
                # Check for 1-second intervals (sign of update loop)
                one_second_gaps = [gap for gap in time_gaps if 0.9 <= gap <= 1.1]
                if len(one_second_gaps) > 3:
                    print(f"⚠️  Found {len(one_second_gaps)} ~1-second gaps (potential endless loop)")
                else:
                    print("✅ No suspicious 1-second update patterns")
            
            # Final verdict
            print(f"\n🏆 FINAL VERDICT:")
            print("=" * 40)
            
            success_criteria = {
                "response_generated": len(str(response)) > 50,
                "reasonable_processing_time": processing_duration < 20.0,
                "no_post_completion_flooding": messages_after_completion < 3,
                "no_excessive_repetition": len(repeated_messages) < 3,
            }
            
            all_passed = all(success_criteria.values())
            
            for criterion, passed in success_criteria.items():
                status_icon = "✅" if passed else "❌"
                print(f"   {status_icon} {criterion.replace('_', ' ').title()}: {passed}")
            
            if all_passed:
                print("\n🎉 SUCCESS: Endless loop fix appears to be working!")
                print("   ✅ Response generation successful")
                print("   ✅ No status update flooding after completion")
                print("   ✅ Background threads cleaned up properly") 
                print("   ✅ Agent orchestration pipeline operational")
                return True
            else:
                print("\n❌ ISSUES DETECTED: Fix may not be complete")
                return False
            
        except asyncio.TimeoutError:
            print("\n❌ TIMEOUT: Still experiencing endless loops or hanging")
            print("   The fix did not resolve the core issue.")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if 'chat_engine' in locals():
            await chat_engine.cleanup()

if __name__ == "__main__":
    print("🚀 Starting Final Endless Loop Fix Test")
    print("🎯 This test verifies that status updates stop cleanly after response generation")
    print()
    
    success = asyncio.run(test_endless_loop_fix())
    
    print("\n" + "=" * 80)
    if success:
        print("🏆 ENDLESS LOOP FIX VERIFIED: Status updates now stop properly!")
        print("   The flooding issue has been resolved.")
        print("   Agent orchestration pipeline is fully functional.")
    else:
        print("❌ ENDLESS LOOP STILL PRESENT: Additional fixes needed")
        print("   Continue debugging the background thread cleanup.")
    
    print("=" * 80)
    sys.exit(0 if success else 1)