#!/usr/bin/env python3
"""
Test script for Step 2: Enable preprocessing, surface output to user, keep sequential thinking disabled.
Tests that preprocessing works correctly without endless loops.
"""

import asyncio
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine

async def test_step2_preprocessing():
    """Test Step 2: Preprocessing enabled, sequential thinking still disabled."""
    print("🧪 Step 2: Testing Preprocessing Enabled (sequential thinking still disabled)")
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
        
        # Process the input with timeout to prevent endless loops
        print("⚡ Processing input with 30-second timeout...")
        
        try:
            response = await asyncio.wait_for(
                chat_engine.process_input(test_query),
                timeout=30.0  # 30-second timeout
            )
            
            print(f"\n📊 Step 2 Results:")
            print("-" * 40)
            
            # Check status messages
            print("Status messages captured:")
            for i, status in enumerate(status_messages, 1):
                print(f"  {i}. {status}")
            
            # Check for expected preprocessing steps 
            preprocessing_indicators = [
                "preprocessing", "Preprocessing", "enhanced",
                "sequential thinking disabled for testing"
            ]
            
            orchestration_indicators = [
                "Delegating to specialist agents",
                "Executing enhanced response"
            ]
            
            preprocessing_found = False
            orchestration_found = False
            
            for status in status_messages:
                for indicator in preprocessing_indicators:
                    if indicator in status:
                        preprocessing_found = True
                        break
                for indicator in orchestration_indicators:
                    if indicator in status:
                        orchestration_found = True
                        break
            
            print(f"\n🎯 Test Results:")
            print(f"{'✅' if preprocessing_found else '❌'} Preprocessing detected: {preprocessing_found}")
            print(f"{'✅' if orchestration_found else '❌'} Orchestration detected: {orchestration_found}")
            print(f"✅ No endless loop (completed within 30 seconds)")
            
            # Check that sequential thinking is still disabled
            sequential_thinking_disabled = any("sequential thinking disabled for testing" in status for status in status_messages)
            print(f"{'✅' if sequential_thinking_disabled else '❌'} Sequential thinking still disabled: {sequential_thinking_disabled}")
            
            # Check message count
            message_count = len(chat_engine.state.messages)
            print(f"💬 Messages generated: {message_count}")
            
            if message_count >= 2:  # User + AI response
                last_message = chat_engine.state.messages[-1]
                print(f"📝 Response preview: {last_message.content[:100]}...")
            
            # Final verdict for Step 2
            if preprocessing_found and orchestration_found and sequential_thinking_disabled and message_count >= 2:
                print("\n🎉 STEP 2 SUCCESS: Preprocessing works without endless loops!")
                print("   ✅ Preprocessing enabled and working")
                print("   ✅ Sequential thinking still disabled")  
                print("   ✅ Agent orchestration working")
                print("   ✅ No endless loops")
                return True
            else:
                print("\n❌ STEP 2 FAILURE: Issues found with preprocessing")
                return False
                
        except asyncio.TimeoutError:
            print("\n❌ STEP 2 FAILURE: Endless loop detected (timed out after 30 seconds)")
            print("   The system is still stuck in an infinite loop even with preprocessing enabled.")
            return False
        
    except Exception as e:
        print(f"❌ Step 2 test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if 'chat_engine' in locals():
            await chat_engine.cleanup()

if __name__ == "__main__":
    success = asyncio.run(test_step2_preprocessing())
    sys.exit(0 if success else 1)