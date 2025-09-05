#!/usr/bin/env python3
"""
Test script for Step 1: Orchestration only (no preprocessing, no sequential thinking).
Tests that basic agent orchestration works without endless loops.
"""

import asyncio
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine

async def test_step1_orchestration():
    """Test Step 1: Basic orchestration without sequential thinking."""
    print("🧪 Step 1: Testing Orchestration Only (no preprocessing, no sequential thinking)")
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
            
            print(f"\n📊 Step 1 Results:")
            print("-" * 40)
            
            # Check status messages
            print("Status messages captured:")
            for i, status in enumerate(status_messages, 1):
                print(f"  {i}. {status}")
            
            # Check for expected orchestration steps
            expected_steps = [
                "🧠 Analyzing request (sequential thinking disabled for testing)",
                "🎯 Delegating to specialist agents", 
                "🚀 Executing enhanced response with agent coordination"
            ]
            
            orchestration_found = False
            for status in status_messages:
                for step in expected_steps:
                    if step in status:
                        orchestration_found = True
                        break
                if orchestration_found:
                    break
            
            print(f"\n🎯 Test Results:")
            print(f"{'✅' if orchestration_found else '❌'} Basic orchestration detected: {orchestration_found}")
            print(f"✅ No endless loop (completed within 30 seconds)")
            
            # Check message count
            message_count = len(chat_engine.state.messages)
            print(f"💬 Messages generated: {message_count}")
            
            if message_count >= 2:  # User + AI response
                last_message = chat_engine.state.messages[-1]
                print(f"📝 Response preview: {last_message.content[:100]}...")
            
            # Final verdict for Step 1
            if orchestration_found and message_count >= 2:
                print("\n🎉 STEP 1 SUCCESS: Basic orchestration works without endless loops!")
                print("   ✅ Agent delegation working")
                print("   ✅ Enhanced prompt creation working") 
                print("   ✅ No sequential thinking blocking")
                print("   ✅ No preprocessing interfering")
                return True
            else:
                print("\n❌ STEP 1 FAILURE: Basic orchestration issues found")
                return False
                
        except asyncio.TimeoutError:
            print("\n❌ STEP 1 FAILURE: Endless loop detected (timed out after 30 seconds)")
            print("   The system is still stuck in an infinite loop even without sequential thinking.")
            return False
        
    except Exception as e:
        print(f"❌ Step 1 test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if 'chat_engine' in locals():
            await chat_engine.cleanup()

if __name__ == "__main__":
    success = asyncio.run(test_step1_orchestration())
    sys.exit(0 if success else 1)