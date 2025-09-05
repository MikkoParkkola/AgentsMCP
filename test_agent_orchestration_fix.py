#!/usr/bin/env python3
"""
Test script to verify agent orchestration fix.
Tests that product assessment queries properly trigger team coordination instead of falling back to direct response.
"""

import asyncio
import sys
import os
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine

async def test_agent_orchestration():
    """Test that agent orchestration works for product assessment queries."""
    print("🧪 Testing Agent Orchestration Fix")
    print("=" * 50)
    
    # Set up logging to capture what's happening
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
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
        
        # Test product assessment query (should trigger agent orchestration)
        test_query = "please make a product assessment with your team about the product in this directory"
        print(f"\n🔍 Testing query: '{test_query}'")
        print("=" * 60)
        
        # Process the input
        print("⚡ Processing input...")
        continue_chat = await chat_engine.process_input(test_query)
        
        print(f"\n📊 Analysis Results:")
        print("-" * 30)
        
        # Check if we saw the expected orchestration steps
        expected_steps = [
            "🧠 Analyzing request and planning approach",
            "🎯 Delegating to specialist agents", 
            "🚀 Executing enhanced response with agent coordination"
        ]
        
        fallback_indicators = [
            "⚠️ Falling back to direct response",
            "🤖 Generating direct response"
        ]
        
        orchestration_found = False
        fallback_found = False
        
        print("Status messages captured:")
        for i, status in enumerate(status_messages):
            print(f"  {i+1}. {status}")
            
            # Check for orchestration steps
            for step in expected_steps:
                if step in status:
                    orchestration_found = True
                    break
            
            # Check for fallback
            for fallback in fallback_indicators:
                if fallback in status:
                    fallback_found = True
                    break
        
        print(f"\n🎯 Test Results:")
        print(f"{'✅' if orchestration_found else '❌'} Agent orchestration detected: {orchestration_found}")
        print(f"{'❌' if fallback_found else '✅'} No fallback to direct response: {not fallback_found}")
        
        # Check message count to see if we got a proper response
        message_count = len(chat_engine.state.messages)
        print(f"💬 Messages generated: {message_count}")
        
        if message_count >= 2:  # User + AI response
            last_message = chat_engine.state.messages[-1]
            print(f"📝 Last response preview: {last_message.content[:100]}...")
        
        # Final verdict
        if orchestration_found and not fallback_found:
            print("\n🎉 SUCCESS: Agent orchestration is working correctly!")
            print("   Product assessment queries now properly trigger team coordination.")
            return True
        else:
            print("\n❌ FAILURE: Agent orchestration still has issues")
            if fallback_found:
                print("   Issue: Still falling back to direct response instead of using agents")
            if not orchestration_found:
                print("   Issue: Expected orchestration steps not found")
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
    success = asyncio.run(test_agent_orchestration())
    sys.exit(0 if success else 1)