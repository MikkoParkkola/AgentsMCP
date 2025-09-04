#!/usr/bin/env python3
"""Manual test for streaming integration - run a single conversation."""

import asyncio
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.ui.v3.chat_engine import ChatEngine


async def test_streaming_manually():
    """Manual test of streaming functionality."""
    print("🧪 Manual Streaming Test")
    print("=" * 30)
    
    # Create chat engine
    engine = ChatEngine()
    
    # Track streaming updates
    streaming_updates = []
    all_messages = []
    
    def capture_status(status):
        if status.startswith("streaming_update:"):
            content = status[17:]
            streaming_updates.append(content)
            print(f"📡 Streaming: {content}")
    
    def capture_message(message):
        all_messages.append(message)
        print(f"💬 Message: {message.role.value} -> {message.content}")
    
    engine.set_callbacks(
        status_callback=capture_status,
        message_callback=capture_message
    )
    
    # Test message
    print("\n🎯 Testing streaming with: 'Hello, can you help me?'")
    print("-" * 50)
    
    try:
        result = await engine.process_input("Hello, can you help me?")
        
        print(f"\n📊 Results:")
        print(f"   • Processing result: {result}")
        print(f"   • Streaming updates: {len(streaming_updates)}")
        print(f"   • Total messages: {len(all_messages)}")
        
        if streaming_updates:
            print(f"   • First update: {streaming_updates[0]}")
            print(f"   • Final update: {streaming_updates[-1]}")
        else:
            print(f"   • No streaming updates (fallback to batch mode)")
        
        if len(all_messages) >= 2:
            print(f"   • User message: {all_messages[0].content}")
            print(f"   • AI response: {all_messages[1].content}")
        
        # Check if streaming is supported
        print(f"\n🔍 LLM Client Status:")
        if engine._llm_client:
            supports_streaming = engine._llm_client.supports_streaming()
            print(f"   • Streaming supported: {supports_streaming}")
            config_status = engine._llm_client.get_configuration_status()
            print(f"   • Current provider: {config_status['current_provider']}")
        else:
            print(f"   • LLM Client: Not initialized")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Manual test completed")


if __name__ == "__main__":
    asyncio.run(test_streaming_manually())