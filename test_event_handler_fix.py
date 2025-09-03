#!/usr/bin/env python3
"""
TEST EVENT HANDLER FIX
Verify that event handlers are now properly registered and counted.
"""

import sys
import os
import asyncio
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

async def test_event_handler_fix():
    """Test that event handlers are properly registered and counted."""
    print("🔍 TESTING EVENT HANDLER FIX")
    print("=" * 50)
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        # Create TUI like real usage
        class TestConfig:
            debug_mode = False
            verbose = False
        
        config = TestConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        print("✅ TUI instance created")
        
        # Test full initialization
        print(f"\n--- FULL INITIALIZATION TEST ---")
        
        try:
            init_result = await tui.initialize()
            print(f"✅ Initialization: {'SUCCESS' if init_result else 'FAILED'}")
            
            if not init_result:
                print("❌ CRITICAL: Initialization failed")
                return False
                
        except Exception as e:
            print(f"❌ CRITICAL: Initialization exception: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test event system statistics AFTER full initialization
        print(f"\n--- EVENT SYSTEM STATS (AFTER FIX) ---")
        
        event_stats = tui.event_system.get_stats()
        print(f"Running: {event_stats['running']}")
        print(f"Handlers: {event_stats['handler_count']}")
        print(f"Queue size: {event_stats['queue_size']}")
        print(f"Events processed: {event_stats['events_processed']}")
        
        if event_stats['handler_count'] == 0:
            print("❌ STILL NO HANDLERS! Fix didn't work.")
            return False
        elif event_stats['handler_count'] > 0:
            print(f"✅ SUCCESS! {event_stats['handler_count']} handlers registered")
        
        # Test specific handler registration
        print(f"\n--- HANDLER DETAILS TEST ---")
        
        # Access the internal handler storage to verify
        enum_handlers = sum(len(handlers) for handlers in tui.event_system._handlers.values())
        named_handlers = sum(len(handlers) for handlers in tui.event_system._named_handlers.values())
        
        print(f"Enum handlers: {enum_handlers}")
        print(f"Named handlers: {named_handlers}")
        print(f"Total: {enum_handlers + named_handlers}")
        
        # List the named event types that have handlers
        print(f"\nRegistered named events:")
        for event_name, handlers in tui.event_system._named_handlers.items():
            handler_count = len([h for h in handlers if h() is not None])  # Count alive handlers
            print(f"  {event_name}: {handler_count} handlers")
        
        # Test event publishing
        print(f"\n--- EVENT PUBLISHING TEST ---")
        
        try:
            # Test publishing a conversation update event
            test_message = {
                "role": "user", 
                "content": "test event handler fix",
                "timestamp": "12:34:56"
            }
            
            print(f"Publishing conversation_updated event...")
            
            # Use the internal publish method
            await tui._publish_conversation_updated(test_message)
            
            print(f"✅ Event published successfully")
            
            # Give the event system a moment to process
            await asyncio.sleep(0.1)
            
            # Check updated stats
            updated_stats = tui.event_system.get_stats()
            print(f"Events processed after test: {updated_stats['events_processed']}")
            
            if updated_stats['events_processed'] > event_stats['events_processed']:
                print(f"✅ Event was processed by handlers!")
            else:
                print(f"⚠️  Event may not have been processed")
                
        except Exception as e:
            print(f"❌ Event publishing failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Final assessment
        print(f"\n--- FINAL ASSESSMENT ---")
        print("=" * 50)
        
        if event_stats['handler_count'] > 0:
            print(f"🎉 EVENT HANDLER FIX SUCCESSFUL!")
            print(f"• {event_stats['handler_count']} handlers registered")
            print(f"• Event system running: {event_stats['running']}")
            print(f"• This should fix UI updates when Enter key is pressed")
            print(f"\n📊 Expected behavior now:")
            print(f"• Type in TUI input -> visible immediately ✅")
            print(f"• Press Enter -> message sent AND UI updates with response")
            print(f"• Conversation history displays properly")
            return True
        else:
            print(f"❌ EVENT HANDLER FIX FAILED")
            print(f"• Still 0 handlers registered")
            print(f"• UI updates will still not work")
            return False
            
    except Exception as e:
        print(f"❌ Event handler fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_event_handler_fix())
    
    print(f"\n{'='*50}")
    print(f"🎯 EVENT HANDLER FIX TEST RESULT")
    print(f"{'='*50}")
    
    if success:
        print(f"✅ FIX SUCCESSFUL - Event handlers now registered!")
        print(f"The TUI should now properly update when Enter is pressed.")
    else:
        print(f"❌ FIX FAILED - Event handlers still not working.")
        print(f"Need further investigation.")