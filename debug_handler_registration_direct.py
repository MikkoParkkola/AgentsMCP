#!/usr/bin/env python3
"""
DEBUG HANDLER REGISTRATION - DIRECT TEST
Test handler registration step by step without relying on logging.
"""

import sys
import os
import asyncio
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

async def test_handler_registration_directly():
    """Test handler registration step by step."""
    print("🔍 DIRECT HANDLER REGISTRATION TEST")
    print("=" * 50)
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        # Create TUI
        class TestConfig:
            debug_mode = False
            verbose = False
        
        config = TestConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        print("✅ TUI instance created")
        
        # Initialize event system manually first
        print(f"\n--- EVENT SYSTEM INITIALIZATION ---")
        
        await tui.event_system.start()
        print(f"✅ Event system started")
        
        initial_stats = tui.event_system.get_stats()
        print(f"Initial handlers: {initial_stats['handler_count']}")
        
        # Test handler method existence
        print(f"\n--- HANDLER METHOD EXISTENCE TEST ---")
        
        handler_methods = [
            '_on_input_changed',
            '_on_agent_status_changed', 
            '_on_metrics_updated',
            '_on_conversation_updated',
            '_on_processing_state_changed',
            '_on_ui_refresh',
            '_handle_user_input_event',
            '_handle_agent_status_change',
            '_handle_performance_update'
        ]
        
        existing_methods = []
        for method_name in handler_methods:
            if hasattr(tui, method_name):
                method = getattr(tui, method_name)
                print(f"✅ {method_name}: {type(method)}")
                existing_methods.append((method_name, method))
            else:
                print(f"❌ {method_name}: NOT FOUND")
        
        # Test manual handler registration
        print(f"\n--- MANUAL HANDLER REGISTRATION TEST ---")
        
        if existing_methods:
            # Try registering the first handler manually
            method_name, method = existing_methods[0]
            print(f"Testing registration of {method_name}...")
            
            try:
                await tui.event_system.subscribe("test_event", method)
                print(f"✅ Manual registration successful")
                
                # Check stats after manual registration
                manual_stats = tui.event_system.get_stats()
                print(f"Handlers after manual registration: {manual_stats['handler_count']}")
                
                if manual_stats['handler_count'] > initial_stats['handler_count']:
                    print(f"✅ Handler count increased! Registration works.")
                else:
                    print(f"❌ Handler count didn't increase. Registration problem.")
                
                # Check internal storage
                print(f"\n--- INTERNAL STORAGE INSPECTION ---")
                enum_handlers = len(tui.event_system._handlers)
                named_handlers = len(tui.event_system._named_handlers)
                print(f"_handlers (enum): {enum_handlers} event types")
                print(f"_named_handlers (string): {named_handlers} event types")
                
                if named_handlers > 0:
                    print(f"Named events registered:")
                    for event_name, handler_list in tui.event_system._named_handlers.items():
                        alive_count = len([h for h in handler_list if h() is not None])
                        print(f"  {event_name}: {alive_count} alive handlers")
                
            except Exception as e:
                print(f"❌ Manual registration failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Test the full registration method
        print(f"\n--- FULL REGISTRATION METHOD TEST ---")
        
        if hasattr(tui, '_register_event_handlers'):
            print(f"Testing _register_event_handlers method...")
            
            try:
                await tui._register_event_handlers()
                print(f"✅ _register_event_handlers completed")
                
                # Check final stats
                final_stats = tui.event_system.get_stats()
                print(f"Handlers after full registration: {final_stats['handler_count']}")
                
                # Detailed breakdown
                enum_count = sum(len(handlers) for handlers in tui.event_system._handlers.values())
                named_count = sum(len(handlers) for handlers in tui.event_system._named_handlers.values())
                
                print(f"Enum handlers: {enum_count}")
                print(f"Named handlers: {named_count}")
                print(f"Total calculated: {enum_count + named_count}")
                print(f"get_stats() returned: {final_stats['handler_count']}")
                
                if final_stats['handler_count'] > 0:
                    print(f"🎉 SUCCESS! Handlers are registered")
                    return True
                else:
                    print(f"❌ FAILURE! No handlers registered")
                    return False
                
            except Exception as e:
                print(f"❌ Full registration failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"❌ _register_event_handlers method not found")
            return False
            
    except Exception as e:
        print(f"❌ Direct handler registration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_handler_registration_directly())
    
    print(f"\n{'='*50}")
    print(f"🎯 DIRECT TEST RESULT")
    print(f"{'='*50}")
    
    if success:
        print(f"✅ Handler registration working!")
    else:
        print(f"❌ Handler registration broken.")