#!/usr/bin/env python3
"""
TEST CONVERSATION FLOW - REAL ISSUE INVESTIGATION
Test the complete conversation flow to see what's missing.
"""

import sys
import os
import time
import asyncio
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def test_conversation_flow():
    """Test the real conversation flow."""
    print("🔍 TESTING COMPLETE CONVERSATION FLOW")
    print("=" * 50)
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        from rich.layout import Layout
        from rich.console import Console
        
        # Create TUI like real usage
        class TestConfig:
            debug_mode = False
            verbose = False
        
        config = TestConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        print("✅ TUI instance created")
        
        # Initialize the TUI components (including event system)
        print(f"\n--- INITIALIZATION TEST ---")
        
        # We can't easily call async initialize from sync context, 
        # but let's test the sync parts and event system manually
        print("Testing event system startup...")
        
        # Start event system manually to test
        async def test_event_system():
            try:
                await tui.event_system.start()
                print("✅ Event system started successfully")
                
                # Check if it's running
                stats = tui.event_system.get_stats()
                print(f"Event system stats: {stats}")
                return True
            except Exception as e:
                print(f"❌ Event system failed to start: {e}")
                return False
        
        # Run the async test
        event_system_works = asyncio.run(test_event_system())
        
        if not event_system_works:
            print("⚠️  Event system not working - conversation updates will be lost")
            return
        
        # Test conversation history functionality
        print(f"\n--- CONVERSATION HISTORY TEST ---")
        
        print("Initial conversation history:")
        print(f"  Messages: {len(tui.state.conversation_history)}")
        
        # Manually add a test message like Enter key would
        timestamp = "12:34:56"
        test_message = {
            "role": "user",
            "content": "test message for conversation flow",
            "timestamp": timestamp
        }
        
        tui.state.conversation_history.append(test_message)
        print(f"Added test message: {test_message}")
        print(f"Updated history length: {len(tui.state.conversation_history)}")
        
        # Test conversation panel creation
        print(f"\n--- CONVERSATION PANEL TEST ---")
        
        try:
            # Check if there's a method to create conversation panel
            conversation_methods = [
                '_create_conversation_panel',
                '_create_chat_panel', 
                '_render_conversation',
                'get_conversation_display'
            ]
            
            found_methods = []
            for method_name in conversation_methods:
                if hasattr(tui, method_name):
                    found_methods.append(method_name)
                    print(f"✅ Found conversation method: {method_name}")
                else:
                    print(f"❌ Missing: {method_name}")
            
            if not found_methods:
                print("🚨 CRITICAL: No conversation display methods found!")
                print("This explains why conversation updates aren't visible!")
            
        except Exception as e:
            print(f"❌ Conversation panel test failed: {e}")
        
        # Test layout structure for conversation display
        print(f"\n--- LAYOUT STRUCTURE TEST ---")
        
        try:
            # Create basic layout like real TUI
            tui.layout = Layout()
            tui.layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="input", size=5)
            )
            
            # Check what goes in main section
            print("Layout structure created:")
            print(f"  Header: {type(tui.layout['header'])}")
            print(f"  Main: {type(tui.layout['main'])}")  
            print(f"  Input: {type(tui.layout['input'])}")
            
            # Try to populate main with conversation
            if len(tui.state.conversation_history) > 0:
                from rich.panel import Panel
                from rich.text import Text
                
                # Create a simple conversation display
                convo_text = Text()
                for msg in tui.state.conversation_history:
                    role_prefix = f"[{msg['role']}] " if 'role' in msg else ""
                    timestamp_suffix = f" ({msg['timestamp']})" if 'timestamp' in msg else ""
                    convo_text.append(f"{role_prefix}{msg['content']}{timestamp_suffix}\n")
                
                conversation_panel = Panel(convo_text, title="Conversation")
                tui.layout["main"].update(conversation_panel)
                print("✅ Conversation panel created and added to main layout")
                
                # Try to render it
                console = Console()
                with console.capture() as capture:
                    console.print(tui.layout)
                rendered = capture.get()
                
                print(f"Rendered layout preview (first 200 chars):")
                print(rendered[:200] + "..." if len(rendered) > 200 else rendered)
                
            else:
                print("⚠️  No conversation history to display")
                
        except Exception as e:
            print(f"❌ Layout structure test failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test the refresh mechanism with conversation updates
        print(f"\n--- REFRESH WITH CONVERSATION TEST ---")
        
        class ConversationTestDisplay:
            def __init__(self):
                self.refresh_count = 0
            
            def refresh(self):
                self.refresh_count += 1
                print(f"    🔄 Refresh #{self.refresh_count} - Layout updated")
        
        tui.live_display = ConversationTestDisplay()
        
        # Test refresh after conversation update
        try:
            print("Testing refresh with conversation...")
            tui._sync_refresh_display()
            print(f"✅ Refresh completed (calls: {tui.live_display.refresh_count})")
        except Exception as e:
            print(f"❌ Refresh with conversation failed: {e}")
        
        # Test event publishing for conversation updates
        print(f"\n--- EVENT PUBLISHING TEST ---")
        
        async def test_conversation_events():
            try:
                # Test publishing a conversation update event
                print("Publishing conversation_updated event...")
                result = await tui._publish_conversation_updated(test_message)
                print(f"Event publish result: {result}")
                
                # Check event system stats after publishing
                stats = tui.event_system.get_stats()
                print(f"Event stats after publish: {stats}")
                
                return True
            except Exception as e:
                print(f"❌ Event publishing failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        events_work = asyncio.run(test_conversation_events())
        
        # Final diagnosis
        print(f"\n--- DIAGNOSIS SUMMARY ---")
        print("=" * 50)
        
        issues_found = []
        solutions_needed = []
        
        if not found_methods:
            issues_found.append("❌ No conversation display methods")
            solutions_needed.append("✅ Create conversation display method")
        
        if not event_system_works:
            issues_found.append("❌ Event system not running")
            solutions_needed.append("✅ Fix event system initialization")
        
        if not events_work:
            issues_found.append("❌ Event publishing not working")
            solutions_needed.append("✅ Fix event publishing mechanism")
        
        print("ISSUES FOUND:")
        for issue in issues_found:
            print(f"  {issue}")
        
        if solutions_needed:
            print(f"\nSOLUTIONS NEEDED:")
            for solution in solutions_needed:
                print(f"  {solution}")
        else:
            print(f"\n🎉 NO CRITICAL ISSUES FOUND!")
        
        print(f"\nROOT CAUSE ANALYSIS:")
        if not found_methods:
            print("• Enter key works, events work, but UI doesn't show conversation updates")
            print("• Missing: Method to display conversation messages in main layout")
            print("• Missing: Event handlers to refresh conversation display")
        else:
            print("• All systems appear functional")
            print("• Issue may be environment-specific (TTY vs non-TTY)")
        
    except Exception as e:
        print(f"❌ Conversation flow test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conversation_flow()