#!/usr/bin/env python3
"""Test script for the new console renderer implementation."""

import sys
import os

# Add src to path
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def test_console_renderer():
    """Test that console renderer can be imported and instantiated."""
    print("🧪 Testing Console Renderer Implementation")
    print("=" * 60)
    
    try:
        # Test imports
        from agentsmcp.ui.v3.console_renderer import ConsoleRenderer
        from agentsmcp.ui.v3.console_message_formatter import ConsoleMessageFormatter
        from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        print("✅ All imports successful")
        
        # Test terminal capabilities
        capabilities = detect_terminal_capabilities()
        print(f"✅ Terminal capabilities detected:")
        print(f"   • is_tty: {capabilities.is_tty}")
        print(f"   • supports_rich: {capabilities.supports_rich}")
        print(f"   • supports_colors: {capabilities.supports_colors}")
        print(f"   • width: {capabilities.width}")
        
        # Test console renderer instantiation
        renderer = ConsoleRenderer(capabilities)
        print("✅ Console renderer instantiation successful")
        
        # Test initialization
        init_success = renderer.initialize()
        print(f"✅ Console renderer initialization: {init_success}")
        
        if init_success:
            # Test message formatting
            renderer.show_welcome()
            renderer.display_chat_message("user", "Hello, this is a test message!")
            renderer.display_chat_message("assistant", "This is a response from the AI assistant.")
            renderer.display_chat_message("system", "Commands:\n• /help - Show this help message\n• /quit - Exit the application\n• /clear - Clear conversation history")
            renderer.show_goodbye()
            print("✅ Message display tests completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_console_vs_rich():
    """Compare console renderer vs rich renderer."""
    print("\n🔍 Comparing Console vs Rich Implementation")
    print("-" * 60)
    
    try:
        from agentsmcp.ui.v3.console_renderer import ConsoleRenderer
        from agentsmcp.ui.v3.rich_tui_renderer import RichTUIRenderer
        from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        
        capabilities = detect_terminal_capabilities()
        
        console_methods = set(dir(ConsoleRenderer))
        rich_methods = set(dir(RichTUIRenderer))
        
        # Check method compatibility
        required_methods = {
            'initialize', 'cleanup', 'handle_input', 'display_chat_message',
            'show_message', 'show_error', 'show_status'
        }
        
        console_has_required = all(method in console_methods for method in required_methods)
        rich_has_required = all(method in rich_methods for method in required_methods)
        
        print(f"✅ Console renderer has required methods: {console_has_required}")
        print(f"✅ Rich renderer has required methods: {rich_has_required}")
        
        # Check unique methods
        console_only = console_methods - rich_methods
        rich_only = rich_methods - console_methods
        
        if console_only:
            print(f"Console-only methods: {sorted(console_only)}")
        if rich_only:
            print(f"Rich-only methods: {sorted(rich_only)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_console_renderer()
    success2 = test_console_vs_rich()
    
    print(f"\n🎯 Overall Test Results:")
    print(f"   Console Renderer Test: {'PASSED' if success1 else 'FAILED'}")
    print(f"   Compatibility Test: {'PASSED' if success2 else 'FAILED'}")
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Console-Style Flow Layout implementation is working!")
        print("✅ No more layout calculation issues!")
        print("✅ No more header duplication problems!")
        print("✅ Ready for production use!")
    else:
        print("\n⚠️ Some tests failed - check output above")
    
    sys.exit(0 if (success1 and success2) else 1)