#!/usr/bin/env python3
"""
Test script specifically for iTerm2 multi-line paste and input functionality.
This tests the enhanced multi-line input system in AgentsMCP.
"""

import sys
import os
sys.path.insert(0, 'src')

def test_iterm2_multiline():
    """Test iTerm2 multi-line input with paste detection"""
    from agentsmcp.ui.command_interface import CommandInterface
    from agentsmcp.ui.theme_manager import ThemeManager
    
    print("🧪 iTerm2 Multi-line Input Test")
    print("=" * 50)
    
    # Mock orchestration manager
    class MockOrchestrationManager:
        def __init__(self):
            self.is_running = True
    
    theme_manager = ThemeManager()
    orchestration_manager = MockOrchestrationManager()
    command_interface = CommandInterface(orchestration_manager, theme_manager)
    
    print("📋 Instructions for iTerm2 Testing:")
    print("-" * 40)
    print("1. Single line test:")
    print("   - Type: 'hello world'")
    print("   - Press Enter → should send immediately")
    print()
    print("2. Multi-line typing test:")
    print("   - Type: 'This is line 1'")
    print("   - Press Option+Enter (⌥+Enter)")
    print("   - Type: 'This is line 2'") 
    print("   - Press Enter → should send both lines")
    print()
    print("3. Multi-line paste test:")
    print("   - Copy this multi-line text:")
    print("     Line 1 of pasted content")
    print("     Line 2 of pasted content") 
    print("     Line 3 of pasted content")
    print("   - Paste it → should be detected and preserved as single input")
    print()
    print("4. Smart completion test:")
    print("   - Type: 'if True:'")
    print("   - Press Enter → should add newline (not send)")
    print("   - Type: '    print(\"hello\")'")
    print("   - Press Enter → should send complete block")
    print()
    print("5. Type 'exit' to quit")
    print()
    
    test_count = 0
    
    while True:
        try:
            test_count += 1
            
            # Test the multi-line input system
            user_input = command_interface._get_input_with_autocomplete(f"🧪 test {test_count} ▶ ")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
                
            # Show what we captured
            print(f"\n📝 Captured input:")
            print("─" * 50)
            print(f"Raw: {repr(user_input)}")
            print("─" * 50)
            print("Formatted:")
            print(user_input)
            print("─" * 50)
            print(f"Lines: {len(user_input.split(chr(10)))}")
            print(f"Characters: {len(user_input)}")
            
            # Analyze the input type
            if '\n' in user_input:
                print(f"✅ Multi-line input detected ({len(user_input.split(chr(10)))} lines)")
            else:
                print(f"✅ Single-line input detected")
            print()
            
        except (KeyboardInterrupt, EOFError):
            break
    
    print("\n👋 iTerm2 multi-line test completed!")
    print("\nResults Summary:")
    print("- Multi-line paste should preserve all line breaks as single input")
    print("- Option+Enter should create new lines during typing")
    print("- Smart Enter should detect incomplete vs complete statements")
    print("- No asyncio warnings should appear")

def test_paste_detection_mechanism():
    """Test the paste detection mechanism standalone"""
    print("\n🔧 Testing Paste Detection Mechanism")
    print("=" * 40)
    
    import select
    import sys
    
    print("Testing select() availability...")
    try:
        # Test if select works on this system
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        print("✅ select() module available and working")
        print(f"Ready streams: {len(ready)}")
    except Exception as e:
        print(f"❌ select() not available: {e}")
    
    print("\n📋 Manual paste test:")
    print("1. Have multi-line content ready to paste")
    print("2. When prompted, paste the content immediately")
    print("3. The system should detect it as paste vs typing")
    
    from agentsmcp.ui.command_interface import CommandInterface
    from agentsmcp.ui.theme_manager import ThemeManager
    
    class MockOrchestrationManager:
        def __init__(self):
            self.is_running = True
    
    theme_manager = ThemeManager()
    orchestration_manager = MockOrchestrationManager()
    command_interface = CommandInterface(orchestration_manager, theme_manager)
    
    try:
        result = command_interface._get_input_with_iterm2_fallback("📋 paste test ▶ ")
        print(f"\n✅ Result: {repr(result)}")
        print(f"Lines captured: {len(result.split(chr(10)))}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 AgentsMCP iTerm2 Multi-line Input Test Suite")
    print("=" * 60)
    print("Environment:")
    print(f"- Python: {sys.version}")
    print(f"- Platform: {sys.platform}")
    print(f"- Terminal: {os.environ.get('TERM', 'unknown')}")
    print(f"- Terminal Program: {os.environ.get('TERM_PROGRAM', 'unknown')}")
    print()
    
    if os.environ.get('TERM_PROGRAM') == 'iTerm.app':
        print("✅ Running in iTerm2 - perfect for testing!")
    else:
        print("⚠️  Not running in iTerm2 - results may vary")
    print()
    
    try:
        # Test paste detection first
        test_paste_detection_mechanism()
        print()
        
        # Then test full functionality
        test_iterm2_multiline()
        
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Expected behavior in iTerm2:")
    print("- Paste: Multi-line content pasted → detected and preserved")
    print("- Typing: Option+Enter → creates new line")  
    print("- Typing: Regular Enter → smart send vs continue")
    print("- No asyncio event loop warnings")
    print("- Consistent behavior across all input scenarios")