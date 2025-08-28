#!/usr/bin/env python3
"""Simple interactive test for multi-line input"""

import sys
import os
sys.path.insert(0, 'src')

def test_interactive():
    """Test interactive input system"""
    from agentsmcp.ui.command_interface import CommandInterface
    from agentsmcp.ui.theme_manager import ThemeManager
    
    print("🧪 Interactive Multi-line Test")
    print("=" * 40)
    
    class MockOrchestrationManager:
        def __init__(self):
            self.is_running = True
    
    theme_manager = ThemeManager()
    orchestration_manager = MockOrchestrationManager()
    command_interface = CommandInterface(orchestration_manager, theme_manager)
    
    print("📋 Test Instructions:")
    print("1. Type 'hello' and press Enter → should send immediately")
    print("2. Copy and paste multiple lines → should detect and capture all")
    print("3. Type 'exit' to quit")
    print()
    
    test_count = 0
    while True:
        try:
            test_count += 1
            result = command_interface._get_input_with_autocomplete(f"test {test_count}> ")
            
            if result.lower() in ['exit', 'quit']:
                break
                
            print(f"✅ Captured: {repr(result)}")
            if '\n' in result:
                print(f"📝 Multi-line: {len(result.split(chr(10)))} lines")
            print()
            
        except (KeyboardInterrupt, EOFError):
            break
    
    print("👋 Test completed!")

if __name__ == "__main__":
    test_interactive()