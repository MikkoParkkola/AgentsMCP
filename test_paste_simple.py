#!/usr/bin/env python3
"""Simple test script to verify paste functionality works"""

import sys
import os
sys.path.insert(0, 'src')

def test_paste_functionality():
    """Test the paste functionality directly"""
    from agentsmcp.ui.command_interface import CommandInterface
    from agentsmcp.ui.theme_manager import ThemeManager
    
    print("🧪 Simple Multi-line Paste Test")
    print("=" * 40)
    
    # Mock orchestration manager
    class MockOrchestrationManager:
        def __init__(self):
            self.is_running = True
    
    theme_manager = ThemeManager()
    orchestration_manager = MockOrchestrationManager()
    command_interface = CommandInterface(orchestration_manager, theme_manager)
    
    print("📋 Instructions:")
    print("1. Get ready to paste multi-line content")  
    print("2. Copy this test content:")
    print("   Line 1: Hello")
    print("   Line 2: World")
    print("   Line 3: Testing")
    print("3. When prompted, paste it immediately")
    print("4. The system should detect and capture all lines")
    print()
    
    try:
        # Test the paste detection directly
        print("Testing paste detection method...")
        result = command_interface._get_input_with_iterm2_paste_support("📋 paste here ▶ ")
        
        print(f"\n✅ Result captured: {repr(result)}")
        print(f"Lines: {len(result.split(chr(10)))}")
        print("Formatted:")
        print(result)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Environment check:")
    print(f"- Python: {sys.version}")
    print(f"- Platform: {sys.platform}")
    print(f"- Terminal: {os.environ.get('TERM_PROGRAM', 'unknown')}")
    print()
    
    test_paste_functionality()