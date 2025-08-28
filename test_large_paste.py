#!/usr/bin/env python3
"""Test large multi-line paste functionality"""

import sys
import os
sys.path.insert(0, 'src')

def test_large_paste():
    """Test the paste functionality with large content"""
    from agentsmcp.ui.command_interface import CommandInterface
    from agentsmcp.ui.theme_manager import ThemeManager
    
    print("🧪 Large Multi-line Paste Test")
    print("=" * 40)
    
    class MockOrchestrationManager:
        def __init__(self):
            self.is_running = True
    
    theme_manager = ThemeManager()
    orchestration_manager = MockOrchestrationManager()
    command_interface = CommandInterface(orchestration_manager, theme_manager)
    
    print("📋 Instructions:")
    print("1. Copy the large analysis content you provided")
    print("2. When prompted, paste it (⌘+V)")  
    print("3. The system should detect and capture all lines as single input")
    print("4. Look for bracketed paste detection or multi-line paste confirmation")
    print()
    
    try:
        result = command_interface._get_input_with_autocomplete("📋 Paste large content here ▶ ")
        
        print(f"\n✅ Result captured!")
        print(f"📊 Length: {len(result)} characters")
        print(f"📄 Lines: {len(result.split(chr(10)))}")
        print("\n🔍 First 200 characters:")
        print("-" * 50)
        print(result[:200] + ("..." if len(result) > 200 else ""))
        print("-" * 50)
        
        if len(result.split('\n')) > 20:
            print(f"✅ Large multi-line content successfully captured!")
        elif len(result) > 500:
            print(f"✅ Large content successfully captured!")
        else:
            print(f"⚠️  Small content - may not have tested large paste scenario")
        
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
    
    test_large_paste()