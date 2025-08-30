#!/usr/bin/env python3
"""
Test script to verify input echo fix in TUI footer.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src directory to Python path
repo_root = Path(__file__).parent
src_dir = repo_root / "src"
sys.path.insert(0, str(src_dir))

async def test_input_echo():
    """Test that input echo renders correctly in footer."""
    try:
        from agentsmcp.ui.components.realtime_input import RealTimeInputField
        from rich.console import Console
        
        console = Console()
        
        # Create input field
        input_field = RealTimeInputField(
            console=console,
            prompt=">>> ",
            max_width=80,
            max_height=3
        )
        
        print("✅ RealTimeInputField initialized")
        
        # Test character insertion
        await input_field.handle_key("h")
        await input_field.handle_key("e")
        await input_field.handle_key("l")
        await input_field.handle_key("l")
        await input_field.handle_key("o")
        
        current_input = input_field.get_current_input()
        print(f"✅ Current input: '{current_input}'")
        
        if current_input == "hello":
            print("✅ Input handling works correctly")
        else:
            print(f"❌ Input handling failed: expected 'hello', got '{current_input}'")
            return False
        
        # Test rendering
        rendered = input_field.render()
        print(f"✅ Rendered type: {type(rendered)}")
        
        # Test with console output
        console.print(rendered)
        print("✅ Rendering completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_input_echo())
    sys.exit(0 if success else 1)