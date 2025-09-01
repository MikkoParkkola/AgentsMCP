#!/usr/bin/env python3
"""
Direct launcher for the Revolutionary TUI Interface.

This script provides a direct way to launch the now-FIXED Revolutionary TUI
with all its advanced features and working input handling.

Usage:
    python -m agentsmcp.ui.v2.launch_revolutionary_tui
    
Or directly:
    python src/agentsmcp/ui/v2/launch_revolutionary_tui.py
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the src path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from agentsmcp.ui.v2.revolutionary_tui_interface import create_revolutionary_interface

async def main():
    """Launch the Revolutionary TUI Interface."""
    print("🚀 Launching Revolutionary TUI Interface...")
    print("✅ Input handling is now FIXED - you will see your typing!")
    
    try:
        # Create and run the revolutionary interface
        interface = await create_revolutionary_interface()
        result = await interface.run()
        return result
        
    except KeyboardInterrupt:
        print("\n👋 Revolutionary TUI interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching Revolutionary TUI: {e}")
        print("\n💡 Try the basic TUI instead:")
        print("   python -m agentsmcp.ui.v2.fixed_working_tui")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))