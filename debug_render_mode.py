#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

from agentsmcp.ui.v2.terminal_manager import create_terminal_manager
from agentsmcp.ui.v2.display_renderer import DisplayRenderer, RenderMode
import asyncio

async def debug_render_mode():
    print("=== Debug: Terminal Capabilities & Render Mode ===")
    
    # Create terminal manager
    term_mgr = create_terminal_manager()
    if not await term_mgr.initialize():
        print("❌ Failed to initialize terminal manager")
        return
        
    # Get capabilities
    caps = term_mgr.detect_capabilities()
    print(f"Terminal capabilities:")
    print(f"  width: {caps.width}")
    print(f"  height: {caps.height}")
    print(f"  colors: {caps.colors}")
    print(f"  unicode_support: {caps.unicode_support}")
    print(f"  mouse_support: {caps.mouse_support}")
    print(f"  alternate_screen: {caps.alternate_screen}")
    print(f"  cursor_control: {caps.cursor_control}")
    print(f"  interactive: {caps.interactive}")
    print(f"  term_program: {caps.term_program}")
    
    # Create display renderer
    renderer = DisplayRenderer(term_mgr)
    if not await renderer.initialize():
        print("❌ Failed to initialize display renderer")
        return
        
    # Check render mode
    render_mode = renderer._render_mode
    print(f"\nRender mode: {render_mode}")
    print(f"Alternate screen active: {renderer._alternate_screen}")
    
    # Clean up
    await renderer.cleanup()
    await term_mgr.cleanup()

if __name__ == "__main__":
    asyncio.run(debug_render_mode())