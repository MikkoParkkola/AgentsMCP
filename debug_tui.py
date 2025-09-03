#!/usr/bin/env python3
"""
Debug TUI Launcher
Forces debug mode to help diagnose input visibility issues
"""

import os
import sys

# Force debug mode
os.environ['AGENTSMCP_DEBUG'] = '1'
os.environ['AGENTSMCP_TUI_DEBUG'] = '1'

# Add project to Python path
project_root = '/Users/mikko/github/AgentsMCP'
if project_root not in sys.path:
    sys.path.insert(0, f'{project_root}/src')

print("🔍 Starting TUI in DEBUG MODE")
print("This will show detailed logging for input panel refreshes")
print("=" * 60)

# Launch the TUI
from agentsmcp.ui.cli_app import CLIApp
from agentsmcp.core.config import AgentsMCPConfig

async def main():
    config = AgentsMCPConfig()
    # Force debug mode in config
    config.debug_mode = True
    
    app = CLIApp(config, mode="tui")
    return await app.start()

if __name__ == "__main__":
    import asyncio
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n🛑 Debug TUI interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Debug TUI error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)