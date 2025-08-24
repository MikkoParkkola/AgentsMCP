"""
Main entry point for AgentsMCP CLI

This allows running the CLI with: python -m agentsmcp
"""
import asyncio
import sys
from .ui.cli_app import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye from AgentsMCP!")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)