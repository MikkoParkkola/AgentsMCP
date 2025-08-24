#!/usr/bin/env python3
"""
AgentsMCP CLI Demo Script

Showcases the revolutionary CLI interface with all its beautiful features:
- Adaptive theme detection (dark/light based on terminal)
- Real-time orchestration dashboard
- Advanced statistics with sparklines and trends
- Interactive command interface with smart completion
- Apple-style design principles throughout
"""
import asyncio
import sys
import os
from pathlib import Path

# Add src directory to path for importing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agentsmcp.ui import CLIApp, CLIConfig


async def demo_interactive_mode():
    """Demo the interactive command interface"""
    print("🎮 Starting Interactive Mode Demo...")
    
    config = CLIConfig(
        interface_mode="interactive",
        show_welcome=True,
        theme_mode="auto",
        auto_refresh=True
    )
    
    app = CLIApp(config)
    await app.start()


async def demo_dashboard_mode():
    """Demo the status dashboard"""
    print("📊 Starting Dashboard Mode Demo...")
    
    config = CLIConfig(
        interface_mode="dashboard", 
        show_welcome=False,
        theme_mode="auto",
        auto_refresh=True,
        refresh_interval=1.5
    )
    
    app = CLIApp(config)
    await app.start()


async def demo_statistics_mode():
    """Demo the statistics display"""
    print("📈 Starting Statistics Mode Demo...")
    
    config = CLIConfig(
        interface_mode="stats",
        show_welcome=False, 
        theme_mode="auto",
        auto_refresh=True,
        refresh_interval=2.0
    )
    
    app = CLIApp(config)
    await app.start()


async def demo_all_modes():
    """Demo all modes in sequence"""
    print("🚀 Welcome to the AgentsMCP CLI Demo!")
    print("This will showcase all three interface modes...")
    print()
    
    modes = [
        ("Interactive Mode", demo_interactive_mode),
        ("Dashboard Mode", demo_dashboard_mode), 
        ("Statistics Mode", demo_statistics_mode)
    ]
    
    for mode_name, demo_func in modes:
        print(f"🎯 Launching {mode_name}...")
        print("   Press Ctrl+C to move to the next mode")
        print()
        
        try:
            await demo_func()
        except KeyboardInterrupt:
            print(f"\n✅ {mode_name} demo completed!")
            print()
            continue
    
    print("🎊 All demos completed! Thank you for trying AgentsMCP!")


def main():
    """Main demo entry point"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "interactive":
            asyncio.run(demo_interactive_mode())
        elif mode == "dashboard":
            asyncio.run(demo_dashboard_mode())
        elif mode == "stats":
            asyncio.run(demo_statistics_mode())
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: interactive, dashboard, stats")
            sys.exit(1)
    else:
        # Show help and run interactive demo
        print("""
🚀 AgentsMCP CLI Demo

Usage:
  python demo_cli.py                  # Interactive mode (default)
  python demo_cli.py interactive      # Interactive command interface  
  python demo_cli.py dashboard        # Real-time orchestration dashboard
  python demo_cli.py stats            # Advanced statistics and trends
  
Features demonstrated:
✨ Adaptive dark/light theme detection based on your terminal
📊 Beautiful real-time metrics with sparklines and trend analysis
🎼 Symphony mode orchestration for multi-agent coordination
🧠 Predictive agent spawning with AI-powered provisioning
⚡ Sub-millisecond response times with smooth animations
🎨 Apple-style design principles throughout the interface

Press Ctrl+C at any time to exit a mode.
        """)
        
        try:
            asyncio.run(demo_interactive_mode())
        except KeyboardInterrupt:
            print("\n👋 Thanks for trying AgentsMCP!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye from AgentsMCP!")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Demo error: {e}")
        sys.exit(1)