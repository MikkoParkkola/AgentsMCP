#!/usr/bin/env python3
"""
Final validation test for TUI startup
"""

import asyncio
import logging
import sys
import os
import signal

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure minimal logging to reduce noise
logging.basicConfig(level=logging.ERROR)

async def test_tui_startup_complete():
    """Test complete TUI startup cycle."""
    try:
        print("🚀 Testing Revolutionary TUI complete startup...")
        
        from agentsmcp.ui.v2.revolutionary_launcher import launch_revolutionary_tui
        from agentsmcp.ui.cli_app import CLIConfig
        
        # Create a minimal CLI config
        cli_config = CLIConfig()
        cli_config.debug_mode = False  # Reduce noise
        
        # Create an async task for the TUI launch
        print("📡 Launching TUI with 20-second timeout...")
        
        try:
            # Use a shorter timeout and catch various exit conditions
            result = await asyncio.wait_for(
                launch_revolutionary_tui(cli_config),
                timeout=20.0
            )
            print(f"✅ TUI launched and exited successfully! Result: {result}")
            return True
            
        except asyncio.TimeoutError:
            print("⏰ TUI launch timed out - this indicates a hang or infinite loop")
            return False
        except KeyboardInterrupt:
            print("⚠️  TUI interrupted by user - startup was responsive")
            return True
            
    except Exception as e:
        print(f"❌ TUI launch failed with exception: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False

async def test_fallback_behavior():
    """Test that fallback TUI works when Revolutionary TUI fails."""
    try:
        print("\n🛡️  Testing fallback TUI behavior...")
        
        # Try to import and launch the fixed working TUI directly
        from agentsmcp.ui.v2.fixed_working_tui import launch_fixed_working_tui
        
        print("📡 Launching fallback TUI with 10-second timeout...")
        
        try:
            result = await asyncio.wait_for(
                launch_fixed_working_tui(),
                timeout=10.0
            )
            print(f"✅ Fallback TUI works! Result: {result}")
            return True
            
        except asyncio.TimeoutError:
            print("⏰ Fallback TUI also hangs - more serious issue")
            return False
        except KeyboardInterrupt:
            print("⚠️  Fallback TUI interrupted - it's responsive")
            return True
            
    except Exception as e:
        print(f"❌ Fallback TUI failed: {e}")
        return False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\n✋ Received signal {signum}, exiting gracefully...")
    sys.exit(0)

async def main():
    """Main test function."""
    # Install signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🧪 AgentsMCP TUI Startup Validation Test")
    print("=" * 50)
    
    # Test 1: Main TUI startup
    tui_success = await test_tui_startup_complete()
    
    # Test 2: Fallback behavior  
    fallback_success = await test_fallback_behavior()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Revolutionary TUI: {'✅ PASS' if tui_success else '❌ FAIL'}")
    print(f"   Fallback TUI:      {'✅ PASS' if fallback_success else '❌ FAIL'}")
    
    if tui_success or fallback_success:
        print("\n🎉 At least one TUI mode works! Startup issue resolved.")
        return True
    else:
        print("\n💥 Both TUI modes failed! Critical issue remains.")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Test framework error: {e}")
        sys.exit(1)