#!/usr/bin/env python3
"""
Debug script to isolate the ModernTUI startup issue.
This reproduces the exact initialization sequence without CLI overhead.
"""

import asyncio
import sys
import traceback
import logging
from pathlib import Path

# Add src to path so we can import
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_modern_tui_startup():
    """Test ModernTUI startup in isolation."""
    
    print("🔍 Testing ModernTUI startup...")
    
    # Enable debug logging to catch all errors
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
    
    try:
        # Step 1: Test basic imports
        print("📦 Testing imports...")
        from agentsmcp.ui.modern_tui import ModernTUI
        from agentsmcp.ui.theme_manager import ThemeManager
        from agentsmcp.ui.cli_app import CLIConfig
        print("✅ All imports successful")
        
        # Step 2: Create basic dependencies
        print("🏗️  Creating dependencies...")
        theme_manager = ThemeManager()
        
        # Create minimal CLI config
        config = CLIConfig(
            theme_mode="auto",
            show_welcome=False,
            refresh_interval=2.0,
            orchestrator_model="gpt-5",
            agent_type="ollama-turbo-coding",
        )
        print("✅ Dependencies created")
        
        # Step 3: Test ModernTUI creation
        print("🎨 Creating ModernTUI instance...")
        tui = ModernTUI(
            config=config,
            theme_manager=theme_manager,
            conversation_manager=None,  # This might be the issue
            orchestration_manager=None,  # This might be the issue  
            theme="auto",
            no_welcome=True,
        )
        print("✅ ModernTUI instance created")
        
        # Step 4: Test the run method with timeout
        print("🚀 Testing tui.run() method...")
        
        # Use asyncio.wait_for to add a timeout
        try:
            await asyncio.wait_for(tui.run(), timeout=5.0)
            print("✅ ModernTUI ran successfully")
        except asyncio.TimeoutError:
            print("⚠️  ModernTUI run() timed out after 5 seconds (likely hanging)")
            print("    This suggests an infinite loop or blocking operation")
            return False
        except Exception as run_error:
            print(f"❌ ModernTUI run() failed: {run_error}")
            print(f"🔬 Error type: {type(run_error).__name__}")
            print("📋 Full traceback:")
            traceback.print_exc()
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error during setup: {e}")
        print(f"🔬 Error type: {type(e).__name__}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return False

async def test_with_real_dependencies():
    """Test with actual dependencies like those used in CLIApp."""
    
    print("\n🔍 Testing ModernTUI with real dependencies...")
    
    try:
        # Import dependencies
        from agentsmcp.ui.modern_tui import ModernTUI
        from agentsmcp.ui.theme_manager import ThemeManager
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.command_interface import CommandInterface
        
        print("✅ All imports successful")
        
        # Create dependencies like CLIApp does
        theme_manager = ThemeManager()
        
        # Create minimal orchestration manager
        class MockOrchestrationManager:
            async def get_system_status(self):
                return {"status": "mock"}
                
            async def initialize(self, mode="hybrid"):
                return {"status": "initialized"}
        
        orchestration_manager = MockOrchestrationManager()
        
        # Create command interface
        command_interface = CommandInterface(
            orchestration_manager=orchestration_manager,
            theme_manager=theme_manager,
            agent_manager=None,
            app_config=None
        )
        
        config = CLIConfig()
        
        print("🏗️  Creating ModernTUI with real dependencies...")
        tui = ModernTUI(
            config=config,
            theme_manager=theme_manager,
            conversation_manager=command_interface.conversation_manager,
            orchestration_manager=orchestration_manager,
            theme="auto",
            no_welcome=True,
        )
        
        print("🚀 Testing tui.run() with real dependencies...")
        
        try:
            await asyncio.wait_for(tui.run(), timeout=5.0)
            print("✅ ModernTUI with real dependencies ran successfully")
            return True
        except asyncio.TimeoutError:
            print("⚠️  ModernTUI timed out (may be working but hanging on input)")
            return False
        except Exception as e:
            print(f"❌ ModernTUI failed: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Error setting up real dependencies: {e}")
        traceback.print_exc()
        return False

def check_terminal_environment():
    """Check if we're in a suitable terminal environment."""
    print("\n🖥️  Checking terminal environment...")
    
    try:
        import sys
        print(f"📊 stdin.isatty(): {sys.stdin.isatty()}")
        print(f"📊 stdout.isatty(): {sys.stdout.isatty()}")
        print(f"📊 stderr.isatty(): {sys.stderr.isatty()}")
        
        if not sys.stdin.isatty():
            print("⚠️  stdin is not a TTY - this may cause TUI issues")
        if not sys.stdout.isatty():
            print("⚠️  stdout is not a TTY - this may cause TUI issues")
            
        # Test terminal size
        try:
            import os
            size = os.get_terminal_size()
            print(f"📏 Terminal size: {size.columns}x{size.lines}")
        except Exception as e:
            print(f"⚠️  Could not get terminal size: {e}")
            
        # Test Rich console
        try:
            from rich.console import Console
            console = Console()
            print(f"🎨 Rich console size: {console.size}")
            print(f"🎨 Rich color support: {console.color_system}")
        except Exception as e:
            print(f"❌ Rich console error: {e}")
            
    except Exception as e:
        print(f"❌ Terminal environment check failed: {e}")

async def main():
    """Main test runner."""
    print("🐛 ModernTUI Debug Session")
    print("=" * 50)
    
    # Check environment first
    check_terminal_environment()
    
    # Test 1: Basic startup
    success1 = await test_modern_tui_startup()
    
    # Test 2: With real dependencies  
    success2 = await test_with_real_dependencies()
    
    print("\n📊 Debug Results Summary:")
    print(f"✅ Basic startup: {'PASS' if success1 else 'FAIL'}")
    print(f"✅ With dependencies: {'PASS' if success2 else 'FAIL'}")
    
    if not (success1 or success2):
        print("\n💡 Recommended Next Steps:")
        print("1. Check ModernTUI.__init__() for required parameters")
        print("2. Check ModernTUI.run() for blocking operations")  
        print("3. Check for missing async/await in Rich Live context")
        print("4. Check for event loop issues in async task creation")
        print("5. Run with: python debug_modern_tui.py 2>&1 | tee debug.log")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Debug session interrupted")
    except Exception as e:
        print(f"\n💥 Debug session crashed: {e}")
        traceback.print_exc()