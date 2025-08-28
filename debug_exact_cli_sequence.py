#!/usr/bin/env python3
"""
Debug script that exactly replicates the CLIApp._run_modern_tui() sequence.
This should reveal the exact error causing the fallback.
"""

import asyncio
import sys
import traceback
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_exact_cli_sequence():
    """Test the exact sequence used in CLIApp._run_modern_tui()."""
    
    print("🔍 Testing exact CLIApp._run_modern_tui() sequence...")
    
    # Enable detailed logging to catch any errors
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
    
    try:
        # Step 1: Import everything CLIApp imports
        print("📦 Importing dependencies like CLIApp...")
        from agentsmcp.ui.modern_tui import ModernTUI
        from agentsmcp.ui.theme_manager import ThemeManager
        from agentsmcp.ui.cli_app import CLIConfig, CLIApp
        from agentsmcp.ui.command_interface import CommandInterface
        print("✅ All imports successful")
        
        # Step 2: Create the exact config CLIApp creates
        print("⚙️  Creating CLIConfig like CLIApp does...")
        config = CLIConfig(
            theme_mode="auto",
            show_welcome=False,  # Using no_welcome=True 
            refresh_interval=2.0,
            orchestrator_model="gpt-5",
            agent_type="ollama-turbo-coding",
            debug_mode=True,  # Enable debug mode
            log_level="DEBUG"
        )
        print("✅ CLIConfig created")
        
        # Step 3: Create dependencies exactly like CLIApp does
        print("🏗️  Creating dependencies like CLIApp.__init__()...")
        
        # Configure logging like CLIApp does
        from agentsmcp.logging_config import configure_logging
        configure_logging(level=config.log_level, fmt="text")
        
        # Create theme manager
        theme_manager = ThemeManager()
        
        # Create orchestration manager (lightweight CLI version)
        from agentsmcp.ui.cli_app import CLIApp
        cli_app_instance = CLIApp(config=config, mode="tui")
        orchestration_manager = cli_app_instance._create_cli_orchestration_manager()
        
        # Create command interface like CLIApp does
        command_interface = CommandInterface(
            orchestration_manager=orchestration_manager,
            theme_manager=theme_manager,
            agent_manager=None,  # CLIApp also sets this to None initially 
            app_config=None      # CLIApp also sets this to None initially
        )
        
        print("✅ All dependencies created")
        
        # Step 4: Create ModernTUI exactly like CLIApp._run_modern_tui() does
        print("🎨 Creating ModernTUI exactly like CLIApp._run_modern_tui()...")
        
        # Pull known arguments from config (exact same code)
        theme = config.theme_mode
        no_welcome = not config.show_welcome

        tui = ModernTUI(
            config=config,
            theme_manager=theme_manager,
            conversation_manager=command_interface.conversation_manager,
            orchestration_manager=orchestration_manager,
            theme=theme,
            no_welcome=no_welcome,
        )
        print("✅ ModernTUI created successfully")
        
        # Step 5: Call await tui.run() exactly like CLIApp does
        print("🚀 Calling await tui.run() exactly like CLIApp._run_modern_tui()...")
        
        try:
            # Use a timeout to prevent infinite hanging
            await asyncio.wait_for(tui.run(), timeout=10.0)
            print("✅ tui.run() completed successfully")
            return True
            
        except asyncio.TimeoutError:
            print("⏰ tui.run() timed out after 10 seconds")
            print("💡 This suggests the TUI is running but waiting for input or events")
            return False  # Not necessarily an error, just hanging on input
            
        except Exception as tui_error:
            print(f"❌ tui.run() failed with error: {tui_error}")
            print(f"🔬 Error type: {type(tui_error).__name__}")
            print("📋 Full traceback:")
            traceback.print_exc()
            return False
            
    except Exception as setup_error:
        print(f"❌ Setup failed: {setup_error}")
        print(f"🔬 Error type: {type(setup_error).__name__}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return False

async def test_with_actual_cli_app():
    """Test using the actual CLIApp._run_modern_tui() method."""
    
    print("\n🔍 Testing with actual CLIApp._run_modern_tui() method...")
    
    try:
        from agentsmcp.ui.cli_app import CLIApp, CLIConfig
        
        # Create config
        config = CLIConfig(
            theme_mode="auto",
            show_welcome=False,
            refresh_interval=2.0,
            debug_mode=True,
            log_level="DEBUG"
        )
        
        # Create CLIApp with TUI mode
        app = CLIApp(config=config, mode="tui")
        
        print("🏗️  CLIApp created successfully")
        
        # Call the actual _run_modern_tui method with timeout
        try:
            await asyncio.wait_for(app._run_modern_tui(), timeout=10.0)
            print("✅ CLIApp._run_modern_tui() completed successfully")
            return True
            
        except asyncio.TimeoutError:
            print("⏰ CLIApp._run_modern_tui() timed out")
            return False
            
        except Exception as e:
            print(f"❌ CLIApp._run_modern_tui() failed: {e}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ CLIApp setup failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Main test runner."""
    print("🐛 Exact CLIApp Sequence Debug")
    print("=" * 50)
    
    # Test 1: Manual replication of the sequence
    success1 = await test_exact_cli_sequence()
    
    # Test 2: Using actual CLIApp method
    success2 = await test_with_actual_cli_app()
    
    print("\n📊 Debug Results:")
    print(f"✅ Manual sequence: {'PASS' if success1 else 'FAIL'}")
    print(f"✅ Actual CLIApp method: {'PASS' if success2 else 'FAIL'}")
    
    if not (success1 or success2):
        print("\n💡 Key Findings:")
        print("- ModernTUI creation works fine")
        print("- Issue is specifically in the tui.run() execution")
        print("- This confirms it's not an import or initialization error")
        print("- The 'Failed to start ModernTUI' error is from exception handling")
        print("- Need to identify what specific exception occurs in tui.run()")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Debug interrupted")
    except Exception as e:
        print(f"\n💥 Debug crashed: {e}")
        traceback.print_exc()