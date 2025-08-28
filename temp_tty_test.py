
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_tui():
    from agentsmcp.ui.modern_tui import ModernTUI
    from agentsmcp.ui.theme_manager import ThemeManager
    from agentsmcp.ui.cli_app import CLIConfig
    from agentsmcp.ui.command_interface import CommandInterface
    
    print("📊 TTY Status in pseudo-TTY:")
    print(f"stdin.isatty(): {sys.stdin.isatty()}")
    print(f"stdout.isatty(): {sys.stdout.isatty()}")
    
    if not sys.stdin.isatty():
        print("❌ Still not a TTY!")
        return False
        
    print("✅ We have a TTY! Testing ModernTUI...")
    
    # Create dependencies
    theme_manager = ThemeManager()
    config = CLIConfig(show_welcome=False)
    
    # Mock orchestration manager
    class MockOrchestrationManager:
        async def get_system_status(self):
            return {"status": "mock"}
        async def initialize(self, mode="hybrid"):
            return {"status": "initialized"}
    
    orchestration_manager = MockOrchestrationManager()
    command_interface = CommandInterface(
        orchestration_manager=orchestration_manager,
        theme_manager=theme_manager,
        agent_manager=None,
        app_config=None
    )
    
    tui = ModernTUI(
        config=config,
        theme_manager=theme_manager,
        conversation_manager=command_interface.conversation_manager,
        orchestration_manager=orchestration_manager,
        theme="auto",
        no_welcome=True,
    )
    
    print("🚀 Starting ModernTUI.run()...")
    
    # Run for a short time then exit
    try:
        # We'll timeout quickly since this is a test
        await asyncio.wait_for(tui.run(), timeout=2.0)
    except asyncio.TimeoutError:
        print("⏰ Timeout (normal for this test)")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

asyncio.run(test_tui())
