#!/usr/bin/env python3
"""
Phase 1 UX Demo - Demonstrate Phase 1 UX improvements for AgentsMCP TUI v2.

This demo showcases:
1. Enhanced status indicators with clear state communication
2. Structured help system with search and categories  
3. Improved error messages with actionable recovery steps
4. Visual hierarchy improvements

Run this demo to see the Phase 1 improvements in action.
"""

import asyncio
import logging
import sys
from typing import Optional

from .application_controller import ApplicationController
from .status_manager import StatusManager, SystemState
from .chat_interface import ChatInterface, ChatInterfaceConfig
from .display_renderer import DisplayRenderer
from .terminal_manager import TerminalManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/agentsmcp_demo.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class Phase1Demo:
    """Demo application showcasing Phase 1 UX improvements."""
    
    def __init__(self):
        """Initialize the demo application."""
        self.app_controller: Optional[ApplicationController] = None
        self.chat_interface: Optional[ChatInterface] = None
        
    async def initialize(self) -> bool:
        """Initialize the demo application."""
        try:
            logger.info("Initializing Phase 1 UX Demo...")
            
            # Create application controller
            self.app_controller = ApplicationController()
            
            # Initialize the controller
            if not await self.app_controller.startup():
                logger.error("Failed to initialize application controller")
                return False
            
            # Create and initialize chat interface
            chat_config = ChatInterfaceConfig(
                enable_history_search=True,
                enable_multiline=True,
                enable_commands=True,
                show_timestamps=True,
                typing_timeout=15.0
            )
            
            self.chat_interface = ChatInterface(self.app_controller, chat_config)
            
            if not await self.chat_interface.initialize():
                logger.error("Failed to initialize chat interface")
                return False
            
            # Activate the chat interface
            await self.chat_interface.activate()
            
            logger.info("Phase 1 Demo initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing demo: {e}")
            return False
    
    async def demo_status_system(self):
        """Demonstrate the enhanced status system."""
        if not self.app_controller.status_manager:
            print("❌ Status manager not available")
            return
        
        status_manager = self.app_controller.status_manager
        
        print("\n" + "="*60)
        print("DEMO: Enhanced Status Indicators")
        print("="*60)
        
        # Demo different status states
        states_to_demo = [
            (SystemState.STARTING, "System initializing...", ""),
            (SystemState.LOADING, "Loading resources...", "Please wait"),
            (SystemState.PROCESSING, "AI processing request...", ""),
            (SystemState.READY, "System ready for input", ""),
            (SystemState.WARNING, "Minor configuration issue", "Check settings"),
            (SystemState.ERROR, "Connection failed", "Network unavailable"),
        ]
        
        for state, message, details in states_to_demo:
            await status_manager.set_status(state, message, details)
            
            # Display formatted status bar
            status_bar = status_manager.format_status_bar(80)
            print(f"\nStatus: {state.name}")
            for line in status_bar:
                print(line)
            
            await asyncio.sleep(1.5)  # Pause to see each status
        
        # Demo context information
        status_manager.update_context(
            agent_name="GPT-4",
            model_name="gpt-4-turbo",
            connection_status="Connected",
            active_connections=2,
            memory_usage="45.2MB"
        )
        
        print("\n" + "-"*60)
        print("Context Information:")
        context_line = status_manager.context_info.format_context_line(80)
        print(f"  {context_line}")
        
        # Reset to ready state
        await status_manager.set_status(SystemState.READY, "Demo complete")
    
    async def demo_help_system(self):
        """Demonstrate the structured help system."""
        print("\n" + "="*60)
        print("DEMO: Structured Help System")
        print("="*60)
        
        if self.app_controller.display_renderer:
            # Demo application controller help
            app_help = self.app_controller._cmd_help()
            print("\nApplication Controller Help:")
            print(app_help)
            
            # Demo chat interface help
            if self.chat_interface:
                print("\n" + "-"*60)
                print("Chat Interface Help:")
                await self.chat_interface._handle_help_command()
        else:
            print("❌ Display renderer not available for help demo")
    
    async def demo_error_handling(self):
        """Demonstrate improved error messages with recovery steps."""
        print("\n" + "="*60)
        print("DEMO: Enhanced Error Messages")
        print("="*60)
        
        if self.chat_interface:
            # Demo different types of errors
            await self.chat_interface._display_error_with_recovery(
                "Connection Error",
                "Failed to connect to the AI service. The remote server is not responding.",
                [
                    "1. Check your internet connection",
                    "2. Verify service status at status.openai.com",
                    "3. Try again in a few minutes",
                    "4. Use /restart to restart the application",
                    "5. Contact support if the problem persists"
                ]
            )
            
            await asyncio.sleep(2)
            
            await self.chat_interface._display_system_check_failure(
                "Input System",
                "Keyboard input handler has limited capabilities in this terminal environment."
            )
        else:
            print("❌ Chat interface not available for error demo")
    
    async def demo_visual_hierarchy(self):
        """Demonstrate visual hierarchy improvements."""
        print("\n" + "="*60)  
        print("DEMO: Visual Hierarchy Improvements")
        print("="*60)
        
        if self.app_controller.display_renderer:
            renderer = self.app_controller.display_renderer
            
            # Demo section headers
            print("\nSection Headers:")
            print(renderer.format_section_header("Main Section", 60, "double"))
            print()
            print(renderer.format_section_header("Subsection", 60, "single"))
            
            # Demo message boxes
            print("\nMessage Boxes:")
            print(renderer.format_message_box(
                "This is an informational message with helpful content.",
                60, "info"
            ))
            print()
            print(renderer.format_message_box(
                "Warning: This action cannot be undone. Please confirm before proceeding.",
                60, "warning"  
            ))
            print()
            print(renderer.format_message_box(
                "Error: The operation failed due to insufficient permissions.",
                60, "error"
            ))
            
            # Demo list formatting
            print("\nFormatted Lists:")
            items = [
                "First item with normal length",
                "Second item that is much longer and should demonstrate the wrapping capabilities of the formatting system",
                "Third item",
                "Fourth item with moderate length for testing"
            ]
            print(renderer.format_list_items(items, 60, "→"))
            
        else:
            print("❌ Display renderer not available for visual demo")
    
    async def run_interactive_demo(self):
        """Run interactive demo mode."""
        print("\n" + "="*60)
        print("INTERACTIVE DEMO MODE")
        print("="*60)
        print("""
Available commands:
  /help    - Show enhanced help system
  /status  - Display detailed system status  
  /error   - Trigger error demonstration
  /clear   - Clear screen
  /quit    - Exit demo

Try typing natural language or use commands to explore the interface!
""")
        
        # Run the chat interface main loop
        if self.chat_interface:
            # The chat interface will handle input and display
            try:
                while self.app_controller.is_running():
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted by user")
        else:
            print("❌ Chat interface not available for interactive demo")
    
    async def run_full_demo(self):
        """Run the complete Phase 1 demo."""
        if not await self.initialize():
            print("❌ Failed to initialize demo")
            return 1
        
        try:
            print("\n🚀 Welcome to AgentsMCP TUI v2 - Phase 1 UX Demo!")
            print("This demo showcases the enhanced user experience improvements.")
            
            # Run individual demos
            await self.demo_status_system()
            await self.demo_help_system() 
            await self.demo_error_handling()
            await self.demo_visual_hierarchy()
            
            # Interactive mode
            user_input = input("\n❓ Would you like to try interactive mode? (y/N): ")
            if user_input.lower() in ('y', 'yes'):
                await self.run_interactive_demo()
            
            print("\n✅ Phase 1 UX Demo completed successfully!")
            return 0
            
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
            return 0
        except Exception as e:
            logger.error(f"Demo error: {e}")
            print(f"❌ Demo failed: {e}")
            return 1
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up demo resources."""
        try:
            if self.chat_interface:
                await self.chat_interface.cleanup()
            
            if self.app_controller:
                await self.app_controller.shutdown(graceful=True)
                
            logger.info("Demo cleanup completed")
        except Exception as e:
            logger.error(f"Error during demo cleanup: {e}")


async def main():
    """Main entry point for the Phase 1 demo."""
    demo = Phase1Demo()
    return await demo.run_full_demo()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)