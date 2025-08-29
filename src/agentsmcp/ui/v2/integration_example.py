"""
Integration example showing how the display and rendering systems work together.

Demonstrates the display_renderer, layout_engine, and themes systems working
together to create a clean TUI without scrollback pollution.
"""

import asyncio
import sys
import logging
from typing import Optional

from .terminal_manager import TerminalManager
from .display_renderer import DisplayRenderer
from .layout_engine import create_standard_tui_layout, TextNode
from .themes import ThemeManager, detect_preferred_scheme
from .event_system import AsyncEventSystem, Event, EventType


logger = logging.getLogger(__name__)


class SimpleTUIDemo:
    """
    Simple TUI demo showing integrated v2 systems.
    
    Demonstrates scrollback-pollution-free rendering with clean layout
    and proper terminal handling.
    """
    
    def __init__(self):
        """Initialize the demo."""
        self.terminal_manager = TerminalManager()
        self.theme_manager = ThemeManager(self.terminal_manager)
        self.display_renderer = DisplayRenderer(self.terminal_manager)
        self.layout_engine, self.node_names = create_standard_tui_layout(self.terminal_manager)
        self.event_system = AsyncEventSystem()
        
        self.running = False
        self.message_count = 0
        
    async def initialize(self) -> bool:
        """Initialize all systems."""
        try:
            # Initialize display renderer
            if not self.display_renderer.initialize():
                logger.error("Failed to initialize display renderer")
                return False
            
            # Set theme scheme
            preferred_scheme = detect_preferred_scheme()
            self.theme_manager.set_color_scheme(preferred_scheme)
            
            # Start event system
            await self.event_system.start()
            
            logger.info(f"TUI Demo initialized with theme: {preferred_scheme}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TUI demo: {e}")
            return False
    
    async def cleanup(self):
        """Clean up all systems."""
        self.running = False
        
        try:
            await self.event_system.stop()
            self.display_renderer.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def setup_layout(self):
        """Set up the layout and register display regions."""
        # Compute layout
        self.layout_engine.compute_layout(force=True)
        
        # Register display regions based on layout
        for node_name in self.node_names.values():
            rect = self.layout_engine.get_node_content_rect(node_name)
            if rect:
                self.display_renderer.define_region(
                    node_name, rect.x, rect.y, rect.width, rect.height
                )
    
    def update_status_bar(self):
        """Update the status bar content."""
        caps = self.terminal_manager.detect_capabilities()
        theme_caps = self.theme_manager.get_capabilities()
        
        status_text = (
            f"Terminal: {caps.width}x{caps.height} | "
            f"Colors: {theme_caps['color_mode']} | "
            f"Messages: {self.message_count} | "
            f"Press 'q' to quit"
        )
        
        # Colorize status bar
        colored_status = self.theme_manager.colorize(
            status_text, 
            self.theme_manager.get_themed_color('foreground'),
            self.theme_manager.get_themed_color('muted')
        )
        
        self.display_renderer.update_region('status_bar', colored_status)
    
    def update_chat_history(self):
        """Update the chat history content."""
        messages = []
        
        # Add some demo messages
        messages.append(
            self.theme_manager.colorize("System: ", self.theme_manager.get_themed_color('info')) +
            "TUI Demo initialized successfully"
        )
        
        messages.append(
            self.theme_manager.colorize("User: ", self.theme_manager.get_themed_color('accent')) +
            "Hello, this is a test message"
        )
        
        messages.append(
            self.theme_manager.colorize("Assistant: ", self.theme_manager.get_themed_color('success')) +
            "This demonstrates clean terminal rendering without scrollback pollution."
        )
        
        if self.message_count > 0:
            for i in range(min(self.message_count, 5)):
                messages.append(
                    self.theme_manager.colorize(f"Demo {i+1}: ", self.theme_manager.get_themed_color('warning')) +
                    f"This is demo message number {i+1}"
                )
        
        # Join messages
        chat_content = "\n".join(messages)
        
        # Update the display region
        self.display_renderer.update_region('chat_history', chat_content)
    
    def update_input_field(self):
        """Update the input field content."""
        # Create a simple input field representation
        input_prompt = self.theme_manager.colorize(">>> ", self.theme_manager.get_themed_color('accent'))
        input_text = "Type your message here..."
        
        # Draw a simple border around the input
        rect = self.layout_engine.get_node_content_rect('input_field')
        if rect:
            border_lines = self.theme_manager.draw_box(
                rect.width, rect.height,
                title="Input",
                border_color=self.theme_manager.get_themed_color('border'),
                title_color=self.theme_manager.get_themed_color('accent')
            )
            
            # Add prompt inside the border
            if len(border_lines) > 2:
                border_lines[1] = (
                    border_lines[1][:2] + 
                    input_prompt + input_text[:rect.width-4] + 
                    border_lines[1][len(input_prompt + input_text) + 2:]
                )
            
            input_content = "\n".join(border_lines)
            self.display_renderer.update_region('input_field', input_content)
    
    def update_input_status(self):
        """Update the input status line."""
        status_text = "[Ctrl+C to quit] [Enter to send] [Demo mode - no actual input]"
        colored_status = self.theme_manager.colorize(
            status_text,
            self.theme_manager.get_themed_color('muted')
        )
        
        self.display_renderer.update_region('input_status', colored_status)
    
    def update_all_regions(self):
        """Update all display regions."""
        self.update_status_bar()
        self.update_chat_history() 
        self.update_input_field()
        self.update_input_status()
        
        # Render all regions
        self.display_renderer.render_all()
    
    def handle_resize(self):
        """Handle terminal resize."""
        self.layout_engine.handle_resize()
        self.display_renderer.handle_resize()
        self.setup_layout()
        self.update_all_regions()
    
    async def run_demo_loop(self):
        """Run the main demo loop."""
        self.running = True
        
        # Initial setup
        self.setup_layout()
        self.update_all_regions()
        
        print(f"\n{self.theme_manager.colorize('TUI v2 Systems Demo', self.theme_manager.get_themed_color('accent'))}")
        print("This demonstrates the new display and rendering systems.")
        print("Key features:")
        print("- Clean terminal output without scrollback pollution")
        print("- Responsive layout engine")
        print("- Color themes with accessibility support")
        print("- Direct cursor control for in-place updates")
        print("\nPress Ctrl+C to exit the demo.\n")
        
        try:
            # Demo loop with periodic updates
            counter = 0
            while self.running:
                await asyncio.sleep(2.0)  # Update every 2 seconds
                
                counter += 1
                self.message_count = counter
                
                # Update content
                self.update_all_regions()
                
                # Simulate resize every 10 iterations
                if counter % 10 == 0:
                    logger.info(f"Demo iteration {counter}, simulating resize check")
                    self.handle_resize()
                
                # Stop after 30 iterations
                if counter >= 30:
                    self.running = False
                    
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        except Exception as e:
            logger.error(f"Error in demo loop: {e}")
        finally:
            await self.cleanup()
    
    async def run(self):
        """Run the complete demo."""
        try:
            if not await self.initialize():
                return False
            
            await self.run_demo_loop()
            return True
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            await self.cleanup()
            return False


async def run_integration_demo():
    """Run the integration demo."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    demo = SimpleTUIDemo()
    
    try:
        success = await demo.run()
        if success:
            print("Demo completed successfully!")
        else:
            print("Demo failed to initialize")
            return 1
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0


def main():
    """Main entry point."""
    try:
        result = asyncio.run(run_integration_demo())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\nDemo interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()