#!/usr/bin/env python3
"""
Integration test for InputController with TUI.

This demonstrates how to replace the hanging _input_loop() method
with the non-blocking InputController for guaranteed responsiveness.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from agentsmcp.ui.v2.reliability.input_controller import (
    InputController, InputEventType, InputEvent
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockTUIState:
    """Mock TUI state for testing."""
    def __init__(self):
        self.current_input = ""
        self.conversation_history = []
        self.running = True


class MockTUI:
    """Mock TUI interface to demonstrate InputController integration."""
    
    def __init__(self):
        self.state = MockTUIState()
        self.input_controller = None
    
    async def initialize(self):
        """Initialize the mock TUI with InputController."""
        logger.info("🚀 Initializing Mock TUI with InputController")
        
        # Create and start input controller
        self.input_controller = InputController(
            response_timeout=0.1,  # 100ms guaranteed response
            setup_timeout=1.0,     # 1s max terminal setup
            exit_timeout=1.0       # 1s max graceful exit
        )
        
        started = await self.input_controller.start()
        if started:
            logger.info("✅ InputController started successfully")
        else:
            logger.warning("⚠️ InputController fell back to simulated mode")
        
        return True
    
    async def run(self):
        """Run the mock TUI with non-blocking input."""
        try:
            logger.info("🎯 Starting TUI main loop...")
            
            # Create background tasks
            input_task = asyncio.create_task(self._handle_input_stream())
            ui_task = asyncio.create_task(self._ui_update_loop())
            
            # Wait for any task to complete (usually input_task on exit)
            done, pending = await asyncio.wait(
                [input_task, ui_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            logger.info("✅ TUI main loop completed")
            return 0
            
        except Exception as e:
            logger.error(f"❌ TUI error: {e}")
            return 1
    
    async def _handle_input_stream(self):
        """Handle input events from InputController - NO MORE HANGS!"""
        logger.info("🎧 Starting input stream processing...")
        
        try:
            # Process input events - this stream never hangs!
            async for event in self.input_controller.get_input_stream():
                logger.info(f"📥 Input event: {event.event_type.value} = '{event.data}'")
                
                # Handle different event types
                if event.event_type == InputEventType.CHARACTER:
                    # Add character to input
                    self.state.current_input += event.data
                    logger.info(f"💬 Current input: '{self.state.current_input}'")
                
                elif event.event_type == InputEventType.BACKSPACE:
                    # Remove last character
                    if self.state.current_input:
                        self.state.current_input = self.state.current_input[:-1]
                        logger.info(f"🔙 Current input: '{self.state.current_input}'")
                
                elif event.event_type == InputEventType.ENTER:
                    # Process completed input
                    if event.data == "enter":
                        # Use current input buffer
                        input_text = self.state.current_input.strip()
                    else:
                        # Direct line input (LINE mode)
                        input_text = event.data.strip()
                    
                    if input_text:
                        await self._process_input(input_text)
                        self.state.current_input = ""  # Clear input buffer
                
                elif event.event_type == InputEventType.CONTROL:
                    if event.data in ("ctrl_c", "ctrl_d"):
                        logger.info(f"🛑 Exit requested via {event.data}")
                        self.state.running = False
                        break
                
                elif event.event_type == InputEventType.HISTORY:
                    # Handle arrow key history navigation
                    logger.info(f"📜 History navigation: {event.data}")
                
                elif event.event_type == InputEventType.ESCAPE:
                    # Clear current input
                    self.state.current_input = ""
                    logger.info("🧹 Input cleared via ESC")
                
                # Check if we should exit
                if not self.state.running:
                    break
        
        except Exception as e:
            logger.error(f"❌ Input stream error: {e}")
        
        logger.info("🔚 Input stream processing ended")
    
    async def _process_input(self, input_text: str):
        """Process user input - simulates orchestrator processing."""
        logger.info(f"⚙️ Processing input: '{input_text}'")
        
        # Add to conversation history
        self.state.conversation_history.append({
            "role": "user",
            "content": input_text,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Handle special commands
        if input_text.lower() in ('quit', 'exit'):
            logger.info("👋 Exit command received")
            self.state.running = False
            return
        
        elif input_text.lower() == 'status':
            status = self.input_controller.get_status()
            response = f"📊 Status: {status['current_mode']} mode, {status['events_processed']} events processed"
            
        elif input_text.lower() == 'help':
            response = "💡 Commands: help, status, quit, exit"
            
        else:
            # Simulate processing delay (but non-blocking!)
            await asyncio.sleep(0.5)
            response = f"✨ Processed: {input_text}"
        
        # Add response to conversation
        self.state.conversation_history.append({
            "role": "assistant", 
            "content": response,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        logger.info(f"🤖 Response: {response}")
    
    async def _ui_update_loop(self):
        """UI update loop - simulates Rich Live display updates.""" 
        logger.info("🎨 Starting UI update loop...")
        
        while self.state.running:
            try:
                # Simulate UI updates (like Rich Live)
                await asyncio.sleep(1.0)  # Update every second
                
                # Show current status
                status = self.input_controller.get_status()
                logger.info(f"🔄 UI Update - Mode: {status['current_mode']}, "
                          f"Events: {status['events_processed']}, "
                          f"Input: '{self.state.current_input}'")
                
            except Exception as e:
                logger.error(f"UI update error: {e}")
                await asyncio.sleep(1.0)
        
        logger.info("🎨 UI update loop ended")
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("🧹 Cleaning up Mock TUI...")
        
        if self.input_controller:
            stopped = await self.input_controller.stop()
            logger.info(f"🛑 InputController stopped: {stopped}")


async def main():
    """Main integration test."""
    logger.info("🎬 Starting InputController Integration Test")
    
    tui = MockTUI()
    
    try:
        # Initialize
        await tui.initialize()
        
        # Run the TUI
        exit_code = await tui.run()
        
        logger.info(f"🏁 TUI completed with exit code: {exit_code}")
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("⌨️ Interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return 1
        
    finally:
        await tui.cleanup()


if __name__ == "__main__":
    # Run the integration test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)