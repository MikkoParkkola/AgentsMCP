"""Main TUI launcher - integrates all v3 architecture components."""

import asyncio
import sys
import signal
import logging
from typing import Optional
from .terminal_capabilities import detect_terminal_capabilities
from .ui_renderer_base import ProgressiveRenderer
from .plain_cli_renderer import PlainCLIRenderer, SimpleTUIRenderer
from .console_renderer import ConsoleRenderer
from .chat_engine import ChatEngine, ChatMessage


class TUILauncher:
    """Main launcher for the progressive enhancement TUI system."""
    
    def __init__(self):
        self.capabilities = None
        self.progressive_renderer = None
        self.current_renderer = None
        self.chat_engine = None
        self.running = False
        self._cleanup_called = False  # Guard against multiple cleanup calls
        self._goodbye_shown = False   # Ensure single goodbye message
        
        # Self-improvement system components
        self.coach_integration_manager = None
        self.continuous_improvement_engine = None
        self._improvement_task = None
        
    def initialize(self) -> bool:
        """Initialize V3 TUI launcher - PHASE 2 COMPLETE: Rich renderer with proper EOF handling."""
        # Initialize TUI with minimal output
        try:
            # COMPLETE LOG SUPPRESSION: Absolutely no DEBUG/INFO messages during TUI
            # This must be done BEFORE any LLM components are imported/initialized
            conversation_loggers = [
                'agentsmcp.conversation.llm_client',
                'agentsmcp.conversation.conversation', 
                'agentsmcp.conversation.orchestrated_conversation',
                'agentsmcp.conversation.dispatcher',
                'agentsmcp.conversation.structured_processor',
                'agentsmcp.conversation',  # Parent logger as fallback
                'agentsmcp.llm',  # LLM module
                'agentsmcp',  # Root agentsmcp logger
                ''  # Root logger itself
            ]
            
            # Set to level 100 (higher than CRITICAL=50) to suppress EVERYTHING
            for logger_name in conversation_loggers:
                logger = logging.getLogger(logger_name)
                logger.setLevel(100)  # Suppress ALL messages including CRITICAL
                logger.propagate = False  # Prevent propagation
                # Remove all handlers to ensure no output
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                # Add null handler to absorb any messages
                logger.addHandler(logging.NullHandler())
                
            # Set root logger to maximum suppression
            root_logger = logging.getLogger()
            root_logger.setLevel(100)  # Higher than CRITICAL
            
            # Remove ALL handlers from root logger
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                
            # Add ONLY a null handler to completely absorb all logging
            root_logger.addHandler(logging.NullHandler())
            
            # Disable logging module entirely for maximum suppression
            logging.disable(logging.CRITICAL)  # Disable all logging below level 50
            
            # Detect terminal capabilities
            from .terminal_capabilities import detect_terminal_capabilities
            self.capabilities = detect_terminal_capabilities()
            
            # Initialize renderer system
            self.progressive_renderer = ProgressiveRenderer(self.capabilities)
            
            # Register renderers
            self.progressive_renderer.register_renderer("plain", PlainCLIRenderer, priority=10)
            
            # Always use plain text interface - Rich TUI disabled per user preference
            # Only register console renderer as fallback to plain
            self.progressive_renderer.register_renderer("console", ConsoleRenderer, priority=20)
            
            # Select renderer
            self.current_renderer = self.progressive_renderer.select_best_renderer()
            if not self.current_renderer:
                print("❌ TUI initialization failed")
                return False
            
            renderer_name = self.current_renderer.__class__.__name__
            # Renderer selected
            
            # Initialize chat engine with current working directory for history persistence
            import os
            launch_directory = os.getcwd()
            self.chat_engine = ChatEngine(launch_directory=launch_directory)
            
            # Set up callbacks
            self.chat_engine.set_callbacks(
                status_callback=self._on_status_change,
                message_callback=self._on_new_message,
                error_callback=self._on_error
            )
            
            # Initialize progress display integration
            self._setup_progress_display_integration()
            
            # Initialize continuous improvement system
            self._initialize_continuous_improvement()
            
            return True
            
        except Exception as e:
            print(f"❌ V3: Critical initialization failure: {e}")
            import traceback
            print("🔍 V3: Full initialization error traceback:")
            traceback.print_exc()
            return False
    
    def _setup_progress_display_integration(self) -> None:
        """Set up progress display integration for enhanced agent visibility."""
        try:
            # Connect task tracker's progress display to the Rich TUI renderer
            if (self.chat_engine and hasattr(self.chat_engine, 'task_tracker') and
                self.chat_engine.task_tracker and self.chat_engine.task_tracker.progress_display):
                
                # If using Rich TUI renderer, connect the progress display
                if (self.current_renderer and hasattr(self.current_renderer, 'set_progress_display')):
                    self.current_renderer.set_progress_display(self.chat_engine.task_tracker.progress_display)
                    
                    # Start Live display if it's a Rich TUI renderer
                    if hasattr(self.current_renderer, 'start_live_display'):
                        self.current_renderer.start_live_display()
                        
        except Exception as e:
            print(f"Progress display integration warning: {e}")
    
    def _initialize_continuous_improvement(self) -> None:
        """Initialize the continuous improvement system for self-improving loops."""
        try:
            # Import improvement system components
            from ...orchestration.coach_integration import initialize_coach_integration, get_integration_manager
            
            print("🔄 Initializing self-improvement system...")
            
            # Initialize coach integration asynchronously
            import asyncio
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Schedule the initialization
            self._improvement_task = asyncio.create_task(self._async_initialize_improvement())
            
            print("✅ Self-improvement system initialization scheduled")
            
        except Exception as e:
            print(f"⚠️ Self-improvement system initialization warning: {e}")
            # Continue without self-improvement - not critical for basic functionality
    
    async def _async_initialize_improvement(self) -> None:
        """Asynchronously initialize the continuous improvement system."""
        try:
            # Import improvement system components
            from ...orchestration.coach_integration import initialize_coach_integration, get_integration_manager
            
            # Initialize coach integration
            success = await initialize_coach_integration()
            if success:
                # Get the integration manager
                self.coach_integration_manager = await get_integration_manager()
                
                # Get the continuous improvement engine
                if (self.coach_integration_manager and 
                    hasattr(self.coach_integration_manager, 'continuous_improvement_engine')):
                    self.continuous_improvement_engine = self.coach_integration_manager.continuous_improvement_engine
                    
                    # Start continuous improvement
                    if self.continuous_improvement_engine:
                        await self.continuous_improvement_engine.start_continuous_improvement()
                        print("✅ Self-improvement loops are now active")
                    else:
                        print("⚠️ Continuous improvement engine not available")
                else:
                    print("⚠️ Coach integration manager not fully initialized")
            else:
                print("⚠️ Coach integration initialization failed")
                
        except Exception as e:
            print(f"⚠️ Async self-improvement initialization error: {e}")
            # Log but don't crash - self-improvement is optional
    def _on_status_change(self, status: str) -> None:
        """Handle status change from chat engine, including streaming updates and orchestration visibility."""
        if status.startswith("streaming_update:"):
            # Handle streaming response update
            content = status[17:]  # Remove "streaming_update:" prefix
            self._handle_streaming_update(content)
        elif status.startswith("thinking_step:"):
            # Handle sequential thinking step update: "thinking_step:3/10:Current thought text"
            try:
                parts = status[14:].split(":", 2)  # Remove "thinking_step:" prefix
                if len(parts) >= 2:
                    step_info = parts[0]  # "3/10"
                    thought_text = parts[1] if len(parts) > 1 else "Processing..."
                    
                    if "/" in step_info:
                        step, total = map(int, step_info.split("/"))
                        
                        # Route to enhanced renderer if available
                        if (self.current_renderer and 
                            hasattr(self.current_renderer, 'handle_sequential_thinking_update')):
                            self.current_renderer.handle_sequential_thinking_update(step, total, thought_text)
                        else:
                            # Fallback display
                            print(f"🧠 Thinking Step {step}/{total}: {thought_text}")
            except Exception as e:
                print(f"Sequential thinking display error: {e}")
        elif status.startswith("agent_activity:"):
            # Handle agent activity update: "agent_activity:coder-1:Writing unit tests:0.75"
            try:
                parts = status[15:].split(":", 3)  # Remove "agent_activity:" prefix
                if len(parts) >= 2:
                    agent_name = parts[0]
                    activity = parts[1]
                    progress = float(parts[2]) if len(parts) > 2 and parts[2] else None
                    
                    # Route to enhanced renderer if available
                    if (self.current_renderer and 
                        hasattr(self.current_renderer, 'handle_agent_activity_update')):
                        self.current_renderer.handle_agent_activity_update(agent_name, activity, progress)
                    else:
                        # Fallback display
                        if progress is not None:
                            print(f"🤖 {agent_name}: {activity} ({progress*100:.0f}%)")
                        else:
                            print(f"🤖 {agent_name}: {activity}")
            except Exception as e:
                print(f"Agent activity display error: {e}")
        elif status.startswith("thinking_complete"):
            # Handle sequential thinking completion
            if (self.current_renderer and 
                hasattr(self.current_renderer, 'complete_sequential_thinking')):
                self.current_renderer.complete_sequential_thinking()
        elif status.startswith("agent_complete:"):
            # Handle agent completion: "agent_complete:coder-1"
            agent_name = status[15:]  # Remove "agent_complete:" prefix
            if (self.current_renderer and 
                hasattr(self.current_renderer, 'clear_agent_activity')):
                self.current_renderer.clear_agent_activity(agent_name)
        else:
            # Check if streaming is active via renderer's streaming manager
            streaming_active = False
            if (self.current_renderer and 
                hasattr(self.current_renderer, 'streaming_manager') and
                self.current_renderer.streaming_manager):
                streaming_active = self.current_renderer.streaming_manager.is_streaming_active()
            
            # Suppress status messages during streaming to avoid mixing with content
            if streaming_active:
                # Special case: "Ready" means streaming/processing is done, but don't show it
                return
                
            # Detect task completion and call renderer cleanup
            if (("completed successfully" in status.lower() or 
                 "task completed" in status.lower() or 
                 status.startswith("✅")) and 
                self.current_renderer and hasattr(self.current_renderer, 'complete_task_display')):
                
                # Call the Rich TUI renderer's completion method to stop endless status updates
                try:
                    self.current_renderer.complete_task_display()
                except Exception as e:
                    print(f"Task completion error: {e}")
            
            # Only show status if not streaming
            if self.current_renderer and hasattr(self.current_renderer, 'show_status'):
                # Rich renderer - use Rich status display with enhanced formatting
                enhanced_status = self._enhance_status_with_orchestration(status)
                self.current_renderer.show_status(enhanced_status)
            elif status and status != "Ready":  # Avoid spamming "Ready" status
                # Enhanced status display for plain renderer
                enhanced_status = self._enhance_status_with_orchestration(status)
                self._display_enhanced_status(enhanced_status)
    
    def _enhance_status_with_orchestration(self, status: str) -> str:
        """Enhance status messages with orchestration visibility, agent role information, and progress bars."""
        try:
            # Check if we have task tracker and active progress to display
            progress_info = ""
            if (self.chat_engine and hasattr(self.chat_engine, 'task_tracker') and 
                self.chat_engine.task_tracker and self.chat_engine.task_tracker.progress_display):
                
                # Get compact status line from progress display
                progress_line = self.chat_engine.task_tracker.progress_display.format_status_line()
                if progress_line and progress_line.strip():
                    progress_info = f" | {progress_line}"
            
            # Detect orchestration patterns and enhance status
            if "orchestrating" in status.lower() or "coordinating" in status.lower():
                return f"🎯 Orchestrator: {status}{progress_info}"
            elif "sequential thinking" in status.lower() or "thinking step" in status.lower():
                return f"🧠 Sequential Thinking: {status}{progress_info}"
            elif "tool:" in status.lower():
                # Extract tool name and show active role
                if "mcp__" in status:
                    # MCP tool execution
                    tool_part = status.split("mcp__")[1].split("__")[0] if "mcp__" in status else "unknown"
                    return f"🛠️ Agent-{tool_part.upper()}: {status}{progress_info}"
                else:
                    return f"🛠️ Tool Agent: {status}{progress_info}"
            elif "analyzing" in status.lower() or "processing" in status.lower():
                return f"🔍 Analyst Agent: {status}{progress_info}"
            elif "generating" in status.lower() or "creating" in status.lower():
                return f"✨ Generator Agent: {status}{progress_info}"
            elif "streaming" in status.lower():
                return f"📡 Stream Manager: {status}{progress_info}"
            else:
                # Default enhancement with coordinator role
                return f"🎯 Coordinator: {status}{progress_info}"
                
        except Exception:
            # Fallback to original status
            return status
    
    def _display_enhanced_status(self, status: str) -> None:
        """Display enhanced status for plain renderer with better formatting."""
        try:
            # Parse different types of status messages for better display
            if "[" in status and "]" in status:
                # Status with timing information
                parts = status.rsplit(" [", 1)
                if len(parts) == 2:
                    message = parts[0]
                    timing = "[" + parts[1]
                    print(f"⏳ {message} {timing}")
                else:
                    print(f"⏳ {status}")
            elif status.startswith("🔍") or status.startswith("🛠️") or status.startswith("📊") or status.startswith("🎯"):
                # Already has emoji, just display
                print(f"  {status}")
            else:
                # Basic status, add icon
                print(f"⏳ {status}")
                
        except Exception:
            # Fallback to simple display
            print(f"⏳ {status}")
    
    def _on_new_message(self, message: ChatMessage) -> None:
        """Handle new message from chat engine - suppress user message echo to avoid duplication."""
        from datetime import datetime
        
        # SKIP displaying user messages to prevent echo duplication 
        # (user already sees their input when typing)
        if message.role.value == "user":
            return
        
        # Only display assistant and system messages
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        
        if self.current_renderer and hasattr(self.current_renderer, 'display_chat_message'):
            # Rich renderer - use Rich message display with timestamp
            self.current_renderer.display_chat_message(message.role.value, message.content, timestamp)
        else:
            # Plain renderer fallback with timestamp
            role_symbols = {"assistant": "🤖", "system": "ℹ️"}
            symbol = role_symbols.get(message.role.value, "❓")
            print(f"{timestamp} {symbol} {message.role.value.title()}: {message.content}")
    
    def _handle_streaming_update(self, content: str) -> None:
        """Handle streaming response update."""
        try:
            if self.current_renderer and hasattr(self.current_renderer, 'handle_streaming_update'):
                self.current_renderer.handle_streaming_update(content)
            else:
                # Fallback for renderers without streaming support
                # Clear current line and show updated content
                print(f"\r🤖 AI: {content[:100]}{'...' if len(content) > 100 else ''}", end="", flush=True)
        except Exception as e:
            print(f"\nStreaming update error: {e}")
    
    def _on_error(self, error: str) -> None:
        """Handle error from chat engine."""
        if self.current_renderer:
            self.current_renderer.show_error(error)
        else:
            print(f"❌ Error: {error}")
    
    async def run_main_loop(self) -> int:
        """Run the V3 TUI main interaction loop with proven EOF handling."""
        try:
            if not self.initialize():
                return 1
            
            # Ready to start
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Show welcome message using the renderer
            if self.current_renderer and hasattr(self.current_renderer, 'show_welcome'):
                self.current_renderer.show_welcome()
            
            # Start Live display for Rich TUI renderer (if not already started)
            if hasattr(self.current_renderer, 'start_live_display'):
                self.current_renderer.start_live_display()
            
            if self.current_renderer and hasattr(self.current_renderer, 'show_ready'):
                self.current_renderer.show_ready()
            
            self.running = True
            
            # Main interaction loop with proven EOF handling
            while self.running:
                try:
                    # Direct input handling with proper error recovery
                    user_input = self.current_renderer.handle_input()
                    
                    if user_input:
                        # Process input through chat engine instead of simple echo
                        should_continue = await self.chat_engine.process_input(user_input)
                        if not should_continue:
                            self._show_goodbye()
                            break
                    
                    # Brief pause to prevent busy waiting
                    await asyncio.sleep(0.01)
                    
                except KeyboardInterrupt:
                    print("\n⚠️ Received Ctrl+C - shutting down gracefully...")
                    self._show_goodbye()
                    break
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    break
            
            return 0
            
        except Exception as e:
            print(f"❌ V3: Fatal error in main loop: {e}")
            import traceback
            print("🔍 V3: Full main loop error traceback:")
            traceback.print_exc()
            return 1
        finally:
            self._cleanup()
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\n⚠️ Received signal {signum} - shutting down...")
            self.running = False
            
            # Force cleanup immediately on signal to prevent hanging
            try:
                self._cleanup()
            except Exception as e:
                print(f"⚠️ Signal cleanup error: {e}")
            
            # Exit after cleanup attempt
            import sys
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _show_welcome(self) -> None:
        """Show welcome message with V3 system identification."""
        renderer_name = self.current_renderer.__class__.__name__ if self.current_renderer else "Unknown"
        welcome_msg = f"""
ℹ️ 🤖 Welcome to AI Command Composer!

This TUI automatically adapts to your terminal capabilities:
• Plain text mode for maximum compatibility
• Simple TUI mode for basic terminal features  
• Rich TUI mode for full-featured experience

Type your message and press Enter to chat.
Type /help to see available commands.
Type /quit to exit.

Let's start chatting!
        """.strip()
        
        if self.current_renderer:
            self.current_renderer.show_message(welcome_msg, "info")
        else:
            print("❌ V3 TUI: No renderer available for welcome message!")
    
    def cleanup(self) -> None:
        """Public cleanup method for external use."""
        self._cleanup()
    
    def _show_goodbye(self) -> None:
        """Show a single goodbye message with proper positioning."""
        if self._goodbye_shown:
            return
        self._goodbye_shown = True
        
        # Use the renderer's goodbye method if available
        if self.current_renderer and hasattr(self.current_renderer, 'show_goodbye'):
            self.current_renderer.show_goodbye()
        else:
            # Fallback
            print("👋 Goodbye!")    
    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._cleanup_called:
            return  # Prevent multiple cleanup calls
        self._cleanup_called = True
        
        try:
            # Clean up continuous improvement system first
            if self.continuous_improvement_engine:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(self.continuous_improvement_engine.stop_continuous_improvement())
                except RuntimeError:
                    try:
                        asyncio.run(self.continuous_improvement_engine.stop_continuous_improvement())
                    except Exception:
                        pass
                        
            if self.coach_integration_manager:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(self.coach_integration_manager.shutdown())
                except RuntimeError:
                    try:
                        asyncio.run(self.coach_integration_manager.shutdown())
                    except Exception:
                        pass
            
            # Cancel improvement initialization task if still running
            if self._improvement_task and not self._improvement_task.done():
                self._improvement_task.cancel()
                        
            # Clean up chat engine
            if self.chat_engine and hasattr(self.chat_engine, 'cleanup'):
                # Run async cleanup in a safe way
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, schedule the cleanup
                    asyncio.create_task(self.chat_engine.cleanup())
                except RuntimeError:
                    # No running loop, create a new one for cleanup
                    try:
                        asyncio.run(self.chat_engine.cleanup())
                    except Exception:
                        pass  # Ignore cleanup errors if we can't run async
                        
            if self.current_renderer:
                self.current_renderer.cleanup()
            if self.progressive_renderer:
                self.progressive_renderer.cleanup()
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")


async def launch_tui() -> int:
    """Launch the V3 TUI application."""
    try:
        launcher = TUILauncher()
        result = await launcher.run_main_loop()
        return result
    except KeyboardInterrupt:
        return 0  # Clean exit on Ctrl+C
    except Exception as e:
        print(f"❌ TUI failed to start: {e}")
        return 1


def main() -> int:
    """Main entry point for the TUI."""
    try:
        return asyncio.run(launch_tui())
    except KeyboardInterrupt:
        # No goodbye here - let TUI launcher handle it
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())