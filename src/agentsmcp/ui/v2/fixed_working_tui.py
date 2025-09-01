"""
Fixed Working TUI - Addresses indentation bug and connects to real LLM

This provides a working TUI that:
1. Fixes the progressive indentation issue by proper cursor management
2. Connects to the real LLM client for actual conversations
3. Maintains immediate character input feedback
"""

import asyncio
import sys
import signal
import logging
import shutil
import termios
import tty
import time
from .improvement_dashboard import ImprovementDashboard

logger = logging.getLogger(__name__)


class FixedWorkingTUI:
    """Enhanced TUI with multi-line input, history, and improved UX."""
    
    def __init__(self):
        self.running = False
        self.input_buffer = ""
        self.original_settings = None
        self.llm_client = None
        self.cursor_col = 0
        self.cursor_row = 0  # For multi-line input
        
        # Multi-line input support
        self.input_lines = [""]  # Start with one empty line
        self.current_line_index = 0
        
        # Input history
        self.input_history = []
        self.history_index = -1
        self.temp_input = ""  # Store current input when navigating history
        
        # Paste detection state
        self.paste_buffer = ""
        self.paste_start_time = 0
        self.is_pasting = False
        self.paste_threshold_ms = 50   # Time threshold to detect paste operations
        self.paste_timeout_ms = 200    # Time to wait after last character for paste completion
        self.paste_task = None
        
        # Progress indicator state
        self.progress_task = None
        self.show_progress = False
        
        # Special key sequence detection
        self.escape_sequence = ""
        self.in_escape = False
        
        self._configure_tui_logging()
    
    def _configure_tui_logging(self):
        """Configure logging for TUI mode - no console output, file only."""
        try:
            import tempfile
            
            # Configure logging to file only for TUI mode
            log_file = tempfile.gettempdir() + "/agentsmcp_tui_debug.log"
            
            # Get root logger and remove all console handlers
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stderr, sys.stdout):
                    root_logger.removeHandler(handler)
            
            # Configure specific loggers that might contaminate TUI
            loggers_to_configure = [
                'agentsmcp.conversation.llm_client',
                'agentsmcp.conversation.conversation', 
                'agentsmcp.orchestration.orchestrator',
                'agentsmcp.orchestration.task_classifier',
                'agentsmcp',  # Main logger
                '',  # Root logger
            ]
            
            # Create file handler for debug logs
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            for logger_name in loggers_to_configure:
                logger_obj = logging.getLogger(logger_name)
                
                # Remove any existing console handlers
                for handler in logger_obj.handlers[:]:
                    if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stderr, sys.stdout):
                        logger_obj.removeHandler(handler)
                
                # Add file handler only
                logger_obj.addHandler(file_handler)
                
                # Set level to ERROR to suppress warnings in TUI (but allow DEBUG for this logger)
                if logger_name == __name__:
                    logger_obj.setLevel(logging.DEBUG)
                else:
                    logger_obj.setLevel(logging.ERROR)
                
                # Prevent propagation to parent loggers
                logger_obj.propagate = False
            
            # Suppress all warnings/info from requests, urllib3, etc
            logging.getLogger('requests').setLevel(logging.ERROR)
            logging.getLogger('urllib3').setLevel(logging.ERROR)
            logging.getLogger('httpx').setLevel(logging.ERROR)
            
            logger.info(f"TUI logging configured - debug logs written to {log_file}")
            
        except Exception as e:
            # Don't print to stdout/stderr in TUI mode
            pass
        
    def setup_terminal(self):
        """Setup terminal for immediate character input while preserving ANSI processing."""
        if sys.stdin.isatty():
            try:
                # Save original terminal settings
                self.original_settings = termios.tcgetattr(sys.stdin.fileno())
                # Set cbreak mode instead of raw mode to preserve ANSI escape sequence processing
                # This allows immediate character input while maintaining color/formatting support
                tty.setcbreak(sys.stdin.fileno())
                return True
            except Exception as e:
                logger.warning(f"Could not setup terminal: {e}")
                return False
        return False
    
    def restore_terminal(self):
        """Restore original terminal settings."""
        if self.original_settings and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.original_settings)
                sys.stdout.flush()
            except Exception as e:
                logger.warning(f"Could not restore terminal: {e}")
    
    def clear_screen_and_show_prompt(self):
        """Clear screen and draw header at column 0, without leaving stray column offsets.

        Note: We deliberately start each printed line with '\r' to ensure the
        cursor returns to column 0. Some terminals keep the current column when
        processing a bare '\n'. Using '\r\n' (or a leading '\r') prevents the
        progressive indentation artifact.
        """
        sys.stdout.write('\033[2J\033[H')  # Clear screen and move to top-left
        sys.stdout.write('\r🚀 AgentsMCP - Fixed Working TUI\n')
        sys.stdout.write('\r' + '─' * 50 + '\n')  # Consistent width
        sys.stdout.write('\rType your message (Ctrl+C to exit, /quit to quit):\n')
        # Do NOT print the prompt here; leave header drawn at a clean column 0.
        sys.stdout.flush()
        self.cursor_col = 0
    
    def show_prompt(self):
        """Show prompt at beginning of line."""
        sys.stdout.write('\r> ')  # Carriage return to beginning, then prompt
        sys.stdout.flush()
        self.cursor_col = 2  # After "> "
        
    def get_current_input(self):
        """Get the complete current input as a single string."""
        return '\n'.join(self.input_lines)
        
    def clear_current_input(self):
        """Clear the current input state."""
        self.input_lines = [""]
        self.current_line_index = 0
        self.input_buffer = ""
        
    def add_to_history(self, input_text):
        """Add input to history if it's not empty and not duplicate."""
        if input_text.strip() and (not self.input_history or self.input_history[-1] != input_text):
            self.input_history.append(input_text)
            # Keep history limited to reasonable size
            if len(self.input_history) > 100:
                self.input_history = self.input_history[-100:]
        self.history_index = -1  # Reset history navigation
        
    def navigate_history(self, direction):
        """Navigate through input history. direction: 1 for up, -1 for down."""
        if not self.input_history:
            return
            
        if self.history_index == -1:  # Starting navigation
            self.temp_input = self.get_current_input()
            
        if direction == 1:  # Up arrow - go back in history
            if self.history_index < len(self.input_history) - 1:
                self.history_index += 1
                self._load_history_item()
        elif direction == -1:  # Down arrow - go forward in history
            if self.history_index >= 0:
                self.history_index -= 1
                if self.history_index == -1:
                    # Restore temporary input
                    self._restore_temp_input()
                else:
                    self._load_history_item()
                    
    def _load_history_item(self):
        """Load a history item into current input."""
        if 0 <= self.history_index < len(self.input_history):
            historical_input = self.input_history[-(self.history_index + 1)]
            self._set_input_content(historical_input)
            
    def _restore_temp_input(self):
        """Restore the temporary input that was being typed."""
        self._set_input_content(self.temp_input)
        
    def _set_input_content(self, content):
        """Set the input content and refresh display."""
        self.input_lines = content.split('\n') if content else [""]
        self.current_line_index = min(len(self.input_lines) - 1, self.current_line_index)
        self.input_buffer = self.input_lines[self.current_line_index] if self.input_lines else ""
        self._refresh_input_display()
        
    def _refresh_input_display(self):
        """Refresh the input display after history navigation."""
        # Clear current line and show new content
        sys.stdout.write('\r\033[K')  # Clear line
        if len(self.input_lines) == 1:
            # Single line input
            sys.stdout.write(f'> {self.input_lines[0]}')
            self.cursor_col = 2 + len(self.input_lines[0])
        else:
            # Multi-line input
            sys.stdout.write(f'> {self.input_lines[0]}')
            for i, line in enumerate(self.input_lines[1:], 1):
                sys.stdout.write(f'\n  {line}')
            self.cursor_col = 2 + len(self.input_lines[self.current_line_index])
        sys.stdout.flush()
    
    def setup_llm_client(self):
        """Setup LLM client and conversation manager for real conversations."""
        try:
            # Set TUI mode environment variable to prevent console log contamination
            import os
            os.environ['AGENTSMCP_TUI_MODE'] = '1'
            
            # Import required components
            sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')
            from agentsmcp.conversation.llm_client import LLMClient
            from agentsmcp.conversation.conversation import ConversationManager
            
            self.llm_client = LLMClient()
            # Initialize conversation manager for proper agent delegation
            self.conversation_manager = ConversationManager()
            logger.info(f"LLM client initialized with provider: {self.llm_client.provider}, model: {self.llm_client.model}")
            return True
        except Exception as e:
            logger.error(f"Failed to setup LLM client: {e}")
            return False
    
    async def handle_input(self):
        """Enhanced keyboard input handling with multi-line support, history, and arrow keys."""
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                # Read one character (non-blocking)
                if sys.stdin.isatty():
                    char = await loop.run_in_executor(None, sys.stdin.read, 1)
                else:
                    # Fallback for non-TTY
                    line = await loop.run_in_executor(None, input, "> ")
                    await self.process_line(line)
                    continue
                
                # Handle escape sequences (arrow keys, etc.)
                if ord(char) == 27:  # ESC
                    self.in_escape = True
                    self.escape_sequence = char
                    continue
                elif self.in_escape:
                    self.escape_sequence += char
                    if self._process_escape_sequence():
                        self.in_escape = False
                        self.escape_sequence = ""
                    continue
                
                # Handle special characters
                if ord(char) == 3:  # Ctrl+C
                    break
                elif ord(char) == 13 or ord(char) == 10:  # Enter
                    await self._handle_enter_key()
                elif ord(char) == 14:  # Ctrl+N for new line (easier to detect than Shift+Enter)
                    self._handle_new_line()
                elif ord(char) == 127 or ord(char) == 8:  # Backspace
                    self._handle_backspace()
                elif ord(char) >= 32:  # Printable characters
                    current_time = time.time() * 1000
                    
                    # Check if this might be part of a paste operation
                    if self._is_paste_event(current_time):
                        self._handle_paste_character(char, current_time)
                    else:
                        self._handle_normal_character(char)
                
                sys.stdout.flush()
                
            except Exception as e:
                logger.error(f"Error in input handling: {e}")
                break
                
    def _process_escape_sequence(self):
        """Process escape sequences for arrow keys and other special keys."""
        if len(self.escape_sequence) >= 3:
            if self.escape_sequence == '\x1b[A':  # Up arrow
                self.navigate_history(1)
                return True
            elif self.escape_sequence == '\x1b[B':  # Down arrow
                self.navigate_history(-1)
                return True
            elif self.escape_sequence == '\x1b[C':  # Right arrow
                # TODO: Implement cursor movement within line
                return True
            elif self.escape_sequence == '\x1b[D':  # Left arrow
                # TODO: Implement cursor movement within line
                return True
            else:
                # Unknown escape sequence, ignore
                return True
        return False  # Need more characters
        
    async def _handle_enter_key(self):
        """Handle Enter key - could be regular enter or Shift+Enter for new line."""
        # For now, treat all Enter as submitting input
        # TODO: Detect Shift+Enter for new line
        if self.is_pasting:
            self._handle_paste_character('\n', time.time() * 1000)
        else:
            # Submit the complete input
            complete_input = self.get_current_input()
            if complete_input.strip():
                self.add_to_history(complete_input)
            
            sys.stdout.write('\n')
            sys.stdout.flush()
            
            await self.process_line(complete_input)
            self.clear_current_input()
            self.show_prompt()
            
    def _handle_new_line(self):
        """Handle Ctrl+N to add a new line in multi-line input."""
        # Add current line to input_lines and start a new line
        self.input_lines.append("")
        self.current_line_index = len(self.input_lines) - 1
        self.input_buffer = ""
        
        # Display new line with proper indentation
        sys.stdout.write('\n  ')  # New line with 2-space indent
        sys.stdout.flush()
        self.cursor_col = 2
        
    def _handle_backspace(self):
        """Handle backspace key."""
        current_line = self.input_lines[self.current_line_index]
        if current_line and self.cursor_col > 2:
            # Remove last character from current line
            self.input_lines[self.current_line_index] = current_line[:-1]
            self.input_buffer = self.input_lines[self.current_line_index]
            sys.stdout.write('\b \b')  # Move back, write space, move back
            self.cursor_col -= 1
    
    def _is_paste_event(self, current_time: float) -> bool:
        """Determine if current character is part of a paste operation."""
        if not self.is_pasting:
            # Not currently pasting, check if rapid input suggests paste
            if self.paste_start_time > 0:
                time_since_last = current_time - self.paste_start_time
                return time_since_last < self.paste_threshold_ms
            return False
        else:
            # Already pasting, continue if within timeout
            time_since_last = current_time - self.paste_start_time
            return time_since_last < self.paste_timeout_ms
    
    def _handle_paste_character(self, char: str, current_time: float):
        """Handle a character as part of a paste operation."""
        if not self.is_pasting:
            # Start new paste operation
            self.is_pasting = True
            self.paste_buffer = char
            logger.debug("Started paste operation")
        else:
            # Add to existing paste buffer
            self.paste_buffer += char
        
        self.paste_start_time = current_time
        
        # Cancel existing paste completion task
        if self.paste_task:
            self.paste_task.cancel()
        
        # Schedule paste completion check
        self.paste_task = asyncio.create_task(self._complete_paste_operation())
    
    def _handle_normal_character(self, char: str):
        """Handle a single character as normal typing."""
        # Add character to current line
        self.input_lines[self.current_line_index] += char
        self.input_buffer = self.input_lines[self.current_line_index]
        sys.stdout.write(char)  # IMMEDIATE ECHO
        self.cursor_col += 1
        
        # Update timestamp for paste detection
        self.paste_start_time = time.time() * 1000
    
    async def _complete_paste_operation(self):
        """Complete the paste operation after timeout."""
        try:
            # Wait for paste timeout
            await asyncio.sleep(self.paste_timeout_ms / 1000.0)
            
            if self.is_pasting and self.paste_buffer:
                # Handle multi-line paste
                paste_lines = self.paste_buffer.split('\n')
                
                if len(paste_lines) == 1:
                    # Single line paste - add to current line
                    self.input_lines[self.current_line_index] += paste_lines[0]
                    sys.stdout.write(paste_lines[0])
                    self.cursor_col += len(paste_lines[0])
                else:
                    # Multi-line paste - split across lines
                    # Add first part to current line
                    self.input_lines[self.current_line_index] += paste_lines[0]
                    sys.stdout.write(paste_lines[0])
                    
                    # Add middle lines as new lines
                    for line in paste_lines[1:-1]:
                        sys.stdout.write(f'\n  {line}')
                        self.input_lines.append(line)
                        
                    # Add last line
                    if paste_lines[-1]:  # Only if not empty
                        sys.stdout.write(f'\n  {paste_lines[-1]}')
                        self.input_lines.append(paste_lines[-1])
                        self.current_line_index = len(self.input_lines) - 1
                        self.cursor_col = 2 + len(paste_lines[-1])
                    else:
                        # Empty last line means paste ended with newline
                        sys.stdout.write('\n  ')
                        self.input_lines.append('')
                        self.current_line_index = len(self.input_lines) - 1
                        self.cursor_col = 2
                
                # Update input_buffer to current line
                self.input_buffer = self.input_lines[self.current_line_index]
                
                # Reset paste state
                self.paste_buffer = ""
                self.is_pasting = False
                self.paste_start_time = 0
                
                sys.stdout.flush()
                logger.info(f"Completed paste operation - {len(self.input_lines)} lines")
                
        except asyncio.CancelledError:
            # Task was cancelled, likely due to more input
            pass
    
    async def process_line(self, line: str):
        """Process a complete line of input and send to LLM."""
        line = line.strip()
        # Built-in commands
        if line.lower() == '/agents':
            try:
                from ...runtime_config import Config
                cfg = Config.load()
                sys.stdout.write('\r\nConfigured agents:\n')
                for name, ac in cfg.agents.items():
                    prov = getattr(ac.provider, 'value', str(ac.provider))
                    sys.stdout.write(f"\r- {name}: provider={prov} model={ac.model}\n")
                sys.stdout.flush()
            except Exception as e:
                sys.stdout.write(f"\rError reading config: {e}\n")
            return
        if line.lower() in ['/quit', '/exit', 'quit', 'exit']:
            self.running = False
            sys.stdout.write('\r\n👋 Goodbye!\n')
            return
        if line.lower() == '/help':
            sys.stdout.write('\r\n📚 Commands:\n')
            sys.stdout.write('\r  /help       - Show this help\n')
            sys.stdout.write('\r  /quit       - Exit TUI\n')
            sys.stdout.write('\r  /clear      - Clear conversation history\n')
            sys.stdout.write('\r  /agents     - List configured agents\n')
            sys.stdout.write('\r  /dashboard  - Show improvement dashboard\n')
            sys.stdout.write('\r  /optimize   - Manual optimization cycle\n')
            sys.stdout.write('\r  /status     - Show system status\n')
            sys.stdout.write('\r\n⌨️  Keyboard Shortcuts:\n')
            sys.stdout.write('\r  Ctrl+C      - Exit TUI\n')
            sys.stdout.write('\r  Ctrl+N      - Add new line (multi-line input)\n')
            sys.stdout.write('\r  ↑/↓         - Navigate input history\n')
            sys.stdout.write('\r  Enter       - Send message\n')
            sys.stdout.write('\r\n💬 Just type normally to chat with the LLM!\n')
            return
        if line.lower() == '/clear':
            if hasattr(self, 'conversation_manager') and self.conversation_manager:
                self.conversation_manager.llm_client.clear_history()
                sys.stdout.write('\r\n🧹 Conversation history cleared!\n')
            elif self.llm_client:
                self.llm_client.clear_history()
                sys.stdout.write('\r\n🧹 Conversation history cleared!\n')
            else:
                sys.stdout.write('\r\n⚠️  LLM client not available\n')
            return
        
        if line.lower() == '/dashboard':
            await self._show_improvement_dashboard()
            return
        
        if line.lower() == '/optimize':
            await self._trigger_manual_optimization()
            return
        
        if line.lower() == '/status':
            await self._show_system_status()
            return
        if not line:
            return
            
        # Ensure clean line break before agent response
        sys.stdout.write('\r\n')
        sys.stdout.flush()
        
        try:
            if self.llm_client:
                from ...orchestration import Orchestrator, OrchestratorConfig, OrchestratorMode
                
                cols, _ = shutil.get_terminal_size(fallback=(100, 40))
                
                # Start progress indicator
                self._start_progress_indicator()
                
                # Initialize orchestrator with better timeouts
                config = OrchestratorConfig(
                    mode=OrchestratorMode.STRICT_ISOLATION,
                    enable_smart_classification=True,
                    fallback_to_simple_response=True,
                    max_agent_wait_time_ms=120000,  # Increased to 2 minutes
                    synthesis_timeout_ms=5000       # Increased synthesis timeout
                )
                orchestrator = Orchestrator(config=config)
                
                # Process the user input through the orchestrator with working directory context
                import os
                context = {
                    "working_directory": os.getcwd(),
                    "project_root": os.getcwd(),
                    "user_initiated_from": "TUI"
                }
                response = await orchestrator.process_user_input(line, context)
                
                # Stop progress indicator
                self._stop_progress_indicator()
                
                # Display the orchestrator's single response with proper formatting
                try:
                    from .ansi_markdown_processor import render_markdown_lines
                except ImportError:
                    render_markdown_lines = None
                
                # Use full terminal width but leave some margin
                safe_width = max(20, cols - 4)
                
                # Ensure agent response starts on a fresh line
                sys.stdout.write('\n🤖 AgentsMCP:\n')
                
                if render_markdown_lines:
                    try:
                        content = response.content or 'No response generated'
                        formatted_lines = render_markdown_lines(content, width=safe_width, indent='')
                        
                        for ln in formatted_lines:
                            # Always ensure we start at column 0, then print the line
                            sys.stdout.write(f'\r{ln}\n')
                    except Exception as e:
                        logger.error(f"Markdown rendering failed: {e}")
                        # Fall back to plain text with basic formatting
                        for ln in (response.content or 'No response generated').split('\n'):
                            sys.stdout.write(f'\r{ln}\n')
                else:
                    # No markdown processor available
                    for ln in (response.content or 'No response generated').split('\n'):
                        sys.stdout.write(f'\r{ln}\n')
                
                # Add extra line break before metadata
                sys.stdout.write('\n')
                
                # Optional: Show metadata in debug mode
                if response.agents_consulted:
                    sys.stdout.write(f'\n💡 Consulted: {", ".join(response.agents_consulted)} '
                                   f'({response.response_type}, {response.processing_time_ms}ms)\n')
                
            else:
                sys.stdout.write(f"⚠️  LLM client unavailable. You said: \"{line}\"\n")
                sys.stdout.write('   Try restarting the TUI to reconnect.\n')
        except Exception as e:
            # Stop progress indicator on error
            self._stop_progress_indicator()
            logger.error(f"Error processing message: {e}")
            sys.stdout.write(f"\n❌ Error: {str(e)}\n")
            sys.stdout.write('   Please try again or use /help for commands.\n')
        
        # Always ensure we end on a fresh line and show prompt
        sys.stdout.write('\n')
        sys.stdout.flush()
        self.show_prompt()

    async def run(self):
        """Run the fixed working TUI."""
        self.running = True
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Setup LLM client
        llm_ready = self.setup_llm_client()
        if not llm_ready:
            sys.stdout.write('⚠️  Warning: LLM client failed to initialize. Chat may not work properly.\n')
            sys.stdout.write('   Continuing in demo mode...\n')
        
        # Setup terminal
        terminal_setup = self.setup_terminal()
        
        try:
            self.clear_screen_and_show_prompt()
            
            if llm_ready:
                # Show connection status on its own line at column 0
                sys.stdout.write(f"\r✅ Connected to {self.llm_client.provider} - {self.llm_client.model}\n")
            # Always show the prompt after header + status
            self.show_prompt()
            
            await self.handle_input()
        
        except KeyboardInterrupt:
            sys.stdout.write('\r\n👋 Goodbye!\n')
        
        finally:
            if terminal_setup:
                self.restore_terminal()
    
    def _start_progress_indicator(self):
        """Start showing progress indicator."""
        self.show_progress = True
        self.progress_task = asyncio.create_task(self._progress_animation())
        
    async def _progress_animation(self):
        """Show animated progress indicator."""
        progress_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        
        try:
            while self.show_progress:
                sys.stdout.write(f'\r🤖 Processing {progress_chars[i % len(progress_chars)]}')
                sys.stdout.flush()
                await asyncio.sleep(0.1)
                i += 1
        except asyncio.CancelledError:
            pass
        finally:
            # Clear the progress line
            sys.stdout.write('\r\033[K')
            sys.stdout.flush()
            
    def _stop_progress_indicator(self):
        """Stop progress indicator."""
        self.show_progress = False
        if self.progress_task:
            self.progress_task.cancel()
            self.progress_task = None

    async def _show_improvement_dashboard(self):
        """Display the improvement dashboard."""
        try:
            sys.stdout.write('\r\n📊 Improvement Dashboard\n')
            sys.stdout.write('─' * 50 + '\n')
            
            # Get orchestrator if available
            orchestrator = await self._get_orchestrator()
            if not orchestrator:
                sys.stdout.write('⚠️  Self-improvement system not available\n')
                return
                
            # Get improvement status
            status = await orchestrator.get_self_improvement_status()
            
            if not status.get('enabled', False):
                sys.stdout.write('ℹ️  Self-improvement system is disabled\n')
                sys.stdout.write('   Use /optimize to enable and run manual optimization\n')
                return
            
            # Display key metrics
            metrics = status.get('metrics', {})
            sys.stdout.write(f"Mode: {status.get('mode', 'unknown')}\n")
            sys.stdout.write(f"Tasks processed: {metrics.get('tasks_processed', 0)}\n")
            sys.stdout.write(f"Improvements applied: {metrics.get('improvements_applied', 0)}\n")
            sys.stdout.write(f"Average completion time: {metrics.get('avg_completion_time', 0):.2f}s\n")
            sys.stdout.write(f"User satisfaction: {metrics.get('user_satisfaction', 0):.1f}/5.0\n")
            
            # Show recent improvements
            recent_improvements = status.get('recent_improvements', [])
            if recent_improvements:
                sys.stdout.write('\n🔧 Recent Improvements:\n')
                for imp in recent_improvements[-3:]:  # Show last 3
                    sys.stdout.write(f"  • {imp.get('description', 'Unknown improvement')}\n")
                    sys.stdout.write(f"    Impact: {imp.get('impact', 'N/A')}\n")
            
            # Show full dashboard using the dashboard component
            dashboard = ImprovementDashboard()
            dashboard_content = await dashboard.render_dashboard(status)
            if dashboard_content:
                sys.stdout.write('\n')
                sys.stdout.write(dashboard_content)
            
        except Exception as e:
            logger.error(f"Error showing improvement dashboard: {e}")
            sys.stdout.write(f'❌ Error displaying dashboard: {str(e)}\n')
        
        sys.stdout.flush()

    async def _trigger_manual_optimization(self):
        """Trigger a manual optimization cycle."""
        try:
            sys.stdout.write('\r\n⚙️  Triggering manual optimization...\n')
            
            # Get orchestrator
            orchestrator = await self._get_orchestrator()
            if not orchestrator:
                sys.stdout.write('⚠️  Self-improvement system not available\n')
                return
                
            # Start optimization
            result = await orchestrator.trigger_manual_optimization()
            
            if result.get('success', False):
                sys.stdout.write('✅ Optimization completed successfully\n')
                
                improvements = result.get('improvements_found', [])
                if improvements:
                    sys.stdout.write(f'🔧 Applied {len(improvements)} improvements:\n')
                    for imp in improvements:
                        sys.stdout.write(f"  • {imp.get('description', 'Unknown improvement')}\n")
                        sys.stdout.write(f"    Expected impact: {imp.get('expected_impact', 'N/A')}\n")
                else:
                    sys.stdout.write('ℹ️  No optimization opportunities found at this time\n')
                    
                # Show updated metrics
                metrics = result.get('updated_metrics', {})
                if metrics:
                    sys.stdout.write(f'\n📈 Updated Performance:\n')
                    sys.stdout.write(f"  Average completion time: {metrics.get('avg_completion_time', 0):.2f}s\n")
                    sys.stdout.write(f"  User satisfaction: {metrics.get('user_satisfaction', 0):.1f}/5.0\n")
            else:
                error_msg = result.get('error', 'Unknown error occurred')
                sys.stdout.write(f'❌ Optimization failed: {error_msg}\n')
                
        except Exception as e:
            logger.error(f"Error triggering manual optimization: {e}")
            sys.stdout.write(f'❌ Error during optimization: {str(e)}\n')
        
        sys.stdout.flush()

    async def _show_system_status(self):
        """Show comprehensive system status."""
        try:
            sys.stdout.write('\r\n🔍 System Status\n')
            sys.stdout.write('─' * 50 + '\n')
            
            # LLM Client Status
            if self.llm_client:
                sys.stdout.write(f'✅ LLM Client: {self.llm_client.provider} - {self.llm_client.model}\n')
            else:
                sys.stdout.write('❌ LLM Client: Not connected\n')
                
            # Conversation Manager Status
            if hasattr(self, 'conversation_manager') and self.conversation_manager:
                sys.stdout.write('✅ Conversation Manager: Active\n')
            else:
                sys.stdout.write('❌ Conversation Manager: Not available\n')
            
            # Orchestrator and Self-Improvement Status
            orchestrator = await self._get_orchestrator()
            if orchestrator:
                sys.stdout.write('✅ Orchestrator: Available\n')
                
                # Get self-improvement status
                si_status = await orchestrator.get_self_improvement_status()
                if si_status.get('enabled', False):
                    mode = si_status.get('mode', 'unknown')
                    sys.stdout.write(f'✅ Self-Improvement: Active ({mode})\n')
                    
                    metrics = si_status.get('metrics', {})
                    sys.stdout.write(f'   Tasks processed: {metrics.get("tasks_processed", 0)}\n')
                    sys.stdout.write(f'   Improvements applied: {metrics.get("improvements_applied", 0)}\n')
                    
                    if metrics.get('last_optimization'):
                        sys.stdout.write(f'   Last optimization: {metrics["last_optimization"]}\n')
                else:
                    sys.stdout.write('⚠️  Self-Improvement: Disabled\n')
            else:
                sys.stdout.write('⚠️  Orchestrator: Not available\n')
                sys.stdout.write('⚠️  Self-Improvement: Not available\n')
            
            # Terminal and Input Status
            sys.stdout.write(f'✅ Terminal: Ready (TTY: {sys.stdin.isatty()})\n')
            sys.stdout.write(f'✅ Input History: {len(self.input_history)} entries\n')
            
            # Working Directory
            import os
            sys.stdout.write(f'📁 Working Directory: {os.getcwd()}\n')
            
        except Exception as e:
            logger.error(f"Error showing system status: {e}")
            sys.stdout.write(f'❌ Error retrieving system status: {str(e)}\n')
        
        sys.stdout.flush()

    async def _get_orchestrator(self):
        """Get orchestrator instance for self-improvement operations."""
        try:
            # Import and create orchestrator similar to how it's done in process_line
            from ...orchestration import Orchestrator, OrchestratorConfig, OrchestratorMode
            
            config = OrchestratorConfig(
                mode=OrchestratorMode.STRICT_ISOLATION,
                enable_smart_classification=True,
                fallback_to_simple_response=True,
                max_agent_wait_time_ms=120000,
                synthesis_timeout_ms=5000
            )
            
            if hasattr(self, 'conversation_manager') and self.conversation_manager:
                return Orchestrator(config=config)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error creating orchestrator: {e}")
            return None

    def __del__(self):
        """Ensure terminal is restored on cleanup."""
        if hasattr(self, 'original_settings') and self.original_settings:
            self.restore_terminal()


async def launch_fixed_working_tui():
    """Launch the fixed working TUI with real LLM connection."""
    tui = FixedWorkingTUI()
    await tui.run()
    return 0


if __name__ == "__main__":
    # Direct execution support
    asyncio.run(launch_fixed_working_tui())
