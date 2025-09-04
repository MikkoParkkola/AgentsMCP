"""Streaming State Manager - Manages streaming output to prevent duplicate lines and flooding."""

import time
import sys
from typing import Optional, Set
from dataclasses import dataclass
from enum import Enum


class StreamingState(Enum):
    """States for streaming output."""
    IDLE = "idle"
    ACTIVE = "active" 
    COMPLETE = "complete"


@dataclass
class StreamingSession:
    """Tracks a streaming session with state management."""
    session_id: str
    start_time: float
    last_update_time: float
    current_content: str
    total_chunks: int
    state: StreamingState
    last_display_length: int = 0  # Track what was last displayed
    
    def should_update_display(self, new_content: str, min_update_interval: float = 0.1) -> bool:
        """Determine if display should be updated based on timing and content changes."""
        current_time = time.time()
        
        # Always update if this is the first content or significantly different
        if not self.current_content or new_content != self.current_content:
            # Always show first update
            if not self.current_content:
                return True
                
            # Always update if significant content change (>50 chars difference)
            content_diff = abs(len(new_content) - len(self.current_content))
            if content_diff > 50:
                return True
                
            # Throttle updates based on time interval for smaller changes
            if current_time - self.last_update_time < min_update_interval:
                return False
                
            return True
        
        return False


class StreamingStateManager:
    """Manages streaming state to prevent duplicate outputs and console flooding."""
    
    def __init__(self, supports_tty: bool = True):
        self.supports_tty = supports_tty
        self.current_session: Optional[StreamingSession] = None
        self.session_history: Set[str] = set()
        self._last_line_cleared = False
        
    def start_streaming_session(self, session_id: str) -> StreamingSession:
        """Start a new streaming session."""
        # Complete any existing session first
        if self.current_session and self.current_session.state == StreamingState.ACTIVE:
            self.complete_streaming_session()
            
        self.current_session = StreamingSession(
            session_id=session_id,
            start_time=time.time(),
            last_update_time=time.time(),
            current_content="",
            total_chunks=0,
            state=StreamingState.ACTIVE
        )
        
        self.session_history.add(session_id)
        self._last_line_cleared = False
        
        return self.current_session
    
    def update_streaming_content(self, content: str) -> tuple[bool, str]:
        """
        Update streaming content and return (should_display, display_content).
        
        Returns:
            tuple: (should_display: bool, display_content: str)
        """
        if not self.current_session or self.current_session.state != StreamingState.ACTIVE:
            return False, ""
            
        # Check if we should update the display
        should_update = self.current_session.should_update_display(content)
        
        if should_update:
            self.current_session.current_content = content
            self.current_session.last_update_time = time.time()
            self.current_session.total_chunks += 1
            
            # Prepare display content with length limiting
            display_content = self._prepare_display_content(content)
            
            return True, display_content
        
        return False, ""
    
    def _prepare_display_content(self, content: str) -> str:
        """Prepare content for display with proper truncation."""
        # Limit display length to prevent line overflow
        max_display_length = 120  # Conservative limit for most terminals
        
        if len(content) <= max_display_length:
            return content
        
        # Truncate with ellipsis, ensuring we don't break mid-word
        truncated = content[:max_display_length - 3]
        
        # Try to break at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_display_length - 20:  # Only if space is reasonably close
            truncated = truncated[:last_space]
            
        return truncated + "..."
    
    def display_streaming_update(self, content: str) -> None:
        """Display a streaming update using appropriate terminal control."""
        should_display, display_content = self.update_streaming_content(content)
        
        if not should_display:
            return
            
        if self.supports_tty:
            self._display_with_terminal_control(display_content)
        else:
            self._display_without_terminal_control(display_content)
    
    def _display_with_terminal_control(self, content: str) -> None:
        """Display update using terminal control sequences for TTY environments."""
        try:
            # Clear current line and return to beginning
            sys.stdout.write('\r\033[K')  # \r = carriage return, \033[K = clear to end of line
            
            # Display the streaming indicator and content
            display_line = f"🤖 AI (streaming): {content}"
            sys.stdout.write(display_line)
            sys.stdout.flush()
            
            # Track what we displayed for cleanup
            self.current_session.last_display_length = len(display_line)
            self._last_line_cleared = False
            
        except Exception as e:
            # Fallback to simple display
            print(f"\r🤖 AI (streaming): {content}", end="", flush=True)
    
    def _display_without_terminal_control(self, content: str) -> None:
        """Display update for non-TTY environments using progress indicators."""
        # For non-TTY, show progress dots instead of full content
        if self.current_session:
            dots_count = min(self.current_session.total_chunks // 10, 20)  # Max 20 dots
            progress_indicator = "." * dots_count
            
            # Only update every few chunks to avoid spam
            if self.current_session.total_chunks % 10 == 0:
                print(f"\r🤖 AI: {progress_indicator}", end="", flush=True)
    
    def complete_streaming_session(self) -> None:
        """Complete the current streaming session and clean up display."""
        if not self.current_session:
            return
            
        self.current_session.state = StreamingState.COMPLETE
        
        # Clear the streaming line to prepare for final message
        if self.supports_tty and not self._last_line_cleared:
            try:
                sys.stdout.write('\r\033[K')  # Clear current line
                sys.stdout.flush()
                self._last_line_cleared = True
            except:
                # Fallback to newline
                print()
        elif not self.supports_tty:
            print(" [Complete]")  # Finish the progress dots
        
        self.current_session = None
    
    def force_cleanup(self) -> None:
        """Force cleanup of any active streaming session."""
        if self.current_session and not self._last_line_cleared:
            try:
                if self.supports_tty:
                    sys.stdout.write('\r\033[K')  # Clear current line
                    sys.stdout.flush()
                else:
                    print()  # Just add a newline
                self._last_line_cleared = True
            except:
                pass
        
        if self.current_session:
            self.current_session.state = StreamingState.COMPLETE
            self.current_session = None
    
    def is_streaming_active(self) -> bool:
        """Check if streaming is currently active."""
        return (self.current_session is not None and 
                self.current_session.state == StreamingState.ACTIVE)
    
    def get_streaming_stats(self) -> dict:
        """Get statistics about current streaming session."""
        if not self.current_session:
            return {"active": False}
            
        return {
            "active": self.current_session.state == StreamingState.ACTIVE,
            "session_id": self.current_session.session_id,
            "duration": time.time() - self.current_session.start_time,
            "total_chunks": self.current_session.total_chunks,
            "content_length": len(self.current_session.current_content)
        }