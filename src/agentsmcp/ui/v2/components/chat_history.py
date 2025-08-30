"""
Chat History Component - Message display and scrolling without terminal pollution.

This component manages the display of chat messages with efficient scrolling
and prevents polluting the terminal's scrollback buffer during normal operation.
"""

import asyncio
import logging
import sys
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque
import re
import textwrap

from ..event_system import AsyncEventSystem, Event, EventType, EventHandler
from ..display_renderer import DisplayRenderer

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Role of the message sender."""
    USER = "user"
    AI = "ai"
    SYSTEM = "system"
    ERROR = "error"


class MessageStatus(Enum):
    """Status of the message."""
    PENDING = "pending"
    COMPLETE = "complete"
    STREAMING = "streaming"
    ERROR = "error"


@dataclass
class ChatMessage:
    """Represents a single chat message."""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: MessageStatus = MessageStatus.COMPLETE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.id:
            # Generate unique ID based on timestamp and content hash
            self.id = f"{self.timestamp.timestamp():.6f}_{hash(self.content)}"


@dataclass 
class DisplayRegion:
    """Represents a display region for message rendering."""
    start_line: int
    end_line: int
    width: int
    visible: bool = True


class ChatHistory:
    """
    Chat history component with efficient message display and scrolling.
    
    Key features:
    - Maintains message history without terminal scrollback pollution
    - Efficient rendering for large message counts (10k+ messages)
    - Smooth scrolling and navigation
    - Search functionality
    - Message formatting (user vs AI vs system)
    - Performance optimizations for memory and display
    """
    
    def __init__(self,
                 event_system: AsyncEventSystem,
                 display_renderer: Optional[DisplayRenderer] = None,
                 max_messages: int = 10000,
                 max_display_height: int = 20):
        """Initialize the chat history component."""
        self.event_system = event_system
        self.display_renderer = display_renderer
        self.max_messages = max_messages
        self.max_display_height = max_display_height
        
        # Message storage using deque for efficient operations
        self.messages: deque[ChatMessage] = deque(maxlen=max_messages)
        self._message_index: Dict[str, ChatMessage] = {}
        
        # Display state
        self.display_region: Optional[DisplayRegion] = None
        self.scroll_position = 0  # 0 = showing latest messages
        self.auto_scroll = True   # Auto-scroll to bottom on new messages
        self.visible = True
        
        # Rendering state
        self._rendered_lines: List[str] = []
        self._needs_refresh = True
        self._last_render_time = datetime.now()
        
        # Search functionality
        self.search_query: str = ""
        self.search_results: List[ChatMessage] = []
        self.search_position = -1
        
        # Performance settings
        self.lazy_loading = True
        self.render_batch_size = 100
        self.max_line_length = 120
        
        # Message formatting
        self.user_prefix = "🧑 You: "
        self.ai_prefix = "🤖 AI: "
        self.system_prefix = "⚙️ System: "
        self.error_prefix = "❌ Error: "
        self.timestamp_format = "%H:%M:%S"
        
        # Callbacks
        self._callbacks: Dict[str, Callable] = {}
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the chat history component."""
        if self._initialized:
            return True
        
        try:
            # Setup display region if renderer is available
            if self.display_renderer:
                # Allocate display region for chat history
                caps = self.display_renderer.terminal_manager.detect_capabilities()
                height = min(self.max_display_height, caps.height - 5)  # Leave room for input
                
                self.display_region = DisplayRegion(
                    start_line=0,
                    end_line=height - 1,
                    width=caps.width
                )
                
                # Define the region in the display renderer
                self.display_renderer.define_region(
                    "chat_history",
                    x=0,
                    y=0,
                    width=caps.width,
                    height=height
                )
                
                logger.info(f"Chat history allocated display region: {height} lines x {caps.width} columns")
            
            # Setup event handlers
            self._setup_event_handlers()
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize chat history: {e}")
            return False
    
    def _setup_event_handlers(self):
        """Setup event handlers for the chat history."""
        # Create resize event handler
        class ResizeHandler(EventHandler):
            def __init__(self, chat_history):
                super().__init__(name="ChatHistoryResizeHandler")
                self.chat_history = chat_history
            
            async def handle_event(self, event: Event) -> bool:
                if event.event_type == EventType.RESIZE:
                    await self.chat_history._handle_terminal_resize(event.data)
                    return True
                return False
        
        self._resize_handler = ResizeHandler(self)
        self.event_system.add_handler(EventType.RESIZE, self._resize_handler)
    
    async def add_message(self, 
                         content: str,
                         role: MessageRole = MessageRole.USER,
                         status: MessageStatus = MessageStatus.COMPLETE,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new message to the history.
        
        Args:
            content: Message content
            role: Role of the message sender
            status: Status of the message
            metadata: Optional metadata
            
        Returns:
            Message ID
        """
        message = ChatMessage(
            id="",  # Will be generated in __post_init__
            role=role,
            content=content,
            status=status,
            metadata=metadata or {}
        )
        
        # Add to collections
        self.messages.append(message)
        self._message_index[message.id] = message
        
        # Auto-scroll to bottom if enabled
        if self.auto_scroll and self.scroll_position <= 5:  # Near bottom
            self.scroll_position = 0
        
        # Mark for refresh
        self._needs_refresh = True
        
        # Emit event
        await self._emit_message_added_event(message)
        
        # Trigger display update if visible
        if self.visible:
            await self._update_display()
        
        logger.debug(f"Added {role.value} message: {content[:50]}{'...' if len(content) > 50 else ''}")
        return message.id
    
    async def update_message(self, 
                           message_id: str,
                           content: Optional[str] = None,
                           status: Optional[MessageStatus] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing message.
        
        Args:
            message_id: ID of the message to update
            content: New content (if provided)
            status: New status (if provided)  
            metadata: New metadata (if provided)
            
        Returns:
            True if message was updated, False if not found
        """
        if message_id not in self._message_index:
            return False
        
        message = self._message_index[message_id]
        
        if content is not None:
            message.content = content
        if status is not None:
            message.status = status
        if metadata is not None:
            message.metadata.update(metadata)
        
        self._needs_refresh = True
        
        # Emit update event
        await self._emit_message_updated_event(message)
        
        # Update display if visible
        if self.visible:
            await self._update_display()
        
        return True
    
    async def stream_message_update(self, message_id: str, partial_content: str) -> bool:
        """
        Update a streaming message with partial content.
        
        Args:
            message_id: ID of the streaming message
            partial_content: Current partial content
            
        Returns:
            True if message was updated
        """
        return await self.update_message(
            message_id, 
            content=partial_content,
            status=MessageStatus.STREAMING
        )
    
    def scroll_up(self, lines: int = 1):
        """Scroll up in the message history."""
        self.scroll_position = min(self.scroll_position + lines, len(self.messages) - 1)
        self.auto_scroll = False
        self._needs_refresh = True
        
        if self.visible:
            asyncio.create_task(self._update_display())
    
    def scroll_down(self, lines: int = 1):
        """Scroll down in the message history."""
        self.scroll_position = max(self.scroll_position - lines, 0)
        
        # Re-enable auto-scroll if we're near the bottom
        if self.scroll_position <= 2:
            self.auto_scroll = True
        
        self._needs_refresh = True
        
        if self.visible:
            asyncio.create_task(self._update_display())
    
    def scroll_to_bottom(self):
        """Scroll to the bottom of the message history."""
        self.scroll_position = 0
        self.auto_scroll = True
        self._needs_refresh = True
        
        if self.visible:
            asyncio.create_task(self._update_display())
    
    def scroll_to_top(self):
        """Scroll to the top of the message history."""
        self.scroll_position = max(len(self.messages) - 1, 0)
        self.auto_scroll = False
        self._needs_refresh = True
        
        if self.visible:
            asyncio.create_task(self._update_display())
    
    async def search_messages(self, query: str) -> List[ChatMessage]:
        """
        Search messages by content.
        
        Args:
            query: Search query
            
        Returns:
            List of matching messages
        """
        if not query.strip():
            self.search_results = []
            return []
        
        self.search_query = query.lower()
        self.search_results = []
        
        for message in self.messages:
            if self.search_query in message.content.lower():
                self.search_results.append(message)
        
        self.search_position = len(self.search_results) - 1 if self.search_results else -1
        
        # Emit search event
        await self._emit_search_event(query, len(self.search_results))
        
        return self.search_results
    
    def navigate_search_next(self) -> Optional[ChatMessage]:
        """Navigate to next search result."""
        if not self.search_results or self.search_position < 0:
            return None
        
        self.search_position = (self.search_position + 1) % len(self.search_results)
        result = self.search_results[self.search_position]
        
        # Scroll to show this message
        self._scroll_to_message(result)
        
        return result
    
    def navigate_search_previous(self) -> Optional[ChatMessage]:
        """Navigate to previous search result."""
        if not self.search_results or self.search_position < 0:
            return None
        
        self.search_position = (self.search_position - 1) % len(self.search_results)
        result = self.search_results[self.search_position]
        
        # Scroll to show this message
        self._scroll_to_message(result)
        
        return result
    
    def _scroll_to_message(self, message: ChatMessage):
        """Scroll to show a specific message."""
        # Find message position in current view
        message_idx = None
        for i, msg in enumerate(self.messages):
            if msg.id == message.id:
                message_idx = i
                break
        
        if message_idx is not None:
            # Calculate scroll position to center this message
            visible_count = self.max_display_height if self.display_region else 10
            self.scroll_position = max(0, len(self.messages) - message_idx - visible_count // 2)
            self.auto_scroll = False
            self._needs_refresh = True
            
            if self.visible:
                asyncio.create_task(self._update_display())
    
    def clear_history(self):
        """Clear all messages from history."""
        self.messages.clear()
        self._message_index.clear()
        self.scroll_position = 0
        self.auto_scroll = True
        self._needs_refresh = True
        
        if self.visible:
            asyncio.create_task(self._update_display())
        
        logger.info("Chat history cleared")
    
    def get_messages(self, 
                    limit: Optional[int] = None,
                    role_filter: Optional[MessageRole] = None) -> List[ChatMessage]:
        """
        Get messages from history with optional filtering.
        
        Args:
            limit: Maximum number of messages to return
            role_filter: Filter by message role
            
        Returns:
            List of messages
        """
        messages = list(self.messages)
        
        if role_filter:
            messages = [msg for msg in messages if msg.role == role_filter]
        
        if limit:
            messages = messages[-limit:]  # Get most recent
        
        return messages
    
    def get_message_by_id(self, message_id: str) -> Optional[ChatMessage]:
        """Get a specific message by ID."""
        return self._message_index.get(message_id)
    
    async def _update_display(self):
        """Update the visual display of chat history."""
        if not self.visible or not self._needs_refresh:
            return
        
        try:
            # Render messages to lines
            lines = self._render_messages()
            
            # Update display via renderer or direct output
            if self.display_renderer and self.display_region:
                await self._render_via_display_renderer(lines)
            else:
                await self._render_direct_output(lines)
            
            self._needs_refresh = False
            self._last_render_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating chat history display: {e}")
    
    def _render_messages(self) -> List[str]:
        """Render messages to display lines."""
        if not self.messages:
            return ["No messages yet. Start typing to begin a conversation."]
        
        lines = []
        visible_height = self.max_display_height
        if self.display_region:
            visible_height = self.display_region.end_line - self.display_region.start_line + 1
        
        # Calculate which messages to show
        total_messages = len(self.messages)
        start_idx = max(0, total_messages - visible_height - self.scroll_position)
        end_idx = total_messages - self.scroll_position
        
        # Render visible messages
        for i in range(start_idx, min(end_idx, total_messages)):
            message = self.messages[i]
            message_lines = self._render_message(message)
            lines.extend(message_lines)
            
            # Stop if we've filled the visible area
            if len(lines) >= visible_height:
                break
        
        # Truncate to fit display height
        lines = lines[-visible_height:] if len(lines) > visible_height else lines
        
        # Pad with empty lines if needed
        while len(lines) < visible_height:
            lines.append("")
        
        return lines
    
    def _render_message(self, message: ChatMessage) -> List[str]:
        """Render a single message to display lines."""
        lines: List[str] = []
        
        # Choose prefix based on role
        if message.role == MessageRole.USER:
            prefix = self.user_prefix
        elif message.role == MessageRole.AI:
            prefix = self.ai_prefix
        elif message.role == MessageRole.SYSTEM:
            prefix = self.system_prefix
        else:  # ERROR
            prefix = self.error_prefix
        
        # Add timestamp if enabled
        if self.timestamp_format:
            timestamp = message.timestamp.strftime(self.timestamp_format)
            prefix = f"[{timestamp}] {prefix}"
        
        # Handle streaming status
        content = message.content or ""
        if message.status == MessageStatus.STREAMING:
            content += "▊"
        elif message.status == MessageStatus.PENDING:
            content = "..."

        # Markdown++ with ANSI styling
        BOLD = "\x1b[1m"
        ITALIC = "\x1b[3m"
        CYAN = "\x1b[36m"
        YELLOW = "\x1b[33m"
        MAGENTA = "\x1b[35m"
        RESET = "\x1b[0m"

        def style_inline(s: str) -> str:
            # Inline code first
            s = re.sub(r"`([^`]+)`", lambda m: f"{CYAN}{m.group(1)}{RESET}", s)
            # Bold **...**
            s = re.sub(r"\*\*(.+?)\*\*", lambda m: f"{BOLD}{m.group(1)}{RESET}", s)
            # Italic *...* or _..._
            s = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", lambda m: f"{ITALIC}{m.group(1)}{RESET}", s)
            s = re.sub(r"_(?!\s)(.+?)(?<!\s)_", lambda m: f"{ITALIC}{m.group(1)}{RESET}", s)
            return s

        avail = self.display_region.width - len(prefix) if self.display_region else max(20, self.max_line_length)
        in_code = False
        first_phys = True
        for raw in (content.split('\n') if content else [""]):
            if raw.strip().startswith("```"):
                in_code = not in_code
                continue
            # Bullets
            bullet = ""
            line = raw
            if not in_code and re.match(r"^\s*[-*]\s+", line):
                line = re.sub(r"^\s*[-*]\s+", "", line)
                bullet = f"{MAGENTA}•{RESET} "
            if in_code:
                # No rewrap in code block; color and cut
                shown = f"{CYAN}{line}{RESET}"
                phys = [shown[:avail]] if len(shown) > avail else [shown]
            else:
                # Headings
                m = re.match(r"^(\s*#+)\s*(.+)$", line)
                if m:
                    line = f"{BOLD}{YELLOW}{m.group(2)}{RESET}"
                else:
                    line = style_inline(line)
                wrap_src = f"{bullet}{line}" if bullet else line
                phys = textwrap.wrap(wrap_src, width=max(1, avail)) or [""]
            for idx, seg in enumerate(phys):
                if first_phys and idx == 0:
                    lines.append(prefix + seg)
                else:
                    lines.append(" " * len(prefix) + seg)
                first_phys = False

        return lines
    
    def _word_wrap(self, text: str, width: int) -> List[str]:
        """Word wrap text to specified width."""
        if not text or width <= 0:
            return [""]
        
        if len(text) <= width:
            return [text]
        
        lines = []
        words = text.split()
        current_line = ""
        
        for word in words:
            if not current_line:
                current_line = word
            elif len(current_line + " " + word) <= width:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [""]
    
    async def _render_via_display_renderer(self, lines: List[str]):
        """Render via the display renderer (preferred method)."""
        if not self.display_renderer or not self.display_region:
            return
        
        # Use display renderer to update the region without scrollback pollution
        region_id = "chat_history"
        
        self.display_renderer.update_region(
            region_id,
            "\n".join(lines)
        )
    
    async def _render_direct_output(self, lines: List[str]):
        """Render via direct terminal output (fallback)."""
        # This is a fallback method that outputs directly to terminal
        # In a production implementation, this should be more sophisticated
        # to avoid scrollback pollution
        
        # Clear the display area and redraw
        # This is a simplified approach - real implementation would
        # use terminal control sequences more carefully
        for line in lines:
            print(line)
    
    async def _handle_terminal_resize(self, resize_data: Dict[str, Any]):
        """Handle terminal resize events."""
        new_width = resize_data.get('width', 80)
        new_height = resize_data.get('height', 24)
        
        if self.display_region:
            # Update display region dimensions
            old_width = self.display_region.width
            self.display_region.width = new_width
            
            # Adjust height if needed
            available_height = new_height - 5  # Leave room for input
            if available_height != (self.display_region.end_line - self.display_region.start_line + 1):
                self.display_region.end_line = self.display_region.start_line + available_height - 1
            
            # Mark for refresh if width changed (affects word wrapping)
            if old_width != new_width:
                self._needs_refresh = True
                await self._update_display()
        
        logger.debug(f"Chat history adapted to terminal resize: {new_width}x{new_height}")
    
    async def _emit_message_added_event(self, message: ChatMessage):
        """Emit message added event."""
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "chat_history",
                "action": "message_added",
                "message_id": message.id,
                "role": message.role.value,
                "content_length": len(message.content)
            }
        )
        await self.event_system.emit_event(event)
        
        if "message_added" in self._callbacks:
            await self._callbacks["message_added"](message)
    
    async def _emit_message_updated_event(self, message: ChatMessage):
        """Emit message updated event."""
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "chat_history",
                "action": "message_updated", 
                "message_id": message.id,
                "status": message.status.value
            }
        )
        await self.event_system.emit_event(event)
        
        if "message_updated" in self._callbacks:
            await self._callbacks["message_updated"](message)
    
    async def _emit_search_event(self, query: str, result_count: int):
        """Emit search event."""
        event = Event(
            event_type=EventType.CUSTOM,
            data={
                "component": "chat_history",
                "action": "search",
                "query": query,
                "result_count": result_count
            }
        )
        await self.event_system.emit_event(event)
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for specific event types."""
        self._callbacks[event_type] = callback
    
    def remove_callback(self, event_type: str):
        """Remove callback for specific event types."""
        if event_type in self._callbacks:
            del self._callbacks[event_type]
    
    def set_visibility(self, visible: bool):
        """Set visibility of the chat history."""
        was_visible = self.visible
        self.visible = visible
        
        if visible and not was_visible:
            # Became visible - trigger refresh
            self._needs_refresh = True
            asyncio.create_task(self._update_display())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chat history statistics."""
        return {
            "message_count": len(self.messages),
            "visible": self.visible,
            "scroll_position": self.scroll_position,
            "auto_scroll": self.auto_scroll,
            "search_results": len(self.search_results),
            "display_height": self.max_display_height,
            "memory_usage_mb": sys.getsizeof(self.messages) / 1024 / 1024,
            "last_render": self._last_render_time.isoformat()
        }
    
    async def cleanup(self):
        """Cleanup the chat history component."""
        self.clear_history()
        self._callbacks.clear()
        
        if self.display_renderer and self.display_region:
            self.display_renderer.update_region("chat_history", "")
        
        self._initialized = False
        logger.debug("Chat history component cleaned up")


# Utility function for easy instantiation
def create_chat_history(event_system: AsyncEventSystem,
                       display_renderer: Optional[DisplayRenderer] = None,
                       max_messages: int = 10000,
                       max_display_height: int = 20) -> ChatHistory:
    """Create and return a new ChatHistory instance."""
    return ChatHistory(event_system, display_renderer, max_messages, max_display_height)
