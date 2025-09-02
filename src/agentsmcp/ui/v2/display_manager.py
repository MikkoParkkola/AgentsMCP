"""
Display Manager - Coordinate all display updates without conflicts or corruption.

Provides centralized display coordination to prevent conflicts between Rich-based
TUI systems and custom v2 TUI components. Manages region allocation, update queuing,
and conflict detection to ensure clean display rendering.

ICD Compliance:
- Inputs: layout_regions, content_updates, refresh_mode
- Outputs: display_updated, conflict_detected, performance_metrics  
- Performance: Full display refresh within 50ms; partial updates within 10ms
- Key Functions: Region management, conflict detection, update coordination
"""

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Union, Tuple
from datetime import datetime
import weakref

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .terminal_controller import get_terminal_controller, TerminalController
from .logging_isolation_manager import get_logging_isolation_manager, LoggingIsolationManager

logger = logging.getLogger(__name__)


class RefreshMode(Enum):
    """Display refresh modes."""
    FULL = "full"           # Full screen refresh
    PARTIAL = "partial"     # Partial region update
    MINIMAL = "minimal"     # Minimal change refresh
    ADAPTIVE = "adaptive"   # Choose based on change size


class RegionType(Enum):
    """Display region types."""
    HEADER = "header"
    MAIN = "main"
    SIDEBAR = "sidebar"
    FOOTER = "footer"
    OVERLAY = "overlay"
    POPUP = "popup"


@dataclass
class DisplayRegion:
    """Represents a display region."""
    region_id: str
    region_type: RegionType
    x: int
    y: int
    width: int
    height: int
    z_index: int = 0
    visible: bool = True
    owner: Optional[str] = None
    content: Any = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ContentUpdate:
    """Represents a content update request."""
    region_id: str
    content: Any
    refresh_mode: RefreshMode = RefreshMode.ADAPTIVE
    priority: int = 0  # Higher numbers = higher priority
    timestamp: datetime = field(default_factory=datetime.now)
    requester: Optional[str] = None


@dataclass
class DisplayMetrics:
    """Performance metrics for display operations."""
    full_refresh_count: int = 0
    partial_refresh_count: int = 0
    conflict_count: int = 0
    average_refresh_time_ms: float = 0.0
    last_refresh_time_ms: float = 0.0
    total_updates: int = 0
    queue_size: int = 0


class ConflictDetector:
    """Detects and resolves display conflicts."""
    
    def __init__(self):
        """Initialize conflict detector."""
        self._active_updates: Set[str] = set()
        self._region_locks: Dict[str, str] = {}  # region_id -> requester
        self._conflict_count = 0
    
    def check_conflicts(self, update: ContentUpdate, regions: Dict[str, DisplayRegion]) -> Tuple[bool, Optional[str]]:
        """
        Check for conflicts with a proposed update.
        
        Args:
            update: The content update to check
            regions: Current display regions
            
        Returns:
            Tuple of (has_conflict, conflict_reason)
        """
        region_id = update.region_id
        
        # Check if region exists
        if region_id not in regions:
            return True, f"Region '{region_id}' does not exist"
        
        region = regions[region_id]
        
        # Check if region is locked by another requester
        if region_id in self._region_locks:
            current_owner = self._region_locks[region_id]
            if current_owner != update.requester:
                return True, f"Region '{region_id}' is locked by '{current_owner}'"
        
        # Check for overlapping updates
        if region_id in self._active_updates:
            return True, f"Region '{region_id}' has pending update"
        
        # Check z-index conflicts for overlays
        if region.region_type in (RegionType.OVERLAY, RegionType.POPUP):
            for other_region in regions.values():
                if (other_region.region_id != region_id and
                    other_region.z_index == region.z_index and
                    other_region.visible and
                    self._regions_overlap(region, other_region)):
                    return True, f"Z-index conflict with region '{other_region.region_id}'"
        
        return False, None
    
    def _regions_overlap(self, region1: DisplayRegion, region2: DisplayRegion) -> bool:
        """Check if two regions overlap."""
        return not (region1.x + region1.width <= region2.x or
                   region2.x + region2.width <= region1.x or
                   region1.y + region1.height <= region2.y or
                   region2.y + region2.height <= region1.y)
    
    def acquire_region_lock(self, region_id: str, requester: str) -> bool:
        """Acquire a lock on a region."""
        if region_id in self._region_locks:
            return self._region_locks[region_id] == requester
        
        self._region_locks[region_id] = requester
        return True
    
    def release_region_lock(self, region_id: str, requester: str) -> bool:
        """Release a lock on a region."""
        if region_id in self._region_locks and self._region_locks[region_id] == requester:
            del self._region_locks[region_id]
            return True
        return False
    
    def mark_update_active(self, region_id: str) -> None:
        """Mark a region as having an active update."""
        self._active_updates.add(region_id)
    
    def mark_update_complete(self, region_id: str) -> None:
        """Mark a region update as complete."""
        self._active_updates.discard(region_id)
    
    def increment_conflict_count(self) -> None:
        """Increment conflict counter."""
        self._conflict_count += 1
    
    def get_conflict_count(self) -> int:
        """Get total conflict count."""
        return self._conflict_count


class DisplayManager:
    """
    Centralized display manager that coordinates all display updates.
    
    Prevents conflicts between Rich-based TUI and custom v2 TUI systems by
    managing regions, queuing updates, and detecting conflicts.
    """
    
    def __init__(self):
        """Initialize the display manager."""
        self._lock = asyncio.Lock()
        self._initialized = False
        
        # Core dependencies
        self._terminal_controller: Optional[TerminalController] = None
        self._logging_manager: Optional[LoggingIsolationManager] = None
        
        # Display state
        self._regions: Dict[str, DisplayRegion] = {}
        self._update_queue: List[ContentUpdate] = []
        self._active_refresh = False
        
        # Rich integration
        self._console: Optional[Console] = None
        self._layout: Optional[Layout] = None
        self._live: Optional[Live] = None
        
        # Conflict management
        self._conflict_detector = ConflictDetector()
        
        # Performance tracking
        self._metrics = DisplayMetrics()
        self._refresh_times: List[float] = []
        self._max_refresh_history = 50
        
        # Event callbacks
        self._update_callbacks: Set[Callable[[str, Any], None]] = set()
        self._conflict_callbacks: Set[Callable[[str, str], None]] = set()
        
        # Configuration
        self._max_queue_size = 100
        self._refresh_interval_ms = 16  # ~60fps
        self._conflict_timeout_ms = 1000
        
        # Background tasks
        self._refresh_task: Optional[asyncio.Task] = None
        self._queue_processor_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> bool:
        """
        Initialize the display manager.
        
        Returns:
            True if initialization successful, False otherwise
        """
        async with self._lock:
            if self._initialized:
                return True
            
            try:
                # Initialize dependencies
                self._terminal_controller = await get_terminal_controller()
                self._logging_manager = await get_logging_isolation_manager()
                
                # Initialize Rich console if available
                if RICH_AVAILABLE:
                    self._console = Console(force_terminal=True)
                    self._layout = Layout()
                    
                    # Setup basic layout structure
                    self._layout.split_column(
                        Layout(name="header", size=3),
                        Layout(name="main"),
                        Layout(name="footer", size=3)
                    )
                    
                    # Split main area
                    self._layout["main"].split_row(
                        Layout(name="content"),
                        Layout(name="sidebar", size=30)
                    )
                
                # Create default regions
                await self._create_default_regions()
                
                # Start background tasks
                self._refresh_task = asyncio.create_task(self._refresh_loop())
                self._queue_processor_task = asyncio.create_task(self._queue_processor_loop())
                
                self._initialized = True
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize display manager: {e}")
                return False
    
    async def register_region(self, region: DisplayRegion) -> bool:
        """
        Register a display region.
        
        Args:
            region: DisplayRegion to register
            
        Returns:
            True if registered successfully, False otherwise
        """
        async with self._lock:
            if not self._initialized:
                return False
            
            # Check for conflicts
            for existing_region in self._regions.values():
                if (existing_region.region_id != region.region_id and
                    existing_region.z_index == region.z_index and
                    self._conflict_detector._regions_overlap(region, existing_region)):
                    logger.warning(f"Region overlap detected: {region.region_id} with {existing_region.region_id}")
            
            self._regions[region.region_id] = region
            
            # Update Rich layout if applicable
            if RICH_AVAILABLE and self._layout and region.region_id in ["header", "main", "footer", "sidebar"]:
                try:
                    self._layout[region.region_id].update(Panel("", title=region.region_id))
                except KeyError:
                    pass  # Region may not exist in layout
            
            return True
    
    async def update_content(self, layout_regions: Dict[str, Any], 
                           content_updates: List[ContentUpdate],
                           refresh_mode: RefreshMode = RefreshMode.ADAPTIVE) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Update display content with conflict detection.
        
        Args:
            layout_regions: Region configurations
            content_updates: List of content updates to apply
            refresh_mode: How to refresh the display
            
        Returns:
            Tuple of (display_updated, conflict_detected, performance_metrics)
        """
        if not self._initialized:
            return False, False, {}
        
        start_time = time.time()
        conflicts_detected = False
        
        try:
            async with self._lock:
                # Check for conflicts first
                valid_updates = []
                for update in content_updates:
                    has_conflict, reason = self._conflict_detector.check_conflicts(update, self._regions)
                    if has_conflict:
                        conflicts_detected = True
                        self._conflict_detector.increment_conflict_count()
                        logger.warning(f"Display conflict detected: {reason}")
                        
                        # Notify conflict callbacks
                        for callback in self._conflict_callbacks:
                            try:
                                callback(update.region_id, reason)
                            except Exception:
                                pass  # Don't let callback errors break display
                    else:
                        valid_updates.append(update)
                
                # Queue valid updates
                for update in valid_updates:
                    if len(self._update_queue) < self._max_queue_size:
                        self._update_queue.append(update)
                        self._conflict_detector.mark_update_active(update.region_id)
                
                # Sort queue by priority and timestamp
                self._update_queue.sort(key=lambda u: (-u.priority, u.timestamp))
                
                # Apply updates immediately for high priority or if queue is small
                immediate_refresh = (
                    any(u.priority > 5 for u in valid_updates) or
                    len(self._update_queue) < 3 or
                    refresh_mode == RefreshMode.FULL
                )
                
                if immediate_refresh:
                    await self._process_updates()
                
                # Update metrics
                operation_time = (time.time() - start_time) * 1000
                self._refresh_times.append(operation_time)
                if len(self._refresh_times) > self._max_refresh_history:
                    self._refresh_times.pop(0)
                
                self._metrics.last_refresh_time_ms = operation_time
                self._metrics.average_refresh_time_ms = sum(self._refresh_times) / len(self._refresh_times)
                self._metrics.total_updates += len(valid_updates)
                self._metrics.conflict_count = self._conflict_detector.get_conflict_count()
                self._metrics.queue_size = len(self._update_queue)
                
                # Performance check (ICD requirement)
                if refresh_mode == RefreshMode.FULL and operation_time > 50.0:
                    logger.warning(f"Full refresh took {operation_time:.1f}ms (>50ms target)")
                elif refresh_mode == RefreshMode.PARTIAL and operation_time > 10.0:
                    logger.warning(f"Partial refresh took {operation_time:.1f}ms (>10ms target)")
                
                return len(valid_updates) > 0, conflicts_detected, {
                    'refresh_time_ms': operation_time,
                    'updates_applied': len(valid_updates),
                    'conflicts_detected': len(content_updates) - len(valid_updates),
                    'queue_size': len(self._update_queue)
                }
                
        except Exception as e:
            logger.error(f"Failed to update display content: {e}")
            return False, True, {'error': str(e)}
    
    async def _process_updates(self) -> None:
        """Process queued updates."""
        if self._active_refresh or not self._update_queue:
            return
        
        self._active_refresh = True
        
        try:
            # Process updates in priority order
            processed_regions = set()
            updates_to_remove = []
            
            for update in self._update_queue[:]:
                if update.region_id in processed_regions:
                    continue  # Skip duplicate updates for same region
                
                # Apply update
                if update.region_id in self._regions:
                    region = self._regions[update.region_id]
                    region.content = update.content
                    region.last_updated = datetime.now()
                    processed_regions.add(update.region_id)
                    
                    # Update Rich layout if applicable
                    if RICH_AVAILABLE and self._layout:
                        await self._update_rich_region(update.region_id, update.content)
                    
                    # Notify update callbacks
                    for callback in self._update_callbacks:
                        try:
                            callback(update.region_id, update.content)
                        except Exception:
                            pass  # Don't let callback errors break display
                
                updates_to_remove.append(update)
                self._conflict_detector.mark_update_complete(update.region_id)
            
            # Remove processed updates
            for update in updates_to_remove:
                self._update_queue.remove(update)
            
            # Refresh display if using Rich Live
            if RICH_AVAILABLE and self._live and self._live.is_started:
                self._live.refresh()
            
            # Update metrics
            if processed_regions:
                self._metrics.partial_refresh_count += 1
            
        finally:
            self._active_refresh = False
    
    async def _update_rich_region(self, region_id: str, content: Any) -> None:
        """Update a Rich layout region with new content."""
        if not self._layout:
            return
        
        try:
            # Convert content to Rich renderable
            if isinstance(content, str):
                renderable = Text(content)
            elif hasattr(content, '__rich__'):
                renderable = content
            else:
                renderable = Text(str(content))
            
            # Update layout region
            if region_id in ["header", "main", "footer", "sidebar", "content"]:
                self._layout[region_id].update(renderable)
                
        except Exception as e:
            logger.debug(f"Failed to update Rich region {region_id}: {e}")
    
    async def _create_default_regions(self) -> None:
        """Create default display regions."""
        if not self._terminal_controller:
            return
        
        terminal_state = await self._terminal_controller.get_terminal_state()
        if not terminal_state:
            # Fallback dimensions
            width, height = 80, 24
        else:
            width = terminal_state.size.width
            height = terminal_state.size.height
        
        # Create standard regions
        regions = [
            DisplayRegion(
                region_id="header",
                region_type=RegionType.HEADER,
                x=0, y=0,
                width=width, height=3,
                z_index=1
            ),
            DisplayRegion(
                region_id="main",
                region_type=RegionType.MAIN,
                x=0, y=3,
                width=width-30, height=height-6,
                z_index=0
            ),
            DisplayRegion(
                region_id="sidebar",
                region_type=RegionType.SIDEBAR,
                x=width-30, y=3,
                width=30, height=height-6,
                z_index=0
            ),
            DisplayRegion(
                region_id="footer",
                region_type=RegionType.FOOTER,
                x=0, y=height-3,
                width=width, height=3,
                z_index=1
            )
        ]
        
        for region in regions:
            await self.register_region(region)
    
    async def _refresh_loop(self) -> None:
        """Background refresh loop."""
        while self._initialized:
            try:
                if not self._active_refresh and self._update_queue:
                    await self._process_updates()
                
                await asyncio.sleep(self._refresh_interval_ms / 1000.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in refresh loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error
    
    async def _queue_processor_loop(self) -> None:
        """Background queue processor loop."""
        while self._initialized:
            try:
                # Check for stale updates
                current_time = datetime.now()
                stale_updates = [
                    update for update in self._update_queue
                    if (current_time - update.timestamp).total_seconds() > 5.0
                ]
                
                for update in stale_updates:
                    self._update_queue.remove(update)
                    self._conflict_detector.mark_update_complete(update.region_id)
                
                await asyncio.sleep(1.0)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1.0)
    
    @asynccontextmanager
    async def display_context(self, enable_rich_live: bool = True):
        """Context manager for display operations."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Activate logging isolation
            if self._logging_manager:
                await self._logging_manager.activate_isolation(tui_active=True)
            
            # Start Rich Live if requested and available
            if enable_rich_live and RICH_AVAILABLE and self._layout:
                self._live = Live(self._layout, refresh_per_second=60)
                self._live.start()
            
            yield self
            
        finally:
            # Cleanup Rich Live
            if self._live and self._live.is_started:
                self._live.stop()
                self._live = None
            
            # Deactivate logging isolation
            if self._logging_manager:
                await self._logging_manager.deactivate_isolation()
    
    def register_update_callback(self, callback: Callable[[str, Any], None]) -> None:
        """Register a callback for content updates."""
        self._update_callbacks.add(callback)
    
    def unregister_update_callback(self, callback: Callable[[str, Any], None]) -> None:
        """Unregister an update callback."""
        self._update_callbacks.discard(callback)
    
    def register_conflict_callback(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback for conflict detection."""
        self._conflict_callbacks.add(callback)
    
    def unregister_conflict_callback(self, callback: Callable[[str, str], None]) -> None:
        """Unregister a conflict callback."""
        self._conflict_callbacks.discard(callback)
    
    async def get_region(self, region_id: str) -> Optional[DisplayRegion]:
        """Get a display region by ID."""
        return self._regions.get(region_id)
    
    async def get_all_regions(self) -> Dict[str, DisplayRegion]:
        """Get all display regions."""
        return dict(self._regions)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get display performance metrics."""
        return {
            'metrics': self._metrics.__dict__,
            'queue_size': len(self._update_queue),
            'active_refresh': self._active_refresh,
            'region_count': len(self._regions),
            'initialized': self._initialized,
            'rich_available': RICH_AVAILABLE,
            'average_refresh_time_ms': self._metrics.average_refresh_time_ms,
            'conflict_rate': self._metrics.conflict_count / max(self._metrics.total_updates, 1)
        }
    
    async def force_refresh(self, refresh_mode: RefreshMode = RefreshMode.FULL) -> bool:
        """Force an immediate display refresh."""
        if not self._initialized:
            return False
        
        start_time = time.time()
        
        try:
            if refresh_mode == RefreshMode.FULL:
                # Full refresh - clear and redraw everything
                if RICH_AVAILABLE and self._live and self._live.is_started:
                    self._live.refresh()
                self._metrics.full_refresh_count += 1
            else:
                # Process pending updates
                await self._process_updates()
            
            operation_time = (time.time() - start_time) * 1000
            self._metrics.last_refresh_time_ms = operation_time
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to force refresh: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup the display manager."""
        self._initialized = False
        
        # Cancel background tasks
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        
        if self._queue_processor_task and not self._queue_processor_task.done():
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
        
        # Stop Rich Live
        if self._live and self._live.is_started:
            self._live.stop()
        
        # Clear state
        self._regions.clear()
        self._update_queue.clear()
        self._update_callbacks.clear()
        self._conflict_callbacks.clear()


# Singleton instance for global access
_display_manager: Optional[DisplayManager] = None


async def get_display_manager() -> DisplayManager:
    """
    Get or create the global display manager instance.
    
    Returns:
        DisplayManager instance
    """
    global _display_manager
    
    if _display_manager is None:
        _display_manager = DisplayManager()
        await _display_manager.initialize()
    
    return _display_manager


async def cleanup_display_manager() -> None:
    """
    Cleanup the global display manager.
    """
    global _display_manager
    
    if _display_manager:
        await _display_manager.cleanup()
        _display_manager = None