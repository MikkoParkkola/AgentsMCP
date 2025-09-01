"""
Progressive Disclosure Manager Component

This component manages the progressive disclosure of information and features
in the TUI, helping users navigate complexity by showing relevant options
at the right time.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DisclosureLevel(Enum):
    """Levels of information disclosure."""
    MINIMAL = "minimal"      # Show only essential information
    BASIC = "basic"          # Show common options and features  
    DETAILED = "detailed"    # Show detailed information and options
    EXPERT = "expert"        # Show all available features and advanced options


@dataclass 
class DisclosureItem:
    """Represents an item that can be disclosed progressively."""
    id: str
    title: str
    description: str
    level: DisclosureLevel
    content: Any
    is_visible: bool = False
    priority: int = 0  # Higher priority items shown first


class ProgressiveDisclosureManager:
    """
    Manages progressive disclosure of information and features.
    
    This component helps users navigate complex interfaces by gradually
    revealing information and options based on context and user expertise level.
    """
    
    def __init__(self):
        """Initialize the progressive disclosure manager."""
        self.disclosure_items: Dict[str, DisclosureItem] = {}
        self.current_level = DisclosureLevel.BASIC
        self.user_expertise_level = DisclosureLevel.BASIC
        self.context_stack = []
        self.is_initialized = False
        
        # Disclosure rules and patterns
        self.disclosure_rules = {}
        self.context_patterns = {}
        
        logger.debug("Progressive disclosure manager initialized")
    
    async def initialize(self, user_preferences=None) -> bool:
        """Initialize the progressive disclosure manager."""
        try:
            # Load user preferences if available
            if user_preferences:
                self.user_expertise_level = DisclosureLevel(
                    user_preferences.get('expertise_level', 'basic')
                )
                self.current_level = self.user_expertise_level
            
            # Set up default disclosure items
            await self._setup_default_items()
            
            self.is_initialized = True
            logger.info(f"Progressive disclosure manager ready (level: {self.current_level.value})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize progressive disclosure manager: {e}")
            return False
    
    async def _setup_default_items(self):
        """Set up default disclosure items for common TUI elements."""
        
        # System information items
        self.add_disclosure_item(DisclosureItem(
            id="system_status",
            title="System Status",
            description="Current system status and health",
            level=DisclosureLevel.MINIMAL,
            content="System running normally",
            priority=100
        ))
        
        self.add_disclosure_item(DisclosureItem(
            id="agent_status",
            title="Agent Status", 
            description="Status of active agents",
            level=DisclosureLevel.BASIC,
            content="Agents ready",
            priority=80
        ))
        
        self.add_disclosure_item(DisclosureItem(
            id="performance_metrics",
            title="Performance Metrics",
            description="System performance information", 
            level=DisclosureLevel.DETAILED,
            content="CPU: Normal, Memory: OK",
            priority=60
        ))
        
        self.add_disclosure_item(DisclosureItem(
            id="debug_info",
            title="Debug Information",
            description="Detailed debugging information",
            level=DisclosureLevel.EXPERT,
            content="Debug logs and internal state",
            priority=20
        ))
        
        # Feature items
        self.add_disclosure_item(DisclosureItem(
            id="basic_commands",
            title="Basic Commands",
            description="Essential commands for daily use",
            level=DisclosureLevel.MINIMAL,
            content=["help", "status", "quit"],
            priority=90
        ))
        
        self.add_disclosure_item(DisclosureItem(
            id="advanced_commands",
            title="Advanced Commands", 
            description="Advanced commands for power users",
            level=DisclosureLevel.DETAILED,
            content=["logs", "stats", "debug", "config"],
            priority=50
        ))
        
        self.add_disclosure_item(DisclosureItem(
            id="expert_commands",
            title="Expert Commands",
            description="Expert-level commands and features",
            level=DisclosureLevel.EXPERT,
            content=["trace", "profile", "internal", "experimental"],
            priority=10
        ))
    
    def add_disclosure_item(self, item: DisclosureItem):
        """Add a disclosure item to the manager."""
        self.disclosure_items[item.id] = item
        
        # Automatically show item if it's at or below current level
        item.is_visible = self._should_show_item(item)
        
        logger.debug(f"Added disclosure item: {item.id} (level: {item.level.value})")
    
    def set_disclosure_level(self, level: DisclosureLevel):
        """Set the current disclosure level."""
        old_level = self.current_level
        self.current_level = level
        
        # Update visibility for all items
        for item in self.disclosure_items.values():
            item.is_visible = self._should_show_item(item)
        
        logger.info(f"Disclosure level changed: {old_level.value} -> {level.value}")
    
    def _should_show_item(self, item: DisclosureItem) -> bool:
        """Determine if an item should be visible at the current level."""
        if not self.is_initialized:
            return False
        
        # Show item if its level is at or below current disclosure level
        levels = list(DisclosureLevel)
        item_index = levels.index(item.level)
        current_index = levels.index(self.current_level)
        
        return item_index <= current_index
    
    def get_visible_items(self) -> List[DisclosureItem]:
        """Get all visible disclosure items."""
        visible_items = [
            item for item in self.disclosure_items.values()
            if item.is_visible
        ]
        
        # Sort by priority (higher first)
        visible_items.sort(key=lambda x: x.priority, reverse=True)
        
        return visible_items
    
    async def cleanup(self):
        """Clean up the progressive disclosure manager."""
        logger.debug("Progressive disclosure manager cleanup")
        self.disclosure_items.clear()
        self.context_stack.clear()
        self.is_initialized = False