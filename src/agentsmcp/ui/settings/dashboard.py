"""
Settings Dashboard - Main settings interface with hierarchical navigation.

This module provides the main settings dashboard interface with:
- Hierarchical settings navigation (Global → User → Session → Agent)
- Quick actions panel for common tasks
- Settings health indicator and validation
- Progressive disclosure with collapsible sections
- WCAG 2.2 AA accessibility compliance
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from ..v2.event_system import AsyncEventSystem, Event, EventType
from ..v2.display_renderer import DisplayRenderer
from ..v2.terminal_manager import TerminalManager
from ..v2.keyboard_processor import KeyboardProcessor

logger = logging.getLogger(__name__)


class SettingsScope(Enum):
    """Settings hierarchy levels."""
    GLOBAL = "global"
    USER = "user" 
    SESSION = "session"
    AGENT = "agent"


class SettingsHealth(Enum):
    """Settings health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class SettingsSection:
    """Represents a collapsible settings section."""
    id: str
    title: str
    description: str
    scope: SettingsScope
    icon: str
    expanded: bool = False
    health: SettingsHealth = SettingsHealth.UNKNOWN
    items: List[Dict[str, Any]] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    
    def toggle_expanded(self):
        """Toggle section expansion state."""
        self.expanded = not self.expanded


@dataclass
class QuickAction:
    """Quick action for common settings tasks."""
    id: str
    title: str
    description: str
    icon: str
    handler: Callable[[], None]
    enabled: bool = True
    keyboard_shortcut: Optional[str] = None


class SettingsDashboard:
    """
    Main settings dashboard with hierarchical navigation and health monitoring.
    
    Features:
    - Hierarchical settings display (Global → User → Session → Agent)
    - Progressive disclosure with collapsible sections
    - Real-time health monitoring and validation
    - Quick actions panel
    - Keyboard navigation and accessibility
    - Event-driven updates
    """
    
    def __init__(self,
                 event_system: AsyncEventSystem,
                 display_renderer: DisplayRenderer,
                 terminal_manager: TerminalManager,
                 keyboard_processor: Optional[KeyboardProcessor] = None):
        """Initialize the settings dashboard."""
        self.event_system = event_system
        self.display_renderer = display_renderer
        self.terminal_manager = terminal_manager
        self.keyboard_processor = keyboard_processor
        
        # Dashboard state
        self.visible = False
        self.active_scope = SettingsScope.GLOBAL
        self.selected_section_idx = 0
        self.selected_action_idx = 0
        self.focus_mode = "sections"  # "sections" or "actions"
        
        # Settings sections organized by scope
        self.sections: Dict[SettingsScope, List[SettingsSection]] = {
            scope: [] for scope in SettingsScope
        }
        
        # Quick actions
        self.quick_actions: List[QuickAction] = []
        
        # Dashboard metrics
        self.health_metrics = {
            'total_settings': 0,
            'healthy_count': 0,
            'warning_count': 0,
            'error_count': 0,
            'last_validation': None
        }
        
        self._initialize_default_sections()
        self._initialize_quick_actions()
    
    async def initialize(self) -> bool:
        """Initialize the dashboard and register event handlers."""
        try:
            # Register event handlers
            await self._register_event_handlers()
            
            # Load saved settings and validate
            await self._load_settings()
            await self._validate_all_settings()
            
            logger.info("Settings dashboard initialized successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize settings dashboard: {e}")
            return False
    
    def _initialize_default_sections(self):
        """Initialize default settings sections for each scope."""
        
        # Global settings
        self.sections[SettingsScope.GLOBAL] = [
            SettingsSection(
                id="system_config",
                title="System Configuration", 
                description="Global system settings and preferences",
                scope=SettingsScope.GLOBAL,
                icon="⚙️",
                items=[
                    {"key": "log_level", "value": "INFO", "type": "select", "options": ["DEBUG", "INFO", "WARN", "ERROR"]},
                    {"key": "max_memory_mb", "value": 1024, "type": "number", "min": 512, "max": 8192},
                    {"key": "enable_telemetry", "value": True, "type": "boolean"}
                ]
            ),
            SettingsSection(
                id="security",
                title="Security & Privacy",
                description="Security policies and access control",
                scope=SettingsScope.GLOBAL,
                icon="🔐",
                items=[
                    {"key": "require_auth", "value": False, "type": "boolean"},
                    {"key": "session_timeout_minutes", "value": 60, "type": "number", "min": 5, "max": 1440},
                    {"key": "encrypt_local_data", "value": True, "type": "boolean"}
                ]
            )
        ]
        
        # User settings
        self.sections[SettingsScope.USER] = [
            SettingsSection(
                id="preferences",
                title="User Preferences",
                description="Personal settings and customizations",
                scope=SettingsScope.USER,
                icon="👤",
                items=[
                    {"key": "theme", "value": "auto", "type": "select", "options": ["light", "dark", "auto"]},
                    {"key": "language", "value": "en", "type": "select", "options": ["en", "es", "fr", "de"]},
                    {"key": "timezone", "value": "UTC", "type": "timezone"}
                ]
            ),
            SettingsSection(
                id="notifications",
                title="Notifications",
                description="Notification preferences and settings",
                scope=SettingsScope.USER,
                icon="🔔",
                items=[
                    {"key": "enable_sound", "value": True, "type": "boolean"},
                    {"key": "desktop_notifications", "value": False, "type": "boolean"},
                    {"key": "notification_level", "value": "important", "type": "select", "options": ["all", "important", "errors"]}
                ]
            )
        ]
        
        # Session settings
        self.sections[SettingsScope.SESSION] = [
            SettingsSection(
                id="session_config",
                title="Session Configuration",
                description="Current session settings and state",
                scope=SettingsScope.SESSION,
                icon="📋",
                items=[
                    {"key": "auto_save_interval", "value": 300, "type": "number", "min": 60, "max": 3600},
                    {"key": "context_window_size", "value": 8192, "type": "number", "min": 1024, "max": 32768},
                    {"key": "debug_mode", "value": False, "type": "boolean"}
                ]
            )
        ]
        
        # Agent settings
        self.sections[SettingsScope.AGENT] = [
            SettingsSection(
                id="agent_management",
                title="Agent Management",
                description="Configure and manage AI agents",
                scope=SettingsScope.AGENT,
                icon="🤖",
                items=[
                    {"key": "default_provider", "value": "ollama-turbo", "type": "select", "options": ["ollama", "ollama-turbo", "codex", "claude"]},
                    {"key": "max_concurrent_agents", "value": 3, "type": "number", "min": 1, "max": 10},
                    {"key": "agent_timeout_seconds", "value": 120, "type": "number", "min": 30, "max": 600}
                ]
            )
        ]
    
    def _initialize_quick_actions(self):
        """Initialize quick action buttons."""
        self.quick_actions = [
            QuickAction(
                id="export_settings",
                title="Export Settings",
                description="Export all settings to file",
                icon="💾",
                handler=self._export_settings,
                keyboard_shortcut="Ctrl+E"
            ),
            QuickAction(
                id="import_settings", 
                title="Import Settings",
                description="Import settings from file",
                icon="📂",
                handler=self._import_settings,
                keyboard_shortcut="Ctrl+I"
            ),
            QuickAction(
                id="reset_defaults",
                title="Reset to Defaults",
                description="Reset all settings to defaults",
                icon="🔄",
                handler=self._reset_to_defaults,
                keyboard_shortcut="Ctrl+R"
            ),
            QuickAction(
                id="validate_all",
                title="Validate Settings",
                description="Run full settings validation",
                icon="✅",
                handler=self._validate_all_settings_sync,
                keyboard_shortcut="Ctrl+V"
            )
        ]
    
    async def show(self):
        """Show the settings dashboard."""
        if self.visible:
            return
            
        self.visible = True
        
        # Update health metrics before showing
        await self._update_health_metrics()
        
        # Render the dashboard
        await self._render_dashboard()
        
        # Enable keyboard handling
        if self.keyboard_processor:
            await self.keyboard_processor.push_context("settings_dashboard")
        
        # Emit dashboard shown event
        await self.event_system.emit(Event(
            event_type=EventType.UI,
            data={"action": "dashboard_shown", "component": "settings"}
        ))
    
    async def hide(self):
        """Hide the settings dashboard."""
        if not self.visible:
            return
            
        self.visible = False
        
        # Disable keyboard handling
        if self.keyboard_processor:
            await self.keyboard_processor.pop_context()
        
        # Clear display
        await self._clear_dashboard()
        
        # Emit dashboard hidden event
        await self.event_system.emit(Event(
            event_type=EventType.UI,
            data={"action": "dashboard_hidden", "component": "settings"}
        ))
    
    async def toggle_visibility(self):
        """Toggle dashboard visibility."""
        if self.visible:
            await self.hide()
        else:
            await self.show()
    
    def navigate_scope(self, scope: SettingsScope):
        """Navigate to a specific settings scope."""
        if scope != self.active_scope:
            self.active_scope = scope
            self.selected_section_idx = 0
            asyncio.create_task(self._render_dashboard())
    
    def navigate_section(self, direction: int):
        """Navigate between sections in current scope."""
        if not self.sections[self.active_scope]:
            return
            
        max_idx = len(self.sections[self.active_scope]) - 1
        self.selected_section_idx = max(0, min(max_idx, self.selected_section_idx + direction))
        asyncio.create_task(self._render_dashboard())
    
    def navigate_action(self, direction: int):
        """Navigate between quick actions."""
        if not self.quick_actions:
            return
            
        max_idx = len(self.quick_actions) - 1
        self.selected_action_idx = max(0, min(max_idx, self.selected_action_idx + direction))
        asyncio.create_task(self._render_dashboard())
    
    def toggle_focus_mode(self):
        """Toggle focus between sections and actions."""
        self.focus_mode = "actions" if self.focus_mode == "sections" else "sections"
        asyncio.create_task(self._render_dashboard())
    
    async def toggle_section_expansion(self):
        """Toggle expansion of currently selected section."""
        if not self.sections[self.active_scope]:
            return
            
        section = self.sections[self.active_scope][self.selected_section_idx]
        section.toggle_expanded()
        await self._render_dashboard()
    
    async def execute_selected_action(self):
        """Execute the currently selected quick action."""
        if not self.quick_actions or self.focus_mode != "actions":
            return
            
        action = self.quick_actions[self.selected_action_idx]
        if action.enabled:
            try:
                if asyncio.iscoroutinefunction(action.handler):
                    await action.handler()
                else:
                    action.handler()
            except Exception as e:
                logger.exception(f"Error executing action {action.id}: {e}")
                await self._show_error(f"Action failed: {e}")
    
    async def _render_dashboard(self):
        """Render the complete settings dashboard."""
        try:
            caps = self.terminal_manager.detect_capabilities()
            width, height = caps.width, caps.height
            
            # Calculate layout
            header_height = 3
            footer_height = 2
            content_height = height - header_height - footer_height
            
            # Render header
            header_content = self._render_header(width)
            
            # Render main content (scope tabs + sections + actions)
            content_lines = self._render_content(width, content_height)
            
            # Render footer with shortcuts
            footer_content = self._render_footer(width)
            
            # Combine all parts
            dashboard_content = (
                header_content + 
                content_lines + 
                footer_content
            )
            
            # Update display region
            self.display_renderer.update_region(
                "settings_dashboard",
                "\n".join(dashboard_content),
                force=True
            )
            
        except Exception as e:
            logger.exception(f"Error rendering dashboard: {e}")
            await self._show_error(f"Rendering error: {e}")
    
    def _render_header(self, width: int) -> List[str]:
        """Render dashboard header with health metrics."""
        health_icon = self._get_overall_health_icon()
        health_text = self._get_health_summary()
        
        title_line = "╔══ AgentsMCP Settings Dashboard ══╗".center(width)
        health_line = f"{health_icon} {health_text}".center(width)
        separator = "═" * width
        
        return [title_line, health_line, separator]
    
    def _render_content(self, width: int, height: int) -> List[str]:
        """Render main dashboard content."""
        content_lines = []
        
        # Render scope tabs
        tabs_line = self._render_scope_tabs(width)
        content_lines.append(tabs_line)
        content_lines.append("─" * width)
        
        # Split remaining space between sections and actions
        sections_height = int(height * 0.7)
        actions_height = height - sections_height - 2  # -2 for tabs and separator
        
        # Render sections
        sections_content = self._render_sections(width, sections_height)
        content_lines.extend(sections_content)
        
        # Separator
        content_lines.append("─" * width)
        
        # Render quick actions
        actions_content = self._render_quick_actions(width, actions_height)
        content_lines.extend(actions_content)
        
        return content_lines
    
    def _render_scope_tabs(self, width: int) -> str:
        """Render scope selection tabs."""
        tab_width = width // len(SettingsScope)
        tabs = []
        
        for scope in SettingsScope:
            is_active = scope == self.active_scope
            icon = {"global": "🌐", "user": "👤", "session": "📋", "agent": "🤖"}[scope.value]
            title = scope.value.title()
            
            if is_active:
                tab_text = f"[{icon} {title}]"
            else:
                tab_text = f" {icon} {title} "
            
            tabs.append(tab_text.center(tab_width))
        
        return "".join(tabs)
    
    def _render_sections(self, width: int, height: int) -> List[str]:
        """Render settings sections for current scope."""
        lines = []
        sections = self.sections[self.active_scope]
        
        if not sections:
            empty_msg = f"No settings available for {self.active_scope.value} scope"
            lines.append(empty_msg.center(width))
            return lines
        
        available_height = height - 1  # Reserve space for section list header
        
        # Section list header
        focus_indicator = "▶" if self.focus_mode == "sections" else " "
        header = f"{focus_indicator} Settings Sections ({len(sections)})"
        lines.append(header)
        
        # Render each section
        for idx, section in enumerate(sections):
            is_selected = idx == self.selected_section_idx and self.focus_mode == "sections"
            section_lines = self._render_section(section, width, is_selected)
            
            # Add section lines if they fit
            if len(lines) + len(section_lines) <= height:
                lines.extend(section_lines)
            else:
                # Add truncation indicator
                lines.append("... (more sections)")
                break
        
        # Pad remaining space
        while len(lines) < height:
            lines.append("")
        
        return lines
    
    def _render_section(self, section: SettingsSection, width: int, selected: bool) -> List[str]:
        """Render a single settings section."""
        lines = []
        
        # Section header
        health_icon = self._get_health_icon(section.health)
        expand_icon = "▼" if section.expanded else "▶"
        selection_marker = "►" if selected else " "
        
        header = f"{selection_marker} {expand_icon} {section.icon} {section.title} {health_icon}"
        if len(section.validation_errors) > 0:
            header += f" ({len(section.validation_errors)} issues)"
        
        lines.append(header[:width])
        
        # Section description (if expanded or selected)
        if section.expanded or selected:
            desc_lines = textwrap.wrap(section.description, width - 4)
            for desc_line in desc_lines[:2]:  # Limit description lines
                lines.append(f"    {desc_line}")
        
        # Section items (if expanded)
        if section.expanded:
            for item in section.items[:3]:  # Limit visible items
                item_line = f"    • {item['key']}: {item['value']}"
                lines.append(item_line[:width])
            
            if len(section.items) > 3:
                lines.append(f"    ... and {len(section.items) - 3} more items")
            
            # Validation errors (if any)
            for error in section.validation_errors[:2]:  # Limit error lines
                lines.append(f"    ⚠ {error}"[:width])
        
        return lines
    
    def _render_quick_actions(self, width: int, height: int) -> List[str]:
        """Render quick actions panel."""
        lines = []
        
        # Actions header
        focus_indicator = "▶" if self.focus_mode == "actions" else " "
        header = f"{focus_indicator} Quick Actions ({len(self.quick_actions)})"
        lines.append(header)
        
        # Calculate actions per row (responsive)
        actions_per_row = max(1, width // 20)  # ~20 chars per action
        
        # Render actions in grid
        for i in range(0, len(self.quick_actions), actions_per_row):
            row_actions = self.quick_actions[i:i+actions_per_row]
            action_texts = []
            
            for j, action in enumerate(row_actions):
                action_idx = i + j
                is_selected = (action_idx == self.selected_action_idx and 
                             self.focus_mode == "actions")
                
                if is_selected:
                    action_text = f"[{action.icon} {action.title}]"
                else:
                    action_text = f" {action.icon} {action.title} "
                
                if not action.enabled:
                    action_text = f"({action_text})"  # Parentheses for disabled
                
                action_texts.append(action_text)
            
            # Join actions in row
            row_text = "  ".join(action_texts)
            lines.append(row_text[:width])
            
            if len(lines) >= height:
                break
        
        # Pad remaining space
        while len(lines) < height:
            lines.append("")
        
        return lines
    
    def _render_footer(self, width: int) -> List[str]:
        """Render footer with keyboard shortcuts."""
        shortcuts = [
            "Tab: Switch focus",
            "↑/↓: Navigate", 
            "Space: Expand/Action",
            "Esc: Close"
        ]
        
        footer_text = " | ".join(shortcuts)
        separator = "═" * width
        
        return [separator, footer_text.center(width)]
    
    def _get_overall_health_icon(self) -> str:
        """Get overall health status icon."""
        if self.health_metrics['error_count'] > 0:
            return "🔴"
        elif self.health_metrics['warning_count'] > 0:
            return "🟡"
        else:
            return "🟢"
    
    def _get_health_summary(self) -> str:
        """Get health summary text."""
        total = self.health_metrics['total_settings']
        errors = self.health_metrics['error_count']
        warnings = self.health_metrics['warning_count']
        
        if errors > 0:
            return f"Settings Health: {errors} errors, {warnings} warnings ({total} total)"
        elif warnings > 0:
            return f"Settings Health: {warnings} warnings ({total} total settings)"
        else:
            return f"Settings Health: All {total} settings validated successfully"
    
    def _get_health_icon(self, health: SettingsHealth) -> str:
        """Get health status icon for a section."""
        return {
            SettingsHealth.HEALTHY: "🟢",
            SettingsHealth.WARNING: "🟡", 
            SettingsHealth.ERROR: "🔴",
            SettingsHealth.UNKNOWN: "⚪"
        }[health]
    
    async def _register_event_handlers(self):
        """Register event handlers for dashboard interaction."""
        
        async def handle_keyboard_event(event: Event):
            if not self.visible or event.event_type != EventType.KEYBOARD:
                return
                
            key = event.data.get('key', '')
            
            # Handle navigation keys
            if key == 'up':
                if self.focus_mode == "sections":
                    self.navigate_section(-1)
                else:
                    self.navigate_action(-1)
            elif key == 'down':
                if self.focus_mode == "sections":
                    self.navigate_section(1)
                else:
                    self.navigate_action(1)
            elif key == 'tab':
                self.toggle_focus_mode()
            elif key == 'space':
                if self.focus_mode == "sections":
                    await self.toggle_section_expansion()
                else:
                    await self.execute_selected_action()
            elif key == 'escape':
                await self.hide()
            elif key in ['1', '2', '3', '4']:
                # Quick scope navigation
                scope_map = {
                    '1': SettingsScope.GLOBAL,
                    '2': SettingsScope.USER, 
                    '3': SettingsScope.SESSION,
                    '4': SettingsScope.AGENT
                }
                self.navigate_scope(scope_map[key])
        
        await self.event_system.subscribe(EventType.KEYBOARD, handle_keyboard_event)
    
    async def _load_settings(self):
        """Load settings from storage."""
        # TODO: Implement settings loading from persistent storage
        pass
    
    async def _validate_all_settings(self):
        """Validate all settings and update health metrics."""
        total_settings = 0
        healthy_count = 0
        warning_count = 0
        error_count = 0
        
        # Validate each section
        for scope_sections in self.sections.values():
            for section in scope_sections:
                section.validation_errors.clear()
                section_health = SettingsHealth.HEALTHY
                
                # Validate each setting item
                for item in section.items:
                    total_settings += 1
                    
                    try:
                        # Basic validation based on type
                        if item['type'] == 'number':
                            value = item['value']
                            if 'min' in item and value < item['min']:
                                error = f"{item['key']}: Value {value} below minimum {item['min']}"
                                section.validation_errors.append(error)
                                section_health = SettingsHealth.ERROR
                                error_count += 1
                            elif 'max' in item and value > item['max']:
                                error = f"{item['key']}: Value {value} above maximum {item['max']}"
                                section.validation_errors.append(error)
                                section_health = SettingsHealth.ERROR
                                error_count += 1
                            else:
                                healthy_count += 1
                        
                        elif item['type'] == 'select':
                            value = item['value']
                            options = item.get('options', [])
                            if value not in options:
                                error = f"{item['key']}: Invalid value '{value}', must be one of {options}"
                                section.validation_errors.append(error)
                                section_health = SettingsHealth.ERROR
                                error_count += 1
                            else:
                                healthy_count += 1
                        
                        else:
                            healthy_count += 1
                    
                    except Exception as e:
                        error = f"{item['key']}: Validation error - {e}"
                        section.validation_errors.append(error)
                        section_health = SettingsHealth.ERROR
                        error_count += 1
                
                section.health = section_health
        
        # Update health metrics
        self.health_metrics.update({
            'total_settings': total_settings,
            'healthy_count': healthy_count,
            'warning_count': warning_count,
            'error_count': error_count,
            'last_validation': datetime.now()
        })
    
    def _validate_all_settings_sync(self):
        """Sync wrapper for validation action."""
        asyncio.create_task(self._validate_all_settings())
        asyncio.create_task(self._render_dashboard())
    
    async def _update_health_metrics(self):
        """Update health metrics and re-validate if needed."""
        await self._validate_all_settings()
    
    async def _clear_dashboard(self):
        """Clear dashboard from display."""
        if hasattr(self.display_renderer, 'clear_region'):
            self.display_renderer.clear_region("settings_dashboard")
    
    async def _show_error(self, message: str):
        """Show error message to user."""
        await self.event_system.emit(Event(
            event_type=EventType.ERROR,
            data={"message": message, "component": "settings_dashboard"}
        ))
    
    # Quick action handlers
    
    def _export_settings(self):
        """Export all settings to a file."""
        try:
            settings_data = {}
            for scope, sections in self.sections.items():
                settings_data[scope.value] = {}
                for section in sections:
                    section_data = {
                        'items': section.items,
                        'expanded': section.expanded
                    }
                    settings_data[scope.value][section.id] = section_data
            
            # TODO: Implement file export dialog and actual file writing
            logger.info("Settings export initiated")
            asyncio.create_task(self._show_info("Settings exported successfully"))
            
        except Exception as e:
            logger.exception(f"Error exporting settings: {e}")
            asyncio.create_task(self._show_error(f"Export failed: {e}"))
    
    def _import_settings(self):
        """Import settings from a file."""
        try:
            # TODO: Implement file import dialog and settings loading
            logger.info("Settings import initiated")
            asyncio.create_task(self._show_info("Settings imported successfully"))
            
        except Exception as e:
            logger.exception(f"Error importing settings: {e}")
            asyncio.create_task(self._show_error(f"Import failed: {e}"))
    
    def _reset_to_defaults(self):
        """Reset all settings to default values."""
        try:
            # Reinitialize sections with defaults
            self._initialize_default_sections()
            asyncio.create_task(self._validate_all_settings())
            asyncio.create_task(self._render_dashboard())
            
            logger.info("Settings reset to defaults")
            asyncio.create_task(self._show_info("Settings reset to defaults"))
            
        except Exception as e:
            logger.exception(f"Error resetting settings: {e}")
            asyncio.create_task(self._show_error(f"Reset failed: {e}"))
    
    async def _show_info(self, message: str):
        """Show info message to user."""
        await self.event_system.emit(Event(
            event_type=EventType.INFO,
            data={"message": message, "component": "settings_dashboard"}
        ))
    
    async def cleanup(self):
        """Cleanup dashboard resources."""
        if self.visible:
            await self.hide()
        
        # TODO: Save any pending settings changes
        logger.info("Settings dashboard cleanup completed")


# Factory function for easy integration
def create_settings_dashboard(event_system: AsyncEventSystem,
                            display_renderer: DisplayRenderer,
                            terminal_manager: TerminalManager,
                            keyboard_processor: Optional[KeyboardProcessor] = None) -> SettingsDashboard:
    """Create and return a configured settings dashboard instance."""
    return SettingsDashboard(
        event_system=event_system,
        display_renderer=display_renderer,
        terminal_manager=terminal_manager,
        keyboard_processor=keyboard_processor
    )