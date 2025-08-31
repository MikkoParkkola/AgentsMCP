"""
Agent Manager Component - Interface for managing AI agents.

This component provides functionality for:
- Adding, modifying, and removing agents
- Agent configuration forms with real-time validation
- Agent status and capability display
- Agent lifecycle management (start/stop/configure)
- Bulk operations and agent templates
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from ..v2.event_system import AsyncEventSystem, Event, EventType
from ..v2.display_renderer import DisplayRenderer
from ..v2.terminal_manager import TerminalManager

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent operational status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"


class AgentProvider(Enum):
    """Available agent providers."""
    OLLAMA = "ollama"
    OLLAMA_TURBO = "ollama-turbo"
    CODEX = "codex"
    CLAUDE = "claude"


@dataclass
class AgentCapability:
    """Represents an agent capability."""
    name: str
    description: str
    supported: bool
    version: Optional[str] = None
    requirements: List[str] = field(default_factory=list)


@dataclass
class AgentConfiguration:
    """Agent configuration data."""
    id: str
    name: str
    provider: AgentProvider
    enabled: bool = True
    status: AgentStatus = AgentStatus.STOPPED
    config: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[AgentCapability] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    
    def update_status(self, status: AgentStatus, error_message: Optional[str] = None):
        """Update agent status."""
        self.status = status
        self.error_message = error_message
        self.last_modified = datetime.now()


@dataclass
class AgentTemplate:
    """Predefined agent configuration template."""
    id: str
    name: str
    description: str
    provider: AgentProvider
    default_config: Dict[str, Any]
    capabilities: List[str]


class AgentValidationError(Exception):
    """Agent configuration validation error."""
    pass


class AgentManager:
    """
    Agent management component with comprehensive agent lifecycle management.
    
    Features:
    - Agent CRUD operations (Create, Read, Update, Delete)
    - Real-time status monitoring and capability detection
    - Configuration validation with immediate feedback
    - Bulk operations (enable/disable/configure multiple agents)
    - Agent templates for quick setup
    - Integration with agent registry and lifecycle management
    """
    
    def __init__(self,
                 event_system: AsyncEventSystem,
                 display_renderer: DisplayRenderer,
                 terminal_manager: TerminalManager):
        """Initialize the agent manager."""
        self.event_system = event_system
        self.display_renderer = display_renderer
        self.terminal_manager = terminal_manager
        
        # Agent management state
        self.agents: Dict[str, AgentConfiguration] = {}
        self.selected_agent_id: Optional[str] = None
        self.visible = False
        self.view_mode = "list"  # "list", "detail", "edit", "create"
        
        # UI state
        self.selected_index = 0
        self.scroll_offset = 0
        self.filter_text = ""
        self.show_only_enabled = False
        
        # Agent templates
        self.templates: List[AgentTemplate] = []
        
        # Form state for agent creation/editing
        self.form_data: Dict[str, Any] = {}
        self.form_errors: Dict[str, str] = {}
        self.form_field_focus = 0
        
        # Bulk operation state
        self.selected_agents: Set[str] = set()
        self.bulk_mode = False
        
        self._initialize_templates()
    
    async def initialize(self) -> bool:
        """Initialize the agent manager and load existing agents."""
        try:
            # Register event handlers
            await self._register_event_handlers()
            
            # Load existing agents from registry
            await self._load_agents()
            
            # Start status monitoring
            await self._start_status_monitoring()
            
            logger.info("Agent manager initialized successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize agent manager: {e}")
            return False
    
    def _initialize_templates(self):
        """Initialize agent templates for common configurations."""
        self.templates = [
            AgentTemplate(
                id="ollama_coding",
                name="Ollama Coding Assistant",
                description="General-purpose coding assistant using Ollama",
                provider=AgentProvider.OLLAMA,
                default_config={
                    "model": "codegemma",
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "context_window": 8192
                },
                capabilities=["code_generation", "code_review", "debugging"]
            ),
            AgentTemplate(
                id="ollama_turbo_fast",
                name="Ollama Turbo (Fast Response)",
                description="High-speed responses with Ollama Turbo",
                provider=AgentProvider.OLLAMA_TURBO,
                default_config={
                    "model": "gpt-oss:20b",
                    "temperature": 0.5,
                    "max_tokens": 2048,
                    "streaming": True
                },
                capabilities=["fast_response", "code_generation", "general_qa"]
            ),
            AgentTemplate(
                id="codex_advanced",
                name="Codex Advanced Coding",
                description="Advanced coding tasks with Codex",
                provider=AgentProvider.CODEX,
                default_config={
                    "temperature": 0.3,
                    "max_tokens": 8192,
                    "context_window": 16384
                },
                capabilities=["advanced_coding", "architecture", "refactoring"]
            ),
            AgentTemplate(
                id="claude_analysis",
                name="Claude Analysis & Documentation",
                description="Large context analysis and documentation with Claude",
                provider=AgentProvider.CLAUDE,
                default_config={
                    "temperature": 0.4,
                    "max_tokens": 8192,
                    "context_window": 1000000
                },
                capabilities=["large_context", "analysis", "documentation", "reasoning"]
            )
        ]
    
    async def show(self):
        """Show the agent manager interface."""
        if self.visible:
            return
        
        self.visible = True
        await self._refresh_agent_status()
        await self._render_interface()
        
        # Emit manager shown event
        await self.event_system.emit(Event(
            event_type=EventType.UI,
            data={"action": "agent_manager_shown"}
        ))
    
    async def hide(self):
        """Hide the agent manager interface."""
        if not self.visible:
            return
        
        self.visible = False
        await self._clear_interface()
        
        # Emit manager hidden event
        await self.event_system.emit(Event(
            event_type=EventType.UI,
            data={"action": "agent_manager_hidden"}
        ))
    
    # Agent CRUD Operations
    
    async def create_agent(self, template_id: Optional[str] = None) -> str:
        """Create a new agent, optionally from a template."""
        # Find template if specified
        template = None
        if template_id:
            template = next((t for t in self.templates if t.id == template_id), None)
            if not template:
                raise AgentValidationError(f"Template {template_id} not found")
        
        # Generate unique agent ID
        agent_id = self._generate_agent_id()
        
        # Create agent configuration
        if template:
            agent_config = AgentConfiguration(
                id=agent_id,
                name=template.name,
                provider=template.provider,
                config=template.default_config.copy()
            )
            
            # Add capabilities from template
            for cap_name in template.capabilities:
                capability = AgentCapability(
                    name=cap_name,
                    description=f"{cap_name.replace('_', ' ').title()} capability",
                    supported=True
                )
                agent_config.capabilities.append(capability)
        else:
            agent_config = AgentConfiguration(
                id=agent_id,
                name=f"Agent {agent_id}",
                provider=AgentProvider.OLLAMA_TURBO  # Default provider
            )
        
        # Validate configuration
        await self._validate_agent_config(agent_config)
        
        # Add to agents dictionary
        self.agents[agent_id] = agent_config
        
        # Detect agent capabilities
        await self._detect_agent_capabilities(agent_id)
        
        # Emit agent created event
        await self.event_system.emit(Event(
            event_type=EventType.AGENT,
            data={
                "action": "agent_created", 
                "agent_id": agent_id,
                "agent_config": agent_config
            }
        ))
        
        logger.info(f"Created agent {agent_id} from template {template_id}")
        return agent_id
    
    async def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing agent configuration."""
        if agent_id not in self.agents:
            raise AgentValidationError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        # Apply updates
        for key, value in updates.items():
            if key == 'config':
                agent.config.update(value)
            else:
                setattr(agent, key, value)
        
        agent.last_modified = datetime.now()
        
        # Validate updated configuration
        await self._validate_agent_config(agent)
        
        # Refresh capabilities if provider changed
        if 'provider' in updates:
            await self._detect_agent_capabilities(agent_id)
        
        # Emit agent updated event
        await self.event_system.emit(Event(
            event_type=EventType.AGENT,
            data={
                "action": "agent_updated",
                "agent_id": agent_id,
                "updates": updates
            }
        ))
        
        logger.info(f"Updated agent {agent_id}")
        return True
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent."""
        if agent_id not in self.agents:
            raise AgentValidationError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        # Stop agent if running
        if agent.status == AgentStatus.RUNNING:
            await self.stop_agent(agent_id)
        
        # Remove from agents dictionary
        del self.agents[agent_id]
        
        # Clear selection if this agent was selected
        if self.selected_agent_id == agent_id:
            self.selected_agent_id = None
        
        # Remove from bulk selection
        self.selected_agents.discard(agent_id)
        
        # Emit agent deleted event
        await self.event_system.emit(Event(
            event_type=EventType.AGENT,
            data={"action": "agent_deleted", "agent_id": agent_id}
        ))
        
        logger.info(f"Deleted agent {agent_id}")
        return True
    
    # Agent Lifecycle Management
    
    async def start_agent(self, agent_id: str) -> bool:
        """Start an agent."""
        if agent_id not in self.agents:
            raise AgentValidationError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        if agent.status == AgentStatus.RUNNING:
            return True  # Already running
        
        try:
            agent.update_status(AgentStatus.STARTING)
            await self._render_interface()
            
            # TODO: Integrate with actual agent runtime
            # For now, simulate startup
            await asyncio.sleep(1)
            
            agent.update_status(AgentStatus.RUNNING)
            
            # Emit agent started event
            await self.event_system.emit(Event(
                event_type=EventType.AGENT,
                data={"action": "agent_started", "agent_id": agent_id}
            ))
            
            logger.info(f"Started agent {agent_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to start: {e}"
            agent.update_status(AgentStatus.ERROR, error_msg)
            logger.exception(f"Error starting agent {agent_id}: {e}")
            return False
        finally:
            await self._render_interface()
    
    async def stop_agent(self, agent_id: str) -> bool:
        """Stop an agent."""
        if agent_id not in self.agents:
            raise AgentValidationError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        
        if agent.status == AgentStatus.STOPPED:
            return True  # Already stopped
        
        try:
            agent.update_status(AgentStatus.STOPPING)
            await self._render_interface()
            
            # TODO: Integrate with actual agent runtime
            # For now, simulate shutdown
            await asyncio.sleep(0.5)
            
            agent.update_status(AgentStatus.STOPPED)
            
            # Emit agent stopped event
            await self.event_system.emit(Event(
                event_type=EventType.AGENT,
                data={"action": "agent_stopped", "agent_id": agent_id}
            ))
            
            logger.info(f"Stopped agent {agent_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to stop: {e}"
            agent.update_status(AgentStatus.ERROR, error_msg)
            logger.exception(f"Error stopping agent {agent_id}: {e}")
            return False
        finally:
            await self._render_interface()
    
    # Bulk Operations
    
    async def enable_agents(self, agent_ids: List[str]) -> int:
        """Enable multiple agents. Returns number of successfully enabled agents."""
        success_count = 0
        for agent_id in agent_ids:
            try:
                await self.update_agent(agent_id, {"enabled": True})
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to enable agent {agent_id}: {e}")
        
        logger.info(f"Bulk enabled {success_count}/{len(agent_ids)} agents")
        return success_count
    
    async def disable_agents(self, agent_ids: List[str]) -> int:
        """Disable multiple agents. Returns number of successfully disabled agents."""
        success_count = 0
        for agent_id in agent_ids:
            try:
                await self.update_agent(agent_id, {"enabled": False})
                # Stop if running
                if self.agents[agent_id].status == AgentStatus.RUNNING:
                    await self.stop_agent(agent_id)
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to disable agent {agent_id}: {e}")
        
        logger.info(f"Bulk disabled {success_count}/{len(agent_ids)} agents")
        return success_count
    
    async def delete_agents(self, agent_ids: List[str]) -> int:
        """Delete multiple agents. Returns number of successfully deleted agents."""
        success_count = 0
        for agent_id in agent_ids:
            try:
                await self.delete_agent(agent_id)
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete agent {agent_id}: {e}")
        
        logger.info(f"Bulk deleted {success_count}/{len(agent_ids)} agents")
        return success_count
    
    # Navigation and UI
    
    def set_view_mode(self, mode: str):
        """Set the current view mode."""
        if mode in ["list", "detail", "edit", "create"]:
            self.view_mode = mode
            asyncio.create_task(self._render_interface())
    
    def navigate_agents(self, direction: int):
        """Navigate through agent list."""
        if not self.agents:
            return
        
        agent_ids = self._get_filtered_agent_ids()
        if not agent_ids:
            return
        
        max_idx = len(agent_ids) - 1
        self.selected_index = max(0, min(max_idx, self.selected_index + direction))
        self.selected_agent_id = agent_ids[self.selected_index]
        
        asyncio.create_task(self._render_interface())
    
    def toggle_agent_selection(self):
        """Toggle selection of current agent for bulk operations."""
        if self.selected_agent_id:
            if self.selected_agent_id in self.selected_agents:
                self.selected_agents.remove(self.selected_agent_id)
            else:
                self.selected_agents.add(self.selected_agent_id)
            
            asyncio.create_task(self._render_interface())
    
    def toggle_bulk_mode(self):
        """Toggle bulk operations mode."""
        self.bulk_mode = not self.bulk_mode
        if not self.bulk_mode:
            self.selected_agents.clear()
        
        asyncio.create_task(self._render_interface())
    
    def set_filter(self, filter_text: str):
        """Set agent filter text."""
        self.filter_text = filter_text.lower()
        self.selected_index = 0
        asyncio.create_task(self._render_interface())
    
    def toggle_show_enabled_only(self):
        """Toggle showing only enabled agents."""
        self.show_only_enabled = not self.show_only_enabled
        self.selected_index = 0
        asyncio.create_task(self._render_interface())
    
    # Rendering Methods
    
    async def _render_interface(self):
        """Render the agent manager interface based on current view mode."""
        if not self.visible:
            return
        
        try:
            caps = self.terminal_manager.detect_capabilities()
            width, height = caps.width, caps.height
            
            if self.view_mode == "list":
                content = self._render_agent_list(width, height)
            elif self.view_mode == "detail":
                content = self._render_agent_detail(width, height)
            elif self.view_mode == "edit":
                content = self._render_agent_edit(width, height)
            elif self.view_mode == "create":
                content = self._render_agent_create(width, height)
            else:
                content = ["Unknown view mode"]
            
            # Update display
            self.display_renderer.update_region(
                "agent_manager",
                "\n".join(content),
                force=True
            )
            
        except Exception as e:
            logger.exception(f"Error rendering agent manager: {e}")
    
    def _render_agent_list(self, width: int, height: int) -> List[str]:
        """Render the agent list view."""
        lines = []
        
        # Header
        bulk_indicator = " [BULK]" if self.bulk_mode else ""
        title = f"╔══ Agent Manager{bulk_indicator} ══╗".center(width)
        lines.append(title)
        
        # Stats line
        total_agents = len(self.agents)
        running_count = len([a for a in self.agents.values() if a.status == AgentStatus.RUNNING])
        enabled_count = len([a for a in self.agents.values() if a.enabled])
        
        stats = f"Total: {total_agents} | Enabled: {enabled_count} | Running: {running_count}"
        if self.bulk_mode:
            stats += f" | Selected: {len(self.selected_agents)}"
        lines.append(stats.center(width))
        lines.append("═" * width)
        
        # Filter info
        filter_info = []
        if self.filter_text:
            filter_info.append(f"Filter: '{self.filter_text}'")
        if self.show_only_enabled:
            filter_info.append("Enabled only")
        
        if filter_info:
            lines.append(" | ".join(filter_info))
            lines.append("─" * width)
        
        # Agent list
        agent_ids = self._get_filtered_agent_ids()
        if not agent_ids:
            lines.append("No agents match current filter".center(width))
            return lines
        
        # Calculate visible range
        content_height = height - len(lines) - 3  # Reserve space for footer
        start_idx = max(0, self.selected_index - content_height // 2)
        end_idx = min(len(agent_ids), start_idx + content_height)
        
        # Render visible agents
        for i in range(start_idx, end_idx):
            agent_id = agent_ids[i]
            agent = self.agents[agent_id]
            
            # Selection indicators
            is_current = i == self.selected_index
            is_selected_bulk = agent_id in self.selected_agents
            
            cursor = "►" if is_current else " "
            bulk_marker = "☑" if is_selected_bulk else "☐" if self.bulk_mode else " "
            
            # Status icon
            status_icon = self._get_status_icon(agent.status)
            
            # Enabled/disabled indicator
            enabled_icon = "●" if agent.enabled else "○"
            
            # Agent line
            agent_line = f"{cursor}{bulk_marker} {status_icon} {enabled_icon} {agent.name}"
            if agent.error_message:
                agent_line += f" ⚠ {agent.error_message}"
            
            lines.append(agent_line[:width])
        
        # Footer with shortcuts
        if len(lines) < height - 2:
            lines.append("─" * width)
            shortcuts = [
                "Enter: Details",
                "E: Edit", 
                "D: Delete",
                "Space: Start/Stop",
                "B: Bulk Mode",
                "N: New"
            ]
            lines.append(" | ".join(shortcuts)[:width])
        
        return lines
    
    def _render_agent_detail(self, width: int, height: int) -> List[str]:
        """Render detailed view of selected agent."""
        lines = []
        
        if not self.selected_agent_id or self.selected_agent_id not in self.agents:
            lines.append("No agent selected".center(width))
            return lines
        
        agent = self.agents[self.selected_agent_id]
        
        # Header
        title = f"╔══ Agent Details: {agent.name} ══╗".center(width)
        lines.append(title)
        lines.append("═" * width)
        
        # Basic info
        lines.append(f"ID: {agent.id}")
        lines.append(f"Provider: {agent.provider.value}")
        lines.append(f"Status: {self._get_status_icon(agent.status)} {agent.status.value}")
        lines.append(f"Enabled: {'Yes' if agent.enabled else 'No'}")
        lines.append(f"Created: {agent.created_at.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Modified: {agent.last_modified.strftime('%Y-%m-%d %H:%M')}")
        
        if agent.error_message:
            lines.append(f"Error: {agent.error_message}")
        
        lines.append("─" * width)
        
        # Configuration
        lines.append("Configuration:")
        for key, value in agent.config.items():
            lines.append(f"  {key}: {value}")
        
        lines.append("─" * width)
        
        # Capabilities
        lines.append("Capabilities:")
        if agent.capabilities:
            for cap in agent.capabilities:
                support_icon = "✓" if cap.supported else "✗"
                lines.append(f"  {support_icon} {cap.name}: {cap.description}")
        else:
            lines.append("  No capabilities detected")
        
        # Footer
        lines.append("─" * width)
        lines.append("E: Edit | D: Delete | Space: Start/Stop | Esc: Back")
        
        return lines
    
    def _render_agent_edit(self, width: int, height: int) -> List[str]:
        """Render agent editing form."""
        # TODO: Implement agent editing form
        return ["Agent editing form - Coming soon"]
    
    def _render_agent_create(self, width: int, height: int) -> List[str]:
        """Render agent creation form."""
        # TODO: Implement agent creation form
        return ["Agent creation form - Coming soon"]
    
    # Helper Methods
    
    def _get_filtered_agent_ids(self) -> List[str]:
        """Get list of agent IDs matching current filters."""
        agent_ids = []
        
        for agent_id, agent in self.agents.items():
            # Apply enabled filter
            if self.show_only_enabled and not agent.enabled:
                continue
            
            # Apply text filter
            if self.filter_text:
                searchable_text = (agent.name + agent.id + agent.provider.value).lower()
                if self.filter_text not in searchable_text:
                    continue
            
            agent_ids.append(agent_id)
        
        # Sort by name
        agent_ids.sort(key=lambda aid: self.agents[aid].name.lower())
        return agent_ids
    
    def _get_status_icon(self, status: AgentStatus) -> str:
        """Get status icon for agent status."""
        return {
            AgentStatus.STOPPED: "⏹",
            AgentStatus.STARTING: "🔄", 
            AgentStatus.RUNNING: "▶️",
            AgentStatus.STOPPING: "⏸",
            AgentStatus.ERROR: "❌",
            AgentStatus.UNKNOWN: "❓"
        }[status]
    
    def _generate_agent_id(self) -> str:
        """Generate unique agent ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    async def _validate_agent_config(self, agent: AgentConfiguration):
        """Validate agent configuration."""
        # TODO: Implement comprehensive validation
        if not agent.name.strip():
            raise AgentValidationError("Agent name cannot be empty")
        
        if not agent.provider:
            raise AgentValidationError("Agent provider must be specified")
    
    async def _detect_agent_capabilities(self, agent_id: str):
        """Detect and update agent capabilities."""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        # TODO: Integrate with actual capability detection
        # For now, provide default capabilities based on provider
        default_caps = {
            AgentProvider.OLLAMA: ["code_generation", "general_qa"],
            AgentProvider.OLLAMA_TURBO: ["fast_response", "code_generation"],
            AgentProvider.CODEX: ["advanced_coding", "code_review"],
            AgentProvider.CLAUDE: ["large_context", "analysis", "reasoning"]
        }
        
        cap_names = default_caps.get(agent.provider, [])
        agent.capabilities = [
            AgentCapability(
                name=cap_name,
                description=f"{cap_name.replace('_', ' ').title()} capability",
                supported=True
            )
            for cap_name in cap_names
        ]
    
    async def _load_agents(self):
        """Load agents from persistent storage."""
        # TODO: Implement loading from agent registry/config
        # For now, create some default agents
        await self._create_default_agents()
    
    async def _create_default_agents(self):
        """Create default agents for demo purposes."""
        default_agents = [
            ("ollama_turbo_default", "Ollama Turbo Default", AgentProvider.OLLAMA_TURBO),
            ("ollama_local", "Ollama Local", AgentProvider.OLLAMA),
            ("codex_advanced", "Codex Advanced", AgentProvider.CODEX),
            ("claude_large_context", "Claude Large Context", AgentProvider.CLAUDE)
        ]
        
        for agent_id, name, provider in default_agents:
            if agent_id not in self.agents:
                agent_config = AgentConfiguration(
                    id=agent_id,
                    name=name,
                    provider=provider,
                    enabled=True
                )
                self.agents[agent_id] = agent_config
                await self._detect_agent_capabilities(agent_id)
    
    async def _refresh_agent_status(self):
        """Refresh status of all agents."""
        # TODO: Implement real status checking
        pass
    
    async def _start_status_monitoring(self):
        """Start background status monitoring."""
        async def monitor_loop():
            while True:
                try:
                    await self._refresh_agent_status()
                    await asyncio.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    logger.warning(f"Status monitoring error: {e}")
                    await asyncio.sleep(10)  # Longer delay on error
        
        asyncio.create_task(monitor_loop())
    
    async def _register_event_handlers(self):
        """Register event handlers for agent manager."""
        
        async def handle_keyboard_event(event: Event):
            if not self.visible or event.event_type != EventType.KEYBOARD:
                return
            
            key = event.data.get('key', '')
            
            # Common navigation
            if key == 'up':
                self.navigate_agents(-1)
            elif key == 'down':
                self.navigate_agents(1)
            elif key == 'escape':
                if self.view_mode != "list":
                    self.set_view_mode("list")
                else:
                    await self.hide()
            
            # View-specific actions
            elif self.view_mode == "list":
                if key == 'enter':
                    self.set_view_mode("detail")
                elif key == 'e':
                    self.set_view_mode("edit")
                elif key == 'd':
                    if self.selected_agent_id:
                        await self.delete_agent(self.selected_agent_id)
                elif key == 'space':
                    if self.selected_agent_id:
                        agent = self.agents[self.selected_agent_id]
                        if agent.status == AgentStatus.RUNNING:
                            await self.stop_agent(self.selected_agent_id)
                        else:
                            await self.start_agent(self.selected_agent_id)
                elif key == 'b':
                    self.toggle_bulk_mode()
                elif key == 'n':
                    self.set_view_mode("create")
                elif key == 'f':
                    self.toggle_show_enabled_only()
                elif key == 'x' and self.bulk_mode:
                    self.toggle_agent_selection()
        
        await self.event_system.subscribe(EventType.KEYBOARD, handle_keyboard_event)
    
    async def _clear_interface(self):
        """Clear the agent manager interface."""
        if hasattr(self.display_renderer, 'clear_region'):
            self.display_renderer.clear_region("agent_manager")
    
    async def cleanup(self):
        """Cleanup agent manager resources."""
        if self.visible:
            await self.hide()
        
        # Stop any running agents
        for agent_id, agent in self.agents.items():
            if agent.status == AgentStatus.RUNNING:
                await self.stop_agent(agent_id)
        
        logger.info("Agent manager cleanup completed")


# Factory function for easy integration
def create_agent_manager(event_system: AsyncEventSystem,
                        display_renderer: DisplayRenderer,
                        terminal_manager: TerminalManager) -> AgentManager:
    """Create and return a configured agent manager instance."""
    return AgentManager(
        event_system=event_system,
        display_renderer=display_renderer,
        terminal_manager=terminal_manager
    )