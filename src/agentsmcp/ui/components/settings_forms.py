"""
Settings Forms Component - Hierarchical settings forms with progressive disclosure.

This component provides:
- Hierarchical settings forms (Global → User → Session → Agent)
- Progressive disclosure with smart defaults
- Real-time validation feedback
- Form field types: text, number, boolean, select, multi-select
- Conditional field visibility based on other field values
- Form templates and presets
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import re

from ..v2.event_system import AsyncEventSystem, Event, EventType
from ..v2.display_renderer import DisplayRenderer
from ..v2.terminal_manager import TerminalManager

logger = logging.getLogger(__name__)


class FieldType(Enum):
    """Form field types."""
    TEXT = "text"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    PASSWORD = "password"
    EMAIL = "email"
    URL = "url"
    FILE_PATH = "file_path"
    JSON = "json"


class ValidationSeverity(Enum):
    """Validation message severity."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of field validation."""
    valid: bool
    severity: ValidationSeverity
    message: str
    field_name: Optional[str] = None


@dataclass
class FormFieldDefinition:
    """Definition of a form field."""
    name: str
    label: str
    field_type: FieldType
    default_value: Any = None
    required: bool = False
    description: Optional[str] = None
    placeholder: Optional[str] = None
    
    # Validation
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None  # Regex pattern
    
    # Select field options
    options: List[Any] = field(default_factory=list)
    
    # Conditional visibility
    visible_when: Optional[Dict[str, Any]] = None  # {"field_name": "expected_value"}
    
    # Help text
    help_text: Optional[str] = None
    
    # Custom validation function
    validator: Optional[Callable[[Any], ValidationResult]] = None


@dataclass
class FormSection:
    """A section of related form fields."""
    name: str
    title: str
    description: Optional[str] = None
    fields: List[FormFieldDefinition] = field(default_factory=list)
    collapsible: bool = True
    collapsed: bool = False
    visible_when: Optional[Dict[str, Any]] = None


@dataclass
class FormTemplate:
    """A complete form template."""
    id: str
    name: str
    description: str
    sections: List[FormSection] = field(default_factory=list)
    scope: Optional[str] = None  # For settings hierarchy


class FormValidationError(Exception):
    """Form validation error."""
    pass


class SettingsForm:
    """
    Advanced settings form with progressive disclosure and real-time validation.
    
    Features:
    - Multiple field types with appropriate input widgets
    - Progressive disclosure with collapsible sections
    - Real-time validation with immediate feedback
    - Conditional field visibility
    - Form templates and presets
    - Smart defaults and auto-completion
    - Accessibility support with keyboard navigation
    """
    
    def __init__(self,
                 event_system: AsyncEventSystem,
                 display_renderer: DisplayRenderer,
                 terminal_manager: TerminalManager):
        """Initialize the settings form."""
        self.event_system = event_system
        self.display_renderer = display_renderer
        self.terminal_manager = terminal_manager
        
        # Form state
        self.template: Optional[FormTemplate] = None
        self.values: Dict[str, Any] = {}
        self.validation_results: Dict[str, ValidationResult] = {}
        self.visible = False
        
        # UI state
        self.current_section_idx = 0
        self.current_field_idx = 0
        self.edit_mode = False
        self.scroll_offset = 0
        
        # Form templates
        self.templates: Dict[str, FormTemplate] = {}
        
        # Callbacks
        self.on_submit: Optional[Callable[[Dict[str, Any]], bool]] = None
        self.on_cancel: Optional[Callable[[], None]] = None
        self.on_change: Optional[Callable[[str, Any], None]] = None
        
        self._initialize_templates()
    
    async def initialize(self) -> bool:
        """Initialize the settings form."""
        try:
            # Register event handlers
            await self._register_event_handlers()
            
            logger.info("Settings form initialized successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to initialize settings form: {e}")
            return False
    
    def _initialize_templates(self):
        """Initialize form templates for common settings types."""
        
        # Global system settings template
        global_template = FormTemplate(
            id="global_settings",
            name="Global System Settings",
            description="System-wide configuration settings",
            scope="global"
        )
        
        # Logging section
        logging_section = FormSection(
            name="logging",
            title="Logging Configuration",
            description="Configure system logging behavior"
        )
        logging_section.fields = [
            FormFieldDefinition(
                name="log_level",
                label="Log Level",
                field_type=FieldType.SELECT,
                default_value="INFO",
                required=True,
                options=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
                help_text="Set the minimum log level for system messages"
            ),
            FormFieldDefinition(
                name="log_file_path",
                label="Log File Path",
                field_type=FieldType.FILE_PATH,
                default_value="./logs/agentsmcp.log",
                help_text="Path where log files will be stored"
            ),
            FormFieldDefinition(
                name="max_log_size_mb",
                label="Max Log File Size (MB)",
                field_type=FieldType.NUMBER,
                default_value=100,
                min_value=1,
                max_value=1024,
                help_text="Maximum size for individual log files"
            ),
            FormFieldDefinition(
                name="log_rotation_count",
                label="Log Rotation Count",
                field_type=FieldType.NUMBER,
                default_value=5,
                min_value=1,
                max_value=50,
                help_text="Number of old log files to keep"
            )
        ]
        
        # Performance section
        perf_section = FormSection(
            name="performance",
            title="Performance Settings",
            description="System performance and resource limits"
        )
        perf_section.fields = [
            FormFieldDefinition(
                name="max_memory_mb",
                label="Max Memory Usage (MB)",
                field_type=FieldType.NUMBER,
                default_value=1024,
                min_value=256,
                max_value=8192,
                help_text="Maximum memory usage for the application"
            ),
            FormFieldDefinition(
                name="max_concurrent_agents",
                label="Max Concurrent Agents",
                field_type=FieldType.NUMBER,
                default_value=5,
                min_value=1,
                max_value=20,
                help_text="Maximum number of agents that can run simultaneously"
            ),
            FormFieldDefinition(
                name="enable_performance_monitoring",
                label="Enable Performance Monitoring",
                field_type=FieldType.BOOLEAN,
                default_value=True,
                help_text="Enable system performance monitoring and metrics collection"
            ),
            FormFieldDefinition(
                name="performance_report_interval_sec",
                label="Performance Report Interval (seconds)",
                field_type=FieldType.NUMBER,
                default_value=300,
                min_value=60,
                max_value=3600,
                visible_when={"enable_performance_monitoring": True},
                help_text="How often to generate performance reports"
            )
        ]
        
        global_template.sections = [logging_section, perf_section]
        self.templates["global_settings"] = global_template
        
        # User preferences template
        user_template = FormTemplate(
            id="user_preferences",
            name="User Preferences",
            description="Personal user settings and preferences",
            scope="user"
        )
        
        # Interface section
        interface_section = FormSection(
            name="interface",
            title="User Interface",
            description="Customize the user interface appearance and behavior"
        )
        interface_section.fields = [
            FormFieldDefinition(
                name="theme",
                label="Theme",
                field_type=FieldType.SELECT,
                default_value="auto",
                options=["light", "dark", "auto"],
                help_text="Color theme for the interface"
            ),
            FormFieldDefinition(
                name="language",
                label="Language",
                field_type=FieldType.SELECT,
                default_value="en",
                options=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja"],
                help_text="Interface language"
            ),
            FormFieldDefinition(
                name="font_size",
                label="Font Size",
                field_type=FieldType.SELECT,
                default_value="medium",
                options=["small", "medium", "large", "x-large"],
                help_text="Text size in the interface"
            ),
            FormFieldDefinition(
                name="show_line_numbers",
                label="Show Line Numbers",
                field_type=FieldType.BOOLEAN,
                default_value=True,
                help_text="Display line numbers in code views"
            )
        ]
        
        # Notifications section
        notifications_section = FormSection(
            name="notifications",
            title="Notifications",
            description="Configure notification preferences"
        )
        notifications_section.fields = [
            FormFieldDefinition(
                name="enable_notifications",
                label="Enable Notifications",
                field_type=FieldType.BOOLEAN,
                default_value=True,
                help_text="Enable system notifications"
            ),
            FormFieldDefinition(
                name="notification_sound",
                label="Notification Sound",
                field_type=FieldType.BOOLEAN,
                default_value=False,
                visible_when={"enable_notifications": True},
                help_text="Play sound with notifications"
            ),
            FormFieldDefinition(
                name="notification_types",
                label="Notification Types",
                field_type=FieldType.MULTI_SELECT,
                default_value=["errors", "completions"],
                options=["errors", "warnings", "completions", "system_updates", "agent_status"],
                visible_when={"enable_notifications": True},
                help_text="Types of notifications to show"
            )
        ]
        
        user_template.sections = [interface_section, notifications_section]
        self.templates["user_preferences"] = user_template
        
        # Agent configuration template
        agent_template = FormTemplate(
            id="agent_config",
            name="Agent Configuration",
            description="Configure individual agent settings",
            scope="agent"
        )
        
        # Basic agent settings
        basic_section = FormSection(
            name="basic",
            title="Basic Settings",
            description="Essential agent configuration"
        )
        basic_section.fields = [
            FormFieldDefinition(
                name="agent_name",
                label="Agent Name",
                field_type=FieldType.TEXT,
                required=True,
                min_length=1,
                max_length=100,
                help_text="Unique name for this agent"
            ),
            FormFieldDefinition(
                name="provider",
                label="Provider",
                field_type=FieldType.SELECT,
                required=True,
                options=["ollama", "ollama-turbo", "codex", "claude"],
                help_text="AI provider for this agent"
            ),
            FormFieldDefinition(
                name="model",
                label="Model",
                field_type=FieldType.TEXT,
                help_text="Specific model to use (provider-dependent)"
            ),
            FormFieldDefinition(
                name="enabled",
                label="Enabled",
                field_type=FieldType.BOOLEAN,
                default_value=True,
                help_text="Whether this agent is active"
            )
        ]
        
        # Advanced agent settings
        advanced_section = FormSection(
            name="advanced",
            title="Advanced Settings",
            description="Advanced agent configuration options",
            collapsed=True
        )
        advanced_section.fields = [
            FormFieldDefinition(
                name="temperature",
                label="Temperature",
                field_type=FieldType.NUMBER,
                default_value=0.7,
                min_value=0.0,
                max_value=2.0,
                help_text="Controls randomness in responses (0.0 = deterministic, 2.0 = very random)"
            ),
            FormFieldDefinition(
                name="max_tokens",
                label="Max Tokens",
                field_type=FieldType.NUMBER,
                default_value=4096,
                min_value=100,
                max_value=32768,
                help_text="Maximum number of tokens in responses"
            ),
            FormFieldDefinition(
                name="context_window",
                label="Context Window",
                field_type=FieldType.NUMBER,
                default_value=8192,
                min_value=1024,
                max_value=1000000,
                help_text="Size of the context window for this agent"
            ),
            FormFieldDefinition(
                name="system_prompt",
                label="System Prompt",
                field_type=FieldType.TEXT,
                help_text="Custom system prompt for this agent (optional)"
            ),
            FormFieldDefinition(
                name="timeout_seconds",
                label="Timeout (seconds)",
                field_type=FieldType.NUMBER,
                default_value=120,
                min_value=10,
                max_value=600,
                help_text="Request timeout for this agent"
            )
        ]
        
        agent_template.sections = [basic_section, advanced_section]
        self.templates["agent_config"] = agent_template
    
    async def show_form(self, template_id: str, initial_values: Optional[Dict[str, Any]] = None):
        """Show form with specified template."""
        if template_id not in self.templates:
            raise FormValidationError(f"Template {template_id} not found")
        
        self.template = self.templates[template_id]
        self.values = initial_values.copy() if initial_values else {}
        self.validation_results.clear()
        
        # Set default values for fields not in initial_values
        for section in self.template.sections:
            for field in section.fields:
                if field.name not in self.values and field.default_value is not None:
                    self.values[field.name] = field.default_value
        
        # Reset UI state
        self.current_section_idx = 0
        self.current_field_idx = 0
        self.edit_mode = False
        self.scroll_offset = 0
        
        self.visible = True
        await self._validate_all_fields()
        await self._render_form()
        
        # Emit form shown event
        await self.event_system.emit(Event(
            event_type=EventType.UI,
            data={"action": "form_shown", "template_id": template_id}
        ))
    
    async def hide_form(self):
        """Hide the form."""
        if not self.visible:
            return
        
        self.visible = False
        await self._clear_form()
        
        # Emit form hidden event
        await self.event_system.emit(Event(
            event_type=EventType.UI,
            data={"action": "form_hidden"}
        ))
    
    async def submit_form(self) -> bool:
        """Submit the form if valid."""
        if not self.template:
            return False
        
        # Validate all fields
        await self._validate_all_fields()
        
        # Check if form has errors
        if any(result.severity == ValidationSeverity.ERROR for result in self.validation_results.values()):
            await self._show_validation_errors()
            return False
        
        # Call submit callback if provided
        if self.on_submit:
            try:
                success = self.on_submit(self.values.copy())
                if success:
                    await self.hide_form()
                    return True
            except Exception as e:
                logger.exception(f"Error in form submit callback: {e}")
                return False
        
        return False
    
    async def cancel_form(self):
        """Cancel form editing."""
        if self.on_cancel:
            self.on_cancel()
        
        await self.hide_form()
    
    # Navigation methods
    
    def navigate_sections(self, direction: int):
        """Navigate between sections."""
        if not self.template:
            return
        
        max_idx = len(self.template.sections) - 1
        self.current_section_idx = max(0, min(max_idx, self.current_section_idx + direction))
        self.current_field_idx = 0  # Reset field index when switching sections
        
        asyncio.create_task(self._render_form())
    
    def navigate_fields(self, direction: int):
        """Navigate between fields in current section."""
        if not self.template or not self.template.sections:
            return
        
        section = self.template.sections[self.current_section_idx]
        visible_fields = self._get_visible_fields(section)
        
        if not visible_fields:
            return
        
        max_idx = len(visible_fields) - 1
        self.current_field_idx = max(0, min(max_idx, self.current_field_idx + direction))
        
        asyncio.create_task(self._render_form())
    
    def toggle_section_collapse(self):
        """Toggle collapse state of current section."""
        if not self.template or not self.template.sections:
            return
        
        section = self.template.sections[self.current_section_idx]
        if section.collapsible:
            section.collapsed = not section.collapsed
            asyncio.create_task(self._render_form())
    
    def enter_edit_mode(self):
        """Enter edit mode for current field."""
        self.edit_mode = True
        asyncio.create_task(self._render_form())
    
    def exit_edit_mode(self, save_changes: bool = True):
        """Exit edit mode."""
        if save_changes:
            asyncio.create_task(self._validate_current_field())
        
        self.edit_mode = False
        asyncio.create_task(self._render_form())
    
    async def update_field_value(self, field_name: str, value: Any):
        """Update a field value and validate."""
        self.values[field_name] = value
        
        # Validate the field
        await self._validate_field(field_name)
        
        # Check conditional visibility changes
        await self._update_field_visibility()
        
        # Call change callback
        if self.on_change:
            self.on_change(field_name, value)
        
        await self._render_form()
    
    # Validation methods
    
    async def _validate_all_fields(self):
        """Validate all form fields."""
        for section in self.template.sections:
            for field in section.fields:
                await self._validate_field(field.name)
    
    async def _validate_field(self, field_name: str) -> ValidationResult:
        """Validate a single field."""
        # Find field definition
        field_def = None
        for section in self.template.sections:
            for field in section.fields:
                if field.name == field_name:
                    field_def = field
                    break
            if field_def:
                break
        
        if not field_def:
            return ValidationResult(True, ValidationSeverity.INFO, "Field not found")
        
        value = self.values.get(field_name)
        
        # Required field validation
        if field_def.required and (value is None or value == ""):
            result = ValidationResult(
                False, 
                ValidationSeverity.ERROR, 
                f"{field_def.label} is required",
                field_name
            )
            self.validation_results[field_name] = result
            return result
        
        # Skip further validation for empty optional fields
        if value is None or value == "":
            result = ValidationResult(True, ValidationSeverity.INFO, "")
            self.validation_results[field_name] = result
            return result
        
        # Type-specific validation
        if field_def.field_type == FieldType.NUMBER:
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                    self.values[field_name] = value
                except (ValueError, TypeError):
                    result = ValidationResult(
                        False,
                        ValidationSeverity.ERROR,
                        f"{field_def.label} must be a number",
                        field_name
                    )
                    self.validation_results[field_name] = result
                    return result
            
            # Range validation
            if field_def.min_value is not None and value < field_def.min_value:
                result = ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"{field_def.label} must be at least {field_def.min_value}",
                    field_name
                )
                self.validation_results[field_name] = result
                return result
            
            if field_def.max_value is not None and value > field_def.max_value:
                result = ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"{field_def.label} must be at most {field_def.max_value}",
                    field_name
                )
                self.validation_results[field_name] = result
                return result
        
        elif field_def.field_type == FieldType.TEXT:
            value_str = str(value)
            
            # Length validation
            if field_def.min_length is not None and len(value_str) < field_def.min_length:
                result = ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"{field_def.label} must be at least {field_def.min_length} characters",
                    field_name
                )
                self.validation_results[field_name] = result
                return result
            
            if field_def.max_length is not None and len(value_str) > field_def.max_length:
                result = ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"{field_def.label} must be at most {field_def.max_length} characters",
                    field_name
                )
                self.validation_results[field_name] = result
                return result
            
            # Pattern validation
            if field_def.pattern and not re.match(field_def.pattern, value_str):
                result = ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"{field_def.label} format is invalid",
                    field_name
                )
                self.validation_results[field_name] = result
                return result
        
        elif field_def.field_type == FieldType.SELECT:
            if value not in field_def.options:
                result = ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"{field_def.label} must be one of: {', '.join(map(str, field_def.options))}",
                    field_name
                )
                self.validation_results[field_name] = result
                return result
        
        elif field_def.field_type == FieldType.EMAIL:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, str(value)):
                result = ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"{field_def.label} must be a valid email address",
                    field_name
                )
                self.validation_results[field_name] = result
                return result
        
        elif field_def.field_type == FieldType.URL:
            url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            if not re.match(url_pattern, str(value)):
                result = ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"{field_def.label} must be a valid URL",
                    field_name
                )
                self.validation_results[field_name] = result
                return result
        
        # Custom validation
        if field_def.validator:
            try:
                custom_result = field_def.validator(value)
                self.validation_results[field_name] = custom_result
                return custom_result
            except Exception as e:
                result = ValidationResult(
                    False,
                    ValidationSeverity.ERROR,
                    f"Validation error: {e}",
                    field_name
                )
                self.validation_results[field_name] = result
                return result
        
        # Field is valid
        result = ValidationResult(True, ValidationSeverity.INFO, "")
        self.validation_results[field_name] = result
        return result
    
    async def _validate_current_field(self):
        """Validate the currently selected field."""
        if not self.template or not self.template.sections:
            return
        
        section = self.template.sections[self.current_section_idx]
        visible_fields = self._get_visible_fields(section)
        
        if self.current_field_idx < len(visible_fields):
            field = visible_fields[self.current_field_idx]
            await self._validate_field(field.name)
    
    async def _update_field_visibility(self):
        """Update field visibility based on conditional rules."""
        # TODO: Implement conditional field visibility logic
        # For now, this is a placeholder
        pass
    
    # Rendering methods
    
    async def _render_form(self):
        """Render the complete form."""
        if not self.visible or not self.template:
            return
        
        try:
            caps = self.terminal_manager.detect_capabilities()
            width, height = caps.width, caps.height
            
            # Render form header
            header_lines = self._render_form_header(width)
            
            # Render sections
            content_height = height - len(header_lines) - 3  # Reserve space for footer
            content_lines = self._render_sections(width, content_height)
            
            # Render footer
            footer_lines = self._render_form_footer(width)
            
            # Combine all parts
            form_content = header_lines + content_lines + footer_lines
            
            # Update display
            self.display_renderer.update_region(
                "settings_form",
                "\n".join(form_content),
                force=True
            )
            
        except Exception as e:
            logger.exception(f"Error rendering form: {e}")
    
    def _render_form_header(self, width: int) -> List[str]:
        """Render form header."""
        lines = []
        
        # Title
        title = f"╔══ {self.template.name} ══╗".center(width)
        lines.append(title)
        
        # Description
        if self.template.description:
            desc_lines = textwrap.wrap(self.template.description, width - 4)
            for desc_line in desc_lines[:2]:  # Limit description lines
                lines.append(f"  {desc_line}")
        
        # Validation summary
        error_count = sum(1 for r in self.validation_results.values() 
                         if r.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for r in self.validation_results.values() 
                           if r.severity == ValidationSeverity.WARNING)
        
        if error_count > 0:
            status = f"❌ {error_count} errors"
            if warning_count > 0:
                status += f", {warning_count} warnings"
        elif warning_count > 0:
            status = f"⚠️ {warning_count} warnings"
        else:
            status = "✅ No validation issues"
        
        lines.append(status.center(width))
        lines.append("═" * width)
        
        return lines
    
    def _render_sections(self, width: int, height: int) -> List[str]:
        """Render form sections."""
        lines = []
        
        if not self.template.sections:
            lines.append("No sections defined".center(width))
            return lines
        
        # Render each section
        for idx, section in enumerate(self.template.sections):
            if not self._is_section_visible(section):
                continue
            
            is_current = idx == self.current_section_idx
            section_lines = self._render_section(section, width, is_current)
            
            # Check if section fits
            if len(lines) + len(section_lines) <= height:
                lines.extend(section_lines)
            else:
                lines.append("... (more sections below)")
                break
        
        # Pad remaining height
        while len(lines) < height:
            lines.append("")
        
        return lines
    
    def _render_section(self, section: FormSection, width: int, is_current: bool) -> List[str]:
        """Render a single form section."""
        lines = []
        
        # Section header
        expand_icon = "▼" if not section.collapsed else "▶"
        current_marker = "►" if is_current else " "
        
        header = f"{current_marker} {expand_icon} {section.title}"
        lines.append(header)
        
        # Section description
        if section.description and not section.collapsed:
            desc_lines = textwrap.wrap(section.description, width - 4)
            for desc_line in desc_lines[:1]:  # Limit description
                lines.append(f"    {desc_line}")
        
        # Section fields (if not collapsed)
        if not section.collapsed:
            visible_fields = self._get_visible_fields(section)
            for idx, field in enumerate(visible_fields):
                is_current_field = (is_current and idx == self.current_field_idx)
                field_lines = self._render_field(field, width, is_current_field)
                lines.extend(field_lines)
        
        lines.append("")  # Add spacing between sections
        return lines
    
    def _render_field(self, field: FormFieldDefinition, width: int, is_current: bool) -> List[str]:
        """Render a single form field."""
        lines = []
        
        # Field label and value
        value = self.values.get(field.name, "")
        validation_result = self.validation_results.get(field.name)
        
        # Current field marker and validation icon
        current_marker = "►" if is_current else " "
        
        if validation_result:
            if validation_result.severity == ValidationSeverity.ERROR:
                status_icon = "❌"
            elif validation_result.severity == ValidationSeverity.WARNING:
                status_icon = "⚠️"
            else:
                status_icon = "✅"
        else:
            status_icon = "  "
        
        # Required field indicator
        required_marker = "*" if field.required else ""
        
        # Field line
        field_line = f"{current_marker}   {field.label}{required_marker}: {value} {status_icon}"
        lines.append(field_line[:width])
        
        # Show validation message if current field or has error
        if validation_result and (is_current or validation_result.severity == ValidationSeverity.ERROR):
            if validation_result.message:
                msg_lines = textwrap.wrap(validation_result.message, width - 8)
                for msg_line in msg_lines[:1]:  # Limit message lines
                    lines.append(f"      {msg_line}")
        
        # Show help text if current field and edit mode
        if is_current and self.edit_mode and field.help_text:
            help_lines = textwrap.wrap(field.help_text, width - 8)
            for help_line in help_lines[:2]:  # Limit help lines
                lines.append(f"      💡 {help_line}")
        
        return lines
    
    def _render_form_footer(self, width: int) -> List[str]:
        """Render form footer with shortcuts."""
        lines = ["═" * width]
        
        if self.edit_mode:
            shortcuts = ["Enter: Save", "Esc: Cancel", "Tab: Next Field"]
        else:
            shortcuts = ["Enter: Edit", "Space: Toggle Section", "Tab: Next Section", "S: Submit", "Esc: Cancel"]
        
        footer_text = " | ".join(shortcuts)
        lines.append(footer_text[:width])
        
        return lines
    
    # Helper methods
    
    def _get_visible_fields(self, section: FormSection) -> List[FormFieldDefinition]:
        """Get list of visible fields in a section."""
        visible_fields = []
        
        for field in section.fields:
            if self._is_field_visible(field):
                visible_fields.append(field)
        
        return visible_fields
    
    def _is_section_visible(self, section: FormSection) -> bool:
        """Check if section should be visible based on conditions."""
        if not section.visible_when:
            return True
        
        for field_name, expected_value in section.visible_when.items():
            current_value = self.values.get(field_name)
            if current_value != expected_value:
                return False
        
        return True
    
    def _is_field_visible(self, field: FormFieldDefinition) -> bool:
        """Check if field should be visible based on conditions."""
        if not field.visible_when:
            return True
        
        for field_name, expected_value in field.visible_when.items():
            current_value = self.values.get(field_name)
            if current_value != expected_value:
                return False
        
        return True
    
    async def _show_validation_errors(self):
        """Show validation errors to user."""
        errors = [r for r in self.validation_results.values() 
                 if r.severity == ValidationSeverity.ERROR]
        
        if errors:
            error_messages = [f"• {error.message}" for error in errors]
            message = "Form has validation errors:\n" + "\n".join(error_messages)
            
            await self.event_system.emit(Event(
                event_type=EventType.ERROR,
                data={"message": message, "component": "settings_form"}
            ))
    
    async def _register_event_handlers(self):
        """Register event handlers for form interaction."""
        
        async def handle_keyboard_event(event: Event):
            if not self.visible or event.event_type != EventType.KEYBOARD:
                return
            
            key = event.data.get('key', '')
            
            if self.edit_mode:
                # Edit mode navigation
                if key == 'enter':
                    self.exit_edit_mode(save_changes=True)
                elif key == 'escape':
                    self.exit_edit_mode(save_changes=False)
                elif key == 'tab':
                    self.exit_edit_mode(save_changes=True)
                    self.navigate_fields(1)
            else:
                # Normal navigation
                if key == 'up':
                    self.navigate_fields(-1)
                elif key == 'down':
                    self.navigate_fields(1)
                elif key == 'left':
                    self.navigate_sections(-1)
                elif key == 'right':
                    self.navigate_sections(1)
                elif key == 'enter':
                    self.enter_edit_mode()
                elif key == 'space':
                    self.toggle_section_collapse()
                elif key == 's':
                    await self.submit_form()
                elif key == 'escape':
                    await self.cancel_form()
        
        await self.event_system.subscribe(EventType.KEYBOARD, handle_keyboard_event)
    
    async def _clear_form(self):
        """Clear form from display."""
        if hasattr(self.display_renderer, 'clear_region'):
            self.display_renderer.clear_region("settings_form")
    
    async def cleanup(self):
        """Cleanup form resources."""
        if self.visible:
            await self.hide_form()
        
        logger.info("Settings form cleanup completed")


# Factory function for easy integration
def create_settings_form(event_system: AsyncEventSystem,
                        display_renderer: DisplayRenderer,
                        terminal_manager: TerminalManager) -> SettingsForm:
    """Create and return a configured settings form instance."""
    return SettingsForm(
        event_system=event_system,
        display_renderer=display_renderer,
        terminal_manager=terminal_manager
    )