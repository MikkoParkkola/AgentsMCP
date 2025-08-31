# AgentsMCP UI/UX Improvement Plan - Revised

*Plan Date: 2025-08-30*  
*Revision: 1.1 - Incorporating specialized agent feedback*
*Priority: Critical - TUI Input Broken, Major UX Issues Identified*

## Executive Summary

Current implementation snapshot (as of this revision)
- Chat CLI: /models, /provider, /apikey, /context, /stream implemented with provider validation, selection UI, and streaming coalescing. Model discovery adapters for OpenAI/OpenRouter/Ollama present.
- TUI v2: Layout and components exist; input pipeline stabilization in progress; ensure per-key input and render loop reliably display typed characters.
- Server: REST endpoints in place; SSE event stream and minimal dashboard not yet implemented; discovery partially present (status + announcer behind flags).

Focus areas to complete Phase 1
- TUI v2 input stabilization + basic command palette and logs view.
- Setup wizard (API keys + config + connection test) and CLI standardization.
- MCP version negotiation (negotiate + downconvert) and finalize provider-native streaming wiring where available.

Based on comprehensive review by specialized agents (Product UX Designer, QA Logic Reviewer, and System Architect), this revised plan addresses critical technical and user experience issues while implementing proper architectural patterns for long-term maintainability.

**Key Changes from Original Plan:**
- Extended Phase 1 from 2 to 3 weeks for proper implementation
- Added mandatory user research phase (Week 2.5)
- Moved accessibility from Phase 4 to Phase 1 (Day 1 implementation)
- Implemented hexagonal architecture pattern
- Added comprehensive testing framework from Day 1
- Fixed critical missing function implementations

## Problem Analysis - Enhanced

### Root Cause Analysis (QA Agent Findings)

**Critical Technical Issues:**
1. **Missing Function Implementation**: `_apply_ansi_markdown` function referenced but not implemented
2. **Flawed Consolidation Approach**: Multiple input systems coexist rather than being unified
3. **Inadequate Testing**: No comprehensive testing framework for TUI interactions
4. **Poor Error Recovery**: Limited fallback mechanisms for input failures

**Architectural Issues (System Architect Findings):**
1. **Monolithic Design**: Single large components instead of modular architecture
2. **Missing Interface Contracts**: No formal ICDs between components
3. **Event Handling Chaos**: No event-driven architecture for reactive updates
4. **Tight Coupling**: Components directly depend on each other instead of interfaces

### User Experience Issues (UX Agent Findings)

**Critical UX Problems:**
1. **No User Research**: Plan lacks understanding of actual user needs
2. **Accessibility Afterthought**: WCAG compliance relegated to final phase
3. **Timeline Too Optimistic**: Complex fixes require more time than allocated
4. **Missing User Feedback Loop**: No mechanism for iterative improvement

## Solution Strategy - Revised

### Phase 0: User Research & Architecture (Week 1)
**Goal: Understand users and establish proper foundation**

#### 0.1 User Research (Week 1 - Days 1-3)
**Research Activities:**
1. **User Interviews**: Interview 5-8 existing and potential users
   - Current pain points and workflows
   - Terminal vs web interface preferences
   - Accessibility needs and requirements
   - Configuration complexity tolerance

2. **Usage Analytics**: Analyze current usage patterns
   ```python
   # src/agentsmcp/analytics/usage_tracker.py
   class UsageTracker:
       def track_command_usage(self, command: str, success: bool):
           """Track command usage and success rates"""
       
       def track_tui_interactions(self, interaction_type: str):
           """Track TUI interaction patterns"""
       
       def generate_usage_report(self) -> Dict[str, Any]:
           """Generate anonymized usage report"""
   ```

3. **Accessibility Assessment**: WCAG 2.2 AA compliance audit
   - Screen reader compatibility testing
   - Keyboard navigation assessment
   - Color contrast validation
   - Text alternative verification

#### 0.2 Architectural Foundation (Week 1 - Days 4-5)
**Implement Hexagonal Architecture Pattern:**

```python
# src/agentsmcp/architecture/
├── ports/                    # Interfaces (contracts)
│   ├── input_port.py        # User input interface
│   ├── agent_port.py        # Agent communication interface
│   ├── config_port.py       # Configuration interface
│   └── ui_port.py           # UI rendering interface
├── adapters/                # Implementations
│   ├── tui_adapter.py       # TUI implementation
│   ├── cli_adapter.py       # CLI implementation
│   ├── web_adapter.py       # Web implementation
│   └── agent_adapter.py     # Agent communication
└── domain/                  # Core business logic
    ├── commands/            # Command handling
    ├── events/              # Domain events
    └── models/              # Core models
```

**Event-Driven Architecture Implementation:**
```python
# src/agentsmcp/events/event_bus.py
from typing import Any, Callable, Dict, List
import asyncio

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_queue = asyncio.Queue()
    
    async def publish(self, event_type: str, data: Any):
        """Publish event to all subscribers"""
        await self._event_queue.put(Event(event_type, data))
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to specific event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    async def process_events(self):
        """Process events from queue"""
        while True:
            event = await self._event_queue.get()
            for handler in self._subscribers.get(event.type, []):
                await handler(event.data)
```

### Phase 1: Critical Fixes with Accessibility (Week 2-4)
**Goal: Fix critical issues with accessibility built-in from Day 1**

#### 1.1 Implement Missing Functions (Priority: P0 - Day 1)
**Fix Critical Missing Implementation:**

```python
# src/agentsmcp/ui/v2/markdown_handler.py
import re
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

def _apply_ansi_markdown(content: str, theme: str = "default") -> str:
    """
    Apply ANSI markdown formatting with accessibility support.
    
    Args:
        content: Raw markdown content
        theme: Color theme (default, high-contrast, dark)
    
    Returns:
        ANSI-formatted string with accessibility attributes
    """
    try:
        console = Console(
            color_system="truecolor" if theme != "high-contrast" else "standard",
            force_terminal=True
        )
        
        # Create markdown renderer with accessibility
        markdown = Markdown(
            content,
            code_theme="github-dark" if theme == "dark" else "default",
            hyperlinks=True
        )
        
        # Render with screen reader friendly output
        with console.capture() as capture:
            console.print(markdown)
        
        result = capture.get()
        
        # Add accessibility markers for screen readers
        if theme == "high-contrast":
            result = _add_accessibility_markers(result)
            
        return result
        
    except Exception as e:
        # Fallback to plain text with accessibility
        return _fallback_accessible_format(content, str(e))

def _add_accessibility_markers(content: str) -> str:
    """Add screen reader navigation markers"""
    # Add ARIA-style markers for TUI screen readers
    content = re.sub(r'^(#+\s+)', r'\1[HEADING] ', content, flags=re.MULTILINE)
    content = re.sub(r'^(\*\s+)', r'\1[LIST ITEM] ', content, flags=re.MULTILINE)
    content = re.sub(r'```(\w+)?', r'[CODE BLOCK START]', content)
    content = re.sub(r'```$', r'[CODE BLOCK END]', content, flags=re.MULTILINE)
    return content

def _fallback_accessible_format(content: str, error: str) -> str:
    """Fallback formatter with accessibility"""
    return f"""[FORMATTED TEXT START]
{content}
[FORMATTED TEXT END]
[NOTE: Advanced formatting unavailable - {error}]"""
```

#### 1.2 Unified Input System with Accessibility (Priority: P0 - Week 2)
**Replace Multiple Competing Handlers with Single Accessible System:**

```python
# src/agentsmcp/ui/unified_input/accessible_input.py
from typing import Optional, Callable, Dict, Any
import asyncio
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import clear
from prompt_toolkit.application import Application

class AccessibleInputHandler:
    """Unified input handler with built-in accessibility support"""
    
    def __init__(self, accessibility_mode: bool = False):
        self.accessibility_mode = accessibility_mode
        self.screen_reader_detected = self._detect_screen_reader()
        self.high_contrast_mode = self._detect_high_contrast()
        
        # Event-driven architecture
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Setup accessible key bindings
        self.key_bindings = self._create_accessible_bindings()
        
        # Create session with accessibility features
        self.session = PromptSession(
            key_bindings=self.key_bindings,
            mouse_support=not self.accessibility_mode,  # Disable mouse if accessibility mode
            complete_style='column' if self.screen_reader_detected else 'multi-column'
        )
    
    def _detect_screen_reader(self) -> bool:
        """Detect if screen reader is active"""
        screen_readers = [
            'NVDA', 'JAWS', 'ORCA', 'VOICEOVER', 'TALKBACK'
        ]
        import os
        return any(sr in os.environ.get('SCREENREADER', '').upper() 
                  for sr in screen_readers)
    
    def _detect_high_contrast(self) -> bool:
        """Detect high contrast preference"""
        import os
        return (os.environ.get('TERM_PROGRAM') == 'HighContrast' or
                os.environ.get('ACCESSIBILITY_HIGH_CONTRAST') == '1')
    
    def _create_accessible_bindings(self) -> KeyBindings:
        """Create keyboard bindings optimized for accessibility"""
        bindings = KeyBindings()
        
        @bindings.add('tab')
        def _(event):
            """Navigate between UI elements"""
            self._announce_navigation("Next element")
            
        @bindings.add('s-tab')  # Shift+Tab
        def _(event):
            """Navigate backwards between UI elements"""
            self._announce_navigation("Previous element")
            
        @bindings.add('f1')
        def _(event):
            """Context-sensitive help"""
            self._show_contextual_help()
            
        @bindings.add('escape', 'h')
        def _(event):
            """Quick help overlay"""
            self._show_help_overlay()
        
        return bindings
    
    def _announce_navigation(self, action: str):
        """Announce navigation for screen readers"""
        if self.screen_reader_detected:
            # Send to screen reader buffer
            sys.stderr.write(f"\a{action}\n")  # Bell + announcement
            sys.stderr.flush()
    
    async def get_input(self, 
                       prompt: str = "AgentsMCP> ",
                       multiline: bool = False,
                       **kwargs) -> str:
        """Get user input with full accessibility support"""
        
        # Announce prompt for screen readers
        if self.screen_reader_detected:
            accessible_prompt = f"[INPUT FIELD] {prompt} [Type your message]"
            sys.stderr.write(f"{accessible_prompt}\n")
        
        try:
            if multiline:
                return await self._get_multiline_input(prompt, **kwargs)
            else:
                return await self.session.prompt_async(
                    prompt,
                    **kwargs
                )
        except KeyboardInterrupt:
            if self.screen_reader_detected:
                sys.stderr.write("[CANCELLED]\n")
            raise
        except Exception as e:
            # Accessible error reporting
            error_msg = f"[INPUT ERROR] {str(e)} [Press F1 for help]"
            sys.stderr.write(f"{error_msg}\n")
            return ""
    
    async def _get_multiline_input(self, prompt: str, **kwargs) -> str:
        """Handle multiline input with accessibility"""
        if self.screen_reader_detected:
            sys.stderr.write("[MULTILINE INPUT MODE] Press Ctrl+D when finished\n")
        
        return await self.session.prompt_async(
            prompt,
            multiline=True,
            **kwargs
        )
```

#### 1.3 Comprehensive Testing Framework (Week 2)
**Implement Testing from Day 1:**

```python
# tests/ui/test_accessible_input.py
import pytest
import asyncio
from unittest.mock import Mock, patch
from agentsmcp.ui.unified_input.accessible_input import AccessibleInputHandler

class TestAccessibleInputHandler:
    
    @pytest.fixture
    async def handler(self):
        """Create test handler instance"""
        return AccessibleInputHandler(accessibility_mode=True)
    
    @pytest.mark.asyncio
    async def test_screen_reader_detection(self, handler):
        """Test screen reader detection logic"""
        with patch.dict('os.environ', {'SCREENREADER': 'NVDA'}):
            assert handler._detect_screen_reader() == True
    
    @pytest.mark.asyncio
    async def test_accessible_input_basic(self, handler):
        """Test basic accessible input functionality"""
        with patch.object(handler.session, 'prompt_async', 
                         return_value=asyncio.Future()) as mock_prompt:
            mock_prompt.return_value.set_result("test input")
            
            result = await handler.get_input("Test prompt> ")
            assert result == "test input"
    
    @pytest.mark.asyncio
    async def test_multiline_accessible_input(self, handler):
        """Test multiline input with accessibility"""
        test_multiline = "line 1\nline 2\nline 3"
        
        with patch.object(handler.session, 'prompt_async',
                         return_value=asyncio.Future()) as mock_prompt:
            mock_prompt.return_value.set_result(test_multiline)
            
            result = await handler.get_input("Multiline> ", multiline=True)
            assert result == test_multiline
    
    @pytest.mark.asyncio 
    async def test_keyboard_navigation(self, handler):
        """Test accessible keyboard navigation"""
        # Test tab navigation
        with patch.object(handler, '_announce_navigation') as mock_announce:
            # Simulate Tab key press
            handler._create_accessible_bindings()
            # Verify navigation announcement
            # This would require more complex event simulation
            pass
    
    @pytest.mark.asyncio
    async def test_error_recovery_accessible(self, handler):
        """Test accessible error handling and recovery"""
        with patch.object(handler.session, 'prompt_async',
                         side_effect=Exception("Test error")):
            
            with patch('sys.stderr') as mock_stderr:
                result = await handler.get_input("Error test> ")
                assert result == ""
                # Verify accessible error message was written
                mock_stderr.write.assert_called()

# tests/ui/test_markdown_handler.py
import pytest
from agentsmcp.ui.v2.markdown_handler import _apply_ansi_markdown

class TestMarkdownHandler:
    
    def test_apply_ansi_markdown_basic(self):
        """Test basic markdown formatting"""
        content = "# Header\n\nSome **bold** text."
        result = _apply_ansi_markdown(content)
        assert result is not None
        assert len(result) > 0
    
    def test_apply_ansi_markdown_accessibility(self):
        """Test accessibility markers in high contrast mode"""
        content = "# Header\n\n* List item\n\n```python\ncode\n```"
        result = _apply_ansi_markdown(content, theme="high-contrast")
        assert "[HEADING]" in result
        assert "[LIST ITEM]" in result
        assert "[CODE BLOCK START]" in result
        assert "[CODE BLOCK END]" in result
    
    def test_fallback_accessible_format(self):
        """Test fallback formatting with accessibility"""
        from agentsmcp.ui.v2.markdown_handler import _fallback_accessible_format
        
        content = "Test content"
        error = "Test error"
        result = _fallback_accessible_format(content, error)
        
        assert "[FORMATTED TEXT START]" in result
        assert content in result
        assert "[FORMATTED TEXT END]" in result
        assert error in result
```

#### 1.4 Interface Contract Documents (ICDs) (Week 3)
**Establish Formal Contracts Between Components:**

```json
// spec/icds/input_handler.json
{
  "name": "input_handler",
  "version": "1.0.0",
  "purpose": "Handle all user input with accessibility support",
  "interfaces": {
    "InputPort": {
      "methods": {
        "get_input": {
          "inputs": {
            "prompt": "string",
            "multiline": "boolean?",
            "accessibility_mode": "boolean?"
          },
          "outputs": {
            "user_input": "string"
          },
          "errors": [
            "InputCancelled",
            "InputTimeout", 
            "InputValidationError"
          ]
        },
        "set_accessibility_mode": {
          "inputs": {
            "enabled": "boolean",
            "screen_reader_detected": "boolean?"
          },
          "outputs": {
            "mode_changed": "boolean"
          }
        }
      }
    }
  },
  "performance": {
    "input_response_time_ms": 50,
    "memory_usage_mb": 10
  },
  "accessibility": {
    "wcag_compliance": "AA",
    "screen_reader_support": true,
    "keyboard_navigation": true,
    "high_contrast_support": true
  }
}
```

#### 1.5 Configuration Wizard with Accessibility (Week 3-4)
**Implement Accessible Setup Process:**

```python
# src/agentsmcp/setup/accessible_wizard.py
from typing import Dict, Any, Optional
import asyncio
from ..ui.unified_input.accessible_input import AccessibleInputHandler

class AccessibleSetupWizard:
    """Setup wizard with built-in accessibility support"""
    
    def __init__(self):
        self.input_handler = AccessibleInputHandler(accessibility_mode=True)
        self.config = {}
        
    async def run_setup(self) -> Dict[str, Any]:
        """Run the complete accessible setup process"""
        
        await self._welcome_screen()
        await self._detect_environment()
        await self._configure_agents()
        await self._test_configuration()
        await self._finalize_setup()
        
        return self.config
    
    async def _welcome_screen(self):
        """Accessible welcome screen"""
        welcome_text = """
        [SETUP WIZARD START]
        Welcome to AgentsMCP Setup!
        
        This wizard will guide you through configuration.
        We've detected your accessibility preferences and
        will provide appropriate support.
        
        Press Enter to continue, or Ctrl+C to exit.
        [SETUP WIZARD WELCOME END]
        """
        
        print(welcome_text)
        await self.input_handler.get_input("")
    
    async def _detect_environment(self):
        """Detect and announce environment capabilities"""
        print("[ENVIRONMENT DETECTION]")
        
        # Check Python version
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"Python version: {python_version} - Compatible")
        
        # Check terminal capabilities
        if self.input_handler.screen_reader_detected:
            print("Screen reader detected - Enabling enhanced accessibility")
            
        if self.input_handler.high_contrast_mode:
            print("High contrast mode detected - Adjusting color scheme")
        
        print("[ENVIRONMENT DETECTION COMPLETE]")
    
    async def _configure_agents(self):
        """Accessible agent configuration"""
        print("[AGENT CONFIGURATION]")
        
        agent_choice = await self._accessible_choice(
            "Which AI agent would you like to use?",
            [
                ("ollama", "Ollama (Local, Free, Privacy-focused)"),
                ("openai", "OpenAI GPT-4 (Requires API key, Most capable)"),
                ("claude", "Claude (Requires API key, Long context)")
            ]
        )
        
        self.config['primary_agent'] = agent_choice
        
        if agent_choice != 'ollama':
            api_key = await self._get_api_key(agent_choice)
            self.config[f'{agent_choice}_api_key'] = api_key
    
    async def _accessible_choice(self, question: str, choices: list) -> str:
        """Present accessible multiple choice"""
        print(f"\n[QUESTION] {question}")
        print("[CHOICES]")
        
        for i, (value, description) in enumerate(choices, 1):
            print(f"{i}. {description}")
        
        print(f"[END CHOICES] Enter 1-{len(choices)}")
        
        while True:
            try:
                choice_input = await self.input_handler.get_input("Choice (1-{}): ".format(len(choices)))
                choice_num = int(choice_input.strip())
                
                if 1 <= choice_num <= len(choices):
                    selected = choices[choice_num - 1][0]
                    print(f"[SELECTED] {choices[choice_num - 1][1]}")
                    return selected
                else:
                    print(f"[ERROR] Please enter a number between 1 and {len(choices)}")
                    
            except ValueError:
                print("[ERROR] Please enter a valid number")
            except KeyboardInterrupt:
                print("[CANCELLED] Setup cancelled by user")
                raise
```

### Phase 2: User Research Integration (Week 5)
**Goal: Incorporate user feedback and refine based on research**

#### 2.1 User Testing Results Integration
**Incorporate Research Findings:**

Based on user research, implement findings:
- Preferred interaction patterns
- Accessibility requirements
- Configuration complexity tolerance
- Feature prioritization

#### 2.2 Iterative UX Improvements
**Implement Research-Driven Changes:**

```python
# src/agentsmcp/ux/user_preferences.py
from typing import Dict, Any
import json
from pathlib import Path

class UserPreferenceManager:
    """Manage user preferences based on research findings"""
    
    def __init__(self):
        self.preferences_file = Path.home() / '.agentsmcp' / 'preferences.json'
        self.preferences = self._load_preferences()
    
    def _load_preferences(self) -> Dict[str, Any]:
        """Load user preferences with research-based defaults"""
        defaults = {
            "interface_style": "auto",  # auto, beginner, advanced
            "accessibility_level": "auto",  # auto, basic, full
            "help_verbosity": "contextual",  # minimal, contextual, detailed
            "error_recovery": "guided",  # auto, guided, manual
            "onboarding_completed": False
        }
        
        if self.preferences_file.exists():
            with open(self.preferences_file) as f:
                loaded = json.load(f)
                defaults.update(loaded)
        
        return defaults
    
    def adapt_interface_for_user(self) -> Dict[str, Any]:
        """Adapt interface based on user preferences and research"""
        adaptations = {
            "show_advanced_options": self.preferences["interface_style"] == "advanced",
            "enable_shortcuts": self.preferences["interface_style"] in ["advanced", "auto"],
            "verbose_feedback": self.preferences["help_verbosity"] in ["contextual", "detailed"],
            "guided_error_recovery": self.preferences["error_recovery"] == "guided"
        }
        
        return adaptations
```

### Phase 3: Modern Implementation (Week 6-7)
**Goal: Implement modern, accessible interfaces**

#### 3.1 Textual Framework Implementation with Accessibility

```python
# src/agentsmcp/ui/modern_tui/accessible_app.py
from textual.app import App
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Input, RichLog, Button
from textual.screen import Screen
from textual import events
from textual.reactive import reactive

class AccessibleTUIApp(App):
    """Modern TUI application with built-in accessibility"""
    
    CSS = """
    .high-contrast {
        background: black;
        color: white;
    }
    
    Input {
        border: thick white;
        background: $surface;
    }
    
    Input:focus {
        border: thick yellow;
        background: $primary;
    }
    
    .screen-reader-announcement {
        position: absolute;
        left: -10000px;
        width: 1px;
        height: 1px;
        overflow: hidden;
    }
    """
    
    accessibility_mode = reactive(False)
    high_contrast = reactive(False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.accessibility_mode = self._detect_accessibility_mode()
        self.high_contrast = self._detect_high_contrast()
        
    def _detect_accessibility_mode(self) -> bool:
        """Detect if accessibility mode should be enabled"""
        import os
        return (os.environ.get('ACCESSIBILITY_MODE') == '1' or
                os.environ.get('SCREENREADER') is not None)
    
    def compose(self):
        """Compose the UI with accessibility features"""
        
        yield Header(show_clock=not self.accessibility_mode)
        
        with Vertical():
            yield RichLog(
                id="chat",
                auto_scroll=True,
                markup=True,
                classes="accessible-log"
            )
            
            with Horizontal():
                yield Input(
                    placeholder="Type your message (Press F1 for help)...",
                    id="user-input",
                    classes="accessible-input"
                )
                yield Button(
                    "Send",
                    id="send-button",
                    variant="primary"
                )
        
        yield Footer()
    
    def on_mount(self):
        """Setup accessibility features on mount"""
        if self.accessibility_mode:
            self.announce("AgentsMCP TUI loaded. Focus is on input field.")
            
        # Set high contrast if needed
        if self.high_contrast:
            self.add_class("high-contrast")
        
        # Focus input field
        self.query_one("#user-input").focus()
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission"""
        message = event.value
        
        if not message.strip():
            if self.accessibility_mode:
                self.announce("Please enter a message")
            return
        
        # Clear input
        event.input.value = ""
        
        # Display user message
        chat_log = self.query_one("#chat")
        chat_log.write(f"[bold blue]You:[/] {message}")
        
        if self.accessibility_mode:
            self.announce(f"Sent message: {message}")
        
        # Process message (placeholder)
        await self._process_user_message(message)
    
    async def _process_user_message(self, message: str):
        """Process user message and generate response"""
        chat_log = self.query_one("#chat")
        
        if self.accessibility_mode:
            self.announce("Processing your request...")
        
        # Placeholder response
        chat_log.write(f"[bold green]Agent:[/] I received your message: {message}")
        
        if self.accessibility_mode:
            self.announce("Response received")
    
    def announce(self, message: str):
        """Announce message to screen readers"""
        if self.accessibility_mode:
            # Create invisible announcement element
            self.bell()  # Audio cue
            # In a real implementation, this would interface with screen reader APIs
    
    def on_key(self, event: events.Key) -> None:
        """Handle keyboard shortcuts with accessibility"""
        if event.key == "f1":
            self.show_help()
            event.prevent_default()
        elif event.key == "ctrl+h":
            self.show_help()
            event.prevent_default()
        elif event.key == "escape":
            self.exit()
    
    def show_help(self):
        """Show contextual help"""
        help_text = """
        [bold]AgentsMCP TUI Help[/]
        
        [yellow]Keyboard Shortcuts:[/]
        • F1 or Ctrl+H: Show this help
        • Enter: Send message
        • Escape: Exit application
        • Tab/Shift+Tab: Navigate elements
        
        [yellow]Accessibility:[/]
        • Full keyboard navigation supported
        • Screen reader announcements active
        • High contrast mode available
        
        Press any key to close this help.
        """
        
        chat_log = self.query_one("#chat")
        chat_log.write(help_text)
        
        if self.accessibility_mode:
            self.announce("Help displayed. Press any key to continue.")
```

### Phase 4: Testing & Quality Assurance (Week 8)
**Goal: Comprehensive testing and quality validation**

#### 4.1 Accessibility Testing Suite

```python
# tests/accessibility/test_wcag_compliance.py
import pytest
import asyncio
from textual.app import App
from agentsmcp.ui.modern_tui.accessible_app import AccessibleTUIApp

class TestWCAGCompliance:
    """Test WCAG 2.2 AA compliance"""
    
    @pytest.mark.accessibility
    def test_keyboard_navigation(self):
        """Test that all UI elements are keyboard accessible"""
        app = AccessibleTUIApp()
        
        # Test tab navigation
        # Test that all interactive elements can be reached by keyboard
        # This would require integration with accessibility testing tools
        pass
    
    @pytest.mark.accessibility
    def test_screen_reader_announcements(self):
        """Test screen reader announcement functionality"""
        app = AccessibleTUIApp()
        
        # Mock screen reader API
        with pytest.patch.object(app, 'announce') as mock_announce:
            app.on_mount()
            mock_announce.assert_called_with("AgentsMCP TUI loaded. Focus is on input field.")
    
    @pytest.mark.accessibility
    def test_high_contrast_mode(self):
        """Test high contrast mode functionality"""
        with pytest.patch.dict('os.environ', {'ACCESSIBILITY_HIGH_CONTRAST': '1'}):
            app = AccessibleTUIApp()
            assert app.high_contrast == True
    
    @pytest.mark.accessibility
    def test_color_contrast_ratios(self):
        """Test that color combinations meet WCAG contrast requirements"""
        # This would integrate with color contrast testing tools
        # to verify all color combinations meet 4.5:1 ratio for AA compliance
        pass
```

## Success Metrics - Revised

### Week 1 Targets (Research & Foundation)
- ✅ User research completed with actionable insights
- ✅ Hexagonal architecture foundation implemented
- ✅ Event-driven system established
- ✅ WCAG 2.2 AA compliance framework in place

### Week 2-4 Targets (Critical Fixes with Accessibility)
- ✅ TUI typing works correctly with accessibility support
- ✅ Missing function implementations completed
- ✅ Comprehensive testing framework operational
- ✅ Interface contracts (ICDs) established
- ✅ Accessible setup wizard functional

### Week 5 Targets (User Research Integration)  
- ✅ User feedback incorporated into design
- ✅ Preference management system implemented
- ✅ Interface adapts based on user research findings

### Week 6-7 Targets (Modern Implementation)
- ✅ Modern Textual-based TUI with accessibility
- ✅ Full keyboard navigation support
- ✅ Screen reader compatibility verified
- ✅ High contrast mode functional

### Week 8 Targets (Testing & QA)
- ✅ WCAG 2.2 AA compliance verified
- ✅ Comprehensive accessibility testing suite
- ✅ Performance targets met (<1s startup, <100ms response)
- ✅ User acceptance testing passed

## Resource Requirements - Updated

### Development Effort
- **Phase 0**: 1 UX researcher + 1 architect × 1 week = 2 dev-weeks
- **Phase 1**: 2 developers × 3 weeks = 6 dev-weeks (Extended from 2 weeks)
- **Phase 2**: 1 UX researcher + 1 developer × 1 week = 2 dev-weeks
- **Phase 3**: 2 developers × 2 weeks = 4 dev-weeks
- **Phase 4**: 1 QA engineer + 1 accessibility specialist × 1 week = 2 dev-weeks
- **Total**: 16 dev-weeks (vs 14 in original plan)

### New Dependencies
- `accessibility-tools>=1.0.0` - Accessibility testing framework
- `wcag-color-contrast>=1.0.0` - Color contrast validation
- `screen-reader-testing>=0.5.0` - Screen reader simulation

### Specialized Skills Required
- **UX Researcher**: User interview and analysis skills
- **Accessibility Specialist**: WCAG compliance expertise  
- **QA Engineer**: Accessibility testing experience

## Risk Mitigation - Enhanced

### Technical Risks (Updated)
- **Risk**: Hexagonal architecture refactoring complexity
  **Mitigation**: Implement incrementally with feature flags, maintain backward compatibility

- **Risk**: Event-driven architecture performance impact
  **Mitigation**: Implement with performance monitoring, async optimization

- **Risk**: Accessibility compliance verification challenges
  **Mitigation**: Use automated tools + manual testing, accessibility expert review

### User Impact Risks (Updated)
- **Risk**: Extended timeline delays user benefits
  **Mitigation**: Release Phase 1 fixes immediately, subsequent phases as updates

- **Risk**: User research reveals fundamental design flaws
  **Mitigation**: Build flexible architecture that can adapt to research findings

## Implementation Schedule - Revised

### Week 1: Research & Architecture Foundation
- **Days 1-3**: User research (interviews, analytics, accessibility audit)
- **Days 4-5**: Implement hexagonal architecture and event system

### Week 2: Critical Implementation Start
- **Day 1**: Implement missing functions (`_apply_ansi_markdown`)
- **Days 2-3**: Build unified input system with accessibility
- **Days 4-5**: Create testing framework and initial ICDs

### Week 3: Testing & Configuration
- **Days 1-3**: Comprehensive testing suite implementation
- **Days 4-5**: Accessible setup wizard development

### Week 4: Interface Contracts & Integration
- **Days 1-3**: Complete ICDs and interface contracts
- **Days 4-5**: Integration testing and refinement

### Week 5: User Research Integration
- **Days 1-2**: Analyze user research findings
- **Days 3-5**: Implement preference management and UX adaptations

### Week 6-7: Modern Implementation
- **Week 6**: Textual framework implementation with accessibility
- **Week 7**: Advanced features and responsive design

### Week 8: Quality Assurance & Launch
- **Days 1-3**: WCAG compliance testing and accessibility validation
- **Days 4-5**: Final integration testing and launch preparation

## Conclusion - Revised

This revised plan addresses critical feedback from specialized agents while maintaining focus on the core goal of fixing AgentsMCP's usability issues. Key improvements include:

**From QA Agent Feedback:**
- Extended timeline for proper implementation
- Added missing function implementations
- Comprehensive testing framework from Day 1
- Fixed architectural approach with proper separation

**From UX Agent Feedback:**  
- Added mandatory user research phase
- Moved accessibility to Phase 1 instead of afterthought
- Incorporated user feedback loop and preference management
- Realistic timeline with proper UX methodology

**From System Architect Feedback:**
- Implemented hexagonal architecture pattern
- Added event-driven design for reactive updates
- Established formal interface contracts (ICDs)
- Proper modular design with clear boundaries

**Success depends on:**
1. **User-Centric Approach**: Research-driven design decisions
2. **Accessibility First**: WCAG compliance from Day 1, not as afterthought  
3. **Proper Architecture**: Hexagonal pattern enables maintainable, testable code
4. **Comprehensive Testing**: Quality assurance throughout development
5. **Iterative Improvement**: User feedback integration and continuous refinement

This plan transforms AgentsMCP into a truly accessible, user-friendly platform while establishing proper architectural foundations for long-term maintainability and extensibility.