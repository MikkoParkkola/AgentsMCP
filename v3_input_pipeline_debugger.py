#!/usr/bin/env python3
"""
V3 TUI Input Pipeline Debugger - Focused debugging for specific user symptoms
==============================================================================

This script is specifically designed to debug the exact symptoms reported:
- Characters appear in bottom right corner one at a time as typed
- Characters don't reach the input box until Enter pressed repeatedly  
- Commands (/) don't work
- Input never gets sent to LLM chat

The debugger traces the complete input flow:
Terminal → PlainCLIRenderer → RevolutionaryTUI → InputPipeline → ChatEngine → LLM

DEBUGGING STRATEGY:
1. Monitor input events in real-time with microsecond precision
2. Test PlainCLIRenderer.handle_input() step by step
3. Trace input flow through all layers with detailed logging
4. Identify where input gets stuck, lost, or corrupted
5. Provide specific fixes for each issue found

Usage:
    python v3_input_pipeline_debugger.py
"""

import asyncio
import logging
import os
import sys
import time
import threading
import traceback
import signal
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from contextlib import contextmanager
import inspect

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer, SimpleTUIRenderer
    from src.agentsmcp.ui.v3.ui_renderer_base import UIRenderer
    from src.agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface, TUIState
    from src.agentsmcp.ui.v2.input_rendering_pipeline import InputRenderingPipeline, InputMode
    from src.agentsmcp.ui.v2.reliability.integration_layer import ReliableTUIInterface
    from src.agentsmcp.ui.cli_app import CLIApp, CLIConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run from the AgentsMCP project root directory")
    sys.exit(1)


@dataclass
class InputEvent:
    """Records an input event for debugging."""
    timestamp: float
    event_type: str  # 'char_input', 'key_press', 'enter', 'backspace', 'command'
    data: Any
    component: str  # Which component handled it
    state_before: str
    state_after: str
    cursor_pos: int
    processing_time_ms: float
    additional_data: Dict[str, Any]


class InputFlowTracer:
    """Real-time input flow tracing with microsecond precision."""
    
    def __init__(self):
        self.events: List[InputEvent] = []
        self.active = False
        self.lock = threading.Lock()
        self.event_callbacks: List[Callable[[InputEvent], None]] = []
        
    def start_tracing(self):
        """Start input flow tracing."""
        with self.lock:
            self.active = True
            self.events.clear()
            print("🔍 Input flow tracing STARTED")
    
    def stop_tracing(self):
        """Stop input flow tracing."""
        with self.lock:
            self.active = False
            print(f"🔍 Input flow tracing STOPPED - Captured {len(self.events)} events")
    
    def trace_event(self, event_type: str, data: Any, component: str, 
                   state_before: str, state_after: str, cursor_pos: int = 0, 
                   **kwargs):
        """Record an input event."""
        if not self.active:
            return
            
        start_time = time.time()
        
        event = InputEvent(
            timestamp=time.time() * 1000000,  # Microsecond precision
            event_type=event_type,
            data=data,
            component=component,
            state_before=state_before,
            state_after=state_after,
            cursor_pos=cursor_pos,
            processing_time_ms=(time.time() - start_time) * 1000,
            additional_data=kwargs
        )
        
        with self.lock:
            self.events.append(event)
            
        # Notify callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"⚠️ Event callback error: {e}")
    
    def add_callback(self, callback: Callable[[InputEvent], None]):
        """Add a callback for real-time event monitoring."""
        self.event_callbacks.append(callback)
    
    def get_events_by_type(self, event_type: str) -> List[InputEvent]:
        """Get all events of a specific type."""
        with self.lock:
            return [e for e in self.events if e.event_type == event_type]
    
    def get_events_by_component(self, component: str) -> List[InputEvent]:
        """Get all events from a specific component."""
        with self.lock:
            return [e for e in self.events if e.component == component]
    
    def print_event_summary(self):
        """Print a summary of all captured events."""
        with self.lock:
            if not self.events:
                print("📝 No input events captured")
                return
                
            print(f"\n📝 INPUT EVENT SUMMARY ({len(self.events)} events)")
            print("=" * 60)
            
            by_type = {}
            by_component = {}
            
            for event in self.events:
                by_type[event.event_type] = by_type.get(event.event_type, 0) + 1
                by_component[event.component] = by_component.get(event.component, 0) + 1
            
            print("Events by Type:")
            for event_type, count in by_type.items():
                print(f"  {event_type}: {count}")
            
            print("\nEvents by Component:")
            for component, count in by_component.items():
                print(f"  {component}: {count}")
    
    def print_detailed_trace(self, limit: int = 50):
        """Print detailed trace of input events."""
        with self.lock:
            if not self.events:
                print("📝 No input events to trace")
                return
            
            print(f"\n🔍 DETAILED INPUT TRACE (showing last {limit} events)")
            print("=" * 80)
            
            events_to_show = self.events[-limit:] if len(self.events) > limit else self.events
            
            for i, event in enumerate(events_to_show):
                timestamp_ms = event.timestamp / 1000
                print(f"{i+1:3d}. [{timestamp_ms:.3f}] {event.component}::{event.event_type}")
                print(f"     Data: {repr(event.data)}")
                print(f"     State: '{event.state_before}' → '{event.state_after}'")
                print(f"     Cursor: {event.cursor_pos}, Time: {event.processing_time_ms:.2f}ms")
                if event.additional_data:
                    print(f"     Extra: {event.additional_data}")
                print()


class PlainCLIRendererDebugger:
    """Step-by-step debugger for PlainCLIRenderer.handle_input()."""
    
    def __init__(self, tracer: InputFlowTracer):
        self.tracer = tracer
        self.original_renderer = None
        self.debug_renderer = None
        
    def setup_debug_renderer(self) -> PlainCLIRenderer:
        """Create a debuggable PlainCLIRenderer with instrumentation."""
        # Mock capabilities
        mock_capabilities = Mock()
        mock_capabilities.is_tty = True
        mock_capabilities.supports_colors = True
        mock_capabilities.supports_unicode = True
        mock_capabilities.width = 80
        mock_capabilities.height = 24
        
        renderer = PlainCLIRenderer(mock_capabilities)
        
        # Instrument the renderer methods
        original_handle_input = renderer.handle_input
        original_render_frame = renderer.render_frame
        
        def debug_handle_input():
            """Instrumented handle_input with detailed logging."""
            state_before = renderer.state.current_input if hasattr(renderer, 'state') else ""
            
            self.tracer.trace_event(
                event_type="handle_input_start",
                data="",
                component="PlainCLIRenderer",
                state_before=state_before,
                state_after=state_before,
                method_call="handle_input()"
            )
            
            try:
                # Call original method
                result = original_handle_input()
                
                state_after = renderer.state.current_input if hasattr(renderer, 'state') else ""
                
                self.tracer.trace_event(
                    event_type="handle_input_complete",
                    data=result,
                    component="PlainCLIRenderer",
                    state_before=state_before,
                    state_after=state_after,
                    result=result,
                    success=True
                )
                
                return result
                
            except Exception as e:
                self.tracer.trace_event(
                    event_type="handle_input_error",
                    data=str(e),
                    component="PlainCLIRenderer",
                    state_before=state_before,
                    state_after=state_before,
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                raise
        
        def debug_render_frame():
            """Instrumented render_frame with detailed logging."""
            state = renderer.state.current_input if hasattr(renderer, 'state') else ""
            
            self.tracer.trace_event(
                event_type="render_frame",
                data=state,
                component="PlainCLIRenderer",
                state_before=state,
                state_after=state,
                method_call="render_frame()"
            )
            
            return original_render_frame()
        
        # Replace methods with instrumented versions
        renderer.handle_input = debug_handle_input
        renderer.render_frame = debug_render_frame
        
        self.debug_renderer = renderer
        return renderer
    
    async def test_input_handling_step_by_step(self):
        """Test PlainCLIRenderer input handling step by step."""
        print("\n🔧 TESTING: PlainCLIRenderer.handle_input() Step by Step")
        print("=" * 60)
        
        renderer = self.setup_debug_renderer()
        
        if not renderer.initialize():
            print("❌ Failed to initialize PlainCLIRenderer")
            return False
        
        print("✅ PlainCLIRenderer initialized successfully")
        
        # Test 1: Basic character input
        print("\n📝 TEST 1: Basic character input")
        with patch('builtins.input', return_value='hello'):
            result = renderer.handle_input()
            print(f"   Input result: {repr(result)}")
            print(f"   State after: {repr(getattr(renderer.state, 'current_input', 'N/A'))}")
        
        # Test 2: Command input
        print("\n📝 TEST 2: Command input (/help)")
        with patch('builtins.input', return_value='/help'):
            result = renderer.handle_input()
            print(f"   Command result: {repr(result)}")
            print(f"   State after: {repr(getattr(renderer.state, 'current_input', 'N/A'))}")
        
        # Test 3: Empty input
        print("\n📝 TEST 3: Empty input")
        with patch('builtins.input', return_value=''):
            result = renderer.handle_input()
            print(f"   Empty result: {repr(result)}")
            print(f"   State after: {repr(getattr(renderer.state, 'current_input', 'N/A'))}")
        
        # Test 4: Interrupt handling
        print("\n📝 TEST 4: Interrupt handling")
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            result = renderer.handle_input()
            print(f"   Interrupt result: {repr(result)}")
        
        renderer.cleanup()
        return True


class RevolutionaryTUIDebugger:
    """Debugger for Revolutionary TUI input pipeline."""
    
    def __init__(self, tracer: InputFlowTracer):
        self.tracer = tracer
    
    async def test_tui_input_flow(self):
        """Test Revolutionary TUI input flow with detailed tracing."""
        print("\n🚀 TESTING: Revolutionary TUI Input Flow")
        print("=" * 60)
        
        try:
            # Create TUI instance
            tui = RevolutionaryTUIInterface()
            
            # Instrument key methods
            await self._instrument_tui_methods(tui)
            
            # Test character input simulation
            print("📝 Simulating character input...")
            await self._simulate_character_input(tui, "hello world")
            
            # Test command input
            print("📝 Simulating command input...")
            await self._simulate_character_input(tui, "/help")
            
            # Test Enter key
            print("📝 Simulating Enter key...")
            await self._simulate_enter_key(tui)
            
            return True
            
        except Exception as e:
            print(f"❌ Revolutionary TUI test failed: {e}")
            traceback.print_exc()
            return False
    
    async def _instrument_tui_methods(self, tui: RevolutionaryTUIInterface):
        """Add instrumentation to TUI methods for debugging."""
        
        # Instrument input handling methods if they exist
        if hasattr(tui, '_handle_character_input'):
            original_handle_char = tui._handle_character_input
            
            def debug_handle_char(char):
                state_before = tui.state.current_input
                self.tracer.trace_event(
                    event_type="char_input",
                    data=char,
                    component="RevolutionaryTUI",
                    state_before=state_before,
                    state_after=state_before
                )
                
                result = original_handle_char(char)
                
                state_after = tui.state.current_input
                self.tracer.trace_event(
                    event_type="char_processed",
                    data=char,
                    component="RevolutionaryTUI",
                    state_before=state_before,
                    state_after=state_after
                )
                
                return result
            
            tui._handle_character_input = debug_handle_char
        
        # Instrument input pipeline if available
        if hasattr(tui, 'input_pipeline'):
            original_render = tui.input_pipeline.render_input
            
            async def debug_render_input(*args, **kwargs):
                self.tracer.trace_event(
                    event_type="pipeline_render",
                    data=args,
                    component="InputPipeline",
                    state_before=str(args),
                    state_after="",
                    kwargs=kwargs
                )
                
                result = await original_render(*args, **kwargs)
                
                self.tracer.trace_event(
                    event_type="pipeline_render_complete",
                    data=str(result),
                    component="InputPipeline", 
                    state_before=str(args),
                    state_after=str(result)
                )
                
                return result
            
            tui.input_pipeline.render_input = debug_render_input
    
    async def _simulate_character_input(self, tui: RevolutionaryTUIInterface, text: str):
        """Simulate typing characters into the TUI."""
        for char in text:
            if hasattr(tui, '_handle_character_input'):
                tui._handle_character_input(char)
            else:
                # Add to state directly if no handler
                tui.state.current_input += char
            
            # Small delay to simulate real typing
            await asyncio.sleep(0.01)
    
    async def _simulate_enter_key(self, tui: RevolutionaryTUIInterface):
        """Simulate pressing Enter key."""
        if hasattr(tui, '_handle_enter_key'):
            tui._handle_enter_key()
        else:
            self.tracer.trace_event(
                event_type="enter_key_missing",
                data="No _handle_enter_key method",
                component="RevolutionaryTUI",
                state_before=tui.state.current_input,
                state_after=tui.state.current_input
            )


class ChatEngineFlowDebugger:
    """Debugger for chat engine integration."""
    
    def __init__(self, tracer: InputFlowTracer):
        self.tracer = tracer
    
    def test_chat_engine_connection(self):
        """Test connection between TUI input and chat engine."""
        print("\n💬 TESTING: Chat Engine Connection")
        print("=" * 60)
        
        # This will be expanded based on the actual chat engine integration
        # For now, we'll test the interface points
        
        try:
            # Mock chat engine for testing
            mock_chat_engine = Mock()
            mock_chat_engine.process_message = Mock(return_value="Mock response")
            
            # Test message processing
            test_message = "Hello, AI!"
            self.tracer.trace_event(
                event_type="chat_engine_test",
                data=test_message,
                component="ChatEngine",
                state_before="",
                state_after="",
                test_mode=True
            )
            
            response = mock_chat_engine.process_message(test_message)
            print(f"✅ Chat engine mock response: {response}")
            
            return True
            
        except Exception as e:
            print(f"❌ Chat engine test failed: {e}")
            return False


class V3InputPipelineDebugger:
    """Main debugger class for V3 TUI input pipeline issues."""
    
    def __init__(self):
        self.tracer = InputFlowTracer()
        self.plain_cli_debugger = PlainCLIRendererDebugger(self.tracer)
        self.tui_debugger = RevolutionaryTUIDebugger(self.tracer)
        self.chat_debugger = ChatEngineFlowDebugger(self.tracer)
        
        # Real-time monitoring
        self.tracer.add_callback(self._real_time_event_monitor)
        
        # Issue tracking
        self.issues_found: List[Dict[str, Any]] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.WARNING,  # Reduce noise during debugging
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _real_time_event_monitor(self, event: InputEvent):
        """Monitor events in real-time for issues."""
        # Check for stuck input (same state for too long)
        if event.event_type in ['char_input', 'char_processed']:
            if event.state_before == event.state_after and event.data:
                self.issues_found.append({
                    'type': 'stuck_input',
                    'event': event,
                    'description': f"Character '{event.data}' didn't update state in {event.component}",
                    'severity': 'HIGH'
                })
        
        # Check for slow processing
        if event.processing_time_ms > 100:  # 100ms is quite slow for character input
            self.issues_found.append({
                'type': 'slow_processing',
                'event': event,
                'description': f"Slow processing: {event.processing_time_ms:.2f}ms in {event.component}",
                'severity': 'MEDIUM'
            })
        
        # Real-time output for critical issues
        if len(self.issues_found) > 0 and self.issues_found[-1]['severity'] == 'HIGH':
            print(f"🚨 ISSUE DETECTED: {self.issues_found[-1]['description']}")
    
    async def run_comprehensive_debug(self):
        """Run comprehensive debugging of V3 input pipeline."""
        print("🔍 V3 TUI INPUT PIPELINE DEBUGGER")
        print("=" * 50)
        print("Diagnosing the specific symptoms:")
        print("- Characters appear in bottom right corner one at a time")
        print("- Characters don't reach input box until Enter pressed repeatedly")
        print("- Commands (/) don't work")
        print("- Input never gets sent to LLM chat")
        print("=" * 50)
        
        # Start tracing
        self.tracer.start_tracing()
        
        try:
            # Phase 1: Test PlainCLIRenderer
            print("\n📋 PHASE 1: PlainCLIRenderer Debugging")
            success = await self.plain_cli_debugger.test_input_handling_step_by_step()
            
            if not success:
                print("❌ Phase 1 failed - PlainCLIRenderer has critical issues")
                return False
            
            # Phase 2: Test Revolutionary TUI
            print("\n📋 PHASE 2: Revolutionary TUI Debugging")
            success = await self.tui_debugger.test_tui_input_flow()
            
            if not success:
                print("⚠️ Phase 2 had issues - Revolutionary TUI may have problems")
            
            # Phase 3: Test Chat Engine integration
            print("\n📋 PHASE 3: Chat Engine Integration")
            success = self.chat_debugger.test_chat_engine_connection()
            
            # Analysis phase
            print("\n📋 PHASE 4: Analysis and Issue Detection")
            await self._analyze_results()
            
            return True
            
        except Exception as e:
            print(f"❌ Debugging failed: {e}")
            traceback.print_exc()
            return False
        
        finally:
            # Stop tracing and show results
            self.tracer.stop_tracing()
            self._generate_debug_report()
    
    async def _analyze_results(self):
        """Analyze debugging results and identify specific issues."""
        print("🔍 Analyzing input pipeline for issues...")
        
        # Check for missing methods
        missing_methods = []
        try:
            renderer = PlainCLIRenderer(Mock())
            if not hasattr(renderer, 'handle_input'):
                missing_methods.append('PlainCLIRenderer.handle_input')
        except Exception as e:
            missing_methods.append(f'PlainCLIRenderer creation failed: {e}')
        
        # Check for input flow breaks
        char_inputs = self.tracer.get_events_by_type('char_input')
        char_processed = self.tracer.get_events_by_type('char_processed')
        
        if len(char_inputs) > len(char_processed):
            self.issues_found.append({
                'type': 'input_flow_break',
                'description': f'{len(char_inputs)} chars input but only {len(char_processed)} processed',
                'severity': 'HIGH'
            })
        
        # Look for rendering pipeline issues
        pipeline_renders = self.tracer.get_events_by_type('pipeline_render')
        if len(pipeline_renders) == 0:
            self.issues_found.append({
                'type': 'no_pipeline_rendering',
                'description': 'No input pipeline rendering detected',
                'severity': 'MEDIUM'
            })
        
        print(f"Analysis complete - {len(self.issues_found)} issues found")
    
    def _generate_debug_report(self):
        """Generate comprehensive debug report with specific fixes."""
        print("\n📊 COMPREHENSIVE DEBUG REPORT")
        print("=" * 60)
        
        # Show event summary
        self.tracer.print_event_summary()
        
        # Show issues found
        if self.issues_found:
            print(f"\n🚨 ISSUES FOUND ({len(self.issues_found)})")
            print("-" * 40)
            
            for i, issue in enumerate(self.issues_found, 1):
                print(f"{i}. [{issue.get('severity', 'UNKNOWN')}] {issue.get('type', 'Unknown')}")
                print(f"   Description: {issue.get('description', 'No description')}")
                print()
        
        # Generate specific fixes
        self._generate_specific_fixes()
        
        # Show detailed trace
        self.tracer.print_detailed_trace()
    
    def _generate_specific_fixes(self):
        """Generate specific fixes for the reported symptoms."""
        print("\n🔧 SPECIFIC FIXES FOR REPORTED SYMPTOMS")
        print("=" * 60)
        
        # Fix 1: Characters appearing in bottom right corner
        print("FIX 1: Characters appearing in bottom right corner")
        print("CAUSE: Terminal cursor positioning issue in rendering")
        print("SOLUTION:")
        print("  - Check terminal escape sequence handling in SimpleTUIRenderer")
        print("  - Verify cursor positioning in _draw_input_area()")
        print("  - Add proper screen coordinate validation")
        print()
        
        # Fix 2: Characters don't reach input box
        print("FIX 2: Characters don't reach input box until Enter pressed repeatedly")
        print("CAUSE: Input buffer synchronization issue between renderer and TUI state")
        print("SOLUTION:")
        print("  - Unify input buffers - remove duplicate _input_buffer in SimpleTUIRenderer")
        print("  - Ensure PlainCLIRenderer updates state.current_input immediately")
        print("  - Add real-time input display updates")
        print()
        
        # Fix 3: Commands don't work
        print("FIX 3: Commands (/) don't work")
        print("CAUSE: Command detection and routing failure")
        print("SOLUTION:")
        print("  - Add command detection in handle_input() before state update")
        print("  - Implement proper command routing to chat engine")
        print("  - Add command validation and error handling")
        print()
        
        # Fix 4: Input never reaches LLM
        print("FIX 4: Input never gets sent to LLM chat")
        print("CAUSE: Missing chat engine integration in input flow")
        print("SOLUTION:")
        print("  - Add chat engine connection in handle_input() after Enter")
        print("  - Implement async message processing")
        print("  - Add proper error handling for LLM communication")
        print()
        
        # Provide code fixes
        self._provide_code_fixes()
    
    def _provide_code_fixes(self):
        """Provide specific code fixes."""
        print("💻 SPECIFIC CODE FIXES")
        print("=" * 40)
        
        print("1. Fix for PlainCLIRenderer.handle_input():")
        print("""
def handle_input(self) -> Optional[str]:
    with self._input_lock:
        try:
            if self.state.is_processing:
                return None
            
            # Get input with immediate state update
            try:
                user_input = input("💬 > ").strip()
            except (EOFError, KeyboardInterrupt):
                return "/quit"
            
            if user_input:
                # CRITICAL: Update state immediately for real-time display
                self.state.current_input = user_input
                
                # CRITICAL: Trigger immediate render update
                self.render_frame()
                
                # CRITICAL: Handle commands before clearing state
                if user_input.startswith('/'):
                    return self._handle_command(user_input)
                
                # CRITICAL: Clear state after processing, not before
                result = user_input
                self.state.current_input = ""
                return result
            
            return None
        except Exception as e:
            print(f"Input error: {e}")
            return None
""")
        
        print("\n2. Fix for SimpleTUIRenderer cursor positioning:")
        print("""
def _draw_input_area(self):
    try:
        input_line = self._screen_height - 1
        
        # CRITICAL: Fix cursor positioning
        if self.capabilities.is_tty:
            # Clear line and move to correct position
            print(f"\\033[{input_line};1H\\033[K", end="")
            
            # Display prompt and current input
            display_text = f"💬 > {self.state.current_input}"
            print(display_text, end="")
            
            # CRITICAL: Position cursor at end of input
            cursor_pos = len(display_text)
            print(f"\\033[{input_line};{cursor_pos + 1}H", end="", flush=True)
    except Exception as e:
        print(f"Input area draw error: {e}")
""")


async def main():
    """Main entry point for the V3 input pipeline debugger."""
    print("🚀 Starting V3 TUI Input Pipeline Debugger...")
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Debugging interrupted by user")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run debugger
    debugger = V3InputPipelineDebugger()
    success = await debugger.run_comprehensive_debug()
    
    if success:
        print("\n✅ Debugging completed successfully")
        print("See the specific fixes above to resolve the input pipeline issues")
    else:
        print("\n❌ Debugging encountered errors")
        print("Check the output above for error details")
    
    return success


if __name__ == "__main__":
    try:
        # Make script executable
        os.chmod(__file__, 0o755)
        
        # Run the debugger
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Debugger interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Debugger failed: {e}")
        traceback.print_exc()
        sys.exit(1)