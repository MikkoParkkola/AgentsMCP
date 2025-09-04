#!/usr/bin/env python3
"""
V3 Command Processing Workflow Debugger
Traces command flow from input to execution in V3 TUI

Debugs the complete command workflow:
1. User types command (e.g., "/help")
2. PlainCLIRenderer.handle_input() captures it
3. TUILauncher main loop receives it
4. ChatEngine.process_input() handles it
5. Command execution and response
6. Callback system updates UI
7. Renderer displays result

This helps identify where commands get stuck or fail.
"""

import sys
import os
import asyncio
import time
import threading
from typing import Optional, List, Dict, Any, Callable
from unittest.mock import patch, Mock
from datetime import datetime

def debugger_header():
    print("=" * 80)
    print("⚙️ V3 Command Processing Workflow Debugger")
    print("=" * 80)
    print("Tracing command flow through the entire V3 TUI pipeline\n")

class WorkflowTracer:
    """Traces command workflow through V3 components."""
    
    def __init__(self):
        self.trace_events = []
        self.start_time = time.time()
        self.component_states = {}
    
    def trace(self, component: str, event: str, data: Any = None, duration_ms: float = None):
        """Record a trace event."""
        event_record = {
            "timestamp": time.time() - self.start_time,
            "component": component,
            "event": event,
            "data": data,
            "duration_ms": duration_ms,
            "datetime": datetime.now().isoformat()
        }
        self.trace_events.append(event_record)
        
        # Show real-time trace
        duration_str = f" ({duration_ms:.1f}ms)" if duration_ms else ""
        data_str = f": {data}" if data else ""
        print(f"[{event_record['timestamp']:7.3f}s] {component} → {event}{duration_str}{data_str}")
    
    def get_trace_summary(self):
        """Get summary of trace events."""
        if not self.trace_events:
            return "No trace events recorded"
        
        components = set(event['component'] for event in self.trace_events)
        total_time = self.trace_events[-1]['timestamp']
        
        return {
            "total_events": len(self.trace_events),
            "components_involved": list(components),
            "total_duration": round(total_time, 3),
            "avg_event_interval": round(total_time / len(self.trace_events), 3) if self.trace_events else 0
        }

class CommandWorkflowDebugger:
    """Main command workflow debugging system."""
    
    def __init__(self):
        self.tracer = WorkflowTracer()
        self.components = {}
        self.mock_setup_complete = False
    
    def setup_traced_components(self):
        """Set up V3 components with tracing."""
        self.tracer.trace("setup", "initializing_components")
        
        try:
            from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
            from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
            from agentsmcp.ui.v3.chat_engine import ChatEngine
            from agentsmcp.ui.v3.tui_launcher import TUILauncher
            
            # Detect terminal capabilities
            caps = detect_terminal_capabilities()
            self.tracer.trace("terminal_capabilities", "detected", {
                "is_tty": caps.is_tty,
                "supports_rich": caps.supports_rich
            })
            
            # Create renderer with tracing
            renderer = TracedPlainCLIRenderer(caps, self.tracer)
            self.components['renderer'] = renderer
            self.tracer.trace("renderer", "created", renderer.__class__.__name__)
            
            # Create chat engine with tracing
            chat_engine = TracedChatEngine(self.tracer)
            self.components['chat_engine'] = chat_engine
            self.tracer.trace("chat_engine", "created")
            
            # Create launcher with tracing
            launcher = TracedTUILauncher(self.tracer)
            self.components['launcher'] = launcher
            self.tracer.trace("launcher", "created")
            
            self.mock_setup_complete = True
            self.tracer.trace("setup", "components_ready")
            return True
            
        except ImportError as e:
            self.tracer.trace("setup", "import_error", str(e))
            return False
        except Exception as e:
            self.tracer.trace("setup", "setup_error", str(e))
            return False
    
    def test_command_workflow(self, command: str):
        """Test complete command workflow with tracing."""
        print(f"\n🔍 TRACING COMMAND WORKFLOW: '{command}'")
        print("-" * 60)
        
        if not self.mock_setup_complete:
            print("❌ Components not set up properly")
            return False
        
        self.tracer.trace("test", "starting_command_test", command)
        
        try:
            # Step 1: Simulate user input through renderer
            renderer = self.components['renderer']
            chat_engine = self.components['chat_engine']
            
            # Step 2: Simulate renderer input handling
            start_time = time.time()
            with patch('builtins.input', return_value=command):
                user_input = renderer.handle_input()
            input_duration = (time.time() - start_time) * 1000
            
            self.tracer.trace("renderer", "input_captured", user_input, input_duration)
            
            if user_input != command.strip():
                self.tracer.trace("test", "input_mismatch", {
                    "expected": command.strip(),
                    "actual": user_input
                })
                return False
            
            # Step 3: Process through chat engine
            start_time = time.time()
            result = asyncio.run(chat_engine.process_input(user_input))
            processing_duration = (time.time() - start_time) * 1000
            
            self.tracer.trace("chat_engine", "processing_complete", result, processing_duration)
            
            # Step 4: Verify expected behavior
            if command == "/quit":
                expected_result = False
            else:
                expected_result = True
            
            if result == expected_result:
                self.tracer.trace("test", "workflow_success", f"result_matches_expected_{expected_result}")
                return True
            else:
                self.tracer.trace("test", "workflow_failure", {
                    "expected": expected_result,
                    "actual": result
                })
                return False
                
        except Exception as e:
            self.tracer.trace("test", "workflow_exception", str(e))
            return False
        finally:
            self.tracer.trace("test", "command_test_complete", command)
    
    def run_comprehensive_command_test(self):
        """Run comprehensive test of all command types."""
        print("\n📋 COMPREHENSIVE COMMAND WORKFLOW TEST")
        print("=" * 60)
        
        if not self.setup_traced_components():
            print("❌ Failed to set up components for testing")
            return {}
        
        # Define test commands
        test_commands = [
            ("/help", "Help command"),
            ("/status", "Status command"),
            ("/clear", "Clear command"),
            ("hello world", "Regular message"),
            ("/nonexistent", "Unknown command"),
            ("/quit", "Quit command")
        ]
        
        results = {}
        
        for command, description in test_commands:
            print(f"\n{'='*20} {description} {'='*20}")
            
            success = self.test_command_workflow(command)
            results[command] = success
            
            if success:
                print(f"✅ {description} workflow completed successfully")
            else:
                print(f"❌ {description} workflow failed")
            
            # Small delay between tests
            time.sleep(0.1)
        
        return results
    
    def analyze_workflow_bottlenecks(self):
        """Analyze trace events for performance bottlenecks."""
        print("\n📊 WORKFLOW BOTTLENECK ANALYSIS")
        print("-" * 40)
        
        if not self.tracer.trace_events:
            print("No trace events to analyze")
            return
        
        # Group events by component
        component_events = {}
        for event in self.tracer.trace_events:
            component = event['component']
            if component not in component_events:
                component_events[component] = []
            component_events[component].append(event)
        
        print("📈 Events by component:")
        for component, events in component_events.items():
            avg_duration = 0
            duration_events = [e for e in events if e.get('duration_ms')]
            if duration_events:
                avg_duration = sum(e['duration_ms'] for e in duration_events) / len(duration_events)
            
            print(f"  {component}: {len(events)} events, avg duration: {avg_duration:.1f}ms")
        
        # Find longest duration events
        duration_events = [e for e in self.tracer.trace_events if e.get('duration_ms')]
        if duration_events:
            slowest = sorted(duration_events, key=lambda x: x['duration_ms'], reverse=True)[:5]
            
            print("\n🐌 Slowest operations:")
            for event in slowest:
                print(f"  {event['component']} → {event['event']}: {event['duration_ms']:.1f}ms")
        
        # Timeline analysis
        print(f"\n⏱️ Timeline analysis:")
        print(f"  First event: {self.tracer.trace_events[0]['event']} at 0.000s")
        print(f"  Last event: {self.tracer.trace_events[-1]['event']} at {self.tracer.trace_events[-1]['timestamp']:.3f}s")
        print(f"  Total workflow time: {self.tracer.trace_events[-1]['timestamp']:.3f}s")

class TracedPlainCLIRenderer:
    """PlainCLIRenderer with workflow tracing."""
    
    def __init__(self, capabilities, tracer: WorkflowTracer):
        self.tracer = tracer
        self.capabilities = capabilities
        
        # Import and wrap the real renderer
        from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
        self._real_renderer = PlainCLIRenderer(capabilities)
        
        self.tracer.trace("renderer", "initialized")
    
    def initialize(self):
        """Initialize with tracing."""
        self.tracer.trace("renderer", "initializing")
        start_time = time.time()
        
        result = self._real_renderer.initialize()
        
        duration = (time.time() - start_time) * 1000
        self.tracer.trace("renderer", "initialized", result, duration)
        return result
    
    def handle_input(self):
        """Handle input with tracing."""
        self.tracer.trace("renderer", "handle_input_start")
        start_time = time.time()
        
        try:
            result = self._real_renderer.handle_input()
            duration = (time.time() - start_time) * 1000
            self.tracer.trace("renderer", "handle_input_complete", result, duration)
            return result
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.tracer.trace("renderer", "handle_input_error", str(e), duration)
            raise
    
    def render_frame(self):
        """Render frame with tracing."""
        self.tracer.trace("renderer", "render_frame_start")
        start_time = time.time()
        
        try:
            result = self._real_renderer.render_frame()
            duration = (time.time() - start_time) * 1000
            self.tracer.trace("renderer", "render_frame_complete", None, duration)
            return result
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.tracer.trace("renderer", "render_frame_error", str(e), duration)
            raise
    
    def cleanup(self):
        """Cleanup with tracing."""
        self.tracer.trace("renderer", "cleanup")
        return self._real_renderer.cleanup()
    
    def __getattr__(self, name):
        """Delegate other attributes to real renderer."""
        return getattr(self._real_renderer, name)

class TracedChatEngine:
    """ChatEngine with workflow tracing."""
    
    def __init__(self, tracer: WorkflowTracer):
        self.tracer = tracer
        
        # Import and wrap the real chat engine
        from agentsmcp.ui.v3.chat_engine import ChatEngine
        self._real_engine = ChatEngine()
        
        self.tracer.trace("chat_engine", "initialized")
    
    async def process_input(self, user_input: str):
        """Process input with tracing."""
        self.tracer.trace("chat_engine", "process_input_start", user_input)
        start_time = time.time()
        
        try:
            # Check if it's a command
            if user_input.startswith('/'):
                self.tracer.trace("chat_engine", "command_detected", user_input)
            else:
                self.tracer.trace("chat_engine", "message_detected", f"{len(user_input)}_chars")
            
            result = await self._real_engine.process_input(user_input)
            
            duration = (time.time() - start_time) * 1000
            self.tracer.trace("chat_engine", "process_input_complete", result, duration)
            return result
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.tracer.trace("chat_engine", "process_input_error", str(e), duration)
            raise
    
    def set_callbacks(self, status_callback=None, message_callback=None, error_callback=None):
        """Set callbacks with tracing."""
        self.tracer.trace("chat_engine", "callbacks_set", {
            "status": status_callback is not None,
            "message": message_callback is not None,
            "error": error_callback is not None
        })
        return self._real_engine.set_callbacks(status_callback, message_callback, error_callback)
    
    def __getattr__(self, name):
        """Delegate other attributes to real engine."""
        return getattr(self._real_engine, name)

class TracedTUILauncher:
    """TUILauncher with workflow tracing."""
    
    def __init__(self, tracer: WorkflowTracer):
        self.tracer = tracer
        
        # Import the real launcher class
        from agentsmcp.ui.v3.tui_launcher import TUILauncher
        self._real_launcher = TUILauncher()
        
        self.tracer.trace("launcher", "initialized")
    
    def initialize(self):
        """Initialize with tracing."""
        self.tracer.trace("launcher", "initializing")
        start_time = time.time()
        
        result = self._real_launcher.initialize()
        
        duration = (time.time() - start_time) * 1000
        self.tracer.trace("launcher", "initialized", result, duration)
        return result
    
    def __getattr__(self, name):
        """Delegate other attributes to real launcher."""
        return getattr(self._real_launcher, name)

def generate_workflow_recommendations(results: Dict[str, bool], tracer: WorkflowTracer):
    """Generate recommendations based on workflow analysis."""
    print("\n" + "=" * 80)
    print("⚙️ COMMAND WORKFLOW ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"📊 Command Workflow Summary: {passed}/{total} commands processed successfully ({passed/total*100:.1f}%)")
    
    print("\n📋 Command Results:")
    for command, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status} '{command}'")
    
    # Analyze trace data
    trace_summary = tracer.get_trace_summary()
    print(f"\n📈 Workflow Metrics:")
    print(f"  Total trace events: {trace_summary['total_events']}")
    print(f"  Components involved: {len(trace_summary['components_involved'])}")
    print(f"  Total workflow time: {trace_summary['total_duration']}s")
    print(f"  Average event interval: {trace_summary['avg_event_interval']}s")
    
    # Identify issues
    failed_commands = [cmd for cmd, success in results.items() if not success]
    
    if failed_commands:
        print(f"\n🔧 FAILED COMMAND ANALYSIS:")
        
        for failed_cmd in failed_commands:
            print(f"  🎯 Command '{failed_cmd}':")
            
            # Find trace events for this command
            cmd_events = [e for e in tracer.trace_events if e.get('data') == failed_cmd]
            
            if cmd_events:
                print(f"    Trace events found: {len(cmd_events)}")
                for event in cmd_events:
                    print(f"      [{event['timestamp']:.3f}s] {event['component']} → {event['event']}")
            
            # Provide specific recommendations
            if failed_cmd.startswith('/'):
                print(f"    Recommendations:")
                print(f"      • Check ChatEngine command handler for '{failed_cmd}'")
                print(f"      • Verify command parsing logic")
                print(f"      • Test command execution path")
            else:
                print(f"    Recommendations:")
                print(f"      • Check LLM client integration")
                print(f"      • Verify message processing pipeline")
                print(f"      • Test async processing handling")
    
    else:
        print("\n🎉 All command workflows completed successfully!")
    
    print(f"\n📋 WORKFLOW OPTIMIZATION RECOMMENDATIONS:")
    if trace_summary['total_duration'] > 1.0:
        print("  ⚠️  High total workflow time detected")
        print("    • Review component initialization overhead")
        print("    • Consider caching initialized components")
        print("    • Optimize async processing")
    
    if trace_summary['avg_event_interval'] > 0.1:
        print("  ⚠️  High average event interval detected")
        print("    • Review synchronous operations in async context")
        print("    • Consider parallel component initialization")
        print("    • Optimize input handling mechanisms")
    
    print("\n📋 NEXT DEBUGGING STEPS:")
    print("  1. Focus on failed command types first")
    print("  2. Use real-time input monitor for detailed input flow")
    print("  3. Test individual components in isolation")
    print("  4. Verify end-to-end integration with actual terminal")

def main():
    debugger_header()
    
    # Create workflow debugger
    debugger = CommandWorkflowDebugger()
    
    try:
        # Run comprehensive command workflow test
        results = debugger.run_comprehensive_command_test()
        
        # Analyze workflow performance
        debugger.analyze_workflow_bottlenecks()
        
        # Generate recommendations
        generate_workflow_recommendations(results, debugger.tracer)
        
    except KeyboardInterrupt:
        print("\n⚠️  Workflow debugging interrupted by user")
    except Exception as e:
        print(f"\n❌ Workflow debugging error: {e}")
    
    print(f"\n" + "=" * 80)
    print("⚙️ Command Workflow Debugging Complete!")
    print("Use this trace data to identify where commands get stuck.")
    print("=" * 80)

if __name__ == "__main__":
    main()