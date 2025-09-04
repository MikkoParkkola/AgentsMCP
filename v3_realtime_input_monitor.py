#!/usr/bin/env python3
"""
V3 Real-Time Input Event Monitor
Live debugging of input flow through the V3 TUI pipeline

This script monitors input events in real-time to debug:
- Character-by-character input visibility
- Input buffer behavior
- Event timing and ordering
- Renderer state changes
"""

import sys
import os
import time
import asyncio
import threading
import select
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

class InputEventLogger:
    """Logs all input events with timestamps for analysis."""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.start_time = time.time()
    
    def log_event(self, event_type: str, data: Any, source: str = "monitor"):
        """Log an input event with timestamp."""
        event = {
            "timestamp": time.time() - self.start_time,
            "datetime": datetime.now().isoformat(),
            "type": event_type,
            "data": data,
            "source": source
        }
        self.events.append(event)
        print(f"[{event['timestamp']:7.3f}s] {event_type}: {data} (from: {source})")
    
    def save_events(self, filename: str):
        """Save events to JSON file for analysis."""
        with open(filename, 'w') as f:
            json.dump(self.events, f, indent=2)
        print(f"📁 Events saved to {filename}")

class InputBufferMonitor:
    """Monitors input buffer state changes."""
    
    def __init__(self, logger: InputEventLogger):
        self.logger = logger
        self.buffer_history: List[str] = []
        self.current_buffer = ""
    
    def update_buffer(self, new_buffer: str):
        """Update buffer and log changes."""
        if new_buffer != self.current_buffer:
            self.logger.log_event("buffer_change", {
                "old": self.current_buffer,
                "new": new_buffer,
                "length": len(new_buffer)
            }, "buffer_monitor")
            
            self.buffer_history.append(self.current_buffer)
            self.current_buffer = new_buffer
    
    def get_stats(self):
        """Get buffer statistics."""
        return {
            "current_length": len(self.current_buffer),
            "changes_count": len(self.buffer_history),
            "current_content": self.current_buffer
        }

class V3RendererMonitor:
    """Monitors V3 renderer state and behavior."""
    
    def __init__(self, logger: InputEventLogger):
        self.logger = logger
        self.renderer = None
        self.renderer_state_history = []
    
    def attach_renderer(self, renderer):
        """Attach to a V3 renderer for monitoring."""
        self.renderer = renderer
        self.logger.log_event("renderer_attached", {
            "type": renderer.__class__.__name__,
            "capabilities": {
                "is_tty": renderer.capabilities.is_tty,
                "width": renderer.capabilities.width,
                "height": renderer.capabilities.height
            }
        }, "renderer_monitor")
    
    def monitor_state_change(self):
        """Monitor renderer state changes."""
        if self.renderer:
            current_state = {
                "current_input": self.renderer.state.current_input,
                "is_processing": self.renderer.state.is_processing,
                "status_message": self.renderer.state.status_message
            }
            
            if not self.renderer_state_history or current_state != self.renderer_state_history[-1]:
                self.logger.log_event("renderer_state_change", current_state, "renderer_monitor")
                self.renderer_state_history.append(current_state.copy())

class CharacterInputTracker:
    """Tracks individual character input events."""
    
    def __init__(self, logger: InputEventLogger):
        self.logger = logger
        self.character_sequence = []
        self.timing_data = []
    
    def track_character(self, char: str, source: str = "stdin"):
        """Track a single character input."""
        char_code = ord(char) if char else None
        char_info = {
            "char": char,
            "ord": char_code,
            "printable": char.isprintable() if char else False,
            "is_control": char_code < 32 if char_code else False
        }
        
        self.character_sequence.append(char_info)
        self.logger.log_event("character_input", char_info, source)
    
    def get_sequence_analysis(self):
        """Analyze character input sequence."""
        return {
            "total_chars": len(self.character_sequence),
            "printable_chars": sum(1 for c in self.character_sequence if c.get("printable", False)),
            "control_chars": sum(1 for c in self.character_sequence if c.get("is_control", False)),
            "sequence": self.character_sequence[-10:]  # Last 10 chars
        }

def monitor_header():
    print("=" * 80)
    print("🔍 V3 Real-Time Input Event Monitor")
    print("=" * 80)
    print("Monitoring input events through V3 TUI pipeline in real-time...")
    print("Press Ctrl+C to stop monitoring and save results\n")

def test_basic_input_monitoring():
    """Test basic input monitoring without V3 components."""
    print("📋 TEST: Basic Input Event Monitoring")
    print("-" * 50)
    
    logger = InputEventLogger()
    char_tracker = CharacterInputTracker(logger)
    
    if not (hasattr(sys.stdin, 'isatty') and sys.stdin.isatty()):
        print("⚠️  Not in TTY mode - character-level monitoring not available")
        return logger
    
    print("🎯 Monitoring basic input events...")
    print("Instructions: Type some characters and press Enter")
    print("Type 'quit' to finish this test\n")
    
    try:
        import termios
        import tty
        
        # Get terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            # Set terminal to raw mode for character-by-character input
            tty.setraw(sys.stdin.fileno())
            logger.log_event("terminal_mode_change", "raw_mode_enabled", "monitor")
            
            input_buffer = ""
            
            while True:
                # Check for available input
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    char_tracker.track_character(char, "raw_stdin")
                    
                    if ord(char) == 3:  # Ctrl+C
                        logger.log_event("keyboard_interrupt", "ctrl_c_pressed", "monitor")
                        break
                    elif ord(char) == 13 or ord(char) == 10:  # Enter
                        logger.log_event("line_complete", {
                            "buffer": input_buffer,
                            "length": len(input_buffer)
                        }, "monitor")
                        
                        if input_buffer.strip().lower() == 'quit':
                            break
                        
                        print(f"\n✅ Line completed: '{input_buffer}'")
                        input_buffer = ""
                        print("💬 > ", end="", flush=True)
                    elif ord(char) == 127:  # Backspace
                        if input_buffer:
                            input_buffer = input_buffer[:-1]
                            logger.log_event("backspace", {
                                "remaining": input_buffer,
                                "length": len(input_buffer)
                            }, "monitor")
                            print("\b \b", end="", flush=True)
                    elif ord(char) >= 32:  # Printable character
                        input_buffer += char
                        logger.log_event("character_added", {
                            "char": char,
                            "buffer": input_buffer,
                            "length": len(input_buffer)
                        }, "monitor")
                        print(char, end="", flush=True)
                
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            logger.log_event("terminal_mode_change", "normal_mode_restored", "monitor")
            print("\n")
            
    except ImportError:
        print("⚠️  termios/tty not available - using line-based input")
        
        try:
            while True:
                line = input("💬 > ").strip()
                logger.log_event("line_input", {
                    "content": line,
                    "length": len(line)
                }, "input_fallback")
                
                if line.lower() == 'quit':
                    break
                    
                char_tracker.track_character(line, "line_input")
                
        except (EOFError, KeyboardInterrupt):
            logger.log_event("input_interrupted", "eof_or_keyboard_interrupt", "monitor")
    
    # Show analysis
    print("\n📊 Input Analysis:")
    char_analysis = char_tracker.get_sequence_analysis()
    for key, value in char_analysis.items():
        if key != 'sequence':
            print(f"  {key}: {value}")
    
    return logger

def test_v3_renderer_input_monitoring():
    """Test input monitoring with V3 renderer components."""
    print("\n📋 TEST: V3 Renderer Input Monitoring")
    print("-" * 50)
    
    logger = InputEventLogger()
    buffer_monitor = InputBufferMonitor(logger)
    renderer_monitor = V3RendererMonitor(logger)
    
    try:
        from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
        
        # Setup V3 components
        caps = detect_terminal_capabilities()
        logger.log_event("capabilities_detected", {
            "is_tty": caps.is_tty,
            "width": caps.width,
            "height": caps.height,
            "supports_colors": caps.supports_colors
        }, "v3_monitor")
        
        renderer = PlainCLIRenderer(caps)
        renderer_monitor.attach_renderer(renderer)
        
        if not renderer.initialize():
            logger.log_event("renderer_init_failed", "PlainCLIRenderer", "v3_monitor")
            return logger
        
        logger.log_event("renderer_initialized", "PlainCLIRenderer", "v3_monitor")
        
        print("🎯 Monitoring V3 renderer input handling...")
        print("Instructions: Type messages and watch for input flow issues")
        print("Type '/quit' to exit\n")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                logger.log_event("input_loop_iteration", iteration, "v3_monitor")
                
                # Monitor renderer state before input
                renderer_monitor.monitor_state_change()
                
                # Render current frame
                renderer.render_frame()
                logger.log_event("frame_rendered", iteration, "v3_monitor")
                
                # Handle input using V3 renderer
                start_time = time.time()
                user_input = renderer.handle_input()
                input_duration = time.time() - start_time
                
                logger.log_event("input_handled", {
                    "input": user_input,
                    "duration_ms": round(input_duration * 1000, 2),
                    "iteration": iteration
                }, "v3_monitor")
                
                # Monitor buffer changes
                if hasattr(renderer.state, 'current_input'):
                    buffer_monitor.update_buffer(renderer.state.current_input)
                
                # Monitor renderer state after input
                renderer_monitor.monitor_state_change()
                
                if user_input:
                    if user_input.lower() == '/quit':
                        logger.log_event("quit_command", "user_requested_exit", "v3_monitor")
                        break
                    
                    print(f"✅ V3 Input processed: '{user_input}'")
                
                # Show buffer stats periodically
                if iteration % 5 == 0:
                    stats = buffer_monitor.get_stats()
                    logger.log_event("buffer_stats", stats, "v3_monitor")
                
                time.sleep(0.05)  # Small delay
                
        except KeyboardInterrupt:
            logger.log_event("monitoring_interrupted", "ctrl_c", "v3_monitor")
        finally:
            renderer.cleanup()
            logger.log_event("renderer_cleanup", "complete", "v3_monitor")
        
    except ImportError as e:
        logger.log_event("v3_import_error", str(e), "v3_monitor")
        print(f"❌ V3 import error: {e}")
    except Exception as e:
        logger.log_event("v3_monitor_error", str(e), "v3_monitor")
        print(f"❌ V3 monitoring error: {e}")
    
    return logger

def test_full_pipeline_monitoring():
    """Test full V3 pipeline with TUILauncher."""
    print("\n📋 TEST: Full V3 Pipeline Monitoring")
    print("-" * 50)
    
    logger = InputEventLogger()
    
    try:
        from agentsmcp.ui.v3.tui_launcher import TUILauncher
        
        logger.log_event("pipeline_test_start", "full_v3_pipeline", "pipeline_monitor")
        
        # Create and initialize TUILauncher
        launcher = TUILauncher()
        if not launcher.initialize():
            logger.log_event("launcher_init_failed", "TUILauncher", "pipeline_monitor")
            return logger
        
        logger.log_event("launcher_initialized", {
            "renderer": launcher.current_renderer.__class__.__name__ if launcher.current_renderer else None,
            "chat_engine": launcher.chat_engine is not None
        }, "pipeline_monitor")
        
        print("🎯 Monitoring full V3 pipeline...")
        print("This will run a simplified version of the main loop with monitoring")
        print("Type messages to test the full pipeline, '/quit' to exit\n")
        
        # Simplified monitoring version of the main loop
        iteration = 0
        try:
            while iteration < 10:  # Limit iterations for testing
                iteration += 1
                logger.log_event("pipeline_iteration", iteration, "pipeline_monitor")
                
                # Render frame
                if launcher.current_renderer:
                    launcher.current_renderer.render_frame()
                    logger.log_event("pipeline_frame_rendered", iteration, "pipeline_monitor")
                
                # Handle input
                start_time = time.time()
                user_input = launcher.current_renderer.handle_input()
                input_duration = time.time() - start_time
                
                logger.log_event("pipeline_input_handled", {
                    "input": user_input,
                    "duration_ms": round(input_duration * 1000, 2),
                    "iteration": iteration
                }, "pipeline_monitor")
                
                if user_input:
                    # Process through chat engine
                    if launcher.chat_engine:
                        try:
                            should_continue = asyncio.run(launcher.chat_engine.process_input(user_input))
                            logger.log_event("chat_engine_processed", {
                                "input": user_input,
                                "should_continue": should_continue
                            }, "pipeline_monitor")
                            
                            if not should_continue:
                                logger.log_event("pipeline_exit", "chat_engine_returned_false", "pipeline_monitor")
                                break
                        except Exception as e:
                            logger.log_event("chat_engine_error", str(e), "pipeline_monitor")
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.log_event("pipeline_interrupted", "ctrl_c", "pipeline_monitor")
        
        logger.log_event("pipeline_test_complete", iteration, "pipeline_monitor")
        
    except ImportError as e:
        logger.log_event("pipeline_import_error", str(e), "pipeline_monitor")
        print(f"❌ Pipeline import error: {e}")
    except Exception as e:
        logger.log_event("pipeline_error", str(e), "pipeline_monitor")
        print(f"❌ Pipeline error: {e}")
    
    return logger

def analyze_input_events(logger: InputEventLogger):
    """Analyze collected input events for issues."""
    print("\n📊 INPUT EVENT ANALYSIS")
    print("-" * 50)
    
    events = logger.events
    
    if not events:
        print("❌ No events collected")
        return
    
    print(f"📈 Total events: {len(events)}")
    print(f"⏱️  Duration: {events[-1]['timestamp']:.2f} seconds")
    
    # Categorize events
    event_types = {}
    for event in events:
        event_type = event['type']
        if event_type not in event_types:
            event_types[event_type] = 0
        event_types[event_type] += 1
    
    print("\n📋 Event Types:")
    for event_type, count in sorted(event_types.items()):
        print(f"  {event_type}: {count}")
    
    # Look for specific issues
    print("\n🔍 Issue Detection:")
    
    # Check for input delays
    input_events = [e for e in events if 'input' in e['type']]
    if input_events:
        delays = []
        for i in range(1, len(input_events)):
            delay = input_events[i]['timestamp'] - input_events[i-1]['timestamp']
            delays.append(delay)
        
        if delays:
            avg_delay = sum(delays) / len(delays)
            max_delay = max(delays)
            print(f"  Input delays: avg={avg_delay:.2f}s, max={max_delay:.2f}s")
            
            if avg_delay > 1.0:
                print("  ⚠️  High average input delay detected!")
            if max_delay > 5.0:
                print("  ❌ Very high maximum input delay detected!")
    
    # Check for errors
    error_events = [e for e in events if 'error' in e['type'] or 'failed' in e['type']]
    if error_events:
        print(f"  ❌ {len(error_events)} error events detected:")
        for event in error_events:
            print(f"    [{event['timestamp']:.2f}s] {event['type']}: {event['data']}")
    
    # Check for missing events
    expected_sequence = ['terminal_mode_change', 'character_input', 'line_complete']
    found_sequence = [e['type'] for e in events if e['type'] in expected_sequence]
    
    if len(set(found_sequence)) < len(expected_sequence):
        missing = set(expected_sequence) - set(found_sequence)
        print(f"  ⚠️  Missing expected events: {missing}")
    
    # Timeline analysis
    print(f"\n📅 Event Timeline (first 10 events):")
    for event in events[:10]:
        print(f"  [{event['timestamp']:7.3f}s] {event['type']} - {event['source']}")

def main():
    monitor_header()
    
    all_loggers = []
    
    try:
        # Test 1: Basic input monitoring
        logger1 = test_basic_input_monitoring()
        all_loggers.append(("basic_input", logger1))
        
        # Test 2: V3 renderer monitoring  
        logger2 = test_v3_renderer_input_monitoring()
        all_loggers.append(("v3_renderer", logger2))
        
        # Test 3: Full pipeline monitoring
        logger3 = test_full_pipeline_monitoring()
        all_loggers.append(("full_pipeline", logger3))
        
    except KeyboardInterrupt:
        print("\n⚠️  Monitoring interrupted by user")
    
    # Analyze and save results
    print("\n" + "=" * 80)
    print("📊 MONITORING RESULTS ANALYSIS")
    print("=" * 80)
    
    for test_name, logger in all_loggers:
        if logger.events:
            print(f"\n🔍 Analysis for {test_name.upper()}:")
            analyze_input_events(logger)
            
            # Save events to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"v3_input_monitor_{test_name}_{timestamp}.json"
            logger.save_events(filename)
    
    print(f"\n" + "=" * 80)
    print("📋 V3 Real-Time Input Monitoring Complete!")
    print("Check the generated JSON files for detailed event analysis.")
    print("=" * 80)

if __name__ == "__main__":
    main()