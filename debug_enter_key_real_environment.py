#!/usr/bin/env python3
"""
ENTER KEY REAL ENVIRONMENT DIAGNOSTIC
Debug why Enter key doesn't work in real TUI environment.
"""

import sys
import os
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def debug_enter_key_handling():
    """Debug Enter key handling in real TUI environment."""
    print("🔍 Debugging Enter key handling in real TUI environment...")
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        import inspect
        
        # Create TUI instance
        class DebugConfig:
            debug_mode = True
            verbose = True
        
        config = DebugConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        print("✅ TUI instance created")
        
        # Check Enter key handling methods
        print(f"\n📊 ENTER KEY HANDLING METHODS:")
        print("=" * 50)
        
        enter_methods = [
            '_handle_enter_key',
            '_handle_enter_input', 
            '_process_user_input',
            'send_message',
            '_send_user_message',
            '_handle_user_input_submit',
            'handle_enter',
            'on_enter',
        ]
        
        existing_methods = []
        for method in enter_methods:
            if hasattr(tui, method):
                method_obj = getattr(tui, method)
                if callable(method_obj):
                    existing_methods.append(method)
                    try:
                        sig = inspect.signature(method_obj)
                        print(f"  ✅ {method}{sig}")
                        
                        # Try to get the method's source
                        try:
                            source_lines = inspect.getsourcelines(method_obj)[1]
                            print(f"     Source starts at line {source_lines}")
                        except:
                            print(f"     Source not available")
                            
                    except Exception as e:
                        print(f"  ✅ {method}(...) - signature unavailable: {e}")
                else:
                    print(f"  ⚠️  {method}: exists but not callable")
            else:
                print(f"  ❌ {method}: not found")
        
        print(f"\nFound {len(existing_methods)} Enter-related methods: {existing_methods}")
        
        # Check orchestrator connection
        print(f"\n📊 ORCHESTRATOR CONNECTION:")
        print("=" * 50)
        
        if hasattr(tui, 'orchestrator'):
            orchestrator = tui.orchestrator
            if orchestrator:
                print(f"✅ Orchestrator exists: {type(orchestrator).__name__}")
                
                # Check orchestrator methods
                orchestrator_methods = [
                    'process_user_input',
                    'handle_message',
                    'send_message',
                    'execute_command',
                ]
                
                for method in orchestrator_methods:
                    if hasattr(orchestrator, method):
                        print(f"  ✅ orchestrator.{method} exists")
                    else:
                        print(f"  ❌ orchestrator.{method} missing")
            else:
                print("❌ Orchestrator is None")
        else:
            print("❌ No orchestrator attribute")
        
        # Test Enter key simulation
        print(f"\n📊 ENTER KEY SIMULATION TEST:")
        print("=" * 50)
        
        # Set up test input
        tui.state.current_input = "test message for enter key"
        print(f"Set test input: '{tui.state.current_input}'")
        
        # Test each available Enter method
        for method_name in existing_methods:
            print(f"\nTesting {method_name}:")
            try:
                method = getattr(tui, method_name)
                
                # Different methods might have different signatures
                sig = inspect.signature(method)
                params = list(sig.parameters.keys())
                
                print(f"  Method signature: {sig}")
                
                if len(params) == 0:
                    # No parameters
                    result = method()
                    print(f"  ✅ Called {method_name}() -> {result}")
                elif len(params) == 1:
                    # One parameter - might be key or message
                    if 'key' in params[0].lower():
                        result = method('\r')  # Carriage return
                        print(f"  ✅ Called {method_name}('\\r') -> {result}")
                    elif 'message' in params[0].lower():
                        result = method(tui.state.current_input)
                        print(f"  ✅ Called {method_name}(input) -> {result}")
                    else:
                        # Try with input
                        result = method(tui.state.current_input)
                        print(f"  ✅ Called {method_name}(input) -> {result}")
                else:
                    print(f"  ⚠️  Method has {len(params)} parameters, skipping test")
                    
            except Exception as e:
                print(f"  ❌ {method_name} failed: {e}")
        
        # Check async methods
        print(f"\n📊 ASYNC METHODS CHECK:")
        print("=" * 50)
        
        async_methods = []
        for method_name in existing_methods:
            method = getattr(tui, method_name)
            if hasattr(method, '__code__') and method.__code__.co_flags & 0x80:  # CO_COROUTINE
                async_methods.append(method_name)
                print(f"  🔄 {method_name} is async")
            else:
                print(f"  📝 {method_name} is sync")
        
        if async_methods:
            print(f"\nFound {len(async_methods)} async methods: {async_methods}")
            print("⚠️  Async methods need to be awaited - this might be the issue!")
        
        # Check input handling pipeline
        print(f"\n📊 INPUT HANDLING PIPELINE:")
        print("=" * 50)
        
        pipeline_methods = [
            'run',
            '_process_user_input', 
            '_handle_character_input',
            '_handle_enter_input',
            '_sync_refresh_display',
        ]
        
        for method_name in pipeline_methods:
            if hasattr(tui, method_name):
                method = getattr(tui, method_name)
                is_async = hasattr(method, '__code__') and method.__code__.co_flags & 0x80
                print(f"  ✅ {method_name}: {'async' if is_async else 'sync'}")
            else:
                print(f"  ❌ {method_name}: missing")
        
        # Check event system
        print(f"\n📊 EVENT SYSTEM:")
        print("=" * 50)
        
        if hasattr(tui, 'event_system'):
            event_system = tui.event_system
            if event_system:
                print(f"✅ Event system exists: {type(event_system).__name__}")
                
                # Check if we can emit events
                try:
                    # Don't actually emit, just check if the method exists
                    if hasattr(event_system, 'emit'):
                        print("  ✅ event_system.emit exists")
                    if hasattr(event_system, 'emit_async'):
                        print("  ✅ event_system.emit_async exists")
                except Exception as e:
                    print(f"  ❌ Event system check failed: {e}")
            else:
                print("❌ Event system is None")
        else:
            print("❌ No event_system attribute")
            
    except Exception as e:
        print(f"❌ Enter key diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

def debug_keyboard_input_mechanisms():
    """Debug the actual keyboard input mechanisms."""
    print(f"\n📊 KEYBOARD INPUT MECHANISMS:")
    print("=" * 50)
    
    # Check for input libraries and methods
    input_libraries = [
        ('keyboard', 'keyboard library'),
        ('pynput', 'pynput library'),
        ('termios', 'termios (Unix terminal control)'),
        ('msvcrt', 'msvcrt (Windows console)'),
        ('select', 'select (Unix I/O multiplexing)'),
    ]
    
    for lib_name, description in input_libraries:
        try:
            __import__(lib_name)
            print(f"  ✅ {description}: available")
        except ImportError:
            print(f"  ❌ {description}: not available")
    
    # Check standard input methods
    print(f"\nStandard input methods:")
    try:
        import sys
        print(f"  ✅ sys.stdin available: {hasattr(sys.stdin, 'read')}")
        print(f"  ✅ sys.stdin readable: {sys.stdin.readable()}")
        print(f"  ✅ sys.stdin is TTY: {sys.stdin.isatty()}")
    except Exception as e:
        print(f"  ❌ sys.stdin check failed: {e}")

if __name__ == "__main__":
    debug_enter_key_handling()
    debug_keyboard_input_mechanisms()
    print(f"\n🎯 ENTER KEY DIAGNOSTIC COMPLETE")
    print("=" * 50)
    print("This diagnostic helps identify:")
    print("• Available Enter key handling methods")
    print("• Method signatures and call patterns")  
    print("• Async vs sync method issues")
    print("• Orchestrator connection problems")
    print("• Event system functionality")
    print("• Keyboard input library availability")
    print("Share this output to help fix the Enter key issue!")