#!/usr/bin/env python3
"""
DEBUG ENTER KEY - REAL ISSUE INVESTIGATION
Focus on why Enter key doesn't work in actual usage.
"""

import sys
import os
import time
import asyncio
import warnings
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

def debug_enter_key():
    """Debug the real Enter key issue."""
    print("🔍 INVESTIGATING REAL ENTER KEY ISSUE")  
    print("=" * 50)
    
    try:
        from agentsmcp.ui.cli_app import CLIConfig
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        
        # Create TUI like real usage
        class TestConfig:
            debug_mode = False
            verbose = False
        
        config = TestConfig()
        tui = RevolutionaryTUIInterface(cli_config=config)
        
        print("✅ TUI instance created")
        
        # Check Enter key methods
        print(f"\n--- ENTER KEY METHODS ANALYSIS ---")
        
        enter_methods = [
            '_handle_enter_input',
            '_handle_enter_input_sync', 
            '_handle_enter_key',
            '_process_user_input',
            'send_message'
        ]
        
        available_methods = []
        for method_name in enter_methods:
            if hasattr(tui, method_name):
                method = getattr(tui, method_name)
                
                import inspect
                is_async = inspect.iscoroutinefunction(method)
                try:
                    sig = inspect.signature(method)
                    method_info = f"{method_name}{sig} ({'async' if is_async else 'sync'})"
                except:
                    method_info = f"{method_name} ({'async' if is_async else 'sync'})"
                
                available_methods.append((method_name, method, is_async))
                print(f"✅ {method_info}")
            else:
                print(f"❌ {method_name}: NOT FOUND")
        
        # Test the sync wrapper specifically  
        print(f"\n--- SYNC WRAPPER TEST ---")
        
        if hasattr(tui, '_handle_enter_input_sync'):
            print("Testing sync wrapper method...")
            
            # Set up test input
            tui.state.current_input = "test message for enter key"
            print(f"Test input: '{tui.state.current_input}'")
            
            # Capture warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                try:
                    result = tui._handle_enter_input_sync()
                    print(f"✅ Sync wrapper called successfully")
                    print(f"   Return value: {result}")
                    
                    # Check for pending tasks
                    if hasattr(tui, '_pending_enter_tasks'):
                        pending = [t for t in tui._pending_enter_tasks if not t.done()]
                        print(f"   Pending async tasks: {len(pending)}")
                    else:
                        print("   No pending task tracking")
                        
                except Exception as e:
                    print(f"❌ Sync wrapper failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Check warnings
                runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
                if runtime_warnings:
                    print(f"⚠️  RuntimeWarnings detected: {len(runtime_warnings)}")
                    for warning in runtime_warnings:
                        print(f"   {warning.message}")
                else:
                    print(f"✅ No RuntimeWarnings")
        else:
            print("❌ Sync wrapper method not found!")
        
        # Test async method directly (this will generate warning, but let's see what happens)
        print(f"\n--- ASYNC METHOD DIRECT TEST ---")
        
        if hasattr(tui, '_handle_enter_input'):
            print("Testing async method directly (will generate warning)...")
            
            tui.state.current_input = "test async method"
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                try:
                    # This should create a coroutine but not execute it
                    coro = tui._handle_enter_input()
                    print(f"✅ Async method returned: {type(coro)}")
                    print(f"   Is coroutine: {asyncio.iscoroutine(coro)}")
                    
                    # Try to actually run it
                    print("   Attempting to run coroutine...")
                    
                    # Check if there's already an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        print(f"   Found running loop: {loop}")
                        
                        # Schedule it properly
                        task = loop.create_task(coro)
                        print(f"   Created task: {task}")
                        
                        # Give it a moment to start
                        time.sleep(0.1)
                        
                        print(f"   Task done: {task.done()}")
                        if task.done():
                            if task.exception():
                                print(f"   Task exception: {task.exception()}")
                            else:
                                print(f"   Task result: {task.result()}")
                        
                    except RuntimeError as e:
                        print(f"   No running loop: {e}")
                        
                        # Try to create a new event loop
                        print("   Creating new event loop...")
                        try:
                            result = asyncio.run(coro)
                            print(f"   ✅ Async method completed: {result}")
                        except Exception as e:
                            print(f"   ❌ Async method failed: {e}")
                    
                except Exception as e:
                    print(f"❌ Async method test failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Check warnings  
                runtime_warnings = [warning for warning in w if issubclass(warning.category, RuntimeWarning)]
                if runtime_warnings:
                    print(f"⚠️  RuntimeWarnings: {len(runtime_warnings)}")
                    for warning in runtime_warnings:
                        print(f"   {warning.message}")
        
        # Test the actual input loop integration
        print(f"\n--- INPUT LOOP INTEGRATION TEST ---")
        
        if hasattr(tui, '_input_loop'):
            print("Analyzing input loop source...")
            
            import inspect
            try:
                source = inspect.getsource(tui._input_loop)
                
                # Look for Enter key handling
                lines = source.split('\n')
                enter_lines = [line.strip() for line in lines if 'enter' in line.lower()]
                
                print(f"Enter key related lines in input loop:")
                for line in enter_lines:
                    print(f"   {line}")
                
                # Check what method is actually called
                if "call_soon_threadsafe(self._handle_enter_input_sync)" in source:
                    print("✅ Input loop uses sync wrapper")
                elif "asyncio.create_task(self._handle_enter_input())" in source:
                    print("❌ Input loop uses broken async task creation")  
                elif "_handle_enter_input" in source:
                    print("⚠️  Input loop references enter method but unclear how")
                else:
                    print("❓ No clear Enter handling found in input loop")
                    
            except Exception as e:
                print(f"❌ Could not analyze input loop: {e}")
        else:
            print("❌ Input loop method not found")
        
        print(f"\n--- SUMMARY ---")
        print("Key findings:")
        for method_name, method, is_async in available_methods:
            print(f"• {method_name}: {'async' if is_async else 'sync'}")
        
        print(f"\nLikely issues:")
        if not hasattr(tui, '_handle_enter_input_sync'):
            print(f"• Missing sync wrapper method")
        if not any(name == 'send_message' for name, _, _ in available_methods):
            print(f"• Missing send_message method (no actual message sending)")
        
        
    except Exception as e:
        print(f"❌ Enter key debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_enter_key()