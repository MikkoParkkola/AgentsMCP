#!/usr/bin/env python3
"""
Systematic TUI Diagnostic Script
Traces exact renderer selection and failure points
"""

import sys
import os
import asyncio
import traceback
import time

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def log(msg, level="INFO"):
    """Structured logging with timestamps."""
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {level}: {msg}", flush=True)

def test_renderer_selection():
    """Test 1: Verify which renderer is being selected."""
    log("=== TEST 1: RENDERER SELECTION ===")
    
    try:
        from src.agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        from src.agentsmcp.ui.v3.ui_renderer_base import ProgressiveRenderer
        from src.agentsmcp.ui.v3.rich_tui_renderer import RichTUIRenderer
        from src.agentsmcp.ui.v3.plain_cli_renderer import SimpleTUIRenderer, PlainCLIRenderer
        
        # Detect capabilities
        capabilities = detect_terminal_capabilities()
        log(f"Terminal capabilities: TTY={capabilities.is_tty}, Rich={capabilities.supports_rich}")
        
        # Initialize progressive renderer
        progressive_renderer = ProgressiveRenderer(capabilities)
        
        # Register renderers
        progressive_renderer.register_renderer("rich", RichTUIRenderer, priority=30)
        progressive_renderer.register_renderer("simple", SimpleTUIRenderer, priority=20)
        progressive_renderer.register_renderer("plain", PlainCLIRenderer, priority=10)
        
        # Select best renderer
        renderer = progressive_renderer.select_best_renderer()
        
        if renderer:
            renderer_name = renderer.__class__.__name__
            log(f"SELECTED RENDERER: {renderer_name}")
            return renderer_name, renderer
        else:
            log("CRITICAL: No renderer selected!", "ERROR")
            return None, None
            
    except Exception as e:
        log(f"Exception in renderer selection: {e}", "ERROR")
        log(f"Traceback: {traceback.format_exc()}", "ERROR")
        return None, None

def test_renderer_initialization(renderer_name, renderer):
    """Test 2: Test renderer initialization."""
    log(f"=== TEST 2: {renderer_name} INITIALIZATION ===")
    
    try:
        log(f"Calling {renderer_name}.initialize()...")
        success = renderer.initialize()
        log(f"{renderer_name}.initialize() returned: {success}")
        return success
        
    except Exception as e:
        log(f"Exception in {renderer_name}.initialize(): {e}", "ERROR")
        log(f"Traceback: {traceback.format_exc()}", "ERROR")
        return False

def test_renderer_input_handling(renderer_name, renderer):
    """Test 3: Test renderer input handling (non-blocking)."""
    log(f"=== TEST 3: {renderer_name} INPUT HANDLING ===")
    
    try:
        log(f"Testing {renderer_name}.handle_input() (non-blocking)...")
        
        # Call handle_input once to see if it crashes
        result = renderer.handle_input()
        log(f"{renderer_name}.handle_input() returned: {result}")
        
        return True
        
    except Exception as e:
        log(f"Exception in {renderer_name}.handle_input(): {e}", "ERROR")
        log(f"Traceback: {traceback.format_exc()}", "ERROR")
        return False

def test_chat_engine():
    """Test 4: Test chat engine integration."""
    log("=== TEST 4: CHAT ENGINE ===")
    
    try:
        from src.agentsmcp.ui.v3.chat_engine import ChatEngine
        
        engine = ChatEngine()
        log("ChatEngine created successfully")
        
        # Test a simple command
        log("Testing /help command...")
        result = asyncio.run(engine.process_input("/help"))
        log(f"ChatEngine.process_input('/help') returned: {result}")
        
        return True
        
    except Exception as e:
        log(f"Exception in ChatEngine test: {e}", "ERROR")
        log(f"Traceback: {traceback.format_exc()}", "ERROR")
        return False

async def test_main_loop_entry():
    """Test 5: Test TUILauncher main loop entry."""
    log("=== TEST 5: MAIN LOOP ENTRY ===")
    
    try:
        from src.agentsmcp.ui.v3.tui_launcher import TUILauncher
        
        launcher = TUILauncher()
        log("TUILauncher created")
        
        # Test initialization
        log("Testing launcher.initialize()...")
        success = launcher.initialize()
        log(f"launcher.initialize() returned: {success}")
        
        if success:
            log("Initialization successful - would enter main loop")
            log("Calling cleanup to exit cleanly...")
            launcher._cleanup()
            return True
        else:
            log("Initialization failed", "ERROR")
            return False
            
    except Exception as e:
        log(f"Exception in main loop entry test: {e}", "ERROR")
        log(f"Traceback: {traceback.format_exc()}", "ERROR")
        return False

def main():
    """Run systematic diagnostic."""
    log("🔬 SYSTEMATIC TUI DIAGNOSTIC STARTING")
    log("This will identify the exact failure point in your TUI")
    print()
    
    # Test 1: Renderer Selection
    renderer_name, renderer = test_renderer_selection()
    if not renderer:
        log("DIAGNOSTIC FAILED: Cannot select renderer", "CRITICAL")
        return 1
    
    print()
    
    # Test 2: Renderer Initialization  
    init_success = test_renderer_initialization(renderer_name, renderer)
    if not init_success:
        log(f"DIAGNOSTIC FAILED: {renderer_name} initialization failed", "CRITICAL")
        return 1
    
    print()
    
    # Test 3: Input Handling
    input_success = test_renderer_input_handling(renderer_name, renderer)
    if not input_success:
        log(f"DIAGNOSTIC FAILED: {renderer_name} input handling failed", "CRITICAL")
        return 1
        
    print()
    
    # Test 4: Chat Engine
    chat_success = test_chat_engine()
    if not chat_success:
        log("DIAGNOSTIC FAILED: ChatEngine failed", "CRITICAL")
        return 1
        
    print()
    
    # Test 5: Main Loop Entry
    loop_success = asyncio.run(test_main_loop_entry())
    if not loop_success:
        log("DIAGNOSTIC FAILED: Main loop entry failed", "CRITICAL")
        return 1
        
    print()
    log("🎉 ALL DIAGNOSTIC TESTS PASSED")
    log(f"Your system should work with {renderer_name}")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        log(f"Diagnostic completed with exit code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log("Diagnostic interrupted by user", "INFO")
        sys.exit(1)
    except Exception as e:
        log(f"Unexpected error in diagnostic: {e}", "CRITICAL")
        log(f"Traceback: {traceback.format_exc()}", "CRITICAL")
        sys.exit(1)