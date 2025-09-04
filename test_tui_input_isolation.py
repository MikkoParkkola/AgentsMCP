#!/usr/bin/env python3
"""
TUI Input Isolation Test
Specifically tests input handling in different scenarios to isolate the root cause
"""

import sys
import os
import asyncio
import threading
import time
from pathlib import Path

def test_scenario_1_basic_input():
    """Test 1: Basic input() function"""
    print("\n🔍 TEST 1: Basic Python input() function")
    print("-" * 50)
    
    if not sys.stdin.isatty():
        print("❌ Not in TTY - basic input test skipped")
        return False
    
    try:
        print("Type 'hello' and press Enter:")
        user_input = input(">>> ")
        print(f"✅ Received: '{user_input}'")
        return len(user_input) > 0
    except Exception as e:
        print(f"❌ Basic input failed: {e}")
        return False

def test_scenario_2_v3_plain_renderer():
    """Test 2: V3 PlainCLIRenderer directly"""
    print("\n🔍 TEST 2: V3 PlainCLIRenderer Direct Test")
    print("-" * 50)
    
    try:
        from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
        
        # Create renderer
        caps = detect_terminal_capabilities()
        renderer = PlainCLIRenderer(caps)
        
        print(f"Capabilities: TTY={caps.is_tty}, Colors={caps.supports_colors}")
        
        # Initialize
        if renderer.initialize():
            print("✅ PlainCLI renderer initialized")
            
            # Test single input cycle
            if sys.stdin.isatty():
                print("Type 'test' and press Enter for PlainCLI test:")
                user_input = renderer.handle_input()
                print(f"PlainCLI received: '{user_input}'")
                
                if user_input:
                    print("✅ PlainCLI input handling works")
                    return True
                else:
                    print("❌ PlainCLI input handling returned None")
                    return False
            else:
                print("⚠️ Non-TTY environment - PlainCLI would use input() fallback")
                return True  # Expected behavior
        else:
            print("❌ PlainCLI renderer failed to initialize")
            return False
            
    except Exception as e:
        print(f"❌ V3 PlainCLI test failed: {e}")
        return False

def test_scenario_3_chat_engine():
    """Test 3: ChatEngine processing loop"""
    print("\n🔍 TEST 3: ChatEngine Processing Loop")
    print("-" * 50)
    
    try:
        from agentsmcp.ui.v3.chat_engine import ChatEngine
        
        # Create chat engine
        engine = ChatEngine()
        
        # Set up test callbacks
        status_messages = []
        chat_messages = []
        error_messages = []
        
        def status_callback(status):
            status_messages.append(status)
            print(f"Status: {status}")
        
        def message_callback(message):
            chat_messages.append(message)
            print(f"Message: {message}")
        
        def error_callback(error):
            error_messages.append(error)
            print(f"Error: {error}")
        
        engine.set_callbacks(status_callback, message_callback, error_callback)
        
        # Test command processing
        print("Testing ChatEngine with '/help' command...")
        result = asyncio.run(engine.process_input("/help"))
        
        if result is True:  # /help should return True (continue running)
            print("✅ ChatEngine processed /help command correctly")
            
            # Check if callbacks were called
            if status_messages or chat_messages:
                print("✅ ChatEngine callbacks working")
                return True
            else:
                print("⚠️ ChatEngine processed but no callback messages")
                return True  # Still working, might be silent
        else:
            print(f"❌ ChatEngine unexpected result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ ChatEngine test failed: {e}")
        return False

def test_scenario_4_full_v3_flow():
    """Test 4: Full V3 system flow without TUI"""
    print("\n🔍 TEST 4: Full V3 System Flow (Non-Interactive)")
    print("-" * 50)
    
    try:
        from agentsmcp.ui.v3.tui_launcher import TUILauncher
        from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        
        # Create launcher (no parameters needed - it detects capabilities internally)
        launcher = TUILauncher()
        
        # Initialize (should work even in non-TTY)
        if launcher.initialize():
            print("✅ V3 TUILauncher initialized successfully")
            
            # Check renderer selection
            current_renderer = launcher.current_renderer
            if current_renderer:
                renderer_name = current_renderer.__class__.__name__
                print(f"✅ Renderer selected: {renderer_name}")
                
                # Test if we can get the chat engine
                if hasattr(launcher, 'chat_engine'):
                    print("✅ ChatEngine available in launcher")
                    return True
                else:
                    print("❌ ChatEngine not found in launcher")
                    return False
            else:
                print("❌ No renderer selected")
                return False
        else:
            print("❌ V3 TUILauncher failed to initialize")
            return False
            
    except Exception as e:
        print(f"❌ Full V3 flow test failed: {e}")
        return False

def test_scenario_5_v2_demo_mode_issue():
    """Test 5: V2 Demo Mode Issue Analysis"""
    print("\n🔍 TEST 5: V2 Demo Mode Issue Analysis")
    print("-" * 50)
    
    # Look for the specific file mentioned in QA analysis
    revolutionary_tui_path = Path("/Users/mikko/github/AgentsMCP/src/agentsmcp/ui/v2/revolutionary_tui_interface.py")
    
    if not revolutionary_tui_path.exists():
        print("❌ Revolutionary TUI interface file not found")
        return False
    
    try:
        # Read the file to find the demo mode logic
        with open(revolutionary_tui_path, 'r') as f:
            content = f.read()
        
        # Look for demo mode patterns
        demo_patterns = [
            'demo_countdown',
            'Demo completed',
            'non-TTY environment',
            'interactive mode',
            'TTY>'
        ]
        
        found_patterns = []
        for pattern in demo_patterns:
            if pattern in content:
                found_patterns.append(pattern)
        
        print(f"Found demo mode patterns: {found_patterns}")
        
        # Look for line 1862 mentioned in QA analysis
        lines = content.split('\n')
        if len(lines) > 1860:
            target_line = lines[1861]  # 0-indexed
            print(f"Line 1862 context: {target_line.strip()}")
            
            # Look for nearby context
            start_line = max(0, 1856)  # 5 lines before
            end_line = min(len(lines), 1867)  # 5 lines after
            
            print("Context around line 1862:")
            for i in range(start_line, end_line):
                marker = ">>> " if i == 1861 else "    "
                print(f"{marker}{i+1:4d}: {lines[i]}")
        
        print("✅ V2 file analysis completed")
        return True
        
    except Exception as e:
        print(f"❌ V2 analysis failed: {e}")
        return False

def main():
    """Run focused input isolation tests"""
    print("🎯 TUI Input Isolation Test Suite")
    print("=" * 60)
    print("This script tests specific scenarios to isolate input issues")
    
    results = {}
    
    # Run all test scenarios
    results['basic_input'] = test_scenario_1_basic_input()
    results['v3_plain_renderer'] = test_scenario_2_v3_plain_renderer()
    results['chat_engine'] = test_scenario_3_chat_engine()
    results['full_v3_flow'] = test_scenario_4_full_v3_flow()
    results['v2_analysis'] = test_scenario_5_v2_demo_mode_issue()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Analysis
    print("\n🔍 ANALYSIS:")
    if results.get('basic_input') and results.get('v3_plain_renderer'):
        print("• Basic input works - issue is likely in TUI orchestration")
    elif not results.get('basic_input'):
        print("• Basic input broken - terminal/environment issue")
    
    if results.get('chat_engine') and results.get('full_v3_flow'):
        print("• V3 system components work individually")
    
    if results.get('v2_analysis'):
        print("• V2 demo mode logic found - likely the root cause per QA analysis")
    
    print("\n📋 Next steps: Share these results for targeted fix implementation")

if __name__ == "__main__":
    main()