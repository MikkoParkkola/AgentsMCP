#!/usr/bin/env python3
"""
V3 TUI Input Pipeline Debugger
Comprehensive step-by-step diagnosis of V3 TUI input issues

This script specifically targets the reported issues:
- Characters visible in bottom right corner one at a time
- Input doesn't reach input box until Enter pressed repeatedly  
- Commands (/) don't work
- No LLM communication
"""

import sys
import os
import time
import asyncio
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

def debug_header():
    print("=" * 80)
    print("🔧 V3 TUI Input Pipeline Debugger")
    print("=" * 80)
    print("Diagnosing V3 TUI input issues step-by-step...")
    print("This will test each component in the V3 input pipeline individually\n")

def test_1_v3_architecture_detection():
    print("📋 TEST 1: V3 Architecture Detection & Import")
    print("-" * 50)
    
    results = {}
    
    try:
        # Test V3 module imports
        from agentsmcp.ui.v3.tui_launcher import TUILauncher
        results['tui_launcher'] = True
        print("  ✅ TUILauncher imported successfully")
        
        from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        results['terminal_capabilities'] = True
        print("  ✅ TerminalCapabilities imported successfully")
        
        from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
        results['plain_cli_renderer'] = True
        print("  ✅ PlainCLIRenderer imported successfully")
        
        from agentsmcp.ui.v3.chat_engine import ChatEngine
        results['chat_engine'] = True
        print("  ✅ ChatEngine imported successfully")
        
        from agentsmcp.ui.v3.ui_renderer_base import ProgressiveRenderer
        results['ui_renderer_base'] = True
        print("  ✅ ProgressiveRenderer imported successfully")
        
        return results
        
    except ImportError as e:
        print(f"  ❌ V3 Import error: {e}")
        results['import_error'] = str(e)
        return results
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        results['unexpected_error'] = str(e)
        return results

def test_2_terminal_capabilities_deep_dive():
    print("\n📋 TEST 2: Terminal Capabilities Deep Dive")
    print("-" * 50)
    
    try:
        from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        
        print("  🔍 Detecting terminal capabilities...")
        caps = detect_terminal_capabilities()
        
        # Deep capability analysis
        print(f"  Raw Capability Details:")
        print(f"    is_tty: {caps.is_tty} (stdin.isatty: {hasattr(sys.stdin, 'isatty') and sys.stdin.isatty()})")
        print(f"    width: {caps.width}, height: {caps.height}")
        print(f"    supports_colors: {caps.supports_colors}")
        print(f"    supports_unicode: {caps.supports_unicode}")
        print(f"    supports_rich: {caps.supports_rich}")
        
        # Environment analysis
        print(f"  Environment Variables:")
        print(f"    TERM: {os.environ.get('TERM', 'NOT_SET')}")
        print(f"    TERM_PROGRAM: {os.environ.get('TERM_PROGRAM', 'NOT_SET')}")
        print(f"    COLORTERM: {os.environ.get('COLORTERM', 'NOT_SET')}")
        print(f"    FORCE_COLOR: {os.environ.get('FORCE_COLOR', 'NOT_SET')}")
        
        # Issue analysis
        issues = []
        if not caps.is_tty:
            issues.append("TTY detection failed - this will cause fallback to V2 demo mode")
        if caps.width < 80 or caps.height < 24:
            issues.append(f"Terminal size too small ({caps.width}x{caps.height}) - may cause rendering issues")
        
        if issues:
            print(f"  ⚠️  Detected Issues:")
            for issue in issues:
                print(f"    • {issue}")
        else:
            print("  ✅ Terminal capabilities look good")
            
        return caps
        
    except Exception as e:
        print(f"  ❌ Terminal capabilities error: {e}")
        return None

def test_3_progressive_renderer_selection():
    print("\n📋 TEST 3: Progressive Renderer Selection Process")
    print("-" * 50)
    
    try:
        from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        from agentsmcp.ui.v3.ui_renderer_base import ProgressiveRenderer
        from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer, SimpleTUIRenderer
        from agentsmcp.ui.v3.rich_tui_renderer import RichTUIRenderer
        
        # Detect capabilities
        caps = detect_terminal_capabilities()
        print(f"  📊 Terminal capabilities: TTY={caps.is_tty}, Rich={caps.supports_rich}")
        
        # Create progressive renderer
        renderer = ProgressiveRenderer(caps)
        print("  ✅ ProgressiveRenderer created")
        
        # Register renderers and see what's available
        print("  🔧 Registering renderers...")
        
        try:
            renderer.register_renderer("rich", RichTUIRenderer, priority=30)
            print("    • RichTUIRenderer registered (priority: 30)")
        except Exception as e:
            print(f"    ❌ RichTUIRenderer registration failed: {e}")
        
        try:
            renderer.register_renderer("simple", SimpleTUIRenderer, priority=20)
            print("    • SimpleTUIRenderer registered (priority: 20)")
        except Exception as e:
            print(f"    ❌ SimpleTUIRenderer registration failed: {e}")
        
        try:
            renderer.register_renderer("plain", PlainCLIRenderer, priority=10)
            print("    • PlainCLIRenderer registered (priority: 10)")
        except Exception as e:
            print(f"    ❌ PlainCLIRenderer registration failed: {e}")
        
        # Select best renderer
        print("  🎯 Selecting best available renderer...")
        selected = renderer.select_best_renderer()
        
        if selected:
            renderer_name = selected.__class__.__name__
            print(f"  ✅ Selected renderer: {renderer_name}")
            
            # Test renderer initialization
            if selected.initialize():
                print(f"  ✅ {renderer_name} initialized successfully")
                selected.cleanup()  # Clean up after test
                return selected
            else:
                print(f"  ❌ {renderer_name} failed to initialize")
                return None
        else:
            print("  ❌ No renderer could be selected!")
            return None
            
    except Exception as e:
        print(f"  ❌ Progressive renderer test error: {e}")
        return None

def test_4_input_handling_mechanics():
    print("\n📋 TEST 4: Input Handling Mechanics Test")
    print("-" * 50)
    
    try:
        from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
        
        caps = detect_terminal_capabilities()
        renderer = PlainCLIRenderer(caps)
        
        if not renderer.initialize():
            print("  ❌ Failed to initialize PlainCLIRenderer")
            return False
        
        print("  🔧 Testing input handling mechanism...")
        print("  📝 Note: This will test the actual input() call that users experience")
        
        # Test the exact input mechanism
        if caps.is_tty:
            print("  ✅ TTY mode detected - testing interactive input")
            print("  👤 Please type 'test123' and press Enter:")
            
            try:
                # This mirrors exactly what PlainCLIRenderer.handle_input() does
                test_input = input("💬 > ").strip()
                print(f"  ✅ Input received successfully: '{test_input}'")
                print(f"  📊 Input length: {len(test_input)} characters")
                
                if test_input == "test123":
                    print("  ✅ Input matches expected value perfectly")
                else:
                    print(f"  ⚠️  Input doesn't match expected 'test123', got '{test_input}'")
                
                return True
                
            except (EOFError, KeyboardInterrupt):
                print("  ⚠️  Input was interrupted (EOFError/KeyboardInterrupt)")
                return False
            except Exception as e:
                print(f"  ❌ Input handling error: {e}")
                return False
        else:
            print("  ⚠️  Not in TTY mode - input handling may not work properly")
            return False
            
    except Exception as e:
        print(f"  ❌ Input handling test error: {e}")
        return False
    finally:
        try:
            if 'renderer' in locals():
                renderer.cleanup()
        except:
            pass

def test_5_chat_engine_command_processing():
    print("\n📋 TEST 5: ChatEngine Command Processing")
    print("-" * 50)
    
    try:
        from agentsmcp.ui.v3.chat_engine import ChatEngine
        
        print("  🚀 Creating ChatEngine instance...")
        engine = ChatEngine()
        print("  ✅ ChatEngine created successfully")
        
        # Test command processing
        test_commands = [
            ("/help", "Should show help message"),
            ("/status", "Should show status"),
            ("/clear", "Should clear screen"),
            ("hello", "Should be treated as user message"),
            ("/nonexistent", "Should handle unknown command"),
        ]
        
        print("  🧪 Testing command processing...")
        
        for cmd, description in test_commands:
            try:
                print(f"    Testing: '{cmd}' - {description}")
                
                # Run the actual async command processing
                result = asyncio.run(engine.process_input(cmd))
                
                if cmd == "/quit":
                    # Special case - /quit should return False
                    expected = False
                else:
                    # Most commands should return True to continue
                    expected = True
                
                print(f"    Result: {result} (expected: {expected})")
                
                if result == expected:
                    print(f"    ✅ Command '{cmd}' processed correctly")
                else:
                    print(f"    ⚠️  Command '{cmd}' returned {result}, expected {expected}")
                    
            except Exception as e:
                print(f"    ❌ Error processing '{cmd}': {e}")
        
        # Test /quit last since it may change engine state
        try:
            print(f"    Testing: '/quit' - Should return False to exit")
            result = asyncio.run(engine.process_input("/quit"))
            print(f"    Result: {result} (expected: False)")
            
            if result == False:
                print(f"    ✅ Command '/quit' processed correctly")
            else:
                print(f"    ⚠️  Command '/quit' returned {result}, expected False")
                
        except Exception as e:
            print(f"    ❌ Error processing '/quit': {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ChatEngine test error: {e}")
        return False

def test_6_v3_launcher_full_initialization():
    print("\n📋 TEST 6: V3 TUILauncher Full Initialization")
    print("-" * 50)
    
    try:
        from agentsmcp.ui.v3.tui_launcher import TUILauncher
        
        print("  🚀 Creating TUILauncher...")
        launcher = TUILauncher()
        print("  ✅ TUILauncher created")
        
        print("  🔧 Running full initialization...")
        if launcher.initialize():
            print("  ✅ TUILauncher initialized successfully")
            
            # Check components
            print("  📊 Component Status:")
            print(f"    capabilities: {launcher.capabilities is not None}")
            print(f"    progressive_renderer: {launcher.progressive_renderer is not None}")
            print(f"    current_renderer: {launcher.current_renderer is not None}")
            print(f"    chat_engine: {launcher.chat_engine is not None}")
            
            if launcher.current_renderer:
                renderer_name = launcher.current_renderer.__class__.__name__
                print(f"    selected_renderer: {renderer_name}")
            
            # Test callback setup
            print("  🔗 Testing callback integration...")
            if launcher.chat_engine:
                # Test status callback
                launcher._on_status_change("Test status message")
                print("    ✅ Status callback test passed")
                
                # Test error callback  
                launcher._on_error("Test error message")
                print("    ✅ Error callback test passed")
            
            launcher._cleanup()
            return True
        else:
            print("  ❌ TUILauncher initialization failed")
            return False
            
    except Exception as e:
        print(f"  ❌ TUILauncher test error: {e}")
        return False

def test_7_real_time_input_debugging():
    print("\n📋 TEST 7: Real-Time Input Flow Debugging")
    print("-" * 50)
    
    try:
        from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
        
        caps = detect_terminal_capabilities()
        
        if not caps.is_tty:
            print("  ⚠️  Not in TTY mode - real-time debugging not available")
            return False
        
        print("  🔄 Starting real-time input flow debugging...")
        print("  📝 Instructions:")
        print("    1. Type characters to see them appear immediately")
        print("    2. Press Enter to process input")  
        print("    3. Type '/quit' and press Enter to exit this test")
        print("    4. Use Ctrl+C to force exit if needed")
        print()
        
        renderer = PlainCLIRenderer(caps)
        if not renderer.initialize():
            print("  ❌ Failed to initialize renderer for debugging")
            return False
        
        print("  🎯 Real-time input debugging active...")
        input_count = 0
        
        try:
            while True:
                input_count += 1
                print(f"\n  [Input #{input_count}] Waiting for input...")
                
                # Use the exact same input mechanism as V3
                user_input = renderer.handle_input()
                
                if user_input:
                    print(f"  ✅ Input received: '{user_input}' (length: {len(user_input)})")
                    
                    if user_input.lower() == '/quit':
                        print("  👋 Exiting real-time debugging...")
                        break
                    elif user_input.startswith('/'):
                        print(f"  🎯 Command detected: {user_input}")
                    else:
                        print(f"  💬 User message: {user_input}")
                else:
                    print("  ⚠️  No input received (returned None)")
                
                # Small delay to prevent overwhelming output
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n  ⚠️  Real-time debugging interrupted by user")
        except Exception as e:
            print(f"\n  ❌ Real-time debugging error: {e}")
        finally:
            renderer.cleanup()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Real-time debugging setup error: {e}")
        return False

def test_8_v2_vs_v3_routing_analysis():
    print("\n📋 TEST 8: V2 vs V3 Routing Analysis")
    print("-" * 50)
    
    print("  🔍 Analyzing why V2 is used instead of V3...")
    
    try:
        # Check current TUI entry point
        current_dir = Path.cwd()
        agentsmcp_script = current_dir / "agentsmcp"
        
        if agentsmcp_script.exists():
            print("  📄 Found ./agentsmcp script - analyzing routing logic...")
            
            with open(agentsmcp_script, 'r') as f:
                script_content = f.read()
            
            # Look for TUI routing logic
            if 'def tui(' in script_content:
                print("  ✅ Found tui() function in agentsmcp script")
                
                # Extract the TUI function
                lines = script_content.split('\n')
                in_tui_func = False
                tui_lines = []
                
                for line in lines:
                    if line.startswith('def tui('):
                        in_tui_func = True
                        tui_lines.append(line)
                    elif in_tui_func:
                        if line.startswith('def ') and not line.startswith('def tui('):
                            break
                        tui_lines.append(line)
                
                print("  📊 TUI function analysis:")
                for i, line in enumerate(tui_lines[:20]):  # Show first 20 lines
                    print(f"    {i+1:2d}: {line}")
                
                if len(tui_lines) > 20:
                    print(f"    ... ({len(tui_lines) - 20} more lines)")
                
                # Check for V3 vs V2 references
                tui_code = '\n'.join(tui_lines)
                v2_refs = tui_code.count('v2')
                v3_refs = tui_code.count('v3')
                
                print(f"  📈 Code analysis:")
                print(f"    V2 references: {v2_refs}")
                print(f"    V3 references: {v3_refs}")
                
                if 'revolutionary' in tui_code.lower():
                    print("  ⚠️  Found 'revolutionary' references - likely using V2 Revolutionary TUI")
                
                if 'tui_launcher' in tui_code.lower() or 'TUILauncher' in tui_code:
                    print("  ✅ Found TUILauncher references - V3 may be available")
                else:
                    print("  ❌ No TUILauncher references found - V3 not being used")
                
            else:
                print("  ❌ No tui() function found in agentsmcp script")
        else:
            print("  ❌ ./agentsmcp script not found")
        
        # Check what happens when we run ./agentsmcp tui
        print("\n  🧪 Testing actual ./agentsmcp tui execution routing...")
        
        try:
            # Run with timeout to see initial output
            result = subprocess.run(
                ["./agentsmcp", "tui", "--help"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            print(f"  📄 Command help output:")
            if result.stdout:
                for line in result.stdout.split('\n')[:10]:
                    if line.strip():
                        print(f"    {line}")
            
            if result.stderr:
                print(f"  ⚠️  Error output:")
                for line in result.stderr.split('\n')[:5]:
                    if line.strip():
                        print(f"    {line}")
                        
        except subprocess.TimeoutExpired:
            print("  ⚠️  Command timed out (expected for interactive mode)")
        except Exception as e:
            print(f"  ❌ Command execution error: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Routing analysis error: {e}")
        return False

def generate_v3_specific_recommendations(test_results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("🔧 V3 TUI INPUT PIPELINE DIAGNOSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    critical_issues = []
    major_issues = []
    minor_issues = []
    
    # Analyze test results for V3-specific issues
    
    # Check if V3 is actually being used
    if not test_results.get('v3_launcher_init', False):
        critical_issues.append("V3 TUILauncher initialization failed - falling back to V2")
    
    # Check terminal capabilities
    if hasattr(test_results.get('terminal_caps'), 'is_tty') and not test_results['terminal_caps'].is_tty:
        critical_issues.append("TTY detection failed - this forces demo mode instead of interactive")
    
    # Check input handling
    if not test_results.get('input_handling', False):
        critical_issues.append("Input handling mechanism failed - users cannot type properly")
    
    # Check command processing
    if not test_results.get('chat_engine', False):
        major_issues.append("ChatEngine command processing has issues - commands like /help won't work")
    
    # Check renderer selection
    if not test_results.get('renderer_selection'):
        major_issues.append("Progressive renderer selection failed - UI may not display correctly")
    
    print(f"📊 ISSUE SUMMARY:")
    print(f"  🔥 Critical Issues: {len(critical_issues)}")
    print(f"  ⚠️  Major Issues: {len(major_issues)}")
    print(f"  ℹ️  Minor Issues: {len(minor_issues)}")
    print()
    
    if critical_issues:
        print("🔥 CRITICAL ISSUES (Must Fix First):")
        for i, issue in enumerate(critical_issues, 1):
            print(f"  {i}. {issue}")
        print()
    
    if major_issues:
        print("⚠️  MAJOR ISSUES:")
        for i, issue in enumerate(major_issues, 1):
            print(f"  {i}. {issue}")
        print()
    
    # V3-specific fix recommendations
    print("🛠️  V3-SPECIFIC FIX RECOMMENDATIONS:")
    print()
    
    if "TTY detection failed" in str(critical_issues):
        print("  🎯 FIX 1: TTY Detection Issue")
        print("     Problem: Terminal not detected as TTY, forcing demo mode")
        print("     Solution: Force TTY mode or improve detection logic")
        print("     Files to check:")
        print("       • src/agentsmcp/ui/v3/terminal_capabilities.py")
        print("       • Environment variables (TERM, TERM_PROGRAM)")
        print()
    
    if "V3 TUILauncher" in str(critical_issues):
        print("  🎯 FIX 2: V3 Routing Issue")
        print("     Problem: agentsmcp script routes to V2 instead of V3")
        print("     Solution: Update agentsmcp script to use V3 by default")
        print("     Files to modify:")
        print("       • ./agentsmcp (main script)")
        print("       • Ensure V3 TUILauncher is called instead of V2")
        print()
    
    if "Input handling mechanism failed" in str(critical_issues):
        print("  🎯 FIX 3: Input Handling")
        print("     Problem: PlainCLIRenderer input handling not working")
        print("     Solution: Debug PlainCLIRenderer.handle_input() method")
        print("     Files to check:")
        print("       • src/agentsmcp/ui/v3/plain_cli_renderer.py")
        print("       • Check for input() blocking issues")
        print()
    
    print("  📋 IMMEDIATE ACTION PLAN:")
    print("     1. Run this debugger in a proper TTY terminal")
    print("     2. Modify agentsmcp script to use V3 instead of V2")
    print("     3. Test V3 PlainCLIRenderer directly")
    print("     4. Verify ChatEngine command processing")
    print("     5. Test end-to-end V3 pipeline")

def main():
    debug_header()
    
    test_results = {}
    
    # Run comprehensive V3-specific tests
    test_results['v3_imports'] = test_1_v3_architecture_detection()
    test_results['terminal_caps'] = test_2_terminal_capabilities_deep_dive()
    test_results['renderer_selection'] = test_3_progressive_renderer_selection()
    test_results['input_handling'] = test_4_input_handling_mechanics()
    test_results['chat_engine'] = test_5_chat_engine_command_processing()
    test_results['v3_launcher_init'] = test_6_v3_launcher_full_initialization()
    
    # Interactive tests (if TTY available)
    if hasattr(test_results.get('terminal_caps'), 'is_tty') and test_results['terminal_caps'].is_tty:
        print("\n🎯 TTY detected - running interactive tests...")
        test_results['realtime_input'] = test_7_real_time_input_debugging()
    else:
        print("\n⚠️  No TTY detected - skipping interactive tests")
        test_results['realtime_input'] = False
    
    test_results['routing_analysis'] = test_8_v2_vs_v3_routing_analysis()
    
    # Generate specific recommendations
    generate_v3_specific_recommendations(test_results)
    
    print(f"\n" + "=" * 80)
    print("📋 V3 TUI Input Pipeline Diagnosis Complete!")
    print("Use these results to target specific V3 components for fixes.")
    print("=" * 80)

if __name__ == "__main__":
    main()