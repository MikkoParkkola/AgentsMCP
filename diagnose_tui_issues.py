#!/usr/bin/env python3
"""
TUI Diagnostic Script for AI Command Composer
Tests various TUI scenarios to isolate input issues
"""

import sys
import os
import asyncio
import subprocess
import platform
import termios
import tty
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_section(title):
    print(f"\n📋 {title}")
    print("-" * 40)

def test_environment():
    """Test terminal environment and capabilities"""
    print_header("ENVIRONMENT DIAGNOSTICS")
    
    # Basic environment info
    print_section("System Information")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"Terminal: {os.environ.get('TERM', 'unknown')}")
    print(f"TTY: {sys.stdin.isatty()}")
    print(f"Stdout TTY: {sys.stdout.isatty()}")
    print(f"Stderr TTY: {sys.stderr.isatty()}")
    
    # Terminal size
    try:
        size = os.get_terminal_size()
        print(f"Terminal Size: {size.columns}x{size.lines}")
    except:
        print("Terminal Size: Unknown")
    
    # Environment variables
    print_section("Relevant Environment Variables")
    env_vars = ['TERM', 'COLORTERM', 'FORCE_COLOR', 'NO_COLOR', 'CLICOLOR']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def test_input_basic():
    """Test basic input functionality"""
    print_header("BASIC INPUT TEST")
    
    if not sys.stdin.isatty():
        print("❌ Not running in TTY - skipping interactive tests")
        return
    
    print("This will test basic input handling...")
    print("Type 'test' and press Enter (or 'skip' to skip):")
    
    try:
        user_input = input(">>> ")
        if user_input.lower() == 'skip':
            print("⏭️  Input test skipped")
            return
        print(f"✅ Received input: '{user_input}'")
        print(f"✅ Length: {len(user_input)} characters")
    except (EOFError, KeyboardInterrupt):
        print("❌ Input test failed - EOF or Ctrl+C")
    except Exception as e:
        print(f"❌ Input test failed: {e}")

def test_v3_system():
    """Test V3 system functionality"""
    print_header("V3 SYSTEM TEST")
    
    try:
        # Try to import V3 components
        print_section("V3 Import Test")
        
        from agentsmcp.ui.v3.terminal_capabilities import detect_terminal_capabilities
        print("✅ terminal_capabilities imported")
        
        from agentsmcp.ui.v3.ui_renderer_base import UIRenderer, ProgressiveRenderer
        print("✅ ui_renderer_base imported")
        
        from agentsmcp.ui.v3.plain_cli_renderer import PlainCLIRenderer
        print("✅ PlainCLIRenderer imported")
        
        from agentsmcp.ui.v3.tui_launcher import TUILauncher
        print("✅ TUILauncher imported")
        
        # Test capability detection
        print_section("V3 Capability Detection")
        caps = detect_terminal_capabilities()
        print(f"✅ TTY: {caps.is_tty}")
        print(f"✅ Size: {caps.width}x{caps.height}")
        print(f"✅ Colors: {caps.supports_colors}")
        print(f"✅ Unicode: {caps.supports_unicode}")
        print(f"✅ Rich: {caps.supports_rich}")
        
        # Test renderer selection
        print_section("V3 Renderer Selection")
        renderer_mgr = ProgressiveRenderer(caps)
        renderer_mgr.register_renderer("plain", PlainCLIRenderer, priority=1)
        
        # This would normally select the best renderer
        print("✅ V3 system appears functional")
        
    except ImportError as e:
        print(f"❌ V3 import failed: {e}")
    except Exception as e:
        print(f"❌ V3 test failed: {e}")

def test_v2_system():
    """Test V2 system components"""
    print_header("V2 SYSTEM TEST")
    
    try:
        print_section("V2 Import Test")
        
        from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
        print("✅ RevolutionaryTUIInterface imported")
        
        from agentsmcp.ui.v2.feature_activation_manager import FeatureActivationManager
        print("✅ FeatureActivationManager imported")
        
        print("✅ V2 system imports successful")
        
        # Test feature detection
        print_section("V2 Feature Detection")
        feature_mgr = FeatureActivationManager()
        
        if hasattr(feature_mgr, 'detect_terminal_capabilities'):
            caps = feature_mgr.detect_terminal_capabilities()
            print(f"✅ V2 capability detection successful")
        else:
            print("⚠️  V2 capability detection method not found")
            
    except ImportError as e:
        print(f"❌ V2 import failed: {e}")
    except Exception as e:
        print(f"❌ V2 test failed: {e}")

def test_cli_entry_points():
    """Test different CLI entry points"""
    print_header("CLI ENTRY POINT TESTS")
    
    print_section("Testing agentsmcp wrapper")
    try:
        # Test help command (should be fast)
        result = subprocess.run(
            ['./agentsmcp', '--help'], 
            capture_output=True, 
            text=True, 
            timeout=10,
            cwd='/Users/mikko/github/AgentsMCP'
        )
        if result.returncode == 0:
            print("✅ agentsmcp wrapper responds to --help")
        else:
            print(f"❌ agentsmcp --help failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ agentsmcp --help timed out")
    except Exception as e:
        print(f"❌ agentsmcp test failed: {e}")

def test_manual_v3_launch():
    """Test manual V3 system launch"""
    print_header("MANUAL V3 LAUNCH TEST")
    
    if not sys.stdin.isatty():
        print("❌ Not in TTY - V3 will use PlainCLI mode")
        print("💡 Run this script in a proper terminal for full test")
        return
        
    print("This test will attempt to launch V3 system manually...")
    print("Press Ctrl+C to interrupt if it hangs")
    
    try:
        from agentsmcp.ui.v3.tui_launcher import launch_tui
        
        print("⚠️  Starting V3 system - press Ctrl+C to exit...")
        
        # Run with timeout to prevent hanging
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("V3 launch test timed out")
            
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # 10 second timeout
        
        try:
            # This should launch the V3 system
            result = asyncio.run(launch_tui())
            print(f"✅ V3 system completed with exit code: {result}")
        except KeyboardInterrupt:
            print("⚠️  V3 system interrupted by user")
        except TimeoutError:
            print("❌ V3 system timed out after 10 seconds")
        finally:
            signal.alarm(0)  # Cancel timeout
            
    except Exception as e:
        print(f"❌ V3 manual launch failed: {e}")

def main():
    """Run all diagnostic tests"""
    print("🚀 AI Command Composer - TUI Diagnostic Script")
    print("This script will test various components to isolate TUI input issues")
    
    # Run all tests
    test_environment()
    test_input_basic()
    test_v3_system()
    test_v2_system()
    test_cli_entry_points()
    
    # Ask user if they want to test manual launch
    if sys.stdin.isatty():
        print(f"\n{'='*60}")
        print("🔴 INTERACTIVE TEST AVAILABLE")
        print("Would you like to test manual V3 launch? (y/N):")
        try:
            answer = input(">>> ").lower().strip()
            if answer in ['y', 'yes']:
                test_manual_v3_launch()
        except:
            print("Skipping manual launch test")
    
    print(f"\n{'='*60}")
    print("✅ Diagnostic script completed!")
    print("📊 Please share this output for analysis")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()