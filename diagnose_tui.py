#!/usr/bin/env python3
"""
TUI Diagnosis Script - Comprehensive testing of AgentsMCP TUI behaviors
Run this in your terminal to diagnose and verify TUI functionality.
"""

import subprocess
import sys
import os
import time
import tempfile
from pathlib import Path

def run_command_with_timeout(cmd, timeout=10, input_text=None):
    """Run command with timeout and capture output."""
    try:
        if input_text:
            result = subprocess.run(
                cmd, shell=True, timeout=timeout, 
                capture_output=True, text=True,
                input=input_text
            )
        else:
            result = subprocess.run(
                cmd, shell=True, timeout=timeout,
                capture_output=True, text=True
            )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -2, "", str(e)

def test_basic_cli():
    """Test basic CLI functionality."""
    print("🔍 Testing basic CLI functionality...")
    
    # Test help
    code, stdout, stderr = run_command_with_timeout("./agentsmcp --help", 5)
    if code == 0:
        print("  ✅ Basic CLI help works")
    else:
        print(f"  ❌ Basic CLI help failed: {stderr}")
    
    # Test tui help
    code, stdout, stderr = run_command_with_timeout("./agentsmcp tui --help", 5)
    if code == 0:
        print("  ✅ TUI help works")
        if "--workspace" in stdout:
            print("  ⚠️  Old behavior: --workspace flag still mentioned in help")
        if "--chat" in stdout:
            print("  ✅ New behavior: --chat flag mentioned in help")
    else:
        print(f"  ❌ TUI help failed: {stderr}")

def test_tui_modes():
    """Test different TUI modes and their behavior."""
    print("\n🎯 Testing TUI modes...")
    
    # Test 1: Default TUI (should be v4 workspace now)
    print("  📋 Test 1: ./agentsmcp tui (should be v4 workspace)")
    code, stdout, stderr = run_command_with_timeout("./agentsmcp tui", 8, "/quit\n")
    
    if "Workspace Controller" in stdout:
        print("  ✅ Default TUI launches v4 workspace controller")
    elif "Conversation" in stdout and ("╭─" in stdout or "╰─" in stdout):
        print("  ❌ Default TUI still shows problematic panels (v3 chat)")
        print(f"    Output preview: {stdout[:200]}...")
    elif "Plain Text Mode" in stdout:
        print("  ⚠️  Default TUI falls back to plain text mode")
    else:
        print(f"  ❓ Unexpected default TUI behavior")
        print(f"    Code: {code}, Stderr: {stderr[:100]}")
    
    # Test 2: Legacy chat mode
    print("\n  📋 Test 2: ./agentsmcp tui --chat (should be legacy v3)")
    code, stdout, stderr = run_command_with_timeout("./agentsmcp tui --chat", 8, "/quit\n")
    
    if "legacy chat interface" in stdout or "V3 chat interface" in stdout:
        print("  ✅ --chat flag launches legacy interface")
    elif "Conversation" in stdout and ("╭─" in stdout or "╰─" in stdout):
        print("  ⚠️  --chat shows panels but no clear legacy messaging")
    else:
        print(f"  ❓ --chat behavior unclear")
        print(f"    Code: {code}, Output: {stdout[:150]}...")
    
    # Test 3: Workspace flag (should still work)
    print("\n  📋 Test 3: ./agentsmcp tui --workspace (backward compatibility)")
    code, stdout, stderr = run_command_with_timeout("./agentsmcp tui --workspace", 8, input_text="q\n")
    
    if "Workspace Controller" in stdout:
        print("  ✅ --workspace flag still works (backward compatibility)")
    else:
        print(f"  ⚠️  --workspace flag behavior changed or broken")

def test_environment_detection():
    """Test environment and terminal detection."""
    print("\n🌍 Testing environment detection...")
    
    # Check TTY status
    is_tty = sys.stdin.isatty()
    print(f"  📟 TTY detected: {is_tty}")
    
    # Check environment variables
    force_rich = os.environ.get('AGENTSMCP_FORCE_RICH')
    print(f"  🎨 AGENTSMCP_FORCE_RICH: {force_rich or 'not set'}")
    
    # Test with FORCE_RICH
    print("\n  📋 Test: AGENTSMCP_FORCE_RICH=1 ./agentsmcp tui")
    code, stdout, stderr = run_command_with_timeout(
        "AGENTSMCP_FORCE_RICH=1 ./agentsmcp tui", 8, input_text="q\n"
    )
    
    if "Workspace Controller" in stdout:
        print("  ✅ FORCE_RICH launches workspace controller")
    elif ("╭─" in stdout or "╰─" in stdout):
        print("  ⚠️  FORCE_RICH shows panels instead of workspace")
    else:
        print(f"  ❓ FORCE_RICH behavior unclear: {stdout[:100]}...")

def test_file_structure():
    """Test that required files exist."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "src/agentsmcp/ui/v4/workspace_controller.py",
        "src/agentsmcp/ui/v4/mcp_agent_orchestrator.py", 
        "src/agentsmcp/ui/v4/sequential_thinking_integrator.py",
        "src/agentsmcp/cli.py",
        "agentsmcp"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path} exists")
        else:
            print(f"  ❌ {file_path} missing")

def test_imports():
    """Test that v4 components can be imported."""
    print("\n🐍 Testing Python imports...")
    
    try:
        # Test workspace controller import
        result = subprocess.run([
            sys.executable, "-c", 
            "from src.agentsmcp.ui.v4.workspace_controller import WorkspaceController; print('✅ WorkspaceController import works')"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("  ✅ v4 imports work correctly")
        else:
            print(f"  ❌ v4 import failed: {result.stderr}")
            
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")

def analyze_current_behavior():
    """Analyze what's actually happening with the current setup."""
    print("\n🔬 Analyzing current behavior...")
    
    # Quick test to see what happens
    print("  Running quick diagnostic...")
    code, stdout, stderr = run_command_with_timeout("timeout 3s ./agentsmcp tui", 5)
    
    # Analyze the output
    if "Workspace Controller" in stdout:
        print("  ✅ GOOD: Default TUI shows Workspace Controller")
        print("  🎯 STATUS: Fix appears to be working!")
    elif "Plain Text Mode" in stdout:
        print("  ⚠️  INFO: TUI falls back to plain text (expected in non-TTY)")
        print("  💡 TIP: Try with AGENTSMCP_FORCE_RICH=1 for rich interface")
    elif ("╭─" in stdout or "╰─" in stdout) and "Conversation" in stdout:
        print("  ❌ ISSUE: Default TUI still shows problematic panels")
        print("  🔧 ACTION: The CLI routing fix may not have taken effect")
    else:
        print(f"  ❓ UNCLEAR: Unexpected output pattern")
        print(f"    Sample: {stdout[:150] if stdout else 'No output'}...")

def main():
    """Run comprehensive TUI diagnosis."""
    print("🚀 AgentsMCP TUI Diagnosis Script")
    print("=" * 50)
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Change to the correct directory if needed
    if not Path("./agentsmcp").exists():
        print("\n❌ Error: ./agentsmcp executable not found")
        print("Please run this script from the AgentsMCP root directory")
        return 1
    
    # Run all diagnostic tests
    test_file_structure()
    test_imports() 
    test_basic_cli()
    test_environment_detection()
    test_tui_modes()
    analyze_current_behavior()
    
    print("\n" + "=" * 50)
    print("🎯 DIAGNOSIS COMPLETE")
    print("\nExpected Results:")
    print("  ✅ ./agentsmcp tui → Should launch Workspace Controller")
    print("  ✅ ./agentsmcp tui --chat → Should launch legacy interface")
    print("  ✅ AGENTSMCP_FORCE_RICH=1 ./agentsmcp tui → Should show rich interface")
    
    print("\nIf you see ❌ issues above, the fix may need adjustment.")
    print("If you see ✅ across the board, the TUI should be working correctly!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())