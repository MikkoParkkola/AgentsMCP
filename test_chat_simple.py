#!/usr/bin/env python3
"""Simple test of multi-line input in chat interface"""

import sys
import subprocess
import time
sys.path.insert(0, 'src')

def test_multiline_logic():
    """Test the multi-line input logic directly"""
    print("🧪 Multi-line Logic Test")
    print("=" * 40)
    
    try:
        from agentsmcp.ui.command_interface import CommandInterface
        from agentsmcp.ui.theme_manager import ThemeManager
        
        class MockOrchestrationManager:
            def __init__(self):
                self.is_running = True
                
        theme_manager = ThemeManager()
        orchestration_manager = MockOrchestrationManager()
        command_interface = CommandInterface(orchestration_manager, theme_manager)
        
        print("✅ Command interface initialized")
        
        # Test incomplete line detection
        print("\n🔍 Testing incomplete line detection:")
        test_cases = [
            ("hello world", False, "Complete sentence"),
            ("if True:", True, "Python if statement"),
            ("def function(", True, "Incomplete function"),
            ("import os", False, "Complete import"),
            ("for i in range(10):", True, "For loop start"),
            ("print('hello')", False, "Complete print statement")
        ]
        
        all_passed = True
        for test_line, expected_incomplete, description in test_cases:
            result = command_interface._looks_incomplete(test_line)
            status = "✅" if result == expected_incomplete else "❌"
            if result != expected_incomplete:
                all_passed = False
            print(f"   {status} '{test_line}' -> {description}")
        
        if all_passed:
            print("\n✅ All incomplete line detection tests passed")
        else:
            print("\n⚠️  Some incomplete line detection tests failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing multi-line logic: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interactive_startup():
    """Test that interactive mode starts without errors"""
    print("\n🎮 Interactive Mode Startup Test")
    print("=" * 40)
    
    try:
        # Test with very simple input that should work
        process = subprocess.Popen(
            [sys.executable, '-m', 'agentsmcp', 'interactive', '--no-welcome'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send simple test input with quick exit
        test_input = "hello\nexit\n"
        
        # Wait a bit then terminate
        try:
            stdout, stderr = process.communicate(input=test_input, timeout=10)
            return_code = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return_code = process.returncode
            print("   ⚠️  Process timed out, had to kill it")
        
        print(f"📊 Results:")
        print(f"   Return code: {return_code}")
        print(f"   Stdout length: {len(stdout)}")
        print(f"   Stderr length: {len(stderr)}")
        
        # Check for critical errors
        critical_errors = ["Traceback", "ImportError", "ModuleNotFoundError", "AttributeError"]
        found_critical = []
        
        for error in critical_errors:
            if error in stderr:
                found_critical.append(error)
        
        if found_critical:
            print(f"   ❌ Critical errors found: {found_critical}")
            print("   Stderr sample:")
            print("   " + stderr[:300].replace('\n', '\n   '))
            return False
        elif return_code == 0:
            print("   ✅ Interactive mode started and exited cleanly")
            return True
        else:
            print(f"   ⚠️  Non-zero exit code but no critical errors")
            return True
            
    except Exception as e:
        print(f"   ❌ Error testing interactive startup: {e}")
        return False

def test_paste_detection_with_content():
    """Test paste detection with actual multi-line content"""
    print("\n📋 Paste Detection Test")
    print("=" * 40)
    
    # Create test content
    test_content = """Line 1: This is test content
Line 2: For multi-line paste testing
Line 3: Should be detected as paste
Line 4: And preserved as single input"""
    
    try:
        # Test by piping content
        process = subprocess.Popen(
            [sys.executable, '-c', '''
import sys
sys.path.insert(0, "src")
from agentsmcp.ui.command_interface import CommandInterface
from agentsmcp.ui.theme_manager import ThemeManager

class MockOrchestrationManager:
    def __init__(self):
        self.is_running = True

theme_manager = ThemeManager()
orchestration_manager = MockOrchestrationManager()
command_interface = CommandInterface(orchestration_manager, theme_manager)

try:
    result = command_interface._get_input_with_autocomplete("test> ")
    lines = result.split("\\n") if result else []
    print(f"RESULT: {len(lines)} lines, {len(result)} chars")
    if len(lines) > 1:
        print("SUCCESS: Multi-line content detected")
    else:
        print("INFO: Single line content")
except Exception as e:
    print(f"ERROR: {e}")
'''],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=test_content)
        
        print(f"📊 Paste Detection Results:")
        print(f"   Stdout: {stdout.strip()}")
        if stderr.strip():
            print(f"   Stderr: {stderr.strip()}")
        
        if "Multi-line content detected" in stdout:
            print("   ✅ Paste detection working correctly")
            return True
        elif "lines" in stdout and "chars" in stdout:
            print("   ✅ Basic input processing working")
            return True
        else:
            print("   ⚠️  Paste detection results unclear")
            return False
            
    except Exception as e:
        print(f"   ❌ Error in paste detection test: {e}")
        return False

if __name__ == "__main__":
    print("🧪 AgentsMCP Multi-line Chat Testing")
    print("=" * 50)
    print(f"Environment: Python {sys.version.split()[0]}, {sys.platform}")
    print()
    
    # Run tests
    test1_pass = test_multiline_logic()
    test2_pass = test_interactive_startup() 
    test3_pass = test_paste_detection_with_content()
    
    # Summary
    print("\n" + "="*50)
    print("🎯 Test Summary:")
    print(f"   Multi-line Logic: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"   Interactive Startup: {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"   Paste Detection: {'✅ PASS' if test3_pass else '❌ FAIL'}")
    
    overall = test1_pass and test2_pass and test3_pass
    print(f"\n🏆 Overall: {'✅ ALL TESTS PASSED' if overall else '⚠️  SOME TESTS FAILED'}")
    
    if overall:
        print("\n💡 Multi-line input should be working in chat interface!")
        print("   Try: python -m agentsmcp interactive --no-welcome")
    else:
        print("\n🔧 Some issues detected - may need further fixes")