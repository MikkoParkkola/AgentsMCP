#!/usr/bin/env python3

"""
Complete Streaming Fixes Verification Test

Tests the critical streaming display issues and ensures:
1. No mixed status/streaming lines
2. Complete final response displays after streaming
3. Clean streaming experience without truncation
4. Proper separation of status updates during streaming
"""

import asyncio
import sys
import os
import subprocess
import tempfile
import time
import threading
from pathlib import Path

# Test scenarios to verify fixes
test_scenarios = [
    {
        "name": "Basic streaming response",
        "command": "What is Python?",
        "expected_patterns": [
            "🤖 AI (streaming):",  # Streaming indicator
            "🤖 Assistant:",       # Final message
        ],
        "anti_patterns": [
            "⏳ 📡 Stream Manager: 🎯 Streaming",  # Should not mix with streaming content
            "...🎯 Coordinator",  # Should not mix with streaming
        ]
    },
    {
        "name": "Long response streaming", 
        "command": "/help",
        "expected_patterns": [
            "🤖 AI Command Composer - Help",  # Complete help response
            "💬 **Chat Commands:**",          # Full content displayed
        ],
        "anti_patterns": [
            "...",  # Should not truncate final response
        ]
    }
]

class StreamingTestHarness:
    """Test harness for streaming display verification."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.agentsmcp_path = self.project_root / "agentsmcp"
        
    def run_streaming_test(self, scenario):
        """Run a single streaming test scenario."""
        print(f"\n🧪 Testing: {scenario['name']}")
        print(f"Command: {scenario['command']}")
        
        # Create temporary script to send command and capture output
        script_content = f"""#!/usr/bin/env python3
import subprocess
import sys
import time
import threading

def send_input():
    time.sleep(1)  # Give TUI time to start
    proc.stdin.write('{scenario['command']}\\n')
    proc.stdin.flush()
    time.sleep(3)  # Wait for response
    proc.stdin.write('/quit\\n')
    proc.stdin.flush()

proc = subprocess.Popen(
    [sys.executable, '-m', 'agentsmcp.cli', 'tui'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    cwd='{self.project_root}',
    env={{**dict(os.environ), 'PYTHONPATH': '{self.project_root}'}}
)

# Start input thread
input_thread = threading.Thread(target=send_input)
input_thread.start()

# Capture output
output = proc.communicate()[0]
input_thread.join()

print("=== CAPTURED OUTPUT ===")
print(output)
print("=== END OUTPUT ===")
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            # Run the test script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.project_root
            )
            
            output = result.stdout + result.stderr
            print("📋 Raw output:")
            print(output)
            
            # Analyze output for patterns
            success = True
            
            # Check expected patterns
            for pattern in scenario['expected_patterns']:
                if pattern not in output:
                    print(f"❌ Missing expected pattern: '{pattern}'")
                    success = False
                else:
                    print(f"✅ Found expected pattern: '{pattern}'")
            
            # Check anti-patterns (things that should NOT appear)
            for anti_pattern in scenario['anti_patterns']:
                if anti_pattern in output:
                    print(f"❌ Found problematic pattern: '{anti_pattern}'")
                    success = False
                else:
                    print(f"✅ Avoided problematic pattern: '{anti_pattern}'")
            
            # Additional streaming-specific checks
            lines = output.split('\n')
            
            # Check for clean streaming lines (no mixing with status)
            streaming_lines = [line for line in lines if "🤖 AI (streaming):" in line]
            for line in streaming_lines:
                if "⏳" in line or "📡" in line:
                    print(f"❌ Streaming line mixed with status: '{line.strip()}'")
                    success = False
                else:
                    print(f"✅ Clean streaming line: '{line.strip()}'")
            
            # Check for final complete message
            final_messages = [line for line in lines if "🤖 Assistant:" in line]
            if not final_messages:
                print("❌ No final complete assistant message found")
                success = False
            else:
                print(f"✅ Found final complete message: '{final_messages[0].strip()[:100]}...'")
            
            return success
            
        except subprocess.TimeoutExpired:
            print("❌ Test timed out")
            return False
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False
        finally:
            # Cleanup
            try:
                os.unlink(script_path)
            except:
                pass
    
    def run_all_tests(self):
        """Run all streaming test scenarios."""
        print("🧪 Complete Streaming Fixes Verification Test")
        print("=" * 60)
        print("Testing critical streaming display issues:")
        print("• No mixed status/streaming lines")
        print("• Complete final response display")
        print("• Clean streaming without truncation")
        print("• Proper status suppression during streaming")
        
        all_passed = True
        results = []
        
        for scenario in test_scenarios:
            success = self.run_streaming_test(scenario)
            results.append((scenario['name'], success))
            if not success:
                all_passed = False
        
        # Summary
        print(f"\n{'='*60}")
        print("🎯 TEST SUMMARY")
        print(f"{'='*60}")
        
        for name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}: {name}")
        
        overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
        print(f"\n{overall_status}")
        
        if all_passed:
            print("\n🎉 Streaming display fixes verified successfully!")
            print("• Status updates properly suppressed during streaming")
            print("• Final complete responses display correctly")
            print("• No more mixed status/streaming output")
            print("• Clean professional streaming experience")
        else:
            print("\n⚠️  Streaming issues still present:")
            print("• Check logs above for specific failures")
            print("• Focus on status/streaming separation")
            print("• Ensure final messages display completely")
        
        return all_passed

def main():
    """Run streaming fixes verification."""
    harness = StreamingTestHarness()
    
    # Verify AgentsMCP is available
    if not harness.agentsmcp_path.exists():
        print("❌ AgentsMCP not found. Run from project root.")
        return 1
    
    success = harness.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())