#!/usr/bin/env python3
"""
Automated tests for TUI flooding and empty line issues.
These tests must pass before the issues are considered fixed.
"""
import subprocess
import time
import sys
import os
import re
from typing import List, Tuple

class TUIIssueTests:
    def __init__(self):
        self.test_results = []
    
    def run_tui_and_capture(self, duration: float = 5.0, debug: bool = False) -> Tuple[List[str], float]:
        """Run TUI for specified duration and capture all output."""
        cmd = ["./agentsmcp", "tui"]
        if debug:
            cmd.append("--debug")
        
        start_time = time.time()
        captured_lines = []
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                bufsize=0,  # Unbuffered
                universal_newlines=True
            )
            
            # Capture output for specified duration
            while time.time() - start_time < duration:
                try:
                    import select
                    ready, _, _ = select.select([process.stdout], [], [], 0.1)
                    if ready:
                        line = process.stdout.readline()
                        if line:
                            captured_lines.append(line.rstrip())
                    
                    # Check if process ended
                    if process.poll() is not None:
                        break
                        
                except Exception:
                    break
            
            # Kill the process
            try:
                process.terminate()
                time.sleep(0.2)
                if process.poll() is None:
                    process.kill()
                    time.sleep(0.2)
            except:
                pass
            
            # Get any remaining output
            try:
                stdout, _ = process.communicate(timeout=1)
                if stdout:
                    for line in stdout.split('\n'):
                        if line.strip():
                            captured_lines.append(line.rstrip())
            except:
                pass
                
        except Exception as e:
            print(f"Error capturing TUI output: {e}")
            return [], 0.0
        
        actual_duration = time.time() - start_time
        return captured_lines, actual_duration
    
    def test_no_flooding(self) -> bool:
        """Test that TUI doesn't flood scrollback with excessive output."""
        print("🔍 Testing: No flooding (reasonable output rate)...")
        
        lines, duration = self.run_tui_and_capture(duration=3.0, debug=False)
        
        if duration < 1.0:
            print(f"❌ Test failed: Duration too short ({duration:.1f}s)")
            return False
        
        lines_per_second = len(lines) / duration
        max_acceptable_rate = 10.0  # Max 10 lines per second is reasonable
        
        print(f"   📊 Captured {len(lines)} lines in {duration:.1f}s")
        print(f"   📈 Output rate: {lines_per_second:.1f} lines/second")
        print(f"   🎯 Acceptable rate: ≤{max_acceptable_rate} lines/second")
        
        if lines_per_second > max_acceptable_rate:
            print(f"❌ FLOODING DETECTED: {lines_per_second:.1f} lines/sec exceeds {max_acceptable_rate}")
            print("   First 10 lines:")
            for i, line in enumerate(lines[:10], 1):
                print(f"     {i:2d}: {line[:80]}")
            if len(lines) > 10:
                print(f"     ... and {len(lines) - 10} more lines")
            self.test_results.append(("flooding", False, f"{lines_per_second:.1f} lines/sec"))
            return False
        else:
            print(f"✅ No flooding: {lines_per_second:.1f} lines/sec is acceptable")
            self.test_results.append(("flooding", True, f"{lines_per_second:.1f} lines/sec"))
            return True
    
    def test_no_excessive_empty_lines(self) -> bool:
        """Test that TUI doesn't have excessive empty lines."""
        print("\n🔍 Testing: No excessive empty lines...")
        
        lines, duration = self.run_tui_and_capture(duration=4.0, debug=False)
        
        if len(lines) < 10:
            print(f"❌ Test failed: Too few lines captured ({len(lines)})")
            return False
        
        empty_count = sum(1 for line in lines if not line.strip())
        empty_percentage = (empty_count / len(lines)) * 100 if lines else 0
        max_acceptable_empty = 20.0  # Max 20% empty lines
        
        print(f"   📊 Total lines: {len(lines)}")
        print(f"   📊 Empty lines: {empty_count}")
        print(f"   📈 Empty percentage: {empty_percentage:.1f}%")
        print(f"   🎯 Acceptable: ≤{max_acceptable_empty}%")
        
        # Also check for consecutive empty lines (more than 2 in a row is suspicious)
        consecutive_empty = 0
        max_consecutive = 0
        for line in lines:
            if not line.strip():
                consecutive_empty += 1
                max_consecutive = max(max_consecutive, consecutive_empty)
            else:
                consecutive_empty = 0
        
        print(f"   📊 Max consecutive empty: {max_consecutive}")
        
        if empty_percentage > max_acceptable_empty:
            print(f"❌ EXCESSIVE EMPTY LINES: {empty_percentage:.1f}% exceeds {max_acceptable_empty}%")
            print("   Sample with empty line markers:")
            for i, line in enumerate(lines[:15], 1):
                marker = " [EMPTY]" if not line.strip() else ""
                print(f"     {i:2d}: {repr(line[:60])}{marker}")
            self.test_results.append(("empty_lines", False, f"{empty_percentage:.1f}%"))
            return False
        elif max_consecutive > 3:
            print(f"❌ TOO MANY CONSECUTIVE EMPTY LINES: {max_consecutive} in a row")
            self.test_results.append(("empty_lines", False, f"{max_consecutive} consecutive"))
            return False
        else:
            print(f"✅ Empty lines acceptable: {empty_percentage:.1f}% with max {max_consecutive} consecutive")
            self.test_results.append(("empty_lines", True, f"{empty_percentage:.1f}%"))
            return True
    
    def test_quit_command_works(self) -> bool:
        """Test that quit command actually works."""
        print("\n🔍 Testing: Quit command functionality...")
        
        try:
            # Test with simulated input
            process = subprocess.Popen(
                ["./agentsmcp", "tui"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it 2 seconds to start up, then send quit
            time.sleep(2)
            stdout, stderr = process.communicate(input="quit\n", timeout=8)
            
            # Check if it exited cleanly
            exit_code = process.returncode
            
            # Look for quit confirmation message
            quit_found = "👋 Exiting Revolutionary TUI" in stdout or "quit" in stdout.lower()
            
            print(f"   📊 Exit code: {exit_code}")
            print(f"   📊 Quit message found: {quit_found}")
            
            if exit_code == 0 and quit_found:
                print("✅ Quit command works correctly")
                self.test_results.append(("quit_command", True, f"exit_code={exit_code}"))
                return True
            else:
                print(f"❌ Quit command failed: exit_code={exit_code}, quit_message={quit_found}")
                print("STDOUT:", stdout[:500])
                self.test_results.append(("quit_command", False, f"exit_code={exit_code}"))
                return False
                
        except subprocess.TimeoutExpired:
            process.kill()
            print("❌ Quit command timed out - command not working")
            self.test_results.append(("quit_command", False, "timeout"))
            return False
        except Exception as e:
            print(f"❌ Quit command test failed: {e}")
            self.test_results.append(("quit_command", False, str(e)))
            return False
    
    def test_scrollback_pollution(self) -> bool:
        """Test that TUI doesn't pollute terminal scrollback history."""
        print("\n🔍 Testing: No scrollback pollution...")
        
        # This is harder to test directly, but we can check for patterns that indicate
        # excessive screen clearing or redrawing
        lines, duration = self.run_tui_and_capture(duration=3.0, debug=False)
        
        # Look for patterns that indicate screen control issues
        clear_patterns = [
            r'\033\[2J',      # Clear screen
            r'\033\[H',       # Home cursor
            r'\033\[\d+;\d+H', # Position cursor
            r'\033\[K',       # Clear to end of line
            r'\033\[0m',      # Reset formatting (excessive use)
        ]
        
        total_control_sequences = 0
        for line in lines:
            for pattern in clear_patterns:
                total_control_sequences += len(re.findall(pattern, line))
        
        control_per_second = total_control_sequences / duration if duration > 0 else 0
        max_acceptable_control = 5.0  # Max 5 control sequences per second
        
        print(f"   📊 Control sequences found: {total_control_sequences}")
        print(f"   📈 Control seq rate: {control_per_second:.1f}/sec")
        print(f"   🎯 Acceptable rate: ≤{max_acceptable_control}/sec")
        
        if control_per_second > max_acceptable_control:
            print(f"❌ SCROLLBACK POLLUTION: {control_per_second:.1f} control seq/sec")
            self.test_results.append(("scrollback", False, f"{control_per_second:.1f}/sec"))
            return False
        else:
            print(f"✅ No scrollback pollution: {control_per_second:.1f}/sec acceptable")
            self.test_results.append(("scrollback", True, f"{control_per_second:.1f}/sec"))
            return True
    
    def run_all_tests(self) -> bool:
        """Run all tests and return True if all pass."""
        print("🧪 Running TUI Issue Tests...")
        print("=" * 60)
        
        tests = [
            self.test_no_flooding,
            self.test_no_excessive_empty_lines, 
            self.test_quit_command_works,
            self.test_scrollback_pollution
        ]
        
        all_passed = True
        for test in tests:
            passed = test()
            if not passed:
                all_passed = False
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY:")
        
        for test_name, passed, details in self.test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {test_name:15s}: {status} ({details})")
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! TUI issues are resolved.")
        else:
            failed_count = sum(1 for _, passed, _ in self.test_results if not passed)
            print(f"\n💥 {failed_count} TEST(S) FAILED! Issues still exist.")
        
        return all_passed

def main():
    """Run the TUI tests."""
    if not os.path.exists("./agentsmcp"):
        print("❌ Error: ./agentsmcp executable not found")
        print("   Make sure you're running this from the correct directory")
        return False
    
    tester = TUIIssueTests()
    return tester.run_all_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)