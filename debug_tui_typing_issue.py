#!/usr/bin/env python3
"""
TUI TYPING ISSUE DIAGNOSTIC SCRIPT

This script provides detailed diagnostics to identify:
1. Why typing is still not visible
2. What's causing the broken layout lines
3. Refresh mechanism behavior
4. State management issues
"""

import os
import re
import sys
from typing import Dict, List, Tuple

class TUITypingDiagnostic:
    """Comprehensive TUI typing issue diagnostics."""
    
    def __init__(self):
        self.project_root = "/Users/mikko/github/AgentsMCP"
        self.tui_file = f"{self.project_root}/src/agentsmcp/ui/v2/revolutionary_tui_interface.py"
    
    def diagnose_refresh_mechanism(self) -> Dict[str, any]:
        """Analyze the current refresh mechanism implementation."""
        print("🔍 DIAGNOSTIC 1: Refresh Mechanism Analysis")
        print("-" * 50)
        
        try:
            with open(self.tui_file, 'r') as f:
                content = f.read()
            
            results = {
                "refresh_calls": [],
                "refresh_conditions": [],
                "tracking_variables": [],
                "potential_issues": []
            }
            
            # Find all refresh calls and their context
            refresh_matches = list(re.finditer(r'self\.live_display\.refresh\(\)', content))
            for match in refresh_matches:
                line_num = content[:match.start()].count('\n') + 1
                # Get 5 lines of context around the call
                lines = content.split('\n')
                start_line = max(0, line_num - 6)
                end_line = min(len(lines), line_num + 5)
                context = lines[start_line:end_line]
                
                results["refresh_calls"].append({
                    "line": line_num,
                    "context": context
                })
            
            # Find refresh conditions
            condition_patterns = [
                r'if.*hasattr.*refresh',
                r'if.*_last_input_refresh_content',
                r'if.*current_input'
            ]
            
            for pattern in condition_patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = content.split('\n')[line_num - 1].strip()
                    results["refresh_conditions"].append({
                        "line": line_num,
                        "condition": line_content
                    })
            
            # Find tracking variables
            tracking_patterns = [
                r'_last_input_refresh_content',
                r'self\.state\.current_input',
                r'input_buffer'
            ]
            
            for pattern in tracking_patterns:
                matches = len(re.findall(pattern, content))
                if matches > 0:
                    results["tracking_variables"].append({
                        "pattern": pattern,
                        "occurrences": matches
                    })
            
            print(f"Found {len(results['refresh_calls'])} refresh calls")
            print(f"Found {len(results['refresh_conditions'])} refresh conditions")
            print(f"Found {len(results['tracking_variables'])} tracking variable patterns")
            
            return results
            
        except Exception as e:
            print(f"❌ Diagnostic failed: {e}")
            return {"error": str(e)}
    
    def diagnose_layout_structure(self) -> Dict[str, any]:
        """Analyze layout creation and update mechanisms."""
        print("\\n🔍 DIAGNOSTIC 2: Layout Structure Analysis")
        print("-" * 50)
        
        try:
            with open(self.tui_file, 'r') as f:
                content = f.read()
            
            results = {
                "layout_creation": [],
                "panel_updates": [],
                "layout_keys": [],
                "potential_issues": []
            }
            
            # Find layout creation patterns
            layout_patterns = [
                r'Layout\(',
                r'self\.layout\s*=',
                r'layout\[.*\]\.update'
            ]
            
            for pattern in layout_patterns:
                matches = list(re.finditer(pattern, content))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = content.split('\n')[line_num - 1].strip()
                    results["layout_creation"].append({
                        "line": line_num,
                        "content": line_content
                    })
            
            # Find panel update patterns
            panel_patterns = [
                r'Panel\(',
                r'\.update\s*\(\s*Panel',
                r'_create_.*_panel'
            ]
            
            for pattern in panel_patterns:
                matches = list(re.finditer(pattern, content))
                results["panel_updates"].extend([
                    {
                        "line": content[:match.start()].count('\n') + 1,
                        "pattern": pattern
                    }
                    for match in matches
                ])
            
            # Look for layout key usage
            key_patterns = [
                r'layout\["input"\]',
                r'layout\["output"\]',
                r'layout\["status"\]'
            ]
            
            for pattern in key_patterns:
                matches = len(re.findall(pattern, content))
                if matches > 0:
                    results["layout_keys"].append({
                        "key": pattern,
                        "usage_count": matches
                    })
            
            print(f"Found {len(results['layout_creation'])} layout creation/assignment patterns")
            print(f"Found {len(results['panel_updates'])} panel update patterns")
            print(f"Found {len(results['layout_keys'])} layout key usage patterns")
            
            return results
            
        except Exception as e:
            print(f"❌ Layout diagnostic failed: {e}")
            return {"error": str(e)}
    
    def diagnose_input_handling(self) -> Dict[str, any]:
        """Analyze input character handling and state management."""
        print("\\n🔍 DIAGNOSTIC 3: Input Handling Analysis")  
        print("-" * 50)
        
        try:
            with open(self.tui_file, 'r') as f:
                content = f.read()
            
            results = {
                "input_handlers": [],
                "state_updates": [],
                "panel_creation": [],
                "debug_logging": []
            }
            
            # Find input handling methods
            handler_patterns = [
                r'def _handle_character_input',
                r'def _create_input_panel',
                r'def _sync_refresh_display'
            ]
            
            for pattern in handler_patterns:
                matches = list(re.finditer(pattern, content))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    results["input_handlers"].append({
                        "method": pattern.replace('def ', '').replace(r'\b', ''),
                        "line": line_num
                    })
            
            # Find state update patterns
            state_patterns = [
                r'self\.state\.current_input\s*=',
                r'self\.state\.current_input\s*\+=',
                r'current_input.*=.*char'
            ]
            
            for pattern in state_patterns:
                matches = list(re.finditer(pattern, content))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = content.split('\n')[line_num - 1].strip()
                    results["state_updates"].append({
                        "line": line_num,
                        "update": line_content
                    })
            
            # Find input panel creation logic
            panel_creation_patterns = [
                r'💬 Input:',
                r'Text\(.*current_input',
                r'return.*Text\('
            ]
            
            for pattern in panel_creation_patterns:
                matches = len(re.findall(pattern, content))
                if matches > 0:
                    results["panel_creation"].append({
                        "pattern": pattern,
                        "occurrences": matches
                    })
            
            # Find debug logging related to input
            debug_patterns = [
                r'_safe_log.*INPUT',
                r'_safe_log.*SYNC_REFRESH',
                r'debug.*current_input'
            ]
            
            for pattern in debug_patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                if matches > 0:
                    results["debug_logging"].append({
                        "pattern": pattern,
                        "occurrences": matches
                    })
            
            print(f"Found {len(results['input_handlers'])} input handling methods")
            print(f"Found {len(results['state_updates'])} state update patterns")  
            print(f"Found {len(results['panel_creation'])} panel creation patterns")
            print(f"Found {len(results['debug_logging'])} debug logging patterns")
            
            return results
            
        except Exception as e:
            print(f"❌ Input handling diagnostic failed: {e}")
            return {"error": str(e)}
    
    def diagnose_recent_changes(self) -> Dict[str, any]:
        """Analyze what changed in the recent fixes that might cause issues."""
        print("\\n🔍 DIAGNOSTIC 4: Recent Changes Analysis")
        print("-" * 50)
        
        try:
            with open(self.tui_file, 'r') as f:
                content = f.read()
            
            # Look for patterns that suggest recent changes
            change_indicators = [
                r'TARGETED REFRESH FIX',
                r'CRITICAL FIX',
                r'# FIXED:',
                r'_last_input_refresh_content'
            ]
            
            results = {"recent_changes": [], "potential_conflicts": []}
            
            for pattern in change_indicators:
                matches = list(re.finditer(pattern, content))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    # Get context around the change
                    lines = content.split('\n')
                    start_line = max(0, line_num - 3)
                    end_line = min(len(lines), line_num + 10)
                    context = lines[start_line:end_line]
                    
                    results["recent_changes"].append({
                        "indicator": pattern,
                        "line": line_num,
                        "context": context
                    })
            
            # Look for potential conflicts
            conflict_patterns = [
                (r'auto_refresh\s*=\s*False', r'self\.live_display\.refresh\(\)'),
                (r'layout\[.*\]\.update', r'refresh\(\)'),
                (r'Panel\(.*input_content', r'_last_input_refresh_content')
            ]
            
            for pattern1, pattern2 in conflict_patterns:
                if re.search(pattern1, content) and re.search(pattern2, content):
                    results["potential_conflicts"].append({
                        "conflict": f"{pattern1} + {pattern2}",
                        "description": "These patterns might interact unexpectedly"
                    })
            
            print(f"Found {len(results['recent_changes'])} recent change indicators")
            print(f"Found {len(results['potential_conflicts'])} potential conflicts")
            
            return results
            
        except Exception as e:
            print(f"❌ Recent changes diagnostic failed: {e}")
            return {"error": str(e)}
    
    def generate_debug_tui_script(self) -> str:
        """Generate a script to run TUI with maximum debug output."""
        script_content = '''#!/usr/bin/env python3
"""
DEBUG TUI RUNNER - Maximum verbosity for troubleshooting
"""

import os
import sys
import subprocess

def run_debug_tui():
    """Run TUI with maximum debug output."""
    print("🔍 Starting TUI with maximum debug output...")
    print("=" * 60)
    
    # Set debug environment
    env = os.environ.copy()
    env['AGENTSMCP_DEBUG'] = 'true'
    env['PYTHONPATH'] = '/Users/mikko/github/AgentsMCP/src'
    
    try:
        # Run with debug mode and capture all output
        result = subprocess.run([
            sys.executable, '-c', 
            """
import sys
sys.path.insert(0, '/Users/mikko/github/AgentsMCP/src')

from agentsmcp.ui.cli_app import CLIConfig
from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface

# Create debug config
class DebugConfig:
    debug_mode = True
    verbose = True

config = DebugConfig()
tui = RevolutionaryTUIInterface(cli_config=config)

print("🔍 TUI created, about to test input handling...")

# Test basic functionality
print("Testing _create_input_panel()...")
try:
    tui.state.current_input = "test input"
    panel = tui._create_input_panel()
    print(f"✅ Input panel created: {type(panel)}")
    print(f"✅ Current input state: '{tui.state.current_input}'")
except Exception as e:
    print(f"❌ Input panel creation failed: {e}")

print("Testing _sync_refresh_display()...")
try:
    tui._sync_refresh_display()
    print("✅ Refresh display completed")
except Exception as e:
    print(f"❌ Refresh display failed: {e}")

print("🔍 Debug test completed")
            """
        ], env=env, capture_output=True, text=True, timeout=30)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("\\nSTDERR:")
            print(result.stderr)
        
        print(f"\\n🎯 Exit code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("⏰ Debug run timed out after 30 seconds")
    except Exception as e:
        print(f"❌ Debug run failed: {e}")

if __name__ == "__main__":
    run_debug_tui()
'''
        
        debug_script_path = "/Users/mikko/github/AgentsMCP/run_debug_tui_test.py"
        with open(debug_script_path, 'w') as f:
            f.write(script_content)
        
        print(f"\\n📝 Generated debug TUI script: {debug_script_path}")
        return debug_script_path
    
    def run_full_diagnostic(self):
        """Run all diagnostics and generate report."""
        print("🚀 TUI TYPING ISSUE - COMPREHENSIVE DIAGNOSTIC")
        print("=" * 70)
        print("Analyzing current implementation to identify root causes...")
        
        # Run all diagnostics
        refresh_results = self.diagnose_refresh_mechanism()
        layout_results = self.diagnose_layout_structure()  
        input_results = self.diagnose_input_handling()
        changes_results = self.diagnose_recent_changes()
        
        # Generate debug script
        debug_script = self.generate_debug_tui_script()
        
        print("\\n" + "=" * 70)
        print("🎯 DIAGNOSTIC SUMMARY")
        print("=" * 70)
        
        print("\\n📋 FINDINGS:")
        print(f"• Refresh mechanism: {len(refresh_results.get('refresh_calls', []))} calls found")
        print(f"• Layout operations: {len(layout_results.get('layout_creation', []))} patterns found")
        print(f"• Input handlers: {len(input_results.get('input_handlers', []))} methods found") 
        print(f"• Recent changes: {len(changes_results.get('recent_changes', []))} indicators found")
        
        print("\\n🛠️  NEXT STEPS FOR TROUBLESHOOTING:")
        print("1. Run the debug script to see runtime behavior:")
        print(f"   python {debug_script}")
        print("\\n2. Check for specific error patterns in the output")
        print("\\n3. Based on results, we can create targeted fixes")
        
        return {
            "refresh_results": refresh_results,
            "layout_results": layout_results,
            "input_results": input_results, 
            "changes_results": changes_results,
            "debug_script": debug_script
        }

def main():
    """Run the comprehensive TUI diagnostic."""
    diagnostic = TUITypingDiagnostic()
    results = diagnostic.run_full_diagnostic()
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)