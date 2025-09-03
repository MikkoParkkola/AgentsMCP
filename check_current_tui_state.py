#!/usr/bin/env python3
"""
SIMPLE TUI STATE CHECKER
Quick diagnostic to see exactly what's happening with the current implementation.
"""

import os
import re
import sys

def check_tui_implementation():
    """Check the current TUI implementation for issues."""
    print("🔍 Checking current TUI implementation...")
    
    tui_file = "/Users/mikko/github/AgentsMCP/src/agentsmcp/ui/v2/revolutionary_tui_interface.py"
    
    with open(tui_file, 'r') as f:
        content = f.read()
    
    # Check the _sync_refresh_display method specifically
    sync_method_match = re.search(
        r'def _sync_refresh_display\(self\):(.*?)(?=def |\Z)', 
        content, re.DOTALL
    )
    
    if sync_method_match:
        method_content = sync_method_match.group(1)
        print("✅ Found _sync_refresh_display method")
        
        # Count refresh calls in this method
        refresh_calls = len(re.findall(r'self\.live_display\.refresh\(\)', method_content))
        print(f"📊 Refresh calls in method: {refresh_calls}")
        
        # Check for tracking variable usage
        tracking_usage = len(re.findall(r'_last_input_refresh_content', method_content))
        print(f"📊 Tracking variable usage: {tracking_usage}")
        
        # Check for error handling
        try_blocks = len(re.findall(r'try:', method_content))
        except_blocks = len(re.findall(r'except:', method_content))
        print(f"📊 Error handling blocks: {try_blocks} try, {except_blocks} except")
        
        # Show recent lines from the method
        lines = method_content.split('\n')
        print(f"\\n📝 Method structure ({len(lines)} lines):")
        for i, line in enumerate(lines[:20], 1):  # First 20 lines
            if line.strip():
                print(f"  {i:2d}: {line}")
        
        if len(lines) > 20:
            print(f"  ... ({len(lines) - 20} more lines)")
    else:
        print("❌ Could not find _sync_refresh_display method!")
    
    # Check for the _create_input_panel method
    input_panel_match = re.search(
        r'def _create_input_panel\(self\):(.*?)(?=def |\Z)', 
        content, re.DOTALL
    )
    
    if input_panel_match:
        method_content = input_panel_match.group(1)
        print(f"\\n✅ Found _create_input_panel method ({len(method_content.split())} lines)")
        
        # Check for input display logic
        input_patterns = [
            r'💬 Input:',
            r'current_input',
            r'Text\(',
            r'return'
        ]
        
        for pattern in input_patterns:
            matches = len(re.findall(pattern, method_content))
            print(f"📊 Pattern '{pattern}': {matches} matches")
    else:
        print("❌ Could not find _create_input_panel method!")

def check_layout_issues():
    """Check for potential layout issues."""
    print("\\n🔍 Checking for layout issues...")
    
    tui_file = "/Users/mikko/github/AgentsMCP/src/agentsmcp/ui/v2/revolutionary_tui_interface.py"
    
    with open(tui_file, 'r') as f:
        content = f.read()
    
    # Look for layout update patterns
    layout_patterns = [
        (r'layout\["input"\]\.update', "Input panel updates"),
        (r'Panel\(', "Panel creations"),
        (r'Live\(', "Live display creations"),
        (r'auto_refresh\s*=', "Auto-refresh settings")
    ]
    
    for pattern, description in layout_patterns:
        matches = len(re.findall(pattern, content))
        print(f"📊 {description}: {matches} occurrences")
        
        if matches > 0:
            # Show context for first match
            match = re.search(pattern, content)
            if match:
                line_num = content[:match.start()].count('\\n') + 1
                lines = content.split('\\n')
                start = max(0, line_num - 2)
                end = min(len(lines), line_num + 2)
                print(f"    Context around line {line_num}:")
                for i in range(start, end):
                    marker = ">>> " if i == line_num - 1 else "    "
                    print(f"    {marker}{i+1}: {lines[i]}")

if __name__ == "__main__":
    check_tui_implementation()
    check_layout_issues()
    print("\\n🎯 Use this output to identify what might be causing the issues!")
    print("\\nNext: Run the comprehensive diagnostic script:")
    print("python debug_tui_typing_issue.py")