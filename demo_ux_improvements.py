#!/usr/bin/env python3
"""Demo script showcasing the Claude Code-level UX improvements in AgentsMCP CLI."""

import subprocess
import sys
import time

def run_command(cmd, description):
    """Run a CLI command and show the output."""
    print(f"\n{'='*60}")
    print(f"🎯 {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")  
            print(result.stderr)
        print(f"Exit code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("Command timed out after 10 seconds")
    except Exception as e:
        print(f"Error running command: {e}")

def main():
    """Demonstrate all UX improvements."""
    print("🚀 AgentsMCP UX Improvements Demo")
    print("=" * 60)
    print("Showcasing Claude Code-level polish and ease of use")
    
    demos = [
        # Progressive Disclosure
        (["./agentsmcp", "run", "--help"], 
         "Progressive Disclosure: Simple vs Advanced Options"),
         
        (["./agentsmcp", "run", "-A", "--help"], 
         "Advanced Mode: Shows Power-User Features"),
        
        # Intelligent Suggestions - Invalid Commands
        (["./agentsmcp", "start"], 
         "Intelligent Suggestions: Typo Detection"),
         
        (["./agentsmcp", "chat"], 
         "Intelligent Suggestions: Command Mapping"), 
         
        # On-demand Suggestions
        (["./agentsmcp", "suggest"], 
         "Context-Aware Suggestions"),
         
        (["./agentsmcp", "suggest", "--all"], 
         "Comprehensive Suggestions with Usage Learning"),
        
        # First-run Experience
        (["./agentsmcp", "init", "--help"], 
         "First-Run Onboarding System"),
        
        # Interactive Mode (will timeout but shows startup)
        (["./agentsmcp", "interactive", "--help"], 
         "Interactive Mode Configuration"),
    ]
    
    for cmd, description in demos:
        run_command(cmd, description)
        time.sleep(1)  # Brief pause between demos
    
    print(f"\n{'='*60}")
    print("🎉 UX IMPROVEMENTS SUMMARY")
    print("=" * 60)
    print("✅ Progressive Disclosure - Simple vs Advanced modes")
    print("✅ Intelligent Command Suggestions - Typo detection & learning")
    print("✅ First-Run Onboarding - Guided setup wizard")
    print("✅ Context-Aware Help - Next-step recommendations")
    print("✅ Enhanced Error Messages - Actionable guidance")
    print("✅ Usage Learning - Personalized suggestions")
    print("\n🚀 AgentsMCP now matches Claude Code CLI level of polish!")

if __name__ == "__main__":
    main()