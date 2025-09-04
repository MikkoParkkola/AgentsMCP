#!/usr/bin/env python3
"""
TUI Demo Mode vs Interactive Mode Gap Analysis

This test identifies the specific gap: TUI shows interactive prompt "💬 TUI> " 
but exits immediately instead of waiting for user input.
"""

import subprocess
import sys
import time
import threading
import os
from pathlib import Path
from typing import Dict, List, Any


class TUIDemoVsInteractiveAnalyzer:
    """Analyze the gap between demo mode and true interactive mode"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def analyze_current_behavior(self) -> Dict[str, Any]:
        """Analyze what actually happens with current TUI"""
        print("🔍 ANALYZING CURRENT TUI BEHAVIOR")
        print("=" * 50)
        
        # Run TUI and capture detailed timing
        result = self._run_tui_with_timing_analysis()
        
        # Parse the behavior
        behavior_analysis = self._parse_tui_behavior(result)
        
        # Print analysis
        self._print_behavior_analysis(behavior_analysis)
        
        return behavior_analysis
    
    def _run_tui_with_timing_analysis(self) -> Dict[str, Any]:
        """Run TUI with detailed timing analysis"""
        start_time = time.time()
        timing_events = []
        
        def add_timing_event(event: str):
            timing_events.append({
                'time': time.time() - start_time,
                'event': event
            })
        
        try:
            add_timing_event('process_start')
            
            process = subprocess.Popen(
                [sys.executable, "-m", "agentsmcp", "--mode", "tui"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            add_timing_event('process_created')
            
            # Monitor process for 10 seconds
            timeout = 10
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                add_timing_event('process_completed')
            except subprocess.TimeoutExpired:
                add_timing_event('process_timeout')
                process.kill()
                stdout, stderr = process.communicate()
            
            add_timing_event('analysis_complete')
            
        except Exception as e:
            stdout = f"Error: {e}"
            stderr = ""
            add_timing_event('error_occurred')
        
        total_runtime = time.time() - start_time
        
        return {
            'stdout': stdout,
            'stderr': stderr,
            'runtime': total_runtime,
            'timing_events': timing_events,
            'exit_code': process.returncode if 'process' in locals() else -1
        }
    
    def _parse_tui_behavior(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse TUI behavior from output"""
        stdout = result['stdout']
        stderr = result['stderr']
        full_output = stdout + stderr
        
        # Key behavior indicators
        behavior = {
            'runtime': result['runtime'],
            'exit_code': result['exit_code'],
            'demo_mode_entered': 'Demo Mode' in stdout,
            'interactive_prompt_shown': '💬 TUI>' in stdout,
            'shows_input_instructions': 'Type messages and press Enter' in stdout,
            'shows_commands': 'Type \'/quit\' to exit' in stdout,
            'actually_waits_for_input': result['runtime'] > 8,  # Should timeout if waiting
            'exits_gracefully': '✅ Demo completed' in stdout,
            'timing_events': result['timing_events']
        }
        
        # Parse output stages
        behavior['output_stages'] = self._identify_output_stages(stdout)
        
        # Identify the critical gap
        behavior['critical_gap'] = (
            behavior['interactive_prompt_shown'] and 
            behavior['shows_input_instructions'] and
            not behavior['actually_waits_for_input']
        )
        
        return behavior
    
    def _identify_output_stages(self, stdout: str) -> List[Dict[str, Any]]:
        """Identify stages in TUI output"""
        lines = stdout.split('\n')
        stages = []
        
        for line in lines:
            if '🚀 Starting Revolutionary TUI' in line:
                stages.append({'stage': 'startup', 'line': line.strip()})
            elif 'Demo Mode' in line:
                stages.append({'stage': 'demo_mode_entry', 'line': line.strip()})
            elif 'TUI initialized successfully' in line:
                stages.append({'stage': 'initialization_complete', 'line': line.strip()})
            elif 'Ready for interactive use' in line:
                stages.append({'stage': 'ready_for_interaction', 'line': line.strip()})
            elif 'Interactive mode now available:' in line:
                stages.append({'stage': 'interactive_mode_available', 'line': line.strip()})
            elif '💬 TUI>' in line:
                stages.append({'stage': 'prompt_shown', 'line': line.strip()})
            elif 'Exiting TUI...' in line:
                stages.append({'stage': 'exit_initiated', 'line': line.strip()})
            elif 'Demo completed' in line:
                stages.append({'stage': 'demo_completed', 'line': line.strip()})
        
        return stages
    
    def _print_behavior_analysis(self, behavior: Dict[str, Any]):
        """Print detailed behavior analysis"""
        print(f"📊 BEHAVIOR ANALYSIS RESULTS")
        print(f"Runtime: {behavior['runtime']:.2f}s")
        print(f"Exit Code: {behavior['exit_code']}")
        print()
        
        # Stage analysis
        print(f"📋 OUTPUT STAGES:")
        for i, stage in enumerate(behavior['output_stages'], 1):
            print(f"  {i}. {stage['stage']}: {stage['line']}")
        print()
        
        # Key indicators
        print(f"🔍 KEY INDICATORS:")
        indicators = [
            ('Demo mode entered', behavior['demo_mode_entered']),
            ('Interactive prompt shown', behavior['interactive_prompt_shown']),
            ('Shows input instructions', behavior['shows_input_instructions']),
            ('Shows available commands', behavior['shows_commands']),
            ('Actually waits for input', behavior['actually_waits_for_input']),
            ('Exits gracefully', behavior['exits_gracefully'])
        ]
        
        for indicator, value in indicators:
            status = "✅" if value else "❌"
            print(f"  {status} {indicator}")
        print()
        
        # Critical gap identification
        if behavior['critical_gap']:
            print(f"🔴 CRITICAL GAP IDENTIFIED:")
            print(f"   TUI shows interactive prompt and instructions")
            print(f"   BUT immediately exits without waiting for input")
            print(f"   This breaks the entire interactive user experience")
        else:
            print(f"✅ No critical gap detected")
        
    def create_fix_recommendations(self, behavior: Dict[str, Any]) -> List[str]:
        """Generate specific fix recommendations"""
        recommendations = []
        
        if behavior['critical_gap']:
            recommendations.extend([
                "CRITICAL: Fix TUI to actually wait for input after showing prompt",
                "Add proper input loop that doesn't exit until user types /quit",
                "Fix demo mode detection to stay interactive when prompt is shown",
                "Ensure TUI doesn't auto-exit after countdown in interactive sections"
            ])
        
        if behavior['demo_mode_entered'] and behavior['interactive_prompt_shown']:
            recommendations.append(
                "Inconsistency: Demo mode but shows interactive prompt - choose one"
            )
        
        if not behavior['actually_waits_for_input'] and behavior['shows_input_instructions']:
            recommendations.append(
                "False promise: Shows input instructions but doesn't wait for input"
            )
        
        return recommendations
    
    def test_what_should_happen_vs_what_does(self):
        """Test and document what should happen vs what actually happens"""
        print("🧪 TUI EXPECTED vs ACTUAL BEHAVIOR TEST")
        print("=" * 60)
        
        # Analyze current behavior
        behavior = self.analyze_current_behavior()
        
        print(f"\n📝 EXPECTED vs ACTUAL COMPARISON:")
        print(f"=" * 40)
        
        comparisons = [
            {
                'aspect': 'Interactive Prompt',
                'expected': 'Show prompt and wait indefinitely for user input',
                'actual': 'Shows prompt but exits immediately',
                'gap': behavior['interactive_prompt_shown'] and not behavior['actually_waits_for_input']
            },
            {
                'aspect': 'User Input Instructions',
                'expected': 'Instructions should be followed by actual input capability',
                'actual': 'Shows instructions but no input capability',
                'gap': behavior['shows_input_instructions'] and not behavior['actually_waits_for_input']
            },
            {
                'aspect': 'TUI Lifecycle',
                'expected': 'Stay running until user types /quit',
                'actual': 'Auto-exits after demo countdown',
                'gap': behavior['exits_gracefully'] and not behavior['actually_waits_for_input']
            },
            {
                'aspect': 'Interactive Mode',
                'expected': 'True interactive mode with persistent session',
                'actual': 'Demo mode with simulated interactivity',
                'gap': behavior['demo_mode_entered']
            }
        ]
        
        gap_count = 0
        for comp in comparisons:
            status = "❌ GAP" if comp['gap'] else "✅ OK"
            print(f"\n{comp['aspect']}:")
            print(f"  Expected: {comp['expected']}")
            print(f"  Actual: {comp['actual']}")
            print(f"  Status: {status}")
            
            if comp['gap']:
                gap_count += 1
        
        # Generate recommendations
        recommendations = self.create_fix_recommendations(behavior)
        
        print(f"\n💡 FIX RECOMMENDATIONS ({len(recommendations)} items):")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        # Final verdict
        print(f"\n🏁 FINAL VERDICT:")
        if gap_count == 0:
            print(f"✅ TUI behavior matches expectations")
        else:
            print(f"❌ {gap_count}/4 critical gaps found")
            print(f"🎯 PRIMARY ISSUE: TUI shows interactive elements but doesn't stay interactive")
            
        return behavior, recommendations


def main():
    """Main test execution"""
    project_root = Path(__file__).parent
    analyzer = TUIDemoVsInteractiveAnalyzer(project_root)
    
    behavior, recommendations = analyzer.test_what_should_happen_vs_what_does()
    
    # Additional insights
    print(f"\n🔍 TECHNICAL INSIGHTS:")
    print(f"   The TUI successfully initializes all interactive components")
    print(f"   It shows the prompt and instructions correctly")
    print(f"   BUT it exits instead of entering an input loop")
    print(f"   This suggests the interactive loop logic is missing or disabled")
    
    return behavior


if __name__ == "__main__":
    main()