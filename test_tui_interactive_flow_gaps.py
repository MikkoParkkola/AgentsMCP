#!/usr/bin/env python3
"""
TUI Interactive Flow Gap Analysis

Tests specific interactive scenarios that should work but don't,
focusing on the user experience flow from startup to interaction.
"""

import subprocess
import sys
import time
import os
import pty  
import select
import signal
from pathlib import Path
from typing import List, Tuple, Optional


class InteractiveFlowTester:
    """Test interactive TUI flow scenarios"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        
    def test_startup_to_ready_flow(self) -> dict:
        """Test the complete startup-to-ready flow"""
        print("🔄 TESTING: Startup to Ready Flow")
        print("Expected: TUI starts → Shows interface → Waits for input → Stays ready")
        
        result = self._run_tui_capture_output(timeout=6)
        
        # Analyze the flow
        flow_analysis = self._analyze_startup_flow(result['output'])
        
        print(f"Actual behavior:")
        print(f"  Runtime: {result['runtime']:.2f}s")  
        print(f"  Exit code: {result['exit_code']}")
        print(f"  Flow stages detected:")
        for stage, detected in flow_analysis.items():
            status = "✅" if detected else "❌"
            print(f"    {status} {stage}")
        
        # Identify the gap
        expected_stages = ['startup', 'interface_ready', 'waiting_for_input', 'stays_running']
        missing_stages = [stage for stage in expected_stages if not flow_analysis.get(stage, False)]
        
        if missing_stages:
            print(f"🔴 MISSING STAGES: {missing_stages}")
            print(f"🔍 Gap Analysis:")
            if 'waiting_for_input' in missing_stages:
                print("   CRITICAL: TUI doesn't wait for user input")
            if 'stays_running' in missing_stages:
                print("   CRITICAL: TUI exits instead of staying interactive")
        
        return {
            'flow_analysis': flow_analysis,
            'missing_stages': missing_stages,
            'output': result['output'],
            'runtime': result['runtime']
        }
    
    def test_user_typing_visibility_flow(self) -> dict:
        """Test if user typing would be visible (if TUI stayed running)"""
        print("\n💬 TESTING: User Typing Visibility Flow")  
        print("Expected: User types → Text appears immediately → Enter sends → Response received")
        
        # Since TUI doesn't stay interactive, we test what SHOULD happen vs what CAN'T happen
        result = self._run_tui_capture_output(timeout=4)
        
        # Check if TUI has input handling components
        has_input_components = self._analyze_input_capabilities(result['output'])
        
        print(f"Actual behavior:")
        print(f"  TUI stayed running for input: {'❌ NO' if result['runtime'] < 3 else '✅ YES'}")
        print(f"  Input handling detected: {'✅ YES' if has_input_components else '❌ NO'}")
        
        # The critical gap
        if result['runtime'] < 3:
            print(f"🔴 CRITICAL GAP: TUI exits before user can type anything")
            print(f"   User typing flow is IMPOSSIBLE because TUI doesn't wait")
        
        return {
            'can_accept_input': result['runtime'] >= 3,
            'has_input_components': has_input_components,
            'output': result['output'],
            'runtime': result['runtime']
        }
    
    def test_command_processing_flow(self) -> dict:
        """Test command processing capabilities"""
        print("\n⚡ TESTING: Command Processing Flow")
        print("Expected: User types /help → Command recognized → Help displayed → Ready for next")
        
        result = self._run_tui_capture_output(timeout=4)
        
        # Check for command processing infrastructure
        command_capabilities = self._analyze_command_capabilities(result['output'])
        
        print(f"Actual behavior:")
        for capability, detected in command_capabilities.items():
            status = "✅" if detected else "❌" 
            print(f"  {status} {capability}")
        
        # Identify command processing gaps
        if not command_capabilities.get('can_process_commands', False):
            print(f"🔴 COMMAND GAP: No evidence of command processing capability")
        
        return {
            'command_capabilities': command_capabilities,
            'output': result['output'],
            'runtime': result['runtime']
        }
    
    def test_ai_integration_flow(self) -> dict:
        """Test AI/LLM integration flow"""
        print("\n🤖 TESTING: AI Integration Flow") 
        print("Expected: User message → AI processing → Response generated → Display in TUI")
        
        result = self._run_tui_capture_output(timeout=5)
        
        # Check for AI integration components
        ai_components = self._analyze_ai_integration(result['output'])
        
        print(f"Actual behavior:")
        for component, detected in ai_components.items():
            status = "✅" if detected else "❌"
            print(f"  {status} {component}")
        
        return {
            'ai_components': ai_components,
            'output': result['output'],
            'runtime': result['runtime']
        }
    
    def _run_tui_capture_output(self, timeout: int = 5) -> dict:
        """Run TUI and capture output with timeout"""
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "agentsmcp", "tui"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                output = stdout + stderr
                exit_code = process.returncode
                
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                output = stdout + stderr
                exit_code = -1  # Killed due to timeout
                
        except Exception as e:
            output = f"Error running TUI: {e}"
            exit_code = -999
        
        runtime = time.time() - start_time
        
        return {
            'output': output,
            'runtime': runtime,
            'exit_code': exit_code
        }
    
    def _analyze_startup_flow(self, output: str) -> dict:
        """Analyze startup flow stages"""
        output_lower = output.lower()
        
        return {
            'startup': any(phrase in output_lower for phrase in [
                'starting', 'initializing', 'launching', 'tui'
            ]),
            'interface_ready': any(phrase in output_lower for phrase in [
                'ready', 'operational', 'initialized', 'interface'
            ]),  
            'waiting_for_input': any(phrase in output_lower for phrase in [
                'waiting', 'input', 'awaiting', 'type', 'command'
            ]),
            'stays_running': 'demo countdown' not in output_lower and 'demo mode' not in output_lower
        }
    
    def _analyze_input_capabilities(self, output: str) -> bool:
        """Check if TUI has input handling capabilities"""
        output_lower = output.lower()
        
        input_indicators = [
            'input', 'typing', 'keyboard', 'command', 'message',
            'enter', 'text', 'user'
        ]
        
        return any(indicator in output_lower for indicator in input_indicators)
    
    def _analyze_command_capabilities(self, output: str) -> dict:
        """Analyze command processing capabilities"""
        output_lower = output.lower()
        
        return {
            'has_command_parser': any(phrase in output_lower for phrase in [
                '/help', '/quit', '/clear', 'command'
            ]),
            'can_process_commands': any(phrase in output_lower for phrase in [
                'command', 'processing', 'execute'
            ]),
            'has_help_system': '/help' in output_lower or 'help' in output_lower
        }
    
    def _analyze_ai_integration(self, output: str) -> dict:
        """Analyze AI integration components"""
        output_lower = output.lower()
        
        return {
            'has_ai_composer': 'ai command composer' in output_lower,
            'has_llm_integration': any(phrase in output_lower for phrase in [
                'llm', 'ai', 'gpt', 'claude', 'model'
            ]),
            'can_generate_responses': any(phrase in output_lower for phrase in [
                'response', 'generate', 'assistant'
            ])
        }
    
    def run_comprehensive_flow_test(self):
        """Run all interactive flow tests"""
        print("🧪 TUI INTERACTIVE FLOW GAP ANALYSIS")
        print("=" * 60)
        print("Testing critical user interaction flows that should work but don't")
        print()
        
        # Test all flows
        startup_result = self.test_startup_to_ready_flow()
        typing_result = self.test_user_typing_visibility_flow()
        command_result = self.test_command_processing_flow()
        ai_result = self.test_ai_integration_flow()
        
        # Comprehensive analysis
        print(f"\n📊 COMPREHENSIVE FLOW ANALYSIS")
        print("=" * 40)
        
        # Critical issues
        critical_issues = []
        
        if startup_result['missing_stages']:
            critical_issues.append(f"Startup Flow: Missing {startup_result['missing_stages']}")
        
        if not typing_result['can_accept_input']:
            critical_issues.append("User Input: TUI exits before accepting input")
        
        if not command_result['command_capabilities'].get('can_process_commands', False):
            critical_issues.append("Command Processing: No command handling detected")
        
        print(f"🔴 CRITICAL ISSUES FOUND: {len(critical_issues)}")
        for i, issue in enumerate(critical_issues, 1):
            print(f"   {i}. {issue}")
        
        # Root cause analysis
        print(f"\n🔍 ROOT CAUSE ANALYSIS:")
        
        # Check runtime patterns
        avg_runtime = sum([
            startup_result['runtime'],
            typing_result['runtime'], 
            command_result['runtime'],
            ai_result['runtime']
        ]) / 4
        
        if avg_runtime < 5:
            print(f"   PRIMARY ISSUE: TUI exits too quickly (avg {avg_runtime:.1f}s)")
            print(f"   This breaks ALL interactive flows")
        
        # Check for demo mode
        demo_detected = any('demo' in result.get('output', '').lower() for result in [
            startup_result, typing_result, command_result, ai_result
        ])
        
        if demo_detected:
            print(f"   DETECTED: TUI running in demo mode instead of interactive mode")
            print(f"   CAUSE: Non-TTY environment detection")
            print(f"   SOLUTION: Force interactive mode or improve TTY handling")
        
        # Recommendations
        print(f"\n💡 RECOMMENDED FIXES:")
        print(f"   1. Fix TTY detection to enable interactive mode")
        print(f"   2. Add fallback interactive mode for non-TTY environments") 
        print(f"   3. Ensure TUI stays running until explicit user exit")
        print(f"   4. Test interactive flows in proper terminal environment")
        
        return {
            'startup': startup_result,
            'typing': typing_result,
            'command': command_result,
            'ai': ai_result,
            'critical_issues': critical_issues,
            'avg_runtime': avg_runtime,
            'demo_mode_detected': demo_detected
        }


def main():
    """Main test execution"""
    project_root = Path(__file__).parent
    tester = InteractiveFlowTester(project_root)
    
    results = tester.run_comprehensive_flow_test()
    
    # Final summary
    print(f"\n🏁 FINAL VERDICT")
    print("=" * 30)
    if len(results['critical_issues']) == 0:
        print("✅ All interactive flows working correctly")
    else:
        print(f"❌ {len(results['critical_issues'])} critical interactive flow issues found")
        print(f"🎯 Primary fix needed: Enable proper interactive mode")


if __name__ == "__main__":
    main()