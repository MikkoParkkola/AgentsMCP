#!/usr/bin/env python3
"""
V3 TUI Debugging Suite - Master Controller
Orchestrates all V3 TUI debugging scripts for comprehensive diagnosis

This master script coordinates all the individual debugging tools:
1. tui_input_diagnostic.py - Updated with V3 tests
2. v3_tui_input_pipeline_debugger.py - Step-by-step V3 pipeline analysis
3. v3_realtime_input_monitor.py - Real-time input event monitoring  
4. v3_plain_cli_renderer_tests.py - PlainCLIRenderer comprehensive testing
5. v3_chat_engine_verifier.py - ChatEngine and LLM communication verification
6. v3_command_workflow_debugger.py - Command processing workflow tracing

Usage: python v3_tui_debugging_suite.py [--quick|--full|--interactive]
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

def suite_header():
    print("=" * 80)
    print("🛠️ V3 TUI Debugging Suite - Master Controller")
    print("=" * 80)
    print("Comprehensive diagnosis of V3 TUI input issues")
    print("Coordinates all debugging tools for systematic problem resolution\n")

class DebuggingSuiteController:
    """Controls and orchestrates all V3 TUI debugging tools."""
    
    def __init__(self):
        self.script_directory = Path.cwd()
        self.debugging_scripts = {
            'diagnostic': 'tui_input_diagnostic.py',
            'pipeline': 'v3_tui_input_pipeline_debugger.py', 
            'monitor': 'v3_realtime_input_monitor.py',
            'renderer': 'v3_plain_cli_renderer_tests.py',
            'chat_engine': 'v3_chat_engine_verifier.py',
            'workflow': 'v3_command_workflow_debugger.py'
        }
        self.results = {}
        self.recommendations = []
    
    def check_script_availability(self):
        """Check which debugging scripts are available."""
        print("📋 DEBUGGING SCRIPT AVAILABILITY CHECK")
        print("-" * 50)
        
        available_scripts = {}
        
        for script_key, script_name in self.debugging_scripts.items():
            script_path = self.script_directory / script_name
            is_available = script_path.exists() and script_path.is_file()
            
            status = "✅ Available" if is_available else "❌ Missing"
            print(f"  {status} {script_name}")
            
            available_scripts[script_key] = is_available
        
        total_available = sum(available_scripts.values())
        total_scripts = len(available_scripts)
        
        print(f"\n📊 Summary: {total_available}/{total_scripts} debugging scripts available")
        
        if total_available < total_scripts:
            print("⚠️  Some debugging scripts are missing. Suite will skip missing scripts.")
        
        return available_scripts
    
    def run_script(self, script_name: str, timeout: int = 300) -> Dict[str, Any]:
        """Run a debugging script and capture results."""
        print(f"\n🔄 Running {script_name}...")
        print("-" * 40)
        
        script_path = self.script_directory / script_name
        
        if not script_path.exists():
            return {
                "success": False,
                "error": f"Script {script_name} not found",
                "output": "",
                "duration": 0
            }
        
        start_time = time.time()
        
        try:
            # Run the script with timeout
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.script_directory
            )
            
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "output": result.stdout,
                "error": result.stderr,
                "duration": duration
            }
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "error": f"Script timed out after {timeout} seconds",
                "output": "",
                "duration": duration
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "success": False,
                "error": f"Failed to run script: {e}",
                "output": "",
                "duration": duration
            }
    
    def run_quick_diagnosis(self):
        """Run quick diagnosis - essential scripts only."""
        print("\n🚀 QUICK DIAGNOSIS MODE")
        print("=" * 50)
        print("Running essential diagnostic scripts for rapid issue identification\n")
        
        # Quick diagnosis scripts in order of importance
        quick_scripts = [
            ('diagnostic', 'Updated TUI Input Diagnostic'),
            ('pipeline', 'V3 Input Pipeline Debugger'),
        ]
        
        for script_key, description in quick_scripts:
            script_name = self.debugging_scripts[script_key]
            
            if script_key in self.results:
                continue  # Skip if already run
            
            print(f"📋 {description}")
            result = self.run_script(script_name, timeout=120)  # 2 minute timeout
            self.results[script_key] = result
            
            if result['success']:
                print(f"✅ {description} completed successfully")
                # Show last few lines of output for context
                if result['output']:
                    lines = result['output'].strip().split('\n')
                    if len(lines) > 5:
                        print("📄 Key results:")
                        for line in lines[-5:]:
                            if line.strip():
                                print(f"    {line}")
            else:
                print(f"❌ {description} failed: {result['error']}")
        
        return self.results
    
    def run_full_diagnosis(self):
        """Run full comprehensive diagnosis - all available scripts."""
        print("\n🔬 FULL COMPREHENSIVE DIAGNOSIS")
        print("=" * 50)
        print("Running all available diagnostic scripts for complete analysis\n")
        
        available_scripts = self.check_script_availability()
        
        # Full diagnosis scripts in logical order
        full_script_order = [
            ('diagnostic', 'Updated TUI Input Diagnostic', 120),
            ('pipeline', 'V3 Input Pipeline Debugger', 180),
            ('renderer', 'PlainCLIRenderer Test Suite', 300),
            ('chat_engine', 'ChatEngine Verifier', 180),
            ('workflow', 'Command Workflow Debugger', 240),
            ('monitor', 'Real-time Input Monitor', 600)  # Longest timeout for interactive
        ]
        
        for script_key, description, timeout in full_script_order:
            if not available_scripts.get(script_key, False):
                print(f"⏭️ Skipping {description} (script not available)")
                continue
            
            if script_key in self.results:
                continue  # Skip if already run
            
            print(f"\n📋 Running {description}...")
            result = self.run_script(self.debugging_scripts[script_key], timeout=timeout)
            self.results[script_key] = result
            
            if result['success']:
                print(f"✅ {description} completed in {result['duration']:.1f}s")
            else:
                print(f"❌ {description} failed: {result['error']}")
                
                # For critical scripts, provide immediate recommendations
                if script_key in ['diagnostic', 'pipeline']:
                    print(f"⚠️  Critical script failed - this may indicate fundamental issues")
        
        return self.results
    
    def run_interactive_diagnosis(self):
        """Run interactive diagnosis - user selects scripts."""
        print("\n🎯 INTERACTIVE DIAGNOSIS MODE")
        print("=" * 50)
        
        available_scripts = self.check_script_availability()
        
        print("\nAvailable debugging scripts:")
        script_options = []
        
        for i, (script_key, script_name) in enumerate(self.debugging_scripts.items(), 1):
            if available_scripts.get(script_key, False):
                description = {
                    'diagnostic': 'Core TUI diagnostic (recommended first)',
                    'pipeline': 'V3 pipeline analysis (recommended)',
                    'renderer': 'PlainCLIRenderer testing',
                    'chat_engine': 'ChatEngine/LLM verification',
                    'workflow': 'Command workflow tracing',
                    'monitor': 'Real-time input monitoring (interactive)'
                }.get(script_key, 'Debug script')
                
                print(f"  {i}. {script_name} - {description}")
                script_options.append((script_key, script_name, description))
            else:
                print(f"  {i}. {script_name} - ❌ Not available")
        
        print(f"\n  {len(script_options)+1}. Run all available scripts")
        print(f"  {len(script_options)+2}. Quick diagnosis (essential scripts only)")
        print(f"  0. Exit")
        
        while True:
            try:
                choice = input(f"\nSelect option (1-{len(script_options)+2}, 0 to exit): ").strip()
                
                if choice == '0':
                    print("Exiting interactive diagnosis.")
                    return self.results
                
                elif choice == str(len(script_options)+1):
                    # Run all scripts
                    return self.run_full_diagnosis()
                
                elif choice == str(len(script_options)+2):
                    # Quick diagnosis
                    return self.run_quick_diagnosis()
                
                elif choice.isdigit() and 1 <= int(choice) <= len(script_options):
                    # Run selected script
                    script_key, script_name, description = script_options[int(choice)-1]
                    
                    print(f"\n📋 Running {description}...")
                    result = self.run_script(script_name, timeout=600)
                    self.results[script_key] = result
                    
                    if result['success']:
                        print(f"✅ {description} completed successfully")
                        print("📄 Output summary:")
                        if result['output']:
                            # Show key parts of output
                            lines = result['output'].strip().split('\n')
                            for line in lines[-10:]:  # Last 10 lines
                                if any(keyword in line.lower() for keyword in 
                                      ['✅', '❌', '⚠️', 'success', 'error', 'failed', 'issue']):
                                    print(f"    {line}")
                    else:
                        print(f"❌ {description} failed: {result['error']}")
                    
                    # Ask if user wants to run another script
                    continue_choice = input("\nRun another script? (y/n): ").lower().strip()
                    if continue_choice not in ['y', 'yes']:
                        break
                
                else:
                    print("Invalid choice. Please try again.")
            
            except KeyboardInterrupt:
                print("\nInteractive diagnosis interrupted.")
                break
            except EOFError:
                print("Input interrupted.")
                break
        
        return self.results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive report from all debugging results."""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE V3 TUI DEBUGGING REPORT")
        print("=" * 80)
        
        if not self.results:
            print("No debugging results available. Run diagnosis first.")
            return
        
        # Summary statistics
        successful_scripts = sum(1 for result in self.results.values() if result.get('success', False))
        total_scripts = len(self.results)
        total_duration = sum(result.get('duration', 0) for result in self.results.values())
        
        print(f"📈 Debugging Session Summary:")
        print(f"  Scripts executed: {total_scripts}")
        print(f"  Successful: {successful_scripts}")
        print(f"  Failed: {total_scripts - successful_scripts}")
        print(f"  Total time: {total_duration:.1f} seconds")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for script_key, result in self.results.items():
            script_name = self.debugging_scripts.get(script_key, script_key)
            status = "✅ SUCCESS" if result.get('success', False) else "❌ FAILED"
            duration = result.get('duration', 0)
            
            print(f"  {status} {script_name} ({duration:.1f}s)")
            
            if not result.get('success', False) and result.get('error'):
                print(f"    Error: {result['error']}")
        
        # Aggregate analysis
        print(f"\n🔍 AGGREGATE ANALYSIS:")
        
        # Check for common patterns in outputs
        all_output = ""
        for result in self.results.values():
            if result.get('output'):
                all_output += result['output'] + "\n"
        
        # Look for key issues mentioned across scripts
        common_issues = []
        
        if 'v2' in all_output.lower() and 'revolutionary' in all_output.lower():
            common_issues.append("V2 Revolutionary TUI detected - likely routing issue")
        
        if 'tty' in all_output.lower() and 'demo mode' in all_output.lower():
            common_issues.append("TTY detection failure causing demo mode")
        
        if 'plainclirenderer' in all_output.lower() and ('failed' in all_output.lower() or 'error' in all_output.lower()):
            common_issues.append("PlainCLIRenderer component issues")
        
        if 'chatengine' in all_output.lower() and ('failed' in all_output.lower() or 'error' in all_output.lower()):
            common_issues.append("ChatEngine/LLM communication problems")
        
        if common_issues:
            print("  🎯 Common issues detected:")
            for issue in common_issues:
                print(f"    • {issue}")
        else:
            print("  ✅ No common patterns of failure detected")
        
        # Priority recommendations
        print(f"\n🛠️ PRIORITY RECOMMENDATIONS:")
        
        if successful_scripts == 0:
            print("  🔥 CRITICAL: All debugging scripts failed")
            print("    • Check Python environment and dependencies")
            print("    • Verify AgentsMCP installation")
            print("    • Run basic import tests")
        
        elif 'diagnostic' in self.results and not self.results['diagnostic'].get('success', False):
            print("  🔥 CRITICAL: Core diagnostic script failed")
            print("    • This indicates fundamental environment issues")
            print("    • Check V3 module imports and basic functionality")
        
        else:
            # Success-based recommendations
            if successful_scripts == total_scripts:
                print("  ✅ All scripts completed successfully")
                print("    • Review individual script outputs for specific issues")
                print("    • Focus on any warnings or recommendations in outputs")
            else:
                print("  ⚠️ Mixed results - some scripts succeeded, others failed")
                print("    • Focus on failed scripts for critical issues")
                print("    • Use successful script results to narrow down problems")
        
        print(f"\n📋 NEXT STEPS:")
        print("  1. Review individual script outputs above")
        print("  2. Address highest priority issues first") 
        print("  3. Re-run specific scripts after fixes")
        print("  4. Test actual TUI functionality: ./agentsmcp tui")

def main():
    suite_header()
    
    # Parse command line arguments
    mode = 'interactive'  # default
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--quick', '-q']:
            mode = 'quick'
        elif arg in ['--full', '-f']:
            mode = 'full'
        elif arg in ['--interactive', '-i']:
            mode = 'interactive'
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python v3_tui_debugging_suite.py [--quick|--full|--interactive]")
            sys.exit(1)
    
    # Create and run debugging suite
    controller = DebuggingSuiteController()
    
    try:
        if mode == 'quick':
            controller.run_quick_diagnosis()
        elif mode == 'full':
            controller.run_full_diagnosis()
        else:  # interactive
            controller.run_interactive_diagnosis()
        
        # Generate comprehensive report
        controller.generate_comprehensive_report()
    
    except KeyboardInterrupt:
        print("\n⚠️ Debugging suite interrupted by user")
    except Exception as e:
        print(f"\n❌ Debugging suite error: {e}")
    
    print(f"\n" + "=" * 80)
    print("🛠️ V3 TUI Debugging Suite Complete!")
    print("Use the analysis above to systematically fix V3 TUI input issues.")
    print("=" * 80)

if __name__ == "__main__":
    main()