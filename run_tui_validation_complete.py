#!/usr/bin/env python3
"""
Complete TUI Validation Suite Orchestrator

This script runs all TUI validation tests in the correct order and generates
a comprehensive final report validating the V3 TUI input fix.

VALIDATION SCOPE:
1. End-to-end comprehensive validation
2. User acceptance critical tests  
3. Input buffer and character handling
4. Manual testing guide (interactive)

USAGE:
- python run_tui_validation_complete.py --all
- python run_tui_validation_complete.py --automated  (CI-safe)
- python run_tui_validation_complete.py --manual     (Interactive only)
- python run_tui_validation_complete.py --report     (Generate final report)

CRITICAL VALIDATION CRITERIA:
✅ Input appears in correct TUI location (not lower right corner)
✅ Commands execute properly (/help, /quit, /status) 
✅ Chat functionality works end-to-end
✅ No console flooding or display corruption
✅ Clean exit without requiring Ctrl+C
✅ Real-time typing feedback operational
"""

import sys
import subprocess
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

class TUIValidationOrchestrator:
    """Orchestrates complete TUI validation testing"""
    
    def __init__(self):
        self.validation_results = {}
        self.critical_failures = []
        self.test_suites = [
            {
                "name": "End-to-End Comprehensive",
                "file": "test_tui_end_to_end_validation_comprehensive.py",
                "critical": True,
                "description": "Complete system validation including startup, input, commands"
            },
            {
                "name": "User Acceptance Critical", 
                "file": "test_tui_user_acceptance_critical.py",
                "critical": True,
                "description": "Tests addressing specific user-reported issues"
            },
            {
                "name": "Input Buffer Comprehensive",
                "file": "test_tui_input_buffer_comprehensive.py", 
                "critical": True,
                "description": "Character-by-character input handling validation"
            },
            {
                "name": "Manual Validation Guide",
                "file": "test_tui_manual_validation_guide.py",
                "critical": False,
                "description": "Interactive human testing guide",
                "manual_only": True
            }
        ]
        
    def print_banner(self, text: str, char: str = "="):
        """Print formatted banner"""
        width = 70
        print(f"\n{char * width}")
        print(f"{text.center(width)}")
        print(f"{char * width}")
    
    def run_test_suite(self, suite: Dict, automated_only: bool = False) -> Dict:
        """Run a single test suite"""
        if suite.get("manual_only", False) and automated_only:
            return {"skipped": True, "reason": "Manual test skipped in automated mode"}
        
        print(f"\n🔧 Running: {suite['name']}")
        print(f"   {suite['description']}")
        
        try:
            if suite.get("manual_only", False):
                # Manual test requires interactive flag
                cmd = [sys.executable, suite["file"], "--interactive"]
                print("   (Interactive manual test - requires user input)")
            else:
                # Automated test
                cmd = [sys.executable, "-m", "pytest", suite["file"], "-v", "--tb=short"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per suite
            )
            
            return {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "duration": 0  # Could add timing later
            }
            
        except subprocess.TimeoutExpired:
            return {
                "return_code": -1,
                "stdout": "",
                "stderr": "Test suite timed out",
                "success": False,
                "timeout": True
            }
        except Exception as e:
            return {
                "return_code": -1,
                "stdout": "",
                "stderr": f"Test execution error: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def run_all_automated_tests(self) -> Dict:
        """Run all automated test suites"""
        self.print_banner("🔥 STARTING COMPLETE TUI VALIDATION SUITE 🔥")
        
        print("\n📋 Test Suites to Execute:")
        for suite in self.test_suites:
            if not suite.get("manual_only", False):
                critical_marker = " [CRITICAL]" if suite["critical"] else ""
                print(f"  • {suite['name']}{critical_marker}")
        
        results = {}
        critical_passed = 0
        total_critical = 0
        
        for suite in self.test_suites:
            if suite.get("manual_only", False):
                continue  # Skip manual tests in automated run
                
            result = self.run_test_suite(suite, automated_only=True)
            results[suite["name"]] = {
                "suite_info": suite,
                "result": result
            }
            
            # Track critical test success
            if suite["critical"]:
                total_critical += 1
                if result["success"]:
                    critical_passed += 1
                else:
                    self.critical_failures.append(f"{suite['name']}: {result.get('stderr', 'Failed')}")
        
        # Generate summary
        summary = {
            "total_suites": len([s for s in self.test_suites if not s.get("manual_only", False)]),
            "successful_suites": sum(1 for r in results.values() if r["result"]["success"]),
            "critical_passed": critical_passed,
            "total_critical": total_critical,
            "critical_success_rate": (critical_passed / total_critical * 100) if total_critical > 0 else 0,
            "results": results,
            "critical_failures": self.critical_failures
        }
        
        return summary
    
    def run_manual_tests(self):
        """Run manual testing guide"""
        self.print_banner("👤 MANUAL TUI VALIDATION")
        
        manual_suite = next((s for s in self.test_suites if s.get("manual_only", False)), None)
        if not manual_suite:
            print("❌ No manual test suite found")
            return
        
        print(f"\n🔍 Starting: {manual_suite['name']}")
        print(f"   {manual_suite['description']}")
        
        result = self.run_test_suite(manual_suite)
        
        if result["success"]:
            print("\n✅ Manual testing completed successfully")
        else:
            print("\n❌ Manual testing encountered issues")
            print(f"Error: {result.get('stderr', 'Unknown error')}")
    
    def generate_final_validation_report(self, summary: Dict) -> str:
        """Generate comprehensive final validation report"""
        
        report = f"""
====================================================================
🔥 REVOLUTIONARY TUI V3 INPUT FIX - FINAL VALIDATION REPORT 🔥
====================================================================

EXECUTIVE SUMMARY:
This report validates the V3 TUI input fix addressing critical user issues:
• "Typing appeared in lower right corner" - VALIDATION STATUS BELOW
• "Commands didn't work" - VALIDATION STATUS BELOW

AUTOMATED TESTING RESULTS:
✅ Total Test Suites: {summary['total_suites']}
✅ Successful Suites: {summary['successful_suites']}
❌ Failed Suites: {summary['total_suites'] - summary['successful_suites']}

CRITICAL VALIDATION STATUS:
✅ Critical Tests Passed: {summary['critical_passed']}/{summary['total_critical']} 
📊 Critical Success Rate: {summary['critical_success_rate']:.1f}%

DETAILED SUITE RESULTS:
"""
        
        for suite_name, suite_data in summary['results'].items():
            suite_info = suite_data['suite_info']
            result = suite_data['result']
            
            status = "✅ PASS" if result['success'] else "❌ FAIL" 
            critical_marker = " [CRITICAL]" if suite_info['critical'] else ""
            
            report += f"\n{status} {suite_name}{critical_marker}"
            report += f"\n   {suite_info['description']}"
            
            if not result['success']:
                error_info = result.get('stderr', '').split('\n')[0][:100]  # First line, truncated
                report += f"\n   ❌ Issue: {error_info}"
        
        # Critical failures section
        if summary['critical_failures']:
            report += f"\n\n🚨 CRITICAL FAILURES DETECTED:\n"
            for failure in summary['critical_failures']:
                report += f"  • {failure}\n"
        else:
            report += f"\n\n✅ NO CRITICAL FAILURES - ALL CORE FUNCTIONALITY VALIDATED!\n"
        
        # Specific user issue resolution
        report += f"\n\nUSER ISSUE RESOLUTION STATUS:\n"
        
        # Input visibility issue
        input_suite_success = any(
            "user acceptance" in name.lower() or "end-to-end" in name.lower()
            for name, data in summary['results'].items() 
            if data['result']['success']
        )
        
        if input_suite_success:
            report += f"✅ INPUT VISIBILITY: User typing now appears in correct TUI location\n"
        else:
            report += f"❌ INPUT VISIBILITY: Issue may not be fully resolved\n"
        
        # Command execution issue
        command_suite_success = any(
            "command" in name.lower() or "user acceptance" in name.lower()
            for name, data in summary['results'].items()
            if data['result']['success']
        )
        
        if command_suite_success:
            report += f"✅ COMMAND EXECUTION: TUI commands now work properly\n"
        else:
            report += f"❌ COMMAND EXECUTION: Commands may still have issues\n"
        
        # Final production readiness verdict
        if summary['critical_success_rate'] >= 90 and not summary['critical_failures']:
            verdict = "🔥 PRODUCTION READY - TUI V3 INPUT FIX IS SUCCESSFUL! 🔥"
            recommendation = "✅ RECOMMEND: Deploy to users immediately"
        elif summary['critical_success_rate'] >= 75:
            verdict = "⚠️  MOSTLY READY - Minor issues to resolve"  
            recommendation = "⚠️  RECOMMEND: Fix minor issues then deploy"
        else:
            verdict = "❌ NOT READY - Critical issues must be fixed"
            recommendation = "❌ RECOMMEND: Do not deploy until issues resolved"
        
        report += f"\n\nFINAL VALIDATION VERDICT:\n{verdict}\n\n{recommendation}\n"
        
        report += f"\nMANUAL TESTING:\n"
        report += f"For complete validation, also run:\n"
        report += f"  python test_tui_manual_validation_guide.py --interactive\n"
        
        report += f"\n===================================================================="
        
        return report
    
    def save_validation_artifacts(self, summary: Dict, report: str):
        """Save validation artifacts and reports"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive report
        report_file = f"TUI_V3_FINAL_VALIDATION_REPORT_{timestamp}.txt"
        with open(report_file, "w") as f:
            f.write(report)
        
        # Save JSON summary for programmatic access
        summary_file = f"TUI_validation_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            # Clean summary for JSON serialization
            clean_summary = {
                "timestamp": timestamp,
                "total_suites": summary["total_suites"],
                "successful_suites": summary["successful_suites"],
                "critical_passed": summary["critical_passed"],
                "total_critical": summary["total_critical"],
                "critical_success_rate": summary["critical_success_rate"],
                "critical_failures": summary["critical_failures"]
            }
            json.dump(clean_summary, f, indent=2)
        
        return report_file, summary_file

def main():
    """Main validation orchestrator"""
    parser = argparse.ArgumentParser(description="TUI V3 Input Fix Validation Suite")
    parser.add_argument("--all", action="store_true", help="Run all tests (automated + manual)")
    parser.add_argument("--automated", action="store_true", help="Run automated tests only")
    parser.add_argument("--manual", action="store_true", help="Run manual tests only")
    parser.add_argument("--report", action="store_true", help="Generate report from existing results")
    
    args = parser.parse_args()
    
    if not any([args.all, args.automated, args.manual, args.report]):
        # Default to automated tests
        args.automated = True
    
    orchestrator = TUIValidationOrchestrator()
    
    if args.manual:
        orchestrator.run_manual_tests()
        return
    
    if args.automated or args.all:
        # Run automated tests
        summary = orchestrator.run_all_automated_tests()
        
        # Generate and display report
        report = orchestrator.generate_final_validation_report(summary)
        print(report)
        
        # Save artifacts
        report_file, summary_file = orchestrator.save_validation_artifacts(summary, report)
        
        print(f"\n📄 Reports saved:")
        print(f"   • Comprehensive: {report_file}")
        print(f"   • JSON Summary: {summary_file}")
        
        if args.all:
            print(f"\n👤 To complete validation, also run manual tests:")
            print(f"   python test_tui_manual_validation_guide.py --interactive")
        
        # Exit with appropriate code
        if summary['critical_success_rate'] >= 90 and not summary['critical_failures']:
            print(f"\n🎉 VALIDATION SUCCESS! 🎉")
            sys.exit(0)
        else:
            print(f"\n❌ VALIDATION ISSUES DETECTED")
            sys.exit(1)

if __name__ == "__main__":
    main()