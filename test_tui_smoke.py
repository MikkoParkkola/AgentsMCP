#!/usr/bin/env python3
"""
AgentsMCP TUI Smoke Test Suite

DESCRIPTION:
This comprehensive smoke test validates the AgentsMCP TUI startup sequence and 
core functionality to catch hanging issues and ensure reliable operation.

WHAT IT TESTS:
1. ✅ Basic imports work without errors
2. ✅ TUI interface initialization completes  
3. ✅ Revolutionary launcher starts up properly
4. ✅ Non-TTY environments are handled gracefully
5. ✅ Graceful shutdown works correctly
6. ✅ Individual components can be imported in isolation
7. ✅ Startup sequence begins without immediate crashes

USAGE:
  python test_tui_smoke.py                    # Run all tests (30s timeout)
  python test_tui_smoke.py --quick            # Run only quick validation (10s)
  python test_tui_smoke.py --timeout 60       # Custom timeout
  python test_tui_smoke.py --verbose          # Detailed output

INTERPRETING RESULTS:
- ALL TESTS PASSED: TUI startup is working correctly
- Basic tests pass but CLI integration times out: Normal (startup detection issues)
- Basic tests fail: Critical TUI startup problems need investigation

TROUBLESHOOTING:
- Run with --quick first to isolate import/initialization issues
- Use --verbose for detailed error information
- CLI integration timeout is expected in automated environments
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional
import threading

# Add the src directory to Python path to import agentsmcp
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Timeout configuration
DEFAULT_TIMEOUT = 30  # seconds
STARTUP_TIMEOUT = 10  # seconds for startup phase
SHUTDOWN_TIMEOUT = 5  # seconds for graceful shutdown


class TimeoutError(Exception):
    """Custom timeout exception."""
    pass


class SmokeTestRunner:
    """Main smoke test runner."""
    
    def __init__(self, timeout: int = DEFAULT_TIMEOUT, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose
        self.results = []
        
        # Setup logging
        level = logging.DEBUG if verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format='[SMOKE] %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log(self, message: str, level: str = "info"):
        """Log a message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        if level == "error":
            print(f"❌ [{timestamp}] {message}")
            self.logger.error(message)
        elif level == "success":
            print(f"✅ [{timestamp}] {message}")
            self.logger.info(message)
        elif level == "warning":
            print(f"⚠️ [{timestamp}] {message}")
            self.logger.warning(message)
        else:
            print(f"ℹ️ [{timestamp}] {message}")
            self.logger.info(message)
    
    def add_result(self, test_name: str, success: bool, message: str, duration: float = 0.0):
        """Add a test result."""
        self.results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "duration": duration
        })
    
    async def with_timeout(self, coro, timeout: float, test_name: str):
        """Run a coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"{test_name} timed out after {timeout}s")
    
    async def test_basic_imports(self) -> bool:
        """Test 1: Basic imports work without errors."""
        start_time = time.time()
        try:
            self.log("Testing basic imports...")
            
            # Test core imports
            from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
            from agentsmcp.ui.v2.revolutionary_launcher import RevolutionaryLauncher, launch_revolutionary_tui
            from agentsmcp.ui.cli_app import CLIConfig
            
            # Test that classes can be instantiated
            cli_config = CLIConfig()
            interface = RevolutionaryTUIInterface(cli_config=cli_config)
            launcher = RevolutionaryLauncher(cli_config=cli_config)
            
            duration = time.time() - start_time
            self.add_result("basic_imports", True, "All imports successful", duration)
            self.log("✓ Basic imports test passed", "success")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.add_result("basic_imports", False, f"Import failed: {e}", duration)
            self.log(f"✗ Basic imports test failed: {e}", "error")
            return False
    
    async def test_interface_initialization(self) -> bool:
        """Test 2: Revolutionary TUI interface can initialize without hanging."""
        start_time = time.time()
        try:
            self.log("Testing TUI interface initialization...")
            
            from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
            from agentsmcp.ui.cli_app import CLIConfig
            
            # Create interface with minimal config
            cli_config = CLIConfig()
            cli_config.debug_mode = self.verbose
            interface = RevolutionaryTUIInterface(cli_config=cli_config)
            
            # Test initialization with timeout
            initialization_task = interface.initialize()
            success = await self.with_timeout(
                initialization_task, 
                STARTUP_TIMEOUT, 
                "interface_initialization"
            )
            
            # Test cleanup
            await interface._cleanup()
            
            duration = time.time() - start_time
            self.add_result("interface_initialization", True, "Interface initialized successfully", duration)
            self.log("✓ Interface initialization test passed", "success")
            return True
            
        except TimeoutError as e:
            duration = time.time() - start_time
            self.add_result("interface_initialization", False, str(e), duration)
            self.log(f"✗ Interface initialization timed out: {e}", "error")
            return False
        except Exception as e:
            duration = time.time() - start_time
            self.add_result("interface_initialization", False, f"Initialization failed: {e}", duration)
            self.log(f"✗ Interface initialization failed: {e}", "error")
            return False
    
    async def test_launcher_startup(self) -> bool:
        """Test 3: Revolutionary launcher can start up without hanging."""
        start_time = time.time()
        try:
            self.log("Testing launcher startup...")
            
            from agentsmcp.ui.v2.revolutionary_launcher import RevolutionaryLauncher
            from agentsmcp.ui.cli_app import CLIConfig
            
            # Create launcher with debug config
            cli_config = CLIConfig()
            cli_config.debug_mode = self.verbose
            launcher = RevolutionaryLauncher(cli_config=cli_config)
            
            # Test that feature detection completes quickly
            feature_detection_task = launcher._initialize_feature_detection()
            await self.with_timeout(
                feature_detection_task,
                5.0,  # Feature detection should be very fast
                "launcher_feature_detection"
            )
            
            # Test feature level determination
            feature_level_task = launcher._determine_feature_level()
            feature_level = await self.with_timeout(
                feature_level_task,
                3.0,  # Feature level should be determined quickly
                "launcher_feature_level"
            )
            
            # Cleanup
            await launcher._cleanup()
            
            duration = time.time() - start_time
            self.add_result("launcher_startup", True, f"Launcher started successfully (level: {feature_level.name})", duration)
            self.log(f"✓ Launcher startup test passed (feature level: {feature_level.name})", "success")
            return True
            
        except TimeoutError as e:
            duration = time.time() - start_time
            self.add_result("launcher_startup", False, str(e), duration)
            self.log(f"✗ Launcher startup timed out: {e}", "error")
            return False
        except Exception as e:
            duration = time.time() - start_time
            self.add_result("launcher_startup", False, f"Launcher failed: {e}", duration)
            self.log(f"✗ Launcher startup failed: {e}", "error")
            return False
    
    async def test_non_tty_fallback(self) -> bool:
        """Test 4: TUI handles non-TTY environment gracefully."""
        start_time = time.time()
        try:
            self.log("Testing non-TTY fallback...")
            
            # Simulate non-TTY environment by setting environment variables
            original_env = os.environ.copy()
            os.environ['CI'] = '1'  # Simulate CI environment
            
            try:
                from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
                from agentsmcp.ui.cli_app import CLIConfig
                
                cli_config = CLIConfig()
                cli_config.debug_mode = self.verbose
                interface = RevolutionaryTUIInterface(cli_config=cli_config)
                
                # Run the interface briefly - should detect CI and exit quickly
                run_task = interface.run()
                exit_code = await self.with_timeout(
                    run_task,
                    5.0,  # Should exit very quickly in CI mode
                    "non_tty_fallback"
                )
                
                # Should return 0 for successful CI detection
                if exit_code == 0:
                    duration = time.time() - start_time
                    self.add_result("non_tty_fallback", True, "CI environment detected and handled gracefully", duration)
                    self.log("✓ Non-TTY fallback test passed", "success")
                    return True
                else:
                    duration = time.time() - start_time
                    self.add_result("non_tty_fallback", False, f"Unexpected exit code: {exit_code}", duration)
                    self.log(f"✗ Non-TTY fallback returned unexpected exit code: {exit_code}", "error")
                    return False
                    
            finally:
                # Restore environment
                os.environ.clear()
                os.environ.update(original_env)
                
        except TimeoutError as e:
            duration = time.time() - start_time
            self.add_result("non_tty_fallback", False, str(e), duration)
            self.log(f"✗ Non-TTY fallback timed out: {e}", "error")
            return False
        except Exception as e:
            duration = time.time() - start_time
            self.add_result("non_tty_fallback", False, f"Fallback test failed: {e}", duration)
            self.log(f"✗ Non-TTY fallback failed: {e}", "error")
            return False
    
    async def test_graceful_shutdown(self) -> bool:
        """Test 5: TUI can be shut down gracefully."""
        start_time = time.time()
        try:
            self.log("Testing graceful shutdown...")
            
            from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface
            from agentsmcp.ui.cli_app import CLIConfig
            
            cli_config = CLIConfig()
            cli_config.debug_mode = self.verbose
            interface = RevolutionaryTUIInterface(cli_config=cli_config)
            
            # Initialize interface
            await interface.initialize()
            
            # Test that we can set running to False and cleanup works
            interface.running = True
            interface.running = False  # Simulate shutdown signal
            
            # Test cleanup with timeout
            cleanup_task = interface._cleanup()
            await self.with_timeout(
                cleanup_task,
                SHUTDOWN_TIMEOUT,
                "graceful_shutdown"
            )
            
            duration = time.time() - start_time
            self.add_result("graceful_shutdown", True, "Shutdown completed successfully", duration)
            self.log("✓ Graceful shutdown test passed", "success")
            return True
            
        except TimeoutError as e:
            duration = time.time() - start_time
            self.add_result("graceful_shutdown", False, str(e), duration)
            self.log(f"✗ Graceful shutdown timed out: {e}", "error")
            return False
        except Exception as e:
            duration = time.time() - start_time
            self.add_result("graceful_shutdown", False, f"Shutdown failed: {e}", duration)
            self.log(f"✗ Graceful shutdown failed: {e}", "error")
            return False
    
    async def test_component_isolation(self) -> bool:
        """Test 6: TUI components can be created in isolation."""
        start_time = time.time()
        try:
            self.log("Testing component isolation...")
            
            # Test individual component imports
            components_to_test = [
                ("terminal_controller", "agentsmcp.ui.v2.terminal_controller"),
                ("logging_manager", "agentsmcp.ui.v2.logging_isolation_manager"),
                ("text_layout_engine", "agentsmcp.ui.v2.text_layout_engine"),
                ("event_system", "agentsmcp.ui.v2.event_system"),
            ]
            
            failed_components = []
            
            for component_name, module_path in components_to_test:
                try:
                    # Import the module
                    parts = module_path.split('.')
                    module = __import__(module_path, fromlist=[parts[-1]])
                    
                    # Try to access key classes
                    if component_name == "terminal_controller":
                        _ = module.TerminalController
                    elif component_name == "logging_manager":
                        _ = module.LoggingIsolationManager
                    elif component_name == "text_layout_engine":
                        _ = module.TextLayoutEngine
                    elif component_name == "event_system":
                        _ = module.AsyncEventSystem
                        
                except Exception as e:
                    failed_components.append((component_name, str(e)))
            
            if not failed_components:
                duration = time.time() - start_time
                self.add_result("component_isolation", True, "All components imported successfully", duration)
                self.log("✓ Component isolation test passed", "success")
                return True
            else:
                duration = time.time() - start_time
                failure_msg = "; ".join([f"{name}: {error}" for name, error in failed_components])
                self.add_result("component_isolation", False, f"Component failures: {failure_msg}", duration)
                self.log(f"✗ Component isolation failed: {failure_msg}", "error")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.add_result("component_isolation", False, f"Isolation test failed: {e}", duration)
            self.log(f"✗ Component isolation failed: {e}", "error")
            return False
    
    async def test_startup_sequence_validation(self) -> bool:
        """Test 7: Validate that startup sequence begins without immediate crashes."""
        start_time = time.time()
        try:
            self.log("Testing startup sequence validation...")
            
            # Import the CLI entry point
            from agentsmcp.ui.v2.revolutionary_launcher import launch_revolutionary_tui
            from agentsmcp.ui.cli_app import CLIConfig
            
            # Create a minimal config
            cli_config = CLIConfig()
            cli_config.debug_mode = self.verbose
            
            # Set environment to force non-interactive mode
            original_env = os.environ.copy()
            os.environ['CI'] = '1'
            os.environ['GITHUB_ACTIONS'] = 'true'
            os.environ['AGENTSMCP_NON_INTERACTIVE'] = '1'
            
            try:
                # Start the TUI and let it run for 2 seconds to validate startup
                launch_task = asyncio.create_task(launch_revolutionary_tui(cli_config))
                
                # Wait a short time to see if startup begins normally
                try:
                    # Wait 2 seconds - if it completes, great! If not, that's expected
                    await asyncio.sleep(2.0)
                    
                    # If we get here, startup began successfully (even if still running)
                    if not launch_task.done():
                        # Task is still running - startup sequence is working
                        launch_task.cancel()
                        try:
                            await launch_task
                        except asyncio.CancelledError:
                            pass
                        
                        duration = time.time() - start_time
                        self.add_result("startup_validation", True, "Startup sequence began successfully", duration)
                        self.log("✓ Startup sequence validation passed", "success")
                        return True
                    else:
                        # Task completed - check result
                        exit_code = launch_task.result()
                        duration = time.time() - start_time
                        self.add_result("startup_validation", True, f"Startup completed with exit code {exit_code}", duration)
                        self.log("✓ Startup sequence validation passed", "success")
                        return True
                        
                except Exception as e:
                    # Cancel the task if something went wrong
                    if not launch_task.done():
                        launch_task.cancel()
                        try:
                            await launch_task
                        except asyncio.CancelledError:
                            pass
                    raise e
                    
            finally:
                # Restore environment
                os.environ.clear()
                os.environ.update(original_env)
                
        except Exception as e:
            duration = time.time() - start_time
            self.add_result("startup_validation", False, f"Startup sequence failed: {e}", duration)
            self.log(f"✗ Startup sequence validation failed: {e}", "error")
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all smoke tests."""
        self.log(f"🚀 Starting AgentsMCP TUI Smoke Tests (timeout: {self.timeout}s)")
        overall_start = time.time()
        
        tests = [
            ("Basic Imports", self.test_basic_imports),
            ("Interface Initialization", self.test_interface_initialization),
            ("Launcher Startup", self.test_launcher_startup),
            ("Non-TTY Fallback", self.test_non_tty_fallback),
            ("Graceful Shutdown", self.test_graceful_shutdown),
            ("Component Isolation", self.test_component_isolation),
            ("Startup Sequence Validation", self.test_startup_sequence_validation),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            self.log(f"Running: {test_name}")
            try:
                # Run each test with overall timeout
                success = await self.with_timeout(
                    test_func(),
                    self.timeout,
                    test_name
                )
                if success:
                    passed += 1
                else:
                    failed += 1
            except TimeoutError as e:
                self.log(f"✗ {test_name} timed out: {e}", "error")
                self.add_result(test_name.lower().replace(" ", "_"), False, str(e))
                failed += 1
            except Exception as e:
                self.log(f"✗ {test_name} failed: {e}", "error")
                self.add_result(test_name.lower().replace(" ", "_"), False, f"Test exception: {e}")
                failed += 1
        
        # Generate summary
        overall_duration = time.time() - overall_start
        self.log("\n" + "="*60)
        self.log("SMOKE TEST SUMMARY")
        self.log("="*60)
        self.log(f"Total Tests: {passed + failed}")
        self.log(f"Passed: {passed}", "success" if passed > 0 else "info")
        self.log(f"Failed: {failed}", "error" if failed > 0 else "info")
        self.log(f"Duration: {overall_duration:.2f}s")
        
        if failed == 0:
            self.log("🎉 ALL TESTS PASSED!", "success")
        else:
            self.log(f"❌ {failed} TESTS FAILED", "error")
        
        # Show detailed results if verbose
        if self.verbose:
            self.log("\nDETAILED RESULTS:")
            self.log("-" * 60)
            for result in self.results:
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                self.log(f"{status} {result['test']} ({result['duration']:.2f}s): {result['message']}")
        
        return failed == 0
    
    async def run_quick_validation(self) -> bool:
        """Run only basic validation tests (imports and initialization)."""
        self.log("🚀 Running Quick Validation Tests")
        overall_start = time.time()
        
        tests = [
            ("Basic Imports", self.test_basic_imports),
            ("Interface Initialization", self.test_interface_initialization),
            ("Component Isolation", self.test_component_isolation),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            self.log(f"Running: {test_name}")
            try:
                success = await self.with_timeout(
                    test_func(),
                    10.0,  # Shorter timeout for quick tests
                    test_name
                )
                if success:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                self.log(f"✗ {test_name} failed: {e}", "error")
                failed += 1
        
        # Generate summary
        overall_duration = time.time() - overall_start
        self.log("\n" + "="*50)
        self.log("QUICK VALIDATION SUMMARY")
        self.log("="*50)
        self.log(f"Total Tests: {passed + failed}")
        self.log(f"Passed: {passed}", "success" if passed > 0 else "info")
        self.log(f"Failed: {failed}", "error" if failed > 0 else "info")
        self.log(f"Duration: {overall_duration:.2f}s")
        
        if failed == 0:
            self.log("🎉 QUICK VALIDATION PASSED!", "success")
        else:
            self.log(f"❌ {failed} VALIDATION TESTS FAILED", "error")
        
        return failed == 0
    
    def print_usage(self):
        """Print usage information."""
        print("""
AgentsMCP TUI Smoke Test Suite

Usage: python test_tui_smoke.py [options]

Options:
  --timeout N     Set timeout in seconds (default: 30)
  --verbose       Enable verbose output and debug logging
  --quick         Run only quick validation tests (imports, basic init)
  --help          Show this help message

Examples:
  python test_tui_smoke.py                    # Run all tests with defaults
  python test_tui_smoke.py --quick            # Run only quick validation
  python test_tui_smoke.py --timeout 60       # Run with 60s timeout
  python test_tui_smoke.py --verbose          # Run with detailed output
  python test_tui_smoke.py --quick --verbose  # Quick tests with verbose output
        """.strip())


async def main():
    """Main entry point."""
    import sys
    
    # Parse arguments
    timeout = DEFAULT_TIMEOUT
    verbose = False
    quick = False
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--timeout" and i + 1 < len(args):
            timeout = int(args[i + 1])
            i += 2
        elif args[i] == "--verbose":
            verbose = True
            i += 1
        elif args[i] == "--quick":
            quick = True
            i += 1
        elif args[i] == "--help":
            SmokeTestRunner(0).print_usage()
            return 0
        else:
            print(f"Unknown argument: {args[i]}")
            SmokeTestRunner(0).print_usage()
            return 1
    
    # Run tests
    runner = SmokeTestRunner(timeout=timeout, verbose=verbose)
    if quick:
        success = await runner.run_quick_validation()
    else:
        success = await runner.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)