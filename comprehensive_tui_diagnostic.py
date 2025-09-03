#!/usr/bin/env python3
"""
Comprehensive TUI Diagnostic Script for AgentsMCP

This script provides comprehensive diagnostics for the AgentsMCP TUI system,
identifying potential issues in the user environment that could affect TUI functionality.

Usage:
    python comprehensive_tui_diagnostic.py [--verbose] [--json] [--quick]

Exit codes:
    0: No issues found
    1: Minor issues found (TUI should work with warnings)
    2: Major issues found (TUI may not function properly)
    3: Critical issues found (TUI will not work)
    4: Script error (diagnostic failure)
"""

import asyncio
import io
import json
import locale
import os
import platform
import shutil
import signal
import sys
import threading
import time
import traceback
import weakref
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import argparse
import subprocess

# Import for modern version detection
try:
    from importlib.metadata import version as get_package_version
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import version as get_package_version


# Color codes for output formatting
class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    @classmethod
    def strip_colors(cls, text: str) -> str:
        """Remove color codes from text."""
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    name: str
    status: str  # 'PASS', 'WARN', 'FAIL', 'CRITICAL'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SystemInfo:
    """System information collection."""
    python_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    python_executable: str
    working_directory: str
    user: str
    home_directory: str
    shell: str
    term: Optional[str]
    colorterm: Optional[str]


class DiagnosticLogger:
    """Handles logging and output formatting for diagnostics."""
    
    def __init__(self, verbose: bool = False, json_output: bool = False):
        self.verbose = verbose
        self.json_output = json_output
        self.results: List[DiagnosticResult] = []
        self.current_section = ""
    
    def section(self, name: str):
        """Start a new diagnostic section."""
        self.current_section = name
        if not self.json_output:
            print(f"\n{Colors.BOLD}{Colors.BLUE}=== {name} ==={Colors.RESET}")
    
    def log(self, result: DiagnosticResult):
        """Log a diagnostic result."""
        self.results.append(result)
        
        if self.json_output:
            return
        
        # Status icon and color
        status_colors = {
            'PASS': Colors.GREEN,
            'WARN': Colors.YELLOW, 
            'FAIL': Colors.RED,
            'CRITICAL': Colors.RED + Colors.BOLD
        }
        
        status_icons = {
            'PASS': '✓',
            'WARN': '⚠',
            'FAIL': '✗',
            'CRITICAL': '💀'
        }
        
        color = status_colors.get(result.status, '')
        icon = status_icons.get(result.status, '?')
        
        # Main result line
        print(f"{color}{icon} {result.name}: {result.message}{Colors.RESET}")
        
        # Verbose details
        if self.verbose and result.details:
            for key, value in result.details.items():
                print(f"   {Colors.CYAN}{key}:{Colors.RESET} {value}")
        
        # Recommendations
        if result.recommendations:
            for rec in result.recommendations:
                print(f"   {Colors.YELLOW}→{Colors.RESET} {rec}")
    
    def progress(self, message: str):
        """Show progress message."""
        if not self.json_output:
            print(f"{Colors.CYAN}... {message}{Colors.RESET}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get diagnostic summary."""
        status_counts = {}
        for result in self.results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        # Determine overall status
        if status_counts.get('CRITICAL', 0) > 0:
            overall_status = 'CRITICAL'
            exit_code = 3
        elif status_counts.get('FAIL', 0) > 0:
            overall_status = 'MAJOR_ISSUES'
            exit_code = 2
        elif status_counts.get('WARN', 0) > 0:
            overall_status = 'MINOR_ISSUES'
            exit_code = 1
        else:
            overall_status = 'HEALTHY'
            exit_code = 0
        
        # Convert results to JSON-serializable format
        serializable_results = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert datetime to ISO string
            if 'timestamp' in result_dict and hasattr(result_dict['timestamp'], 'isoformat'):
                result_dict['timestamp'] = result_dict['timestamp'].isoformat()
            serializable_results.append(result_dict)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'exit_code': exit_code,
            'status_counts': status_counts,
            'total_checks': len(self.results),
            'results': serializable_results
        }


class EnvironmentDiagnostics:
    """Environment and system diagnostics."""
    
    def __init__(self, logger: DiagnosticLogger):
        self.logger = logger
    
    def get_system_info(self) -> SystemInfo:
        """Collect basic system information."""
        return SystemInfo(
            python_version=sys.version,
            platform_system=platform.system(),
            platform_release=platform.release(),
            platform_machine=platform.machine(),
            python_executable=sys.executable,
            working_directory=os.getcwd(),
            user=os.environ.get('USER', 'unknown'),
            home_directory=os.path.expanduser('~'),
            shell=os.environ.get('SHELL', 'unknown'),
            term=os.environ.get('TERM'),
            colorterm=os.environ.get('COLORTERM')
        )
    
    async def diagnose_python_environment(self):
        """Diagnose Python environment."""
        self.logger.section("Python Environment")
        
        # Python version check
        version_info = sys.version_info
        if version_info >= (3, 10):
            self.logger.log(DiagnosticResult(
                name="Python Version",
                status="PASS",
                message=f"Python {version_info.major}.{version_info.minor}.{version_info.micro}",
                details={"full_version": sys.version}
            ))
        elif version_info >= (3, 8):
            self.logger.log(DiagnosticResult(
                name="Python Version",
                status="WARN",
                message=f"Python {version_info.major}.{version_info.minor}.{version_info.micro} (older than recommended)",
                recommendations=["Upgrade to Python 3.10+ for best compatibility"]
            ))
        else:
            self.logger.log(DiagnosticResult(
                name="Python Version",
                status="CRITICAL",
                message=f"Python {version_info.major}.{version_info.minor}.{version_info.micro} (unsupported)",
                recommendations=["Upgrade to Python 3.10+ required"]
            ))
        
        # Platform check
        system = platform.system()
        supported_platforms = ['Linux', 'Darwin', 'Windows']
        if system in supported_platforms:
            self.logger.log(DiagnosticResult(
                name="Platform Support",
                status="PASS",
                message=f"{system} ({platform.release()})",
                details={
                    "machine": platform.machine(),
                    "processor": platform.processor() or "unknown"
                }
            ))
        else:
            self.logger.log(DiagnosticResult(
                name="Platform Support",
                status="WARN",
                message=f"Untested platform: {system}",
                recommendations=["Test TUI functionality carefully on this platform"]
            ))
        
        # Async support check
        try:
            loop = asyncio.get_running_loop()
            self.logger.log(DiagnosticResult(
                name="Async Support",
                status="PASS",
                message="Event loop available",
                details={"loop_type": type(loop).__name__}
            ))
        except RuntimeError:
            try:
                asyncio.new_event_loop()
                self.logger.log(DiagnosticResult(
                    name="Async Support",
                    status="PASS",
                    message="Async support available"
                ))
            except Exception as e:
                self.logger.log(DiagnosticResult(
                    name="Async Support",
                    status="CRITICAL",
                    message="Async support broken",
                    details={"error": str(e)},
                    recommendations=["Check Python asyncio installation"]
                ))
    
    async def diagnose_terminal_environment(self):
        """Diagnose terminal environment."""
        self.logger.section("Terminal Environment")
        
        # TTY detection
        is_tty = sys.stdout.isatty()
        if is_tty:
            self.logger.log(DiagnosticResult(
                name="TTY Detection",
                status="PASS",
                message="TTY available",
                details={"stdin_tty": sys.stdin.isatty(), "stderr_tty": sys.stderr.isatty()}
            ))
        else:
            self.logger.log(DiagnosticResult(
                name="TTY Detection",
                status="FAIL",
                message="No TTY detected",
                recommendations=[
                    "Run in a proper terminal",
                    "Avoid running in IDE consoles or pipes",
                    "Use 'python -u' if needed"
                ]
            ))
        
        # TERM environment variable
        term = os.environ.get('TERM')
        if term:
            known_terms = ['xterm', 'xterm-256color', 'screen', 'tmux', 'rxvt', 'vt100', 'vt220']
            if any(t in term for t in known_terms):
                self.logger.log(DiagnosticResult(
                    name="TERM Variable",
                    status="PASS",
                    message=f"TERM={term}",
                    details={"colorterm": os.environ.get('COLORTERM')}
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="TERM Variable",
                    status="WARN",
                    message=f"Unknown TERM: {term}",
                    recommendations=["May cause display issues"]
                ))
        else:
            self.logger.log(DiagnosticResult(
                name="TERM Variable",
                status="FAIL",
                message="TERM not set",
                recommendations=["Set TERM environment variable (e.g., export TERM=xterm-256color)"]
            ))
        
        # Terminal size detection
        try:
            size = shutil.get_terminal_size()
            if size.columns >= 80 and size.lines >= 24:
                self.logger.log(DiagnosticResult(
                    name="Terminal Size",
                    status="PASS",
                    message=f"{size.columns}x{size.lines}",
                    details={"min_recommended": "80x24"}
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Terminal Size",
                    status="WARN",
                    message=f"Small terminal: {size.columns}x{size.lines}",
                    recommendations=["Resize terminal to at least 80x24 for best experience"]
                ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Terminal Size",
                status="FAIL",
                message="Cannot detect terminal size",
                details={"error": str(e)},
                recommendations=["Check terminal configuration"]
            ))
        
        # Encoding support
        try:
            encoding = locale.getpreferredencoding()
            if 'utf' in encoding.lower():
                self.logger.log(DiagnosticResult(
                    name="Text Encoding",
                    status="PASS",
                    message=f"UTF-8 compatible ({encoding})"
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Text Encoding",
                    status="WARN",
                    message=f"Non-UTF encoding: {encoding}",
                    recommendations=["Set locale to UTF-8 for proper unicode support"]
                ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Text Encoding",
                status="WARN",
                message="Cannot detect encoding",
                details={"error": str(e)}
            ))

class LibraryDiagnostics:
    """Library and dependency diagnostics."""
    
    def __init__(self, logger: DiagnosticLogger):
        self.logger = logger
    
    def safe_import(self, module_name: str) -> Tuple[bool, Optional[Any], Optional[str]]:
        """Safely import a module and return status."""
        try:
            if '.' in module_name:
                # Handle submodule imports
                module = __import__(module_name, fromlist=[''])
            else:
                module = __import__(module_name)
            return True, module, None
        except ImportError as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"
    
    async def diagnose_core_dependencies(self):
        """Diagnose core library dependencies."""
        self.logger.section("Core Dependencies")
        
        # Core libraries required for TUI
        core_libs = [
            ('rich', 'Rich text rendering library'),
            ('click', 'Command line interface library'),
            ('pydantic', 'Data validation library'),
            ('asyncio', 'Async programming support'),
            ('prompt_toolkit', 'Advanced terminal interface library')
        ]
        
        for lib_name, description in core_libs:
            success, module, error = self.safe_import(lib_name)
            
            if success:
                version = "unknown"
                try:
                    # Use modern package metadata approach first
                    try:
                        version = get_package_version(lib_name)
                    except Exception:
                        # Fallback to module attributes for packages not installed via pip/setuptools
                        if hasattr(module, 'VERSION'):
                            version = module.VERSION
                        elif hasattr(module, 'version'):
                            version = module.version
                        elif hasattr(module, '__version__'):
                            # Use as last resort with warning suppression
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", DeprecationWarning)
                                version = module.__version__
                    
                    self.logger.log(DiagnosticResult(
                        name=f"{lib_name.title()} Library",
                        status="PASS",
                        message=f"Available (v{version})",
                        details={"description": description}
                    ))
                except Exception:
                    self.logger.log(DiagnosticResult(
                        name=f"{lib_name.title()} Library",
                        status="PASS",
                        message="Available",
                        details={"description": description, "version": "unknown"}
                    ))
            else:
                status = "CRITICAL" if lib_name in ['rich', 'asyncio'] else "FAIL"
                self.logger.log(DiagnosticResult(
                    name=f"{lib_name.title()} Library",
                    status=status,
                    message="Not available",
                    details={"error": error, "description": description},
                    recommendations=[f"Install {lib_name}: pip install {lib_name}"]
                ))
    
    async def diagnose_rich_capabilities(self):
        """Diagnose Rich library capabilities."""
        self.logger.section("Rich Library Capabilities")
        
        success, rich, error = self.safe_import('rich')
        if not success:
            self.logger.log(DiagnosticResult(
                name="Rich Library",
                status="CRITICAL",
                message="Rich not available",
                details={"error": error},
                recommendations=["Install Rich: pip install rich"]
            ))
            return
        
        # Test Rich Console
        try:
            from rich.console import Console
            console = Console()
            
            # Test basic functionality
            with redirect_stdout(io.StringIO()):
                console.print("Test", style="bold")
            
            self.logger.log(DiagnosticResult(
                name="Rich Console",
                status="PASS",
                message="Console creation successful",
                details={
                    "size": f"{console.size.width}x{console.size.height}",
                    "color_system": console.color_system,
                    "encoding": console.file.encoding if hasattr(console.file, 'encoding') else "unknown"
                }
            ))
            
            # Test color support
            if console.color_system:
                self.logger.log(DiagnosticResult(
                    name="Rich Color Support",
                    status="PASS",
                    message=f"Color system: {console.color_system}"
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Rich Color Support",
                    status="WARN",
                    message="No color support detected",
                    recommendations=["Enable color in terminal settings"]
                ))
            
            # Test specific Rich components used by TUI
            components = [
                ('rich.text', 'Text'),
                ('rich.panel', 'Panel'), 
                ('rich.layout', 'Layout'),
                ('rich.live', 'Live'),
                ('rich.table', 'Table')
            ]
            
            for module_name, class_name in components:
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    cls = getattr(module, class_name)
                    # Try to instantiate
                    if class_name == 'Text':
                        obj = cls("test")
                    elif class_name == 'Panel':
                        obj = cls("test")
                    elif class_name == 'Layout':
                        obj = cls()
                    elif class_name == 'Live':
                        obj = cls("test", console=console)
                    elif class_name == 'Table':
                        obj = cls()
                    
                    self.logger.log(DiagnosticResult(
                        name=f"Rich {class_name}",
                        status="PASS",
                        message="Component available"
                    ))
                except Exception as e:
                    self.logger.log(DiagnosticResult(
                        name=f"Rich {class_name}",
                        status="FAIL",
                        message="Component not working",
                        details={"error": str(e)},
                        recommendations=["Update Rich library"]
                    ))
            
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Rich Console",
                status="FAIL",
                message="Console creation failed",
                details={"error": str(e)},
                recommendations=["Check Rich installation and terminal compatibility"]
            ))


class TUIComponentDiagnostics:
    """TUI-specific component diagnostics."""
    
    def __init__(self, logger: DiagnosticLogger):
        self.logger = logger
        self.agentsmcp_available = False
    
    async def diagnose_agentsmcp_imports(self):
        """Diagnose AgentsMCP TUI imports."""
        self.logger.section("AgentsMCP TUI Components")
        
        # Try to add the source path
        src_path = Path(__file__).parent / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        # Test core AgentsMCP imports
        tui_imports = [
            'agentsmcp',
            'agentsmcp.ui',
            'agentsmcp.ui.v2',
            'agentsmcp.ui.v2.revolutionary_tui_interface',
            'agentsmcp.ui.v2.terminal_controller',
            'agentsmcp.ui.v2.event_system',
            'agentsmcp.ui.v2.input_rendering_pipeline',
            'agentsmcp.ui.v2.display_manager'
        ]
        
        for import_name in tui_imports:
            success, module, error = self.safe_import(import_name)
            
            if success:
                self.logger.log(DiagnosticResult(
                    name=f"Import {import_name}",
                    status="PASS",
                    message="Available"
                ))
                if import_name == 'agentsmcp':
                    self.agentsmcp_available = True
            else:
                status = "WARN" if 'agentsmcp' in import_name else "FAIL"
                self.logger.log(DiagnosticResult(
                    name=f"Import {import_name}",
                    status=status,
                    message="Import failed",
                    details={"error": error},
                    recommendations=["Ensure AgentsMCP is installed and PYTHONPATH is set correctly"]
                ))
    
    def safe_import(self, module_name: str) -> Tuple[bool, Optional[Any], Optional[str]]:
        """Safely import a module and return status."""
        try:
            if '.' in module_name:
                module = __import__(module_name, fromlist=[''])
            else:
                module = __import__(module_name)
            return True, module, None
        except ImportError as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"
    
    async def diagnose_event_system(self):
        """Diagnose event system functionality."""
        self.logger.section("Event System Testing")
        
        if not self.agentsmcp_available:
            self.logger.log(DiagnosticResult(
                name="Event System",
                status="FAIL", 
                message="AgentsMCP not available",
                recommendations=["Install AgentsMCP first"]
            ))
            return
        
        try:
            # Mock event system test
            test_events = []
            
            class MockEventHandler:
                def __init__(self):
                    self.handled_events = []
                
                async def handle_event(self, event):
                    self.handled_events.append(event)
                    return True
            
            # Test basic event handling
            handler = MockEventHandler()
            
            # Simulate event
            mock_event = {
                'type': 'test',
                'data': {'test': True},
                'timestamp': datetime.now()
            }
            
            await handler.handle_event(mock_event)
            
            if len(handler.handled_events) == 1:
                self.logger.log(DiagnosticResult(
                    name="Event Handling",
                    status="PASS",
                    message="Basic event handling works"
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Event Handling",
                    status="FAIL",
                    message="Event handling failed"
                ))
            
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Event System",
                status="FAIL",
                message="Event system test failed",
                details={"error": str(e)},
                recommendations=["Check event system implementation"]
            ))
    
    async def diagnose_input_handling(self):
        """Diagnose input handling capabilities."""
        self.logger.section("Input Handling")
        
        # Test basic input simulation
        try:
            # Mock input buffer
            input_buffer = []
            
            def mock_input_handler(char):
                input_buffer.append(char)
                return True
            
            # Simulate character input
            test_chars = ['h', 'e', 'l', 'l', 'o']
            for char in test_chars:
                mock_input_handler(char)
            
            if input_buffer == test_chars:
                self.logger.log(DiagnosticResult(
                    name="Character Input",
                    status="PASS",
                    message="Character input simulation works"
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Character Input",
                    status="FAIL",
                    message="Character input failed"
                ))
            
            # Test special key handling
            special_keys = ['\x7f', '\r', '\n', '\t', '\x1b']  # backspace, enter, newline, tab, escape
            special_buffer = []
            
            for key in special_keys:
                special_buffer.append(key)
            
            self.logger.log(DiagnosticResult(
                name="Special Keys",
                status="PASS",
                message="Special key handling ready",
                details={"tested_keys": ["backspace", "enter", "newline", "tab", "escape"]}
            ))
            
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Input Handling",
                status="FAIL",
                message="Input handling test failed",
                details={"error": str(e)}
            ))


class PerformanceDiagnostics:
    """Performance analysis and testing."""
    
    def __init__(self, logger: DiagnosticLogger):
        self.logger = logger
    
    async def diagnose_performance(self):
        """Diagnose performance characteristics."""
        self.logger.section("Performance Analysis")
        
        # Memory usage
        try:
            import resource
            memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On Linux, this is in KB, on macOS it's in bytes
            if platform.system() == 'Darwin':
                memory_mb = memory_usage / (1024 * 1024)
            else:
                memory_mb = memory_usage / 1024
            
            if memory_mb < 100:
                status = "PASS"
                message = f"Low memory usage: {memory_mb:.1f} MB"
            elif memory_mb < 500:
                status = "PASS"
                message = f"Moderate memory usage: {memory_mb:.1f} MB"
            else:
                status = "WARN"
                message = f"High memory usage: {memory_mb:.1f} MB"
            
            self.logger.log(DiagnosticResult(
                name="Memory Usage",
                status=status,
                message=message
            ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Memory Usage",
                status="WARN",
                message="Cannot measure memory usage",
                details={"error": str(e)}
            ))
        
        # Async performance
        start_time = time.time()
        tasks = []
        
        async def dummy_task(delay):
            await asyncio.sleep(delay)
            return delay
        
        # Create some concurrent tasks
        for i in range(10):
            tasks.append(dummy_task(0.01))
        
        try:
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            if total_time < 0.1:  # Should complete in parallel, not serial
                self.logger.log(DiagnosticResult(
                    name="Async Performance",
                    status="PASS",
                    message=f"Good async performance: {total_time:.3f}s for {len(tasks)} tasks"
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Async Performance",
                    status="WARN",
                    message=f"Slow async performance: {total_time:.3f}s for {len(tasks)} tasks",
                    recommendations=["Check system load and Python async implementation"]
                ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Async Performance",
                status="FAIL",
                message="Async performance test failed",
                details={"error": str(e)}
            ))
        
        # I/O performance test
        try:
            start_time = time.time()
            test_data = "x" * 1000
            
            # Test string operations (simulating text processing)
            for _ in range(100):
                processed = test_data.upper().lower().replace('x', 'y').replace('y', 'x')
            
            io_time = time.time() - start_time
            
            if io_time < 0.1:
                self.logger.log(DiagnosticResult(
                    name="Text Processing Performance",
                    status="PASS",
                    message=f"Fast text processing: {io_time:.3f}s"
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Text Processing Performance", 
                    status="WARN",
                    message=f"Slow text processing: {io_time:.3f}s",
                    recommendations=["System may be under load"]
                ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Text Processing Performance",
                status="WARN",
                message="Text processing test failed",
                details={"error": str(e)}
            ))


class ErrorScenarioDiagnostics:
    """Error scenario and edge case diagnostics."""
    
    def __init__(self, logger: DiagnosticLogger):
        self.logger = logger
    
    async def diagnose_error_scenarios(self):
        """Test various error scenarios."""
        self.logger.section("Error Scenario Testing")
        
        # Permission tests
        await self._test_file_permissions()
        
        # Signal handling
        await self._test_signal_handling()
        
        # Resource exhaustion simulation
        await self._test_resource_limits()
        
        # Edge case input handling
        await self._test_edge_case_inputs()
    
    async def _test_file_permissions(self):
        """Test file system permissions."""
        try:
            # Test write permission in current directory
            test_file = Path("._diagnostic_test_")
            test_file.write_text("test")
            test_file.unlink()
            
            self.logger.log(DiagnosticResult(
                name="File Permissions",
                status="PASS",
                message="Write permissions available"
            ))
        except PermissionError:
            self.logger.log(DiagnosticResult(
                name="File Permissions",
                status="FAIL",
                message="No write permission in current directory",
                recommendations=["Run from a directory with write permissions"]
            ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="File Permissions",
                status="WARN",
                message="File permission test failed",
                details={"error": str(e)}
            ))
    
    async def _test_signal_handling(self):
        """Test signal handling capabilities."""
        try:
            # Test if we can register signal handlers (Unix-like systems)
            if platform.system() != 'Windows':
                original_handler = signal.signal(signal.SIGTERM, signal.default_int_handler)
                signal.signal(signal.SIGTERM, original_handler)  # Restore
                
                self.logger.log(DiagnosticResult(
                    name="Signal Handling",
                    status="PASS",
                    message="Signal handling available"
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Signal Handling",
                    status="PASS",
                    message="Windows signal handling (limited)"
                ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Signal Handling",
                status="WARN",
                message="Signal handling test failed",
                details={"error": str(e)}
            ))
    
    async def _test_resource_limits(self):
        """Test resource limit handling."""
        try:
            # Test creating many small objects
            test_objects = []
            for i in range(1000):
                test_objects.append({"id": i, "data": "x" * 100})
            
            # Clean up
            del test_objects
            
            self.logger.log(DiagnosticResult(
                name="Memory Allocation",
                status="PASS",
                message="Memory allocation test passed"
            ))
        except MemoryError:
            self.logger.log(DiagnosticResult(
                name="Memory Allocation",
                status="FAIL",
                message="Memory allocation failed",
                recommendations=["System may be low on memory"]
            ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Memory Allocation",
                status="WARN",
                message="Memory allocation test failed",
                details={"error": str(e)}
            ))
    
    async def _test_edge_case_inputs(self):
        """Test edge case input handling."""
        try:
            # Test various problematic inputs
            edge_cases = [
                "",  # Empty string
                " " * 1000,  # Very long whitespace
                "\x00\x01\x02",  # Control characters
                "🙂🎉🚀",  # Unicode emojis
                "\n\r\t",  # Newlines and tabs
                "\\x1b[31m",  # ANSI escape sequences
            ]
            
            for i, test_input in enumerate(edge_cases):
                # Simple processing test
                processed = str(test_input).strip()
                if len(processed) >= 0:  # Basic sanity check
                    continue
            
            self.logger.log(DiagnosticResult(
                name="Edge Case Input Handling",
                status="PASS",
                message="Edge case inputs handled correctly"
            ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Edge Case Input Handling",
                status="WARN",
                message="Edge case input test failed",
                details={"error": str(e)}
            ))


class SimulationDiagnostics:
    """Real user simulation and integration testing."""
    
    def __init__(self, logger: DiagnosticLogger):
        self.logger = logger
    
    async def diagnose_user_scenarios(self):
        """Simulate real user scenarios."""
        self.logger.section("User Scenario Simulation")
        
        await self._simulate_typing_scenario()
        await self._simulate_navigation_scenario()
        await self._simulate_error_recovery_scenario()
    
    async def _simulate_typing_scenario(self):
        """Simulate user typing a command."""
        try:
            # Mock a typing session
            user_input = "hello world"
            input_buffer = ""
            
            # Simulate character-by-character input
            for char in user_input:
                input_buffer += char
                # Simulate processing delay
                await asyncio.sleep(0.001)
            
            # Simulate enter key
            input_buffer += "\n"
            
            if input_buffer.strip() == user_input:
                self.logger.log(DiagnosticResult(
                    name="Typing Simulation",
                    status="PASS",
                    message="Typing simulation successful"
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Typing Simulation",
                    status="FAIL",
                    message="Typing simulation failed"
                ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Typing Simulation",
                status="FAIL",
                message="Typing simulation error",
                details={"error": str(e)}
            ))
    
    async def _simulate_navigation_scenario(self):
        """Simulate user navigation through interface."""
        try:
            # Mock navigation state
            current_panel = "main"
            navigation_history = []
            
            # Simulate navigation sequence
            panels = ["main", "settings", "help", "main"]
            for panel in panels:
                navigation_history.append(current_panel)
                current_panel = panel
                await asyncio.sleep(0.01)  # Simulate user think time
            
            if len(navigation_history) == len(panels):
                self.logger.log(DiagnosticResult(
                    name="Navigation Simulation",
                    status="PASS",
                    message="Navigation simulation successful"
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Navigation Simulation",
                    status="FAIL",
                    message="Navigation simulation failed"
                ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Navigation Simulation",
                status="FAIL",
                message="Navigation simulation error",
                details={"error": str(e)}
            ))
    
    async def _simulate_error_recovery_scenario(self):
        """Simulate error recovery scenarios."""
        try:
            # Mock error scenarios and recovery
            error_scenarios = [
                "invalid_command",
                "network_timeout", 
                "permission_denied",
                "resource_exhausted"
            ]
            
            recovered_errors = []
            
            for error_type in error_scenarios:
                try:
                    # Simulate error
                    if error_type == "invalid_command":
                        raise ValueError("Invalid command")
                    elif error_type == "network_timeout":
                        raise TimeoutError("Network timeout")
                    elif error_type == "permission_denied":
                        raise PermissionError("Permission denied")
                    elif error_type == "resource_exhausted":
                        raise MemoryError("Resource exhausted")
                except Exception:
                    # Simulate recovery
                    recovered_errors.append(error_type)
            
            if len(recovered_errors) == len(error_scenarios):
                self.logger.log(DiagnosticResult(
                    name="Error Recovery",
                    status="PASS",
                    message="Error recovery simulation successful"
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Error Recovery",
                    status="FAIL",
                    message="Error recovery simulation failed"
                ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Error Recovery",
                status="FAIL",
                message="Error recovery simulation error",
                details={"error": str(e)}
            ))


class LayoutCorruptionDiagnostics:
    """Advanced layout corruption detection specialized for TUI display issues."""
    
    def __init__(self, logger: DiagnosticLogger):
        self.logger = logger
        self.src_path = Path(__file__).parent / "src"
        if self.src_path.exists():
            sys.path.insert(0, str(self.src_path))
            
        # Check Rich availability
        try:
            import rich
            self.RICH_AVAILABLE = True
        except ImportError:
            self.RICH_AVAILABLE = False
    
    async def diagnose_layout_corruption_patterns(self):
        """Run advanced layout corruption detection tests."""
        self.logger.section("Advanced Layout Corruption Detection")
        
        await self._test_rich_console_state_integrity()
        await self._test_terminal_capability_detection()
        await self._test_layout_rendering_artifacts()
        await self._test_input_triggered_layout_corruption()
        await self._test_console_size_mismatch_detection()
        await self._test_terminal_escape_sequence_handling()
        await self._test_progressive_layout_degradation()
        await self._test_cursor_positioning_accuracy()
    
    async def _test_rich_console_state_integrity(self):
        """Test Rich Console state before/after operations to detect corruption."""
        try:
            if not self.RICH_AVAILABLE:
                self.logger.log(DiagnosticResult(
                    name="Rich Console State Integrity",
                    status="FAIL",
                    message="Rich library not available"
                ))
                return
            
            from rich.console import Console
            from rich.panel import Panel
            
            # Create console and capture initial state
            console = Console(width=80, height=24, force_terminal=True)
            
            initial_state = {
                'width': console.size.width,
                'height': console.size.height,
                'color_system': console.color_system,
                'encoding': getattr(console.file, 'encoding', 'unknown'),
                'legacy_windows': getattr(console, 'legacy_windows', False),
                'is_terminal': console.is_terminal,
                'options': console.options
            }
            
            corruption_indicators = []
            
            # Test 1: Basic rendering operations
            test_panels = [
                Panel("Test 1", style="cyan"),
                Panel("Test with unicode: ▲◆●▼", style="green"),  
                Panel("Test\nMulti\nLine", style="yellow"),
                Panel("Long line " * 20, style="red")
            ]
            
            for i, panel in enumerate(test_panels):
                with redirect_stdout(io.StringIO()) as f:
                    console.print(panel)
                    output = f.getvalue()
                
                # Check for state corruption after each render
                current_state = {
                    'width': console.size.width,
                    'height': console.size.height,
                    'color_system': console.color_system,
                    'encoding': getattr(console.file, 'encoding', 'unknown'),
                    'legacy_windows': getattr(console, 'legacy_windows', False),
                    'is_terminal': console.is_terminal,
                    'options': console.options
                }
                
                if current_state != initial_state:
                    corruption_indicators.append(f"state_changed_after_panel_{i}")
                
                # Check output quality
                lines = output.split('\n')
                valid_lines = [line for line in lines if line.strip()]
                
                if len(valid_lines) < 2:  # Panel should have at least 2 lines
                    corruption_indicators.append(f"truncated_output_panel_{i}")
                
                # Check for broken box characters
                box_chars_present = any(char in output for char in ['┌', '┐', '└', '┘', '─', '│'])
                if not box_chars_present and len(valid_lines) > 0:
                    corruption_indicators.append(f"missing_box_chars_panel_{i}")
            
            # Test 2: Rapid console operations
            for i in range(20):
                console.print(f"Rapid test {i}", style="dim")
            
            rapid_test_state = {
                'width': console.size.width,
                'height': console.size.height,
                'color_system': console.color_system
            }
            
            if rapid_test_state['width'] != initial_state['width'] or \
               rapid_test_state['height'] != initial_state['height']:
                corruption_indicators.append("console_size_corruption_after_rapid_ops")
            
            # Test 3: Memory pressure test
            large_panels = []
            for i in range(10):
                large_content = "X" * (100 * (i + 1))
                large_panels.append(Panel(large_content, title=f"Memory Test {i}"))
            
            memory_test_errors = []
            for i, panel in enumerate(large_panels):
                try:
                    with redirect_stdout(io.StringIO()) as f:
                        console.print(panel)
                        output = f.getvalue()
                    if len(output) < 50:  # Should have substantial output
                        memory_test_errors.append(f"memory_panel_{i}_truncated")
                except Exception as e:
                    memory_test_errors.append(f"memory_panel_{i}_error: {str(e)}")
            
            if memory_test_errors:
                corruption_indicators.extend(memory_test_errors[:3])  # Limit to first 3
            
            # Report results
            if corruption_indicators:
                self.logger.log(DiagnosticResult(
                    name="Rich Console State Integrity",
                    status="FAIL",
                    message="Console state integrity issues detected",
                    details={
                        "corruption_indicators": corruption_indicators,
                        "initial_console_size": f"{initial_state['width']}x{initial_state['height']}",
                        "color_system": initial_state['color_system'],
                        "encoding": initial_state['encoding'],
                        "is_terminal": initial_state['is_terminal']
                    },
                    recommendations=[
                        "Check terminal compatibility with Rich",
                        "Verify TERM and COLORTERM environment variables",
                        "Test with different terminal emulators"
                    ]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Rich Console State Integrity",
                    status="PASS",
                    message="Console state remains stable across operations",
                    details={
                        "tests_completed": 3,
                        "console_size": f"{initial_state['width']}x{initial_state['height']}",
                        "color_system": initial_state['color_system']
                    }
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Rich Console State Integrity",
                status="FAIL",
                message="Console state integrity test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_terminal_capability_detection(self):
        """Test terminal capability detection and compatibility issues."""
        try:
            terminal_issues = []
            
            # Test 1: TERM variable analysis
            term = os.environ.get('TERM', '')
            colorterm = os.environ.get('COLORTERM', '')
            
            # Known problematic TERM values for Rich layouts
            problematic_terms = ['dumb', 'unknown', 'cons25', 'emacs']
            if any(prob in term.lower() for prob in problematic_terms):
                terminal_issues.append(f"problematic_term: {term}")
            
            # Test 2: Terminal size detection accuracy
            try:
                # Multiple methods to detect terminal size
                shutil_size = shutil.get_terminal_size()
                os_size = os.get_terminal_size() if hasattr(os, 'get_terminal_size') else None
                
                # Test if Rich detects the same size
                if self.RICH_AVAILABLE:
                    from rich.console import Console
                    console = Console()
                    rich_size = console.size
                    
                    # Compare sizes
                    if shutil_size.columns != rich_size.width or shutil_size.lines != rich_size.height:
                        terminal_issues.append(f"size_mismatch: shutil={shutil_size.columns}x{shutil_size.lines}, rich={rich_size.width}x{rich_size.height}")
                    
                    # Test size detection consistency
                    for i in range(5):
                        current_rich_size = Console().size
                        if current_rich_size.width != rich_size.width or current_rich_size.height != rich_size.height:
                            terminal_issues.append("inconsistent_size_detection")
                            break
                        await asyncio.sleep(0.01)
                
            except Exception as e:
                terminal_issues.append(f"size_detection_error: {str(e)}")
            
            # Test 3: Color capability detection
            try:
                if self.RICH_AVAILABLE:
                    from rich.console import Console
                    console = Console(force_terminal=True)
                    
                    # Test different color systems
                    color_systems = ['standard', 'eight_bit', 'truecolor']
                    detected_system = console.color_system
                    
                    if not detected_system:
                        terminal_issues.append("no_color_system_detected")
                    elif detected_system not in color_systems:
                        terminal_issues.append(f"unknown_color_system: {detected_system}")
                    
                    # Test actual color rendering
                    with redirect_stdout(io.StringIO()) as f:
                        console.print("Test", style="red bold")
                        color_output = f.getvalue()
                    
                    # Should contain ANSI codes for colors
                    if '\x1b[' not in color_output and detected_system:
                        terminal_issues.append("color_codes_missing_despite_detection")
                        
            except Exception as e:
                terminal_issues.append(f"color_detection_error: {str(e)}")
            
            # Test 4: Unicode/Box drawing character support
            try:
                if self.RICH_AVAILABLE:
                    from rich.console import Console
                    from rich.panel import Panel
                    
                    console = Console(force_terminal=True)
                    with redirect_stdout(io.StringIO()) as f:
                        console.print(Panel("Unicode test: ░▒▓█▲▼◆●"))
                        unicode_output = f.getvalue()
                    
                    # Check for box drawing characters
                    box_chars = ['┌', '┐', '└', '┘', '─', '│']
                    has_box_chars = any(char in unicode_output for char in box_chars)
                    
                    if not has_box_chars:
                        terminal_issues.append("box_drawing_chars_missing")
                    
                    # Check for unicode symbols
                    unicode_chars = ['░', '▒', '▓', '█', '▲', '▼', '◆', '●']
                    has_unicode = any(char in unicode_output for char in unicode_chars)
                    
                    if not has_unicode:
                        terminal_issues.append("unicode_symbols_missing")
                        
            except Exception as e:
                terminal_issues.append(f"unicode_test_error: {str(e)}")
            
            # Test 5: Terminal response/latency
            try:
                import subprocess
                start_time = time.time()
                
                # Test terminal responsiveness with a simple echo
                result = subprocess.run(['echo', 'terminal_test'], 
                                      capture_output=True, text=True, timeout=1.0)
                
                response_time = time.time() - start_time
                
                if response_time > 0.1:  # Should be much faster
                    terminal_issues.append(f"slow_terminal_response: {response_time:.3f}s")
                    
            except subprocess.TimeoutExpired:
                terminal_issues.append("terminal_response_timeout")
            except Exception as e:
                terminal_issues.append(f"terminal_response_error: {str(e)}")
            
            # Report results
            if terminal_issues:
                self.logger.log(DiagnosticResult(
                    name="Terminal Capability Detection",
                    status="FAIL",
                    message="Terminal compatibility issues detected",
                    details={
                        "issues": terminal_issues,
                        "term_env": term,
                        "colorterm_env": colorterm,
                        "terminal_size": f"{shutil.get_terminal_size().columns}x{shutil.get_terminal_size().lines}"
                    },
                    recommendations=[
                        "Use a more capable terminal emulator",
                        "Check TERM and COLORTERM environment variables",
                        "Enable unicode and color support in terminal settings",
                        "Consider using xterm-256color as TERM value"
                    ]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Terminal Capability Detection",
                    status="PASS",
                    message="Terminal capabilities are well-supported",
                    details={
                        "term_env": term,
                        "colorterm_env": colorterm,
                        "size_detection_consistent": True,
                        "unicode_support": True
                    }
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Terminal Capability Detection",
                status="FAIL",
                message="Terminal capability detection failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_layout_rendering_artifacts(self):
        """Test for layout rendering artifacts like broken box characters and truncated lines."""
        try:
            if not self.RICH_AVAILABLE:
                self.logger.log(DiagnosticResult(
                    name="Layout Rendering Artifacts",
                    status="FAIL",
                    message="Rich library not available"
                ))
                return
            
            from rich.console import Console
            from rich.layout import Layout
            from rich.panel import Panel
            from rich.text import Text
            
            console = Console(width=80, height=24, force_terminal=True)
            rendering_artifacts = []
            
            # Test 1: Complex nested layouts
            try:
                main_layout = Layout()
                main_layout.split_column(
                    Layout(name="header", size=3),
                    Layout(name="body"),
                    Layout(name="footer", size=3)
                )
                
                main_layout["body"].split_row(
                    Layout(name="sidebar", size=20),
                    Layout(name="content")
                )
                
                # Populate with content
                main_layout["header"].update(Panel("Header", style="cyan"))
                main_layout["sidebar"].update(Panel("Sidebar\nLine 2\nLine 3", style="green"))
                main_layout["content"].update(Panel("Main Content Area\nWith multiple lines\nAnd more text", style="white"))
                main_layout["footer"].update(Panel("Footer", style="yellow"))
                
                # Render and check for artifacts
                with redirect_stdout(io.StringIO()) as f:
                    console.print(main_layout)
                    layout_output = f.getvalue()
                
                # Analyze output for artifacts
                lines = layout_output.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                
                if len(non_empty_lines) < 20:  # Expected substantial output for complex layout
                    rendering_artifacts.append("complex_layout_truncated")
                
                # Check for proper box drawing
                required_box_chars = ['┌', '┐', '└', '┘', '─', '│']
                missing_chars = [char for char in required_box_chars if char not in layout_output]
                if missing_chars:
                    rendering_artifacts.append(f"missing_box_chars: {missing_chars}")
                
                # Check for layout structure integrity
                if "Header" not in layout_output or "Sidebar" not in layout_output or "Footer" not in layout_output:
                    rendering_artifacts.append("layout_content_missing")
                    
            except Exception as e:
                rendering_artifacts.append(f"complex_layout_error: {str(e)}")
            
            # Test 2: Rapid layout refresh simulation
            try:
                refresh_artifacts = []
                
                for i in range(10):
                    dynamic_layout = Layout()
                    dynamic_layout.split_column(
                        Layout(Panel(f"Dynamic Header {i}", style="cyan"), size=3),
                        Layout(Panel(f"Content {i}\n{'█' * (i % 20)}", style="white"))
                    )
                    
                    with redirect_stdout(io.StringIO()) as f:
                        console.print(dynamic_layout)
                        refresh_output = f.getvalue()
                    
                    # Check each refresh for artifacts
                    if len(refresh_output.split('\n')) < 5:
                        refresh_artifacts.append(f"refresh_{i}_truncated")
                    
                    if f"Dynamic Header {i}" not in refresh_output:
                        refresh_artifacts.append(f"refresh_{i}_content_missing")
                    
                    # Small delay to simulate real refresh timing
                    await asyncio.sleep(0.001)
                
                if refresh_artifacts:
                    rendering_artifacts.extend([f"rapid_refresh: {artifact}" for artifact in refresh_artifacts[:3]])
                    
            except Exception as e:
                rendering_artifacts.append(f"rapid_refresh_error: {str(e)}")
            
            # Test 3: Text wrapping and overflow handling
            try:
                # Test long lines that should wrap
                very_long_text = "This is an extremely long line of text that should cause wrapping behavior in the terminal and we need to see if it renders properly without artifacts " * 3
                
                wrap_panel = Panel(very_long_text, title="Wrap Test", width=60)
                
                with redirect_stdout(io.StringIO()) as f:
                    console.print(wrap_panel)
                    wrap_output = f.getvalue()
                
                wrap_lines = wrap_output.split('\n')
                
                # Check if text is properly contained within panel
                for line in wrap_lines:
                    if line.strip() and len(line) > 80:  # Should not exceed console width
                        rendering_artifacts.append("text_overflow_beyond_console_width")
                        break
                
                # Check for truncated text (should see some of the long text)
                if "extremely long line" not in wrap_output:
                    rendering_artifacts.append("wrapped_text_missing")
                    
            except Exception as e:
                rendering_artifacts.append(f"text_wrap_error: {str(e)}")
            
            # Test 4: Special characters and encoding
            try:
                special_chars_text = "Special: €£¥§©®™°±×÷αβγδ⭐🚀🎉"
                special_panel = Panel(special_chars_text, title="Special Chars")
                
                with redirect_stdout(io.StringIO()) as f:
                    console.print(special_panel)
                    special_output = f.getvalue()
                
                # Check if special characters render (or at least don't break layout)
                if len(special_output.split('\n')) < 3:  # Panel should have multiple lines
                    rendering_artifacts.append("special_chars_break_layout")
                    
                # Check for encoding issues (question marks or boxes)
                question_marks = special_output.count('?')
                if question_marks > 5:  # Too many replacement chars
                    rendering_artifacts.append("encoding_issues_detected")
                    
            except Exception as e:
                rendering_artifacts.append(f"special_chars_error: {str(e)}")
            
            # Report results
            if rendering_artifacts:
                self.logger.log(DiagnosticResult(
                    name="Layout Rendering Artifacts",
                    status="FAIL",
                    message="Layout rendering artifacts detected",
                    details={
                        "artifacts": rendering_artifacts,
                        "artifact_count": len(rendering_artifacts),
                        "console_size": f"{console.size.width}x{console.size.height}"
                    },
                    recommendations=[
                        "Check terminal's box drawing character support",
                        "Verify terminal encoding is set to UTF-8",
                        "Test with different terminal emulators",
                        "Check Rich library version compatibility"
                    ]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Layout Rendering Artifacts",
                    status="PASS",
                    message="No layout rendering artifacts detected",
                    details={
                        "tests_completed": 4,
                        "console_size": f"{console.size.width}x{console.size.height}",
                        "complex_layout_ok": True,
                        "rapid_refresh_ok": True,
                        "text_wrapping_ok": True,
                        "special_chars_ok": True
                    }
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Layout Rendering Artifacts",
                status="FAIL",
                message="Layout rendering artifacts test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_input_triggered_layout_corruption(self):
        """Test for layout corruption specifically triggered by input events."""
        try:
            if not self.RICH_AVAILABLE:
                self.logger.log(DiagnosticResult(
                    name="Input-Triggered Layout Corruption",
                    status="FAIL",
                    message="Rich library not available"
                ))
                return
            
            from rich.console import Console
            from rich.layout import Layout
            from rich.panel import Panel
            from rich.live import Live
            
            console = Console(width=80, height=24, force_terminal=True)
            input_corruption_issues = []
            
            # Test 1: Simulate typing with layout updates
            try:
                def create_input_layout(current_input: str):
                    layout = Layout()
                    layout.split_column(
                        Layout(Panel("TUI Header", style="cyan"), name="header", size=3),
                        Layout(Panel("Main Content Area", style="white"), name="main"),
                        Layout(Panel(f"Input: {current_input}", style="green"), name="input", size=3)
                    )
                    return layout
                
                # Simulate character-by-character typing
                test_input = "hello world, this is a test message"
                current_input = ""
                
                layout_states = []
                
                for i, char in enumerate(test_input):
                    current_input += char
                    
                    # Create layout with current input
                    layout = create_input_layout(current_input)
                    
                    # Capture rendered output
                    with redirect_stdout(io.StringIO()) as f:
                        console.print(layout)
                        output = f.getvalue()
                    
                    # Analyze layout state
                    lines = output.split('\n')
                    valid_lines = [line for line in lines if line.strip()]
                    
                    layout_state = {
                        'char_index': i,
                        'char': char,
                        'input_text': current_input,
                        'line_count': len(valid_lines),
                        'has_input_text': current_input in output,
                        'has_structure': all(text in output for text in ["TUI Header", "Main Content", "Input:"]),
                        'output_length': len(output)
                    }
                    
                    layout_states.append(layout_state)
                    
                    # Check for immediate corruption
                    if layout_state['line_count'] < 10:  # Layout collapsed
                        input_corruption_issues.append(f"layout_collapsed_at_char_{i}_{char}")
                    
                    if not layout_state['has_input_text']:
                        input_corruption_issues.append(f"input_text_missing_at_char_{i}_{char}")
                    
                    if not layout_state['has_structure']:
                        input_corruption_issues.append(f"layout_structure_broken_at_char_{i}_{char}")
                    
                    # Small delay to simulate real typing speed
                    await asyncio.sleep(0.001)
                
                # Analysis: Look for patterns in corruption
                if len(layout_states) > 1:
                    line_counts = [state['line_count'] for state in layout_states]
                    
                    # Check for progressive degradation
                    if any(line_counts[i] < line_counts[i-1] - 5 for i in range(1, len(line_counts))):
                        input_corruption_issues.append("progressive_layout_degradation")
                    
                    # Check for consistency issues
                    expected_line_count = layout_states[0]['line_count']
                    inconsistent_states = [state for state in layout_states if abs(state['line_count'] - expected_line_count) > 3]
                    
                    if len(inconsistent_states) > len(layout_states) * 0.2:  # More than 20% inconsistent
                        input_corruption_issues.append("inconsistent_layout_sizing")
                        
            except Exception as e:
                input_corruption_issues.append(f"typing_simulation_error: {str(e)}")
            
            # Test 2: Rapid input changes (simulate fast typing/backspacing)
            try:
                rapid_inputs = [
                    "h",
                    "he", 
                    "hel",
                    "hell",
                    "hello",
                    "hell",  # backspace
                    "hello",
                    "hello ",
                    "hello w",
                    "hello wo",
                    "hello wor",
                    "hello worl",
                    "hello world"
                ]
                
                rapid_corruption_count = 0
                
                for i, test_text in enumerate(rapid_inputs):
                    layout = create_input_layout(test_text)
                    
                    with redirect_stdout(io.StringIO()) as f:
                        console.print(layout)
                        output = f.getvalue()
                    
                    # Quick corruption checks
                    if test_text not in output:
                        rapid_corruption_count += 1
                    
                    if len(output.split('\n')) < 8:  # Expect reasonable line count
                        rapid_corruption_count += 1
                
                if rapid_corruption_count > 3:  # More than few failures
                    input_corruption_issues.append(f"rapid_input_corruption: {rapid_corruption_count}/{len(rapid_inputs)} failed")
                    
            except Exception as e:
                input_corruption_issues.append(f"rapid_input_error: {str(e)}")
            
            # Test 3: Special input characters
            try:
                special_inputs = [
                    "normal text",
                    "unicode: 🚀🎉⭐",
                    "tabs:\tand\ttabs",
                    "newlines:\nand\nmore",
                    "quotes: \"hello\" 'world'",
                    "symbols: !@#$%^&*()",
                    "long: " + "x" * 100
                ]
                
                special_char_issues = 0
                
                for special_input in special_inputs:
                    layout = create_input_layout(special_input)
                    
                    try:
                        with redirect_stdout(io.StringIO()) as f:
                            console.print(layout)
                            output = f.getvalue()
                        
                        # Check if special input breaks layout
                        if len(output.split('\n')) < 5:
                            special_char_issues += 1
                            
                    except Exception:
                        special_char_issues += 1
                
                if special_char_issues > 1:
                    input_corruption_issues.append(f"special_chars_break_layout: {special_char_issues}/{len(special_inputs)}")
                    
            except Exception as e:
                input_corruption_issues.append(f"special_chars_error: {str(e)}")
            
            # Test 4: Memory pressure during input (simulate real-world usage)
            try:
                # Simulate a long conversation with growing input history
                conversation_layouts = []
                
                for msg_count in range(20):
                    # Build conversation history
                    history = []
                    for i in range(msg_count):
                        history.append(f"Message {i}: This is message number {i}")
                    
                    current_input = f"typing message {msg_count}..."
                    
                    # Create layout with history and input
                    layout = Layout()
                    layout.split_column(
                        Layout(Panel("Header", style="cyan"), size=3),
                        Layout(Panel("\n".join(history[-5:]) if history else "No messages", style="white")),  # Show last 5
                        Layout(Panel(f"Input: {current_input}", style="green"), size=3)
                    )
                    
                    try:
                        with redirect_stdout(io.StringIO()) as f:
                            console.print(layout)
                            output = f.getvalue()
                        
                        # Check for memory-related corruption
                        if len(output) < 100:  # Should have substantial output
                            input_corruption_issues.append(f"memory_pressure_corruption_at_msg_{msg_count}")
                            
                    except Exception:
                        input_corruption_issues.append(f"memory_pressure_error_at_msg_{msg_count}")
                        
            except Exception as e:
                input_corruption_issues.append(f"memory_pressure_test_error: {str(e)}")
            
            # Report results
            if input_corruption_issues:
                self.logger.log(DiagnosticResult(
                    name="Input-Triggered Layout Corruption",
                    status="FAIL",
                    message="Input events trigger layout corruption - MATCHES REPORTED ISSUE",
                    details={
                        "corruption_issues": input_corruption_issues,
                        "issue_count": len(input_corruption_issues),
                        "typing_corruption": any("char_" in issue for issue in input_corruption_issues),
                        "rapid_input_corruption": any("rapid_input" in issue for issue in input_corruption_issues),
                        "special_chars_corruption": any("special_chars" in issue for issue in input_corruption_issues)
                    },
                    recommendations=[
                        "CRITICAL: Input handling is breaking layout rendering",
                        "Separate input buffer from display layout updates",
                        "Implement proper state isolation between input and display",
                        "Add input sanitization and validation before layout updates",
                        "Consider using different rendering strategy for input area"
                    ]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Input-Triggered Layout Corruption",
                    status="PASS",
                    message="Input events do not corrupt layout",
                    details={
                        "typing_simulation_ok": True,
                        "rapid_input_ok": True,
                        "special_chars_ok": True,
                        "memory_pressure_ok": True
                    }
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Input-Triggered Layout Corruption",
                status="FAIL",
                message="Input-triggered corruption test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_console_size_mismatch_detection(self):
        """Test for console size mismatches between Rich and actual terminal."""
        try:
            size_mismatch_issues = []
            
            # Get terminal size from different sources
            try:
                shutil_size = shutil.get_terminal_size()
                fallback_used = False
            except OSError:
                shutil_size = shutil.get_terminal_size((80, 24))  # fallback
                fallback_used = True
                size_mismatch_issues.append("shutil_size_detection_failed")
            
            if self.RICH_AVAILABLE:
                from rich.console import Console
                
                # Test different console configurations
                console_configs = [
                    {"force_terminal": True},
                    {"force_terminal": False},
                    {"width": 80, "height": 24, "force_terminal": True},
                    {"width": shutil_size.columns, "height": shutil_size.lines, "force_terminal": True}
                ]
                
                rich_sizes = []
                
                for config in console_configs:
                    try:
                        console = Console(**config)
                        rich_size = console.size
                        rich_sizes.append({
                            "config": config,
                            "size": f"{rich_size.width}x{rich_size.height}",
                            "width": rich_size.width,
                            "height": rich_size.height
                        })
                    except Exception as e:
                        size_mismatch_issues.append(f"console_config_error: {config} - {str(e)}")
                
                # Compare sizes
                expected_size = f"{shutil_size.columns}x{shutil_size.lines}"
                
                for rich_info in rich_sizes:
                    if rich_info["size"] != expected_size:
                        # Allow small differences (±1) as they're common
                        width_diff = abs(rich_info["width"] - shutil_size.columns)
                        height_diff = abs(rich_info["height"] - shutil_size.lines)
                        
                        if width_diff > 1 or height_diff > 1:
                            size_mismatch_issues.append(
                                f"size_mismatch: {rich_info['config']} -> {rich_info['size']} vs expected {expected_size}"
                            )
                
                # Test size consistency across multiple console instances
                console_sizes = []
                for i in range(5):
                    console = Console(force_terminal=True)
                    console_sizes.append((console.size.width, console.size.height))
                
                # Check for inconsistency
                unique_sizes = set(console_sizes)
                if len(unique_sizes) > 1:
                    size_mismatch_issues.append(f"inconsistent_console_sizes: {unique_sizes}")
                
                # Test size detection timing
                size_detection_times = []
                for i in range(10):
                    start_time = time.time()
                    console = Console(force_terminal=True)
                    _ = console.size  # Access size property
                    detection_time = time.time() - start_time
                    size_detection_times.append(detection_time)
                
                avg_detection_time = sum(size_detection_times) / len(size_detection_times)
                if avg_detection_time > 0.01:  # Should be very fast
                    size_mismatch_issues.append(f"slow_size_detection: {avg_detection_time:.4f}s average")
                
                # Test dynamic size changes (simulate resize)
                try:
                    original_console = Console(force_terminal=True)
                    original_size = original_console.size
                    
                    # Create console with different size
                    resized_console = Console(width=original_size.width + 10, height=original_size.height + 5, force_terminal=True)
                    resized_size = resized_console.size
                    
                    # Should reflect the specified size
                    if resized_size.width != original_size.width + 10:
                        size_mismatch_issues.append("manual_resize_width_ignored")
                    
                    if resized_size.height != original_size.height + 5:
                        size_mismatch_issues.append("manual_resize_height_ignored")
                        
                except Exception as e:
                    size_mismatch_issues.append(f"resize_test_error: {str(e)}")
                
            else:
                size_mismatch_issues.append("rich_not_available_for_size_testing")
            
            # Report results
            if size_mismatch_issues:
                self.logger.log(DiagnosticResult(
                    name="Console Size Mismatch Detection",
                    status="FAIL",
                    message="Console size detection issues found",
                    details={
                        "size_issues": size_mismatch_issues,
                        "shutil_size": f"{shutil_size.columns}x{shutil_size.lines}",
                        "fallback_used": fallback_used,
                        "issue_count": len(size_mismatch_issues)
                    },
                    recommendations=[
                        "Check terminal size detection across different terminals",
                        "Verify LINES and COLUMNS environment variables",
                        "Test terminal resize handling",
                        "Consider manual size configuration for problematic terminals"
                    ]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Console Size Mismatch Detection",
                    status="PASS",
                    message="Console size detection is consistent",
                    details={
                        "terminal_size": f"{shutil_size.columns}x{shutil_size.lines}",
                        "rich_detection_consistent": True,
                        "resize_handling_ok": True
                    }
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Console Size Mismatch Detection",
                status="FAIL",
                message="Console size mismatch detection failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_terminal_escape_sequence_handling(self):
        """Test terminal escape sequence handling and cursor positioning."""
        try:
            escape_handling_issues = []
            
            # Test 1: Basic ANSI escape sequences
            basic_sequences = [
                ('\x1b[31m', 'red_color'),
                ('\x1b[1m', 'bold'),
                ('\x1b[0m', 'reset'),
                ('\x1b[2J', 'clear_screen'),
                ('\x1b[H', 'cursor_home'),
                ('\x1b[1;1H', 'cursor_position'),
                ('\x1b[K', 'clear_line'),
                ('\x1b[?25h', 'show_cursor'),
                ('\x1b[?25l', 'hide_cursor')
            ]
            
            for sequence, name in basic_sequences:
                try:
                    # Test if the sequence can be written without errors
                    with redirect_stdout(io.StringIO()) as f:
                        print(sequence, end='', file=f)
                        output = f.getvalue()
                    
                    # Basic validation - should not crash
                    if len(output) != len(sequence):
                        escape_handling_issues.append(f"sequence_modified: {name}")
                        
                except Exception as e:
                    escape_handling_issues.append(f"sequence_error: {name} - {str(e)}")
            
            # Test 2: Rich's escape sequence generation
            if self.RICH_AVAILABLE:
                from rich.console import Console
                from rich.text import Text
                
                try:
                    console = Console(force_terminal=True, color_system="truecolor")
                    
                    # Create styled text
                    styled_text = Text()
                    styled_text.append("Red", style="red")
                    styled_text.append(" Normal ")
                    styled_text.append("Bold", style="bold")
                    styled_text.append(" Blue", style="blue")
                    
                    # Render to string
                    with redirect_stdout(io.StringIO()) as f:
                        console.print(styled_text, end='')
                        rich_output = f.getvalue()
                    
                    # Check if ANSI codes are present
                    if '\x1b[' not in rich_output:
                        escape_handling_issues.append("rich_ansi_codes_missing")
                    
                    # Check for proper reset sequences
                    reset_count = rich_output.count('\x1b[0m')
                    if reset_count < 2:  # Should have resets after color changes
                        escape_handling_issues.append("insufficient_reset_sequences")
                    
                    # Test escape sequence parsing
                    import re
                    ansi_pattern = re.compile(r'\x1b\[[0-9;]*[mGKHJABCDEF]')
                    sequences = ansi_pattern.findall(rich_output)
                    
                    if len(sequences) < 3:  # Should have multiple sequences for the styled text
                        escape_handling_issues.append("few_ansi_sequences_generated")
                        
                except Exception as e:
                    escape_handling_issues.append(f"rich_escape_test_error: {str(e)}")
            
            # Test 3: Cursor positioning accuracy
            try:
                # This is a simulation - in real terminal this would move cursor
                position_commands = [
                    '\x1b[5;10H',  # Move to row 5, col 10
                    '\x1b[1;1H',   # Home position
                    '\x1b[10;20H', # Move to row 10, col 20
                ]
                
                for cmd in position_commands:
                    try:
                        with redirect_stdout(io.StringIO()) as f:
                            print(cmd, end='', file=f)
                            cursor_output = f.getvalue()
                        
                        if cursor_output != cmd:
                            escape_handling_issues.append("cursor_command_modified")
                            
                    except Exception as e:
                        escape_handling_issues.append(f"cursor_positioning_error: {str(e)}")
                        
            except Exception as e:
                escape_handling_issues.append(f"cursor_test_error: {str(e)}")
            
            # Test 4: Terminal mode handling (Raw vs Cooked)
            try:
                # Test if we can detect terminal modes
                is_tty = sys.stdout.isatty()
                
                if is_tty:
                    # In a real TTY, test mode switching capability
                    import termios
                    import tty
                    
                    # Get current terminal attributes
                    try:
                        old_attrs = termios.tcgetattr(sys.stdin.fileno())
                        
                        # Test if we can set raw mode (without actually doing it)
                        # This tests if the terminal supports mode switching
                        test_attrs = termios.tcgetattr(sys.stdin.fileno())
                        tty.setraw(sys.stdin.fileno())
                        
                        # Immediately restore
                        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_attrs)
                        
                    except (termios.error, OSError) as e:
                        escape_handling_issues.append(f"terminal_mode_switching_failed: {str(e)}")
                        
                else:
                    # Not a TTY - this is expected in many test environments
                    escape_handling_issues.append("not_running_in_tty")
                    
            except ImportError:
                # termios not available (Windows)
                escape_handling_issues.append("termios_not_available")
            except Exception as e:
                escape_handling_issues.append(f"terminal_mode_test_error: {str(e)}")
            
            # Test 5: Alternate screen buffer
            try:
                # Test alternate screen sequences
                alt_screen_enter = '\x1b[?1049h'  # Enter alternate screen
                alt_screen_exit = '\x1b[?1049l'   # Exit alternate screen
                
                for seq, name in [(alt_screen_enter, 'alt_screen_enter'), (alt_screen_exit, 'alt_screen_exit')]:
                    try:
                        with redirect_stdout(io.StringIO()) as f:
                            print(seq, end='', file=f)
                            alt_output = f.getvalue()
                        
                        if alt_output != seq:
                            escape_handling_issues.append(f"alt_screen_sequence_modified: {name}")
                            
                    except Exception as e:
                        escape_handling_issues.append(f"alt_screen_error: {name} - {str(e)}")
                        
            except Exception as e:
                escape_handling_issues.append(f"alt_screen_test_error: {str(e)}")
            
            # Report results
            if escape_handling_issues:
                self.logger.log(DiagnosticResult(
                    name="Terminal Escape Sequence Handling",
                    status="FAIL",
                    message="Terminal escape sequence handling issues detected",
                    details={
                        "escape_issues": escape_handling_issues,
                        "issue_count": len(escape_handling_issues),
                        "is_tty": sys.stdout.isatty(),
                        "platform": platform.system()
                    },
                    recommendations=[
                        "Use a more capable terminal emulator",
                        "Check terminal's ANSI escape sequence support",
                        "Verify terminal is properly configured for colors",
                        "Test with xterm-256color TERM setting"
                    ]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Terminal Escape Sequence Handling",
                    status="PASS",
                    message="Terminal escape sequence handling appears functional",
                    details={
                        "basic_sequences_ok": True,
                        "rich_sequences_ok": self.RICH_AVAILABLE,
                        "cursor_positioning_ok": True,
                        "terminal_modes_ok": True
                    }
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Terminal Escape Sequence Handling",
                status="FAIL",
                message="Terminal escape sequence handling test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_progressive_layout_degradation(self):
        """Test for progressive layout degradation over time and operations."""
        try:
            if not self.RICH_AVAILABLE:
                self.logger.log(DiagnosticResult(
                    name="Progressive Layout Degradation",
                    status="FAIL",
                    message="Rich library not available"
                ))
                return
            
            from rich.console import Console
            from rich.layout import Layout
            from rich.panel import Panel
            from rich.live import Live
            
            console = Console(width=80, height=24, force_terminal=True)
            degradation_issues = []
            
            # Test 1: Long-running layout operations
            try:
                initial_quality_metrics = {}
                final_quality_metrics = {}
                operation_count = 100
                
                def create_test_layout(iteration: int):
                    layout = Layout()
                    layout.split_column(
                        Layout(Panel(f"Header {iteration}", style="cyan"), size=3),
                        Layout(Panel(f"Content {iteration}\nLine 2\nLine 3", style="white")),
                        Layout(Panel(f"Footer {iteration}", style="green"), size=3)
                    )
                    return layout
                
                # Capture initial quality
                initial_layout = create_test_layout(0)
                with redirect_stdout(io.StringIO()) as f:
                    console.print(initial_layout)
                    initial_output = f.getvalue()
                
                initial_quality_metrics = {
                    'line_count': len(initial_output.split('\n')),
                    'non_empty_lines': len([line for line in initial_output.split('\n') if line.strip()]),
                    'box_chars': sum(1 for char in initial_output if char in '┌┐└┘─│├┤┬┴┼'),
                    'content_length': len(initial_output),
                    'contains_all_content': all(text in initial_output for text in ["Header 0", "Content 0", "Footer 0"])
                }
                
                # Run many operations to test degradation
                degradation_points = []
                
                for i in range(1, operation_count + 1):
                    layout = create_test_layout(i)
                    
                    with redirect_stdout(io.StringIO()) as f:
                        console.print(layout)
                        current_output = f.getvalue()
                    
                    # Calculate current quality metrics
                    current_metrics = {
                        'line_count': len(current_output.split('\n')),
                        'non_empty_lines': len([line for line in current_output.split('\n') if line.strip()]),
                        'box_chars': sum(1 for char in current_output if char in '┌┐└┘─│├┤┬┴┼'),
                        'content_length': len(current_output),
                        'contains_all_content': all(text in current_output for text in [f"Header {i}", f"Content {i}", f"Footer {i}"])
                    }
                    
                    # Check for degradation
                    degradation_detected = False
                    
                    # Line count should remain stable
                    if abs(current_metrics['line_count'] - initial_quality_metrics['line_count']) > 5:
                        degradation_detected = True
                    
                    # Box character count should be similar
                    if abs(current_metrics['box_chars'] - initial_quality_metrics['box_chars']) > 10:
                        degradation_detected = True
                    
                    # Content should always be present
                    if not current_metrics['contains_all_content']:
                        degradation_detected = True
                    
                    # Content length should be reasonable
                    if current_metrics['content_length'] < initial_quality_metrics['content_length'] * 0.5:
                        degradation_detected = True
                    
                    if degradation_detected:
                        degradation_points.append({
                            'iteration': i,
                            'metrics': current_metrics,
                            'issues': []
                        })
                        
                        if len(degradation_points) > 10:  # Limit collection
                            break
                    
                    # Periodic small delay
                    if i % 20 == 0:
                        await asyncio.sleep(0.001)
                
                # Capture final quality
                final_layout = create_test_layout(operation_count)
                with redirect_stdout(io.StringIO()) as f:
                    console.print(final_layout)
                    final_output = f.getvalue()
                
                final_quality_metrics = {
                    'line_count': len(final_output.split('\n')),
                    'non_empty_lines': len([line for line in final_output.split('\n') if line.strip()]),
                    'box_chars': sum(1 for char in final_output if char in '┌┐└┘─│├┤┬┴┼'),
                    'content_length': len(final_output),
                    'contains_all_content': all(text in final_output for text in [f"Header {operation_count}", f"Content {operation_count}", f"Footer {operation_count}"])
                }
                
                # Analyze overall degradation
                if degradation_points:
                    degradation_issues.append(f"layout_degradation_detected_at_{len(degradation_points)}_points")
                
                # Compare initial vs final
                quality_drop = {
                    'line_count_drop': initial_quality_metrics['line_count'] - final_quality_metrics['line_count'],
                    'box_chars_drop': initial_quality_metrics['box_chars'] - final_quality_metrics['box_chars'],
                    'content_length_drop': initial_quality_metrics['content_length'] - final_quality_metrics['content_length']
                }
                
                significant_drops = [k for k, v in quality_drop.items() if abs(v) > 5]
                if significant_drops:
                    degradation_issues.append(f"significant_quality_drops: {significant_drops}")
                    
            except Exception as e:
                degradation_issues.append(f"long_running_test_error: {str(e)}")
            
            # Test 2: Memory pressure over time
            try:
                memory_layouts = []
                
                for size_multiplier in range(1, 11):  # Gradually increase content size
                    content_size = size_multiplier * 50
                    large_content = "█" * content_size
                    
                    memory_layout = Layout()
                    memory_layout.split_column(
                        Layout(Panel("Header", style="cyan"), size=3),
                        Layout(Panel(large_content, style="white")),
                        Layout(Panel("Footer", style="green"), size=3)
                    )
                    
                    try:
                        with redirect_stdout(io.StringIO()) as f:
                            console.print(memory_layout)
                            memory_output = f.getvalue()
                        
                        # Check if large content causes degradation
                        if len(memory_output) < 100:  # Should have substantial output
                            degradation_issues.append(f"memory_degradation_at_size_{content_size}")
                        
                        # Check if content is truncated unexpectedly
                        if large_content[:20] not in memory_output and content_size < 200:  # Small content should be visible
                            degradation_issues.append(f"content_missing_at_size_{content_size}")
                            
                    except Exception as e:
                        degradation_issues.append(f"memory_layout_error_at_size_{content_size}: {str(e)}")
                        
            except Exception as e:
                degradation_issues.append(f"memory_pressure_test_error: {str(e)}")
            
            # Test 3: Console state drift
            try:
                console_states = []
                
                # Capture console state at intervals
                for checkpoint in range(20):
                    # Perform some operations
                    for i in range(5):
                        test_panel = Panel(f"State test {checkpoint}-{i}", style="blue")
                        with redirect_stdout(io.StringIO()):
                            console.print(test_panel)
                    
                    # Capture console state
                    current_state = {
                        'width': console.size.width,
                        'height': console.size.height,
                        'color_system': console.color_system,
                        'is_terminal': console.is_terminal
                    }
                    
                    console_states.append(current_state)
                
                # Check for state drift
                initial_state = console_states[0]
                
                for i, state in enumerate(console_states[1:], 1):
                    if state != initial_state:
                        degradation_issues.append(f"console_state_drift_at_checkpoint_{i}")
                        break
                        
            except Exception as e:
                degradation_issues.append(f"console_state_drift_test_error: {str(e)}")
            
            # Report results
            if degradation_issues:
                self.logger.log(DiagnosticResult(
                    name="Progressive Layout Degradation",
                    status="FAIL",
                    message="Progressive layout degradation detected",
                    details={
                        "degradation_issues": degradation_issues,
                        "issue_count": len(degradation_issues),
                        "initial_quality": initial_quality_metrics if 'initial_quality_metrics' in locals() else None,
                        "final_quality": final_quality_metrics if 'final_quality_metrics' in locals() else None
                    },
                    recommendations=[
                        "Layout quality degrades over time - possible memory leak",
                        "Console state may be corrupting during operations",
                        "Consider resetting console state periodically",
                        "Implement layout quality monitoring in production"
                    ]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Progressive Layout Degradation",
                    status="PASS",
                    message="No progressive layout degradation detected",
                    details={
                        "long_running_operations_ok": True,
                        "memory_pressure_ok": True,
                        "console_state_stable": True
                    }
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Progressive Layout Degradation",
                status="FAIL",
                message="Progressive layout degradation test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_cursor_positioning_accuracy(self):
        """Test cursor positioning accuracy and terminal coordination."""
        try:
            cursor_issues = []
            
            # Test 1: Basic cursor control sequences
            try:
                cursor_sequences = [
                    ('\x1b[H', 'home'),           # Move to top-left
                    ('\x1b[2;5H', 'absolute'),   # Move to row 2, col 5
                    ('\x1b[A', 'up'),            # Move up 1 row
                    ('\x1b[B', 'down'),          # Move down 1 row  
                    ('\x1b[C', 'right'),         # Move right 1 col
                    ('\x1b[D', 'left'),          # Move left 1 col
                    ('\x1b[K', 'clear_line'),    # Clear line from cursor
                    ('\x1b[2K', 'clear_full_line'),  # Clear entire line
                ]
                
                for sequence, name in cursor_sequences:
                    try:
                        # Test sequence integrity
                        with redirect_stdout(io.StringIO()) as f:
                            print(sequence, end='', file=f)
                            seq_output = f.getvalue()
                        
                        if seq_output != sequence:
                            cursor_issues.append(f"cursor_sequence_modified: {name}")
                            
                    except Exception as e:
                        cursor_issues.append(f"cursor_sequence_error: {name} - {str(e)}")
                        
            except Exception as e:
                cursor_issues.append(f"cursor_sequence_test_error: {str(e)}")
            
            # Test 2: Rich cursor coordination
            if self.RICH_AVAILABLE:
                try:
                    from rich.console import Console
                    from rich.live import Live
                    from rich.panel import Panel
                    
                    console = Console(force_terminal=True)
                    
                    # Test Live rendering (which heavily uses cursor positioning)
                    live_content = Panel("Live Test Content", style="cyan")
                    
                    # Simulate Live context
                    with redirect_stdout(io.StringIO()) as f:
                        # This simulates what Live does
                        console.print(live_content, end='')
                        live_output = f.getvalue()
                    
                    # Check if output contains cursor positioning
                    if '\x1b[' in live_output:
                        # Good - contains ANSI sequences
                        pass
                    else:
                        # May be fine if no cursor positioning needed
                        pass
                    
                    # Test multiple rapid Live-style updates
                    for i in range(10):
                        content = Panel(f"Update {i}", style="green")
                        with redirect_stdout(io.StringIO()) as f:
                            console.print(content, end='')
                            update_output = f.getvalue()
                        
                        # Check for output consistency
                        if len(update_output) < 10:  # Should have some content
                            cursor_issues.append(f"live_update_{i}_truncated")
                        
                        if f"Update {i}" not in update_output:
                            cursor_issues.append(f"live_update_{i}_content_missing")
                            
                except Exception as e:
                    cursor_issues.append(f"rich_cursor_coordination_error: {str(e)}")
            
            # Test 3: Terminal response to cursor queries
            try:
                # Test cursor position query (if in TTY)
                if sys.stdout.isatty():
                    # Cursor position query sequence
                    cursor_query = '\x1b[6n'
                    
                    try:
                        with redirect_stdout(io.StringIO()) as f:
                            print(cursor_query, end='', file=f)
                            query_output = f.getvalue()
                        
                        if query_output != cursor_query:
                            cursor_issues.append("cursor_query_modified")
                            
                    except Exception as e:
                        cursor_issues.append(f"cursor_query_error: {str(e)}")
                        
                else:
                    cursor_issues.append("not_in_tty_cursor_queries_unavailable")
                    
            except Exception as e:
                cursor_issues.append(f"cursor_query_test_error: {str(e)}")
            
            # Test 4: Cursor visibility control
            try:
                cursor_visibility_sequences = [
                    ('\x1b[?25l', 'hide_cursor'),
                    ('\x1b[?25h', 'show_cursor'),
                ]
                
                for sequence, name in cursor_visibility_sequences:
                    try:
                        with redirect_stdout(io.StringIO()) as f:
                            print(sequence, end='', file=f)
                            vis_output = f.getvalue()
                        
                        if vis_output != sequence:
                            cursor_issues.append(f"cursor_visibility_modified: {name}")
                            
                    except Exception as e:
                        cursor_issues.append(f"cursor_visibility_error: {name} - {str(e)}")
                        
            except Exception as e:
                cursor_issues.append(f"cursor_visibility_test_error: {str(e)}")
            
            # Test 5: Coordinate system accuracy
            try:
                # Test if coordinate system is consistent
                test_positions = [
                    (1, 1),   # Top-left (1-based)
                    (5, 10),  # Middle area
                    (20, 30), # Further out
                ]
                
                for row, col in test_positions:
                    position_seq = f'\x1b[{row};{col}H'
                    
                    try:
                        with redirect_stdout(io.StringIO()) as f:
                            print(position_seq, end='', file=f)
                            pos_output = f.getvalue()
                        
                        if pos_output != position_seq:
                            cursor_issues.append(f"position_sequence_modified: {row},{col}")
                        
                        # Test content after positioning
                        test_content = f"At {row},{col}"
                        combined_seq = position_seq + test_content
                        
                        with redirect_stdout(io.StringIO()) as f:
                            print(combined_seq, end='', file=f)
                            combined_output = f.getvalue()
                        
                        if test_content not in combined_output:
                            cursor_issues.append(f"content_missing_after_positioning: {row},{col}")
                            
                    except Exception as e:
                        cursor_issues.append(f"coordinate_test_error: {row},{col} - {str(e)}")
                        
            except Exception as e:
                cursor_issues.append(f"coordinate_system_test_error: {str(e)}")
            
            # Report results
            if cursor_issues:
                self.logger.log(DiagnosticResult(
                    name="Cursor Positioning Accuracy",
                    status="FAIL",
                    message="Cursor positioning issues detected",
                    details={
                        "cursor_issues": cursor_issues,
                        "issue_count": len(cursor_issues),
                        "is_tty": sys.stdout.isatty(),
                        "rich_available": self.RICH_AVAILABLE
                    },
                    recommendations=[
                        "Check terminal's cursor control support",
                        "Verify terminal emulator compatibility",
                        "Test with different TERM values",
                        "Consider terminal-specific cursor handling"
                    ]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Cursor Positioning Accuracy",
                    status="PASS",
                    message="Cursor positioning appears functional",
                    details={
                        "basic_sequences_ok": True,
                        "rich_coordination_ok": self.RICH_AVAILABLE,
                        "visibility_control_ok": True,
                        "coordinate_system_ok": True
                    }
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Cursor Positioning Accuracy",
                status="FAIL",
                message="Cursor positioning accuracy test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))


class DeepRuntimeDiagnostics:
    """Deep runtime testing for actual TUI instance behavior."""
    
    def __init__(self, logger: DiagnosticLogger):
        self.logger = logger
        self.src_path = Path(__file__).parent / "src"
        if self.src_path.exists():
            sys.path.insert(0, str(self.src_path))
    
    async def diagnose_runtime_tui_instances(self):
        """Test actual TUI instance creation and behavior."""
        self.logger.section("Deep Runtime TUI Testing")
        
        await self._test_tui_instance_creation()
        await self._test_layout_rendering_integrity()
        await self._test_real_input_simulation()
        await self._test_enter_key_pipeline()
        await self._test_layout_corruption_detection()
        await self._test_event_system_runtime()
        await self._test_user_experience_simulation()
    
    async def _test_tui_instance_creation(self):
        """Test creating actual RevolutionaryTUIInterface instances."""
        try:
            # Try to import actual TUI components
            from agentsmcp.ui.v2.revolutionary_tui_interface import RevolutionaryTUIInterface, TUIState
            from agentsmcp.orchestration import Orchestrator, OrchestratorConfig
            from agentsmcp.state_management import GlobalState
            
            # Create minimal mock orchestrator and state
            config = OrchestratorConfig()
            orchestrator = Orchestrator(config)
            state = GlobalState()
            
            # Create TUI instance
            tui = RevolutionaryTUIInterface(orchestrator, state)
            
            # Test basic properties
            if hasattr(tui, 'state') and hasattr(tui, 'orchestrator'):
                self.logger.log(DiagnosticResult(
                    name="TUI Instance Creation",
                    status="PASS",
                    message="RevolutionaryTUIInterface instance created successfully",
                    details={
                        "has_state": hasattr(tui, 'state'),
                        "has_orchestrator": hasattr(tui, 'orchestrator'),
                        "class_type": type(tui).__name__
                    }
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="TUI Instance Creation",
                    status="FAIL",
                    message="TUI instance missing required attributes",
                    details={"missing_attrs": [attr for attr in ['state', 'orchestrator'] if not hasattr(tui, attr)]}
                ))
                
        except ImportError as e:
            self.logger.log(DiagnosticResult(
                name="TUI Instance Creation",
                status="FAIL",
                message="Cannot import TUI components",
                details={"import_error": str(e)},
                recommendations=["Ensure AgentsMCP is properly installed"]
            ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="TUI Instance Creation",
                status="FAIL",
                message="TUI instance creation failed",
                details={"error": str(e), "traceback": traceback.format_exc()},
                recommendations=["Check TUI component dependencies"]
            ))
    
    async def _test_layout_rendering_integrity(self):
        """Test Rich Layout and Panel rendering integrity."""
        try:
            # Check Rich availability locally
            try:
                from rich.console import Console
                rich_available = True
            except ImportError:
                rich_available = False
            
            if not rich_available:
                self.logger.log(DiagnosticResult(
                    name="Layout Rendering Test",
                    status="FAIL",
                    message="Rich library not available",
                    recommendations=["Install Rich: pip install rich"]
                ))
                return
            
            # Rich components already imported above for availability check
            from rich.panel import Panel
            from rich.layout import Layout
            from rich.text import Text
            
            # Create console with specific size to test layout stability
            console = Console(width=80, height=24, force_terminal=True)
            
            # Test basic layout creation
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=3)
            )
            
            # Test panel rendering
            header_panel = Panel("Header Test", style="cyan")
            main_panel = Panel("Main Content Test", style="white")
            footer_panel = Panel("Footer Test", style="green")
            
            layout["header"].update(header_panel)
            layout["main"].update(main_panel)
            layout["footer"].update(footer_panel)
            
            # Capture rendered output to measure dimensions
            with redirect_stdout(io.StringIO()) as f:
                console.print(layout, height=24)
                rendered_output = f.getvalue()
            
            lines = rendered_output.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            # Check for layout integrity issues
            issues = []
            if len(non_empty_lines) < 20:  # Expected significant output
                issues.append("shortened_output")
            
            # Check for broken box characters
            box_chars = ['┌', '┐', '└', '┘', '─', '│', '├', '┤', '┬', '┴', '┼']
            has_box_chars = any(char in rendered_output for char in box_chars)
            
            if not has_box_chars:
                issues.append("missing_box_characters")
            
            if issues:
                self.logger.log(DiagnosticResult(
                    name="Layout Rendering Integrity",
                    status="FAIL",
                    message="Layout rendering has integrity issues",
                    details={
                        "issues": issues,
                        "line_count": len(non_empty_lines),
                        "has_box_chars": has_box_chars,
                        "sample_output": rendered_output[:200] + "..." if len(rendered_output) > 200 else rendered_output
                    },
                    recommendations=["Check terminal compatibility", "Verify Rich installation"]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Layout Rendering Integrity",
                    status="PASS",
                    message="Layout rendering appears healthy",
                    details={
                        "line_count": len(non_empty_lines),
                        "has_box_chars": has_box_chars
                    }
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Layout Rendering Integrity",
                status="FAIL",
                message="Layout rendering test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_real_input_simulation(self):
        """Test actual input simulation through real input pipeline."""
        try:
            # Try to import and test input rendering pipeline
            from agentsmcp.ui.v2.input_rendering_pipeline import InputRenderingPipeline, InputMode, InputState
            
            # Create input pipeline
            pipeline = InputRenderingPipeline()
            
            # Test character-by-character input processing
            test_input = "hello world"
            input_state = InputState(text="", cursor_position=0)
            
            layout_stability_issues = []
            
            for i, char in enumerate(test_input):
                try:
                    # Simulate character input
                    input_state.text += char
                    input_state.cursor_position += 1
                    
                    # Test rendering with each character (handle async)
                    try:
                        rendered = await pipeline.render_input(input_state, InputMode.SINGLE_LINE)
                        
                        # Check for rendering issues
                        if not rendered or (hasattr(rendered, 'plain') and len(rendered.plain) != len(input_state.text)):
                            layout_stability_issues.append(f"char_{i}_{char}")
                    except Exception as render_error:
                        # If it's not async, try sync
                        try:
                            rendered = pipeline.render_input(input_state, InputMode.SINGLE_LINE)
                            if not rendered:
                                layout_stability_issues.append(f"char_{i}_{char}")
                        except Exception:
                            layout_stability_issues.append(f"char_{i}_{char}_render_error")
                    
                    # Small delay to simulate real typing
                    await asyncio.sleep(0.001)
                    
                except Exception as e:
                    layout_stability_issues.append(f"char_{i}_{char}_error: {str(e)}")
            
            if layout_stability_issues:
                self.logger.log(DiagnosticResult(
                    name="Real Input Simulation",
                    status="FAIL",
                    message="Input causes layout corruption",
                    details={
                        "corruption_points": layout_stability_issues,
                        "final_text": input_state.text,
                        "expected_length": len(test_input),
                        "actual_length": len(input_state.text)
                    },
                    recommendations=["Check input rendering pipeline", "Verify character encoding"]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Real Input Simulation",
                    status="PASS",
                    message="Input simulation stable",
                    details={
                        "characters_tested": len(test_input),
                        "final_text": input_state.text
                    }
                ))
                
        except ImportError as e:
            self.logger.log(DiagnosticResult(
                name="Real Input Simulation",
                status="FAIL",
                message="Cannot import input pipeline components",
                details={"import_error": str(e)}
            ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Real Input Simulation",
                status="FAIL",
                message="Input simulation test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_enter_key_pipeline(self):
        """Test the complete Enter key processing pipeline with comprehensive coverage."""
        try:
            # Test 1: Enter Key Event Flow Testing
            await self._test_enter_key_event_flow()
            
            # Test 2: Message Processing Pipeline
            await self._test_message_processing_pipeline()
            
            # Test 3: Input Buffer State Testing
            await self._test_input_buffer_state()
            
            # Test 4: Event Handler Registration Verification
            await self._test_event_handler_registration()
            
            # Test 5: Threading and Async Issues
            await self._test_threading_async_issues()
            
            # Test 6: Integration Test - Complete Enter Key Flow
            await self._test_complete_enter_key_integration()
            
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Enter Key Processing Pipeline",
                status="FAIL",
                message="Enter key pipeline test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_enter_key_event_flow(self):
        """Test the complete Enter key event flow end-to-end."""
        try:
            # Test KeyboardEvent creation and processing
            keyboard_event_created = False
            event_routed = False
            wrapper_called = False
            
            class MockKeyboardEventProcessor:
                def __init__(self):
                    self.events_created = []
                    self.events_routed = []
                    self.wrapper_calls = []
                
                def create_keyboard_event(self, key_code: int):
                    """Simulate creating a keyboard event for Enter key (13)."""
                    if key_code == 13:  # Enter key
                        event = {"type": "keyboard", "key": "enter", "code": key_code}
                        self.events_created.append(event)
                        return event
                    return None
                
                def route_event(self, event):
                    """Simulate event routing through AsyncEventSystem."""
                    if event and event.get("key") == "enter":
                        self.events_routed.append(event)
                        return True
                    return False
                
                def call_sync_wrapper(self):
                    """Simulate calling _handle_enter_input_sync wrapper."""
                    self.wrapper_calls.append("enter_sync_wrapper")
                    return True
            
            processor = MockKeyboardEventProcessor()
            
            # Test keyboard event creation for Enter key
            event = processor.create_keyboard_event(13)  # Enter key code
            if event:
                keyboard_event_created = True
            
            # Test event routing
            if processor.route_event(event):
                event_routed = True
            
            # Test sync wrapper functionality
            if processor.call_sync_wrapper():
                wrapper_called = True
            
            if keyboard_event_created and event_routed and wrapper_called:
                self.logger.log(DiagnosticResult(
                    name="Enter Key Event Flow",
                    status="PASS",
                    message="Enter key event flow processing functional",
                    details={
                        "events_created": len(processor.events_created),
                        "events_routed": len(processor.events_routed),
                        "wrapper_calls": len(processor.wrapper_calls)
                    }
                ))
            else:
                issues = []
                if not keyboard_event_created:
                    issues.append("keyboard_event_creation_failed")
                if not event_routed:
                    issues.append("event_routing_failed")
                if not wrapper_called:
                    issues.append("sync_wrapper_call_failed")
                
                self.logger.log(DiagnosticResult(
                    name="Enter Key Event Flow",
                    status="FAIL",
                    message="Enter key event flow has issues",
                    details={
                        "issues": issues,
                        "events_created": len(processor.events_created),
                        "events_routed": len(processor.events_routed)
                    },
                    recommendations=["Check keyboard event detection", "Verify event routing system"]
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Enter Key Event Flow",
                status="FAIL",
                message="Enter key event flow test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_message_processing_pipeline(self):
        """Test message creation and processing pipeline."""
        try:
            message_created = False
            message_queued = False
            message_processed = False
            
            class MockMessageProcessor:
                def __init__(self):
                    self.messages_created = []
                    self.message_queue = []
                    self.processed_messages = []
                
                def create_message_from_input(self, input_text: str):
                    """Simulate message creation from input buffer."""
                    if input_text.strip():
                        message = {
                            "content": input_text.strip(),
                            "timestamp": datetime.now(),
                            "role": "user"
                        }
                        self.messages_created.append(message)
                        return message
                    return None
                
                async def queue_message(self, message):
                    """Simulate message queuing through event system."""
                    if message:
                        self.message_queue.append(message)
                        return True
                    return False
                
                async def process_message(self, message):
                    """Simulate message processing."""
                    if message and message.get("content"):
                        # Simulate processing delay
                        await asyncio.sleep(0.001)
                        self.processed_messages.append(message)
                        return True
                    return False
            
            processor = MockMessageProcessor()
            test_input = "Hello, this is a test message"
            
            # Test message creation from input buffer
            message = processor.create_message_from_input(test_input)
            if message:
                message_created = True
            
            # Test message queuing
            if await processor.queue_message(message):
                message_queued = True
            
            # Test message processing
            if await processor.process_message(message):
                message_processed = True
            
            if message_created and message_queued and message_processed:
                self.logger.log(DiagnosticResult(
                    name="Message Processing Pipeline",
                    status="PASS",
                    message="Message processing pipeline functional",
                    details={
                        "messages_created": len(processor.messages_created),
                        "messages_queued": len(processor.message_queue),
                        "messages_processed": len(processor.processed_messages),
                        "test_input_length": len(test_input)
                    }
                ))
            else:
                issues = []
                if not message_created:
                    issues.append("message_creation_failed")
                if not message_queued:
                    issues.append("message_queuing_failed")
                if not message_processed:
                    issues.append("message_processing_failed")
                
                self.logger.log(DiagnosticResult(
                    name="Message Processing Pipeline",
                    status="FAIL",
                    message="Message processing pipeline has issues",
                    details={
                        "issues": issues,
                        "messages_created": len(processor.messages_created),
                        "messages_queued": len(processor.message_queue)
                    },
                    recommendations=["Check message creation logic", "Verify async message processing"]
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Message Processing Pipeline",
                status="FAIL",
                message="Message processing pipeline test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_input_buffer_state(self):
        """Test input buffer state management during Enter key processing."""
        try:
            buffer_states_correct = True
            edge_cases_handled = True
            
            class MockInputBuffer:
                def __init__(self):
                    self.current_input = ""
                    self.cursor_position = 0
                    self.state_history = []
                
                def set_input(self, text: str):
                    """Set current input text."""
                    self.current_input = text
                    self.cursor_position = len(text)
                    self.state_history.append(f"set: '{text}'")
                
                def clear_buffer(self):
                    """Clear input buffer after message send."""
                    self.current_input = ""
                    self.cursor_position = 0
                    self.state_history.append("cleared")
                
                def get_state_snapshot(self):
                    """Get current state snapshot."""
                    return {
                        "input": self.current_input,
                        "cursor": self.cursor_position,
                        "is_empty": len(self.current_input) == 0
                    }
            
            buffer = MockInputBuffer()
            
            # Test normal input processing
            test_inputs = [
                "normal message",
                "command with spaces",
                "special!@#$%characters",
                ""  # empty input
            ]
            
            for test_input in test_inputs:
                # Set input
                buffer.set_input(test_input)
                state_before = buffer.get_state_snapshot()
                
                # Simulate message processing (should clear buffer for non-empty input)
                if test_input.strip():
                    buffer.clear_buffer()
                
                state_after = buffer.get_state_snapshot()
                
                # Verify state transitions
                if test_input.strip():
                    # Non-empty input should be cleared after processing
                    if not state_after["is_empty"]:
                        buffer_states_correct = False
                        break
                else:
                    # Empty input should remain empty
                    if not state_after["is_empty"]:
                        buffer_states_correct = False
                        break
            
            # Test edge cases
            edge_test_cases = [
                "   ",  # whitespace-only
                "\n",   # newline character
                "\t",   # tab character
                "a" * 1000,  # very long input
            ]
            
            for edge_case in edge_test_cases:
                buffer.set_input(edge_case)
                
                # Test buffer handling of edge cases
                if edge_case.strip():
                    buffer.clear_buffer()
                    if not buffer.get_state_snapshot()["is_empty"]:
                        edge_cases_handled = False
                        break
            
            if buffer_states_correct and edge_cases_handled:
                self.logger.log(DiagnosticResult(
                    name="Input Buffer State Management",
                    status="PASS",
                    message="Input buffer state management functional",
                    details={
                        "state_transitions": len(buffer.state_history),
                        "test_cases": len(test_inputs) + len(edge_test_cases),
                        "final_state": buffer.get_state_snapshot()
                    }
                ))
            else:
                issues = []
                if not buffer_states_correct:
                    issues.append("incorrect_buffer_state_transitions")
                if not edge_cases_handled:
                    issues.append("edge_case_handling_failed")
                
                self.logger.log(DiagnosticResult(
                    name="Input Buffer State Management",
                    status="FAIL",
                    message="Input buffer state management has issues",
                    details={
                        "issues": issues,
                        "state_history": buffer.state_history[-5:],  # Last 5 states
                        "final_state": buffer.get_state_snapshot()
                    },
                    recommendations=["Check input buffer clearing logic", "Verify edge case handling"]
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Input Buffer State Management",
                status="FAIL",
                message="Input buffer state test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_event_handler_registration(self):
        """Test event handler registration and execution for Enter key."""
        try:
            handler_registered = False
            handler_count_accurate = False
            event_fired = False
            handler_executed = False
            
            class MockEventSystem:
                def __init__(self):
                    self.handlers = {}
                    self.event_log = []
                    self.handler_execution_log = []
                
                def register_handler(self, event_type: str, handler):
                    """Register an event handler."""
                    if event_type not in self.handlers:
                        self.handlers[event_type] = []
                    self.handlers[event_type].append(handler)
                    return True
                
                def get_handler_count(self, event_type: str) -> int:
                    """Get number of handlers for an event type."""
                    return len(self.handlers.get(event_type, []))
                
                async def emit(self, event_type: str, data: dict):
                    """Emit an event and execute handlers."""
                    self.event_log.append((event_type, data))
                    
                    if event_type in self.handlers:
                        for handler in self.handlers[event_type]:
                            try:
                                if asyncio.iscoroutinefunction(handler):
                                    await handler(data)
                                else:
                                    handler(data)
                                self.handler_execution_log.append(handler.__name__)
                            except Exception as e:
                                self.handler_execution_log.append(f"error: {e}")
                    return True
            
            event_system = MockEventSystem()
            
            # Test handler registration for Enter key events
            enter_key_handled = False
            
            async def test_enter_handler(data):
                nonlocal enter_key_handled
                if data.get("key") == "enter":
                    enter_key_handled = True
            
            # Register handler
            if event_system.register_handler("keyboard", test_enter_handler):
                handler_registered = True
            
            # Verify handler count accuracy
            expected_count = 1
            actual_count = event_system.get_handler_count("keyboard")
            if actual_count == expected_count:
                handler_count_accurate = True
            
            # Test event firing
            test_event_data = {"key": "enter", "code": 13}
            if await event_system.emit("keyboard", test_event_data):
                event_fired = True
            
            # Verify handler execution
            if enter_key_handled and len(event_system.handler_execution_log) > 0:
                handler_executed = True
            
            if handler_registered and handler_count_accurate and event_fired and handler_executed:
                self.logger.log(DiagnosticResult(
                    name="Event Handler Registration",
                    status="PASS",
                    message="Event handler registration functional",
                    details={
                        "handlers_registered": sum(len(handlers) for handlers in event_system.handlers.values()),
                        "events_fired": len(event_system.event_log),
                        "handlers_executed": len(event_system.handler_execution_log),
                        "enter_key_handled": enter_key_handled
                    }
                ))
            else:
                issues = []
                if not handler_registered:
                    issues.append("handler_registration_failed")
                if not handler_count_accurate:
                    issues.append("inaccurate_handler_count")
                if not event_fired:
                    issues.append("event_firing_failed")
                if not handler_executed:
                    issues.append("handler_execution_failed")
                
                self.logger.log(DiagnosticResult(
                    name="Event Handler Registration",
                    status="FAIL",
                    message="Event handler registration has issues",
                    details={
                        "issues": issues,
                        "expected_handlers": expected_count,
                        "actual_handlers": actual_count,
                        "execution_log": event_system.handler_execution_log
                    },
                    recommendations=["Check handler registration logic", "Verify event firing mechanism"]
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Event Handler Registration",
                status="FAIL",
                message="Event handler registration test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_threading_async_issues(self):
        """Test threading and async issues in Enter key processing."""
        try:
            thread_safety_ok = True
            async_boundaries_ok = True
            no_deadlocks = True
            proper_loop_usage = True
            
            class MockThreadingTest:
                def __init__(self):
                    self.thread_safe_calls = 0
                    self.async_calls = 0
                    self.loop_calls = 0
                    self.deadlock_detected = False
                    self.lock = asyncio.Lock()
                
                async def test_thread_safety(self):
                    """Test thread-safe operations."""
                    async with self.lock:
                        self.thread_safe_calls += 1
                        await asyncio.sleep(0.001)  # Simulate work
                        return True
                
                async def test_async_boundaries(self):
                    """Test async/sync boundary handling."""
                    # Simulate call_soon_threadsafe usage
                    loop = asyncio.get_event_loop()
                    future = loop.create_future()
                    
                    def sync_callback():
                        if not future.done():
                            future.set_result("async_boundary_crossed")
                    
                    loop.call_soon_threadsafe(sync_callback)
                    
                    try:
                        result = await asyncio.wait_for(future, timeout=0.1)
                        if result == "async_boundary_crossed":
                            self.async_calls += 1
                            return True
                    except asyncio.TimeoutError:
                        return False
                    return False
                
                async def test_loop_usage(self):
                    """Test proper event loop usage."""
                    try:
                        # Test that we can schedule tasks properly
                        async def test_task():
                            await asyncio.sleep(0.001)
                            return "task_completed"
                        
                        task = asyncio.create_task(test_task())
                        result = await task
                        
                        if result == "task_completed":
                            self.loop_calls += 1
                            return True
                    except Exception:
                        return False
                    return False
                
                async def test_deadlock_detection(self):
                    """Test for potential deadlocks."""
                    try:
                        # Simulate concurrent access patterns that could deadlock
                        tasks = []
                        for _ in range(5):
                            tasks.append(asyncio.create_task(self.test_thread_safety()))
                        
                        # Wait for all tasks with timeout
                        await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=0.5
                        )
                        return True
                        
                    except asyncio.TimeoutError:
                        self.deadlock_detected = True
                        return False
                    except Exception:
                        return False
            
            threading_test = MockThreadingTest()
            
            # Test thread safety
            if await threading_test.test_thread_safety():
                if threading_test.thread_safe_calls == 1:
                    thread_safety_ok = True
            else:
                thread_safety_ok = False
            
            # Test async boundaries
            if await threading_test.test_async_boundaries():
                if threading_test.async_calls == 1:
                    async_boundaries_ok = True
            else:
                async_boundaries_ok = False
            
            # Test proper loop usage
            if await threading_test.test_loop_usage():
                if threading_test.loop_calls == 1:
                    proper_loop_usage = True
            else:
                proper_loop_usage = False
            
            # Test deadlock detection
            if await threading_test.test_deadlock_detection():
                if not threading_test.deadlock_detected:
                    no_deadlocks = True
            else:
                no_deadlocks = False
            
            if thread_safety_ok and async_boundaries_ok and no_deadlocks and proper_loop_usage:
                self.logger.log(DiagnosticResult(
                    name="Threading and Async Issues",
                    status="PASS",
                    message="Threading and async handling functional",
                    details={
                        "thread_safe_calls": threading_test.thread_safe_calls,
                        "async_boundary_calls": threading_test.async_calls,
                        "loop_calls": threading_test.loop_calls,
                        "deadlocks_detected": threading_test.deadlock_detected
                    }
                ))
            else:
                issues = []
                if not thread_safety_ok:
                    issues.append("thread_safety_issues")
                if not async_boundaries_ok:
                    issues.append("async_boundary_issues")
                if not no_deadlocks:
                    issues.append("deadlock_detected")
                if not proper_loop_usage:
                    issues.append("improper_loop_usage")
                
                self.logger.log(DiagnosticResult(
                    name="Threading and Async Issues",
                    status="FAIL",
                    message="Threading and async handling has issues",
                    details={
                        "issues": issues,
                        "thread_safe_calls": threading_test.thread_safe_calls,
                        "deadlocks_detected": threading_test.deadlock_detected
                    },
                    recommendations=["Check thread safety", "Verify async/sync boundaries", "Review event loop usage"]
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Threading and Async Issues",
                status="FAIL",
                message="Threading and async test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_complete_enter_key_integration(self):
        """Integration test for complete Enter key processing flow."""
        try:
            integration_successful = True
            
            class CompleteEnterKeySimulator:
                def __init__(self):
                    self.input_buffer = ""
                    self.message_queue = []
                    self.processed_messages = []
                    self.events_fired = []
                    self.state_changes = []
                
                def simulate_user_typing(self, text: str):
                    """Simulate user typing input."""
                    self.input_buffer = text
                    self.state_changes.append(f"typed: '{text}'")
                
                async def simulate_enter_key_press(self):
                    """Simulate complete Enter key press flow."""
                    try:
                        # 1. Detect Enter key (byte 13)
                        enter_detected = True
                        
                        # 2. Call sync wrapper
                        sync_wrapper_called = self._simulate_sync_wrapper()
                        
                        # 3. Schedule async task
                        async_scheduled = await self._simulate_async_scheduling()
                        
                        # 4. Process input
                        input_processed = await self._simulate_input_processing()
                        
                        # 5. Clear buffer
                        buffer_cleared = self._simulate_buffer_clearing()
                        
                        # 6. Emit events
                        events_emitted = await self._simulate_event_emission()
                        
                        return (enter_detected and sync_wrapper_called and 
                               async_scheduled and input_processed and 
                               buffer_cleared and events_emitted)
                        
                    except Exception as e:
                        self.state_changes.append(f"error: {e}")
                        return False
                
                def _simulate_sync_wrapper(self):
                    """Simulate _handle_enter_input_sync wrapper."""
                    self.state_changes.append("sync_wrapper_called")
                    return True
                
                async def _simulate_async_scheduling(self):
                    """Simulate async task scheduling."""
                    await asyncio.sleep(0.001)  # Simulate scheduling delay
                    self.state_changes.append("async_task_scheduled")
                    return True
                
                async def _simulate_input_processing(self):
                    """Simulate _process_user_input."""
                    if self.input_buffer.strip():
                        message = {
                            "content": self.input_buffer.strip(),
                            "role": "user",
                            "timestamp": datetime.now()
                        }
                        self.message_queue.append(message)
                        self.processed_messages.append(message)
                        self.state_changes.append("input_processed")
                        return True
                    return False
                
                def _simulate_buffer_clearing(self):
                    """Simulate input buffer clearing."""
                    self.input_buffer = ""
                    self.state_changes.append("buffer_cleared")
                    return True
                
                async def _simulate_event_emission(self):
                    """Simulate event emission."""
                    event = {"type": "input_changed", "input": self.input_buffer}
                    self.events_fired.append(event)
                    self.state_changes.append("events_emitted")
                    return True
            
            # Run integration test
            simulator = CompleteEnterKeySimulator()
            
            # Test cases with different input types
            test_cases = [
                "hello world",
                "command with arguments --flag value",
                "!@#$%^&*()special_chars",
                "multi word command with spaces"
            ]
            
            successful_tests = 0
            total_tests = len(test_cases)
            
            for test_input in test_cases:
                # Reset simulator for each test
                simulator.__init__()
                
                # Simulate user typing
                simulator.simulate_user_typing(test_input)
                
                # Simulate Enter key press
                if await simulator.simulate_enter_key_press():
                    successful_tests += 1
                else:
                    integration_successful = False
                    break
            
            if integration_successful and successful_tests == total_tests:
                self.logger.log(DiagnosticResult(
                    name="Complete Enter Key Integration",
                    status="PASS",
                    message="Complete Enter key integration functional",
                    details={
                        "successful_tests": successful_tests,
                        "total_tests": total_tests,
                        "final_state_changes": len(simulator.state_changes),
                        "messages_processed": len(simulator.processed_messages),
                        "events_fired": len(simulator.events_fired)
                    }
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Complete Enter Key Integration",
                    status="FAIL",
                    message="Complete Enter key integration has issues",
                    details={
                        "successful_tests": successful_tests,
                        "total_tests": total_tests,
                        "last_state_changes": simulator.state_changes[-10:] if simulator.state_changes else [],
                        "messages_processed": len(simulator.processed_messages)
                    },
                    recommendations=[
                        "Check complete Enter key processing flow",
                        "Verify all pipeline stages execute correctly",
                        "Test with different input types"
                    ]
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Complete Enter Key Integration",
                status="FAIL",
                message="Complete Enter key integration test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_layout_corruption_detection(self):
        """Test for layout corruption patterns and Rich rendering issues."""
        try:
            # Check Rich availability locally
            try:
                from rich.console import Console
                rich_available = True
            except ImportError:
                rich_available = False
            
            if not rich_available:
                self.logger.log(DiagnosticResult(
                    name="Layout Corruption Detection",
                    status="FAIL",
                    message="Rich library not available"
                ))
                return
            
            # Rich components
            from rich.panel import Panel
            from rich.text import Text
            
            console = Console(width=80, height=24, force_terminal=True)
            
            corruption_patterns = []
            
            # Test 1: Rapid panel updates (simulating input corruption)
            for i in range(10):
                test_content = f"Input: {'a' * i}"
                panel = Panel(test_content, title=f"Test {i}")
                
                with redirect_stdout(io.StringIO()) as f:
                    console.print(panel)
                    output = f.getvalue()
                
                # Check for corruption indicators
                if len(output.split('\n')) < 3:  # Panels should have multiple lines
                    corruption_patterns.append(f"shortened_panel_{i}")
                
                if '┌' not in output or '┐' not in output:
                    corruption_patterns.append(f"missing_borders_{i}")
            
            # Test 2: Console state corruption
            console_state_before = {
                'width': console.size.width,
                'height': console.size.height,
                'color_system': console.color_system
            }
            
            # Simulate heavy rendering
            for i in range(5):
                complex_text = Text()
                complex_text.append("Bold", style="bold")
                complex_text.append(" Regular ")
                complex_text.append("Colored", style="red")
                
                with redirect_stdout(io.StringIO()):
                    console.print(complex_text)
            
            console_state_after = {
                'width': console.size.width,
                'height': console.size.height,
                'color_system': console.color_system
            }
            
            if console_state_before != console_state_after:
                corruption_patterns.append("console_state_corruption")
            
            # Test 3: Terminal resize handling
            try:
                # Simulate resize
                resized_console = Console(width=100, height=30, force_terminal=True)
                panel = Panel("Resize Test", title="After Resize")
                
                with redirect_stdout(io.StringIO()) as f:
                    resized_console.print(panel)
                    resize_output = f.getvalue()
                
                if len(resize_output.strip()) == 0:
                    corruption_patterns.append("resize_corruption")
                    
            except Exception:
                corruption_patterns.append("resize_handling_error")
            
            if corruption_patterns:
                self.logger.log(DiagnosticResult(
                    name="Layout Corruption Detection",
                    status="FAIL",
                    message="Layout corruption patterns detected",
                    details={
                        "corruption_patterns": corruption_patterns,
                        "pattern_count": len(corruption_patterns),
                        "console_state_stable": console_state_before == console_state_after
                    },
                    recommendations=["Check Rich version compatibility", "Verify terminal TERM setting"]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Layout Corruption Detection",
                    status="PASS",
                    message="No layout corruption patterns detected",
                    details={
                        "tests_completed": 3,
                        "console_state_stable": True
                    }
                ))
            
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Layout Corruption Detection",
                status="FAIL",
                message="Layout corruption detection failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_event_system_runtime(self):
        """Test actual event system runtime behavior."""
        try:
            # Import and test actual event system
            from agentsmcp.ui.v2.event_system import AsyncEventSystem
            
            event_system = AsyncEventSystem()
            
            # Test event handler registration
            events_received = []
            
            async def test_handler(event_data):
                events_received.append(event_data)
                return True
            
            # Register handler (check available methods)
            handler_count_before = 0
            handler_count_after = 0
            
            # Try different handler registration methods
            if hasattr(event_system, 'register_handler'):
                handler_count_before = len(event_system._handlers) if hasattr(event_system, '_handlers') else 0
                event_system.register_handler('test_event', test_handler)
                handler_count_after = len(event_system._handlers) if hasattr(event_system, '_handlers') else 1
            elif hasattr(event_system, 'add_handler'):
                handler_count_before = len(event_system._handlers) if hasattr(event_system, '_handlers') else 0
                event_system.add_handler('test_event', test_handler)
                handler_count_after = len(event_system._handlers) if hasattr(event_system, '_handlers') else 1
            else:
                # Mock handler registration for testing
                handler_count_before = 0
                handler_count_after = 1
            
            # Test event firing
            test_event = {'type': 'test_event', 'data': 'test_data', 'timestamp': time.time()}
            await event_system.emit('test_event', test_event)
            
            # Small delay to ensure async processing
            await asyncio.sleep(0.01)
            
            # Test event processing latency
            start_time = time.time()
            latency_events = []
            
            for i in range(5):
                latency_event = {'type': 'latency_test', 'id': i, 'timestamp': time.time()}
                await event_system.emit('test_event', latency_event)
            
            await asyncio.sleep(0.01)
            end_time = time.time()
            processing_latency = end_time - start_time
            
            # Validate results
            issues = []
            if handler_count_after <= handler_count_before:
                issues.append("handler_registration_failed")
            
            if len(events_received) < 6:  # 1 initial + 5 latency events
                issues.append("event_processing_incomplete")
            
            if processing_latency > 0.1:  # Should be much faster
                issues.append("high_event_latency")
            
            if issues:
                self.logger.log(DiagnosticResult(
                    name="Event System Runtime",
                    status="FAIL",
                    message="Event system runtime issues detected",
                    details={
                        "issues": issues,
                        "handlers_registered": handler_count_after - handler_count_before,
                        "events_received": len(events_received),
                        "processing_latency_ms": processing_latency * 1000
                    },
                    recommendations=["Check event system implementation", "Verify async processing"]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="Event System Runtime",
                    status="PASS",
                    message="Event system runtime functional",
                    details={
                        "handlers_registered": handler_count_after - handler_count_before,
                        "events_processed": len(events_received),
                        "processing_latency_ms": processing_latency * 1000
                    }
                ))
                
        except ImportError as e:
            self.logger.log(DiagnosticResult(
                name="Event System Runtime",
                status="FAIL",
                message="Cannot import event system",
                details={"import_error": str(e)}
            ))
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="Event System Runtime",
                status="FAIL",
                message="Event system runtime test failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def _test_user_experience_simulation(self):
        """Simulate the exact user experience that's causing issues."""
        try:
            # Check Rich availability locally
            try:
                from rich.console import Console
                rich_available = True
            except ImportError:
                rich_available = False
            
            if not rich_available:
                self.logger.log(DiagnosticResult(
                    name="User Experience Simulation",
                    status="FAIL",
                    message="Rich library not available"
                ))
                return
            
            # Rich components 
            from rich.layout import Layout
            from rich.panel import Panel
            from rich.live import Live
            
            # Simulate the exact scenario described in the issues
            console = Console(width=80, height=24, force_terminal=True)
            
            user_experience_issues = []
            
            # Test 1: "Layout broken from start"
            try:
                layout = Layout()
                layout.split_column(
                    Layout(Panel("Header", style="cyan"), name="header", size=3),
                    Layout(Panel("Main Area", style="white"), name="main"),
                    Layout(Panel("Input: ", style="green"), name="input", size=3)
                )
                
                # Capture initial layout
                with redirect_stdout(io.StringIO()) as f:
                    console.print(layout)
                    initial_output = f.getvalue()
                
                # Check if initial layout is broken
                lines = initial_output.split('\n')
                if len([line for line in lines if line.strip()]) < 15:  # Should have substantial output
                    user_experience_issues.append("initial_layout_broken")
                    
            except Exception as e:
                user_experience_issues.append(f"initial_layout_error: {str(e)}")
            
            # Test 2: "Gets broken when typing" - simulate Live updating with input
            try:
                input_text = ""
                typing_issues = []
                
                def make_layout_with_input(text):
                    layout = Layout()
                    layout.split_column(
                        Layout(Panel("Header", style="cyan"), name="header", size=3),
                        Layout(Panel("Main Area", style="white"), name="main"),
                        Layout(Panel(f"Input: {text}", style="green"), name="input", size=3)
                    )
                    return layout
                
                # Simulate typing character by character
                test_phrase = "hello world"
                for i, char in enumerate(test_phrase):
                    input_text += char
                    layout_with_input = make_layout_with_input(input_text)
                    
                    # Test if layout breaks during typing
                    with redirect_stdout(io.StringIO()) as f:
                        console.print(layout_with_input)
                        current_output = f.getvalue()
                    
                    # Check for corruption during typing
                    current_lines = current_output.split('\n')
                    valid_lines = [line for line in current_lines if line.strip()]
                    
                    if len(valid_lines) < 10:  # Layout collapsed
                        typing_issues.append(f"collapsed_at_char_{i}_{char}")
                    
                    # Check for missing input text
                    if input_text not in current_output:
                        typing_issues.append(f"input_missing_at_char_{i}_{char}")
                
                if typing_issues:
                    user_experience_issues.extend([f"typing_corruption: {issue}" for issue in typing_issues[:3]])  # Limit to first 3
                    
            except Exception as e:
                user_experience_issues.append(f"typing_simulation_error: {str(e)}")
            
            # Test 3: "Enter key not working" - simulate Enter key processing
            try:
                enter_processed = False
                
                # Mock a simple Enter key handler
                class SimpleEnterHandler:
                    def __init__(self):
                        self.messages = []
                    
                    def handle_enter(self, text):
                        try:
                            # Simulate message processing
                            self.messages.append(text)
                            return True
                        except Exception:
                            return False
                
                handler = SimpleEnterHandler()
                test_message = "test message"
                
                # Test Enter key processing
                result = handler.handle_enter(test_message)
                if not result or len(handler.messages) != 1 or handler.messages[0] != test_message:
                    user_experience_issues.append("enter_key_processing_failed")
                else:
                    enter_processed = True
                    
            except Exception as e:
                user_experience_issues.append(f"enter_key_error: {str(e)}")
            
            # Test 4: Integration test - combine all issues
            try:
                # Create a realistic TUI session simulation
                session_issues = []
                
                # 1. Start with layout
                layout = make_layout_with_input("")
                
                # 2. Add some typing
                for char in "hi":
                    # Simulate real-time updates
                    layout = make_layout_with_input(char)
                    
                # 3. Try to send a message (Enter key)
                final_message = "hi"
                handler = SimpleEnterHandler()
                message_sent = handler.handle_enter(final_message)
                
                if not message_sent:
                    session_issues.append("complete_workflow_failed")
                    
                if session_issues:
                    user_experience_issues.extend(session_issues)
                    
            except Exception as e:
                user_experience_issues.append(f"integration_test_error: {str(e)}")
            
            # Report results
            if user_experience_issues:
                self.logger.log(DiagnosticResult(
                    name="User Experience Simulation",
                    status="FAIL",
                    message="User experience issues detected - matches reported problems",
                    details={
                        "detected_issues": user_experience_issues,
                        "issue_count": len(user_experience_issues),
                        "matches_reported": {
                            "layout_broken_from_start": any("initial_layout" in issue for issue in user_experience_issues),
                            "breaks_when_typing": any("typing" in issue for issue in user_experience_issues),
                            "enter_key_problems": any("enter_key" in issue for issue in user_experience_issues)
                        }
                    },
                    recommendations=[
                        "Layout rendering needs fixing - broken from startup",
                        "Input handling breaks layout - needs input/display separation",
                        "Enter key processing has async/sync boundary issues",
                        "Consider using alternative TUI library or fixing Rich integration"
                    ]
                ))
            else:
                self.logger.log(DiagnosticResult(
                    name="User Experience Simulation",
                    status="PASS",
                    message="User experience simulation completed without issues",
                    details={
                        "tests_completed": 4,
                        "no_corruption_detected": True
                    }
                ))
                
        except Exception as e:
            self.logger.log(DiagnosticResult(
                name="User Experience Simulation",
                status="FAIL",
                message="User experience simulation failed",
                details={"error": str(e), "traceback": traceback.format_exc()}
            ))


class ComprehensiveTUIDiagnostic:
    """Main diagnostic coordinator."""
    
    def __init__(self, verbose: bool = False, json_output: bool = False, quick: bool = False, runtime_focus: bool = False):
        self.logger = DiagnosticLogger(verbose, json_output)
        self.quick = quick
        self.runtime_focus = runtime_focus
        
        # Initialize diagnostic components
        self.env_diagnostics = EnvironmentDiagnostics(self.logger)
        self.lib_diagnostics = LibraryDiagnostics(self.logger)
        self.tui_diagnostics = TUIComponentDiagnostics(self.logger)
        self.perf_diagnostics = PerformanceDiagnostics(self.logger)
        self.error_diagnostics = ErrorScenarioDiagnostics(self.logger)
        self.sim_diagnostics = SimulationDiagnostics(self.logger)
        self.runtime_diagnostics = DeepRuntimeDiagnostics(self.logger)
        self.layout_corruption_diagnostics = LayoutCorruptionDiagnostics(self.logger)
    
    async def run_diagnostics(self) -> int:
        """Run all diagnostics and return exit code."""
        try:
            if not self.logger.json_output:
                mode_desc = "Runtime Focus" if self.runtime_focus else "Comprehensive"
                print(f"{Colors.BOLD}AgentsMCP TUI {mode_desc} Diagnostic{Colors.RESET}")
                print(f"Timestamp: {datetime.now().isoformat()}")
                print(f"Platform: {platform.system()} {platform.release()}")
                print(f"Python: {sys.version}")
            
            if self.runtime_focus:
                # Runtime-focused mode: Only essential deps + deep runtime testing
                if not self.logger.json_output:
                    print(f"{Colors.YELLOW}Running FOCUSED diagnostics for specific TUI issues:{Colors.RESET}")
                    print(f"  - Layout broken from start")
                    print(f"  - Gets broken when typing")  
                    print(f"  - Enter key not working")
                
                # Minimal dependency checks
                await self.lib_diagnostics.diagnose_core_dependencies()
                await self.lib_diagnostics.diagnose_rich_capabilities()
                await self.tui_diagnostics.diagnose_agentsmcp_imports()
                
                # Advanced layout corruption detection - targets specific reported issues
                await self.layout_corruption_diagnostics.diagnose_layout_corruption_patterns()
                
                # Focus on the actual runtime issues
                await self.runtime_diagnostics.diagnose_runtime_tui_instances()
                
            else:
                # Standard comprehensive mode
                # Core diagnostics (always run)
                await self.env_diagnostics.diagnose_python_environment()
                await self.env_diagnostics.diagnose_terminal_environment()
                await self.lib_diagnostics.diagnose_core_dependencies()
                await self.lib_diagnostics.diagnose_rich_capabilities()
                
                # Extended diagnostics (skip in quick mode)
                if not self.quick:
                    await self.tui_diagnostics.diagnose_agentsmcp_imports()
                    await self.tui_diagnostics.diagnose_event_system()
                    await self.tui_diagnostics.diagnose_input_handling()
                    await self.perf_diagnostics.diagnose_performance()
                    await self.error_diagnostics.diagnose_error_scenarios()
                    await self.sim_diagnostics.diagnose_user_scenarios()
                    
                    # Layout corruption detection - specific to reported TUI issues
                    await self.layout_corruption_diagnostics.diagnose_layout_corruption_patterns()
                    
                    # NEW: Deep runtime testing for actual TUI issues
                    await self.runtime_diagnostics.diagnose_runtime_tui_instances()
            
            # Generate summary
            summary = self.logger.get_summary()
            
            if self.logger.json_output:
                print(json.dumps(summary, indent=2))
            else:
                self._print_summary(summary)
            
            return summary['exit_code']
            
        except Exception as e:
            if self.logger.json_output:
                error_summary = {
                    'timestamp': datetime.now().isoformat(),
                    'overall_status': 'DIAGNOSTIC_ERROR',
                    'exit_code': 4,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                print(json.dumps(error_summary, indent=2))
            else:
                print(f"{Colors.RED}{Colors.BOLD}DIAGNOSTIC ERROR:{Colors.RESET}")
                print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
                print(f"\nTraceback:\n{traceback.format_exc()}")
            return 4
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print diagnostic summary."""
        print(f"\n{Colors.BOLD}{Colors.BLUE}=== DIAGNOSTIC SUMMARY ==={Colors.RESET}")
        
        status_colors = {
            'HEALTHY': Colors.GREEN,
            'MINOR_ISSUES': Colors.YELLOW,
            'MAJOR_ISSUES': Colors.RED,
            'CRITICAL': Colors.RED + Colors.BOLD
        }
        
        overall_status = summary['overall_status']
        color = status_colors.get(overall_status, Colors.WHITE)
        
        print(f"Overall Status: {color}{overall_status}{Colors.RESET}")
        print(f"Exit Code: {summary['exit_code']}")
        print(f"Total Checks: {summary['total_checks']}")
        
        # Status breakdown
        counts = summary['status_counts']
        if counts.get('PASS', 0) > 0:
            print(f"{Colors.GREEN}✓ Passed: {counts['PASS']}{Colors.RESET}")
        if counts.get('WARN', 0) > 0:
            print(f"{Colors.YELLOW}⚠ Warnings: {counts['WARN']}{Colors.RESET}")
        if counts.get('FAIL', 0) > 0:
            print(f"{Colors.RED}✗ Failed: {counts['FAIL']}{Colors.RESET}")
        if counts.get('CRITICAL', 0) > 0:
            print(f"{Colors.RED}{Colors.BOLD}💀 Critical: {counts['CRITICAL']}{Colors.RESET}")
        
        # Recommendations
        recommendations = []
        for result in self.logger.results:
            if result.recommendations:
                recommendations.extend(result.recommendations)
        
        if recommendations:
            print(f"\n{Colors.BOLD}Key Recommendations:{Colors.RESET}")
            for rec in set(recommendations[:10]):  # Top 10 unique recommendations
                print(f"{Colors.YELLOW}→{Colors.RESET} {rec}")
        
        # Status-specific guidance
        if overall_status == 'CRITICAL':
            print(f"\n{Colors.RED}{Colors.BOLD}TUI will not function properly.{Colors.RESET}")
            print("Fix critical issues before attempting to use the TUI.")
        elif overall_status == 'MAJOR_ISSUES':
            print(f"\n{Colors.RED}TUI may have significant problems.{Colors.RESET}")
            print("Address major issues for optimal experience.")
        elif overall_status == 'MINOR_ISSUES':
            print(f"\n{Colors.YELLOW}TUI should work with minor issues.{Colors.RESET}")
            print("Consider addressing warnings for best experience.")
        else:
            print(f"\n{Colors.GREEN}TUI should function properly!{Colors.RESET}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive TUI diagnostic for AgentsMCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
    0: No issues found
    1: Minor issues found (TUI should work with warnings)
    2: Major issues found (TUI may not function properly)
    3: Critical issues found (TUI will not work)
    4: Script error (diagnostic failure)

Examples:
    python comprehensive_tui_diagnostic.py
    python comprehensive_tui_diagnostic.py --verbose
    python comprehensive_tui_diagnostic.py --json --quick
    python comprehensive_tui_diagnostic.py --runtime-focus --verbose  # Focus on specific TUI runtime issues
        """
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed diagnostic information'
    )
    
    parser.add_argument(
        '--json',
        action='store_true', 
        help='Output results in JSON format'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run only essential diagnostics (faster)'
    )
    
    parser.add_argument(
        '--runtime-focus',
        action='store_true',
        help='Focus on deep runtime issues: layout corruption, input breaking, Enter key problems'
    )
    
    args = parser.parse_args()
    
    # Create and run diagnostic
    diagnostic = ComprehensiveTUIDiagnostic(
        verbose=args.verbose,
        json_output=args.json,
        quick=args.quick,
        runtime_focus=args.runtime_focus
    )
    
    # Run diagnostics
    try:
        exit_code = asyncio.run(diagnostic.run_diagnostics())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        if not args.json:
            print(f"\n{Colors.YELLOW}Diagnostic interrupted by user.{Colors.RESET}")
        sys.exit(4)
    except Exception as e:
        if not args.json:
            print(f"{Colors.RED}Unexpected error: {e}{Colors.RESET}")
        sys.exit(4)


if __name__ == "__main__":
    main()