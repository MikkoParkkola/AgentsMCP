# AgentsMCP TUI Comprehensive Diagnostic

This diagnostic script provides comprehensive analysis of the user environment to identify potential issues that could affect the AgentsMCP TUI system functionality.

## Quick Start

```bash
# Basic diagnostic
python comprehensive_tui_diagnostic.py

# Quick diagnostic (essential checks only)
python comprehensive_tui_diagnostic.py --quick

# Verbose output with detailed information
python comprehensive_tui_diagnostic.py --verbose

# JSON output for automation
python comprehensive_tui_diagnostic.py --json
```

## What It Checks

### Environment Diagnostics
- Python version compatibility (requires 3.10+)
- Platform support (Linux, macOS, Windows)
- TTY detection and terminal capabilities
- TERM/COLORTERM environment variables
- Terminal size and encoding settings
- Async/event loop functionality

### Library Compatibility
- Core dependencies (Rich, Click, Pydantic, etc.)
- Rich library capabilities and components
- Version detection and compatibility
- Import dependency validation

### TUI Component Testing
- AgentsMCP TUI module imports
- Event system initialization
- Input handling simulation
- Layout creation and rendering
- Display refresh mechanisms

### Performance Analysis
- Memory usage monitoring
- Async operation performance
- Text processing speed
- I/O performance metrics

### Error Scenarios
- File permission testing
- Signal handling capabilities
- Resource exhaustion simulation
- Edge case input handling
- Error recovery scenarios

### Real User Simulation
- Typing scenario simulation
- Navigation flow testing
- Error recovery workflows
- Integration testing

## Exit Codes

- **0**: No issues found - TUI should work perfectly
- **1**: Minor issues found - TUI should work with warnings
- **2**: Major issues found - TUI may not function properly
- **3**: Critical issues found - TUI will not work
- **4**: Script error - Diagnostic failure

## Output Formats

### Standard Output
Colorized, human-readable format with clear status indicators:
- ✓ Green for passed checks
- ⚠ Yellow for warnings
- ✗ Red for failures
- 💀 Bold red for critical issues

### JSON Output
Machine-readable format suitable for automation:
```json
{
  "timestamp": "2025-09-03T19:31:46.345922",
  "overall_status": "MAJOR_ISSUES",
  "exit_code": 2,
  "status_counts": {
    "PASS": 17,
    "FAIL": 1,
    "WARN": 1
  },
  "total_checks": 19,
  "results": [...]
}
```

## Common Issues and Solutions

### TTY Not Detected
**Problem**: Running in IDE console or piped environment
**Solution**: Run in a proper terminal emulator

### Rich Color Support Missing
**Problem**: Terminal doesn't support colors
**Solution**: Enable color support in terminal settings or use a modern terminal

### AgentsMCP Import Failures
**Problem**: AgentsMCP not installed or not in PYTHONPATH
**Solution**: Install AgentsMCP and ensure proper Python path configuration

### Python Version Too Old
**Problem**: Using Python < 3.10
**Solution**: Upgrade to Python 3.10 or newer

## Troubleshooting

1. **Run in a proper terminal**: Avoid IDE consoles, use Terminal.app, iTerm2, or similar
2. **Check environment variables**: Ensure TERM and COLORTERM are set appropriately
3. **Verify dependencies**: Make sure all required libraries are installed
4. **Test in isolation**: Try running the diagnostic in a clean Python environment

## Integration

This diagnostic can be integrated into:
- CI/CD pipelines for environment validation
- Automated testing workflows
- User support troubleshooting
- Development environment setup

## Example Usage

```bash
# For user support - get comprehensive diagnostics
python comprehensive_tui_diagnostic.py --verbose > diagnostics.txt

# For CI/CD - quick JSON check
python comprehensive_tui_diagnostic.py --json --quick

# For development - interactive debugging
python comprehensive_tui_diagnostic.py --verbose
```

## Notes

- The script is designed to be safe and non-destructive
- It handles import failures gracefully
- All tests are isolated and won't affect system state
- Runs in any Python 3.8+ environment (though 3.10+ recommended)
- No external dependencies beyond standard library + project requirements