# TUI Alignment Test Suite

This directory contains comprehensive tests for TUI (Terminal User Interface) console alignment issues in AgentsMCP. The test suite is designed to catch and prevent the progressive indentation bug and other console positioning problems.

## Problem Description

The AgentsMCP TUI was experiencing console alignment issues including:

1. **Progressive indentation** - each line getting more indented than the previous
2. **Debug logging interference** - debug logs mixing with user interface 
3. **Inconsistent cursor positioning** - prompts not aligned properly
4. **Line break handling issues** - newlines not properly managed

Example of the problematic output:
```
🚀 AgentsMCP - Fixed Working TUI
                                ──────────────────────────────────────────────────
                                                                                  Type your message (Ctrl+C to exit, /quit to quit):
                                                                                                                                    >
                                                                                                                                      ✅ Connected to ollama-turbo - gpt-oss:120b
> hello

       🤔 Thinking...
                     2025-08-31T11:34:12+0300 DEBUG agentsmcp.conversation.llm_client: Tool execution turn 1/3
```

## Test Suite Structure

### Core Test Files

- **`test_tui_alignment.py`** - Tests for cursor positioning and alignment issues
- **`test_tui_logging_isolation.py`** - Tests for logging isolation from UI output
- **`test_tui_comprehensive_suite.py`** - Orchestrates all tests and provides analysis
- **`test_tui_utilities.py`** - Utility classes and mock terminals for testing
- **`conftest.py`** - Pytest configuration and shared fixtures

### Integration Tests

- **`../integration/test_tui_console_output.py`** - Integration tests for complete conversation flows

## Test Categories

### 1. Console Alignment Tests (`test_tui_alignment.py`)

Tests cursor position tracking and alignment:

- ✅ Cursor position tracking accuracy
- ✅ Prompts stay at column 0 after responses  
- ✅ Multi-line responses maintain consistent indentation
- ✅ Carriage returns properly reset cursor position
- ✅ Backspace handling doesn't affect alignment
- ✅ No progressive indentation over multiple messages

#### Key Test Classes:
- `TestTUICursorAlignment` - Basic cursor positioning tests
- `TestTUIStateConsistency` - State management across operations
- `TestTUIEdgeCases` - Edge cases that could cause alignment issues
- `TestTUIPerformanceAlignment` - Performance impact of alignment operations

### 2. Logging Isolation Tests (`test_tui_logging_isolation.py`)

Tests that debug logs don't interfere with TUI display:

- ✅ Debug logs don't appear in stdout
- ✅ LLM client logs are properly separated
- ✅ Thinking indicators work without log pollution
- ✅ Clean console output in production mode

#### Key Test Classes:
- `TestLoggingIsolation` - Basic logging separation
- `TestLoggerConfiguration` - Logger setup verification  
- `TestProductionModeLogging` - Production vs development logging
- `TestAsyncLoggingIsolation` - Async operation logging
- `TestLogFormattingIsolation` - ANSI code handling

### 3. Integration Tests (`test_tui_console_output.py`)

Tests complete conversation flows:

- ✅ Complete conversation flow with proper alignment
- ✅ Status messages, prompts, and responses align correctly
- ✅ Edge cases like very long responses
- ✅ Command output alignment (/help, /quit, etc.)
- ✅ Error message alignment
- ✅ Performance doesn't degrade over time

#### Key Test Classes:
- `TestTUICompleteConversationFlow` - End-to-end conversation testing
- `TestTUIEdgeCaseIntegration` - Integration edge cases
- `TestTUIPerformanceIntegration` - Extended performance validation
- `TestTUIRealWorldScenarios` - Realistic usage patterns

## Test Utilities

### `test_tui_utilities.py`

Provides comprehensive testing infrastructure:

#### Core Classes:

- **`TerminalSimulator`** - Advanced terminal emulation with cursor tracking
- **`AlignmentAnalyzer`** - Analyzes output for alignment issues
- **`OutputPatternMatcher`** - Verifies conversation flow patterns
- **`PerformanceBenchmarker`** - Performance measurement and analysis

#### Key Features:

- ANSI escape sequence processing
- Screen buffer simulation with scrolling
- Cursor position tracking
- Progressive indentation detection
- Response formatting analysis
- Performance benchmarking

## Running the Tests

### Run All TUI Tests
```bash
pytest tests/ui/ -v
```

### Run Specific Test Categories
```bash
# Alignment tests only
pytest tests/ui/test_tui_alignment.py -v

# Logging isolation tests only  
pytest tests/ui/test_tui_logging_isolation.py -v

# Integration tests only
pytest tests/integration/test_tui_console_output.py -v
```

### Run Comprehensive Test Suite
```bash
# From project root
cd tests/ui && python test_tui_comprehensive_suite.py

# Or with pytest
pytest tests/ui/test_tui_comprehensive_suite.py -v
```

### Run Tests by Markers
```bash
# Alignment-related tests
pytest -m alignment -v

# Logging-related tests
pytest -m logging -v

# Integration tests
pytest -m integration -v

# Performance tests
pytest -m performance -v

# Regression tests for known issues
pytest -m regression -v
```

## Test Configuration

### Pytest Configuration (`conftest.py`)

Provides shared fixtures and utilities:

- `tui_instance` - Fresh TUI instance for each test
- `mock_llm_client` - Mock LLM client with configurable responses  
- `capture_logs` - Capture logging output for analysis
- `isolated_stdout` - Isolate stdout for UI testing
- `mock_terminal` - Terminal simulator for testing
- `tui_assertions` - Custom assertions for TUI testing

### Environment Variables

- `AGENTSMCP_TEST_MODE=1` - Disables real terminal interactions during testing
- `AGENTSMCP_ENV=production` - Tests production logging behavior
- `AGENTSMCP_ENV=development` - Tests development logging behavior

## Key Technical Areas

### Terminal Control Testing

Tests proper handling of:
- `\r` (carriage return) for cursor positioning
- `\n` (newline) for line breaks  
- ANSI escape sequences for cursor control
- Terminal width detection and wrapping

### State Management Testing

Verifies:
- Cursor column tracking accuracy
- Line position state between operations
- Buffer management for input/output separation

### Logging Integration Testing

Ensures:
- Logger configuration prevents console interference
- Proper separation of debug output from user interface
- Log level management for different output streams

## Expected Deliverables ✅

The test suite provides:

1. **Comprehensive test coverage** for all alignment scenarios
2. **Mock terminal utilities** for testing cursor positioning  
3. **Console capture mechanisms** for verifying clean output
4. **Performance benchmarks** for alignment operations
5. **Edge case coverage** for unusual input/output scenarios

## Integration Notes

The tests work with:
- `src/agentsmcp/ui/v2/fixed_working_tui.py` - Current TUI implementation
- `src/agentsmcp/conversation/llm_client.py` - LLM integration
- Standard terminal emulation for CI/CD compatibility

## Test Results

Current status: **🎉 All tests passing**

```
Running TUI Comprehensive Test Suite...

Alignment Tests:
  ✅ progressive_indentation
  ✅ cursor_tracking  
  ✅ multiline_formatting
  ✅ boundary_conditions

Logging Tests:
  ✅ debug_isolation
  ✅ ui_log_separation
  ✅ async_logging
  ✅ error_handling

Integration Tests:
  ✅ conversation_flow
  ✅ command_integration
  ✅ error_scenarios
  ✅ performance

Summary: 12/12 tests passed
🎉 All TUI tests passed! No alignment issues detected.
```

## Contributing

When adding new TUI features or fixes:

1. Run the comprehensive test suite to ensure no regressions
2. Add tests for new functionality in the appropriate test file
3. Update this README if adding new test categories or utilities
4. Ensure all tests pass before submitting changes

The test suite is designed to catch alignment issues early and guide fixes for a clean, professional TUI experience.