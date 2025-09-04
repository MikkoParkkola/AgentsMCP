# V3 TUI Input Debugging Suite

## Overview

This comprehensive debugging suite diagnoses V3 TUI input issues where:
- Characters appear in bottom right corner one at a time
- Input doesn't reach input box until Enter pressed repeatedly  
- Commands (/) don't work
- No LLM communication occurs

## Root Cause Analysis

Based on the background TUI processes, the main issue is **routing**: `./agentsmcp tui` is using V2 Revolutionary TUI (which runs in demo mode) instead of V3 TUI (which has proper input handling).

```
🚀 Revolutionary TUI Interface - Demo Mode
💬 Interactive mode now available:
💬 TUI> 
👋 Exiting TUI...
✅ Demo completed - TUI shutting down gracefully
```

The V2 system shows "Interactive mode now available" but never actually waits for input - it just exits after a countdown.

## Debugging Scripts

### 1. Master Controller
**`v3_tui_debugging_suite.py`** - Orchestrates all debugging tools
```bash
python v3_tui_debugging_suite.py --quick    # Quick diagnosis
python v3_tui_debugging_suite.py --full     # Full comprehensive analysis
python v3_tui_debugging_suite.py            # Interactive mode (default)
```

### 2. Core Diagnostic (Updated)
**`tui_input_diagnostic.py`** - Updated with V3-specific tests
- Environment detection
- Basic input functionality
- V3 component imports and initialization
- TUI routing detection (V2 vs V3)
- Terminal capability analysis

### 3. V3 Pipeline Debugger
**`v3_tui_input_pipeline_debugger.py`** - Step-by-step V3 pipeline analysis
- Terminal capabilities detection
- Progressive renderer selection  
- Input handling mechanics testing
- ChatEngine command processing
- Full TUILauncher initialization
- V2 vs V3 routing analysis

### 4. Real-Time Input Monitor
**`v3_realtime_input_monitor.py`** - Live input event monitoring
- Character-by-character input tracking
- Input buffer state monitoring
- V3 renderer behavior analysis
- Event timing and performance analysis
- Full pipeline monitoring with TUILauncher

### 5. PlainCLIRenderer Test Suite
**`v3_plain_cli_renderer_tests.py`** - Comprehensive renderer testing
- Renderer creation and initialization
- State management testing
- Mock input handling verification
- Render frame behavior analysis
- Message display functionality
- Threading safety tests
- Input stress testing
- ChatEngine integration verification

### 6. ChatEngine Verifier
**`v3_chat_engine_verifier.py`** - LLM communication verification
- ChatEngine initialization testing
- Command processing (/help, /quit, /status)
- Message processing with mock LLM
- Error handling verification
- Callback system integration
- Message history management
- Concurrent processing tests

### 7. Command Workflow Debugger
**`v3_command_workflow_debugger.py`** - End-to-end command tracing
- Complete workflow tracing from input to execution
- Component interaction analysis
- Performance bottleneck identification
- Traced versions of V3 components
- Command processing timing analysis

## Quick Start

1. **Immediate Fix** (Most Likely Solution):
   ```bash
   # The issue is probably routing to V2 instead of V3
   # Check ./agentsmcp script and modify tui() function to use V3
   python tui_input_diagnostic.py  # Will detect routing issue
   ```

2. **Comprehensive Analysis**:
   ```bash
   python v3_tui_debugging_suite.py --quick
   ```

3. **Detailed Investigation**:
   ```bash
   python v3_tui_debugging_suite.py --full
   ```

## Expected Findings

Based on the background processes, you should see:

1. **Routing Issue**: `./agentsmcp tui` calls V2 Revolutionary TUI instead of V3
2. **TTY Detection**: Terminal capabilities not detected properly
3. **Demo Mode**: V2 runs in demo mode without real input handling
4. **V3 Available**: V3 components exist and should work properly when routed correctly

## Priority Fix Order

1. **Fix Routing** (CRITICAL): Modify `./agentsmcp` script to use V3 TUILauncher
2. **Test V3 Components**: Verify V3 input handling works properly
3. **TTY Detection**: Improve terminal capability detection if needed
4. **Integration**: Ensure V3 components integrate properly

## Files to Check

- `./agentsmcp` - Main script routing (modify tui() function)
- `src/agentsmcp/ui/v3/tui_launcher.py` - V3 entry point
- `src/agentsmcp/ui/v3/plain_cli_renderer.py` - Input handling
- `src/agentsmcp/ui/v3/terminal_capabilities.py` - TTY detection

## Success Criteria

After fixes, you should see:
- Immediate character visibility when typing
- Commands like `/help` work properly
- Real LLM communication occurs
- No "demo mode" messages
- Proper interactive TUI experience

## Troubleshooting

If debugging scripts fail:
1. Check Python environment and AgentsMCP installation
2. Verify all V3 modules can be imported
3. Test basic terminal capabilities
4. Run individual scripts to isolate issues

The debugging suite provides targeted analysis to identify exactly where the V3 TUI pipeline breaks and how to fix it.