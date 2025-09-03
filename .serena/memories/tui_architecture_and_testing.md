# Revolutionary TUI Architecture and Testing

## TUI Architecture Overview
The Revolutionary TUI Interface is the main focus of recent development work, with a sophisticated architecture:

### Core Components
- **Revolutionary TUI Interface** (`src/agentsmcp/ui/v2/revolutionary_tui_interface.py`) - Main TUI implementation
- **Revolutionary Launcher** (`src/agentsmcp/ui/v2/revolutionary_launcher.py`) - Entry point and capability detection
- **Unified TUI Coordinator** (`src/agentsmcp/ui/v2/unified_tui_coordinator.py`) - Central coordination
- **Integration Layer** (`src/agentsmcp/ui/v2/reliability/integration_layer.py`) - Reliability integration
- **Input Rendering Pipeline** (`src/agentsmcp/ui/v2/input_rendering_pipeline.py`) - Input handling

### Key Features
- Rich multi-panel layout with status bars and interactive sections
- 60fps animations and visual effects (without console flooding)
- AI Command Composer integration with smart suggestions
- Symphony Dashboard with live metrics
- Real-time status updates and agent monitoring
- Typewriter effects and visual feedback
- Advanced input handling with command completion
- Zero dotted line pollution using unified architecture

### Entry Points
1. `./agentsmcp tui` - Main TUI command (uses Revolutionary TUI by default)
2. `./agentsmcp tui-alias` - TUI with capability detection and options
3. `./agentsmcp tui-v2-dev` - Development TUI (direct v2 access)
4. `./agentsmcp tui-v2-raw` - Minimal raw TTY tester

## Testing Requirements for TUI
Based on recent git history, there have been critical issues with:
- Input visibility (user typing not showing up)
- TUI execution bypassing
- Input buffer systems conflicts
- Guardian state management
- Integration layer reliability

### Critical Test Areas
1. **Input Visibility** - Ensure user typing is immediately visible
2. **LLM Integration** - Verify actual AI responses work
3. **Command Functionality** - Test help, clear, quit commands
4. **Rich Interface** - Confirm Rich Live display (not fallback mode)
5. **Error Handling** - Graceful error recovery
6. **TTY Detection** - Proper enhanced/Rich mode activation

### Test Infrastructure
- Use pytest with `@pytest.mark.ui` for TUI tests
- Use `@pytest.mark.interactive` for tests requiring terminal interaction
- Test files should be in root directory (following existing pattern)
- Use subprocess to run actual `./agentsmcp tui` commands
- Capture and validate output patterns