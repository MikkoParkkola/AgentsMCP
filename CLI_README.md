# 🚀 AgentsMCP Revolutionary CLI Interface

A beautiful, intelligent, and adaptive command-line interface for multi-agent orchestration. Inspired by the best practices from Claude Code, Codex CLI, and Gemini CLI, with Apple-style design principles throughout.

## ✨ Key Features

### 🎨 Adaptive Theme System
- **Automatic Detection**: Intelligently detects your terminal's dark/light theme
- **Multiple Detection Methods**: Environment variables, system preferences, terminal info, time-based heuristics
- **Seamless Switching**: Switch themes on-the-fly without restarting
- **Accessibility Compliant**: High contrast ratios for optimal readability

### 📊 Real-Time Orchestration Dashboard  
- **Live Metrics**: Monitor active agents, symphony sessions, and task queues
- **Beautiful Visualizations**: Progress bars, status indicators, and multi-column layouts
- **Performance Monitoring**: CPU, memory, response times, and throughput tracking
- **Auto-Refresh**: Configurable refresh intervals with smooth updates

### 📈 Advanced Statistics Display
- **Trend Analysis**: Sparklines show metric trends at a glance
- **Historical Data**: Automatic data collection with configurable retention
- **Smart Formatting**: Context-aware units (%, ms, MB/s, etc.)
- **Interactive Views**: Overview, detailed, and trends modes

### 🎮 Interactive Command Interface
- **Smart Completion**: Tab completion with context-aware suggestions
- **Command History**: Persistent history with search and filtering
- **Parameter Validation**: Real-time input validation with helpful error messages
- **Conversational UI**: Natural language command processing

### 🎼 Symphony Mode Integration
- **Visual Conducting**: Watch agents coordinate in real-time
- **Harmony Metrics**: See orchestration efficiency and conflict resolution
- **Agent Status**: Monitor individual agent states and performance
- **Task Visualization**: Beautiful task flow and dependency tracking

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default settings (interactive mode)
python -m agentsmcp

# Or use the demo script
python demo_cli.py
```

### Mode-Specific Launch

```bash
# Interactive command interface (default)
python demo_cli.py interactive

# Real-time dashboard
python demo_cli.py dashboard  

# Statistics and trends
python demo_cli.py stats
```

### Command Line Options

```bash
# Force theme mode
python -m agentsmcp --theme dark|light|auto

# Set refresh interval  
python -m agentsmcp --refresh-interval 1.5

# Start in specific mode
python -m agentsmcp --mode dashboard

# Skip welcome screen
python -m agentsmcp --no-welcome

# Enable debug mode
python -m agentsmcp --debug
```

## 🎯 Interface Modes

### 1. Interactive Mode (Default)
The full-featured command interface with smart completion and conversational interactions.

**Features:**
- Tab completion for all commands and parameters
- Command history with search (Ctrl+R)
- Real-time parameter validation
- Built-in help system
- Context-aware suggestions

**Commands:**
```bash
status          # Show system status
dashboard       # Launch dashboard view
execute <task>  # Execute orchestration task
agents          # List and manage agents
symphony <cmd>  # Symphony mode controls
theme <mode>    # Switch theme
history         # Show command history
help [command]  # Get help
config          # Show configuration
clear           # Clear screen
exit            # Exit application
```

### 2. Dashboard Mode  
Real-time monitoring of the orchestration system with beautiful visualizations.

**Features:**
- Live agent status and metrics
- Task queue monitoring
- Performance indicators
- Symphony session tracking
- Auto-refreshing displays

**Controls:**
- `q` - Exit dashboard
- `r` - Manual refresh
- `p` - Pause/resume auto-refresh
- `↑/↓` - Scroll through metrics

### 3. Statistics Mode
Advanced metrics visualization with trends and historical analysis.

**Features:**
- Sparkline trend indicators
- Historical data analysis
- Percentage change calculations
- Category-based organization
- Interactive metric exploration

**Controls:**
- `1-4` - Switch between metric categories
- `o` - Overview mode
- `d` - Detailed view
- `t` - Trends analysis
- `q` - Exit statistics

## 🎨 Theme System

### Automatic Detection
The CLI automatically detects your terminal's theme using multiple methods:

1. **Environment Variables**: `TERM_THEME`, `COLORTERM`
2. **System Preferences**: macOS/Windows system theme
3. **Terminal Information**: Background color analysis
4. **Time Heuristics**: Nighttime defaults to dark theme

### Manual Control
Force a specific theme:
```bash
# Command line
python -m agentsmcp --theme dark

# Interactive mode
> theme light
> theme dark  
> theme auto
```

### Color Schemes

**Dark Theme:**
- Primary: Bright cyan for headings
- Secondary: Light blue for descriptions  
- Accent: Bright magenta for highlights
- Success: Bright green for positive states
- Warning: Bright yellow for cautions
- Error: Bright red for issues
- Muted: Dark gray for secondary text

**Light Theme:**  
- Primary: Dark blue for headings
- Secondary: Medium blue for descriptions
- Accent: Dark magenta for highlights
- Success: Dark green for positive states
- Warning: Dark orange for cautions
- Error: Dark red for issues
- Muted: Light gray for secondary text

## 📊 Metrics and Analytics

### Orchestration Metrics
- **Active Agents**: Currently running agent instances
- **Symphony Sessions**: Concurrent orchestration sessions
- **Task Queue Size**: Pending tasks awaiting processing
- **Completion Rate**: Percentage of successfully completed tasks

### Performance Metrics
- **CPU Usage**: System processor utilization
- **Memory Usage**: RAM consumption percentage
- **Response Time**: Average API response latency
- **Throughput**: Operations processed per second

### Agent Metrics  
- **Spawned Agents**: Total agents created
- **Agent Efficiency**: Average task completion efficiency
- **Error Rate**: Percentage of failed operations
- **Uptime**: System operational time

### System Metrics
- **Disk Usage**: Storage utilization percentage
- **Network I/O**: Data transfer rates  
- **API Calls**: External service requests
- **Cache Hit Rate**: Cache effectiveness percentage

## 🔧 Configuration

### Configuration File
Create `~/.agentsmcp/config.json`:

```json
{
    "theme_mode": "auto",
    "auto_refresh": true,
    "refresh_interval": 2.0,
    "show_welcome": true,
    "enable_colors": true,
    "interface_mode": "interactive",
    "history_size": 1000,
    "log_level": "INFO"
}
```

### Environment Variables
```bash
export AGENTSMCP_THEME=dark
export AGENTSMCP_REFRESH_INTERVAL=1.5
export AGENTSMCP_NO_COLORS=false
```

## 🎼 Symphony Mode Integration

The CLI provides beautiful visualization of Symphony Mode orchestration:

### Visual Elements
- **Conductor Status**: Shows orchestration state and harmony levels
- **Agent Ensemble**: Visual representation of active agents
- **Task Flow**: Beautiful task progression indicators
- **Conflict Resolution**: Real-time conflict detection and resolution

### Monitoring Features
- **Harmony Metrics**: Overall orchestration efficiency
- **Agent Coordination**: Inter-agent communication patterns
- **Resource Utilization**: System resource distribution
- **Performance Optimization**: Real-time performance tuning

## 🔍 Advanced Features

### Smart Completion System
- **Context-Aware**: Suggestions based on current mode and state
- **Parameter Hints**: Show expected parameter types and formats
- **Command Discovery**: Find commands by typing partial matches
- **Error Recovery**: Intelligent error correction suggestions

### Real-Time Updates
- **Async Architecture**: Non-blocking UI updates
- **Smooth Animations**: Transition effects for state changes
- **Progressive Loading**: Graceful handling of loading states
- **Error Resilience**: Automatic recovery from display errors

### Accessibility Features
- **High Contrast**: Optimal color contrast ratios
- **Clear Typography**: Readable fonts and spacing
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader Support**: Compatible with assistive technologies

## 🚀 Performance

### Optimizations
- **Sub-millisecond Response**: Lightning-fast command processing
- **Efficient Rendering**: Minimal screen redraws
- **Memory Management**: Automatic cleanup of old metrics
- **Async Operations**: Non-blocking I/O operations

### Resource Usage
- **Low CPU**: Efficient event loops and rendering
- **Minimal Memory**: Smart data structure management
- **Network Efficient**: Batched API calls and caching
- **Battery Friendly**: Optimized refresh rates

## 🎨 Design Philosophy

### Apple-Style Principles
- **Simplicity**: Clean, uncluttered interfaces
- **Consistency**: Unified design language throughout
- **Intuitiveness**: Natural interaction patterns
- **Delight**: Subtle animations and beautiful typography

### User Experience Focus
- **Zero Configuration**: Works perfectly out of the box
- **Progressive Disclosure**: Advanced features when needed
- **Forgiving Interface**: Helpful error messages and recovery
- **Contextual Help**: Just-in-time assistance

## 🛠 Development

### Architecture
```
src/agentsmcp/ui/
├── theme_manager.py      # Adaptive theme detection and management
├── ui_components.py      # Reusable UI component library  
├── status_dashboard.py   # Real-time orchestration dashboard
├── command_interface.py  # Interactive command interface
├── statistics_display.py # Advanced metrics visualization
├── cli_app.py           # Main application orchestrator
└── __init__.py          # Module exports
```

### Key Components

**ThemeManager**: Intelligent theme detection and color management
**UIComponents**: Beautiful, reusable CLI components (boxes, tables, progress bars)  
**StatusDashboard**: Real-time system monitoring with live updates
**CommandInterface**: Smart command processing with completion and history
**StatisticsDisplay**: Advanced metrics with trends and sparklines
**CLIApp**: Main application that orchestrates all components

### Extension Points
- **Custom Themes**: Add new color schemes
- **New Components**: Extend the UI component library
- **Additional Metrics**: Integrate new data sources
- **Command Plugins**: Add custom command handlers

## 🎯 Best Practices

### Theme Usage
```python
# Always use theme manager for colors
theme = self.theme_manager.current_theme
print(f"{theme.colors['success']}Success!{theme.colors['reset']}")
```

### Component Integration
```python
# Reuse UI components for consistency
box = self.ui.box("Content", title="My Box", style='light')
table = self.ui.table(data, headers=["Col1", "Col2"])
```

### Async Patterns
```python
# Use async for non-blocking operations
async def update_display(self):
    while self.is_running:
        await self.render_content()
        await asyncio.sleep(self.refresh_interval)
```

## 📱 Platform Support

### Tested Terminals
- **macOS**: Terminal.app, iTerm2, Hyper
- **Linux**: GNOME Terminal, Konsole, xterm, Alacritty
- **Windows**: Windows Terminal, PowerShell, WSL

### Requirements
- Python 3.8+
- Unicode support
- ANSI color support (automatic fallback for limited terminals)

## 🎊 What Makes It Revolutionary

1. **Adaptive Intelligence**: Automatically adapts to your terminal and preferences
2. **Apple-Style Design**: Beautiful, intuitive interfaces inspired by Apple's design principles
3. **Real-Time Everything**: Live metrics, instant feedback, responsive interactions
4. **Zero Configuration**: Perfect experience out of the box with smart defaults
5. **Advanced Analytics**: Sparklines, trend analysis, and predictive insights
6. **Orchestration Focus**: Purpose-built for multi-agent coordination and management

---

*Ready to orchestrate the future? Launch the CLI and experience the revolution in command-line interfaces! 🚀*