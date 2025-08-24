# 🤖 AgentsMCP Web UI Documentation

## Overview

AgentsMCP features a spectacular interactive web interface that demonstrates AI agent orchestration in real-time. Watch as Codex, Claude, and Ollama work together like a symphony, with beautiful animations and live performance metrics.

## Features

### 🎼 Live Agent Orchestra
- **Interactive agent network** showing Codex, Claude, and Ollama
- **Real-time status indicators** with pulsing animations
- **Dynamic task routing** visualization
- **Live performance metrics** updating in real-time

### 📋 Task Management Dashboard
- **Animated task queue** with progress indicators
- **Parallel execution visualization** 
- **Success rate tracking** and performance analytics
- **Resource utilization monitoring**

### ✨ Interactive Effects
- **Floating sparkle particles** creating magical atmosphere
- **Smooth hover animations** on all interactive elements  
- **Gradient backgrounds** that respond to activity
- **Responsive design** optimized for all screen sizes

## Quick Access

### Method 1: Direct Binary Command (Simplest)
```bash
# Launch the web UI instantly
agentsmcp ui

# Or with custom configuration
agentsmcp ui --port 8080 --theme dark
```

### Method 2: Python Module
```bash
# From the project directory
python -m agentsmcp.ui

# Or with pip installed
agentsmcp-ui
```

### Method 3: Direct File Access
```bash
# Open directly in browser
open web-ui.html

# Or serve via FastAPI
uvicorn agentsmcp.ui:app --host 0.0.0.0 --port 8000
# Then visit: http://localhost:8000
```

## Integration with AgentsMCP Binary

The web UI provides a real-time window into your AgentsMCP orchestration system:

### Live Agent Monitoring
- **Real agent status** from your running AgentsMCP server
- **Active task tracking** showing what each agent is working on
- **Performance metrics** from actual agent execution
- **Resource utilization** showing CPU, memory, and API usage

### Interactive Controls
```bash
# Launch UI with full control panel
agentsmcp ui --admin

# Features:
# - Start/stop individual agents
# - Adjust task priorities
# - Monitor system health
# - View detailed logs
```

### Session Management
The UI connects to your live AgentsMCP sessions:
- **Session lifecycle visualization** 
- **Heartbeat monitoring** with health indicators
- **Error tracking** with automatic recovery
- **Performance benchmarking** across different agents

## Configuration

### Agent Provider Setup
```bash
# Configure providers for UI display
agentsmcp ui --providers openai,anthropic,ollama

# Set API endpoints
agentsmcp ui --openai-endpoint https://api.openai.com/v1
agentsmcp ui --claude-endpoint https://api.anthropic.com/v1
```

### Visual Customization
```bash
# Theme options
agentsmcp ui --theme dark      # Dark theme (default)
agentsmcp ui --theme light     # Light theme  
agentsmcp ui --theme neon      # Cyberpunk neon theme
agentsmcp ui --theme minimal   # Clean minimal theme

# Animation settings
agentsmcp ui --animations fast    # Fast animations
agentsmcp ui --animations slow    # Slower, more detailed animations
agentsmcp ui --no-particles      # Disable particle effects
```

### Performance Tuning
```bash
# Update intervals
agentsmcp ui --metrics-interval 1000    # Update metrics every 1s
agentsmcp ui --task-interval 500        # Update tasks every 500ms

# Data limits
agentsmcp ui --max-tasks 100            # Show max 100 tasks
agentsmcp ui --history-days 7           # Show 7 days of history
```

## Screenshots

### Agent Orchestra Dashboard
![Agent Orchestra](screenshots/orchestra.png)
*Live visualization of AI agents working in harmony*

### Task Management Interface
![Task Management](screenshots/tasks.png)
*Real-time task queue with progress tracking*

### Performance Analytics  
![Analytics](screenshots/performance.png)
*Comprehensive performance metrics and trends*

### Control Panel
![Control Panel](screenshots/admin.png)
*Administrative interface for managing agents*

## Real-Time Features

### Live Metrics
The UI displays real-time data from your AgentsMCP system:

```javascript
// Metrics updated every second
{
  "active_agents": 3,
  "tasks_completed": 47,
  "success_rate": 98.9,
  "avg_response_time": "1.2s",
  "queue_depth": 5,
  "cpu_usage": 23.1,
  "memory_usage": 45.6
}
```

### Agent Status Monitoring
Each agent shows detailed status information:
- **Health status**: Healthy, Warning, Error
- **Current task**: What the agent is currently processing  
- **Performance**: Response times and success rates
- **Resource usage**: CPU, memory, and API quota

### Task Visualization
Watch tasks flow through your system:
- **Task creation** with priority indicators
- **Agent assignment** with intelligent routing
- **Execution progress** with real-time updates
- **Completion status** with success/failure indicators

## API Integration

### WebSocket Endpoints
The UI connects to real-time WebSocket endpoints:

```bash
# Task updates
ws://localhost:8000/ws/tasks

# Agent status
ws://localhost:8000/ws/agents  

# Performance metrics
ws://localhost:8000/ws/metrics

# System events
ws://localhost:8000/ws/events
```

### REST API Integration
Full REST API support for programmatic control:

```bash
# Get current system status
curl http://localhost:8000/api/status

# List active agents
curl http://localhost:8000/api/agents

# Submit new task
curl -X POST http://localhost:8000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"task": "Generate unit tests", "priority": 1}'
```

## Advanced Features

### Multi-Environment Support
```bash
# Connect to different environments
agentsmcp ui --env production
agentsmcp ui --env staging  
agentsmcp ui --env development

# Multi-cluster monitoring
agentsmcp ui --clusters cluster1,cluster2,cluster3
```

### Custom Dashboards
Create custom dashboard layouts:
```bash
# Predefined layouts
agentsmcp ui --layout executive    # High-level overview
agentsmcp ui --layout technical    # Detailed technical metrics
agentsmcp ui --layout minimal      # Clean, focused view

# Save custom layout
agentsmcp ui --save-layout my-dashboard
```

### Export and Reporting
```bash
# Export performance data
agentsmcp ui --export-csv performance.csv
agentsmcp ui --export-json metrics.json

# Generate reports
agentsmcp ui --report-daily
agentsmcp ui --report-weekly
```

## Development

### Local Development Setup
```bash
# Clone repository
git clone <repository-url>
cd AgentsMCP

# Install dependencies
pip install -e ".[ui,dev]"

# Start development server
agentsmcp ui --dev --reload

# Enable debug mode
agentsmcp ui --debug --verbose
```

### Custom Widgets
Extend the UI with custom widgets:

```python
# plugins/custom_widget.py
from agentsmcp.ui import Widget

class CustomMetricWidget(Widget):
    template = "custom_metric.html"
    
    def get_data(self):
        return {
            "metric_value": self.calculate_custom_metric(),
            "trend": self.get_trend_data()
        }
    
    def calculate_custom_metric(self):
        # Your custom calculation logic
        return 42.5
```

### Theme Development
Create custom themes:

```css
/* themes/cyberpunk.css */
:root {
  --primary-color: #00ff41;
  --secondary-color: #ff0080;  
  --background: linear-gradient(135deg, #0a0a0a, #1a1a2e);
  --glass-effect: rgba(0, 255, 65, 0.1);
}
```

## Troubleshooting

### Common Issues

#### UI Won't Connect
```bash
# Check AgentsMCP server status
agentsmcp status

# Verify WebSocket endpoints
agentsmcp test-websockets

# Check firewall settings
agentsmcp ui --debug-connection
```

#### Performance Issues
```bash
# Reduce update frequency
agentsmcp ui --slow-mode

# Disable animations
agentsmcp ui --no-animations --no-particles

# Enable performance monitoring
agentsmcp ui --performance-mode
```

#### Agent Status Not Updating
```bash
# Restart agent monitoring
agentsmcp restart-monitoring

# Check agent connectivity
agentsmcp test-agents

# Reset UI cache
agentsmcp ui --clear-cache
```

### Logs and Debugging
```bash
# Enable debug logging
agentsmcp ui --log-level debug

# View UI-specific logs
agentsmcp logs --component ui

# Export debug bundle
agentsmcp debug-export --include-ui
```

## Future Roadmap

### Planned Features
- 🎯 **3D Agent Visualization** with WebGL rendering
- 🔔 **Smart Notifications** with ML-powered alerts
- 📱 **Mobile App** with native performance
- 🎨 **Theme Marketplace** with community themes
- 📊 **Advanced Analytics** with predictive insights
- 🔗 **Third-party Integrations** (Slack, Discord, etc.)

### Community Contributions
We welcome contributions for:
- New visualization types
- Performance optimizations  
- Theme designs
- Widget development
- Mobile responsiveness
- Accessibility improvements

See our [Contributing Guide](../CONTRIBUTING.md) for detailed information.