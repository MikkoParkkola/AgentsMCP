# Live Dashboard Guide

AgentsMCP provides real-time monitoring and management capabilities through multiple dashboard interfaces. This guide covers setup, usage, and customization of the live dashboard system.

## Dashboard Interfaces

### 1. Terminal UI Dashboard (Default)
```bash
# Launch interactive TUI dashboard
agentsmcp dashboard

# Monitor specific agent team
agentsmcp dashboard --team backend-development

# Focus on performance metrics
agentsmcp dashboard --view performance
```

### 2. Web Dashboard
```bash
# Start web dashboard server
agentsmcp dashboard --web --port 8080

# Access at http://localhost:8080
# Supports real-time WebSocket updates
```

### 3. CLI Status Commands
```bash
# Quick agent status overview
agentsmcp agent list --live

# Resource usage monitoring
agentsmcp system status --refresh 5s

# Task queue monitoring
agentsmcp queue monitor --continuous
```

## Dashboard Components

### Agent Status Panel
```
┌─ Active Agents ─────────────────────────────────────────┐
│ ID    Type                Status    Task               CPU│
│ a001  backend-engineer    working   API endpoints      8% │
│ a002  web-frontend-eng    waiting   UI components      2% │
│ a003  qa-engineer         testing   integration       15% │
│ a004  architect           planning  system design      4% │
└─────────────────────────────────────────────────────────┘
```

### Resource Monitoring
```
┌─ System Resources ──────────────────────────────────────┐
│ Memory: ████████████░░░░ 75% (12.3GB / 16GB)          │
│ CPU:    ██████░░░░░░░░░░ 45% (8 cores active)          │
│ Disk:   ███░░░░░░░░░░░░░ 23% (45GB / 200GB)            │
│ Network: ↑ 125KB/s ↓ 2.3MB/s                          │
└─────────────────────────────────────────────────────────┘
```

### Task Queue Overview
```
┌─ Task Queue Status ─────────────────────────────────────┐
│ Pending:    23 tasks                                    │
│ Processing: 5 tasks                                     │
│ Completed:  1,247 tasks (today)                        │
│ Failed:     3 tasks (0.24% failure rate)               │
│ Avg Time:   4.2s per task                              │
└─────────────────────────────────────────────────────────┘
```

### Provider Performance
```
┌─ Provider Statistics ───────────────────────────────────┐
│ ollama-turbo  █████████████████████████ 87% (primary)  │
│ openai        ████████░░░░░░░░░░░░░░░░░ 8%  (fallback) │
│ anthropic     ███░░░░░░░░░░░░░░░░░░░░░░ 5%  (specialist)│
│ Avg Latency: 2.3s | Success Rate: 94.2%               │
└─────────────────────────────────────────────────────────┘
```

## Dashboard Configuration

### Configuration File
```yaml
# ~/.agentsmcp/dashboard.yaml
dashboard:
  interface: tui  # tui, web, or hybrid
  refresh_rate: 2  # seconds
  
  panels:
    agents:
      enabled: true
      position: top-left
      height: 10
    
    resources:
      enabled: true
      position: top-right
      height: 6
      
    queue:
      enabled: true
      position: bottom-left
      width: 50
      
    logs:
      enabled: true
      position: bottom-right
      max_lines: 50
  
  alerts:
    memory_threshold: 90
    cpu_threshold: 80
    failure_rate_threshold: 5
    response_time_threshold: 10
    
  theme:
    primary_color: blue
    warning_color: yellow
    error_color: red
    success_color: green
```

### Environment Variables
```bash
export AGENTSMCP_DASHBOARD_REFRESH=1
export AGENTSMCP_DASHBOARD_THEME=dark
export AGENTSMCP_DASHBOARD_PANELS=agents,resources,queue
export AGENTSMCP_DASHBOARD_PORT=8080
```

## Real-Time Monitoring

### Agent Lifecycle Tracking
```bash
# Monitor agent creation and destruction
agentsmcp dashboard --track-lifecycle

# View agent conversation history
agentsmcp dashboard --agent a001 --show-history

# Monitor agent performance over time
agentsmcp dashboard --agent a001 --performance-graph
```

### Task Flow Visualization
```
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Task Queue  │───▶│ Processing  │───▶│ Completed   │
    │    23       │    │      5      │    │   1,247     │
    └─────────────┘    └─────────────┘    └─────────────┘
           │                                      │
           ▼                                      ▼
    ┌─────────────┐                        ┌─────────────┐
    │ Failed      │                        │ Archived    │
    │     3       │                        │   45,892    │
    └─────────────┘                        └─────────────┘
```

### Performance Metrics
```bash
# Real-time performance dashboard
agentsmcp dashboard --metrics performance

# Historical performance trends
agentsmcp dashboard --metrics trends --period 7d

# Provider performance comparison
agentsmcp dashboard --metrics providers --compare
```

## Web Dashboard Features

### Setup
```bash
# Install web dashboard dependencies
pip install agentsmcp[web-dashboard]

# Start with authentication
agentsmcp dashboard --web --auth --port 8080

# Start with SSL
agentsmcp dashboard --web --ssl-cert cert.pem --ssl-key key.pem
```

### Web Interface Features
- **Real-time Updates**: WebSocket-based live data
- **Interactive Graphs**: Click and drag for historical data
- **Agent Chat**: Direct communication with agents
- **Task Management**: Create, modify, and cancel tasks
- **System Controls**: Start/stop agents, adjust resources
- **Export Capabilities**: Download logs, metrics, reports

### API Endpoints
```bash
# REST API for dashboard data
GET /api/dashboard/agents
GET /api/dashboard/system/status
GET /api/dashboard/tasks/queue
GET /api/dashboard/metrics/performance
POST /api/dashboard/agents/{id}/action
PUT /api/dashboard/config
```

## Custom Monitoring

### Custom Metrics
```python
# custom_metrics.py
from agentsmcp.monitoring import DashboardPlugin

class CustomMetricsPlugin(DashboardPlugin):
    def get_metrics(self):
        return {
            'custom_kpi': self.calculate_kpi(),
            'business_metric': self.get_business_value(),
            'quality_score': self.calculate_quality()
        }
    
    def get_panel_config(self):
        return {
            'title': 'Custom KPIs',
            'position': 'center',
            'refresh': 5,
            'chart_type': 'line'
        }

# Register custom plugin
agentsmcp dashboard --plugin custom_metrics.py
```

### Alerting Integration
```python
# alerts.py
from agentsmcp.monitoring import AlertHandler

class SlackAlertHandler(AlertHandler):
    def handle_alert(self, alert_type, message, severity):
        if severity >= AlertSeverity.HIGH:
            self.send_slack_message(
                channel='#agentsmcp-alerts',
                message=f"🚨 {alert_type}: {message}"
            )

# Configure alerting
agentsmcp dashboard --alerts slack_alerts.py
```

## Dashboard Automation

### Automated Reports
```bash
# Generate daily reports
agentsmcp dashboard report --daily --email team@company.com

# Weekly performance summaries
agentsmcp dashboard report --weekly --format pdf --output weekly-report.pdf

# Custom report generation
agentsmcp dashboard report --template custom.html --data metrics.json
```

### Scheduled Tasks
```bash
# Schedule dashboard snapshots
agentsmcp dashboard schedule --snapshot daily --time 06:00

# Automated performance baselines
agentsmcp dashboard schedule --baseline weekly --metrics performance

# Alert rule updates
agentsmcp dashboard schedule --update-alerts --source config.yaml
```

## Troubleshooting

### Common Issues

**Dashboard not updating:**
```bash
# Check WebSocket connection
agentsmcp dashboard --debug --verbose

# Verify agent communication
agentsmcp agent ping --all

# Restart dashboard service
agentsmcp dashboard restart
```

**High memory usage:**
```bash
# Reduce dashboard refresh rate
agentsmcp dashboard --refresh 10

# Limit historical data retention
agentsmcp dashboard --history-limit 1000

# Disable heavy panels
agentsmcp dashboard --panels agents,queue
```

**Performance issues:**
```bash
# Use lightweight mode
agentsmcp dashboard --lightweight

# Reduce panel count
agentsmcp dashboard --minimal

# Use CLI instead of web interface
agentsmcp agent list --watch
```

### Debug Commands
```bash
# Dashboard debug information
agentsmcp dashboard --debug --output debug.log

# Component health check
agentsmcp dashboard healthcheck

# Performance profiling
agentsmcp dashboard --profile --output dashboard-profile.json
```

## Advanced Features

### Multi-Environment Monitoring
```yaml
# environments.yaml
environments:
  development:
    endpoint: http://localhost:8000
    refresh: 2
  staging:
    endpoint: https://staging.agentsmcp.com
    refresh: 5
  production:
    endpoint: https://prod.agentsmcp.com
    refresh: 10
    alerts: critical_only
```

### Team Dashboards
```bash
# Team-specific views
agentsmcp dashboard --team backend --view tasks
agentsmcp dashboard --team frontend --view performance
agentsmcp dashboard --team qa --view test-results

# Role-based access control
agentsmcp dashboard --role developer --permissions read-only
agentsmcp dashboard --role admin --permissions full-access
```

### Integration with External Tools
```bash
# Grafana integration
agentsmcp dashboard --export grafana --config grafana.json

# Datadog metrics
agentsmcp dashboard --metrics-export datadog --api-key $DD_API_KEY

# Prometheus metrics endpoint
agentsmcp dashboard --prometheus --port 9090
```