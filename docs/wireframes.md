# AgentsMCP CLI/TUI Wireframes

## 1. Welcome Screen (First Run)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ AgentsMCP 🎭 Multi-Agent Orchestra                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│     ╭─────────────────────────────────────────────────────────────╮     │
│     │  Welcome to the Future of Agent Orchestration              │     │
│     │                                                             │     │
│     │  Transform complex multi-agent workflows into              │     │
│     │  beautiful, intuitive experiences.                         │     │
│     ╰─────────────────────────────────────────────────────────────╯     │
│                                                                         │
│  How familiar are you with command-line interfaces?                    │
│                                                                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐   │
│  │  👋 New Here  │  │ 💻 Developer  │  │  ⚡ Command Line Expert   │   │
│  │               │  │               │  │                           │   │
│  │ Guide me      │  │ I know some   │  │ Skip to advanced setup    │   │
│  │ through       │  │ CLI basics    │  │                           │   │
│  │ everything    │  │               │  │                           │   │
│  └───────────────┘  └───────────────┘  └───────────────────────────┘   │
│                                                                         │
│                              [Continue]                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Natural Language Command Interface

```
┌─────────────────────────────────────────────────────────────────────────┐
│ AgentsMCP > Natural Language Mode                               [?] [⚙] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ What would you like to do? (Try speaking naturally)                    │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ > Create a Python agent to check my code for security issues       │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ ╭─ AI Command Composer ───────────────────────────────────────────────╮ │
│ │                                                                     │ │
│ │ I understand you want to:                                           │ │
│ │ • Create a new agent                                                │ │
│ │ • Language: Python                                                  │ │
│ │ • Task: Security analysis                                           │ │
│ │                                                                     │ │
│ │ This will run:                                                      │ │
│ │ ┌─────────────────────────────────────────────────────────────────┐ │ │
│ │ │ agentsmcp create --type python \                                │ │ │
│ │ │                  --tool semgrep \                               │ │ │
│ │ │                  --name security-checker \                      │ │ │
│ │ │                  --target ./                                    │ │ │
│ │ └─────────────────────────────────────────────────────────────────┘ │ │
│ │                                                                     │ │
│ │ ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │ │
│ │ │   🔧 Edit    │  │  ▶️ Execute   │  │  💡 Explain Step-by-Step │   │ │
│ │ └──────────────┘  └──────────────┘  └──────────────────────────┘   │ │
│ ╰─────────────────────────────────────────────────────────────────────╯ │
│                                                                         │
│ Recent commands:                                                        │
│ • "List all my agents" → agentsmcp list                                │
│ • "Deploy to staging" → agentsmcp deploy --env staging                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3. Symphony Mode Dashboard

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Symphony Mode 🎼 - Multi-Agent Coordination                    [×] [□] │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ ╭─ Agent Orchestra (5 active) ────────────┬─ Task Flow ──────────────╮  │
│ │                                         │                         │  │
│ │ ┌────────┐ ┌────────┐ ┌────────┐        │    ┌─────┐              │  │
│ │ │   🟢   │ │   🟡   │ │   🔴   │        │ ┌─▶│Task1│──┐           │  │
│ │ │ Python │ │   Go   │ │  Node  │        │ │  └─────┘  │           │  │
│ │ │  92%   │ │  67%   │ │ ERROR  │        │ │     │     ▼           │  │
│ │ └────────┘ └────────┘ └────────┘        │ │  ┌─────┐ ┌─────┐      │  │
│ │                                         │ │  │Task2│ │Task3│      │  │
│ │ ┌────────┐ ┌────────┐                   │ │  └─────┘ └─────┘      │  │
│ │ │   🟢   │ │   ⚪   │                   │ │     │     │           │  │
│ │ │ Rust   │ │ Docker │                   │ │     ▼     ▼           │  │
│ │ │  88%   │ │  idle  │                   │ │  ┌─────────────┐     │  │
│ │ └────────┘ └────────┘                   │ │  │   Results   │     │  │
│ │                                         │ └─▶│   Merge     │     │  │
│ ├─────────────────────────────────────────┤    └─────────────┘     │  │
│ │ 🎵 Harmony Score: 87/100                │                         │  │
│ │                                         │                         │  │
│ │ ████████████████████████████████▒▒▒▒▒   │                         │  │
│ │                                         │                         │  │
│ │ Coordination:  ████████████████████▒    │                         │  │
│ │ Resource Use:  ██████████████████▒▒▒    │                         │  │
│ │ Error Rate:    ████████████████████▒    │                         │  │
│ ├─────────────────────────────────────────┼─────────────────────────│  │
│ │ Active Issues (1):                      │ Performance Metrics:    │  │
│ │ ⚠️  Node agent connection timeout       │ • Throughput: 2.3/min   │  │
│ │    → [Restart] [Check Logs] [Ignore]   │ • Memory: 1.2GB         │  │
│ │                                         │ • CPU: 23%              │  │
│ │ Quick Actions:                          │ • Queue: 7 pending      │  │
│ │ [🔄 Restart All] [📊 Full Report]       │                         │  │
│ ╰─────────────────────────────────────────┴─────────────────────────╯  │
│                                                                         │
│ [F1: Help] [F2: Logs] [F3: Config] [Esc: Exit] [Space: Pause/Resume]  │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4. Expert Command Line Interface

```
┌─────────────────────────────────────────────────────────────────────────┐
│ AgentsMCP Expert Mode                          Active: 5 | Queue: 2    │
├─────────────────────────────────────────────────────────────────────────┤
│ $ agentsmcp symphony status --live                                      │
│                                                                         │
│ Agent Status:                            Task Pipeline:                 │
│ • python-sec   🟢 92% (∆+5%)           ┌─→ analyze-repo → security-scan │
│ • go-api       🟡 67% (⚠ mem high)      │  ├─→ lint-check → deploy-prep │
│ • node-ui      🔴 ERR (conn timeout)    │  └─→ test-suite → integration │
│ • rust-core    🟢 88% (idle 2min)       │                              │
│ • docker-ops   ⚪ idle                   Pipeline Health: 87% 🎵         │
│                                                                         │
│ $ █                                                                     │
│                                                                         │
│ ╭─ Quick Actions ───────────────────────────────────────────────────╮   │
│ │ Ctrl+R  Restart failed agents     │  Ctrl+L  View live logs      │   │
│ │ Ctrl+S  Symphony dashboard        │  Ctrl+H  Show all shortcuts  │   │
│ │ Ctrl+Q  Queue management          │  Ctrl+D  Debug mode          │   │
│ │ Ctrl+P  Performance metrics       │  Ctrl+C  Cancel current      │   │
│ ╰───────────────────────────────────────────────────────────────────╯   │
│                                                                         │
│ Recent commands:                                                        │
│ • agentsmcp create python --tool semgrep --name sec-check              │
│ • agentsmcp symphony start --agents 5 --mode balanced                  │
│ • agentsmcp deploy --env prod --confirm                                 │
│                                                                         │
│ Command suggestions:                                                    │
│ • agentsmcp fix node-ui --auto        (fix connection timeout)         │
│ • agentsmcp scale --agents +2          (handle increased queue)        │
│ • agentsmcp report --format json       (export current metrics)        │
└─────────────────────────────────────────────────────────────────────────┘
```

## 5. Error Recovery Interface

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 🔧 Error Recovery Assistant                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ ╭─ Issue Detected ─────────────────────────────────────────────────────╮ │
│ │ 🔴 Agent 'node-ui' failed to connect to remote service              │ │
│ │                                                                     │ │
│ │ Last successful connection: 2 minutes ago                           │ │
│ │ Error: ECONNREFUSED 127.0.0.1:3000                                  │ │
│ ╰─────────────────────────────────────────────────────────────────────╯ │
│                                                                         │
│ ╭─ Smart Analysis ─────────────────────────────────────────────────────╮ │
│ │ Based on context analysis, this is likely a:                       │ │
│ │                                                                     │ │
│ │ 🎯 Network Connection Issue (99% confidence)                        │ │
│ │                                                                     │ │
│ │ Contributing factors:                                               │ │
│ │ • Service may have crashed or restarted                            │ │
│ │ • Network configuration changed                                     │ │
│ │ • Port conflict with another process                               │ │
│ ╰─────────────────────────────────────────────────────────────────────╯ │
│                                                                         │
│ ╭─ Recommended Solutions ──────────────────────────────────────────────╮ │
│ │                                                                     │ │
│ │ 1. 🔄 Restart Service (Quick Fix)                                   │ │
│ │    └─ agentsmcp restart node-ui --force                            │ │
│ │       Success rate: 85% | Time: ~30 seconds                       │ │
│ │                                                                     │ │
│ │ 2. 🔍 Check Network Status                                          │ │
│ │    └─ Verify port 3000 availability and service health            │ │
│ │       Success rate: 70% | Time: ~2 minutes                        │ │
│ │                                                                     │ │
│ │ 3. 📋 Switch to Offline Mode                                        │ │
│ │    └─ Continue with cached data and local processing              │ │
│ │       Success rate: 100% | Time: Immediate                        │ │
│ │                                                                     │ │
│ │ ┌───────────────┐  ┌───────────────┐  ┌──────────────────────┐    │ │
│ │ │  Auto-Fix #1  │  │  Guide Me     │  │  Show Detailed Logs  │    │ │
│ │ └───────────────┘  └───────────────┘  └──────────────────────┘    │ │
│ ╰─────────────────────────────────────────────────────────────────────╯ │
│                                                                         │
│ Previous similar issues: 3 (all resolved with restart)                 │
│ Would you like me to remember this solution for future issues?         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 6. Mobile-Responsive Terminal (Collapsed States)

```
┌─────────────────────────┐
│ AgentsMCP 🎭           │
├─────────────────────────┤
│ ┌─ Status ─────────────┐ │
│ │ 🟢 3  🟡 1  🔴 1     │ │
│ │ Harmony: 87%  🎵     │ │
│ └─────────────────────┘ │
│                         │
│ Quick Actions:          │
│ • [R] Restart failed    │
│ • [S] Symphony view     │
│ • [L] Logs              │
│ • [H] Help              │
│                         │
│ Recent:                 │
│ • Created python-sec ✓  │
│ • Started symphony ✓    │
│ • Error on node-ui ⚠️   │
│                         │
│ > Say what to do...     │
│ ┌─────────────────────┐ │
│ │ restart node        │ │
│ └─────────────────────┘ │
│ [Send] [Voice] [Menu]   │
└─────────────────────────┘
```