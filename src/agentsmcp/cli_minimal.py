"""Minimal CLI entry point for maximum startup speed."""

import sys

# Inline version to avoid module loading overhead
__version__ = "1.0.0"


def show_help():
    """Show help for hierarchical CLI structure."""
    help_text = f"""Usage: agentsmcp [OPTIONS] COMMAND [ARGS]...

  AgentsMCP - Revolutionary Multi-Agent Orchestration with Cost Intelligence.

Options:
  --version                 Show the version and exit.
  --log-level TEXT          Log level (override env)  
  --log-format [json|text]
  --config TEXT             Path to YAML config
  -h, --help                Show this message and exit.

Commands:
  init         Getting started – discovery & first-time configuration.
  run          Run the core AgentsMCP workflows.
  monitor      Observability, cost-tracking and budget alerts.
  knowledge    Knowledge-base management and model selection.
  server       Server lifecycle and integration utilities.
  config       Advanced configuration utilities.

Backward-compatible aliases:
  simple       Execute a task using simple orchestration (recommended...)
  interactive  Launch enhanced interactive CLI with AI chat...
  dashboard    Launch real-time dashboard with cost monitoring...
  costs        Display current costs and budget status...
  budget       Manage monthly budget and get cost alerts...
  models       Show available orchestrator models...
  optimize     Optimize model selection for cost-effectiveness...
  setup        Interactive wizard for AgentsMCP configuration...
  rag          Manage RAG knowledge base...
  mcp          Manage MCP servers in configuration...
  roles        Role‑based orchestration commands...

Use 'agentsmcp <command> --help' for more information on a specific command.
"""
    print(help_text)


def main():
    """Ultra-fast entry point with minimal overhead."""
    args = sys.argv[1:]
    
    # Handle fast-path cases without importing anything
    if not args or args[0] in ['-h', '--help']:
        show_help()
        return
    
    if args[0] == '--version':
        print(f"agentsmcp, version {__version__}")
        return
    
    # For actual commands, delegate to hierarchical CLI
    from agentsmcp.cli import main as hierarchical_main
    hierarchical_main()


if __name__ == "__main__":
    main()