"""CLI module for AgentsMCP."""

# Import main functions from the consolidated CLI
import sys
from pathlib import Path

# Add parent directory to path to import from cli.py
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from agentsmcp.cli import main as _main
    
    # Re-export main function and other key functions
    def main():
        """Main CLI entry point."""
        return _main()
    
    # For backward compatibility, also export other expected functions
    def run_interactive():
        """Run interactive mode - placeholder."""
        pass
    
    def monitor_dashboard():
        """Monitor dashboard - placeholder.""" 
        pass
    
    def monitor_costs():
        """Monitor costs - placeholder."""
        pass
        
    def monitor_budget():
        """Monitor budget - placeholder."""
        pass
        
    def knowledge_optimize():
        """Knowledge optimization - placeholder."""
        pass
        
except ImportError:
    # Fallback if imports fail
    def main():
        print("CLI not available")
        return 1
    
    def run_interactive():
        pass
    def monitor_dashboard():
        pass
    def monitor_costs():
        pass
    def monitor_budget():
        pass  
    def knowledge_optimize():
        pass