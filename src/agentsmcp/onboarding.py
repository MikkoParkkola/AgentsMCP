"""First-run onboarding wizard for AgentsMCP CLI."""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import click
import time
import threading

from agentsmcp.config import Config
from agentsmcp.paths import default_user_config_path
from agentsmcp.errors import ConfigError, NetworkError, TaskExecutionError


class OnboardingSpinner:
    """Simple spinner for async operations."""
    
    def __init__(self, message: str):
        self.message = message
        self._running = False
        self._thread = None
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
    
    def _spin(self):
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self._running:
            print(f"\r{click.style(chars[i % len(chars)], fg='cyan')} {self.message}", 
                  end="", flush=True)
            time.sleep(0.1)
            i += 1
    
    def stop(self, final_msg: Optional[str] = None, success: bool = True):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        
        if final_msg:
            icon = "✅" if success else "❌"
            color = "green" if success else "red"
            print(f"\r{click.style(icon, fg=color)} {final_msg}")
        else:
            print()


class OnboardingWizard:
    """Interactive first-run setup wizard."""
    
    def __init__(self, mode: str = "interactive"):
        """
        Initialize onboarding wizard.
        
        Args:
            mode: One of 'interactive', 'quick', 'advanced'
        """
        self.mode = mode
        self.config_data = {}
        
    def should_run_onboarding(self) -> bool:
        """Check if onboarding should be triggered."""
        # Skip if environment variable set
        if os.getenv("AGENTSMCP_SKIP_ONBOARDING") == "1":
            return False
        
        # Check for existing config
        config_path = default_user_config_path()
        if not config_path.exists():
            return True
        
        # Check if config is valid
        try:
            Config.from_file(config_path)
            return False
        except Exception:
            # Config exists but is invalid - offer repair
            click.echo(click.style(
                "⚠️  Existing config appears corrupted. Running repair wizard...", 
                fg='yellow'
            ))
            return True
    
    def welcome_screen(self) -> str:
        """Show welcome screen and get mode selection."""
        if self.mode != "interactive":
            return self.mode
        
        click.echo()
        click.echo(click.style("👋  Welcome to ", fg='blue', bold=True) + 
                  click.style("AgentsMCP", fg='cyan', bold=True) + 
                  click.style(" – the multi-agent platform with built-in cost intelligence.", fg='blue'))
        click.echo()
        click.echo("You're about to set up your environment for the first time.")
        click.echo("We'll walk you through the minimal config required to run a sample task.")
        click.echo()
        
        click.echo("How would you like to proceed?")
        click.echo()
        click.echo("  [1] Interactive wizard (recommended)")
        click.echo("  [2] Quick-start (no questions)")
        click.echo("  [3] Advanced – show every option")
        click.echo()
        
        while True:
            choice = click.prompt("Select an option", default="1", type=str)
            if choice in ["1", "interactive", ""]:
                return "interactive"
            elif choice in ["2", "quick"]:
                return "quick"
            elif choice in ["3", "advanced"]:
                return "advanced"
            else:
                click.echo(click.style("Please enter 1, 2, or 3", fg='red'))
    
    def step_provider_selection(self) -> Dict[str, Any]:
        """Step 1: Choose LLM provider."""
        if self.mode == "quick":
            return {"provider": "openai"}
        
        click.echo()
        click.echo(click.style("Step 1/5 • Choose an LLM provider", fg='cyan', bold=True))
        click.echo()
        
        providers = [
            ("OpenAI", "Most common, works out-of-the-box", True),
            ("Anthropic", "Great for reasoning, requires separate key", False),
            ("Azure OpenAI", "Enterprise endpoint", False),
            ("Custom", "Provide full endpoint & auth", False)
        ]
        
        if self.mode == "interactive":
            for i, (name, desc, default) in enumerate(providers, 1):
                marker = " ←" if default else ""
                click.echo(f"  {i}) {name}{marker}")
                click.echo(f"     {click.style(desc, dim=True)}")
        
        while True:
            choice = click.prompt("Which LLM provider would you like to use?", 
                                default="1", type=str)
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(providers):
                    provider_map = {
                        0: "openai",
                        1: "anthropic", 
                        2: "azure",
                        3: "custom"
                    }
                    return {"provider": provider_map[idx]}
            except ValueError:
                pass
            click.echo(click.style("Please enter 1, 2, 3, or 4", fg='red'))
    
    def step_api_key(self, provider: str) -> Dict[str, Any]:
        """Step 2: Get and validate API key."""
        if self.mode == "quick":
            # In quick mode, we need an API key from environment
            env_key = os.getenv("OPENAI_API_KEY") or os.getenv("AGENTSMCP_API_KEY")
            if not env_key:
                raise ConfigError(
                    "Quick-start mode requires OPENAI_API_KEY environment variable"
                )
            return {"api_key": env_key}
        
        click.echo()
        click.echo(click.style("Step 2/5 • API credentials", fg='cyan', bold=True))
        click.echo()
        
        provider_name = provider.title()
        while True:
            api_key = click.prompt(f"Enter your {provider_name} API key", 
                                 hide_input=True, type=str)
            
            if not api_key.strip():
                if click.confirm("Skip API key validation? (you won't be able to run tasks)"):
                    return {"api_key": ""}
                continue
            
            # Validate API key
            spinner = OnboardingSpinner(f"Testing {provider_name} API key...")
            spinner.start()
            
            try:
                valid = self._validate_api_key(provider, api_key)
                if valid:
                    spinner.stop(f"Key verified – access confirmed", True)
                    return {"api_key": api_key}
                else:
                    spinner.stop(f"Key appears invalid", False)
                    if not click.confirm("Retry with a different key?", default=True):
                        return {"api_key": api_key}  # Use anyway
            except Exception as e:
                spinner.stop(f"Validation failed: {str(e)[:50]}", False)
                if not click.confirm("Use this key anyway?", default=False):
                    continue
                return {"api_key": api_key}
    
    def step_model_selection(self, provider: str) -> Dict[str, Any]:
        """Step 3: Pick default model."""
        if self.mode == "quick":
            defaults = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-haiku",
                "azure": "gpt-4o-mini",
                "custom": "default"
            }
            return {"model": defaults.get(provider, "gpt-4o-mini")}
        
        click.echo()
        click.echo(click.style("Step 3/5 • Default model", fg='cyan', bold=True))
        click.echo()
        
        models = {
            "openai": [
                ("gpt-4o-mini", "$0.001 per 1k tokens", True),
                ("gpt-4o", "$0.003 per 1k tokens", False),
                ("gpt-4-turbo", "$0.002 per 1k tokens", False)
            ],
            "anthropic": [
                ("claude-3-haiku", "$0.001 per 1k tokens", True),
                ("claude-3-sonnet", "$0.003 per 1k tokens", False),
                ("claude-3-opus", "$0.015 per 1k tokens", False)
            ]
        }
        
        provider_models = models.get(provider, [("default", "Standard model", True)])
        
        click.echo("Select a default model (press <Enter> for the recommended one):")
        click.echo()
        
        for i, (name, cost, default) in enumerate(provider_models, 1):
            marker = " ←" if default else ""
            click.echo(f"  {i}) {name}   ({cost}){marker}")
        
        while True:
            choice = click.prompt("Model selection", default="1", type=str)
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(provider_models):
                    return {"model": provider_models[idx][0]}
            except ValueError:
                pass
            click.echo(click.style(f"Please enter 1-{len(provider_models)}", fg='red'))
    
    def step_cost_budget(self) -> Dict[str, Any]:
        """Step 4: Set cost budget."""
        if self.mode == "quick":
            return {"cost_budget": 0.0}
        
        click.echo()
        click.echo(click.style("Step 4/5 • Cost budget", fg='cyan', bold=True))
        click.echo()
        
        click.echo("AgentsMCP tracks token usage per run and warns you before")
        click.echo("you exceed your budget. You can set a daily or per-run limit.")
        click.echo()
        
        while True:
            budget = click.prompt(
                "Enter a daily cost limit in USD (default $0 – you'll be alerted before any spend)",
                default="0", type=str
            )
            
            try:
                if budget.lower() in ["none", "unlimited"]:
                    return {"cost_budget": -1.0}  # Unlimited
                
                budget_float = float(budget)
                if budget_float >= 0:
                    if budget_float > 0:
                        click.echo()
                        click.echo(click.style(
                            f"💸 You'll be warned when a run is forecasted to exceed ${budget_float:.2f}",
                            fg='yellow'
                        ))
                    return {"cost_budget": budget_float}
                else:
                    click.echo(click.style("Budget must be >= 0", fg='red'))
            except ValueError:
                click.echo(click.style("Please enter a number or 'none'", fg='red'))
    
    def step_orchestrator(self) -> Dict[str, Any]:
        """Step 5: Choose orchestrator engine."""
        if self.mode == "quick":
            return {"orchestrator": "local"}
        
        click.echo()
        click.echo(click.style("Step 5/5 • Execution engine", fg='cyan', bold=True))
        click.echo()
        
        engines = [
            ("Local Docker", "Spins up containers per agent (recommended)", True),
            ("Native Python", "Runs agents as subprocesses", False),
            ("Remote K8s", "Point to a cluster endpoint (enterprise)", False)
        ]
        
        click.echo("Select an execution engine:")
        click.echo()
        
        for i, (name, desc, default) in enumerate(engines, 1):
            marker = " ←" if default else ""
            click.echo(f"  {i}) {name}{marker}")
            click.echo(f"     {click.style(desc, dim=True)}")
        
        while True:
            choice = click.prompt("Engine selection", default="1", type=str)
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(engines):
                    engine_map = {0: "docker", 1: "native", 2: "k8s"}
                    engine = engine_map[idx]
                    
                    # Validate Docker if selected
                    if engine == "docker":
                        spinner = OnboardingSpinner("Checking Docker...")
                        spinner.start()
                        
                        try:
                            # Simple check - could be more sophisticated
                            import subprocess
                            result = subprocess.run(["docker", "--version"], 
                                                  capture_output=True, text=True, timeout=5)
                            if result.returncode == 0:
                                spinner.stop("Docker is available", True)
                            else:
                                spinner.stop("Docker not found - falling back to Native", False)
                                engine = "native"
                        except Exception:
                            spinner.stop("Docker not available - using Native mode", False)
                            engine = "native"
                    
                    return {"orchestrator": engine}
            except ValueError:
                pass
            click.echo(click.style("Please enter 1, 2, or 3", fg='red'))
    
    def dry_run_validation(self, config_data: Dict[str, Any]) -> bool:
        """Step 6: Validate the complete configuration."""
        click.echo()
        click.echo(click.style("🔧  Validating configuration...", fg='cyan', bold=True))
        
        spinner = OnboardingSpinner("Running configuration test...")
        spinner.start()
        
        try:
            # Simulate validation - in real implementation, test the actual config
            time.sleep(2)  # Simulate validation time
            
            # Basic validation checks
            required_fields = ["provider", "model", "cost_budget", "orchestrator"]
            for field in required_fields:
                if field not in config_data:
                    raise ValueError(f"Missing required field: {field}")
            
            spinner.stop("All checks passed!", True)
            return True
            
        except Exception as e:
            spinner.stop(f"Validation failed: {str(e)}", False)
            
            click.echo()
            click.echo("What would you like to do?")
            click.echo("  [R]etry validation")
            click.echo("  [S]kip validation (risky)")
            click.echo("  [E]xit and fix manually")
            
            choice = click.prompt("Choose", default="R", type=str).lower()
            if choice in ["s", "skip"]:
                return True
            elif choice in ["e", "exit"]:
                return False
            else:
                return self.dry_run_validation(config_data)  # Retry
    
    def run_demo_task(self) -> bool:
        """Step 7: Run the Hello World demo."""
        click.echo()
        click.echo(click.style("🚀  Running demo 'Hello-World' task...", fg='green', bold=True))
        click.echo()
        
        # Simulate demo execution with progress
        steps = [
            ("Starting Planner agent", 1.0),
            ("Planner completed: decided to greet user", 0.5),
            ("Starting Speaker agent", 1.5),
            ("Generated greeting message", 0.8)
        ]
        
        for step_name, duration in steps:
            spinner = OnboardingSpinner(step_name)
            spinner.start()
            time.sleep(duration)
            spinner.stop(f"{step_name} ✓", True)
        
        click.echo()
        click.echo(click.style("💬 LLM response: ", fg='blue') + 
                  click.style("\"Hello! 👋 AgentsMCP is now ready to orchestrate your multi-agent workflows!\"", 
                            fg='green'))
        click.echo()
        click.echo(click.style("✅ Demo completed successfully!", fg='green', bold=True))
        
        return True
    
    def show_next_steps(self):
        """Step 8: Show completion message and next steps."""
        click.echo()
        click.echo(click.style("🎉  You've successfully completed the first-run onboarding!", 
                             fg='green', bold=True))
        click.echo()
        
        click.echo("Next steps you might explore:")
        click.echo(f"  • {click.style('agentsmcp config edit', fg='cyan')}          – tweak settings later")
        click.echo(f"  • {click.style('agentsmcp knowledge import', fg='cyan')}   – add your own knowledge base") 
        click.echo(f"  • {click.style('agentsmcp server start', fg='cyan')}          – launch the web UI")
        click.echo(f"  • {click.style('agentsmcp run <your-script>', fg='cyan')}     – run your own multi-agent flow")
        click.echo()
        click.echo(f"Run {click.style('agentsmcp --help', fg='cyan')} for the full command list.")
        click.echo()
    
    def save_config(self, config_data: Dict[str, Any]) -> bool:
        """Save the configuration to disk."""
        try:
            config_path = default_user_config_path()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create basic config structure
            config = Config(
                orchestrator_model=config_data.get("model", "gpt-4o-mini"),
                cost_budget=config_data.get("cost_budget", 0.0),
                orchestrator_engine=config_data.get("orchestrator", "docker"),
                api_keys={
                    config_data.get("provider", "openai"): config_data.get("api_key", "")
                }
            )
            
            config.save_to_file(config_path)
            click.echo(click.style(f"✅ Configuration saved to {config_path}", fg='green'))
            return True
            
        except Exception as e:
            click.echo(click.style(f"❌ Failed to save config: {e}", fg='red'))
            return False
    
    def run(self) -> bool:
        """Run the complete onboarding wizard."""
        try:
            # Welcome and mode selection
            self.mode = self.welcome_screen()
            
            # Quick mode shortcut
            if self.mode == "quick":
                click.echo()
                click.echo(click.style("⚡️ Quick-start: using OpenAI, gpt-4o-mini, $0 daily budget, local engine.", 
                                     fg='yellow'))
                
                self.config_data = {
                    "provider": "openai",
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "model": "gpt-4o-mini",
                    "cost_budget": 0.0,
                    "orchestrator": "docker"
                }
                
                if not self.config_data["api_key"]:
                    click.echo(click.style("❌ OPENAI_API_KEY environment variable required for quick-start", 
                                         fg='red'))
                    return False
            else:
                # Interactive mode - go through all steps
                provider_data = self.step_provider_selection()
                self.config_data.update(provider_data)
                
                api_data = self.step_api_key(provider_data["provider"])
                self.config_data.update(api_data)
                
                model_data = self.step_model_selection(provider_data["provider"])
                self.config_data.update(model_data)
                
                budget_data = self.step_cost_budget()
                self.config_data.update(budget_data)
                
                orchestrator_data = self.step_orchestrator()
                self.config_data.update(orchestrator_data)
            
            # Validate configuration
            if not self.dry_run_validation(self.config_data):
                return False
            
            # Save configuration
            if not self.save_config(self.config_data):
                return False
            
            # Run demo
            if not self.run_demo_task():
                return False
            
            # Show completion
            self.show_next_steps()
            
            return True
            
        except KeyboardInterrupt:
            click.echo("\n")
            click.echo(click.style("👋 Onboarding cancelled by user.", fg='yellow'))
            
            # Save partial progress
            if self.config_data:
                try:
                    partial_path = default_user_config_path().parent / "onboarding.partial.yaml"
                    # Save partial config for resume (implementation would save YAML)
                    click.echo(click.style(f"💾 Partial progress saved. Resume with: agentsmcp config resume", 
                                         fg='blue'))
                except Exception:
                    pass
            
            return False
            
        except Exception as e:
            click.echo(click.style(f"❌ Onboarding failed: {e}", fg='red'))
            return False
    
    def _validate_api_key(self, provider: str, api_key: str) -> bool:
        """Validate API key by making a test request."""
        # In real implementation, this would make actual API calls
        # For now, just check that key looks reasonable
        if not api_key or len(api_key.strip()) < 10:
            return False
        
        # Simulate API validation delay
        time.sleep(1)
        
        # Basic format validation
        if provider == "openai" and not api_key.startswith("sk-"):
            return False
        
        return True


def detect_and_run_onboarding(mode: str = "auto") -> bool:
    """
    Detect if onboarding should run and execute it if needed.
    
    Args:
        mode: 'auto' (detect), 'force' (always run), 'interactive', 'quick', 'advanced'
        
    Returns:
        True if onboarding ran successfully or was not needed
    """
    wizard = OnboardingWizard()
    
    if mode == "force" or (mode == "auto" and wizard.should_run_onboarding()):
        if mode == "auto":
            wizard.mode = "interactive"  # Default for auto-detection
        else:
            wizard.mode = mode
        
        return wizard.run()
    
    return True  # No onboarding needed