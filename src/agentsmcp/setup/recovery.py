"""Error Recovery and Guidance System for AgentsMCP Installation.

Provides intelligent recovery guidance for installation failures:
- Analyzes failure patterns and suggests specific solutions
- Offers multiple recovery paths based on error type
- Provides step-by-step remediation instructions
- Supports offline/air-gapped installation scenarios
- Maintains failure rate <10% through comprehensive error handling
"""

from __future__ import annotations

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from .health_checker import HealthStatus, HealthCheckResult, HealthReport

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Types of recovery actions available."""
    RETRY = "retry"
    MANUAL_FIX = "manual_fix"
    ALTERNATIVE_APPROACH = "alternative"
    SKIP_OPTIONAL = "skip_optional"
    REINSTALL = "reinstall"
    CONTACT_SUPPORT = "contact_support"


@dataclass
class RecoveryStep:
    """Individual recovery step with instructions."""
    title: str
    description: str
    command: Optional[str] = None
    platform_specific: Optional[Dict[str, str]] = None
    requires_admin: bool = False
    estimated_time_minutes: int = 1
    success_indicator: Optional[str] = None


@dataclass
class RecoveryPlan:
    """Complete recovery plan for a specific error."""
    error_type: str
    severity: str
    description: str
    steps: List[RecoveryStep] = field(default_factory=list)
    alternative_plans: List['RecoveryPlan'] = field(default_factory=list)
    success_probability: float = 0.8
    estimated_total_time_minutes: int = 5


@dataclass
class RecoveryResult:
    """Result of executing a recovery plan."""
    success: bool
    plan_executed: str
    steps_completed: int
    total_steps: int
    execution_time_minutes: float
    error_message: Optional[str] = None
    next_action: Optional[RecoveryAction] = None


class RecoveryGuide:
    """Intelligent error recovery and guidance system."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize recovery guide.
        
        Args:
            verbose: Enable detailed recovery output
        """
        self.verbose = verbose
        self.recovery_database = self._build_recovery_database()
        
    def analyze_failure(self, error: Exception, context: Dict[str, Any]) -> RecoveryPlan:
        """
        Analyze an installation failure and generate recovery plan.
        
        Args:
            error: The exception that occurred
            context: Additional context about the failure
            
        Returns:
            Comprehensive recovery plan with multiple options
        """
        error_type = self._classify_error(error)
        
        # Get base recovery plan
        plan = self.recovery_database.get(error_type, self._get_generic_recovery_plan())
        
        # Customize plan based on context
        plan = self._customize_plan_for_context(plan, context)
        
        if self.verbose:
            print(f"🔍 Analyzing failure: {error_type}")
            print(f"📋 Generated recovery plan with {len(plan.steps)} steps")
            
        return plan
    
    def analyze_health_report(self, report: HealthReport) -> List[RecoveryPlan]:
        """
        Analyze health check report and generate recovery plans for failures.
        
        Args:
            report: Health check report with failure details
            
        Returns:
            List of recovery plans for each failure found
        """
        recovery_plans = []
        
        for check in report.checks:
            if check.status in [HealthStatus.ERROR, HealthStatus.CRITICAL]:
                plan = self._create_plan_from_health_check(check)
                if plan:
                    recovery_plans.append(plan)
        
        # Sort by severity (critical first)
        recovery_plans.sort(key=lambda p: 0 if p.severity == "critical" else 1)
        
        return recovery_plans
    
    def execute_recovery_plan(self, plan: RecoveryPlan, 
                            interactive: bool = True) -> RecoveryResult:
        """
        Execute a recovery plan with user interaction.
        
        Args:
            plan: Recovery plan to execute
            interactive: Whether to prompt user for confirmation
            
        Returns:
            Result of recovery plan execution
        """
        import time
        start_time = time.time()
        
        if self.verbose:
            print(f"🔧 Executing recovery plan: {plan.error_type}")
            print(f"📊 Estimated time: {plan.estimated_total_time_minutes} minutes")
            
        steps_completed = 0
        
        try:
            for i, step in enumerate(plan.steps, 1):
                if interactive:
                    self._display_step(step, i, len(plan.steps))
                    
                    if step.requires_admin:
                        if not self._confirm_admin_step(step):
                            continue
                            
                    if not self._confirm_step_execution(step):
                        continue
                
                # Execute the step
                success = self._execute_step(step)
                
                if success:
                    steps_completed += 1
                    if self.verbose:
                        print(f"✅ Step {i} completed: {step.title}")
                else:
                    if self.verbose:
                        print(f"❌ Step {i} failed: {step.title}")
                    
                    if interactive and len(plan.alternative_plans) > 0:
                        if self._should_try_alternative():
                            alt_plan = plan.alternative_plans[0]
                            return self.execute_recovery_plan(alt_plan, interactive)
                    
                    break
            
            execution_time = (time.time() - start_time) / 60
            
            return RecoveryResult(
                success=steps_completed == len(plan.steps),
                plan_executed=plan.error_type,
                steps_completed=steps_completed,
                total_steps=len(plan.steps),
                execution_time_minutes=execution_time,
                next_action=RecoveryAction.RETRY if steps_completed == len(plan.steps) else RecoveryAction.MANUAL_FIX
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) / 60
            
            return RecoveryResult(
                success=False,
                plan_executed=plan.error_type,
                steps_completed=steps_completed,
                total_steps=len(plan.steps),
                execution_time_minutes=execution_time,
                error_message=str(e),
                next_action=RecoveryAction.CONTACT_SUPPORT
            )
    
    def get_offline_recovery_guide(self) -> str:
        """
        Generate comprehensive offline recovery guide.
        
        Returns:
            Formatted text guide for offline troubleshooting
        """
        guide = """
# AgentsMCP Installation Recovery Guide

## Quick Diagnostics

1. **Check Python Version**
   ```bash
   python --version
   # Should be 3.8 or higher
   ```

2. **Check pip Installation**
   ```bash
   pip --version
   # If missing: python -m ensurepip --upgrade
   ```

3. **Check Internet Connectivity**
   ```bash
   ping pypi.org
   # For offline installs, skip this step
   ```

## Common Issues and Solutions

### Issue 1: Permission Denied Errors

**Windows:**
```cmd
# Run as Administrator
pip install --user agentsmcp
```

**macOS/Linux:**
```bash
# Use --user flag or virtual environment
pip install --user agentsmcp
# OR create venv
python -m venv agentsmcp_env
source agentsmcp_env/bin/activate  # Linux/macOS
agentsmcp_env\\Scripts\\activate  # Windows
pip install agentsmcp
```

### Issue 2: Network/SSL Errors

**Solution:**
```bash
# Try with trusted hosts
pip install --trusted-host pypi.org --trusted-host pypi.python.org agentsmcp

# For corporate networks
pip install --proxy http://proxy:port agentsmcp
```

### Issue 3: Missing Dependencies

**Solution:**
```bash
# Install build tools first
# Windows: Install Visual Studio Build Tools
# macOS: xcode-select --install  
# Linux: sudo apt-get install build-essential python3-dev
```

### Issue 4: Configuration Errors

1. **Remove existing config:**
   ```bash
   # Linux/macOS
   rm -rf ~/.agentsmcp
   
   # Windows
   rmdir /s %USERPROFILE%\\.agentsmcp
   ```

2. **Run setup again:**
   ```bash
   agentsmcp init --reset
   ```

### Issue 5: API Key Problems

1. **Check environment variables:**
   ```bash
   echo $OPENAI_API_KEY
   echo $ANTHROPIC_API_KEY
   ```

2. **Set temporarily:**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

3. **Test API connectivity:**
   ```bash
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \\
        https://api.openai.com/v1/models
   ```

## Emergency Contacts

- GitHub Issues: https://github.com/AgentsMCP/issues
- Documentation: https://agentsmcp.readthedocs.io
- Discord: https://discord.gg/agentsmcp

## System Information for Support

When reporting issues, include:

```bash
# System info
uname -a  # Linux/macOS
systeminfo  # Windows

# Python info  
python --version
pip list | grep -i agent

# Environment
env | grep -i agent
env | grep -i api
```
"""
        return guide.strip()
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for recovery planning."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        if "permission" in error_str or "access denied" in error_str:
            return "permission_denied"
        elif "network" in error_str or "connection" in error_str or "timeout" in error_str:
            return "network_error"
        elif "ssl" in error_str or "certificate" in error_str:
            return "ssl_certificate_error"
        elif "api" in error_str and "key" in error_str:
            return "api_key_error"
        elif "import" in error_str or "module" in error_str:
            return "dependency_error"
        elif "config" in error_str or "yaml" in error_str or "json" in error_str:
            return "configuration_error"
        elif "docker" in error_str:
            return "docker_error"
        elif error_type in ["filenotfounderror", "ioerror"]:
            return "file_not_found"
        else:
            return "unknown_error"
    
    def _build_recovery_database(self) -> Dict[str, RecoveryPlan]:
        """Build database of recovery plans for common errors."""
        database = {}
        
        # Permission denied errors
        database["permission_denied"] = RecoveryPlan(
            error_type="permission_denied",
            severity="error",
            description="Insufficient permissions to complete installation",
            steps=[
                RecoveryStep(
                    title="Use user installation",
                    description="Install to user directory instead of system-wide",
                    command="pip install --user agentsmcp",
                    estimated_time_minutes=2
                ),
                RecoveryStep(
                    title="Create virtual environment",
                    description="Install in isolated Python environment",
                    command="python -m venv agentsmcp_env && source agentsmcp_env/bin/activate && pip install agentsmcp",
                    platform_specific={
                        "Windows": "python -m venv agentsmcp_env && agentsmcp_env\\Scripts\\activate && pip install agentsmcp"
                    },
                    estimated_time_minutes=3
                )
            ],
            success_probability=0.9,
            estimated_total_time_minutes=5
        )
        
        # Network errors
        database["network_error"] = RecoveryPlan(
            error_type="network_error", 
            severity="error",
            description="Network connectivity issues preventing download",
            steps=[
                RecoveryStep(
                    title="Check internet connection",
                    description="Verify network connectivity",
                    command="ping pypi.org"
                ),
                RecoveryStep(
                    title="Use trusted hosts",
                    description="Bypass SSL verification for PyPI",
                    command="pip install --trusted-host pypi.org --trusted-host pypi.python.org agentsmcp",
                    estimated_time_minutes=2
                ),
                RecoveryStep(
                    title="Configure proxy",
                    description="Set up proxy if behind corporate firewall",
                    command="pip install --proxy http://proxy:port agentsmcp",
                    estimated_time_minutes=1
                )
            ],
            success_probability=0.8,
            estimated_total_time_minutes=4
        )
        
        # API key errors
        database["api_key_error"] = RecoveryPlan(
            error_type="api_key_error",
            severity="warning", 
            description="Invalid or missing API keys",
            steps=[
                RecoveryStep(
                    title="Check environment variables",
                    description="Verify API keys are set correctly",
                    command="echo $OPENAI_API_KEY | head -c 20",
                    platform_specific={
                        "Windows": "echo %OPENAI_API_KEY%"
                    }
                ),
                RecoveryStep(
                    title="Set API key temporarily",
                    description="Set API key for current session",
                    command="export OPENAI_API_KEY=your_key_here",
                    platform_specific={
                        "Windows": "set OPENAI_API_KEY=your_key_here"
                    }
                ),
                RecoveryStep(
                    title="Test API connectivity",
                    description="Verify API key works with provider",
                    command="agentsmcp run simple 'test connection'",
                    estimated_time_minutes=1
                )
            ],
            success_probability=0.85,
            estimated_total_time_minutes=3
        )
        
        # Configuration errors  
        database["configuration_error"] = RecoveryPlan(
            error_type="configuration_error",
            severity="error",
            description="Invalid or corrupted configuration file",
            steps=[
                RecoveryStep(
                    title="Remove existing configuration",
                    description="Delete corrupted config files",
                    command="rm -rf ~/.agentsmcp",
                    platform_specific={
                        "Windows": "rmdir /s %USERPROFILE%\\.agentsmcp"
                    }
                ),
                RecoveryStep(
                    title="Regenerate configuration",
                    description="Run setup wizard to create new config",
                    command="agentsmcp init --reset",
                    estimated_time_minutes=2
                )
            ],
            success_probability=0.95,
            estimated_total_time_minutes=3
        )
        
        # Dependency errors
        database["dependency_error"] = RecoveryPlan(
            error_type="dependency_error",
            severity="critical",
            description="Missing Python dependencies or packages",
            steps=[
                RecoveryStep(
                    title="Update pip",
                    description="Ensure latest pip version",
                    command="python -m pip install --upgrade pip",
                    estimated_time_minutes=1
                ),
                RecoveryStep(
                    title="Install build dependencies",
                    description="Install system build tools if needed",
                    command="# Platform specific - see recovery guide",
                    requires_admin=True,
                    estimated_time_minutes=5
                ),
                RecoveryStep(
                    title="Reinstall AgentsMCP",
                    description="Clean install of AgentsMCP with dependencies",
                    command="pip uninstall -y agentsmcp && pip install agentsmcp",
                    estimated_time_minutes=2
                )
            ],
            success_probability=0.7,
            estimated_total_time_minutes=8
        )
        
        return database
    
    def _get_generic_recovery_plan(self) -> RecoveryPlan:
        """Generic recovery plan for unknown errors."""
        return RecoveryPlan(
            error_type="unknown_error",
            severity="error", 
            description="Unknown installation error occurred",
            steps=[
                RecoveryStep(
                    title="Check system requirements",
                    description="Verify Python version and dependencies",
                    command="python --version && pip --version"
                ),
                RecoveryStep(
                    title="Clean reinstall",
                    description="Remove and reinstall AgentsMCP",
                    command="pip uninstall -y agentsmcp && pip install agentsmcp",
                    estimated_time_minutes=3
                ),
                RecoveryStep(
                    title="Reset configuration",
                    description="Start fresh with configuration",
                    command="agentsmcp init --reset",
                    estimated_time_minutes=2
                )
            ],
            success_probability=0.6,
            estimated_total_time_minutes=6
        )
    
    def _customize_plan_for_context(self, plan: RecoveryPlan, context: Dict[str, Any]) -> RecoveryPlan:
        """Customize recovery plan based on execution context."""
        # Add platform-specific adjustments
        current_platform = platform.system()
        
        for step in plan.steps:
            if step.platform_specific and current_platform in step.platform_specific:
                step.command = step.platform_specific[current_platform]
        
        # Adjust based on detected environment
        if context.get("in_venv"):
            # Remove --user flags if in virtual environment
            for step in plan.steps:
                if step.command and "--user" in step.command:
                    step.command = step.command.replace(" --user", "")
        
        return plan
    
    def _create_plan_from_health_check(self, check: HealthCheckResult) -> Optional[RecoveryPlan]:
        """Create recovery plan from health check failure."""
        if not check.fix_suggestion:
            return None
            
        severity = "critical" if check.status == HealthStatus.CRITICAL else "error"
        
        return RecoveryPlan(
            error_type=f"health_check_{check.name.lower().replace(' ', '_')}",
            severity=severity,
            description=check.message,
            steps=[
                RecoveryStep(
                    title=f"Fix {check.name}",
                    description=check.fix_suggestion,
                    command=check.fix_suggestion if check.fix_suggestion.startswith(("pip", "python", "agentsmcp")) else None
                )
            ],
            success_probability=0.8,
            estimated_total_time_minutes=2
        )
    
    def _display_step(self, step: RecoveryStep, current: int, total: int):
        """Display recovery step to user."""
        print(f"\n🔧 Step {current}/{total}: {step.title}")
        print(f"   {step.description}")
        if step.command:
            print(f"   Command: {step.command}")
        if step.requires_admin:
            print("   ⚠️  This step requires administrator privileges")
    
    def _confirm_step_execution(self, step: RecoveryStep) -> bool:
        """Ask user to confirm step execution."""
        try:
            response = input(f"Execute this step? [Y/n]: ").strip().lower()
            return response in ["", "y", "yes"]
        except (EOFError, KeyboardInterrupt):
            return False
    
    def _confirm_admin_step(self, step: RecoveryStep) -> bool:
        """Confirm execution of step requiring admin privileges."""
        print("⚠️  This step requires administrator/root privileges.")
        try:
            response = input("Do you want to continue? [y/N]: ").strip().lower()
            return response in ["y", "yes"]
        except (EOFError, KeyboardInterrupt):
            return False
    
    def _should_try_alternative(self) -> bool:
        """Ask if user wants to try alternative recovery plan."""
        try:
            response = input("Step failed. Try alternative approach? [Y/n]: ").strip().lower()
            return response in ["", "y", "yes"]
        except (EOFError, KeyboardInterrupt):
            return False
    
    def _execute_step(self, step: RecoveryStep) -> bool:
        """Execute a single recovery step."""
        if not step.command:
            return True  # Manual step, assume success
            
        try:
            result = subprocess.run(
                step.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=step.estimated_time_minutes * 60
            )
            
            success = result.returncode == 0
            
            if not success and self.verbose:
                print(f"Command failed: {result.stderr}")
                
            return success
            
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"Command timed out after {step.estimated_time_minutes} minutes")
            return False
        except Exception as e:
            if self.verbose:
                print(f"Command execution failed: {e}")
            return False


def create_recovery_plan(error: Exception, context: Dict[str, Any]) -> RecoveryPlan:
    """
    Convenience function to create recovery plan for an error.
    
    Args:
        error: Exception that occurred
        context: Additional context information
        
    Returns:
        Recovery plan for the error
    """
    guide = RecoveryGuide()
    return guide.analyze_failure(error, context)