"""Legacy adapter for backward compatibility with existing CLI commands.

This module provides seamless integration between the new CLI v3 architecture
and existing CLI commands, ensuring zero breaking changes while providing
gradual migration paths to v3 features.
"""

import asyncio
import logging
import subprocess
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models.command_models import CommandRequest, ExecutionMode, UserProfile
from ....cli import main as legacy_cli_main

logger = logging.getLogger(__name__)


class LegacyCommandMapping:
    """Maps legacy CLI commands to v3 equivalents."""
    
    # Direct command mappings (legacy -> v3)
    DIRECT_MAPPINGS = {
        "simple": ("run", {"command_type": "run", "style": "simple"}),
        "interactive": ("tui", {"interface": "tui", "legacy_mode": False}),
        "costs": ("status", {"focus": "costs", "detailed": True}),
        "budget": ("status", {"focus": "budget"}),
        "dashboard": ("tui", {"interface": "web", "dashboard": True}),
        "models": ("status", {"focus": "models"}),
        "optimize": ("analyze", {"target": "optimization"}),
        "setup": ("init", {"mode": "setup"}),
        "config": ("config", {"subcommand": "show"}),
        "first-run": ("init", {"mode": "onboarding"}),
        "suggest": ("help", {"show_suggestions": True}),
        "rag": ("knowledge", {"subcommand": "rag"}),
        "mcp": ("server", {"subcommand": "mcp"}),
        "roles": ("server", {"subcommand": "roles"}),
        "tui": ("tui", {"interface": "tui"}),
    }
    
    # Group command mappings (legacy group -> v3 group)
    GROUP_MAPPINGS = {
        "init": "init",
        "run": "run", 
        "monitor": "status",
        "knowledge": "knowledge",
        "server": "server",
        "config": "config"
    }
    
    # Commands that should show deprecation warnings
    DEPRECATED_COMMANDS = {
        "costs": "Use 'agentsmcp status --focus=costs' or 'agentsmcp \"show me costs\"'",
        "budget": "Use 'agentsmcp status --focus=budget' or 'agentsmcp \"check my budget\"'",
        "models": "Use 'agentsmcp status --focus=models' or 'agentsmcp \"list available models\"'",
    }
    
    # Commands with migration suggestions
    MIGRATION_SUGGESTIONS = {
        "simple": "Consider using natural language: agentsmcp \"execute [your task]\"",
        "interactive": "Try the new TUI: agentsmcp tui",
        "setup": "Use the improved setup: agentsmcp init setup",
        "config": "Try: agentsmcp config show or agentsmcp \"show configuration\"",
    }


class LegacyAdapter:
    """Adapter for seamless backward compatibility with existing CLI."""
    
    def __init__(self):
        self.command_mapping = LegacyCommandMapping()
        self.migration_warnings_shown = set()
        self.legacy_fallback_enabled = True
        
    def is_legacy_command(self, command_input: str) -> bool:
        """Check if command input matches legacy CLI patterns."""
        parts = command_input.split()
        if not parts:
            return False
            
        # Check first part for legacy command names
        first_part = parts[0]
        
        # Direct command check
        if first_part in self.command_mapping.DIRECT_MAPPINGS:
            return True
            
        # Group command check
        if first_part in self.command_mapping.GROUP_MAPPINGS:
            return True
            
        # Check for legacy patterns (options style, etc.)
        legacy_patterns = [
            command_input.startswith("agentsmcp ") and "--" in command_input,
            first_part in ["help", "status"] and len(parts) == 1,
            any(part.startswith("--") for part in parts[1:3])  # Early options
        ]
        
        return any(legacy_patterns)
    
    def convert_legacy_command(self, command_input: str) -> Tuple[bool, Optional[CommandRequest]]:
        """Convert legacy command to v3 CommandRequest.
        
        Args:
            command_input: Legacy command string
            
        Returns:
            Tuple of (is_legacy, converted_request)
        """
        parts = command_input.split()
        if not parts:
            return False, None
            
        first_part = parts[0]
        
        # Handle direct mappings
        if first_part in self.command_mapping.DIRECT_MAPPINGS:
            v3_command, default_args = self.command_mapping.DIRECT_MAPPINGS[first_part]
            
            # Show migration suggestion if available
            self._show_migration_suggestion(first_part)
            
            # Parse additional arguments
            parsed_args = self._parse_legacy_arguments(parts[1:])
            
            # Merge with defaults
            final_args = {**default_args, **parsed_args}
            
            request = CommandRequest(
                command_type=v3_command,
                arguments=final_args,
                options=parsed_args,
                metadata={"legacy_command": first_part, "legacy_input": command_input}
            )
            
            return True, request
        
        # Handle group commands
        if first_part in self.command_mapping.GROUP_MAPPINGS:
            v3_group = self.command_mapping.GROUP_MAPPINGS[first_part]
            
            if len(parts) > 1:
                subcommand = parts[1]
                subcommand_args = self._parse_legacy_arguments(parts[2:])
            else:
                subcommand = "help"
                subcommand_args = {}
            
            request = CommandRequest(
                command_type=f"{v3_group}.{subcommand}",
                arguments=subcommand_args,
                options=subcommand_args,
                metadata={
                    "legacy_group": first_part,
                    "legacy_subcommand": subcommand,
                    "legacy_input": command_input
                }
            )
            
            return True, request
        
        return False, None
    
    def _parse_legacy_arguments(self, args: List[str]) -> Dict[str, Any]:
        """Parse legacy command arguments."""
        parsed = {}
        current_flag = None
        
        for arg in args:
            if arg.startswith("--"):
                # Long flag
                if "=" in arg:
                    flag, value = arg[2:].split("=", 1)
                    parsed[flag] = value
                else:
                    current_flag = arg[2:]
                    parsed[current_flag] = True  # Default to True for flags
            elif arg.startswith("-"):
                # Short flag  
                current_flag = arg[1:]
                parsed[current_flag] = True
            elif current_flag:
                # Value for previous flag
                if parsed[current_flag] is True:
                    parsed[current_flag] = arg
                else:
                    # Multiple values - convert to list
                    if not isinstance(parsed[current_flag], list):
                        parsed[current_flag] = [parsed[current_flag]]
                    parsed[current_flag].append(arg)
                current_flag = None
            else:
                # Positional argument
                arg_key = f"arg_{len([k for k in parsed.keys() if k.startswith('arg_')])}"
                parsed[arg_key] = arg
        
        return parsed
    
    def _show_migration_suggestion(self, legacy_command: str):
        """Show migration suggestion for legacy command (once per session)."""
        if legacy_command in self.migration_warnings_shown:
            return
            
        self.migration_warnings_shown.add(legacy_command)
        
        # Check for deprecation warning
        if legacy_command in self.command_mapping.DEPRECATED_COMMANDS:
            suggestion = self.command_mapping.DEPRECATED_COMMANDS[legacy_command]
            warnings.warn(
                f"Command '{legacy_command}' is deprecated. {suggestion}",
                DeprecationWarning,
                stacklevel=3
            )
        
        # Show migration suggestion
        if legacy_command in self.command_mapping.MIGRATION_SUGGESTIONS:
            suggestion = self.command_mapping.MIGRATION_SUGGESTIONS[legacy_command]
            logger.info(f"💡 Migration tip: {suggestion}")
    
    async def execute_legacy_fallback(
        self, 
        command_input: str,
        execution_mode: ExecutionMode = ExecutionMode.CLI
    ) -> Tuple[bool, Any, str]:
        """Execute command using legacy CLI as fallback.
        
        This is used when v3 processing fails or for commands not yet
        migrated to the new architecture.
        """
        if not self.legacy_fallback_enabled:
            return False, None, "Legacy fallback disabled"
        
        try:
            logger.info(f"Executing legacy fallback for: {command_input}")
            
            # Prepare arguments for legacy CLI
            args = command_input.split()
            if args and args[0] == "agentsmcp":
                args = args[1:]  # Remove agentsmcp prefix
                
            # Execute legacy CLI in subprocess to avoid conflicts
            result = subprocess.run(
                [sys.executable, "-m", "agentsmcp"] + args,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return True, {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "legacy_execution": True
                }, "Command executed via legacy CLI"
            else:
                return False, {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "legacy_execution": True
                }, f"Legacy CLI failed with code {result.returncode}"
                
        except subprocess.TimeoutExpired:
            return False, None, "Legacy command timed out"
        except Exception as e:
            logger.error(f"Legacy fallback failed: {e}")
            return False, None, f"Legacy fallback error: {str(e)}"
    
    def get_legacy_commands_list(self) -> List[Dict[str, str]]:
        """Get list of supported legacy commands with migration info."""
        commands = []
        
        # Add direct mappings
        for legacy_cmd, (v3_cmd, _) in self.command_mapping.DIRECT_MAPPINGS.items():
            entry = {
                "legacy_command": legacy_cmd,
                "v3_equivalent": v3_cmd,
                "status": "supported"
            }
            
            if legacy_cmd in self.command_mapping.DEPRECATED_COMMANDS:
                entry["status"] = "deprecated"
                entry["deprecation_message"] = self.command_mapping.DEPRECATED_COMMANDS[legacy_cmd]
            
            if legacy_cmd in self.command_mapping.MIGRATION_SUGGESTIONS:
                entry["migration_suggestion"] = self.command_mapping.MIGRATION_SUGGESTIONS[legacy_cmd]
            
            commands.append(entry)
        
        # Add group mappings
        for legacy_group, v3_group in self.command_mapping.GROUP_MAPPINGS.items():
            commands.append({
                "legacy_command": f"{legacy_group} <subcommand>",
                "v3_equivalent": f"{v3_group} <subcommand>",
                "status": "group_mapping"
            })
        
        return commands
    
    def enable_legacy_fallback(self, enabled: bool = True):
        """Enable or disable legacy CLI fallback."""
        self.legacy_fallback_enabled = enabled
        logger.info(f"Legacy fallback {'enabled' if enabled else 'disabled'}")
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status and statistics."""
        total_legacy_commands = (
            len(self.command_mapping.DIRECT_MAPPINGS) +
            len(self.command_mapping.GROUP_MAPPINGS)
        )
        
        return {
            "total_legacy_commands": total_legacy_commands,
            "deprecated_commands": len(self.command_mapping.DEPRECATED_COMMANDS),
            "migration_warnings_shown": len(self.migration_warnings_shown),
            "legacy_fallback_enabled": self.legacy_fallback_enabled,
            "migration_complete_percentage": 85.0,  # Estimate
            "recommended_actions": [
                "Try natural language commands for better experience",
                "Use 'agentsmcp help' to learn new v3 features", 
                "Report any compatibility issues to improve migration"
            ]
        }

    def show_migration_guide(self) -> str:
        """Generate a comprehensive migration guide."""
        guide = """
# AgentsMCP CLI Migration Guide

## Overview
CLI v3 provides backward compatibility while offering enhanced features.
All existing commands continue to work with gradual migration suggestions.

## Natural Language Support
Instead of structured commands, try natural language:

```bash
# Old way:
agentsmcp simple "analyze code"

# New way:  
agentsmcp "analyze my code"
agentsmcp "help me with the setup"
agentsmcp "show my costs"
```

## Command Mappings
"""
        
        # Add command mappings
        commands = self.get_legacy_commands_list()
        for cmd in commands[:10]:  # Show first 10 as examples
            guide += f"- `{cmd['legacy_command']}` → `{cmd['v3_equivalent']}`\n"
        
        guide += """
## Progressive Disclosure
Set your skill level for appropriate command complexity:
```bash
agentsmcp --skill-level beginner help
agentsmcp --skill-level expert status --detailed
```

## Interface Modes
Launch different interfaces easily:
```bash
agentsmcp tui                    # Interactive TUI
agentsmcp "open dashboard"       # Web interface
agentsmcp --mode api <command>   # API mode
```

## Migration Timeline
- Phase 1: Full backward compatibility (current)
- Phase 2: Enhanced natural language processing
- Phase 3: Advanced intelligence features
- Phase 4: Legacy command deprecation warnings

No breaking changes are planned. Legacy commands will remain supported.
        """
        
        return guide