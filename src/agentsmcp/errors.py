"""Enhanced error handling for AgentsMCP CLI with friendly, actionable messages."""

from __future__ import annotations
import sys
import click
from typing import Optional


class AgentsMCPError(click.ClickException):
    """Base class for all CLI-visible errors with enhanced formatting."""

    #: Emoji shown at the beginning of every error line
    emoji: str = "❌"

    #: Short, actionable suggestion shown after the main message.
    #: Sub-classes override this in __init__ or default_hint.
    hint: Optional[str] = None

    def __init__(self, message: str, hint: Optional[str] = None) -> None:
        super().__init__(message)
        if hint is not None:
            self.hint = hint

    @property
    def formatted_message(self) -> str:
        """
        Returns the final string that Click writes to stderr.
        Includes:
        * emoji + main message (bold)
        * optional hint on a new line (dim colour)
        """
        lines = [f"{self.emoji}  {click.style(self.message, fg='red', bold=True)}"]
        if self.hint:
            lines.append(click.style(f"💡 {self.hint}", fg='yellow'))
        return "\n".join(lines)

    # Click calls show to emit the message.
    def show(self, file=None) -> None:
        click.echo(self.formatted_message, err=True, file=file)


class InvalidCommandError(AgentsMCPError):
    """Raised when user provides an invalid command."""
    emoji = "🚫"
    
    def __init__(self, cmd: str):
        hint = f"Run {click.style('agentsmcp --help', fg='cyan')} to see available commands."
        super().__init__(f"The command {click.style(cmd, fg='magenta')} is not recognized.", hint)
        
    def show(self, file=None) -> None:
        # Show the error message without suggestions (suggestions shown by CLI handler)
        click.echo(self.formatted_message, err=True, file=file)


class MissingParameterError(AgentsMCPError):
    """Raised when a required parameter is missing."""
    emoji = "⚠️"
    
    def __init__(self, param: str, cmd: str | None = None):
        hint = f"Pass the missing argument: {click.style(f'--{param}', fg='cyan')}"
        if cmd:
            hint += f" (or see {click.style(f'agentsmcp {cmd} --help', fg='cyan')})"
        super().__init__(f"Missing required parameter {click.style(param, fg='magenta')}.", hint)


class ConfigError(AgentsMCPError):
    """Raised when there's a configuration problem."""
    emoji = "🔧"
    
    def __init__(self, details: str):
        hint = f"Run {click.style('agentsmcp init setup', fg='cyan')} to configure AgentsMCP."
        super().__init__(f"Configuration problem – {details}", hint)


class NetworkError(AgentsMCPError):
    """Raised when there's a network connectivity issue."""
    emoji = "🌐"
    
    def __init__(self, details: str):
        hint = (
            "• Verify you are online\n"
            "• If behind a proxy, set the environment variable "
            f"{click.style('HTTPS_PROXY', fg='cyan')}\n"
            "• Retry later, the service could be temporarily unavailable"
        )
        super().__init__(f"Network error – {details}", hint)


class PermissionError(AgentsMCPError):
    """Raised when there's a file/directory permission issue."""
    emoji = "🛡️"
    
    def __init__(self, path: str):
        hint = f"Make sure you have read/write access to {click.style(path, fg='cyan')}."
        super().__init__(f"Permission denied while accessing {click.style(path, fg='magenta')}.", hint)


class ResourceNotFoundError(AgentsMCPError):
    """Raised when a requested resource cannot be found."""
    emoji = "🔍"
    
    def __init__(self, resource: str, cmd: str | None = None):
        hint = f"Double‑check the name or path. "
        if cmd:
            hint += f"Run {click.style(f'agentsmcp {cmd} --help', fg='cyan')} for the correct syntax."
        super().__init__(f"The requested {click.style(resource, fg='magenta')} could not be found.", hint)


class VersionCompatibilityError(AgentsMCPError):
    """Raised when there's a version compatibility issue."""
    emoji = "⚙️"
    
    def __init__(self, required: str, current: str):
        hint = (
            f"Upgrade with {click.style('pip install -U agentsmcp', fg='cyan')} "
            f"or install the compatible version of the server."
        )
        super().__init__(
            f"Version mismatch – required {click.style(required, fg='magenta')}, "
            f"but you have {click.style(current, fg='magenta')}.",
            hint,
        )


class TaskExecutionError(AgentsMCPError):
    """Raised when task execution fails."""
    emoji = "💥"
    
    def __init__(self, details: str, task: str | None = None):
        hint = "Try with a simpler task or check your configuration."
        if task:
            cmd_suggestion = f'agentsmcp run simple "{task}" --cost-sensitive'
            hint = f"Try running: {click.style(cmd_suggestion, fg='cyan')}"
        super().__init__(f"Task execution failed – {details}", hint)


class AuthenticationError(AgentsMCPError):
    """Raised when authentication fails."""
    emoji = "🔐"
    
    def __init__(self, service: str):
        hint = f"Check your API keys in {click.style('agentsmcp config show', fg='cyan')}."
        super().__init__(f"Authentication failed for {click.style(service, fg='magenta')}.", hint)


class RateLimitError(AgentsMCPError):
    """Raised when API rate limits are exceeded."""
    emoji = "⏱️"
    
    def __init__(self, service: str, retry_after: str | None = None):
        hint = "Wait a moment and try again"
        if retry_after:
            hint += f" (retry after {retry_after})"
        hint += f" or use {click.style('--cost-sensitive', fg='cyan')} to reduce API calls."
        super().__init__(f"Rate limit exceeded for {click.style(service, fg='magenta')}.", hint)


def require_option(value, name: str, cmd: str | None = None):
    """Helper function to validate required options."""
    if value is None:
        raise MissingParameterError(name, cmd)
    return value


def suggest_command(invalid_cmd: str) -> str:
    """Suggest the closest valid command based on user input."""
    # Common command mappings
    suggestions = {
        'start': 'run simple',
        'exec': 'run simple', 
        'execute': 'run simple',
        'launch': 'run interactive',
        'chat': 'run interactive',
        'talk': 'run interactive',
        'price': 'monitor costs',
        'cost': 'monitor costs',
        'spend': 'monitor costs',
        'money': 'monitor budget',
        'ai': 'knowledge models',
        'model': 'knowledge models',
        'llm': 'knowledge models',
        'learn': 'knowledge rag',
        'search': 'knowledge rag',
        'configure': 'init config',
        'settings': 'config show',
        'install': 'init setup',
        'create': 'init setup',
        'new': 'init setup',
    }
    
    return suggestions.get(invalid_cmd.lower(), 'init setup')