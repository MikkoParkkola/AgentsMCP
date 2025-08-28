"""Enhanced CLI with friendly error handling."""

import sys
import click
from typing import Optional
from pathlib import Path

from agentsmcp.errors import (
    AgentsMCPError,
    InvalidCommandError, 
    MissingParameterError,
    ConfigError,
    NetworkError,
    PermissionError,
    ResourceNotFoundError,
    VersionCompatibilityError,
    TaskExecutionError,
    AuthenticationError,
    RateLimitError,
    require_option,
    suggest_command
)
from agentsmcp.intelligent_suggestions import get_suggestion_system, display_suggestions


class EnhancedAgentsMCPCLI(click.Group):
    """Enhanced Click group with friendly error handling and suggestions."""

    def get_command(self, ctx, cmd_name):
        """Override to provide suggestions for invalid commands."""
        rv = super().get_command(ctx, cmd_name)
        if rv is not None:
            return rv
            
        # Command not found - provide intelligent suggestions
        suggestion_system = get_suggestion_system()
        suggestions = suggestion_system.suggest_for_invalid_command(cmd_name)
        
        # Display suggestions before raising error
        if suggestions:
            click.echo()  # Extra spacing
            display_suggestions(suggestions, "💡 Did you mean?")
        
        raise InvalidCommandError(cmd_name)

    def invoke(self, ctx):
        """Wrap the regular invoke with enhanced error handling."""
        try:
            return super().invoke(ctx)
        
        except click.ClickException as ce:
            # Already a ClickException – check if it's our enhanced type
            if isinstance(ce, AgentsMCPError):
                ce.show()
            else:
                # Unknown ClickException – add a gentle tip
                click.echo(
                    f"❓  {click.style('Oops! Something went wrong.', fg='red', bold=True)}\n"
                    f"   {click.style(str(ce), fg='yellow')}\n"
                    "💡 Run the command with --help to see the expected usage.",
                    err=True,
                )
            ctx.exit(1)

        except KeyboardInterrupt:
            click.echo("\n👋 Operation cancelled by user.", err=True)
            ctx.exit(130)  # Standard exit code for SIGINT
            
        except Exception as exc:
            # SystemExit with code 0 is success - don't show as error
            if isinstance(exc, SystemExit) and exc.code == 0:
                ctx.exit(0)
                
            # Anything else (programming error, unexpected library error...)
            # Show helpful message but preserve debugging capability
            if "--debug" in sys.argv or ctx.find_root().params.get('debug'):
                raise  # Let the full traceback bubble up

            click.echo(
                f"💥  {click.style('Unexpected error:', fg='red', bold=True)} {exc}\n"
                f"💡 If this keeps happening, please:\n"
                f"   • Re-run with {click.style('--debug', fg='cyan')} to see the full traceback\n"
                f"   • Report the issue at {click.style('https://github.com/yourorg/agentsmcp/issues', fg='cyan')}",
                err=True,
            )
            ctx.exit(1)

    def main(self, *args, **kwargs):
        """Override main to add debug option handling."""
        try:
            return super().main(*args, **kwargs)
        except SystemExit as e:
            # Preserve system exit codes - don't show error for successful exits
            if e.code != 0:
                raise e
            else:
                # Success exit - just exit cleanly
                sys.exit(0)
        except Exception:
            # Last resort error handling
            if "--debug" in sys.argv:
                raise
            click.echo("💥 Fatal error occurred. Use --debug for details.", err=True)
            sys.exit(1)


def handle_common_errors(func):
    """Decorator to handle common errors in command functions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise ResourceNotFoundError(f"file {e.filename}")
        except OSError as e:
            if e.errno == 13:  # Permission denied
                raise PermissionError(str(e.filename or "resource"))
            raise
        except ConnectionError as e:
            raise NetworkError(str(e))
        except ImportError as e:
            if "cost" in str(e).lower():
                raise ConfigError("Cost tracking features not available. Install with: pip install agentsmcp[cost]")
            elif "rag" in str(e).lower():
                raise ConfigError("RAG features not available. Install with: pip install agentsmcp[rag]")
            raise
    return wrapper


def with_intelligent_suggestions(func):
    """Decorator to add intelligent suggestions after command execution."""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract command name from function context
        import inspect
        frame = inspect.currentframe()
        try:
            # Get the click context
            ctx = None
            for arg in args:
                if isinstance(arg, click.Context):
                    ctx = arg
                    break
            
            # Execute the original function
            result = func(*args, **kwargs)
            
            # Get command name from context or function name
            command_name = func.__name__ if not ctx else ctx.info_name
            if ctx and ctx.parent:
                # Build full command path
                parent_names = []
                current_ctx = ctx.parent
                while current_ctx and current_ctx.info_name != 'agentsmcp':
                    parent_names.append(current_ctx.info_name)
                    current_ctx = current_ctx.parent
                if parent_names:
                    command_name = ' '.join(reversed(parent_names)) + ' ' + command_name
            
            # Record successful usage and show suggestions
            suggestion_system = get_suggestion_system()
            suggestion_system.record_command_usage(command_name, success=True)
            
            # Show next-step suggestions (but only in advanced mode or if specifically helpful)
            should_show_suggestions = True
            if ctx and ctx.obj and not ctx.obj.get('advanced', False):
                # In simple mode, only show suggestions for key moments
                key_moments = ['init setup', 'run simple', 'monitor costs']
                should_show_suggestions = any(key in command_name for key in key_moments)
            
            if should_show_suggestions:
                suggestions = suggestion_system.suggest_next_actions(command_name)
                if suggestions:
                    display_suggestions(suggestions[:3], "💡 What's next?")  # Limit to 3
                    
            return result
            
        except Exception as e:
            # Record failed usage
            if 'command_name' in locals():
                suggestion_system = get_suggestion_system()
                suggestion_system.record_command_usage(command_name, success=False)
            raise
        finally:
            if frame:
                del frame
                
    return wrapper


def validate_task_input(task: str) -> str:
    """Validate and clean task input."""
    if not task or not task.strip():
        raise MissingParameterError("task", "run simple")
    
    task = task.strip()
    if len(task) > 1000:
        raise TaskExecutionError("Task description too long (max 1000 characters)")
    
    return task


def check_config_exists(config_path: Optional[str] = None) -> Path:
    """Check if config exists and provide helpful error if not."""
    if config_path:
        path = Path(config_path)
    else:
        from agentsmcp.paths import default_user_config_path
        path = default_user_config_path()
    
    if not path.exists():
        raise ConfigError(
            f"Configuration file not found at {path}. "
            "Run 'agentsmcp init setup' to create one."
        )
    
    return path


def friendly_exception_handler(exc_type, exc_value, exc_traceback):
    """Global exception handler for better user experience."""
    if exc_type is KeyboardInterrupt:
        click.echo("\n👋 Operation cancelled by user.", err=True)
        sys.exit(130)
    
    # Don't handle if we're in debug mode
    if "--debug" in sys.argv:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Convert some common exceptions to our friendly format
    if exc_type is FileNotFoundError:
        error = ResourceNotFoundError(f"file {exc_value.filename}")
        error.show()
        sys.exit(1)
    elif exc_type is PermissionError:
        error = PermissionError(str(exc_value.filename or "resource"))
        error.show()
        sys.exit(1)
    elif exc_type is ConnectionError:
        error = NetworkError(str(exc_value))
        error.show()
        sys.exit(1)
    
    # Fallback to default handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


# Install the friendly exception handler globally
sys.excepthook = friendly_exception_handler