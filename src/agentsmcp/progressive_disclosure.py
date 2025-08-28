"""Progressive disclosure system for AgentsMCP CLI."""

from __future__ import annotations
import click
import functools
from typing import Callable, Any, Optional


class ProgressiveDisclosureGroup(click.Group):
    """A click.Group that adds progressive disclosure capabilities."""

    def invoke(self, ctx):
        """Store the advanced flag in context for all subcommands."""
        if ctx.obj is None:
            ctx.obj = {}
        
        # Check if advanced mode is enabled
        advanced = ctx.params.get('advanced', False)
        ctx.obj['advanced'] = advanced
        
        # Store original params for help filtering
        if 'original_params' not in ctx.obj:
            ctx.obj['original_params'] = {}
        
        return super().invoke(ctx)

    def get_command(self, ctx, cmd_name):
        """Override to add advanced flag to all commands."""
        cmd = super().get_command(ctx, cmd_name)
        if cmd and not hasattr(cmd, '_progressive_disclosure_wrapped'):
            cmd = self._wrap_command_with_advanced_flag(cmd)
        return cmd

    def _wrap_command_with_advanced_flag(self, cmd: click.Command) -> click.Command:
        """Add the global --advanced flag to a command."""
        if any(opt.name == "advanced" for opt in cmd.params):
            return cmd
        
        @click.pass_context
        def new_callback(ctx: click.Context, *args, **kwargs):
            # Store advanced flag
            if ctx.obj is None:
                ctx.obj = {}
            
            advanced = kwargs.pop('advanced', False)
            ctx.obj['advanced'] = advanced
            
            # Run original command
            result = ctx.invoke(cmd.callback, *args, **kwargs)
            
            # Show feature discovery hint in simple mode
            if not advanced and not ctx.resilient_parsing:
                click.echo(
                    click.style(
                        "\n💡 Tip: Use --advanced (-A) to see more options and power-user features.",
                        fg='blue', dim=True
                    ),
                    err=True,
                )
            
            return result

        # Add advanced flag
        advanced_opt = click.Option(
            ["-A", "--advanced"],
            is_flag=True,
            help="Show advanced options and enable power-user features.",
            hidden=False,
        )

        new_params = [advanced_opt] + list(cmd.params)
        
        new_cmd = click.Command(
            name=cmd.name,
            callback=new_callback,
            params=new_params,
            help=cmd.help,
            short_help=cmd.short_help,
            epilog=cmd.epilog,
            context_settings=cmd.context_settings,
            deprecated=cmd.deprecated,
        )
        
        new_cmd._progressive_disclosure_wrapped = True
        return new_cmd


def advanced_option(
    *param_decls: str,
    advanced: bool = False,
    default: Any = None,
    **attrs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Enhanced click.option that supports progressive disclosure.
    
    Parameters
    ----------
    advanced: bool
        If True, this option is hidden in simple mode
    default: Any
        Smart default value for beginners
    """
    
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        # Modify help text to indicate level
        if advanced and 'help' in attrs:
            attrs['help'] = f"{attrs['help']} (advanced)"
        elif not advanced and 'help' in attrs:
            attrs['help'] = f"{attrs['help']} (simple)"
        
        # Create the click option decorator
        option_decorator = click.option(*param_decls, default=default, **attrs)
        
        # Apply the option decorator to get the updated function
        updated_f = option_decorator(f)
        
        # Find the option that was just added and mark it
        if hasattr(updated_f, '__click_params__'):
            # The most recently added parameter is at the beginning
            latest_param = updated_f.__click_params__[0]
            latest_param._is_advanced = advanced
            latest_param._original_help = attrs.get('help', '')
        
        return updated_f
    
    return decorator


def filter_options_by_mode(ctx: click.Context, options: list) -> list:
    """Filter options based on current mode (simple/advanced)."""
    if not ctx.obj:
        return options
    
    advanced_mode = ctx.obj.get('advanced', False)
    
    if advanced_mode:
        return options
    else:
        # Simple mode - filter out advanced options
        return [opt for opt in options if not getattr(opt, '_is_advanced', False)]


# Monkey patch click.Command to respect progressive disclosure
_original_format_help_text = click.Command.format_help_text
_original_get_params = click.Command.get_params

def _patched_format_help_text(self, ctx, formatter):
    """Override help formatting to hide advanced options in simple mode."""
    # Store original params
    original_params = self.params
    
    try:
        # Check if we're in advanced mode
        advanced_mode = False
        if ctx.obj and ctx.obj.get('advanced', False):
            advanced_mode = True
        
        # Check if --advanced flag was passed in current context params
        if ctx.params and ctx.params.get('advanced', False):
            advanced_mode = True
            
        # In simple mode, filter out advanced options
        if not advanced_mode:
            filtered_params = []
            for p in self.params:
                if getattr(p, '_is_advanced', False):
                    # Skip advanced options in simple mode
                    continue
                filtered_params.append(p)
            self.params = filtered_params
        
        return _original_format_help_text(self, ctx, formatter)
    finally:
        # Restore original params
        self.params = original_params

# Apply the patch
click.Command.format_help_text = _patched_format_help_text


class SmartDefaults:
    """Smart defaults for beginner-friendly CLI experience."""
    
    @staticmethod
    def get_complexity_default() -> str:
        """Default complexity for task execution."""
        return "moderate"  # Balance between capable and cost-effective
    
    @staticmethod
    def get_timeout_default() -> int:
        """Default timeout for operations."""
        return 300  # 5 minutes - enough for most tasks
    
    @staticmethod
    def get_port_default() -> int:
        """Default port for server operations."""
        return 8000  # Standard development port
    
    @staticmethod
    def get_log_level_default() -> str:
        """Default log level."""
        return "INFO"  # Informative but not overwhelming
    
    @staticmethod
    def should_cost_optimize() -> bool:
        """Default cost optimization setting."""
        return True  # Be cost-conscious by default


def contextual_help_hint(command: str, context: str = "") -> str:
    """Generate contextual help hints based on command and context."""
    hints = {
        "init": "💡 Need help getting started? Run: agentsmcp init setup",
        "run": "💡 For interactive mode, try: agentsmcp run interactive", 
        "monitor": "💡 Track costs with: agentsmcp monitor costs",
        "knowledge": "💡 Manage your knowledge base: agentsmcp knowledge rag",
        "server": "💡 Start the API server: agentsmcp server start",
        "config": "💡 Edit configuration: agentsmcp config edit",
    }
    
    return hints.get(command, "💡 Use --help to see available options")


def suggest_next_steps(completed_command: str) -> str:
    """Suggest logical next steps after completing a command."""
    suggestions = {
        "init setup": "Next: agentsmcp run simple 'your first task'",
        "run simple": "Next: agentsmcp monitor costs (to check usage)",
        "run interactive": "Next: agentsmcp knowledge models (to explore AI models)",
        "monitor costs": "Next: agentsmcp monitor budget --check",
        "knowledge rag": "Next: agentsmcp run simple 'ask about your knowledge'",
        "server start": "Next: Open http://localhost:8000 in your browser",
    }
    
    return suggestions.get(completed_command, "")