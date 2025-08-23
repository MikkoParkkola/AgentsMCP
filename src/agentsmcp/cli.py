"""CLI entry point for AgentsMCP."""

import click

from . import __version__
from .placeholder import add as add_fn


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="agentsmcp")
def main() -> None:
    """AgentsMCP CLI."""
    # This function is the console script entry point.
    # Subcommands are defined below.
    pass


@main.command("add")
@click.argument("a", type=int)
@click.argument("b", type=int)
def add_cmd(a: int, b: int) -> None:
    """Add two integers and print the result."""
    click.echo(str(add_fn(a, b)))


if __name__ == "__main__":
    main()
