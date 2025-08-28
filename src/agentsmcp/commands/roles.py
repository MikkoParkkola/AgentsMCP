# -*- coding: utf-8 -*-
"""
agentsmcp.commands.roles
~~~~~~~~~~~~~~~~~~~~~~~~

Click command‑group that exposes the *role‑based orchestration* API to the end‑user.

The group provides five sub‑commands:

* ``roles list``                – show every role that lives in ``ROLE_REGISTRY``
* ``roles task``                – fire a new task through ``MainCoordinator``
* ``roles status``              – poll the status / result of a previously launched task
* ``roles test``                – run the golden‑tests for *all* roles
* ``roles validate``            – validate a JSON envelope against the contract schemas
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import uuid
from typing import Any, Dict, Optional

import click

# --------------------------------------------------------------------------- #
#   Orchestration / Role imports (they already exist in the repo)
# --------------------------------------------------------------------------- #
from agentsmcp.roles import ROLE_REGISTRY               # dict[str, RoleBase]
from agentsmcp.orchestration.coordinator import MainCoordinator
from agentsmcp.models import TaskEnvelopeV1, ResultEnvelopeV1
from agentsmcp.testing.golden_tests import main as run_golden_tests_main
from agentsmcp.contracts.validation import validate_task_envelope, validate_result_envelope

# --------------------------------------------------------------------------- #
#   Helper utilities
# --------------------------------------------------------------------------- #
def _print_header(msg: str) -> None:
    """Print a colourful header with an emoji."""
    click.secho(f"🚀  {msg}", fg="bright_blue", bold=True)


def _pretty_json(data: Dict[str, Any]) -> str:
    """Indent JSON for terminal display."""
    return json.dumps(data, indent=2, sort_keys=True)


def _run_async(coro):
    """Run an async coroutine from a sync click command."""
    return asyncio.get_event_loop().run_until_complete(coro)


# --------------------------------------------------------------------------- #
#   Click command group
# --------------------------------------------------------------------------- #
@click.group(name="roles")
def roles() -> None:
    """🛠️  Role‑based orchestration commands."""
    pass


# --------------------------------------------------------------------------- #
#   roles list
# --------------------------------------------------------------------------- #
@roles.command(name="list")
def _list() -> None:
    """Show every role that is registered and a short description of its capabilities."""
    _print_header("Available Roles")
    if not ROLE_REGISTRY:
        click.secho("⚠️  No roles have been registered yet.", fg="yellow")
        return

    for role_name, role_cls in ROLE_REGISTRY.items():
        # Get role info from class methods
        responsibilities = role_cls.responsibilities() if hasattr(role_cls, 'responsibilities') else []
        decision_rights = role_cls.decision_rights() if hasattr(role_cls, 'decision_rights') else []
        
        click.secho(f"🔹 {role_name.value}", fg="green", bold=True)
        if responsibilities:
            click.secho(f"    📋 Responsibilities: {', '.join(responsibilities)}", fg="white")
        if decision_rights:
            click.secho(f"    ⚖️  Decision Rights: {', '.join(decision_rights)}", fg="white")
        click.echo()


# --------------------------------------------------------------------------- #
#   roles task
# --------------------------------------------------------------------------- #
@roles.command(name="task")
@click.argument("objective", type=str)
@click.option(
    "--role",
    "-r",
    "role_name",
    type=str,
    help="Name of the role that should handle the task.",
)
@click.option(
    "--priority",
    "-p",
    default=3,
    type=int,
    help="Numeric priority (1=highest, 5=lowest).",
)
@click.option(
    "--context",
    "-c",
    help="Bounded context to limit task scope.",
)
def _task(objective: str, role_name: Optional[str], priority: int, context: Optional[str]) -> None:
    """
    Submit a new task to the orchestrator using the selected ROLE.

    The command prints the generated TASK_ID which can later be used with
    ``agentsmcp roles status <task-id>``.
    """
    _print_header("Submitting Task")

    # ------------------------------------------------------------------- #
    #   Build the envelope that the orchestrator expects
    # ------------------------------------------------------------------- #
    task_id = str(uuid.uuid4())
    envelope = TaskEnvelopeV1(
        id=task_id,
        objective=objective,
        priority=priority,
        bounded_context=context,
        requested_agent_type=role_name,
    )

    click.secho("🔸 Task envelope created:", fg="magenta")
    click.echo(_pretty_json(envelope.model_dump()))

    # ------------------------------------------------------------------- #
    #   Fire the task through the MainCoordinator
    # ------------------------------------------------------------------- #
    coordinator = MainCoordinator()

    async def _submit() -> str:
        # ``submit_task`` returns the same task_id we gave
        result = await coordinator.submit_task(
            objective=envelope.objective,
            bounded_context=envelope.bounded_context,
            priority=envelope.priority,
            requested_agent_type=envelope.requested_agent_type,
        )
        return result["id"]

    try:
        task_id = _run_async(_submit())
    except Exception as exc:   # pragma: no cover – defensive
        click.secho(f"❌  Failed to submit task: {exc}", fg="red")
        raise click.Abort()

    click.secho("\n✅  Task accepted!", fg="green", bold=True)
    click.secho(f"🆔  Task ID: {task_id}", fg="cyan")
    click.secho("💡  Check progress later with:", fg="white")
    click.secho(f"    agentsmcp roles status {task_id}", fg="bright_black")


# --------------------------------------------------------------------------- #
#   roles status
# --------------------------------------------------------------------------- #
@roles.command(name="status")
@click.argument("task_id", type=str)
def _status(task_id: str) -> None:
    """Query the orchestrator for the current status / result of a task."""
    _print_header(f"Status for task {task_id}")

    coordinator = MainCoordinator()

    async def _poll() -> Optional[Dict[str, Any]]:
        # ``get_result`` may return ``None`` while the task is still running.
        return await coordinator.get_result(task_id)

    try:
        result = _run_async(_poll())
    except Exception as exc:   # pragma: no cover – defensive
        click.secho(f"❌  Could not retrieve status: {exc}", fg="red")
        raise click.Abort()

    if result is None:
        click.secho("⌛  Task is still in progress.", fg="yellow")
        return

    click.secho("✅  Task completed!", fg="green", bold=True)
    click.secho("📦  Result:", fg="magenta")
    click.echo(_pretty_json(result))


# --------------------------------------------------------------------------- #
#   roles test
# --------------------------------------------------------------------------- #
@roles.command(name="test")
@click.option(
    "--role",
    "-r",
    "role_filter",
    type=str,
    default=None,
    help="If supplied, run golden‑tests only for the given role.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show the detailed stdout of each golden test.",
)
@click.option(
    "--json-dir",
    default="tests/golden",
    help="Root directory containing golden JSON test files",
)
def _test(role_filter: Optional[str], verbose: bool, json_dir: str) -> None:
    """
    Execute the *Golden Tests* suite for every role (or a single role).
    """
    _print_header("Running Golden Tests")

    # Use the golden test main function directly
    import sys
    from unittest import mock
    
    # Mock sys.argv for the golden test main function
    test_args = ["golden_tests.py", "--json-dir", json_dir]
    if role_filter:
        test_args.extend(["--role", role_filter])
    if verbose:
        test_args.append("--verbose")
    
    try:
        with mock.patch.object(sys, 'argv', test_args):
            run_golden_tests_main()
    except SystemExit as e:
        if e.code == 0:
            click.secho("\n✅  Golden‑test run finished successfully.", fg="green")
        else:
            click.secho(f"\n❌  Golden‑test run failed with exit code {e.code}.", fg="red")
            raise click.Abort()
    except Exception as exc:   # pragma: no cover – defensive
        click.secho(f"❌  Golden‑test execution failed: {exc}", fg="red")
        raise click.Abort()


# --------------------------------------------------------------------------- #
#   roles validate
# --------------------------------------------------------------------------- #
@roles.command(name="validate")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option(
    "--type",
    "envelope_type",
    type=click.Choice(["task", "result"]),
    default="task",
    help="Type of envelope to validate (task or result).",
)
def _validate(file: pathlib.Path, envelope_type: str) -> None:
    """
    Validate a JSON envelope (task or result) against the contract schemas.

    The command will exit with a non‑zero status code if validation fails.
    """
    _print_header(f"Validating {file.name}")

    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        click.secho(f"❌  Invalid JSON: {exc}", fg="red")
        raise click.Abort()

    try:
        if envelope_type == "task":
            validate_task_envelope(payload)
        else:
            validate_result_envelope(payload)
    except Exception as exc:                # pragma: no cover – defensive
        click.secho(f"❌  Contract validation failed: {exc}", fg="red")
        raise click.Abort()

    click.secho("✅  Envelope complies with the contract.", fg="green")


# --------------------------------------------------------------------------- #
#   Export the group for the top‑level CLI to pick up
# --------------------------------------------------------------------------- #
__all__ = ["roles"]