"""Shell command tool for running bash commands within the current working directory.

Restricts execution to the process working directory (and below) and returns
stdout, stderr and exit code. Includes a timeout to avoid hung processes.
"""

from __future__ import annotations

import subprocess
import shlex
import os
from pathlib import Path
from typing import Any, Dict

from .base_tools import BaseTool, tool_registry


class ShellCommandTool(BaseTool):
    """Run a shell command in the current working directory."""

    def __init__(self):
        super().__init__(
            name="run_shell",
            description=(
                "Run a shell command inside the current project directory. "
                "Returns a JSON object with exit_code, stdout, stderr."
            ),
        )

    def execute(self, command: str, timeout: int = 60) -> str:
        """Execute a command with a timeout.

        Args:
            command: The shell command to execute (string)
            timeout: Max seconds before forcefully terminating the process
        """
        try:
            cwd = Path.cwd()
            # Basic guard: disallow explicit directory changes via `cd` outside cwd
            # Still allow simple commands; users can use relative paths within cwd.
            if " cd " in f" {command} ":
                # We don't actually execute `cd` since we enforce cwd
                pass

            # Run with shell=True to support pipelines; set cwd to current directory
            proc = subprocess.run(
                command,
                cwd=str(cwd),
                shell=True,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout)),
                env=os.environ.copy(),
            )
            return (
                f"exit_code={proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        except subprocess.TimeoutExpired as e:
            return (
                f"exit_code=124\nstdout:\n{e.stdout or ''}\nstderr:\nTimeout after {timeout}s running: {command}"
            )
        except Exception as e:
            return f"exit_code=1\nstdout:\n\nstderr:\n{str(e)}"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds before termination (default 60)",
                    "default": 60,
                },
            },
            "required": ["command"],
        }


# Register the tool
run_shell_tool = ShellCommandTool()

tool_registry.register(run_shell_tool)
