from __future__ import annotations

"""Centralized user data directories for AgentsMCP.

Creates and returns per-user directories under ~/.agentsmcp for:
- state/: runtime state (pid files, ephemeral state that should persist across sessions)
- tmp/: temp working files (safe to delete)
- logs/: log output (if file logging enabled)
- cache/: caches
- configs/: auxiliary JSON/YAML configs generated at runtime
"""

import os
from pathlib import Path
from typing import Tuple


def data_dir() -> Path:
    return Path(os.path.expanduser("~/.agentsmcp"))


def get_app_data_dir() -> Path:
    """Alias for data_dir() to match expected interface."""
    return data_dir()


def subdirs() -> Tuple[Path, Path, Path, Path, Path]:
    base = data_dir()
    return (
        base / "state",
        base / "tmp",
        base / "logs",
        base / "cache",
        base / "configs",
    )


def ensure_dirs() -> None:
    base = data_dir()
    base.mkdir(parents=True, exist_ok=True)
    for d in subdirs():
        d.mkdir(parents=True, exist_ok=True)


def pid_file_path() -> Path:
    return data_dir() / "state" / "agentsmcp.pid"


def default_user_config_path() -> Path:
    return data_dir() / "agentsmcp.yaml"


def temp_dir() -> Path:
    return data_dir() / "tmp"

