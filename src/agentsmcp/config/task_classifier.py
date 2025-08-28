"""
Task classifier module
~~~~~~~~~~~~~~~~~~~~~~

Maps goal/descriptive task names to an ordered list of agent roles that
should be instantiated by the system.  The mapping can be loaded from a
YAML file and is validated with Pydantic.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
#   Domain model
# --------------------------------------------------------------------------- #

class TaskMapping(BaseModel):
    """
    Validation model for a single task → roles mapping.

    Attributes
    ----------
    task : str
        Human‑readable task description (e.g. "Write a blog post").
    roles : List[str]
        Ordered list of role names expected to carry out the task.
    """

    task: str = Field(..., min_length=1)
    roles: List[str] = Field(..., min_items=1)

    @field_validator("roles")
    @classmethod
    def _ensure_non_empty(cls, value: List[str]) -> List[str]:
        result = []
        for role in value:
            if not role.strip():
                raise ValueError("Role name cannot be empty or whitespace")
            result.append(role.strip())
        return result


class TaskClassifier(BaseModel):
    """
    Aggregates multiple ``TaskMapping`` objects.

    Attributes
    ----------
    mappings : Dict[str, List[str]]
        Mapping from task name to ordered list of role names.
    """

    mappings: Dict[str, List[str]] = Field(
        {}, description="Task → role mapping dictionary"
    )

    @staticmethod
    def _load_yaml(file: Path) -> Any:
        try:
            with file.open(encoding="utf-8") as fp:
                return yaml.safe_load(fp) or {}
        except FileNotFoundError:
            return {}
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Error parsing YAML at {file}: {exc}") from exc

    @classmethod
    def load_from_file(
        cls,
        yaml_file: Optional[Path] = None,
        *,
        fallback: bool = True,
    ) -> "TaskClassifier":
        """
        Load the mapping from a YAML file or use an embedded default.

        Parameters
        ----------
        yaml_file : Path | None
            Path to a YAML config file. If ``None`` the default file
            inside the package is used.
        fallback : bool
            When ``True``, embed a basic mapping when the file is missing.
        """
        if yaml_file is None:
            yaml_file = Path(__file__).parent / "default_task_map.yaml"

        data = cls._load_yaml(yaml_file)

        # ``data`` can be either a list or a mapping.
        if isinstance(data, list):
            mappings: Dict[str, List[str]] = {}
            for item in data:
                try:
                    tm = TaskMapping(**item)
                    mappings[tm.task] = tm.roles
                except Exception as exc:
                    raise ValueError(
                        f"Invalid mapping entry {item!r} in {yaml_file}"
                    ) from exc
        elif isinstance(data, dict):
            # Ensure keys are non‑empty and values are lists of strings
            mappings = {}
            for task, roles in data.items():
                if not task.strip():
                    raise ValueError("Task key cannot be empty")
                if not isinstance(roles, list) or not roles:
                    raise ValueError(
                        f"Roles for task `{task}` must be a non-empty list"
                    )
                # Stripping role names
                mappings[task.strip()] = [r.strip() for r in roles]
        else:
            raise TypeError(f"Invalid YAML format in {yaml_file}")

        if fallback and not mappings:
            # Provide a minimal default mapping
            mappings = {
                "Write a blog post": ["Researcher", "Writer", "Editor"],
                "Plan a marketing campaign": [
                    "Strategist",
                    "Designer",
                    "Copywriter",
                ],
                "Code review": ["ARCHITECT", "CODER", "QA"],
                "Deploy application": ["CODER", "MERGE_BOT", "QA"],
                "Write documentation": ["DOCS", "QA"],
                "Analyze performance": ["ARCHITECT", "METRICS_COLLECTOR", "QA"],
                "Process improvement": ["PROCESS_COACH", "ARCHITECT", "QA"],
            }

        return cls(mappings=mappings)

    def get_roles(self, task_name: str) -> List[str]:
        """
        Return the ordered list of roles for a task.

        Parameters
        ----------
        task_name : str
            The task description.

        Raises
        ------
        KeyError
            If the task is unknown.
        """
        try:
            return self.mappings[task_name]
        except KeyError as exc:
            raise KeyError(
                f"Task `{task_name}` is not defined in the task mapping."
            ) from exc


# --------------------------------------------------------------------------- #
#   Public API
# --------------------------------------------------------------------------- #

__all__ = ["TaskMapping", "TaskClassifier"]