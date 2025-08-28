"""
Role preferences module
~~~~~~~~~~~~~~~~~~~~~~~

Defines user preference profiles that influence agent optimisation
(e.g. speed vs. quality, cost vs. capability).  The profiles are
validated with Pydantic and can be loaded from a YAML file.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from enum import Enum
from typing import List, Mapping, Optional, Any

from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
#   Enumerations
# --------------------------------------------------------------------------- #

class PreferenceSet(str, Enum):
    """High‑level preference groups."""
    FAST = "fast"
    QUALITY = "quality"
    COST = "cost"
    BALANCED = "balanced"
    CUSTOM = "custom"


# --------------------------------------------------------------------------- #
#   Domain model
# --------------------------------------------------------------------------- #

class PreferenceProfile(BaseModel):
    """
    Validation model for a single preference profile.

    Attributes
    ----------
    name : str
        Human readable profile name.
    profile : PreferenceSet
        One of the predefined groups or ``CUSTOM``.
    speed : float
        Weight (0–1) for speed optimisation.  1 gives priority to speed.
    quality : float
        Weight (0–1) for quality optimisation.
    cost : float
        Weight (0–1) for cost optimisation.
    capability : float
        Weight (0–1) for capability (model sophistication).
    """

    name: str = Field(..., min_length=1, description="Profile name")
    profile: PreferenceSet = Field(
        PreferenceSet.BALANCED,
        description="High‑level preference group",
    )
    speed: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Weight for speed optimisation",
    )
    quality: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Weight for quality optimisation",
    )
    cost: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Weight for cost optimisation",
    )
    capability: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description="Weight for capability optimisation",
    )

    @model_validator(mode='after')
    def _check_sum_is_unit(self):
        """Ensure that the sum of weights equals 1.0."""
        weights = ["speed", "quality", "cost", "capability"]
        total = sum(getattr(self, w) for w in weights)
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"Total weight must sum to 1.0 (got {total:.2f});"
                f" profile `{self.name}` has {total:.2f}"
            )
        return self


# --------------------------------------------------------------------------- #
#   Helper functions
# --------------------------------------------------------------------------- #

def _load_yaml(path: Path) -> Any:
    try:  # pragma: no cover - YAML reading rarely fails during tests
        with path.open(encoding="utf-8") as fp:
            return yaml.safe_load(fp) or {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Failed to parse YAML config at {path}: {exc}") from exc


def load_profiles(
    file_path: Optional[Path] = None,
    /,
    *,
    strict: bool = False,
) -> List[PreferenceProfile]:
    """
    Load preference profiles from a YAML file.

    Parameters
    ----------
    file_path : Path | None
        Path to the YAML file. If ``None`` defaults are used.
    strict : bool
        If ``True``, any malformed profile will raise an exception.

    Returns
    -------
    List[PreferenceProfile]
        Validated profile objects.
    """
    if file_path is None:
        # Use a built‑in default file in the package if present.
        file_path = Path(__file__).parent / "default_profiles.yaml"

    raw = _load_yaml(file_path)
    profiles_data = raw.get("profiles") or raw  # support top‑level mapping

    if not isinstance(profiles_data, list):
        raise TypeError(f"Expected a list of profiles; got {type(profiles_data).__name__}")

    profiles: List[PreferenceProfile] = []
    for idx, data in enumerate(profiles_data):
        try:
            pref = PreferenceProfile(**data)
            profiles.append(pref)
        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"Failed to load profile at index {idx} from {file_path}"
                ) from exc
            # Skip malformed profile but log warning
            print(
                f"Warning: ignoring malformed preference profile at index {idx}"
                f" from {file_path}: {exc}"
            )
    return profiles


# --------------------------------------------------------------------------- #
#   Default profiles
# --------------------------------------------------------------------------- #

DEFAULT_PROFILES: List[PreferenceProfile] = [
    PreferenceProfile(
        name="Fast",
        profile=PreferenceSet.FAST,
        speed=0.6,
        quality=0.1,
        cost=0.2,
        capability=0.1,
    ),
    PreferenceProfile(
        name="High Quality",
        profile=PreferenceSet.QUALITY,
        speed=0.1,
        quality=0.6,
        cost=0.1,
        capability=0.2,
    ),
    PreferenceProfile(
        name="Cost‑Efficient",
        profile=PreferenceSet.COST,
        speed=0.2,
        quality=0.1,
        cost=0.6,
        capability=0.1,
    ),
    PreferenceProfile(
        name="Balanced",
        profile=PreferenceSet.BALANCED,
        speed=0.25,
        quality=0.25,
        cost=0.25,
        capability=0.25,
    ),
]

# --------------------------------------------------------------------------- #
#   Public API
# --------------------------------------------------------------------------- #

__all__ = [
    "PreferenceSet",
    "PreferenceProfile",
    "load_profiles",
    "DEFAULT_PROFILES",
]