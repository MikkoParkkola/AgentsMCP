"""
Configuration loader
~~~~~~~~~~~~~~~~~~~~

Combines :pymod:`defaults`, environment detection, and user‑supplied
overrides into a single validated configuration object.  Supports:

* YAML files for user‑defined settings
* Environment variable overrides (`ENVVAR_*` style)
* Built‑in default values
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Mapping, List, Literal

from pydantic import BaseModel, Field, ValidationError
# Note: this module does not use BaseSettings directly; avoid import-time
# failures in environments without pydantic-settings by providing a safe stub.
try:  # pragma: no cover - defensive import
    from pydantic_settings import BaseSettings  # type: ignore
except Exception:  # pragma: no cover
    class BaseSettings:  # type: ignore
        pass

from .defaults import DEFAULT_CONFIG as DEFAULTS
from .defaults import Provider, RAGConfig
from .env_detector import detect_api_keys, detect_system_info, list_available_models, detect_rag_capabilities
from .role_preferences import PreferenceSet, PreferenceProfile, load_profiles, DEFAULT_PROFILES
from .task_classifier import TaskClassifier

# --------------------------------------------------------------------------- #
#   Main configuration model
# --------------------------------------------------------------------------- #

class ConfigLoader(BaseModel):
    """
    Root configuration model for the AgentsMCP system.
    It pulls values from defaults, environment detection and YAML overrides.
    """

    # --- LLM / Provider ----------------------------------------------------

    provider: Provider = Field(
        DEFAULTS.provider,
        description="Which LLM provider will be used",
    )
    model: str = Field(
        DEFAULTS.model,
        description="Model name for the chosen provider",
    )

    # --- System level -----------------------------------------------------

    concurrent_agents: int = Field(
        DEFAULTS.concurrent_agents,
        ge=1,
        description="Maximum concurrent agent work units",
    )
    memory_limit_mb: int = Field(
        DEFAULTS.memory_limit_mb,
        ge=512,
        description="Soft memory limit in MB",
    )
    system_info: Dict[str, Any] = Field(
        None,
        description="Auto‑detected system info",
    )
    available_models: List[str] = Field(
        None,
        description="List of usable models based on detected keys",
    )
    api_keys: Dict[str, str] = Field(
        None,
        description="API keys captured from the environment",
    )
    rag_capabilities: Dict[str, bool] = Field(
        None,
        description="Auto-detected RAG library and service availability",
    )

    # --- Role preferences --------------------------------------------------

    role_pref_profiles: List[PreferenceProfile] = Field(
        None,
        description="List of loaded user preference profiles",
    )

    # --- Task mapping -----------------------------------------------------

    task_classifier: TaskClassifier = Field(
        None,
        description="Task → role mapping service",
    )

    # --- Misc ----------------------------------------------------------------

    # Use Literal for strict validation (Pydantic v2-compatible)
    cost_profile: Literal["high", "medium", "low"] = Field(
        DEFAULTS.cost_profile.value,
        description="Cost tier per 1 K tokens",
    )

    # --- RAG Configuration ------------------------------------------------

    rag: RAGConfig = Field(
        default_factory=lambda: DEFAULTS.rag,
        description="Retrieval Augmented Generation configuration",
    )

    model_config = {
        "extra": "forbid",  # disallow unknown keys
    }

    # ------------------------------------------------------------------------- #
    #   Construction helpers
    # ------------------------------------------------------------------------- #

    @staticmethod
    def _load_yaml(file_path: Path) -> Mapping[str, Any]:
        try:
            with file_path.open(encoding="utf-8") as fp:
                return yaml.safe_load(fp) or {}
        except FileNotFoundError:
            return {}
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Failed to parse YAML at {file_path}") from exc

    @classmethod
    def _merge_dicts(cls, base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
        """
        Merge two dictionaries with override taking precedence.
        """
        result = dict(base)
        for key, val in override.items():
            if isinstance(val, dict) and key in result:
                result[key] = cls._merge_dicts(result[key], val)
            else:
                result[key] = val
        return result

    @classmethod
    def from_env(cls) -> "ConfigLoader":
        """
        Load configuration from defaults, environment variables and
        user YAML overrides.  Raises ``ValidationError`` if anything
        is invalid.
        """
        # 1. Base defaults from code
        cfg: Dict[str, Any] = DEFAULTS.model_dump()

        # 2. Inject system / detector values
        cfg["system_info"] = detect_system_info()
        cfg["available_models"] = list_available_models()
        cfg["api_keys"] = detect_api_keys()
        cfg["rag_capabilities"] = detect_rag_capabilities()

        # 3. Override from environment variables
        #    Prefix: AGENTSMCP_
        env_overrides: Dict[str, Any] = {}
        for key, val in os.environ.items():
            if key.startswith("AGENTSMCP_"):
                # Normalize key to snake_case
                norm = key[len("AGENTSMCP_") :].lower()
                env_overrides[norm] = val

        # 4. Load user overrides from YAML (if present)
        #    The user can specify the path in env var or use the default
        yaml_path = Path(env_overrides.get("role_preferences_path") or cfg["role_preferences_path"])
        yaml_content = cls._load_yaml(yaml_path)
        if yaml_content:
            cfg = cls._merge_dicts(cfg, yaml_content)

        # 5. Merge environment overrides on top of YAML+defaults
        cfg = cls._merge_dicts(cfg, env_overrides)

        # 6. Validate & populate complex fields
        #    * role preference profiles
        pref_profiles: List[PreferenceProfile] = []
        if "role_pref_profiles" in cfg:
            pref_profiles = [PreferenceProfile(**p) for p in cfg["role_pref_profiles"]]
        else:
            pref_profiles = DEFAULT_PROFILES

        #    * task mapping
        task_file = Path(cfg.get("task_mapping_path", "configs/task_map.yaml"))
        task_classifier = TaskClassifier.load_from_file(task_file)

        # 7. Build final dict for Pydantic
        final_dict = {
            **cfg,
            "role_pref_profiles": pref_profiles,
            "task_classifier": task_classifier,
        }

        # 8. Construct model (this will validate)
        try:
            return cls(**final_dict)
        except ValidationError as ev:
            raise RuntimeError("Configuration validation failed") from ev

    @classmethod
    def load(cls, yaml_override: Path | None = None) -> "ConfigLoader":
        """
        Public entry point – read optional YAML overrides and
        process environment / default values.
        """
        cfg = cls.from_env()

        if yaml_override is not None:
            yaml_overrides = cls._load_yaml(yaml_override)
            merged = cls._merge_dicts(cfg.model_dump(exclude_unset=True), yaml_overrides)
            try:
                cfg = cls(**merged)
            except ValidationError as ev:
                raise RuntimeError(f"Failed to apply override at {yaml_override}") from ev
        return cfg

# --------------------------------------------------------------------------- #
#   Convenience function
# --------------------------------------------------------------------------- #

def get_config() -> ConfigLoader:
    """
    Singleton accessor that lazily loads the configuration on first call.
    Subsequent imports will share the same instance.

    Returns
    -------
    ConfigLoader
        Global configuration for the application.
    """
    if not hasattr(get_config, "_config"):
        get_config._config = ConfigLoader.from_env()
    return get_config._config


__all__ = ["ConfigLoader", "get_config"]
