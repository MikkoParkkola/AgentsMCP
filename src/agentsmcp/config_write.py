"""Helpers to persist config changes (K2).

These helpers merge provider credentials into the active Config and write to disk
using the same YAML schema as Config.save_to_file.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config, ProviderType, ProviderConfig


def persist_provider_api_key(cfg: Config, path: Path, provider: ProviderType, api_key: str) -> Path:
    """Persist an API key for a given provider into the YAML config.

    - Updates cfg.providers dict in-memory
    - Writes merged config to the provided path (relative paths are saved under ~/.agentsmcp)
    - Returns the resolved path written
    """
    existing = cfg.providers.get(provider.value)
    if existing:
        existing.api_key = api_key
    else:
        cfg.providers[provider.value] = ProviderConfig(name=provider, api_key=api_key)

    # Reuse Config.save_to_file logic for path handling
    cfg.save_to_file(path)
    return path


__all__ = ["persist_provider_api_key"]

