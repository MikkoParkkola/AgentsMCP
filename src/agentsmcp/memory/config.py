"""
Configuration helper for the memory subsystem.

Environment variables are the primary source of truth – this mirrors the
common pattern used throughout AgentsMCP and keeps the codebase consistent
and easy to deploy in containerised environments.

All values are type‑annotated and validated with `pydantic.BaseSettings`
for safety.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import pydantic
except ImportError:
    # Fallback for environments without pydantic
    pydantic = None


if pydantic:
    class MemoryBackend(pydantic.BaseSettings):
        """
        Environment variables specific to the chosen memory backend.

        These are intentionally simple – the real implementation will validate
        and expose only what we need.
        """

        REDIS_URL: str = pydantic.Field(
            default="redis://localhost:6379/0",
            description="Redis connection URL (including database index).",
        )
        REDIS_PERSISTENCE: bool = pydantic.Field(
            default=True,
            description="Enable Redis persistence (RDB/AOF).",
        )

        POSTGRES_DSN: Optional[str] = pydantic.Field(
            default=None,
            description="PostgreSQL DSN for optional persistence.",
        )

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            env_prefix = ""
            use_enum_values = True
else:
    # Simple fallback without pydantic validation
    class MemoryBackend:
        def __init__(self):
            self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.REDIS_PERSISTENCE = os.getenv("REDIS_PERSISTENCE", "true").lower() == "true"
            self.POSTGRES_DSN = os.getenv("POSTGRES_DSN")


@dataclass
class MemoryConfig:
    """
    Consolidated configuration object that other modules import.
    """

    backend: str = field(default="redis")
    redis_url: str = field(default="redis://localhost:6379/0")
    redis_persistence: bool = field(default=True)
    postgres_dsn: Optional[str] = field(default=None)

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """
        Build a ``MemoryConfig`` instance from the process environment.

        Raises:
            ValueError: If the required values are missing.
        """
        settings = MemoryBackend()
        if not settings.REDIS_URL:
            raise ValueError("REDIS_URL must be set")
        return cls(
            backend="redis",
            redis_url=settings.REDIS_URL,
            redis_persistence=settings.REDIS_PERSISTENCE,
            postgres_dsn=settings.POSTGRES_DSN,
        )