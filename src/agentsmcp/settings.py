"""Environment-based configuration using pydantic-settings.

This module defines Settings classes that load configuration from environment
variables and optional .env files, and provides helpers to construct the
runtime Config model used across the app.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .runtime_config import (
    Config,
    RAGConfig,
    ServerConfig,
    StorageConfig,
    StorageType,
    TransportConfig,
)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AGENTS_", case_sensitive=False
    )

    # Logging
    log_level: str = Field(default="INFO", description="Application log level")
    log_format: str = Field(default="json", description="Log format: json or text")

    # Server
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8000)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    # Storage
    storage_type: StorageType = Field(default=StorageType.MEMORY)
    sqlite_path: Optional[str] = Field(default="agentsmcp.db")
    postgres_host: Optional[str] = None
    postgres_port: Optional[int] = None
    postgres_database: Optional[str] = None
    postgres_username: Optional[str] = None
    postgres_password: Optional[str] = None
    redis_host: Optional[str] = None
    redis_port: Optional[int] = None

    # RAG
    rag_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    rag_chunk_size: int = Field(default=512)
    rag_chunk_overlap: int = Field(default=50)
    rag_max_results: int = Field(default=10)
    rag_similarity_threshold: float = Field(default=0.7)

    # Observability (default off for minimal startup)
    prometheus_enabled: bool = Field(default=False)

    # REST API toggles
    mcp_api_enabled: bool = Field(default=False, description="Enable /mcp REST endpoints")
    # MCP transport toggles
    mcp_stdio_enabled: bool = Field(default=True)
    mcp_ws_enabled: bool = Field(default=False)
    mcp_sse_enabled: bool = Field(default=False)

    def to_runtime_config(self, base: Optional[Config] = None) -> Config:
        """Merge environment settings into a runtime Config object.

        If a base Config is provided (e.g., loaded from YAML), environment
        variables take precedence.
        """
        if base is None:
            base = Config()

        # Server
        base.server = ServerConfig(
            host=self.server_host or base.server.host,
            port=self.server_port or base.server.port,
            cors_origins=self.cors_origins or base.server.cors_origins,
        )

        # Storage
        storage_config = {}
        if self.storage_type == StorageType.SQLITE:
            storage_config = {"database_path": self.sqlite_path or "agentsmcp.db"}
        elif self.storage_type == StorageType.POSTGRESQL:
            storage_config = {
                "host": self.postgres_host,
                "port": self.postgres_port or 5432,
                "database": self.postgres_database,
                "username": self.postgres_username,
                "password": self.postgres_password,
            }
        elif self.storage_type == StorageType.REDIS:
            storage_config = {
                "host": self.redis_host or "localhost",
                "port": self.redis_port or 6379,
            }

        base.storage = StorageConfig(type=self.storage_type, config=storage_config)

        # RAG
        base.rag = RAGConfig(
            embedding_model=self.rag_embedding_model,
            chunk_size=self.rag_chunk_size,
            chunk_overlap=self.rag_chunk_overlap,
            max_results=self.rag_max_results,
            similarity_threshold=self.rag_similarity_threshold,
        )

        # Transport unchanged for now
        base.transport = base.transport or TransportConfig()

        # Feature toggles
        try:
            # Prefer explicit env toggle when set
            base.mcp_api_enabled = bool(self.mcp_api_enabled or getattr(base, "mcp_api_enabled", False))  # type: ignore[attr-defined]
        except Exception:
            pass

        # Transport flags
        try:
            base.mcp_stdio_enabled = bool(self.mcp_stdio_enabled)  # type: ignore[attr-defined]
            base.mcp_ws_enabled = bool(self.mcp_ws_enabled)  # type: ignore[attr-defined]
            base.mcp_sse_enabled = bool(self.mcp_sse_enabled)  # type: ignore[attr-defined]
        except Exception:
            pass

        return base

# Rebuild model to resolve forward references
AppSettings.model_rebuild()
