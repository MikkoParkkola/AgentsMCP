import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class TransportType(str, Enum):
    HTTP = "http"
    WEBSOCKET = "websocket"
    STDIO = "stdio"
    SSE = "sse"


class StorageType(str, Enum):
    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    REDIS = "redis"


class ServerConfig(BaseModel):
    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=8000, description="Server port")
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"], description="CORS allowed origins"
    )

    @field_validator("port")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class TransportConfig(BaseModel):
    type: TransportType = TransportType.HTTP
    config: Dict[str, Any] = Field(default_factory=dict)


class StorageConfig(BaseModel):
    type: StorageType = StorageType.MEMORY
    config: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("config")
    def validate_storage_config(cls, v, info):
        storage_type = info.data.get("type") if info.data else None
        if storage_type == StorageType.SQLITE:
            if "database_path" not in v:
                v["database_path"] = "agentsmcp.db"
        elif storage_type == StorageType.POSTGRESQL:
            required = ["host", "port", "database", "username", "password"]
            missing = [key for key in required if key not in v]
            if missing:
                raise ValueError(f"PostgreSQL storage requires: {', '.join(missing)}")
        elif storage_type == StorageType.REDIS:
            if "host" not in v:
                v["host"] = "localhost"
            if "port" not in v:
                v["port"] = 6379
        return v


class ToolConfig(BaseModel):
    name: str
    type: str
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class AgentConfig(BaseModel):
    type: str
    model: Optional[str] = None
    max_tokens: int = Field(
        default=4000, description="Maximum tokens for agent responses"
    )
    temperature: float = Field(default=0.7, description="Model temperature")
    timeout: int = Field(default=300, description="Agent timeout in seconds")
    tools: List[str] = Field(
        default_factory=list, description="Available tools for this agent"
    )
    system_prompt: Optional[str] = None

    @field_validator("temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @field_validator("timeout")
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v


class RAGConfig(BaseModel):
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = Field(default=512, description="Document chunk size")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size")
    max_results: int = Field(default=10, description="Maximum retrieval results")
    similarity_threshold: float = Field(
        default=0.7, description="Similarity threshold for retrieval"
    )

    @field_validator("chunk_size")
    def validate_chunk_size(cls, v):
        if v <= 0:
            raise ValueError("Chunk size must be positive")
        return v

    @field_validator("similarity_threshold")
    def validate_similarity_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v


class Config(BaseModel):
    """Main configuration for AgentsMCP."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)

    agents: Dict[str, AgentConfig] = Field(
        default_factory=lambda: {
            "codex": AgentConfig(
                type="codex",
                model="gpt-4",
                system_prompt="You are a code generation and analysis expert.",
                tools=["filesystem", "git", "bash"],
            ),
            "claude": AgentConfig(
                type="claude",
                model="claude-3-sonnet",
                system_prompt=(
                    "You are a helpful AI assistant with deep reasoning capabilities."
                ),
                tools=["filesystem", "web_search"],
            ),
            "ollama": AgentConfig(
                type="ollama",
                model="llama2",
                system_prompt=(
                    "You are a cost-effective AI assistant for general tasks."
                ),
                tools=["filesystem"],
            ),
        }
    )

    tools: List[ToolConfig] = Field(
        default_factory=lambda: [
            ToolConfig(
                name="filesystem",
                type="filesystem",
                config={"allowed_paths": [tempfile.gettempdir(), "."]},
            ),
            ToolConfig(name="git", type="git", config={}),
            ToolConfig(name="bash", type="bash", config={"timeout": 60}),
            ToolConfig(name="web_search", type="web_search", config={}),
        ]
    )

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables.

        Environment variables (all optional):
          - AGENTSMCP_SERVER_HOST
          - AGENTSMCP_SERVER_PORT
          - AGENTSMCP_STORAGE_TYPE (memory|sqlite|postgresql|redis)
          - AGENTSMCP_STORAGE_DB_PATH (for sqlite)
          - AGENTSMCP_CORS_ORIGINS (comma-separated)
        """
        server_host = os.getenv("AGENTSMCP_SERVER_HOST")
        server_port = os.getenv("AGENTSMCP_SERVER_PORT")
        storage_type = os.getenv("AGENTSMCP_STORAGE_TYPE")
        sqlite_path = os.getenv("AGENTSMCP_STORAGE_DB_PATH")
        cors_origins = os.getenv("AGENTSMCP_CORS_ORIGINS")

        # start from defaults then override
        server = ServerConfig()
        if server_host:
            server.host = server_host
        if server_port:
            server.port = int(server_port)
        if cors_origins:
            server.cors_origins = [
                o.strip() for o in cors_origins.split(",") if o.strip()
            ]

        storage = StorageConfig()
        if storage_type:
            storage.type = StorageType(storage_type)
        if sqlite_path and storage.type == StorageType.SQLITE:
            storage.config["database_path"] = sqlite_path

        return cls(server=server, storage=storage)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        """Load configuration using precedence: explicit path -> env var -> env defaults -> defaults.

        If `path` not provided, uses AGENTSMCP_CONFIG if set.
        """
        cfg_path = path or (
            Path(os.getenv("AGENTSMCP_CONFIG"))
            if os.getenv("AGENTSMCP_CONFIG")
            else None
        )
        if cfg_path:
            return cls.from_file(cfg_path)
        # Merge env overrides onto defaults
        return cls.from_env()

    def save_to_file(self, path: Path):
        """Save configuration to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(
                self.dict(exclude_unset=True), f, default_flow_style=False, indent=2
            )

    def get_agent_config(self, agent_type: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent type."""
        return self.agents.get(agent_type)

    def get_tool_config(self, tool_name: str) -> Optional[ToolConfig]:
        """Get configuration for a specific tool."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
