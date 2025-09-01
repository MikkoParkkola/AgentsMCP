import os
import tempfile
import re
import logging
import secrets
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .lazy_loading import lazy_import, memoized_property

# Lazy imports for heavy dependencies
yaml = lazy_import('yaml')
pydantic = lazy_import('pydantic')
pydantic_settings = lazy_import('pydantic_settings')


def _setup_yaml_representers():
    """Setup YAML representers for enum types to ensure proper serialization."""
    def represent_enum(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data.value)
    
    yaml.add_representer(StorageType, represent_enum)
    yaml.add_representer(TransportType, represent_enum)
    yaml.add_representer(ProviderType, represent_enum)


def generate_local_jwt_secret() -> str:
    """Generate a secure random JWT secret for local development."""
    return secrets.token_hex(32)


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


class ServerConfig(pydantic.BaseModel):
    host: str = pydantic.Field(default="localhost", description="Server host")
    port: int = pydantic.Field(default=8000, description="Server port")
    cors_origins: List[str] = pydantic.Field(
        default_factory=lambda: ["*"], description="CORS allowed origins"
    )

    @pydantic.field_validator("port")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class TransportConfig(pydantic.BaseModel):
    type: TransportType = TransportType.HTTP
    config: Dict[str, Any] = pydantic.Field(default_factory=dict)


class StorageConfig(pydantic.BaseModel):
    type: StorageType = StorageType.MEMORY
    config: Dict[str, Any] = pydantic.Field(default_factory=dict)

    @pydantic.field_validator("config")
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


class ToolConfig(pydantic.BaseModel):
    name: str
    type: str
    config: Dict[str, Any] = pydantic.Field(default_factory=dict)
    enabled: bool = True


class ProviderType(str, Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    OLLAMA_TURBO = "ollama-turbo"
    CUSTOM = "custom"


# Initialize YAML representers after enum definitions
_setup_yaml_representers()


class ProviderConfig(pydantic.BaseModel):
    name: ProviderType
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class MCPServerConfig(pydantic.BaseModel):
    """Config for a single MCP server.

    Supports simple stdio (command) or URL-based transports (sse/websocket).
    """

    name: str
    enabled: bool = True
    transport: Optional[str] = pydantic.Field(
        default="stdio", description="Transport: stdio|sse|websocket"
    )
    command: Optional[List[str]] = pydantic.Field(
        default=None, description="Executable + args for stdio transport"
    )
    url: Optional[str] = pydantic.Field(default=None, description="URL for sse/websocket")
    env: Dict[str, str] = pydantic.Field(default_factory=dict)
    cwd: Optional[str] = None


    # moved up


class AgentConfig(pydantic.BaseModel):
    type: str
    model: Optional[str] = None
    # Optional prioritized list of models; first is preferred when no explicit model is set
    model_priority: List[str] = pydantic.Field(default_factory=list)
    provider: ProviderType = pydantic.Field(default=ProviderType.OPENAI)
    api_base: Optional[str] = pydantic.Field(
        default=None,
        description="Override API base URL (e.g., OpenRouter: https://openrouter.ai/api/v1)",
    )
    api_key_env: Optional[str] = pydantic.Field(
        default=None, description="Env var name to read API key from (for CUSTOM)"
    )

    max_tokens: int = pydantic.Field(
        default=4000, description="Maximum tokens for agent responses"
    )
    temperature: float = pydantic.Field(default=0.7, description="Model temperature")
    timeout: int = pydantic.Field(default=300, description="Agent timeout in seconds")
    tools: List[str] = pydantic.Field(
        default_factory=list, description="Available tools for this agent"
    )
    system_prompt: Optional[str] = None
    mcp: List[str] = pydantic.Field(
        default_factory=list,
        description=(
            "List of MCP server names this agent can access via the generic mcp_call tool."
        ),
    )

    @pydantic.field_validator("temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v

    @pydantic.field_validator("timeout")
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v


class RAGConfig(pydantic.BaseModel):
    embedding_model: str = pydantic.Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = pydantic.Field(default=512, description="Document chunk size")
    chunk_overlap: int = pydantic.Field(default=50, description="Chunk overlap size")
    max_results: int = pydantic.Field(default=10, description="Maximum retrieval results")
    similarity_threshold: float = pydantic.Field(
        default=0.7, description="Similarity threshold for retrieval"
    )

    @pydantic.field_validator("chunk_size")
    def validate_chunk_size(cls, v):
        if v <= 0:
            raise ValueError("Chunk size must be positive")
        return v

    @pydantic.field_validator("similarity_threshold")
    def validate_similarity_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v


class Config(pydantic_settings.BaseSettings):
    """Main configuration for AgentsMCP."""

    server: ServerConfig = pydantic.Field(default_factory=ServerConfig)
    transport: TransportConfig = pydantic.Field(default_factory=TransportConfig)
    storage: StorageConfig = pydantic.Field(default_factory=StorageConfig)
    rag: RAGConfig = pydantic.Field(default_factory=RAGConfig)

    # Optional per-provider credentials/base URLs
    providers: Dict[str, ProviderConfig] = pydantic.Field(
        default_factory=lambda: {
            ProviderType.OLLAMA_TURBO.value: ProviderConfig(
                name=ProviderType.OLLAMA_TURBO,
                api_base="https://ollama.com/",
                api_key=os.getenv("OLLAMA_API_KEY"),
            )
        }
    )

    agents: Dict[str, AgentConfig] = pydantic.Field(
        default_factory=lambda: {
            "business_analyst": AgentConfig(type="business_analyst", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a business analyst. Elicit requirements, define acceptance criteria, clarify scope, and translate needs into engineering-ready tasks. Conduct market research and competitive analysis to inform product decisions.", tools=["filesystem", "bash", "git", "web_search"]),
            "backend_engineer": AgentConfig(type="backend_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a backend engineer. Design and implement robust services, data models, persistence layers, and APIs with performance and security in mind. Research best practices and emerging technologies.", tools=["filesystem", "bash", "git", "web_search"]),
            "web_frontend_engineer": AgentConfig(type="web_frontend_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a web frontend engineer. Build accessible, responsive, and maintainable UI components with great UX. Stay current with frontend frameworks, design systems, and web standards.", tools=["filesystem", "bash", "git", "web_search"]),
            "api_engineer": AgentConfig(type="api_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are an API engineer. Define contracts/ICDs, versioning, and error semantics; ensure clarity, stability, and testability. Research API best practices and industry standards.", tools=["filesystem", "bash", "git", "web_search"]),
            "tui_frontend_engineer": AgentConfig(type="tui_frontend_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a TUI frontend engineer. Design/implement terminal UIs with clean layout, great keyboard interaction, and broad terminal compatibility. Research terminal technologies and TUI frameworks.", tools=["filesystem", "bash", "git", "web_search"]),
            "backend_qa_engineer": AgentConfig(type="backend_qa_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a backend QA engineer. Design and execute tests for services and data layers, covering contracts, errors, and load scenarios. Research testing frameworks and quality assurance best practices.", tools=["filesystem", "bash", "git", "web_search"]),
            "web_frontend_qa_engineer": AgentConfig(type="web_frontend_qa_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a web frontend QA engineer. Validate accessibility, rendering, and interaction across browsers; prevent UX regressions. Research browser compatibility and testing tools.", tools=["filesystem", "bash", "git", "web_search"]),
            "tui_frontend_qa_engineer": AgentConfig(type="tui_frontend_qa_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a TUI frontend QA engineer. Test TUI across terminals, inputs, and edge cases; ensure reliable behavior. Research terminal compatibility and TUI testing strategies.", tools=["filesystem", "bash", "git", "web_search"]),
            "chief_qa_engineer": AgentConfig(type="chief_qa_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a chief QA engineer. Define QA strategy, quality gates, and approve releases; drive continuous quality improvements. Research industry QA trends and best practices.", tools=["filesystem", "bash", "git", "web_search"]),
            "it_lawyer": AgentConfig(type="it_lawyer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are an IT lawyer. Advise on licensing, privacy/GDPR, and compliance; flag legal risks and propose mitigations. Research current regulations and legal precedents.", tools=["filesystem", "bash", "git", "web_search"]),
            "marketing_manager": AgentConfig(type="marketing_manager", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a marketing manager. Craft positioning, messaging, and content/SEO plans aligned to audience and product goals. Conduct market research and competitor analysis.", tools=["filesystem", "bash", "git", "web_search"]),
            "ci_cd_engineer": AgentConfig(type="ci_cd_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a CI/CD engineer. Design reliable build/test/deploy pipelines and safe release flows. Research DevOps tools, practices, and infrastructure technologies.", tools=["filesystem", "bash", "git", "web_search"]),
            "dev_tooling_engineer": AgentConfig(type="dev_tooling_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a developer tooling engineer. Improve developer experience with effective tooling and automation. Research development tools, IDEs, and productivity solutions.", tools=["filesystem", "bash", "git", "web_search"]),
            "data_analyst": AgentConfig(type="data_analyst", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a data analyst. Perform exploratory analysis, build metrics and dashboards, and communicate insights. Research data visualization tools and analytical techniques.", tools=["filesystem", "bash", "git", "web_search"]),
            "data_scientist": AgentConfig(type="data_scientist", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a data scientist. Design experiments and models to answer questions and validate hypotheses. Research machine learning techniques and statistical methods.", tools=["filesystem", "bash", "git", "web_search"]),
            "ml_scientist": AgentConfig(type="ml_scientist", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a machine learning scientist. Explore and evaluate novel ML approaches and research directions. Stay current with latest ML research papers and emerging techniques.", tools=["filesystem", "bash", "git", "web_search"]),
            "ml_engineer": AgentConfig(type="ml_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a machine learning engineer. Build reliable training, data, and inference systems for ML models. Research MLOps tools and production ML best practices.", tools=["filesystem", "bash", "git", "web_search"]),
            
            # Product & Design Agents
            "product_manager": AgentConfig(type="product_manager", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a product manager. Define product vision, roadmaps, user stories, and requirements. Prioritize features based on user value and business impact. Coordinate with design and engineering teams to deliver successful products.", tools=["filesystem", "bash", "git", "web_search"]),
            "ux_ui_designer": AgentConfig(type="ux_ui_designer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a UX/UI designer. Design user-centered interfaces, create wireframes, prototypes, and design systems. Focus on usability, accessibility, and visual design. Work with user research to create intuitive and engaging experiences.", tools=["filesystem", "bash", "git", "web_search"]),
            "tui_ux_designer": AgentConfig(type="tui_ux_designer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a specialized TUI UX designer. Design terminal-based user interfaces optimized for keyboard navigation, ASCII art, and text-based interactions. Focus on information architecture, layout efficiency, and terminal compatibility across different environments.", tools=["filesystem", "bash", "git", "web_search"]),
            
            # Research Agents
            "user_researcher": AgentConfig(type="user_researcher", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a user researcher. Conduct user interviews, surveys, and usability studies. Research target user personas, behaviors, needs, and pain points. Provide insights to inform product and design decisions through qualitative and quantitative research methods.", tools=["filesystem", "bash", "git", "web_search"]),
            "market_researcher": AgentConfig(type="market_researcher", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a market researcher. Analyze markets, competitors, trends, and opportunities. Gather real-time market data, news, and competitive intelligence. Provide strategic insights about market positioning, pricing, and go-to-market strategies.", tools=["filesystem", "bash", "git", "web_search"]),
            
            # Additional Software Development Agents
            "security_engineer": AgentConfig(type="security_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a security engineer. Conduct security reviews, vulnerability assessments, and penetration testing. Implement security controls, encryption, and secure coding practices. Stay updated on latest threats and security best practices.", tools=["filesystem", "bash", "git", "web_search"]),
            "site_reliability_engineer": AgentConfig(type="site_reliability_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a site reliability engineer (SRE). Design and maintain scalable, reliable systems. Implement monitoring, alerting, and incident response. Focus on availability, latency, performance, and capacity planning.", tools=["filesystem", "bash", "git", "web_search"]),
            "database_engineer": AgentConfig(type="database_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a database engineer. Design, optimize, and maintain database systems. Handle schema design, query optimization, replication, backups, and performance tuning. Work with both SQL and NoSQL databases.", tools=["filesystem", "bash", "git"]),
            "mobile_engineer": AgentConfig(type="mobile_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a mobile engineer. Develop native and cross-platform mobile applications. Focus on mobile UX patterns, performance optimization, platform-specific features, and app store guidelines.", tools=["filesystem", "bash", "git"]),
            "performance_engineer": AgentConfig(type="performance_engineer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a performance engineer. Analyze and optimize system performance, identify bottlenecks, conduct load testing, and implement performance monitoring. Focus on scalability, efficiency, and resource optimization.", tools=["filesystem", "bash", "git", "web_search"]),
            "accessibility_specialist": AgentConfig(type="accessibility_specialist", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are an accessibility specialist. Ensure digital products are accessible to users with disabilities. Implement WCAG guidelines, conduct accessibility audits, and design inclusive user experiences. Work with design and engineering teams to build accessible solutions.", tools=["filesystem", "bash", "web_search"]),
            "technical_writer": AgentConfig(type="technical_writer", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a technical writer. Create clear, comprehensive documentation including API docs, user guides, tutorials, and technical specifications. Collaborate with engineers and product teams to make complex technical concepts accessible.", tools=["filesystem", "git", "web_search"]),
            "solutions_architect": AgentConfig(type="solutions_architect", model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", system_prompt="You are a solutions architect. Design high-level system architecture, technology stacks, and integration patterns. Balance technical requirements with business needs, scalability, and maintainability. Create architectural documentation and technical specifications.", tools=["filesystem", "git", "web_search"]),
        }
    )

    tools: List[ToolConfig] = pydantic.Field(
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

    # Optional MCP servers (disabled by default)
    mcp: List[MCPServerConfig] = pydantic.Field(default_factory=list)
    # Expose REST endpoints for MCP management (/mcp) when true
    mcp_api_enabled: bool = pydantic.Field(default=False, description="Enable /mcp REST API for managing MCP servers")
    # Transport feature flags
    mcp_stdio_enabled: bool = pydantic.Field(default=True, description="Allow stdio transport for MCP")
    mcp_ws_enabled: bool = pydantic.Field(default=False, description="Allow WebSocket transport for MCP")
    mcp_sse_enabled: bool = pydantic.Field(default=False, description="Allow SSE transport for MCP")

    # Discovery & coordination flags (AD5)
    discovery_enabled: bool = pydantic.Field(default=False, description="Enable local discovery announcer/registry")
    discovery_allowlist: List[str] = pydantic.Field(default_factory=list, description="Allowed agent IDs or names for coordination")
    discovery_token: Optional[str] = pydantic.Field(default=None, description="Optional shared secret for discovery entries")
    discovery_registry_endpoint: Optional[str] = pydantic.Field(default=None, description="Remote registry HTTP API endpoint")
    discovery_registry_token: Optional[str] = pydantic.Field(default=None, description="Authentication token for remote registry")
    
    # Security configuration (AD5)
    security_enabled: bool = pydantic.Field(default=True, description="Enable security features (signatures, TLS validation)")
    private_key_path: Optional[str] = pydantic.Field(default=None, description="Path to PEM-encoded private key file")
    public_key_path: Optional[str] = pydantic.Field(default=None, description="Path to PEM-encoded public key file")
    key_rotation_interval_hours: int = pydantic.Field(default=24, description="Key rotation interval in hours")
    tls_cert_path: Optional[str] = pydantic.Field(default=None, description="Path to TLS certificate file")
    tls_key_path: Optional[str] = pydantic.Field(default=None, description="Path to TLS private key file")
    require_tls: bool = pydantic.Field(default=False, description="Require TLS for all connections")
    jwt_secret: str = pydantic.Field(
        default_factory=generate_local_jwt_secret,
        description="JWT signing secret - auto-generated for local dev, set JWT_SECRET for production"
    )
    jwt_issuer: str = pydantic.Field(default="agents-mcp", description="JWT issuer name")
    jwt_expiry_minutes: int = pydantic.Field(default=60, description="JWT token expiry in minutes")
    trust_store_path: Optional[str] = pydantic.Field(default=None, description="Path to trusted public keys directory")

    @pydantic.field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: str) -> str:
        """Enforce a strong JWT secret, auto-generating for local development."""
        DEFAULT_PLACEHOLDER = "change-me-in-production"
        
        # If explicitly set to placeholder, reject it
        if v == DEFAULT_PLACEHOLDER:
            raise ValueError(
                "JWT_SECRET cannot be the default 'change-me-in-production'. "
                "Provide a production-ready secret via the JWT_SECRET environment variable."
            )
        
        # Validate minimum length
        if len(v) < 32:
            raise ValueError(
                f"JWT_SECRET is too short ({len(v)} chars). "
                "Use at least a 32-character random string."
            )
        
        # Check for whitespace
        if re.search(r"\s", v):
            raise ValueError("JWT_SECRET must not contain whitespace characters.")
        
        # Log appropriate message
        logger = logging.getLogger(__name__)
        env_jwt = os.getenv("JWT_SECRET")
        if env_jwt:
            logger.info("JWT secret loaded from JWT_SECRET environment variable.")
        else:
            logger.info("JWT secret auto-generated for local development session.")
        
        return v

    # Web UI (WUI6): enable/disable minimal built-in dashboard
    ui_enabled: bool = pydantic.Field(default=False, description="Mount /ui static dashboard when true")

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
        # Determine default config path under user's home directory
        home_cfg = Path(os.path.expanduser("~/.agentsmcp/agentsmcp.yaml"))
        explicit = path or (Path(os.getenv("AGENTSMCP_CONFIG")) if os.getenv("AGENTSMCP_CONFIG") else None)
        cfg_path = explicit or (home_cfg if home_cfg.exists() else None)
        if cfg_path and Path(cfg_path).exists():
            cfg = cls.from_file(Path(cfg_path))
            try:
                cfg._ensure_human_role_agents()
                cfg.save_to_file(Path(cfg_path))
            except Exception:
                pass
            return cfg
        # Fallback: legacy cwd file if present
        legacy = Path("agentsmcp.yaml")
        if legacy.exists():
            cfg = cls.from_file(legacy)
            try:
                cfg._ensure_human_role_agents()
                cfg.save_to_file(legacy)
            except Exception:
                pass
            return cfg
        # Merge env overrides onto defaults
        cfg = cls.from_env()
        try:
            cfg._ensure_human_role_agents()
        except Exception:
            pass
        return cfg

    def save_to_file(self, path: Path):
        """Save configuration to a YAML file."""
        # Default to ~/.agentsmcp/agentsmcp.yaml if a bare filename was provided
        if not path.is_absolute():
            path = Path(os.path.expanduser("~/.agentsmcp")) / path
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(exclude_unset=True), f, default_flow_style=False, indent=2
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

    @staticmethod
    def default_config_path() -> Path:
        """Return the default per-user config path under ~/.agentsmcp."""
        from .paths import default_user_config_path
        return default_user_config_path()


    def _ensure_human_role_agents(self) -> None:
        roles = [
            "business_analyst","backend_engineer","web_frontend_engineer","api_engineer","tui_frontend_engineer",
            "backend_qa_engineer","web_frontend_qa_engineer","tui_frontend_qa_engineer","chief_qa_engineer",
            "it_lawyer","marketing_manager","ci_cd_engineer","dev_tooling_engineer",
            "data_analyst","data_scientist","ml_scientist","ml_engineer",
            "product_manager","ux_ui_designer","tui_ux_designer","user_researcher","market_researcher",
            "security_engineer","site_reliability_engineer","database_engineer","mobile_engineer",
            "performance_engineer","accessibility_specialist","technical_writer","solutions_architect",
        ]
        self.providers.setdefault(
            ProviderType.OLLAMA_TURBO.value,
            ProviderConfig(name=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/", api_key=os.getenv("OLLAMA_API_KEY")),
        )
        for legacy in ["codex","claude","ollama"]:
            if legacy in self.agents:
                try:
                    del self.agents[legacy]
                except Exception:
                    pass
        for r in roles:
            ac = self.agents.get(r)
            if not ac:
                self.agents[r] = AgentConfig(
                    type=r, model="gpt-oss:120b", provider=ProviderType.OLLAMA_TURBO, api_base="https://ollama.com/",
                )
            else:
                ac.provider = ProviderType.OLLAMA_TURBO
                ac.model = ac.model or "gpt-oss:120b"
                ac.api_base = "https://ollama.com/"
