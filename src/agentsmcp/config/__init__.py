"""
AgentsMCP Configuration Package

This package provides a robust configuration system for AgentsMCP with:
- User preference profiles for role-based optimization
- Task classification and team composition mapping  
- Environment detection for API keys and system capabilities
- Smart defaults for out-of-the-box operation
- Configuration loading with YAML and environment overrides
"""

from .defaults import (
    DEFAULT_CONFIG, Provider, CostModel, DefaultConfig,
    EmbedderModel, VectorBackend, NotificationLevel,
    RAGConfig, RAGEmbedderConfig, RAGVectorStoreConfig,
    RAGIngestionConfig, RAGFreshnessPolicyConfig
)
from .env_detector import detect_api_keys, list_available_models, detect_system_info, detect_rag_capabilities
from .loader import ConfigLoader, get_config
from .role_preferences import PreferenceProfile, PreferenceSet, DEFAULT_PROFILES, load_profiles
from .task_classifier import TaskClassifier, TaskMapping

# Re-export runtime configuration models from the dedicated runtime module.
from agentsmcp.runtime_config import (
    AgentConfig,
    ProviderType,
    ProviderConfig,
    Config,
    ServerConfig,
    StorageConfig,
    StorageType,
    TransportConfig,
    RAGConfig,
    MCPServerConfig,
)

__all__ = [
    # Defaults
    "DEFAULT_CONFIG", "Provider", "CostModel", "DefaultConfig",
    
    # RAG Configuration
    "EmbedderModel", "VectorBackend", "NotificationLevel",
    "RAGConfig", "RAGEmbedderConfig", "RAGVectorStoreConfig",
    "RAGIngestionConfig", "RAGFreshnessPolicyConfig",
    
    # Environment detection
    "detect_api_keys", "list_available_models", "detect_system_info", "detect_rag_capabilities",
    
    # Configuration loading
    "ConfigLoader", "get_config",

    # Role preferences
    "PreferenceProfile", "PreferenceSet", "DEFAULT_PROFILES", "load_profiles",

    # Task classification
    "TaskClassifier", "TaskMapping",

    # Runtime config (Pydantic v2 models used by the app)
    "AgentConfig", "ProviderType", "ProviderConfig", "Config",
    "ServerConfig", "StorageConfig", "StorageType", "TransportConfig", "RAGConfig", "MCPServerConfig",
]
