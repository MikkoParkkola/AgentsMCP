"""
Smart defaults module
~~~~~~~~~~~~~~~~~~~~~

Provides sane, out‑of‑the‑box configuration values that work well
for non‑technical users.  These defaults can be overridden by the
user via environment variables or custom YAML files.
"""

from __future__ import annotations

from typing import Dict, List
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

# --------------------------------------------------------------------------- #
#   Enums for provider choice
# --------------------------------------------------------------------------- #

class Provider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"


class CostModel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --------------------------------------------------------------------------- #
#   RAG-specific enums
# --------------------------------------------------------------------------- #

class EmbedderModel(str, Enum):
    SENTENCE_TRANSFORMERS = "all-MiniLM-L6-v2"
    OLLAMA_LLAMA3 = "ollama:llama3:embedding"


class VectorBackend(str, Enum):
    FAISS = "faiss"
    LANCEDB = "lancedb"


class NotificationLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QueryExpansionMode(str, Enum):
    DISABLED = "disabled"
    SIMPLE = "simple"
    LLM_BASED = "llm_based"


# --------------------------------------------------------------------------- #
#   RAG configuration models
# --------------------------------------------------------------------------- #

class RAGEmbedderConfig(BaseModel):
    """Configuration for the RAG embedder component."""
    
    model: EmbedderModel = Field(
        EmbedderModel.SENTENCE_TRANSFORMERS,
        description="Embedding model to use for document vectors",
    )
    batch_size: int = Field(
        128,
        ge=1,
        description="Batch size for embedding generation",
    )


class RAGVectorStoreConfig(BaseModel):
    """Configuration for the RAG vector storage backend."""
    
    backend: VectorBackend = Field(
        VectorBackend.FAISS,
        description="Vector storage backend to use",
    )
    path: str = Field(
        "$HOME/.agentsmcp/knowl_base/faiss.index",
        description="Path to vector store index file",
    )


class RAGIngestionConfig(BaseModel):
    """Configuration for RAG document ingestion."""
    
    chunk_size: int = Field(
        1000,
        ge=100,
        description="Size of document chunks in tokens/lines",
    )
    overlap: int = Field(
        200,
        ge=0,
        description="Overlap between chunks in tokens",
    )
    max_file_bytes: int = Field(
        5_000_000,
        ge=1024,
        description="Maximum file size for ingestion (bytes)",
    )
    encoding: str = Field(
        "utf-8",
        description="File encoding for text ingestion",
    )


class RAGQueryInterfaceConfig(BaseModel):
    """Configuration for the RAG query interface and agent integration."""
    
    enabled: bool = Field(
        True,
        description="Enable RAG query interface for agent integration",
    )
    expand_query: bool = Field(
        False,
        description="Enable query expansion for better context retrieval",
    )
    expansion_mode: QueryExpansionMode = Field(
        QueryExpansionMode.SIMPLE,
        description="Query expansion strategy",
    )
    top_k: int = Field(
        3,
        ge=1,
        le=50,
        description="Number of top documents to retrieve for context",
    )
    relevance_threshold: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for including documents in context",
    )
    cache_ttl: int = Field(
        600,
        ge=0,
        description="Cache time-to-live for search results (seconds, 0 to disable)",
    )
    cache_maxsize: int = Field(
        128,
        ge=1,
        description="Maximum number of cached query results",
    )
    context_format: str = Field(
        "Document {idx}: {text}",
        description="Format template for context documents",
    )
    max_context_length: int = Field(
        2000,
        ge=100,
        description="Maximum total length of context to include in prompts",
    )


class RAGFreshnessPolicyConfig(BaseModel):
    """Configuration for RAG knowledge freshness management."""
    
    ttl_days: int = Field(
        90,
        ge=1,
        description="Knowledge expiry time in days",
    )
    notification_level: NotificationLevel = Field(
        NotificationLevel.WARNING,
        description="Notification level for stale content",
    )
    auto_remove_stale: bool = Field(
        True,
        description="Automatically remove stale content after TTL",
    )
    confirmation_prompt: bool = Field(
        True,
        description="Prompt for confirmation before removing stale content",
    )


class RAGConfig(BaseModel):
    """Complete RAG configuration model."""
    
    enabled: bool = Field(
        False,
        description="Enable/disable RAG functionality (disabled by default)",
    )
    embedder: RAGEmbedderConfig = Field(
        default_factory=RAGEmbedderConfig,
        description="Embedder configuration",
    )
    vector_store: RAGVectorStoreConfig = Field(
        default_factory=RAGVectorStoreConfig,
        description="Vector store configuration",
    )
    ingestion: RAGIngestionConfig = Field(
        default_factory=RAGIngestionConfig,
        description="Document ingestion configuration",
    )
    query_interface: RAGQueryInterfaceConfig = Field(
        default_factory=RAGQueryInterfaceConfig,
        description="Query interface and agent integration configuration",
    )
    freshness_policy: RAGFreshnessPolicyConfig = Field(
        default_factory=RAGFreshnessPolicyConfig,
        description="Knowledge freshness management configuration",
    )


# --------------------------------------------------------------------------- #
#   Base configuration model
# --------------------------------------------------------------------------- #

class DefaultConfig(BaseModel):
    """
    Root default configuration model used by the main loading routine.

    Attributes
    ----------
    provider : Provider
        Default LLM provider.
    model : str
        Default model name for the chosen provider.
    concurrent_agents : int
        Maximum number of agents that can run simultaneously.
    memory_limit_mb : int
        Soft memory limit in megabytes for heavy inference.
    role_preferences_path : str
        Relative path to the YAML file containing role preferences.
    task_mapping_path : str
        Relative path to the task‑mapping YAML file.
    cost_profile : CostModel
        Default expected cost per 1 k tokens (high/medium/low).
    """

    provider: Provider = Field(
        Provider.OPENAI,
        description="Default LLM provider",
    )
    model: str = Field(
        "gpt-4o-mini", description="Default language model identifier"
    )
    concurrent_agents: int = Field(
        4,
        ge=1,
        description="Number of agents that can run concurrently",
    )
    memory_limit_mb: int = Field(
        8192,
        ge=512,
        description="Soft memory limit per inference job (MB)",
    )
    role_preferences_path: str = Field(
        "configs/role_preferences.yaml",
        description="YAML path containing role preference profiles",
    )
    task_mapping_path: str = Field(
        "configs/task_map.yaml",
        description="YAML path describing task → role mappings",
    )
    cost_profile: CostModel = Field(
        CostModel.MEDIUM,
        description="Default cost tier per 1 K tokens",
    )
    rag: RAGConfig = Field(
        default_factory=RAGConfig,
        description="Retrieval Augmented Generation configuration (disabled by default)",
    )

    # Pydantic v2 configuration
    model_config = ConfigDict(frozen=True)


# --------------------------------------------------------------------------- #
#   Concrete instance
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG: DefaultConfig = DefaultConfig()

__all__ = [
    "Provider", 
    "CostModel", 
    "EmbedderModel", 
    "VectorBackend", 
    "NotificationLevel",
    "RAGEmbedderConfig",
    "RAGVectorStoreConfig", 
    "RAGIngestionConfig",
    "RAGFreshnessPolicyConfig",
    "RAGConfig",
    "DefaultConfig", 
    "DEFAULT_CONFIG"
]
