"""
AgentsMCP - CLI-driven MCP agent system with extensible RAG pipeline.
"""

__version__ = "1.0.0"

from .config import Config
from .server import AgentServer  
from .agent_manager import AgentManager
from .rag.pipeline import RAGPipeline

__all__ = [
    "Config",
    "AgentServer", 
    "AgentManager",
    "RAGPipeline",
]
