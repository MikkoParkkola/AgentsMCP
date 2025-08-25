"""
AgentsMCP - CLI-driven MCP agent system with extensible RAG pipeline.
"""

__version__ = "0.1.1"

# Import memory subsystem for easy access
try:
    from . import memory
    __all__ = ["__version__", "memory"]
except ImportError:
    # Memory subsystem optional dependencies may not be available
    __all__ = ["__version__"]
