"""
Agent Management System

This module provides a flexible, modular system for managing AI agents in AgentsMCP.
Agent descriptions are stored in JSON files and loaded dynamically, allowing for
easy modification and extension of agent capabilities.
"""

from .agent_loader import AgentLoader, AgentDescription, get_agent_loader

__all__ = [
    "AgentLoader",
    "AgentDescription", 
    "get_agent_loader"
]