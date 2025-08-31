"""
Managed Agent Loader - Ensures only predefined, managed agents are loaded dynamically.

This module implements controlled agent loading that prevents auto-discovery of
unmanaged agents while supporting efficient on-demand loading of known agents.
"""

import logging
from typing import Set, Dict, List, Optional, Type
from enum import Enum

from ..roles.base import BaseRole, RoleName
from ..roles.registry import RoleRegistry

logger = logging.getLogger(__name__)


class AgentLoadingStrategy(Enum):
    """Strategy for loading agents."""
    LAZY = "lazy"  # Load only when needed
    EAGER = "eager"  # Load all at startup
    SELECTIVE = "selective"  # Load based on configuration


class ManagedAgentLoader:
    """
    Controlled agent loading system that only works with predefined, managed agents.
    
    This ensures that:
    1. Only agents from the role registry can be loaded
    2. No auto-discovery or scanning of unmanaged agents
    3. Efficient on-demand loading without loading all agents at startup
    4. Clear tracking of which agents are loaded vs available
    """
    
    def __init__(self, loading_strategy: AgentLoadingStrategy = AgentLoadingStrategy.LAZY):
        """Initialize the managed agent loader."""
        self.loading_strategy = loading_strategy
        self.role_registry = RoleRegistry()
        
        # Track which agents are available vs loaded
        self.available_agents: Set[RoleName] = set(self.role_registry.ROLE_CLASSES.keys())
        self.loaded_agents: Dict[RoleName, Type[BaseRole]] = {}
        
        # Performance tracking
        self._load_count = 0
        self._load_errors: List[str] = []
        
        logger.info(f"ManagedAgentLoader initialized with {len(self.available_agents)} available agents")
        
        # Eager loading if requested
        if self.loading_strategy == AgentLoadingStrategy.EAGER:
            self._load_all_agents()
    
    def _load_all_agents(self) -> None:
        """Load all available agents (used for eager loading)."""
        logger.info("Eagerly loading all managed agents...")
        for role_name in self.available_agents:
            try:
                self._load_agent(role_name)
            except Exception as e:
                logger.warning(f"Failed to load agent {role_name}: {e}")
                self._load_errors.append(f"{role_name}: {str(e)}")
    
    def _load_agent(self, role_name: RoleName) -> Type[BaseRole]:
        """Load a specific agent if not already loaded."""
        if role_name in self.loaded_agents:
            return self.loaded_agents[role_name]
        
        # Ensure this is a managed agent
        if role_name not in self.available_agents:
            raise ValueError(f"Agent {role_name} is not a managed agent. Only predefined agents are allowed.")
        
        # Load from registry
        try:
            role_class = self.role_registry.ROLE_CLASSES[role_name]
            self.loaded_agents[role_name] = role_class
            self._load_count += 1
            
            logger.debug(f"Loaded managed agent: {role_name}")
            return role_class
            
        except Exception as e:
            error_msg = f"Failed to load managed agent {role_name}: {e}"
            logger.error(error_msg)
            self._load_errors.append(error_msg)
            raise
    
    def get_agent_class(self, role_name: RoleName) -> Type[BaseRole]:
        """Get the agent class for a role, loading it if necessary."""
        return self._load_agent(role_name)
    
    def is_agent_available(self, role_name: RoleName) -> bool:
        """Check if an agent is available (but not necessarily loaded)."""
        return role_name in self.available_agents
    
    def is_agent_loaded(self, role_name: RoleName) -> bool:
        """Check if an agent is currently loaded."""
        return role_name in self.loaded_agents
    
    def get_available_agents(self) -> List[RoleName]:
        """Get list of all available managed agents."""
        return list(self.available_agents)
    
    def get_loaded_agents(self) -> List[RoleName]:
        """Get list of currently loaded agents."""
        return list(self.loaded_agents.keys())
    
    def validate_team_composition(self, requested_roles: List[str]) -> List[RoleName]:
        """
        Validate that all requested roles are managed agents and return RoleName objects.
        
        Args:
            requested_roles: List of role names as strings
            
        Returns:
            List of validated RoleName objects
            
        Raises:
            ValueError: If any requested role is not a managed agent
        """
        validated_roles = []
        
        for role_str in requested_roles:
            # Convert string to RoleName if possible
            try:
                role_name = RoleName(role_str)
            except ValueError:
                raise ValueError(f"Unknown role: {role_str}. Only managed agents are allowed.")
            
            # Ensure it's a managed agent
            if not self.is_agent_available(role_name):
                raise ValueError(f"Agent {role_name} is not a managed agent. Only predefined agents are allowed.")
            
            validated_roles.append(role_name)
        
        logger.info(f"Validated team composition: {[r.value for r in validated_roles]}")
        return validated_roles
    
    def preload_agents_for_team(self, role_names: List[RoleName]) -> Dict[RoleName, Type[BaseRole]]:
        """
        Preload a specific set of agents for a team.
        
        This is useful for performance optimization when you know which agents
        will be needed for a specific task.
        """
        loaded_agents = {}
        
        for role_name in role_names:
            try:
                loaded_agents[role_name] = self.get_agent_class(role_name)
            except Exception as e:
                logger.warning(f"Failed to preload agent {role_name}: {e}")
        
        logger.info(f"Preloaded {len(loaded_agents)} agents for team")
        return loaded_agents
    
    def unload_unused_agents(self, keep_roles: Optional[List[RoleName]] = None) -> int:
        """
        Unload agents that are not in the keep list to free memory.
        
        Args:
            keep_roles: List of roles to keep loaded, or None to keep all
            
        Returns:
            Number of agents unloaded
        """
        if keep_roles is None:
            return 0
        
        keep_set = set(keep_roles)
        to_unload = [role for role in self.loaded_agents.keys() if role not in keep_set]
        
        for role in to_unload:
            del self.loaded_agents[role]
        
        unload_count = len(to_unload)
        if unload_count > 0:
            logger.info(f"Unloaded {unload_count} unused agents")
        
        return unload_count
    
    def get_loading_statistics(self) -> Dict[str, any]:
        """Get statistics about agent loading."""
        return {
            "loading_strategy": self.loading_strategy.value,
            "available_agents": len(self.available_agents),
            "loaded_agents": len(self.loaded_agents),
            "total_loads": self._load_count,
            "load_errors": len(self._load_errors),
            "memory_efficiency": len(self.loaded_agents) / len(self.available_agents) if self.available_agents else 0,
            "loaded_agent_list": [role.value for role in self.loaded_agents.keys()],
            "recent_errors": self._load_errors[-5:] if self._load_errors else []
        }
    
    def clear_load_errors(self) -> int:
        """Clear the load error history and return the number of errors cleared."""
        error_count = len(self._load_errors)
        self._load_errors.clear()
        return error_count


# Global instance for the orchestration system
_managed_agent_loader = ManagedAgentLoader(AgentLoadingStrategy.LAZY)


def get_managed_agent_loader() -> ManagedAgentLoader:
    """Get the global managed agent loader instance."""
    return _managed_agent_loader


def set_agent_loading_strategy(strategy: AgentLoadingStrategy) -> None:
    """Set the global agent loading strategy."""
    global _managed_agent_loader
    _managed_agent_loader = ManagedAgentLoader(strategy)
