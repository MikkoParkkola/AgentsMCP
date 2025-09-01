"""
Role-based agent system with provider/model separation.

This module defines agent roles (analyst, coder, project_manager, etc.) 
and their corresponding provider/model configurations, allowing flexible
LLM backend selection while maintaining consistent role-based task routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Available agent roles for task delegation."""
    ANALYST = "analyst"
    CODER = "coder" 
    PROJECT_MANAGER = "project_manager"
    ARCHITECT = "architect"
    QA_REVIEWER = "qa_reviewer"
    GENERAL = "general"

@dataclass
class RoleConfig:
    """Configuration for an agent role."""
    role: AgentRole
    provider: str
    model: str
    description: str
    capabilities: List[str]
    context_length: Optional[int] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    
class RoleManager:
    """Manages agent roles and their provider/model configurations."""
    
    def __init__(self):
        self._roles = self._initialize_default_roles()
    
    def _initialize_default_roles(self) -> Dict[AgentRole, RoleConfig]:
        """Initialize default role configurations with ollama-turbo."""
        return {
            AgentRole.ANALYST: RoleConfig(
                role=AgentRole.ANALYST,
                provider="ollama-turbo",
                model="gpt-oss:120b",
                description="Analyzes code, architecture, and provides quality assessments",
                capabilities=["analysis", "review", "assessment", "evaluation", "documentation"],
                context_length=128000,
                temperature=0.3  # Lower temperature for analytical tasks
            ),
            
            AgentRole.CODER: RoleConfig(
                role=AgentRole.CODER,
                provider="ollama-turbo", 
                model="gpt-oss:120b",
                description="Writes, debugs, and refactors code",
                capabilities=["coding", "debugging", "refactoring", "implementation"],
                context_length=128000,
                temperature=0.2  # Lower temperature for code generation
            ),
            
            AgentRole.PROJECT_MANAGER: RoleConfig(
                role=AgentRole.PROJECT_MANAGER,
                provider="ollama-turbo",
                model="gpt-oss:120b", 
                description="Manages project priorities, backlogs, and planning",
                capabilities=["planning", "prioritization", "project_management", "roadmap"],
                context_length=128000,
                temperature=0.5  # Balanced temperature for planning
            ),
            
            AgentRole.ARCHITECT: RoleConfig(
                role=AgentRole.ARCHITECT,
                provider="ollama-turbo",
                model="gpt-oss:120b",
                description="Designs system architecture and technical solutions",
                capabilities=["architecture", "design", "system_design", "technical_planning"],
                context_length=128000, 
                temperature=0.4  # Balanced temperature for design
            ),
            
            AgentRole.QA_REVIEWER: RoleConfig(
                role=AgentRole.QA_REVIEWER,
                provider="ollama-turbo",
                model="gpt-oss:120b",
                description="Reviews code quality, tests, and provides QA feedback",
                capabilities=["qa", "testing", "review", "quality_assurance"],
                context_length=128000,
                temperature=0.3  # Lower temperature for quality checks
            ),
            
            AgentRole.GENERAL: RoleConfig(
                role=AgentRole.GENERAL,
                provider="ollama-turbo", 
                model="gpt-oss:120b",
                description="Handles general tasks and fallback scenarios",
                capabilities=["general", "conversation", "help", "status"],
                context_length=128000,
                temperature=0.7  # Higher temperature for conversation
            )
        }
    
    def get_role_config(self, role: AgentRole) -> RoleConfig:
        """Get configuration for a specific role."""
        return self._roles.get(role, self._roles[AgentRole.GENERAL])
    
    def get_role_by_name(self, role_name: str) -> Optional[AgentRole]:
        """Get role enum by string name."""
        try:
            return AgentRole(role_name.lower())
        except ValueError:
            return None
    
    def update_role_config(self, role: AgentRole, **kwargs) -> None:
        """Update configuration for a role."""
        if role in self._roles:
            current_config = self._roles[role]
            # Update only provided fields
            for key, value in kwargs.items():
                if hasattr(current_config, key):
                    setattr(current_config, key, value)
    
    def set_global_provider_model(self, provider: str, model: str) -> None:
        """Set provider and model for all roles."""
        for role_config in self._roles.values():
            role_config.provider = provider
            role_config.model = model
    
    def get_available_roles(self) -> List[AgentRole]:
        """Get list of all available roles."""
        return list(self._roles.keys())
    
    def get_role_for_capabilities(self, capabilities: List[str]) -> AgentRole:
        """Find the best role for given capabilities."""
        best_role = AgentRole.GENERAL
        best_score = 0
        
        for role, config in self._roles.items():
            # Score based on capability overlap
            overlap = len(set(capabilities) & set(config.capabilities))
            if overlap > best_score:
                best_score = overlap
                best_role = role
        
        return best_role
    
    def get_role_info(self) -> Dict[str, Any]:
        """Get information about all configured roles."""
        return {
            role.value: {
                "provider": config.provider,
                "model": config.model,
                "description": config.description,
                "capabilities": config.capabilities,
                "context_length": config.context_length,
                "temperature": config.temperature
            }
            for role, config in self._roles.items()
        }

# Global role manager instance
role_manager = RoleManager()