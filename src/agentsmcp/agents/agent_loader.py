"""
Agent Loader - Dynamic Agent Configuration System

This module provides functionality to load agent descriptions from JSON files
and create AgentConfig objects dynamically. This separation allows for better
maintainability and flexibility in agent management.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, ValidationError

from ..runtime_config import AgentConfig, ProviderType

logger = logging.getLogger(__name__)


class AgentDescription(BaseModel):
    """Pydantic model for agent descriptions loaded from JSON files."""
    
    type: str
    name: str
    category: str
    description: str
    system_prompt: str
    tools: List[str]
    capabilities: List[str]
    specializations: Optional[List[str]] = None
    collaboration_patterns: Optional[List[str]] = None
    quality_gates: Optional[List[Dict[str, Any]]] = None


class AgentLoader:
    """Loads and manages agent descriptions from JSON files."""
    
    def __init__(self, descriptions_path: Optional[Path] = None):
        """Initialize the agent loader.
        
        Args:
            descriptions_path: Path to the agent descriptions directory.
                              If None, uses default location.
        """
        if descriptions_path is None:
            descriptions_path = Path(__file__).parent / "descriptions"
        
        self.descriptions_path = descriptions_path
        self.loaded_descriptions: Dict[str, AgentDescription] = {}
        
    def load_all_descriptions(self) -> Dict[str, AgentDescription]:
        """Load all agent descriptions from JSON files.
        
        Returns:
            Dictionary mapping agent type to AgentDescription.
        """
        if not self.descriptions_path.exists():
            logger.warning(f"Agent descriptions directory not found: {self.descriptions_path}")
            return {}
            
        loaded_count = 0
        
        # Load from main descriptions directory
        for json_file in self.descriptions_path.glob("*.json"):
            if json_file.name == "agent_description_schema.json":
                continue  # Skip schema file
                
            try:
                agent_desc = self.load_description(json_file)
                if agent_desc:
                    self.loaded_descriptions[agent_desc.type] = agent_desc
                    loaded_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to load agent description from {json_file}: {e}")
        
        # Load from roles subdirectory if it exists
        roles_path = self.descriptions_path / "roles"
        if roles_path.exists():
            for json_file in roles_path.glob("*.json"):
                try:
                    agent_desc = self.load_description(json_file)
                    if agent_desc:
                        self.loaded_descriptions[agent_desc.type] = agent_desc
                        loaded_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to load agent description from {json_file}: {e}")
                
        logger.info(f"Loaded {loaded_count} agent descriptions from {self.descriptions_path}")
        return self.loaded_descriptions
    
    def load_description(self, file_path: Path) -> Optional[AgentDescription]:
        """Load a single agent description from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing agent description.
            
        Returns:
            AgentDescription object or None if loading failed.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            return AgentDescription(**data)
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Invalid agent description in {file_path}: {e}")
            return None
            
        except FileNotFoundError:
            logger.error(f"Agent description file not found: {file_path}")
            return None
    
    def create_agent_config(
        self,
        agent_desc: AgentDescription,
        model: str = "gpt-oss:120b",
        provider: ProviderType = ProviderType.OLLAMA_TURBO,
        api_base: str = "https://ollama.com/"
    ) -> AgentConfig:
        """Create an AgentConfig from an AgentDescription.
        
        Args:
            agent_desc: The agent description to convert.
            model: Model name to use for the agent.
            provider: Provider type for the agent.
            api_base: API base URL for the provider.
            
        Returns:
            AgentConfig object ready for use.
        """
        return AgentConfig(
            type=agent_desc.type,
            model=model,
            provider=provider,
            api_base=api_base,
            system_prompt=agent_desc.system_prompt,
            tools=agent_desc.tools
        )
    
    def create_all_agent_configs(
        self,
        model: str = "gpt-oss:120b",
        provider: ProviderType = ProviderType.OLLAMA_TURBO,
        api_base: str = "https://ollama.com/"
    ) -> Dict[str, AgentConfig]:
        """Create AgentConfig objects for all loaded descriptions.
        
        Args:
            model: Default model name to use for agents.
            provider: Default provider type for agents.
            api_base: Default API base URL for the provider.
            
        Returns:
            Dictionary mapping agent type to AgentConfig.
        """
        if not self.loaded_descriptions:
            self.load_all_descriptions()
            
        agent_configs = {}
        
        for agent_type, agent_desc in self.loaded_descriptions.items():
            try:
                config = self.create_agent_config(
                    agent_desc=agent_desc,
                    model=model,
                    provider=provider,
                    api_base=api_base
                )
                agent_configs[agent_type] = config
                
            except Exception as e:
                logger.error(f"Failed to create config for agent {agent_type}: {e}")
                
        return agent_configs
    
    def get_agent_capabilities(self, agent_type: str) -> List[str]:
        """Get capabilities for a specific agent type.
        
        Args:
            agent_type: The type of agent to get capabilities for.
            
        Returns:
            List of capabilities or empty list if agent not found.
        """
        if agent_type not in self.loaded_descriptions:
            return []
            
        return self.loaded_descriptions[agent_type].capabilities
    
    def get_collaboration_patterns(self, agent_type: str) -> List[str]:
        """Get collaboration patterns for a specific agent type.
        
        Args:
            agent_type: The type of agent to get collaboration patterns for.
            
        Returns:
            List of collaborating agent types or empty list if not found.
        """
        if agent_type not in self.loaded_descriptions:
            return []
            
        patterns = self.loaded_descriptions[agent_type].collaboration_patterns
        return patterns if patterns else []
    
    def get_quality_gates(self, agent_type: str) -> List[Dict[str, Any]]:
        """Get quality gates for a specific agent type.
        
        Args:
            agent_type: The type of agent to get quality gates for.
            
        Returns:
            List of quality gate definitions or empty list if not found.
        """
        if agent_type not in self.loaded_descriptions:
            return []
            
        gates = self.loaded_descriptions[agent_type].quality_gates
        return gates if gates else []
    
    def suggest_collaborators(self, task_description: str, current_agent: str) -> List[str]:
        """Suggest collaborating agents based on task description.
        
        Args:
            task_description: Description of the task to be performed.
            current_agent: Current agent type performing the task.
            
        Returns:
            List of suggested collaborating agent types.
        """
        if current_agent not in self.loaded_descriptions:
            return []
            
        # Get predefined collaboration patterns
        collaborators = self.get_collaboration_patterns(current_agent)
        
        # Add task-specific suggestions based on keywords
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ['security', 'vulnerability', 'encrypt', 'auth']):
            if 'security_engineer' not in collaborators:
                collaborators.append('security_engineer')
                
        if any(word in task_lower for word in ['performance', 'load', 'scale', 'optimize']):
            if 'performance_engineer' not in collaborators:
                collaborators.append('performance_engineer')
                
        if any(word in task_lower for word in ['ui', 'ux', 'design', 'interface', 'user']):
            if 'ux_ui_designer' not in collaborators:
                collaborators.append('ux_ui_designer')
                
        if any(word in task_lower for word in ['test', 'quality', 'bug', 'qa']):
            if 'chief_qa_engineer' not in collaborators:
                collaborators.append('chief_qa_engineer')
                
        return collaborators


# Global agent loader instance
_agent_loader = None


def get_agent_loader() -> AgentLoader:
    """Get the global agent loader instance."""
    global _agent_loader
    if _agent_loader is None:
        _agent_loader = AgentLoader()
    return _agent_loader