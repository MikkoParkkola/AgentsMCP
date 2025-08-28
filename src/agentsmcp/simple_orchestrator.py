"""
Simple Orchestrator - Default mode for AgentsMCP

Implements Claude Code best practices:
- Single main loop architecture 
- Configurable model selection with user preferences
- Fallback model support
- Cost-optimized model routing
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum

from .config import Config
from .agents.base import BaseAgent
from .agents.claude_agent import ClaudeAgent
from .agents.codex_agent import CodexAgent
from .agents.ollama_agent import OllamaAgent


class TaskComplexity(Enum):
    SIMPLE = "simple"      # Basic queries, simple analysis
    MODERATE = "moderate"  # Multi-step tasks, code generation
    COMPLEX = "complex"    # Architecture, complex reasoning


class ModelRole(Enum):
    WORKHORSE = "workhorse"    # High-volume, cost-sensitive tasks
    ORCHESTRATOR = "orchestrator"  # Main loop, coordination
    SPECIALIST = "specialist"   # Complex reasoning, architecture


@dataclass
class ModelPreference:
    role: ModelRole
    primary_model: str
    fallback_models: List[str]
    cost_threshold: Optional[float] = None


@dataclass 
class TaskRequest:
    task: str
    complexity: TaskComplexity = TaskComplexity.MODERATE
    preferred_role: Optional[ModelRole] = None
    context_size_estimate: int = 1000
    cost_sensitive: bool = False


class SimpleOrchestrator:
    """
    Simplified orchestrator implementing Claude Code best practices.
    
    Default model preferences (configurable by user):
    - ollama-turbo: cheap workhorse for high-volume tasks
    - claude: orchestrator with main loop coordination  
    - codex: specialist for complex reasoning
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.log = logging.getLogger(__name__)
        
        # Agent instances cache
        self._agents: Dict[str, BaseAgent] = {}
        
        # Default model preferences (user configurable)
        self.model_preferences = self._load_model_preferences()
        
        # Task routing rules
        self.routing_rules = self._setup_routing_rules()
        
    def _load_model_preferences(self) -> Dict[ModelRole, ModelPreference]:
        """Load user-configurable model preferences from config."""
        user_prefs = getattr(self.config, 'model_preferences', {})
        
        # Default preferences based on user's stated requirements
        defaults = {
            ModelRole.WORKHORSE: ModelPreference(
                role=ModelRole.WORKHORSE,
                primary_model="ollama",
                fallback_models=["codex", "claude"],
                cost_threshold=0.01  # Very cost sensitive
            ),
            ModelRole.ORCHESTRATOR: ModelPreference(
                role=ModelRole.ORCHESTRATOR, 
                primary_model="claude",
                fallback_models=["codex", "ollama"],
                cost_threshold=0.05
            ),
            ModelRole.SPECIALIST: ModelPreference(
                role=ModelRole.SPECIALIST,
                primary_model="codex", 
                fallback_models=["claude"],
                cost_threshold=0.10  # Allow higher cost for complex tasks
            )
        }
        
        # Override with user preferences
        for role, pref_config in user_prefs.items():
            if isinstance(role, str):
                role = ModelRole(role)
            if role in defaults:
                if 'primary_model' in pref_config:
                    defaults[role].primary_model = pref_config['primary_model']
                if 'fallback_models' in pref_config:
                    defaults[role].fallback_models = pref_config['fallback_models']
                if 'cost_threshold' in pref_config:
                    defaults[role].cost_threshold = pref_config['cost_threshold']
                    
        return defaults
    
    def _setup_routing_rules(self) -> Dict[TaskComplexity, ModelRole]:
        """Setup default task complexity to model role routing."""
        return {
            TaskComplexity.SIMPLE: ModelRole.WORKHORSE,
            TaskComplexity.MODERATE: ModelRole.ORCHESTRATOR, 
            TaskComplexity.COMPLEX: ModelRole.SPECIALIST
        }
    
    async def _get_agent(self, model_type: str) -> BaseAgent:
        """Get or create agent instance with caching."""
        if model_type not in self._agents:
            agent_config = self.config.get_agent_config(model_type)
            if not agent_config:
                raise ValueError(f"No configuration found for model: {model_type}")
                
            if model_type == "claude":
                self._agents[model_type] = ClaudeAgent(agent_config, self.config)
            elif model_type == "codex":  
                self._agents[model_type] = CodexAgent(agent_config, self.config)
            elif model_type == "ollama":
                self._agents[model_type] = OllamaAgent(agent_config, self.config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        return self._agents[model_type]
    
    async def _select_model(self, request: TaskRequest) -> str:
        """
        Select best model based on task requirements and user preferences.
        Implements fallback logic for unavailable models.
        """
        # Determine role based on task complexity or explicit preference
        target_role = request.preferred_role or self.routing_rules[request.complexity]
        
        # Override for cost-sensitive tasks
        if request.cost_sensitive and target_role != ModelRole.WORKHORSE:
            target_role = ModelRole.WORKHORSE
            
        # Override for large context requirements  
        if request.context_size_estimate > 100000:
            target_role = ModelRole.ORCHESTRATOR  # Claude has largest context
            
        preference = self.model_preferences[target_role]
        
        # Try primary model first
        try:
            agent = await self._get_agent(preference.primary_model)
            return preference.primary_model
        except Exception as e:
            self.log.warning(f"Primary model {preference.primary_model} unavailable: {e}")
            
        # Try fallback models
        for fallback_model in preference.fallback_models:
            try:
                agent = await self._get_agent(fallback_model)
                self.log.info(f"Using fallback model {fallback_model} for role {target_role.value}")
                return fallback_model
            except Exception as e:
                self.log.warning(f"Fallback model {fallback_model} unavailable: {e}")
                continue
                
        raise RuntimeError(f"No available models for role {target_role.value}")
    
    async def execute_task(
        self, 
        task: Union[str, TaskRequest],
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Execute a task using the simple orchestration approach.
        
        Single main loop with intelligent model selection.
        """
        # Normalize input
        if isinstance(task, str):
            request = TaskRequest(task=task)
        else:
            request = task
            
        self.log.info(f"Executing task with complexity: {request.complexity.value}")
        
        try:
            # Select optimal model
            selected_model = await self._select_model(request)
            
            # Get agent and execute
            agent = await self._get_agent(selected_model)
            
            self.log.info(f"Using {selected_model} for task execution")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.execute_task(request.task),
                timeout=timeout
            )
            
            return {
                "status": "success",
                "result": result,
                "model_used": selected_model,
                "complexity": request.complexity.value,
                "execution_time": None  # TODO: Add timing
            }
            
        except asyncio.TimeoutError:
            self.log.error(f"Task timed out after {timeout}s")
            return {
                "status": "timeout", 
                "error": f"Task timed out after {timeout} seconds",
                "model_used": selected_model if 'selected_model' in locals() else None
            }
            
        except Exception as e:
            self.log.exception(f"Task execution failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_used": selected_model if 'selected_model' in locals() else None
            }
    
    async def chat(
        self,
        messages: List[Dict[str, str]], 
        preferred_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simplified chat interface for conversational workflows.
        Uses orchestrator model by default for conversation continuity.
        """
        # Estimate task complexity from conversation length and content
        total_tokens = sum(len(msg.get('content', '')) for msg in messages) // 4
        
        if total_tokens > 50000:
            complexity = TaskComplexity.COMPLEX
        elif total_tokens > 10000:
            complexity = TaskComplexity.MODERATE  
        else:
            complexity = TaskComplexity.SIMPLE
            
        # Create task request
        last_message = messages[-1].get('content', '') if messages else ""
        request = TaskRequest(
            task=last_message,
            complexity=complexity,
            preferred_role=ModelRole.ORCHESTRATOR,  # Chat needs continuity
            context_size_estimate=total_tokens * 4
        )
        
        # Override model if specified
        if preferred_model:
            try:
                agent = await self._get_agent(preferred_model)
                result = await agent.execute_task(last_message)
                return {
                    "status": "success",
                    "result": result, 
                    "model_used": preferred_model
                }
            except Exception as e:
                self.log.warning(f"Preferred model {preferred_model} failed: {e}")
                # Fall through to standard selection
        
        return await self.execute_task(request)
    
    def configure_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Runtime configuration of model preferences.
        Allows users to adjust model selection without restart.
        """
        for role_str, pref_config in preferences.items():
            try:
                role = ModelRole(role_str)
                if role in self.model_preferences:
                    pref = self.model_preferences[role]
                    if 'primary_model' in pref_config:
                        pref.primary_model = pref_config['primary_model']
                    if 'fallback_models' in pref_config:
                        pref.fallback_models = pref_config['fallback_models']
                    if 'cost_threshold' in pref_config:
                        pref.cost_threshold = pref_config['cost_threshold']
                        
                    self.log.info(f"Updated preferences for {role.value}")
            except (ValueError, KeyError) as e:
                self.log.warning(f"Invalid preference config for {role_str}: {e}")
    
    async def cleanup(self):
        """Clean up agent resources."""
        for agent in self._agents.values():
            try:
                await agent.cleanup()
            except Exception as e:
                self.log.warning(f"Error cleaning up agent: {e}")
        self._agents.clear()