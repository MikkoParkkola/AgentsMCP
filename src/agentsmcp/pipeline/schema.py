"""
Pipeline Schema Models

Defines Pydantic models for the Multi-Agent CI Pipeline configuration,
including agent assignments, stage specifications, and complete pipeline specs
with comprehensive validation and default merging capabilities.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator, RootModel, ConfigDict

# Type aliases for better readability
AgentType = Literal["ollama-turbo", "codex", "claude"]
OnFailurePolicy = Literal["retry", "skip", "abort"]


class AgentPayload(RootModel[Dict[str, Any]]):
    """
    Free-form payload data for agent tasks.
    
    The orchestrator decides which keys it needs based on the task type.
    This allows flexibility for different task types and agent implementations.
    """
    root: Dict[str, Any] = Field(default_factory=dict)
    
    def __getitem__(self, item):
        return self.root[item]
    
    def __setitem__(self, key, value):
        self.root[key] = value
    
    def get(self, key, default=None):
        return self.root.get(key, default)


class AgentAssignment(BaseModel):
    """
    Specification for assigning a specific agent to perform a task.
    
    Attributes
    ----------
    type : AgentType
        Which LLM provider to use (ollama-turbo, codex, claude)
    model : str
        Specific model identifier for the agent
    task : str
        High-level task name that the agent should perform
    payload : AgentPayload
        Task-specific configuration and parameters
    timeout_seconds : int, optional
        Per-agent timeout override (inherits from stage/pipeline if not set)
    retries : int, optional
        Number of retry attempts on failure (inherits from stage/pipeline if not set)
    on_failure : OnFailurePolicy, optional
        What to do if this agent fails (retry, skip, abort)
    """
    
    type: AgentType = Field(..., description="Which LLM provider to use")
    model: str = Field(..., description="Model identifier")
    task: str = Field(..., description="High-level task name")
    payload: AgentPayload = Field(default_factory=AgentPayload, description="Task-specific parameters")
    timeout_seconds: Optional[int] = Field(None, ge=1, description="Per-agent timeout in seconds")
    retries: Optional[int] = Field(None, ge=0, description="Number of retry attempts")
    on_failure: Optional[OnFailurePolicy] = Field(None, description="Failure handling policy")

    @field_validator("task")
    def task_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Task name cannot be empty")
        return v.strip()

    @field_validator("model")
    def model_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "ollama-turbo",
                "model": "codellama:7b",
                "task": "run_pytest",
                "payload": {
                    "test_path": "tests/",
                    "coverage": True
                },
                "timeout_seconds": 300,
                "retries": 2,
                "on_failure": "retry"
            }
        }
    }


class StageDefaults(BaseModel):
    """
    Default configuration values that can be inherited by stages.
    
    Attributes
    ----------
    timeout_seconds : int
        Default timeout for agents in this context
    retries : int
        Default number of retry attempts
    on_failure : OnFailurePolicy
        Default failure handling policy
    agents : List[AgentAssignment]
        Default agent assignments (can be overridden by stages)
    """
    
    timeout_seconds: Optional[int] = Field(300, ge=1, description="Default timeout in seconds")
    retries: Optional[int] = Field(1, ge=0, description="Default retry attempts")
    on_failure: OnFailurePolicy = Field("retry", description="Default failure handling")
    agents: List[AgentAssignment] = Field(default_factory=list, description="Default agent assignments")


class StageSpec(BaseModel):
    """
    Specification for a single pipeline stage.
    
    A stage represents a logical CI step (e.g., 'compile', 'unit-test', 'deploy')
    and contains one or more agent assignments that can run in parallel or sequentially.
    
    Attributes
    ----------
    name : str
        Unique stage identifier
    description : str, optional
        Human-readable description of what this stage does
    parallel : bool
        Whether agents in this stage should run in parallel (default: True)
    agents : List[AgentAssignment]
        Agent assignments for this stage
    timeout_seconds : int, optional
        Stage-level timeout override
    retries : int, optional
        Stage-level retry override
    on_failure : OnFailurePolicy, optional
        Stage-level failure policy override
    depends_on : List[str], optional
        List of stage names this stage depends on (future feature)
    """
    
    name: str = Field(..., description="Unique stage identifier")
    description: Optional[str] = Field(None, description="Stage description")
    parallel: bool = Field(True, description="Run agents in parallel")
    agents: List[AgentAssignment] = Field(default_factory=list, description="Agent assignments")
    timeout_seconds: Optional[int] = Field(None, ge=1, description="Stage timeout override")
    retries: Optional[int] = Field(None, ge=0, description="Stage retry override")  
    on_failure: Optional[OnFailurePolicy] = Field(None, description="Stage failure policy")
    depends_on: List[str] = Field(default_factory=list, description="Stage dependencies")

    @field_validator("name")
    def name_must_be_valid(cls, v):
        if not v or not v.strip():
            raise ValueError("Stage name cannot be empty")
        # Ensure name is filesystem/URL safe
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v.strip()):
            raise ValueError("Stage name must contain only alphanumeric characters, hyphens, and underscores")
        return v.strip()

    @field_validator("agents")
    def agents_must_have_tasks(cls, v):
        for agent in v:
            if not agent.task:
                raise ValueError("Every agent assignment must declare a task")
        return v

    def apply_defaults(self, defaults: StageDefaults) -> StageSpec:
        """
        Apply default values to this stage specification.
        
        Parameters
        ----------
        defaults : StageDefaults
            Default values to apply where stage values are None
            
        Returns
        -------
        StageSpec
            New stage spec with defaults applied
        """
        # Create updated agents with defaults applied
        updated_agents = []
        for agent in self.agents:
            updated_agent = agent.model_copy()
            if updated_agent.timeout_seconds is None:
                updated_agent.timeout_seconds = self.timeout_seconds or defaults.timeout_seconds
            if updated_agent.retries is None:
                updated_agent.retries = self.retries or defaults.retries
            if updated_agent.on_failure is None:
                updated_agent.on_failure = self.on_failure or defaults.on_failure
            updated_agents.append(updated_agent)
        
        # If no agents specified, inherit default agents
        if not updated_agents and defaults.agents:
            updated_agents = [agent.model_copy() for agent in defaults.agents]
        
        return self.model_copy(update={"agents": updated_agents})

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "run-tests",
                "description": "Execute unit tests and static analysis",
                "parallel": True,
                "agents": [
                    {
                        "type": "claude",
                        "model": "claude-3.5-sonnet",
                        "task": "run_pytest",
                        "payload": {"test_path": "tests/"}
                    }
                ]
            }
        }
    }


class PipelineSpec(BaseModel):
    """
    Complete pipeline specification.
    
    Defines the entire CI pipeline including stages, default values,
    and metadata. This is the root configuration object.
    
    Attributes
    ----------
    name : str
        Pipeline name (used for run IDs and logging)
    description : str, optional
        Human-readable pipeline description
    version : str
        Pipeline specification version
    defaults : StageDefaults
        Default values inherited by all stages
    stages : List[StageSpec]
        Ordered list of pipeline stages
    notifications : dict, optional
        Notification configuration (Slack, email, etc.)
    """
    
    name: str = Field(..., description="Pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    version: str = Field("0.1.0", description="Pipeline version")
    defaults: StageDefaults = Field(default_factory=StageDefaults, description="Default stage values")
    stages: List[StageSpec] = Field(..., min_items=1, description="Pipeline stages")
    notifications: Dict[str, Any] = Field(default_factory=dict, description="Notification settings")

    @field_validator("name")
    def name_must_be_valid(cls, v):
        if not v or not v.strip():
            raise ValueError("Pipeline name cannot be empty")
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', v.strip()):
            raise ValueError("Pipeline name must contain only alphanumeric characters, hyphens, and underscores")
        return v.strip()

    @model_validator(mode='after')
    def validate_unique_stage_names(self):
        stages = self.stages
        if not stages:
            return self
            
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate stage names found: {duplicates}")
        return self

    @model_validator(mode='after')
    def validate_stage_dependencies(self):
        stages = self.stages
        if not stages:
            return self
            
        stage_names = {stage.name for stage in stages}
        for stage in stages:
            for dep in stage.depends_on:
                if dep not in stage_names:
                    raise ValueError(f"Stage '{stage.name}' depends on undefined stage '{dep}'")
        return self

    def apply_defaults(self) -> PipelineSpec:
        """
        Apply default values to all stages in the pipeline.
        
        Returns
        -------
        PipelineSpec
            New pipeline spec with defaults applied to all stages
        """
        updated_stages = [stage.apply_defaults(self.defaults) for stage in self.stages]
        return self.model_copy(update={"stages": updated_stages})

    def get_stage(self, name: str) -> Optional[StageSpec]:
        """
        Get a stage by name.
        
        Parameters
        ----------
        name : str
            Stage name to find
            
        Returns
        -------
        StageSpec or None
            Stage specification if found, None otherwise
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def validate_execution_order(self) -> List[List[str]]:
        """
        Calculate stage execution order respecting dependencies.
        
        Returns
        -------
        List[List[str]]
            List of stage name lists, where each inner list can be executed in parallel
            
        Raises
        ------
        ValueError
            If circular dependencies are detected
        """
        # Simple topological sort implementation
        # This is a placeholder for future dependency support
        return [[stage.name for stage in self.stages]]

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "python-package-ci",
            "description": "CI pipeline for Python packages",
            "version": "1.0.0",
            "defaults": {
                "timeout_seconds": 300,
                "retries": 2,
                "on_failure": "retry"
            },
            "stages": [
                {
                    "name": "install-deps",
                    "description": "Install dependencies",
                    "parallel": False,
                    "agents": [
                        {
                            "type": "ollama-turbo",
                            "model": "codellama:7b", 
                            "task": "install_requirements",
                            "payload": {"requirements_file": "requirements.txt"}
                        }
                    ]
                }
            ]
        }
    })


# Export for easier imports
__all__ = [
    "AgentType",
    "OnFailurePolicy", 
    "AgentPayload",
    "AgentAssignment",
    "StageDefaults",
    "StageSpec",
    "PipelineSpec"
]