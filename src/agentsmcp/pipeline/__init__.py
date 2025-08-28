"""
Multi-Agent CI Pipeline Package

Provides a complete CI/CD pipeline system that orchestrates multiple AI agents
(ollama-turbo, codex, claude) for different pipeline stages with parallel execution,
beautiful UX, and comprehensive status monitoring.
"""

from .schema import (
    AgentAssignment,
    StageSpec, 
    PipelineSpec,
    AgentType,
    OnFailurePolicy
)

__all__ = [
    "AgentAssignment",
    "StageSpec", 
    "PipelineSpec",
    "AgentType",
    "OnFailurePolicy",
    "load_pipeline_spec"
]


def load_pipeline_spec(file_path_or_dict) -> PipelineSpec:
    """
    Convenience function to load a pipeline specification.
    
    Parameters
    ----------
    file_path_or_dict : str | Path | dict
        Either a path to a YAML file or a dictionary containing pipeline config
        
    Returns
    -------
    PipelineSpec
        Validated pipeline specification with defaults applied
        
    Example
    -------
    >>> spec = load_pipeline_spec("agentsmcp.pipeline.yml")
    >>> spec = load_pipeline_spec({"name": "test", "stages": [...]})
    """
    if isinstance(file_path_or_dict, (str, Path)):
        from ..config.pipeline_config import PipelineConfig
        config = PipelineConfig.load_from_yaml(file_path_or_dict)
        return config.spec
    else:
        return PipelineSpec(**file_path_or_dict)