"""
Pipeline Configuration Management

Provides integration between the pipeline schema and the AgentsMCP configuration system.
Handles loading, validation, and management of pipeline specifications from YAML files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Union

import yaml
from pydantic import ValidationError

from ..pipeline.schema import PipelineSpec

logger = logging.getLogger(__name__)


class PipelineConfigError(Exception):
    """Raised when there are issues with pipeline configuration."""
    pass


class PipelineConfig:
    """
    Pipeline configuration manager.
    
    Handles loading and validation of pipeline specifications from YAML files,
    with integration into the broader AgentsMCP configuration system.
    
    Attributes
    ----------
    spec : PipelineSpec
        The validated pipeline specification
    source_path : Path, optional
        Path to the source configuration file
    """
    
    def __init__(self, spec: PipelineSpec, source_path: Path = None):
        """
        Initialize pipeline configuration.
        
        Parameters
        ----------
        spec : PipelineSpec
            Validated pipeline specification
        source_path : Path, optional
            Path to source configuration file
        """
        self.spec = spec
        self.source_path = source_path
    
    @classmethod
    def load_from_yaml(cls, file_path: Union[str, Path]) -> PipelineConfig:
        """
        Load pipeline configuration from a YAML file.
        
        Parameters
        ----------
        file_path : str or Path
            Path to the YAML configuration file
            
        Returns
        -------
        PipelineConfig
            Loaded and validated pipeline configuration
            
        Raises
        ------
        PipelineConfigError
            If the file cannot be read or contains invalid configuration
        ValidationError
            If the configuration doesn't match the schema
            
        Example
        -------
        >>> config = PipelineConfig.load_from_yaml("agentsmcp.pipeline.yml")
        >>> spec = config.spec
        >>> print(spec.name)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise PipelineConfigError(f"Pipeline configuration file not found: {path}")
        
        if not path.is_file():
            raise PipelineConfigError(f"Path is not a file: {path}")
        
        try:
            with path.open('r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise PipelineConfigError(f"Invalid YAML in {path}: {e}")
        except Exception as e:
            raise PipelineConfigError(f"Could not read {path}: {e}")
        
        if not raw_config:
            raise PipelineConfigError(f"Empty configuration file: {path}")
        
        if not isinstance(raw_config, dict):
            raise PipelineConfigError(f"Configuration must be a YAML object, got {type(raw_config)}")
        
        # Extract pipeline section
        if "pipeline" not in raw_config:
            raise PipelineConfigError("Configuration must contain a 'pipeline' section")
        
        pipeline_data = raw_config["pipeline"]
        if not isinstance(pipeline_data, dict):
            raise PipelineConfigError("Pipeline section must be a YAML object")
        
        try:
            spec = PipelineSpec(**pipeline_data)
        except ValidationError as e:
            logger.error("Pipeline validation failed for %s: %s", path, e)
            raise
        
        return cls(spec, path)
    
    @classmethod
    def load_from_dict(cls, config_dict: Dict[str, Any]) -> PipelineConfig:
        """
        Load pipeline configuration from a dictionary.
        
        Parameters
        ----------
        config_dict : dict
            Configuration dictionary (must contain 'pipeline' key)
            
        Returns
        -------
        PipelineConfig
            Loaded and validated pipeline configuration
            
        Raises
        ------
        PipelineConfigError
            If the dictionary structure is invalid
        ValidationError
            If the configuration doesn't match the schema
        """
        if not isinstance(config_dict, dict):
            raise PipelineConfigError(f"Configuration must be a dictionary, got {type(config_dict)}")
        
        if "pipeline" not in config_dict:
            raise PipelineConfigError("Configuration must contain a 'pipeline' section")
        
        pipeline_data = config_dict["pipeline"]
        if not isinstance(pipeline_data, dict):
            raise PipelineConfigError("Pipeline section must be a dictionary")
        
        try:
            spec = PipelineSpec(**pipeline_data)
        except ValidationError as e:
            logger.error("Pipeline validation failed: %s", e)
            raise
        
        return cls(spec)
    
    def save_to_yaml(self, file_path: Union[str, Path]) -> None:
        """
        Save pipeline configuration to a YAML file.
        
        Parameters
        ----------
        file_path : str or Path
            Path where to save the configuration
            
        Raises
        ------
        PipelineConfigError
            If the file cannot be written
        """
        path = Path(file_path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert spec to dict and wrap in pipeline section
        config_dict = {
            "pipeline": self.spec.model_dump(exclude_unset=True)
        }
        
        try:
            with path.open('w', encoding='utf-8') as f:
                yaml.safe_dump(
                    config_dict, 
                    f, 
                    default_flow_style=False,
                    sort_keys=False,
                    indent=2
                )
            logger.info("Pipeline configuration saved to %s", path)
        except Exception as e:
            raise PipelineConfigError(f"Could not write configuration to {path}: {e}")
    
    def validate(self) -> bool:
        """
        Validate the pipeline configuration.
        
        Returns
        -------
        bool
            True if configuration is valid
            
        Raises
        ------
        ValidationError
            If validation fails
        """
        # Apply defaults and re-validate
        try:
            applied_spec = self.spec.apply_defaults()
            # Additional custom validation can go here
            return True
        except Exception as e:
            logger.error("Pipeline validation failed: %s", e)
            raise
    
    def get_stage_names(self) -> list[str]:
        """
        Get list of all stage names in the pipeline.
        
        Returns
        -------
        list[str]
            Stage names in execution order
        """
        return [stage.name for stage in self.spec.stages]
    
    def get_agent_types(self) -> set[str]:
        """
        Get set of all agent types used in the pipeline.
        
        Returns
        -------
        set[str]
            Unique agent types (ollama-turbo, codex, claude)
        """
        agent_types = set()
        for stage in self.spec.stages:
            for agent in stage.agents:
                agent_types.add(agent.type)
        return agent_types
    
    def __str__(self) -> str:
        """String representation of the pipeline configuration."""
        source = f" from {self.source_path}" if self.source_path else ""
        return f"PipelineConfig(name='{self.spec.name}', stages={len(self.spec.stages)}){source}"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"PipelineConfig("
            f"name='{self.spec.name}', "
            f"version='{self.spec.version}', "
            f"stages={len(self.spec.stages)}, "
            f"source_path={self.source_path}"
            f")"
        )


def load_pipeline_config(file_path: Union[str, Path]) -> PipelineConfig:
    """
    Convenience function to load a pipeline configuration.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the YAML configuration file
        
    Returns
    -------
    PipelineConfig
        Loaded and validated pipeline configuration
        
    Example
    -------
    >>> config = load_pipeline_config("my-pipeline.yml")
    >>> print(config.spec.name)
    """
    return PipelineConfig.load_from_yaml(file_path)


def create_default_pipeline_config(
    name: str,
    stages: list[str] = None,
    output_path: Path = None
) -> PipelineConfig:
    """
    Create a default pipeline configuration.
    
    Parameters
    ----------
    name : str
        Pipeline name
    stages : list[str], optional
        Stage names (default: ["build", "test", "deploy"])
    output_path : Path, optional
        If provided, save the config to this path
        
    Returns
    -------
    PipelineConfig
        Default pipeline configuration
    """
    if stages is None:
        stages = ["build", "test", "deploy"]
    
    from ..pipeline.schema import StageSpec, AgentAssignment
    
    stage_specs = []
    for stage_name in stages:
        # Create a simple ollama-turbo assignment for each stage
        agent = AgentAssignment(
            type="ollama-turbo",
            model="codellama:7b",
            task=f"run_{stage_name}",
            payload={"stage": stage_name}
        )
        stage_spec = StageSpec(
            name=stage_name,
            description=f"Execute {stage_name} stage",
            agents=[agent]
        )
        stage_specs.append(stage_spec)
    
    spec = PipelineSpec(
        name=name,
        description=f"Default pipeline for {name}",
        stages=stage_specs
    )
    
    config = PipelineConfig(spec)
    
    if output_path:
        config.save_to_yaml(output_path)
    
    return config


# Export for easier imports
__all__ = [
    "PipelineConfig",
    "PipelineConfigError",
    "load_pipeline_config",
    "create_default_pipeline_config"
]