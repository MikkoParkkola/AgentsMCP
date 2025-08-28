"""
Orchestrator Factory - Choose between simple and complex orchestration modes

Implements the pattern: simple mode as default, complex orchestration optional.
"""

import logging
from typing import Optional, Union
from enum import Enum

from .config import Config
from .simple_orchestrator import SimpleOrchestrator


class OrchestratorMode(Enum):
    SIMPLE = "simple"      # Default: Single main loop, optimized for most use cases
    COMPLEX = "complex"    # Optional: Full distributed orchestration for enterprise


class OrchestratorFactory:
    """
    Factory for creating appropriate orchestrator based on requirements.
    
    Follows principle: Simple mode as default, complex only when needed.
    """
    
    @staticmethod
    def create(
        config: Config,
        mode: Optional[Union[str, OrchestratorMode]] = None,
        auto_detect: bool = True
    ):
        """
        Create orchestrator instance based on mode and requirements.
        
        Args:
            config: Application configuration
            mode: Explicit orchestrator mode (simple/complex)  
            auto_detect: Automatically detect if complex mode needed
            
        Returns:
            Appropriate orchestrator instance
        """
        logger = logging.getLogger(__name__)
        
        # Normalize mode
        if isinstance(mode, str):
            mode = OrchestratorMode(mode.lower())
        
        # Auto-detection logic for complex mode requirements
        if mode is None and auto_detect:
            mode = OrchestratorFactory._detect_required_mode(config)
            
        # Default to simple mode
        if mode is None:
            mode = OrchestratorMode.SIMPLE
            
        logger.info(f"Creating orchestrator in {mode.value} mode")
        
        if mode == OrchestratorMode.SIMPLE:
            return SimpleOrchestrator(config)
        elif mode == OrchestratorMode.COMPLEX:
            # Lazy import to avoid loading complex orchestrator unless needed
            from .distributed.orchestrator import DistributedOrchestrator
            return DistributedOrchestrator(config)
        else:
            raise ValueError(f"Unknown orchestrator mode: {mode}")
    
    @staticmethod
    def _detect_required_mode(config: Config) -> OrchestratorMode:
        """
        Auto-detect if complex orchestration mode is required.
        
        Complex mode indicators:
        - Multiple distributed nodes configured
        - Enterprise features enabled
        - High-scale requirements 
        - Advanced workflow orchestration needed
        """
        # Check for distributed configuration
        distributed_config = getattr(config, 'distributed', None)
        if distributed_config:
            node_count = getattr(distributed_config, 'node_count', 1)
            if node_count > 1:
                return OrchestratorMode.COMPLEX
                
        # Check for enterprise features
        enterprise_features = getattr(config, 'enterprise_features', {})
        if enterprise_features.get('enabled', False):
            return OrchestratorMode.COMPLEX
            
        # Check for high-scale indicators
        scale_config = getattr(config, 'scale', {})
        if scale_config.get('concurrent_jobs', 0) > 10:
            return OrchestratorMode.COMPLEX
            
        # Check for advanced workflow requirements
        workflow_config = getattr(config, 'workflows', {})  
        if workflow_config.get('complex_orchestration', False):
            return OrchestratorMode.COMPLEX
            
        # Default to simple mode
        return OrchestratorMode.SIMPLE
    
    @staticmethod
    def get_mode_recommendations(config: Config) -> dict:
        """
        Get recommendations for orchestrator mode selection.
        
        Returns:
            Dict with mode recommendations and reasoning
        """
        detected_mode = OrchestratorFactory._detect_required_mode(config)
        
        recommendations = {
            "detected_mode": detected_mode.value,
            "reasoning": [],
            "benefits": {
                "simple": [
                    "Lower resource usage",
                    "Faster startup time", 
                    "Simpler debugging",
                    "Cost-optimized model routing",
                    "Claude Code best practices"
                ],
                "complex": [
                    "Distributed execution",
                    "Enterprise-grade scaling",
                    "Advanced workflow orchestration", 
                    "Multi-node coordination",
                    "Full observability"
                ]
            },
            "when_to_use": {
                "simple": [
                    "Single-node deployment",
                    "< 10 concurrent jobs",
                    "Cost-sensitive workloads",
                    "Quick prototyping",
                    "Most production use cases"
                ],
                "complex": [
                    "Multi-node clusters", 
                    "> 10 concurrent jobs",
                    "Enterprise requirements",
                    "Complex workflow dependencies",
                    "Advanced monitoring needs"
                ]
            }
        }
        
        # Add specific reasoning
        if detected_mode == OrchestratorMode.COMPLEX:
            recommendations["reasoning"].append("Complex mode detected based on configuration")
        else:
            recommendations["reasoning"].append("Simple mode recommended for optimal performance")
            
        return recommendations