'''Strict Orchestrator Implementation for AgentsMCP

This orchestrator enforces the architectural principle that ONLY the orchestrator 
communicates directly with users. All agent communications are internal.

Key Principles:

The orchestrator acts as the single point of contact for users while coordinating
with agents behind the scenes to provide intelligent, consolidated responses.
'''"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .task_classifier import TaskClassifier, TaskClassification, ClassificationResult
from .response_synthesizer import ResponseSynthesizer, SynthesisStrategy
from .communication_interceptor import CommunicationInterceptor

logger = logging.getLogger(__name__)


class OrchestratorMode(Enum):
    """Orchestrator operation modes."""
    STRICT_ISOLATION = "strict_isolation"  # No direct agent-user communication allowed
    SUPERVISED = "supervised"  # Agent communications monitored and filtered
    TRANSPARENT = "transparent"  # Agent communications visible but orchestrator-mediated


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    mode: OrchestratorMode = OrchestratorMode.STRICT_ISOLATION
    
    # Task classification settings
    enable_smart_classification: bool = True
    simple_task_threshold: float = 0.8
    single_agent_threshold: float = 0.6
    
    # Response synthesis settings
    default_synthesis_strategy: SynthesisStrategy = SynthesisStrategy.SUMMARIZE
    synthesis_timeout_ms: int = 2000
    
    # Communication settings
    intercept_all_agent_output: bool = True
    allow_agent_status_messages: bool = False
    consolidate_error_messages: bool = True
    
    # Performance settings
    max_agent_wait_time_ms: int = 30000
    max_parallel_agents: int = 8
    
    # Fallback behavior
    fallback_to_simple_response: bool = True
    orchestrator_persona: str = "helpful AI assistant"
 
 
 @dataclass 
 class OrchestratorResponse:
         logger.info("Orchestrator shutdown complete")
