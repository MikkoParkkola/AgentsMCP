"""
Distributed AgentsMCP Architecture

Separates the system into:
- Orchestrator: Central coordination with 64K-128K context window
- Workers: Task execution with 8K-32K context windows
- Message Queue: Reliable communication and task distribution

Enables cost-effective scaling and specialized resource allocation.
"""

from .orchestrator import DistributedOrchestrator
from .worker import AgentWorker
from .message_queue import MessageQueue, Task, TaskResult
from .mesh_coordinator import (AgentMeshCoordinator, CollaborationRequest, 
                             CollaborationType, TrustLevel, AgentCapability)
from .governance import (GovernanceEngine, GovernancePolicy, AutonomyLevel, 
                        RiskLevel, EscalationRequest, EscalationResponse)
from .context_intelligence_clean import (ContextIntelligenceEngine, ContextItem, 
                                        ContextPriority, ContextType, ContextBudget)
from .multimodal_engine import (MultiModalEngine, AgentCapabilityProfile, ModalContent,
                              ModalityType, ProcessingCapability, ModalProcessor,
                              CodeProcessor, DataProcessor, ImageProcessor)
from .ollama_turbo_integration import (OllamaHybridOrchestrator, OllamaMode, ModelTier, 
                                     OllamaRequest, OllamaResponse, create_ollama_orchestrator,
                                     get_ollama_config_from_env)

__all__ = [
    "DistributedOrchestrator", 
    "AgentWorker", 
    "MessageQueue", 
    "Task", 
    "TaskResult",
    "AgentMeshCoordinator",
    "CollaborationRequest",
    "CollaborationType",
    "TrustLevel",
    "AgentCapability",
    "GovernanceEngine",
    "GovernancePolicy",
    "AutonomyLevel",
    "RiskLevel",
    "EscalationRequest",
    "EscalationResponse",
    "ContextIntelligenceEngine",
    "ContextItem",
    "ContextPriority",
    "ContextType",
    "ContextBudget",
    "MultiModalEngine",
    "AgentCapabilityProfile",
    "ModalContent",
    "ModalityType",
    "ProcessingCapability",
    "ModalProcessor",
    "CodeProcessor",
    "DataProcessor",
    "ImageProcessor",
    "OllamaHybridOrchestrator",
    "OllamaMode",
    "ModelTier",
    "OllamaRequest",
    "OllamaResponse",
    "create_ollama_orchestrator",
    "get_ollama_config_from_env"
]