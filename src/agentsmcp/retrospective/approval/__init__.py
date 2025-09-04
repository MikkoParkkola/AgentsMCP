"""
Comprehensive user approval system for retrospective improvements.

This module provides a complete approval system for retrospective improvements,
including configuration management, user interfaces (CLI, TUI, auto), criteria-based
filtering, workflow orchestration, and integration with the safety framework.

Key Components:
- ApprovalConfig: Configuration and settings management
- ApprovalDecision: Data structures for approval outcomes  
- ApprovalInterface: Abstract interfaces for different approval methods
- ApprovalCriteria: Extensible criteria and filtering system
- ApprovalWorkflow: Main workflow orchestration
- ApprovalOrchestrator: Integration with safety framework

Usage Examples:

1. Simple auto-approval:
    ```python
    from agentsmcp.retrospective.approval import create_auto_orchestrator
    
    orchestrator = create_auto_orchestrator()
    result = await orchestrator.orchestrate_approval_process(improvements)
    ```

2. Interactive approval with TUI:
    ```python
    from agentsmcp.retrospective.approval import (
        ApprovalConfig, ApprovalMode, ApprovalInterfaceType, ApprovalOrchestrator
    )
    
    config = ApprovalConfig()
    config.approval_mode = ApprovalMode.INTERACTIVE
    config.interface_type = ApprovalInterfaceType.TUI
    
    orchestrator = ApprovalOrchestrator(config)
    result = await orchestrator.orchestrate_approval_process(improvements)
    ```

3. Custom criteria-based approval:
    ```python
    from agentsmcp.retrospective.approval import (
        ApprovalOrchestrator, ApprovalCriteriaEngine, CategoryCriterion,
        FilterAction, ImprovementCategory
    )
    
    criteria_engine = ApprovalCriteriaEngine()
    criteria_engine.add_global_criterion(CategoryCriterion(
        "auto_approve_performance",
        target_categories={ImprovementCategory.PERFORMANCE},
        action=FilterAction.AUTO_APPROVE
    ))
    
    orchestrator = ApprovalOrchestrator(criteria_engine=criteria_engine)
    result = await orchestrator.orchestrate_approval_process(improvements)
    ```
"""

# Core data structures
from .approval_decision import (
    ApprovalStatus,
    ApprovalMode, 
    ApprovalInterfaceType,
    RejectionReason,
    ApprovalDecision,
    BatchApprovalDecision,
    ApprovalHistory,
    ApprovalContext
)

# Configuration management
from .approval_config import (
    CategoryApprovalRule,
    ApprovalTimeouts,
    ApprovalThresholds,
    UserPreferences,
    ApprovalConfig,
    get_default_config_path,
    load_approval_config
)

# User interfaces
from .approval_interfaces import (
    ApprovalInterface,
    ApprovalPromptData,
    AutoApprovalInterface,
    CLIApprovalInterface,
    TUIApprovalInterface,
    create_approval_interface
)

# Criteria and filtering
from .approval_criteria import (
    CriterionType,
    FilterAction,
    CriterionResult,
    ApprovalCriterion,
    CategoryCriterion,
    PriorityCriterion,
    ContentCriterion,
    ImpactCriterion,
    RiskCriterion,
    TimeCriterion,
    CustomCriterion,
    ApprovalFilter,
    ApprovalCriteriaEngine
)

# Workflow orchestration
from .approval_workflow import (
    WorkflowState,
    WorkflowMetrics,
    ApprovalWorkflow,
    WorkflowFactory
)

# Main orchestrator
from .approval_orchestrator import (
    OrchestrationPhase,
    OrchestrationResult,
    ApprovalOrchestrator,
    create_production_orchestrator,
    create_development_orchestrator,
    create_testing_orchestrator
)

# Version information
__version__ = "1.0.0"
__author__ = "AgentsMCP Development Team"
__description__ = "Comprehensive user approval system for retrospective improvements"

# Main exports
__all__ = [
    # Data structures
    "ApprovalStatus",
    "ApprovalMode", 
    "ApprovalInterfaceType",
    "RejectionReason",
    "ApprovalDecision",
    "BatchApprovalDecision",
    "ApprovalHistory",
    "ApprovalContext",
    
    # Configuration
    "CategoryApprovalRule",
    "ApprovalTimeouts",
    "ApprovalThresholds", 
    "UserPreferences",
    "ApprovalConfig",
    "get_default_config_path",
    "load_approval_config",
    
    # Interfaces
    "ApprovalInterface",
    "ApprovalPromptData",
    "AutoApprovalInterface",
    "CLIApprovalInterface", 
    "TUIApprovalInterface",
    "create_approval_interface",
    
    # Criteria
    "CriterionType",
    "FilterAction",
    "CriterionResult",
    "ApprovalCriterion",
    "CategoryCriterion",
    "PriorityCriterion",
    "ContentCriterion",
    "ImpactCriterion",
    "RiskCriterion",
    "TimeCriterion",
    "CustomCriterion",
    "ApprovalFilter",
    "ApprovalCriteriaEngine",
    
    # Workflow
    "WorkflowState",
    "WorkflowMetrics",
    "ApprovalWorkflow",
    "WorkflowFactory",
    
    # Orchestration
    "OrchestrationPhase",
    "OrchestrationResult", 
    "ApprovalOrchestrator",
    "create_production_orchestrator",
    "create_development_orchestrator",
    "create_testing_orchestrator"
]


# Convenience factory functions
def create_auto_orchestrator(safety_validation: bool = False) -> ApprovalOrchestrator:
    """Create an orchestrator configured for automatic approvals.
    
    Args:
        safety_validation: Whether to enable safety validation
        
    Returns:
        ApprovalOrchestrator configured for auto-approval
    """
    config = ApprovalConfig()
    config.approval_mode = ApprovalMode.AUTO
    config.interface_type = ApprovalInterfaceType.AUTO
    config.require_safety_validation = safety_validation
    
    return ApprovalOrchestrator(approval_config=config)


def create_interactive_orchestrator(interface: str = "tui", 
                                  safety_validation: bool = True) -> ApprovalOrchestrator:
    """Create an orchestrator configured for interactive approvals.
    
    Args:
        interface: Interface type ("cli", "tui", or "auto")
        safety_validation: Whether to enable safety validation
        
    Returns:
        ApprovalOrchestrator configured for interactive approval
    """
    config = ApprovalConfig()
    config.approval_mode = ApprovalMode.INTERACTIVE
    
    interface_map = {
        "cli": ApprovalInterfaceType.CLI,
        "tui": ApprovalInterfaceType.TUI,
        "auto": ApprovalInterfaceType.AUTO
    }
    config.interface_type = interface_map.get(interface.lower(), ApprovalInterfaceType.TUI)
    config.require_safety_validation = safety_validation
    
    return ApprovalOrchestrator(approval_config=config)


def create_batch_orchestrator(mode: str = "approve", 
                            safety_validation: bool = True) -> ApprovalOrchestrator:
    """Create an orchestrator configured for batch operations.
    
    Args:
        mode: Batch mode ("approve" or "reject")
        safety_validation: Whether to enable safety validation
        
    Returns:
        ApprovalOrchestrator configured for batch operations
    """
    config = ApprovalConfig()
    
    if mode.lower() == "approve":
        config.approval_mode = ApprovalMode.BATCH_APPROVE
    else:
        config.approval_mode = ApprovalMode.BATCH_REJECT
    
    config.interface_type = ApprovalInterfaceType.CLI
    config.require_safety_validation = safety_validation
    
    return ApprovalOrchestrator(approval_config=config)


# CLI integration helper
def update_config_from_cli_args(**kwargs) -> ApprovalConfig:
    """Update approval configuration from CLI arguments.
    
    Supported arguments:
    - approval_mode: "auto", "manual", "interactive", "batch_approve", "batch_reject"  
    - approval_interface: "cli", "tui", "auto"
    - approval_timeout: timeout in seconds
    - safety_validation: boolean
    
    Returns:
        Updated ApprovalConfig
    """
    config = load_approval_config()
    config.update_from_cli_args(**kwargs)
    return config


# Integration with existing retrospective system
def integrate_with_retrospective_engine(retrospective_engine, 
                                      approval_config = None) -> ApprovalOrchestrator:
    """Integrate approval system with existing retrospective engine.
    
    Args:
        retrospective_engine: Existing retrospective engine instance
        approval_config: Optional approval configuration
        
    Returns:
        ApprovalOrchestrator integrated with retrospective engine
    """
    config = approval_config or load_approval_config()
    
    # Create orchestrator with integration hooks
    orchestrator = ApprovalOrchestrator(approval_config=config)
    
    # Set up integration callbacks (placeholder for actual implementation)
    # This would connect to the retrospective engine's improvement generation
    # and safety validation systems
    
    return orchestrator