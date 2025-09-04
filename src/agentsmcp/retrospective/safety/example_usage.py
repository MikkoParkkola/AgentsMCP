"""Example usage of the safety validation framework."""

import asyncio
import logging
from typing import List, Union

from ..data_models import ActionPoint, SystemicImprovement, PriorityLevel, ImprovementCategory
from .safety_config import SafetyConfig, SafetyLevel
from .safety_orchestrator import SafetyOrchestrator, SafetyWorkflowResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_improvement_implementer(
    improvements: List[Union[ActionPoint, SystemicImprovement]]
):
    """Example improvement implementer function."""
    logger.info(f"Implementing {len(improvements)} improvements...")
    
    for improvement in improvements:
        improvement_id = getattr(improvement, 'action_id', None) or getattr(improvement, 'improvement_id', 'unknown')
        logger.info(f"- Implementing: {improvement.title} (ID: {improvement_id})")
        
        # Simulate implementation work
        await asyncio.sleep(0.1)
    
    logger.info("All improvements implemented successfully")


async def create_sample_improvements() -> List[ActionPoint]:
    """Create sample improvement actions for testing."""
    improvements = [
        ActionPoint(
            title="Optimize response caching",
            description="Implement response caching to improve performance metrics",
            category="performance",
            priority=PriorityLevel.HIGH,
            implementation_steps=[
                "Add caching middleware",
                "Configure cache TTL",
                "Monitor cache hit rates"
            ],
            estimated_effort_hours=4.0
        ),
        ActionPoint(
            title="Enhanced error logging",
            description="Add detailed error logging for better debugging",
            category="monitoring",
            priority=PriorityLevel.MEDIUM,
            implementation_steps=[
                "Add structured logging",
                "Include error context",
                "Ensure no sensitive data in logs"
            ],
            estimated_effort_hours=2.0
        ),
        ActionPoint(
            title="Critical system modification",
            description="Modify orchestrator core functionality for better coordination",
            category="system",
            priority=PriorityLevel.CRITICAL,
            implementation_steps=[
                "Update orchestrator configuration",
                "Modify core processing logic",
                "Update decision making process"
            ],
            estimated_effort_hours=8.0
        )
    ]
    
    return improvements


async def example_safe_improvement_workflow():
    """Example of using the complete safety workflow."""
    logger.info("=== Safety Framework Example ===")
    
    # Create configuration
    config = SafetyConfig.create_production_config()
    logger.info(f"Using safety level: {config.safety_level}")
    
    # Create safety orchestrator
    orchestrator = SafetyOrchestrator(config)
    
    try:
        # Initialize the orchestrator
        await orchestrator.initialize()
        logger.info("Safety orchestrator initialized")
        
        # Create sample improvements
        improvements = await create_sample_improvements()
        logger.info(f"Created {len(improvements)} sample improvements")
        
        # Execute safe improvement workflow
        result = await orchestrator.safe_apply_improvements(
            improvements=improvements,
            implementer_function=example_improvement_implementer,
            timeout_seconds=120
        )
        
        logger.info(f"Safety workflow result: {result}")
        
        if result == SafetyWorkflowResult.SUCCESS:
            logger.info("✅ All improvements applied safely")
        elif result == SafetyWorkflowResult.VALIDATION_FAILED:
            logger.warning("⚠️  Some improvements failed validation")
        elif result == SafetyWorkflowResult.ROLLBACK_TRIGGERED:
            logger.warning("🔄 Rollback was triggered due to safety concerns")
        else:
            logger.error(f"❌ Workflow failed with result: {result}")
    
    finally:
        # Cleanup
        await orchestrator.shutdown()
        logger.info("Safety orchestrator shutdown")


async def example_validation_only():
    """Example of validation-only workflow."""
    logger.info("=== Validation Only Example ===")
    
    # Create development configuration for testing
    config = SafetyConfig.create_development_config()
    orchestrator = SafetyOrchestrator(config)
    
    try:
        await orchestrator.initialize()
        
        # Create sample improvements
        improvements = await create_sample_improvements()
        
        # Validate improvements without applying them
        validation_results = await orchestrator.validate_improvements_only(improvements)
        
        logger.info(f"Validation results for {len(validation_results)} improvements:")
        
        for improvement_id, result in validation_results.items():
            status = "✅ PASS" if result.passed else "❌ FAIL"
            logger.info(f"  {improvement_id}: {status}")
            
            if result.issues:
                for issue in result.issues:
                    logger.info(f"    - {issue.severity.upper()}: {issue.message}")
            
            if result.warnings:
                for warning in result.warnings:
                    logger.info(f"    - WARNING: {warning}")
    
    finally:
        await orchestrator.shutdown()


async def example_manual_rollback():
    """Example of creating and using manual rollback points."""
    logger.info("=== Manual Rollback Example ===")
    
    config = SafetyConfig.create_standard_config()
    orchestrator = SafetyOrchestrator(config)
    
    try:
        await orchestrator.initialize()
        
        # Create a manual safety checkpoint
        rollback_point = await orchestrator.create_safety_checkpoint(
            name="Pre-deployment checkpoint",
            description="Safety checkpoint before deploying new features"
        )
        
        logger.info(f"Created rollback point: {rollback_point.rollback_id}")
        
        # Simulate some work that might need rollback
        logger.info("Performing risky operations...")
        await asyncio.sleep(1)
        
        # Get available rollback points
        rollback_points = await orchestrator.rollback_manager.get_rollback_points()
        logger.info(f"Available rollback points: {len(rollback_points)}")
        
        for rp in rollback_points:
            logger.info(f"  - {rp.name} (ID: {rp.rollback_id}, State: {rp.state})")
    
    finally:
        await orchestrator.shutdown()


async def example_health_monitoring():
    """Example of health monitoring capabilities."""
    logger.info("=== Health Monitoring Example ===")
    
    config = SafetyConfig(
        health_monitoring_enabled=True,
        health_check_interval_seconds=5,
        baseline_collection_duration_seconds=10  # Short duration for example
    )
    
    orchestrator = SafetyOrchestrator(config)
    
    try:
        await orchestrator.initialize()
        
        # Collect health baseline
        logger.info("Collecting health baseline...")
        baseline = await orchestrator.health_monitor.collect_baseline(duration_seconds=5)
        
        logger.info(f"Baseline collected with {len(baseline.metrics)} metrics over {baseline.sample_count} samples")
        
        # Show current health metrics
        current_metrics = await orchestrator.health_monitor.collect_current_metrics()
        logger.info(f"Current health status: {current_metrics.overall_status}")
        logger.info(f"Health summary: {current_metrics.summary}")
        
        # Show individual metrics
        for metric_name, metric in current_metrics.metrics.items():
            logger.info(f"  {metric_name}: {metric.value:.2f} {metric.unit} ({metric.status})")
    
    finally:
        await orchestrator.shutdown()


async def run_all_examples():
    """Run all example workflows."""
    examples = [
        ("Validation Only", example_validation_only),
        ("Health Monitoring", example_health_monitoring),
        ("Manual Rollback", example_manual_rollback),
        ("Complete Safety Workflow", example_safe_improvement_workflow),
    ]
    
    for name, example_func in examples:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {name}")
        logger.info('='*50)
        
        try:
            await example_func()
            logger.info(f"✅ {name} completed successfully")
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
        
        # Small delay between examples
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(run_all_examples())