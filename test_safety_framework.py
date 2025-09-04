#!/usr/bin/env python3
"""Standalone test for the safety validation framework."""

import asyncio
import logging
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agentsmcp.retrospective.safety.safety_config import SafetyConfig, SafetyLevel
from agentsmcp.retrospective.safety.safety_validator import SafetyValidator
from agentsmcp.retrospective.safety.health_monitor import HealthMonitor
from agentsmcp.retrospective.safety.rollback_manager import RollbackManager
from agentsmcp.retrospective.safety.safety_orchestrator import SafetyOrchestrator
from agentsmcp.retrospective.data_models import ActionPoint, PriorityLevel, SystemicImprovement, ImprovementCategory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_safety_config():
    """Test safety configuration."""
    print("\n=== Testing Safety Configuration ===")
    
    # Test different config types
    dev_config = SafetyConfig.create_development_config()
    prod_config = SafetyConfig.create_production_config()
    emergency_config = SafetyConfig.create_emergency_config()
    
    print(f"✅ Development config: {dev_config.safety_level}, skip_checks={dev_config.should_skip_safety_checks()}")
    print(f"✅ Production config: {prod_config.safety_level}, auto_rollback={prod_config.enable_auto_rollback}")
    print(f"✅ Emergency config: {emergency_config.safety_level}, rollback_timeout={emergency_config.rollback_timeout_seconds}s")
    
    # Test validation
    errors = dev_config.validate()
    print(f"✅ Configuration validation: {len(errors)} errors")
    
    return True


async def test_safety_validator():
    """Test safety validator."""
    print("\n=== Testing Safety Validator ===")
    
    config = SafetyConfig.create_development_config()
    validator = SafetyValidator(config)
    
    print(f"✅ Validator created with {len(validator.rules)} validation rules")
    
    # Test safe improvement
    safe_action = ActionPoint(
        title="Add performance monitoring",
        description="Add monitoring for response times and error rates",
        priority=PriorityLevel.MEDIUM,
        estimated_effort_hours=3.0,
        implementation_steps=[
            "Install monitoring library",
            "Configure metrics collection",
            "Set up dashboards"
        ]
    )
    
    result = await validator.validate_improvement(safe_action)
    print(f"✅ Safe action validation: passed={result.passed}, issues={len(result.issues)}")
    
    # Test risky improvement
    risky_action = ActionPoint(
        title="Replace orchestrator core",
        description="Delete orchestrator decision making and replace with new algorithm",
        priority=PriorityLevel.CRITICAL,
        estimated_effort_hours=20.0,
        implementation_steps=[
            "Remove existing orchestrator logic",
            "Implement new decision engine",
            "Update all agent communications"
        ]
    )
    
    risky_result = await validator.validate_improvement(risky_action)
    print(f"✅ Risky action validation: passed={risky_result.passed}, issues={len(risky_result.issues)}")
    
    if risky_result.issues:
        for issue in risky_result.issues[:3]:  # Show first 3 issues
            print(f"  🚨 {issue.severity.upper()}: {issue.message}")
    
    # Test batch validation
    improvements = [safe_action, risky_action]
    batch_results = await validator.validate_improvements_batch(improvements)
    print(f"✅ Batch validation: {len(batch_results)} results")
    
    return len(risky_result.issues) > 0  # Should have validation issues


async def test_health_monitor():
    """Test health monitoring."""
    print("\n=== Testing Health Monitor ===")
    
    config = SafetyConfig(
        health_monitoring_enabled=True,
        health_check_interval_seconds=1,
        baseline_collection_duration_seconds=3
    )
    
    monitor = HealthMonitor(config)
    
    # Test current metrics collection
    current_metrics = await monitor.collect_current_metrics()
    print(f"✅ Current metrics collected: {len(current_metrics.metrics)} metrics")
    print(f"   Overall health: {current_metrics.overall_status}")
    
    # Show a few metrics
    for i, (name, metric) in enumerate(current_metrics.metrics.items()):
        if i >= 3:  # Show first 3 metrics
            break
        print(f"   {name}: {metric.value:.2f} {metric.unit} ({metric.status})")
    
    # Test baseline collection (short duration for demo)
    print("Collecting baseline...")
    baseline = await monitor.collect_baseline(duration_seconds=2)
    print(f"✅ Baseline collected: {len(baseline.metrics)} metric types, {baseline.sample_count} samples")
    
    return True


async def test_rollback_manager():
    """Test rollback manager."""
    print("\n=== Testing Rollback Manager ===")
    
    config = SafetyConfig.create_development_config()
    rollback_manager = RollbackManager(config)
    
    await rollback_manager.initialize()
    print("✅ Rollback manager initialized")
    
    # Create a rollback point
    rollback_point = await rollback_manager.create_rollback_point(
        name="Test rollback point",
        description="Testing rollback functionality",
        created_by="test_framework",
        capture_configuration=True,
        expires_in_hours=1
    )
    
    print(f"✅ Rollback point created: {rollback_point.rollback_id}")
    print(f"   State: {rollback_point.state}")
    print(f"   Has config snapshot: {rollback_point.configuration_snapshot is not None}")
    
    # List rollback points
    rollback_points = await rollback_manager.get_rollback_points()
    print(f"✅ Available rollback points: {len(rollback_points)}")
    
    return True


async def test_safety_orchestrator():
    """Test safety orchestrator."""
    print("\n=== Testing Safety Orchestrator ===")
    
    config = SafetyConfig.create_development_config()
    # Disable health monitoring for simpler test
    config.health_monitoring_enabled = False
    config.enable_dry_run_mode = True
    
    orchestrator = SafetyOrchestrator(config)
    await orchestrator.initialize()
    print("✅ Safety orchestrator initialized")
    
    # Create test improvements
    improvements = [
        ActionPoint(
            title="Add caching layer",
            description="Implement Redis caching for database queries",
            priority=PriorityLevel.HIGH,
            estimated_effort_hours=4.0
        ),
        SystemicImprovement(
            title="Optimize communication patterns",
            description="Reduce message passing overhead between agents",
            category=ImprovementCategory.PERFORMANCE,
            priority=PriorityLevel.MEDIUM,
            impact_assessment="Should improve response times by 15%",
            effort_assessment="Medium effort, requires coordination updates"
        )
    ]
    
    # Test validation-only workflow
    validation_results = await orchestrator.validate_improvements_only(improvements)
    print(f"✅ Validation-only workflow: {len(validation_results)} results")
    
    passed_count = sum(1 for result in validation_results.values() if result.passed)
    print(f"   {passed_count}/{len(validation_results)} improvements passed validation")
    
    # Test safe application workflow
    async def mock_implementer(improvements_list):
        """Mock improvement implementer."""
        print(f"   Mock implementing {len(improvements_list)} improvements...")
        await asyncio.sleep(0.1)  # Simulate work
    
    result = await orchestrator.safe_apply_improvements(
        improvements=improvements,
        implementer_function=mock_implementer,
        timeout_seconds=30
    )
    
    print(f"✅ Safe application workflow result: {result}")
    
    await orchestrator.shutdown()
    return True


async def test_integration_example():
    """Test a complete integration example."""
    print("\n=== Integration Example ===")
    
    config = SafetyConfig.create_development_config()
    config.enable_dry_run_mode = True
    config.health_monitoring_enabled = False  # Simplified for demo
    
    orchestrator = SafetyOrchestrator(config)
    await orchestrator.initialize()
    
    # Create a mix of safe and risky improvements
    improvements = [
        ActionPoint(
            title="Add error logging",
            description="Enhance error logging with more context",
            priority=PriorityLevel.LOW,
            estimated_effort_hours=1.0
        ),
        ActionPoint(
            title="Critical system change",
            description="Modify orchestrator core decision making logic",
            priority=PriorityLevel.CRITICAL,
            estimated_effort_hours=15.0
        )
    ]
    
    print(f"Testing {len(improvements)} improvements through safety workflow...")
    
    async def mock_implementer(improvements_list):
        for imp in improvements_list:
            print(f"  Implementing: {imp.title}")
            await asyncio.sleep(0.1)
    
    # Execute complete safety workflow
    context = await orchestrator.execute_safe_improvement_workflow(
        improvements=improvements,
        improvement_implementer=mock_implementer,
        timeout_seconds=60
    )
    
    print(f"✅ Workflow completed:")
    print(f"   State: {context.state}")
    print(f"   Result: {context.result}")
    print(f"   Validation results: {len(context.validation_results)}")
    print(f"   Duration: {(context.completed_at - context.started_at).total_seconds():.2f}s")
    
    if context.validation_results:
        failed_validations = [
            result for result in context.validation_results.values()
            if not result.passed
        ]
        if failed_validations:
            print(f"   ⚠️  {len(failed_validations)} improvements failed validation")
    
    await orchestrator.shutdown()
    return True


async def run_all_tests():
    """Run all safety framework tests."""
    tests = [
        ("Safety Configuration", test_safety_config),
        ("Safety Validator", test_safety_validator),
        ("Health Monitor", test_health_monitor),
        ("Rollback Manager", test_rollback_manager),
        ("Safety Orchestrator", test_safety_orchestrator),
        ("Integration Example", test_integration_example),
    ]
    
    print("🚀 Safety Validation Framework Test Suite")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n▶️  Running: {test_name}")
            result = await test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            logger.exception(f"Test {test_name} failed with exception")
    
    print(f"\n{'=' * 50}")
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Safety framework is working correctly.")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)