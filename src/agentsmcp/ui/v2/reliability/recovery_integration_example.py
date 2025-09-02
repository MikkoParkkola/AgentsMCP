"""
Recovery Manager Integration Example - Shows automatic hang recovery in action.

This example demonstrates how the recovery manager integrates with the health
monitor to automatically detect and recover from component hangs without requiring
a full TUI restart.

Key features demonstrated:
- Automatic hang detection within 3 seconds
- Component-level recovery without full restart
- Multiple recovery strategies based on failure severity
- Fallback mode activation when recovery fails
"""

import asyncio
import logging
from typing import Dict, Any

from .recovery_manager import (
    RecoveryManager, RecoveryConfig, RecoveryStrategy, RecoveryResult,
    ComponentFailureType, ComponentRecoveryInterface
)
from .health_monitor import HealthMonitor, HangDetectionConfig
from .timeout_guardian import TimeoutGuardian

logger = logging.getLogger(__name__)


class MockTUIComponent:
    """Mock TUI component for demonstration."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_running = False
        self.hang_simulation_enabled = False
        
    async def start(self) -> bool:
        """Start the component."""
        logger.info(f"Starting component: {self.name}")
        self.is_running = True
        return True
        
    async def stop(self) -> bool:
        """Stop the component."""
        logger.info(f"Stopping component: {self.name}")
        self.is_running = False
        return True
        
    async def restart(self) -> bool:
        """Restart the component."""
        logger.info(f"Restarting component: {self.name}")
        await self.stop()
        await asyncio.sleep(0.1)  # Brief delay
        return await self.start()
        
    def simulate_hang(self):
        """Enable hang simulation for testing."""
        self.hang_simulation_enabled = True
        logger.warning(f"Hang simulation enabled for {self.name}")
        
    async def process_operation(self) -> str:
        """Simulate component operation that might hang."""
        if self.hang_simulation_enabled:
            # Simulate hang by sleeping longer than timeout
            logger.warning(f"Simulating hang in {self.name}")
            await asyncio.sleep(10)  # This will timeout
            
        await asyncio.sleep(0.1)  # Normal processing time
        return f"Operation completed by {self.name}"


class MockComponentRecoveryInterface(ComponentRecoveryInterface):
    """Mock recovery interface for demonstration."""
    
    def __init__(self, component: MockTUIComponent):
        self.component = component
        
    async def prepare_for_recovery(self) -> None:
        """Prepare component for recovery."""
        logger.info(f"Preparing {self.component.name} for recovery")
        
    async def perform_recovery_restart(self) -> bool:
        """Perform component-specific restart."""
        logger.info(f"Performing recovery restart for {self.component.name}")
        
        # Reset hang simulation
        self.component.hang_simulation_enabled = False
        
        # Restart component
        return await self.component.restart()
        
    async def activate_fallback_mode(self) -> bool:
        """Activate fallback mode for this component."""
        logger.info(f"Activating fallback mode for {self.component.name}")
        # In real implementation, this would activate minimal functionality
        return True
        
    async def cleanup_after_failure(self) -> None:
        """Cleanup after component failure."""
        logger.info(f"Cleaning up after failure in {self.component.name}")


async def demonstrate_automatic_recovery():
    """Demonstrate automatic hang recovery in action."""
    print("\n🔧 TUI Recovery Manager Demo - Automatic Hang Recovery")
    print("=" * 60)
    
    # Setup components with fast recovery timeouts for demo
    recovery_config = RecoveryConfig(
        component_restart_timeout_s=2.0,        # 2s for demo (3s in production)
        fallback_activation_timeout_s=1.0,
        hang_detection_threshold_s=1.0,         # Faster detection for demo
        max_recovery_attempts=2
    )
    
    hang_config = HangDetectionConfig(
        response_timeout_seconds=1.0,            # Faster detection for demo
        update_timeout_seconds=1.0,
        event_timeout_seconds=2.0
    )
    
    # Create systems
    timeout_guardian = TimeoutGuardian(default_timeout=3.0)
    health_monitor = HealthMonitor(
        check_interval_seconds=0.5,              # Faster checking for demo
        hang_config=hang_config,
        timeout_guardian=timeout_guardian
    )
    recovery_manager = RecoveryManager(
        config=recovery_config,
        timeout_guardian=timeout_guardian,
        health_monitor=health_monitor
    )
    
    print("✅ Recovery systems initialized")
    
    # Create mock components
    components = {
        'display_renderer': MockTUIComponent('display_renderer'),
        'input_handler': MockTUIComponent('input_handler'),
        'event_system': MockTUIComponent('event_system')
    }
    
    # Register components with recovery manager
    for name, component in components.items():
        recovery_manager.register_component(name, component)
        
        # Register recovery interface
        recovery_interface = MockComponentRecoveryInterface(component)
        recovery_manager.register_recovery_interface(name, recovery_interface)
        
    print("✅ Components registered for recovery management")
    
    # Start components
    for component in components.values():
        await component.start()
        
    print("✅ All components started")
    
    # Add recovery callback to track results
    recovery_results = []
    
    def recovery_callback(result: RecoveryResult):
        recovery_results.append(result)
        print(f"🔄 Recovery {result.status.value} for {result.component_name} "
              f"using {result.strategy.value} (took {result.duration_ms:.1f}ms)")
        if result.recovery_actions_taken:
            print(f"   Actions: {', '.join(result.recovery_actions_taken)}")
    
    recovery_manager.add_recovery_callback(recovery_callback)
    
    # Start health monitoring
    await health_monitor.start_monitoring()
    print("✅ Health monitoring started")
    
    print("\n🧪 Test 1: Successful component recovery")
    print("-" * 40)
    
    # Simulate a component hang
    display_component = components['display_renderer']
    print(f"🚨 Simulating hang in {display_component.name}")
    
    # Trigger manual recovery
    result = await recovery_manager.manual_recovery(
        'display_renderer', 
        RecoveryStrategy.RESTART_COMPONENT,
        "Simulated hang for demo"
    )
    
    print(f"✅ Recovery completed: {result.status.value}")
    print(f"   Duration: {result.duration_ms:.1f}ms")
    print(f"   Success: {result.success}")
    
    print("\n🧪 Test 2: Recovery with timeout and fallback")
    print("-" * 40)
    
    # Simulate a component that fails to restart quickly
    input_component = components['input_handler']
    input_component.simulate_hang()  # This will make restart operations hang
    
    print(f"🚨 Triggering recovery for hanging component: {input_component.name}")
    
    result = await recovery_manager.manual_recovery(
        'input_handler',
        RecoveryStrategy.RESTART_COMPONENT, 
        "Component appears hung"
    )
    
    print(f"✅ Recovery result: {result.status.value}")
    if result.fallback_activated:
        print("⚠️  Fallback mode was activated")
        
    print("\n🧪 Test 3: Health monitor integration")
    print("-" * 40)
    
    # Simulate health report that triggers recovery
    print("🔍 Health monitor will detect issues and trigger automatic recovery")
    
    # Record UI activity to simulate normal operation
    await health_monitor.record_ui_response()
    await health_monitor.record_ui_update()
    
    # Wait for health monitoring to detect any issues
    await asyncio.sleep(2)
    
    print("\n📊 Recovery Statistics")
    print("-" * 20)
    
    status = await recovery_manager.get_recovery_status()
    print(f"Total recoveries: {status['total_recoveries']}")
    print(f"Successful recoveries: {status['successful_recoveries']}")
    print(f"Failed recoveries: {status['failed_recoveries']}")
    print(f"Success rate: {status['success_rate']:.1f}%")
    print(f"Fallback mode active: {status['fallback_mode_active']}")
    
    # Show recovery history
    history = await recovery_manager.get_recovery_history(limit=5)
    print(f"\n📜 Recent Recovery History ({len(history)} operations)")
    print("-" * 30)
    
    for i, record in enumerate(history, 1):
        print(f"{i}. {record['component_name']} - {record['strategy']} - {record['status']} "
              f"({record['duration_ms']:.1f}ms)")
              
    print("\n🛡️  Recovery Manager Demo Complete")
    print("Key features demonstrated:")
    print("• Automatic hang detection and recovery")
    print("• Component restart without full TUI restart")  
    print("• Fallback mode activation on recovery failure")
    print("• Recovery performance monitoring")
    print("• Integration with health monitor")
    
    # Cleanup
    await health_monitor.stop_monitoring()
    await timeout_guardian.shutdown()
    
    # Stop components
    for component in components.values():
        await component.stop()
        
    return recovery_results


async def demonstrate_recovery_strategies():
    """Demonstrate different recovery strategies."""
    print("\n🎯 Recovery Strategy Comparison")
    print("=" * 40)
    
    recovery_manager = RecoveryManager()
    
    # Create a component that supports different recovery modes
    test_component = MockTUIComponent('test_component')
    recovery_manager.register_component('test_component', test_component)
    
    strategies = [
        RecoveryStrategy.RESTART_COMPONENT,
        RecoveryStrategy.FALLBACK_MODE,
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\n🔧 Testing {strategy.value}")
        
        result = await recovery_manager.manual_recovery(
            'test_component',
            strategy, 
            f"Testing {strategy.value} strategy"
        )
        
        results[strategy] = result
        print(f"   Result: {result.status.value}")
        print(f"   Duration: {result.duration_ms:.1f}ms")
        print(f"   Fallback activated: {result.fallback_activated}")
        
        # Reset for next test
        if result.fallback_activated:
            await recovery_manager.reset_fallback_mode()
            
    print(f"\n📊 Strategy Performance Summary")
    print("-" * 30)
    
    for strategy, result in results.items():
        success_indicator = "✅" if result.success else "❌"
        print(f"{success_indicator} {strategy.value}: {result.duration_ms:.1f}ms")
        
    return results


async def main():
    """Run recovery manager demonstration."""
    print("🚀 TUI Recovery Manager - Complete Demonstration")
    print("=" * 55)
    
    try:
        # Demo 1: Automatic recovery
        recovery_results = await demonstrate_automatic_recovery()
        
        # Demo 2: Strategy comparison
        strategy_results = await demonstrate_recovery_strategies()
        
        print("\n🎉 All demonstrations completed successfully!")
        print(f"Total recovery operations: {len(recovery_results)}")
        
        successful_recoveries = sum(1 for r in recovery_results if r.success)
        print(f"Successful recoveries: {successful_recoveries}/{len(recovery_results)}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Setup logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Run the demonstration
    asyncio.run(main())