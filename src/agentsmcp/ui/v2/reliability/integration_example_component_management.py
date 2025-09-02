"""
Component Management Integration Example.

Demonstrates how to use the component management layer (ComponentInitializer and 
HealthMonitor) with the startup orchestrator to prevent TUI hangs during initialization.

This shows the complete flow:
1. Component initialization with timeout protection
2. Health monitoring startup
3. Integration with startup orchestrator
4. Graceful handling of failures and recovery

Usage:
    python integration_example_component_management.py
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import component management components
from .component_initializer import (
    ComponentInitializer,
    ComponentType,
    ComponentSpec,
    InitializationMode,
    initialize_tui_components,
    get_global_initializer
)

from .health_monitor import (
    HealthMonitor,
    HealthStatus,
    PerformanceReport,
    HangDetectionConfig,
    start_tui_health_monitoring,
    get_global_health_monitor
)

from .startup_orchestrator import (
    StartupOrchestrator,
    StartupResult,
    coordinate_tui_startup
)

from .timeout_guardian import get_global_guardian


class TUILifecycleManager:
    """
    Complete TUI lifecycle manager demonstrating component management integration.
    
    This shows how all the pieces work together:
    - Component initialization with parallel execution and timeouts
    - Health monitoring with hang detection
    - Startup orchestration with fallback modes
    - Recovery mechanisms when problems occur
    """
    
    def __init__(self):
        """Initialize the TUI lifecycle manager."""
        self.component_initializer = get_global_initializer()
        self.health_monitor = get_global_health_monitor()
        self.startup_orchestrator = StartupOrchestrator()
        self.timeout_guardian = get_global_guardian()
        
        self.components: Dict[str, Any] = {}
        self.startup_successful = False
        self.monitoring_active = False
        
    async def start_tui_system(self) -> bool:
        """
        Start the complete TUI system with component management.
        
        Returns:
            bool: True if startup successful, False otherwise
        """
        logger.info("Starting TUI system with component management")
        
        try:
            # Step 1: Initialize components with timeout protection
            logger.info("Step 1: Initializing TUI components...")
            init_result = await self._initialize_components()
            
            if not init_result:
                logger.error("Component initialization failed")
                return False
                
            # Step 2: Start health monitoring
            logger.info("Step 2: Starting health monitoring...")
            monitoring_result = await self._start_health_monitoring()
            
            if not monitoring_result:
                logger.warning("Health monitoring failed to start - continuing without it")
                
            # Step 3: Coordinate startup with orchestrator
            logger.info("Step 3: Coordinating TUI startup...")
            startup_result = await self._coordinate_startup()
            
            if startup_result == StartupResult.SUCCESS:
                logger.info("✅ TUI system startup completed successfully")
                self.startup_successful = True
                return True
            elif startup_result == StartupResult.FALLBACK:
                logger.warning("⚠️ TUI system started in fallback mode")
                self.startup_successful = True
                return True
            else:
                logger.error("❌ TUI system startup failed")
                return False
                
        except Exception as e:
            logger.error(f"TUI system startup failed with exception: {e}")
            await self._cleanup()
            return False
            
    async def _initialize_components(self) -> bool:
        """Initialize TUI components with timeout protection."""
        try:
            # Use the component initializer to set up all TUI components
            result = await initialize_tui_components(
                timeout_seconds=5.0,  # Total 5 second timeout
                mode=InitializationMode.ADAPTIVE  # Use adaptive mode
            )
            
            components = result['components']
            failed = result['failed']
            metrics = result['metrics']
            
            logger.info(f"Component initialization complete: {len(components)} successful, {len(failed)} failed")
            logger.info(f"Initialization took {metrics.total_duration_seconds:.2f}s")
            
            # Store successfully initialized components
            self.components.update(components)
            
            # Report on failed components
            if failed:
                logger.warning("Some components failed to initialize:")
                for comp_name, error in failed.items():
                    logger.warning(f"  - {comp_name}: {error}")
                    
            # Determine if we have enough components to proceed
            essential_components = ['terminal_controller', 'display_manager']
            has_essentials = all(comp in components for comp in essential_components)
            
            if not has_essentials:
                logger.error("Essential components failed to initialize")
                return False
                
            logger.info("✅ Component initialization successful")
            return True
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            return False
            
    async def _start_health_monitoring(self) -> bool:
        """Start health monitoring with hang detection."""
        try:
            # Configure hang detection for TUI-specific thresholds
            hang_config = HangDetectionConfig(
                response_timeout_seconds=5.0,     # UI must respond within 5s
                update_timeout_seconds=5.0,       # UI must update within 5s  
                event_timeout_seconds=10.0,       # Events must be processed within 10s
                memory_threshold_mb=1000.0,       # Alert if memory > 1GB
                cpu_threshold_percent=90.0,       # Alert if CPU > 90%
                fps_minimum=5.0                   # Alert if FPS < 5
            )
            
            # Start health monitoring
            self.health_monitor = await start_tui_health_monitoring(
                check_interval_seconds=1.0,  # Check every second
                hang_config=hang_config
            )
            
            # Set up callbacks for health issues
            self.health_monitor.add_health_callback(self._on_health_status_change)
            self.health_monitor.add_hang_callback(self._on_hang_detected)
            
            self.monitoring_active = True
            logger.info("✅ Health monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start health monitoring: {e}")
            return False
            
    async def _coordinate_startup(self) -> StartupResult:
        """Coordinate TUI startup using the startup orchestrator."""
        try:
            # Use startup orchestrator to coordinate final startup
            startup_components = {
                'component_initializer': self.component_initializer,
                'health_monitor': self.health_monitor,
                'components': self.components
            }
            
            result = await coordinate_tui_startup(startup_components)
            
            if result == StartupResult.SUCCESS:
                logger.info("Startup coordination successful - full TUI mode")
            elif result == StartupResult.FALLBACK:
                logger.warning("Startup coordination fallback - basic TUI mode")
            else:
                logger.error("Startup coordination failed")
                
            return result
            
        except Exception as e:
            logger.error(f"Startup coordination failed: {e}")
            return StartupResult.FAILURE
            
    async def _on_health_status_change(self, report: PerformanceReport) -> None:
        """Handle health status changes."""
        if report.overall_status == HealthStatus.UNHEALTHY:
            logger.warning(f"TUI health degraded: {report.overall_status.value}")
            if report.alerts:
                for alert in report.alerts:
                    logger.warning(f"  Alert: {alert}")
                    
        elif report.overall_status == HealthStatus.HANGING:
            logger.critical("TUI hang detected - initiating recovery")
            await self._handle_hang_recovery()
            
    async def _on_hang_detected(self, reason: str) -> None:
        """Handle hang detection."""
        logger.critical(f"TUI hang detected: {reason}")
        await self._handle_hang_recovery()
        
    async def _handle_hang_recovery(self) -> None:
        """Handle TUI hang recovery."""
        logger.info("Attempting TUI hang recovery...")
        
        try:
            # Attempt to recover by reinitializing components
            logger.info("Reinitializing components for recovery...")
            recovery_result = await self._initialize_components()
            
            if recovery_result:
                logger.info("✅ TUI hang recovery successful")
            else:
                logger.error("❌ TUI hang recovery failed")
                
        except Exception as e:
            logger.error(f"TUI hang recovery failed: {e}")
            
    async def stop_tui_system(self) -> None:
        """Stop the TUI system gracefully."""
        logger.info("Stopping TUI system...")
        await self._cleanup()
        logger.info("✅ TUI system stopped")
        
    async def _cleanup(self) -> None:
        """Clean up all TUI resources."""
        try:
            # Stop health monitoring
            if self.monitoring_active and self.health_monitor._monitoring_active:
                await self.health_monitor.stop_monitoring()
                self.monitoring_active = False
                
            # Clean up components
            for component_name, component in self.components.items():
                if hasattr(component, 'cleanup') and callable(component.cleanup):
                    try:
                        if asyncio.iscoroutinefunction(component.cleanup):
                            await component.cleanup()
                        else:
                            component.cleanup()
                    except Exception as e:
                        logger.warning(f"Component {component_name} cleanup failed: {e}")
                        
            self.components.clear()
            self.startup_successful = False
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current TUI system status."""
        status = {
            'startup_successful': self.startup_successful,
            'monitoring_active': self.monitoring_active,
            'components_count': len(self.components),
            'components': list(self.components.keys())
        }
        
        if self.monitoring_active:
            health_status = await self.health_monitor.get_current_health_status()
            status['health_status'] = health_status.value
            status['performance_summary'] = await self.health_monitor.get_performance_summary()
            
        return status


async def main():
    """Main example execution."""
    print("Component Management Integration Example")
    print("=" * 50)
    
    # Create TUI lifecycle manager
    tui_manager = TUILifecycleManager()
    
    try:
        # Start the TUI system
        print("Starting TUI system...")
        success = await tui_manager.start_tui_system()
        
        if success:
            print("✅ TUI system started successfully!")
            
            # Let it run for a few seconds to demonstrate monitoring
            print("Running TUI system for 5 seconds...")
            for i in range(5):
                await asyncio.sleep(1)
                status = await tui_manager.get_system_status()
                print(f"  Status check {i+1}/5: {status['health_status'] if 'health_status' in status else 'Unknown'}")
                
                # Simulate UI activity to prevent hang detection
                if tui_manager.monitoring_active:
                    await tui_manager.health_monitor.record_ui_response()
                    await tui_manager.health_monitor.record_ui_update()
                    
        else:
            print("❌ TUI system failed to start")
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always clean up
        print("Cleaning up...")
        await tui_manager.stop_tui_system()
        print("✅ Example completed")


if __name__ == '__main__':
    asyncio.run(main())