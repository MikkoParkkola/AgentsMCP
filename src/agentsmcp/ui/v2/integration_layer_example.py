"""
Integration Layer Example - Demonstrates unified TUI system usage.

Shows how to use the display_manager and unified_tui_coordinator together
to create a conflict-free, integrated TUI experience.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from .unified_tui_coordinator import (
    get_unified_tui_coordinator,
    TUIMode, 
    ComponentConfig,
    start_revolutionary_tui,
    start_basic_tui,
    switch_tui_mode,
    stop_tui
)
from .display_manager import (
    get_display_manager,
    DisplayRegion,
    ContentUpdate,
    RefreshMode,
    RegionType
)
from .orchestrator_integration import OrchestratorTUIIntegration

logger = logging.getLogger(__name__)


class IntegratedTUIExample:
    """
    Example demonstrating the integrated TUI system.
    
    Shows how to properly initialize, use, and coordinate the
    display manager and unified TUI coordinator together.
    """
    
    def __init__(self):
        """Initialize the example."""
        self.display_manager = None
        self.coordinator = None
        self.running = False
        
    async def setup(self) -> bool:
        """Setup the integrated TUI system."""
        try:
            logger.info("Setting up integrated TUI system...")
            
            # Get singleton instances
            self.display_manager = await get_display_manager()
            self.coordinator = await get_unified_tui_coordinator()
            
            # Configure the coordinator with custom settings
            config = ComponentConfig(
                enable_animations=True,
                enable_rich_rendering=True,
                max_fps=60,
                enable_logging_isolation=True,
                enable_alternate_screen=True,
                performance_monitoring=True,
                error_recovery=True
            )
            
            # Initialize with configuration
            await self.coordinator.initialize(component_config=config)
            
            # Setup display regions for our example
            await self._setup_display_regions()
            
            logger.info("Integrated TUI system setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup integrated TUI system: {e}")
            return False
    
    async def _setup_display_regions(self) -> None:
        """Setup display regions for the example."""
        regions = [
            DisplayRegion(
                region_id="example_header",
                region_type=RegionType.HEADER,
                x=0, y=0,
                width=100, height=3,
                z_index=1
            ),
            DisplayRegion(
                region_id="example_main",
                region_type=RegionType.MAIN,
                x=0, y=3,
                width=70, height=18,
                z_index=0
            ),
            DisplayRegion(
                region_id="example_sidebar",
                region_type=RegionType.SIDEBAR,
                x=70, y=3,
                width=30, height=18,
                z_index=0
            ),
            DisplayRegion(
                region_id="example_footer",
                region_type=RegionType.FOOTER,
                x=0, y=21,
                width=100, height=3,
                z_index=1
            )
        ]
        
        for region in regions:
            success = await self.display_manager.register_region(region)
            if not success:
                logger.warning(f"Failed to register region: {region.region_id}")
    
    async def run_revolutionary_mode_example(self) -> None:
        """Run an example using revolutionary TUI mode."""
        logger.info("Running Revolutionary TUI mode example...")
        
        try:
            # Start revolutionary TUI
            tui_instance, mode_active, status = await self.coordinator.start_tui(TUIMode.REVOLUTIONARY)
            
            if not mode_active:
                logger.error(f"Failed to start revolutionary TUI: {status}")
                return
            
            logger.info(f"Revolutionary TUI started in {status['startup_time_seconds']:.2f}s")
            
            # Use display manager to update content while TUI is running
            await self._update_example_content("Revolutionary Mode Active!")
            
            # Simulate some activity
            await self._simulate_activity()
            
            # Show performance metrics
            await self._show_performance_metrics()
            
        except Exception as e:
            logger.error(f"Error in revolutionary mode example: {e}")
    
    async def run_basic_mode_example(self) -> None:
        """Run an example using basic TUI mode."""
        logger.info("Running Basic TUI mode example...")
        
        try:
            # Switch to basic mode
            success, metrics = await self.coordinator.switch_mode(TUIMode.BASIC)
            
            if not success:
                logger.error(f"Failed to switch to basic TUI: {metrics}")
                return
            
            logger.info(f"Switched to basic TUI in {metrics['switch_time_seconds']:.3f}s")
            
            # Update content for basic mode
            await self._update_example_content("Basic Mode Active!")
            
            # Show simpler content suitable for basic mode
            await self._simulate_basic_activity()
            
        except Exception as e:
            logger.error(f"Error in basic mode example: {e}")
    
    async def run_mode_switching_example(self) -> None:
        """Demonstrate smooth mode switching."""
        logger.info("Running mode switching example...")
        
        modes = [TUIMode.REVOLUTIONARY, TUIMode.BASIC, TUIMode.FALLBACK, TUIMode.REVOLUTIONARY]
        
        for mode in modes:
            try:
                logger.info(f"Switching to {mode.value} mode...")
                
                success, metrics = await self.coordinator.switch_mode(mode)
                if success:
                    logger.info(f"Switched to {mode.value} in {metrics['switch_time_seconds']:.3f}s")
                    await self._update_example_content(f"{mode.value.title()} Mode Active")
                    
                    # Brief pause to show the mode
                    await asyncio.sleep(1.0)
                else:
                    logger.error(f"Failed to switch to {mode.value}: {metrics}")
                
            except Exception as e:
                logger.error(f"Error switching to {mode.value}: {e}")
    
    async def run_orchestrator_integration_example(self) -> None:
        """Demonstrate orchestrator integration."""
        logger.info("Running orchestrator integration example...")
        
        try:
            # Create orchestrator integration
            orchestrator_integration = OrchestratorTUIIntegration()
            
            # Start TUI with orchestrator integration
            tui_instance, mode_active, status = await self.coordinator.start_tui(
                TUIMode.REVOLUTIONARY,
                orchestrator_integration=orchestrator_integration
            )
            
            if mode_active:
                logger.info("TUI started with orchestrator integration")
                await self._update_example_content("Orchestrator Integration Active!")
                
                # Show integration status
                integration_status = self.coordinator.get_integration_status()
                logger.info(f"Orchestrator connected: {integration_status.orchestrator_connected}")
                logger.info(f"Strict mode: {integration_status.strict_mode_enabled}")
                
            else:
                logger.error(f"Failed to start TUI with orchestrator: {status}")
                
        except Exception as e:
            logger.error(f"Error in orchestrator integration example: {e}")
    
    async def _update_example_content(self, header_text: str) -> None:
        """Update example content in display regions."""
        try:
            # Get current regions
            regions = await self.display_manager.get_all_regions()
            
            # Prepare updates
            updates = [
                ContentUpdate(
                    region_id="example_header",
                    content=f"🚀 AgentsMCP TUI - {header_text}",
                    refresh_mode=RefreshMode.PARTIAL,
                    priority=2,
                    requester="example"
                ),
                ContentUpdate(
                    region_id="example_main",
                    content=f"Main Content Area\\n\\nTime: {datetime.now()}\\nMode: {self.coordinator.get_current_mode().value}\\nStatus: {self.coordinator.get_status().value}",
                    refresh_mode=RefreshMode.PARTIAL,
                    priority=1,
                    requester="example"
                ),
                ContentUpdate(
                    region_id="example_sidebar",
                    content="📊 Sidebar\\n\\nMetrics:\\n- CPU: 45%\\n- Memory: 62%\\n- Network: OK\\n\\nComponents:\\n✅ Terminal\\n✅ Display\\n✅ Logging\\n✅ Layout\\n✅ Input",
                    refresh_mode=RefreshMode.PARTIAL,
                    priority=1,
                    requester="example"
                ),
                ContentUpdate(
                    region_id="example_footer",
                    content="🎛️  Integration Layer Example - Press Ctrl+C to exit",
                    refresh_mode=RefreshMode.PARTIAL,
                    priority=2,
                    requester="example"
                )
            ]
            
            # Apply updates
            display_updated, conflict_detected, metrics = await self.display_manager.update_content(
                layout_regions=regions,
                content_updates=updates,
                refresh_mode=RefreshMode.ADAPTIVE
            )
            
            if not display_updated:
                logger.warning("Display update failed")
            if conflict_detected:
                logger.warning("Display conflicts detected")
                
        except Exception as e:
            logger.error(f"Failed to update example content: {e}")
    
    async def _simulate_activity(self) -> None:
        """Simulate TUI activity."""
        logger.info("Simulating TUI activity...")
        
        activities = [
            "Processing user input...",
            "Updating display regions...", 
            "Managing layout engine...",
            "Coordinating components...",
            "Optimizing performance..."
        ]
        
        for activity in activities:
            await self._update_activity_status(activity)
            await asyncio.sleep(0.5)
    
    async def _simulate_basic_activity(self) -> None:
        """Simulate basic TUI activity."""
        logger.info("Simulating basic TUI activity...")
        
        basic_activities = [
            "Basic mode: Simple operations",
            "Basic mode: Stable performance",
            "Basic mode: Minimal resources"
        ]
        
        for activity in basic_activities:
            await self._update_activity_status(activity)
            await asyncio.sleep(0.7)
    
    async def _update_activity_status(self, activity: str) -> None:
        """Update activity status in main region."""
        try:
            regions = await self.display_manager.get_all_regions()
            
            update = ContentUpdate(
                region_id="example_main",
                content=f"Main Content Area\\n\\nCurrent Activity:\\n{activity}\\n\\nTime: {datetime.now()}\\nMode: {self.coordinator.get_current_mode().value}",
                refresh_mode=RefreshMode.PARTIAL,
                requester="activity_simulator"
            )
            
            await self.display_manager.update_content(
                layout_regions=regions,
                content_updates=[update]
            )
            
        except Exception as e:
            logger.error(f"Failed to update activity status: {e}")
    
    async def _show_performance_metrics(self) -> None:
        """Show performance metrics."""
        logger.info("Performance Metrics:")
        
        # Coordinator metrics
        coord_metrics = self.coordinator.get_performance_metrics()
        logger.info(f"  Current Mode: {coord_metrics['current_mode']}")
        logger.info(f"  Status: {coord_metrics['status']}")
        logger.info(f"  Switch Count: {coord_metrics['switch_count']}")
        logger.info(f"  Average Switch Time: {coord_metrics['average_switch_time_ms']:.1f}ms")
        
        # Display manager metrics
        display_metrics = self.display_manager.get_performance_metrics()
        logger.info(f"  Region Count: {display_metrics['region_count']}")
        logger.info(f"  Average Refresh Time: {display_metrics['average_refresh_time_ms']:.1f}ms")
        logger.info(f"  Conflict Rate: {display_metrics['conflict_rate']:.2%}")
    
    async def cleanup(self) -> None:
        """Cleanup the integrated TUI system."""
        logger.info("Cleaning up integrated TUI system...")
        
        try:
            if self.coordinator:
                await self.coordinator.shutdown()
            
            if self.display_manager:
                await self.display_manager.cleanup()
                
            logger.info("Cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def run_complete_example():
    """Run the complete integrated TUI example."""
    logging.basicConfig(level=logging.INFO)
    
    example = IntegratedTUIExample()
    
    try:
        # Setup
        success = await example.setup()
        if not success:
            logger.error("Failed to setup integrated TUI system")
            return
        
        # Run different examples
        logger.info("=== Starting Integrated TUI Examples ===")
        
        # Revolutionary mode example
        await example.run_revolutionary_mode_example()
        await asyncio.sleep(2)
        
        # Basic mode example  
        await example.run_basic_mode_example()
        await asyncio.sleep(2)
        
        # Mode switching example
        await example.run_mode_switching_example()
        await asyncio.sleep(1)
        
        # Orchestrator integration example
        await example.run_orchestrator_integration_example()
        await asyncio.sleep(2)
        
        logger.info("=== All examples completed successfully ===")
        
    except KeyboardInterrupt:
        logger.info("Example interrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}")
    finally:
        await example.cleanup()


# Convenience functions for quick testing

async def quick_revolutionary_demo():
    """Quick demo of revolutionary TUI."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    try:
        logger.info("Starting quick revolutionary TUI demo...")
        
        # Start revolutionary TUI
        mode_active, status = await start_revolutionary_tui()
        
        if mode_active:
            logger.info(f"Revolutionary TUI started in {status['startup_time_seconds']:.2f}s")
            await asyncio.sleep(3)
            
            # Stop TUI
            success, _ = await stop_tui()
            if success:
                logger.info("Revolutionary TUI stopped successfully")
        else:
            logger.error(f"Failed to start revolutionary TUI: {status}")
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")


async def quick_mode_switching_demo():
    """Quick demo of mode switching."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    
    try:
        logger.info("Starting quick mode switching demo...")
        
        # Start in basic mode
        await start_basic_tui()
        logger.info("Started in basic mode")
        await asyncio.sleep(1)
        
        # Switch to revolutionary
        success, metrics = await switch_tui_mode(TUIMode.REVOLUTIONARY)
        if success:
            logger.info(f"Switched to revolutionary in {metrics['switch_time_seconds']:.3f}s")
        await asyncio.sleep(1)
        
        # Switch to fallback
        success, metrics = await switch_tui_mode(TUIMode.FALLBACK) 
        if success:
            logger.info(f"Switched to fallback in {metrics['switch_time_seconds']:.3f}s")
        await asyncio.sleep(1)
        
        # Stop TUI
        await stop_tui()
        logger.info("Mode switching demo completed")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")


if __name__ == "__main__":
    # Run the complete example
    asyncio.run(run_complete_example())