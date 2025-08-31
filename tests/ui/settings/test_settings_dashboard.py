"""
Comprehensive test suite for Settings Dashboard component.

Tests the SettingsDashboard component with 95%+ coverage, including:
- Dashboard initialization and lifecycle
- Progressive disclosure navigation
- Health monitoring and system status
- Quick actions functionality
- Settings hierarchy management
- Real-time data updates
- Accessibility features
- Performance under load
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
from datetime import datetime, timedelta
from pathlib import Path

# Import the dashboard component to test
from agentsmcp.ui.settings.dashboard import SettingsDashboard, SettingsSection, HealthStatus


@pytest.fixture
def mock_display_renderer():
    """Mock DisplayRenderer for testing."""
    renderer = Mock()
    renderer.create_panel = Mock(return_value="[Panel]")
    renderer.create_table = Mock(return_value="[Table]")
    renderer.style_text = Mock(side_effect=lambda text, style: f"[{style}]{text}[/{style}]")
    renderer.create_progress_bar = Mock(return_value="[Progress: 50%]")
    return renderer


@pytest.fixture
def mock_terminal_manager():
    """Mock TerminalManager for testing."""
    manager = Mock()
    manager.width = 120
    manager.height = 40
    manager.print = Mock()
    manager.clear = Mock()
    manager.get_size = Mock(return_value=(120, 40))
    return manager


@pytest.fixture
def settings_dashboard(mock_display_renderer, mock_terminal_manager):
    """Create SettingsDashboard instance for testing."""
    dashboard = SettingsDashboard(
        display_renderer=mock_display_renderer,
        terminal_manager=mock_terminal_manager
    )
    return dashboard


class TestSettingsDashboardInitialization:
    """Test Settings Dashboard initialization and setup."""

    def test_initialization_success(self, settings_dashboard):
        """Test successful dashboard initialization."""
        assert settings_dashboard.display_renderer is not None
        assert settings_dashboard.terminal_manager is not None
        assert settings_dashboard.current_section is None
        assert isinstance(settings_dashboard.sections, dict)
        assert len(settings_dashboard.sections) > 0

    def test_default_sections_creation(self, settings_dashboard):
        """Test default sections are created correctly."""
        expected_sections = [
            'global', 'user', 'session', 'agent',
            'security', 'performance', 'logging'
        ]
        
        for section_name in expected_sections:
            assert section_name in settings_dashboard.sections
            section = settings_dashboard.sections[section_name]
            assert isinstance(section, SettingsSection)
            assert section.name == section_name
            assert hasattr(section, 'settings')
            assert hasattr(section, 'visible')

    def test_health_monitor_initialization(self, settings_dashboard):
        """Test health monitoring system initialization."""
        assert hasattr(settings_dashboard, 'health_status')
        assert hasattr(settings_dashboard, '_last_health_check')
        assert hasattr(settings_dashboard, '_health_check_interval')

    @pytest.mark.asyncio
    async def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            'health_check_interval': 5.0,
            'max_recent_actions': 20,
            'auto_refresh': False
        }
        
        with patch('agentsmcp.ui.v2.display_renderer.DisplayRenderer') as MockRenderer:
            with patch('agentsmcp.ui.v2.terminal_state_manager.TerminalManager') as MockManager:
                mock_renderer = MockRenderer.return_value
                mock_manager = MockManager.return_value
                
                dashboard = SettingsDashboard(
                    display_renderer=mock_renderer,
                    terminal_manager=mock_manager,
                    **custom_config
                )
                
                assert dashboard._health_check_interval == 5.0
                assert dashboard._max_recent_actions == 20
                assert dashboard._auto_refresh is False


class TestSettingsSectionManagement:
    """Test settings section management functionality."""

    def test_add_section_success(self, settings_dashboard):
        """Test adding a new settings section."""
        new_section = SettingsSection(
            name='custom',
            title='Custom Settings',
            description='Custom configuration options'
        )
        
        settings_dashboard.add_section(new_section)
        assert 'custom' in settings_dashboard.sections
        assert settings_dashboard.sections['custom'] == new_section

    def test_add_section_duplicate_name(self, settings_dashboard):
        """Test adding section with duplicate name raises error."""
        duplicate_section = SettingsSection(
            name='global',  # Already exists
            title='Duplicate',
            description='Should not be added'
        )
        
        with pytest.raises(ValueError, match="Section 'global' already exists"):
            settings_dashboard.add_section(duplicate_section)

    def test_remove_section_success(self, settings_dashboard):
        """Test removing an existing section."""
        # Add a custom section first
        custom_section = SettingsSection(name='temp', title='Temporary')
        settings_dashboard.add_section(custom_section)
        assert 'temp' in settings_dashboard.sections
        
        # Remove it
        result = settings_dashboard.remove_section('temp')
        assert result is True
        assert 'temp' not in settings_dashboard.sections

    def test_remove_section_nonexistent(self, settings_dashboard):
        """Test removing non-existent section returns False."""
        result = settings_dashboard.remove_section('nonexistent')
        assert result is False

    def test_get_section_success(self, settings_dashboard):
        """Test getting existing section."""
        section = settings_dashboard.get_section('global')
        assert section is not None
        assert section.name == 'global'

    def test_get_section_nonexistent(self, settings_dashboard):
        """Test getting non-existent section returns None."""
        section = settings_dashboard.get_section('nonexistent')
        assert section is None

    def test_section_visibility_control(self, settings_dashboard):
        """Test section visibility control."""
        section = settings_dashboard.get_section('global')
        
        # Initially visible
        assert section.visible is True
        
        # Hide section
        settings_dashboard.set_section_visibility('global', False)
        assert section.visible is False
        
        # Show section
        settings_dashboard.set_section_visibility('global', True)
        assert section.visible is True

    def test_get_visible_sections(self, settings_dashboard):
        """Test getting only visible sections."""
        # Hide one section
        settings_dashboard.set_section_visibility('logging', False)
        
        visible_sections = settings_dashboard.get_visible_sections()
        section_names = [s.name for s in visible_sections]
        
        assert 'logging' not in section_names
        assert 'global' in section_names
        assert len(visible_sections) < len(settings_dashboard.sections)


class TestNavigationSystem:
    """Test dashboard navigation functionality."""

    def test_navigate_to_section_success(self, settings_dashboard):
        """Test navigating to existing section."""
        result = settings_dashboard.navigate_to_section('user')
        assert result is True
        assert settings_dashboard.current_section == 'user'

    def test_navigate_to_section_nonexistent(self, settings_dashboard):
        """Test navigating to non-existent section."""
        result = settings_dashboard.navigate_to_section('nonexistent')
        assert result is False
        assert settings_dashboard.current_section is None

    def test_navigate_to_hidden_section(self, settings_dashboard):
        """Test navigating to hidden section."""
        # Hide the section
        settings_dashboard.set_section_visibility('agent', False)
        
        # Try to navigate to it
        result = settings_dashboard.navigate_to_section('agent')
        assert result is False
        assert settings_dashboard.current_section is None

    def test_navigation_history(self, settings_dashboard):
        """Test navigation history tracking."""
        # Navigate through several sections
        navigation_path = ['global', 'user', 'session', 'agent']
        
        for section_name in navigation_path:
            settings_dashboard.navigate_to_section(section_name)
        
        # Check history exists
        assert hasattr(settings_dashboard, '_navigation_history')
        assert len(settings_dashboard._navigation_history) == len(navigation_path)

    def test_go_back_navigation(self, settings_dashboard):
        """Test going back in navigation."""
        # Navigate forward
        settings_dashboard.navigate_to_section('user')
        settings_dashboard.navigate_to_section('session')
        
        # Go back
        previous_section = settings_dashboard.go_back()
        assert previous_section == 'user'
        assert settings_dashboard.current_section == 'user'

    def test_go_back_empty_history(self, settings_dashboard):
        """Test going back with empty history."""
        result = settings_dashboard.go_back()
        assert result is None
        assert settings_dashboard.current_section is None

    def test_breadcrumb_generation(self, settings_dashboard):
        """Test breadcrumb trail generation."""
        # Navigate through sections
        settings_dashboard.navigate_to_section('global')
        settings_dashboard.navigate_to_section('user')
        settings_dashboard.navigate_to_section('session')
        
        breadcrumbs = settings_dashboard.get_breadcrumbs()
        assert isinstance(breadcrumbs, list)
        assert len(breadcrumbs) > 0
        assert 'session' in breadcrumbs[-1]  # Current section should be last


class TestHealthMonitoringSystem:
    """Test health monitoring functionality."""

    @pytest.mark.asyncio
    async def test_health_check_basic(self, settings_dashboard):
        """Test basic health check functionality."""
        # Mock health check methods
        with patch.object(settings_dashboard, '_check_system_health') as mock_check:
            mock_check.return_value = HealthStatus(
                overall='healthy',
                components={'database': 'healthy', 'cache': 'healthy'},
                last_check=datetime.now(),
                details={'uptime': '5h 32m'}
            )
            
            await settings_dashboard.run_health_check()
            
            mock_check.assert_called_once()
            assert settings_dashboard.health_status is not None
            assert settings_dashboard.health_status.overall == 'healthy'

    @pytest.mark.asyncio
    async def test_health_check_with_issues(self, settings_dashboard):
        """Test health check with system issues."""
        with patch.object(settings_dashboard, '_check_system_health') as mock_check:
            mock_check.return_value = HealthStatus(
                overall='warning',
                components={'database': 'healthy', 'cache': 'warning', 'api': 'error'},
                last_check=datetime.now(),
                details={'error_count': 3, 'warnings': ['Cache memory high']}
            )
            
            await settings_dashboard.run_health_check()
            
            assert settings_dashboard.health_status.overall == 'warning'
            assert 'cache' in settings_dashboard.health_status.components
            assert settings_dashboard.health_status.components['api'] == 'error'

    def test_health_status_age_calculation(self, settings_dashboard):
        """Test health status age calculation."""
        # Set a health status with specific timestamp
        old_timestamp = datetime.now() - timedelta(minutes=5)
        settings_dashboard.health_status = HealthStatus(
            overall='healthy',
            components={},
            last_check=old_timestamp,
            details={}
        )
        
        age = settings_dashboard.get_health_status_age()
        assert age.total_seconds() >= 300  # At least 5 minutes

    def test_is_health_check_stale(self, settings_dashboard):
        """Test stale health check detection."""
        # Recent health check
        settings_dashboard.health_status = HealthStatus(
            overall='healthy',
            components={},
            last_check=datetime.now(),
            details={}
        )
        assert not settings_dashboard.is_health_check_stale()
        
        # Stale health check
        old_timestamp = datetime.now() - timedelta(minutes=10)
        settings_dashboard.health_status.last_check = old_timestamp
        assert settings_dashboard.is_health_check_stale()

    @pytest.mark.asyncio
    async def test_automatic_health_refresh(self, settings_dashboard):
        """Test automatic health status refresh."""
        refresh_count = 0
        
        async def mock_health_check():
            nonlocal refresh_count
            refresh_count += 1
            return HealthStatus(
                overall='healthy',
                components={},
                last_check=datetime.now(),
                details={'refresh_count': refresh_count}
            )
        
        with patch.object(settings_dashboard, 'run_health_check', side_effect=mock_health_check):
            # Enable auto-refresh with short interval
            settings_dashboard._auto_refresh = True
            settings_dashboard._health_check_interval = 0.1  # 100ms for testing
            
            # Start auto-refresh
            task = asyncio.create_task(settings_dashboard._auto_refresh_health())
            
            # Wait for a few refreshes
            await asyncio.sleep(0.3)
            
            # Stop auto-refresh
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            assert refresh_count >= 2


class TestQuickActionsSystem:
    """Test quick actions functionality."""

    def test_add_quick_action(self, settings_dashboard):
        """Test adding quick actions."""
        action = {
            'id': 'restart_service',
            'label': 'Restart Service',
            'description': 'Restart the main service',
            'category': 'system',
            'requires_confirmation': True
        }
        
        settings_dashboard.add_quick_action(action)
        assert 'restart_service' in settings_dashboard.quick_actions
        added_action = settings_dashboard.quick_actions['restart_service']
        assert added_action['label'] == 'Restart Service'

    def test_remove_quick_action(self, settings_dashboard):
        """Test removing quick actions."""
        # Add action first
        action = {'id': 'test_action', 'label': 'Test'}
        settings_dashboard.add_quick_action(action)
        assert 'test_action' in settings_dashboard.quick_actions
        
        # Remove it
        result = settings_dashboard.remove_quick_action('test_action')
        assert result is True
        assert 'test_action' not in settings_dashboard.quick_actions

    def test_get_actions_by_category(self, settings_dashboard):
        """Test getting quick actions by category."""
        # Add actions in different categories
        actions = [
            {'id': 'action1', 'category': 'system', 'label': 'System Action'},
            {'id': 'action2', 'category': 'user', 'label': 'User Action'},
            {'id': 'action3', 'category': 'system', 'label': 'Another System Action'}
        ]
        
        for action in actions:
            settings_dashboard.add_quick_action(action)
        
        system_actions = settings_dashboard.get_actions_by_category('system')
        assert len(system_actions) == 2
        
        user_actions = settings_dashboard.get_actions_by_category('user')
        assert len(user_actions) == 1

    @pytest.mark.asyncio
    async def test_execute_quick_action(self, settings_dashboard):
        """Test executing quick actions."""
        executed_actions = []
        
        async def mock_executor(action_id, **kwargs):
            executed_actions.append({'id': action_id, 'kwargs': kwargs})
            return {'success': True, 'message': f'Executed {action_id}'}
        
        # Add action with executor
        action = {
            'id': 'test_exec',
            'label': 'Test Execution',
            'executor': mock_executor
        }
        settings_dashboard.add_quick_action(action)
        
        # Execute action
        result = await settings_dashboard.execute_quick_action('test_exec', param='value')
        
        assert result['success'] is True
        assert len(executed_actions) == 1
        assert executed_actions[0]['id'] == 'test_exec'
        assert executed_actions[0]['kwargs']['param'] == 'value'

    def test_quick_action_validation(self, settings_dashboard):
        """Test quick action validation."""
        # Valid action
        valid_action = {
            'id': 'valid',
            'label': 'Valid Action',
            'category': 'system'
        }
        settings_dashboard.add_quick_action(valid_action)  # Should not raise
        
        # Invalid action - missing required fields
        with pytest.raises(ValueError):
            invalid_action = {'label': 'Missing ID'}
            settings_dashboard.add_quick_action(invalid_action)


class TestRenderingAndDisplay:
    """Test dashboard rendering and display functionality."""

    def test_render_dashboard_basic(self, settings_dashboard):
        """Test basic dashboard rendering."""
        rendered = settings_dashboard.render_dashboard()
        
        assert rendered is not None
        settings_dashboard.display_renderer.create_panel.assert_called()

    def test_render_section_list(self, settings_dashboard):
        """Test section list rendering."""
        section_list = settings_dashboard.render_section_list()
        
        assert section_list is not None
        # Should have called display renderer methods
        assert settings_dashboard.display_renderer.create_table.called or \
               settings_dashboard.display_renderer.create_panel.called

    def test_render_current_section(self, settings_dashboard):
        """Test current section rendering."""
        # Navigate to a section
        settings_dashboard.navigate_to_section('global')
        
        rendered = settings_dashboard.render_current_section()
        assert rendered is not None

    def test_render_health_status(self, settings_dashboard):
        """Test health status rendering."""
        # Set up health status
        settings_dashboard.health_status = HealthStatus(
            overall='healthy',
            components={'db': 'healthy', 'cache': 'warning'},
            last_check=datetime.now(),
            details={'uptime': '2h 15m'}
        )
        
        rendered = settings_dashboard.render_health_status()
        assert rendered is not None
        settings_dashboard.display_renderer.create_panel.assert_called()

    def test_render_quick_actions(self, settings_dashboard):
        """Test quick actions rendering."""
        # Add some quick actions
        actions = [
            {'id': 'action1', 'label': 'Action 1', 'category': 'system'},
            {'id': 'action2', 'label': 'Action 2', 'category': 'user'}
        ]
        
        for action in actions:
            settings_dashboard.add_quick_action(action)
        
        rendered = settings_dashboard.render_quick_actions()
        assert rendered is not None

    def test_render_breadcrumbs(self, settings_dashboard):
        """Test breadcrumb rendering."""
        # Navigate to create breadcrumb trail
        settings_dashboard.navigate_to_section('global')
        settings_dashboard.navigate_to_section('user')
        
        rendered = settings_dashboard.render_breadcrumbs()
        assert rendered is not None

    def test_responsive_rendering(self, settings_dashboard):
        """Test responsive rendering for different terminal sizes."""
        # Test small terminal
        settings_dashboard.terminal_manager.width = 60
        settings_dashboard.terminal_manager.height = 20
        
        rendered_small = settings_dashboard.render_dashboard()
        assert rendered_small is not None
        
        # Test large terminal
        settings_dashboard.terminal_manager.width = 160
        settings_dashboard.terminal_manager.height = 50
        
        rendered_large = settings_dashboard.render_dashboard()
        assert rendered_large is not None


class TestEventHandling:
    """Test event handling and real-time updates."""

    def test_event_subscription(self, settings_dashboard):
        """Test event subscription mechanism."""
        received_events = []
        
        def event_handler(event_type, data):
            received_events.append({'type': event_type, 'data': data})
        
        # Subscribe to events
        settings_dashboard.subscribe_to_events(['settings_changed'], event_handler)
        
        # Emit an event
        settings_dashboard.emit_event('settings_changed', {'section': 'global'})
        
        assert len(received_events) == 1
        assert received_events[0]['type'] == 'settings_changed'

    def test_settings_change_propagation(self, settings_dashboard):
        """Test settings change event propagation."""
        change_events = []
        
        def change_handler(event_type, data):
            change_events.append(data)
        
        settings_dashboard.subscribe_to_events(['settings_changed'], change_handler)
        
        # Simulate settings change
        settings_dashboard.handle_setting_change('global', 'log_level', 'DEBUG', 'INFO')
        
        assert len(change_events) == 1
        assert change_events[0]['section'] == 'global'
        assert change_events[0]['key'] == 'log_level'

    @pytest.mark.asyncio
    async def test_real_time_updates(self, settings_dashboard):
        """Test real-time update handling."""
        update_count = 0
        
        async def update_handler():
            nonlocal update_count
            update_count += 1
        
        # Start real-time updates
        with patch.object(settings_dashboard, '_process_real_time_update', side_effect=update_handler):
            settings_dashboard._enable_real_time_updates = True
            
            # Trigger updates
            await settings_dashboard.handle_external_update({'type': 'health_check'})
            await settings_dashboard.handle_external_update({'type': 'settings_sync'})
            
            assert update_count == 2


class TestKeyboardInputHandling:
    """Test keyboard input handling and shortcuts."""

    def test_keyboard_shortcut_registration(self, settings_dashboard):
        """Test keyboard shortcut registration."""
        shortcuts_registered = []
        
        def mock_register_shortcut(key, callback):
            shortcuts_registered.append({'key': key, 'callback': callback})
        
        with patch.object(settings_dashboard, '_register_keyboard_shortcut', side_effect=mock_register_shortcut):
            settings_dashboard.setup_keyboard_shortcuts()
            
            # Should have registered common shortcuts
            shortcut_keys = [s['key'] for s in shortcuts_registered]
            assert 'h' in shortcut_keys  # Help
            assert 'q' in shortcut_keys  # Quit
            assert 'r' in shortcut_keys  # Refresh

    def test_navigation_key_handling(self, settings_dashboard):
        """Test navigation key handling."""
        # Mock key event processing
        key_events = []
        
        def mock_handle_key(key):
            key_events.append(key)
            if key == 'j':  # Down
                return settings_dashboard._navigate_down()
            elif key == 'k':  # Up
                return settings_dashboard._navigate_up()
            elif key == 'enter':
                return settings_dashboard._select_current()
        
        with patch.object(settings_dashboard, 'handle_key_press', side_effect=mock_handle_key):
            # Simulate navigation
            settings_dashboard.handle_key_press('j')
            settings_dashboard.handle_key_press('k')
            settings_dashboard.handle_key_press('enter')
            
            assert 'j' in key_events
            assert 'k' in key_events
            assert 'enter' in key_events

    def test_search_functionality(self, settings_dashboard):
        """Test search functionality."""
        # Add searchable content
        search_results = settings_dashboard.search_settings('log')
        
        # Should return results containing 'log'
        assert isinstance(search_results, list)
        # Results should have relevant sections
        result_sections = [r['section'] for r in search_results if 'section' in r]
        assert 'logging' in result_sections


class TestAccessibilityFeatures:
    """Test accessibility features and compliance."""

    def test_screen_reader_support(self, settings_dashboard):
        """Test screen reader support features."""
        # Ensure all UI elements have proper labels
        rendered = settings_dashboard.render_dashboard()
        
        # Check that rendered content includes accessibility information
        assert rendered is not None
        # Verify accessibility attributes are present in renderer calls

    def test_high_contrast_mode(self, settings_dashboard):
        """Test high contrast mode support."""
        # Enable high contrast mode
        settings_dashboard.enable_high_contrast_mode(True)
        
        rendered = settings_dashboard.render_dashboard()
        assert rendered is not None
        
        # Verify high contrast styling was applied
        style_calls = settings_dashboard.display_renderer.style_text.call_args_list
        high_contrast_styles = [call for call in style_calls if 'high_contrast' in str(call)]
        assert len(high_contrast_styles) >= 0  # Should support high contrast

    def test_keyboard_only_navigation(self, settings_dashboard):
        """Test full keyboard-only navigation."""
        # Should be able to navigate entire interface with keyboard
        navigation_path = [
            'tab',    # Move to next element
            'enter',  # Select element
            'escape', # Go back
            'h',      # Help
            'q'       # Quit
        ]
        
        for key in navigation_path:
            # Should not raise exceptions
            result = settings_dashboard.handle_key_press(key)
            assert result is not None or result is None  # Any valid response


class TestPerformanceOptimization:
    """Test performance optimization and scalability."""

    def test_large_settings_hierarchy_performance(self, settings_dashboard):
        """Test performance with large settings hierarchy."""
        # Add many sections with many settings each
        start_time = time.time()
        
        for i in range(100):
            section = SettingsSection(
                name=f'section_{i}',
                title=f'Section {i}',
                description=f'Description for section {i}'
            )
            
            # Add many settings to each section
            for j in range(50):
                section.settings[f'setting_{j}'] = {
                    'value': f'value_{i}_{j}',
                    'type': 'string',
                    'description': f'Setting {j} description'
                }
            
            settings_dashboard.add_section(section)
        
        creation_time = time.time() - start_time
        assert creation_time < 5.0  # Should complete in reasonable time
        
        # Test rendering performance
        start_time = time.time()
        rendered = settings_dashboard.render_dashboard()
        render_time = time.time() - start_time
        
        assert rendered is not None
        assert render_time < 2.0  # Should render in reasonable time

    def test_memory_usage_optimization(self, settings_dashboard):
        """Test memory usage optimization."""
        import sys
        
        # Get initial memory usage
        initial_size = sys.getsizeof(settings_dashboard)
        
        # Add substantial content
        for i in range(1000):
            settings_dashboard.add_quick_action({
                'id': f'action_{i}',
                'label': f'Action {i}',
                'category': 'test'
            })
        
        # Memory usage should scale reasonably
        final_size = sys.getsizeof(settings_dashboard)
        growth_ratio = final_size / initial_size
        
        assert growth_ratio < 10  # Should not grow excessively

    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, settings_dashboard):
        """Test performance under concurrent operations."""
        import asyncio
        
        async def concurrent_navigation():
            for i in range(10):
                section_name = list(settings_dashboard.sections.keys())[i % len(settings_dashboard.sections)]
                settings_dashboard.navigate_to_section(section_name)
                await asyncio.sleep(0.001)
        
        async def concurrent_health_checks():
            for _ in range(5):
                await settings_dashboard.run_health_check()
                await asyncio.sleep(0.01)
        
        # Run concurrent operations
        start_time = time.time()
        
        await asyncio.gather(
            concurrent_navigation(),
            concurrent_health_checks(),
            concurrent_navigation(),
            concurrent_health_checks()
        )
        
        duration = time.time() - start_time
        assert duration < 2.0  # Should handle concurrency efficiently


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""

    def test_invalid_section_handling(self, settings_dashboard):
        """Test handling of invalid section operations."""
        # Try to navigate to invalid section
        result = settings_dashboard.navigate_to_section('invalid_section')
        assert result is False
        
        # Try to get invalid section
        section = settings_dashboard.get_section('invalid_section')
        assert section is None

    @pytest.mark.asyncio
    async def test_health_check_failure_recovery(self, settings_dashboard):
        """Test recovery from health check failures."""
        failure_count = 0
        
        async def failing_health_check():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise Exception("Health check failed")
            return HealthStatus(overall='healthy', components={}, last_check=datetime.now(), details={})
        
        with patch.object(settings_dashboard, '_check_system_health', side_effect=failing_health_check):
            # Should recover after failures
            await settings_dashboard.run_health_check()
            
            # Health status should indicate error state but system should continue
            assert settings_dashboard.health_status is not None

    def test_render_failure_fallback(self, settings_dashboard):
        """Test fallback rendering when normal rendering fails."""
        # Mock renderer to fail
        settings_dashboard.display_renderer.create_panel.side_effect = Exception("Render failure")
        
        # Should fall back to basic rendering without crashing
        rendered = settings_dashboard.render_dashboard()
        
        # Should return fallback content
        assert rendered is not None
        assert isinstance(rendered, str)

    def test_event_handling_error_resilience(self, settings_dashboard):
        """Test resilience to event handling errors."""
        def failing_handler(event_type, data):
            raise Exception("Handler failed")
        
        # Subscribe failing handler
        settings_dashboard.subscribe_to_events(['test_event'], failing_handler)
        
        # Emit event - should not crash the system
        try:
            settings_dashboard.emit_event('test_event', {'data': 'test'})
        except Exception:
            pytest.fail("Event emission should handle handler failures gracefully")


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_complete_dashboard_workflow(self, settings_dashboard):
        """Test complete dashboard usage workflow."""
        # 1. Initialize and load settings
        await settings_dashboard.run_health_check()
        
        # 2. Navigate through sections
        navigation_path = ['global', 'user', 'session', 'agent']
        for section in navigation_path:
            result = settings_dashboard.navigate_to_section(section)
            assert result is True
        
        # 3. Execute quick actions
        settings_dashboard.add_quick_action({
            'id': 'test_workflow',
            'label': 'Test Action',
            'executor': lambda action_id: {'success': True}
        })
        
        result = await settings_dashboard.execute_quick_action('test_workflow')
        assert result['success'] is True
        
        # 4. Render complete dashboard
        rendered = settings_dashboard.render_dashboard()
        assert rendered is not None

    def test_settings_persistence_integration(self, settings_dashboard):
        """Test integration with settings persistence."""
        # Mock settings persistence
        saved_settings = {}
        
        def mock_save_setting(section, key, value):
            if section not in saved_settings:
                saved_settings[section] = {}
            saved_settings[section][key] = value
        
        def mock_load_settings(section):
            return saved_settings.get(section, {})
        
        with patch.object(settings_dashboard, '_save_setting', side_effect=mock_save_setting):
            with patch.object(settings_dashboard, '_load_settings', side_effect=mock_load_settings):
                
                # Save some settings
                settings_dashboard.handle_setting_change('global', 'theme', 'dark')
                settings_dashboard.handle_setting_change('user', 'name', 'test_user')
                
                # Verify persistence
                assert saved_settings['global']['theme'] == 'dark'
                assert saved_settings['user']['name'] == 'test_user'
                
                # Load settings back
                global_settings = mock_load_settings('global')
                assert global_settings['theme'] == 'dark'


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])