"""Performance optimization tests for AgentsMCP settings management system.

Tests performance characteristics with large settings hierarchies, memory usage,
rendering efficiency, and scalability limits across all UI components.
"""

import pytest
import asyncio
import time
import memory_profiler
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List, Optional, Generator
from datetime import datetime, timedelta
import threading
import gc
import sys

from src.agentsmcp.ui.settings.dashboard import SettingsDashboard
from src.agentsmcp.ui.components.agent_manager import AgentManager
from src.agentsmcp.ui.components.settings_forms import SettingsForm
from src.agentsmcp.ui.components.config_assistant import ConfigurationAssistant
from src.agentsmcp.ui.components.access_control import AccessControlInterface


class PerformanceProfiler:
    """Helper class for performance profiling and measurement."""
    
    def __init__(self):
        self.measurements = []
        self.memory_snapshots = []
    
    async def measure_execution_time(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure execution time of an async function."""
        start_time = time.perf_counter()
        start_memory = memory_profiler.memory_usage()[0]
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            end_time = time.perf_counter()
            end_memory = memory_profiler.memory_usage()[0]
            
            measurement = {
                'function': func.__name__,
                'execution_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'memory_peak': max(memory_profiler.memory_usage()),
                'success': True,
                'result_size': sys.getsizeof(result) if result else 0
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            measurement = {
                'function': func.__name__,
                'execution_time': end_time - start_time,
                'memory_used': 0,
                'memory_peak': 0,
                'success': False,
                'error': str(e)
            }
        
        self.measurements.append(measurement)
        return measurement
    
    def create_large_settings_hierarchy(self, 
                                      global_settings_count: int = 100,
                                      users_count: int = 50,
                                      agents_count: int = 200,
                                      settings_per_level: int = 20) -> Dict[str, Any]:
        """Create a large settings hierarchy for testing."""
        hierarchy = {
            'global': {},
            'users': {},
            'agents': {}
        }
        
        # Global settings
        for i in range(global_settings_count):
            hierarchy['global'][f'global_setting_{i}'] = {
                'value': f'global_value_{i}',
                'type': 'string',
                'default': f'default_{i}',
                'validation': {'required': i % 3 == 0},
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'modified': datetime.now().isoformat(),
                    'version': 1
                }
            }
        
        # User-specific settings
        for user_id in range(users_count):
            user_key = f'user_{user_id}'
            hierarchy['users'][user_key] = {}
            
            for setting_id in range(settings_per_level):
                setting_key = f'user_setting_{setting_id}'
                hierarchy['users'][user_key][setting_key] = {
                    'value': f'user_{user_id}_value_{setting_id}',
                    'type': 'string',
                    'inherited_from': 'global' if setting_id % 2 == 0 else None,
                    'overrides': f'global_setting_{setting_id}' if setting_id < global_settings_count else None
                }
        
        # Agent-specific settings
        for agent_id in range(agents_count):
            agent_key = f'agent_{agent_id}'
            hierarchy['agents'][agent_key] = {}
            
            for setting_id in range(settings_per_level):
                setting_key = f'agent_setting_{setting_id}'
                hierarchy['agents'][agent_key][setting_key] = {
                    'value': f'agent_{agent_id}_value_{setting_id}',
                    'type': 'string',
                    'inherited_from': 'user' if setting_id % 3 == 0 else 'global' if setting_id % 3 == 1 else None,
                    'dependencies': [f'global_setting_{(setting_id + 1) % global_settings_count}']
                }
        
        return hierarchy
    
    def simulate_concurrent_users(self, user_count: int = 10) -> List[Dict[str, Any]]:
        """Simulate concurrent user sessions."""
        sessions = []
        for i in range(user_count):
            session = {
                'user_id': f'concurrent_user_{i}',
                'session_id': f'session_{i}_{int(time.time())}',
                'active_components': ['dashboard', 'settings_form'] if i % 2 == 0 else ['agent_manager'],
                'settings_cache': {},
                'last_activity': datetime.now(),
                'concurrent_operations': []
            }
            sessions.append(session)
        
        return sessions


class TestSettingsDashboardPerformance:
    """Test performance of Settings Dashboard with large datasets."""
    
    @pytest.fixture
    async def profiler(self):
        return PerformanceProfiler()
    
    @pytest.fixture
    async def large_dashboard(self, profiler):
        """Create Settings Dashboard with large dataset."""
        large_hierarchy = profiler.create_large_settings_hierarchy(
            global_settings_count=200,
            users_count=100,
            agents_count=500
        )
        
        mock_settings_service = MagicMock()
        mock_settings_service.get_settings = AsyncMock(return_value=large_hierarchy)
        mock_settings_service.get_health_info = AsyncMock(return_value={
            'total_settings': 10000,
            'active_agents': 500,
            'concurrent_users': 100,
            'system_load': 'high'
        })
        
        dashboard = SettingsDashboard(
            event_system=MagicMock(),
            settings_service=mock_settings_service,
            auth_service=MagicMock()
        )
        
        return dashboard, large_hierarchy
    
    @pytest.mark.asyncio
    async def test_dashboard_initialization_performance(self, large_dashboard, profiler):
        """Test dashboard initialization with large datasets."""
        dashboard, hierarchy = large_dashboard
        
        # Measure initialization time
        measurement = await profiler.measure_execution_time(dashboard.initialize)
        
        # Performance thresholds
        assert measurement['execution_time'] < 2.0, f"Dashboard initialization took {measurement['execution_time']:.2f}s, should be < 2.0s"
        assert measurement['memory_used'] < 50, f"Dashboard initialization used {measurement['memory_used']:.2f}MB, should be < 50MB"
        assert measurement['success'], "Dashboard initialization should succeed"
    
    @pytest.mark.asyncio
    async def test_section_navigation_performance(self, large_dashboard, profiler):
        """Test section navigation performance with large datasets."""
        dashboard, hierarchy = large_dashboard
        await dashboard.initialize()
        
        sections = ['global', 'users', 'agents', 'system', 'security']
        navigation_times = []
        
        for section in sections:
            measurement = await profiler.measure_execution_time(
                dashboard.navigate_to_section, section
            )
            navigation_times.append(measurement['execution_time'])
            
            # Each navigation should be fast
            assert measurement['execution_time'] < 0.5, \
                f"Navigation to {section} took {measurement['execution_time']:.2f}s, should be < 0.5s"
        
        # Average navigation time should be reasonable
        avg_time = sum(navigation_times) / len(navigation_times)
        assert avg_time < 0.2, f"Average navigation time {avg_time:.2f}s should be < 0.2s"
    
    @pytest.mark.asyncio
    async def test_search_performance_large_dataset(self, large_dashboard, profiler):
        """Test search performance across large settings hierarchy."""
        dashboard, hierarchy = large_dashboard
        await dashboard.initialize()
        
        search_queries = [
            'global_setting_1',
            'user_50',
            'agent_100',
            'validation',
            'string'
        ]
        
        for query in search_queries:
            measurement = await profiler.measure_execution_time(
                dashboard.search_settings, query
            )
            
            # Search should be fast even with large datasets
            assert measurement['execution_time'] < 1.0, \
                f"Search for '{query}' took {measurement['execution_time']:.2f}s, should be < 1.0s"
    
    @pytest.mark.asyncio
    async def test_concurrent_dashboard_operations(self, large_dashboard, profiler):
        """Test concurrent operations on dashboard."""
        dashboard, hierarchy = large_dashboard
        await dashboard.initialize()
        
        # Create concurrent operations
        operations = [
            dashboard.navigate_to_section('agents'),
            dashboard.search_settings('test'),
            dashboard.refresh_health_status(),
            dashboard.get_quick_actions(),
            dashboard.navigate_to_section('users')
        ]
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*operations, return_exceptions=True)
        end_time = time.perf_counter()
        
        # Verify all operations completed successfully
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Concurrent operations failed: {exceptions}"
        
        # Total time should be reasonable
        total_time = end_time - start_time
        assert total_time < 3.0, f"Concurrent operations took {total_time:.2f}s, should be < 3.0s"
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, large_dashboard, profiler):
        """Test memory usage patterns under heavy load."""
        dashboard, hierarchy = large_dashboard
        await dashboard.initialize()
        
        initial_memory = memory_profiler.memory_usage()[0]
        
        # Perform memory-intensive operations
        for i in range(50):
            await dashboard.navigate_to_section(f'section_{i % 5}')
            await dashboard.search_settings(f'query_{i}')
            
            if i % 10 == 0:
                # Force garbage collection periodically
                gc.collect()
        
        final_memory = memory_profiler.memory_usage()[0]
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be bounded
        assert memory_growth < 100, f"Memory growth {memory_growth:.2f}MB should be < 100MB"


class TestAgentManagerPerformance:
    """Test performance of Agent Manager with large numbers of agents."""
    
    @pytest.fixture
    async def large_agent_manager(self):
        """Create Agent Manager with large number of agents."""
        # Create mock data for 1000+ agents
        agents = []
        for i in range(1000):
            agents.append({
                'id': f'agent_{i}',
                'name': f'Agent {i}',
                'type': 'chat' if i % 3 == 0 else 'task' if i % 3 == 1 else 'analysis',
                'status': 'active' if i % 4 != 0 else 'inactive',
                'config': {
                    'model': f'model_{i % 5}',
                    'temperature': 0.1 + (i % 10) * 0.1,
                    'max_tokens': 1000 + i * 10
                },
                'metrics': {
                    'requests': i * 100,
                    'success_rate': 0.95 + (i % 10) * 0.005,
                    'avg_response_time': 0.5 + (i % 20) * 0.1
                },
                'created': datetime.now() - timedelta(days=i % 365)
            })
        
        mock_agent_service = MagicMock()
        mock_agent_service.list_agents = AsyncMock(return_value=agents)
        mock_agent_service.get_agent_status = AsyncMock(
            side_effect=lambda agent_id: {'status': 'active', 'health': 'good'}
        )
        
        agent_manager = AgentManager(
            event_system=MagicMock(),
            agent_service=mock_agent_service,
            auth_service=MagicMock()
        )
        
        return agent_manager, agents
    
    @pytest.mark.asyncio
    async def test_agent_list_rendering_performance(self, large_agent_manager):
        """Test agent list rendering with large datasets."""
        agent_manager, agents = large_agent_manager
        
        profiler = PerformanceProfiler()
        
        # Measure initialization
        init_measurement = await profiler.measure_execution_time(agent_manager.initialize)
        assert init_measurement['execution_time'] < 3.0, \
            f"Agent manager initialization took {init_measurement['execution_time']:.2f}s"
        
        # Measure list rendering
        render_measurement = await profiler.measure_execution_time(
            agent_manager.render_agent_list
        )
        assert render_measurement['execution_time'] < 1.0, \
            f"Agent list rendering took {render_measurement['execution_time']:.2f}s"
    
    @pytest.mark.asyncio
    async def test_agent_filtering_performance(self, large_agent_manager):
        """Test agent filtering performance with large datasets."""
        agent_manager, agents = large_agent_manager
        await agent_manager.initialize()
        
        profiler = PerformanceProfiler()
        
        # Test different filter types
        filters = [
            {'type': 'status', 'value': 'active'},
            {'type': 'name', 'value': 'Agent 1'},
            {'type': 'model', 'value': 'model_0'},
            {'type': 'created', 'value': 'last_month'}
        ]
        
        for filter_config in filters:
            measurement = await profiler.measure_execution_time(
                agent_manager.apply_filter, filter_config
            )
            
            assert measurement['execution_time'] < 0.5, \
                f"Filter {filter_config} took {measurement['execution_time']:.2f}s"
    
    @pytest.mark.asyncio
    async def test_bulk_operations_performance(self, large_agent_manager):
        """Test bulk operations performance."""
        agent_manager, agents = large_agent_manager
        await agent_manager.initialize()
        
        profiler = PerformanceProfiler()
        
        # Select 100 agents
        agent_ids = [agent['id'] for agent in agents[:100]]
        
        # Test bulk status update
        measurement = await profiler.measure_execution_time(
            agent_manager.bulk_update_status, agent_ids, 'maintenance'
        )
        
        assert measurement['execution_time'] < 2.0, \
            f"Bulk status update took {measurement['execution_time']:.2f}s"
        
        # Test bulk configuration update
        config_updates = {'temperature': 0.7, 'max_tokens': 2000}
        measurement = await profiler.measure_execution_time(
            agent_manager.bulk_update_config, agent_ids, config_updates
        )
        
        assert measurement['execution_time'] < 3.0, \
            f"Bulk config update took {measurement['execution_time']:.2f}s"


class TestSettingsFormsPerformance:
    """Test performance of Settings Forms with complex form structures."""
    
    @pytest.fixture
    async def complex_settings_form(self):
        """Create Settings Form with complex structure."""
        # Create form with many fields and complex validation
        form_config = {
            'fields': []
        }
        
        # Add 200 fields of various types
        for i in range(200):
            field_type = ['text', 'number', 'select', 'checkbox', 'textarea'][i % 5]
            field = {
                'name': f'field_{i}',
                'type': field_type,
                'label': f'Field {i}',
                'validation': {
                    'required': i % 3 == 0,
                    'min_length': 5 if field_type == 'text' else None,
                    'max_length': 100 if field_type in ['text', 'textarea'] else None,
                    'pattern': r'^\w+$' if i % 10 == 0 else None
                },
                'depends_on': f'field_{i-1}' if i > 0 and i % 7 == 0 else None,
                'options': [f'option_{j}' for j in range(10)] if field_type == 'select' else None
            }
            form_config['fields'].append(field)
        
        settings_form = SettingsForm(
            event_system=MagicMock(),
            settings_service=MagicMock(),
            validation_service=MagicMock()
        )
        
        return settings_form, form_config
    
    @pytest.mark.asyncio
    async def test_form_creation_performance(self, complex_settings_form):
        """Test form creation performance with many fields."""
        settings_form, form_config = complex_settings_form
        
        profiler = PerformanceProfiler()
        
        measurement = await profiler.measure_execution_time(
            settings_form.create_form, 'performance_test', form_config
        )
        
        assert measurement['execution_time'] < 1.0, \
            f"Form creation took {measurement['execution_time']:.2f}s"
        assert measurement['memory_used'] < 30, \
            f"Form creation used {measurement['memory_used']:.2f}MB"
    
    @pytest.mark.asyncio
    async def test_real_time_validation_performance(self, complex_settings_form):
        """Test real-time validation performance."""
        settings_form, form_config = complex_settings_form
        await settings_form.create_form('performance_test', form_config)
        
        profiler = PerformanceProfiler()
        
        # Test validation of multiple fields rapidly
        validation_times = []
        for i in range(50):
            field_name = f'field_{i}'
            test_value = f'test_value_{i}'
            
            measurement = await profiler.measure_execution_time(
                settings_form.validate_field, field_name, test_value
            )
            validation_times.append(measurement['execution_time'])
        
        # Each validation should be fast
        max_validation_time = max(validation_times)
        assert max_validation_time < 0.1, f"Max validation time {max_validation_time:.3f}s should be < 0.1s"
        
        # Average validation time should be very fast
        avg_validation_time = sum(validation_times) / len(validation_times)
        assert avg_validation_time < 0.05, f"Average validation time {avg_validation_time:.3f}s should be < 0.05s"
    
    @pytest.mark.asyncio
    async def test_form_submission_performance(self, complex_settings_form):
        """Test form submission performance with large datasets."""
        settings_form, form_config = complex_settings_form
        await settings_form.create_form('performance_test', form_config)
        
        # Fill form with test data
        for i, field in enumerate(form_config['fields']):
            if field['type'] == 'text':
                await settings_form.update_field_value(field['name'], f'value_{i}')
            elif field['type'] == 'number':
                await settings_form.update_field_value(field['name'], i)
            elif field['type'] == 'checkbox':
                await settings_form.update_field_value(field['name'], i % 2 == 0)
            elif field['type'] == 'select':
                await settings_form.update_field_value(field['name'], 'option_0')
        
        profiler = PerformanceProfiler()
        
        # Test form submission
        measurement = await profiler.measure_execution_time(settings_form.submit_form)
        
        assert measurement['execution_time'] < 2.0, \
            f"Form submission took {measurement['execution_time']:.2f}s"


class TestConfigurationAssistantPerformance:
    """Test performance of Configuration Assistant with AI operations."""
    
    @pytest.fixture
    async def config_assistant_with_load(self):
        """Create Configuration Assistant with simulated AI load."""
        mock_ai_service = MagicMock()
        
        # Simulate AI response delays
        async def mock_ai_call(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate AI processing time
            return {
                'recommendations': [
                    {'type': 'setting', 'value': 'recommended_value', 'confidence': 0.9},
                    {'type': 'config', 'value': 'optimal_config', 'confidence': 0.85}
                ]
            }
        
        mock_ai_service.get_recommendation = AsyncMock(side_effect=mock_ai_call)
        
        config_assistant = ConfigurationAssistant(
            event_system=MagicMock(),
            ai_service=mock_ai_service,
            settings_service=MagicMock()
        )
        
        return config_assistant
    
    @pytest.mark.asyncio
    async def test_concurrent_ai_requests_performance(self, config_assistant_with_load):
        """Test handling of concurrent AI requests."""
        config_assistant = config_assistant_with_load
        await config_assistant.initialize()
        
        profiler = PerformanceProfiler()
        
        # Create multiple concurrent AI requests
        requests = [
            config_assistant.get_smart_recommendation(f'context_{i}', {})
            for i in range(10)
        ]
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*requests)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        
        # Concurrent requests should be processed efficiently
        assert total_time < 2.0, f"10 concurrent AI requests took {total_time:.2f}s, should be < 2.0s"
        assert len(results) == 10, "All AI requests should complete"
    
    @pytest.mark.asyncio
    async def test_recommendation_caching_performance(self, config_assistant_with_load):
        """Test recommendation caching performance."""
        config_assistant = config_assistant_with_load
        await config_assistant.initialize()
        
        profiler = PerformanceProfiler()
        
        # First request (should hit AI service)
        first_measurement = await profiler.measure_execution_time(
            config_assistant.get_smart_recommendation, 'test_context', {'param': 'value'}
        )
        
        # Second request with same parameters (should hit cache)
        second_measurement = await profiler.measure_execution_time(
            config_assistant.get_smart_recommendation, 'test_context', {'param': 'value'}
        )
        
        # Cached request should be much faster
        assert second_measurement['execution_time'] < first_measurement['execution_time'] / 2, \
            "Cached request should be at least 2x faster"


class TestAccessControlInterfacePerformance:
    """Test performance of Access Control Interface with large user bases."""
    
    @pytest.fixture
    async def large_access_control(self):
        """Create Access Control Interface with large user base."""
        # Create large user/role dataset
        users = []
        roles = []
        permissions = []
        
        # Create 1000 users
        for i in range(1000):
            users.append({
                'id': f'user_{i}',
                'username': f'user_{i}',
                'email': f'user_{i}@example.com',
                'roles': [f'role_{i % 10}'],
                'permissions': [f'perm_{j}' for j in range(i % 20)],
                'groups': [f'group_{i % 50}'],
                'created': datetime.now() - timedelta(days=i % 365)
            })
        
        # Create 50 roles
        for i in range(50):
            roles.append({
                'id': f'role_{i}',
                'name': f'Role {i}',
                'permissions': [f'perm_{j}' for j in range(i % 30)],
                'hierarchy_level': i % 5
            })
        
        # Create 100 permissions
        for i in range(100):
            permissions.append({
                'id': f'perm_{i}',
                'name': f'Permission {i}',
                'resource': f'resource_{i % 20}',
                'action': ['read', 'write', 'delete', 'admin'][i % 4]
            })
        
        mock_auth_service = MagicMock()
        mock_auth_service.list_users = AsyncMock(return_value=users)
        mock_auth_service.list_roles = AsyncMock(return_value=roles)
        mock_auth_service.list_permissions = AsyncMock(return_value=permissions)
        
        access_control = AccessControlInterface(
            event_system=MagicMock(),
            auth_service=mock_auth_service,
            user_service=MagicMock()
        )
        
        return access_control, {'users': users, 'roles': roles, 'permissions': permissions}
    
    @pytest.mark.asyncio
    async def test_user_list_performance(self, large_access_control):
        """Test user list performance with large datasets."""
        access_control, data = large_access_control
        
        profiler = PerformanceProfiler()
        
        # Test initialization
        init_measurement = await profiler.measure_execution_time(access_control.initialize)
        assert init_measurement['execution_time'] < 2.0, \
            f"Access control initialization took {init_measurement['execution_time']:.2f}s"
        
        # Test user list rendering
        render_measurement = await profiler.measure_execution_time(
            access_control.render_user_list
        )
        assert render_measurement['execution_time'] < 1.5, \
            f"User list rendering took {render_measurement['execution_time']:.2f}s"
    
    @pytest.mark.asyncio
    async def test_permission_matrix_performance(self, large_access_control):
        """Test permission matrix performance with large datasets."""
        access_control, data = large_access_control
        await access_control.initialize()
        
        profiler = PerformanceProfiler()
        
        # Test permission matrix generation
        measurement = await profiler.measure_execution_time(
            access_control.generate_permission_matrix
        )
        
        assert measurement['execution_time'] < 3.0, \
            f"Permission matrix generation took {measurement['execution_time']:.2f}s"
    
    @pytest.mark.asyncio
    async def test_bulk_permission_updates_performance(self, large_access_control):
        """Test bulk permission updates performance."""
        access_control, data = large_access_control
        await access_control.initialize()
        
        profiler = PerformanceProfiler()
        
        # Update permissions for 100 users
        user_ids = [user['id'] for user in data['users'][:100]]
        new_permissions = ['new_perm_1', 'new_perm_2', 'new_perm_3']
        
        measurement = await profiler.measure_execution_time(
            access_control.bulk_update_user_permissions, user_ids, new_permissions
        )
        
        assert measurement['execution_time'] < 5.0, \
            f"Bulk permission update took {measurement['execution_time']:.2f}s"


class TestSystemWidePerformance:
    """Test system-wide performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_integrated_system_performance_under_load(self):
        """Test integrated system performance under heavy load."""
        profiler = PerformanceProfiler()
        
        # Create large datasets
        large_hierarchy = profiler.create_large_settings_hierarchy(
            global_settings_count=500,
            users_count=200,
            agents_count=1000
        )
        
        concurrent_sessions = profiler.simulate_concurrent_users(50)
        
        # Mock services
        mock_services = {
            'event_system': MagicMock(),
            'settings_service': MagicMock(),
            'agent_service': MagicMock(),
            'auth_service': MagicMock(),
            'validation_service': MagicMock(),
            'ai_service': MagicMock(),
            'user_service': MagicMock()
        }
        
        # Configure mock returns
        mock_services['settings_service'].get_settings = AsyncMock(return_value=large_hierarchy)
        mock_services['agent_service'].list_agents = AsyncMock(return_value=[])
        mock_services['auth_service'].list_users = AsyncMock(return_value=[])
        
        # Initialize all components
        components = {}
        
        start_time = time.perf_counter()
        
        components['dashboard'] = SettingsDashboard(
            event_system=mock_services['event_system'],
            settings_service=mock_services['settings_service'],
            auth_service=mock_services['auth_service']
        )
        await components['dashboard'].initialize()
        
        components['agent_manager'] = AgentManager(
            event_system=mock_services['event_system'],
            agent_service=mock_services['agent_service'],
            auth_service=mock_services['auth_service']
        )
        await components['agent_manager'].initialize()
        
        components['settings_form'] = SettingsForm(
            event_system=mock_services['event_system'],
            settings_service=mock_services['settings_service'],
            validation_service=mock_services['validation_service']
        )
        await components['settings_form'].initialize()
        
        components['config_assistant'] = ConfigurationAssistant(
            event_system=mock_services['event_system'],
            ai_service=mock_services['ai_service'],
            settings_service=mock_services['settings_service']
        )
        await components['config_assistant'].initialize()
        
        components['access_control'] = AccessControlInterface(
            event_system=mock_services['event_system'],
            auth_service=mock_services['auth_service'],
            user_service=mock_services['user_service']
        )
        await components['access_control'].initialize()
        
        initialization_time = time.perf_counter() - start_time
        
        # Test concurrent operations across all components
        operations = []
        
        # Dashboard operations
        operations.extend([
            components['dashboard'].navigate_to_section('agents'),
            components['dashboard'].search_settings('test_query'),
            components['dashboard'].refresh_health_status()
        ])
        
        # Agent manager operations
        operations.extend([
            components['agent_manager'].get_agent_status('test_agent'),
            components['agent_manager'].render_agent_list()
        ])
        
        # Settings form operations
        operations.extend([
            components['settings_form'].create_form('test_form', {'fields': []}),
            components['settings_form'].validate_form()
        ])
        
        # Configuration assistant operations
        operations.extend([
            components['config_assistant'].get_smart_recommendation('test_context', {}),
            components['config_assistant'].start_wizard('test_wizard')
        ])
        
        # Access control operations
        operations.extend([
            components['access_control'].render_user_list(),
            components['access_control'].check_user_permissions('test_user', ['read'])
        ])
        
        # Execute all operations concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*operations, return_exceptions=True)
        operations_time = time.perf_counter() - start_time
        
        # Performance assertions
        assert initialization_time < 10.0, f"System initialization took {initialization_time:.2f}s, should be < 10.0s"
        assert operations_time < 5.0, f"Concurrent operations took {operations_time:.2f}s, should be < 5.0s"
        
        # Verify no exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Operations failed: {exceptions}"
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test for memory leaks over extended usage."""
        profiler = PerformanceProfiler()
        
        # Initial memory snapshot
        initial_memory = memory_profiler.memory_usage()[0]
        
        # Simulate extended usage
        for iteration in range(100):
            # Create and destroy components repeatedly
            dashboard = SettingsDashboard(
                event_system=MagicMock(),
                settings_service=MagicMock(),
                auth_service=MagicMock()
            )
            
            await dashboard.initialize()
            await dashboard.navigate_to_section('test')
            
            # Explicit cleanup
            del dashboard
            
            # Force garbage collection every 10 iterations
            if iteration % 10 == 0:
                gc.collect()
                current_memory = memory_profiler.memory_usage()[0]
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be bounded
                assert memory_growth < 100, f"Memory leak detected: {memory_growth:.2f}MB growth at iteration {iteration}"
        
        # Final memory check
        final_memory = memory_profiler.memory_usage()[0]
        total_growth = final_memory - initial_memory
        
        assert total_growth < 50, f"Total memory growth {total_growth:.2f}MB should be < 50MB"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])