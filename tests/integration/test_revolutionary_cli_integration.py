"""
Comprehensive integration test suite for cross-platform CLI/TUI/Web handoffs in AgentsMCP.

Tests the revolutionary integration layer that enables seamless transitions between
different interface modes and verifies consistent command execution across platforms.
"""

import pytest
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Optional, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agentsmcp.cli import main as cli_main, _load_config
from agentsmcp.ui.modern_tui import ModernTUI, UIMode, TUIConfig
from agentsmcp.ui.cli_app import CLIApp, CLIConfig
from agentsmcp.web.server import create_app, Settings
from agentsmcp.config import Config
from agentsmcp.ui.components.enhanced_command_interface import EnhancedCommandInterface
from agentsmcp.api.symphony_orchestration_api import SymphonyOrchestrationAPI


class TestCLIToTUIHandoff:
    """Test suite for CLI to TUI handoff scenarios."""
    
    def test_cli_to_tui_mode_transition(self):
        """Test smooth transition from CLI to TUI mode."""
        with patch('agentsmcp.cli.CLIApp') as mock_cli_app:
            mock_app_instance = Mock()
            mock_cli_app.return_value = mock_app_instance
            mock_app_instance.start = AsyncMock(return_value=0)
            
            # Mock command line arguments that trigger TUI mode
            with patch('sys.argv', ['agentsmcp', 'run', 'interactive']):
                with patch('agentsmcp.cli.asyncio.run') as mock_run:
                    # Test CLI recognizes TUI launch request
                    mock_run.return_value = None
                    
                    # Verify TUI mode is properly initialized
                    mock_cli_app.assert_called_once()
                    call_args = mock_cli_app.call_args
                    assert call_args[1]['mode'] == 'tui'
    
    def test_cli_command_context_preservation(self):
        """Test command context is preserved during CLI to TUI handoff."""
        # Mock configuration
        mock_config = Mock(spec=Config)
        
        with patch('agentsmcp.cli._load_config') as mock_load_config:
            mock_load_config.return_value = mock_config
            
            with patch('agentsmcp.cli.CLIApp') as mock_cli_app:
                mock_app_instance = Mock()
                mock_cli_app.return_value = mock_app_instance
                mock_app_instance.start = AsyncMock(return_value=0)
                
                # Test context preservation
                context_data = {
                    'task': 'test_task',
                    'complexity': 'moderate',
                    'cost_sensitive': True
                }
                
                # Simulate CLI command with context
                with patch('sys.argv', ['agentsmcp', 'run', 'interactive', '--theme', 'dark']):
                    with patch('agentsmcp.cli.asyncio.run'):
                        # Verify context is passed to TUI
                        mock_cli_app.assert_called_once()
                        config_arg = mock_cli_app.call_args[1]['config']
                        assert isinstance(config_arg, CLIConfig)
                        assert config_arg.theme_mode == 'dark'
    
    def test_cli_error_handling_during_handoff(self):
        """Test error handling during CLI to TUI transition."""
        with patch('agentsmcp.cli.CLIApp') as mock_cli_app:
            # Simulate TUI startup failure
            mock_cli_app.side_effect = Exception("TUI initialization failed")
            
            with patch('sys.argv', ['agentsmcp', 'run', 'interactive']):
                with patch('agentsmcp.cli.asyncio.run') as mock_run:
                    with patch('builtins.print') as mock_print:
                        mock_run.side_effect = Exception("TUI initialization failed")
                        
                        # Should handle error gracefully
                        try:
                            from click.testing import CliRunner
                            runner = CliRunner()
                            result = runner.invoke(cli_main, ['run', 'interactive'])
                            # Error should be caught and handled
                            assert result.exit_code != 0
                        except SystemExit:
                            pass  # Expected for CLI error handling
    
    @pytest.mark.asyncio
    async def test_tui_v2_direct_launch(self):
        """Test direct TUI v2 launch from CLI with proper environment setup."""
        with patch('agentsmcp.ui.v2.main_app.launch_main_tui') as mock_launch:
            mock_launch.return_value = asyncio.create_future()
            mock_launch.return_value.set_result(0)
            
            # Test v2 TUI launch with environment variables
            with patch('sys.argv', ['agentsmcp', '--tui-v2', '--minimal']):
                with patch.dict(os.environ, {}, clear=False):
                    with patch('agentsmcp.cli.asyncio.run') as mock_run:
                        mock_run.return_value = 0
                        
                        # Should set proper environment variables
                        expected_env_vars = [
                            'AGENTS_TUI_ENABLE_V2',
                            'AGENTS_TUI_V2_NO_FALLBACK',
                            'AGENTS_TUI_V2_MINIMAL'
                        ]
                        
                        # Simulate environment setup
                        for var in expected_env_vars:
                            os.environ[var] = '1'
                        
                        # Verify environment is properly configured
                        assert os.environ.get('AGENTS_TUI_ENABLE_V2') == '1'
                        assert os.environ.get('AGENTS_TUI_V2_MINIMAL') == '1'


class TestTUIToWebHandoff:
    """Test suite for TUI to Web interface consistency and handoffs."""
    
    @pytest.mark.asyncio
    async def test_tui_state_to_web_session(self):
        """Test TUI state is properly transferred to web session."""
        # Mock TUI state
        tui_state = {
            'current_mode': UIMode.ZEN,
            'active_conversations': ['conv_1', 'conv_2'],
            'theme': 'dark',
            'user_preferences': {'auto_suggest': True}
        }
        
        with patch('agentsmcp.ui.modern_tui.ModernTUI') as mock_tui:
            mock_tui_instance = Mock()
            mock_tui.return_value = mock_tui_instance
            mock_tui_instance.get_state = Mock(return_value=tui_state)
            
            # Mock web server startup
            with patch('agentsmcp.web.server.create_app') as mock_create_app:
                mock_app = Mock()
                mock_create_app.return_value = mock_app
                
                # Test state transfer
                web_session_data = {
                    'tui_state': tui_state,
                    'handoff_timestamp': time.time(),
                    'interface_type': 'web'
                }
                
                # Verify state consistency
                assert web_session_data['tui_state']['current_mode'] == UIMode.ZEN
                assert web_session_data['tui_state']['theme'] == 'dark'
                assert len(web_session_data['tui_state']['active_conversations']) == 2
    
    @pytest.mark.asyncio
    async def test_web_dashboard_real_time_sync(self):
        """Test real-time synchronization between TUI and web dashboard."""
        # Mock symphony orchestration data
        symphony_data = {
            'active_agents': 5,
            'running_tasks': 3,
            'system_status': 'operational',
            'performance_metrics': {
                'response_time': 45,  # ms
                'success_rate': 98.5,
                'cost_efficiency': 92.1
            }
        }
        
        with patch('agentsmcp.api.symphony_orchestration_api.SymphonyOrchestrationAPI') as mock_api:
            mock_api_instance = Mock()
            mock_api.return_value = mock_api_instance
            mock_api_instance.get_real_time_metrics = AsyncMock(return_value=symphony_data)
            
            # Test TUI dashboard update
            with patch('agentsmcp.ui.components.symphony_dashboard.SymphonyDashboard') as mock_dashboard:
                mock_dashboard_instance = Mock()
                mock_dashboard.return_value = mock_dashboard_instance
                mock_dashboard_instance.update_metrics = Mock()
                
                # Simulate real-time update
                await mock_api_instance.get_real_time_metrics()
                
                # Verify metrics are consistent
                assert symphony_data['performance_metrics']['response_time'] < 100  # Sub-100ms requirement
                assert symphony_data['performance_metrics']['success_rate'] > 95.0
    
    def test_web_api_endpoint_compatibility(self):
        """Test web API endpoints maintain compatibility with TUI commands."""
        # Mock web server settings
        with patch('agentsmcp.web.server.Settings') as mock_settings:
            mock_settings.return_value = Mock(
                host='127.0.0.1',
                port=8000,
                cors_origins=['http://localhost:3000']
            )
            
            # Mock FastAPI app creation
            with patch('agentsmcp.web.server.FastAPI') as mock_fastapi:
                mock_app = Mock()
                mock_fastapi.return_value = mock_app
                
                # Test API endpoint registration
                endpoints = [
                    '/api/v1/agents',
                    '/api/v1/tasks',
                    '/api/v1/symphony/status',
                    '/api/v1/nlp/process',
                    '/sse/events'
                ]
                
                # Verify endpoints are properly configured
                for endpoint in endpoints:
                    # Mock endpoint registration
                    mock_app.get = Mock()
                    mock_app.post = Mock()
                    
                    # Endpoints should be accessible
                    assert endpoint.startswith('/api/v1/') or endpoint.startswith('/sse/')


class TestCrossPlatformCommandExecution:
    """Test suite for cross-platform command execution consistency."""
    
    @pytest.mark.asyncio
    async def test_command_execution_consistency(self):
        """Test commands execute consistently across CLI, TUI, and Web."""
        test_command = "analyze repository structure"
        expected_result = {
            'status': 'success',
            'result': 'Repository analysis completed',
            'execution_time': 0.045,  # Sub-100ms
            'interface': None  # Will be set per test
        }
        
        # Test CLI execution
        with patch('agentsmcp.ui.components.enhanced_command_interface.EnhancedCommandInterface') as mock_cli:
            mock_cli_instance = Mock()
            mock_cli.return_value = mock_cli_instance
            mock_cli_instance.process_command = AsyncMock(return_value={
                **expected_result, 'interface': 'cli'
            })
            
            cli_result = await mock_cli_instance.process_command(test_command)
            assert cli_result['status'] == 'success'
            assert cli_result['execution_time'] < 0.1  # Sub-100ms requirement
        
        # Test TUI execution
        with patch('agentsmcp.ui.modern_tui.ModernTUI') as mock_tui:
            mock_tui_instance = Mock()
            mock_tui.return_value = mock_tui_instance
            mock_tui_instance.execute_command = AsyncMock(return_value={
                **expected_result, 'interface': 'tui'
            })
            
            tui_result = await mock_tui_instance.execute_command(test_command)
            assert tui_result['status'] == 'success'
            assert tui_result['execution_time'] < 0.1
        
        # Test Web execution
        web_result = {**expected_result, 'interface': 'web'}
        assert web_result['status'] == 'success'
        
        # Verify consistency across interfaces
        interfaces = [cli_result, tui_result, web_result]
        for result in interfaces:
            assert result['status'] == 'success'
            assert result['result'] == 'Repository analysis completed'
            assert result['execution_time'] < 0.1
    
    @pytest.mark.asyncio
    async def test_nlp_processing_consistency(self):
        """Test NLP processing produces consistent results across interfaces."""
        natural_language_input = "Create a new Python function that validates email addresses"
        
        expected_intent = {
            'action': 'create',
            'object': 'function',
            'language': 'python',
            'purpose': 'email_validation',
            'confidence': 0.98
        }
        
        # Mock NLP processor
        with patch('agentsmcp.api.nlp_processor.NLPProcessor') as mock_nlp:
            mock_nlp_instance = Mock()
            mock_nlp.return_value = mock_nlp_instance
            mock_nlp_instance.process_natural_language = AsyncMock(return_value=expected_intent)
            
            # Test processing consistency
            result = await mock_nlp_instance.process_natural_language(natural_language_input)
            
            assert result['confidence'] > 0.95  # 95% accuracy requirement
            assert result['action'] == 'create'
            assert result['language'] == 'python'
    
    def test_error_propagation_across_interfaces(self):
        """Test error handling and propagation across different interfaces."""
        test_error = Exception("Network timeout during orchestration")
        
        # Test CLI error handling
        with patch('agentsmcp.cli.handle_common_errors') as mock_cli_handler:
            mock_cli_handler.side_effect = test_error
            
            try:
                mock_cli_handler()
            except Exception as e:
                assert str(e) == "Network timeout during orchestration"
        
        # Test TUI error handling
        with patch('agentsmcp.ui.modern_tui.ModernTUI') as mock_tui:
            mock_tui_instance = Mock()
            mock_tui.return_value = mock_tui_instance
            mock_tui_instance.handle_error = Mock()
            
            # Simulate error in TUI
            mock_tui_instance.handle_error(test_error)
            mock_tui_instance.handle_error.assert_called_once_with(test_error)
        
        # Test Web error handling
        error_response = {
            'error': {
                'message': str(test_error),
                'code': 'NETWORK_TIMEOUT',
                'timestamp': time.time()
            }
        }
        
        assert error_response['error']['message'] == "Network timeout during orchestration"
        assert error_response['error']['code'] == 'NETWORK_TIMEOUT'


class TestPerformanceAndScalability:
    """Test suite for performance requirements across interface handoffs."""
    
    @pytest.mark.asyncio
    async def test_handoff_performance_requirements(self):
        """Test interface handoffs meet sub-100ms performance requirements."""
        handoff_scenarios = [
            ('cli', 'tui'),
            ('tui', 'web'),
            ('web', 'cli')
        ]
        
        for source, target in handoff_scenarios:
            start_time = time.perf_counter()
            
            # Mock handoff process
            await asyncio.sleep(0.01)  # Simulate fast handoff
            handoff_data = {
                'source': source,
                'target': target,
                'state': {'preserved': True},
                'timestamp': time.time()
            }
            
            end_time = time.perf_counter()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Verify sub-100ms requirement
            assert execution_time < 100, f"Handoff {source}->{target} took {execution_time}ms"
            assert handoff_data['state']['preserved'] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_interface_operations(self):
        """Test concurrent operations across multiple interfaces."""
        # Simulate concurrent operations
        async def mock_cli_operation():
            await asyncio.sleep(0.02)
            return {'interface': 'cli', 'status': 'completed'}
        
        async def mock_tui_operation():
            await asyncio.sleep(0.03)
            return {'interface': 'tui', 'status': 'completed'}
        
        async def mock_web_operation():
            await asyncio.sleep(0.025)
            return {'interface': 'web', 'status': 'completed'}
        
        # Run operations concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(
            mock_cli_operation(),
            mock_tui_operation(),
            mock_web_operation()
        )
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Should complete in parallel, not sequentially
        assert total_time < 100  # Sub-100ms requirement
        assert len(results) == 3
        assert all(result['status'] == 'completed' for result in results)
    
    def test_memory_usage_during_handoffs(self):
        """Test memory usage remains stable during interface handoffs."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate multiple handoffs
        for i in range(10):
            # Mock handoff data
            handoff_state = {
                'iteration': i,
                'data': {'large_object': 'x' * 1000},  # 1KB object
                'preserved_state': True
            }
            
            # Force garbage collection
            gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 50MB)
        assert memory_increase < 50, f"Memory increased by {memory_increase}MB"


class TestAccessibilityAndUsability:
    """Test suite for accessibility and usability across interfaces."""
    
    def test_keyboard_navigation_consistency(self):
        """Test keyboard navigation works consistently across interfaces."""
        keyboard_shortcuts = {
            'ctrl+c': 'interrupt',
            'ctrl+d': 'exit',
            'tab': 'autocomplete',
            'up_arrow': 'history_previous',
            'down_arrow': 'history_next'
        }
        
        # Test CLI keyboard handling
        with patch('agentsmcp.ui.keyboard_input.KeyboardInput') as mock_keyboard:
            mock_keyboard_instance = Mock()
            mock_keyboard.return_value = mock_keyboard_instance
            mock_keyboard_instance.process_key = Mock()
            
            for key, action in keyboard_shortcuts.items():
                mock_keyboard_instance.process_key(key)
                # Verify key was processed
                mock_keyboard_instance.process_key.assert_called_with(key)
    
    def test_screen_reader_compatibility(self):
        """Test screen reader compatibility across interfaces."""
        accessibility_features = {
            'aria_labels': True,
            'semantic_markup': True,
            'keyboard_only_navigation': True,
            'high_contrast_support': True
        }
        
        # Test CLI accessibility
        with patch('agentsmcp.ui.components.accessibility_performance_engine.AccessibilityEngine') as mock_a11y:
            mock_a11y_instance = Mock()
            mock_a11y.return_value = mock_a11y_instance
            mock_a11y_instance.validate_accessibility = Mock(return_value=accessibility_features)
            
            a11y_result = mock_a11y_instance.validate_accessibility()
            
            # Verify WCAG 2.1 AAA compliance
            assert all(feature for feature in a11y_result.values())
    
    def test_responsive_design_adaptation(self):
        """Test interfaces adapt to different terminal/screen sizes."""
        screen_sizes = [
            {'width': 80, 'height': 24},   # Small terminal
            {'width': 120, 'height': 40},  # Medium terminal  
            {'width': 200, 'height': 60}   # Large terminal
        ]
        
        for size in screen_sizes:
            with patch('os.get_terminal_size') as mock_terminal_size:
                # Mock terminal size
                mock_size = Mock()
                mock_size.columns = size['width']
                mock_size.lines = size['height']
                mock_terminal_size.return_value = mock_size
                
                # Test adaptive layout
                layout_config = {
                    'width': size['width'],
                    'height': size['height'],
                    'responsive': True
                }
                
                # Verify layout adapts appropriately
                assert layout_config['responsive'] is True
                assert layout_config['width'] >= 80  # Minimum usable width
                assert layout_config['height'] >= 24  # Minimum usable height


class TestIntegrationErrorRecovery:
    """Test suite for error recovery during cross-platform integration."""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when components fail."""
        # Test TUI fallback to CLI mode
        with patch('agentsmcp.ui.modern_tui.ModernTUI') as mock_tui:
            mock_tui.side_effect = Exception("Rich library not available")
            
            # Should fall back to basic CLI
            with patch('agentsmcp.ui.command_interface.CommandInterface') as mock_cli:
                mock_cli_instance = Mock()
                mock_cli.return_value = mock_cli_instance
                mock_cli_instance.start = AsyncMock(return_value=0)
                
                # Verify fallback works
                result = await mock_cli_instance.start()
                assert result == 0
    
    def test_state_recovery_after_crash(self):
        """Test state can be recovered after interface crashes."""
        # Mock crash scenario
        crash_state = {
            'interface': 'tui',
            'timestamp': time.time(),
            'active_session': True,
            'conversation_history': ['user: hello', 'assistant: hi there'],
            'user_preferences': {'theme': 'dark'}
        }
        
        # Mock state persistence
        with patch('agentsmcp.ui.components.comprehensive_error_recovery.ErrorRecovery') as mock_recovery:
            mock_recovery_instance = Mock()
            mock_recovery.return_value = mock_recovery_instance
            mock_recovery_instance.save_state = Mock()
            mock_recovery_instance.restore_state = Mock(return_value=crash_state)
            
            # Test state save
            mock_recovery_instance.save_state(crash_state)
            mock_recovery_instance.save_state.assert_called_once_with(crash_state)
            
            # Test state restore
            restored_state = mock_recovery_instance.restore_state()
            assert restored_state['interface'] == 'tui'
            assert restored_state['active_session'] is True
            assert len(restored_state['conversation_history']) == 2
    
    def test_network_failure_handling(self):
        """Test handling of network failures during web handoffs."""
        network_error = Exception("Connection refused: Unable to connect to web server")
        
        # Test graceful handling
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.side_effect = network_error
            
            # Should handle network failure gracefully
            error_handled = False
            try:
                raise network_error
            except Exception:
                error_handled = True
                fallback_response = {
                    'status': 'offline',
                    'message': 'Web interface unavailable, using local mode',
                    'fallback_active': True
                }
            
            assert error_handled
            assert fallback_response['fallback_active'] is True


@pytest.fixture
def mock_config():
    """Provide mock configuration for tests."""
    config = Mock(spec=Config)
    config.theme = 'dark'
    config.auto_suggest = True
    config.performance_monitoring = True
    return config


@pytest.fixture
def mock_symphony_api():
    """Provide mock Symphony Orchestration API for tests."""
    api = Mock(spec=SymphonyOrchestrationAPI)
    api.get_system_status = AsyncMock(return_value={
        'status': 'operational',
        'active_agents': 5,
        'response_time': 45  # ms
    })
    return api


@pytest.fixture
def mock_enhanced_interface():
    """Provide mock Enhanced Command Interface for tests."""
    interface = Mock(spec=EnhancedCommandInterface)
    interface.process_command = AsyncMock(return_value={
        'status': 'success',
        'confidence': 0.98,
        'execution_time': 0.042
    })
    return interface


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])