"""Integration tests for CLI v3 main application.

Tests the complete integration of all CLI v3 components including:
- Natural language processing
- Command execution pipeline
- Progressive disclosure
- Cross-modal coordination
- Legacy compatibility
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock

from agentsmcp.cli.v3.main import CliV3Main, create_cli_v3, display_result
from agentsmcp.cli.v3.models.command_models import ExecutionMode, SkillLevel, UserProfile
from agentsmcp.cli.v3.models.nlp_models import ConversationContext, LLMConfig


class TestCliV3MainIntegration:
    """Integration tests for the main CLI v3 application."""
    
    @pytest.fixture
    async def cli_app(self):
        """Create CLI v3 application for testing."""
        app = await create_cli_v3()
        yield app
        await app.shutdown()
    
    @pytest.fixture
    def user_profile(self):
        """Create test user profile."""
        return UserProfile(
            user_id="test_user",
            skill_level=SkillLevel.INTERMEDIATE,
            command_history=[],
            preferences=UserProfile.UserPreferences()
        )
    
    async def test_initialization_performance(self):
        """Test that CLI v3 initializes within performance targets (<200ms)."""
        start_time = time.time()
        
        app = await create_cli_v3()
        
        initialization_time = time.time() - start_time
        performance_metrics = app.get_startup_performance()
        
        await app.shutdown()
        
        # Verify performance targets
        assert initialization_time < 0.2, f"Initialization took {initialization_time*1000:.1f}ms, target <200ms"
        assert performance_metrics["startup_time_ms"] < 200
        assert performance_metrics["ready"] is True
        assert performance_metrics["components_loaded"] == 6
    
    async def test_natural_language_command_execution(self, cli_app, user_profile):
        """Test natural language command processing end-to-end."""
        # Test natural language help request
        success, result, message = await cli_app.execute_command(
            "help me get started",
            ExecutionMode.CLI,
            user_profile
        )
        
        assert success is True
        assert "help" in message.lower() or "completed" in message.lower()
        assert isinstance(result, dict)
        
        # Test natural language status request
        success, result, message = await cli_app.execute_command(
            "show me the system status", 
            ExecutionMode.CLI,
            user_profile
        )
        
        assert success is True
        assert result is not None
    
    async def test_structured_command_execution(self, cli_app, user_profile):
        """Test structured command processing."""
        # Test basic help command
        success, result, message = await cli_app.execute_command(
            "help",
            ExecutionMode.CLI, 
            user_profile
        )
        
        assert success is True
        assert isinstance(result, dict)
        assert "title" in result
        assert "commands" in result
        
        # Test status command with options
        success, result, message = await cli_app.execute_command(
            "status --detailed",
            ExecutionMode.CLI,
            user_profile
        )
        
        assert success is True
        assert isinstance(result, dict)
        assert "system" in result
    
    async def test_progressive_disclosure(self, cli_app):
        """Test progressive disclosure based on skill level."""
        # Test beginner user
        beginner_profile = UserProfile(
            user_id="beginner",
            skill_level=SkillLevel.BEGINNER,
            command_history=[],
            preferences=UserProfile.UserPreferences()
        )
        
        success, result, message = await cli_app.execute_command(
            "help",
            ExecutionMode.CLI,
            beginner_profile
        )
        
        assert success is True
        beginner_commands = result["commands"]
        
        # Test expert user
        expert_profile = UserProfile(
            user_id="expert",
            skill_level=SkillLevel.EXPERT,
            command_history=[],
            preferences=UserProfile.UserPreferences()
        )
        
        success, result, message = await cli_app.execute_command(
            "help",
            ExecutionMode.CLI,
            expert_profile
        )
        
        assert success is True
        expert_commands = result["commands"]
        
        # Expert should have more commands than beginner
        assert len(expert_commands) > len(beginner_commands)
        
        # Verify skill level adaptation
        assert result["skill_level"] == "EXPERT"
    
    async def test_command_history_tracking(self, cli_app, user_profile):
        """Test that command history is properly tracked."""
        initial_history_length = len(user_profile.command_history)
        
        # Execute several commands
        commands = ["help", "status", "help natural"]
        
        for cmd in commands:
            await cli_app.execute_command(cmd, ExecutionMode.CLI, user_profile)
        
        # Verify history was updated
        assert len(user_profile.command_history) == initial_history_length + len(commands)
        assert user_profile.command_history[-len(commands):] == commands
    
    async def test_natural_language_detection(self, cli_app):
        """Test natural language vs structured command detection."""
        test_cases = [
            # Natural language examples
            ("help me analyze my code", True),
            ("show me the system status", True),
            ("I want to check my costs", True), 
            ("can you help with setup", True),
            ("what is the current status", True),
            ('"analyze my Python files"', True),
            
            # Structured command examples
            ("help", False),
            ("status --detailed", False),
            ("run task", False),
            ("config edit", False),
            ("-h", False),
            ("--version", False)
        ]
        
        for command, expected_is_natural in test_cases:
            is_natural = cli_app._is_natural_language_input(command)
            assert is_natural == expected_is_natural, f"Command '{command}' detection failed"
    
    async def test_error_handling_and_recovery(self, cli_app, user_profile):
        """Test error handling and recovery suggestions."""
        # Test invalid command
        success, result, message = await cli_app.execute_command(
            "nonexistent_command",
            ExecutionMode.CLI,
            user_profile
        )
        
        assert success is False
        assert "not found" in message.lower() or "error" in message.lower()
        
        # Test malformed natural language
        success, result, message = await cli_app.execute_command(
            "asdf qwerty nonsense input",
            ExecutionMode.CLI,
            user_profile
        )
        
        # Should either succeed with fallback or fail gracefully
        assert message is not None
        assert len(message) > 0
    
    async def test_cross_modal_coordination(self, cli_app, user_profile):
        """Test cross-modal coordination features."""
        # Test mode switching notification
        await cli_app.cross_modal_coordinator.notify_mode_switch(
            ExecutionMode.CLI, 
            ExecutionMode.TUI,
            Mock(user_profile=user_profile)
        )
        
        # Verify mode was tracked
        assert ExecutionMode.TUI in cli_app.cross_modal_coordinator.active_modes
        assert ExecutionMode.CLI not in cli_app.cross_modal_coordinator.active_modes
    
    async def test_telemetry_collection(self, cli_app, user_profile):
        """Test telemetry and metrics collection."""
        initial_metrics = cli_app.telemetry_collector.metrics.copy()
        
        # Execute some commands
        await cli_app.execute_command("help", ExecutionMode.CLI, user_profile)
        await cli_app.execute_command("analyze my code", ExecutionMode.CLI, user_profile)
        
        # Verify metrics were updated
        final_metrics = cli_app.telemetry_collector.metrics
        assert final_metrics["total_commands"] > initial_metrics["total_commands"]
        assert final_metrics["natural_language_commands"] >= initial_metrics["natural_language_commands"]
        assert final_metrics["direct_commands"] >= initial_metrics["direct_commands"]
    
    @pytest.mark.asyncio
    async def test_concurrent_command_execution(self, cli_app):
        """Test handling of concurrent command executions."""
        # Create multiple user profiles
        users = [
            UserProfile(user_id=f"user_{i}", skill_level=SkillLevel.INTERMEDIATE, command_history=[], preferences=UserProfile.UserPreferences())
            for i in range(3)
        ]
        
        # Execute commands concurrently
        tasks = [
            cli_app.execute_command(f"help topic_{i}", ExecutionMode.CLI, users[i])
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All commands should complete successfully
        for success, result, message in results:
            assert success is True
            assert result is not None
    
    async def test_performance_under_load(self, cli_app, user_profile):
        """Test performance with multiple rapid commands."""
        start_time = time.time()
        commands_executed = 0
        
        # Execute commands rapidly for 1 second
        while time.time() - start_time < 1.0:
            success, result, message = await cli_app.execute_command(
                "help",
                ExecutionMode.CLI,
                user_profile
            )
            commands_executed += 1
            
            # All commands should complete within reasonable time
            assert success is True
        
        total_time = time.time() - start_time
        avg_command_time = total_time / commands_executed
        
        # Each command should average <500ms as per requirements
        assert avg_command_time < 0.5, f"Average command time {avg_command_time*1000:.0f}ms exceeds 500ms target"
        assert commands_executed >= 2, "Should execute at least 2 commands per second"


class TestLegacyCompatibility:
    """Test legacy CLI compatibility through integration."""
    
    @pytest.fixture
    async def cli_app(self):
        """Create CLI v3 application for testing."""
        app = await create_cli_v3()
        yield app
        await app.shutdown()
    
    async def test_legacy_command_detection(self, cli_app):
        """Test detection of legacy commands."""
        legacy_commands = [
            "simple execute task",
            "costs --detailed",
            "budget --check",
            "models --recommend analysis",
            "setup --mode interactive"
        ]
        
        for cmd in legacy_commands:
            is_legacy = cli_app.legacy_adapter.is_legacy_command(cmd)
            assert is_legacy is True, f"Failed to detect legacy command: {cmd}"
    
    async def test_legacy_command_conversion(self, cli_app):
        """Test conversion of legacy commands to v3 format."""
        test_cases = [
            ("simple execute task", "run"),
            ("costs --detailed", "status"),
            ("budget --check", "status"),
            ("setup --mode interactive", "init"),
        ]
        
        for legacy_cmd, expected_v3_type in test_cases:
            is_legacy, request = cli_app.legacy_adapter.convert_legacy_command(legacy_cmd)
            
            assert is_legacy is True
            assert request is not None
            assert expected_v3_type in request.command_type
            assert request.metadata["legacy_input"] == legacy_cmd
    
    async def test_migration_suggestions(self, cli_app):
        """Test that migration suggestions are provided appropriately."""
        # This would test the warning/suggestion system
        # For now, verify the mapping data is available
        suggestions = cli_app.legacy_adapter.get_legacy_commands_list()
        
        assert len(suggestions) > 0
        assert all("legacy_command" in cmd for cmd in suggestions)
        assert all("v3_equivalent" in cmd for cmd in suggestions)
        assert all("status" in cmd for cmd in suggestions)


class TestDisplaySystem:
    """Test the display and help system."""
    
    def test_display_result_beginner_level(self, capsys):
        """Test result display for beginner skill level."""
        result_data = {
            "title": "Test Command",
            "commands": ["help - Show help", "status - Show status"],
            "natural_language": True
        }
        
        display_result(True, result_data, "Command completed", SkillLevel.BEGINNER)
        captured = capsys.readouterr()
        
        assert "✅" in captured.out
        assert "Test Command" in captured.out
        assert "help - Show help" in captured.out
    
    def test_display_result_expert_level(self, capsys):
        """Test result display for expert skill level."""
        result_data = {
            "title": "Advanced Status",
            "description": "System status information",
            "commands": ["status --detailed", "status --json"]
        }
        
        display_result(True, result_data, "Status retrieved", SkillLevel.EXPERT)
        captured = capsys.readouterr()
        
        assert "Success" in captured.out
        assert "Advanced Status" in captured.out
        assert "natural language" in captured.out.lower()
    
    def test_display_error_with_suggestions(self, capsys):
        """Test error display with helpful suggestions."""
        display_result(False, None, "Command not found", SkillLevel.BEGINNER)
        captured = capsys.readouterr()
        
        assert "❌" in captured.out
        assert "Error" in captured.out
        assert "help" in captured.out.lower()
    
    def test_beautiful_help_display(self, capsys):
        """Test the beautiful help system display."""
        from agentsmcp.cli.v3.main import display_beautiful_help
        
        display_beautiful_help()
        captured = capsys.readouterr()
        
        # Should contain key elements
        assert "AgentsMCP CLI v3" in captured.out
        assert "Natural Language Commands" in captured.out
        assert "Structured Commands" in captured.out
        assert "Progressive Disclosure" in captured.out


class TestEndToEndWorkflows:
    """Test complete end-to-end user workflows."""
    
    @pytest.fixture
    async def cli_app(self):
        """Create CLI v3 application for testing."""
        app = await create_cli_v3()
        yield app
        await app.shutdown()
    
    async def test_new_user_onboarding_workflow(self, cli_app):
        """Test complete new user onboarding workflow."""
        # New user profile
        new_user = UserProfile(
            user_id="new_user",
            skill_level=SkillLevel.BEGINNER,
            command_history=[],
            preferences=UserProfile.UserPreferences()
        )
        
        # Step 1: User asks for help
        success, result, message = await cli_app.execute_command(
            "help me get started",
            ExecutionMode.CLI,
            new_user
        )
        
        assert success is True
        assert "help" in result or "commands" in result
        
        # Step 2: User checks status
        success, result, message = await cli_app.execute_command(
            "what is the current status",
            ExecutionMode.CLI,
            new_user
        )
        
        assert success is True
        
        # Step 3: User launches TUI
        success, result, message = await cli_app.execute_command(
            "start the interactive interface",
            ExecutionMode.CLI,
            new_user
        )
        
        # Should either succeed or fail gracefully
        assert message is not None
        
        # Verify command history was tracked
        assert len(new_user.command_history) == 3
    
    async def test_expert_user_workflow(self, cli_app):
        """Test expert user advanced workflow."""
        expert_user = UserProfile(
            user_id="expert_user", 
            skill_level=SkillLevel.EXPERT,
            command_history=["status", "config", "help advanced"],
            preferences=UserProfile.UserPreferences()
        )
        
        # Expert uses structured commands with options
        success, result, message = await cli_app.execute_command(
            "status --detailed",
            ExecutionMode.CLI,
            expert_user
        )
        
        assert success is True
        assert isinstance(result, dict)
        
        # Expert should get more detailed information
        if "performance" in result:
            assert "avg_lookup_time_ms" in result["performance"]
    
    async def test_mixed_mode_workflow(self, cli_app):
        """Test workflow mixing natural language and structured commands."""
        user = UserProfile(
            user_id="mixed_user",
            skill_level=SkillLevel.INTERMEDIATE, 
            command_history=[],
            preferences=UserProfile.UserPreferences()
        )
        
        commands = [
            ("help me understand the system", True),  # Natural language
            ("status", False),                        # Structured
            ("show me available commands", True),     # Natural language
            ("help natural", False)                   # Structured with args
        ]
        
        for cmd, is_natural in commands:
            success, result, message = await cli_app.execute_command(
                cmd,
                ExecutionMode.CLI,
                user
            )
            
            assert success is True, f"Command '{cmd}' failed: {message}"
            assert result is not None
        
        # Verify mixed command history
        assert len(user.command_history) == len(commands)
    
    async def test_error_recovery_workflow(self, cli_app):
        """Test error recovery and suggestion workflow."""
        user = UserProfile(
            user_id="error_user",
            skill_level=SkillLevel.INTERMEDIATE,
            command_history=[],
            preferences=UserProfile.UserPreferences()
        )
        
        # User makes an error
        success, result, message = await cli_app.execute_command(
            "invalid_command_here",
            ExecutionMode.CLI,
            user
        )
        
        assert success is False
        
        # User follows up with help
        success, result, message = await cli_app.execute_command(
            "help",
            ExecutionMode.CLI,
            user
        )
        
        assert success is True
        assert "commands" in result
        
        # User tries a corrected command
        success, result, message = await cli_app.execute_command(
            "status",
            ExecutionMode.CLI,
            user
        )
        
        assert success is True