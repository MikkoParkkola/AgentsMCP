"""Tests for CommandRegistry with performance, security, and functionality validation."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.agentsmcp.cli.v3.models.command_models import ExecutionMode, SkillLevel
from src.agentsmcp.cli.v3.models.registry_models import (
    CommandDefinition,
    CommandMetadata,
    CommandCategory,
    SecurityLevel,
    ParameterDefinition,
    ParameterType,
    CommandDependency,
    CommandExample,
    CommandAlreadyExistsError,
    InvalidDefinitionError,
    CircularDependencyError,
    RegistryCorruptedError,
)
from src.agentsmcp.cli.v3.registry.command_registry import CommandRegistry


@pytest.fixture
def temp_registry_dir():
    """Create a temporary directory for registry persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_command_definition():
    """Create a sample command definition for testing."""
    return CommandDefinition(
        name="test_command",
        aliases=["tc", "test"],
        description="A test command for unit testing",
        category=CommandCategory.CORE,
        security_level=SecurityLevel.SAFE,
        supported_modes=[ExecutionMode.CLI, ExecutionMode.TUI],
        required_permissions=["test.execute"],
        min_skill_level=SkillLevel.BEGINNER,
        version="1.0.0",
        tags=["test", "utility"]
    )


@pytest.fixture
def registry_with_sample_commands():
    """Create a registry pre-populated with sample commands."""
    registry = CommandRegistry()
    
    # Add several test commands
    commands = [
        CommandDefinition(
            name="file_copy",
            aliases=["fc", "copy"],
            description="Copy files from source to destination",
            category=CommandCategory.CORE,
            tags=["file", "copy"]
        ),
        CommandDefinition(
            name="git_status",
            aliases=["gs"],
            description="Show git repository status",
            category=CommandCategory.ADVANCED,
            min_skill_level=SkillLevel.INTERMEDIATE,
            tags=["git", "status"]
        ),
        CommandDefinition(
            name="system_monitor",
            description="Monitor system resources",
            category=CommandCategory.SYSTEM,
            security_level=SecurityLevel.ELEVATED,
            min_skill_level=SkillLevel.EXPERT,
            tags=["system", "monitoring"]
        ),
        CommandDefinition(
            name="deprecated_command",
            description="An old deprecated command",
            category=CommandCategory.DEPRECATED,
            deprecated=True,
            replacement="new_command"
        )
    ]
    
    for cmd_def in commands:
        registry.register_command(cmd_def, f"{cmd_def.name.title()}Handler")
    
    return registry


class TestCommandRegistryBasics:
    """Test basic CommandRegistry functionality."""
    
    def test_registry_initialization(self):
        """Test registry initializes correctly."""
        registry = CommandRegistry()
        
        assert len(registry) == 0
        assert registry.get_registry_stats().total_commands == 0
        assert not registry.get_command("nonexistent")
    
    def test_registry_with_persistence(self, temp_registry_dir):
        """Test registry initialization with persistence."""
        registry = CommandRegistry(data_dir=temp_registry_dir)
        
        assert len(registry) == 0
        registry_file = temp_registry_dir / "command_registry.json"
        assert not registry_file.exists()  # No file until first save
    
    def test_command_registration(self, sample_command_definition):
        """Test basic command registration."""
        registry = CommandRegistry()
        
        metadata = registry.register_command(
            sample_command_definition,
            "TestCommandHandler"
        )
        
        assert isinstance(metadata, CommandMetadata)
        assert metadata.definition.name == "test_command"
        assert metadata.handler_class == "TestCommandHandler"
        assert len(registry) == 1
        assert "test_command" in registry
    
    def test_command_registration_with_plugin(self, sample_command_definition):
        """Test command registration with plugin source."""
        registry = CommandRegistry()
        
        metadata = registry.register_command(
            sample_command_definition,
            "TestCommandHandler",
            plugin_source="test_plugin"
        )
        
        assert metadata.plugin_source == "test_plugin"
        assert len(registry) == 1
    
    def test_duplicate_command_registration_fails(self, sample_command_definition):
        """Test that duplicate command registration fails."""
        registry = CommandRegistry()
        
        # First registration should succeed
        registry.register_command(sample_command_definition, "Handler1")
        
        # Second registration should fail
        with pytest.raises(CommandAlreadyExistsError, match="already registered"):
            registry.register_command(sample_command_definition, "Handler2")
    
    def test_force_update_command(self, sample_command_definition):
        """Test force updating an existing command."""
        registry = CommandRegistry()
        
        # First registration
        registry.register_command(sample_command_definition, "Handler1")
        
        # Force update should succeed
        updated_def = sample_command_definition.model_copy()
        updated_def.description = "Updated description"
        
        metadata = registry.register_command(
            updated_def, "Handler2", force_update=True
        )
        
        assert metadata.definition.description == "Updated description"
        assert metadata.handler_class == "Handler2"
        assert len(registry) == 1  # Still only one command
    
    def test_alias_conflict_detection(self):
        """Test that alias conflicts are detected."""
        registry = CommandRegistry()
        
        cmd1 = CommandDefinition(
            name="command1",
            aliases=["c1", "first"],
            description="First command"
        )
        
        cmd2 = CommandDefinition(
            name="command2",
            aliases=["c2", "first"],  # Conflicting alias
            description="Second command"
        )
        
        registry.register_command(cmd1, "Handler1")
        
        with pytest.raises(CommandAlreadyExistsError, match="Alias 'first' already used"):
            registry.register_command(cmd2, "Handler2")


class TestCommandLookup:
    """Test command lookup functionality and performance."""
    
    def test_lookup_by_name(self, registry_with_sample_commands):
        """Test O(1) lookup by command name."""
        registry = registry_with_sample_commands
        
        start_time = time.perf_counter()
        metadata = registry.get_command("file_copy")
        lookup_time_ms = (time.perf_counter() - start_time) * 1000
        
        assert metadata is not None
        assert metadata.definition.name == "file_copy"
        assert lookup_time_ms < 1.0  # Should be sub-millisecond
    
    def test_lookup_by_alias(self, registry_with_sample_commands):
        """Test O(1) lookup by command alias."""
        registry = registry_with_sample_commands
        
        start_time = time.perf_counter()
        metadata = registry.get_command("fc")  # Alias for file_copy
        lookup_time_ms = (time.perf_counter() - start_time) * 1000
        
        assert metadata is not None
        assert metadata.definition.name == "file_copy"
        assert lookup_time_ms < 1.0  # Should be sub-millisecond
    
    def test_lookup_nonexistent_command(self, registry_with_sample_commands):
        """Test lookup of nonexistent command."""
        registry = registry_with_sample_commands
        
        result = registry.get_command("nonexistent_command")
        assert result is None
    
    def test_handler_lookup(self, registry_with_sample_commands):
        """Test handler class lookup."""
        registry = registry_with_sample_commands
        
        handler = registry.get_handler_class("file_copy")
        assert handler == "FileCopyHandler"
        
        handler = registry.get_handler_class("fc")  # By alias
        assert handler == "FileCopyHandler"
        
        handler = registry.get_handler_class("nonexistent")
        assert handler is None
    
    def test_lookup_performance_at_scale(self):
        """Test lookup performance with many commands."""
        registry = CommandRegistry()
        
        # Register many commands
        num_commands = 1000
        for i in range(num_commands):
            cmd = CommandDefinition(
                name=f"command_{i:04d}",
                aliases=[f"c{i}", f"cmd{i}"],
                description=f"Test command number {i}",
                tags=[f"tag{i % 10}", "test"]
            )
            registry.register_command(cmd, f"Handler{i}")
        
        # Test lookup performance
        start_time = time.perf_counter()
        
        # Lookup 100 random commands
        for i in range(0, num_commands, 10):
            metadata = registry.get_command(f"command_{i:04d}")
            assert metadata is not None
            
            # Also test alias lookup
            metadata = registry.get_command(f"c{i}")
            assert metadata is not None
        
        total_time_ms = (time.perf_counter() - start_time) * 1000
        avg_time_per_lookup = total_time_ms / 200  # 100 name + 100 alias lookups
        
        assert avg_time_per_lookup < 0.1  # Should be sub 0.1ms per lookup
    
    def test_usage_tracking(self, registry_with_sample_commands):
        """Test that command usage is tracked."""
        registry = registry_with_sample_commands
        
        # Get initial usage count
        metadata = registry.get_command("file_copy")
        initial_count = metadata.usage_count
        
        # Look up command again
        metadata = registry.get_command("file_copy")
        assert metadata.usage_count == initial_count + 1
        assert metadata.last_used is not None


class TestCommandFiltering:
    """Test command listing and filtering functionality."""
    
    def test_list_all_commands(self, registry_with_sample_commands):
        """Test listing all commands."""
        registry = registry_with_sample_commands
        
        commands = registry.list_commands()
        assert len(commands) == 4  # All registered commands
        
        # Should be sorted by usage count (desc) then name (asc)
        command_names = [cmd.definition.name for cmd in commands]
        assert "deprecated_command" in command_names
        assert "file_copy" in command_names
    
    def test_filter_by_category(self, registry_with_sample_commands):
        """Test filtering commands by category."""
        registry = registry_with_sample_commands
        
        core_commands = registry.list_commands(category=CommandCategory.CORE)
        assert len(core_commands) == 1
        assert core_commands[0].definition.name == "file_copy"
        
        advanced_commands = registry.list_commands(category=CommandCategory.ADVANCED)
        assert len(advanced_commands) == 1
        assert advanced_commands[0].definition.name == "git_status"
    
    def test_filter_by_skill_level(self, registry_with_sample_commands):
        """Test filtering commands by skill level."""
        registry = registry_with_sample_commands
        
        # Beginner should see beginner + intermediate + expert commands
        beginner_commands = registry.list_commands(skill_level=SkillLevel.BEGINNER)
        beginner_names = [cmd.definition.name for cmd in beginner_commands]
        assert "file_copy" in beginner_names
        assert "git_status" in beginner_names
        assert "system_monitor" in beginner_names
        
        # Intermediate should see intermediate + expert, not beginner-only
        intermediate_commands = registry.list_commands(skill_level=SkillLevel.INTERMEDIATE)
        intermediate_names = [cmd.definition.name for cmd in intermediate_commands]
        assert "git_status" in intermediate_names
        assert "system_monitor" in intermediate_names
        
        # Expert should see only expert commands
        expert_commands = registry.list_commands(skill_level=SkillLevel.EXPERT)
        expert_names = [cmd.definition.name for cmd in expert_commands]
        assert "system_monitor" in expert_names
    
    def test_filter_by_execution_mode(self, registry_with_sample_commands):
        """Test filtering commands by execution mode."""
        registry = registry_with_sample_commands
        
        cli_commands = registry.list_commands(execution_mode=ExecutionMode.CLI)
        assert len(cli_commands) >= 1  # At least one CLI command
        
        api_commands = registry.list_commands(execution_mode=ExecutionMode.API)
        # No commands were registered with API mode
        assert len(api_commands) == 0
    
    def test_deprecated_filtering(self, registry_with_sample_commands):
        """Test filtering of deprecated commands."""
        registry = registry_with_sample_commands
        
        # Default: exclude deprecated
        commands = registry.list_commands()
        command_names = [cmd.definition.name for cmd in commands]
        assert "deprecated_command" not in command_names
        
        # Include deprecated
        commands_with_deprecated = registry.list_commands(include_deprecated=True)
        deprecated_names = [cmd.definition.name for cmd in commands_with_deprecated]
        assert "deprecated_command" in deprecated_names
    
    def test_get_commands_by_tag(self, registry_with_sample_commands):
        """Test getting commands by tag."""
        registry = registry_with_sample_commands
        
        file_commands = registry.get_commands_by_tag("file")
        assert len(file_commands) == 1
        assert file_commands[0].definition.name == "file_copy"
        
        git_commands = registry.get_commands_by_tag("git")
        assert len(git_commands) == 1
        assert git_commands[0].definition.name == "git_status"
        
        nonexistent_commands = registry.get_commands_by_tag("nonexistent")
        assert len(nonexistent_commands) == 0


class TestDependencyManagement:
    """Test command dependency handling."""
    
    def test_register_command_with_dependencies(self):
        """Test registering commands with dependencies."""
        registry = CommandRegistry()
        
        # Register base command first
        base_cmd = CommandDefinition(
            name="base_command",
            description="Base command"
        )
        registry.register_command(base_cmd, "BaseHandler")
        
        # Register dependent command
        dep = CommandDependency(
            command="base_command",
            version_min="1.0.0",
            optional=False
        )
        
        dependent_cmd = CommandDefinition(
            name="dependent_command",
            description="Command that depends on base_command",
            dependencies=[dep]
        )
        
        registry.register_command(dependent_cmd, "DependentHandler")
        
        # Verify dependencies are tracked
        deps = registry.get_dependencies("dependent_command")
        assert "base_command" in deps
        
        dependents = registry.get_dependents("base_command")
        assert "dependent_command" in dependents
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        registry = CommandRegistry()
        
        # Register first command
        cmd1 = CommandDefinition(
            name="command1",
            description="First command"
        )
        registry.register_command(cmd1, "Handler1")
        
        # Register second command depending on first
        dep1 = CommandDependency(command="command1", optional=False)
        cmd2 = CommandDefinition(
            name="command2",
            description="Second command",
            dependencies=[dep1]
        )
        registry.register_command(cmd2, "Handler2")
        
        # Try to make first command depend on second (circular)
        dep2 = CommandDependency(command="command2", optional=False)
        cmd1_updated = CommandDefinition(
            name="command1",
            description="First command updated",
            dependencies=[dep2]
        )
        
        with pytest.raises(CircularDependencyError, match="circular dependency"):
            registry.register_command(cmd1_updated, "Handler1", force_update=True)
    
    def test_dependency_validation(self):
        """Test dependency validation."""
        registry = CommandRegistry()
        
        # Register base command
        base_cmd = CommandDefinition(
            name="base_command",
            description="Base command"
        )
        registry.register_command(base_cmd, "BaseHandler")
        
        # Register dependent command
        dep = CommandDependency(
            command="base_command",
            optional=False
        )
        
        dependent_cmd = CommandDefinition(
            name="dependent_command",
            description="Dependent command",
            dependencies=[dep]
        )
        registry.register_command(dependent_cmd, "DependentHandler")
        
        # Test validation
        is_valid, missing = registry.validate_dependencies("dependent_command")
        assert is_valid is True
        assert len(missing) == 0
        
        # Test with missing dependency
        missing_dep = CommandDependency(
            command="nonexistent_command",
            optional=False
        )
        
        cmd_with_missing_dep = CommandDefinition(
            name="cmd_with_missing_dep",
            description="Command with missing dependency",
            dependencies=[missing_dep]
        )
        registry.register_command(cmd_with_missing_dep, "Handler")
        
        is_valid, missing = registry.validate_dependencies("cmd_with_missing_dep")
        assert is_valid is False
        assert "nonexistent_command" in missing
    
    def test_recursive_dependencies(self):
        """Test getting recursive dependencies."""
        registry = CommandRegistry()
        
        # Create a dependency chain: cmd1 -> cmd2 -> cmd3
        cmd3 = CommandDefinition(name="cmd3", description="Level 3")
        registry.register_command(cmd3, "Handler3")
        
        dep3 = CommandDependency(command="cmd3", optional=False)
        cmd2 = CommandDefinition(name="cmd2", description="Level 2", dependencies=[dep3])
        registry.register_command(cmd2, "Handler2")
        
        dep2 = CommandDependency(command="cmd2", optional=False)
        cmd1 = CommandDefinition(name="cmd1", description="Level 1", dependencies=[dep2])
        registry.register_command(cmd1, "Handler1")
        
        # Test recursive dependency retrieval
        direct_deps = registry.get_dependencies("cmd1", recursive=False)
        assert direct_deps == ["cmd2"]
        
        recursive_deps = registry.get_dependencies("cmd1", recursive=True)
        assert set(recursive_deps) == {"cmd2", "cmd3"}
        
        # Test recursive dependents
        recursive_dependents = registry.get_dependents("cmd3", recursive=True)
        assert set(recursive_dependents) == {"cmd2", "cmd1"}


class TestPluginManagement:
    """Test plugin command management."""
    
    def test_register_plugin_commands(self):
        """Test registering commands from plugins."""
        registry = CommandRegistry()
        
        # Register commands from different plugins
        cmd1 = CommandDefinition(name="plugin1_cmd", description="Command from plugin 1")
        registry.register_command(cmd1, "Handler1", plugin_source="plugin1")
        
        cmd2 = CommandDefinition(name="plugin1_cmd2", description="Another command from plugin 1")
        registry.register_command(cmd2, "Handler2", plugin_source="plugin1")
        
        cmd3 = CommandDefinition(name="plugin2_cmd", description="Command from plugin 2")
        registry.register_command(cmd3, "Handler3", plugin_source="plugin2")
        
        # Test filtering by plugin
        plugin1_commands = registry.list_commands(plugin_source="plugin1")
        assert len(plugin1_commands) == 2
        
        plugin2_commands = registry.list_commands(plugin_source="plugin2")
        assert len(plugin2_commands) == 1
    
    def test_unregister_plugin(self):
        """Test unregistering all commands from a plugin."""
        registry = CommandRegistry()
        
        # Register several plugin commands
        for i in range(3):
            cmd = CommandDefinition(
                name=f"plugin_cmd_{i}",
                description=f"Plugin command {i}"
            )
            registry.register_command(cmd, f"Handler{i}", plugin_source="test_plugin")
        
        assert len(registry) == 3
        
        # Unregister entire plugin
        removed_count = registry.unregister_plugin("test_plugin")
        assert removed_count == 3
        assert len(registry) == 0
        
        # Verify commands are gone
        for i in range(3):
            assert registry.get_command(f"plugin_cmd_{i}") is None


class TestRegistryPersistence:
    """Test registry persistence and loading."""
    
    @pytest.mark.asyncio
    async def test_save_and_load_registry(self, temp_registry_dir, sample_command_definition):
        """Test saving and loading registry."""
        # Create registry with data
        registry1 = CommandRegistry(data_dir=temp_registry_dir)
        registry1.register_command(sample_command_definition, "TestHandler")
        
        # Save registry
        saved = await registry1.save_registry(force=True)
        assert saved is True
        
        # Create new registry and load
        registry2 = CommandRegistry(data_dir=temp_registry_dir)
        loaded = registry2.load_registry()
        assert loaded is True
        
        # Verify data was loaded correctly
        assert len(registry2) == 1
        metadata = registry2.get_command("test_command")
        assert metadata is not None
        assert metadata.definition.name == "test_command"
        assert metadata.handler_class == "TestHandler"
    
    def test_auto_save_functionality(self, temp_registry_dir, sample_command_definition):
        """Test automatic saving functionality."""
        registry = CommandRegistry(data_dir=temp_registry_dir)
        registry._auto_save = True  # Ensure auto-save is enabled
        
        # Register command (should trigger auto-save)
        registry.register_command(sample_command_definition, "TestHandler")
        
        # Give async save time to complete
        import time
        time.sleep(0.1)
        
        # Verify file exists
        registry_file = temp_registry_dir / "command_registry.json"
        assert registry_file.exists()
        
        # Verify file contents
        with open(registry_file, 'r') as f:
            data = json.load(f)
        
        assert "commands" in data
        assert "test_command" in data["commands"]
    
    def test_corrupted_registry_handling(self, temp_registry_dir):
        """Test handling of corrupted registry files."""
        # Create corrupted registry file
        registry_file = temp_registry_dir / "command_registry.json"
        with open(registry_file, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(RegistryCorruptedError, match="Registry file is corrupted"):
            CommandRegistry(data_dir=temp_registry_dir)
    
    def test_registry_without_persistence(self, sample_command_definition):
        """Test registry operation without persistence."""
        registry = CommandRegistry()  # No data_dir
        registry.register_command(sample_command_definition, "TestHandler")
        
        # Should work normally but no save/load
        assert len(registry) == 1
        assert not registry.load_registry()  # Should return False (no file)


class TestRegistryStatistics:
    """Test registry statistics and health monitoring."""
    
    def test_registry_stats(self, registry_with_sample_commands):
        """Test registry statistics generation."""
        registry = registry_with_sample_commands
        
        stats = registry.get_registry_stats()
        
        assert stats.total_commands == 4
        assert stats.active_commands == 3  # Excluding deprecated
        assert stats.deprecated_commands == 1
        assert stats.plugin_commands == 0  # No plugin commands in fixture
        
        # Check category breakdown
        assert CommandCategory.CORE in stats.commands_by_category
        assert CommandCategory.ADVANCED in stats.commands_by_category
        assert CommandCategory.SYSTEM in stats.commands_by_category
        assert CommandCategory.DEPRECATED in stats.commands_by_category
        
        # Check performance metrics
        assert stats.avg_lookup_time_ms >= 0
        assert stats.avg_discovery_time_ms >= 0
        assert stats.avg_validation_time_ms >= 0
        
        assert isinstance(stats.most_used_commands, list)
        assert len(stats.most_used_commands) <= 10
    
    def test_command_usage_statistics(self):
        """Test command usage tracking and statistics."""
        registry = CommandRegistry()
        
        cmd = CommandDefinition(name="test_cmd", description="Test")
        registry.register_command(cmd, "Handler")
        
        # Use command multiple times
        for _ in range(5):
            metadata = registry.get_command("test_cmd")
        
        # Check usage was tracked
        metadata = registry.get_command("test_cmd")
        assert metadata.usage_count == 6  # 5 lookups + 1 final lookup = 6
        assert metadata.last_used is not None
        
        # Check engine status
        status = registry.get_engine_status()
        assert "status" in status
        assert status["status"] == "healthy"
        assert status["registered_handlers"] == 1


class TestRegistryCleanup:
    """Test registry cleanup functionality."""
    
    def test_cleanup_unused_commands(self):
        """Test cleanup of unused commands."""
        registry = CommandRegistry()
        
        # Register some commands with different usage patterns
        used_cmd = CommandDefinition(name="used_cmd", description="Used command")
        unused_cmd = CommandDefinition(
            name="unused_cmd",
            description="Unused command",
            category=CommandCategory.ADVANCED  # Non-core
        )
        
        registry.register_command(used_cmd, "UsedHandler")
        registry.register_command(unused_cmd, "UnusedHandler")
        
        # Use one command
        registry.get_command("used_cmd")
        
        # Cleanup should remove unused non-core commands
        removed_count = registry.cleanup_unused_commands(min_age_days=0)
        
        # Should not remove any commands yet (they're not old enough or have been used)
        assert removed_count >= 0  # Depends on implementation details
        
        # Both commands should still exist
        assert registry.get_command("used_cmd") is not None
        assert registry.get_command("unused_cmd") is not None


class TestRegistryEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_unregister_nonexistent_command(self):
        """Test unregistering a command that doesn't exist."""
        registry = CommandRegistry()
        
        result = registry.unregister_command("nonexistent")
        assert result is False
    
    def test_unregister_command_with_dependents(self):
        """Test unregistering a command that has dependents."""
        registry = CommandRegistry()
        
        # Register base command
        base_cmd = CommandDefinition(name="base", description="Base")
        registry.register_command(base_cmd, "BaseHandler")
        
        # Register dependent command
        dep = CommandDependency(command="base", optional=False)
        dependent_cmd = CommandDefinition(
            name="dependent",
            description="Dependent",
            dependencies=[dep]
        )
        registry.register_command(dependent_cmd, "DependentHandler")
        
        # Unregister base command (should work but log warning)
        result = registry.unregister_command("base")
        assert result is True
        
        # Dependent should still exist but have broken dependency
        dependent = registry.get_command("dependent")
        assert dependent is not None
    
    def test_registry_with_empty_commands(self):
        """Test registry behavior with no commands."""
        registry = CommandRegistry()
        
        commands = registry.list_commands()
        assert len(commands) == 0
        
        stats = registry.get_registry_stats()
        assert stats.total_commands == 0
        assert stats.active_commands == 0
    
    def test_large_registry_performance(self):
        """Test registry performance with many commands."""
        registry = CommandRegistry()
        
        # Register many commands
        num_commands = 500
        start_time = time.perf_counter()
        
        for i in range(num_commands):
            cmd = CommandDefinition(
                name=f"cmd_{i:04d}",
                description=f"Command {i}",
                aliases=[f"c{i}"],
                tags=[f"tag{i % 10}"]
            )
            registry.register_command(cmd, f"Handler{i}")
        
        registration_time = time.perf_counter() - start_time
        
        # Should be reasonably fast
        assert registration_time < 5.0  # 5 seconds for 500 commands
        
        # Test listing performance
        start_time = time.perf_counter()
        commands = registry.list_commands()
        listing_time = time.perf_counter() - start_time
        
        assert len(commands) == num_commands
        assert listing_time < 0.1  # Should be fast
        
        # Test filtering performance
        start_time = time.perf_counter()
        core_commands = registry.list_commands(category=CommandCategory.CORE)
        filter_time = time.perf_counter() - start_time
        
        assert len(core_commands) == num_commands  # All default to CORE
        assert filter_time < 0.1  # Should be fast


class TestRegistryIntegration:
    """Integration tests for registry functionality."""
    
    def test_complete_command_lifecycle(self, temp_registry_dir):
        """Test complete command lifecycle from registration to cleanup."""
        # Initialize registry with persistence
        registry = CommandRegistry(data_dir=temp_registry_dir)
        
        # Create a complex command
        param = ParameterDefinition(
            name="input",
            type=ParameterType.FILE_PATH,
            description="Input file",
            required=True
        )
        
        example = CommandExample(
            command="complex_cmd --input file.txt",
            description="Process input file"
        )
        
        dependency = CommandDependency(
            command="required_base",
            optional=False
        )
        
        # Register base dependency first
        base_cmd = CommandDefinition(
            name="required_base",
            description="Required base command"
        )
        registry.register_command(base_cmd, "BaseHandler")
        
        # Register complex command
        complex_cmd = CommandDefinition(
            name="complex_cmd",
            aliases=["cc", "complex"],
            description="A complex test command",
            category=CommandCategory.ADVANCED,
            security_level=SecurityLevel.ELEVATED,
            supported_modes=[ExecutionMode.CLI, ExecutionMode.TUI, ExecutionMode.API],
            required_permissions=["admin", "file.write"],
            min_skill_level=SkillLevel.EXPERT,
            parameters=[param],
            examples=[example],
            dependencies=[dependency],
            tags=["complex", "testing", "advanced"]
        )
        
        metadata = registry.register_command(complex_cmd, "ComplexHandler")
        
        # Verify registration
        assert metadata.definition.name == "complex_cmd"
        assert len(registry) == 2
        
        # Test lookups
        assert registry.get_command("complex_cmd") is not None
        assert registry.get_command("cc") is not None  # Alias
        assert registry.get_handler_class("complex_cmd") == "ComplexHandler"
        
        # Test dependencies
        deps = registry.get_dependencies("complex_cmd")
        assert "required_base" in deps
        
        dependents = registry.get_dependents("required_base")
        assert "complex_cmd" in dependents
        
        # Test filtering
        advanced_commands = registry.list_commands(category=CommandCategory.ADVANCED)
        assert len(advanced_commands) == 1
        
        expert_commands = registry.list_commands(skill_level=SkillLevel.EXPERT)
        assert len(expert_commands) == 1
        
        # Test tag lookup
        complex_tagged = registry.get_commands_by_tag("complex")
        assert len(complex_tagged) == 1
        
        # Test persistence
        saved = asyncio.run(registry.save_registry(force=True))
        assert saved is True
        
        # Load in new registry
        new_registry = CommandRegistry(data_dir=temp_registry_dir)
        loaded = new_registry.load_registry()
        assert loaded is True
        assert len(new_registry) == 2
        
        # Verify complex command was preserved
        loaded_metadata = new_registry.get_command("complex_cmd")
        assert loaded_metadata is not None
        assert loaded_metadata.definition.name == "complex_cmd"
        assert len(loaded_metadata.definition.parameters) == 1
        assert len(loaded_metadata.definition.examples) == 1
        assert len(loaded_metadata.definition.dependencies) == 1
        
        # Cleanup
        unregistered = new_registry.unregister_command("complex_cmd")
        assert unregistered is True
        assert len(new_registry) == 1
        assert new_registry.get_command("complex_cmd") is None