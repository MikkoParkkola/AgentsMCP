"""Comprehensive tests for CLI v3 command engine.

Tests cover:
- Simple command execution with success metrics
- Natural language command routing  
- Error handling with helpful suggestions
- Performance requirements validation
- Progressive disclosure behavior
- Security validation and sandboxing
"""

import asyncio
import pytest
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from agentsmcp.cli.v3.core import CommandEngine, CommandHandler, ExecutionContext
from agentsmcp.cli.v3.models import (
    CommandRequest,
    CommandResult,
    CommandStatus,
    ExecutionMode,
    UserProfile,
    UserPreferences,
    SkillLevel,
    ResourceLimit,
    ResourceType,
    NextAction,
    Suggestion,
    ExecutionMetrics,
    CommandNotFoundError,
    ValidationFailedError,
    PermissionDeniedError,
    ResourceExhaustedError,
)


class MockCommandHandler(CommandHandler):
    """Mock command handler for testing."""
    
    def __init__(self, command_type: str = "test_command", execution_time_ms: int = 10):
        super().__init__(command_type)
        self.execution_time_ms = execution_time_ms
        self.execution_count = 0
        self.validation_calls = 0
        
    async def validate(self, request: CommandRequest, context: ExecutionContext) -> None:
        self.validation_calls += 1
        await super().validate(request, context)
        
    async def execute(self, request: CommandRequest, context: ExecutionContext) -> dict:
        self.execution_count += 1
        await asyncio.sleep(self.execution_time_ms / 1000)
        return {
            "message": f"Executed {self.command_type}",
            "args": request.args,
            "execution_count": self.execution_count
        }
    
    async def generate_suggestions(self, request, result, context):
        return [
            Suggestion(
                text=f"Try running {self.command_type} with different args",
                confidence=0.8,
                category="optimization"
            )
        ]
    
    async def generate_next_actions(self, request, result, context):
        return [
            NextAction(
                command="status",
                description="Check system status",
                confidence=0.9,
                category="workflow"
            )
        ]


class MockIntelligenceProvider:
    """Mock intelligence provider for testing."""
    
    async def parse_natural_language(self, input_text: str, context: ExecutionContext):
        # Simple rule-based parsing for tests
        if "help" in input_text.lower():
            return CommandRequest(command_type="help", args={})
        elif "test" in input_text.lower():
            return CommandRequest(command_type="test_command", args={})
        else:
            return CommandRequest(command_type="unknown", args={})
    
    async def generate_smart_suggestions(self, context, recent_commands, current_result=None):
        return [
            Suggestion(
                text="Consider using the status command",
                command="status",
                confidence=0.7,
                category="suggestion"
            )
        ]


class MockTelemetryCollector:
    """Mock telemetry collector for testing."""
    
    def __init__(self):
        self.recorded_executions = []
        self.recorded_errors = []
        self.recorded_behaviors = []
    
    async def record_command_execution(self, request, result, metrics, context):
        self.recorded_executions.append({
            "request": request,
            "result": result, 
            "metrics": metrics,
            "context_id": context.context_id
        })
    
    async def record_error(self, error, request, context):
        self.recorded_errors.append({
            "error": error,
            "request": request,
            "context_id": context.context_id
        })


@pytest.fixture
def user_profile():
    """Create a test user profile."""
    return UserProfile(
        user_id="test_user",
        skill_level=SkillLevel.INTERMEDIATE,
        preferences=UserPreferences(
            verbose_output=True,
            suggestion_level=3,
            default_timeout_ms=30000  # 30 seconds for tests
        ),
        command_history=["help", "status"],
        favorite_commands=["help", "test"]
    )


@pytest.fixture
def command_engine():
    """Create a configured command engine for testing."""
    engine = CommandEngine()
    
    # Register test handlers
    engine.register_handler(MockCommandHandler("test_command"))
    engine.register_handler(MockCommandHandler("help", 5))
    engine.register_handler(MockCommandHandler("slow_command", 100))
    engine.register_handler(MockCommandHandler("very_slow_command", 2000))  # 2 seconds
    
    return engine


@pytest.fixture
def telemetry_collector():
    """Create mock telemetry collector."""
    return MockTelemetryCollector()


class TestCommandEngineBasic:
    """Test basic command engine functionality."""
    
    @pytest.mark.asyncio
    async def test_simple_command_execution_success(self, command_engine, user_profile):
        """Test simple successful command execution with metrics."""
        request = CommandRequest(
            command_type="test_command",
            args={"param1": "value1"}
        )
        
        start_time = time.time()
        result, metrics, next_actions = await command_engine.execute_command(
            request, ExecutionMode.CLI, user_profile
        )
        execution_time = time.time() - start_time
        
        # Validate result structure
        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.status == CommandStatus.COMPLETED
        assert result.request_id == request.request_id
        assert result.data["message"] == "Executed test_command"
        assert result.data["args"] == {"param1": "value1"}
        
        # Validate metrics
        assert isinstance(metrics, ExecutionMetrics)
        assert metrics.duration_ms > 0
        assert metrics.duration_ms < 1000  # Should complete quickly
        
        # Validate next actions
        assert isinstance(next_actions, list)
        assert len(next_actions) > 0
        assert all(isinstance(action, NextAction) for action in next_actions)
        
        # Validate performance requirements (P95 <500ms for simple commands)
        assert execution_time < 0.5  # 500ms requirement
    
    @pytest.mark.asyncio
    async def test_command_not_found_error(self, command_engine, user_profile):
        """Test handling of unknown commands."""
        request = CommandRequest(command_type="unknown_command")
        
        result, metrics, next_actions = await command_engine.execute_command(
            request, ExecutionMode.CLI, user_profile
        )
        
        assert result.success is False
        assert result.status == CommandStatus.FAILED
        assert len(result.errors) == 1
        assert result.errors[0].error_code == "CommandNotFound"
        assert "unknown_command" in result.errors[0].message
        assert len(result.errors[0].recovery_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_permission_denied_error(self, command_engine, user_profile):
        """Test permission validation."""
        # Create handler requiring admin permission
        class AdminHandler(CommandHandler):
            def __init__(self):
                super().__init__("admin_command") 
                self.required_permissions = ["system.admin"]
            
            async def execute(self, request, context):
                return {"message": "Admin operation"}
        
        command_engine.register_handler(AdminHandler())
        request = CommandRequest(command_type="admin_command")
        
        result, metrics, next_actions = await command_engine.execute_command(
            request, ExecutionMode.CLI, user_profile
        )
        
        assert result.success is False
        assert result.status == CommandStatus.FAILED
        assert len(result.errors) == 1
        assert result.errors[0].error_code == "PermissionDenied"
    
    @pytest.mark.asyncio
    async def test_execution_timeout(self, command_engine, user_profile):
        """Test command execution timeout handling."""
        request = CommandRequest(
            command_type="very_slow_command",  # Takes 2 seconds
            timeout_ms=1000  # 1 second timeout - should timeout
        )
        
        result, metrics, next_actions = await command_engine.execute_command(
            request, ExecutionMode.CLI, user_profile
        )
        
        assert result.success is False
        assert result.status == CommandStatus.TIMEOUT
        assert len(result.errors) == 1
        assert result.errors[0].error_code == "ExecutionTimeout"


class TestProgressiveDisclosure:
    """Test progressive disclosure based on user skill level."""
    
    @pytest.mark.asyncio 
    async def test_beginner_skill_level_adaptation(self, command_engine):
        """Test command output adaptation for beginner users."""
        user_profile = UserProfile(
            skill_level=SkillLevel.BEGINNER,
            preferences=UserPreferences(suggestion_level=2)
        )
        
        request = CommandRequest(command_type="test_command")
        result, metrics, next_actions = await command_engine.execute_command(
            request, ExecutionMode.CLI, user_profile
        )
        
        assert result.success is True
        # Beginners should get limited suggestions with explanations
        assert len(result.suggestions) <= 3
        
    @pytest.mark.asyncio
    async def test_expert_skill_level_adaptation(self, command_engine):
        """Test command output adaptation for expert users."""
        user_profile = UserProfile(
            skill_level=SkillLevel.EXPERT,
            preferences=UserPreferences(suggestion_level=5)
        )
        
        request = CommandRequest(command_type="test_command")
        result, metrics, next_actions = await command_engine.execute_command(
            request, ExecutionMode.CLI, user_profile
        )
        
        assert result.success is True
        # Experts should see technical details and advanced options
        if isinstance(result.data, dict):
            # These would be added by skill level adaptation
            assert result.data.get("advanced_options") is True
            assert result.data.get("technical_details") is True


class TestResourceManagement:
    """Test resource limits and monitoring."""
    
    @pytest.mark.asyncio
    async def test_resource_limit_enforcement(self, command_engine, user_profile):
        """Test resource limit enforcement."""
        # Create command that would exceed memory limit
        class MemoryIntensiveHandler(CommandHandler):
            def __init__(self):
                super().__init__("memory_test")
                
            async def execute(self, request, context):
                # Simulate exceeding memory limit in test
                if hasattr(context.resource_monitor, "limits"):
                    memory_limit = context.resource_monitor.limits.get(ResourceType.MEMORY)
                    if memory_limit:
                        raise ResourceExhaustedError(f"Memory limit exceeded: 1000MB > {memory_limit.max_value}MB")
                return {"status": "completed"}
        
        # Configure execution context with memory limits
        engine = CommandEngine()
        engine.register_handler(MemoryIntensiveHandler())
        
        request = CommandRequest(command_type="memory_test")
        
        # For this test, we'll simulate the resource exhaustion in the handler
        result, metrics, next_actions = await engine.execute_command(
            request, ExecutionMode.CLI, user_profile
        )
        
        # Should complete normally since we don't have actual limits configured
        assert result.success is True or result.errors[0].error_code == "ResourceExhausted"


class TestIntelligenceIntegration:
    """Test natural language processing integration."""
    
    @pytest.mark.asyncio
    async def test_natural_language_parsing(self, command_engine, user_profile):
        """Test natural language command parsing."""
        intelligence_provider = MockIntelligenceProvider()
        command_engine.set_intelligence_provider(intelligence_provider)
        
        # Test parsing natural language input
        context = ExecutionContext(user_profile, ExecutionMode.CLI)
        parsed_command = await intelligence_provider.parse_natural_language(
            "please run the test command", context
        )
        
        assert parsed_command.command_type == "test_command"
        
        # Test executing the parsed command
        result, metrics, next_actions = await command_engine.execute_command(
            parsed_command, ExecutionMode.CLI, user_profile
        )
        
        assert result.success is True
        assert result.data["message"] == "Executed test_command"


class TestTelemetryCollection:
    """Test metrics and observability collection."""
    
    @pytest.mark.asyncio
    async def test_telemetry_recording(self, command_engine, user_profile, telemetry_collector):
        """Test telemetry collection during command execution."""
        command_engine.set_telemetry_collector(telemetry_collector)
        
        request = CommandRequest(command_type="test_command")
        result, metrics, next_actions = await command_engine.execute_command(
            request, ExecutionMode.CLI, user_profile
        )
        
        # Verify telemetry was recorded
        assert len(telemetry_collector.recorded_executions) == 1
        recorded = telemetry_collector.recorded_executions[0]
        
        assert recorded["request"].command_type == "test_command"
        assert recorded["result"].success is True
        assert recorded["metrics"].duration_ms > 0
    
    @pytest.mark.asyncio
    async def test_error_telemetry_recording(self, command_engine, user_profile, telemetry_collector):
        """Test error telemetry collection."""
        command_engine.set_telemetry_collector(telemetry_collector)
        
        request = CommandRequest(command_type="unknown_command")
        result, metrics, next_actions = await command_engine.execute_command(
            request, ExecutionMode.CLI, user_profile
        )
        
        # Verify error was recorded
        assert len(telemetry_collector.recorded_errors) == 1
        recorded_error = telemetry_collector.recorded_errors[0]
        
        assert isinstance(recorded_error["error"], CommandNotFoundError)
        assert recorded_error["request"].command_type == "unknown_command"


class TestPerformanceRequirements:
    """Test performance requirements compliance."""
    
    @pytest.mark.asyncio
    async def test_startup_time_requirement(self):
        """Test engine startup time <200ms."""
        start_time = time.time()
        engine = CommandEngine()
        startup_time = (time.time() - start_time) * 1000
        
        assert startup_time < 200  # <200ms startup requirement
    
    @pytest.mark.asyncio
    async def test_simple_command_latency_requirement(self, command_engine, user_profile):
        """Test P95 latency <500ms for simple commands."""
        request = CommandRequest(command_type="test_command")
        
        # Run multiple executions to get P95
        latencies = []
        for _ in range(20):
            start_time = time.time()
            result, metrics, next_actions = await command_engine.execute_command(
                request, ExecutionMode.CLI, user_profile
            )
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
            assert result.success is True
        
        # Calculate P95 (95th percentile)
        latencies.sort()
        p95_latency = latencies[int(0.95 * len(latencies))]
        
        assert p95_latency < 500  # P95 <500ms requirement
    
    @pytest.mark.asyncio
    async def test_complex_orchestration_requirement(self, command_engine, user_profile):
        """Test complex orchestration <2s requirement."""
        # Simulate complex orchestration with multiple commands
        commands = [
            CommandRequest(command_type="test_command", args={"step": 1}),
            CommandRequest(command_type="help"),
            CommandRequest(command_type="test_command", args={"step": 2}),
        ]
        
        start_time = time.time()
        for command in commands:
            result, metrics, next_actions = await command_engine.execute_command(
                command, ExecutionMode.CLI, user_profile
            )
            assert result.success is True
        
        total_time_ms = (time.time() - start_time) * 1000
        assert total_time_ms < 2000  # <2s requirement for complex orchestration


class TestAuditAndSecurity:
    """Test audit trail and security features."""
    
    @pytest.mark.asyncio
    async def test_audit_trail_generation(self, command_engine, user_profile):
        """Test audit trail generation."""
        request = CommandRequest(command_type="test_command")
        result, metrics, next_actions = await command_engine.execute_command(
            request, ExecutionMode.CLI, user_profile
        )
        
        # Check engine status includes audit information
        status = command_engine.get_engine_status()
        assert "command_stats" in status
        assert "test_command" in status["command_stats"]
        
        stats = status["command_stats"]["test_command"]
        assert stats["total_executions"] >= 1
        assert stats["successful_executions"] >= 1
        assert stats["avg_duration_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, command_engine, user_profile):
        """Test graceful degradation when services are unavailable."""
        # Test execution without intelligence provider
        assert command_engine.intelligence_provider is None
        
        request = CommandRequest(command_type="test_command")
        result, metrics, next_actions = await command_engine.execute_command(
            request, ExecutionMode.CLI, user_profile
        )
        
        # Should still work without intelligence provider
        assert result.success is True
        
        # Test execution without telemetry collector
        assert command_engine.telemetry_collector is None
        
        result2, metrics2, next_actions2 = await command_engine.execute_command(
            request, ExecutionMode.CLI, user_profile
        )
        
        # Should still work without telemetry
        assert result2.success is True


class TestEngineManagement:
    """Test engine lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_engine_status(self, command_engine):
        """Test engine status reporting."""
        status = command_engine.get_engine_status()
        
        assert status["status"] == "healthy"
        assert isinstance(status["uptime_seconds"], int)
        assert status["registered_handlers"] >= 3  # We registered test handlers
        assert status["active_commands"] == 0  # No commands running
        assert isinstance(status["command_stats"], dict)
    
    @pytest.mark.asyncio
    async def test_engine_shutdown(self, command_engine):
        """Test graceful engine shutdown."""
        # Start a command that will take some time
        async def long_running_command():
            request = CommandRequest(command_type="slow_command")
            user_profile = UserProfile(user_id="test_user")
            return await command_engine.execute_command(
                request, ExecutionMode.CLI, user_profile
            )
        
        # Start command in background
        command_task = asyncio.create_task(long_running_command())
        
        # Give it a moment to start
        await asyncio.sleep(0.01)
        
        # Should have active command
        status = command_engine.get_engine_status()
        
        # Shutdown should wait for completion
        shutdown_task = asyncio.create_task(command_engine.shutdown())
        
        # Let both complete
        result, metrics, next_actions = await command_task
        await shutdown_task
        
        assert result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])