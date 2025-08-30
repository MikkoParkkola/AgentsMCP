"""Integration tests for CLI v3 command engine pipeline.

These tests demonstrate the complete command execution pipeline including
natural language processing, progressive disclosure, and cross-modal coordination.
"""

import asyncio
import pytest
import time
from typing import Any, Dict, List

from agentsmcp.cli.v3.core import CommandEngine, CommandHandler, ExecutionContext
from agentsmcp.cli.v3.models import (
    CommandRequest,
    CommandResult,
    CommandStatus,
    ExecutionMode,
    UserProfile,
    UserPreferences,
    SkillLevel,
    NextAction,
    Suggestion,
    ExecutionMetrics,
)


# Sample command handlers for integration testing
class HelpCommandHandler(CommandHandler):
    """Handler for help commands."""
    
    def __init__(self):
        super().__init__("help")
        
    async def execute(self, request: CommandRequest, context: ExecutionContext) -> Dict[str, Any]:
        topic = request.args.get("topic", "general")
        
        help_content = {
            "general": "AgentsMCP CLI v3 - Intelligent command processor",
            "commands": "Available commands: help, status, test, config",
            "advanced": "Advanced features: natural language processing, smart suggestions"
        }
        
        return {
            "topic": topic,
            "content": help_content.get(topic, "No help available for this topic"),
            "available_topics": list(help_content.keys())
        }
    
    async def generate_suggestions(self, request, result, context):
        if context.user_profile.skill_level == SkillLevel.BEGINNER:
            return [
                Suggestion(
                    text="Try 'help commands' to see what you can do",
                    command="help",
                    confidence=0.9,
                    category="tutorial"
                )
            ]
        else:
            return [
                Suggestion(
                    text="Use 'status' to check system health",
                    command="status",
                    confidence=0.8,
                    category="workflow"
                )
            ]
    
    async def generate_next_actions(self, request, result, context):
        return [
            NextAction(
                command="status",
                description="Check system status",
                confidence=0.7,
                category="workflow"
            ),
            NextAction(
                command="config show",
                description="View current configuration",
                confidence=0.6,
                category="configuration"
            )
        ]


class StatusCommandHandler(CommandHandler):
    """Handler for status commands."""
    
    def __init__(self):
        super().__init__("status")
        
    async def execute(self, request: CommandRequest, context: ExecutionContext) -> Dict[str, Any]:
        return {
            "system_status": "healthy",
            "active_commands": 0,
            "user_session": {
                "user_id": context.user_profile.user_id,
                "skill_level": context.user_profile.skill_level.value,
                "session_duration": 0,  # Would calculate actual duration
            },
            "capabilities": context.capabilities,
            "execution_mode": context.execution_mode.value
        }
    
    async def generate_next_actions(self, request, result, context):
        actions = []
        
        if context.user_profile.skill_level != SkillLevel.BEGINNER:
            actions.append(
                NextAction(
                    command="config edit",
                    description="Modify system configuration",
                    confidence=0.5,
                    category="configuration"
                )
            )
        
        return actions


class ConfigCommandHandler(CommandHandler):
    """Handler for configuration commands."""
    
    def __init__(self):
        super().__init__("config")
        self.required_permissions = ["config.view"]
        
    async def validate(self, request: CommandRequest, context: ExecutionContext) -> None:
        await super().validate(request, context)
        
        # Additional validation for edit operations
        action = request.args.get("action", "show")
        if action in ["edit", "set", "delete"]:
            context.require_permission("config.edit")
    
    async def execute(self, request: CommandRequest, context: ExecutionContext) -> Dict[str, Any]:
        action = request.args.get("action", "show")
        
        if action == "show":
            return {
                "action": "show",
                "config": {
                    "user_preferences": context.user_profile.preferences.model_dump(),
                    "execution_mode": context.execution_mode.value,
                    "capabilities": context.capabilities
                }
            }
        elif action == "edit":
            key = request.args.get("key")
            value = request.args.get("value")
            
            if not key:
                raise ValueError("Key is required for edit action")
                
            return {
                "action": "edit",
                "key": key,
                "old_value": None,  # Would get from actual config
                "new_value": value,
                "status": "updated"
            }
        else:
            raise ValueError(f"Unknown config action: {action}")
    
    async def generate_next_actions(self, request, result, context):
        """Generate next actions for config command."""
        action = request.args.get("action", "show")
        
        actions = []
        
        if action == "show":
            # After showing config, suggest editing if user has permission
            if context.check_permission("config.edit"):
                actions.append(
                    NextAction(
                        command="config edit",
                        description="Edit configuration settings",
                        confidence=0.6,
                        category="configuration"
                    )
                )
                
            # Always suggest viewing help
            actions.append(
                NextAction(
                    command="help config",
                    description="Get help with configuration commands",
                    confidence=0.5,
                    category="help"
                )
            )
        elif action == "edit":
            # After editing, suggest viewing the changes
            actions.append(
                NextAction(
                    command="config show",
                    description="View updated configuration",
                    confidence=0.8,
                    category="configuration"
                )
            )
            
        return actions


class NaturalLanguageIntelligenceProvider:
    """Enhanced intelligence provider for integration testing."""
    
    def __init__(self):
        # Simple keyword-based NL understanding for testing
        self.command_patterns = {
            "help": ["help", "assist", "guide", "how", "understand", "available", "commands"],
            "status": ["status", "current", "system", "health", "check", "state"],
            "config": ["config", "configuration", "setting", "settings", "preference", "configure", "show"],
        }
    
    async def parse_natural_language(self, input_text: str, context: ExecutionContext) -> CommandRequest:
        """Parse natural language into command request."""
        text_lower = input_text.lower()
        
        # Extract command type by checking patterns
        command_type = "help"  # default
        best_match_score = 0
        
        for cmd, patterns in self.command_patterns.items():
            # Count how many patterns match
            matches = sum(1 for pattern in patterns if pattern in text_lower)
            if matches > best_match_score:
                best_match_score = matches
                command_type = cmd
        
        # Extract arguments based on command type
        args = {}
        
        if command_type == "help":
            if "command" in text_lower:
                args["topic"] = "commands"
            elif "advanced" in text_lower:
                args["topic"] = "advanced"
                
        elif command_type == "config":
            # Check for explicit edit actions vs just mentioning "settings"
            if ("edit" in text_lower and "config" in text_lower) or (" set " in text_lower):
                args["action"] = "edit"
                # Simple key=value extraction
                if "=" in input_text:
                    parts = input_text.split("=")
                    if len(parts) == 2:
                        args["key"] = parts[0].strip().split()[-1]
                        args["value"] = parts[1].strip()
            else:
                args["action"] = "show"
        
        return CommandRequest(
            command_type=command_type,
            args=args,
            raw_input=input_text
        )
    
    async def analyze_user_intent(self, request: CommandRequest, context: ExecutionContext) -> Dict[str, Any]:
        """Analyze user intent for additional context."""
        intent_analysis = {
            "confidence": 0.8,
            "primary_intent": request.command_type,
            "user_expertise": context.user_profile.skill_level.value,
            "context_clues": []
        }
        
        # Analyze raw input if available
        if request.raw_input:
            if "?" in request.raw_input:
                intent_analysis["context_clues"].append("questioning")
            if "please" in request.raw_input.lower():
                intent_analysis["context_clues"].append("polite")
            if "urgent" in request.raw_input.lower() or "quickly" in request.raw_input.lower():
                intent_analysis["context_clues"].append("urgent")
        
        return intent_analysis
    
    async def generate_smart_suggestions(self, context, recent_commands, current_result=None):
        """Generate contextual suggestions."""
        suggestions = []
        
        # Suggest based on recent command patterns
        if "help" in recent_commands and "status" not in recent_commands:
            suggestions.append(
                Suggestion(
                    text="Check system status to see current state",
                    command="status",
                    confidence=0.8,
                    category="workflow"
                )
            )
        
        if context.user_profile.skill_level == SkillLevel.EXPERT:
            suggestions.append(
                Suggestion(
                    text="Use natural language: 'show me the config settings'",
                    confidence=0.7,
                    category="advanced"
                )
            )
        
        return suggestions


class IntegrationTelemetryCollector:
    """Comprehensive telemetry collector for integration testing."""
    
    def __init__(self):
        self.metrics = {
            "command_executions": [],
            "errors": [],
            "user_behaviors": [],
            "performance_metrics": {},
        }
    
    async def record_command_execution(self, request, result, metrics, context):
        """Record detailed command execution metrics."""
        self.metrics["command_executions"].append({
            "timestamp": result.timestamp,
            "command_type": request.command_type,
            "success": result.success,
            "duration_ms": metrics.duration_ms,
            "user_id": context.user_profile.user_id,
            "execution_mode": context.execution_mode.value,
            "skill_level": context.user_profile.skill_level.value,
            "suggestions_count": len(result.suggestions),
            "raw_input_used": request.raw_input is not None,
        })
        
        # Track performance by command type
        cmd_type = request.command_type
        if cmd_type not in self.metrics["performance_metrics"]:
            self.metrics["performance_metrics"][cmd_type] = {
                "total_calls": 0,
                "total_duration_ms": 0,
                "success_count": 0,
                "avg_duration_ms": 0
            }
        
        perf = self.metrics["performance_metrics"][cmd_type]
        perf["total_calls"] += 1
        perf["total_duration_ms"] += metrics.duration_ms
        if result.success:
            perf["success_count"] += 1
        perf["avg_duration_ms"] = perf["total_duration_ms"] // perf["total_calls"]
    
    async def record_error(self, error, request, context):
        """Record error details."""
        self.metrics["errors"].append({
            "error_type": type(error).__name__,
            "error_message": str(error),
            "command_type": request.command_type,
            "user_id": context.user_profile.user_id,
            "skill_level": context.user_profile.skill_level.value,
        })
    
    async def record_user_behavior(self, event, data, context):
        """Record user behavior patterns."""
        self.metrics["user_behaviors"].append({
            "event": event,
            "data": data,
            "user_id": context.user_profile.user_id,
            "timestamp": data.get("timestamp"),
        })
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Generate analytics summary."""
        total_commands = len(self.metrics["command_executions"])
        successful_commands = sum(1 for cmd in self.metrics["command_executions"] if cmd["success"])
        
        return {
            "total_commands": total_commands,
            "success_rate": successful_commands / total_commands if total_commands > 0 else 0,
            "avg_duration_ms": (
                sum(cmd["duration_ms"] for cmd in self.metrics["command_executions"]) 
                // total_commands if total_commands > 0 else 0
            ),
            "command_types_used": list(self.metrics["performance_metrics"].keys()),
            "error_count": len(self.metrics["errors"]),
            "natural_language_usage": sum(
                1 for cmd in self.metrics["command_executions"] 
                if cmd["raw_input_used"]
            ),
        }


@pytest.fixture
def integrated_engine():
    """Create a fully integrated command engine."""
    engine = CommandEngine()
    
    # Register handlers
    engine.register_handler(HelpCommandHandler())
    engine.register_handler(StatusCommandHandler())
    engine.register_handler(ConfigCommandHandler())
    
    # Set up intelligence provider
    intelligence = NaturalLanguageIntelligenceProvider()
    engine.set_intelligence_provider(intelligence)
    
    # Set up telemetry
    telemetry = IntegrationTelemetryCollector()
    engine.set_telemetry_collector(telemetry)
    
    return engine, intelligence, telemetry


@pytest.fixture
def test_users():
    """Create test users with different skill levels."""
    return {
        "beginner": UserProfile(
            user_id="beginner_user",
            skill_level=SkillLevel.BEGINNER,
            preferences=UserPreferences(
                verbose_output=True,
                suggestion_level=2
            ),
            command_history=[]
        ),
        "intermediate": UserProfile(
            user_id="intermediate_user", 
            skill_level=SkillLevel.INTERMEDIATE,
            preferences=UserPreferences(
                suggestion_level=3,
                auto_confirm=False
            ),
            command_history=["help", "status"]
        ),
        "expert": UserProfile(
            user_id="expert_user",
            skill_level=SkillLevel.EXPERT,
            preferences=UserPreferences(
                verbose_output=False,
                suggestion_level=5,
                auto_confirm=True
            ),
            command_history=["config", "status", "help", "config edit"]
        )
    }


class TestFullPipeline:
    """Test complete command execution pipeline."""
    
    @pytest.mark.asyncio
    async def test_natural_language_to_execution_pipeline(self, integrated_engine, test_users):
        """Test complete natural language to command execution pipeline."""
        engine, intelligence, telemetry = integrated_engine
        user = test_users["intermediate"]
        
        # Natural language inputs
        natural_inputs = [
            "Help me understand the available commands",
            "What's the current system status?",
            "Show me the configuration settings"
        ]
        
        results = []
        
        for nl_input in natural_inputs:
            # Parse natural language
            parsed_command = await intelligence.parse_natural_language(
                nl_input, ExecutionContext(user, ExecutionMode.CLI)
            )
            
            # Execute command
            result, metrics, next_actions = await engine.execute_command(
                parsed_command, ExecutionMode.CLI, user
            )
            
            results.append((nl_input, parsed_command, result, metrics, next_actions))
            
            # Verify success
            assert result.success is True
            assert result.status == CommandStatus.COMPLETED
            assert len(next_actions) > 0
        
        # Analyze results
        command_types = [parsed_cmd.command_type for _, parsed_cmd, _, _, _ in results]
        assert "help" in command_types
        assert "status" in command_types
        assert "config" in command_types
        
        # Check telemetry collected data
        analytics = telemetry.get_analytics_summary()
        assert analytics["total_commands"] == 3
        assert analytics["success_rate"] == 1.0
        assert analytics["natural_language_usage"] == 3
    
    @pytest.mark.asyncio
    async def test_progressive_disclosure_across_skill_levels(self, integrated_engine, test_users):
        """Test progressive disclosure adapts to different skill levels."""
        engine, intelligence, telemetry = integrated_engine
        
        # Same command executed by users with different skill levels
        help_request = CommandRequest(command_type="help", args={"topic": "commands"})
        
        results_by_skill = {}
        
        for skill_level, user in test_users.items():
            result, metrics, next_actions = await engine.execute_command(
                help_request, ExecutionMode.CLI, user
            )
            
            results_by_skill[skill_level] = {
                "result": result,
                "metrics": metrics,
                "next_actions": next_actions
            }
        
        # Beginner should get simplified output with more guidance
        beginner_result = results_by_skill["beginner"]["result"]
        assert beginner_result.success is True
        
        # Expert should get more technical details
        expert_result = results_by_skill["expert"]["result"]
        assert expert_result.success is True
        
        # Content should be adapted based on skill level
        if isinstance(expert_result.data, dict):
            assert expert_result.data.get("advanced_options") is True
            assert expert_result.data.get("technical_details") is True
    
    @pytest.mark.asyncio
    async def test_cross_modal_execution_consistency(self, integrated_engine, test_users):
        """Test command execution consistency across different modes."""
        engine, intelligence, telemetry = integrated_engine
        user = test_users["intermediate"]
        
        # Execute same command in different modes
        status_request = CommandRequest(command_type="status")
        modes = [ExecutionMode.CLI, ExecutionMode.TUI, ExecutionMode.WEB_UI, ExecutionMode.API]
        
        results_by_mode = {}
        
        for mode in modes:
            result, metrics, next_actions = await engine.execute_command(
                status_request, mode, user
            )
            
            results_by_mode[mode] = {
                "result": result,
                "metrics": metrics,
                "next_actions": next_actions
            }
            
            # All modes should succeed
            assert result.success is True
            assert result.status == CommandStatus.COMPLETED
        
        # Core data should be consistent across modes
        cli_data = results_by_mode[ExecutionMode.CLI]["result"].data
        tui_data = results_by_mode[ExecutionMode.TUI]["result"].data
        
        assert cli_data["system_status"] == tui_data["system_status"]
        assert cli_data["user_session"]["user_id"] == tui_data["user_session"]["user_id"]
        
        # But capabilities should differ
        assert cli_data["capabilities"] != tui_data["capabilities"]
    
    @pytest.mark.asyncio
    async def test_smart_suggestion_workflow(self, integrated_engine, test_users):
        """Test smart suggestion generation and workflow continuation."""
        engine, intelligence, telemetry = integrated_engine
        user = test_users["expert"]
        
        # Execute help command
        help_request = CommandRequest(command_type="help")
        help_result, help_metrics, help_next_actions = await engine.execute_command(
            help_request, ExecutionMode.CLI, user
        )
        
        assert help_result.success is True
        assert len(help_next_actions) > 0
        
        # Follow suggested next action
        suggested_action = help_next_actions[0]
        follow_up_request = CommandRequest(command_type=suggested_action.command.split()[0])
        
        follow_up_result, follow_up_metrics, _ = await engine.execute_command(
            follow_up_request, ExecutionMode.CLI, user
        )
        
        assert follow_up_result.success is True
        
        # Verify workflow continuity through telemetry
        analytics = telemetry.get_analytics_summary()
        assert analytics["total_commands"] == 2
        assert analytics["success_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_performance_requirements_under_load(self, integrated_engine, test_users):
        """Test performance requirements under concurrent load."""
        engine, intelligence, telemetry = integrated_engine
        user = test_users["intermediate"]
        
        # Create multiple concurrent command executions
        concurrent_requests = [
            CommandRequest(command_type="help"),
            CommandRequest(command_type="status"),
            CommandRequest(command_type="config", args={"action": "show"}),
            CommandRequest(command_type="help", args={"topic": "commands"}),
            CommandRequest(command_type="status"),
        ]
        
        # Execute concurrently
        start_time = time.time()
        
        tasks = [
            engine.execute_command(request, ExecutionMode.CLI, user)
            for request in concurrent_requests
        ]
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # All commands should succeed
        for result, metrics, next_actions in results:
            assert result.success is True
            assert result.status == CommandStatus.COMPLETED
            assert isinstance(metrics, ExecutionMetrics)
            assert metrics.duration_ms > 0
        
        # Performance requirements
        # Total time should be reasonable for concurrent execution
        assert total_time < 2.0  # Should complete within 2 seconds
        
        # Individual command latencies should meet requirements
        for result, metrics, next_actions in results:
            assert metrics.duration_ms < 500  # P95 <500ms for simple commands
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_suggestions(self, integrated_engine, test_users):
        """Test error handling and recovery suggestions."""
        engine, intelligence, telemetry = integrated_engine
        user = test_users["beginner"]
        
        # Execute command that will fail (unknown command)
        invalid_request = CommandRequest(command_type="unknown_command")
        
        result, metrics, next_actions = await engine.execute_command(
            invalid_request, ExecutionMode.CLI, user
        )
        
        # Should fail gracefully
        assert result.success is False
        assert result.status == CommandStatus.FAILED
        assert len(result.errors) == 1
        
        error = result.errors[0]
        assert error.error_code == "CommandNotFound"
        assert len(error.recovery_suggestions) > 0
        
        # Recovery suggestions should be helpful
        suggestions = error.recovery_suggestions
        assert any("help" in suggestion.lower() for suggestion in suggestions)
        
        # Telemetry should record the error
        analytics = telemetry.get_analytics_summary()
        assert analytics["error_count"] == 1
    
    @pytest.mark.asyncio
    async def test_audit_trail_and_analytics(self, integrated_engine, test_users):
        """Test comprehensive audit trail and analytics collection."""
        engine, intelligence, telemetry = integrated_engine
        
        # Execute commands with different users
        test_scenarios = [
            (test_users["beginner"], "help", ExecutionMode.CLI),
            (test_users["intermediate"], "status", ExecutionMode.TUI),
            (test_users["expert"], "config", ExecutionMode.API),
        ]
        
        for user, command_type, mode in test_scenarios:
            request = CommandRequest(command_type=command_type)
            result, metrics, next_actions = await engine.execute_command(
                request, mode, user
            )
            
            assert result.success is True
        
        # Analyze collected analytics
        analytics = telemetry.get_analytics_summary()
        
        assert analytics["total_commands"] == 3
        assert analytics["success_rate"] == 1.0
        assert len(analytics["command_types_used"]) == 3
        assert analytics["error_count"] == 0
        
        # Performance metrics should be collected per command type
        for cmd_type in ["help", "status", "config"]:
            assert cmd_type in telemetry.metrics["performance_metrics"]
            perf = telemetry.metrics["performance_metrics"][cmd_type]
            assert perf["total_calls"] == 1
            assert perf["success_count"] == 1
            assert perf["avg_duration_ms"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])