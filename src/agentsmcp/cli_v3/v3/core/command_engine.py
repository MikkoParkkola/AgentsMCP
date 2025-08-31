"""Core command engine for CLI v3 architecture.

This module provides the central CommandEngine class that handles intelligent 
command routing, validation, execution lifecycle management, and smart suggestions.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from ..models.command_models import (
    CommandRequest,
    CommandResult,
    CommandStatus,
    CommandError,
    ExecutionMetrics,
    NextAction,
    Suggestion,
    UserProfile,
    ExecutionMode,
    AuditLogEntry,
    CommandEngineError,
    CommandNotFoundError,
    ValidationFailedError,
    ExecutionTimeoutError,
    PermissionDeniedError,
    ResourceExhaustedError,
)
from .execution_context import ExecutionContext


logger = logging.getLogger(__name__)


class CommandHook:
    """Hook configuration for command pipeline integration."""
    
    def __init__(self, name: str, handler: Callable, priority: int = 5):
        self.name = name
        self.handler = handler
        self.priority = priority  # 1=highest, 10=lowest
        self.enabled = True


class CommandHandler:
    """Base class for command handlers."""
    
    def __init__(self, command_type: str):
        self.command_type = command_type
        self.supported_modes = [ExecutionMode.CLI, ExecutionMode.TUI, ExecutionMode.WEB_UI, ExecutionMode.API]
        self.required_permissions: List[str] = []
    
    async def validate(self, request: CommandRequest, context: ExecutionContext) -> None:
        """Validate command request. Raise ValidationFailedError if invalid."""
        # Check permissions
        for perm in self.required_permissions:
            context.require_permission(perm)
        
        # Check execution mode support
        if context.execution_mode not in self.supported_modes:
            raise ValidationFailedError(
                f"Command {self.command_type} not supported in {context.execution_mode} mode"
            )
    
    async def execute(self, request: CommandRequest, context: ExecutionContext) -> Any:
        """Execute the command and return result data."""
        raise NotImplementedError(f"Command handler {self.command_type} must implement execute()")
    
    async def generate_suggestions(
        self, 
        request: CommandRequest, 
        result: Any,
        context: ExecutionContext
    ) -> List[Suggestion]:
        """Generate smart suggestions based on command result."""
        return []
    
    async def generate_next_actions(
        self,
        request: CommandRequest,
        result: Any, 
        context: ExecutionContext
    ) -> List[NextAction]:
        """Generate suggested next actions."""
        return []


class IntelligenceProvider:
    """Interface for natural language processing and user intelligence."""
    
    async def parse_natural_language(self, input_text: str, context: ExecutionContext) -> CommandRequest:
        """Parse natural language input into structured command request."""
        raise NotImplementedError("Intelligence provider must implement parse_natural_language()")
    
    async def analyze_user_intent(
        self, 
        request: CommandRequest,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Analyze user intent and provide additional context."""
        return {}
    
    async def generate_smart_suggestions(
        self,
        context: ExecutionContext,
        recent_commands: List[str],
        current_result: Any = None
    ) -> List[Suggestion]:
        """Generate contextual smart suggestions."""
        return []


class TelemetryCollector:
    """Interface for metrics and observability collection."""
    
    async def record_command_execution(
        self,
        request: CommandRequest,
        result: CommandResult, 
        metrics: ExecutionMetrics,
        context: ExecutionContext
    ) -> None:
        """Record command execution metrics."""
        pass
    
    async def record_error(
        self,
        error: Exception,
        request: CommandRequest,
        context: ExecutionContext
    ) -> None:
        """Record error occurrence."""
        pass
    
    async def record_user_behavior(
        self,
        event: str,
        data: Dict[str, Any],
        context: ExecutionContext
    ) -> None:
        """Record user behavior patterns."""
        pass


class CrossModalCoordinator:
    """Interface for coordinating across different interface modes."""
    
    async def notify_mode_switch(
        self,
        from_mode: ExecutionMode,
        to_mode: ExecutionMode, 
        context: ExecutionContext
    ) -> None:
        """Notify of interface mode switch."""
        pass
    
    async def sync_session_state(
        self,
        context: ExecutionContext,
        target_modes: List[ExecutionMode]
    ) -> None:
        """Synchronize session state across modes."""
        pass


class CommandEngine:
    """Central command execution engine with intelligent routing and lifecycle management.
    
    This engine provides:
    - Command validation and preprocessing
    - Intelligent routing based on context
    - Resource monitoring and limits enforcement
    - Progressive disclosure based on user skill level
    - Smart suggestions and next action recommendations
    - Comprehensive audit trail and metrics
    """
    
    def __init__(self):
        # Core components
        self._handlers: Dict[str, CommandHandler] = {}
        self._pre_hooks: List[CommandHook] = []
        self._post_hooks: List[CommandHook] = []
        
        # Integration points
        self.intelligence_provider: Optional[IntelligenceProvider] = None
        self.telemetry_collector: Optional[TelemetryCollector] = None
        self.cross_modal_coordinator: Optional[CrossModalCoordinator] = None
        
        # State management
        self._active_commands: Set[str] = set()
        self._command_stats: Dict[str, Dict[str, Any]] = {}
        self._startup_time = datetime.now(timezone.utc)
        
        logger.info("CommandEngine initialized")
    
    def register_handler(self, handler: CommandHandler) -> None:
        """Register a command handler."""
        self._handlers[handler.command_type] = handler
        logger.debug(f"Registered handler for command: {handler.command_type}")
    
    def add_pre_hook(self, hook: CommandHook) -> None:
        """Add a pre-execution hook."""
        self._pre_hooks.append(hook)
        self._pre_hooks.sort(key=lambda h: h.priority)
        logger.debug(f"Added pre-hook: {hook.name}")
    
    def add_post_hook(self, hook: CommandHook) -> None:
        """Add a post-execution hook.""" 
        self._post_hooks.append(hook)
        self._post_hooks.sort(key=lambda h: h.priority)
        logger.debug(f"Added post-hook: {hook.name}")
    
    def set_intelligence_provider(self, provider: IntelligenceProvider) -> None:
        """Set natural language intelligence provider."""
        self.intelligence_provider = provider
        logger.info("Intelligence provider configured")
    
    def set_telemetry_collector(self, collector: TelemetryCollector) -> None:
        """Set telemetry collector for metrics."""
        self.telemetry_collector = collector
        logger.info("Telemetry collector configured")
    
    def set_cross_modal_coordinator(self, coordinator: CrossModalCoordinator) -> None:
        """Set cross-modal coordinator."""
        self.cross_modal_coordinator = coordinator
        logger.info("Cross-modal coordinator configured")
    
    async def execute_command(
        self,
        command: CommandRequest,
        execution_mode: ExecutionMode,
        user_profile: UserProfile
    ) -> Tuple[CommandResult, ExecutionMetrics, List[NextAction]]:
        """Execute a command following the exact ICD interface.
        
        Args:
            command: CommandRequest with type, args, and context
            execution_mode: Interface mode (CLI, TUI, WebUI, API)
            user_profile: User context with skill level, preferences, history
            
        Returns:
            Tuple of (CommandResult, ExecutionMetrics, List[NextAction])
            
        Raises:
            CommandNotFound: Command type not registered
            ValidationFailed: Command validation failed
            ExecutionTimeout: Command execution timed out
            PermissionDenied: User lacks required permissions
            ResourceExhausted: Resource limits exceeded
        """
        start_time = time.perf_counter()
        context = ExecutionContext(user_profile, execution_mode)
        
        # Track active command
        self._active_commands.add(command.request_id)
        
        try:
            # Set up timeout for entire execution if specified
            timeout_seconds = None
            if command.timeout_ms:
                timeout_seconds = command.timeout_ms / 1000
            elif user_profile.preferences.default_timeout_ms:
                timeout_seconds = user_profile.preferences.default_timeout_ms / 1000
            
            async with context.command_execution(command.command_type):
                # Execute the command pipeline with timeout
                if timeout_seconds:
                    try:
                        result_data, suggestions = await asyncio.wait_for(
                            self._execute_pipeline(command, context),
                            timeout=timeout_seconds
                        )
                    except asyncio.TimeoutError:
                        timeout_ms = int(timeout_seconds * 1000)
                        raise asyncio.TimeoutError(f"Command {command.command_type} timed out after {timeout_ms}ms")
                else:
                    result_data, suggestions = await self._execute_pipeline(command, context)
                
                # Generate final result
                result = CommandResult(
                    request_id=command.request_id,
                    success=True,
                    status=CommandStatus.COMPLETED,
                    data=context.adapt_for_skill_level(result_data),
                    suggestions=suggestions
                )
                
                # Generate next actions
                handler = self._handlers.get(command.command_type)
                next_actions = []
                if handler:
                    next_actions = await handler.generate_next_actions(command, result_data, context)
                
                # Add intelligent suggestions
                if self.intelligence_provider:
                    smart_suggestions = await self.intelligence_provider.generate_smart_suggestions(
                        context, user_profile.command_history, result_data
                    )
                    next_actions.extend([
                        NextAction(
                            command=s.command or "help",
                            description=s.text,
                            confidence=s.confidence,
                            category=s.category
                        ) for s in smart_suggestions if s.command
                    ])
                
                # Calculate metrics
                duration_ms = max(1, int((time.perf_counter() - start_time) * 1000))
                metrics = ExecutionMetrics(
                    duration_ms=duration_ms,
                    tokens_used=0,  # Would be populated by actual handlers
                    cost_usd=0.0,
                    cpu_time_ms=0,
                    memory_peak_mb=0
                )
                
                # Record telemetry
                if self.telemetry_collector:
                    await self.telemetry_collector.record_command_execution(
                        command, result, metrics, context
                    )
                
                # Update command statistics
                self._update_command_stats(command.command_type, duration_ms, True)
                
                logger.info(f"Command {command.command_type} completed successfully in {duration_ms}ms")
                return result, metrics, next_actions
                
        except Exception as e:
            # Handle and categorize errors
            error_result, metrics = await self._handle_execution_error(
                e, command, context, start_time
            )
            
            # Record error telemetry
            if self.telemetry_collector:
                await self.telemetry_collector.record_error(e, command, context)
            
            self._update_command_stats(command.command_type, metrics.duration_ms, False)
            
            return error_result, metrics, []
        
        finally:
            self._active_commands.discard(command.request_id)
    
    async def _execute_pipeline(
        self, 
        command: CommandRequest, 
        context: ExecutionContext
    ) -> Tuple[Any, List[Suggestion]]:
        """Execute the full command pipeline with hooks."""
        
        # Pre-execution hooks
        for hook in self._pre_hooks:
            if hook.enabled:
                try:
                    await hook.handler(command, context)
                except Exception as e:
                    logger.warning(f"Pre-hook {hook.name} failed: {e}")
        
        # Find and validate handler
        handler = self._handlers.get(command.command_type)
        if not handler:
            raise CommandNotFoundError(f"Command not found: {command.command_type}")
        
        # Validate command
        try:
            await handler.validate(command, context)
        except Exception as e:
            if isinstance(e, (PermissionDeniedError, ValidationFailedError)):
                raise
            raise ValidationFailedError(f"Command validation failed: {e}")
        
        # Execute command
        result_data = await handler.execute(command, context)
        
        # Generate suggestions
        suggestions = await handler.generate_suggestions(command, result_data, context)
        
        # Post-execution hooks
        for hook in self._post_hooks:
            if hook.enabled:
                try:
                    await hook.handler(command, context, result_data)
                except Exception as e:
                    logger.warning(f"Post-hook {hook.name} failed: {e}")
        
        return result_data, suggestions
    
    async def _handle_execution_error(
        self,
        error: Exception,
        command: CommandRequest, 
        context: ExecutionContext,
        start_time: float
    ) -> Tuple[CommandResult, ExecutionMetrics]:
        """Handle and categorize execution errors."""
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Categorize error and create appropriate response
        if isinstance(error, asyncio.TimeoutError):
            status = CommandStatus.TIMEOUT
            error_code = "ExecutionTimeout"
            message = f"Command timed out after {duration_ms}ms"
        elif isinstance(error, PermissionDeniedError):
            status = CommandStatus.FAILED
            error_code = "PermissionDenied" 
            message = str(error)
        elif isinstance(error, ResourceExhaustedError):
            status = CommandStatus.FAILED
            error_code = "ResourceExhausted"
            message = str(error)
        elif isinstance(error, ValidationFailedError):
            status = CommandStatus.FAILED
            error_code = "ValidationFailed"
            message = str(error)
        elif isinstance(error, CommandNotFoundError):
            status = CommandStatus.FAILED
            error_code = "CommandNotFound"
            message = str(error)
        else:
            status = CommandStatus.FAILED
            error_code = "InternalError"
            message = f"Internal error: {str(error)}"
        
        # Generate recovery suggestions
        recovery_suggestions = self._generate_recovery_suggestions(error_code, command, context)
        
        command_error = CommandError(
            error_code=error_code,
            message=message,
            recovery_suggestions=recovery_suggestions
        )
        
        result = CommandResult(
            request_id=command.request_id,
            success=False,
            status=status,
            data=None,
            errors=[command_error]
        )
        
        metrics = ExecutionMetrics(
            duration_ms=duration_ms,
            tokens_used=0,
            cost_usd=0.0
        )
        
        logger.error(f"Command {command.command_type} failed: {message}")
        return result, metrics
    
    def _generate_recovery_suggestions(
        self, 
        error_code: str,
        command: CommandRequest,
        context: ExecutionContext
    ) -> List[str]:
        """Generate contextual recovery suggestions for errors."""
        
        suggestions = []
        
        if error_code == "CommandNotFound":
            suggestions.extend([
                f"Use 'help' to see available commands",
                f"Check spelling of command: {command.command_type}",
                "Try 'search <partial_name>' to find similar commands"
            ])
        elif error_code == "PermissionDenied":
            suggestions.extend([
                "Contact administrator for required permissions",
                "Try using 'whoami' to check current permissions",
                f"Consider switching to expert mode if you have access"
            ])
        elif error_code == "ValidationFailed":
            suggestions.extend([
                f"Use 'help {command.command_type}' for usage information",
                "Check command arguments and try again",
                "Use '--validate' flag to check arguments without executing"
            ])
        elif error_code == "ExecutionTimeout":
            suggestions.extend([
                "Try breaking the task into smaller parts",
                "Use '--timeout <seconds>' to allow more time",
                "Check system resources and try again later"
            ])
        elif error_code == "ResourceExhausted":
            suggestions.extend([
                "Close other applications to free resources",
                "Try running the command with '--low-memory' flag",
                "Consider running during off-peak hours"
            ])
        
        return suggestions
    
    def _update_command_stats(self, command_type: str, duration_ms: int, success: bool) -> None:
        """Update internal command execution statistics."""
        if command_type not in self._command_stats:
            self._command_stats[command_type] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_duration_ms": 0,
                "avg_duration_ms": 0,
                "last_execution": None
            }
        
        stats = self._command_stats[command_type]
        stats["total_executions"] += 1
        if success:
            stats["successful_executions"] += 1
        stats["total_duration_ms"] += duration_ms
        stats["avg_duration_ms"] = stats["total_duration_ms"] // stats["total_executions"]
        stats["last_execution"] = datetime.now(timezone.utc)
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status information."""
        uptime_seconds = (datetime.now(timezone.utc) - self._startup_time).total_seconds()
        
        return {
            "status": "healthy",
            "uptime_seconds": int(uptime_seconds),
            "registered_handlers": len(self._handlers),
            "active_commands": len(self._active_commands),
            "pre_hooks": len(self._pre_hooks),
            "post_hooks": len(self._post_hooks),
            "intelligence_provider": self.intelligence_provider is not None,
            "telemetry_collector": self.telemetry_collector is not None,
            "cross_modal_coordinator": self.cross_modal_coordinator is not None,
            "command_stats": dict(self._command_stats)
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the command engine."""
        logger.info("Shutting down command engine...")
        
        # Wait for active commands to complete (with timeout)
        if self._active_commands:
            logger.info(f"Waiting for {len(self._active_commands)} active commands to complete...")
            timeout = 30  # 30 second shutdown timeout
            
            while self._active_commands and timeout > 0:
                await asyncio.sleep(0.1)
                timeout -= 0.1
            
            if self._active_commands:
                logger.warning(f"Forced shutdown with {len(self._active_commands)} active commands")
        
        logger.info("Command engine shutdown complete")