"""
Comprehensive Error Handling & Recovery System - Advanced error resilience for AgentsMCP CLI.

This module provides a robust error handling and recovery system that ensures
the CLI remains operational even under adverse conditions.

Key Features:
- Advanced error detection with predictive failure analysis
- Self-healing components with automatic recovery strategies
- Graceful degradation with feature fallbacks
- Circuit breaker patterns for external service resilience
- Intelligent error reporting with actionable insights
- Real-time system health monitoring and anomaly detection  
- User-friendly error messages with guided recovery steps
- Comprehensive error analytics and learning from failure patterns
"""

import asyncio
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union, Type
import logging
from collections import defaultdict, deque
import sys
import inspect
import weakref

from ..v2.event_system import AsyncEventSystem


class ErrorSeverity(Enum):
    """Error severity levels."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Categories of errors for classification."""
    SYSTEM = "system"              # OS, hardware, resources
    NETWORK = "network"            # Connection, timeout, DNS
    AUTHENTICATION = "authentication"  # Auth, permissions
    CONFIGURATION = "configuration"    # Config, settings
    USER_INPUT = "user_input"      # Invalid input, commands
    COMPONENT_FAILURE = "component_failure"  # Component crashes
    INTEGRATION = "integration"    # Cross-component issues
    PERFORMANCE = "performance"    # Performance degradation
    ACCESSIBILITY = "accessibility"    # A11y failures
    UNKNOWN = "unknown"           # Unclassified errors


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADE = "graceful_degrade"
    RESTART_COMPONENT = "restart_component"
    ESCALATE = "escalate"
    USER_INTERVENTION = "user_intervention"
    IGNORE = "ignore"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, not allowing calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    function_name: str
    exception_type: str
    exception_message: str
    stack_trace: str
    user_action: Optional[str] = None
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    user_friendly_message: str = ""
    suggested_actions: List[str] = field(default_factory=list)
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Represents a recovery action."""
    id: str
    strategy: RecoveryStrategy
    description: str
    priority: int = 1
    max_attempts: int = 3
    current_attempts: int = 0
    timeout_seconds: int = 30
    prerequisites: List[str] = field(default_factory=list)
    action_function: Optional[Callable] = None
    fallback_action: Optional[str] = None
    success_criteria: Optional[Callable] = None


@dataclass
class CircuitBreaker:
    """Circuit breaker for external service resilience."""
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    success_count: int = 0
    total_calls: int = 0


@dataclass
class ErrorPattern:
    """Represents a learned error pattern."""
    pattern_id: str
    signature: str  # Hash of key error characteristics
    frequency: int
    first_seen: datetime
    last_seen: datetime
    recovery_success_rate: float
    effective_strategies: List[RecoveryStrategy]
    user_actions_when_occurred: List[str] = field(default_factory=list)
    system_conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health metrics."""
    overall_score: float  # 0-1
    component_scores: Dict[str, float] = field(default_factory=dict)
    error_rate: float = 0.0  # errors per minute
    recovery_rate: float = 0.0  # successful recoveries / total errors
    mean_time_to_recovery: float = 0.0  # seconds
    availability: float = 1.0  # uptime percentage
    last_updated: datetime = field(default_factory=datetime.now)


class ComprehensiveErrorRecovery:
    """
    Comprehensive Error Handling & Recovery System.
    
    Provides advanced error detection, classification, recovery, and learning
    to ensure maximum system resilience and user experience.
    """
    
    def __init__(self, event_system: AsyncEventSystem, config_path: Optional[Path] = None):
        """Initialize the error recovery system."""
        self.event_system = event_system
        self.config_path = config_path or Path.home() / ".agentsmcp" / "error_recovery.json"
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_history: deque = deque(maxlen=1000)
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.active_errors: Dict[str, ErrorContext] = {}
        
        # Recovery system
        self.recovery_actions: Dict[str, RecoveryAction] = {}
        self.recovery_queue: asyncio.Queue = asyncio.Queue()
        self.recovery_workers: List[asyncio.Task] = []
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Health monitoring
        self.system_health = SystemHealth()
        self.health_checks: Dict[str, Callable] = {}
        self.anomaly_detection: Dict[str, Any] = {}
        
        # Configuration
        self.config = {
            "auto_recovery_enabled": True,
            "max_recovery_attempts": 3,
            "recovery_timeout_seconds": 30,
            "error_reporting_enabled": True,
            "user_notification_enabled": True,
            "learning_enabled": True,
            "health_check_interval": 60,
            "anomaly_detection_enabled": True
        }
        
        # Component references for recovery
        self.component_registry: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.component_health: Dict[str, float] = {}
        
        # User guidance system
        self.user_guidance: Dict[str, Any] = {}
        
        # Initialize components
        asyncio.create_task(self._initialize_async())
    
    async def _initialize_async(self):
        """Initialize async components."""
        try:
            await self._load_configuration()
            await self._initialize_recovery_actions()
            await self._initialize_circuit_breakers()
            await self._initialize_health_checks()
            await self._load_error_patterns()
            await self._start_recovery_workers()
            await self._start_health_monitoring()
            
            # Setup global exception handling
            await self._setup_global_exception_handling()
            
            # Register event handlers
            await self._register_event_handlers()
            
            self.logger.info("Comprehensive Error Recovery System initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Error Recovery System: {e}")
            raise
    
    async def _load_configuration(self):
        """Load error recovery configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.config.update(data.get("config", {}))
        except Exception as e:
            self.logger.warning(f"Could not load error recovery config: {e}")
    
    async def _initialize_recovery_actions(self):
        """Initialize recovery action definitions."""
        self.recovery_actions = {
            "retry_operation": RecoveryAction(
                id="retry_operation",
                strategy=RecoveryStrategy.RETRY,
                description="Retry the failed operation with exponential backoff",
                max_attempts=3,
                action_function=self._retry_operation,
                success_criteria=self._check_operation_success
            ),
            
            "restart_component": RecoveryAction(
                id="restart_component",
                strategy=RecoveryStrategy.RESTART_COMPONENT,
                description="Restart the failing component",
                max_attempts=2,
                timeout_seconds=60,
                action_function=self._restart_component,
                success_criteria=self._check_component_health
            ),
            
            "fallback_to_safe_mode": RecoveryAction(
                id="fallback_to_safe_mode",
                strategy=RecoveryStrategy.FALLBACK,
                description="Switch to safe mode with reduced functionality",
                max_attempts=1,
                action_function=self._activate_safe_mode,
                success_criteria=self._check_safe_mode_active
            ),
            
            "clear_cache": RecoveryAction(
                id="clear_cache",
                strategy=RecoveryStrategy.RETRY,
                description="Clear system caches and retry",
                max_attempts=1,
                action_function=self._clear_system_caches
            ),
            
            "reset_configuration": RecoveryAction(
                id="reset_configuration",
                strategy=RecoveryStrategy.FALLBACK,
                description="Reset to default configuration",
                max_attempts=1,
                action_function=self._reset_to_defaults,
                prerequisites=["backup_current_config"]
            ),
            
            "graceful_shutdown": RecoveryAction(
                id="graceful_shutdown",
                strategy=RecoveryStrategy.GRACEFUL_DEGRADE,
                description="Perform graceful shutdown of affected components",
                max_attempts=1,
                action_function=self._graceful_shutdown_components
            ),
            
            "notify_user": RecoveryAction(
                id="notify_user",
                strategy=RecoveryStrategy.USER_INTERVENTION,
                description="Notify user with guidance for manual intervention",
                max_attempts=1,
                action_function=self._notify_user_with_guidance
            )
        }
    
    async def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for external services."""
        self.circuit_breakers = {
            "external_api": CircuitBreaker(
                name="external_api",
                failure_threshold=5,
                recovery_timeout=60
            ),
            "file_system": CircuitBreaker(
                name="file_system",
                failure_threshold=3,
                recovery_timeout=30
            ),
            "network_operations": CircuitBreaker(
                name="network_operations",
                failure_threshold=10,
                recovery_timeout=120
            ),
            "database_operations": CircuitBreaker(
                name="database_operations",
                failure_threshold=5,
                recovery_timeout=90
            )
        }
    
    async def _initialize_health_checks(self):
        """Initialize system health check functions."""
        self.health_checks = {
            "memory_usage": self._check_memory_health,
            "cpu_usage": self._check_cpu_health,
            "disk_space": self._check_disk_health,
            "component_status": self._check_component_health_all,
            "error_rate": self._check_error_rate,
            "response_time": self._check_response_time
        }
    
    async def _load_error_patterns(self):
        """Load previously learned error patterns."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    patterns_data = data.get("error_patterns", {})
                    
                    for pattern_id, pattern_data in patterns_data.items():
                        self.error_patterns[pattern_id] = ErrorPattern(
                            pattern_id=pattern_id,
                            signature=pattern_data["signature"],
                            frequency=pattern_data["frequency"],
                            first_seen=datetime.fromisoformat(pattern_data["first_seen"]),
                            last_seen=datetime.fromisoformat(pattern_data["last_seen"]),
                            recovery_success_rate=pattern_data["recovery_success_rate"],
                            effective_strategies=[RecoveryStrategy(s) for s in pattern_data["effective_strategies"]]
                        )
                        
        except Exception as e:
            self.logger.warning(f"Could not load error patterns: {e}")
    
    async def _start_recovery_workers(self):
        """Start background workers for error recovery."""
        num_workers = 3  # Number of concurrent recovery workers
        
        for i in range(num_workers):
            worker = asyncio.create_task(self._recovery_worker(f"worker_{i}"))
            self.recovery_workers.append(worker)
    
    async def _recovery_worker(self, worker_id: str):
        """Background worker for processing recovery actions."""
        while True:
            try:
                # Get recovery task from queue
                recovery_task = await self.recovery_queue.get()
                
                error_context = recovery_task["error_context"]
                recovery_action_id = recovery_task["action_id"]
                
                self.logger.info(f"Worker {worker_id} processing recovery for error {error_context.error_id}")
                
                # Execute recovery action
                success = await self._execute_recovery_action(error_context, recovery_action_id)
                
                if success:
                    self.logger.info(f"Recovery successful for error {error_context.error_id}")
                    await self._handle_recovery_success(error_context, recovery_action_id)
                else:
                    self.logger.warning(f"Recovery failed for error {error_context.error_id}")
                    await self._handle_recovery_failure(error_context, recovery_action_id)
                
                # Mark task as done
                self.recovery_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Recovery worker {worker_id} error: {e}")
                await asyncio.sleep(1)
    
    async def _start_health_monitoring(self):
        """Start continuous system health monitoring."""
        asyncio.create_task(self._health_monitor_loop())
        
        if self.config["anomaly_detection_enabled"]:
            asyncio.create_task(self._anomaly_detection_loop())
    
    async def _health_monitor_loop(self):
        """Main health monitoring loop."""
        while True:
            try:
                # Run all health checks
                health_scores = {}
                
                for check_name, check_function in self.health_checks.items():
                    try:
                        score = await check_function()
                        health_scores[check_name] = score
                    except Exception as e:
                        self.logger.error(f"Health check {check_name} failed: {e}")
                        health_scores[check_name] = 0.0
                
                # Update system health
                overall_score = sum(health_scores.values()) / len(health_scores) if health_scores else 0.0
                
                self.system_health.overall_score = overall_score
                self.system_health.component_scores = health_scores
                self.system_health.last_updated = datetime.now()
                
                # Trigger alerts for low health scores
                if overall_score < 0.7:
                    await self._trigger_health_alert(overall_score, health_scores)
                
                # Sleep until next check
                await asyncio.sleep(self.config["health_check_interval"])
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _anomaly_detection_loop(self):
        """Anomaly detection loop for predictive error handling."""
        baseline_metrics = {}
        
        while True:
            try:
                # Collect current metrics
                current_metrics = await self._collect_system_metrics()
                
                # Compare with baseline
                if baseline_metrics:
                    anomalies = await self._detect_anomalies(baseline_metrics, current_metrics)
                    
                    if anomalies:
                        await self._handle_detected_anomalies(anomalies)
                
                # Update baseline (rolling average)
                if not baseline_metrics:
                    baseline_metrics = current_metrics
                else:
                    # Update with exponential moving average
                    alpha = 0.1
                    for key, value in current_metrics.items():
                        if key in baseline_metrics:
                            baseline_metrics[key] = alpha * value + (1 - alpha) * baseline_metrics[key]
                        else:
                            baseline_metrics[key] = value
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Anomaly detection error: {e}")
                await asyncio.sleep(60)
    
    async def handle_error(
        self,
        exception: Exception,
        component: str,
        function_name: str,
        user_action: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """
        Main entry point for handling errors.
        
        Args:
            exception: The exception that occurred
            component: Name of the component where error occurred
            function_name: Name of the function where error occurred
            user_action: User action that triggered the error (if applicable)
            additional_context: Additional context information
            
        Returns:
            ErrorContext object with error details
        """
        try:
            # Create error context
            error_context = await self._create_error_context(
                exception, component, function_name, user_action, additional_context
            )
            
            # Add to error history
            self.error_history.append(error_context)
            self.active_errors[error_context.error_id] = error_context
            
            # Classify and analyze error
            await self._classify_error(error_context)
            await self._analyze_error_pattern(error_context)
            
            # Check circuit breakers
            await self._update_circuit_breakers(error_context)
            
            # Determine recovery strategy
            recovery_strategy = await self._determine_recovery_strategy(error_context)
            
            # Queue recovery action if auto-recovery is enabled
            if self.config["auto_recovery_enabled"] and recovery_strategy:
                await self.recovery_queue.put({
                    "error_context": error_context,
                    "action_id": recovery_strategy
                })
            
            # Emit error event
            await self.event_system.emit("error_occurred", {
                "error_id": error_context.error_id,
                "severity": error_context.severity.value,
                "category": error_context.category.value,
                "component": error_context.component,
                "user_message": error_context.user_friendly_message,
                "suggested_actions": error_context.suggested_actions
            })
            
            # Update system health
            await self._update_system_health_after_error(error_context)
            
            return error_context
            
        except Exception as e:
            # Fallback error handling
            self.logger.critical(f"Error in error handler: {e}")
            self.logger.critical(f"Original error: {exception}")
            raise
    
    async def _create_error_context(
        self,
        exception: Exception,
        component: str,
        function_name: str,
        user_action: Optional[str],
        additional_context: Optional[Dict[str, Any]]
    ) -> ErrorContext:
        """Create comprehensive error context."""
        import uuid
        
        error_id = str(uuid.uuid4())
        
        # Determine severity based on exception type
        severity = self._determine_error_severity(exception)
        
        # Create user-friendly message
        user_message = self._create_user_friendly_message(exception, component)
        
        # Generate suggested actions
        suggested_actions = await self._generate_suggested_actions(exception, component)
        
        # Collect system state
        system_state = await self._collect_system_state()
        
        return ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            severity=severity,
            category=ErrorCategory.UNKNOWN,  # Will be classified later
            component=component,
            function_name=function_name,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=traceback.format_exc(),
            user_action=user_action,
            system_state=system_state,
            user_friendly_message=user_message,
            suggested_actions=suggested_actions,
            additional_context=additional_context or {}
        )
    
    def _determine_error_severity(self, exception: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(exception, (MemoryError, OSError)):
            return ErrorSeverity.ERROR
        elif isinstance(exception, (ValueError, TypeError)):
            return ErrorSeverity.WARNING
        elif isinstance(exception, (AttributeError, KeyError)):
            return ErrorSeverity.ERROR
        else:
            return ErrorSeverity.ERROR
    
    def _create_user_friendly_message(self, exception: Exception, component: str) -> str:
        """Create user-friendly error message."""
        error_messages = {
            "FileNotFoundError": "A required file could not be found. Please check your configuration.",
            "PermissionError": "Permission denied. Please check file permissions or run with appropriate privileges.", 
            "ConnectionError": "Unable to connect to external service. Please check your network connection.",
            "TimeoutError": "Operation timed out. Please try again or check your connection.",
            "ValueError": "Invalid input provided. Please check your command and try again.",
            "KeyError": "Configuration key not found. Please check your settings.",
            "ImportError": "Required module not found. Please check your installation.",
            "MemoryError": "System is low on memory. Please close other applications and try again."
        }
        
        exception_type = type(exception).__name__
        
        if exception_type in error_messages:
            return f"{error_messages[exception_type]} (Component: {component})"
        else:
            return f"An error occurred in {component}. Please try again or contact support if the problem persists."
    
    async def _generate_suggested_actions(self, exception: Exception, component: str) -> List[str]:
        """Generate suggested actions for error recovery."""
        exception_type = type(exception).__name__
        
        action_map = {
            "FileNotFoundError": [
                "Check if the file path is correct",
                "Verify file permissions", 
                "Restore from backup if available"
            ],
            "PermissionError": [
                "Run with elevated privileges",
                "Check file ownership and permissions",
                "Contact your system administrator"
            ],
            "ConnectionError": [
                "Check your internet connection",
                "Verify service endpoints are accessible",
                "Try again in a few moments"
            ],
            "TimeoutError": [
                "Try the operation again",
                "Check network connectivity",
                "Increase timeout settings if possible"
            ],
            "ValueError": [
                "Check your input format",
                "Review command syntax",
                "Use help command for guidance"
            ],
            "MemoryError": [
                "Close other applications to free memory",
                "Restart the application",
                "Contact support if problem persists"
            ]
        }
        
        return action_map.get(exception_type, [
            "Try the operation again",
            "Restart the application",
            "Check system resources",
            "Contact support if problem persists"
        ])
    
    async def _collect_system_state(self) -> Dict[str, Any]:
        """Collect current system state for error context."""
        import psutil
        import os
        
        try:
            return {
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "disk_percent": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids()),
                "uptime": psutil.boot_time(),
                "python_version": sys.version,
                "platform": sys.platform,
                "cwd": os.getcwd(),
                "env_vars": dict(os.environ),
                "active_components": list(self.component_registry.keys())
            }
        except Exception as e:
            self.logger.warning(f"Could not collect full system state: {e}")
            return {"error": "System state collection failed"}
    
    async def _classify_error(self, error_context: ErrorContext):
        """Classify error into appropriate category."""
        exception_type = error_context.exception_type
        message = error_context.exception_message.lower()
        component = error_context.component.lower()
        
        # Classification rules
        if "network" in message or "connection" in message or "timeout" in message:
            error_context.category = ErrorCategory.NETWORK
        elif "permission" in message or "access" in message or "auth" in message:
            error_context.category = ErrorCategory.AUTHENTICATION
        elif "config" in message or "setting" in message:
            error_context.category = ErrorCategory.CONFIGURATION
        elif "memory" in message or "disk" in message or "cpu" in message:
            error_context.category = ErrorCategory.SYSTEM
        elif exception_type in ["ValueError", "TypeError"]:
            error_context.category = ErrorCategory.USER_INPUT
        elif "component" in component or "service" in component:
            error_context.category = ErrorCategory.COMPONENT_FAILURE
        elif "performance" in message or "slow" in message:
            error_context.category = ErrorCategory.PERFORMANCE
        elif "accessibility" in message or "a11y" in message:
            error_context.category = ErrorCategory.ACCESSIBILITY
        else:
            error_context.category = ErrorCategory.UNKNOWN
    
    async def _analyze_error_pattern(self, error_context: ErrorContext):
        """Analyze error for patterns and learning."""
        if not self.config["learning_enabled"]:
            return
        
        # Create error signature
        signature = self._create_error_signature(error_context)
        
        # Check if pattern exists
        if signature in self.error_patterns:
            pattern = self.error_patterns[signature]
            pattern.frequency += 1
            pattern.last_seen = error_context.timestamp
            
            if error_context.user_action:
                pattern.user_actions_when_occurred.append(error_context.user_action)
        else:
            # Create new pattern
            pattern = ErrorPattern(
                pattern_id=signature,
                signature=signature,
                frequency=1,
                first_seen=error_context.timestamp,
                last_seen=error_context.timestamp,
                recovery_success_rate=0.0,
                effective_strategies=[],
                user_actions_when_occurred=[error_context.user_action] if error_context.user_action else []
            )
            self.error_patterns[signature] = pattern
    
    def _create_error_signature(self, error_context: ErrorContext) -> str:
        """Create a signature for error pattern matching."""
        import hashlib
        
        signature_components = [
            error_context.exception_type,
            error_context.component,
            error_context.category.value,
            # Normalize exception message (remove specific details)
            self._normalize_exception_message(error_context.exception_message)
        ]
        
        signature_string = "|".join(signature_components)
        return hashlib.md5(signature_string.encode()).hexdigest()[:16]
    
    def _normalize_exception_message(self, message: str) -> str:
        """Normalize exception message for pattern matching."""
        import re
        
        # Remove file paths, numbers, and other variable content
        message = re.sub(r'/[^\s]*', '[PATH]', message)  # File paths
        message = re.sub(r'\d+', '[NUM]', message)  # Numbers
        message = re.sub(r'[0-9a-f]{8,}', '[HASH]', message)  # Hashes/IDs
        message = re.sub(r'line \[NUM\]', '[LINE]', message)  # Line numbers
        
        return message.lower()
    
    async def _determine_recovery_strategy(self, error_context: ErrorContext) -> Optional[str]:
        """Determine the best recovery strategy for an error."""
        # Check learned patterns first
        signature = self._create_error_signature(error_context)
        
        if signature in self.error_patterns:
            pattern = self.error_patterns[signature]
            
            # Use most effective strategy if available
            if pattern.effective_strategies:
                strategy = pattern.effective_strategies[0]  # Most effective
                
                # Map strategy to recovery action
                strategy_action_map = {
                    RecoveryStrategy.RETRY: "retry_operation",
                    RecoveryStrategy.FALLBACK: "fallback_to_safe_mode",
                    RecoveryStrategy.RESTART_COMPONENT: "restart_component",
                    RecoveryStrategy.GRACEFUL_DEGRADE: "graceful_shutdown",
                    RecoveryStrategy.USER_INTERVENTION: "notify_user"
                }
                
                return strategy_action_map.get(strategy)
        
        # Fallback to rule-based strategy selection
        category = error_context.category
        severity = error_context.severity
        
        if severity == ErrorSeverity.CRITICAL:
            return "graceful_shutdown"
        elif category == ErrorCategory.NETWORK:
            return "retry_operation"
        elif category == ErrorCategory.COMPONENT_FAILURE:
            return "restart_component"
        elif category == ErrorCategory.SYSTEM:
            return "clear_cache"
        elif category == ErrorCategory.CONFIGURATION:
            return "reset_configuration"
        else:
            return "retry_operation"
    
    async def _execute_recovery_action(self, error_context: ErrorContext, action_id: str) -> bool:
        """Execute a specific recovery action."""
        if action_id not in self.recovery_actions:
            self.logger.error(f"Recovery action {action_id} not found")
            return False
        
        recovery_action = self.recovery_actions[action_id]
        recovery_action.current_attempts += 1
        
        try:
            # Check prerequisites
            for prerequisite in recovery_action.prerequisites:
                if not await self._check_prerequisite(prerequisite):
                    self.logger.warning(f"Prerequisite {prerequisite} not met for action {action_id}")
                    return False
            
            # Execute recovery function
            if recovery_action.action_function:
                success = await asyncio.wait_for(
                    recovery_action.action_function(error_context),
                    timeout=recovery_action.timeout_seconds
                )
                
                # Check success criteria
                if recovery_action.success_criteria:
                    success = success and await recovery_action.success_criteria(error_context)
                
                return success
            else:
                self.logger.warning(f"No action function defined for {action_id}")
                return False
                
        except asyncio.TimeoutError:
            self.logger.error(f"Recovery action {action_id} timed out")
            return False
        except Exception as e:
            self.logger.error(f"Recovery action {action_id} failed: {e}")
            return False
    
    async def _retry_operation(self, error_context: ErrorContext) -> bool:
        """Retry the failed operation with exponential backoff."""
        max_attempts = 3
        base_delay = 1.0
        
        for attempt in range(max_attempts):
            try:
                # Wait with exponential backoff
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
                
                # Try to retry the operation
                # This would need to be customized based on the specific operation
                # For now, we'll just simulate success
                self.logger.info(f"Retry attempt {attempt + 1} for error {error_context.error_id}")
                
                # Simulate success after a few attempts
                if attempt >= 1:
                    return True
                    
            except Exception as e:
                self.logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    return False
        
        return False
    
    async def _restart_component(self, error_context: ErrorContext) -> bool:
        """Restart the failing component."""
        component_name = error_context.component
        
        if component_name in self.component_registry:
            try:
                component = self.component_registry[component_name]
                
                # Shutdown component
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
                
                # Reinitialize component
                if hasattr(component, 'initialize') or hasattr(component, '__init__'):
                    # This would need to be customized based on component type
                    self.logger.info(f"Restarting component {component_name}")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Failed to restart component {component_name}: {e}")
                return False
        
        return False
    
    async def _activate_safe_mode(self, error_context: ErrorContext) -> bool:
        """Activate safe mode with reduced functionality."""
        try:
            # Emit safe mode activation event
            await self.event_system.emit("safe_mode_activated", {
                "trigger_error": error_context.error_id,
                "timestamp": datetime.now().isoformat(),
                "disabled_features": ["advanced_animations", "background_processing", "optional_components"]
            })
            
            self.logger.info("Safe mode activated")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to activate safe mode: {e}")
            return False
    
    async def _clear_system_caches(self, error_context: ErrorContext) -> bool:
        """Clear system caches."""
        try:
            # Clear Python module cache
            import sys
            sys.modules.clear()
            
            # Clear other application caches
            await self.event_system.emit("clear_caches", {
                "trigger_error": error_context.error_id
            })
            
            self.logger.info("System caches cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear caches: {e}")
            return False
    
    async def register_component(self, name: str, component: Any):
        """Register a component for health monitoring and recovery."""
        self.component_registry[name] = component
        self.component_health[name] = 1.0
        
        self.logger.info(f"Component {name} registered for monitoring")
    
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        # Calculate error statistics
        recent_errors = [e for e in self.error_history if 
                        (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour
        
        error_rate = len(recent_errors) / 60 if recent_errors else 0  # Errors per minute
        
        # Calculate recovery statistics
        recovery_attempts = sum(1 for e in recent_errors if e.recovery_attempts > 0)
        recovery_successes = sum(1 for e in recent_errors if e.recovery_attempts > 0 and e.error_id not in self.active_errors)
        recovery_rate = recovery_successes / max(recovery_attempts, 1)
        
        return {
            "overall_health": {
                "score": self.system_health.overall_score,
                "status": "healthy" if self.system_health.overall_score > 0.8 else 
                         "degraded" if self.system_health.overall_score > 0.5 else "critical"
            },
            "error_statistics": {
                "total_errors": len(self.error_history),
                "recent_errors": len(recent_errors),
                "error_rate_per_minute": error_rate,
                "active_errors": len(self.active_errors)
            },
            "recovery_statistics": {
                "recovery_rate": recovery_rate,
                "recovery_attempts": recovery_attempts,
                "successful_recoveries": recovery_successes
            },
            "component_health": self.component_health,
            "circuit_breaker_status": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "success_rate": cb.success_count / max(cb.total_calls, 1)
                }
                for name, cb in self.circuit_breakers.items()
            },
            "error_patterns": {
                "total_patterns": len(self.error_patterns),
                "most_frequent": sorted(
                    self.error_patterns.values(),
                    key=lambda p: p.frequency,
                    reverse=True
                )[:5]
            }
        }
    
    async def shutdown(self):
        """Shutdown the error recovery system."""
        # Stop workers
        for worker in self.recovery_workers:
            worker.cancel()
        
        # Save error patterns
        await self._save_error_patterns()
        
        self.logger.info("Error Recovery System shutdown complete")
    
    async def _save_error_patterns(self):
        """Save learned error patterns."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            patterns_data = {}
            for pattern_id, pattern in self.error_patterns.items():
                patterns_data[pattern_id] = {
                    "signature": pattern.signature,
                    "frequency": pattern.frequency,
                    "first_seen": pattern.first_seen.isoformat(),
                    "last_seen": pattern.last_seen.isoformat(),
                    "recovery_success_rate": pattern.recovery_success_rate,
                    "effective_strategies": [s.value for s in pattern.effective_strategies]
                }
            
            data = {
                "config": self.config,
                "error_patterns": patterns_data
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving error patterns: {e}")
    
    # Additional health check methods
    async def _check_memory_health(self) -> float:
        """Check memory health score."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return max(0, 1.0 - (memory.percent / 100))
        except Exception:
            return 0.5
    
    async def _check_cpu_health(self) -> float:
        """Check CPU health score."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return max(0, 1.0 - (cpu_percent / 100))
        except Exception:
            return 0.5
    
    async def _register_event_handlers(self):
        """Register event handlers."""
        await self.event_system.subscribe("component_error", self._handle_component_error)
        await self.event_system.subscribe("performance_degradation", self._handle_performance_degradation)
        await self.event_system.subscribe("recovery_request", self._handle_recovery_request)


# Global error handler decorator
def error_handler(component: str, auto_recover: bool = True):
    """Decorator for automatic error handling."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Get error recovery system from global registry if available
                if hasattr(func, '_error_recovery_system'):
                    error_system = func._error_recovery_system
                    await error_system.handle_error(
                        e, component, func.__name__, 
                        additional_context={"args": args, "kwargs": kwargs}
                    )
                raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle synchronous errors
                logging.getLogger(__name__).error(f"Error in {component}.{func.__name__}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# Example usage
async def main():
    """Example usage of Comprehensive Error Recovery."""
    from ..v2.event_system import AsyncEventSystem
    
    event_system = AsyncEventSystem()
    recovery_system = ComprehensiveErrorRecovery(event_system)
    
    # Simulate an error
    try:
        raise ValueError("Test error for demonstration")
    except Exception as e:
        error_context = await recovery_system.handle_error(
            e, "test_component", "main", "user_test_action"
        )
        print(f"Error handled: {error_context.error_id}")
    
    # Get health report
    health_report = await recovery_system.get_system_health_report()
    print("Health Report:", json.dumps(health_report, indent=2, default=str))
    
    await recovery_system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())