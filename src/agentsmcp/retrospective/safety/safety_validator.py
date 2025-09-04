"""Safety validation logic for retrospective system improvements."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path

from ..data_models import ActionPoint, SystemicImprovement, ImplementationStatus, PriorityLevel
from .safety_config import SafetyConfig, SafetyLevel

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation results."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(str, Enum):
    """Categories of validation checks."""
    SYNTAX = "syntax"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"
    FUNCTIONALITY = "functionality"
    PERFORMANCE = "performance"
    SAFETY = "safety"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    issue_id: str
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    details: str = ""
    suggested_fix: Optional[str] = None
    blocking: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ValidationResult:
    """Result of safety validation."""
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def blocking_issues(self) -> List[ValidationIssue]:
        """Get issues that block implementation."""
        return [issue for issue in self.issues if issue.blocking]
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Get critical severity issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.CRITICAL]
    
    @property
    def error_issues(self) -> List[ValidationIssue]:
        """Get error severity issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
    
    def has_blocking_issues(self) -> bool:
        """Check if there are any blocking issues."""
        return len(self.blocking_issues) > 0
    
    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Get issues by category."""
        return [issue for issue in self.issues if issue.category == category]


class ValidationRule(ABC):
    """Abstract base class for validation rules."""
    
    def __init__(self, name: str, category: ValidationCategory, blocking: bool = False):
        self.name = name
        self.category = category
        self.blocking = blocking
    
    @abstractmethod
    async def validate(self, improvement: Union[ActionPoint, SystemicImprovement], 
                      context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate an improvement and return any issues."""
        pass


class CriticalRoleProtectionRule(ValidationRule):
    """Validation rule to protect critical system roles."""
    
    def __init__(self, critical_roles: Set[str]):
        super().__init__("critical_role_protection", ValidationCategory.SAFETY, blocking=True)
        self.critical_roles = critical_roles
    
    async def validate(self, improvement: Union[ActionPoint, SystemicImprovement], 
                      context: Dict[str, Any]) -> List[ValidationIssue]:
        issues = []
        
        # Check if improvement affects critical roles
        description = improvement.description.lower()
        title = improvement.title.lower()
        
        for role in self.critical_roles:
            if role in description or role in title:
                # Allow only specific safe operations
                safe_operations = [
                    "monitoring", "logging", "metrics", "reporting",
                    "optimization", "performance", "efficiency"
                ]
                
                if not any(op in description or op in title for op in safe_operations):
                    issues.append(ValidationIssue(
                        issue_id=f"critical_role_{role}_change",
                        category=self.category,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Modification to critical role '{role}' detected",
                        details=f"Changes to critical roles require manual approval: {improvement.title}",
                        blocking=True
                    ))
        
        return issues


class CoreFunctionalityRule(ValidationRule):
    """Validation rule to preserve core functionality."""
    
    def __init__(self):
        super().__init__("core_functionality_preservation", ValidationCategory.FUNCTIONALITY, blocking=True)
        self.protected_functions = {
            "task_execution", "agent_communication", "decision_making",
            "memory_management", "error_handling", "security"
        }
    
    async def validate(self, improvement: Union[ActionPoint, SystemicImprovement], 
                      context: Dict[str, Any]) -> List[ValidationIssue]:
        issues = []
        
        description = improvement.description.lower()
        
        # Check for dangerous operations
        dangerous_operations = [
            "delete", "remove", "disable", "stop", "kill", "terminate",
            "replace", "override", "bypass", "skip"
        ]
        
        for operation in dangerous_operations:
            if operation in description:
                for function in self.protected_functions:
                    if function in description:
                        issues.append(ValidationIssue(
                            issue_id=f"core_function_{function}_{operation}",
                            category=self.category,
                            severity=ValidationSeverity.ERROR,
                            message=f"Potentially dangerous operation on core function: {operation} {function}",
                            details=f"Review required: {improvement.title}",
                            blocking=True
                        ))
        
        return issues


class ConfigurationSyntaxRule(ValidationRule):
    """Validation rule for configuration syntax."""
    
    def __init__(self):
        super().__init__("configuration_syntax", ValidationCategory.SYNTAX, blocking=False)
    
    async def validate(self, improvement: Union[ActionPoint, SystemicImprovement], 
                      context: Dict[str, Any]) -> List[ValidationIssue]:
        issues = []
        
        # Check for configuration changes
        if "config" in improvement.description.lower() or "configuration" in improvement.description.lower():
            # Look for JSON in implementation steps
            for step in getattr(improvement, 'implementation_steps', []):
                if '{' in step and '}' in step:
                    try:
                        # Try to extract and validate JSON
                        json_match = re.search(r'\{.*\}', step, re.DOTALL)
                        if json_match:
                            json.loads(json_match.group())
                    except json.JSONDecodeError as e:
                        issues.append(ValidationIssue(
                            issue_id=f"json_syntax_error",
                            category=self.category,
                            severity=ValidationSeverity.ERROR,
                            message="Invalid JSON syntax in configuration",
                            details=f"JSON parsing error: {e}",
                            suggested_fix="Validate JSON syntax before implementation"
                        ))
        
        return issues


class SecurityConstraintsRule(ValidationRule):
    """Validation rule for security constraints."""
    
    def __init__(self):
        super().__init__("security_constraints", ValidationCategory.SECURITY, blocking=True)
        self.security_keywords = {
            "password", "token", "key", "secret", "credential", "auth",
            "permission", "access", "privilege", "security", "crypto"
        }
    
    async def validate(self, improvement: Union[ActionPoint, SystemicImprovement], 
                      context: Dict[str, Any]) -> List[ValidationIssue]:
        issues = []
        
        description = improvement.description.lower()
        
        # Check for security-related changes
        if any(keyword in description for keyword in self.security_keywords):
            # Security changes require additional validation
            issues.append(ValidationIssue(
                issue_id="security_change_detected",
                category=self.category,
                severity=ValidationSeverity.WARNING,
                message="Security-related change detected",
                details=f"Security review recommended: {improvement.title}",
                suggested_fix="Perform security impact assessment"
            ))
            
            # Check for credential exposure risks
            if "log" in description or "print" in description or "output" in description:
                issues.append(ValidationIssue(
                    issue_id="credential_exposure_risk",
                    category=self.category,
                    severity=ValidationSeverity.CRITICAL,
                    message="Risk of credential exposure in logging",
                    details="Security-related changes that involve logging may expose sensitive data",
                    blocking=True,
                    suggested_fix="Ensure sensitive data is properly masked in logs"
                ))
        
        return issues


class ResourceAvailabilityRule(ValidationRule):
    """Validation rule for resource availability."""
    
    def __init__(self):
        super().__init__("resource_availability", ValidationCategory.PERFORMANCE, blocking=False)
    
    async def validate(self, improvement: Union[ActionPoint, SystemicImprovement], 
                      context: Dict[str, Any]) -> List[ValidationIssue]:
        issues = []
        
        # Check estimated effort
        effort_hours = getattr(improvement, 'estimated_effort_hours', 0.0)
        if effort_hours > 8.0:  # More than one working day
            issues.append(ValidationIssue(
                issue_id="high_effort_improvement",
                category=self.category,
                severity=ValidationSeverity.WARNING,
                message=f"High-effort improvement detected: {effort_hours} hours",
                details="Consider breaking down into smaller increments",
                suggested_fix="Split into multiple smaller action points"
            ))
        
        # Check for resource-intensive operations
        description = improvement.description.lower()
        resource_intensive_terms = [
            "migration", "rebuild", "reindex", "recompile", "restructure"
        ]
        
        for term in resource_intensive_terms:
            if term in description:
                issues.append(ValidationIssue(
                    issue_id=f"resource_intensive_{term}",
                    category=self.category,
                    severity=ValidationSeverity.WARNING,
                    message=f"Resource-intensive operation detected: {term}",
                    details="Monitor system resources during implementation",
                    suggested_fix="Schedule during low-traffic periods"
                ))
        
        return issues


class SafetyValidator:
    """Main safety validator for retrospective improvements."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.rules: List[ValidationRule] = []
        self._setup_rules()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_rules(self):
        """Setup validation rules based on configuration."""
        if self.config.validation_rules_enabled.get("no_critical_role_changes", True):
            self.rules.append(
                CriticalRoleProtectionRule(self.config.thresholds.critical_roles)
            )
        
        if self.config.validation_rules_enabled.get("preserve_core_functionality", True):
            self.rules.append(CoreFunctionalityRule())
        
        if self.config.validation_rules_enabled.get("validate_configuration_syntax", True):
            self.rules.append(ConfigurationSyntaxRule())
        
        if self.config.validation_rules_enabled.get("validate_security_constraints", True):
            self.rules.append(SecurityConstraintsRule())
        
        if self.config.validation_rules_enabled.get("verify_resource_availability", True):
            self.rules.append(ResourceAvailabilityRule())
    
    async def validate_improvement(
        self, 
        improvement: Union[ActionPoint, SystemicImprovement],
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Validate a single improvement."""
        if context is None:
            context = {}
        
        start_time = datetime.now(timezone.utc)
        issues = []
        warnings = []
        
        try:
            # Skip validation in development if configured
            if self.config.should_skip_safety_checks():
                self.logger.info("Skipping safety validation in development mode")
                return ValidationResult(
                    passed=True,
                    warnings=["Safety validation skipped in development mode"]
                )
            
            # Run all validation rules
            for rule in self.rules:
                try:
                    rule_issues = await rule.validate(improvement, context)
                    issues.extend(rule_issues)
                    self.logger.debug(f"Rule {rule.name} found {len(rule_issues)} issues")
                except Exception as e:
                    self.logger.error(f"Validation rule {rule.name} failed: {e}")
                    issues.append(ValidationIssue(
                        issue_id=f"rule_failure_{rule.name}",
                        category=ValidationCategory.SAFETY,
                        severity=ValidationSeverity.WARNING,
                        message=f"Validation rule {rule.name} failed",
                        details=str(e)
                    ))
            
            # Additional validation based on safety level
            if self.config.get_effective_safety_level() == SafetyLevel.STRICT:
                issues.extend(await self._strict_validation(improvement, context))
            
            # Check priority and implementation status
            if hasattr(improvement, 'priority') and improvement.priority == PriorityLevel.CRITICAL:
                warnings.append("Critical priority improvement requires careful monitoring")
            
            # Determine overall result
            blocking_issues = [issue for issue in issues if issue.blocking]
            passed = len(blocking_issues) == 0
            
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = ValidationResult(
                passed=passed,
                issues=issues,
                warnings=warnings,
                validation_duration_seconds=duration,
                metadata={
                    "safety_level": self.config.get_effective_safety_level(),
                    "rules_executed": len(self.rules),
                    "improvement_type": type(improvement).__name__
                }
            )
            
            self.logger.info(
                f"Validation completed: passed={passed}, "
                f"issues={len(issues)}, warnings={len(warnings)}, "
                f"duration={duration:.2f}s"
            )
            
            return result
        
        except Exception as e:
            self.logger.error(f"Validation failed with exception: {e}")
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return ValidationResult(
                passed=False,
                issues=[ValidationIssue(
                    issue_id="validation_exception",
                    category=ValidationCategory.SAFETY,
                    severity=ValidationSeverity.CRITICAL,
                    message="Validation failed with exception",
                    details=str(e),
                    blocking=True
                )],
                validation_duration_seconds=duration
            )
    
    async def _strict_validation(
        self, 
        improvement: Union[ActionPoint, SystemicImprovement],
        context: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Additional validation for strict safety level."""
        issues = []
        
        # Require manual approval for certain changes
        if self.config.require_manual_approval_for_critical_changes:
            description = improvement.description.lower()
            critical_terms = ["system", "infrastructure", "database", "network", "security"]
            
            if any(term in description for term in critical_terms):
                issues.append(ValidationIssue(
                    issue_id="manual_approval_required",
                    category=ValidationCategory.SAFETY,
                    severity=ValidationSeverity.WARNING,
                    message="Manual approval required for critical system change",
                    details=f"Strict mode requires manual approval: {improvement.title}",
                    suggested_fix="Obtain manual approval before implementation"
                ))
        
        return issues
    
    async def validate_improvements_batch(
        self, 
        improvements: List[Union[ActionPoint, SystemicImprovement]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, ValidationResult]:
        """Validate multiple improvements in batch."""
        if context is None:
            context = {}
        
        results = {}
        
        # Use semaphore to limit concurrent validations
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent validations
        
        async def validate_single(improvement):
            async with semaphore:
                improvement_id = getattr(improvement, 'action_id', None) or getattr(improvement, 'improvement_id', str(id(improvement)))
                result = await self.validate_improvement(improvement, context)
                return improvement_id, result
        
        # Run validations concurrently
        tasks = [validate_single(improvement) for improvement in improvements]
        validation_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in validation_results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch validation error: {result}")
                continue
            
            improvement_id, validation_result = result
            results[improvement_id] = validation_result
        
        self.logger.info(f"Batch validation completed: {len(results)} improvements validated")
        return results
    
    def add_custom_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self.rules.append(rule)
        self.logger.info(f"Added custom validation rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a validation rule by name."""
        original_count = len(self.rules)
        self.rules = [rule for rule in self.rules if rule.name != rule_name]
        removed = len(self.rules) < original_count
        
        if removed:
            self.logger.info(f"Removed validation rule: {rule_name}")
        
        return removed