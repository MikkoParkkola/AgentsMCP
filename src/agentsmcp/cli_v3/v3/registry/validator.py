"""Command validation engine for CLI v3.

Provides comprehensive command validation including parameter validation,
security validation, malicious command detection, dependency verification,
and permission checking with detailed error reporting.
"""

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from ..models.command_models import ExecutionMode, SkillLevel
from ..models.registry_models import (
    ValidationRequest,
    ValidationResult,
    ValidationIssue,
    ParameterDefinition,
    ParameterType,
    ValidationConstraint,
    CommandMetadata,
    SecurityLevel,
)
from .command_registry import CommandRegistry

logger = logging.getLogger(__name__)


class SecurityValidator:
    """Advanced security validation for commands and parameters."""
    
    def __init__(self):
        # Dangerous command patterns
        self.dangerous_patterns = [
            # System modification commands
            r'rm\s+-rf\s+/',
            r'sudo\s+rm',
            r'dd\s+if=.*of=/dev/',
            r'mkfs\.',
            r'fdisk',
            r'parted',
            
            # Network/remote execution
            r'nc\s+.*-e',
            r'ncat\s+.*-e',
            r'telnet\s+.*\|\s*sh',
            r'curl\s+.*\|\s*sh',
            r'wget\s+.*\|\s*sh',
            
            # Code injection
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'os\.popen',
            
            # File system manipulation
            r'chmod\s+777',
            r'chown\s+.*:.*\s+/',
            r'mount\s+.*',
            r'umount\s+.*',
            
            # Process manipulation
            r'kill\s+-9\s+1$',
            r'killall\s+.*',
            r'pkill\s+.*',
            
            # Privilege escalation
            r'sudo\s+su\s*-',
            r'su\s+root',
            r'passwd\s+root',
        ]
        
        # Suspicious file paths
        self.dangerous_paths = [
            '/etc/passwd',
            '/etc/shadow',
            '/etc/sudoers',
            '/boot/',
            '/sys/',
            '/proc/sys/',
            '~/.ssh/',
            '~/.aws/',
            '/dev/null',
            '/dev/zero',
            '/dev/urandom',
        ]
        
        # Dangerous environment variables
        self.dangerous_env_vars = [
            'PATH',
            'LD_LIBRARY_PATH',
            'PYTHONPATH',
            'HOME',
            'USER',
            'SUDO_USER',
        ]
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            r"';.*--",
            r'";.*--',
            r"'\s*or\s*'1'\s*=\s*'1",
            r'"\s*or\s*"1"\s*=\s*"1',
            r"union\s+select",
            r"drop\s+table",
            r"delete\s+from",
            r"insert\s+into",
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r';\s*rm\s+',
            r'&&\s*rm\s+',
            r'\|\s*rm\s+',
            r'`.*`',
            r'\$\(.*\)',
            r'>\s*/dev/',
            r'<\s*/dev/',
        ]
    
    def validate_security(
        self,
        command: str,
        args: Dict[str, Any],
        context: Dict[str, Any],
        security_level: SecurityLevel = SecurityLevel.SAFE
    ) -> Tuple[bool, List[ValidationIssue]]:
        """Perform comprehensive security validation.
        
        Args:
            command: Command name
            args: Command arguments
            context: Execution context
            security_level: Required security level
            
        Returns:
            Tuple of (is_safe, security_issues)
        """
        issues = []
        
        # Build full command string for analysis
        full_command = self._build_command_string(command, args)
        
        # Check for dangerous patterns
        pattern_issues = self._check_dangerous_patterns(full_command)
        issues.extend(pattern_issues)
        
        # Check file path arguments
        path_issues = self._check_file_paths(args)
        issues.extend(path_issues)
        
        # Check for injection attacks
        injection_issues = self._check_injection_attacks(args)
        issues.extend(injection_issues)
        
        # Check environment variable manipulation
        env_issues = self._check_environment_vars(args)
        issues.extend(env_issues)
        
        # Check URL arguments
        url_issues = self._check_urls(args)
        issues.extend(url_issues)
        
        # Check permission requirements vs context
        perm_issues = self._check_permissions(command, args, context, security_level)
        issues.extend(perm_issues)
        
        # Determine if safe based on security level and issues
        critical_issues = [i for i in issues if i.severity == 'error']
        is_safe = len(critical_issues) == 0
        
        # Adjust safety based on security level
        if security_level == SecurityLevel.SYSTEM and issues:
            # System level commands require extra scrutiny
            warning_issues = [i for i in issues if i.severity == 'warning']
            if warning_issues:
                issues.append(ValidationIssue(
                    severity='error',
                    code='SYSTEM_SECURITY_WARNING',
                    message='System-level command has security warnings',
                    suggestion='Review all warnings before proceeding'
                ))
                is_safe = False
        
        return is_safe, issues
    
    def _build_command_string(self, command: str, args: Dict[str, Any]) -> str:
        """Build full command string for pattern analysis."""
        parts = [command]
        
        for key, value in args.items():
            if isinstance(value, bool):
                if value:
                    parts.append(f"--{key}")
            elif isinstance(value, (str, int, float)):
                parts.extend([f"--{key}", str(value)])
            elif isinstance(value, list):
                for item in value:
                    parts.extend([f"--{key}", str(item)])
        
        return ' '.join(parts)
    
    def _check_dangerous_patterns(self, command_string: str) -> List[ValidationIssue]:
        """Check for dangerous command patterns."""
        issues = []
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, command_string, re.IGNORECASE):
                issues.append(ValidationIssue(
                    severity='error',
                    code='DANGEROUS_PATTERN',
                    message=f'Detected potentially dangerous pattern: {pattern}',
                    suggestion='Review command carefully and ensure it is safe'
                ))
        
        return issues
    
    def _check_file_paths(self, args: Dict[str, Any]) -> List[ValidationIssue]:
        """Check file path arguments for dangerous locations."""
        issues = []
        
        for key, value in args.items():
            paths_to_check = []
            
            # Collect path values
            if isinstance(value, str) and ('path' in key.lower() or 'file' in key.lower()):
                paths_to_check.append(value)
            elif isinstance(value, list):
                paths_to_check.extend([str(v) for v in value if isinstance(v, str)])
            
            for path_str in paths_to_check:
                # Expand path for analysis
                try:
                    expanded_path = os.path.expanduser(os.path.expandvars(path_str))
                    
                    # Check against dangerous paths
                    for dangerous_path in self.dangerous_paths:
                        if expanded_path.startswith(dangerous_path):
                            issues.append(ValidationIssue(
                                severity='error',
                                code='DANGEROUS_PATH',
                                message=f'Path "{path_str}" points to sensitive location: {dangerous_path}',
                                parameter=key,
                                suggestion='Use a safer file location'
                            ))
                            break
                    
                    # Check for directory traversal
                    if '..' in path_str or path_str.startswith('/'):
                        if not self._is_safe_absolute_path(expanded_path):
                            issues.append(ValidationIssue(
                                severity='warning',
                                code='PATH_TRAVERSAL_RISK',
                                message=f'Path "{path_str}" may allow directory traversal',
                                parameter=key,
                                suggestion='Use relative paths within safe directories'
                            ))
                    
                    # Check file permissions (if file exists)
                    if os.path.exists(expanded_path):
                        stat_info = os.stat(expanded_path)
                        if stat_info.st_mode & 0o002:  # World-writable
                            issues.append(ValidationIssue(
                                severity='warning',
                                code='WORLD_WRITABLE_FILE',
                                message=f'File "{path_str}" is world-writable',
                                parameter=key,
                                suggestion='Check file permissions'
                            ))
                
                except (OSError, ValueError) as e:
                    issues.append(ValidationIssue(
                        severity='warning',
                        code='PATH_VALIDATION_ERROR',
                        message=f'Could not validate path "{path_str}": {e}',
                        parameter=key
                    ))
        
        return issues
    
    def _check_injection_attacks(self, args: Dict[str, Any]) -> List[ValidationIssue]:
        """Check for various injection attack patterns."""
        issues = []
        
        for key, value in args.items():
            if not isinstance(value, str):
                continue
            
            value_lower = value.lower()
            
            # SQL injection
            for pattern in self.sql_injection_patterns:
                if re.search(pattern, value_lower):
                    issues.append(ValidationIssue(
                        severity='error',
                        code='SQL_INJECTION_RISK',
                        message=f'Detected potential SQL injection in parameter "{key}"',
                        parameter=key,
                        suggestion='Sanitize input or use parameterized queries'
                    ))
                    break
            
            # Command injection
            for pattern in self.command_injection_patterns:
                if re.search(pattern, value):
                    issues.append(ValidationIssue(
                        severity='error',
                        code='COMMAND_INJECTION_RISK',
                        message=f'Detected potential command injection in parameter "{key}"',
                        parameter=key,
                        suggestion='Sanitize input to prevent command execution'
                    ))
                    break
            
            # Script injection (common patterns)
            script_patterns = [
                r'<script.*>',
                r'javascript:',
                r'vbscript:',
                r'onload=',
                r'onerror=',
            ]
            
            for pattern in script_patterns:
                if re.search(pattern, value_lower):
                    issues.append(ValidationIssue(
                        severity='warning',
                        code='SCRIPT_INJECTION_RISK',
                        message=f'Detected potential script injection in parameter "{key}"',
                        parameter=key,
                        suggestion='Escape or sanitize script-like content'
                    ))
                    break
        
        return issues
    
    def _check_environment_vars(self, args: Dict[str, Any]) -> List[ValidationIssue]:
        """Check for dangerous environment variable manipulation."""
        issues = []
        
        for key, value in args.items():
            if 'env' in key.lower() and isinstance(value, dict):
                for env_key, env_value in value.items():
                    if env_key in self.dangerous_env_vars:
                        issues.append(ValidationIssue(
                            severity='warning',
                            code='DANGEROUS_ENV_VAR',
                            message=f'Modifying sensitive environment variable: {env_key}',
                            parameter=key,
                            suggestion='Be cautious when modifying system environment variables'
                        ))
                    
                    # Check for injection in environment values
                    if isinstance(env_value, str) and any(char in env_value for char in '`$();'):
                        issues.append(ValidationIssue(
                            severity='warning',
                            code='ENV_INJECTION_RISK',
                            message=f'Environment variable "{env_key}" contains shell metacharacters',
                            parameter=key,
                            suggestion='Escape shell metacharacters in environment values'
                        ))
        
        return issues
    
    def _check_urls(self, args: Dict[str, Any]) -> List[ValidationIssue]:
        """Check URL arguments for security issues."""
        issues = []
        
        for key, value in args.items():
            if not isinstance(value, str):
                continue
            
            # Check if value looks like a URL
            if re.match(r'^https?://', value, re.IGNORECASE):
                try:
                    parsed = urlparse(value)
                    
                    # Check for private/internal networks
                    hostname = parsed.hostname
                    if hostname:
                        # Check for localhost/internal IPs
                        internal_patterns = [
                            r'^localhost$',
                            r'^127\.',
                            r'^192\.168\.',
                            r'^10\.',
                            r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',
                            r'^::1$',
                            r'^fc00:',
                            r'^fe80:',
                        ]
                        
                        hostname_lower = hostname.lower()
                        for pattern in internal_patterns:
                            if re.match(pattern, hostname_lower):
                                issues.append(ValidationIssue(
                                    severity='warning',
                                    code='INTERNAL_URL',
                                    message=f'URL "{value}" points to internal/private network',
                                    parameter=key,
                                    suggestion='Verify this is intended and safe'
                                ))
                                break
                    
                    # Check for non-HTTPS URLs
                    if parsed.scheme.lower() == 'http':
                        issues.append(ValidationIssue(
                            severity='info',
                            code='INSECURE_URL',
                            message=f'URL "{value}" uses insecure HTTP protocol',
                            parameter=key,
                            suggestion='Consider using HTTPS for better security'
                        ))
                
                except Exception:
                    issues.append(ValidationIssue(
                        severity='warning',
                        code='INVALID_URL',
                        message=f'Invalid URL format: "{value}"',
                        parameter=key,
                        suggestion='Verify URL format is correct'
                    ))
        
        return issues
    
    def _check_permissions(
        self,
        command: str,
        args: Dict[str, Any],
        context: Dict[str, Any],
        security_level: SecurityLevel
    ) -> List[ValidationIssue]:
        """Check permission requirements vs context."""
        issues = []
        
        # Check if running with elevated privileges
        if hasattr(os, 'geteuid') and os.geteuid() == 0:
            if security_level == SecurityLevel.SAFE:
                issues.append(ValidationIssue(
                    severity='warning',
                    code='ELEVATED_PRIVILEGES',
                    message='Running with root privileges while executing safe command',
                    suggestion='Consider running with lower privileges'
                ))
        
        # Check user permissions from context
        user_permissions = context.get('user_permissions', [])
        required_permissions = context.get('required_permissions', [])
        
        for perm in required_permissions:
            if perm not in user_permissions:
                issues.append(ValidationIssue(
                    severity='error',
                    code='INSUFFICIENT_PERMISSIONS',
                    message=f'Missing required permission: {perm}',
                    suggestion=f'Obtain "{perm}" permission before running this command'
                ))
        
        return issues
    
    def _is_safe_absolute_path(self, path: str) -> bool:
        """Check if absolute path is in a safe location."""
        safe_prefixes = [
            '/tmp/',
            '/var/tmp/',
            '/home/',
            '/Users/',
            os.path.expanduser('~/'),
        ]
        
        return any(path.startswith(prefix) for prefix in safe_prefixes)


class ParameterValidator:
    """Parameter validation with type checking and constraint enforcement."""
    
    def validate_parameters(
        self,
        parameters: List[ParameterDefinition],
        args: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, bool], List[ValidationIssue]]:
        """Validate all parameters against their definitions.
        
        Args:
            parameters: Parameter definitions
            args: Provided argument values
            
        Returns:
            Tuple of (all_valid, per_param_results, issues)
        """
        issues = []
        param_results = {}
        
        # Create parameter lookup
        param_map = {p.name: p for p in parameters}
        
        # Add aliases to lookup
        for param in parameters:
            for alias in param.aliases:
                param_map[alias] = param
        
        # Check all provided arguments
        for arg_name, arg_value in args.items():
            if arg_name not in param_map:
                issues.append(ValidationIssue(
                    severity='warning',
                    code='UNKNOWN_PARAMETER',
                    message=f'Unknown parameter: {arg_name}',
                    parameter=arg_name,
                    suggestion='Check parameter name spelling'
                ))
                param_results[arg_name] = False
                continue
            
            param_def = param_map[arg_name]
            param_issues = self._validate_single_parameter(param_def, arg_value)
            
            param_results[param_def.name] = len([i for i in param_issues if i.severity == 'error']) == 0
            issues.extend(param_issues)
        
        # Check for missing required parameters
        provided_params = set(args.keys())
        for param in parameters:
            if param.required and param.name not in provided_params:
                # Check if any aliases were provided
                alias_provided = any(alias in provided_params for alias in param.aliases)
                if not alias_provided:
                    issues.append(ValidationIssue(
                        severity='error',
                        code='MISSING_REQUIRED_PARAMETER',
                        message=f'Missing required parameter: {param.name}',
                        parameter=param.name,
                        suggestion=f'Provide value for required parameter "{param.name}"'
                    ))
                    param_results[param.name] = False
        
        all_valid = len([i for i in issues if i.severity == 'error']) == 0
        return all_valid, param_results, issues
    
    def _validate_single_parameter(
        self,
        param_def: ParameterDefinition,
        value: Any
    ) -> List[ValidationIssue]:
        """Validate a single parameter value."""
        issues = []
        
        # Type validation
        type_valid, type_issues = self._validate_type(param_def, value)
        issues.extend(type_issues)
        
        if not type_valid:
            return issues  # Skip constraint validation if type is wrong
        
        # Constraint validation
        for constraint in param_def.constraints:
            constraint_issues = self._validate_constraint(param_def, value, constraint)
            issues.extend(constraint_issues)
        
        return issues
    
    def _validate_type(
        self,
        param_def: ParameterDefinition,
        value: Any
    ) -> Tuple[bool, List[ValidationIssue]]:
        """Validate parameter type."""
        issues = []
        expected_type = param_def.type
        
        # Type checking logic
        type_valid = True
        
        if expected_type == ParameterType.STRING:
            if not isinstance(value, str):
                type_valid = False
                issues.append(ValidationIssue(
                    severity='error',
                    code='INVALID_TYPE',
                    message=f'Parameter "{param_def.name}" must be a string, got {type(value).__name__}',
                    parameter=param_def.name
                ))
        
        elif expected_type == ParameterType.INTEGER:
            if not isinstance(value, int):
                # Try to convert
                try:
                    int(value)
                except (ValueError, TypeError):
                    type_valid = False
                    issues.append(ValidationIssue(
                        severity='error',
                        code='INVALID_TYPE',
                        message=f'Parameter "{param_def.name}" must be an integer',
                        parameter=param_def.name
                    ))
        
        elif expected_type == ParameterType.FLOAT:
            if not isinstance(value, (int, float)):
                try:
                    float(value)
                except (ValueError, TypeError):
                    type_valid = False
                    issues.append(ValidationIssue(
                        severity='error',
                        code='INVALID_TYPE',
                        message=f'Parameter "{param_def.name}" must be a number',
                        parameter=param_def.name
                    ))
        
        elif expected_type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                # Accept string representations
                if isinstance(value, str) and value.lower() not in ['true', 'false', '1', '0']:
                    type_valid = False
                    issues.append(ValidationIssue(
                        severity='error',
                        code='INVALID_TYPE',
                        message=f'Parameter "{param_def.name}" must be a boolean (true/false)',
                        parameter=param_def.name
                    ))
        
        elif expected_type == ParameterType.LIST:
            if not isinstance(value, list):
                type_valid = False
                issues.append(ValidationIssue(
                    severity='error',
                    code='INVALID_TYPE',
                    message=f'Parameter "{param_def.name}" must be a list',
                    parameter=param_def.name
                ))
        
        elif expected_type == ParameterType.DICT:
            if not isinstance(value, dict):
                type_valid = False
                issues.append(ValidationIssue(
                    severity='error',
                    code='INVALID_TYPE',
                    message=f'Parameter "{param_def.name}" must be a dictionary',
                    parameter=param_def.name
                ))
        
        elif expected_type == ParameterType.FILE_PATH:
            if not isinstance(value, str):
                type_valid = False
                issues.append(ValidationIssue(
                    severity='error',
                    code='INVALID_TYPE',
                    message=f'Parameter "{param_def.name}" must be a file path string',
                    parameter=param_def.name
                ))
            else:
                # Additional file path validation
                expanded_path = os.path.expanduser(os.path.expandvars(value))
                if not os.path.isfile(expanded_path):
                    issues.append(ValidationIssue(
                        severity='warning',
                        code='FILE_NOT_FOUND',
                        message=f'File not found: {value}',
                        parameter=param_def.name,
                        suggestion='Verify the file path is correct'
                    ))
        
        elif expected_type == ParameterType.DIRECTORY_PATH:
            if not isinstance(value, str):
                type_valid = False
                issues.append(ValidationIssue(
                    severity='error',
                    code='INVALID_TYPE',
                    message=f'Parameter "{param_def.name}" must be a directory path string',
                    parameter=param_def.name
                ))
            else:
                expanded_path = os.path.expanduser(os.path.expandvars(value))
                if not os.path.isdir(expanded_path):
                    issues.append(ValidationIssue(
                        severity='warning',
                        code='DIRECTORY_NOT_FOUND',
                        message=f'Directory not found: {value}',
                        parameter=param_def.name,
                        suggestion='Verify the directory path is correct'
                    ))
        
        elif expected_type == ParameterType.URL:
            if not isinstance(value, str):
                type_valid = False
                issues.append(ValidationIssue(
                    severity='error',
                    code='INVALID_TYPE',
                    message=f'Parameter "{param_def.name}" must be a URL string',
                    parameter=param_def.name
                ))
            else:
                try:
                    parsed = urlparse(value)
                    if not parsed.scheme or not parsed.netloc:
                        issues.append(ValidationIssue(
                            severity='error',
                            code='INVALID_URL',
                            message=f'Invalid URL format: {value}',
                            parameter=param_def.name
                        ))
                        type_valid = False
                except Exception:
                    issues.append(ValidationIssue(
                        severity='error',
                        code='INVALID_URL',
                        message=f'Invalid URL: {value}',
                        parameter=param_def.name
                    ))
                    type_valid = False
        
        elif expected_type == ParameterType.EMAIL:
            if not isinstance(value, str):
                type_valid = False
                issues.append(ValidationIssue(
                    severity='error',
                    code='INVALID_TYPE',
                    message=f'Parameter "{param_def.name}" must be an email string',
                    parameter=param_def.name
                ))
            else:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if not re.match(email_pattern, value):
                    issues.append(ValidationIssue(
                        severity='error',
                        code='INVALID_EMAIL',
                        message=f'Invalid email format: {value}',
                        parameter=param_def.name
                    ))
                    type_valid = False
        
        elif expected_type == ParameterType.REGEX:
            if not isinstance(value, str):
                type_valid = False
                issues.append(ValidationIssue(
                    severity='error',
                    code='INVALID_TYPE',
                    message=f'Parameter "{param_def.name}" must be a regex string',
                    parameter=param_def.name
                ))
            else:
                try:
                    re.compile(value)
                except re.error as e:
                    issues.append(ValidationIssue(
                        severity='error',
                        code='INVALID_REGEX',
                        message=f'Invalid regex pattern: {e}',
                        parameter=param_def.name
                    ))
                    type_valid = False
        
        return type_valid, issues
    
    def _validate_constraint(
        self,
        param_def: ParameterDefinition,
        value: Any,
        constraint: ValidationConstraint
    ) -> List[ValidationIssue]:
        """Validate a single constraint."""
        issues = []
        constraint_type = constraint.type
        constraint_value = constraint.value
        
        if constraint_type == 'min_length' and isinstance(value, str):
            if len(value) < constraint_value:
                issues.append(ValidationIssue(
                    severity='error',
                    code='CONSTRAINT_VIOLATION',
                    message=constraint.message or f'Parameter "{param_def.name}" must be at least {constraint_value} characters',
                    parameter=param_def.name
                ))
        
        elif constraint_type == 'max_length' and isinstance(value, str):
            if len(value) > constraint_value:
                issues.append(ValidationIssue(
                    severity='error',
                    code='CONSTRAINT_VIOLATION',
                    message=constraint.message or f'Parameter "{param_def.name}" must be at most {constraint_value} characters',
                    parameter=param_def.name
                ))
        
        elif constraint_type == 'min_value' and isinstance(value, (int, float)):
            if value < constraint_value:
                issues.append(ValidationIssue(
                    severity='error',
                    code='CONSTRAINT_VIOLATION',
                    message=constraint.message or f'Parameter "{param_def.name}" must be at least {constraint_value}',
                    parameter=param_def.name
                ))
        
        elif constraint_type == 'max_value' and isinstance(value, (int, float)):
            if value > constraint_value:
                issues.append(ValidationIssue(
                    severity='error',
                    code='CONSTRAINT_VIOLATION',
                    message=constraint.message or f'Parameter "{param_def.name}" must be at most {constraint_value}',
                    parameter=param_def.name
                ))
        
        elif constraint_type == 'pattern' and isinstance(value, str):
            if not re.match(str(constraint_value), value):
                issues.append(ValidationIssue(
                    severity='error',
                    code='CONSTRAINT_VIOLATION',
                    message=constraint.message or f'Parameter "{param_def.name}" does not match required pattern',
                    parameter=param_def.name
                ))
        
        elif constraint_type == 'choices' and isinstance(constraint_value, list):
            if value not in constraint_value:
                choices_str = ', '.join(str(c) for c in constraint_value)
                issues.append(ValidationIssue(
                    severity='error',
                    code='CONSTRAINT_VIOLATION',
                    message=constraint.message or f'Parameter "{param_def.name}" must be one of: {choices_str}',
                    parameter=param_def.name
                ))
        
        return issues


class DependencyValidator:
    """Validates command dependencies and availability."""
    
    def __init__(self, registry: CommandRegistry):
        self.registry = registry
    
    def validate_dependencies(
        self,
        command_name: str
    ) -> Tuple[bool, Dict[str, bool], List[ValidationIssue]]:
        """Validate all dependencies for a command.
        
        Args:
            command_name: Name of command to validate dependencies for
            
        Returns:
            Tuple of (all_satisfied, dependency_status, issues)
        """
        issues = []
        dependency_status = {}
        
        # Get command metadata
        metadata = self.registry.get_command(command_name)
        if not metadata:
            issues.append(ValidationIssue(
                severity='error',
                code='COMMAND_NOT_FOUND',
                message=f'Command "{command_name}" not found in registry'
            ))
            return False, {}, issues
        
        # Check each dependency
        for dep in metadata.definition.dependencies:
            dep_metadata = self.registry.get_command(dep.command)
            
            if not dep_metadata:
                if not dep.optional:
                    issues.append(ValidationIssue(
                        severity='error',
                        code='MISSING_DEPENDENCY',
                        message=f'Required dependency "{dep.command}" not available',
                        suggestion=f'Install or enable dependency "{dep.command}"'
                    ))
                    dependency_status[dep.command] = False
                else:
                    issues.append(ValidationIssue(
                        severity='info',
                        code='OPTIONAL_DEPENDENCY_MISSING',
                        message=f'Optional dependency "{dep.command}" not available',
                        suggestion=f'Some features may be limited without "{dep.command}"'
                    ))
                    dependency_status[dep.command] = True  # Optional deps don't fail validation
            else:
                # Check version compatibility if specified
                if dep.version_min or dep.version_max:
                    version_issues = self._check_version_compatibility(
                        dep_metadata, dep.version_min, dep.version_max
                    )
                    issues.extend(version_issues)
                    dependency_status[dep.command] = len([i for i in version_issues if i.severity == 'error']) == 0
                else:
                    dependency_status[dep.command] = True
        
        all_satisfied = all(status for status in dependency_status.values())
        return all_satisfied, dependency_status, issues
    
    def _check_version_compatibility(
        self,
        dep_metadata: CommandMetadata,
        min_version: Optional[str],
        max_version: Optional[str]
    ) -> List[ValidationIssue]:
        """Check version compatibility for a dependency."""
        issues = []
        current_version = dep_metadata.definition.version
        
        try:
            # Simple version comparison (assumes semantic versioning)
            current_parts = [int(x) for x in current_version.split('.')]
            
            if min_version:
                min_parts = [int(x) for x in min_version.split('.')]
                if current_parts < min_parts:
                    issues.append(ValidationIssue(
                        severity='error',
                        code='VERSION_TOO_OLD',
                        message=f'Dependency "{dep_metadata.definition.name}" version {current_version} is older than required {min_version}',
                        suggestion=f'Update dependency to version {min_version} or newer'
                    ))
            
            if max_version:
                max_parts = [int(x) for x in max_version.split('.')]
                if current_parts > max_parts:
                    issues.append(ValidationIssue(
                        severity='error',
                        code='VERSION_TOO_NEW',
                        message=f'Dependency "{dep_metadata.definition.name}" version {current_version} is newer than supported {max_version}',
                        suggestion=f'Use dependency version {max_version} or older'
                    ))
        
        except (ValueError, AttributeError) as e:
            issues.append(ValidationIssue(
                severity='warning',
                code='VERSION_PARSE_ERROR',
                message=f'Could not parse version numbers for dependency "{dep_metadata.definition.name}": {e}'
            ))
        
        return issues


class CommandValidator:
    """Comprehensive command validation engine.
    
    Provides multi-layered validation including:
    - Parameter validation with type checking and constraints
    - Security validation with malicious command detection
    - Dependency verification
    - Permission checking
    - Performance estimation
    """
    
    def __init__(self, registry: CommandRegistry):
        """Initialize validator with command registry.
        
        Args:
            registry: CommandRegistry instance
        """
        self.registry = registry
        self.security_validator = SecurityValidator()
        self.parameter_validator = ParameterValidator()
        self.dependency_validator = DependencyValidator(registry)
        
        # Validation statistics
        self._stats = {
            'total_validations': 0,
            'validation_errors': 0,
            'security_blocks': 0,
            'avg_validation_time_ms': 0,
            'last_validation': None
        }
        
        logger.info("CommandValidator initialized")
    
    async def validate_command(self, request: ValidationRequest) -> ValidationResult:
        """Validate a complete command request.
        
        Args:
            request: ValidationRequest with command and context
            
        Returns:
            ValidationResult with comprehensive validation details
        """
        start_time = time.perf_counter()
        self._stats['total_validations'] += 1
        
        try:
            # Get command metadata
            metadata = self.registry.get_command(request.command)
            if not metadata:
                return ValidationResult(
                    valid=False,
                    command_found=False,
                    issues=[ValidationIssue(
                        severity='error',
                        code='COMMAND_NOT_FOUND',
                        message=f'Command "{request.command}" not found',
                        suggestion='Check command spelling or use "help" to list available commands'
                    )]
                )
            
            issues = []
            all_validations_passed = True
            
            # Parameter validation
            param_valid = True
            param_results = {}
            if request.parameter_validation:
                param_valid, param_results, param_issues = self.parameter_validator.validate_parameters(
                    metadata.definition.parameters, request.args
                )
                issues.extend(param_issues)
                if not param_valid:
                    all_validations_passed = False
            
            # Security validation
            security_valid = True
            security_assessment = {}
            if request.security_check:
                security_valid, security_issues = self.security_validator.validate_security(
                    request.command, request.args, request.context, 
                    metadata.definition.security_level
                )
                issues.extend(security_issues)
                
                security_assessment = {
                    'security_level': metadata.definition.security_level.value,
                    'safe': security_valid,
                    'risk_factors': [i.code for i in security_issues if i.severity == 'error']
                }
                
                if not security_valid:
                    all_validations_passed = False
                    self._stats['security_blocks'] += 1
            
            # Dependency validation
            deps_valid = True
            dep_status = {}
            if request.dependency_check:
                deps_valid, dep_status, dep_issues = self.dependency_validator.validate_dependencies(
                    request.command
                )
                issues.extend(dep_issues)
                if not deps_valid:
                    all_validations_passed = False
            
            # Permission validation
            if request.permission_check:
                perm_issues = self._validate_permissions(metadata, request)
                issues.extend(perm_issues)
                perm_errors = [i for i in perm_issues if i.severity == 'error']
                if perm_errors:
                    all_validations_passed = False
            
            # Interface compatibility
            interface_issues = self._validate_interface_compatibility(metadata, request)
            issues.extend(interface_issues)
            interface_errors = [i for i in interface_issues if i.severity == 'error']
            if interface_errors:
                all_validations_passed = False
            
            # Generate suggestions
            suggestions = self._generate_suggestions(metadata, issues, request)
            
            # Estimate performance
            estimated_time = self._estimate_execution_time(metadata, request.args)
            resource_estimates = self._estimate_resource_usage(metadata, request.args)
            
            # Track validation errors
            error_count = len([i for i in issues if i.severity == 'error'])
            if error_count > 0:
                self._stats['validation_errors'] += 1
            
            # Create result
            result = ValidationResult(
                valid=all_validations_passed,
                command_found=True,
                issues=issues,
                suggestions=suggestions,
                parameter_validation=param_results,
                security_assessment=security_assessment,
                dependency_status=dep_status,
                estimated_execution_time_ms=estimated_time,
                estimated_resource_usage=resource_estimates
            )
            
            # Update performance stats
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._update_validation_stats(duration_ms)
            
            logger.debug(f"Validated command '{request.command}' in {duration_ms:.2f}ms")
            return result
        
        except Exception as e:
            logger.error(f"Validation failed for command '{request.command}': {e}")
            return ValidationResult(
                valid=False,
                command_found=False,
                issues=[ValidationIssue(
                    severity='error',
                    code='VALIDATION_ERROR',
                    message=f'Validation failed: {str(e)}'
                )]
            )
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics.
        
        Returns:
            Dictionary with validation metrics
        """
        return dict(self._stats)
    
    # Internal methods
    
    def _validate_permissions(
        self,
        metadata: CommandMetadata,
        request: ValidationRequest
    ) -> List[ValidationIssue]:
        """Validate user permissions for command."""
        issues = []
        
        required_perms = metadata.definition.required_permissions
        user_perms = request.user_permissions
        
        for perm in required_perms:
            if perm not in user_perms:
                issues.append(ValidationIssue(
                    severity='error',
                    code='INSUFFICIENT_PERMISSIONS',
                    message=f'Missing required permission: {perm}',
                    suggestion=f'Obtain "{perm}" permission to run this command'
                ))
        
        # Check skill level requirements
        skill_order = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE, SkillLevel.EXPERT]
        required_idx = skill_order.index(metadata.definition.min_skill_level)
        user_idx = skill_order.index(request.user_skill_level)
        
        if user_idx < required_idx:
            issues.append(ValidationIssue(
                severity='warning',
                code='INSUFFICIENT_SKILL_LEVEL',
                message=f'Command requires {metadata.definition.min_skill_level.value} skill level, user is {request.user_skill_level.value}',
                suggestion='This command may be complex for your current skill level'
            ))
        
        return issues
    
    def _validate_interface_compatibility(
        self,
        metadata: CommandMetadata,
        request: ValidationRequest
    ) -> List[ValidationIssue]:
        """Validate execution mode compatibility."""
        issues = []
        
        if request.execution_mode not in metadata.definition.supported_modes:
            mode_names = [m.value for m in metadata.definition.supported_modes]
            issues.append(ValidationIssue(
                severity='error',
                code='UNSUPPORTED_EXECUTION_MODE',
                message=f'Command "{metadata.definition.name}" not supported in {request.execution_mode.value} mode',
                suggestion=f'Use one of the supported modes: {", ".join(mode_names)}'
            ))
        
        return issues
    
    def _generate_suggestions(
        self,
        metadata: CommandMetadata,
        issues: List[ValidationIssue],
        request: ValidationRequest
    ) -> List[str]:
        """Generate helpful suggestions based on validation results."""
        suggestions = []
        
        # Add suggestions from issues
        for issue in issues:
            if issue.suggestion:
                suggestions.append(issue.suggestion)
        
        # Add command-specific suggestions
        if metadata.definition.examples:
            suggestions.append(f'See examples: {metadata.definition.examples[0].command}')
        
        # Add deprecation suggestions
        if metadata.definition.deprecated and metadata.definition.replacement:
            suggestions.append(f'Consider using "{metadata.definition.replacement}" instead (this command is deprecated)')
        
        # Performance suggestions
        if len(request.args) == 0 and metadata.definition.parameters:
            required_params = [p.name for p in metadata.definition.parameters if p.required]
            if required_params:
                suggestions.append(f'This command requires parameters: {", ".join(required_params)}')
        
        return list(set(suggestions))  # Remove duplicates
    
    def _estimate_execution_time(self, metadata: CommandMetadata, args: Dict[str, Any]) -> Optional[int]:
        """Estimate command execution time in milliseconds."""
        base_time = metadata.avg_execution_time_ms or 1000  # Default 1 second
        
        # Adjust based on arguments
        complexity_multiplier = 1.0
        
        # More arguments = potentially more complexity
        if len(args) > 5:
            complexity_multiplier *= 1.2
        
        # File operations take longer
        for key, value in args.items():
            if 'file' in key.lower() or 'path' in key.lower():
                if isinstance(value, str) and os.path.exists(value):
                    try:
                        file_size = os.path.getsize(value)
                        if file_size > 1024 * 1024:  # > 1MB
                            complexity_multiplier *= 1.5
                    except OSError:
                        pass
        
        estimated = int(base_time * complexity_multiplier)
        return min(estimated, 300000)  # Cap at 5 minutes
    
    def _estimate_resource_usage(self, metadata: CommandMetadata, args: Dict[str, Any]) -> Dict[str, Union[int, float]]:
        """Estimate resource usage for command execution."""
        estimates = {
            'memory_mb': 50,  # Base memory usage
            'cpu_percent': 10,  # Base CPU usage
            'disk_io_mb': 0,   # Disk I/O
            'network_kb': 0    # Network usage
        }
        
        # Adjust based on command category and arguments
        definition = metadata.definition
        
        if definition.category.value in ['system', 'advanced']:
            estimates['memory_mb'] *= 2
            estimates['cpu_percent'] *= 1.5
        
        # File operations increase disk I/O
        file_args = [v for k, v in args.items() 
                    if 'file' in k.lower() or 'path' in k.lower()]
        
        if file_args:
            estimates['disk_io_mb'] = len(file_args) * 10  # Rough estimate
        
        # URL arguments suggest network usage
        url_args = [v for v in args.values() 
                   if isinstance(v, str) and v.startswith(('http://', 'https://'))]
        
        if url_args:
            estimates['network_kb'] = len(url_args) * 100  # Rough estimate
        
        return estimates
    
    def _update_validation_stats(self, duration_ms: float) -> None:
        """Update validation performance statistics."""
        self._stats['last_validation'] = datetime.now(timezone.utc)
        
        # Update running average
        current_avg = self._stats['avg_validation_time_ms']
        total_validations = self._stats['total_validations']
        
        if total_validations == 1:
            self._stats['avg_validation_time_ms'] = duration_ms
        else:
            self._stats['avg_validation_time_ms'] = (
                (current_avg * (total_validations - 1) + duration_ms) / total_validations
            )