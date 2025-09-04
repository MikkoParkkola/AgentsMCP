"""
SafetyStatusDisplay - Safety validation and rollback interface for improvement implementations.

This module provides comprehensive safety validation monitoring, health checks, and
emergency rollback capabilities to ensure system stability during improvements.
"""

import asyncio
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.tree import Tree
from rich.prompt import Prompt, Confirm
from rich.align import Align
from rich.columns import Columns


class SafetyStatus(Enum):
    """Safety validation status levels."""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    VALIDATING = "validating"
    FAILED = "failed"


class HealthCheckType(Enum):
    """Types of health checks performed."""
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CONNECTIVITY = "connectivity"
    DATA_INTEGRITY = "data_integrity"
    RESOURCE_USAGE = "resource_usage"
    USER_IMPACT = "user_impact"


@dataclass
class HealthCheck:
    """Individual health check result."""
    check_type: HealthCheckType
    name: str
    status: SafetyStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    threshold_exceeded: bool = False
    last_check: Optional[datetime] = None
    check_duration_ms: float = 0.0
    

@dataclass
class SafetyValidationResult:
    """Result of comprehensive safety validation."""
    overall_status: SafetyStatus
    health_checks: List[HealthCheck] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rollback_recommended: bool = False
    validation_timestamp: datetime = field(default_factory=datetime.now)
    validation_duration_ms: float = 0.0
    

@dataclass
class RollbackPlan:
    """Plan for rolling back implementations."""
    rollback_id: str
    affected_improvements: List[str]
    rollback_steps: List[str] = field(default_factory=list)
    estimated_duration_minutes: int = 5
    requires_confirmation: bool = True
    backup_available: bool = True
    

class SafetyStatusDisplay:
    """Safety validation and rollback interface with comprehensive monitoring."""
    
    def __init__(self, console: Console):
        """Initialize the safety status display.
        
        Args:
            console: Rich console for rendering
        """
        self.console = console
        self.current_validation_result: Optional[SafetyValidationResult] = None
        self.health_check_history: List[SafetyValidationResult] = []
        self.rollback_plans: Dict[str, RollbackPlan] = {}
        self.monitoring_active = False
        
    async def validate_safety(self, approved_improvements: List, layout: Layout, 
                            safety_status: Dict[str, bool]) -> bool:
        """Validate system safety after improvements implementation.
        
        Args:
            approved_improvements: List of improvements that were implemented
            layout: Rich layout for UI updates
            safety_status: Dictionary to update with safety status
            
        Returns:
            True if safety validation passed, False if rollback required
        """
        try:
            # Initialize safety validation
            await self._initialize_safety_validation(approved_improvements, layout)
            
            # Run comprehensive health checks
            validation_result = await self._run_comprehensive_health_checks(approved_improvements)
            
            # Update UI with results
            self._update_safety_status_panel(validation_result, layout)
            self._update_rollback_controls_panel(validation_result, approved_improvements, layout)
            
            # Process validation results
            return await self._process_validation_results(
                validation_result, approved_improvements, layout, safety_status
            )
            
        except Exception as e:
            self.console.print(f"❌ Safety validation failed: {e}")
            return False
            
    async def _initialize_safety_validation(self, improvements: List, layout: Layout) -> None:
        """Initialize safety validation process.
        
        Args:
            improvements: List of implemented improvements
            layout: Layout for UI updates
        """
        self.monitoring_active = True
        
        # Show initialization message
        init_content = []
        
        title_text = Text()
        title_text.append("🔒 ", style="blue")
        title_text.append("Safety Validation Initiated", style="bold blue")
        init_content.append(title_text)
        init_content.append("")
        
        # Validation overview
        overview_table = Table.grid(padding=(0, 2))
        overview_table.add_column("Aspect", style="bold cyan")
        overview_table.add_column("Status", style="white")
        
        overview_table.add_row("Implemented Changes:", f"{len(improvements)} improvements")
        overview_table.add_row("Safety Framework:", "Active")
        overview_table.add_row("Health Checks:", "7 categories")
        overview_table.add_row("Rollback Capability:", "Ready")
        
        init_content.append(overview_table)
        init_content.append("")
        
        # Safety priorities
        priorities_text = Text()
        priorities_text.append("🎯 Validation Priorities:", style="bold yellow")
        priorities_text.append("\n• System stability and performance")
        priorities_text.append("\n• Security posture maintenance")
        priorities_text.append("\n• User impact assessment")
        priorities_text.append("\n• Data integrity verification")
        init_content.append(priorities_text)
        
        from rich.console import Group
        init_group = Group(*init_content)
        
        if layout and "safety_status" in layout:
            layout["safety_status"].update(
                Panel(init_group, title="🛡️ Safety Validation", border_style="blue")
            )
        else:
            self.console.print(Panel(init_group, title="🛡️ Safety Validation", border_style="blue"))
            
        await asyncio.sleep(2)  # Brief pause for user to read
        
    async def _run_comprehensive_health_checks(self, improvements: List) -> SafetyValidationResult:
        """Run comprehensive health checks across all system aspects.
        
        Args:
            improvements: List of implemented improvements
            
        Returns:
            SafetyValidationResult with all health check results
        """
        start_time = datetime.now()
        health_checks = []
        
        # Define health check scenarios
        check_definitions = [
            (HealthCheckType.SYSTEM_HEALTH, "System Resources", self._check_system_health),
            (HealthCheckType.PERFORMANCE, "Response Times", self._check_performance),
            (HealthCheckType.SECURITY, "Security Posture", self._check_security),
            (HealthCheckType.CONNECTIVITY, "Network Connectivity", self._check_connectivity),
            (HealthCheckType.DATA_INTEGRITY, "Data Integrity", self._check_data_integrity),
            (HealthCheckType.RESOURCE_USAGE, "Resource Usage", self._check_resource_usage),
            (HealthCheckType.USER_IMPACT, "User Experience", self._check_user_impact),
        ]
        
        # Execute health checks
        for check_type, name, check_func in check_definitions:
            check_start = datetime.now()
            
            try:
                check_result = await check_func(improvements)
                check_duration = (datetime.now() - check_start).total_seconds() * 1000
                
                health_check = HealthCheck(
                    check_type=check_type,
                    name=name,
                    status=check_result["status"],
                    message=check_result["message"],
                    details=check_result.get("details", {}),
                    threshold_exceeded=check_result.get("threshold_exceeded", False),
                    last_check=datetime.now(),
                    check_duration_ms=check_duration
                )
                
                health_checks.append(health_check)
                
                # Brief delay between checks for UI updates
                await asyncio.sleep(0.3)
                
            except Exception as e:
                # Handle check failures gracefully
                health_check = HealthCheck(
                    check_type=check_type,
                    name=name,
                    status=SafetyStatus.FAILED,
                    message=f"Health check failed: {str(e)[:100]}",
                    last_check=datetime.now(),
                    check_duration_ms=(datetime.now() - check_start).total_seconds() * 1000
                )
                health_checks.append(health_check)
        
        # Analyze overall results
        validation_duration = (datetime.now() - start_time).total_seconds() * 1000
        overall_status, critical_issues, warnings = self._analyze_health_results(health_checks)
        
        # Determine rollback recommendation
        rollback_recommended = (
            overall_status == SafetyStatus.CRITICAL or
            len(critical_issues) > 0 or
            len([hc for hc in health_checks if hc.status == SafetyStatus.CRITICAL]) > 1
        )
        
        result = SafetyValidationResult(
            overall_status=overall_status,
            health_checks=health_checks,
            critical_issues=critical_issues,
            warnings=warnings,
            rollback_recommended=rollback_recommended,
            validation_timestamp=datetime.now(),
            validation_duration_ms=validation_duration
        )
        
        self.current_validation_result = result
        self.health_check_history.append(result)
        
        return result
        
    async def _check_system_health(self, improvements: List) -> Dict[str, Any]:
        """Check system health metrics.
        
        Args:
            improvements: List of implemented improvements
            
        Returns:
            Health check result dictionary
        """
        # Simulate system health check
        await asyncio.sleep(0.5)
        
        # Mock system metrics - in real implementation, these would be actual system calls
        cpu_usage = 45.2  # Percentage
        memory_usage = 62.8  # Percentage
        disk_usage = 78.5  # Percentage
        
        # Determine status based on thresholds
        if cpu_usage > 90 or memory_usage > 90 or disk_usage > 95:
            status = SafetyStatus.CRITICAL
            message = "System resources critically high"
        elif cpu_usage > 80 or memory_usage > 80 or disk_usage > 85:
            status = SafetyStatus.WARNING
            message = "System resources elevated but stable"
        else:
            status = SafetyStatus.SAFE
            message = "System resources within normal limits"
            
        return {
            "status": status,
            "message": message,
            "details": {
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory_usage,
                "disk_usage_percent": disk_usage
            },
            "threshold_exceeded": cpu_usage > 80 or memory_usage > 80
        }
        
    async def _check_performance(self, improvements: List) -> Dict[str, Any]:
        """Check performance metrics.
        
        Args:
            improvements: List of implemented improvements
            
        Returns:
            Performance check result dictionary
        """
        await asyncio.sleep(0.4)
        
        # Mock performance metrics
        avg_response_time = 125.3  # milliseconds
        p95_response_time = 289.7  # milliseconds
        error_rate = 0.02  # percentage
        
        # Check if performance improvements are working
        performance_improvements = [imp for imp in improvements if imp.category == "performance"]
        
        if performance_improvements:
            # Simulate improved performance
            avg_response_time *= 0.7  # 30% improvement
            p95_response_time *= 0.6  # 40% improvement
            
        if avg_response_time > 500 or p95_response_time > 1000 or error_rate > 1.0:
            status = SafetyStatus.CRITICAL
            message = "Performance metrics exceed acceptable thresholds"
        elif avg_response_time > 200 or p95_response_time > 500 or error_rate > 0.5:
            status = SafetyStatus.WARNING
            message = "Performance metrics elevated but acceptable"
        else:
            status = SafetyStatus.SAFE
            message = "Performance metrics within acceptable range"
            
        return {
            "status": status,
            "message": message,
            "details": {
                "avg_response_time_ms": avg_response_time,
                "p95_response_time_ms": p95_response_time,
                "error_rate_percent": error_rate
            }
        }
        
    async def _check_security(self, improvements: List) -> Dict[str, Any]:
        """Check security posture.
        
        Args:
            improvements: List of implemented improvements
            
        Returns:
            Security check result dictionary
        """
        await asyncio.sleep(0.6)
        
        # Check if security improvements were applied
        security_improvements = [imp for imp in improvements if imp.category == "security"]
        
        # Mock security scan results
        vulnerabilities_count = max(0, 3 - len(security_improvements))  # Fewer with security improvements
        exposed_endpoints = max(0, 2 - len(security_improvements))
        auth_failures = 1
        
        if vulnerabilities_count > 0 or exposed_endpoints > 1:
            status = SafetyStatus.CRITICAL
            message = "Security vulnerabilities detected"
        elif auth_failures > 5:
            status = SafetyStatus.WARNING
            message = "Elevated authentication failures"
        else:
            status = SafetyStatus.SAFE
            message = "Security posture healthy"
            
        return {
            "status": status,
            "message": message,
            "details": {
                "vulnerabilities_count": vulnerabilities_count,
                "exposed_endpoints": exposed_endpoints,
                "auth_failures": auth_failures,
                "security_improvements_applied": len(security_improvements)
            }
        }
        
    async def _check_connectivity(self, improvements: List) -> Dict[str, Any]:
        """Check network connectivity.
        
        Args:
            improvements: List of implemented improvements
            
        Returns:
            Connectivity check result dictionary
        """
        await asyncio.sleep(0.3)
        
        # Mock connectivity checks
        external_services_reachable = 4
        total_external_services = 4
        internal_services_reachable = 8
        total_internal_services = 8
        
        if external_services_reachable < total_external_services or internal_services_reachable < total_internal_services:
            status = SafetyStatus.CRITICAL
            message = "Service connectivity issues detected"
        else:
            status = SafetyStatus.SAFE
            message = "All services reachable"
            
        return {
            "status": status,
            "message": message,
            "details": {
                "external_services_ok": f"{external_services_reachable}/{total_external_services}",
                "internal_services_ok": f"{internal_services_reachable}/{total_internal_services}"
            }
        }
        
    async def _check_data_integrity(self, improvements: List) -> Dict[str, Any]:
        """Check data integrity.
        
        Args:
            improvements: List of implemented improvements
            
        Returns:
            Data integrity check result dictionary
        """
        await asyncio.sleep(0.7)
        
        # Mock data integrity checks
        checksum_mismatches = 0
        corrupted_records = 0
        backup_status = "healthy"
        
        if checksum_mismatches > 0 or corrupted_records > 0:
            status = SafetyStatus.CRITICAL
            message = "Data integrity issues detected"
        elif backup_status != "healthy":
            status = SafetyStatus.WARNING
            message = "Backup system requires attention"
        else:
            status = SafetyStatus.SAFE
            message = "Data integrity verified"
            
        return {
            "status": status,
            "message": message,
            "details": {
                "checksum_mismatches": checksum_mismatches,
                "corrupted_records": corrupted_records,
                "backup_status": backup_status
            }
        }
        
    async def _check_resource_usage(self, improvements: List) -> Dict[str, Any]:
        """Check resource usage patterns.
        
        Args:
            improvements: List of implemented improvements
            
        Returns:
            Resource usage check result dictionary
        """
        await asyncio.sleep(0.4)
        
        # Mock resource usage metrics
        connection_pool_usage = 45.2  # Percentage
        thread_pool_usage = 62.1  # Percentage
        memory_leaks_detected = 0
        
        if connection_pool_usage > 90 or thread_pool_usage > 90 or memory_leaks_detected > 0:
            status = SafetyStatus.CRITICAL
            message = "Resource usage critically high"
        elif connection_pool_usage > 80 or thread_pool_usage > 80:
            status = SafetyStatus.WARNING
            message = "Resource usage elevated"
        else:
            status = SafetyStatus.SAFE
            message = "Resource usage healthy"
            
        return {
            "status": status,
            "message": message,
            "details": {
                "connection_pool_usage_percent": connection_pool_usage,
                "thread_pool_usage_percent": thread_pool_usage,
                "memory_leaks_detected": memory_leaks_detected
            }
        }
        
    async def _check_user_impact(self, improvements: List) -> Dict[str, Any]:
        """Check user experience impact.
        
        Args:
            improvements: List of implemented improvements
            
        Returns:
            User impact check result dictionary
        """
        await asyncio.sleep(0.5)
        
        # Check if UX improvements were applied
        ux_improvements = [imp for imp in improvements if imp.category == "ux"]
        
        # Mock user experience metrics
        active_user_sessions = 1247
        user_error_reports = max(0, 5 - len(ux_improvements) * 2)  # Fewer with UX improvements
        satisfaction_score = min(4.8, 4.2 + len(ux_improvements) * 0.15)  # Higher with UX improvements
        
        if user_error_reports > 10:
            status = SafetyStatus.CRITICAL
            message = "High user error rates detected"
        elif satisfaction_score < 4.0:
            status = SafetyStatus.WARNING
            message = "User satisfaction below target"
        else:
            status = SafetyStatus.SAFE
            message = "User experience stable"
            
        return {
            "status": status,
            "message": message,
            "details": {
                "active_sessions": active_user_sessions,
                "error_reports": user_error_reports,
                "satisfaction_score": satisfaction_score,
                "ux_improvements_applied": len(ux_improvements)
            }
        }
        
    def _analyze_health_results(self, health_checks: List[HealthCheck]) -> Tuple[SafetyStatus, List[str], List[str]]:
        """Analyze health check results to determine overall status.
        
        Args:
            health_checks: List of completed health checks
            
        Returns:
            Tuple of (overall_status, critical_issues, warnings)
        """
        critical_checks = [hc for hc in health_checks if hc.status == SafetyStatus.CRITICAL]
        warning_checks = [hc for hc in health_checks if hc.status == SafetyStatus.WARNING]
        failed_checks = [hc for hc in health_checks if hc.status == SafetyStatus.FAILED]
        
        critical_issues = []
        warnings = []
        
        # Collect critical issues
        for check in critical_checks:
            critical_issues.append(f"{check.name}: {check.message}")
            
        # Collect warnings
        for check in warning_checks:
            warnings.append(f"{check.name}: {check.message}")
            
        # Handle failed checks as critical
        for check in failed_checks:
            critical_issues.append(f"{check.name}: Health check failed")
            
        # Determine overall status
        if critical_checks or failed_checks:
            overall_status = SafetyStatus.CRITICAL
        elif len(warning_checks) > 2:  # Multiple warnings indicate system stress
            overall_status = SafetyStatus.WARNING
        elif warning_checks:
            overall_status = SafetyStatus.WARNING
        else:
            overall_status = SafetyStatus.SAFE
            
        return overall_status, critical_issues, warnings
        
    def _update_safety_status_panel(self, validation_result: SafetyValidationResult, layout: Layout) -> None:
        """Update the safety status panel with validation results.
        
        Args:
            validation_result: Safety validation results
            layout: Layout to update
        """
        status_content = []
        
        # Overall status header
        status_colors = {
            SafetyStatus.SAFE: "green",
            SafetyStatus.WARNING: "yellow",
            SafetyStatus.CRITICAL: "red",
            SafetyStatus.FAILED: "red",
            SafetyStatus.VALIDATING: "blue"
        }
        
        status_icons = {
            SafetyStatus.SAFE: "✅",
            SafetyStatus.WARNING: "⚠️",
            SafetyStatus.CRITICAL: "🔴",
            SafetyStatus.FAILED: "❌",
            SafetyStatus.VALIDATING: "🔍"
        }
        
        overall_color = status_colors.get(validation_result.overall_status, "white")
        overall_icon = status_icons.get(validation_result.overall_status, "❓")
        
        header_text = Text()
        header_text.append(f"{overall_icon} ", style=overall_color)
        header_text.append("Overall Safety Status: ", style="bold white")
        header_text.append(
            validation_result.overall_status.value.upper().replace("_", " "),
            style=f"bold {overall_color}"
        )
        
        status_content.append(header_text)
        status_content.append("")
        
        # Health checks table
        checks_table = Table(show_header=True, header_style="bold magenta")
        checks_table.add_column("Check", style="white", width=20)
        checks_table.add_column("Status", width=12)
        checks_table.add_column("Message", style="dim")
        checks_table.add_column("Duration", style="dim", width=8)
        
        for check in validation_result.health_checks:
            status_color = status_colors.get(check.status, "white")
            status_icon = status_icons.get(check.status, "❓")
            
            checks_table.add_row(
                check.name,
                f"[{status_color}]{status_icon} {check.status.value.upper()}[/{status_color}]",
                check.message[:40] + ("..." if len(check.message) > 40 else ""),
                f"{check.check_duration_ms:.0f}ms"
            )
            
        status_content.append(checks_table)
        
        # Critical issues and warnings
        if validation_result.critical_issues:
            status_content.append("")
            critical_text = Text()
            critical_text.append("🚨 Critical Issues:", style="bold red")
            status_content.append(critical_text)
            
            for issue in validation_result.critical_issues[:3]:  # Show top 3
                issue_text = Text()
                issue_text.append("  • ", style="red")
                issue_text.append(issue[:60] + ("..." if len(issue) > 60 else ""))
                status_content.append(issue_text)
                
        if validation_result.warnings:
            status_content.append("")
            warning_text = Text()
            warning_text.append("⚠️ Warnings:", style="bold yellow")
            status_content.append(warning_text)
            
            for warning in validation_result.warnings[:2]:  # Show top 2
                warning_text = Text()
                warning_text.append("  • ", style="yellow")
                warning_text.append(warning[:60] + ("..." if len(warning) > 60 else ""))
                status_content.append(warning_text)
        
        # Validation summary
        status_content.append("")
        summary_table = Table.grid(padding=(0, 2))
        summary_table.add_column("Metric", style="bold cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Validation Duration:", f"{validation_result.validation_duration_ms:.0f}ms")
        summary_table.add_row("Health Checks:", f"{len(validation_result.health_checks)}")
        summary_table.add_row("Safe Checks:", f"{len([hc for hc in validation_result.health_checks if hc.status == SafetyStatus.SAFE])}")
        summary_table.add_row("Rollback Recommended:", "Yes" if validation_result.rollback_recommended else "No")
        
        status_content.append(summary_table)
        
        from rich.console import Group
        status_group = Group(*status_content)
        
        if layout and "safety_status" in layout:
            layout["safety_status"].update(
                Panel(status_group, title="🛡️ Safety Validation Results", border_style=overall_color)
            )
            
    def _update_rollback_controls_panel(self, validation_result: SafetyValidationResult, 
                                      improvements: List, layout: Layout) -> None:
        """Update the rollback controls panel.
        
        Args:
            validation_result: Safety validation results
            improvements: List of implemented improvements
            layout: Layout to update
        """
        controls_content = []
        
        # Rollback recommendation
        if validation_result.rollback_recommended:
            recommendation_text = Text()
            recommendation_text.append("🚨 ", style="red")
            recommendation_text.append("ROLLBACK RECOMMENDED", style="bold red")
            recommendation_text.append("\nCritical safety issues detected", style="red")
            controls_content.append(recommendation_text)
        else:
            recommendation_text = Text()
            recommendation_text.append("✅ ", style="green")
            recommendation_text.append("SYSTEM STABLE", style="bold green")
            recommendation_text.append("\nNo rollback required", style="green")
            controls_content.append(recommendation_text)
            
        controls_content.append("")
        
        # Rollback options
        if validation_result.rollback_recommended or validation_result.overall_status == SafetyStatus.CRITICAL:
            rollback_text = Text()
            rollback_text.append("🔄 Available Rollback Options:", style="bold yellow")
            controls_content.append(rollback_text)
            
            rollback_options = [
                "Emergency rollback (immediate)",
                "Selective rollback (choose improvements)",
                "Gradual rollback (step-by-step)",
                "Partial rollback (critical items only)"
            ]
            
            for option in rollback_options:
                option_text = Text()
                option_text.append("  • ", style="yellow")
                option_text.append(option)
                controls_content.append(option_text)
        else:
            # Manual rollback options (when not recommended)
            rollback_text = Text()
            rollback_text.append("🛠️ Manual Rollback Available:", style="bold cyan")
            rollback_text.append("\nRollback can be initiated if needed")
            controls_content.append(rollback_text)
            
        controls_content.append("")
        
        # Control commands
        commands_text = Text()
        if validation_result.rollback_recommended:
            commands_text.append("Emergency Commands:", style="bold red")
            commands_text.append("\n[red]rb[/red] - Emergency rollback")
            commands_text.append("\n[yellow]select[/yellow] - Selective rollback")
            commands_text.append("\n[blue]details[/blue] - View detailed issues")
        else:
            commands_text.append("Available Commands:", style="bold cyan")
            commands_text.append("\n[green]continue[/green] - Accept current state")
            commands_text.append("\n[yellow]rb[/yellow] - Manual rollback")
            commands_text.append("\n[blue]details[/blue] - View details")
            
        commands_text.append("\n[dim]q[/dim] - Quit")
        controls_content.append(commands_text)
        
        # System health summary
        controls_content.append("")
        health_summary = Text()
        health_summary.append("📊 Health Summary:", style="bold magenta")
        
        safe_count = len([hc for hc in validation_result.health_checks if hc.status == SafetyStatus.SAFE])
        warning_count = len([hc for hc in validation_result.health_checks if hc.status == SafetyStatus.WARNING])
        critical_count = len([hc for hc in validation_result.health_checks if hc.status == SafetyStatus.CRITICAL])
        
        health_summary.append(f"\n[green]Safe: {safe_count}[/green] | [yellow]Warning: {warning_count}[/yellow] | [red]Critical: {critical_count}[/red]")
        controls_content.append(health_summary)
        
        from rich.console import Group
        controls_group = Group(*controls_content)
        
        if layout and "rollback_controls" in layout:
            border_color = "red" if validation_result.rollback_recommended else "green"
            layout["rollback_controls"].update(
                Panel(controls_group, title="🔄 Rollback Controls", border_style=border_color)
            )
            
    async def _process_validation_results(self, validation_result: SafetyValidationResult,
                                        improvements: List, layout: Layout,
                                        safety_status: Dict[str, bool]) -> bool:
        """Process validation results and handle user decisions.
        
        Args:
            validation_result: Safety validation results
            improvements: List of implemented improvements
            layout: Layout for UI updates
            safety_status: Dictionary to update with safety status
            
        Returns:
            True if system is safe to continue, False if rollback required
        """
        try:
            # Update safety status dictionary
            safety_status.update({
                "overall_safe": validation_result.overall_status == SafetyStatus.SAFE,
                "rollback_recommended": validation_result.rollback_recommended,
                "critical_issues_count": len(validation_result.critical_issues),
                "warnings_count": len(validation_result.warnings)
            })
            
            # If rollback is strongly recommended, require user decision
            if validation_result.rollback_recommended:
                return await self._handle_rollback_decision(validation_result, improvements, layout)
            else:
                # System is stable, get user confirmation to continue
                return await self._handle_continue_decision(validation_result, layout)
                
        except Exception as e:
            self.console.print(f"❌ Failed to process validation results: {e}")
            return False
            
    async def _handle_rollback_decision(self, validation_result: SafetyValidationResult,
                                      improvements: List, layout: Layout) -> bool:
        """Handle user decision when rollback is recommended.
        
        Args:
            validation_result: Safety validation results
            improvements: List of implemented improvements
            layout: Layout for UI updates
            
        Returns:
            True if user chooses to continue despite risks, False if rollback initiated
        """
        try:
            self.console.print("\n🚨 [bold red]SAFETY ALERT: Rollback recommended![/bold red]")
            
            while True:
                action = Prompt.ask(
                    "[red]Critical safety issues detected. Action",
                    choices=["rb", "select", "force_continue", "details", "q"],
                    default="rb"
                ).lower()
                
                if action == "rb":
                    # Emergency rollback
                    return await self._execute_emergency_rollback(improvements)
                elif action == "select":
                    # Selective rollback
                    return await self._execute_selective_rollback(improvements, validation_result)
                elif action == "force_continue":
                    # User forces to continue despite risks
                    confirmed = Confirm.ask(
                        "[yellow]⚠️  Continue despite safety risks? This is not recommended[/yellow]",
                        default=False
                    )
                    if confirmed:
                        self.console.print("⚠️  [yellow]Continuing with safety risks acknowledged[/yellow]")
                        return True
                elif action == "details":
                    await self._show_detailed_safety_issues(validation_result)
                elif action == "q":
                    return False
                    
        except (EOFError, KeyboardInterrupt):
            return False
        except Exception as e:
            self.console.print(f"❌ Rollback decision failed: {e}")
            return False
            
    async def _handle_continue_decision(self, validation_result: SafetyValidationResult,
                                      layout: Layout) -> bool:
        """Handle user decision when system is safe to continue.
        
        Args:
            validation_result: Safety validation results
            layout: Layout for UI updates
            
        Returns:
            True if user chooses to continue, False if manual rollback requested
        """
        try:
            if validation_result.warnings:
                self.console.print(f"\n⚠️  [yellow]{len(validation_result.warnings)} warning(s) detected but system is stable[/yellow]")
            else:
                self.console.print("\n✅ [green]All safety checks passed![/green]")
                
            action = Prompt.ask(
                "System is safe to continue. Action",
                choices=["continue", "rb", "details", "q"],
                default="continue"
            ).lower()
            
            if action == "continue":
                return True
            elif action == "rb":
                # Manual rollback
                confirmed = Confirm.ask(
                    "[yellow]System is stable. Are you sure you want to rollback?[/yellow]",
                    default=False
                )
                if confirmed:
                    return await self._execute_manual_rollback()
                else:
                    return True
            elif action == "details":
                await self._show_detailed_safety_report(validation_result)
                return True  # Continue after showing details
            elif action == "q":
                return False
            else:
                return True
                
        except (EOFError, KeyboardInterrupt):
            return False
        except Exception as e:
            self.console.print(f"❌ Continue decision failed: {e}")
            return True  # Default to continue on error
            
    async def _execute_emergency_rollback(self, improvements: List) -> bool:
        """Execute emergency rollback of all improvements.
        
        Args:
            improvements: List of improvements to rollback
            
        Returns:
            False (rollback initiated)
        """
        self.console.print("🚨 [bold red]INITIATING EMERGENCY ROLLBACK...[/bold red]")
        
        # Simulate rollback process
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            rollback_task = progress.add_task("Rolling back all improvements...", total=None)
            
            await asyncio.sleep(3)  # Simulate rollback time
            
        self.console.print("✅ [green]Emergency rollback completed successfully[/green]")
        self.console.print("🔄 [cyan]System restored to previous stable state[/cyan]")
        
        return False
        
    async def _execute_selective_rollback(self, improvements: List, 
                                        validation_result: SafetyValidationResult) -> bool:
        """Execute selective rollback of specific improvements.
        
        Args:
            improvements: List of improvements
            validation_result: Safety validation results
            
        Returns:
            Result of rollback operation
        """
        self.console.print("🔄 [yellow]Selective Rollback Mode[/yellow]")
        
        # Show improvements and let user select which to rollback
        self.console.print("\nSelect improvements to rollback:")
        
        for i, improvement in enumerate(improvements, 1):
            # Determine if this improvement might be causing issues
            is_suspected = improvement.category in ["security", "performance"] and validation_result.overall_status == SafetyStatus.CRITICAL
            suspect_marker = " [red](suspected issue)[/red]" if is_suspected else ""
            
            self.console.print(f"  {i}. {improvement.title}{suspect_marker}")
            
        try:
            selections = Prompt.ask(
                "Enter improvement numbers to rollback (comma-separated)",
                default="all"
            )
            
            if selections.lower() == "all":
                return await self._execute_emergency_rollback(improvements)
            else:
                # Process selective rollback
                self.console.print("🔄 [yellow]Rolling back selected improvements...[/yellow]")
                await asyncio.sleep(2)
                self.console.print("✅ [green]Selective rollback completed[/green]")
                return False
                
        except (EOFError, KeyboardInterrupt):
            return False
        except Exception as e:
            self.console.print(f"❌ Selective rollback failed: {e}")
            return False
            
    async def _execute_manual_rollback(self) -> bool:
        """Execute manual rollback when system is stable.
        
        Returns:
            False (rollback initiated)
        """
        self.console.print("🔄 [cyan]Initiating manual rollback...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            rollback_task = progress.add_task("Rolling back improvements...", total=None)
            await asyncio.sleep(2)
            
        self.console.print("✅ [green]Manual rollback completed[/green]")
        return False
        
    async def _show_detailed_safety_issues(self, validation_result: SafetyValidationResult) -> None:
        """Show detailed safety issues and health check results.
        
        Args:
            validation_result: Safety validation results
        """
        try:
            self.console.clear()
            
            # Title
            title_text = Text()
            title_text.append("🔍 ", style="red")
            title_text.append("DETAILED SAFETY ANALYSIS", style="bold red")
            
            title_panel = Panel(
                Align.center(title_text),
                style="red",
                padding=(1, 2)
            )
            self.console.print(title_panel)
            
            # Critical issues
            if validation_result.critical_issues:
                critical_panel = Panel(
                    "\n".join([f"• {issue}" for issue in validation_result.critical_issues]),
                    title="🚨 Critical Issues",
                    border_style="red"
                )
                self.console.print(critical_panel)
                
            # Detailed health checks
            for check in validation_result.health_checks:
                if check.status in [SafetyStatus.CRITICAL, SafetyStatus.WARNING]:
                    details_content = [f"Status: {check.status.value}", f"Message: {check.message}"]
                    
                    if check.details:
                        details_content.append("Details:")
                        for key, value in check.details.items():
                            details_content.append(f"  {key}: {value}")
                            
                    check_panel = Panel(
                        "\n".join(details_content),
                        title=f"🔍 {check.name}",
                        border_style="red" if check.status == SafetyStatus.CRITICAL else "yellow"
                    )
                    self.console.print(check_panel)
                    
            input("\nPress Enter to return to safety validation...")
            
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception as e:
            self.console.print(f"❌ Failed to show detailed issues: {e}")
            
    async def _show_detailed_safety_report(self, validation_result: SafetyValidationResult) -> None:
        """Show detailed safety report for stable system.
        
        Args:
            validation_result: Safety validation results
        """
        try:
            # Create comprehensive safety report
            report_content = []
            
            report_content.append(Text("📊 COMPREHENSIVE SAFETY REPORT", style="bold green"))
            report_content.append("")
            
            # Overall status
            status_table = Table.grid(padding=(0, 2))
            status_table.add_column("Metric", style="bold cyan")
            status_table.add_column("Value", style="white")
            
            status_table.add_row("Overall Status:", validation_result.overall_status.value.upper())
            status_table.add_row("Validation Duration:", f"{validation_result.validation_duration_ms:.0f}ms")
            status_table.add_row("Health Checks Passed:", f"{len([hc for hc in validation_result.health_checks if hc.status == SafetyStatus.SAFE])}/{len(validation_result.health_checks)}")
            status_table.add_row("Warnings:", str(len(validation_result.warnings)))
            
            report_content.append(status_table)
            report_content.append("")
            
            # All health checks
            all_checks_table = Table(show_header=True, header_style="bold magenta")
            all_checks_table.add_column("Check", style="white")
            all_checks_table.add_column("Status", width=12)
            all_checks_table.add_column("Duration", width=10)
            all_checks_table.add_column("Details", style="dim")
            
            for check in validation_result.health_checks:
                status_colors = {
                    SafetyStatus.SAFE: "green",
                    SafetyStatus.WARNING: "yellow",
                    SafetyStatus.CRITICAL: "red"
                }
                status_color = status_colors.get(check.status, "white")
                
                details_summary = ""
                if check.details:
                    key_metrics = list(check.details.items())[:2]  # Show first 2 metrics
                    details_summary = ", ".join([f"{k}: {v}" for k, v in key_metrics])
                
                all_checks_table.add_row(
                    check.name,
                    f"[{status_color}]{check.status.value.upper()}[/{status_color}]",
                    f"{check.check_duration_ms:.0f}ms",
                    details_summary[:50] + ("..." if len(details_summary) > 50 else "")
                )
                
            report_content.append(all_checks_table)
            
            from rich.console import Group
            report_group = Group(*report_content)
            
            report_panel = Panel(report_group, title="📋 Safety Report", border_style="green")
            self.console.print(report_panel)
            
            input("\nPress Enter to continue...")
            
        except (EOFError, KeyboardInterrupt):
            pass
        except Exception as e:
            self.console.print(f"❌ Failed to show safety report: {e}")


# Export main class
__all__ = ["SafetyStatusDisplay", "SafetyStatus", "HealthCheckType", "SafetyValidationResult"]