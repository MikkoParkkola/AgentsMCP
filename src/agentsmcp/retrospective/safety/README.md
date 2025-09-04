# Safety Validation Framework

A comprehensive safety validation framework for the retrospective system that ensures improvements are applied safely with automatic rollback capabilities.

## Overview

The Safety Validation Framework provides production-ready safety mechanisms for validating, applying, and rolling back improvements identified by the retrospective system. It implements a multi-layered approach to safety including validation rules, health monitoring, rollback management, and orchestrated workflows.

## Key Features

### 🛡️ Safety Validation
- **Comprehensive Validation Rules**: Protects critical system components
- **Configurable Safety Levels**: Development, production, strict, and emergency modes
- **Custom Rule Support**: Extensible validation rule system
- **Batch Validation**: Efficient validation of multiple improvements

### 🏥 Health Monitoring
- **Real-time Metrics**: CPU, memory, disk usage, and application metrics
- **Baseline Comparison**: Automated health degradation detection
- **Custom Metrics**: Support for application-specific metrics
- **Trend Analysis**: Historical health data analysis

### ⏪ Rollback Management
- **Point-in-time Snapshots**: Configuration and file system backups
- **Automatic Rollback**: Triggered by health degradation
- **Manual Rollback**: Emergency rollback capabilities
- **Rollback History**: Persistent rollback point management

### 🎛️ Safety Orchestration
- **End-to-end Workflow**: Complete safety validation process
- **Timeout Management**: Configurable operation timeouts
- **Hook System**: Pre/post validation and application hooks
- **Integration Ready**: Easy integration with improvement systems

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Safety Orchestrator                         │
│  Coordinates the complete safety validation workflow            │
└─────────────────────┬───────────────────────┬───────────────────┘
                      │                       │
              ┌───────▼────────┐      ┌──────▼──────┐
              │ Safety Validator│      │Health Monitor│
              │ • Validation rules    │ • System metrics│
              │ • Risk assessment     │ • Health baselines│
              │ • Batch processing    │ • Trend analysis │
              └───────┬────────┘      └──────┬──────┘
                      │                       │
              ┌───────▼────────┐      ┌──────▼──────┐
              │Rollback Manager│      │Safety Config│
              │ • Backup/restore     │ • Safety levels │
              │ • State management   │ • Thresholds    │
              │ • Recovery ops       │ • Policies      │
              └─────────────────┘      └─────────────┘
```

## Components

### SafetyValidator
Core validation logic that evaluates improvements against safety rules:
- Critical role protection (orchestrator, process-coach, architect)
- Core functionality preservation
- Configuration syntax validation
- Security constraint checking
- Resource availability verification

### HealthMonitor
System health monitoring and baseline comparison:
- Collects system metrics (CPU, memory, disk, processes)
- Establishes health baselines
- Detects performance degradation
- Supports custom metric collectors

### RollbackManager
Manages rollback points and recovery operations:
- Creates configuration snapshots
- Backs up critical files
- Executes automatic rollbacks
- Manages rollback history and cleanup

### SafetyOrchestrator
Coordinates the complete safety workflow:
- Validation → Rollback Point → Baseline → Apply → Monitor
- Handles timeouts and error conditions
- Provides integration hooks
- Manages workflow state

### SafetyConfig
Centralized configuration management:
- Development, production, and emergency configurations
- Safety thresholds and timeouts
- Validation rule enablement
- Environment-specific settings

## Usage Examples

### Basic Integration

```python
from agentsmcp.retrospective.safety import SafetyOrchestrator, SafetyConfig
from agentsmcp.retrospective.data_models import ActionPoint

# Initialize safety framework
config = SafetyConfig.create_production_config()
orchestrator = SafetyOrchestrator(config)
await orchestrator.initialize()

# Define improvements
improvements = [
    ActionPoint(
        title="Add caching layer",
        description="Implement Redis caching for database queries",
        priority=PriorityLevel.HIGH,
        estimated_effort_hours=4.0
    )
]

# Safe improvement implementation
async def implement_improvements(improvements):
    for improvement in improvements:
        # Your implementation logic here
        pass

# Execute with safety validation
result = await orchestrator.safe_apply_improvements(
    improvements=improvements,
    implementer_function=implement_improvements,
    timeout_seconds=300
)

print(f"Safety workflow result: {result}")
await orchestrator.shutdown()
```

### Validation Only

```python
# Validate improvements without applying them
validation_results = await orchestrator.validate_improvements_only(improvements)

for improvement_id, result in validation_results.items():
    if result.passed:
        print(f"✅ {improvement_id}: Validation passed")
    else:
        print(f"❌ {improvement_id}: Validation failed")
        for issue in result.issues:
            print(f"  - {issue.severity}: {issue.message}")
```

### Manual Rollback Points

```python
# Create manual safety checkpoint
rollback_point = await orchestrator.create_safety_checkpoint(
    name="Pre-deployment checkpoint",
    description="Safety checkpoint before deploying new features"
)

# Later, if needed, trigger manual rollback
await orchestrator.trigger_manual_rollback(workflow_id)
```

## Configuration

### Safety Levels

- **MINIMAL**: Basic validation, suitable for development
- **STANDARD**: Production-ready safety checks
- **STRICT**: Maximum safety with comprehensive validation
- **EMERGENCY**: Immediate rollback mode

### Environment Configurations

```python
# Development environment
dev_config = SafetyConfig.create_development_config()

# Production environment  
prod_config = SafetyConfig.create_production_config()

# Emergency mode
emergency_config = SafetyConfig.create_emergency_config()
```

### Custom Thresholds

```python
config = SafetyConfig(
    safety_level=SafetyLevel.STRICT,
    thresholds=SafetyThresholds(
        max_response_time_increase_percent=30.0,
        max_error_rate_increase_percent=10.0,
        max_memory_increase_percent=20.0
    )
)
```

## Safety Rules

### Built-in Rules

1. **Critical Role Protection**: Prevents modifications to orchestrator, process-coach, architect
2. **Core Functionality Preservation**: Protects essential system functions
3. **Configuration Syntax**: Validates JSON/YAML configurations
4. **Security Constraints**: Reviews security-related changes
5. **Resource Availability**: Checks resource requirements

### Custom Rules

```python
from agentsmcp.retrospective.safety import ValidationRule, ValidationIssue

class CustomValidationRule(ValidationRule):
    def __init__(self):
        super().__init__("custom_rule", ValidationCategory.SAFETY, blocking=True)
    
    async def validate(self, improvement, context):
        issues = []
        # Custom validation logic
        if "dangerous_operation" in improvement.description:
            issues.append(ValidationIssue(
                issue_id="dangerous_op",
                category=self.category,
                severity=ValidationSeverity.CRITICAL,
                message="Dangerous operation detected",
                blocking=True
            ))
        return issues

# Add custom rule
validator.add_custom_rule(CustomValidationRule())
```

## Integration with Improvement System

The safety framework is designed to integrate seamlessly with the existing improvement implementation system:

```python
from agentsmcp.self_improvement import ImprovementImplementer

class SafeImprovementImplementer(ImprovementImplementer):
    def __init__(self):
        super().__init__()
        self.safety_config = SafetyConfig.create_production_config()
        self.safety_orchestrator = SafetyOrchestrator(self.safety_config)
    
    async def implement_improvements(self, improvements):
        # Use safety framework for implementation
        result = await self.safety_orchestrator.safe_apply_improvements(
            improvements=improvements,
            implementer_function=self._internal_implementation,
            timeout_seconds=600
        )
        return result
    
    async def _internal_implementation(self, improvements):
        # Your existing implementation logic
        return await super().implement_improvements(improvements)
```

## Monitoring and Observability

### Health Metrics
- System metrics: CPU, memory, disk usage
- Process metrics: memory usage, file descriptors, thread count
- Application metrics: response times, error rates (customizable)

### Rollback Tracking
- Rollback point creation and expiration
- Rollback execution success/failure
- Rollback history and cleanup

### Workflow State
- Validation results and timing
- Health monitoring duration
- Implementation success/failure

## Testing

Run the comprehensive test suite:

```bash
python test_safety_framework.py
```

The test suite validates:
- Configuration management
- Validation rule execution
- Health monitoring
- Rollback operations
- End-to-end workflows

## Best Practices

### Development
- Use `SafetyConfig.create_development_config()` for local development
- Enable dry-run mode for testing
- Set shorter monitoring durations for faster feedback

### Production
- Use `SafetyConfig.create_production_config()` for production deployments
- Enable health monitoring and automatic rollback
- Set appropriate safety thresholds for your environment
- Create manual rollback points before major changes

### Emergency Response
- Use `SafetyConfig.create_emergency_config()` for incident response
- Enables immediate rollback with minimal thresholds
- Reduced monitoring windows for faster response

## Security Considerations

- Configuration backups may contain sensitive data - ensure proper access controls
- Rollback points are stored locally by default - consider secure storage for production
- Health metrics collection respects privacy - no sensitive data is logged
- Validation rules check for potential credential exposure in logs

## Performance Impact

- Minimal overhead in normal operation
- Validation: ~10-50ms per improvement depending on rules
- Health collection: ~100-500ms depending on metrics
- Rollback point creation: ~1-5s depending on data volume
- Monitoring: Configurable interval, default 30s

## Dependencies

- `psutil`: System metrics collection
- `aiofiles`: Async file operations
- `asyncio`: Async/await support
- Standard library: `json`, `logging`, `pathlib`, etc.

## Future Enhancements

- Integration with external monitoring systems (Prometheus, Datadog)
- Advanced rollback strategies (blue-green, canary)
- Machine learning-based anomaly detection
- Distributed rollback coordination
- Custom metric aggregation and alerting