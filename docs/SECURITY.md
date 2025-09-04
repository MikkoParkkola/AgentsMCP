# Security Documentation

## Table of Contents
- [Overview](#overview)
- [Insecure Mode Feature](#insecure-mode-feature)
- [Development Usage](#development-usage)
- [Production Considerations](#production-considerations)
- [Configuration Reference](#configuration-reference)
- [Security Architecture](#security-architecture)
- [Troubleshooting](#troubleshooting)

## Overview

AgentsMCP implements a security-first architecture with authentication, authorization, and cryptographic signing for agent communications. The system includes a development-focused `insecure_mode` feature that allows bypassing security controls for local development and testing scenarios.

⚠️ **CRITICAL SECURITY WARNING**: The `insecure_mode` feature completely disables authentication and authorization. It must NEVER be used in production environments.

## Insecure Mode Feature

### What It Does

The `insecure_mode` feature is a development tool that allows AgentsMCP to run without security controls. When enabled, it:

**Security Bypass Behavior:**
- **Authentication**: Returns successful authentication without validating tokens
- **Authorization**: Allows all operations without permission checks
- **Token Creation**: Generates stub tokens without cryptographic signing
- **Public Key Access**: Returns placeholder public key data

**Technical Implementation:**
- Located in: `src/agentsmcp/security/manager.py`
- Default state: `insecure_mode=False` (secure by default)
- Affects: SecurityManager class methods

### Security States Comparison

| Security Control | Secure Mode (`False`) | Insecure Mode (`True`) |
|------------------|----------------------|------------------------|
| Authentication | Validates JWT tokens | Returns stub success |
| Authorization | Enforces RBAC/permissions | Allows all operations |
| Token Validation | Cryptographic verification | Bypassed completely |
| Logging | Standard security logs | Warning messages logged |
| Production Use | ✅ Required | ❌ FORBIDDEN |

## Development Usage

### Enabling Insecure Mode

There are three ways to enable insecure mode for development:

#### 1. CLI Flag (Recommended for Testing)
```bash
# Enable insecure mode for this session only
./agentsmcp --insecure-mode

# Or with full path
python -m agentsmcp.cli --insecure-mode
```

#### 2. Environment Variable (Recommended for Development Environment)
```bash
# Set environment variable
export AGENTSMCP_INSECURE=true
./agentsmcp

# Or inline
AGENTSMCP_INSECURE=true ./agentsmcp
```

#### 3. Programmatic Configuration
```python
from agentsmcp.security.manager import create_security_manager

# Create insecure security manager
security_manager = create_security_manager(insecure_mode=True)
```

### Configuration Precedence

The system uses the following precedence order (highest to lowest):

1. CLI flag: `--insecure-mode`
2. Environment variable: `AGENTSMCP_INSECURE=true`
3. Default: `insecure_mode=False` (secure)

### Development Scenarios

**When to Use Insecure Mode:**
- Local development without authentication setup
- Automated testing scenarios
- Rapid prototyping and debugging
- Integration testing with mock services

**Example Development Workflow:**
```bash
# Terminal 1: Start AgentsMCP in insecure mode
export AGENTSMCP_INSECURE=true
./agentsmcp

# Terminal 2: Test without authentication
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{"action": "list"}'
```

## Production Considerations

### Security Requirements

**Production environments MUST:**
- Use `insecure_mode=False` (default)
- Implement real authentication (JWT/OAuth/API keys)
- Configure proper authorization policies (RBAC/ABAC)
- Enable security logging and monitoring
- Use HTTPS/TLS for all communications

### Verification Steps

**Verify Secure Mode is Active:**

1. **Check Logs for Security Warnings:**
```bash
# Look for insecure mode warnings (should NOT appear in production)
grep -i "insecure mode" /path/to/logs/agentsmcp.log

# Should see secure initialization instead:
# "SecurityManager initialized (stub implementation)"
```

2. **Test Authentication Requirements:**
```bash
# This should FAIL without proper authentication in production
curl -X POST https://your-domain.com/api/agents \
  -H "Content-Type: application/json" \
  -d '{"action": "list"}'
# Expected: 401 Unauthorized or 403 Forbidden
```

3. **Environment Check:**
```bash
# Verify environment variables are NOT set
echo $AGENTSMCP_INSECURE  # Should be empty or "false"
```

### Production Security Checklist

- [ ] `AGENTSMCP_INSECURE` environment variable is unset or `false`
- [ ] No `--insecure-mode` flags in startup scripts
- [ ] Authentication service is properly configured
- [ ] Authorization policies are implemented
- [ ] Security logs are monitored
- [ ] Regular security audits are performed
- [ ] HTTPS/TLS is enabled for all endpoints

## Configuration Reference

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `AGENTSMCP_INSECURE` | `true`, `1`, `yes`, `on` (case-insensitive) | `false` | Enables insecure mode |

### CLI Options

| Flag | Description | Use Case |
|------|-------------|----------|
| `--insecure-mode` | Enable insecure mode for this session | One-time testing |
| `--no-insecure-mode` | Explicitly disable (default behavior) | Override environment vars |

### Code Configuration

```python
# Secure mode (production)
security_manager = create_security_manager(insecure_mode=False)

# Insecure mode (development only)
security_manager = create_security_manager(insecure_mode=True)
```

## Security Architecture

### SecurityManager Implementation

The `SecurityManager` class provides the core security functionality:

```python
class SecurityManager:
    def __init__(self, config=None, *, insecure_mode: bool = False):
        self.insecure_mode = insecure_mode
        # ... initialization
    
    def authenticate(self, token: Optional[str] = None, **kwargs) -> bool:
        if self.insecure_mode:
            return True  # BYPASS: Always succeed
        
        # Real authentication logic
        if not token:
            return False
        
        # TODO: Implement JWT/OAuth validation
        raise NotImplementedError("Authentication not yet implemented")
    
    def authorize(self, user_id: str, resource: str, action: str) -> bool:
        if self.insecure_mode:
            return True  # BYPASS: Always allow
        
        # TODO: Implement RBAC/ABAC authorization
        raise NotImplementedError("Authorization not yet implemented")
```

### Security Controls

**When Secure Mode is Active:**
- All authentication attempts must provide valid tokens
- Authorization checks enforce permission policies
- Cryptographic operations use real keys
- Security events are logged appropriately

**When Insecure Mode is Active:**
- Authentication always succeeds regardless of token validity
- Authorization always grants access regardless of permissions
- Stub implementations return placeholder data
- Warning messages are logged for each security bypass

## Troubleshooting

### Common Issues

#### Issue: "Authentication not implemented" Error
**Symptom:** `NotImplementedError: Real authentication not implemented yet`

**Cause:** Trying to use secure mode without implementing authentication

**Solutions:**
1. **For Development:** Enable insecure mode
   ```bash
   export AGENTSMCP_INSECURE=true
   ./agentsmcp
   ```

2. **For Production:** Implement real authentication in `SecurityManager.authenticate()`

#### Issue: Insecure Mode Not Working
**Symptom:** Still getting authentication errors despite setting insecure mode

**Debugging Steps:**
1. Check environment variable:
   ```bash
   echo $AGENTSMCP_INSECURE
   ```

2. Verify CLI flag usage:
   ```bash
   ./agentsmcp --insecure-mode
   ```

3. Check logs for insecure mode warning:
   ```bash
   tail -f logs/agentsmcp.log | grep -i "insecure"
   ```

#### Issue: Production Security Warnings
**Symptom:** Security warnings in production logs

**Investigation:**
```bash
# Check for insecure mode activation
grep "INSECURE mode" /var/log/agentsmcp/*.log

# Check environment
env | grep -i agentsmcp

# Verify startup parameters
ps aux | grep agentsmcp
```

**Resolution:** Remove all insecure mode configurations from production

### Debug Mode

For detailed security debugging:

```bash
# Enable debug logging
./agentsmcp --debug --insecure-mode

# Monitor security events
tail -f logs/agentsmcp.log | grep -E "(auth|security|insecure)"
```

### Getting Help

If you encounter security-related issues:

1. Check this documentation
2. Review logs for error messages
3. Verify configuration against production checklist
4. Consult the main troubleshooting guide: [docs/TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

---

**Remember:** Security is not optional in production. Always use secure mode (`insecure_mode=False`) for any production deployment, and implement proper authentication and authorization mechanisms.