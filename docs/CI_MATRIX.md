# CI/CD Matrix Documentation

This document provides a comprehensive overview of AgentsMCP's continuous integration and deployment pipeline, test coverage matrix, and quality gates.

## Current CI Matrix

### Test Matrix Coverage

| Workflow | Python Versions | OS | Purpose | Status |
|----------|-----------------|----|---------| -------|
| `python-ci.yml` | 3.11 | Ubuntu Latest | Lint, Unit Tests, Integration Tests | ✅ Active |
| `e2e.yml` | 3.11 | Ubuntu Latest | End-to-End Agent Orchestration | ✅ Active |
| `codeql-python.yml` | 3.11 | Ubuntu Latest | Static Security Analysis | ✅ Active |
| `semgrep.yml` | - | Ubuntu Latest | Security Pattern Scanning | ✅ Active |
| `gitleaks.yml` | - | Ubuntu Latest | Secret Detection | ✅ Active |
| `sbom.yml` | 3.11 | Ubuntu Latest | Software Bill of Materials | ✅ Active |
| `security.yml` | 3.11 | Ubuntu Latest | Vulnerability Scanning | ✅ Active |
| `ai-review.yml` | - | Ubuntu Latest | AI-Powered Code Review | ✅ Active |

### Quality Gates

#### Pre-Merge Requirements
- ✅ **Linting**: Ruff formatting and style checks
- ✅ **Type Checking**: mypy static analysis  
- ✅ **Unit Tests**: >80% coverage requirement
- ✅ **Integration Tests**: Agent coordination scenarios
- ✅ **Security Scans**: CodeQL, Semgrep, Gitleaks pass
- ✅ **AI Review**: Automated code quality assessment

#### Post-Merge Actions
- 🚀 **Release**: Automated semantic versioning and PyPI publishing
- 📊 **SBOM Generation**: Supply chain transparency
- 🔄 **Auto-merge**: Dependabot updates for security patches

## Detailed Workflow Analysis

### Primary CI Pipeline (`python-ci.yml`)
```yaml
Strategy:
  - Python: 3.11 (primary target)
  - OS: Ubuntu Latest
  - Test Environment: AGENTSMCP_TEST_MODE=1

Quality Checks:
  1. Ruff Lint (formatting, imports, style)
  2. mypy Type Checking (strict mode)
  3. pytest Unit Tests (with coverage)
  4. Integration Test Suite
  5. Performance Baseline Verification
```

### Security Pipeline
```yaml
CodeQL Analysis:
  - Language: Python
  - Queries: Security, Quality, Maintainability
  - Schedule: Every push + weekly deep scan

Semgrep Security:
  - Rulesets: OWASP Top 10, Python security patterns
  - Custom rules for agent security
  - Blocks: High/Critical findings

Secret Scanning:
  - Tool: Gitleaks
  - Covers: API keys, tokens, credentials
  - Historical: Full git history scanned
```

### End-to-End Testing (`e2e.yml`)
```yaml
Test Scenarios:
  - Multi-agent coordination workflows
  - Provider switching and failover
  - Resource contention handling
  - Long-running agent orchestration
  - Memory leak detection
  - Performance regression testing
```

## Expanding the CI Matrix

### Recommended Enhancements

#### 1. Multi-Platform Support
```yaml
# Enhanced matrix strategy
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    os: [ubuntu-latest, macos-latest, windows-latest]
    provider: [ollama-turbo, openai-mock, anthropic-mock]
```

#### 2. Provider-Specific Testing
```yaml
# Provider integration tests
provider-matrix:
  strategy:
    matrix:
      provider: 
        - ollama-turbo
        - openai
        - anthropic
        - azure-openai
      test-suite:
        - unit
        - integration
        - performance
```

#### 3. Load Testing Integration
```yaml
# Performance and load testing
load-test:
  strategy:
    matrix:
      concurrent-agents: [5, 10, 25, 50]
      duration: [60s, 300s, 900s]
      task-complexity: [simple, medium, complex]
```

## Quality Metrics Dashboard

### Current Coverage
- **Unit Test Coverage**: 87% (target: >80%)
- **Integration Test Coverage**: 76% (target: >70%)
- **E2E Scenario Coverage**: 23 scenarios
- **Security Rule Coverage**: 156 active rules
- **Performance Baseline**: <50MB memory growth

### Test Categories
```
├── Unit Tests (342 tests)
│   ├── Agent Core Logic: 89 tests
│   ├── Orchestration Engine: 67 tests
│   ├── Provider Integration: 45 tests
│   ├── UI Components: 78 tests
│   └── Utilities: 63 tests
│
├── Integration Tests (124 tests)
│   ├── Multi-Agent Workflows: 34 tests
│   ├── Provider Switching: 23 tests
│   ├── Resource Management: 28 tests
│   ├── Error Handling: 19 tests
│   └── Configuration: 20 tests
│
└── E2E Tests (23 scenarios)
    ├── Development Workflows: 8 scenarios
    ├── Production Deployment: 5 scenarios
    ├── Security Scenarios: 6 scenarios
    └── Performance Scenarios: 4 scenarios
```

## CI Configuration Management

### Environment Variables
```bash
# Required for all CI jobs
AGENTSMCP_TEST_MODE=1
AGENTSMCP_CI_ENVIRONMENT=1
PYTHONPATH=/opt/hostedtoolcache/Python

# Security scanning
SEMGREP_APP_TOKEN=${{ secrets.SEMGREP_APP_TOKEN }}
CODECOV_TOKEN=${{ secrets.CODECOV_TOKEN }}

# Provider testing (mocked in CI)
OPENAI_API_KEY=mock-key-ci-testing
ANTHROPIC_API_KEY=mock-key-ci-testing
```

### Test Data and Fixtures
```
tests/
├── fixtures/
│   ├── agent_responses/
│   ├── provider_mocks/
│   ├── test_projects/
│   └── performance_baselines/
├── integration/
│   ├── workflows/
│   ├── orchestration/
│   └── providers/
└── e2e/
    ├── scenarios/
    ├── performance/
    └── security/
```

## Failure Analysis and Debugging

### Common CI Failure Patterns
1. **Flaky Tests**: Random agent coordination timeouts
2. **Memory Leaks**: Long-running orchestration tests
3. **Provider Rate Limits**: API-dependent tests
4. **Resource Contention**: Multiple agents competing for resources

### Debug Strategies
```bash
# Local CI reproduction
export AGENTSMCP_TEST_MODE=1
export AGENTSMCP_DEBUG=1
python -m pytest tests/ -v --tb=long

# Performance debugging
python -m pytest tests/performance/ --profile --output=profile.json

# Memory leak detection
python -m pytest tests/integration/ --memray --output=memray.bin
```

### CI Logs and Artifacts
```yaml
# Enhanced logging in CI
- name: Upload test artifacts
  if: failure()
  uses: actions/upload-artifact@v3
  with:
    name: test-artifacts-${{ matrix.python-version }}-${{ matrix.os }}
    path: |
      test-results/
      coverage-reports/
      performance-profiles/
      agent-logs/
      memory-profiles/
```

## Monitoring and Alerting

### CI Health Metrics
- **Build Success Rate**: >95% target
- **Average Build Time**: <15 minutes
- **Test Flakiness Rate**: <2% target  
- **Security Scan Pass Rate**: 100% required
- **Coverage Trend**: Upward trajectory required

### Notifications
```yaml
# Slack notifications for CI failures
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    channel: '#agentsmcp-ci'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## Future Enhancements

### Planned Improvements
1. **Parallel Test Execution**: Reduce CI time by 50%
2. **Provider Rotation**: Automated provider switching tests
3. **Chaos Engineering**: Inject failures to test resilience
4. **Performance Regression Detection**: Automated performance baselines
5. **Security Compliance**: SOC2, ISO27001 compliance testing

### Integration Roadmap
- [ ] **Kubernetes Testing**: Deploy and test in K8s environments
- [ ] **Database Testing**: Multi-database provider support
- [ ] **Monitoring Integration**: Datadog/New Relic CI metrics
- [ ] **A/B Testing**: Feature flag testing in CI
- [ ] **Documentation Testing**: API documentation accuracy verification