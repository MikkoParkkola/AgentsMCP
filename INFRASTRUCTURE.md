# AgentsMCP Infrastructure Guide

This document provides comprehensive guidance for deploying AgentsMCP in production using Docker and Kubernetes.

## 🏗️ Architecture Overview

AgentsMCP uses a production-ready containerized architecture with:

- **Multi-stage Docker builds** for optimized security and size
- **Kubernetes manifests** for scalable orchestration
- **Helm charts** for configurable deployments
- **Enterprise security** with non-root containers, network policies, and RBAC
- **Auto-scaling** with HPA and VPA support
- **Health monitoring** with comprehensive probes and metrics

## 📋 Prerequisites

### Required Tools
- Docker 20.10+
- Kubernetes 1.24+
- Helm 3.8+
- kubectl configured for your cluster

### Optional Tools (enhance validation)
- yamllint
- kubeval
- conftest
- Prometheus (for monitoring)

## 🚀 Quick Start

### 1. Validate Infrastructure
```bash
./scripts/validate-infrastructure.sh
```

### 2. Build and Push Docker Image
```bash
# Build the image
docker build -t your-registry.io/agentsmcp:1.0.0 .

# Push to registry
docker push your-registry.io/agentsmcp:1.0.0
```

### 3. Configure Secrets
```bash
# Create namespace
kubectl create namespace agentsmcp

# Create secrets (update with your values)
kubectl create secret generic agentsmcp-secrets \
  --from-literal=JWT_SECRET_KEY="your-jwt-secret" \
  --from-literal=OPENAI_API_KEY="your-openai-key" \
  --from-literal=ANTHROPIC_API_KEY="your-anthropic-key" \
  -n agentsmcp
```

### 4. Deploy with Helm
```bash
# Development
helm install agentsmcp ./charts/agentsmcp \
  -f charts/agentsmcp/values-development.yaml \
  -n agentsmcp

# Production
helm install agentsmcp ./charts/agentsmcp \
  -f charts/agentsmcp/values-production.yaml \
  --set image.registry=your-registry.io \
  --set image.tag=1.0.0 \
  -n agentsmcp
```

## 🐳 Docker Configuration

### Production Dockerfile Features
- **Multi-stage build** reduces final image size
- **Non-root user** (UID 1000) for security
- **Distroless base** minimizes attack surface
- **Health checks** for container orchestration
- **Proper signal handling** for graceful shutdown
- **Security labels** for compliance scanning

### Build Arguments
```bash
docker build \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  -t agentsmcp:latest .
```

### Security Scanning
```bash
# Scan with Docker Scout
docker scout cves agentsmcp:latest

# Scan with Trivy
trivy image agentsmcp:latest
```

## ⚓ Kubernetes Deployment

### Resource Structure
```
k8s/
├── deployment.yaml      # Main application deployment
├── service.yaml         # Service definitions (LoadBalancer, ClusterIP, Headless)
├── configmap.yaml       # Configuration management
├── secret.yaml          # Secret templates
├── rbac.yaml           # ServiceAccount, Role, RoleBinding
├── networkpolicy.yaml   # Network security policies
└── hpa.yaml            # Horizontal Pod Autoscaler + VPA
```

### Security Features
- **Pod Security Standards** (restricted profile)
- **Network Policies** (default-deny with explicit allows)
- **RBAC** with minimal required permissions
- **Security Contexts** with non-root, read-only filesystem
- **Resource Limits** to prevent resource exhaustion

### Deployment Commands
```bash
# Deploy all manifests
kubectl apply -f k8s/ -n agentsmcp

# Check deployment status
kubectl get pods,svc,ingress -n agentsmcp

# View logs
kubectl logs -l app=agentsmcp -n agentsmcp -f

# Scale deployment
kubectl scale deployment agentsmcp --replicas=5 -n agentsmcp
```

## 📦 Helm Charts

### Chart Structure
```
charts/agentsmcp/
├── Chart.yaml                 # Chart metadata
├── values.yaml               # Default configuration
├── values-production.yaml    # Production overrides
├── values-staging.yaml       # Staging overrides
├── values-development.yaml   # Development overrides
├── templates/
│   ├── _helpers.tpl          # Template helpers
│   ├── deployment.yaml       # Deployment template
│   ├── service.yaml          # Service templates
│   ├── configmap.yaml        # ConfigMap template
│   ├── secret.yaml           # Secret templates
│   ├── rbac.yaml             # RBAC templates
│   ├── hpa.yaml              # Autoscaler templates
│   ├── ingress.yaml          # Ingress template
│   ├── networkpolicy.yaml    # NetworkPolicy template
│   └── tests/
│       └── test-connection.yaml # Connection tests
└── NOTES.txt                 # Post-install instructions
```

### Environment-Specific Deployments

#### Development
```bash
helm install agentsmcp ./charts/agentsmcp \
  -f charts/agentsmcp/values-development.yaml \
  -n agentsmcp-dev --create-namespace
```

#### Staging
```bash
helm install agentsmcp ./charts/agentsmcp \
  -f charts/agentsmcp/values-staging.yaml \
  --set image.tag=staging-latest \
  -n agentsmcp-staging --create-namespace
```

#### Production
```bash
helm install agentsmcp ./charts/agentsmcp \
  -f charts/agentsmcp/values-production.yaml \
  --set image.registry=your-prod-registry.io \
  --set image.tag=1.0.0 \
  --set ingress.hosts[0].host=api.agentsmcp.com \
  -n agentsmcp-prod --create-namespace
```

### Helm Commands
```bash
# Upgrade deployment
helm upgrade agentsmcp ./charts/agentsmcp \
  -f charts/agentsmcp/values-production.yaml \
  -n agentsmcp

# Rollback
helm rollback agentsmcp 1 -n agentsmcp

# Test deployment
helm test agentsmcp -n agentsmcp

# Uninstall
helm uninstall agentsmcp -n agentsmcp
```

## 🔧 Configuration Management

### Environment Variables
Key configuration via environment variables:

- `AGENTSMCP_ENVIRONMENT`: deployment environment
- `AGENTSMCP_LOG_LEVEL`: logging level (DEBUG, INFO, WARNING, ERROR)
- `AGENTSMCP_LOG_FORMAT`: log format (json, text)
- `AGENTSMCP_PROMETHEUS_ENABLED`: enable metrics collection
- `AGENTSMCP_RATE_LIMIT_ENABLED`: enable rate limiting

### ConfigMap Configuration
Application configuration via YAML:
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "https://your-domain.com"

agents:
  coding:
    type: "openai"
    model: "gpt-4"
    provider: "openai"
    timeout: 300

providers:
  openai:
    api_base: "https://api.openai.com/v1"
```

### Secret Management
Secrets are managed separately from configuration:
- JWT signing keys
- API keys for external services
- Database credentials
- TLS certificates

Production secret management options:
- AWS Secrets Manager + External Secrets Operator
- HashiCorp Vault + Vault Agent
- Azure Key Vault + CSI Secret Store
- Kubernetes native secrets with encryption at rest

## 📊 Monitoring and Observability

### Health Endpoints
- `/health` - Basic health check
- `/health/ready` - Readiness probe (checks dependencies)
- `/health/live` - Liveness probe (basic service health)

### Metrics
Prometheus metrics available at `/metrics`:
- HTTP request metrics
- Application performance metrics
- Resource utilization metrics
- Custom business metrics

### Monitoring Stack Integration
```yaml
# ServiceMonitor for Prometheus
monitoring:
  serviceMonitor:
    enabled: true
    interval: 30s
    path: /metrics

# PrometheusRule for alerts
prometheusRule:
  enabled: true
  rules:
    - alert: AgentsMCPDown
      expr: up{job="agentsmcp"} == 0
      for: 1m
```

### Logging
Structured logging with configurable output:
- JSON format for production (machine-readable)
- Text format for development (human-readable)
- Configurable log levels
- Request tracing with correlation IDs

## 🔒 Security Best Practices

### Container Security
- ✅ Non-root user (UID 1000)
- ✅ Read-only root filesystem
- ✅ Dropped all Linux capabilities
- ✅ No privilege escalation
- ✅ Security context constraints
- ✅ Multi-stage builds
- ✅ Regular base image updates

### Kubernetes Security
- ✅ Pod Security Standards (restricted)
- ✅ Network Policies (default-deny)
- ✅ RBAC with minimal permissions
- ✅ Resource quotas and limits
- ✅ Security context enforcement
- ✅ Service mesh compatibility

### Network Security
- ✅ Default-deny network policies
- ✅ Explicit ingress/egress rules
- ✅ TLS encryption in transit
- ✅ mTLS for service-to-service communication
- ✅ Rate limiting and DDoS protection

### Secret Management
- ✅ No hardcoded secrets
- ✅ Encrypted secrets at rest
- ✅ Secret rotation policies
- ✅ External secret management integration
- ✅ Principle of least privilege access

## ⚖️ Auto-scaling Configuration

### Horizontal Pod Autoscaler (HPA)
```yaml
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

### Vertical Pod Autoscaler (VPA)
```yaml
vpa:
  enabled: true
  updateMode: "Auto"
  minAllowed:
    cpu: 100m
    memory: 128Mi
  maxAllowed:
    cpu: 2000m
    memory: 4Gi
```

### Custom Metrics Scaling
- HTTP requests per second
- Queue depth
- Custom business metrics
- External metrics integration

## 🌐 Ingress and Load Balancing

### Ingress Configuration
```yaml
ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "1000"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: api.agentsmcp.com
      paths:
        - path: /
          pathType: Prefix
```

### Service Types
- **LoadBalancer**: External access with cloud LB
- **ClusterIP**: Internal service discovery
- **Headless**: Direct pod access for clustering

## 🔄 CI/CD Integration

### GitOps Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy AgentsMCP
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build and push
        run: |
          docker build -t ${{ secrets.REGISTRY }}/agentsmcp:${{ github.ref_name }} .
          docker push ${{ secrets.REGISTRY }}/agentsmcp:${{ github.ref_name }}
      
      - name: Deploy to production
        run: |
          helm upgrade agentsmcp ./charts/agentsmcp \
            -f charts/agentsmcp/values-production.yaml \
            --set image.tag=${{ github.ref_name }} \
            -n agentsmcp-prod
```

### Validation Pipeline
```bash
# Automated validation in CI
./scripts/validate-infrastructure.sh
```

## 🧪 Testing and Validation

### Infrastructure Tests
```bash
# Run all validation tests
./scripts/validate-infrastructure.sh

# Test Docker build
docker build -t agentsmcp:test .

# Test Kubernetes manifests
kubectl apply --dry-run=client -f k8s/

# Test Helm templates
helm template agentsmcp ./charts/agentsmcp

# Test with different environments
helm template agentsmcp ./charts/agentsmcp -f values-production.yaml
```

### Connection Tests
```bash
# Run Helm tests
helm test agentsmcp -n agentsmcp

# Manual health check tests
kubectl port-forward svc/agentsmcp 8080:8000 -n agentsmcp
curl http://localhost:8080/health
curl http://localhost:8080/health/ready
curl http://localhost:8080/health/live
```

## 🚨 Troubleshooting

### Common Issues

#### Pod Won't Start
```bash
# Check pod status
kubectl describe pod <pod-name> -n agentsmcp

# Check events
kubectl get events -n agentsmcp --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n agentsmcp
```

#### Health Checks Failing
```bash
# Test health endpoints
kubectl port-forward svc/agentsmcp 8080:8000 -n agentsmcp
curl -v http://localhost:8080/health

# Check probe configuration
kubectl describe pod <pod-name> -n agentsmcp
```

#### Resource Issues
```bash
# Check resource usage
kubectl top pods -n agentsmcp
kubectl describe hpa agentsmcp-hpa -n agentsmcp

# Check resource quotas
kubectl describe resourcequota -n agentsmcp
```

#### Network Issues
```bash
# Check network policies
kubectl describe networkpolicy -n agentsmcp

# Test connectivity
kubectl run debug --image=curlimages/curl -it --rm -- sh
# Inside pod: curl http://agentsmcp:8000/health
```

### Debug Mode
Enable debug mode for troubleshooting:
```yaml
debug:
  enabled: true
  command:
    - /bin/sh
    - -c
    - "while true; do sleep 3600; done"
```

### Log Analysis
```bash
# Structured log analysis
kubectl logs -l app=agentsmcp -n agentsmcp | jq '.'

# Error filtering
kubectl logs -l app=agentsmcp -n agentsmcp | grep ERROR

# Performance analysis
kubectl logs -l app=agentsmcp -n agentsmcp | grep "response_time"
```

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [CNCF Security Best Practices](https://github.com/cncf/sig-security)
- [Prometheus Monitoring](https://prometheus.io/docs/)

## 🤝 Contributing

When contributing to infrastructure:

1. Run validation tests: `./scripts/validate-infrastructure.sh`
2. Test with all environment configurations
3. Update documentation for changes
4. Follow security best practices
5. Test rollback procedures

## 📄 License

This infrastructure configuration is part of the AgentsMCP project and follows the same licensing terms.

---

🚀 **Ready to deploy AgentsMCP in production!** Follow the quick start guide and customize the configuration for your environment.