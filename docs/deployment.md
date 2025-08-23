# AgentsMCP Deployment Guide

This guide covers how to deploy AgentsMCP in production environments.

## Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- PostgreSQL or Redis (for persistent storage)

## Environment Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Key configuration options:

### Server Configuration
- `AGENTSMCP_HOST`: Server bind address (default: localhost)
- `AGENTSMCP_PORT`: Server port (default: 8000)
- `AGENTSMCP_LOG_LEVEL`: Logging level (debug, info, warning, error)

### Storage Configuration
- `AGENTSMCP_STORAGE_TYPE`: Storage backend (memory, redis, postgresql)
- `AGENTSMCP_STORAGE_DATABASE_URL`: PostgreSQL connection string
- `AGENTSMCP_STORAGE_REDIS_URL`: Redis connection string

### Agent Configuration
- `AGENTSMCP_CODEX_API_KEY`: API key for Codex agent
- `AGENTSMCP_CLAUDE_API_KEY`: API key for Claude agent
- `AGENTSMCP_OLLAMA_HOST`: Ollama server URL

## Deployment Methods

### 1. Docker Compose (Recommended)

The easiest way to deploy AgentsMCP with all dependencies:

```bash
# Clone the repository
git clone <repository-url>
cd AgentsMCP

# Copy and configure environment
cp .env.example .env
# Edit .env with your configuration

# Start services
docker-compose up -d

# View logs
docker-compose logs -f agentsmcp
```

This starts:
- AgentsMCP server on port 8000
- PostgreSQL database
- Redis cache

### 2. Docker Only

Build and run just the AgentsMCP container:

```bash
# Build image
docker build -t agentsmcp .

# Run container
docker run -d \
  --name agentsmcp \
  -p 8000:8000 \
  --env-file .env \
  agentsmcp
```

### 3. Local Installation

For development or when you need more control:

```bash
# Install in development mode
pip install -e ".[dev,rag]"

# Start server
agentsmcp server start --host 0.0.0.0 --port 8000
```

## Health Checks

AgentsMCP provides multiple health check endpoints:

- `GET /health` - Basic health check
- `GET /health/ready` - Readiness probe (checks dependencies)
- `GET /health/live` - Liveness probe
- `GET /metrics` - Basic metrics

For Kubernetes:

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Scaling and Production

### Load Balancing

AgentsMCP is stateless and can be horizontally scaled:

```yaml
# docker-compose.override.yml
services:
  agentsmcp:
    deploy:
      replicas: 3
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Database Setup

For production, use PostgreSQL:

```bash
# Create database
createdb agentsmcp

# Set environment variable
AGENTSMCP_STORAGE_TYPE=postgresql
AGENTSMCP_STORAGE_DATABASE_URL=postgresql://user:pass@localhost/agentsmcp
```

### Security Considerations

1. **API Keys**: Store sensitive keys in environment variables or secrets management
2. **HTTPS**: Use a reverse proxy (nginx/traefik) with SSL/TLS termination
3. **CORS**: Configure `cors_origins` in your config for web clients
4. **Rate Limiting**: Implement rate limiting at the proxy level

### Monitoring

AgentsMCP logs are structured and can be ingested by:

- ELK Stack (Elasticsearch, Logstash, Kibana)
- Grafana + Loki
- Datadog, New Relic, etc.

Example log format:
```json
{
  "timestamp": "2024-01-01T10:00:00Z",
  "level": "INFO",
  "logger": "agentsmcp.agent_manager",
  "message": "Agent spawned successfully",
  "job_id": "abc123",
  "agent_type": "codex"
}
```

## Kubernetes Deployment

Example Kubernetes manifests:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentsmcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentsmcp
  template:
    metadata:
      labels:
        app: agentsmcp
    spec:
      containers:
      - name: agentsmcp
        image: agentsmcp:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: agentsmcp-config
        - secretRef:
            name: agentsmcp-secrets
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: agentsmcp-service
spec:
  selector:
    app: agentsmcp
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

## Troubleshooting

### Common Issues

1. **Connection Errors**: Check database/Redis connectivity
2. **Agent Failures**: Verify API keys and agent configurations
3. **Memory Issues**: Monitor job cleanup and storage usage

### Logs

Check application logs:

```bash
# Docker Compose
docker-compose logs -f agentsmcp

# Kubernetes
kubectl logs -f deployment/agentsmcp

# Local
agentsmcp server start --log-level debug
```

### CLI Debugging

Test agent functionality:

```bash
# List available agents
agentsmcp agent list

# Test agent spawn
agentsmcp agent spawn codex "Hello world"

# Check server status
curl http://localhost:8000/health
```