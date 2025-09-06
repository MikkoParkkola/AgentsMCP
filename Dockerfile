# syntax=docker/dockerfile:1

# Build stage - includes build dependencies
FROM python:3.11-slim AS builder

# Security: Run as non-root user during build
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set build-time environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better layer caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install . --user --no-warn-script-location

# Copy source code
COPY src ./src

# Production stage - minimal runtime image
FROM python:3.11-slim AS production

# Security labels and metadata
LABEL maintainer="AgentsMCP Team" \
      description="Production-ready AgentsMCP container" \
      version="1.0.0" \
      security.scan="required" \
      org.opencontainers.image.title="AgentsMCP" \
      org.opencontainers.image.description="CLI-driven MCP agent system with extensible RAG pipeline" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="AgentsMCP" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/MikkoParkkola/AgentsMCP"

# Create non-root user for production
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set production environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PATH="/home/appuser/.local/bin:$PATH" \
    AGENTSMCP_HOST="0.0.0.0" \
    AGENTSMCP_PORT="8000" \
    AGENTSMCP_LOG_LEVEL="INFO" \
    AGENTSMCP_LOG_FORMAT="json"

# Set working directory
WORKDIR /app

# Copy built application from builder stage
COPY --from=builder --chown=appuser:appgroup /root/.local /home/appuser/.local
COPY --from=builder --chown=appuser:appgroup /app/src /app/src
COPY --from=builder --chown=appuser:appgroup /app/pyproject.toml /app/pyproject.toml

# Create necessary directories with proper ownership
RUN mkdir -p /app/data /app/logs /app/temp && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use exec form for proper signal handling
CMD ["python", "-m", "uvicorn", "agentsmcp.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--access-log", "--log-config", "/dev/null"]
