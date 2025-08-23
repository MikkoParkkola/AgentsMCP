# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system deps (if any needed later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src ./src

RUN pip install --upgrade pip && pip install .

EXPOSE 8000

# Default command runs FastAPI via uvicorn factory
CMD ["uvicorn", "agentsmcp.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
