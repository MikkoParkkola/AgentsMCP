"""
agentsmcp.web
~~~~~~~~~~~~~

Web interface package for the AgentsMCP project.

This package provides a production-ready FastAPI-based web server that offers:

* Real-time monitoring dashboard with Server-Sent Events (SSE)
* RESTful API for agent management and task control
* JWT-based authentication and authorization
* WebSocket endpoints for bidirectional communication
* Static file serving for dashboard UI assets
* Health monitoring and metrics endpoints
* Integration with core AgentsMCP components

The web server is designed to be self-contained and can gracefully handle
missing optional dependencies by disabling features or providing fallback
implementations.

Main Components:
    server: FastAPI application with all routes and middleware
    
Example usage::

    from agentsmcp.web.server import app
    import uvicorn
    
    # Run development server
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    
    # Or in production
    uvicorn.run(
        "agentsmcp.web.server:app", 
        host="0.0.0.0", 
        port=8000,
        workers=4
    )

Environment Configuration:
    The server can be configured via environment variables with the 
    AGENTSMCP_ prefix:
    
    * AGENTSMCP_HOST: Server bind host (default: "0.0.0.0")
    * AGENTSMCP_PORT: Server port (default: 8000)  
    * AGENTSMCP_JWT_SECRET_KEY: JWT signing secret (default: "change-me-please")
    * AGENTSMCP_CORS_ORIGINS: Allowed CORS origins (default: ["*"])
    * AGENTSMCP_STATIC_DIR: Static files directory (default: "./static")

Optional Dependencies:
    * sse-starlette or fastapi-sse: Server-Sent Events support
    * slowapi: Rate limiting and WebSocket support
    * python-jose[cryptography]: JWT token handling
    * structlog: Structured logging
    * prometheus-client: Metrics export

The server will start without these dependencies but some features will be
disabled with appropriate log messages.
"""

from .server import app

__all__ = ["app"]