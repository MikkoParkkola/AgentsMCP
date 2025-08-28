"""
OpenTelemetry observability bootstrap for AgentsMCP.

* Auto‑traces all FastAPI endpoints
* Exposes Prometheus metrics at `/metrics`
* Provides helpers to create custom spans for agent lifecycle events
* Adds a correlation ID header to every request
"""
import asyncio
import os
import uuid
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Callable, Any, Awaitable, Dict, Optional

try:
    from fastapi import Request, FastAPI
    from fastapi.responses import PlainTextResponse
    from starlette.types import ASGIApp, Receive, Scope, Send
except ImportError:
    # Graceful fallback if FastAPI not available
    Request = None
    FastAPI = None

try:
    from opentelemetry import trace, metrics, baggage, context
    from opentelemetry.context import Context
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricsExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.propagate import set_global_textmap
    from opentelemetry.propagators.b3 import B3Propagator
    OTEL_AVAILABLE = True
except ImportError:
    # Graceful fallback - create no-op implementations
    OTEL_AVAILABLE = False
    trace = None
    metrics = None

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#   Configuration helpers
# --------------------------------------------------------------------------- #
def _get_env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")

# --------------------------------------------------------------------------- #
#   Correlation ID context variable
# --------------------------------------------------------------------------- #
CORRELATION_ID_CTX: ContextVar[str] = ContextVar("correlation_id", default="unknown-correlation-id")

def get_correlation_id() -> str:
    """Retrieve the correlation ID from the context variable."""
    return CORRELATION_ID_CTX.get()

def set_correlation_id(cid: str) -> None:
    """Set a new correlation ID in the context variable."""
    CORRELATION_ID_CTX.set(cid)

# --------------------------------------------------------------------------- #
#   OpenTelemetry initialization
# --------------------------------------------------------------------------- #
_initialized = False
_tracer = None
_meter = None

def init_observability() -> tuple[Any, Any]:
    """Initialize OpenTelemetry providers and exporters."""
    global _initialized, _tracer, _meter
    
    if _initialized:
        return _tracer, _meter
        
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available - observability disabled")
        return None, None

    try:
        # Service resource
        SERVICE_RESOURCE = Resource.create({
            SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME", "agentsmcp"),
            SERVICE_VERSION: os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
        })

        # Configure tracing
        trace.set_tracer_provider(TracerProvider(resource=SERVICE_RESOURCE))
        
        # Configure Jaeger exporter
        jaeger_endpoint = os.getenv("OTEL_EXPORTER_JAEGER_ENDPOINT")
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_endpoint=jaeger_endpoint
            )
        elif os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            jaeger_exporter = OTLPSpanExporter(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                insecure=_get_env_bool("OTEL_EXPORTER_OTLP_INSECURE", True)
            )
        else:
            # Console fallback
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            jaeger_exporter = ConsoleSpanExporter()

        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )

        # Configure metrics
        prometheus_exporter = PrometheusMetricsExporter()
        metrics.set_meter_provider(MeterProvider(resource=SERVICE_RESOURCE))
        metrics.get_meter_provider().add_metric_reader(prometheus_exporter)

        # Set propagator
        set_global_textmap(B3Propagator())

        _tracer = trace.get_tracer(__name__)
        _meter = metrics.get_meter(__name__)
        _initialized = True
        
        logger.info("OpenTelemetry observability initialized successfully")
        return _tracer, _meter
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        return None, None

# --------------------------------------------------------------------------- #
#   Middleware: Correlation ID
# --------------------------------------------------------------------------- #
class CorrelationIDMiddleware:
    """
    Middleware that extracts or generates correlation IDs for requests.
    """
    
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract or generate correlation ID
        headers = dict(scope.get("headers", []))
        corr_id_header = b"x-correlation-id"
        corr_id = None
        
        for name, value in headers.items():
            if name == corr_id_header:
                corr_id = value.decode("utf-8")
                break
                
        if not corr_id:
            corr_id = str(uuid.uuid4())

        # Set in context
        set_correlation_id(corr_id)
        
        # Store in scope for FastAPI access
        scope["correlation_id"] = corr_id

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                message["headers"].append((b"x-correlation-id", corr_id.encode("utf-8")))
            await send(message)

        await self.app(scope, receive, send_wrapper)

# --------------------------------------------------------------------------- #
#   Custom instrumentation helpers
# ----------- --------------------------------------------------------------------------- #
def instrument_agent_event(event_name: str, **attributes) -> None:
    """Create a span for an agent lifecycle event."""
    if not _tracer:
        return
        
    with _tracer.start_as_current_span(f"agent.{event_name}") as span:
        # Add correlation ID
        span.set_attribute("correlation_id", get_correlation_id())
        # Add custom attributes
        for key, value in attributes.items():
            span.set_attribute(key, str(value))

def instrument_agent_operation(operation_name: Optional[str] = None):
    """Decorator to instrument agent operations."""
    def decorator(func):
        op_name = operation_name or func.__name__
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not _tracer:
                # No-op if tracing not available
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
                
            with _tracer.start_as_current_span(f"operation.{op_name}") as span:
                span.set_attribute("correlation_id", get_correlation_id())
                span.set_attribute("operation", op_name)
                
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error", str(e))
                    raise
                    
        return wrapper
    return decorator

# --------------------------------------------------------------------------- #
#   FastAPI integration
# --------------------------------------------------------------------------- #
def setup_fastapi_observability(app: FastAPI) -> None:
    """Set up observability for a FastAPI application."""
    if not OTEL_AVAILABLE or not FastAPI:
        logger.warning("FastAPI or OpenTelemetry not available - skipping instrumentation")
        return

    try:
        # Initialize observability
        init_observability()
        
        # Add correlation ID middleware
        app.add_middleware(CorrelationIDMiddleware)
        
        # Auto-instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        
        logger.info("FastAPI observability setup completed")
        
    except Exception as e:
        logger.error(f"Failed to setup FastAPI observability: {e}")

# --------------------------------------------------------------------------- #
#   Logging configuration
# --------------------------------------------------------------------------- #
class CorrelationIDFilter(logging.Filter):
    """Logging filter that adds correlation ID to log records."""
    
    def filter(self, record):
        record.correlation_id = get_correlation_id()
        return True

def configure_observability_logging():
    """Configure logging to include correlation IDs."""
    # Add correlation ID filter to root logger
    correlation_filter = CorrelationIDFilter()
    root_logger = logging.getLogger()
    
    for handler in root_logger.handlers:
        handler.addFilter(correlation_filter)
    
    # Update formatter to include correlation ID
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s [%(name)s] [cid:%(correlation_id)s] %(message)s'
    )
    
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

# --------------------------------------------------------------------------- #
#   Metrics helpers
# --------------------------------------------------------------------------- #
def get_metrics_endpoint_handler():
    """Get the Prometheus metrics endpoint handler."""
    if not OTEL_AVAILABLE:
        def fallback_handler(*args, **kwargs):
            return PlainTextResponse("# Metrics not available\n", media_type="text/plain")
        return fallback_handler
    
    try:
        from prometheus_client import generate_latest, REGISTRY
        def metrics_handler(*args, **kwargs):
            return PlainTextResponse(generate_latest(REGISTRY), media_type="text/plain")
        return metrics_handler
    except ImportError:
        def fallback_handler(*args, **kwargs):
            return PlainTextResponse("# Prometheus client not available\n", media_type="text/plain")
        return fallback_handler

# --------------------------------------------------------------------------- #
#   Public API
# --------------------------------------------------------------------------- #
def get_tracer(name: str = __name__):
    """Get a tracer instance."""
    if not _initialized:
        init_observability()
    return _tracer

def get_meter(name: str = __name__):
    """Get a meter instance.""" 
    if not _initialized:
        init_observability()
    return _meter

# Convenience imports
__all__ = [
    "init_observability",
    "setup_fastapi_observability", 
    "configure_observability_logging",
    "get_correlation_id",
    "set_correlation_id",
    "instrument_agent_event",
    "instrument_agent_operation",
    "get_metrics_endpoint_handler",
    "get_tracer",
    "get_meter",
    "CorrelationIDMiddleware",
]