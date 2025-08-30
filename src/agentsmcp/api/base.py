"""
Base API Components for Revolutionary Backend Architecture

Provides foundational classes and utilities for high-performance API services
with comprehensive error handling, logging, and security features.
"""

import time
import asyncio
import structlog
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
import uuid


class APIStatus(str, Enum):
    """API response status codes."""
    SUCCESS = "success"
    ERROR = "error" 
    PARTIAL = "partial"
    PROCESSING = "processing"
    DEGRADED = "degraded"


class APIError(Exception):
    """Base API error with structured error information."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow()


class APIResponse(BaseModel):
    """Standardized API response format with performance metrics."""
    
    status: APIStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class APIBase:
    """Base class for all API services with common functionality."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = structlog.get_logger(service=service_name)
        self.metrics = {}
        self.start_time = time.time()
        
    async def _execute_with_metrics(
        self, 
        operation_name: str,
        func,
        *args,
        **kwargs
    ) -> APIResponse:
        """Execute an operation with comprehensive metrics and error handling."""
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            self.logger.info(
                "operation_started",
                operation=operation_name,
                correlation_id=correlation_id
            )
            
            # Execute the operation
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            self.logger.info(
                "operation_completed",
                operation=operation_name,
                correlation_id=correlation_id,
                latency_ms=latency_ms
            )
            
            return APIResponse(
                status=APIStatus.SUCCESS,
                data=result,
                correlation_id=correlation_id,
                latency_ms=latency_ms
            )
            
        except APIError as e:
            latency_ms = (time.time() - start_time) * 1000
            
            self.logger.error(
                "operation_failed",
                operation=operation_name,
                correlation_id=correlation_id,
                error_code=e.error_code,
                error_message=e.message,
                latency_ms=latency_ms
            )
            
            return APIResponse(
                status=APIStatus.ERROR,
                error=e.message,
                error_code=e.error_code,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                metadata={"details": e.details}
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            self.logger.error(
                "operation_unexpected_error",
                operation=operation_name,
                correlation_id=correlation_id,
                error=str(e),
                latency_ms=latency_ms
            )
            
            return APIResponse(
                status=APIStatus.ERROR,
                error="Internal service error",
                error_code="INTERNAL_ERROR",
                correlation_id=correlation_id,
                latency_ms=latency_ms
            )
    
    def _validate_input(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate input data against schema."""
        # Basic validation - can be extended with more sophisticated validation
        for required_field in schema.get("required", []):
            if required_field not in data:
                raise APIError(
                    f"Missing required field: {required_field}",
                    "VALIDATION_ERROR",
                    status_code=400
                )
        return True
    
    def _check_rate_limit(self, client_id: str, limit: int = 100) -> bool:
        """Basic rate limiting check."""
        # Simplified rate limiting - production should use Redis or similar
        current_time = time.time()
        if client_id not in self.metrics:
            self.metrics[client_id] = {"requests": 0, "window_start": current_time}
        
        client_metrics = self.metrics[client_id]
        
        # Reset window if needed (1 minute windows)
        if current_time - client_metrics["window_start"] > 60:
            client_metrics["requests"] = 0
            client_metrics["window_start"] = current_time
        
        if client_metrics["requests"] >= limit:
            raise APIError(
                "Rate limit exceeded",
                "RATE_LIMIT_EXCEEDED", 
                status_code=429
            )
        
        client_metrics["requests"] += 1
        return True
    
    async def health_check(self) -> APIResponse:
        """Service health check endpoint."""
        return await self._execute_with_metrics(
            "health_check",
            lambda: {
                "service": self.service_name,
                "status": "healthy",
                "uptime_seconds": time.time() - self.start_time,
                "timestamp": datetime.utcnow().isoformat()
            }
        )