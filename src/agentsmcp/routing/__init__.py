"""
agentsmcp.routing
-----------------
Entry‑point for the routing subsystem.

All public helpers for the new OpenRouter integration live here:

    * `client`   – Async HTTP wrapper around OpenRouter.ai.
    * `models`   – Static (or JSON‑loadable) database of model capabilities.
    * `selector` – Intializes the routing decision engine.
    * `tracker`  – Collects per‑request cost/latency metrics.
    * `optimizer`– Implements the cost‑optimisation algorithm.

A consumer can simply::

    from agentsmcp.routing import openrouter_request

and the request will be routed through the full stack.
"""

from .client import OpenRouterClient, OpenRouterError
from .models import ModelDB, Model
from .selector import ModelSelector, TaskSpec, SelectionResult
from .tracker import MetricsTracker, RequestMetrics
from .optimizer import CostOptimizer, OptimizationResult

__version__ = "1.0.0"

# Public API -------------------------------------------------------------- #
__all__ = [
    "OpenRouterClient",
    "OpenRouterError", 
    "ModelDB",
    "Model",
    "ModelSelector",
    "TaskSpec", 
    "SelectionResult",
    "MetricsTracker",
    "RequestMetrics",
    "CostOptimizer",
    "OptimizationResult",
]