"""
Cost‑tracking and performance‑monitoring subsystem for the AgentsMCP routing
architecture.

The module exposes two public objects:

    * :class:`RequestMetrics` – a data container that describes a single
      request/response cycle.
    * :class:`MetricsTracker` – a thread‑safe, in‑memory counter that
      aggregates :class:`RequestMetrics` objects and exposes a variety
      of reporting / alerting APIs.

The tracker is intentionally lightweight – it stores the most recent
metrics in memory and writes them to disk on demand.  For a production
deployment the :class:`MetricsTracker` can be subclassed and plugged into a
real database or a time series store; the public API remains unchanged.
"""

from __future__ import annotations

import logging
import os
import json
import datetime
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union, Iterable, Any
from collections import defaultdict, Counter

__all__ = [
    "RequestMetrics",
    "MetricsTracker",
]

# --------------------------------------------------------------------------- #
# 1. Logging configuration
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 2. RequestMetrics dataclass
# --------------------------------------------------------------------------- #
@dataclass
class RequestMetrics:
    """
    A single request/response snapshot for cost and performance analytics.

    Parameters
    ----------
    model : str
        The model identifier that handled the request.
    request_ts : datetime.datetime
        Timestamp of when the request was initiated.
    tokens_prompt : int
        Number of tokens sent to the model.
    tokens_completion : int
        Number of tokens returned by the model.
    response_time : float
        Wall‑clock time in seconds to receive the full answer.
    provider : str
        The LLM provider (e.g. "openrouter").
    success : bool
        True iff the request finished without an API error.
    task_type : Optional[str]
        A user‑defined classification of the request (e.g. "chat",
        "classification").
    session_id : Optional[str]
        Identifier for the current user session.
    user_id : Optional[str]
        Identifier for the end user.
    response_quality : Optional[Dict[str, float]]
        Optional quality metrics returned from the downstream
        evaluation pipeline.
    error_code : Optional[str]
        OpenRouter error code (if any) – e.g. "limit_exceeded".
    """

    model: str
    request_ts: datetime.datetime
    tokens_prompt: int
    tokens_completion: int
    response_time: float
    provider: str
    success: bool
    task_type: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    response_quality: Optional[Dict[str, float]] = None
    error_code: Optional[str] = None

    # -------------------------------------------------------------------- #
    #  Derived read‑only properties
    # -------------------------------------------------------------------- #
    @property
    def total_tokens(self) -> int:
        return self.tokens_prompt + self.tokens_completion

    @property
    def cost(self) -> float:
        """
        Compute the request cost in USD based on the provider's pricing schema.
        This uses a simplified calculation and should be enhanced with real pricing data.
        """
        # Default pricing for OpenRouter - simplified calculation
        # Production code should fetch from ModelDB or pricing service
        rate_per_1k = 0.002  # $0.002 per 1k tokens as baseline
        return (self.total_tokens / 1000.0) * rate_per_1k

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON‑serialisable representation of the metrics."""
        d = asdict(self)
        d["request_ts"] = self.request_ts.isoformat()
        d["cost"] = self.cost
        d["total_tokens"] = self.total_tokens
        return d


# --------------------------------------------------------------------------- #
# 3. MetricsTracker
# --------------------------------------------------------------------------- #
class MetricsTracker:
    """
    Thread‑safe aggregator for RequestMetrics that maintains
    running totals/averages, supports time‑based filtering, and can export
    to JSON or Prometheus.
    """

    def __init__(self,
                 persistence_path: Optional[str] = None,
                 budget_limits: Optional[Dict[str, float]] = None):
        """
        Initialise the tracker.

        Parameters
        ----------
        persistence_path : Optional[str]
            Path to a JSON file where snapshots are written.
            If None, snapshots are kept only in memory.
        budget_limits : Optional[Dict[str, float]]
            Mapping from budget key (e.g. ``"provider:openrouter"``) to a
            monetary limit threshold.  When a threshold is crossed an
            event is logged.
        """
        self._metrics: List[RequestMetrics] = []
        self._persistence_path = persistence_path
        self._budget_limits = budget_limits or {}
        self._budget_violated = set()

        # Pre‑calculated aggregates for fast query
        self._by_model = Counter()
        self._by_provider = Counter()
        self._by_task = Counter()
        self._by_user = Counter()
        self._total_cost = 0.0
        self._total_tokens = 0
        self._total_response_time = 0.0
        self._total_requests = 0
        self._success_counter = Counter()
        self._error_counter = Counter()

        # Chronological index by day
        self._daily_index: Dict[str, List[RequestMetrics]] = defaultdict(list)

    # ----------------------------------------------------------------------- #
    # Core API – Recording
    # ----------------------------------------------------------------------- #
    def record(self, request_metrics: RequestMetrics) -> None:
        """
        Record a new request metric.

        This operation updates all running totals, aggregates and
        triggers any budget alerts.

        Parameters
        ----------
        request_metrics : RequestMetrics
            Data to be incorporated.
        """
        try:
            self._metrics.append(request_metrics)
            self._total_requests += 1
            self._total_cost += request_metrics.cost
            self._total_tokens += request_metrics.total_tokens
            self._total_response_time += request_metrics.response_time

            self._by_model[request_metrics.model] += 1
            self._by_provider[request_metrics.provider] += 1
            if request_metrics.task_type:
                self._by_task[request_metrics.task_type] += 1
            if request_metrics.user_id:
                self._by_user[request_metrics.user_id] += 1

            if request_metrics.success:
                self._success_counter[request_metrics.model] += 1
            else:
                self._error_counter[request_metrics.error_code or "unknown"] += 1

            # Time‑based index
            day = request_metrics.request_ts.date().isoformat()
            self._daily_index[day].append(request_metrics)

            # Budget alerting
            self._check_budget_alerts(request_metrics)

            logger.debug(f"Recorded metrics for {request_metrics.model}: "
                        f"${request_metrics.cost:.4f}, {request_metrics.total_tokens} tokens")

        except Exception:
            logger.exception("Failed to record request metrics.")

    # ----------------------------------------------------------------------- #
    # Budget Alerts
    # ----------------------------------------------------------------------- #
    def _check_budget_alerts(self, request_metrics: RequestMetrics) -> None:
        """Check if any budget limits have been exceeded."""
        key = f"{request_metrics.provider}:{request_metrics.model}"
        if key not in self._budget_limits:
            return

        threshold = self._budget_limits[key]
        if self._total_cost >= threshold and key not in self._budget_violated:
            self._budget_violated.add(key)
            logger.warning(
                f"Budget limit exceeded for {key} – "
                f"cost ${self._total_cost:.3f} >= threshold ${threshold:.3f}"
            )

    # ----------------------------------------------------------------------- #
    # Query helpers
    # ----------------------------------------------------------------------- #
    def _filter_by_time(self,
                        start: Optional[datetime.datetime],
                        end: Optional[datetime.datetime]) -> List[RequestMetrics]:
        """Filter metrics by time range."""
        if not start and not end:
            return self._metrics[:]
            
        filtered = []
        for m in self._metrics:
            if start and m.request_ts < start:
                continue
            if end and m.request_ts > end:
                continue
            filtered.append(m)
        return filtered

    def daily_stats(self, days: int = 7) -> Dict[str, Dict[str, Any]]:
        """Get daily statistics for the last N days."""
        stats = {}
        today = datetime.date.today()
        
        for i in range(days):
            day = (today - datetime.timedelta(days=i)).isoformat()
            day_metrics = self._daily_index.get(day, [])
            
            if day_metrics:
                total_cost = sum(m.cost for m in day_metrics)
                total_tokens = sum(m.total_tokens for m in day_metrics)
                avg_response = statistics.mean(m.response_time for m in day_metrics)
                success_rate = sum(1 for m in day_metrics if m.success) / len(day_metrics)
            else:
                total_cost = total_tokens = avg_response = success_rate = 0
                
            stats[day] = {
                "requests": len(day_metrics),
                "cost": round(total_cost, 4),
                "tokens": total_tokens,
                "avg_response_time": round(avg_response, 3),
                "success_rate": round(success_rate * 100, 2)
            }
            
        return stats

    # ----------------------------------------------------------------------- #
    # Report generation
    # ----------------------------------------------------------------------- #
    def generate_report(self,
                        start: Optional[datetime.datetime] = None,
                        end: Optional[datetime.datetime] = None,
                        include_metrics: bool = False) -> Dict[str, Any]:
        """
        Generate a usage report summarising cost, tokens, latency, success rates,
        and per‑task breakdown for an optional time window.

        Parameters
        ----------
        start : Optional[datetime.datetime]
            Start filter – inclusive.  If None, from the earliest data.
        end : Optional[datetime.datetime]
            End filter – inclusive.  If None, to the latest data.
        include_metrics : bool
            If True, the raw RequestMetrics list is included in the
            output (useful for JSON export but can blow up memory).

        Returns
        -------
        Dict[str, Any]
            Structured representation of the report.
        """
        filt_metrics = self._filter_by_time(start, end)

        if not filt_metrics:
            return {
                "time_window": {
                    "start": start.isoformat() if start else None,
                    "end": end.isoformat() if end else None,
                },
                "totals": {
                    "requests": 0,
                    "total_cost_usd": 0.0,
                    "total_tokens": 0,
                    "average_response_time_sec": 0.0,
                    "success_rate_percent": 0.0,
                },
                "by_model": {},
                "by_task_type": {},
                "by_user_id": {},
                "error_codes": {},
            }

        # Totals
        total_cost = sum(m.cost for m in filt_metrics)
        total_tokens = sum(m.total_tokens for m in filt_metrics)
        avg_response = statistics.mean(m.response_time for m in filt_metrics)
        success_rate = sum(1 for m in filt_metrics if m.success) / len(filt_metrics)

        # Aggregations
        by_model = Counter(m.model for m in filt_metrics)
        by_task = Counter(m.task_type for m in filt_metrics if m.task_type)
        by_user = Counter(m.user_id for m in filt_metrics if m.user_id)
        error_codes = Counter(m.error_code for m in filt_metrics if m.error_code and not m.success)

        report = {
            "time_window": {
                "start": start.isoformat() if start else None,
                "end": end.isoformat() if end else None,
            },
            "totals": {
                "requests": len(filt_metrics),
                "total_cost_usd": round(total_cost, 4),
                "total_tokens": total_tokens,
                "average_response_time_sec": round(avg_response, 3),
                "success_rate_percent": round(success_rate * 100, 2),
            },
            "by_model": dict(by_model),
            "by_task_type": dict(by_task),
            "by_user_id": dict(by_user),
            "error_codes": dict(error_codes),
        }

        if include_metrics:
            report["metrics"] = [m.to_dict() for m in filt_metrics]
            
        return report

    # ----------------------------------------------------------------------- #
    # Cost optimization
    # ----------------------------------------------------------------------- #
    def cost_optimization_recommendations(self) -> List[str]:
        """
        Analyze current data and return cost optimization recommendations.

        Returns
        -------
        List[str]
            Human‑readable recommendation messages.
        """
        if not self._metrics:
            return ["No metrics available for analysis."]

        recs: List[str] = []

        # Analyze model usage patterns
        model_stats = defaultdict(lambda: {"cost": 0.0, "tokens": 0, "requests": 0})

        for m in self._metrics:
            stats = model_stats[m.model]
            stats["cost"] += m.cost
            stats["tokens"] += m.total_tokens
            stats["requests"] += 1

        # Calculate cost per token for each model
        cost_per_token = {}
        for model, stats in model_stats.items():
            if stats["tokens"] > 0:
                cost_per_token[model] = stats["cost"] / stats["tokens"]
            else:
                cost_per_token[model] = 0.0

        if cost_per_token:
            # Find most and least efficient models
            cheapest_model = min(cost_per_token, key=cost_per_token.get)
            most_expensive = max(cost_per_token, key=cost_per_token.get)
            
            cheapest_cost = cost_per_token[cheapest_model]
            expensive_cost = cost_per_token[most_expensive]

            if expensive_cost > cheapest_cost * 1.5:  # 50% more expensive
                potential_savings = (expensive_cost - cheapest_cost) * model_stats[most_expensive]["tokens"]
                recs.append(
                    f"Consider switching from '{most_expensive}' to '{cheapest_model}' "
                    f"for potential savings of ${potential_savings:.3f} "
                    f"({((expensive_cost - cheapest_cost) / expensive_cost * 100):.1f}% reduction)"
                )

        # Check for high-cost, low-usage models
        for model, stats in model_stats.items():
            if stats["requests"] < 5 and stats["cost"] > self._total_cost * 0.1:
                recs.append(
                    f"Model '{model}' has low usage ({stats['requests']} requests) "
                    f"but accounts for {stats['cost'] / self._total_cost * 100:.1f}% of total cost. "
                    f"Consider if this model is necessary."
                )

        # Check average response times
        slow_models = []
        if self._total_requests > 0:
            avg_response = self._total_response_time / self._total_requests
            for model in model_stats:
                model_metrics = [m for m in self._metrics if m.model == model]
                if model_metrics:
                    model_avg = statistics.mean(m.response_time for m in model_metrics)
                    if model_avg > avg_response * 2:  # 2x slower than average
                        slow_models.append((model, model_avg))
        
        if slow_models:
            recs.append(
                f"Models with slow response times: "
                + ", ".join(f"'{model}' ({time:.2f}s)" for model, time in slow_models)
                + ". Consider faster alternatives."
            )

        return recs or ["No specific optimization recommendations at this time."]

    # ----------------------------------------------------------------------- #
    # Persistence & Exports
    # ----------------------------------------------------------------------- #
    def persist_snapshot(self,
                         path: Optional[str] = None,
                         include_metrics: bool = False) -> None:
        """
        Persist a snapshot of all metrics to a JSON file.

        Parameters
        ----------
        path : Optional[str]
            File path to write.  If None, uses persistence_path.
        include_metrics : bool
            If True, all RequestMetrics objects are written.
        """
        if path is None:
            path = self._persistence_path
            
        if path is None:
            logger.warning("No persistence path configured – skipping snapshot.")
            return

        data = {
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "total_cost_usd": round(self._total_cost, 4),
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "average_response_time": (
                self._total_response_time / max(self._total_requests, 1)
            ),
            "by_model": dict(self._by_model),
            "by_provider": dict(self._by_provider),
            "by_task_type": dict(self._by_task),
            "by_user_id": dict(self._by_user),
            "success_counts": dict(self._success_counter),
            "error_codes": dict(self._error_counter),
        }
        
        if include_metrics:
            data["metrics"] = [m.to_dict() for m in self._metrics]

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            logger.info("Metrics snapshot persisted to %s", path)
        except Exception:
            logger.exception("Failed to persist metrics snapshot to %s", path)

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns
        -------
        str
            Prometheus-compatible metrics text.
        """
        lines = [
            "# HELP agentsmcp_total_cost_usd Total cost of all requests in USD",
            "# TYPE agentsmcp_total_cost_usd gauge",
            f"agentsmcp_total_cost_usd {self._total_cost:.6f}",
            "",
            "# HELP agentsmcp_total_requests Total number of requests processed",
            "# TYPE agentsmcp_total_requests counter",
            f"agentsmcp_total_requests {self._total_requests}",
            "",
            "# HELP agentsmcp_total_tokens Total tokens processed",
            "# TYPE agentsmcp_total_tokens counter", 
            f"agentsmcp_total_tokens {self._total_tokens}",
            "",
            "# HELP agentsmcp_requests_by_model Number of requests per model",
            "# TYPE agentsmcp_requests_by_model counter",
        ]

        # Per-model metrics
        for model, count in self._by_model.items():
            model_safe = model.replace("-", "_").replace("/", "_")
            lines.append(f'agentsmcp_requests_by_model{{model="{model}"}} {count}')

        lines.extend([
            "",
            "# HELP agentsmcp_success_rate Success rate by model",
            "# TYPE agentsmcp_success_rate gauge",
        ])

        for model, successes in self._success_counter.items():
            total_for_model = self._by_model[model]
            rate = successes / total_for_model if total_for_model > 0 else 0
            lines.append(f'agentsmcp_success_rate{{model="{model}"}} {rate:.3f}')

        return "\n".join(lines)

    def as_json(self, 
                start: Optional[datetime.datetime] = None,
                end: Optional[datetime.datetime] = None) -> str:
        """
        Produce a JSON string for web dashboard consumption.

        Parameters
        ----------
        start, end : Optional[datetime.datetime]
            Optional time window for the report.

        Returns
        -------
        str
            Pretty‑printed JSON.
        """
        report = self.generate_report(start=start, end=end, include_metrics=False)
        return json.dumps(report, indent=2)

    # ----------------------------------------------------------------------- #
    # Budget management
    # ----------------------------------------------------------------------- #
    def set_budget_alert(self, key: str, threshold_usd: float) -> None:
        """
        Register a new budget alert.

        Parameters
        ----------
        key : str
            Identifier like "provider:model" or "user:userid".
        threshold_usd : float
            Amount in USD after which a warning is logged.
        """
        self._budget_limits[key] = threshold_usd
        self._budget_violated.discard(key)
        logger.info(f"Budget alert set: {key} threshold ${threshold_usd:.3f}")

    # ----------------------------------------------------------------------- #
    # Factory method for OpenRouter integration
    # ----------------------------------------------------------------------- #
    @staticmethod
    def from_openrouter_response(
        *,
        request_ts: datetime.datetime,
        model: str,
        tokens_prompt: int,
        tokens_completion: int,
        response_time: float,
        provider: str = "openrouter",
        task_type: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        response_quality: Optional[Dict[str, float]] = None,
        error_code: Optional[str] = None
    ) -> RequestMetrics:
        """
        Factory helper that transforms OpenRouter response data into RequestMetrics.

        Returns
        -------
        RequestMetrics
            Metrics object ready for recording.
        """
        success = error_code is None
        return RequestMetrics(
            model=model,
            request_ts=request_ts,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            response_time=response_time,
            provider=provider,
            success=success,
            task_type=task_type,
            session_id=session_id,
            user_id=user_id,
            response_quality=response_quality,
            error_code=error_code,
        )


# --------------------------------------------------------------------------- #
# Demo usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    import time
    
    # Create tracker with persistence
    tracker = MetricsTracker(
        persistence_path="./metrics_demo.json",
        budget_limits={"openrouter:gpt-4": 10.0}
    )

    # Simulate some requests
    print("Simulating API requests...")
    for i in range(5):
        now = datetime.datetime.utcnow()
        metrics = MetricsTracker.from_openrouter_response(
            request_ts=now,
            model=f"gpt-4{'o-mini' if i % 2 else ''}",
            tokens_prompt=100 + i * 20,
            tokens_completion=50 + i * 10,
            response_time=0.5 + i * 0.2,
            provider="openrouter",
            task_type="chat" if i % 2 else "coding",
            session_id=f"session-{i // 2}",
            user_id=f"user-{i % 3}",
        )
        tracker.record(metrics)
        time.sleep(0.1)

    # Generate reports
    print("\n=== Usage Report ===")
    print(tracker.as_json())
    
    print("\n=== Cost Optimization ===")
    for rec in tracker.cost_optimization_recommendations():
        print(f"• {rec}")
    
    print("\n=== Daily Stats ===")
    for day, stats in tracker.daily_stats(3).items():
        print(f"{day}: {stats['requests']} requests, ${stats['cost']}, {stats['success_rate']}% success")
    
    # Export formats
    print("\n=== Prometheus Export ===")
    print(tracker.export_prometheus()[:300] + "...")
    
    # Persist to file
    tracker.persist_snapshot(include_metrics=True)
    print("\nMetrics saved to disk.")