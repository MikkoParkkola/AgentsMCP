"""
AgentsMCP – Model Selection Utilities

The :mod:`selector` module implements a generic model selection engine for AgentsMCP
that takes into account task specifications, cost constraints, performance requirements,
context‑length needs and optional preferences.

It is designed to work seamlessly with the ModelDB class from the routing package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple, Any

from .models import ModelDB, Model

__all__ = [
    "TaskSpec",
    "SelectionResult", 
    "ModelSelector",
]

# --------------------------------------------------------------------------- #
# 1. Task Specification
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TaskSpec:
    """
    Describes the requirements of a given user task.

    Parameters
    ----------
    task_type : str
        One of ``"reasoning"``, ``"coding"``, ``"general"``, ``"multimodal"``.
    max_cost_per_1k_tokens : Optional[float]
        Upper bound on the cost per 1 000 tokens. ``None`` means no budget restriction.
    min_performance_tier : int
        Minimum required performance tier. Lower numbers are more permissive.
    required_context_length : Optional[int]
        Minimum number of tokens the model must support. ``None`` means no requirement.
    preferences : Dict[str, Any] = field(default_factory=dict)
        Arbitrary key/value pairs, e.g. ``{"preferred_provider": "openai"}``.
    """
    task_type: str
    max_cost_per_1k_tokens: Optional[float] = None
    min_performance_tier: int = 1
    required_context_length: Optional[int] = None
    preferences: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 2. Selection Result
# --------------------------------------------------------------------------- #

class SelectionResult(NamedTuple):
    """
    Encapsulates the selected model and explanatory information.

    Attributes
    ----------
    model : Model
        The chosen model instance.
    explanation : str
        Human‑readable description of why this model was chosen.
    score : float
        Numerical score used for ranking.
    """
    model: Model
    explanation: str
    score: float


# --------------------------------------------------------------------------- #
# 3. Model Selector
# --------------------------------------------------------------------------- #

class ModelSelector:
    """
    Core selector that evaluates all models in the ModelDB against
    a TaskSpec and returns the best candidate.

    Parameters
    ----------
    model_db : ModelDB
        Database of available models.
    log_level : int, optional
        Logging verbosity for internal decision tracing.
    """

    # Default weights for the scoring formula
    _DEFAULT_WEIGHTS = {
        "capability_match": 5.0,
        "performance": 3.0,
        "cost_efficiency": 4.0,
        "context_length": 2.0,
        "provider_preference": 1.5,
    }

    def __init__(self, model_db: ModelDB, log_level: int = logging.INFO) -> None:
        self.model_db = model_db
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.weights = self._DEFAULT_WEIGHTS.copy()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def select_model(self, spec: TaskSpec) -> SelectionResult:
        """
        Select the best model for the supplied TaskSpec.

        The algorithm scores each candidate, optionally breaking ties by
        provider diversity. If no model satisfies all hard constraints,
        the model with the highest composite score is returned anyway.

        Parameters
        ----------
        spec : TaskSpec
            The specification that the chosen model must satisfy.

        Returns
        -------
        SelectionResult
            The selected model and a textual explanation.

        Raises
        ------
        ValueError
            If the model database is empty or the request cannot be fulfilled.
        """
        models = self.model_db.all_models()
        if not models:
            raise ValueError("Model database is empty – cannot perform selection.")

        scored: List[Tuple[float, Model, List[str]]] = []

        for model in models:
            try:
                score, reasons = self._score_model(model, spec)
                scored.append((score, model, reasons))
                self.logger.debug(
                    "Model %s scored %.2f: %s",
                    model.name,
                    score,
                    "; ".join(reasons),
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to score model %s: %s",
                    model.name,
                    exc,
                )

        if not scored:
            raise ValueError("No models could be evaluated.")

        # Sort descending by score (higher is better)
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_model, best_reasons = scored[0]
        explanation = self._generate_explanation(best_model, best_reasons)

        self.logger.info(
            "Selected model %s (score: %.2f) for task %s",
            best_model.name,
            best_score,
            spec.task_type,
        )

        return SelectionResult(best_model, explanation, best_score)

    # --------------------------------------------------------------------- #
    # Convenience Scenario Helpers
    # --------------------------------------------------------------------- #

    def best_cost_effective_coding(self, max_budget: float = 1.0) -> SelectionResult:
        """
        Return the most cost‑efficient model that meets the coding requirement.
        
        Parameters
        ----------
        max_budget : float
            Maximum cost per 1k tokens (default: $1.0)
        """
        spec = TaskSpec(
            task_type="coding",
            max_cost_per_1k_tokens=max_budget,
            min_performance_tier=3,  # Require decent performance for coding
        )
        return self.select_model(spec)

    def fastest_reasoning_under_budget(self, budget: float) -> SelectionResult:
        """
        Return the reasoning model with the highest performance tier 
        that stays within budget.
        
        Parameters
        ----------
        budget : float
            Maximum cost per 1k tokens
        """
        spec = TaskSpec(
            task_type="reasoning",
            max_cost_per_1k_tokens=budget,
            min_performance_tier=4,  # High performance for reasoning
        )
        return self.select_model(spec)

    def most_capable_regardless_of_cost(self) -> SelectionResult:
        """
        Return the model with the highest performance tier, ignoring cost.
        """
        spec = TaskSpec(
            task_type="general",
            min_performance_tier=5,  # Only top-tier models
            max_cost_per_1k_tokens=None,
        )
        # Temporarily boost performance weight
        saved_weights = self.weights.copy()
        self.weights["performance"] = 10.0
        self.weights["cost_efficiency"] = 0.1
        try:
            return self.select_model(spec)
        finally:
            self.weights = saved_weights

    def cheapest_meeting_requirements(self,
                                     min_tier: int = 3,
                                     context_needed: Optional[int] = None
                                     ) -> SelectionResult:
        """
        Return the cheapest model that satisfies minimum requirements.
        
        Parameters
        ----------
        min_tier : int
            Minimum performance tier required
        context_needed : Optional[int]
            Minimum context length needed
        """
        spec = TaskSpec(
            task_type="general",
            min_performance_tier=min_tier,
            required_context_length=context_needed,
            max_cost_per_1k_tokens=None,
        )
        # Boost cost efficiency weight
        saved_weights = self.weights.copy()
        self.weights["cost_efficiency"] = 10.0
        self.weights["performance"] = 1.0
        try:
            return self.select_model(spec)
        finally:
            self.weights = saved_weights

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _score_model(self, model: Model, spec: TaskSpec) -> Tuple[float, List[str]]:
        """
        Compute a composite score for a single model relative to a task spec.

        Returns a tuple of (score, list_of_reason_strings).
        """
        reasons: List[str] = []
        scores = []

        # 1. Capability match
        capabilities = {c.lower() for c in model.categories}
        if spec.task_type.lower() in capabilities:
            cap_score = self.weights["capability_match"]
            reasons.append(f"Capability match for {spec.task_type} (+{cap_score})")
            scores.append(cap_score)
        else:
            # Check for partial matches (e.g., "general" can handle many tasks)
            if "general" in capabilities or len(capabilities) > 2:
                cap_score = self.weights["capability_match"] * 0.3  # Partial credit
                reasons.append(f"Partial capability match (+{cap_score:.1f})")
                scores.append(cap_score)
            else:
                cap_score = -5.0  # Penalty for poor match
                reasons.append(f"Poor capability match for {spec.task_type} ({cap_score})")
                scores.append(cap_score)

        # 2. Performance tier
        if model.performance_tier >= spec.min_performance_tier:
            perf_bonus = model.performance_tier - spec.min_performance_tier
            perf_score = self.weights["performance"] * (1 + 0.2 * perf_bonus)
            reasons.append(f"Performance tier {model.performance_tier} (+{perf_score:.1f})")
            scores.append(perf_score)
        else:
            perf_score = -10.0
            reasons.append(f"Performance tier {model.performance_tier} below minimum ({perf_score})")
            scores.append(perf_score)

        # 3. Cost efficiency
        if spec.max_cost_per_1k_tokens is not None:
            if model.cost_per_input_token <= spec.max_cost_per_1k_tokens:
                # Reward efficiency: lower cost within budget gets higher score
                efficiency_ratio = spec.max_cost_per_1k_tokens / max(model.cost_per_input_token, 0.001)
                cost_score = self.weights["cost_efficiency"] * min(efficiency_ratio, 5.0)  # Cap bonus
                reasons.append(f"Cost ${model.cost_per_input_token}/1k within budget (+{cost_score:.1f})")
                scores.append(cost_score)
            else:
                cost_score = -15.0  # Heavy penalty for exceeding budget
                reasons.append(f"Cost ${model.cost_per_input_token}/1k exceeds budget ({cost_score})")
                scores.append(cost_score)
        else:
            # No budget constraint - reward lower cost
            cost_efficiency = 1.0 / max(model.cost_per_input_token, 0.001)
            cost_score = self.weights["cost_efficiency"] * min(cost_efficiency, 10.0)
            reasons.append(f"Cost efficiency (+{cost_score:.1f})")
            scores.append(cost_score)

        # 4. Context length adequacy
        if spec.required_context_length is not None:
            if model.context_length and model.context_length >= spec.required_context_length:
                ctx_score = self.weights["context_length"]
                reasons.append(f"Context {model.context_length} meets requirement (+{ctx_score})")
                scores.append(ctx_score)
            else:
                ctx_score = -8.0
                reasons.append(f"Insufficient context length ({ctx_score})")
                scores.append(ctx_score)
        else:
            # Bonus for large context length
            if model.context_length and model.context_length > 32000:
                ctx_score = self.weights["context_length"] * 0.5
                reasons.append(f"Large context length bonus (+{ctx_score})")
                scores.append(ctx_score)

        # 5. Provider preference
        pref_provider = spec.preferences.get("preferred_provider")
        if pref_provider:
            if model.provider.lower() == pref_provider.lower():
                prov_score = self.weights["provider_preference"]
                reasons.append(f"Preferred provider {model.provider} (+{prov_score})")
                scores.append(prov_score)
            else:
                prov_score = -0.5
                reasons.append(f"Non-preferred provider {model.provider} ({prov_score})")
                scores.append(prov_score)

        total_score = sum(scores)
        return total_score, reasons

    def _generate_explanation(self, model: Model, reasons: List[str]) -> str:
        """
        Generate a human-readable explanation for the selection.
        """
        return (
            f"Selected {model.name} by {model.provider}: "
            f"{'; '.join(reasons[:3])}..."  # Show top 3 reasons
        )


# --------------------------------------------------------------------------- #
# Test and demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    from .models import ModelDB
    
    # Initialize with default model database
    db = ModelDB()
    selector = ModelSelector(db, log_level=logging.DEBUG)
    
    print("=== Model Selection Demo ===\n")
    
    # Test different scenarios
    print("1. Best cost-effective coding model:")
    result = selector.best_cost_effective_coding(max_budget=2.0)
    print(f"   {result.model.name} (score: {result.score:.2f})")
    print(f"   {result.explanation}\n")
    
    print("2. Fastest reasoning under $5 budget:")
    result = selector.fastest_reasoning_under_budget(5.0)
    print(f"   {result.model.name} (score: {result.score:.2f})")
    print(f"   {result.explanation}\n")
    
    print("3. Most capable model regardless of cost:")
    result = selector.most_capable_regardless_of_cost()
    print(f"   {result.model.name} (score: {result.score:.2f})")
    print(f"   {result.explanation}\n")
    
    print("4. Cheapest model meeting tier 4+ requirements:")
    result = selector.cheapest_meeting_requirements(min_tier=4, context_needed=100000)
    print(f"   {result.model.name} (score: {result.score:.2f})")
    print(f"   {result.explanation}")