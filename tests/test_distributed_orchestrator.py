# test_distributed_orchestrator.py
"""
Integration tests for the DistributedOrchestrator in the AgentsMCP orchestrator
module.

The tests cover:
  1. Successful initialisation with supported models.
  2. Fallback to the default model (`gpt‑5`) when an unsupported model is supplied.
  3. Correct loading of model configuration data.
  4. Availability of supported model list and recommendation logic.
  5. Basic cost estimation logic.
"""

import pytest

# --------------------------------------------------------------------------- #
# Import the class under test.  If the import fails, mark all tests as
# "xfail" so that CI can surface the problem cleanly.
# --------------------------------------------------------------------------- #
try:
    from agentsmcp.distributed.orchestrator import DistributedOrchestrator
except Exception as exc:   # pragma: no cover – executed only when import fails
    DistributedOrchestrator = None

    @pytest.mark.skip(reason="Failed to import DistributedOrchestrator: %s" % exc)
    def test_import():
        """Placeholder to keep the test module importable."""
        pass


# --------------------------------------------------------------------------- #
# Helper constants
# --------------------------------------------------------------------------- #
# A list of known good models (pulled from the orchestrator's public API).  If
# this list changes in a future release, simply update the values below.
KNOWN_MODELS = [
    "gpt-5",          # the default model
    "claude-4.1-opus",
    "claude-4.1-sonnet",
    "gemini-2.5-pro",
    "qwen3-235b-a22b",
    "qwen3-32b",
]

# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_name", KNOWN_MODELS)
def test_initialise_with_valid_model(model_name):
    """
    Instantiating with a supported model should keep that model in the
    orchestrator instance.
    """
    orchestrator = DistributedOrchestrator(orchestrator_model=model_name)
    # The orchestrator should expose the name that was requested
    assert orchestrator.orchestrator_model == model_name
    # The model configuration should be available and not empty
    config = getattr(orchestrator, "model_config", None)
    assert config is not None, f"Model config for {model_name} is None"
    assert isinstance(config, dict), f"Model config for {model_name} is not a dict"


def test_invalid_model_fallback():
    """
    Providing an unknown model should cause the orchestrator to fall back to the
    default (`gpt-5`) and not raise an exception.
    """
    invalid_name = "nonexistent-model-xyz"
    orchestrator = DistributedOrchestrator(orchestrator_model=invalid_name)
    # The fallback should be the default model name
    assert orchestrator.orchestrator_model == "gpt-5"
    # The config for the default model should still be present
    config = getattr(orchestrator, "model_config", None)
    assert config is not None, "Fallback config is None"


def test_get_available_models():
    """
    The static method should return a non‑empty dict containing all supported
    models.  The returned dict must contain all KNOWN_MODELS as keys.
    """
    available = DistributedOrchestrator.get_available_models()
    assert isinstance(available, dict), "get_available_models() did not return a dict"
    assert available, "Available models dict is empty"
    for model in KNOWN_MODELS:
        assert model in available, f"Model {model} missing from available models"


@pytest.mark.parametrize(
    "use_case,expected",
    [
        ("default", "gpt-5"),
        ("cost_effective", "gpt-5"), 
        ("premium", "claude-4.1-opus"),
        ("balanced", "claude-4.1-sonnet"),
        ("massive_context", "gemini-2.5-pro"),
        ("local", "qwen3-235b-a22b"),
        ("privacy", "qwen3-235b-a22b"),
        ("offline", "qwen3-32b"),
        ("unknown_use_case", "gpt-5"),  # fallback for unknown use case
    ],
)
def test_get_model_recommendation(use_case, expected):
    """
    Verify that the recommendation logic picks the expected model for
    various use cases.  The mapping mirrors the actual implementation.
    """
    recommendation = DistributedOrchestrator.get_model_recommendation(use_case)
    assert recommendation == expected, (
        f"Recommendation for use case '{use_case}' should be '{expected}', "
        f"got '{recommendation}' instead"
    )


@pytest.mark.parametrize(
    "input_tokens,output_tokens",
    [
        (1000, 2000),      # small task
        (0, 0),            # zero cost edge case
        (5000, 5000),      # medium task
        (50000, 5000),     # large task
    ],
)
def test_estimate_orchestration_cost(input_tokens, output_tokens):
    """
    The cost estimator should return a non‑negative numeric value.
    We test various token combinations to ensure the calculation is stable.
    """
    orchestrator = DistributedOrchestrator()  # use default model
    cost = orchestrator.estimate_orchestration_cost(input_tokens, output_tokens)
    assert isinstance(cost, (float, int)), "Cost estimate is not numeric"
    assert cost >= 0.0, "Cost estimate is negative"
    
    # For local models (cost_per_input/output = 0), cost should be 0
    if orchestrator.model_config["cost_per_input"] == 0:
        assert cost == 0.0, "Local model should have zero cost"


def test_model_config_contains_expected_keys():
    """
    Each model configuration should contain expected keys like:
      * context_limit
      * cost_per_input  
      * cost_per_output
      * performance_score
    """
    orchestrator = DistributedOrchestrator()
    config = orchestrator.model_config
    required_keys = {"context_limit", "cost_per_input", "cost_per_output", "performance_score"}
    missing = required_keys - config.keys()
    assert not missing, f"Missing config keys: {missing}"


def test_context_budget_adjustment():
    """
    Test that context budget is adjusted when it exceeds model limits.
    """
    # Test with a model that has a known context limit
    orchestrator = DistributedOrchestrator(
        orchestrator_model="gpt-5",
        context_budget_tokens=500000  # Exceeds GPT-5's 400K limit
    )
    # Should be adjusted down to model's context limit
    assert orchestrator.context_budget_tokens <= orchestrator.model_config["context_limit"]
    assert orchestrator.context_budget_tokens == 400000  # GPT-5's limit