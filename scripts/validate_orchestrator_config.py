#!/usr/bin/env python3
"""
validate_orchestrator_config.py
================================

A comprehensive validator for the **AgentsMCP orchestrator** model
configuration.  The script can be used both as a command‑line tool and
as an importable library.

The validator performs the following checks:

1. **Required fields** – Every model configuration must contain
   `context_limit`, `cost_per_input`, `cost_per_output`,
   `performance_score`, and any other fields you consider mandatory.
2. **Performance scores** – Must be a float or int in the inclusive
   range 0‑100.
3. **Cost values** – `cost_per_input` and `cost_per_output` must be
   non‑negative floats (or ints).
4. **Context limits** – Must be a positive integer.
5. **Default model** – The key `gpt-5` must exist in the configuration.
6. **Duplicate models** – Duplicate model names are impossible in a
   Python dictionary, but if you load from a raw file you can pass a
   custom dictionary that contains duplicates.  The validator checks
   for repeated keys anyway.
7. **Model name pattern** – Names must match the regex
   ``^[a-zA-Z0-9_-]+$`` (letters, digits, underscores and hyphens
   only, no spaces).
8. **Summary report** – Human‑readable report with a count of passed
   / failed models and detailed error messages.
9. **CLI & importable** – Use the command line or import functions
   into your own Python code.
10. **Exit codes** – `0` if all checks pass, otherwise `1`.  Useful for
    CI/CD pipelines.

The script expects the dictionary to be available as
`distributed.orchestrator.ORCHESTRATOR_MODELS`.  If that module
cannot be imported, an informative error is shown and the script
exits with status 1.

---------------------------------------------------------------------

Example CLI usage
-----------------
```bash
$ python validate_orchestrator_config.py
✔ All 12 models passed validation.

$ python validate_orchestrator_config.py --verbose
🔍 Validating 12 models...
✖ Model 'bad-model' failed 3 checks:
  • performance_score 150.0 is outside 0–100 range
  • cost_per_input -0.01 is negative
  • context_limit 0 is not a positive integer
✔ All models passed validation.
```

---------------------------------------------------------------------

Importable usage
----------------
```python
from validate_orchestrator_config import validate_all_models

results = validate_all_models()
if not results['passed']:
    print("Validation failed.")
    for name, errors in results['errors'].items():
        print(f"{name}: {errors}")
```

---------------------------------------------------------------------

"""

import argparse
import sys
import re
from typing import Dict, Tuple, List, Any

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #

# List of fields that every model must contain
REQUIRED_FIELDS = [
    "context_limit",
    "cost_per_input",
    "cost_per_output",
    "performance_score",
    # Add more mandatory fields here if needed
]

# Allowed regex for model names: letters, digits, underscores and hyphens
MODEL_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]+$")

# The key for the default model
DEFAULT_MODEL_KEY = "gpt-5"

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _load_orchestrator_models() -> Dict[str, Any]:
    """
    Import the ORCHESTRATOR_MODELS dictionary from the distributed.orchestrator
    module.  The function is isolated so that it can be mocked or replaced
    during testing.

    Returns
    -------
    dict
        Dictionary of model configurations.

    Raises
    ------
    RuntimeError
        If the module or dictionary cannot be imported.
    """
    try:
        from agentsmcp.distributed.orchestrator import ORCHESTRATOR_MODELS
        return ORCHESTRATOR_MODELS
    except Exception as exc:
        raise RuntimeError(
            "Could not import ORCHESTRATOR_MODELS from "
            "agentsmcp.distributed.orchestrator.  Ensure the module is "
            "available on sys.path."
        ) from exc


def _check_required_fields(
    model_name: str, config: Dict[str, Any]
) -> List[str]:
    """Return a list of missing required fields."""
    missing = [field for field in REQUIRED_FIELDS if field not in config]
    return missing


def _check_performance_score(
    model_name: str, score: Any
) -> List[str]:
    """Return a list of errors related to the performance_score."""
    errors = []
    try:
        val = float(score)
    except Exception:
        errors.append(
            f"performance_score '{score}' is not a numeric value"
        )
        return errors

    if not (0.0 <= val <= 100.0):
        errors.append(
            f"performance_score {score} is outside 0–100 range"
        )
    return errors


def _check_cost_value(
    model_name: str, key: str, value: Any
) -> List[str]:
    """Return errors if the cost value is invalid."""
    errors = []
    try:
        val = float(value)
    except Exception:
        errors.append(f"{key} '{value}' is not a numeric value")
        return errors

    if val < 0.0:
        errors.append(f"{key} {value} is negative")
    return errors


def _check_context_limit(
    model_name: str, value: Any
) -> List[str]:
    """Return errors if the context limit is not a positive int."""
    errors = []
    if not isinstance(value, int):
        errors.append(f"context_limit '{value}' is not an integer")
        return errors
    if value <= 0:
        errors.append(f"context_limit {value} is not a positive integer")
    return errors


def _check_model_name(
    model_name: str
) -> List[str]:
    """Return errors if the model name does not match the expected pattern."""
    if not MODEL_NAME_REGEX.match(model_name):
        return [f"model name '{model_name}' does not match pattern {MODEL_NAME_REGEX.pattern}"]
    return []


def _check_default_model_exists(
    models: Dict[str, Any]
) -> List[str]:
    """Return an error if DEFAULT_MODEL_KEY is not present."""
    if DEFAULT_MODEL_KEY not in models:
        return [f"Default model '{DEFAULT_MODEL_KEY}' is missing from the configuration"]
    return []


def _check_duplicate_models(
    models: Dict[str, Any]
) -> List[str]:
    """Return an error if duplicate keys are found.

    In a Python dict duplicates cannot exist, but if the caller
    passes a custom mapping that may contain duplicates (e.g. after
    merging dictionaries manually) this function can detect that.
    """
    seen = set()
    duplicates = set()
    for key in models.keys():
        if key in seen:
            duplicates.add(key)
        else:
            seen.add(key)
    if duplicates:
        return [f"Duplicate model names found: {', '.join(sorted(duplicates))}"]
    return []


def _validate_single_model(
    model_name: str, config: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Validate a single model configuration.

    Parameters
    ----------
    model_name : str
        The name of the model.
    config : dict
        The configuration dictionary for this model.

    Returns
    -------
    (bool, list[str])
        ``True`` if the model passes all checks, otherwise ``False`` and a
        list of error messages.
    """
    errors = []

    # 1. Required fields
    missing = _check_required_fields(model_name, config)
    if missing:
        errors.append(
            f"Missing required fields: {', '.join(missing)}"
        )

    # 2. Performance score
    if "performance_score" in config:
        errors.extend(_check_performance_score(model_name, config["performance_score"]))
    else:
        # Already reported in missing fields
        pass

    # 3. Cost values
    for key in ("cost_per_input", "cost_per_output"):
        if key in config:
            errors.extend(_check_cost_value(model_name, key, config[key]))
        else:
            # Already reported in missing fields
            pass

    # 4. Context limit
    if "context_limit" in config:
        errors.extend(_check_context_limit(model_name, config["context_limit"]))
    else:
        # Already reported in missing fields
        pass

    # 7. Model name pattern
    errors.extend(_check_model_name(model_name))

    passed = len(errors) == 0
    return passed, errors


def validate_all_models(
    models: Dict[str, Any] | None = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Validate the entire orchestrator model configuration.

    Parameters
    ----------
    models : dict | None
        The dictionary of model configurations.  If ``None``, the
        default dictionary will be loaded from
        ``agentsmcp.distributed.orchestrator``.
    verbose : bool
        If ``True``, detailed per‑model messages are printed to stdout.

    Returns
    -------
    dict
        Dictionary with keys:
            * ``passed`` (bool) – ``True`` if all models passed.
            * ``count`` (int) – number of models checked.
            * ``errors`` (dict[str, list[str]]) – mapping of model names to
              lists of error strings.
            * ``summary`` (str) – human‑readable summary message.
    """
    if models is None:
        models = _load_orchestrator_models()

    # 5. Default model existence
    errors = _check_default_model_exists(models)

    # 6. Duplicate model names
    errors.extend(_check_duplicate_models(models))

    # 1‑4. Individual model checks
    model_errors: Dict[str, List[str]] = {}
    for name, cfg in models.items():
        passed, err_list = _validate_single_model(name, cfg)
        if not passed:
            model_errors[name] = err_list

    # Consolidate all errors
    errors.extend([f"{name}: {', '.join(errs)}" for name, errs in model_errors.items()])

    passed_all = len(errors) == 0
    summary = (
        f"✔ All {len(models)} models passed validation."
        if passed_all
        else f"✖ {len(model_errors)} model(s) failed validation."
    )

    # Optional verbose output
    if verbose:
        print(f"🔍 Validating {len(models)} models...")
        for name, cfg in models.items():
            passed, err_list = _validate_single_model(name, cfg)
            if passed:
                print(f"✔ {name}")
            else:
                print(f"✖ {name} failed {len(err_list)} check(s):")
                for err in err_list:
                    print(f"  • {err}")
        print(summary)

    return {
        "passed": passed_all,
        "count": len(models),
        "errors": model_errors,
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the AgentsMCP orchestrator model configuration "
            "stored in agentsmcp.distributed.orchestrator.ORCHESTRATOR_MODELS."
        )
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed validation output for each model.",
    )
    return parser


def _cli_main(argv: List[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = validate_all_models(verbose=args.verbose)
    except RuntimeError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(EXIT_FAILURE)

    if result["passed"]:
        print(result["summary"])
        sys.exit(EXIT_SUCCESS)
    else:
        print(result["summary"], file=sys.stderr)
        # Dump all errors for CI/CD visibility
        for name, errs in result["errors"].items():
            for err in errs:
                print(f"{name}: {err}", file=sys.stderr)
        sys.exit(EXIT_FAILURE)


# --------------------------------------------------------------------------- #
# Main guard
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    _cli_main()