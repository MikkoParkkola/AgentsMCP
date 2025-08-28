# --------------------------------------------------------------------------- #
# File: /Users/mikko/github/AgentsMCP/src/agentsmcp/contracts/validation.py
# --------------------------------------------------------------------------- #
"""
Contract‑validation framework for AgentsMCP.

Features
--------
*  JSON‑schema driven validation of TaskEnvelopeV{n} and ResultEnvelopeV{n}
*  Role‑specific contract enforcement (input & output)
*  Chain‑of‑custody validation for task flow integrity
*  Version compatibility checking (Envelope versions)
*  Runtime contract‑violation detection with rich error reporting
*  Zero‑dependency integration with the envelope‑schema utilities
*  Decorator helper (`@enforce_role_contract`) that wraps the `apply` method of any
   role to automatically perform all checks before and after execution
"""

from __future__ import annotations

import json
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import jsonschema
from jsonschema import ValidationError as JsonSchemaError
from pydantic import BaseModel, ValidationError as PydanticError, Field

# Import the envelope JSON schemas generated in P4.1
try:
    from agentsmcp.schemas.envelope_schemas import (
        get_task_envelope_schema,
        get_result_envelope_schema,
        validate_task_envelope as schema_validate_task,
        validate_result_envelope as schema_validate_result,
    )
    from agentsmcp.models import TaskEnvelopeV1, ResultEnvelopeV1, EnvelopeStatus
    from agentsmcp.roles.base import RoleName
except Exception as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "Failed to import envelope schemas or models. Ensure that "
        "agentsmcp.schemas.envelope_schemas and agentsmcp.models are available."
    ) from exc

# --------------------------------------------------------------------------- #
# Error classes
# --------------------------------------------------------------------------- #
class ContractViolationError(Exception):
    """Base class for all contract violations."""
    pass

class SchemaValidationError(ContractViolationError):
    """
    Raised when JSON schema validation of an envelope fails.
    Attributes:
        envelope_type: str
        errors: list[str] – human readable description of each error
    """
    def __init__(self, envelope_type: str, errors: List[str]):
        super().__init__(f"Schema validation error in {envelope_type}")
        self.envelope_type = envelope_type
        self.errors = errors

class RoleContractError(ContractViolationError):
    """Raised when a role contract is not satisfied."""
    def __init__(self, role: str, action: str, details: str):
        super().__init__(f"Role contract violation: {role} {action}")
        self.role = role
        self.action = action  # "input" or "output"
        self.details = details

class ChainOfCustodyError(ContractViolationError):
    """Raised when the integrity of the task flow is broken."""
    def __init__(self, message: str):
        super().__init__(f"Chain‑of‑custody error: {message}")
        self.message = message

class VersionCompatibilityError(ContractViolationError):
    """Raised when envelope version mismatch occurs."""
    def __init__(self, envelope_type: str, current: str, expected: str):
        super().__init__(
            f"{envelope_type} version mismatch: {current} != {expected}"
        )
        self.envelope_type = envelope_type
        self.current = current
        self.expected = expected

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def _json_schema_validate(
    data: Dict[str, Any], schema: Dict[str, Any], schema_name: str
) -> None:
    """
    Validate *data* against the JSON schema.
    On failure a ``SchemaValidationError`` is raised.
    """
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        # Build a user‑friendly representation of the errors
        err_desc = [
            f"{list(error.path)}: {error.message}" for error in errors
        ]
        raise SchemaValidationError(schema_name, err_desc)


def _check_role_contract(
    role: str,
    payload: Dict[str, Any],
    stage: str,
    required_fields: Iterable[Tuple[str, str]],
) -> None:
    """
    Generic enforcement of role‑specific contract rules.

    Parameters
    ----------
    role : str
        Name of the role that received the payload.
    payload : dict
        The envelope body that should contain the required fields.
    stage : str
        One of ``"input"`` or ``"output"``.
    required_fields : iterable
        Iterable of ``(field_name, field_type)`` tuples.
        ``field_type`` is used only in the error message.

    Raises
    ------
    RoleContractError
        If any required field is missing or has an incorrect type.
    """
    missing = [name for name, _ in required_fields if name not in payload]
    if missing:
        raise RoleContractError(
            role=role, action=stage,
            details=f"missing required fields: {', '.join(missing)}"
        )

    # Basic type checking for required fields
    for field_name, field_type in required_fields:
        if field_name in payload:
            value = payload[field_name]
            if value is None:
                raise RoleContractError(
                    role=role, action=stage,
                    details=f"field '{field_name}' is null"
                )


def _ensure_version_compatibility(schema_name: str, envelope_data: Dict[str, Any]) -> None:
    """
    Check that the envelope in *envelope_data* matches the expected version.
    """
    expected_version = "1.0"  # Current system version
    actual_version = envelope_data.get("version")
    if actual_version and actual_version != expected_version:
        raise VersionCompatibilityError(
            envelope_type=schema_name,
            current=str(actual_version),
            expected=str(expected_version),
        )


# --------------------------------------------------------------------------- #
# Public validation functions
# --------------------------------------------------------------------------- #
def validate_task_envelope(
    task_data: Union[TaskEnvelopeV1, Dict[str, Any]],
    *,
    role_name: Optional[str] = None,
    required_fields: Optional[Iterable[Tuple[str, str]]] = None,
) -> TaskEnvelopeV1:
    """
    Validate *task_data* against the TaskEnvelopeV1 JSON schema and the
    pydantic model. Optionally checks a role‑specific rule set.

    Parameters
    ----------
    task_data : TaskEnvelopeV1 | dict
        Either a fully‑constructed Pydantic model or a raw dictionary.
    role_name : str | None
        Name of the role that will consume the envelope.
        When supplied, role contract rules are checked.
    required_fields : iterable | None
        Iterable ``(field_name, field_type)`` tuples to enforce for this role.

    Returns
    -------
    TaskEnvelopeV1
        The validated Pydantic instance.

    Raises
    ------
    SchemaValidationError | RoleContractError
    """
    if isinstance(task_data, TaskEnvelopeV1):
        raw = task_data.model_dump(by_alias=True, exclude_unset=False)
    else:
        raw = task_data

    # Version compatibility check
    _ensure_version_compatibility("TaskEnvelopeV1", raw)

    # JSON‑schema validation
    schema = get_task_envelope_schema()
    _json_schema_validate(raw, schema, "TaskEnvelopeV1")

    # Pydantic validation (reuse schema utility)
    model = schema_validate_task(raw)

    # Role‑specific contract validation
    if role_name and required_fields:
        _check_role_contract(
            role=role_name,
            payload=raw,
            stage="input",
            required_fields=required_fields,
        )

    return model


def validate_result_envelope(
    result_data: Union[ResultEnvelopeV1, Dict[str, Any]],
    *,
    role_name: Optional[str] = None,
    required_fields: Optional[Iterable[Tuple[str, str]]] = None,
) -> ResultEnvelopeV1:
    """
    Validate *result_data* against the ResultEnvelopeV1 JSON schema and
    the pydantic model. Optionally checks a role‑specific rule set.

    Parameters
    ----------
    result_data : ResultEnvelopeV1 | dict
        Either a fully‑constructed Pydantic model or a raw dictionary.
    role_name : str | None
        Name of the role that produced the envelope.
    required_fields : iterable | None
        Iterable ``(field_name, field_type)`` tuples to enforce for this role.

    Returns
    -------
    ResultEnvelopeV1
        The validated Pydantic instance.

    Raises
    ------
    SchemaValidationError | RoleContractError
    """
    if isinstance(result_data, ResultEnvelopeV1):
        raw = result_data.model_dump(by_alias=True, exclude_unset=False)
    else:
        raw = result_data

    _ensure_version_compatibility("ResultEnvelopeV1", raw)
    
    # JSON‑schema validation
    schema = get_result_envelope_schema()
    _json_schema_validate(raw, schema, "ResultEnvelopeV1")
    
    # Pydantic validation (reuse schema utility)  
    model = schema_validate_result(raw)

    if role_name and required_fields:
        _check_role_contract(
            role=role_name,
            payload=raw,
            stage="output",
            required_fields=required_fields,
        )

    return model


def validate_chain_of_custody(
    current_result: ResultEnvelopeV1,
    input_task: TaskEnvelopeV1,
) -> None:
    """
    Validate that the result envelope properly references the input task.

    Raises
    ------
    ChainOfCustodyError
    """
    if current_result.id != input_task.id:
        raise ChainOfCustodyError(
            f"Result envelope ID {current_result.id} does not match task ID {input_task.id}"
        )


# --------------------------------------------------------------------------- #
# Decorator that wraps the `apply` method of a role
# --------------------------------------------------------------------------- #
def enforce_role_contract(
    *,
    task_required_fields: Optional[Iterable[Tuple[str, str]]] = None,
    result_required_fields: Optional[Iterable[Tuple[str, str]]] = None,
) -> Callable[[Callable[[Any, TaskEnvelopeV1], ResultEnvelopeV1]], Callable]:
    """
    Decorator for role implementations.

    Applies the following workflow:

    1. Receives a ``TaskEnvelopeV1`` instance or raw dict.
    2. Validates the envelope against the Task schema and role‑specific
       required fields (if any).
    3. Calls the wrapped `apply` function (the role logic).
    4. Validates the returned ``ResultEnvelopeV1`` against the Result
       schema and role‑specific required fields (if any).
    5. Performs a basic chain‑of‑custody check.

    Returns
    -------
    Callable
        The wrapped role method.
    """
    def decorator(func: Callable[[Any, TaskEnvelopeV1], ResultEnvelopeV1]) -> Callable:
        @wraps(func)
        def wrapper(self: Any, task: Union[TaskEnvelopeV1, Dict[str, Any]]) -> ResultEnvelopeV1:
            role_name = getattr(self, "role_name", None) or getattr(self, "__class__").__name__

            # Validate the incoming TaskEnvelope
            task_model = validate_task_envelope(
                task_data=task,
                role_name=role_name,
                required_fields=task_required_fields,
            )

            # Execute the actual role logic
            result_raw = func(self, task_model)

            # Validate the output ResultEnvelope
            result_model = validate_result_envelope(
                result_data=result_raw,
                role_name=role_name,
                required_fields=result_required_fields,
            )

            # Chain‑of‑custody check
            validate_chain_of_custody(result_model, task_model)

            return result_model

        return wrapper

    return decorator


# Role-specific validation rules
ROLE_CONTRACT_RULES = {
    "QA": {
        "input": [],  # No specific input requirements
        "output": [("risks", "list"), ("followups", "list")],
    },
    "ARCHITECT": {
        "input": [],
        "output": [("decisions", "list")],
    },
    "CODER": {
        "input": [],
        "output": [("decisions", "list")],
    },
    "MERGE_BOT": {
        "input": [],
        "output": [("decisions", "list")],
    },
}


def get_role_contract(role_name: str) -> Dict[str, List[Tuple[str, str]]]:
    """Get the contract rules for a specific role."""
    return ROLE_CONTRACT_RULES.get(role_name, {"input": [], "output": []})