"""
Utilities for generating and validating JSON schemas for the Envelope models
defined in :mod:`agentsmcp.models`.

The module provides:

* Automatic JSON schema generation using Pydantic's built‑in
  ``BaseModel.schema`` method.
* Validation helpers that raise :class:`pydantic.ValidationError` on bad
  data and return the corresponding Pydantic model instance otherwise.
* Functions that export the generated schemas to JSON files.  The file
  names contain the envelope name **and** the current schema version,
  keeping them easily identifiable and versioned.
* A small helper to fetch the envelope's own version (read from the
  ``version`` field defined on the model).

The code is intentionally lightweight, fully typed, and documented
according to the project style guidelines.  It is a drop‑in
replacement – simply be sure that the package ``agentsmcp`` is on
your :pydata:`sys.path` so that the import below resolves correctly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError

# Import the envelope models – they live in the project's `models` module.
# The import is deliberately located at module import time to make the
# functions below fully typed.  If the model names change, update the
# import accordingly.
try:
    from agentsmcp.models import ResultEnvelopeV1, TaskEnvelopeV1
except Exception as exc:  # pragma: no cover – import failure is critical
    logging.critical(
        "Failed to import envelope models from agentsmcp.models: %s",
        exc,
    )
    raise

__all__ = [
    "get_task_envelope_schema",
    "get_result_envelope_schema",
    "validate_task_envelope",
    "validate_result_envelope",
    "export_task_schema",
    "export_result_schema",
    "export_schema",
]

log = logging.getLogger(__name__)


def _get_schema(model: BaseModel) -> Dict[str, Any]:
    """
    Return a JSON‑serialisable schema dictionary for *model*.

    The function also ensures the optional ``$schema`` attribute that
    Pydantic usually emits is present – if not, it inserts the default
    ``http://json-schema.org/draft-07/schema#`` value.  This makes the
    output more self‑describing for consumers that do not rely on the
    Pydantic runtime.

    Parameters
    ----------
    model : BaseModel
        Pydantic model class for which to generate the schema.

    Returns
    -------
    dict
        JSON‑serialisable schema representation.
    """
    # Pydantic v2 uses `model_json_schema` for class-level schema generation
    try:
        schema = model.model_json_schema()  # type: ignore[attr-defined]
    except AttributeError:
        # Fallback for older Pydantic versions
        schema = model.schema()
    # Pydantic does not always include the $schema key by default.  Adding
    # it provides a standard header for validation tools that recognise
    # JSON‑Schema drafts.
    schema.setdefault("$schema", "http://json-schema.org/draft-07/schema#")
    return schema


def get_task_envelope_schema() -> Dict[str, Any]:
    """
    Return the JSON schema for :class:`TaskEnvelopeV1`.

    The schema is generated on-demand.  Returning the dict allows callers
    to embed the schema into documentation, JSON‑schema‑based validators,
    or to further modify it if required.

    Returns
    -------
    dict
        JSON‑serialisable schema for ``TaskEnvelopeV1``.
    """
    return _get_schema(TaskEnvelopeV1)


def get_result_envelope_schema() -> Dict[str, Any]:
    """
    Return the JSON schema for :class:`ResultEnvelopeV1`.

    Returns
    -------
    dict
        JSON‑serialisable schema for ``ResultEnvelopeV1``.
    """
    return _get_schema(ResultEnvelopeV1)


def _extract_version(schema: Dict[str, Any]) -> str:
    """
    Retrieve the envelope version from a generated schema.

    The ``version`` field is defined on every envelope model.  We
    fetch the *default* value for that field – if it is missing we
    fall back to ``unknown``.

    Parameters
    ----------
    schema : dict
        Schema dictionary from :func:`_get_schema`.

    Returns
    -------
    str
        Version string read from the schema.
    """
    return schema.get("properties", {}).get("version", {}).get("default", "unknown")


def validate_task_envelope(data: Dict[str, Any]) -> TaskEnvelopeV1:
    """
    Validate raw data against the :class:`TaskEnvelopeV1` schema.

    Parameters
    ----------
    data : dict
        Payload to validate.

    Returns
    -------
    TaskEnvelopeV1
        A fully initialised model instance.

    Raises
    ------
    pydantic.ValidationError
        If the payload does not conform to the schema.
    """
    try:
        return TaskEnvelopeV1.model_validate(data)  # type: ignore[attr-defined]
    except ValidationError as exc:
        log.error("TaskEnvelopeV1 validation failed: %s", exc, exc_info=True)
        raise


def validate_result_envelope(data: Dict[str, Any]) -> ResultEnvelopeV1:
    """
    Validate raw data against the :class:`ResultEnvelopeV1` schema.

    Parameters
    ----------
    data : dict
        Payload to validate.

    Returns
    -------
    ResultEnvelopeV1
        A fully initialised model instance.

    Raises
    ------
    pydantic.ValidationError
        If the payload does not conform to the schema.
    """
    try:
        return ResultEnvelopeV1.model_validate(data)  # type: ignore[attr-defined]
    except ValidationError as exc:
        log.error("ResultEnvelopeV1 validation failed: %s", exc, exc_info=True)
        raise


def export_schema(schema: Dict[str, Any], path: Path) -> None:
    """
    Persist a schema dictionary to a JSON file.

    The function guarantees that the parent directory exists;
    otherwise it will be created.  It logs a message with the
    resulting path.

    Parameters
    ----------
    schema : dict
        JSON‑serialisable schema to write.
    path : pathlib.Path
        Destination file.

    Raises
    ------
    OSError
        If writing to *path* fails.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(schema, fh, indent=2, sort_keys=True)
        log.info("Schema written to %s", path.resolve())
    except OSError as exc:
        log.exception("Failed to write schema to %s: %s", path.resolve(), exc)
        raise


def _schema_export_path(
    schema: Dict[str, Any], *, base_dir: Optional[Path] = None
) -> Path:
    """
    Compute a file path for a given schema that contains the correct
    envelope name and version.

    Example: ``TaskEnvelopeV1_1.0.json``

    Parameters
    ----------
    schema : dict
        Schema dictionary.
    base_dir : pathlib.Path | None
        Directory in which to place the file.  Defaults to a ``schemas``
        subdirectory of this module.

    Returns
    -------
    pathlib.Path
        Full path to the schema file.
    """
    name = schema.get("title") or "Envelope"
    version = _extract_version(schema)
    filename = f"{name}_{version}.json"
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent / "schemas"
    return base_dir / filename


def export_task_schema(base_dir: Optional[Path] = None) -> Path:
    """
    Generate the :class:`TaskEnvelopeV1` JSON schema and write it to a file.

    Returns the full path to the created file.

    Parameters
    ----------
    base_dir : pathlib.Path | None
        Destination directory.  If omitted, the schema will live in
        ``<this_module>/schemas`` next to the source code.

    Returns
    -------
    pathlib.Path
        Resolved path of the written file.
    """
    schema = get_task_envelope_schema()
    path = _schema_export_path(schema, base_dir=base_dir)
    export_schema(schema, path)
    return path


def export_result_schema(base_dir: Optional[Path] = None) -> Path:
    """
    Generate the :class:`ResultEnvelopeV1` JSON schema and write it to a file.

    Returns the full path to the created file.

    Parameters
    ----------
    base_dir : pathlib.Path | None
        Destination directory.  If omitted, the schema will live in
        ``<this_module>/schemas`` next to the source code.

    Returns
    -------
    pathlib.Path
        Resolved path of the written file.
    """
    schema = get_result_envelope_schema()
    path = _schema_export_path(schema, base_dir=base_dir)
    export_schema(schema, path)
    return path
