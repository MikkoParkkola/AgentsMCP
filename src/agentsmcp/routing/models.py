"""
models.py – A lightweight, production‑ready model database for OpenRouter.ai routing.

This module defines a :class:`Model` dataclass that encapsulates the capabilities and
costs of a single language model, and a :class:`ModelDB` class that loads a JSON
database and exposes convenient filtering and lookup utilities.

The JSON schema (see ``_models.json``) is deliberately simple to allow quick
updates and minimal maintenance:

{
    "id": "unique‑model‑identifier",
    "name": "Human‑friendly display name",
    "provider": "OpenAI / Anthropic / Meta / Google / ...",
    "context_length": 128000,
    "cost_per_input_token": 3.0,   # USD per 1 000 tokens
    "cost_per_output_token": 6.0,  # USD per 1 000 tokens
    "performance_tier": 5,         # 1–5 (higher is faster / richer)
    "categories": ["reasoning", "coding", "multimodal"]
}
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

__all__ = [
    "Model",
    "ModelDB",
]

# Type alias for path-like objects
PathLike = Union[str, pathlib.Path]

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
if not logger.handlers:  # Guard against double configuration
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Model dataclass
# --------------------------------------------------------------------------- #

@dataclass(order=True, frozen=True)
class Model:
    """
    Represents a single language model's metadata.

    Parameters
    ----------
    id : str
        Unique identifier used by OpenRouter for lookup.
    name : str
        Human‑readable name.
    provider : str
        The organization that owns the model.
    context_length : Optional[int]
        Maximum input token length. ``None`` indicates unlimited or not applicable.
    cost_per_input_token : float
        Cost in USD for each 1 000 input tokens.
    cost_per_output_token : float
        Cost in USD for each 1 000 output tokens.
    performance_tier : int
        Integer rating 1–5 (5 = highest performance).
    categories : Sequence[str]
        Broad capability tags (e.g. ``['coding', 'reasoning']``).
    """

    id: str
    name: str
    provider: str
    context_length: Optional[int]
    cost_per_input_token: float
    cost_per_output_token: float
    performance_tier: int
    categories: Tuple[str, ...] = field(default_factory=lambda: ())

    def __post_init__(self):
        # Validate numeric fields
        if self.cost_per_input_token < 0:
            raise ValueError(f"Negative cost_per_input_token for model {self.id}")
        if self.cost_per_output_token < 0:
            raise ValueError(f"Negative cost_per_output_token for model {self.id}")
        if not 1 <= self.performance_tier <= 5:
            raise ValueError(f"performance_tier out of bounds for model {self.id}")

    # Convenience aliases ----------------------------------------------------- #
    @property
    def has_context(self) -> bool:
        return self.context_length is not None

    @property
    def cost_per_token(self) -> float:
        """Mean cost per token (input + output) for 1 000 tokens."""
        return (self.cost_per_input_token + self.cost_per_output_token) / 2

    # String representation --------------------------------------------------- #
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"<Model id={self.id!r} provider={self.provider!r} "
            f"context={self.context_length} cost=({self.cost_per_input_token:.3f}/"
            f"{self.cost_per_output_token:.3f} per 1k) tier={self.performance_tier}>"
        )

# --------------------------------------------------------------------------- #
# ModelDB
# --------------------------------------------------------------------------- #

class ModelDB:
    """
    Loads a JSON database of model metadata and offers filtering utilities.

    Parameters
    ----------
    db_path : str | pathlib.Path, optional
        Path to the JSON file. Defaults to ``_models.json`` in the same
        directory as this module.
    """

    def __init__(self, db_path: Optional[PathLike] = None):
        self._path: pathlib.Path = (
            pathlib.Path(db_path) if db_path else pathlib.Path(__file__).with_name("_models.json")
        )
        self._models: Dict[str, Model] = {}
        self._load()

    # ----------------------------------------------------------------------- #
    # Core loading
    # ----------------------------------------------------------------------- #

    def _load(self) -> None:
        """Read JSON file and instantiate :class:`Model` objects."""
        if not self._path.exists():
            logger.error(f"Model database not found at {self._path}")
            raise FileNotFoundError(f"Model database '{self._path}' does not exist")

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.exception(
                f"Failed to parse JSON model database at {self._path}: {exc}"
            )
            raise

        if not isinstance(raw, (list, tuple)):
            logger.error(
                f"Expected top‑level list from model database, got {type(raw).__name__}"
            )
            raise ValueError(f"Invalid JSON format for model database: {self._path}")

        for entry in raw:
            try:
                model = self._model_from_dict(entry)
                if model.id in self._models:
                    logger.warning(f"Duplicate model id {model.id!r} ignored")
                else:
                    self._models[model.id] = model
                    logger.debug(f"Loaded model {model.id}")
            except Exception as exc:  # pragma: no cover
                logger.exception(f"Error loading model entry {entry!r}: {exc}")

    def _model_from_dict(self, data: Dict[str, Any]) -> Model:
        """Instantiate a :class:`Model` from a JSON entry."""
        # Normalise fields & validate mandatory keys
        required = ["id", "name", "provider", "context_length",
                    "cost_per_input_token", "cost_per_output_token",
                    "performance_tier", "categories"]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing required model field(s): {', '.join(missing)}")

        # Some entry may use 'context_length': null, converting to None is fine
        context = data["context_length"]
        if context is None:
            context = None
        elif isinstance(context, int) and context > 0:
            context = int(context)
        else:
            raise ValueError(
                f"Invalid context_length {context!r} for model {data['id']}"
            )

        # Ensure categories is a tuple of strings
        cats = tuple(str(c) for c in data["categories"])

        return Model(
            id=str(data["id"]),
            name=str(data["name"]),
            provider=str(data["provider"]),
            context_length=context,
            cost_per_input_token=float(data["cost_per_input_token"]),
            cost_per_output_token=float(data["cost_per_output_token"]),
            performance_tier=int(data["performance_tier"]),
            categories=cats,
        )

    # ----------------------------------------------------------------------- #
    # Convenience getters
    # ----------------------------------------------------------------------- #

    def get_by_id(self, model_id: str) -> Model:
        """Return a :class:`Model` instance or raise ``KeyError``."""
        try:
            return self._models[model_id]
        except KeyError:
            logger.error(f"Model id {model_id!r} not found")
            raise

    def get_by_name(self, name: str) -> Model:
        """Return first model with matching display name."""
        for m in self._models.values():
            if m.name.lower() == name.lower():
                return m
        logger.error(f"Model name {name!r} not found")
        raise KeyError(f"No model named '{name}'")

    def all_models(self) -> List[Model]:
        """Return a list of all models (unfiltered)."""
        return list(self._models.values())

    # ----------------------------------------------------------------------- #
    # Filtering helpers
    # ----------------------------------------------------------------------- #

    def filter_by_category(self, category: str) -> List[Model]:
        """Return models containing the given category (case‑insensitive)."""
        category = category.lower()
        result = [m for m in self._models.values() if category in (c.lower() for c in m.categories)]
        logger.debug(f"filter_by_category('{category}') -> {len(result)} hits")
        return result

    def filter_by_performance_min(self, min_tier: int) -> List[Model]:
        """Return models with performance_tier >= min_tier."""
        if not 1 <= min_tier <= 5:
            raise ValueError("min_tier must be between 1 and 5")
        result = [m for m in self._models.values() if m.performance_tier >= min_tier]
        logger.debug(f"filter_by_performance_min({min_tier}) -> {len(result)} hits")
        return result

    def filter_by_cost_max(self, max_cost_per_input: float) -> List[Model]:
        """
        Return models whose cost_per_input_token is <= ``max_cost_per_input``.

        ``max_cost_per_input`` is specified in USD per 1 000 tokens.
        """
        if max_cost_per_input < 0:
            raise ValueError("max_cost_per_input must be non‑negative")
        result = [m for m in self._models.values() if m.cost_per_input_token <= max_cost_per_input]
        logger.debug(f"filter_by_cost_max({max_cost_per_input}) -> {len(result)} hits")
        return result

    def filter_by_capability(self, capability: str) -> List[Model]:
        """Alias for :meth:`filter_by_category` to emphasise capability."""
        return self.filter_by_category(capability)

    # ----------------------------------------------------------------------- #
    # Composite filtering
    # ----------------------------------------------------------------------- #

    def filter(
        self,
        *,
        category: Optional[str] = None,
        min_performance: Optional[int] = None,
        max_cost_per_input: Optional[float] = None,
        capability: Optional[str] = None,
        custom_filters: Optional[Iterable[callable]] = None,
    ) -> List[Model]:
        """Perform multi‑condition filtering.

        Parameters
        ----------
        category: str, optional
            Category to match.
        min_performance: int, optional
            Minimum performance tier (1–5).
        max_cost_per_input: float, optional
            Upper bound on cost per 1 000 input tokens.
        capability: str, optional
            Alias for category/search by capability tag.
        custom_filters: iterable of predicates
            Each predicate receives a :class:`Model` and returns bool.

        Returns
        -------
        List[Model]
        """
        filtered = set(self._models.values())

        if category:
            filtered = filtered.intersection(self.filter_by_category(category))
        if min_performance:
            filtered = filtered.intersection(self.filter_by_performance_min(min_performance))
        if max_cost_per_input:
            filtered = filtered.intersection(self.filter_by_cost_max(max_cost_per_input))
        if capability:
            filtered = filtered.intersection(self.filter_by_category(capability))
        if custom_filters:
            for pred in custom_filters:
                filtered = {m for m in filtered if pred(m)}

        result = sorted(filtered, key=lambda m: m.performance_tier, reverse=True)
        logger.debug(f"Composite filter -> {len(result)} hits")
        return result

    # ----------------------------------------------------------------------- #
    # Mutator helpers
    # ----------------------------------------------------------------------- #

    def add_model(self, model: Model) -> None:
        """Add a new model to the in‑memory database."""
        if model.id in self._models:
            raise KeyError(f"Model id {model.id!r} already exists")
        self._models[model.id] = model
        logger.info(f"Added new model {model.id}")

    def remove_model(self, model_id: str) -> None:
        """Remove a model by id."""
        if model_id not in self._models:
            raise KeyError(f"Model id {model_id!r} does not exist")
        del self._models[model_id]
        logger.info(f"Removed model {model_id}")

    def save(self, path: Optional[PathLike] = None) -> None:
        """Persist the current state back to JSON."""
        path = pathlib.Path(path) if path else self._path
        json_data = []
        for m in self._models.values():
            # Convert Model back to dict format
            model_dict = {
                "id": m.id,
                "name": m.name,
                "provider": m.provider,
                "context_length": m.context_length,
                "cost_per_input_token": m.cost_per_input_token,
                "cost_per_output_token": m.cost_per_output_token,
                "performance_tier": m.performance_tier,
                "categories": list(m.categories),
            }
            json_data.append(model_dict)
        
        path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Model database persisted to {path}")

# --------------------------------------------------------------------------- #
# Test harness (would normally be in a separate test file)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    db = ModelDB()
    all_models = db.all_models()
    print(f"Total models: {len(all_models)}")
    print("Top 5 by performance tier:")
    for m in sorted(all_models, key=lambda x: x.performance_tier, reverse=True)[:5]:
        print(f"  {m.id}: tier {m.performance_tier} – {m.name}")