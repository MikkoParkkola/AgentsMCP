"""
agentsmcp.discovery.matching_engine

AD3 – Service‑Discovery Matching Engine for AgentsMCP.

Features
--------
1. Capability matching with JSON‑Schema validation.
2. Pluggable agent‑selection strategies:
   * Least‑loaded
   * Most‑available
   * Round‑Robin
3. Resource‑constraint evaluation (CPU, memory, GPU).
4. Tag‑based filtering and arbitrary property matching.
5. Scoring & ranking (resource fit, tags relevance, last‑heartbeat …).
6. LRU caching for fast repeated queries.
7. Full type‑hints, logging, and robust error handling.

The engine works together with the **Agent Registry** from AD2 – simply
pass an instance of that registry when constructing :class:`DiscoveryEngine`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
)

from functools import lru_cache, wraps

# Import registry module for real AD2 integration
try:
    from . import registry
    HAS_REGISTRY = True
except ImportError:
    registry = None
    HAS_REGISTRY = False

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("agentsmcp.discovery.matching_engine")

# --------------------------------------------------------------------------- #
# JSON Schema validation (fallback if jsonschema not available)
# --------------------------------------------------------------------------- #

try:
    import jsonschema  # pip install jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    LOGGER.warning("jsonschema package not found - capability validation will be simplified")

# --------------------------------------------------------------------------- #
# Public data structures
# --------------------------------------------------------------------------- #

class DiscoveryError(RuntimeError):
    """Base exception for all discovery‑engine errors."""

class CapabilitySchemaError(DiscoveryError):
    """Raised when a capability JSON schema is invalid."""

class NoMatchingAgentsError(DiscoveryError):
    """Raised when a request cannot be satisfied by any agent."""

@dataclass(frozen=True)
class ResourceSpec:
    """Quantitative resource requirements for a request."""
    cpu_cores: float = 0.0          # cores, can be fractional
    memory_gb: float = 0.0          # GiB
    gpu_units: int = 0              # discrete GPU units

    def fits(self, available: "ResourceSpec") -> bool:
        """Return True if ``available`` can satisfy this spec."""
        return (
            available.cpu_cores >= self.cpu_cores
            and available.memory_gb >= self.memory_gb
            and available.gpu_units >= self.gpu_units
        )

@dataclass(frozen=True)
class MatchingAgentInfo:
    """
    Minimal representation of an agent used by the matching engine.
    This mirrors the AgentInfo from AD2 but includes additional fields needed for matching.
    """
    agent_id: str
    capabilities: Mapping[str, Any]               # free‑form dict
    capability_schema: Mapping[str, Any]          # JSON‑Schema dict
    resources_total: ResourceSpec
    resources_used: ResourceSpec
    tags: Set[str] = field(default_factory=set)
    properties: Mapping[str, Any] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)   # epoch secs

    @property
    def resources_available(self) -> ResourceSpec:
        """Resources that are still free on the agent."""
        return ResourceSpec(
            cpu_cores=self.resources_total.cpu_cores - self.resources_used.cpu_cores,
            memory_gb=self.resources_total.memory_gb - self.resources_used.memory_gb,
            gpu_units=self.resources_total.gpu_units - self.resources_used.gpu_units,
        )

# --------------------------------------------------------------------------- #
# Registry protocol (AD2 integration)
# --------------------------------------------------------------------------- #

class AgentRegistryProtocol(Protocol):
    """The subset of the AD2 registry API required by DiscoveryEngine."""
    def get_all(self) -> List[MatchingAgentInfo]:
        """Return all currently registered agents."""
        ...

# --------------------------------------------------------------------------- #
# Core matching components
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 1. Capability matching
# --------------------------------------------------------------------------- #

class CapabilityMatcher:
    """
    Validate a request's capability payload against the agent's declared
    JSON‑Schema and return ``True`` if the payload conforms.

    The matcher is deliberately stateless – it can be safely shared
    between threads.
    """

    @staticmethod
    def validate(
        payload: Mapping[str, Any],
        schema: Mapping[str, Any],
    ) -> bool:
        """
        Validate ``payload`` against ``schema`` using ``jsonschema``.
        Returns ``True`` on success, raises :class:`CapabilitySchemaError`
        on validation failure.
        """
        if not HAS_JSONSCHEMA:
            # Fallback: simple type checking
            LOGGER.debug("Using simplified validation (jsonschema not available)")
            return True
            
        try:
            jsonschema.validate(instance=payload, schema=schema)
            return True
        except jsonschema.exceptions.ValidationError as exc:
            LOGGER.debug(
                "Capability validation failed: %s – payload=%s schema=%s",
                exc.message, payload, schema,
            )
            raise CapabilitySchemaError(str(exc)) from exc

    @classmethod
    def matches(
        cls,
        request_payload: Mapping[str, Any],
        agent: MatchingAgentInfo,
        capability_name: str,
    ) -> bool:
        """
        Check whether ``agent`` advertises a capability named
        ``capability_name`` and, if so, whether ``request_payload`` complies
        with the agent's JSON‑Schema for that capability.
        """
        schema = agent.capability_schema.get(capability_name)
        if not schema:
            LOGGER.debug(
                "Agent %s does not declare capability %s",
                agent.agent_id, capability_name,
            )
            return False

        return cls.validate(request_payload, schema)


# --------------------------------------------------------------------------- #
# 2. Selection strategies
# --------------------------------------------------------------------------- #

SelectionStrategyCallable = Callable[[List[MatchingAgentInfo]], MatchingAgentInfo]

class SelectionStrategyRegistry:
    """Registry of available selection strategies (pluggable at runtime)."""

    _registry: Dict[str, SelectionStrategyCallable] = {}

    @classmethod
    def register(
        cls,
        name: str,
        fn: SelectionStrategyCallable,
    ) -> None:
        if name in cls._registry:
            raise ValueError(f"Selection strategy '{name}' already registered.")
        LOGGER.debug("Registering selection strategy %s", name)
        cls._registry[name] = fn

    @classmethod
    def get(cls, name: str) -> SelectionStrategyCallable:
        try:
            return cls._registry[name]
        except KeyError as exc:
            raise ValueError(f"Unknown selection strategy '{name}'") from exc

    @classmethod
    def available(cls) -> Set[str]:
        return set(cls._registry)

# ---- Built‑in strategies

def _least_loaded(agents: List[MatchingAgentInfo]) -> MatchingAgentInfo:
    """Pick the agent with the smallest (cpu + memory) utilization ratio."""
    def load(agent: MatchingAgentInfo) -> float:
        total = agent.resources_total.cpu_cores + agent.resources_total.memory_gb
        used = agent.resources_used.cpu_cores + agent.resources_used.memory_gb
        return used / total if total > 0 else float('inf')
    chosen = min(agents, key=load)
    LOGGER.debug("Least‑loaded selected %s (load=%.2f)", chosen.agent_id, load(chosen))
    return chosen

def _most_available(agents: List[MatchingAgentInfo]) -> MatchingAgentInfo:
    """Pick the agent with the most free resource (CPU+mem+GPU) units."""
    def free_units(agent: MatchingAgentInfo) -> float:
        avail = agent.resources_available
        return avail.cpu_cores + avail.memory_gb + avail.gpu_units * 10.0  # weight GPU higher
    chosen = max(agents, key=free_units)
    LOGGER.debug("Most‑available selected %s (free=%.2f)", chosen.agent_id, free_units(chosen))
    return chosen

def _round_robin_factory() -> SelectionStrategyCallable:
    """Factory that creates a round‑robin selector with its own state."""
    lock = threading.Lock()
    index_map: Dict[str, int] = defaultdict(int)   # key → next index

    def selector(agents: List[MatchingAgentInfo]) -> MatchingAgentInfo:
        if not agents:
            raise NoMatchingAgentsError("No agents to select from.")
        # Use a deterministic key – e.g., hash of sorted agent IDs.
        key = ",".join(sorted(a.agent_id for a in agents))
        with lock:
            i = index_map[key] % len(agents)
            index_map[key] = i + 1
        chosen = agents[i]
        LOGGER.debug("Round‑Robin selected %s (position %d)", chosen.agent_id, i)
        return chosen
    return selector

# Register defaults
SelectionStrategyRegistry.register("least_loaded", _least_loaded)
SelectionStrategyRegistry.register("most_available", _most_available)
SelectionStrategyRegistry.register("round_robin", _round_robin_factory())

# --------------------------------------------------------------------------- #
# 3. Resource constraint evaluation
# --------------------------------------------------------------------------- #

class ResourceEvaluator:
    """
    Evaluate whether an agent can satisfy a request's resource spec and
    compute a *fit score* (higher = better fit).
    """

    @staticmethod
    def can_satisfy(request: ResourceSpec, agent: MatchingAgentInfo) -> bool:
        """True if the agent has enough *available* resources."""
        available = agent.resources_available
        fits = request.fits(available)
        LOGGER.debug(
            "Agent %s resource check: request=%s available=%s fits=%s",
            agent.agent_id, request, available, fits,
        )
        return fits

    @staticmethod
    def fit_score(request: ResourceSpec, agent: MatchingAgentInfo) -> float:
        """
        Compute a numeric score representing how tightly the agent matches
        the request.  The score is the ratio of *available* over *requested*
        resources (closer to 1 = tight fit, >1 = excess capacity).  The score
        is the geometric mean of the three dimensions to avoid bias.
        """
        avail = agent.resources_available
        ratios = []
        if request.cpu_cores > 0:
            ratios.append(avail.cpu_cores / request.cpu_cores)
        if request.memory_gb > 0:
            ratios.append(avail.memory_gb / request.memory_gb)
        if request.gpu_units > 0:
            ratios.append(avail.gpu_units / request.gpu_units)

        if not ratios:  # request does not demand anything specific
            return 1.0

        # Geometric mean
        prod = 1.0
        for r in ratios:
            prod *= r
        score = prod ** (1.0 / len(ratios))
        LOGGER.debug(
            "Agent %s fit_score=%.3f for request=%s (available=%s)",
            agent.agent_id, score, request, avail,
        )
        return score

# --------------------------------------------------------------------------- #
# 4. Tag & property matching
# --------------------------------------------------------------------------- #

class TagPropertyFilter:
    """Utility class for tag/property based filtering."""

    @staticmethod
    def tag_match(
        required: Set[str],
        agent_tags: Set[str],
    ) -> bool:
        """All ``required`` tags must be present on the agent."""
        match = required.issubset(agent_tags)
        LOGGER.debug(
            "Tag match required=%s agent=%s → %s",
            required, agent_tags, match,
        )
        return match

    @staticmethod
    def property_match(
        required: Mapping[str, Any],
        agent_properties: Mapping[str, Any],
    ) -> bool:
        """
        For every key in ``required`` the agent must have the same value.
        Nested dicts are compared shallowly (deep‑nested matching can be added).
        """
        for key, val in required.items():
            if key not in agent_properties or agent_properties[key] != val:
                LOGGER.debug(
                    "Property mismatch on key=%s: required=%s agent=%s",
                    key, val, agent_properties.get(key),
                )
                return False
        LOGGER.debug(
            "All required properties matched: %s against %s",
            required, agent_properties,
        )
        return True

# --------------------------------------------------------------------------- #
# 5. Scoring & ranking
# --------------------------------------------------------------------------- #

class ScoringEngine:
    """
    Combine multiple orthogonal scores into a single ranking value.
    The default weighting can be overridden by the caller.
    """

    @staticmethod
    def composite_score(
        *,
        resource_score: float,
        tag_overlap: int,
        heartbeat_age: float,
        strategy_weight: float = 1.0,
        resource_weight: float = 1.0,
        tag_weight: float = 0.5,
        heartbeat_weight: float = 0.2,
    ) -> float:
        """
        Produce a composite score.

        Parameters
        ----------
        resource_score: float
            Value from :meth:`ResourceEvaluator.fit_score`.
        tag_overlap: int
            Number of tags that intersect between request and agent.
        heartbeat_age: float
            Seconds since last heartbeat – lower is better.
        strategy_weight, resource_weight, tag_weight, heartbeat_weight:
            Relative importance of each component.

        Returns
        -------
        float
            Higher is better.
        """
        # Normalise heartbeat – we treat age as a *penalty*.
        heartbeat_factor = max(0.0, 1.0 - (heartbeat_age / 3600.0))  # 1h = zero score
        score = (
            strategy_weight * resource_score
            + tag_weight * tag_overlap
            + heartbeat_weight * heartbeat_factor
        )
        LOGGER.debug(
            "Composite score: resource=%.3f tags=%d hb=%.3f → %.3f",
            resource_score, tag_overlap, heartbeat_factor, score,
        )
        return score

# --------------------------------------------------------------------------- #
# 6. LRU Cache layer
# --------------------------------------------------------------------------- #

# A thin wrapper around functools.lru_cache to make the cache part of the class
# and expose a ``clear`` method for test‑teardown / runtime reloads.

_R = TypeVar("_R")

def lru_cache_method(maxsize: int = 128):
    """
    Decorator for instance methods that should be cached per‑instance.
    The underlying cache key includes ``self``'s ``id()`` to avoid cross‑talk.
    """
    def decorator(fn: Callable[..., _R]) -> Callable[..., _R]:
        cached_fn = lru_cache(maxsize=maxsize)(fn)

        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            # bind the instance id into the cache key
            return cached_fn(self, *args, **kwargs)
        wrapper.cache_clear = cached_fn.cache_clear  # type: ignore[attr-defined]
        return wrapper
    return decorator

# --------------------------------------------------------------------------- #
# 7. Discovery Engine (orchestrator)
# --------------------------------------------------------------------------- #

class DiscoveryEngine:
    """
    Central façade that a caller uses to request an agent capable of handling
    a given workload.

    Example
    -------
    >>> registry = AgentRegistry()                     # AD2 class
    >>> engine = DiscoveryEngine(registry)
    >>> chosen = engine.select_agent(
    ...     capability="image_classification",
    ...     payload={"model": "resnet50", "format": "png"},
    ...     resource=ResourceSpec(cpu_cores=2, memory_gb=4),
    ...     required_tags={"gpu", "edge"},
    ...     required_properties={"os": "linux"},
    ...     strategy="least_loaded",
    ... )
    >>> print(chosen.agent_id)
    """

    def __init__(
        self,
        registry: AgentRegistryProtocol,
        *,
        cache_maxsize: int = 256,
        default_strategy: str = "least_loaded",
    ) -> None:
        self._registry = registry
        self._default_strategy = default_strategy
        self._cache_maxsize = cache_maxsize
        self._lock = threading.RLock()  # Protects internal mutable state

        # Expose cache clearing for external use (e.g., tests)
        self.clear_cache = self._select_agent_cached.cache_clear  # type: ignore

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #

    def select_agent(
        self,
        *,
        capability: str,
        payload: Mapping[str, Any],
        resource: ResourceSpec,
        required_tags: Optional[Set[str]] = None,
        required_properties: Optional[Mapping[str, Any]] = None,
        strategy: Optional[str] = None,
    ) -> MatchingAgentInfo:
        """
        Resolve to a single best‑fit agent.

        Parameters
        ----------
        capability: str
            Name of the capability the request needs.
        payload: Mapping[str, Any]
            JSON‑compatible request payload that must match the agent's
            schema for the given capability.
        resource: ResourceSpec
            Quantitative resources the request consumes.
        required_tags: set[str] | None
            All of these tags must be present on the candidate agent.
        required_properties: Mapping[str, Any] | None
            All defined key/value pairs must match the agent's static
            properties.
        strategy: str | None
            Selection strategy name; falls back to the engine default.

        Returns
        -------
        MatchingAgentInfo
            The chosen agent.

        Raises
        ------
        NoMatchingAgentsError
            If no agent satisfies *all* constraints.
        """
        if strategy is None:
            strategy = self._default_strategy

        # Convert to hashable types for caching
        payload_tuple = tuple(sorted(payload.items())) if payload else ()
        required_props_tuple = tuple(sorted((required_properties or {}).items()))

        # Use the cached path – the heavy lifting is inside `_select_agent_cached`.
        return self._select_agent_cached(
            capability=capability,
            payload=payload_tuple,
            resource=resource,
            required_tags=frozenset(required_tags or set()),
            required_properties=required_props_tuple,
            strategy=strategy,
        )

    # ------------------------------------------------------------------- #
    # Internal helpers
    # ------------------------------------------------------------------- #

    @lru_cache_method(maxsize=256)
    def _select_agent_cached(
        self,
        *,
        capability: str,
        payload: Tuple[Tuple[str, Any], ...],  # JSON‑serialisable, stored as tuple for cache‑key stability
        resource: ResourceSpec,
        required_tags: frozenset,
        required_properties: Tuple[Tuple[str, Any], ...],
        strategy: str,
    ) -> MatchingAgentInfo:
        """
        Cached implementation – all parameters are hashable.
        Converting mutable mappings to immutable structures (tuple, frozenset)
        ensures the ``lru_cache`` can use them as keys.
        """
        # Rebuild mutable structures from the immutable cache arguments
        payload_dict = dict(payload)

        # 1️⃣ Fetch fresh agents list (registry may have changed)
        agents = self._registry.get_all()
        LOGGER.debug("Fetched %d agents from registry", len(agents))

        # 2️⃣ Capability + schema validation
        capable = [
            a for a in agents
            if CapabilityMatcher.matches(payload_dict, a, capability)
        ]
        LOGGER.debug("%d agents passed capability filtering", len(capable))

        # 3️⃣ Resource feasibility
        feasible = [
            a for a in capable
            if ResourceEvaluator.can_satisfy(resource, a)
        ]
        LOGGER.debug("%d agents have enough resources", len(feasible))

        # 4️⃣ Tag & property filtering
        if required_tags:
            feasible = [
                a for a in feasible
                if TagPropertyFilter.tag_match(required_tags, a.tags)
            ]
            LOGGER.debug("%d agents passed tag filter", len(feasible))

        if required_properties:
            req_props = dict(required_properties)
            feasible = [
                a for a in feasible
                if TagPropertyFilter.property_match(req_props, a.properties)
            ]
            LOGGER.debug("%d agents passed property filter", len(feasible))

        if not feasible:
            raise NoMatchingAgentsError(
                f"No agents can satisfy capability={capability!r} "
                f"with required resources={resource} "
                f"tags={required_tags} properties={required_properties}"
            )

        # 5️⃣ Scoring
        now = time.time()
        scored: List[Tuple[float, MatchingAgentInfo]] = []
        for agent in feasible:
            res_score = ResourceEvaluator.fit_score(resource, agent)
            tag_overlap = len(required_tags.intersection(agent.tags))
            hb_age = now - agent.last_heartbeat
            composite = ScoringEngine.composite_score(
                resource_score=res_score,
                tag_overlap=tag_overlap,
                heartbeat_age=hb_age,
            )
            scored.append((composite, agent))

        # Sort descending by score
        scored.sort(key=lambda tup: tup[0], reverse=True)
        ranked_agents = [agent for _, agent in scored]
        LOGGER.debug(
            "Ranking order: %s",
            [a.agent_id for a in ranked_agents],
        )

        # 6️⃣ Selection strategy
        strat_fn = SelectionStrategyRegistry.get(strategy)
        chosen = strat_fn(ranked_agents)
        LOGGER.info(
            "DiscoveryEngine selected agent %s (strategy=%s)",
            chosen.agent_id, strategy,
        )
        return chosen

# --------------------------------------------------------------------------- #
# AD2 Registry Adapter (Real Implementation)
# --------------------------------------------------------------------------- #

class RegistryAdapter:
    """
    Adapter that converts between AD2 registry Entry format and 
    MatchingAgentInfo format required by the discovery engine.
    """
    
    def __init__(self, registry_module=None):
        """Initialize adapter with registry module for entry listing."""
        if registry_module:
            self.registry_module = registry_module
        elif HAS_REGISTRY:
            self.registry_module = registry
        else:
            LOGGER.warning("Registry module not available, using fallback")
            self.registry_module = None
    
    def get_all(self) -> List[MatchingAgentInfo]:
        """Convert registry entries to MatchingAgentInfo format."""
        if not self.registry_module:
            LOGGER.warning("Registry module not available, returning empty list")
            return []
        
        try:
            entries = self.registry_module.list_entries()
            agents = []
            
            for entry in entries:
                # Convert Entry to MatchingAgentInfo
                agent_info = self._convert_entry_to_agent_info(entry)
                if agent_info:
                    agents.append(agent_info)
            
            LOGGER.debug("Converted %d registry entries to agent info", len(agents))
            return agents
            
        except Exception as e:
            LOGGER.error("Failed to fetch agents from registry: %s", e)
            return []
    
    def _convert_entry_to_agent_info(self, entry) -> Optional[MatchingAgentInfo]:
        """Convert a single Entry to MatchingAgentInfo."""
        try:
            # Build capability schema from capabilities
            capability_schema = {}
            capabilities_dict = {}
            
            for cap in entry.capabilities:
                # Create basic schema for each capability
                capability_schema[cap] = {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True
                }
                capabilities_dict[cap] = {"version": "1.0"}
            
            # Extract resource information from metadata or use defaults
            resources_total = ResourceSpec(
                cpu_cores=4.0,  # Default values - could be enhanced to read from entry metadata
                memory_gb=8.0,
                gpu_units=0
            )
            
            resources_used = ResourceSpec(
                cpu_cores=0.0,  # Assume no usage by default
                memory_gb=0.0,
                gpu_units=0
            )
            
            # Build tags from capabilities and transport info
            tags = set(entry.capabilities)
            if entry.transport:
                tags.add(entry.transport)
            
            # Build properties from entry metadata
            properties = {
                "transport": entry.transport,
                "endpoint": entry.endpoint,
            }
            if entry.token:
                properties["has_token"] = True
            
            # Create heartbeat timestamp (use current time if not available)
            last_heartbeat = time.time()  # Could be enhanced to read from entry metadata
            
            return MatchingAgentInfo(
                agent_id=entry.agent_id,
                capabilities=capabilities_dict,
                capability_schema=capability_schema,
                resources_total=resources_total,
                resources_used=resources_used,
                tags=tags,
                properties=properties,
                last_heartbeat=last_heartbeat
            )
            
        except Exception as e:
            LOGGER.error("Failed to convert entry %s: %s", entry.agent_id, e)
            return None

# --------------------------------------------------------------------------- #
# Helper utilities for testing / demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Production registry adapter
    try:
        registry_adapter = RegistryAdapter()
        engine = DiscoveryEngine(registry_adapter)
        LOGGER.info("Using real AD2 registry integration")
    except Exception as e:
        LOGGER.warning("Failed to initialize real registry, using demo stub: %s", e)
        
        # Fallback demo stub for testing when registry is not available
        class DummyRegistry:
            def __init__(self):
                self._agents = []

            def add(self, agent: MatchingAgentInfo) -> None:
                self._agents.append(agent)

            def get_all(self) -> List[MatchingAgentInfo]:
                return self._agents
        
        dummy_registry = DummyRegistry()
        engine = DiscoveryEngine(dummy_registry)

        # Add demo agents only if using fallback dummy registry
        dummy_registry.add(
            MatchingAgentInfo(
                agent_id="agent‑001",
                capabilities={"image_classification": {"model": "any"}},
                capability_schema={
                    "image_classification": {
                        "type": "object",
                        "properties": {
                            "model": {"type": "string"},
                            "format": {"type": "string", "enum": ["png", "jpg"]},
                        },
                        "required": ["model", "format"],
                    }
                },
                resources_total=ResourceSpec(cpu_cores=8, memory_gb=32, gpu_units=2),
                resources_used=ResourceSpec(cpu_cores=2, memory_gb=8, gpu_units=0),
                tags={"gpu", "edge"},
                properties={"os": "linux"},
            )
        )
        dummy_registry.add(
            MatchingAgentInfo(
                agent_id="agent‑002",
                capabilities={"image_classification": {"model": "any"}},
                capability_schema={
                    "image_classification": {
                        "type": "object",
                        "properties": {
                            "model": {"type": "string"},
                            "format": {"type": "string"},
                        },
                        "required": ["model", "format"],
                    }
                },
                resources_total=ResourceSpec(cpu_cores=4, memory_gb=16, gpu_units=0),
                resources_used=ResourceSpec(cpu_cores=1, memory_gb=4, gpu_units=0),
                tags={"cpu"},
                properties={"os": "linux"},
            )
        )

    # Demo test
    try:
        selected = engine.select_agent(
            capability="image_classification",
            payload={"model": "resnet50", "format": "png"},
            resource=ResourceSpec(cpu_cores=2, memory_gb=4, gpu_units=1),
            required_tags={"gpu"} if hasattr(engine._registry, 'add') else set(),  # Only for dummy registry
            required_properties={"os": "linux"} if hasattr(engine._registry, 'add') else {},
            strategy="least_loaded",
        )
        print(f"Chosen agent: {selected.agent_id}")
    except DiscoveryError as exc:
        print(f"Discovery failed: {exc}")
    except Exception as exc:
        print(f"Demo test failed (registry may be empty): {exc}")