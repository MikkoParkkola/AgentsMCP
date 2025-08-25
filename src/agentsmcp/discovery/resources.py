"""Resource discovery models and utilities.

Enhanced resource model for production-grade discovery system.
Implements Phase 1 improvements: rich resource descriptions, versioning, and health tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Standard resource types for discovery."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    SENSOR = "sensor"
    SERVICE = "service"
    DATABASE = "database"
    QUEUE = "queue"
    CACHE = "cache"
    COMPUTE = "compute"
    ML_MODEL = "ml_model"
    CUSTOM = "custom"


class HealthState(str, Enum):
    """Resource health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    DRAINING = "draining"
    UNKNOWN = "unknown"


class AgentState(str, Enum):
    """Agent lifecycle states."""
    ACTIVE = "active"
    DRAINING = "draining"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class HealthInfo:
    """Health information for a resource or agent."""
    state: HealthState = HealthState.UNKNOWN
    reason: Optional[str] = None
    score: float = 1.0  # 0-1 normalized health score
    last_check: float = field(default_factory=time.time)
    check_interval: float = 60.0  # seconds
    
    def is_healthy(self) -> bool:
        """Check if resource is in a healthy state."""
        return self.state in (HealthState.HEALTHY, HealthState.DEGRADED)
    
    def is_available(self) -> bool:
        """Check if resource is available for new work."""
        return self.state == HealthState.HEALTHY
    
    def update(self, state: HealthState, reason: Optional[str] = None, score: float = 1.0):
        """Update health information."""
        self.state = state
        self.reason = reason
        self.score = max(0.0, min(1.0, score))  # Clamp to 0-1
        self.last_check = time.time()


@dataclass
class ResourceCapacity:
    """Resource capacity and utilization information."""
    total: Dict[str, float] = field(default_factory=dict)
    used: Dict[str, float] = field(default_factory=dict)
    reserved: Dict[str, float] = field(default_factory=dict)
    
    def get_available(self, metric: str) -> float:
        """Get available capacity for a metric."""
        total_val = self.total.get(metric, 0.0)
        used_val = self.used.get(metric, 0.0)
        reserved_val = self.reserved.get(metric, 0.0)
        return max(0.0, total_val - used_val - reserved_val)
    
    def get_utilization(self, metric: str) -> float:
        """Get utilization percentage (0-1) for a metric."""
        total_val = self.total.get(metric, 0.0)
        if total_val <= 0:
            return 0.0
        used_val = self.used.get(metric, 0.0)
        return min(1.0, used_val / total_val)
    
    def can_allocate(self, requirements: Dict[str, float]) -> bool:
        """Check if resource can satisfy the given requirements."""
        for metric, required in requirements.items():
            if self.get_available(metric) < required:
                return False
        return True


@dataclass
class Capability:
    """Enhanced capability description with versioning."""
    name: str
    version: str = "1.0"
    min_version: Optional[str] = None
    max_version: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def is_compatible(self, required_version: str) -> bool:
        """Check if capability version is compatible with required version."""
        # Simplified version comparison - in production use semantic versioning
        try:
            current = tuple(map(int, self.version.split('.')))
            required = tuple(map(int, required_version.split('.')))
            return current >= required
        except ValueError:
            logger.warning(f"Invalid version format: {self.version} vs {required_version}")
            return self.version == required_version


@dataclass
class Resource:
    """Enhanced resource description with rich metadata."""
    type: ResourceType
    id: str
    name: Optional[str] = None
    version: str = "1.0"
    labels: Dict[str, str] = field(default_factory=dict)
    capacity: ResourceCapacity = field(default_factory=ResourceCapacity)
    endpoint: Optional[str] = None
    health: HealthInfo = field(default_factory=HealthInfo)
    location: Optional[Dict[str, Any]] = None  # Geographic or logical location
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def matches_labels(self, required_labels: Dict[str, str]) -> bool:
        """Check if resource matches all required labels."""
        for key, value in required_labels.items():
            if self.labels.get(key) != value:
                return False
        return True
    
    def update_health(self, state: HealthState, reason: Optional[str] = None, score: float = 1.0):
        """Update resource health information."""
        self.health.update(state, reason, score)
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass  
class EnhancedAgent:
    """Enhanced agent with rich resource model and health tracking."""
    agent_id: str
    agent_name: Optional[str] = None
    resources: List[Resource] = field(default_factory=list)
    capabilities: List[Capability] = field(default_factory=list)
    public_key: Optional[str] = None
    transport: Dict[str, str] = field(default_factory=lambda: {"type": "http", "endpoint": "localhost:8000"})
    state: AgentState = AgentState.ACTIVE
    health: HealthInfo = field(default_factory=HealthInfo)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def add_resource(self, resource: Resource):
        """Add a resource to this agent."""
        self.resources.append(resource)
        self.updated_at = time.time()
    
    def remove_resource(self, resource_id: str) -> bool:
        """Remove a resource by ID."""
        original_count = len(self.resources)
        self.resources = [r for r in self.resources if r.id != resource_id]
        if len(self.resources) < original_count:
            self.updated_at = time.time()
            return True
        return False
    
    def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get a resource by ID."""
        for resource in self.resources:
            if resource.id == resource_id:
                return resource
        return None
    
    def get_resources_by_type(self, resource_type: ResourceType) -> List[Resource]:
        """Get all resources of a specific type."""
        return [r for r in self.resources if r.type == resource_type]
    
    def find_resources(self, 
                      resource_type: Optional[ResourceType] = None,
                      labels: Optional[Dict[str, str]] = None,
                      healthy_only: bool = True) -> List[Resource]:
        """Find resources matching criteria."""
        results = self.resources.copy()
        
        if resource_type:
            results = [r for r in results if r.type == resource_type]
        
        if labels:
            results = [r for r in results if r.matches_labels(labels)]
        
        if healthy_only:
            results = [r for r in results if r.health.is_available()]
        
        return results
    
    def has_capability(self, capability_name: str, min_version: Optional[str] = None) -> bool:
        """Check if agent has a specific capability."""
        for cap in self.capabilities:
            if cap.name == capability_name:
                if min_version is None or cap.is_compatible(min_version):
                    return True
        return False
    
    def is_available(self) -> bool:
        """Check if agent is available for new work."""
        return (self.state == AgentState.ACTIVE and 
                self.health.is_available())
    
    def update_health(self, state: HealthState, reason: Optional[str] = None, score: float = 1.0):
        """Update agent health information."""
        self.health.update(state, reason, score)
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedAgent':
        """Create EnhancedAgent from dictionary."""
        # Handle nested objects
        resources_data = data.get('resources', [])
        resources = []
        for res_data in resources_data:
            if 'capacity' in res_data and isinstance(res_data['capacity'], dict):
                res_data['capacity'] = ResourceCapacity(**res_data['capacity'])
            if 'health' in res_data and isinstance(res_data['health'], dict):
                res_data['health'] = HealthInfo(**res_data['health'])
            resources.append(Resource(**res_data))
        
        capabilities_data = data.get('capabilities', [])
        capabilities = []
        for cap_data in capabilities_data:
            if isinstance(cap_data, dict):
                capabilities.append(Capability(**cap_data))
            else:
                # Handle legacy string capabilities
                capabilities.append(Capability(name=str(cap_data)))
        
        health_data = data.get('health', {})
        health = HealthInfo(**health_data) if isinstance(health_data, dict) else HealthInfo()
        
        return cls(
            agent_id=data['agent_id'],
            agent_name=data.get('agent_name'),
            resources=resources,
            capabilities=capabilities,
            public_key=data.get('public_key'),
            transport=data.get('transport', {}),
            state=AgentState(data.get('state', AgentState.ACTIVE)),
            health=health,
            metadata=data.get('metadata', {}),
            schema_version=data.get('schema_version', 2),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time())
        )


# Utility functions for resource discovery
def create_compute_resource(
    resource_id: str,
    cores: int,
    memory_gb: float,
    endpoint: str,
    labels: Optional[Dict[str, str]] = None
) -> Resource:
    """Create a compute resource with CPU and memory capacity."""
    capacity = ResourceCapacity(
        total={"cores": float(cores), "memory_gb": memory_gb},
        used={"cores": 0.0, "memory_gb": 0.0}
    )
    
    return Resource(
        type=ResourceType.COMPUTE,
        id=resource_id,
        capacity=capacity,
        endpoint=endpoint,
        labels=labels or {},
        health=HealthInfo(state=HealthState.HEALTHY)
    )


def create_gpu_resource(
    resource_id: str,
    gpu_count: int,
    vram_gb: float,
    endpoint: str,
    gpu_type: str = "generic",
    labels: Optional[Dict[str, str]] = None
) -> Resource:
    """Create a GPU resource."""
    capacity = ResourceCapacity(
        total={"gpu_count": float(gpu_count), "vram_gb": vram_gb},
        used={"gpu_count": 0.0, "vram_gb": 0.0}
    )
    
    resource_labels = labels or {}
    resource_labels["gpu_type"] = gpu_type
    
    return Resource(
        type=ResourceType.GPU,
        id=resource_id,
        capacity=capacity,
        endpoint=endpoint,
        labels=resource_labels,
        health=HealthInfo(state=HealthState.HEALTHY)
    )


def create_service_resource(
    resource_id: str,
    service_name: str,
    endpoint: str,
    version: str = "1.0",
    labels: Optional[Dict[str, str]] = None
) -> Resource:
    """Create a service resource."""
    return Resource(
        type=ResourceType.SERVICE,
        id=resource_id,
        name=service_name,
        version=version,
        endpoint=endpoint,
        labels=labels or {},
        health=HealthInfo(state=HealthState.HEALTHY)
    )