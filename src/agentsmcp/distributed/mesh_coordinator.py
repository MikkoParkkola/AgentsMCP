"""
Agentic AI Mesh Architecture for peer-to-peer agent collaboration.

This module implements a sophisticated mesh network where agents can collaborate
directly without bottlenecking through a central orchestrator.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import networkx as nx
import json
import uuid

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels for agent interactions."""
    UNTRUSTED = 0.0
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    FULL = 1.0


class CollaborationType(Enum):
    """Types of agent collaboration patterns."""
    DELEGATION = "delegation"
    CONSULTATION = "consultation"
    PARALLEL_EXECUTION = "parallel_execution"
    PIPELINE = "pipeline"
    CONSENSUS = "consensus"


@dataclass
class AgentCapability:
    """Represents a specific capability an agent possesses."""
    name: str
    proficiency: float  # 0.0 to 1.0
    cost_per_token: float
    avg_execution_time: float
    reliability_score: float
    specializations: List[str] = field(default_factory=list)


@dataclass
class CollaborationRequest:
    """Request for agent-to-agent collaboration."""
    request_id: str = field(default_factory=lambda: f"collab_{uuid.uuid4().hex[:8]}")
    requesting_agent: str = ""
    target_agent: Optional[str] = None  # None for broadcast
    collaboration_type: CollaborationType = CollaborationType.CONSULTATION
    task_description: str = ""
    required_capabilities: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    max_cost: Optional[float] = None
    deadline: Optional[datetime] = None
    trust_required: TrustLevel = TrustLevel.MEDIUM


@dataclass
class CollaborationResponse:
    """Response to a collaboration request."""
    request_id: str
    responding_agent: str
    accepted: bool
    cost_estimate: Optional[float] = None
    time_estimate: Optional[timedelta] = None
    confidence: float = 0.0
    alternative_agents: List[str] = field(default_factory=list)
    message: str = ""


class AgentMeshCoordinator:
    """
    Coordinates peer-to-peer agent collaboration in a mesh network.
    
    Key features:
    - Dynamic trust scoring based on performance
    - Capability-based agent matching
    - Direct peer-to-peer communication
    - Collaboration pattern optimization
    - Performance monitoring and adaptation
    """
    
    def __init__(self, max_mesh_size: int = 50):
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.capabilities_map: Dict[str, List[AgentCapability]] = {}
        self.collaboration_graph = nx.DiGraph()
        self.trust_scores: Dict[Tuple[str, str], float] = {}
        self.performance_history: Dict[str, List[Dict]] = {}
        self.active_collaborations: Dict[str, CollaborationRequest] = {}
        self.max_mesh_size = max_mesh_size
        self.collaboration_patterns: Dict[str, Any] = {}
        
        # Initialize collaboration pattern templates
        self._initialize_collaboration_patterns()
    
    def _initialize_collaboration_patterns(self):
        """Initialize common collaboration patterns."""
        self.collaboration_patterns = {
            "code_review": {
                "type": CollaborationType.CONSULTATION,
                "required_capabilities": ["code_analysis", "security_review"],
                "trust_required": TrustLevel.HIGH,
                "parallel_reviewers": 2
            },
            "data_pipeline": {
                "type": CollaborationType.PIPELINE,
                "stages": ["extraction", "transformation", "analysis", "visualization"],
                "trust_required": TrustLevel.MEDIUM
            },
            "creative_brainstorm": {
                "type": CollaborationType.CONSENSUS,
                "required_capabilities": ["creative_thinking", "problem_solving"],
                "min_participants": 3,
                "trust_required": TrustLevel.LOW
            }
        }
    
    async def register_agent(self, 
                           agent_id: str, 
                           capabilities: List[AgentCapability],
                           agent_metadata: Dict[str, Any] = None) -> bool:
        """Register a new agent in the mesh network."""
        if len(self.agent_registry) >= self.max_mesh_size:
            logger.warning(f"Mesh at capacity ({self.max_mesh_size}), rejecting {agent_id}")
            return False
        
        self.agent_registry[agent_id] = {
            "registered_at": datetime.utcnow(),
            "status": "active",
            "metadata": agent_metadata or {},
            "collaboration_count": 0,
            "success_rate": 1.0
        }
        
        self.capabilities_map[agent_id] = capabilities
        self.performance_history[agent_id] = []
        
        # Add to collaboration graph
        self.collaboration_graph.add_node(agent_id, **self.agent_registry[agent_id])
        
        # Initialize trust scores with other agents
        for other_agent in self.agent_registry:
            if other_agent != agent_id:
                self.trust_scores[(agent_id, other_agent)] = TrustLevel.MEDIUM.value
                self.trust_scores[(other_agent, agent_id)] = TrustLevel.MEDIUM.value
        
        logger.info(f"Registered agent {agent_id} with {len(capabilities)} capabilities")
        return True
    
    def find_best_collaborator(self, 
                              request: CollaborationRequest) -> Optional[str]:
        """Find the best agent to collaborate with for a given request."""
        if request.target_agent and request.target_agent in self.agent_registry:
            return request.target_agent
        
        candidates = []
        
        for agent_id, capabilities in self.capabilities_map.items():
            if agent_id == request.requesting_agent:
                continue
            
            # Check trust level
            trust = self.trust_scores.get((request.requesting_agent, agent_id), 0.5)
            if trust < request.trust_required.value:
                continue
            
            # Check capabilities match
            capability_match = 0.0
            total_cost = 0.0
            
            for req_cap in request.required_capabilities:
                for agent_cap in capabilities:
                    if req_cap in agent_cap.name or req_cap in agent_cap.specializations:
                        capability_match += agent_cap.proficiency
                        total_cost += agent_cap.cost_per_token * 1000  # Estimated tokens
                        break
            
            if capability_match == 0:
                continue
            
            # Apply cost constraint
            if request.max_cost and total_cost > request.max_cost:
                continue
            
            # Calculate overall score
            performance = self.agent_registry[agent_id]["success_rate"]
            availability = 1.0 if self.agent_registry[agent_id]["status"] == "active" else 0.0
            
            score = (capability_match * 0.4 + 
                    trust * 0.3 + 
                    performance * 0.2 + 
                    availability * 0.1)
            
            candidates.append((agent_id, score, total_cost))
        
        if not candidates:
            return None
        
        # Sort by score, then by cost
        candidates.sort(key=lambda x: (-x[1], x[2]))
        return candidates[0][0]
    
    async def request_collaboration(self, request: CollaborationRequest) -> Optional[str]:
        """Initiate a collaboration request."""
        best_collaborator = self.find_best_collaborator(request)
        
        if not best_collaborator:
            logger.warning(f"No suitable collaborator found for request {request.request_id}")
            return None
        
        # Store active collaboration
        self.active_collaborations[request.request_id] = request
        
        # Add collaboration edge to graph
        self.collaboration_graph.add_edge(
            request.requesting_agent, 
            best_collaborator,
            collaboration_type=request.collaboration_type.value,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Initiated collaboration {request.request_id}: "
                   f"{request.requesting_agent} -> {best_collaborator}")
        
        return best_collaborator
    
    def update_trust_score(self, 
                          agent_a: str, 
                          agent_b: str, 
                          performance_score: float,
                          collaboration_outcome: str):
        """Update trust score based on collaboration outcome."""
        current_trust = self.trust_scores.get((agent_a, agent_b), 0.5)
        
        # Adaptive trust update based on outcome
        if collaboration_outcome == "success":
            trust_delta = 0.1 * (1.0 - current_trust)  # Diminishing returns
        elif collaboration_outcome == "partial_success":
            trust_delta = 0.05 * (1.0 - current_trust)
        elif collaboration_outcome == "failure":
            trust_delta = -0.2 * current_trust  # Faster trust degradation
        else:
            trust_delta = 0
        
        new_trust = max(0.0, min(1.0, current_trust + trust_delta))
        self.trust_scores[(agent_a, agent_b)] = new_trust
        
        # Update performance history
        performance_entry = {
            "timestamp": datetime.utcnow(),
            "collaborator": agent_b,
            "outcome": collaboration_outcome,
            "performance_score": performance_score,
            "trust_after": new_trust
        }
        
        if agent_a not in self.performance_history:
            self.performance_history[agent_a] = []
        self.performance_history[agent_a].append(performance_entry)
        
        # Update agent success rate
        recent_performances = self.performance_history[agent_a][-10:]  # Last 10
        success_count = sum(1 for p in recent_performances 
                           if p["outcome"] in ["success", "partial_success"])
        self.agent_registry[agent_a]["success_rate"] = success_count / len(recent_performances)
    
    def get_collaboration_recommendations(self, 
                                       agent_id: str, 
                                       task_type: str) -> List[Dict[str, Any]]:
        """Get recommended collaboration patterns for a specific task."""
        recommendations = []
        
        # Check if we have a predefined pattern
        if task_type in self.collaboration_patterns:
            pattern = self.collaboration_patterns[task_type]
            
            # Find agents matching the pattern requirements
            suitable_agents = []
            for candidate_id, capabilities in self.capabilities_map.items():
                if candidate_id == agent_id:
                    continue
                
                trust = self.trust_scores.get((agent_id, candidate_id), 0.5)
                if trust >= pattern["trust_required"].value:
                    capability_match = any(
                        req_cap in cap.name or req_cap in cap.specializations
                        for req_cap in pattern["required_capabilities"]
                        for cap in capabilities
                    )
                    if capability_match:
                        suitable_agents.append({
                            "agent_id": candidate_id,
                            "trust": trust,
                            "capabilities": [cap.name for cap in capabilities]
                        })
            
            recommendations.append({
                "pattern": task_type,
                "type": pattern["type"].value,
                "suitable_agents": suitable_agents[:5],  # Top 5
                "estimated_cost": self._estimate_pattern_cost(pattern, suitable_agents),
                "confidence": min(1.0, len(suitable_agents) / 3.0)  # Higher confidence with more agents
            })
        
        # Generate dynamic recommendations based on graph analysis
        if agent_id in self.collaboration_graph:
            # Find agents with successful collaboration history
            successful_collaborators = []
            for neighbor in self.collaboration_graph.neighbors(agent_id):
                edge_data = self.collaboration_graph[agent_id][neighbor]
                trust = self.trust_scores.get((agent_id, neighbor), 0.5)
                if trust > 0.6:
                    successful_collaborators.append({
                        "agent_id": neighbor,
                        "trust": trust,
                        "last_collaboration": edge_data.get("timestamp"),
                        "collaboration_type": edge_data.get("collaboration_type")
                    })
            
            if successful_collaborators:
                recommendations.append({
                    "pattern": "proven_collaborators",
                    "type": "historical_success",
                    "suitable_agents": successful_collaborators,
                    "estimated_cost": None,
                    "confidence": 0.8
                })
        
        return recommendations
    
    def _estimate_pattern_cost(self, pattern: Dict, agents: List[Dict]) -> Optional[float]:
        """Estimate cost for a collaboration pattern."""
        if not agents:
            return None
        
        # Simple cost estimation based on pattern complexity
        base_cost = 0.1  # Base collaboration overhead
        agent_costs = []
        
        for agent_data in agents[:3]:  # Consider top 3 agents
            agent_id = agent_data["agent_id"]
            if agent_id in self.capabilities_map:
                avg_cost = sum(cap.cost_per_token for cap in self.capabilities_map[agent_id]) / len(self.capabilities_map[agent_id])
                agent_costs.append(avg_cost * 1000)  # Estimated 1000 tokens
        
        if agent_costs:
            return base_cost + sum(agent_costs) / len(agent_costs)
        
        return base_cost
    
    def get_mesh_analytics(self) -> Dict[str, Any]:
        """Get analytics and insights about the mesh network."""
        total_agents = len(self.agent_registry)
        active_agents = sum(1 for agent in self.agent_registry.values() 
                           if agent["status"] == "active")
        
        # Calculate network density
        possible_edges = total_agents * (total_agents - 1)
        actual_edges = self.collaboration_graph.number_of_edges()
        network_density = actual_edges / possible_edges if possible_edges > 0 else 0
        
        # Top performers
        top_performers = sorted(
            self.agent_registry.items(),
            key=lambda x: x[1]["success_rate"] * x[1]["collaboration_count"],
            reverse=True
        )[:5]
        
        # Trust network statistics
        trust_values = list(self.trust_scores.values())
        avg_trust = sum(trust_values) / len(trust_values) if trust_values else 0
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "network_density": network_density,
            "active_collaborations": len(self.active_collaborations),
            "average_trust": avg_trust,
            "top_performers": [{"agent_id": agent_id, **stats} 
                             for agent_id, stats in top_performers],
            "collaboration_patterns": len(self.collaboration_patterns),
            "total_collaborations": sum(agent["collaboration_count"] 
                                      for agent in self.agent_registry.values())
        }
    
    async def optimize_mesh(self):
        """Optimize mesh configuration based on performance data."""
        # Identify underperforming agents
        underperformers = [
            agent_id for agent_id, stats in self.agent_registry.items()
            if stats["success_rate"] < 0.3 and stats["collaboration_count"] > 5
        ]
        
        # Identify highly trusted agent pairs for priority routing
        high_trust_pairs = [
            (agent_a, agent_b) for (agent_a, agent_b), trust in self.trust_scores.items()
            if trust > 0.8
        ]
        
        # Update collaboration patterns based on successful patterns
        successful_patterns = self._analyze_successful_collaboration_patterns()
        
        optimization_report = {
            "underperformers": underperformers,
            "high_trust_pairs": len(high_trust_pairs),
            "successful_patterns": successful_patterns,
            "optimization_timestamp": datetime.utcnow()
        }
        
        logger.info(f"Mesh optimization completed: {len(underperformers)} underperformers identified, "
                   f"{len(high_trust_pairs)} high-trust pairs found")
        
        return optimization_report
    
    def _analyze_successful_collaboration_patterns(self) -> Dict[str, Any]:
        """Analyze successful collaboration patterns to improve future matching."""
        patterns = {}
        
        for agent_id, history in self.performance_history.items():
            successful_collabs = [h for h in history if h["outcome"] == "success"]
            
            if len(successful_collabs) < 3:
                continue
            
            # Group by collaboration type or capabilities
            for collab in successful_collabs:
                collaborator = collab["collaborator"]
                if collaborator in self.capabilities_map:
                    caps = [cap.name for cap in self.capabilities_map[collaborator]]
                    pattern_key = f"{agent_id}_with_{'+'.join(sorted(caps[:2]))}"
                    
                    if pattern_key not in patterns:
                        patterns[pattern_key] = {
                            "success_count": 0,
                            "avg_performance": 0.0,
                            "participants": set()
                        }
                    
                    patterns[pattern_key]["success_count"] += 1
                    patterns[pattern_key]["avg_performance"] = (
                        (patterns[pattern_key]["avg_performance"] * (patterns[pattern_key]["success_count"] - 1) +
                         collab["performance_score"]) / patterns[pattern_key]["success_count"]
                    )
                    patterns[pattern_key]["participants"].add(agent_id)
                    patterns[pattern_key]["participants"].add(collaborator)
        
        # Convert sets to lists for JSON serialization
        for pattern in patterns.values():
            pattern["participants"] = list(pattern["participants"])
        
        return patterns