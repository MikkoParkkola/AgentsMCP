"""
agentsmcp.discovery

Agent Discovery & Coordination subsystem for AgentsMCP.

This package implements the distributed agent discovery protocol that enables
agents to register themselves, advertise capabilities, monitor health, and
coordinate with each other in a production-ready distributed environment.

Modules:
    agent_service: AD2 - Agent registration & health monitoring service
    matching_engine: AD3 - Service discovery with capability matching
    load_balancer: AD4 - Load balancing & coordination protocols  
    raft_cluster: AD5 - Discovery service clustering with Raft consensus

The discovery subsystem integrates with the delegation system to provide
secure, scalable agent orchestration with network partition tolerance.
"""

from .agent_service import create_app as create_agent_service_app
from .agent_service import AgentRegistry, AgentInfo

__all__ = [
    "create_agent_service_app",
    "AgentRegistry", 
    "AgentInfo",
]