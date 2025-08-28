"""Agent discovery announcer and registry implementation (AD2).

Enhanced implementation following the Agent Discovery Protocol specification.
Provides mDNS/registry publishing with full protocol compliance.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import struct
import threading
import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple

from ..config import Config
from .registry import Entry, write_entry, list_entries
from agentsmcp.security import SecurityManager
from .resources import EnhancedAgent, Resource, ResourceType, HealthInfo, HealthState, AgentState

logger = logging.getLogger(__name__)

# mDNS/Multicast constants
MDNS_MULTICAST_GROUP = "224.0.0.251"
MDNS_PORT = 5353
AGENTSMCP_MULTICAST_GROUP = "224.1.1.1"  # Custom group for AgentsMCP
AGENTSMCP_MULTICAST_PORT = 5353


class MDNSSocketManager:
    """Singleton, thread-safe mDNS UDP socket manager."""
    _instance: Optional["MDNSSocketManager"] = None
    _lock_cls = threading.Lock()
    
    def __init__(self):
        if MDNSSocketManager._instance is not None:
            raise RuntimeError("Use MDNSSocketManager.get_instance() instead")
        self._sock_lock = threading.Lock()
        self._socket: Optional[socket.socket] = None
    
    @classmethod
    def get_instance(cls) -> "MDNSSocketManager":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock_cls:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _ensure_socket(self) -> socket.socket:
        """Ensure socket exists and is configured for multicast."""
        if self._socket is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Set TTL for multicast
            ttl = struct.pack('b', 2)  # TTL=2 for local network
            self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        
        return self._socket
    
    def sendto(self, data: bytes, addr: Tuple[str, int]) -> int:
        """Thread-safe sendto operation."""
        with self._sock_lock:
            sock = self._ensure_socket()
            try:
                return sock.sendto(data, addr)
            except Exception as e:
                logger.error(f"Socket sendto failed: {e}")
                # Close and reset socket on error
                if self._socket:
                    self._socket.close()
                    self._socket = None
                raise
    
    async def sendto_async(self, data: bytes, addr: Tuple[str, int]) -> int:
        """Async sendto operation for compatibility with asyncio."""
        with self._sock_lock:
            # Create a temporary socket for async operation to avoid blocking
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Set TTL for multicast
            ttl = struct.pack('b', 2)  # TTL=2 for local network
            temp_sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
            
            temp_sock.setblocking(False)
            
            try:
                # Use asyncio event loop for async socket operation
                loop = asyncio.get_event_loop()
                result = await loop.sock_sendto(temp_sock, data, addr)
                return result
            except Exception as e:
                logger.error(f"Async socket sendto failed: {e}")
                raise
            finally:
                temp_sock.close()
    
    def close(self) -> None:
        """Close the socket safely."""
        with self._sock_lock:
            if self._socket:
                self._socket.close()
                self._socket = None
    
    def __del__(self):
        """Cleanup on destruction."""
        self.close()


class Agent:
    """Agent identity and metadata for discovery protocol (AD1 compliant)."""
    
    def __init__(
        self,
        agent_id: str,
        agent_name: Optional[str] = None,
        capabilities: Optional[Dict[str, str]] = None,
        public_key: Optional[str] = None,
        transport: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.capabilities = capabilities or {}
        self.public_key = public_key
        self.transport = transport or {"type": "tcp", "endpoint": "localhost:9000"}
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "capabilities": self.capabilities,
            "public_key": self.public_key,
            "transport": self.transport,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """Create Agent from dictionary."""
        return cls(
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name"),
            capabilities=data.get("capabilities", {}),
            public_key=data.get("public_key"),
            transport=data.get("transport", {}),
            metadata=data.get("metadata", {}),
        )


class Announcer:
    """Enhanced announcer implementing the full discovery protocol (AD2)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Generate deterministic but unique agent_id based on hostname and config
        hostname = socket.gethostname()
        config_hash = str(hash(str(cfg.__dict__)))[:8]
        self.agent_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{hostname}:{config_hash}"))
        
        # AD5: Initialize security manager
        self.security = SecurityManager(cfg)
        
        # Create full Agent representation (legacy)
        self.agent = self._create_agent()
        
        # Create enhanced agent representation with rich resources
        self.enhanced_agent = self._create_enhanced_agent()
        
        # Announcement state
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self.heartbeat_interval = getattr(cfg, "discovery_heartbeat", 30)
        self.ttl = getattr(cfg, "discovery_ttl", 120)

    def _create_agent(self) -> Agent:
        """Create Agent representation from config."""
        capabilities = {}
        
        # Add agent types as capabilities
        for agent_name in self.cfg.agents.keys():
            capabilities[agent_name] = "1.0"
        
        # Add server capabilities if enabled
        if hasattr(self.cfg, "server") and self.cfg.server:
            capabilities["http-server"] = "1.0"
        
        # Add MCP capabilities if present
        if hasattr(self.cfg, "mcp") and self.cfg.mcp:
            capabilities["mcp"] = "1.0"
        
        # Determine transport
        transport = {"type": "tcp", "endpoint": "localhost:9000"}
        if hasattr(self.cfg, "server") and self.cfg.server:
            transport = {
                "type": "http",
                "endpoint": f"http://{self.cfg.server.host}:{self.cfg.server.port}"
            }
        
        # Gather metadata
        metadata = {
            "__os": os.name,
            "__arch": os.uname().machine if hasattr(os, "uname") else "unknown",
            "__hostname": socket.gethostname(),
        }
        
        return Agent(
            agent_id=self.agent_id,
            agent_name="agentsmcp",
            capabilities=capabilities,
            public_key=self.security.get_public_key_pem(),  # AD5: Include public key
            transport=transport,
            metadata=metadata,
        )

    def _create_enhanced_agent(self) -> EnhancedAgent:
        """Create EnhancedAgent with rich resource model."""
        from .resources import Capability, create_compute_resource, create_service_resource
        import psutil
        
        # Create capabilities list
        capabilities = []
        for agent_name in self.cfg.agents.keys():
            capabilities.append(Capability(name=agent_name, version="1.0"))
        
        if hasattr(self.cfg, "server") and self.cfg.server:
            capabilities.append(Capability(name="http-server", version="1.0"))
        
        if hasattr(self.cfg, "mcp") and self.cfg.mcp:
            capabilities.append(Capability(name="mcp", version="1.0"))
        
        # Create resources based on system capabilities
        resources = []
        
        # Add compute resource if available
        try:
            cpu_count = psutil.cpu_count()
            memory_info = psutil.virtual_memory()
            memory_gb = memory_info.total / (1024**3)
            
            endpoint = "localhost:8000"
            if hasattr(self.cfg, "server") and self.cfg.server:
                endpoint = f"http://{self.cfg.server.host}:{self.cfg.server.port}"
            
            compute_labels = {
                "os": os.name,
                "arch": os.uname().machine if hasattr(os, "uname") else "unknown",
                "hostname": socket.gethostname(),
            }
            
            compute_resource = create_compute_resource(
                resource_id="system-compute",
                cores=cpu_count or 1,
                memory_gb=memory_gb,
                endpoint=endpoint,
                labels=compute_labels
            )
            resources.append(compute_resource)
            
        except (ImportError, AttributeError):
            logger.debug("psutil not available, creating basic compute resource")
            # Fallback without psutil
            endpoint = "localhost:8000"
            if hasattr(self.cfg, "server") and self.cfg.server:
                endpoint = f"http://{self.cfg.server.host}:{self.cfg.server.port}"
            
            basic_resource = create_compute_resource(
                resource_id="system-compute",
                cores=1,
                memory_gb=1.0,
                endpoint=endpoint,
                labels={"hostname": socket.gethostname()}
            )
            resources.append(basic_resource)
        
        # Add service resources for each configured agent
        for agent_name, agent_config in self.cfg.agents.items():
            service_endpoint = "localhost:8000"
            if hasattr(self.cfg, "server") and self.cfg.server:
                service_endpoint = f"http://{self.cfg.server.host}:{self.cfg.server.port}"
            
            service_resource = create_service_resource(
                resource_id=f"agent-{agent_name}",
                service_name=agent_name,
                endpoint=service_endpoint,
                version="1.0",
                labels={
                    "agent_type": agent_config.type,
                    "model": agent_config.model or "unknown"
                }
            )
            resources.append(service_resource)
        
        # Determine transport
        transport = {"type": "tcp", "endpoint": "localhost:9000"}
        if hasattr(self.cfg, "server") and self.cfg.server:
            transport = {
                "type": "http",
                "endpoint": f"http://{self.cfg.server.host}:{self.cfg.server.port}"
            }
        
        # Gather metadata
        metadata = {
            "__os": os.name,
            "__arch": os.uname().machine if hasattr(os, "uname") else "unknown",
            "__hostname": socket.gethostname(),
            "__version": "1.0.0",  # AgentsMCP version
        }
        
        return EnhancedAgent(
            agent_id=self.agent_id,
            agent_name="agentsmcp",
            resources=resources,
            capabilities=capabilities,
            public_key=self.security.get_public_key_pem(),
            transport=transport,
            state=AgentState.ACTIVE,
            health=HealthInfo(state=HealthState.HEALTHY),
            metadata=metadata,
            schema_version=2
        )

    def capabilities(self) -> List[str]:
        """Legacy method for backward compatibility."""
        return list(self.agent.capabilities.keys())

    def announce(self) -> None:
        """Synchronous announce method for backward compatibility."""
        if not getattr(self.cfg, "discovery_enabled", False):
            return
        
        # Convert Agent to legacy Entry format
        token = getattr(self.cfg, "discovery_token", None)
        e = Entry(
            agent_id=self.agent.agent_id,
            name=self.agent.agent_name or "agentsmcp",
            capabilities=list(self.agent.capabilities.keys()),
            transport=self.agent.transport.get("type", "http"),
            endpoint=self.agent.transport.get("endpoint", ""),
            token=token,
        )
        write_entry(e)
        logger.info(f"Announced agent {self.agent_id}")

    async def start_async(self) -> None:
        """Start the async announcer service with heartbeat."""
        if self._running:
            return
            
        self._running = True
        logger.info(f"Starting async announcer for agent {self.agent_id}")
        
        # Initial announcement
        await self._announce_async()
        
        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
    async def stop_async(self) -> None:
        """Stop the async announcer service."""
        if not self._running:
            return
            
        self._running = False
        logger.info(f"Stopping async announcer for agent {self.agent_id}")
        
        # Cancel heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Send cleanup announcement to inform peers of graceful shutdown
        await self._send_cleanup_announcement()

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop for periodic announcements."""
        try:
            while self._running:
                await asyncio.sleep(self.heartbeat_interval)
                if self._running:
                    # AD5: Check for key rotation
                    if self.security.check_key_rotation():
                        # Regenerate agent with new public key
                        self.agent = self._create_agent()
                        logger.info("Rotated keys and updated agent representation")
                    
                    await self._announce_async()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat loop error: {e}")
            
    async def _announce_async(self) -> None:
        """Async announcement with full protocol support."""
        try:
            # Always register in local registry
            self.announce()  # Use sync method for backward compatibility
            
            # Send mDNS announcement
            await self._announce_mdns()
            
            # Send remote registry announcement
            await self._announce_registry()
            
        except Exception as e:
            logger.error(f"Async announcement error: {e}")
            
    async def _announce_mdns(self) -> None:
        """Announce via mDNS/UDP multicast."""
        try:
            # Create announcement message
            announcement = self.create_announce_message()
            message_data = json.dumps(announcement).encode('utf-8')
            
            # Check message size (UDP limit is ~1500 bytes, be conservative)
            if len(message_data) > 1400:
                logger.warning(f"mDNS message too large ({len(message_data)} bytes), truncating")
                # Could implement fragmentation here if needed
                message_data = message_data[:1400]
            
            # Use singleton socket manager for thread-safe operation
            socket_mgr = MDNSSocketManager.get_instance()
            await socket_mgr.sendto_async(message_data, (AGENTSMCP_MULTICAST_GROUP, AGENTSMCP_MULTICAST_PORT))
            
            logger.debug(f"Sent mDNS announcement for agent {self.agent_id} ({len(message_data)} bytes)")
            
        except Exception as e:
            logger.error(f"mDNS announcement failed for agent {self.agent_id}: {e}")
        
    async def _announce_registry(self) -> None:
        """Announce to remote registry via HTTP API."""
        # Check if registry endpoint is configured
        registry_endpoint = getattr(self.cfg, 'discovery_registry_endpoint', None)
        if not registry_endpoint:
            logger.debug("No registry endpoint configured, skipping registry announcement")
            return
            
        try:
            import aiohttp
            
            # Create announcement message
            announcement = self.create_announce_message()
            
            # Add TTL and timestamp
            announcement.update({
                "ttl": self.ttl,
                "timestamp": time.time()
            })
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": f"AgentsMCP/{self.agent_id}"
                }
                
                # Add authentication if configured
                registry_token = getattr(self.cfg, 'discovery_registry_token', None)
                if registry_token:
                    headers["Authorization"] = f"Bearer {registry_token}"
                
                # POST to registry endpoint
                async with session.post(
                    f"{registry_endpoint}/agents",
                    json=announcement,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Successfully announced to registry for agent {self.agent_id}")
                    elif response.status == 409:
                        logger.debug(f"Agent {self.agent_id} already registered in registry")
                    else:
                        logger.warning(
                            f"Registry announcement failed for agent {self.agent_id}: "
                            f"HTTP {response.status}"
                        )
                        
        except ImportError:
            logger.warning("aiohttp not available, skipping registry announcement")
        except Exception as e:
            logger.error(f"Registry announcement failed for agent {self.agent_id}: {e}")
    
    async def _send_cleanup_announcement(self) -> None:
        """Send cleanup announcement on shutdown."""
        try:
            # Create cleanup message
            cleanup_message = {
                "protocol_version": "v1.0",
                "agent": self.get_agent_dict(),
                "action": "shutdown",
                "timestamp": time.time()
            }
            
            # Add signature if security is enabled
            signature = self.security.sign_announcement(cleanup_message["agent"])
            if signature:
                cleanup_message["signature"] = signature
            
            # Send via mDNS
            await self._send_cleanup_mdns(cleanup_message)
            
            # Send to registry
            await self._send_cleanup_registry(cleanup_message)
            
            logger.info(f"Sent cleanup announcement for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Cleanup announcement failed for agent {self.agent_id}: {e}")
    
    async def _send_cleanup_mdns(self, cleanup_message: Dict[str, Any]) -> None:
        """Send cleanup via mDNS multicast."""
        try:
            message_data = json.dumps(cleanup_message).encode('utf-8')
            
            # Use singleton socket manager for thread-safe operation
            socket_mgr = MDNSSocketManager.get_instance()
            await socket_mgr.sendto_async(message_data, (AGENTSMCP_MULTICAST_GROUP, AGENTSMCP_MULTICAST_PORT))
            
            logger.debug(f"Sent cleanup mDNS message for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Cleanup mDNS failed for agent {self.agent_id}: {e}")
    
    async def _send_cleanup_registry(self, cleanup_message: Dict[str, Any]) -> None:
        """Send cleanup to remote registry."""
        registry_endpoint = getattr(self.cfg, 'discovery_registry_endpoint', None)
        if not registry_endpoint:
            return
            
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": f"AgentsMCP/{self.agent_id}"
                }
                
                registry_token = getattr(self.cfg, 'discovery_registry_token', None)
                if registry_token:
                    headers["Authorization"] = f"Bearer {registry_token}"
                
                # DELETE from registry
                async with session.delete(
                    f"{registry_endpoint}/agents/{self.agent_id}",
                    json=cleanup_message,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if 200 <= response.status < 300:
                        logger.debug(f"Successfully removed from registry: agent {self.agent_id}")
                    else:
                        logger.warning(f"Registry cleanup failed: HTTP {response.status}")
                        
        except ImportError:
            logger.debug("aiohttp not available, skipping registry cleanup")
        except Exception as e:
            logger.error(f"Registry cleanup failed for agent {self.agent_id}: {e}")

    def get_agent_dict(self) -> Dict[str, Any]:
        """Get the full Agent dictionary for protocol messages."""
        return self.agent.to_dict()
        
    def create_announce_message(self, protocol_version: str = "v1.0") -> Dict[str, Any]:
        """Create enhanced announcement message with schema versioning.
        
        Supports both legacy (v1) and enhanced (v2) agent formats with
        rich resource metadata, health info, and capabilities.
        """
        # Determine schema version based on enhanced_agent availability
        if self.enhanced_agent is not None:
            schema_version = 2
            agent_dict = self._enhanced_agent_to_dict()
        else:
            schema_version = 1
            agent_dict = self.get_agent_dict()
        
        message = {
            "protocol_version": protocol_version,
            "schema_version": schema_version,
            "timestamp": int(time.time()),
            "message_type": "announce",
            "agent": agent_dict,
        }
        
        # AD5: Add signature if security is enabled
        signature = self.security.sign_announcement(agent_dict)
        if signature:
            message["signature"] = signature
        
        return message
    
    def _resource_to_dict(self, resource) -> Dict[str, Any]:
        """Convert a Resource to dictionary for serialization."""
        data = {
            "resource_id": resource.id,
            "resource_type": resource.type,
        }
        
        # Add optional fields if present
        if hasattr(resource, 'name') and resource.name:
            data["name"] = resource.name
        if hasattr(resource, 'version'):
            data["version"] = resource.version
        if hasattr(resource, 'labels') and resource.labels:
            data["labels"] = resource.labels
        if hasattr(resource, 'endpoint') and resource.endpoint:
            data["endpoint"] = resource.endpoint
        if hasattr(resource, 'location') and resource.location:
            data["location"] = resource.location
        
        # Capacity information
        if hasattr(resource, 'capacity') and resource.capacity:
            data["capacity"] = {
                "total": resource.capacity.total,
                "used": resource.capacity.used,
                "reserved": resource.capacity.reserved
            }
        
        # Health information
        if hasattr(resource, 'health') and resource.health:
            data["health"] = {
                "state": resource.health.state,
                "score": resource.health.score,
                "last_check": resource.health.last_check,
                "check_interval": resource.health.check_interval
            }
            if resource.health.reason:
                data["health"]["reason"] = resource.health.reason
        
        return data
    
    def _enhanced_agent_to_dict(self) -> Dict[str, Any]:
        """Convert EnhancedAgent to dictionary with rich metadata."""
        agent = self.enhanced_agent
        
        # Convert capabilities to dictionary format
        capabilities_dict = {}
        for cap in agent.capabilities:
            capabilities_dict[cap.name] = {
                "version": cap.version,
                "parameters": cap.parameters
            }
            if cap.min_version:
                capabilities_dict[cap.name]["min_version"] = cap.min_version
            if cap.max_version:
                capabilities_dict[cap.name]["max_version"] = cap.max_version
        
        # Convert resources to dictionary format
        resources_list = []
        for resource in agent.resources:
            resources_list.append(self._resource_to_dict(resource))
        
        payload = {
            "agent_id": agent.agent_id,
            "agent_name": agent.agent_name,
            "schema_version": agent.schema_version,
            "capabilities": capabilities_dict,
            "resources": resources_list,
            "transport": agent.transport,
            "state": agent.state,
            "metadata": agent.metadata,
            "created_at": agent.created_at,
            "updated_at": agent.updated_at,
        }
        
        # Add optional fields
        if agent.public_key:
            payload["public_key"] = agent.public_key
        
        # Add health information
        if agent.health:
            payload["health"] = {
                "state": agent.health.state,
                "score": agent.health.score,
                "last_check": agent.health.last_check,
                "check_interval": agent.health.check_interval
            }
            if agent.health.reason:
                payload["health"]["reason"] = agent.health.reason
        
        return payload


# Utility functions
def create_agent_id() -> str:
    """Generate a new RFC 4122 UUID for agent identity."""
    return str(uuid.uuid4())


def discover_agents() -> List[Agent]:
    """Discover agents from local registry."""
    agents = []
    for entry in list_entries():
        # Convert legacy Entry to Agent format
        agent = Agent(
            agent_id=entry.agent_id,
            agent_name=entry.name,
            capabilities={cap: "1.0" for cap in entry.capabilities},
            transport={"type": entry.transport, "endpoint": entry.endpoint},
            metadata={"token": entry.token} if entry.token else {},
        )
        agents.append(agent)
    return agents
