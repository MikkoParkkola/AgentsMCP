"""Discovery client implementation for the AgentsMCP system (AD3).

Enhanced implementation following the Agent Discovery Protocol specification.
Provides comprehensive query capabilities and event-driven announcement handling.
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Set

from .registry import list_entries, Entry
from .announcer import Agent, MDNSSocketManager
from ..config import Config
from ..security import SecurityManager
from .exceptions import DiscoveryProtocolError, NetworkError, ServiceUnavailableError
from .circuit_breaker import CircuitBreaker
from .retry import exponential_backoff
from .cache import read_cache, write_cache
from .health import register_circuit_breaker

logger = logging.getLogger(__name__)

# Protocol constants
DEFAULT_MULTICAST_GROUP = "224.1.1.1"
DEFAULT_MULTICAST_PORT = 5353  # mDNS standard port
DEFAULT_PROTOCOL_VERSION = "v1.0"

# Global circuit breaker for discovery operations
DISCOVERY_BREAKER = CircuitBreaker(
    name="discovery_service",
    failure_threshold=3,
    recovery_timeout=20.0,
    expected_exceptions=(NetworkError, DiscoveryProtocolError, ServiceUnavailableError),
)


class DiscoveryClient:
    """Enhanced discovery client implementing the full discovery protocol (AD3)."""

    def __init__(
        self,
        multicast_group: str = DEFAULT_MULTICAST_GROUP,
        multicast_port: int = DEFAULT_MULTICAST_PORT,
        config: Optional[Config] = None,
    ):
        """Initialize discovery client with configuration."""
        self.multicast_group = multicast_group
        self.multicast_port = multicast_port
        self.config = config or self._load_config()
        
        # AD5: Initialize security manager
        self.security = SecurityManager(self.config)
        
        # Local registry for caching discovered agents
        self._local_registry: Dict[str, Agent] = {}
        
        # Event listeners
        self._announcement_listeners: List[Callable[[Agent], None]] = []
        
        # Background listening state
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None
        self._socket: Optional[socket.socket] = None
        
        # Register circuit breaker for health monitoring
        register_circuit_breaker("discovery_service", DISCOVERY_BREAKER)

    def _load_config(self) -> Config:
        """Load configuration with fallback to default."""
        try:
            return Config.load()
        except Exception:
            return Config()

    def discover(self) -> List[Entry]:
        """Legacy discover method for backward compatibility (AD3).
        
        Applies allowlist filtering and graceful degradation with caching.
        """
        try:
            # Try to get entries from registry
            entries = DISCOVERY_BREAKER.call(list_entries)
            
            # Cache the results
            entries_data = [entry.__dict__ for entry in entries]
            write_cache("discovery_entries", {"entries": entries_data}, ttl=60)
            
            # Apply allowlist filtering if configured
            try:
                allow = set(getattr(self.config, "discovery_allowlist", []) or [])
                if allow:
                    entries = [e for e in entries if e.agent_id in allow or e.name in allow]
            except Exception as exc:
                logger.warning("Error applying allowlist filter: %s", exc)
            
            return entries
            
        except ServiceUnavailableError:
            # Circuit breaker is open - try cache fallback
            logger.warning("Discovery service unavailable, attempting cache fallback")
            cached = read_cache("discovery_entries")
            if cached and "entries" in cached:
                try:
                    # Reconstruct Entry objects from cached data
                    entries = [Entry(**entry_data) for entry_data in cached["entries"]]
                    logger.info("Using cached discovery entries (%d entries)", len(entries))
                    return entries
                except Exception as exc:
                    logger.error("Failed to reconstruct entries from cache: %s", exc)
            
            # No cache available - raise the original error
            raise DiscoveryProtocolError(
                "Discovery service unavailable and no cached data available",
                payload={"fallback": "none", "cache_available": cached is not None}
            )
        except Exception as exc:
            logger.error("Discovery failed: %s", exc)
            # Try cache as last resort
            cached = read_cache("discovery_entries")
            if cached and "entries" in cached:
                try:
                    entries = [Entry(**entry_data) for entry_data in cached["entries"]]
                    logger.warning("Using stale cache due to error: %s", exc)
                    return entries
                except Exception:
                    pass
            
            # Convert to our domain exception
            raise DiscoveryProtocolError("Discovery operation failed", cause=exc) from exc

    def query_capabilities(self, capabilities: List[str], timeout: float = 5.0) -> List[Agent]:
        """Query for agents by capability synchronously."""
        # First check local registry
        local_matches = []
        for agent in self._local_registry.values():
            if self._capabilities_match(agent, capabilities):
                local_matches.append(agent)
        
        # Then query network
        network_matches = self._query_network_sync(capabilities, timeout)
        
        # Combine results, avoiding duplicates
        all_matches = local_matches.copy()
        for agent in network_matches:
            if agent.agent_id not in {a.agent_id for a in all_matches}:
                all_matches.append(agent)
        
        return all_matches

    async def query_capabilities_async(
        self, 
        capabilities: List[str], 
        timeout: float = 5.0
    ) -> List[Agent]:
        """Query for agents by capability asynchronously."""
        # Check local registry first
        local_matches = []
        for agent in self._local_registry.values():
            if self._capabilities_match(agent, capabilities):
                local_matches.append(agent)
        
        # Query network asynchronously
        network_matches = await self._query_network_async(capabilities, timeout)
        
        # Combine results, avoiding duplicates
        all_matches = local_matches.copy()
        for agent in network_matches:
            if agent.agent_id not in {a.agent_id for a in all_matches}:
                all_matches.append(agent)
        
        return all_matches

    @exponential_backoff(attempts=3, base_delay=0.2, retry_on=(NetworkError, OSError))
    def _query_network_sync(self, capabilities: List[str], timeout: float) -> List[Agent]:
        """Send query message and collect responses synchronously with retry logic."""
        return DISCOVERY_BREAKER.call(self._perform_network_query_sync, capabilities, timeout)
    
    def _perform_network_query_sync(self, capabilities: List[str], timeout: float) -> List[Agent]:
        """Actual network query implementation (wrapped by circuit breaker and retry)."""
        query_message = self._create_query_message(capabilities)
        data = json.dumps(query_message).encode('utf-8')
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        responses = []
        
        try:
            # Send query to multicast group using socket manager
            socket_mgr = MDNSSocketManager.get_instance()
            socket_mgr.sendto(data, (self.multicast_group, self.multicast_port))
            logger.debug(f"Sent query for capabilities: {capabilities}")
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response_data, addr = sock.recvfrom(4096)
                    response = self._parse_response(response_data)
                    if response and self._capabilities_match(response, capabilities):
                        responses.append(response)
                        # Update local registry
                        self._local_registry[response.agent_id] = response
                except socket.timeout:
                    break
                except socket.error as e:
                    # Convert socket errors to our domain exceptions
                    raise NetworkError(f"Socket error during query: {e}", cause=e) from e
                except Exception as e:
                    logger.debug(f"Error parsing response: {e}")
                    continue
                    
        except socket.error as e:
            # Convert socket errors to our domain exceptions
            raise NetworkError(f"Network error during capability query: {e}", cause=e) from e
        except Exception as e:
            logger.error(f"Network query error: {e}")
            raise DiscoveryProtocolError(f"Query operation failed: {e}", cause=e) from e
        finally:
            sock.close()
        
        return responses

    @exponential_backoff(attempts=3, base_delay=0.2, retry_on=(NetworkError, OSError))
    async def _query_network_async(self, capabilities: List[str], timeout: float) -> List[Agent]:
        """Send query message and collect responses asynchronously with retry logic."""
        return await DISCOVERY_BREAKER.acall(self._perform_network_query_async, capabilities, timeout)
    
    async def _perform_network_query_async(self, capabilities: List[str], timeout: float) -> List[Agent]:
        """Core async network query implementation with proper error handling."""
        query_message = self._create_query_message(capabilities)
        data = json.dumps(query_message).encode('utf-8')
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        responses = []
        
        try:
            # Send query to multicast group using socket manager
            socket_mgr = MDNSSocketManager.get_instance()
            await socket_mgr.sendto_async(data, (self.multicast_group, self.multicast_port))
            logger.debug(f"Sent async query for capabilities: {capabilities}")
            
            start_time = asyncio.get_event_loop().time()
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    response_data, addr = await asyncio.wait_for(
                        asyncio.get_event_loop().sock_recvfrom(sock, 4096),
                        timeout=0.5
                    )
                    response = self._parse_response(response_data)
                    if response and self._capabilities_match(response, capabilities):
                        responses.append(response)
                        # Update local registry
                        self._local_registry[response.agent_id] = response
                except asyncio.TimeoutError:
                    continue
                except OSError as e:
                    # Convert OS errors to our domain exceptions
                    raise NetworkError(f"Socket error during async query: {e}", cause=e) from e
                except Exception as e:
                    logger.debug(f"Error parsing async response: {e}")
                    continue
                    
        except OSError as e:
            # Convert socket errors to our domain exceptions
            raise NetworkError(f"Network error during async capability query: {e}", cause=e) from e
        except Exception as e:
            logger.error(f"Async network query error: {e}")
            raise DiscoveryProtocolError(f"Async query operation failed: {e}", cause=e) from e
        finally:
            sock.close()
        
        return responses

    def _create_query_message(self, capabilities: List[str]) -> Dict[str, Any]:
        """Create a query message per protocol specification."""
        return {
            "protocol_version": DEFAULT_PROTOCOL_VERSION,
            "query": {
                "capability": capabilities[0] if capabilities else "",
                # Could be enhanced to support multiple capabilities
            }
        }

    def _parse_response(self, data: bytes) -> Optional[Agent]:
        """Parse response message and extract Agent with signature validation.
        
        Supports both v1 (legacy) and v2 (enhanced) schema versions.
        """
        try:
            message = json.loads(data.decode('utf-8'))
            
            # Check if it's a valid response message
            if (message.get("protocol_version") == DEFAULT_PROTOCOL_VERSION and
                "agent" in message and message.get("matched", False)):
                
                agent_data = message["agent"]
                agent_id = agent_data.get("agent_id")
                signature = message.get("signature")
                schema_version = message.get("schema_version", 1)
                
                # AD5: Validate signature if present and security enabled
                if self.security.is_enabled():
                    if not signature:
                        logger.warning(f"Missing signature in response from agent {agent_id}")
                        return None
                    
                    if not self.security.validate_announcement(agent_id, agent_data, signature):
                        logger.warning(f"Invalid signature in response from agent {agent_id}")
                        return None
                    
                    # Check if agent is allowed
                    if not self.security.is_agent_allowed(agent_id):
                        logger.warning(f"Agent {agent_id} not in allowlist")
                        return None
                
                # Handle different schema versions
                if schema_version == 2:
                    # Enhanced agent with rich resource model
                    from .resources import EnhancedAgent
                    enhanced_agent = EnhancedAgent.from_dict(agent_data)
                    # Convert to legacy Agent format for compatibility
                    return self._convert_enhanced_to_legacy_agent(enhanced_agent)
                else:
                    # Legacy v1 agent format
                    return Agent.from_dict(agent_data)
                
        except Exception as e:
            logger.debug(f"Failed to parse response: {e}")
        
        return None
    
    def _convert_enhanced_to_legacy_agent(self, enhanced_agent) -> Agent:
        """Convert EnhancedAgent to legacy Agent format for compatibility."""
        # Extract capabilities as simple string keys
        capabilities = {}
        for cap in enhanced_agent.capabilities:
            capabilities[cap.name] = cap.version
        
        return Agent(
            agent_id=enhanced_agent.agent_id,
            agent_name=enhanced_agent.agent_name,
            capabilities=capabilities,
            public_key=enhanced_agent.public_key,
            transport=enhanced_agent.transport,
            metadata=enhanced_agent.metadata
        )

    def _capabilities_match(self, agent: Agent, required_capabilities: List[str]) -> bool:
        """Check if agent has all required capabilities."""
        if not required_capabilities:
            return True
        
        agent_caps = set(agent.capabilities.keys())
        return all(cap in agent_caps for cap in required_capabilities)

    # Event-driven announcement listening
    async def start_listening(self) -> None:
        """Start listening for announcements."""
        if self._running:
            return
        
        self._running = True
        logger.info("Starting discovery client listener")
        
        self._listen_task = asyncio.create_task(self._listen_loop())

    async def stop_listening(self) -> None:
        """Stop listening for announcements."""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping discovery client listener")
        
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        if self._socket:
            self._socket.close()
            self._socket = None

    async def _listen_loop(self) -> None:
        """Background loop for listening to announcements."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            # Bind to multicast port
            self._socket.bind(('', self.multicast_port))
            
            # Join multicast group
            mreq = struct.pack("4sl", socket.inet_aton(self.multicast_group), socket.INADDR_ANY)
            self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            
            self._socket.setblocking(False)
            
            while self._running:
                try:
                    data, addr = await asyncio.get_event_loop().sock_recvfrom(self._socket, 4096)
                    await self._handle_announcement(data, addr)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug(f"Error in listen loop: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to start listening: {e}")
        finally:
            if self._socket:
                self._socket.close()
                self._socket = None

    async def _handle_announcement(self, data: bytes, addr: tuple) -> None:
        """Handle incoming announcement message with signature validation.
        
        Supports both v1 (legacy) and v2 (enhanced) schema versions.
        """
        try:
            message = json.loads(data.decode('utf-8'))
            
            # Check if it's an announcement (has agent but no matched field)
            if ("agent" in message and 
                "matched" not in message and 
                message.get("protocol_version") == DEFAULT_PROTOCOL_VERSION):
                
                agent_data = message["agent"]
                agent_id = agent_data.get("agent_id")
                signature = message.get("signature")
                schema_version = message.get("schema_version", 1)
                
                # AD5: Validate signature if present and security enabled
                if self.security.is_enabled():
                    if not signature:
                        logger.warning(f"Missing signature in announcement from agent {agent_id}")
                        return
                    
                    if not self.security.validate_announcement(agent_id, agent_data, signature):
                        logger.warning(f"Invalid signature in announcement from agent {agent_id}")
                        return
                    
                    # Check if agent is allowed
                    if not self.security.is_agent_allowed(agent_id):
                        logger.warning(f"Agent {agent_id} not in allowlist")
                        return
                
                # Handle different schema versions
                if schema_version == 2:
                    # Enhanced agent with rich resource model
                    from .resources import EnhancedAgent
                    enhanced_agent = EnhancedAgent.from_dict(agent_data)
                    agent = self._convert_enhanced_to_legacy_agent(enhanced_agent)
                    logger.debug(f"Processed enhanced announcement (v2) from {agent.agent_name} with {len(enhanced_agent.resources)} resources")
                else:
                    # Legacy v1 agent format
                    agent = Agent.from_dict(agent_data)
                    logger.debug(f"Processed legacy announcement (v1) from {agent.agent_name}")
                
                # Update local registry
                self._local_registry[agent.agent_id] = agent
                
                # Notify listeners
                for listener in self._announcement_listeners:
                    try:
                        listener(agent)
                    except Exception as e:
                        logger.error(f"Error in announcement listener: {e}")
                
                logger.debug(f"Updated registry with agent {agent.agent_name} ({agent.agent_id})")
                
        except Exception as e:
            logger.debug(f"Failed to handle announcement: {e}")

    def add_announcement_listener(self, listener: Callable[[Agent], None]) -> None:
        """Add a listener for announcement events."""
        if listener not in self._announcement_listeners:
            self._announcement_listeners.append(listener)

    def remove_announcement_listener(self, listener: Callable[[Agent], None]) -> None:
        """Remove an announcement listener."""
        if listener in self._announcement_listeners:
            self._announcement_listeners.remove(listener)

    def get_local_registry(self) -> Dict[str, Agent]:
        """Get the current local registry of discovered agents."""
        return self._local_registry.copy()

    def clear_local_registry(self) -> None:
        """Clear the local registry."""
        self._local_registry.clear()

    # Context manager support
    async def __aenter__(self):
        await self.start_listening()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_listening()


# Legacy compatibility function
def discover() -> List[Entry]:
    """Legacy discover function for backward compatibility (AD3).

    Applies allowlist filtering if configured (AD5).
    """
    client = DiscoveryClient()
    return client.discover()


# Utility functions
def query_by_capability(capabilities: List[str], timeout: float = 5.0) -> List[Agent]:
    """Query for agents by capability - convenience function."""
    client = DiscoveryClient()
    return client.query_capabilities(capabilities, timeout)


async def query_by_capability_async(capabilities: List[str], timeout: float = 5.0) -> List[Agent]:
    """Query for agents by capability asynchronously - convenience function."""
    client = DiscoveryClient()
    return await client.query_capabilities_async(capabilities, timeout)
