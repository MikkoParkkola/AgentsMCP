"""
agentsmcp.discovery.coordination

Production‑ready implementation of load‑balancing, health‑monitoring,
circuit‑breaker and peer‑to‑peer coordination for the **AgentsMCP** platform.

The module is deliberately self‑contained but hooks into the discovery engine
(from matching_engine) and the agent service module from AD‑2.

Typical usage
-------------

>>> from agentsmcp.discovery.coordination import CoordinationNode
>>> node = CoordinationNode(
...     service_name="agents-mcp",
...     load_balancing="weighted",
...     jwt_secret="super‑secret",
... )
>>> await node.start()
>>> # later …
>>> await node.shutdown()
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

# External dependencies (add to requirements.txt)
try:
    import aiohttp
    import jwt
    import websockets
    from websockets import WebSocketClientProtocol
    HAS_WEBSOCKET_DEPS = True
except ImportError:
    HAS_WEBSOCKET_DEPS = False

# --------------------------------------------------------------------------- #
# Integration with other AgentsMCP modules
# --------------------------------------------------------------------------- #
try:
    from .matching_engine import MatchingAgentInfo
    from .agent_service import AgentRegistry
except ImportError:
    # Fallback stubs for when modules aren't available
    @dataclass
    class MatchingAgentInfo:
        agent_id: str
        address: str = ""
        weight: int = 1
        
    class AgentRegistry:
        def get_all(self) -> List[MatchingAgentInfo]:
            return []

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logger = logging.getLogger("agentsmcp.discovery.coordination")

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class CoordinationError(RuntimeError):
    """Base class for all coordination‑related errors."""


class LoadBalancingError(CoordinationError):
    """Raised when a load‑balancing operation cannot be satisfied."""


class CircuitBreakerOpen(CoordinationError):
    """Raised when a request is blocked because the circuit breaker is open."""


class TokenValidationError(CoordinationError):
    """Raised when a JWT load‑token cannot be verified."""


# --------------------------------------------------------------------------- #
# Helper data structures
# --------------------------------------------------------------------------- #
class LBStrategy(str, Enum):
    """Supported load‑balancing strategies."""

    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_CONNECTIONS = "least_connections"


class MessageType(str, Enum):
    """Logical message envelope types."""

    # Agent Discovery & Coordination Protocol messages (AD4)
    HELLO = "hello"
    CAPABILITY_EXCHANGE = "capability_exchange"
    ACK = "ack"
    
    # Existing load balancing messages
    REQUEST = "request"
    RESPONSE = "response"
    HEARTBEAT = "heartbeat"
    CONTROL = "control"  # e.g., for scaling, re‑balancing, etc.


class Envelope(NamedTuple):
    """The envelope that travels across the WebSocket channel."""

    msg_id: str
    msg_type: MessageType
    payload: dict
    token: str  # JWT load‑token


# --------------------------------------------------------------------------- #
# Load‑token management (JWT)
# --------------------------------------------------------------------------- #
@dataclass
class LoadTokenManager:
    """
    Generates and validates JWT load tokens used for every message envelope.

    The token contains:
    * ``iss`` – the issuing service name
    * ``sub`` – the target agent id (optional)
    * ``exp`` – expiry (defaults to 30 seconds)
    * ``jti`` – a unique identifier
    * ``weight`` – a numeric weight used by the weighted load‑balancer
    """

    secret: str
    issuer: str
    algorithm: str = "HS256"
    ttl_seconds: int = 30

    def generate(
        self,
        *,
        target: Optional[str] = None,
        weight: Optional[int] = None,
    ) -> str:
        """Create a signed token."""
        if not HAS_WEBSOCKET_DEPS:
            logger.warning("JWT dependencies not available - using mock token")
            return f"mock-token-{uuid.uuid4()}"
            
        now = int(time.time())
        payload: Dict[str, Any] = {
            "iss": self.issuer,
            "iat": now,
            "exp": now + self.ttl_seconds,
            "jti": str(uuid.uuid4()),
        }
        if target:
            payload["sub"] = target
        if weight is not None:
            payload["weight"] = weight
        token = jwt.encode(payload, self.secret, algorithm=self.algorithm)
        logger.debug("Generated load token: %s", payload)
        return token

    def validate(self, token: str) -> dict:
        """
        Validate a JWT token and return its payload.

        :raises TokenValidationError: if verification fails.
        """
        if not HAS_WEBSOCKET_DEPS:
            # Mock validation for when dependencies aren't available
            if token.startswith("mock-token-"):
                return {"iss": self.issuer, "jti": token}
            raise TokenValidationError("Invalid mock token format")
            
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                options={"require": ["exp", "iat", "jti"]},
            )
            logger.debug("Validated token payload: %s", payload)
            return payload
        except jwt.PyJWTError as exc:
            logger.warning("Invalid load token: %s", exc)
            raise TokenValidationError(str(exc)) from exc


# --------------------------------------------------------------------------- #
# Circuit Breaker
# --------------------------------------------------------------------------- #
@dataclass
class CircuitBreaker:
    """
    A classic "closed → open → half‑open" circuit breaker.

    - ``failure_threshold`` – number of consecutive failures before opening.
    - ``recovery_timeout`` – seconds to stay open before moving to half‑open.
    - ``expected_successes`` – consecutive successful calls required to close.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    expected_successes: int = 3

    _failure_count: int = field(init=False, default=0)
    _state: str = field(init=False, default="closed")  # closed|open|half-open
    _opened_at: float = field(init=False, default=0.0)
    _half_open_successes: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._failure_count = 0
        self._state = "closed"
        self._opened_at = 0.0
        self._half_open_successes = 0

    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """
        Wrap an awaitable ``func`` with circuit‑breaker logic.

        :raises CircuitBreakerOpen: if the circuit is open.
        """
        if self._state == "open":
            elapsed = time.time() - self._opened_at
            if elapsed >= self.recovery_timeout:
                self._state = "half-open"
                logger.info("Circuit breaker transitioning to half‑open")
            else:
                logger.debug(
                    "Circuit breaker OPEN – %s seconds remaining", self.recovery_timeout - elapsed
                )
                raise CircuitBreakerOpen("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
        except Exception as exc:
            await self._record_failure()
            raise exc
        else:
            await self._record_success()
            return result

    async def _record_failure(self) -> None:
        self._failure_count += 1
        logger.warning(
            "Circuit breaker failure %d/%d", self._failure_count, self.failure_threshold
        )
        if self._failure_count >= self.failure_threshold:
            self._open_circuit()

    async def _record_success(self) -> None:
        if self._state == "half-open":
            self._half_open_successes += 1
            if self._half_open_successes >= self.expected_successes:
                logger.info("Circuit breaker CLOSED after successful half‑open period")
                self._reset()
        else:
            # In the closed state a single success resets failures.
            self._failure_count = 0

    def _open_circuit(self) -> None:
        self._state = "open"
        self._opened_at = time.time()
        logger.error(
            "Circuit breaker OPEN – will stay open for %.1f seconds", self.recovery_timeout
        )


# --------------------------------------------------------------------------- #
# Agent Coordination Handshake Implementation (AD4)
# --------------------------------------------------------------------------- #
@dataclass
class HandshakeState:
    """State tracking for agent-to-agent coordination handshake."""
    agent_id: str
    remote_agent_id: Optional[str] = None
    state: str = "disconnected"  # disconnected|connecting|handshaking|established|failed
    capabilities: Set[str] = field(default_factory=set)
    remote_capabilities: Set[str] = field(default_factory=set)
    connection_time: Optional[float] = None
    last_heartbeat: Optional[float] = None


class CoordinationHandshake:
    """
    Implements the agent-to-agent coordination handshake protocol (AD4).
    
    Handshake sequence:
    1. HELLO: Initiator sends agent_id, capabilities, supported versions
    2. CAPABILITY_EXCHANGE: Responder replies with their capabilities
    3. ACK: Initiator acknowledges and connection is established
    
    Supports both direct peer-to-peer and registry-mediated coordination.
    """
    
    def __init__(self, local_agent_id: str, capabilities: Set[str]):
        self.local_agent_id = local_agent_id
        self.capabilities = capabilities
        self.handshake_states: Dict[str, HandshakeState] = {}
        self._handshake_timeout = 10.0
        self._heartbeat_interval = 30.0
        
    async def initiate_handshake(
        self, 
        remote_address: str, 
        pool: "WebSocketPool"
    ) -> bool:
        """
        Initiate handshake with a remote agent.
        
        Returns True if handshake succeeds, False otherwise.
        """
        logger.info(f"Initiating handshake with {remote_address}")
        
        # Create handshake state
        state = HandshakeState(
            agent_id=self.local_agent_id,
            state="connecting",
            capabilities=self.capabilities
        )
        self.handshake_states[remote_address] = state
        
        try:
            # Step 1: Send HELLO message
            hello_payload = {
                "agent_id": self.local_agent_id,
                "capabilities": list(self.capabilities),
                "supported_versions": ["v1.0"],
                "timestamp": time.time()
            }
            
            state.state = "handshaking"
            
            response = await pool.send_request(
                remote_address,
                hello_payload,
                timeout=self._handshake_timeout,
                msg_type=MessageType.HELLO
            )
            
            # Step 2: Process CAPABILITY_EXCHANGE response
            if not self._validate_capability_response(response):
                state.state = "failed"
                return False
            
            state.remote_agent_id = response.get("agent_id")
            state.remote_capabilities = set(response.get("capabilities", []))
            
            # Step 3: Send ACK to complete handshake
            ack_payload = {
                "agent_id": self.local_agent_id,
                "accepted": True,
                "negotiated_capabilities": list(
                    self.capabilities.intersection(state.remote_capabilities)
                ),
                "timestamp": time.time()
            }
            
            ack_response = await pool.send_request(
                remote_address,
                ack_payload,
                timeout=self._handshake_timeout,
                msg_type=MessageType.ACK
            )
            
            if ack_response.get("status") == "established":
                state.state = "established"
                state.connection_time = time.time()
                state.last_heartbeat = time.time()
                logger.info(f"Handshake established with {remote_address}")
                return True
            else:
                state.state = "failed"
                return False
                
        except Exception as e:
            logger.error(f"Handshake failed with {remote_address}: {e}")
            state.state = "failed"
            return False
    
    async def handle_hello(self, hello_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming HELLO message and respond with CAPABILITY_EXCHANGE."""
        remote_agent_id = hello_payload.get("agent_id")
        remote_capabilities = set(hello_payload.get("capabilities", []))
        
        logger.info(f"Received HELLO from agent {remote_agent_id}")
        
        # Create handshake state for this remote agent
        state = HandshakeState(
            agent_id=self.local_agent_id,
            remote_agent_id=remote_agent_id,
            state="handshaking",
            capabilities=self.capabilities,
            remote_capabilities=remote_capabilities
        )
        
        # Response with our capabilities
        response = {
            "agent_id": self.local_agent_id,
            "capabilities": list(self.capabilities),
            "supported_versions": ["v1.0"],
            "timestamp": time.time()
        }
        
        return response
    
    async def handle_ack(self, ack_payload: Dict[str, Any], remote_address: str) -> Dict[str, Any]:
        """Handle incoming ACK message and finalize handshake."""
        remote_agent_id = ack_payload.get("agent_id")
        accepted = ack_payload.get("accepted", False)
        
        logger.info(f"Received ACK from agent {remote_agent_id}, accepted: {accepted}")
        
        if accepted:
            # Update handshake state
            if remote_address in self.handshake_states:
                state = self.handshake_states[remote_address]
                state.state = "established"
                state.connection_time = time.time()
                state.last_heartbeat = time.time()
            
            return {"status": "established", "agent_id": self.local_agent_id}
        else:
            if remote_address in self.handshake_states:
                self.handshake_states[remote_address].state = "failed"
            return {"status": "rejected", "agent_id": self.local_agent_id}
    
    def _validate_capability_response(self, response: Dict[str, Any]) -> bool:
        """Validate the capability exchange response."""
        required_fields = ["agent_id", "capabilities", "supported_versions"]
        return all(field in response for field in required_fields)
    
    def get_established_connections(self) -> List[str]:
        """Get list of addresses with established handshakes."""
        return [
            addr for addr, state in self.handshake_states.items()
            if state.state == "established"
        ]
    
    def get_handshake_state(self, address: str) -> Optional[HandshakeState]:
        """Get handshake state for a specific address."""
        return self.handshake_states.get(address)
    
    async def maintain_heartbeat(self, pool: "WebSocketPool") -> None:
        """Maintain heartbeat with established connections."""
        current_time = time.time()
        
        for address, state in self.handshake_states.items():
            if state.state == "established":
                if (state.last_heartbeat is None or 
                    current_time - state.last_heartbeat > self._heartbeat_interval):
                    
                    try:
                        heartbeat_payload = {
                            "agent_id": self.local_agent_id,
                            "timestamp": current_time,
                            "type": "heartbeat"
                        }
                        
                        await pool.send_request(
                            address,
                            heartbeat_payload,
                            timeout=5.0,
                            msg_type=MessageType.HEARTBEAT
                        )
                        
                        state.last_heartbeat = current_time
                        
                    except Exception as e:
                        logger.warning(f"Heartbeat failed for {address}: {e}")
                        state.state = "failed"


# --------------------------------------------------------------------------- #
# Load‑balancing strategies
# --------------------------------------------------------------------------- #
@dataclass
class ServiceInstance:
    """
    Simple DTO that represents a discovered service instance.

    ``address`` – ``ws://host:port`` URL for the websocket endpoint.
    ``weight`` – integer weight used by the weighted balancer (default=1).
    ``metadata`` – optional free‑form dict (e.g., version, region).
    """

    address: str
    weight: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoadBalancer(ABC):
    """Base class for load balancing strategies.
    
    Concrete subclasses must implement :meth:`select` which returns a
    :class:`ServiceInstance` chosen according to the strategy's logic.
    """

    def __init__(self, instances: List[ServiceInstance]) -> None:
        self.instances: List[ServiceInstance] = instances

    def update_instances(self, instances: List[ServiceInstance]) -> None:
        """Replace the current instance list – called by the health monitor."""
        self.instances = instances
        logger.debug("Load balancer refreshed %d instances", len(instances))

    @abstractmethod
    def select(self) -> ServiceInstance:
        """Return a single instance chosen according to the concrete strategy.
        
        The method is abstract; concrete subclasses must provide an implementation.
        """
        ...


class RoundRobinLoadBalancer(LoadBalancer):
    """Simple round‑robin balancer."""

    def __init__(self, instances: List[ServiceInstance]) -> None:
        super().__init__(instances)
        self._idx: int = 0

    def select(self) -> ServiceInstance:
        if not self.instances:
            raise LoadBalancingError("No available instances for round‑robin")
        self._idx = (self._idx + 1) % len(self.instances)
        chosen = self.instances[self._idx]
        logger.debug("RoundRobin selected %s", chosen.address)
        return chosen


class WeightedLoadBalancer(LoadBalancer):
    """Weighted random selection – each weight = relative probability."""

    def __init__(self, instances: List[ServiceInstance]) -> None:
        super().__init__(instances)
        self._total_weight: int = 0
        self._cum_weights: List[int] = []
        self._recompute_weights()

    def update_instances(self, instances: List[ServiceInstance]) -> None:
        super().update_instances(instances)
        self._recompute_weights()

    def _recompute_weights(self) -> None:
        self._cum_weights = []
        cum = 0
        for inst in self.instances:
            cum += max(inst.weight, 1)
            self._cum_weights.append(cum)
        self._total_weight = cum
        logger.debug(
            "Weighted balancer recomputed cum_weights=%s total=%d",
            self._cum_weights,
            self._total_weight,
        )

    def select(self) -> ServiceInstance:
        if not self.instances:
            raise LoadBalancingError("No available instances for weighted balancer")
        rnd = random.randint(1, self._total_weight)
        # Linear search is fine for typical small cluster sizes.
        for idx, cum_weight in enumerate(self._cum_weights):
            if rnd <= cum_weight:
                chosen = self.instances[idx]
                logger.debug(
                    "Weighted selected %s (rnd=%d, cum=%d)", chosen.address, rnd, cum_weight
                )
                return chosen
        # Fallback – should never happen.
        chosen = self.instances[-1]
        logger.warning("Weighted selection fell back to last instance")
        return chosen


class LeastConnectionsLoadBalancer(LoadBalancer):
    """
    Keeps a live count of active connections per instance.  The instance with the
    fewest active streams is chosen.  The balancer is *stateful* – callers must
    invoke ``increment`` / ``decrement`` when they open/close a logical stream.
    """

    def __init__(self, instances: List[ServiceInstance]) -> None:
        super().__init__(instances)
        self._connection_counts: Dict[str, int] = {}
        self._reset_counts()

    def _reset_counts(self) -> None:
        self._connection_counts = {inst.address: 0 for inst in self.instances}
        logger.debug("Least‑connections counters reset")

    def update_instances(self, instances: List[ServiceInstance]) -> None:
        super().update_instances(instances)
        self._reset_counts()

    def increment(self, address: str) -> None:
        self._connection_counts[address] = self._connection_counts.get(address, 0) + 1
        logger.debug("Inc connections %s → %d", address, self._connection_counts[address])

    def decrement(self, address: str) -> None:
        cnt = self._connection_counts.get(address, 0)
        if cnt > 0:
            self._connection_counts[address] = cnt - 1
            logger.debug(
                "Dec connections %s → %d", address, self._connection_counts[address]
            )

    def select(self) -> ServiceInstance:
        if not self.instances:
            raise LoadBalancingError("No available instances for least‑connections")
        # Choose the address with the minimal counter; break ties randomly.
        min_cnt = min(self._connection_counts.values())
        candidates = [
            inst for inst in self.instances if self._connection_counts[inst.address] == min_cnt
        ]
        chosen = random.choice(candidates)
        logger.debug(
            "LeastConnections selected %s (active=%d)", chosen.address, min_cnt
        )
        return chosen


# --------------------------------------------------------------------------- #
# Connection pooling & multiplexing
# --------------------------------------------------------------------------- #
class WebSocketPool:
    """
    Maintains a single WebSocket per remote peer but allows many logical
    streams (identified by ``msg_id``) to share the underlying TCP connection.
    """

    def __init__(self, token_manager: LoadTokenManager, handshake: Optional[CoordinationHandshake] = None) -> None:
        self._token_manager = token_manager
        self._handshake = handshake
        self._connections: Dict[str, Any] = {}  # WebSocketClientProtocol when available
        self._send_locks: Dict[str, asyncio.Lock] = {}
        self._receive_queues: Dict[str, Deque[Envelope]] = defaultdict(deque)
        self._pending_acks: Dict[str, asyncio.Event] = {}
        self._listener_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Kick‑off background listening task."""
        if not HAS_WEBSOCKET_DEPS:
            logger.warning("WebSocket dependencies not available - running in mock mode")
            return
        self._listener_task = asyncio.create_task(self._listener_loop())
        logger.info("WebSocketPool listener started")

    async def shutdown(self) -> None:
        """Close all connections and stop the listener."""
        self._stop_event.set()
        if self._listener_task:
            await self._listener_task
        if HAS_WEBSOCKET_DEPS:
            close_tasks = [ws.close() for ws in self._connections.values()]
            await asyncio.gather(*close_tasks, return_exceptions=True)
        self._connections.clear()
        logger.info("WebSocketPool shut down")

    async def _listener_loop(self) -> None:
        """Continuously reads from all open sockets and dispatches messages."""
        while not self._stop_event.is_set():
            if not self._connections:
                await asyncio.sleep(0.1)
                continue
            # Use ``asyncio.wait`` to read from any socket that becomes ready.
            ws_futs = {
                asyncio.create_task(ws.recv()): addr for addr, ws in self._connections.items()
            }
            done, _ = await asyncio.wait(ws_futs, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)

            for fut in done:
                addr = ws_futs[fut]
                try:
                    raw = fut.result()
                    envelope = self._parse_envelope(raw)
                    await self._handle_incoming(envelope, addr)
                except Exception as exc:
                    if "websockets" in str(type(exc)):
                        logger.warning("WebSocket to %s closed – will be removed", addr)
                        await self._remove_connection(addr)
                    else:
                        logger.exception("Error processing inbound message from %s: %s", addr, exc)

    async def _remove_connection(self, address: str) -> None:
        ws = self._connections.pop(address, None)
        if ws and HAS_WEBSOCKET_DEPS:
            await ws.close()
        self._send_locks.pop(address, None)

    @staticmethod
    def _parse_envelope(raw: str) -> Envelope:
        data = json.loads(raw)
        return Envelope(
            msg_id=data["msg_id"],
            msg_type=MessageType(data["msg_type"]),
            payload=data["payload"],
            token=data["token"],
        )

    async def _handle_incoming(self, envelope: Envelope, address: str) -> None:
        """Process inbound envelopes – acks, messages, heartbeats, handshake messages."""
        # First, validate token.
        try:
            self._token_manager.validate(envelope.token)
        except TokenValidationError:
            logger.warning("Dropping message with invalid token from %s", address)
            return

        if envelope.msg_type == MessageType.RESPONSE:
            # Resolve pending ack if any.
            evt = self._pending_acks.get(envelope.msg_id)
            if evt:
                self._receive_queues[address].append(envelope)
                evt.set()
        elif envelope.msg_type == MessageType.HEARTBEAT:
            # Respond with a lightweight heartbeat.
            await self._send_raw(
                address,
                Envelope(
                    msg_id=envelope.msg_id,
                    msg_type=MessageType.HEARTBEAT,
                    payload={},
                    token=self._token_manager.generate(),
                ),
            )
        elif envelope.msg_type == MessageType.HELLO and self._handshake:
            # Handle incoming HELLO message with handshake response
            response_payload = await self._handshake.handle_hello(envelope.payload)
            await self._send_raw(
                address,
                Envelope(
                    msg_id=envelope.msg_id,
                    msg_type=MessageType.RESPONSE,
                    payload=response_payload,
                    token=self._token_manager.generate(),
                ),
            )
        elif envelope.msg_type == MessageType.ACK and self._handshake:
            # Handle incoming ACK message
            response_payload = await self._handshake.handle_ack(envelope.payload, address)
            await self._send_raw(
                address,
                Envelope(
                    msg_id=envelope.msg_id,
                    msg_type=MessageType.RESPONSE,
                    payload=response_payload,
                    token=self._token_manager.generate(),
                ),
            )
        else:
            # For REQUEST / CONTROL we simply queue; callers can poll.
            self._receive_queues[address].append(envelope)

    async def _send_raw(self, address: str, envelope: Envelope) -> None:
        """Serialize and send directly – assumes connection already exists."""
        if address not in self._connections:
            raise CoordinationError(f"No open WS connection to {address}")

        if not HAS_WEBSOCKET_DEPS:
            # Mock mode - just log the send
            logger.debug("Mock sending envelope to %s: %s", address, envelope.msg_id)
            return

        async with self._send_locks[address]:
            ws = self._connections[address]
            await ws.send(json.dumps(envelope._asdict()))

    async def send_request(
        self,
        address: str,
        payload: dict,
        timeout: float = 10.0,
        *,
        msg_type: MessageType = MessageType.REQUEST,
    ) -> dict:
        """
        Send a request envelope and await the paired response.

        :returns: ``payload`` of the response message.
        :raises asyncio.TimeoutError: if the peer does not answer in time.
        """
        # Ensure the connection exists (lazy connect).
        if address not in self._connections:
            await self._ensure_connection(address)

        msg_id = str(uuid.uuid4())
        token = self._token_manager.generate()
        envelope = Envelope(
            msg_id=msg_id,
            msg_type=msg_type,
            payload=payload,
            token=token,
        )
        ack_evt = asyncio.Event()
        self._pending_acks[msg_id] = ack_evt

        await self._send_raw(address, envelope)

        if not HAS_WEBSOCKET_DEPS:
            # Mock mode - return a dummy response
            return {"mock_response": True, "original_payload": payload}

        try:
            await asyncio.wait_for(ack_evt.wait(), timeout=timeout)
        finally:
            # Cleanup bookkeeping regardless of outcome.
            self._pending_acks.pop(msg_id, None)

        # Retrieve the queued response.
        resp_queue = self._receive_queues[address]
        for i, env in enumerate(resp_queue):
            if env.msg_id == msg_id:
                resp_queue.remove(env)  # type: ignore[arg-type]
                return env.payload
        raise CoordinationError("Response received but not found in queue")

    async def _ensure_connection(self, address: str) -> None:
        """Create a new WebSocket connection if not already present."""
        if address in self._connections:
            return
            
        logger.info("Opening WS connection to %s", address)
        
        if not HAS_WEBSOCKET_DEPS:
            # Mock connection
            self._connections[address] = f"mock-connection-{address}"
            self._send_locks[address] = asyncio.Lock()
            return
            
        ws = await websockets.connect(address, ping_interval=20, ping_timeout=20)
        self._connections[address] = ws
        self._send_locks[address] = asyncio.Lock()
        # Send a greeting / health ping so the remote knows we are alive.
        await self._send_raw(
            address,
            Envelope(
                msg_id=str(uuid.uuid4()),
                msg_type=MessageType.HEARTBEAT,
                payload={},
                token=self._token_manager.generate(),
            ),
        )

    async def broadcast(self, payload: dict, timeout: float = 5.0) -> Dict[str, dict]:
        """
        Send the same payload to *all* connected peers concurrently.

        :returns: ``address → response payload`` mapping.
        """
        tasks = {
            addr: self.send_request(addr, payload, timeout=timeout)
            for addr in self._connections.keys()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return {
            addr: (res if not isinstance(res, Exception) else {"error": str(res)})
            for addr, res in zip(tasks.keys(), results)
        }


# --------------------------------------------------------------------------- #
# Health monitoring & automatic fail‑over
# --------------------------------------------------------------------------- #
class HealthMonitor:
    """
    Periodically checks the health of each discovered instance and notifies the
    load balancer when instances become unhealthy or recover.

    Health checks are performed via HTTP ``/health`` endpoint (configurable).
    """

    def __init__(
        self,
        service_name: str,
        balancer: LoadBalancer,
        registry: Optional[AgentRegistry] = None,
        check_interval: float = 10.0,
        health_path: str = "/health",
        timeout: float = 3.0,
    ) -> None:
        self.service_name = service_name
        self.balancer = balancer
        self.registry = registry or AgentRegistry()
        self.check_interval = check_interval
        self.health_path = health_path
        self.timeout = timeout
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        logger.info("HealthMonitor started for %s", self.service_name)

    async def shutdown(self) -> None:
        self._stop_event.set()
        if self._task:
            await self._task
        logger.info("HealthMonitor stopped")

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            await self._refresh_instances()
            await asyncio.sleep(self.check_interval)

    async def _refresh_instances(self) -> None:
        """
        Query the discovery engine, run health checks and feed the *healthy* list
        to the load balancer.
        """
        # Get instances from registry
        raw_instances = self.registry.get_all()
        candidates = [
            ServiceInstance(
                address=f"ws://{agent.address if hasattr(agent, 'address') and agent.address else 'localhost:8080'}",
                weight=getattr(agent, 'weight', 1),
                metadata={"agent_id": agent.agent_id},
            )
            for agent in raw_instances
        ]
        
        healthy: List[ServiceInstance] = []

        if not HAS_WEBSOCKET_DEPS:
            # In mock mode, consider all instances healthy
            healthy = candidates
        else:
            async with aiohttp.ClientSession() as session:
                check_tasks = [
                    self._probe(session, inst) for inst in candidates
                ]
                results = await asyncio.gather(*check_tasks, return_exceptions=True)

            for inst, ok in zip(candidates, results):
                if isinstance(ok, Exception):
                    logger.warning("Health check failed for %s: %s", inst.address, ok)
                elif ok:
                    healthy.append(inst)

        if not healthy:
            logger.error("No healthy instances discovered for %s!", self.service_name)
        else:
            logger.debug("Discovered %d healthy instances", len(healthy))
        self.balancer.update_instances(healthy)

    async def _probe(self, session: aiohttp.ClientSession, inst: ServiceInstance) -> bool:
        """
        Perform a single HTTP GET to ``address + health_path``.
        Returns ``True`` if status is 2xx within ``self.timeout``.
        """
        # Convert WebSocket URL to HTTP for health check
        http_url = inst.address.replace("ws://", "http://").replace("wss://", "https://")
        url = f"{http_url.rstrip('/')}{self.health_path}"
        try:
            async with session.get(url, timeout=self.timeout) as resp:
                ok = 200 <= resp.status < 300
                logger.debug("Health probe %s → %s", url, resp.status)
                return ok
        except Exception as exc:
            logger.debug("Health probe exception for %s: %s", url, exc)
            return False


# --------------------------------------------------------------------------- #
# High‑level Coordination Node (public API)
# --------------------------------------------------------------------------- #
class CoordinationNode:
    """
    Facade that combines discovery, load‑balancing, circuit‑breaker,
    health monitoring and a websocket pool to provide *distributed coordination*
    for AgentsMCP.

    Example
    -------
    >>> node = CoordinationNode(
    ...     service_name="agents-mcp",
    ...     load_balancing="weighted",
    ...     jwt_secret="very‑secret",
    ... )
    >>> await node.start()
    >>> response = await node.send_task({"action": "ping"})
    >>> await node.shutdown()
    """

    def __init__(
        self,
        service_name: str,
        agent_id: Optional[str] = None,
        capabilities: Optional[Set[str]] = None,
        load_balancing: Union[LBStrategy, str] = LBStrategy.ROUND_ROBIN,
        jwt_secret: str = "change‑me",
        jwt_issuer: str = "agents-mcp",
        health_check_interval: float = 10.0,
        circuit_breaker_cfg: Optional[Dict[str, Any]] = None,
        registry: Optional[AgentRegistry] = None,
    ) -> None:
        self.service_name = service_name
        self.agent_id = agent_id or f"agent-{uuid.uuid4()}"
        self.capabilities = capabilities or {"coordination", "discovery"}
        self.lb_strategy = LBStrategy(load_balancing)
        self.token_manager = LoadTokenManager(secret=jwt_secret, issuer=jwt_issuer)

        # Resolve the concrete balancer.
        self.balancer: LoadBalancer = self._make_balancer(self.lb_strategy, [])

        self.circuit_breaker = CircuitBreaker(**(circuit_breaker_cfg or {}))

        # AD4: Coordination handshake functionality
        self.handshake = CoordinationHandshake(self.agent_id, self.capabilities)
        
        self.pool = WebSocketPool(token_manager=self.token_manager, handshake=self.handshake)

        self.health_monitor = HealthMonitor(
            service_name=service_name,
            balancer=self.balancer,
            registry=registry,
            check_interval=health_check_interval,
        )

        self._started = False

    # --------------------------------------------------------------------- #
    # Construction helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _make_balancer(
        strategy: LBStrategy,
        instances: List[ServiceInstance],
    ) -> LoadBalancer:
        if strategy == LBStrategy.ROUND_ROBIN:
            return RoundRobinLoadBalancer(instances)
        if strategy == LBStrategy.WEIGHTED:
            return WeightedLoadBalancer(instances)
        if strategy == LBStrategy.LEAST_CONNECTIONS:
            return LeastConnectionsLoadBalancer(instances)
        raise ValueError(f"Unsupported load‑balancing strategy: {strategy}")

    # --------------------------------------------------------------------- #
    # Lifecycle management
    # --------------------------------------------------------------------- #
    async def start(self) -> None:
        """
        Initialise / start all background tasks:
        * health monitoring + discovery refresh
        * websocket pool listener
        """
        if self._started:
            logger.warning("CoordinationNode already started")
            return
        await self.pool.start()
        await self.health_monitor.start()
        self._started = True
        logger.info("CoordinationNode started (service=%s)", self.service_name)

    async def shutdown(self) -> None:
        """Gracefully stop background tasks and close all sockets."""
        if not self._started:
            logger.warning("CoordinationNode shutdown called before start")
            return
        await self.health_monitor.shutdown()
        await self.pool.shutdown()
        self._started = False
        logger.info("CoordinationNode shut down")

    # --------------------------------------------------------------------- #
    # Public request API
    # --------------------------------------------------------------------- #
    async def send_task(self, payload: dict, timeout: float = 15.0) -> dict:
        """
        High‑level helper that selects a healthy instance, wraps the payload into
        a request envelope, runs it through the circuit‑breaker and returns the
        remote response payload.

        :param payload: arbitrary JSON‑serialisable dict that represents the work.
        :param timeout: overall timeout for the operation (including retries).
        :raises CircuitBreakerOpen: if the chosen peer is currently tripped.
        """
        # 1️⃣ Choose an instance using the load‑balancer.
        target = self.balancer.select()
        logger.debug("send_task → selected %s", target.address)

        # 2️⃣ (optional) Update connection count for least‑connections balancer.
        if isinstance(self.balancer, LeastConnectionsLoadBalancer):
            self.balancer.increment(target.address)

        # 3️⃣ Wrap the network call with the circuit‑breaker.
        try:
            response = await self.circuit_breaker.call(
                self.pool.send_request,
                target.address,
                payload,
                timeout,
                msg_type=MessageType.REQUEST,
            )
        finally:
            if isinstance(self.balancer, LeastConnectionsLoadBalancer):
                self.balancer.decrement(target.address)

        logger.debug("send_task response from %s: %s", target.address, response)
        return response

    async def broadcast_task(self, payload: dict, timeout: float = 10.0) -> Dict[str, dict]:
        """
        Send ``payload`` to *all* currently known healthy peers and collect their
        responses.  Errors from individual peers are returned as ``{"error": "..."}``.

        :returns: Mapping ``address → response payload``.
        """
        # Ensure we have connections to all known healthy peers.
        for inst in self.balancer.instances:
            await self.pool._ensure_connection(inst.address)

        results = await self.pool.broadcast(payload, timeout=timeout)
        logger.debug("broadcast_task results: %s", results)
        return results
    
    # --------------------------------------------------------------------- #
    # AD4: Coordination API methods
    # --------------------------------------------------------------------- #
    async def initiate_handshake_with(self, remote_address: str) -> bool:
        """
        Initiate coordination handshake with a remote agent.
        
        Returns True if handshake succeeds, False otherwise.
        """
        return await self.handshake.initiate_handshake(remote_address, self.pool)
    
    def get_established_connections(self) -> List[str]:
        """Get list of addresses with established handshakes."""
        return self.handshake.get_established_connections()
    
    def get_handshake_state(self, address: str) -> Optional[HandshakeState]:
        """Get handshake state for a specific address."""
        return self.handshake.get_handshake_state(address)
    
    async def maintain_coordination_heartbeat(self) -> None:
        """Maintain heartbeat with all established coordination connections."""
        await self.handshake.maintain_heartbeat(self.pool)
    
    def get_negotiated_capabilities(self, address: str) -> Set[str]:
        """Get negotiated capabilities with a specific peer."""
        state = self.handshake.get_handshake_state(address)
        if state and state.state == "established":
            return self.capabilities.intersection(state.remote_capabilities)
        return set()


# --------------------------------------------------------------------------- #
# Module exports
# --------------------------------------------------------------------------- #
__all__ = [
    "CoordinationNode",
    "CoordinationHandshake",
    "HandshakeState",
    "LoadBalancer", 
    "RoundRobinLoadBalancer",
    "WeightedLoadBalancer", 
    "LeastConnectionsLoadBalancer",
    "ServiceInstance",
    "CircuitBreaker",
    "LoadTokenManager",
    "WebSocketPool",
    "HealthMonitor",
    "CoordinationError",
    "LoadBalancingError",
    "CircuitBreakerOpen",
    "TokenValidationError",
]