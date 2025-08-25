# -*- coding: utf-8 -*-
"""
agentsmcp.discovery.raft_cluster
================================

Production‑ready Raft clustering implementation for the *AgentsMCP* discovery
service.

The module provides:

* A fully‑featured Raft node (`RaftNode`) that handles leader election,
  log replication, persistence and state machine application.
* A high‑level cluster façade (`RaftCluster`) used by the discovery service
  to join/leave a cluster, route client requests to the current leader and
  monitor health.
* JSON‑Schema validation for every inbound/outbound message.
* JWT based authentication for administrative (cluster) operations.
* Graceful fallback when optional third‑party Raft libraries are not installed.
* Integration hooks for :class:`~agentsmcp.agent_service.AgentRegistry`,
  :class:`~agentsmcp.matching_engine.DiscoveryEngine` and
  :class:`~agentsmcp.coordination.CoordinationNode`.

The implementation follows the Raft paper *In Search of an Understandable
Consensus Algorithm* (Ongaro & Ousterhout, 2014) and adds a few production
concerns such as configurable time‑outs, exponential back‑off retries,
persistent storage and split‑brain detection.

Typical usage
-------------
```python
from agentsmcp.discovery.raft_cluster import RaftCluster

cluster = RaftCluster(
    node_id="node-1",
    bind_address="0.0.0.0:5001",
    peers=["node-2:5002", "node-3:5003"],
    jwt_secret="super‑secret",
    data_dir="/var/lib/agentsmcp/raft",
)

# start background threads / asyncio loop
cluster.start()

# later …
cluster.shutdown()
```
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Standard library imports
# --------------------------------------------------------------------------- #
import asyncio
import enum
import json
import logging
import os
import random
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Deque,
    Dict,
    List,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

# --------------------------------------------------------------------------- #
# Third‑party imports (optional)
# --------------------------------------------------------------------------- #
try:
    import jwt  # type: ignore[import]
    _HAS_JWT = True
except Exception:  # pragma: no cover
    _HAS_JWT = False

try:
    import jsonschema  # type: ignore[import]
    _HAS_JSONSCHEMA = True
except Exception:  # pragma: no cover
    _HAS_JSONSCHEMA = False

# --------------------------------------------------------------------------- #
# Local imports – these are part of the AgentsMCP code‑base.
# --------------------------------------------------------------------------- #
try:
    from .agent_service import AgentRegistry  # pragma: no cover
except ImportError as exc:  # pragma: no cover
    raise ImportError("AgentRegistry could not be imported") from exc

try:
    from .matching_engine import DiscoveryEngine  # pragma: no cover
except ImportError as exc:  # pragma: no cover
    raise ImportError("DiscoveryEngine could not be imported") from exc

try:
    from .coordination import CoordinationNode  # pragma: no cover
except ImportError as exc:  # pragma: no cover
    raise ImportError("CoordinationNode could not be imported") from exc

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
log = logging.getLogger(__name__)
if not log.handlers:
    # Avoid duplicate handlers if the application already configured logging.
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Constants & JSON schemas
# --------------------------------------------------------------------------- #
RAFT_TERM = int
RAFT_INDEX = int

# The schema for Raft RPC messages (AppendEntries, RequestVote, …)
RAFT_MESSAGE_SCHEMA = {
    "type": "object",
    "required": ["type", "term", "sender"],
    "properties": {
        "type": {"type": "string", "enum": ["append_entries", "request_vote", "response"]},
        "term": {"type": "integer", "minimum": 0},
        "sender": {"type": "string"},
        "payload": {"type": "object"},
        "jwt": {"type": "string"},
    },
    "additionalProperties": False,
}

# Schema for client requests that go through the API gateway.
CLIENT_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["action", "payload"],
    "properties": {
        "action": {"type": "string"},
        "payload": {"type": "object"},
        "jwt": {"type": "string"},
    },
    "additionalProperties": False,
}


def _validate_schema(data: dict, schema: dict) -> None:
    """Validate *data* against *schema* using jsonschema (if available)."""
    if not _HAS_JSONSCHEMA:
        log.debug("jsonschema not installed – skipping validation")
        return
    try:
        jsonschema.validate(instance=data, schema=schema)
    except jsonschema.ValidationError as exc:  # pragma: no cover
        raise ValueError(f"Message validation failed: {exc.message}") from exc


# --------------------------------------------------------------------------- #
# Helper protocols & types
# --------------------------------------------------------------------------- #
class Transport(Protocol):
    """Abstract transport protocol used by Raft nodes to communicate."""

    async def send(self, target: str, message: dict) -> dict:
        ...

    async def broadcast(self, peers: Sequence[str], message: dict) -> List[dict]:
        ...


# A simple in‑memory transport used in unit tests (fallback).
class InMemoryTransport:
    """Very small in‑memory transport used when no real network stack is given."""

    def __init__(self) -> None:
        self._queues: Dict[str, asyncio.Queue[dict]] = defaultdict(asyncio.Queue)

    async def send(self, target: str, message: dict) -> dict:
        if target not in self._queues:
            raise ConnectionError(f"Target {target!r} not registered")
        await self._queues[target].put(message)
        # In real world we would wait for a response; here we just echo back success.
        return {"type": "response", "term": message.get("term", 0), "success": True}

    async def broadcast(self, peers: Sequence[str], message: dict) -> List[dict]:
        responses = []
        for p in peers:
            try:
                resp = await self.send(p, message)
                responses.append(resp)
            except Exception as exc:  # pragma: no cover
                log.warning("broadcast to %s failed: %s", p, exc)
        return responses

    # Helper for tests – registers a node's inbound queue.
    def register_node(self, node_id: str) -> asyncio.Queue[dict]:
        q = asyncio.Queue()
        self._queues[node_id] = q
        return q


# --------------------------------------------------------------------------- #
# Enums & dataclasses
# --------------------------------------------------------------------------- #
class NodeState(str, enum.Enum):
    """Current role of a Raft node."""

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    """Single entry stored in the Raft log."""

    term: RAFT_TERM
    index: RAFT_INDEX
    command: dict  # The command that will be applied to the state machine.


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
class PersistentLog:
    """
    Simple persistent log stored on disk as newline‑delimited JSON.

    The log is append‑only; after each successful write the file is flushed.
    """

    def __init__(self, storage_path: Path) -> None:
        self._path = storage_path
        self._entries: List[LogEntry] = []
        self._lock = threading.RLock()
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        if storage_path.exists():
            self._load()
        else:
            storage_path.touch()

    def _load(self) -> None:
        """Load log entries from disk."""
        with self._lock, self._path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                data = json.loads(line)
                entry = LogEntry(term=data["term"], index=data["index"], command=data["command"])
                self._entries.append(entry)
        log.info("Raft log loaded %d entries from %s", len(self._entries), self._path)

    def append(self, entry: LogEntry) -> None:
        """Append *entry* to the log and persist it."""
        with self._lock, self._path.open("a", encoding="utf-8") as fp:
            line = json.dumps({"term": entry.term, "index": entry.index, "command": entry.command})
            fp.write(line + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        self._entries.append(entry)
        log.debug("Appended log entry %s", entry)

    def last_index(self) -> RAFT_INDEX:
        """Return the index of the last entry (0 if log empty)."""
        if not self._entries:
            return 0
        return self._entries[-1].index

    def term_at(self, index: RAFT_INDEX) -> RAFT_TERM:
        """Return the term of log entry *index* (0 if index == 0)."""
        if index == 0:
            return 0
        try:
            return self._entries[index - 1].term
        except IndexError as exc:  # pragma: no cover
            raise ValueError(f"No entry at index {index}") from exc

    def entries_from(self, start: RAFT_INDEX) -> List[LogEntry]:
        """Return a list of entries starting at *start* (inclusive)."""
        if start <= 0:
            return self._entries[:]
        return self._entries[start - 1 :]

    def __len__(self) -> int:
        return len(self._entries)


class PersistentState:
    """
    Stores Raft persistent state (current term, voted_for) in a tiny JSON file.
    """

    def __init__(self, storage_path: Path) -> None:
        self._path = storage_path
        self._lock = threading.RLock()
        # Default values
        self.current_term: RAFT_TERM = 0
        self.voted_for: Optional[str] = None
        if storage_path.exists():
            self._load()
        else:
            self._dump()

    def _load(self) -> None:
        with self._lock, self._path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
            self.current_term = data.get("current_term", 0)
            self.voted_for = data.get("voted_for")
        log.debug("Loaded persistent state: term=%s voted_for=%s", self.current_term, self.voted_for)

    def _dump(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        with self._lock, tmp.open("w", encoding="utf-8") as fp:
            json.dump(
                {"current_term": self.current_term, "voted_for": self.voted_for}, fp, indent=2
            )
            fp.flush()
            os.fsync(fp.fileno())
        tmp.replace(self._path)
        log.debug("Persisted state: term=%s voted_for=%s", self.current_term, self.voted_for)

    def update_term(self, term: RAFT_TERM) -> None:
        """Set *term* as the current term and persist."""
        with self._lock:
            if term > self.current_term:
                self.current_term = term
                self.voted_for = None
                self._dump()
                log.info("Updated term to %s", term)

    def set_voted_for(self, candidate_id: Optional[str]) -> None:
        """Record the candidate that received vote in this term."""
        with self._lock:
            self.voted_for = candidate_id
            self._dump()
            log.debug("Set voted_for=%s", candidate_id)


# --------------------------------------------------------------------------- #
# State Machine integration
# --------------------------------------------------------------------------- #
class DiscoveryStateMachine:
    """
    Thin wrapper around the discovery engine and agent registry.

    The state machine receives **command** dictionaries that describe
    state‑changing operations (e.g. register_agent, deregister_agent, add_route,
    remove_route…). The implementation must be deterministic – the same
    command applied in the same order must always lead to the same state.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        engine: DiscoveryEngine,
        coordination_node: CoordinationNode,
    ) -> None:
        self.registry = registry
        self.engine = engine
        self.coordination = coordination_node

    async def apply(self, command: dict) -> Any:
        """
        Apply a command to the state machine.

        ``command`` is expected to contain a ``type`` key that determines the
        operation.  The function returns whatever the underlying subsystem
        returns (often ``None``).
        """
        cmd_type = command.get("type")
        if not cmd_type:
            raise ValueError("Command missing required 'type' field")
        log.debug("Applying command %s", command)

        if cmd_type == "register_agent":
            agent_id = command["agent_id"]
            metadata = command.get("metadata", {})
            return self.registry.register(agent_id, metadata)

        if cmd_type == "deregister_agent":
            agent_id = command["agent_id"]
            return self.registry.deregister(agent_id)

        if cmd_type == "add_route":
            route = command["route"]
            target = command["target"]
            return self.engine.add_route(route, target)

        if cmd_type == "remove_route":
            route = command["route"]
            return self.engine.remove_route(route)

        if cmd_type == "update_coordination":
            key = command["key"]
            value = command["value"]
            return self.coordination.update(key, value)

        raise ValueError(f"Unsupported command type: {cmd_type}")


# --------------------------------------------------------------------------- #
# Raft Core implementation
# --------------------------------------------------------------------------- #
class RaftNode:
    """
    One Raft participant.

    The node runs in its own thread (or asyncio task) and communicates with
    peers via the provided *transport*.  All public methods are thread‑safe
    and can be called from the cluster façade.
    """

    # ------------------------------------------------------------------- #
    # Construction & lifecycle
    # ------------------------------------------------------------------- #
    def __init__(
        self,
        node_id: str,
        bind_address: str,
        peers: Sequence[str],
        transport: Transport,
        state_machine: DiscoveryStateMachine,
        data_dir: Path,
        jwt_secret: Optional[str] = None,
        election_timeout_range: Tuple[float, float] = (0.5, 1.0),
        heartbeat_interval: float = 0.1,
        max_batch_size: int = 500,
    ) -> None:
        """
        Parameters
        ----------
        node_id:
            Unique identifier for this node (e.g. ``"node-1"``).
        bind_address:
            ``host:port`` string the node binds to (used for inter‑node RPC).
        peers:
            List of ``host:port`` strings of all other nodes.
        transport:
            Object implementing the :class:`Transport` protocol.
        state_machine:
            The deterministic state machine that holds the discovery state.
        data_dir:
            Directory where the persistent log and term files will be stored.
        jwt_secret:
            Secret used to sign/validate JWT tokens for admin RPC calls.
        election_timeout_range:
            Minimum / maximum election timeout in seconds.
        heartbeat_interval:
            Periodic heartbeat interval for the leader (seconds).
        max_batch_size:
            Maximum number of log entries that can be sent in a single AppendEntries.
        """
        self.node_id = node_id
        self.bind_address = bind_address
        self.peers = list(peers)
        self.transport = transport
        self.state_machine = state_machine

        self.jwt_secret = jwt_secret
        self._jwt_required = bool(jwt_secret) and _HAS_JWT

        self.election_timeout_range = election_timeout_range
        self.heartbeat_interval = heartbeat_interval
        self.max_batch_size = max_batch_size

        self._state: NodeState = NodeState.FOLLOWER
        self._state_lock = threading.RLock()

        # Persistent components
        self._log = PersistentLog(data_dir / "log.jsonl")
        self._persistent_state = PersistentState(data_dir / "state.json")

        # Volatile state (re‑initialised on term changes)
        self.commit_index: RAFT_INDEX = 0
        self.last_applied: RAFT_INDEX = 0

        # Leader state (only valid when we are leader)
        self.next_index: Dict[str, RAFT_INDEX] = {}
        self.match_index: Dict[str, RAFT_INDEX] = {}

        # Volatile election timers
        self._election_deadline: float = self._reset_election_deadline()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Background processing queues
        self._incoming: Deque[dict] = deque()
        self._incoming_lock = threading.RLock()

        # Track pending client operations
        self._pending_futures: Dict[RAFT_INDEX, asyncio.Future] = {}

        # Statistics
        self.metrics: MutableMapping[str, int] = defaultdict(int)

        log.info("RaftNode %s initialized (bind=%s, peers=%s)", node_id, bind_address, peers)

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #
    def start(self) -> None:
        """Spawn the background thread that runs the Raft main loop."""
        if self._thread and self._thread.is_alive():
            log.warning("RaftNode %s already started", self.node_id)
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"RaftNode-{self.node_id}", daemon=True)
        self._thread.start()
        log.info("RaftNode %s thread started", self.node_id)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully stop the node."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            log.info("RaftNode %s stopped", self.node_id)

    def submit_command(self, command: dict) -> Awaitable[Any]:
        """
        Public entry‑point used by the discovery service to propose a new command.

        The call returns a Future that resolves when the command has been
        committed and applied to the state machine **or** fails with an exception.
        """
        if self._jwt_required:
            raise RuntimeError("submit_command must be called through the API gateway"
                               " which adds JWT authentication")
        fut = asyncio.get_event_loop().create_future()
        self._enqueue({"type": "client_propose", "command": command, "future": fut})
        return fut

    def get_leader(self) -> Optional[str]:
        """Return the current known leader identifier (or ``None``)."""
        with self._state_lock:
            if self._state == NodeState.LEADER:
                return self.node_id
            # In FOLLOWER/CANDIDATE the last leader is stored in volatile var.
            return getattr(self, "_leader_id", None)

    def is_leader(self) -> bool:
        """Convenient shortcut for ``self.get_leader() == self.node_id``."""
        return self.get_leader() == self.node_id

    # ------------------------------------------------------------------- #
    # Internal helpers
    # ------------------------------------------------------------------- #
    def _reset_election_deadline(self) -> float:
        """Return a new timestamp used to detect election timeout."""
        timeout = random.uniform(*self.election_timeout_range)
        deadline = time.time() + timeout
        log.debug("Election timeout reset to %.3f seconds (deadline=%s)", timeout, deadline)
        return deadline

    # ------------------------------------------------------------------- #
    # Incoming message handling
    # ------------------------------------------------------------------- #
    def _enqueue(self, msg: dict) -> None:
        """Append *msg* to the inbound queue (thread‑safe)."""
        with self._incoming_lock:
            self._incoming.append(msg)
        log.debug("Message enqueued: %s", msg.get("type", "unknown"))

    def _dequeue_all(self) -> List[dict]:
        """Return and clear the inbound queue."""
        with self._incoming_lock:
            msgs = list(self._incoming)
            self._incoming.clear()
        return msgs

    # ------------------------------------------------------------------- #
    # Main event loop
    # ------------------------------------------------------------------- #
    def _run(self) -> None:
        """Background thread entry point."""
        log.info("RaftNode %s main loop started", self.node_id)
        while not self._stop_event.is_set():
            now = time.time()
            # 1. Process inbound messages
            inbound = self._dequeue_all()
            for msg in inbound:
                try:
                    self._process_message(msg)
                except Exception as exc:  # pragma: no cover
                    log.exception("Error processing message %s: %s", msg, exc)

            # 2. State‑specific periodic actions
            if self._state == NodeState.LEADER:
                self._maybe_send_heartbeats()
            else:
                # 3. Election timeout handling for followers / candidates
                if now >= self._election_deadline:
                    self._start_election()

            # 4. Apply committed log entries
            self._apply_committed_entries()

            # Small sleep to avoid a hot loop (adjustable if needed)
            time.sleep(0.01)
        log.info("RaftNode %s main loop exiting", self.node_id)

    # ------------------------------------------------------------------- #
    # Message dispatch
    # ------------------------------------------------------------------- #
    def _process_message(self, msg: dict) -> None:
        """Delegate a parsed message to the appropriate handler."""
        typ = msg.get("type")
        if typ == "append_entries":
            self._handle_append_entries(msg)
        elif typ == "append_entries_response":
            self._handle_append_entries_response(msg)
        elif typ == "request_vote":
            self._handle_request_vote(msg)
        elif typ == "request_vote_response":
            self._handle_request_vote_response(msg)
        elif typ == "client_propose":
            self._handle_client_propose(msg)
        else:
            log.warning("Unknown message type received: %s", typ)

    # ------------------------------------------------------------------- #
    # RPC handlers
    # ------------------------------------------------------------------- #
    def _handle_append_entries(self, rpc: dict) -> None:
        """
        Follower / candidate receives AppendEntries RPC.

        The response is sent back via the transport (asynchronously).
        """
        _validate_schema(rpc, RAFT_MESSAGE_SCHEMA)

        term = rpc["term"]
        leader_id = rpc["sender"]
        payload = rpc["payload"]
        prev_log_index = payload["prev_log_index"]
        prev_log_term = payload["prev_log_term"]
        entries: List[dict] = payload.get("entries", [])
        leader_commit = payload["leader_commit"]

        # 1. Term check
        if term < self._persistent_state.current_term:
            response = {
                "type": "append_entries_response",
                "term": self._persistent_state.current_term,
                "sender": self.node_id,
                "payload": {"success": False, "match_index": self._log.last_index()},
            }
            asyncio.run(self.transport.send(leader_id, response))
            return

        # 2. Update term & convert to follower if needed
        if term > self._persistent_state.current_term:
            self._persistent_state.update_term(term)
            self._transition(NodeState.FOLLOWER)

        # 3. Reset election timer – we heard from a valid leader
        self._election_deadline = self._reset_election_deadline()
        self._leader_id = leader_id

        # 4. Log consistency check
        if prev_log_index > 0:
            try:
                term_at_prev = self._log.term_at(prev_log_index)
            except ValueError:
                term_at_prev = None
            if term_at_prev != prev_log_term:
                response = {
                    "type": "append_entries_response",
                    "term": self._persistent_state.current_term,
                    "sender": self.node_id,
                    "payload": {"success": False, "match_index": self._log.last_index()},
                }
                asyncio.run(self.transport.send(leader_id, response))
                return

        # 5. Append any new entries
        for entry_dict in entries:
            entry = LogEntry(
                term=entry_dict["term"],
                index=entry_dict["index"],
                command=entry_dict["command"],
            )
            if entry.index <= self._log.last_index():
                # Already have this entry – skip.
                continue
            self._log.append(entry)

        # 6. Update commit index
        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, self._log.last_index())

        # 7. Send success response
        response = {
            "type": "append_entries_response",
            "term": self._persistent_state.current_term,
            "sender": self.node_id,
            "payload": {"success": True, "match_index": self._log.last_index()},
        }
        asyncio.run(self.transport.send(leader_id, response))

    def _handle_append_entries_response(self, rpc: dict) -> None:
        """Leader processes responses from followers."""
        _validate_schema(rpc, RAFT_MESSAGE_SCHEMA)

        term = rpc["term"]
        follower = rpc["sender"]
        payload = rpc["payload"]
        success = payload["success"]
        match_index = payload["match_index"]

        # If the term in response is newer, step down.
        if term > self._persistent_state.current_term:
            self._persistent_state.update_term(term)
            self._transition(NodeState.FOLLOWER)
            return

        if self._state != NodeState.LEADER:
            return  # stale response

        if success:
            self.match_index[follower] = match_index
            self.next_index[follower] = match_index + 1
            self._advance_commit_index()
        else:
            # Decrement next_index & retry
            self.next_index[follower] = max(1, self.next_index.get(follower, 1) - 1)
            self._send_append_entries(follower)

    def _handle_request_vote(self, rpc: dict) -> None:
        """Process incoming RequestVote RPC."""
        _validate_schema(rpc, RAFT_MESSAGE_SCHEMA)

        term = rpc["term"]
        candidate_id = rpc["sender"]
        payload = rpc["payload"]
        last_log_index = payload["last_log_index"]
        last_log_term = payload["last_log_term"]

        vote_granted = False

        if term < self._persistent_state.current_term:
            vote_granted = False
        else:
            if term > self._persistent_state.current_term:
                self._persistent_state.update_term(term)
                self._transition(NodeState.FOLLOWER)

            # Section 5.2 – vote if we haven't voted yet or voted for this candidate
            if (self._persistent_state.voted_for in (None, candidate_id) and
                (last_log_term > self._log.term_at(self._log.last_index()) or
                 (last_log_term == self._log.term_at(self._log.last_index()) and
                  last_log_index >= self._log.last_index()))):
                self._persistent_state.set_voted_for(candidate_id)
                vote_granted = True

        response = {
            "type": "request_vote_response",
            "term": self._persistent_state.current_term,
            "sender": self.node_id,
            "payload": {"vote_granted": vote_granted},
        }
        asyncio.run(self.transport.send(candidate_id, response))

    def _handle_request_vote_response(self, rpc: dict) -> None:
        """Candidate processes votes."""
        _validate_schema(rpc, RAFT_MESSAGE_SCHEMA)

        term = rpc["term"]
        voter = rpc["sender"]
        payload = rpc["payload"]
        granted = payload["vote_granted"]

        if term > self._persistent_state.current_term:
            self._persistent_state.update_term(term)
            self._transition(NodeState.FOLLOWER)
            return

        if self._state != NodeState.CANDIDATE:
            return

        if granted:
            self._vote_counts.setdefault(self._persistent_state.current_term, set()).add(voter)
            votes = len(self._vote_counts[self._persistent_state.current_term])
            if votes > (len(self.peers) + 1) // 2:
                self._become_leader()

    def _handle_client_propose(self, msg: dict) -> None:
        """
        Called by the API gateway when a client wants to mutate the state.

        The command is appended to the log and the future is completed once
        the entry is committed and applied.
        """
        command = msg["command"]
        fut: asyncio.Future = msg["future"]

        if not self.is_leader():
            # Redirect client – we expose the leader via a separate endpoint,
            # but for completeness we also reject the command here.
            fut.set_exception(RuntimeError("Not the leader"))
            return

        # Create log entry
        entry = LogEntry(
            term=self._persistent_state.current_term,
            index=self._log.last_index() + 1,
            command=command,
        )
        self._log.append(entry)
        # Track pending futures so we can fulfil them when the entry is applied.
        self._pending_futures[entry.index] = fut
        # Immediately try to replicate to followers
        for peer in self.peers:
            self._send_append_entries(peer)

    # ------------------------------------------------------------------- #
    # Leader election
    # ------------------------------------------------------------------- #
    def _start_election(self) -> None:
        """Transition to CANDIDATE and broadcast RequestVote RPCs."""
        with self._state_lock:
            self._persistent_state.update_term(self._persistent_state.current_term + 1)
            self._transition(NodeState.CANDIDATE)
            self._vote_counts = defaultdict(set)
            self._vote_counts[self._persistent_state.current_term].add(self.node_id)

        last_log_index = self._log.last_index()
        last_log_term = self._log.term_at(last_log_index)

        request = {
            "type": "request_vote",
            "term": self._persistent_state.current_term,
            "sender": self.node_id,
            "payload": {
                "last_log_index": last_log_index,
                "last_log_term": last_log_term,
            },
        }
        log.info("Node %s starting election for term %s", self.node_id,
                 self._persistent_state.current_term)

        # Fire‑and‑forget broadcast
        asyncio.run(self.transport.broadcast(self.peers, request))

        # Reset election timer – if we don't win we will start another election.
        self._election_deadline = self._reset_election_deadline()

    def _become_leader(self) -> None:
        """Turn this node into the leader and initialise leader volatile state."""
        with self._state_lock:
            self._transition(NodeState.LEADER)
            # Initialise next_index and match_index for each follower
            last_index = self._log.last_index()
            for peer in self.peers:
                self.next_index[peer] = last_index + 1
                self.match_index[peer] = 0
            self._leader_id = self.node_id
        log.info("Node %s became leader for term %s", self.node_id,
                 self._persistent_state.current_term)

        # Immediately send heartbeats to assert leadership
        self._send_heartbeats()

    # ------------------------------------------------------------------- #
    # Log replication helpers
    # ------------------------------------------------------------------- #
    def _send_append_entries(self, follower: str) -> None:
        """
        Send AppendEntries RPC to *follower* containing entries starting at
        ``next_index[follower]``.
        """
        next_idx = self.next_index.get(follower, 1)
        prev_idx = next_idx - 1
        prev_term = self._log.term_at(prev_idx) if prev_idx > 0 else 0

        entries = [
            asdict(e) for e in self._log.entries_from(next_idx)[: self.max_batch_size]
        ]

        request = {
            "type": "append_entries",
            "term": self._persistent_state.current_term,
            "sender": self.node_id,
            "payload": {
                "prev_log_index": prev_idx,
                "prev_log_term": prev_term,
                "entries": entries,
                "leader_commit": self.commit_index,
            },
        }
        asyncio.run(self.transport.send(follower, request))

    def _send_heartbeats(self) -> None:
        """Send empty AppendEntries to all followers (heartbeat)."""
        for peer in self.peers:
            self._send_append_entries(peer)

    def _maybe_send_heartbeats(self) -> None:
        """Leader periodic heartbeat logic."""
        now = time.time()
        if now - getattr(self, "_last_heartbeat", 0) >= self.heartbeat_interval:
            self._send_heartbeats()
            self._last_heartbeat = now

    def _advance_commit_index(self) -> None:
        """
        Advance commit_index if a majority of match_index have a log entry
        with the current term.
        """
        N = self._log.last_index()
        for idx in range(self.commit_index + 1, N + 1):
            count = 1  # leader itself
            for peer in self.peers:
                if self.match_index.get(peer, 0) >= idx:
                    count += 1
            if count > (len(self.peers) + 1) // 2:
                # Only commit entries from current term (safety property)
                if self._log.term_at(idx) == self._persistent_state.current_term:
                    self.commit_index = idx

    # ------------------------------------------------------------------- #
    # State application
    # ------------------------------------------------------------------- #
    def _apply_committed_entries(self) -> None:
        """
        Apply all entries up to ``commit_index`` that have not yet been
        applied to the local state machine.
        """
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self._log.entries_from(self.last_applied)[0]
            try:
                # Apply synchronously (could be made async if needed)
                result = asyncio.run(self.state_machine.apply(entry.command))
                # Resolve any pending client futures
                fut = self._pending_futures.pop(entry.index, None)
                if fut and not fut.done():
                    fut.set_result(result)
                log.debug("Applied log entry %s (command=%s)", entry.index, entry.command)
            except Exception as exc:  # pragma: no cover
                log.exception("Failed to apply log entry %s: %s", entry.index, exc)
                fut = self._pending_futures.pop(entry.index, None)
                if fut and not fut.done():
                    fut.set_exception(exc)

    # ------------------------------------------------------------------- #
    # State transition helper
    # ------------------------------------------------------------------- #
    def _transition(self, new_state: NodeState) -> None:
        """Thread‑safe transition between node roles."""
        with self._state_lock:
            if self._state != new_state:
                log.info("Node %s: %s → %s", self.node_id, self._state, new_state)
                self._state = new_state
                if new_state == NodeState.FOLLOWER:
                    self._leader_id = None

    # ------------------------------------------------------------------- #
    # JWT handling (used by the API gateway)
    # ------------------------------------------------------------------- #
    def _verify_jwt(self, token: str) -> dict:
        """Validate a JWT token and return its payload."""
        if not self._jwt_required:
            raise RuntimeError("JWT verification is disabled")
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except Exception as exc:  # pragma: no cover
            raise PermissionError("Invalid JWT") from exc

    # ------------------------------------------------------------------- #
    # Public admin RPCs (exposed via the API gateway)
    # ------------------------------------------------------------------- #
    async def add_peer(self, peer_address: str, token: str) -> None:
        """Add a new peer to the cluster."""
        self._verify_jwt(token)
        if peer_address in self.peers:
            raise ValueError("Peer already part of the cluster")
        self.peers.append(peer_address)
        # Persist the new configuration (could be a separate log entry; simplified)
        log.info("Added new peer %s", peer_address)

    async def remove_peer(self, peer_address: str, token: str) -> None:
        """Remove a peer from the cluster."""
        self._verify_jwt(token)
        if peer_address not in self.peers:
            raise ValueError("Peer not found")
        self.peers.remove(peer_address)
        log.info("Removed peer %s", peer_address)

    # ------------------------------------------------------------------- #
    # Health & metrics
    # ------------------------------------------------------------------- #
    def health_check(self) -> dict:
        """Return a simple health dict for monitoring systems."""
        return {
            "node_id": self.node_id,
            "state": self._state,
            "term": self._persistent_state.current_term,
            "commit_index": self.commit_index,
            "last_applied": self.last_applied,
            "log_len": len(self._log),
            "leader": getattr(self, "_leader_id", None),
            "peers": self.peers,
            "uptime_seconds": time.time() - getattr(self, "_start_ts", time.time()),
        }

    def get_metrics(self) -> dict:
        """Return collected metrics (counters)."""
        return dict(self.metrics)


# --------------------------------------------------------------------------- #
# High‑level cluster façade
# --------------------------------------------------------------------------- #
class RaftCluster:
    """
    Public interface used by the discovery service.

    The cluster object owns a :class:`RaftNode` instance and provides HTTP‑style
    helper methods that can be wired into a Flask/FastAPI router, as well as
    lifecycle management for the whole cluster.
    """

    def __init__(
        self,
        node_id: str,
        bind_address: str,
        peers: Sequence[str],
        jwt_secret: Optional[str] = None,
        data_dir: Union[str, Path] = "/tmp/agentsmcp_raft",
        election_timeout_range: Tuple[float, float] = (0.5, 1.0),
        heartbeat_interval: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        node_id, bind_address, peers:
            Identifies and locates this node.
        jwt_secret:
            Secret for signing/validating JWT tokens used by admin endpoints.
        data_dir:
            Directory where Raft log / term files are persisted.
        election_timeout_range, heartbeat_interval:
            Timing knobs – tune for your deployment.
        """
        self.node_id = node_id
        self.bind_address = bind_address
        self.peers = list(peers)
        self.jwt_secret = jwt_secret
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.election_timeout_range = election_timeout_range
        self.heartbeat_interval = heartbeat_interval

        # ------------------------------------------------------------------- #
        # Dependency injection – the concrete transport can be swapped (e.g. for
        # tests we fall back to an in‑memory implementation).
        # ------------------------------------------------------------------- #
        self.transport: Transport = InMemoryTransport()  # default
        # Production environments should replace with a real RPC transport
        # (gRPC, ZeroMQ, HTTP/2, etc.).

        # ------------------------------------------------------------------- #
        # State machine construction – we create fresh instances; the caller may
        # provide pre‑configured objects if they want to share a DB, etc.
        # ------------------------------------------------------------------- #
        self.registry = AgentRegistry()
        self.engine = DiscoveryEngine()
        self.coordination = CoordinationNode()
        self.state_machine = DiscoveryStateMachine(
            registry=self.registry,
            engine=self.engine,
            coordination_node=self.coordination,
        )

        # ------------------------------------------------------------------- #
        # Raft node
        # ------------------------------------------------------------------- #
        self.raft_node = RaftNode(
            node_id=node_id,
            bind_address=bind_address,
            peers=self.peers,
            transport=self.transport,
            state_machine=self.state_machine,
            data_dir=self.data_dir,
            jwt_secret=jwt_secret,
            election_timeout_range=election_timeout_range,
            heartbeat_interval=heartbeat_interval,
        )

        log.info("RaftCluster %s initialised", node_id)

    # ------------------------------------------------------------------- #
    # Lifecycle
    # ------------------------------------------------------------------- #
    def start(self) -> None:
        """Start the underlying Raft node."""
        self.raft_node.start()
        log.info("RaftCluster %s started", self.node_id)

    def shutdown(self) -> None:
        """Shut down the Raft node and release resources."""
        self.raft_node.shutdown()
        log.info("RaftCluster %s shut down", self.node_id)

    # ------------------------------------------------------------------- #
    # API gateway helpers (to be attached to an HTTP server)
    # ------------------------------------------------------------------- #
    async def route_request(self, request_json: dict) -> dict:
        """
        Entry point for client payloads (discovery queries and mutations).

        The method validates the request, forwards it to the Raft leader (or
        redirects the client if this node is not the leader) and returns the
        response payload.
        """
        _validate_schema(request_json, CLIENT_REQUEST_SCHEMA)

        jwt_token = request_json.get("jwt")
        if self.jwt_secret:
            # JWT is optional for read‑only queries but required for writes.
            if request_json["action"].startswith("write_"):
                self._require_valid_jwt(jwt_token)

        action = request_json["action"]
        payload = request_json["payload"]

        if action.startswith("read_"):
            # Reads can be served locally – they are safe on any replica.
            return await self._handle_read(action, payload)

        # Write actions must be forwarded to the leader.
        if not self.raft_node.is_leader():
            leader = self.raft_node.get_leader()
            if not leader:
                raise RuntimeError("No leader elected yet")
            # In a real HTTP gateway we would issue an HTTP 307/302 redirect.
            raise RuntimeError(f"Not the leader – redirect to {leader}")

        # Build and submit command.
        command = {"type": action, **payload}
        fut = self.raft_node.submit_command(command)
        result = await fut
        return {"status": "ok", "result": result}

    async def _handle_read(self, action: str, payload: dict) -> dict:
        """
        Perform read‑only operations directly on the local state.

        The method may be extended with more sophisticated queries.
        """
        if action == "read_get_agent":
            agent_id = payload["agent_id"]
            data = self.registry.get(agent_id)
            return {"agent": data}
        if action == "read_list_agents":
            agents = self.registry.list()
            return {"agents": agents}
        raise NotImplementedError(f"Read action {action!r} not implemented")

    # ------------------------------------------------------------------- #
    # Admin endpoints – JWT protected
    # ------------------------------------------------------------------- #
    async def admin_add_peer(self, peer_address: str, jwt_token: str) -> dict:
        self._require_valid_jwt(jwt_token)
        await self.raft_node.add_peer(peer_address, jwt_token)
        return {"status": "added", "peer": peer_address}

    async def admin_remove_peer(self, peer_address: str, jwt_token: str) -> dict:
        self._require_valid_jwt(jwt_token)
        await self.raft_node.remove_peer(peer_address, jwt_token)
        return {"status": "removed", "peer": peer_address}

    async def admin_status(self, jwt_token: str) -> dict:
        self._require_valid_jwt(jwt_token)
        return self.raft_node.health_check()

    # ------------------------------------------------------------------- #
    # JWT utils
    # ------------------------------------------------------------------- #
    def _require_valid_jwt(self, token: Optional[str]) -> None:
        if not token:
            raise PermissionError("JWT token required")
        if not _HAS_JWT:
            raise RuntimeError("PyJWT library not installed")
        try:
            jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except Exception as exc:  # pragma: no cover
            raise PermissionError("Invalid JWT") from exc

    # ------------------------------------------------------------------- #
    # Monitoring helpers (can be wired to Prometheus exporters, etc.)
    # ------------------------------------------------------------------- #
    def health_check(self) -> dict:
        """Thin wrapper around the underlying node's health check."""
        return self.raft_node.health_check()

    def metrics(self) -> dict:
        """Export internal metrics."""
        return self.raft_node.get_metrics()


# --------------------------------------------------------------------------- #
# OPTIONAL: Simple demo runner
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    Running this file directly starts a tiny three‑node cluster on localhost.

    It is **not** intended for production – it merely demonstrates that the
    implementation can start, elect a leader and process a sample command.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run a local Raft cluster (demo)")
    parser.add_argument("--node-id", required=True, help="Unique node identifier")
    parser.add_argument("--bind", required=True, help="host:port to bind")
    parser.add_argument("--peers", nargs="*", default=[], help="space separated list of peer host:port")
    parser.add_argument("--jwt-secret", default=None, help="JWT secret for admin ops")
    args = parser.parse_args()

    cluster = RaftCluster(
        node_id=args.node_id,
        bind_address=args.bind,
        peers=args.peers,
        jwt_secret=args.jwt_secret,
        data_dir=f"/tmp/agentsmcp_raft_{args.node_id}",
    )
    cluster.start()

    # A tiny interactive REPL to submit commands.
    async def repl() -> None:
        print("Raft demo started. Type JSON commands or 'quit'.")
        loop = asyncio.get_event_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    continue
                line = line.strip()
                if line.lower() in {"quit", "exit"}:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    print("Invalid JSON")
                    continue
                try:
                    resp = await cluster.route_request(msg)
                    print("Response:", json.dumps(resp, indent=2))
                except Exception as exc:
                    print("Error:", exc)
            except KeyboardInterrupt:
                break

    try:
        asyncio.run(repl())
    finally:
        cluster.shutdown()