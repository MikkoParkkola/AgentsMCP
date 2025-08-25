from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable


REGISTRY_PATH = Path(os.path.expanduser("~/.agentsmcp/registry.json"))
ENTRY_TTL_SEC = 120

logger = logging.getLogger(__name__)

# Thread safety lock for concurrent access
_REGISTRY_LOCK = threading.RLock()


@dataclass
class Entry:
    agent_id: str
    name: str
    capabilities: List[str]
    transport: str
    endpoint: str
    token: Optional[str] = None
    ts: float = 0.0


def _load() -> List[Dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return []
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except Exception:
        return []


def _save(items: List[Dict[str, Any]]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(items, indent=2))


def write_entry(e: Entry) -> None:
    with _REGISTRY_LOCK:
        items = _load()
        # remove existing by agent_id or name
        items = [it for it in items if it.get("agent_id") != e.agent_id and it.get("name") != e.name]
        it = asdict(e)
        it["ts"] = time.time()
        items.append(it)
        _save(items)


def list_entries() -> List[Entry]:
    now = time.time()
    out: List[Entry] = []
    changed = False
    for it in _load():
        try:
            if now - float(it.get("ts", 0)) > ENTRY_TTL_SEC:
                changed = True
                continue
            out.append(
                Entry(
                    agent_id=it.get("agent_id", ""),
                    name=it.get("name", ""),
                    capabilities=list(it.get("capabilities", [])),
                    transport=it.get("transport", "stdio"),
                    endpoint=it.get("endpoint", ""),
                    token=it.get("token"),
                    ts=float(it.get("ts", 0.0)),
                )
            )
        except Exception:
            continue
    if changed:
        _save([asdict(e) for e in out])
    return out


# ============================================================================
# Registry Cleanup Extension
# ============================================================================

# Maintenance throttling
_LAST_MAINTENANCE: float = 0.0
_MAINTENANCE_INTERVAL: float = 30.0  # seconds


class RegistryError(RuntimeError):
    """Base class for registry-related problems (corruption, I/O, etc.)."""
    pass


def validate_entry(raw: Any) -> Optional[Entry]:
    """
    Convert raw JSON dict into an Entry.
    Returns None if the payload is missing required keys or contains bogus data.
    """
    if not isinstance(raw, dict):
        logger.debug("Entry not a dict: %r", raw)
        return None

    required = {"agent_id", "name", "capabilities", "transport", "endpoint"}
    missing = required - raw.keys()
    if missing:
        logger.debug("Entry missing keys %s: %r", missing, raw)
        return None

    try:
        entry = Entry(
            agent_id=str(raw["agent_id"]),
            name=str(raw["name"]),
            capabilities=list(raw["capabilities"]),
            transport=str(raw["transport"]),
            endpoint=str(raw["endpoint"]),
            token=raw.get("token"),
            ts=float(raw.get("ts", 0.0)),
        )
    except Exception as exc:
        logger.debug("Entry failed validation (%s): %r", exc, raw)
        return None

    return entry


def cleanup_by_agent(agent_id: str) -> bool:
    """
    Remove all entries belonging to agent_id.
    Returns True if one or more entries were removed.
    """
    with _REGISTRY_LOCK:
        items = _load()
        original_count = len(items)
        
        # Filter out entries with matching agent_id
        filtered_items = [
            item for item in items
            if item.get("agent_id") != agent_id
        ]
        
        removed = original_count != len(filtered_items)
        if removed:
            _save(filtered_items)
            logger.info("Removed %d entries for agent_id=%s", 
                       original_count - len(filtered_items), agent_id)
        else:
            logger.debug("No entries found for cleanup_by_agent(%s)", agent_id)
    
    return removed


def cleanup_stale(now: Optional[float] = None) -> int:
    """
    Remove all entries older than ENTRY_TTL_SEC.
    Returns number of entries that were deleted.
    """
    if now is None:
        now = time.time()
        
    with _REGISTRY_LOCK:
        items = _load()
        original_count = len(items)
        
        # Filter out stale entries
        fresh_items = [
            item for item in items
            if (now - item.get("ts", 0)) <= ENTRY_TTL_SEC
        ]
        
        removed_count = original_count - len(fresh_items)
        if removed_count > 0:
            _save(fresh_items)
            logger.info("Stale-cleanup removed %d entries", removed_count)
        else:
            logger.debug("Stale-cleanup found no entries to delete")
    
    return removed_count


def batch_cleanup(agent_ids: Iterable[str]) -> Dict[str, bool]:
    """
    Remove entries for all supplied agent identifiers in one operation.
    Returns mapping agent_id → bool where True means at least one row was removed.
    """
    ids_set = {str(i) for i in agent_ids}
    result: Dict[str, bool] = {i: False for i in ids_set}
    
    with _REGISTRY_LOCK:
        items = _load()
        original_count = len(items)
        
        # Track which agent_ids had entries removed
        remaining_items = []
        for item in items:
            agent_id = item.get("agent_id")
            if agent_id in ids_set:
                result[agent_id] = True
            else:
                remaining_items.append(item)
        
        removed_count = original_count - len(remaining_items)
        if removed_count > 0:
            _save(remaining_items)
            removed_agents = sum(result.values())
            logger.info("Batch cleanup removed %d entries for %d agents",
                       removed_count, removed_agents)
        else:
            logger.debug("Batch cleanup found no matching entries")
    
    return result


def compact() -> None:
    """
    Rewrite the registry file, removing any corruption and optimizing storage.
    """
    with _REGISTRY_LOCK:
        items = _load()
        validated_items = []
        
        for item in items:
            validated_entry = validate_entry(item)
            if validated_entry:
                validated_items.append(asdict(validated_entry))
            else:
                logger.warning("Dropping corrupted entry during compaction: %r", item)
        
        _save(validated_items)
        logger.debug("Registry compaction completed - validated %d entries", 
                    len(validated_items))


def run_maintenance(force: bool = False) -> None:
    """
    Execute routine housekeeping: prune stale entries and compact storage.
    Throttled to _MAINTENANCE_INTERVAL seconds unless force is True.
    """
    global _LAST_MAINTENANCE
    now = time.time()
    
    if not force and (now - _LAST_MAINTENANCE) < _MAINTENANCE_INTERVAL:
        logger.debug("Maintenance throttled - last run %.1f s ago", 
                    now - _LAST_MAINTENANCE)
        return

    logger.debug("Running registry maintenance")
    try:
        cleanup_stale(now)
        compact()
        _LAST_MAINTENANCE = now
    except Exception as exc:
        logger.exception("Registry maintenance failed")
        raise RegistryError("Maintenance error") from exc


def remove_by_name(name: str) -> bool:
    """
    Remove all entries with the specified name.
    Returns True if one or more entries were removed.
    """
    with _REGISTRY_LOCK:
        items = _load()
        original_count = len(items)
        
        # Filter out entries with matching name
        filtered_items = [
            item for item in items
            if item.get("name") != name
        ]
        
        removed = original_count != len(filtered_items)
        if removed:
            _save(filtered_items)
            logger.info("Removed %d entries for name=%s", 
                       original_count - len(filtered_items), name)
        else:
            logger.debug("No entries found for remove_by_name(%s)", name)
    
    return removed

