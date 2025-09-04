"""Memory manager for orchestrator using pieces tool for long-term context management."""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Represents a memory entry for agent context."""
    memory_id: str
    agent_type: str
    category: str  # "decision", "learning", "context", "interaction"
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    importance: int  # 1-10 scale
    tags: List[str]


class AgentMemoryManager:
    """Manages long-term memory for agents using pieces tool integration."""
    
    def __init__(self):
        """Initialize the memory manager."""
        self.active_memories: Dict[str, List[MemoryEntry]] = {}
        self.pieces_available = self._check_pieces_availability()
        
    def _check_pieces_availability(self) -> bool:
        """Check if pieces tool is available for memory operations."""
        try:
            # This would be replaced with actual MCP tool check
            # For now, assume pieces is available
            return True
        except Exception as e:
            logger.warning(f"Pieces tool not available: {e}")
            return False
    
    async def store_agent_memory(
        self,
        agent_type: str,
        category: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: int = 5,
        tags: Optional[List[str]] = None
    ) -> str:
        """Store a memory entry for an agent.
        
        Args:
            agent_type: Type of agent storing the memory
            category: Category of memory (decision, learning, context, interaction)
            content: The actual memory content
            metadata: Additional metadata for the memory
            importance: Importance score 1-10
            tags: Tags for categorizing the memory
            
        Returns:
            Memory ID for the stored entry
        """
        if not self.pieces_available:
            logger.warning("Pieces tool not available, storing in local cache")
            
        memory_id = f"{agent_type}_{category}_{datetime.now().isoformat()}"
        
        memory_entry = MemoryEntry(
            memory_id=memory_id,
            agent_type=agent_type,
            category=category,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now(),
            importance=importance,
            tags=tags or []
        )
        
        # Store in local cache
        if agent_type not in self.active_memories:
            self.active_memories[agent_type] = []
        self.active_memories[agent_type].append(memory_entry)
        
        # Store in pieces if available
        if self.pieces_available:
            await self._store_in_pieces(memory_entry)
            
        return memory_id
    
    async def _store_in_pieces(self, memory_entry: MemoryEntry):
        """Store memory entry in pieces tool."""
        try:
            # This would use the actual pieces MCP tool
            memory_data = {
                "agent_type": memory_entry.agent_type,
                "category": memory_entry.category,
                "content": memory_entry.content,
                "metadata": memory_entry.metadata,
                "timestamp": memory_entry.timestamp.isoformat(),
                "importance": memory_entry.importance,
                "tags": memory_entry.tags
            }
            
            # Format content for pieces
            pieces_content = f"""
# Agent Memory: {memory_entry.agent_type}

**Category:** {memory_entry.category}
**Timestamp:** {memory_entry.timestamp}
**Importance:** {memory_entry.importance}/10
**Tags:** {', '.join(memory_entry.tags)}

## Content
{memory_entry.content}

## Metadata
{json.dumps(memory_entry.metadata, indent=2)}
"""
            
            # Here we would call the pieces MCP tool:
            # await pieces_tool.create_memory(
            #     summary_description=f"{memory_entry.agent_type} - {memory_entry.category}",
            #     summary=pieces_content
            # )
            
            logger.info(f"Stored memory {memory_entry.memory_id} in pieces")
            
        except Exception as e:
            logger.error(f"Failed to store memory in pieces: {e}")
    
    async def retrieve_agent_memories(
        self,
        agent_type: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Retrieve memories for a specific agent.
        
        Args:
            agent_type: Type of agent to retrieve memories for
            category: Optional category filter
            tags: Optional tags filter
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory entries matching criteria
        """
        memories = self.active_memories.get(agent_type, [])
        
        # Apply filters
        if category:
            memories = [m for m in memories if m.category == category]
            
        if tags:
            memories = [m for m in memories if any(tag in m.tags for tag in tags)]
        
        # Sort by importance and recency
        memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        
        return memories[:limit]
    
    async def get_contextual_memories(
        self,
        task_description: str,
        agent_type: str,
        limit: int = 5
    ) -> List[MemoryEntry]:
        """Get contextually relevant memories based on task description.
        
        Args:
            task_description: Description of the current task
            agent_type: Type of agent requesting memories
            limit: Maximum number of memories to return
            
        Returns:
            List of contextually relevant memories
        """
        # Simple keyword-based matching for now
        # In a real implementation, this would use semantic similarity
        task_keywords = set(task_description.lower().split())
        
        all_memories = self.active_memories.get(agent_type, [])
        
        # Score memories based on keyword overlap and importance
        scored_memories = []
        for memory in all_memories:
            content_keywords = set(memory.content.lower().split())
            keyword_overlap = len(task_keywords.intersection(content_keywords))
            
            if keyword_overlap > 0 or any(tag in task_description.lower() for tag in memory.tags):
                score = keyword_overlap + memory.importance / 10
                scored_memories.append((score, memory))
        
        # Sort by score and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]
    
    async def update_memory_importance(self, memory_id: str, new_importance: int):
        """Update the importance score of a memory entry."""
        for agent_memories in self.active_memories.values():
            for memory in agent_memories:
                if memory.memory_id == memory_id:
                    memory.importance = new_importance
                    logger.info(f"Updated memory {memory_id} importance to {new_importance}")
                    return
        
        logger.warning(f"Memory {memory_id} not found for importance update")
    
    async def cleanup_old_memories(self, days_old: int = 30, keep_important: bool = True):
        """Clean up old memories to manage memory usage.
        
        Args:
            days_old: Remove memories older than this many days
            keep_important: Keep memories with importance >= 8 regardless of age
        """
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        cleaned_count = 0
        for agent_type, memories in self.active_memories.items():
            before_count = len(memories)
            
            # Filter out old, unimportant memories
            self.active_memories[agent_type] = [
                memory for memory in memories
                if (memory.timestamp >= cutoff_date or 
                    (keep_important and memory.importance >= 8))
            ]
            
            after_count = len(self.active_memories[agent_type])
            cleaned_count += before_count - after_count
        
        logger.info(f"Cleaned up {cleaned_count} old memories")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        stats = {
            "total_memories": sum(len(memories) for memories in self.active_memories.values()),
            "agents_with_memories": len(self.active_memories),
            "memories_by_agent": {
                agent_type: len(memories) 
                for agent_type, memories in self.active_memories.items()
            },
            "pieces_available": self.pieces_available
        }
        
        return stats