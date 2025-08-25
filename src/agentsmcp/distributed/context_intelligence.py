"""
Enhanced Context Intelligence

Provides intelligent context management with dynamic prioritization, semantic understanding,
and efficient memory utilization for multi-agent systems.

Key features:
- Semantic similarity analysis for context relevance
- Dynamic priority-based context compression
- Intelligent memory management with forgetting curves
- Cross-agent context sharing and optimization  
- Context embedding and retrieval systems
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
import json
import uuid
import hashlib
import math
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ContextPriority(Enum):
    """Priority levels for context information."""
    CRITICAL = 1.0      # Essential for current operations
    HIGH = 0.8         # Important for decision making
    MEDIUM = 0.6       # Useful background information
    LOW = 0.4          # Nice to have context
    MINIMAL = 0.2      # Can be compressed or discarded


class ContextType(Enum):
    """Types of context information."""
    TASK_STATE = "task_state"
    USER_INTENT = "user_intent"
    EXECUTION_HISTORY = "execution_history"
    AGENT_CAPABILITY = "agent_capability"
    ERROR_CONTEXT = "error_context"
    PERFORMANCE_METRICS = "performance_metrics"
    COLLABORATION_STATE = "collaboration_state"
    DOMAIN_KNOWLEDGE = "domain_knowledge"


@dataclass
class ContextItem:
    """Individual context item with metadata."""
    item_id: str = field(default_factory=lambda: f"ctx_{uuid.uuid4().hex[:8]}")
    content: Any = None
    content_type: ContextType = ContextType.TASK_STATE
    priority: ContextPriority = ContextPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    relevance_score: float = 0.5
    semantic_embedding: Optional[List[float]] = None
    token_estimate: int = 0
    agent_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    expiry_time: Optional[datetime] = None
    compression_level: int = 0  # 0 = uncompressed, higher = more compressed


@dataclass
class ContextBudget:
    """Context budget configuration."""
    total_tokens: int = 64000
    reserved_tokens: int = 8000  # Reserved for system operations
    priority_allocation: Dict[ContextPriority, float] = field(default_factory=lambda: {
        ContextPriority.CRITICAL: 0.4,    # 40% for critical context
        ContextPriority.HIGH: 0.3,        # 30% for high priority
        ContextPriority.MEDIUM: 0.2,      # 20% for medium priority  
        ContextPriority.LOW: 0.08,        # 8% for low priority
        ContextPriority.MINIMAL: 0.02     # 2% for minimal priority
    })


class ContextIntelligenceEngine:
    """
    Advanced context management with semantic understanding and optimization.
    
    Features:
    - Semantic similarity for relevance scoring
    - Dynamic context prioritization based on usage patterns
    - Intelligent compression and summarization
    - Cross-agent context sharing
    - Forgetting curves for memory optimization
    """
    
    def __init__(self, context_budget: ContextBudget = None):
        self.context_budget = context_budget or ContextBudget()
        self.context_store: Dict[str, ContextItem] = {}
        self.agent_contexts: Dict[str, Set[str]] = defaultdict(set)
        self.semantic_index: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.priority_queues: Dict[ContextPriority, deque] = {
            priority: deque() for priority in ContextPriority
        }
        
        # Context analytics
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.compression_stats: Dict[str, int] = defaultdict(int)
        self.sharing_network: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Memory management
        self.forgetting_curve_decay = 0.1  # Rate of memory decay
        self.similarity_threshold = 0.7    # Threshold for semantic similarity
        
        # Background optimization
        asyncio.create_task(self._context_optimization_loop())
        asyncio.create_task(self._memory_cleanup_loop())
    
    async def add_context(self, 
                         content: Any, 
                         agent_id: str,
                         content_type: ContextType = ContextType.TASK_STATE,
                         priority: ContextPriority = None,
                         tags: Set[str] = None,
                         expiry_minutes: int = None) -> str:
        """Add new context with intelligent prioritization."""
        
        # Estimate token usage
        content_str = json.dumps(content) if not isinstance(content, str) else content
        token_estimate = len(content_str) // 4  # Rough token estimate
        
        # Auto-determine priority if not specified
        if priority is None:
            priority = await self._determine_context_priority(content, content_type, agent_id)
        
        # Create context item
        context_item = ContextItem(
            content=content,
            content_type=content_type,
            priority=priority,
            token_estimate=token_estimate,
            agent_id=agent_id,
            tags=tags or set(),
            expiry_time=datetime.utcnow() + timedelta(minutes=expiry_minutes) if expiry_minutes else None
        )
        
        # Generate semantic embedding for similarity analysis
        context_item.semantic_embedding = await self._generate_embedding(content_str)
        
        # Check if context budget allows addition
        if not await self._check_context_budget(context_item):
            # Need to make room - trigger compression/cleanup
            await self._optimize_context_for_new_item(context_item)
        
        # Store context
        self.context_store[context_item.item_id] = context_item
        self.agent_contexts[agent_id].add(context_item.item_id)
        self.priority_queues[priority].append(context_item.item_id)
        
        # Update semantic index
        await self._update_semantic_index(context_item)
        
        logger.info(f"🧠 Context added: {context_item.item_id} ({priority.name}, {token_estimate} tokens)")
        return context_item.item_id
    
    async def get_relevant_context(self, 
                                 query: str, 
                                 agent_id: str,
                                 max_tokens: int = None,
                                 include_types: List[ContextType] = None) -> List[ContextItem]:
        """Retrieve most relevant context for a query with intelligent ranking."""
        
        max_tokens = max_tokens or (self.context_budget.total_tokens // 4)
        query_embedding = await self._generate_embedding(query)
        
        # Get candidate context items
        candidates = []
        
        # Include agent's own context
        agent_context_ids = self.agent_contexts.get(agent_id, set())
        for ctx_id in agent_context_ids:
            if ctx_id in self.context_store:
                candidates.append(self.context_store[ctx_id])
        
        # Include shared/global context that might be relevant
        for ctx_id, ctx_item in self.context_store.items():
            if (ctx_item.agent_id != agent_id and 
                (include_types is None or ctx_item.content_type in include_types)):
                candidates.append(ctx_item)
        
        # Calculate relevance scores
        scored_candidates = []
        for ctx_item in candidates:
            relevance_score = await self._calculate_relevance_score(
                ctx_item, query_embedding, agent_id
            )
            scored_candidates.append((ctx_item, relevance_score))
        
        # Sort by relevance and priority
        scored_candidates.sort(key=lambda x: (x[1], x[0].priority.value), reverse=True)
        
        # Select context within token budget
        selected_context = []
        total_tokens = 0
        
        for ctx_item, score in scored_candidates:
            if total_tokens + ctx_item.token_estimate <= max_tokens:
                selected_context.append(ctx_item)
                total_tokens += ctx_item.token_estimate
                
                # Update access patterns
                ctx_item.last_accessed = datetime.utcnow()
                ctx_item.access_count += 1
                self.access_patterns[ctx_item.item_id].append(datetime.utcnow())
            else:
                break
        
        logger.info(f"🎯 Retrieved {len(selected_context)} context items ({total_tokens} tokens) for query")
        return selected_context
    
    async def _determine_context_priority(self, 
                                        content: Any, 
                                        content_type: ContextType, 
                                        agent_id: str) -> ContextPriority:
        """Automatically determine context priority based on content and patterns."""
        
        content_str = json.dumps(content) if not isinstance(content, str) else content
        content_lower = content_str.lower()
        
        # Type-based priority
        type_priorities = {
            ContextType.USER_INTENT: ContextPriority.CRITICAL,
            ContextType.ERROR_CONTEXT: ContextPriority.HIGH,
            ContextType.TASK_STATE: ContextPriority.HIGH,
            ContextType.COLLABORATION_STATE: ContextPriority.MEDIUM,
            ContextType.PERFORMANCE_METRICS: ContextPriority.LOW,
            ContextType.DOMAIN_KNOWLEDGE: ContextPriority.MEDIUM
        }
        
        base_priority = type_priorities.get(content_type, ContextPriority.MEDIUM)
        
        # Content-based priority adjustments
        priority_boost = 0
        priority_penalty = 0
        
        # High priority indicators
        if any(keyword in content_lower for keyword in 
               ['error', 'critical', 'urgent', 'deadline', 'failure']):
            priority_boost += 0.2
        
        if any(keyword in content_lower for keyword in 
               ['user', 'customer', 'requirement', 'goal', 'objective']):
            priority_boost += 0.15
        
        # Low priority indicators  
        if any(keyword in content_lower for keyword in 
               ['debug', 'verbose', 'trace', 'info']):
            priority_penalty += 0.1
        
        if len(content_str) > 5000:  # Very long content might be less critical
            priority_penalty += 0.05
        
        # Agent usage patterns
        agent_access_history = self.access_patterns.get(agent_id, [])
        if len(agent_access_history) > 10:  # Frequently accessed agent
            priority_boost += 0.05
        
        # Calculate final priority
        final_score = base_priority.value + priority_boost - priority_penalty
        final_score = max(0.2, min(1.0, final_score))  # Clamp to valid range
        
        # Map back to priority enum
        if final_score >= 0.9:
            return ContextPriority.CRITICAL
        elif final_score >= 0.7:
            return ContextPriority.HIGH
        elif final_score >= 0.5:
            return ContextPriority.MEDIUM
        elif final_score >= 0.3:
            return ContextPriority.LOW
        else:
            return ContextPriority.MINIMAL\n    \n    async def _generate_embedding(self, text: str) -> List[float]:\n        \"\"\"Generate semantic embedding for text (simplified implementation).\"\"\"\n        # Simplified embedding generation using text characteristics\n        # In production, you'd use a proper embedding model like sentence-transformers\n        \n        words = text.lower().split()\n        if not words:\n            return [0.0] * 128  # Default empty embedding\n        \n        # Create feature vector based on text properties\n        features = [0.0] * 128\n        \n        # Word frequency features\n        word_freq = defaultdict(int)\n        for word in words:\n            word_freq[word] += 1\n        \n        # Length features\n        features[0] = min(1.0, len(text) / 1000)  # Text length\n        features[1] = min(1.0, len(words) / 100)  # Word count\n        features[2] = sum(len(word) for word in words) / len(words) if words else 0  # Avg word length\n        \n        # Content type indicators (simple keyword matching)\n        technical_keywords = ['function', 'class', 'method', 'api', 'database', 'server']\n        business_keywords = ['user', 'customer', 'requirement', 'goal', 'revenue', 'cost']\n        error_keywords = ['error', 'exception', 'failed', 'timeout', 'crashed']\n        \n        features[3] = sum(1 for word in words if word in technical_keywords) / len(words)\n        features[4] = sum(1 for word in words if word in business_keywords) / len(words)\n        features[5] = sum(1 for word in words if word in error_keywords) / len(words)\n        \n        # Simple hash-based features for the rest\n        for i, word in enumerate(words[:50]):  # Use first 50 words\n            hash_val = abs(hash(word)) % 123  # Map to remaining feature space\n            if hash_val < len(features) - 10:\n                features[hash_val + 10] = min(1.0, features[hash_val + 10] + 0.1)\n        \n        return features\n    \n    async def _calculate_relevance_score(self, \n                                       ctx_item: ContextItem, \n                                       query_embedding: List[float],\n                                       agent_id: str) -> float:\n        \"\"\"Calculate relevance score for context item against query.\"\"\"\n        \n        # Semantic similarity (cosine similarity)\n        if ctx_item.semantic_embedding and query_embedding:\n            dot_product = sum(a * b for a, b in zip(ctx_item.semantic_embedding, query_embedding))\n            norm_a = math.sqrt(sum(a * a for a in ctx_item.semantic_embedding))\n            norm_b = math.sqrt(sum(b * b for b in query_embedding))\n            semantic_similarity = dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0\n        else:\n            semantic_similarity = 0.5  # Default similarity\n        \n        # Recency bonus (more recent = more relevant)\n        time_diff = datetime.utcnow() - ctx_item.last_accessed\n        recency_score = math.exp(-time_diff.total_seconds() / 3600)  # Decay over hours\n        \n        # Access frequency bonus\n        frequency_score = min(1.0, ctx_item.access_count / 10)  # Normalize to 0-1\n        \n        # Agent relationship bonus\n        agent_bonus = 0.2 if ctx_item.agent_id == agent_id else 0.0\n        \n        # Priority weight\n        priority_weight = ctx_item.priority.value\n        \n        # Forgetting curve (items naturally lose relevance over time)\n        age_hours = (datetime.utcnow() - ctx_item.created_at).total_seconds() / 3600\n        forgetting_factor = math.exp(-self.forgetting_curve_decay * age_hours)\n        \n        # Combined relevance score\n        relevance_score = (\n            semantic_similarity * 0.4 +\n            recency_score * 0.2 +\n            frequency_score * 0.15 +\n            priority_weight * 0.15 +\n            agent_bonus * 0.1\n        ) * forgetting_factor\n        \n        return max(0.0, min(1.0, relevance_score))\n    \n    async def _check_context_budget(self, new_item: ContextItem) -> bool:\n        \"\"\"Check if adding new context item fits within budget.\"\"\"\n        current_usage = sum(item.token_estimate for item in self.context_store.values())\n        available_budget = self.context_budget.total_tokens - self.context_budget.reserved_tokens\n        \n        return (current_usage + new_item.token_estimate) <= available_budget\n    \n    async def _optimize_context_for_new_item(self, new_item: ContextItem):\n        \"\"\"Optimize context storage to make room for new item.\"\"\"\n        \n        # Calculate how much space we need\n        current_usage = sum(item.token_estimate for item in self.context_store.values())\n        available_budget = self.context_budget.total_tokens - self.context_budget.reserved_tokens\n        needed_space = current_usage + new_item.token_estimate - available_budget\n        \n        if needed_space <= 0:\n            return  # No optimization needed\n        \n        logger.info(f\"🗜️ Optimizing context to free {needed_space} tokens\")\n        \n        # Strategy 1: Remove expired items\n        expired_items = [\n            item_id for item_id, item in self.context_store.items()\n            if item.expiry_time and datetime.utcnow() > item.expiry_time\n        ]\n        \n        for item_id in expired_items:\n            await self._remove_context_item(item_id)\n            needed_space -= self.context_store.get(item_id, ContextItem()).token_estimate\n            if needed_space <= 0:\n                return\n        \n        # Strategy 2: Compress low priority items\n        low_priority_items = [\n            (item_id, item) for item_id, item in self.context_store.items()\n            if item.priority in [ContextPriority.LOW, ContextPriority.MINIMAL] and item.compression_level == 0\n        ]\n        \n        # Sort by lowest relevance first\n        low_priority_items.sort(key=lambda x: (x[1].access_count, x[1].last_accessed))\n        \n        for item_id, item in low_priority_items:\n            if needed_space <= 0:\n                break\n            \n            original_tokens = item.token_estimate\n            await self._compress_context_item(item)\n            tokens_saved = original_tokens - item.token_estimate\n            needed_space -= tokens_saved\n        \n        # Strategy 3: Remove least relevant items as last resort\n        if needed_space > 0:\n            all_items = list(self.context_store.items())\n            # Calculate relevance scores for removal candidates\n            removal_candidates = []\n            \n            for item_id, item in all_items:\n                if item.priority != ContextPriority.CRITICAL:  # Don't remove critical items\n                    # Simple relevance score for removal\n                    age_penalty = (datetime.utcnow() - item.last_accessed).total_seconds() / 3600\n                    removal_score = item.access_count - age_penalty * 0.1\n                    removal_candidates.append((item_id, removal_score, item.token_estimate))\n            \n            # Sort by removal score (lowest first)\n            removal_candidates.sort(key=lambda x: x[1])\n            \n            for item_id, score, tokens in removal_candidates:\n                if needed_space <= 0:\n                    break\n                \n                await self._remove_context_item(item_id)\n                needed_space -= tokens\n                \n        logger.info(f\"📊 Context optimization completed - freed space for new item\")\n    \n    async def _compress_context_item(self, item: ContextItem):\n        \"\"\"Compress context item to save tokens.\"\"\"\n        if item.compression_level >= 3:  # Already heavily compressed\n            return\n        \n        content_str = json.dumps(item.content) if not isinstance(item.content, str) else item.content\n        \n        # Simple compression strategies\n        if item.compression_level == 0:\n            # Level 1: Remove extra whitespace, shorten field names\n            compressed = content_str.replace('  ', ' ').replace('\\n\\n', '\\n')\n            item.content = compressed\n            item.compression_level = 1\n            \n        elif item.compression_level == 1:\n            # Level 2: Keep only essential information\n            if isinstance(item.content, dict):\n                essential_keys = ['id', 'status', 'result', 'error', 'cost']\n                compressed_content = {k: v for k, v in item.content.items() if k in essential_keys}\n                item.content = compressed_content\n                item.compression_level = 2\n            \n        elif item.compression_level == 2:\n            # Level 3: Create summary\n            summary = f\"Compressed context: {item.content_type.value} from {item.created_at.strftime('%H:%M')}\"\n            if isinstance(item.content, dict) and 'status' in item.content:\n                summary += f\" status: {item.content['status']}\"\n            item.content = summary\n            item.compression_level = 3\n        \n        # Recalculate token estimate\n        new_content_str = json.dumps(item.content) if not isinstance(item.content, str) else item.content\n        old_tokens = item.token_estimate\n        item.token_estimate = len(new_content_str) // 4\n        \n        self.compression_stats[f\"level_{item.compression_level}\"] += 1\n        \n        logger.debug(f\"🗜️ Compressed context {item.item_id}: {old_tokens} -> {item.token_estimate} tokens\")\n    \n    async def _remove_context_item(self, item_id: str):\n        \"\"\"Remove context item from all storage.\"\"\"\n        if item_id not in self.context_store:\n            return\n        \n        item = self.context_store[item_id]\n        \n        # Remove from main storage\n        del self.context_store[item_id]\n        \n        # Remove from agent contexts\n        if item.agent_id and item_id in self.agent_contexts.get(item.agent_id, set()):\n            self.agent_contexts[item.agent_id].remove(item_id)\n        \n        # Remove from priority queues\n        try:\n            self.priority_queues[item.priority].remove(item_id)\n        except ValueError:\n            pass  # Item not in queue\n        \n        # Remove from access patterns\n        if item_id in self.access_patterns:\n            del self.access_patterns[item_id]\n        \n        logger.debug(f\"🗑️ Removed context item: {item_id}\")\n    \n    async def _update_semantic_index(self, context_item: ContextItem):\n        \"\"\"Update semantic index for fast similarity searches.\"\"\"\n        if not context_item.semantic_embedding:\n            return\n        \n        # Find similar existing items\n        similar_items = []\n        for existing_id, existing_item in self.context_store.items():\n            if existing_id == context_item.item_id or not existing_item.semantic_embedding:\n                continue\n            \n            similarity = await self._cosine_similarity(\n                context_item.semantic_embedding, \n                existing_item.semantic_embedding\n            )\n            \n            if similarity > self.similarity_threshold:\n                similar_items.append((existing_id, similarity))\n        \n        # Update semantic index\n        content_hash = hashlib.md5(\n            json.dumps(context_item.content, sort_keys=True).encode()\n        ).hexdigest()[:8]\n        \n        self.semantic_index[content_hash] = similar_items\n    \n    async def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:\n        \"\"\"Calculate cosine similarity between two vectors.\"\"\"\n        if len(vec1) != len(vec2):\n            return 0.0\n        \n        dot_product = sum(a * b for a, b in zip(vec1, vec2))\n        norm_a = math.sqrt(sum(a * a for a in vec1))\n        norm_b = math.sqrt(sum(b * b for b in vec2))\n        \n        if norm_a == 0 or norm_b == 0:\n            return 0.0\n        \n        return dot_product / (norm_a * norm_b)\n    \n    async def share_context_between_agents(self, \n                                         from_agent: str, \n                                         to_agent: str, \n                                         context_ids: List[str]) -> int:\n        \"\"\"Share context between agents with intelligent filtering.\"\"\"\n        shared_count = 0\n        \n        for ctx_id in context_ids:\n            if ctx_id not in self.context_store:\n                continue\n            \n            ctx_item = self.context_store[ctx_id]\n            \n            # Check if context is appropriate for sharing\n            if await self._should_share_context(ctx_item, from_agent, to_agent):\n                # Add to target agent's context\n                self.agent_contexts[to_agent].add(ctx_id)\n                \n                # Update sharing network\n                self.sharing_network[from_agent][to_agent] += 1\n                \n                # Boost priority slightly for shared context\n                if ctx_item.priority != ContextPriority.CRITICAL:\n                    priority_values = list(ContextPriority)\n                    current_idx = priority_values.index(ctx_item.priority)\n                    if current_idx > 0:\n                        ctx_item.priority = priority_values[current_idx - 1]\n                \n                shared_count += 1\n        \n        logger.info(f\"🤝 Shared {shared_count} context items: {from_agent} -> {to_agent}\")\n        return shared_count\n    \n    async def _should_share_context(self, \n                                  ctx_item: ContextItem, \n                                  from_agent: str, \n                                  to_agent: str) -> bool:\n        \"\"\"Determine if context should be shared between agents.\"\"\"\n        \n        # Don't share very low priority context\n        if ctx_item.priority == ContextPriority.MINIMAL:\n            return False\n        \n        # Don't share expired context\n        if ctx_item.expiry_time and datetime.utcnow() > ctx_item.expiry_time:\n            return False\n        \n        # Check if agents have collaborated before (higher sharing likelihood)\n        collaboration_history = self.sharing_network[from_agent].get(to_agent, 0)\n        if collaboration_history > 5:  # Frequent collaborators\n            return True\n        \n        # Share high priority or recently accessed context\n        if (ctx_item.priority in [ContextPriority.CRITICAL, ContextPriority.HIGH] or\n            ctx_item.access_count > 3):\n            return True\n        \n        # Don't share by default for privacy/security\n        return False\n    \n    async def _context_optimization_loop(self):\n        \"\"\"Background loop for context optimization.\"\"\"\n        while True:\n            try:\n                await self._optimize_context_storage()\n                await self._update_priority_scores()\n                await self._analyze_access_patterns()\n            except Exception as e:\n                logger.error(f\"Context optimization error: {e}\")\n            \n            await asyncio.sleep(300)  # Optimize every 5 minutes\n    \n    async def _memory_cleanup_loop(self):\n        \"\"\"Background loop for memory cleanup.\"\"\"\n        while True:\n            try:\n                await self._cleanup_expired_context()\n                await self._apply_forgetting_curve()\n            except Exception as e:\n                logger.error(f\"Memory cleanup error: {e}\")\n            \n            await asyncio.sleep(600)  # Cleanup every 10 minutes\n    \n    async def _optimize_context_storage(self):\n        \"\"\"Optimize overall context storage efficiency.\"\"\"\n        current_usage = sum(item.token_estimate for item in self.context_store.values())\n        target_usage = self.context_budget.total_tokens * 0.8  # Aim for 80% utilization\n        \n        if current_usage > target_usage:\n            # Need to compress or remove items\n            excess = current_usage - target_usage\n            \n            # Find compression candidates\n            compression_candidates = [\n                item for item in self.context_store.values()\n                if (item.compression_level < 2 and \n                    item.priority in [ContextPriority.LOW, ContextPriority.MINIMAL] and\n                    item.access_count < 3)\n            ]\n            \n            for item in compression_candidates[:10]:  # Compress up to 10 items\n                await self._compress_context_item(item)\n                if excess <= 0:\n                    break\n    \n    async def _update_priority_scores(self):\n        \"\"\"Update priority scores based on access patterns.\"\"\"\n        for item in self.context_store.values():\n            # Calculate new relevance score based on recent access\n            recent_access = [\n                access_time for access_time in self.access_patterns.get(item.item_id, [])\n                if (datetime.utcnow() - access_time).days <= 7\n            ]\n            \n            if len(recent_access) > 5:  # Highly accessed recently\n                # Boost priority if not already critical\n                if item.priority not in [ContextPriority.CRITICAL, ContextPriority.HIGH]:\n                    priority_values = list(ContextPriority)\n                    current_idx = priority_values.index(item.priority)\n                    if current_idx > 0:\n                        item.priority = priority_values[current_idx - 1]\n            \n            elif len(recent_access) == 0 and item.access_count < 2:\n                # Rarely accessed, consider lowering priority\n                if item.priority not in [ContextPriority.MINIMAL]:\n                    priority_values = list(ContextPriority)\n                    current_idx = priority_values.index(item.priority)\n                    if current_idx < len(priority_values) - 1:\n                        item.priority = priority_values[current_idx + 1]\n    \n    async def _analyze_access_patterns(self):\n        \"\"\"Analyze context access patterns for insights.\"\"\"\n        # Find frequently accessed context types\n        type_access_counts = defaultdict(int)\n        for item in self.context_store.values():\n            type_access_counts[item.content_type] += item.access_count\n        \n        # Log insights\n        most_accessed_type = max(type_access_counts, key=type_access_counts.get) if type_access_counts else None\n        if most_accessed_type:\n            logger.info(f\"📊 Most accessed context type: {most_accessed_type.value}\")\n    \n    async def _cleanup_expired_context(self):\n        \"\"\"Remove expired context items.\"\"\"\n        expired_items = [\n            item_id for item_id, item in self.context_store.items()\n            if item.expiry_time and datetime.utcnow() > item.expiry_time\n        ]\n        \n        for item_id in expired_items:\n            await self._remove_context_item(item_id)\n        \n        if expired_items:\n            logger.info(f\"🧹 Cleaned up {len(expired_items)} expired context items\")\n    \n    async def _apply_forgetting_curve(self):\n        \"\"\"Apply forgetting curve to reduce relevance of old unused context.\"\"\"\n        current_time = datetime.utcnow()\n        \n        for item in self.context_store.values():\n            # Apply forgetting curve based on last access time\n            time_since_access = current_time - item.last_accessed\n            hours_since_access = time_since_access.total_seconds() / 3600\n            \n            # Exponential decay\n            forgetting_factor = math.exp(-self.forgetting_curve_decay * hours_since_access)\n            item.relevance_score = item.relevance_score * forgetting_factor\n            \n            # If relevance drops too low, consider for removal\n            if (item.relevance_score < 0.1 and \n                item.priority in [ContextPriority.LOW, ContextPriority.MINIMAL] and\n                hours_since_access > 48):  # Not accessed for 2+ days\n                \n                await self._remove_context_item(item.item_id)\n    \n    def get_context_analytics(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive context intelligence analytics.\"\"\"\n        total_items = len(self.context_store)\n        total_tokens = sum(item.token_estimate for item in self.context_store.values())\n        \n        # Priority distribution\n        priority_distribution = defaultdict(int)\n        for item in self.context_store.values():\n            priority_distribution[item.priority.name] += 1\n        \n        # Type distribution\n        type_distribution = defaultdict(int)\n        for item in self.context_store.values():\n            type_distribution[item.content_type.value] += 1\n        \n        # Compression stats\n        compression_distribution = defaultdict(int)\n        for item in self.context_store.values():\n            compression_distribution[f\"level_{item.compression_level}\"] += 1\n        \n        # Agent distribution\n        agent_context_counts = {agent: len(contexts) for agent, contexts in self.agent_contexts.items()}\n        \n        # Budget utilization\n        budget_utilization = {\n            \"total_budget\": self.context_budget.total_tokens,\n            \"reserved_tokens\": self.context_budget.reserved_tokens,\n            \"used_tokens\": total_tokens,\n            \"available_tokens\": self.context_budget.total_tokens - self.context_budget.reserved_tokens - total_tokens,\n            \"utilization_percentage\": (total_tokens / self.context_budget.total_tokens) * 100\n        }\n        \n        return {\n            \"total_context_items\": total_items,\n            \"total_tokens_used\": total_tokens,\n            \"priority_distribution\": dict(priority_distribution),\n            \"type_distribution\": dict(type_distribution),\n            \"compression_distribution\": dict(compression_distribution),\n            \"agent_context_counts\": agent_context_counts,\n            \"budget_utilization\": budget_utilization,\n            \"sharing_network_size\": len(self.sharing_network),\n            \"semantic_index_size\": len(self.semantic_index),\n            \"average_access_count\": sum(item.access_count for item in self.context_store.values()) / total_items if total_items > 0 else 0,\n            \"compression_stats\": dict(self.compression_stats)\n        }"