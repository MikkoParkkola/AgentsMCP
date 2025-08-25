"""
Enhanced Context Intelligence Engine for AgentsMCP.
Provides intelligent context management with dynamic prioritization, 
semantic understanding, and adaptive memory systems.
"""

import asyncio
import hashlib
import json
import math
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

import logging
logger = logging.getLogger(__name__)


class ContextPriority(Enum):
    """Context priority levels."""
    MINIMAL = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    CRITICAL = 0.9


class ContextType(Enum):
    """Types of context content."""
    TASK_RESULT = "task_result"
    TASK_STATE = "task_state"
    ERROR_LOG = "error_log"
    PERFORMANCE_METRIC = "performance_metric"
    USER_INTERACTION = "user_interaction"
    AGENT_COMMUNICATION = "agent_communication"
    SYSTEM_STATE = "system_state"
    LEARNING_INSIGHT = "learning_insight"


@dataclass
class ContextBudget:
    """Context budget management."""
    total_tokens: int = 50000
    reserved_tokens: int = 10000  # Reserved for critical context
    warning_threshold: float = 0.8  # Warn when 80% full


@dataclass
class ContextItem:
    """Individual context item with metadata."""
    item_id: str
    content: Any
    content_type: ContextType
    agent_id: str
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    priority: ContextPriority = ContextPriority.MEDIUM
    token_estimate: int = 0
    relevance_score: float = 1.0
    compression_level: int = 0  # 0=none, 1=light, 2=medium, 3=heavy
    semantic_embedding: Optional[List[float]] = None
    expiry_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextIntelligenceEngine:
    """
    Advanced context intelligence engine with dynamic prioritization,
    semantic understanding, and intelligent memory management.
    """
    
    def __init__(self, 
                 context_budget: ContextBudget = None,
                 forgetting_curve_decay: float = 0.01,
                 similarity_threshold: float = 0.7):
        
        # Core storage
        self.context_store: Dict[str, ContextItem] = {}
        self.agent_contexts: Dict[str, Set[str]] = defaultdict(set)
        
        # Prioritization and access patterns
        self.priority_queues: Dict[ContextPriority, deque] = {
            priority: deque() for priority in ContextPriority
        }
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        
        # Memory management
        self.context_budget = context_budget or ContextBudget()
        self.forgetting_curve_decay = forgetting_curve_decay
        
        # Semantic intelligence
        self.semantic_analyzer = SemanticAnalyzer()
        self.semantic_index: Dict[str, List[Tuple[str, float]]] = {}
        self.similarity_threshold = similarity_threshold
        
        # Collaboration and sharing
        self.sharing_network: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Performance tracking
        self.compression_stats: Dict[str, int] = defaultdict(int)
        
        # Background optimization
        self.optimization_task = None
        self.cleanup_task = None
        
        logger.info("🧠 Context Intelligence Engine initialized")
    
    async def start_background_tasks(self):
        """Start background optimization and cleanup tasks."""
        if not self.optimization_task:
            self.optimization_task = asyncio.create_task(self._context_optimization_loop())
        
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._memory_cleanup_loop())
        
        logger.info("🔄 Background context optimization tasks started")
    
    async def stop_background_tasks(self):
        """Stop background tasks."""
        if self.optimization_task:
            self.optimization_task.cancel()
            self.optimization_task = None
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            self.cleanup_task = None
        
        logger.info("⏹️ Background context tasks stopped")
    
    async def add_context(self, 
                         content: Any, 
                         content_type: ContextType,
                         agent_id: str,
                         priority: Optional[ContextPriority] = None,
                         expiry_hours: Optional[int] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add new context with intelligent prioritization."""
        
        # Generate unique item ID
        content_str = json.dumps(content) if not isinstance(content, str) else content
        item_id = hashlib.md5(f"{agent_id}_{datetime.utcnow().isoformat()}_{content_str[:100]}".encode()).hexdigest()[:12]
        
        # Intelligent priority determination
        if priority is None:
            priority = await self._determine_intelligent_priority(content, content_type, agent_id)
        
        # Create context item
        context_item = ContextItem(
            item_id=item_id,
            content=content,
            content_type=content_type,
            agent_id=agent_id,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            priority=priority,
            token_estimate=len(content_str) // 4,  # Rough token estimate
            semantic_embedding=await self._generate_embedding(content_str),
            expiry_time=datetime.utcnow() + timedelta(hours=expiry_hours) if expiry_hours else None,
            metadata=metadata or {}
        )
        
        # Check budget and optimize if needed
        if not await self._check_context_budget(context_item):
            await self._optimize_context_for_new_item(context_item)
        
        # Store context item
        self.context_store[item_id] = context_item
        self.agent_contexts[agent_id].add(item_id)
        self.priority_queues[priority].append(item_id)
        
        # Update semantic index
        await self._update_semantic_index(context_item)
        
        logger.debug(f"📝 Added context: {item_id} (priority: {priority.name}, tokens: {context_item.token_estimate})")
        return item_id
    
    async def retrieve_context(self, 
                             agent_id: str, 
                             query: str,
                             max_items: int = 10,
                             min_relevance: float = 0.3) -> List[ContextItem]:
        """Retrieve most relevant context for agent and query."""
        
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Get candidate context items for this agent
        candidate_ids = self.agent_contexts.get(agent_id, set())
        candidates = [self.context_store[item_id] for item_id in candidate_ids if item_id in self.context_store]
        
        # Calculate relevance scores
        scored_items = []
        for item in candidates:
            relevance_score = await self._calculate_relevance_score(item, query_embedding, agent_id)
            if relevance_score >= min_relevance:
                scored_items.append((item, relevance_score))
        
        # Sort by relevance and return top items
        scored_items.sort(key=lambda x: x[1], reverse=True)
        relevant_items = [item for item, score in scored_items[:max_items]]
        
        # Update access patterns
        for item in relevant_items:
            item.last_accessed = datetime.utcnow()
            item.access_count += 1
            self.access_patterns[item.item_id].append(datetime.utcnow())
        
        logger.debug(f"🔍 Retrieved {len(relevant_items)} context items for agent {agent_id}")
        return relevant_items
    
    async def analyze_task_context(self, task_content: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task context for intelligent agent assignment recommendations."""
        
        # Generate task embedding
        task_embedding = await self._generate_embedding(task_content)
        
        # Analyze context elements
        context_analysis = {
            'task_complexity': self._assess_task_complexity(task_content, task_context),
            'domain_indicators': self._extract_domain_indicators(task_content),
            'priority_signals': self._detect_priority_signals(task_content, task_context),
            'resource_requirements': self._estimate_resource_requirements(task_content, task_context)
        }
        
        # Find similar historical contexts
        similar_contexts = await self._find_similar_contexts(task_embedding, limit=5)
        
        # Agent recommendations based on context
        agent_recommendations = self._generate_agent_recommendations(context_analysis, similar_contexts)
        
        # Calculate overall context score
        context_score = self._calculate_context_intelligence_score(context_analysis, similar_contexts)
        
        return {
            'context_score': context_score,
            'task_complexity': context_analysis['task_complexity'],
            'domain_indicators': context_analysis['domain_indicators'],
            'recommended_agent_profiles': agent_recommendations,
            'similar_contexts': len(similar_contexts),
            'resource_requirements': context_analysis['resource_requirements'],
            'priority_adjustment': self._determine_priority_adjustment(context_analysis),
            'confidence': min(1.0, context_score * 1.2)  # Boost confidence slightly
        }
    
    def _assess_task_complexity(self, task_content: str, task_context: Dict[str, Any]) -> str:
        """Assess the complexity level of a task."""
        complexity_indicators = {
            'high': ['complex', 'advanced', 'sophisticated', 'multi-step', 'integrate', 'architecture'],
            'medium': ['analyze', 'implement', 'optimize', 'refactor', 'design'],
            'low': ['simple', 'basic', 'quick', 'fix', 'update', 'check']
        }
        
        content_lower = task_content.lower()
        scores = {}
        
        for level, keywords in complexity_indicators.items():
            scores[level] = sum(1 for keyword in keywords if keyword in content_lower)
        
        # Consider task context
        if task_context.get('project_context', {}).get('domain') in ['ai', 'ml', 'distributed_systems']:
            scores['high'] += 1
        
        if len(task_content.split()) > 100:  # Long description suggests complexity
            scores['medium'] += 1
        
        return max(scores, key=scores.get) if scores else 'medium'
    
    def _extract_domain_indicators(self, task_content: str) -> List[str]:
        """Extract domain indicators from task content."""
        domain_keywords = {
            'ai_ml': ['machine learning', 'neural', 'model', 'training', 'inference', 'ai'],
            'web_dev': ['frontend', 'backend', 'api', 'web', 'http', 'html', 'css', 'javascript'],
            'data': ['database', 'sql', 'analytics', 'data', 'etl', 'pipeline'],
            'devops': ['deployment', 'ci/cd', 'docker', 'kubernetes', 'infrastructure'],
            'security': ['security', 'authentication', 'encryption', 'vulnerability', 'audit'],
            'performance': ['performance', 'optimization', 'speed', 'latency', 'throughput']
        }
        
        content_lower = task_content.lower()
        detected_domains = []
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_domains.append(domain)
        
        return detected_domains
    
    def _detect_priority_signals(self, task_content: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect priority signals in task content and context."""
        content_lower = task_content.lower()
        
        priority_signals = {
            'urgency': any(keyword in content_lower for keyword in ['urgent', 'asap', 'critical', 'emergency']),
            'business_impact': any(keyword in content_lower for keyword in ['revenue', 'customer', 'production', 'live']),
            'deadline_mentioned': 'deadline' in content_lower or 'due' in content_lower,
            'user_facing': any(keyword in content_lower for keyword in ['user', 'client', 'customer', 'ui', 'ux'])
        }
        
        # Context-based signals
        if task_context.get('deadline') == 'urgent':
            priority_signals['context_urgent'] = True
        
        if task_context.get('user_preferences', {}).get('fast_execution', False):
            priority_signals['speed_preferred'] = True
        
        return priority_signals
    
    def _estimate_resource_requirements(self, task_content: str, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource requirements for the task."""
        content_length = len(task_content)
        word_count = len(task_content.split())
        
        # Base estimates
        estimated_tokens = max(1000, word_count * 3)  # Rough token estimation
        memory_intensive = any(keyword in task_content.lower() for keyword in 
                             ['large dataset', 'big data', 'image processing', 'video'])
        
        cpu_intensive = any(keyword in task_content.lower() for keyword in 
                          ['optimization', 'algorithm', 'computation', 'analysis'])
        
        return {
            'estimated_tokens': estimated_tokens,
            'memory_intensive': memory_intensive,
            'cpu_intensive': cpu_intensive,
            'complexity_score': min(1.0, content_length / 1000),
            'parallel_suitable': 'parallel' in task_content.lower() or 'concurrent' in task_content.lower()
        }
    
    async def _find_similar_contexts(self, query_embedding: List[float], limit: int = 5) -> List[ContextItem]:
        """Find similar contexts using semantic similarity."""
        if not query_embedding:
            return []
        
        similarity_scores = []
        
        for item_id, context_item in self.context_store.items():
            if context_item.semantic_embedding:
                similarity = await self._cosine_similarity(query_embedding, context_item.semantic_embedding)
                if similarity > self.similarity_threshold:
                    similarity_scores.append((context_item, similarity))
        
        # Sort by similarity and return top items
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in similarity_scores[:limit]]
    
    def _generate_agent_recommendations(self, context_analysis: Dict[str, Any], similar_contexts: List[ContextItem]) -> List[Dict[str, Any]]:
        """Generate agent recommendations based on context analysis."""
        recommendations = []
        
        # Base recommendations based on domain
        domain_agent_mapping = {
            'ai_ml': {'agent_type': 'claude', 'reasoning': 'complex_ai_analysis'},
            'web_dev': {'agent_type': 'codex', 'reasoning': 'web_development_expertise'},
            'data': {'agent_type': 'codex', 'reasoning': 'data_processing_capabilities'},
            'devops': {'agent_type': 'ollama', 'reasoning': 'infrastructure_management'},
            'security': {'agent_type': 'claude', 'reasoning': 'security_analysis'},
            'performance': {'agent_type': 'ollama', 'reasoning': 'optimization_focused'}
        }
        
        # Recommend based on detected domains
        for domain in context_analysis['domain_indicators']:
            if domain in domain_agent_mapping:
                recommendation = domain_agent_mapping[domain].copy()
                recommendation['match_score'] = 0.8
                recommendation['domain'] = domain
                recommendations.append(recommendation)
        
        # Cost-conscious recommendations
        if context_analysis['task_complexity'] == 'low':
            recommendations.append({
                'agent_type': 'ollama',
                'match_score': 0.75,
                'reasoning': 'cost_effective_for_simple_tasks',
                'domain': 'general'
            })
        
        # High complexity recommendations
        elif context_analysis['task_complexity'] == 'high':
            recommendations.append({
                'agent_type': 'claude',
                'match_score': 0.85,
                'reasoning': 'advanced_reasoning_for_complex_tasks',
                'domain': 'complex_analysis'
            })
        
        # Remove duplicates and sort by match score
        unique_recommendations = {rec['agent_type']: rec for rec in recommendations}
        sorted_recommendations = sorted(unique_recommendations.values(), 
                                      key=lambda x: x['match_score'], reverse=True)
        
        return sorted_recommendations[:3]  # Return top 3 recommendations
    
    def _calculate_context_intelligence_score(self, context_analysis: Dict[str, Any], similar_contexts: List[ContextItem]) -> float:
        """Calculate overall context intelligence score."""
        base_score = 0.5
        
        # Boost for domain detection
        if context_analysis['domain_indicators']:
            base_score += 0.2 * min(1.0, len(context_analysis['domain_indicators']) / 3)
        
        # Boost for priority signals
        priority_count = sum(1 for signal in context_analysis['priority_signals'].values() if signal)
        base_score += 0.15 * min(1.0, priority_count / 4)
        
        # Boost for similar contexts (learning from history)
        if similar_contexts:
            base_score += 0.25 * min(1.0, len(similar_contexts) / 5)
        
        return min(1.0, base_score)
    
    def _determine_priority_adjustment(self, context_analysis: Dict[str, Any]) -> str:
        """Determine priority adjustment based on context analysis."""
        priority_signals = context_analysis['priority_signals']
        
        # Count high-priority signals
        urgent_signals = sum(1 for key, value in priority_signals.items() 
                           if value and key in ['urgency', 'business_impact', 'deadline_mentioned'])
        
        if urgent_signals >= 2:
            return 'high'
        elif urgent_signals == 1 or priority_signals.get('user_facing', False):
            return 'medium'
        else:
            return 'low'
    
    async def _determine_intelligent_priority(self, 
                                            content: Any, 
                                            content_type: ContextType, 
                                            agent_id: str) -> ContextPriority:
        """Intelligently determine context priority based on multiple factors."""
        
        # Base priority from content type
        type_priorities = {
            ContextType.ERROR_LOG: ContextPriority.HIGH,
            ContextType.USER_INTERACTION: ContextPriority.MEDIUM,
            ContextType.PERFORMANCE_METRIC: ContextPriority.MEDIUM,
            ContextType.TASK_RESULT: ContextPriority.MEDIUM,
            ContextType.SYSTEM_STATE: ContextPriority.LOW,
            ContextType.AGENT_COMMUNICATION: ContextPriority.LOW,
            ContextType.LEARNING_INSIGHT: ContextPriority.MEDIUM
        }
        
        base_priority = type_priorities.get(content_type, ContextPriority.MEDIUM)
        
        # Content analysis for priority adjustment
        content_str = json.dumps(content) if not isinstance(content, str) else content
        content_lower = content_str.lower()
        
        priority_boost = 0.0
        priority_penalty = 0.0
        
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
            return ContextPriority.MINIMAL
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate semantic embedding for text (simplified implementation)."""
        # Simplified embedding generation using text characteristics
        # In production, you'd use a proper embedding model like sentence-transformers
        
        words = text.lower().split()
        if not words:
            return [0.0] * 128  # Default empty embedding
        
        # Create feature vector based on text properties
        features = [0.0] * 128
        
        # Word frequency features
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # Length features
        features[0] = min(1.0, len(text) / 1000)  # Text length
        features[1] = min(1.0, len(words) / 100)  # Word count
        features[2] = sum(len(word) for word in words) / len(words) if words else 0  # Avg word length
        
        # Content type indicators (simple keyword matching)
        technical_keywords = ['function', 'class', 'method', 'api', 'database', 'server']
        business_keywords = ['user', 'customer', 'requirement', 'goal', 'revenue', 'cost']
        error_keywords = ['error', 'exception', 'failed', 'timeout', 'crashed']
        
        features[3] = sum(1 for word in words if word in technical_keywords) / len(words)
        features[4] = sum(1 for word in words if word in business_keywords) / len(words)
        features[5] = sum(1 for word in words if word in error_keywords) / len(words)
        
        # Simple hash-based features for the rest
        for i, word in enumerate(words[:50]):  # Use first 50 words
            hash_val = abs(hash(word)) % 123  # Map to remaining feature space
            if hash_val < len(features) - 10:
                features[hash_val + 10] = min(1.0, features[hash_val + 10] + 0.1)
        
        return features
    
    async def _calculate_relevance_score(self, 
                                       ctx_item: ContextItem, 
                                       query_embedding: List[float],
                                       agent_id: str) -> float:
        """Calculate relevance score for context item against query."""
        
        # Semantic similarity (cosine similarity)
        if ctx_item.semantic_embedding and query_embedding:
            dot_product = sum(a * b for a, b in zip(ctx_item.semantic_embedding, query_embedding))
            norm_a = math.sqrt(sum(a * a for a in ctx_item.semantic_embedding))
            norm_b = math.sqrt(sum(b * b for b in query_embedding))
            semantic_similarity = dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0
        else:
            semantic_similarity = 0.5  # Default similarity
        
        # Recency bonus (more recent = more relevant)
        time_diff = datetime.utcnow() - ctx_item.last_accessed
        recency_score = math.exp(-time_diff.total_seconds() / 3600)  # Decay over hours
        
        # Access frequency bonus
        frequency_score = min(1.0, ctx_item.access_count / 10)  # Normalize to 0-1
        
        # Agent relationship bonus
        agent_bonus = 0.2 if ctx_item.agent_id == agent_id else 0.0
        
        # Priority weight
        priority_weight = ctx_item.priority.value
        
        # Forgetting curve (items naturally lose relevance over time)
        age_hours = (datetime.utcnow() - ctx_item.created_at).total_seconds() / 3600
        forgetting_factor = math.exp(-self.forgetting_curve_decay * age_hours)
        
        # Combined relevance score
        relevance_score = (
            semantic_similarity * 0.4 +
            recency_score * 0.2 +
            frequency_score * 0.15 +
            priority_weight * 0.15 +
            agent_bonus * 0.1
        ) * forgetting_factor
        
        return max(0.0, min(1.0, relevance_score))
    
    async def _check_context_budget(self, new_item: ContextItem) -> bool:
        """Check if adding new context item fits within budget."""
        current_usage = sum(item.token_estimate for item in self.context_store.values())
        available_budget = self.context_budget.total_tokens - self.context_budget.reserved_tokens
        
        return (current_usage + new_item.token_estimate) <= available_budget
    
    async def _optimize_context_for_new_item(self, new_item: ContextItem):
        """Optimize context storage to make room for new item."""
        
        # Calculate how much space we need
        current_usage = sum(item.token_estimate for item in self.context_store.values())
        available_budget = self.context_budget.total_tokens - self.context_budget.reserved_tokens
        needed_space = current_usage + new_item.token_estimate - available_budget
        
        if needed_space <= 0:
            return  # No optimization needed
        
        logger.info(f"🗜️ Optimizing context to free {needed_space} tokens")
        
        # Strategy 1: Remove expired items
        expired_items = [
            item_id for item_id, item in self.context_store.items()
            if item.expiry_time and datetime.utcnow() > item.expiry_time
        ]
        
        for item_id in expired_items:
            await self._remove_context_item(item_id)
            needed_space -= self.context_store.get(item_id, ContextItem()).token_estimate
            if needed_space <= 0:
                return
        
        # Strategy 2: Compress low priority items
        low_priority_items = [
            (item_id, item) for item_id, item in self.context_store.items()
            if item.priority in [ContextPriority.LOW, ContextPriority.MINIMAL] and item.compression_level == 0
        ]
        
        # Sort by lowest relevance first
        low_priority_items.sort(key=lambda x: (x[1].access_count, x[1].last_accessed))
        
        for item_id, item in low_priority_items:
            if needed_space <= 0:
                break
            
            original_tokens = item.token_estimate
            await self._compress_context_item(item)
            tokens_saved = original_tokens - item.token_estimate
            needed_space -= tokens_saved
        
        # Strategy 3: Remove least relevant items as last resort
        if needed_space > 0:
            all_items = list(self.context_store.items())
            # Calculate relevance scores for removal candidates
            removal_candidates = []
            
            for item_id, item in all_items:
                if item.priority != ContextPriority.CRITICAL:  # Don't remove critical items
                    # Simple relevance score for removal
                    age_penalty = (datetime.utcnow() - item.last_accessed).total_seconds() / 3600
                    removal_score = item.access_count - age_penalty * 0.1
                    removal_candidates.append((item_id, removal_score, item.token_estimate))
            
            # Sort by removal score (lowest first)
            removal_candidates.sort(key=lambda x: x[1])
            
            for item_id, score, tokens in removal_candidates:
                if needed_space <= 0:
                    break
                
                await self._remove_context_item(item_id)
                needed_space -= tokens
                
        logger.info(f"📊 Context optimization completed - freed space for new item")
    
    async def _compress_context_item(self, item: ContextItem):
        """Compress context item to save tokens."""
        if item.compression_level >= 3:  # Already heavily compressed
            return
        
        content_str = json.dumps(item.content) if not isinstance(item.content, str) else item.content
        
        # Simple compression strategies
        if item.compression_level == 0:
            # Level 1: Remove extra whitespace, shorten field names
            compressed = content_str.replace('  ', ' ').replace('\n\n', '\n')
            item.content = compressed
            item.compression_level = 1
            
        elif item.compression_level == 1:
            # Level 2: Keep only essential information
            if isinstance(item.content, dict):
                essential_keys = ['id', 'status', 'result', 'error', 'cost']
                compressed_content = {k: v for k, v in item.content.items() if k in essential_keys}
                item.content = compressed_content
                item.compression_level = 2
            
        elif item.compression_level == 2:
            # Level 3: Create summary
            summary = f"Compressed context: {item.content_type.value} from {item.created_at.strftime('%H:%M')}"
            if isinstance(item.content, dict) and 'status' in item.content:
                summary += f" status: {item.content['status']}"
            item.content = summary
            item.compression_level = 3
        
        # Recalculate token estimate
        new_content_str = json.dumps(item.content) if not isinstance(item.content, str) else item.content
        old_tokens = item.token_estimate
        item.token_estimate = len(new_content_str) // 4
        
        self.compression_stats[f"level_{item.compression_level}"] += 1
        
        logger.debug(f"🗜️ Compressed context {item.item_id}: {old_tokens} -> {item.token_estimate} tokens")
    
    async def _remove_context_item(self, item_id: str):
        """Remove context item from all storage."""
        if item_id not in self.context_store:
            return
        
        item = self.context_store[item_id]
        
        # Remove from main storage
        del self.context_store[item_id]
        
        # Remove from agent contexts
        if item.agent_id and item_id in self.agent_contexts.get(item.agent_id, set()):
            self.agent_contexts[item.agent_id].remove(item_id)
        
        # Remove from priority queues
        try:
            self.priority_queues[item.priority].remove(item_id)
        except ValueError:
            pass  # Item not in queue
        
        # Remove from access patterns
        if item_id in self.access_patterns:
            del self.access_patterns[item_id]
        
        logger.debug(f"🗑️ Removed context item: {item_id}")
    
    async def _update_semantic_index(self, context_item: ContextItem):
        """Update semantic index for fast similarity searches."""
        if not context_item.semantic_embedding:
            return
        
        # Find similar existing items
        similar_items = []
        for existing_id, existing_item in self.context_store.items():
            if existing_id == context_item.item_id or not existing_item.semantic_embedding:
                continue
            
            similarity = await self._cosine_similarity(
                context_item.semantic_embedding, 
                existing_item.semantic_embedding
            )
            
            if similarity > self.similarity_threshold:
                similar_items.append((existing_id, similarity))
        
        # Update semantic index
        content_hash = hashlib.md5(
            json.dumps(context_item.content, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        self.semantic_index[content_hash] = similar_items
    
    async def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def share_context_between_agents(self, 
                                         from_agent: str, 
                                         to_agent: str, 
                                         context_ids: List[str]) -> int:
        """Share context between agents with intelligent filtering."""
        shared_count = 0
        
        for ctx_id in context_ids:
            if ctx_id not in self.context_store:
                continue
            
            ctx_item = self.context_store[ctx_id]
            
            # Check if context is appropriate for sharing
            if await self._should_share_context(ctx_item, from_agent, to_agent):
                # Add to target agent's context
                self.agent_contexts[to_agent].add(ctx_id)
                
                # Update sharing network
                self.sharing_network[from_agent][to_agent] += 1
                
                # Boost priority slightly for shared context
                if ctx_item.priority != ContextPriority.CRITICAL:
                    priority_values = list(ContextPriority)
                    current_idx = priority_values.index(ctx_item.priority)
                    if current_idx > 0:
                        ctx_item.priority = priority_values[current_idx - 1]
                
                shared_count += 1
        
        logger.info(f"🤝 Shared {shared_count} context items: {from_agent} -> {to_agent}")
        return shared_count
    
    async def _should_share_context(self, 
                                  ctx_item: ContextItem, 
                                  from_agent: str, 
                                  to_agent: str) -> bool:
        """Determine if context should be shared between agents."""
        
        # Don't share very low priority context
        if ctx_item.priority == ContextPriority.MINIMAL:
            return False
        
        # Don't share expired context
        if ctx_item.expiry_time and datetime.utcnow() > ctx_item.expiry_time:
            return False
        
        # Check if agents have collaborated before (higher sharing likelihood)
        collaboration_history = self.sharing_network[from_agent].get(to_agent, 0)
        if collaboration_history > 5:  # Frequent collaborators
            return True
        
        # Share high priority or recently accessed context
        if (ctx_item.priority in [ContextPriority.CRITICAL, ContextPriority.HIGH] or
            ctx_item.access_count > 3):
            return True
        
        # Don't share by default for privacy/security
        return False
    
    async def _context_optimization_loop(self):
        """Background loop for context optimization."""
        while True:
            try:
                await self._optimize_context_storage()
                await self._update_priority_scores()
                await self._analyze_access_patterns()
            except Exception as e:
                logger.error(f"Context optimization error: {e}")
            
            await asyncio.sleep(300)  # Optimize every 5 minutes
    
    async def _memory_cleanup_loop(self):
        """Background loop for memory cleanup."""
        while True:
            try:
                await self._cleanup_expired_context()
                await self._apply_forgetting_curve()
            except Exception as e:
                logger.error(f"Memory cleanup error: {e}")
            
            await asyncio.sleep(600)  # Cleanup every 10 minutes
    
    async def _optimize_context_storage(self):
        """Optimize overall context storage efficiency."""
        current_usage = sum(item.token_estimate for item in self.context_store.values())
        target_usage = self.context_budget.total_tokens * 0.8  # Aim for 80% utilization
        
        if current_usage > target_usage:
            # Need to compress or remove items
            excess = current_usage - target_usage
            
            # Find compression candidates
            compression_candidates = [
                item for item in self.context_store.values()
                if (item.compression_level < 2 and 
                    item.priority in [ContextPriority.LOW, ContextPriority.MINIMAL] and
                    item.access_count < 3)
            ]
            
            for item in compression_candidates[:10]:  # Compress up to 10 items
                await self._compress_context_item(item)
                if excess <= 0:
                    break
    
    async def _update_priority_scores(self):
        """Update priority scores based on access patterns."""
        for item in self.context_store.values():
            # Calculate new relevance score based on recent access
            recent_access = [
                access_time for access_time in self.access_patterns.get(item.item_id, [])
                if (datetime.utcnow() - access_time).days <= 7
            ]
            
            if len(recent_access) > 5:  # Highly accessed recently
                # Boost priority if not already critical
                if item.priority not in [ContextPriority.CRITICAL, ContextPriority.HIGH]:
                    priority_values = list(ContextPriority)
                    current_idx = priority_values.index(item.priority)
                    if current_idx > 0:
                        item.priority = priority_values[current_idx - 1]
            
            elif len(recent_access) == 0 and item.access_count < 2:
                # Rarely accessed, consider lowering priority
                if item.priority not in [ContextPriority.MINIMAL]:
                    priority_values = list(ContextPriority)
                    current_idx = priority_values.index(item.priority)
                    if current_idx < len(priority_values) - 1:
                        item.priority = priority_values[current_idx + 1]
    
    async def _analyze_access_patterns(self):
        """Analyze context access patterns for insights."""
        # Find frequently accessed context types
        type_access_counts = defaultdict(int)
        for item in self.context_store.values():
            type_access_counts[item.content_type] += item.access_count
        
        # Log insights
        most_accessed_type = max(type_access_counts, key=type_access_counts.get) if type_access_counts else None
        if most_accessed_type:
            logger.info(f"📊 Most accessed context type: {most_accessed_type.value}")
    
    async def _cleanup_expired_context(self):
        """Remove expired context items."""
        expired_items = [
            item_id for item_id, item in self.context_store.items()
            if item.expiry_time and datetime.utcnow() > item.expiry_time
        ]
        
        for item_id in expired_items:
            await self._remove_context_item(item_id)
        
        if expired_items:
            logger.info(f"🧹 Cleaned up {len(expired_items)} expired context items")
    
    async def _apply_forgetting_curve(self):
        """Apply forgetting curve to reduce relevance of old unused context."""
        current_time = datetime.utcnow()
        
        for item in self.context_store.values():
            # Apply forgetting curve based on last access time
            time_since_access = current_time - item.last_accessed
            hours_since_access = time_since_access.total_seconds() / 3600
            
            # Exponential decay
            forgetting_factor = math.exp(-self.forgetting_curve_decay * hours_since_access)
            item.relevance_score = item.relevance_score * forgetting_factor
            
            # If relevance drops too low, consider for removal
            if (item.relevance_score < 0.1 and 
                item.priority in [ContextPriority.LOW, ContextPriority.MINIMAL] and
                hours_since_access > 48):  # Not accessed for 2+ days
                
                await self._remove_context_item(item.item_id)
    
    def get_context_analytics(self) -> Dict[str, Any]:
        """Get comprehensive context intelligence analytics."""
        total_items = len(self.context_store)
        total_tokens = sum(item.token_estimate for item in self.context_store.values())
        
        # Priority distribution
        priority_distribution = defaultdict(int)
        for item in self.context_store.values():
            priority_distribution[item.priority.name] += 1
        
        # Type distribution
        type_distribution = defaultdict(int)
        for item in self.context_store.values():
            type_distribution[item.content_type.value] += 1
        
        # Compression stats
        compression_distribution = defaultdict(int)
        for item in self.context_store.values():
            compression_distribution[f"level_{item.compression_level}"] += 1
        
        # Agent distribution
        agent_context_counts = {agent: len(contexts) for agent, contexts in self.agent_contexts.items()}
        
        # Budget utilization
        budget_utilization = {
            "total_budget": self.context_budget.total_tokens,
            "reserved_tokens": self.context_budget.reserved_tokens,
            "used_tokens": total_tokens,
            "available_tokens": self.context_budget.total_tokens - self.context_budget.reserved_tokens - total_tokens,
            "utilization_percentage": (total_tokens / self.context_budget.total_tokens) * 100
        }
        
        return {
            "total_context_items": total_items,
            "total_tokens_used": total_tokens,
            "priority_distribution": dict(priority_distribution),
            "type_distribution": dict(type_distribution),
            "compression_distribution": dict(compression_distribution),
            "agent_context_counts": agent_context_counts,
            "budget_utilization": budget_utilization,
            "sharing_network_size": len(self.sharing_network),
            "semantic_index_size": len(self.semantic_index),
            "average_access_count": sum(item.access_count for item in self.context_store.values()) / total_items if total_items > 0 else 0,
            "compression_stats": dict(self.compression_stats)
        }


class SemanticAnalyzer:
    """Simplified semantic analyzer for context understanding."""
    
    def __init__(self):
        self.concept_mappings = {
            'programming': ['code', 'function', 'class', 'method', 'api', 'database'],
            'business': ['user', 'customer', 'requirement', 'goal', 'revenue', 'cost'],
            'technical': ['performance', 'optimization', 'algorithm', 'system', 'architecture'],
            'support': ['error', 'bug', 'issue', 'problem', 'fix', 'help'],
            'planning': ['strategy', 'roadmap', 'timeline', 'milestone', 'objective']
        }
    
    def analyze_content(self, content: str) -> Dict[str, float]:
        """Analyze content and return concept scores."""
        content_lower = content.lower()
        concept_scores = {}
        
        for concept, keywords in self.concept_mappings.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            concept_scores[concept] = score / len(keywords)  # Normalize
        
        return concept_scores
    
    def extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content (simplified)."""
        # Simplified entity extraction
        words = content.split()
        entities = []
        
        for word in words:
            if word.isupper() and len(word) > 2:  # Potential acronym
                entities.append(word)
            elif word[0].isupper() and len(word) > 3:  # Potential proper noun
                entities.append(word)
        
        return entities[:10]  # Return top 10 entities