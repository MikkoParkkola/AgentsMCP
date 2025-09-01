"""
Conversation Context Manager

Manages conversation history, context, and learning from previous interactions.
Provides memory and context continuity for improved understanding and response quality.
"""

import logging
import json
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import uuid

from .intent_analyzer import IntentAnalysis, IntentType, TechnicalDomain

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context information.""" 
    USER_PREFERENCE = "user_preference"
    TECHNICAL_CONTEXT = "technical_context"
    PROJECT_CONTEXT = "project_context"
    CONVERSATION_HISTORY = "conversation_history"
    LEARNED_PATTERN = "learned_pattern"
    ERROR_PATTERN = "error_pattern"
    SUCCESS_PATTERN = "success_pattern"


class ContextScope(Enum):
    """Scope of context applicability."""
    SESSION = "session"          # Current session only
    USER = "user"               # Across user sessions
    PROJECT = "project"         # Project-specific
    GLOBAL = "global"           # Global patterns


@dataclass
class ContextEntry:
    """A single context entry."""
    id: str
    context_type: ContextType
    scope: ContextScope
    content: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    confidence: float = 1.0
    expires_at: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    source: str = "user_input"  # user_input, system_learning, external


@dataclass
class ConversationTurn:
    """A single conversation turn."""
    turn_id: str
    user_input: str
    intent_analysis: Optional[IntentAnalysis]
    system_response: str
    timestamp: datetime
    success_indicators: Dict[str, Any] = field(default_factory=dict)
    user_satisfaction: Optional[float] = None  # 0-1 scale
    follow_up_questions: List[str] = field(default_factory=list)
    context_used: List[str] = field(default_factory=list)  # Context IDs used


@dataclass
class ConversationSession:
    """A conversation session."""
    session_id: str
    user_id: Optional[str]
    project_context: Optional[str]
    start_time: datetime
    last_activity: datetime
    turns: List[ConversationTurn] = field(default_factory=list)
    session_context: Dict[str, Any] = field(default_factory=dict)
    active: bool = True


class ConversationContext:
    """
    Comprehensive conversation context manager.
    
    Manages conversation history, user preferences, technical context,
    and learned patterns to improve understanding and response quality.
    """
    
    def __init__(self, max_context_entries: int = 1000, max_session_turns: int = 100):
        """Initialize conversation context manager."""
        self.max_context_entries = max_context_entries
        self.max_session_turns = max_session_turns
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Storage
        self.context_entries: Dict[str, ContextEntry] = {}
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.user_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Learning patterns
        self.learned_patterns: Dict[str, Any] = {}
        self.success_patterns: List[Dict[str, Any]] = []
        self.error_patterns: List[Dict[str, Any]] = []
        
        # Statistics
        self.total_contexts_stored = 0
        self.total_sessions = 0
        self.context_hits = 0
        self.learning_events = 0
        
        self.logger.info(f"ConversationContext initialized with max_entries={max_context_entries}")
    
    async def start_session(self, user_id: Optional[str] = None, project_context: Optional[str] = None) -> str:
        """Start a new conversation session."""
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now()
        
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            project_context=project_context,
            start_time=now,
            last_activity=now
        )
        
        self.active_sessions[session_id] = session
        self.total_sessions += 1
        
        # Load relevant context for the session
        await self._load_session_context(session)
        
        self.logger.info(f"Started conversation session {session_id} for user {user_id}")
        return session_id
    
    async def add_conversation_turn(self, 
                                 session_id: str,
                                 user_input: str,
                                 intent_analysis: Optional[IntentAnalysis],
                                 system_response: str,
                                 success_indicators: Optional[Dict] = None) -> ConversationTurn:
        """Add a new conversation turn to the session."""
        if session_id not in self.active_sessions:
            # Auto-create session if it doesn't exist
            await self.start_session()
            session_id = list(self.active_sessions.keys())[-1]
        
        session = self.active_sessions[session_id]
        turn_id = f"{session_id}_{len(session.turns)}"
        
        turn = ConversationTurn(
            turn_id=turn_id,
            user_input=user_input,
            intent_analysis=intent_analysis,
            system_response=system_response,
            timestamp=datetime.now(),
            success_indicators=success_indicators or {}
        )
        
        session.turns.append(turn)
        session.last_activity = datetime.now()
        
        # Limit turn history
        if len(session.turns) > self.max_session_turns:
            session.turns = session.turns[-self.max_session_turns:]
        
        # Learn from this interaction
        await self._learn_from_turn(turn, session)
        
        self.logger.debug(f"Added conversation turn {turn_id} to session {session_id}")
        return turn
    
    async def get_relevant_context(self, 
                                 session_id: str,
                                 current_input: str,
                                 intent_analysis: Optional[IntentAnalysis] = None,
                                 limit: int = 10) -> List[ContextEntry]:
        """Get relevant context entries for current input."""
        if session_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[session_id]
        relevant_contexts = []
        
        # Score all context entries for relevance
        context_scores = []
        
        for context_id, context in self.context_entries.items():
            score = await self._calculate_relevance_score(
                context, current_input, intent_analysis, session
            )
            
            if score > 0.1:  # Minimum relevance threshold
                context_scores.append((context, score))
        
        # Sort by relevance and return top results
        context_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_contexts = [ctx for ctx, score in context_scores[:limit]]
        
        # Update access statistics
        for context in relevant_contexts:
            context.access_count += 1
            context.last_accessed = datetime.now()
        
        self.context_hits += len(relevant_contexts)
        
        self.logger.debug(f"Retrieved {len(relevant_contexts)} relevant contexts for session {session_id}")
        return relevant_contexts
    
    async def add_context_entry(self,
                              context_type: ContextType,
                              scope: ContextScope,
                              content: Dict[str, Any],
                              session_id: Optional[str] = None,
                              expires_in: Optional[timedelta] = None,
                              tags: Optional[Set[str]] = None,
                              source: str = "user_input") -> str:
        """Add a new context entry."""
        context_id = str(uuid.uuid4())
        now = datetime.now()
        
        expires_at = None
        if expires_in:
            expires_at = now + expires_in
        elif scope == ContextScope.SESSION:
            expires_at = now + timedelta(hours=24)  # Session contexts expire in 24h
        
        context = ContextEntry(
            id=context_id,
            context_type=context_type,
            scope=scope,
            content=content,
            created_at=now,
            last_accessed=now,
            expires_at=expires_at,
            tags=tags or set(),
            source=source
        )
        
        self.context_entries[context_id] = context
        self.total_contexts_stored += 1
        
        # Cleanup old entries if at capacity
        if len(self.context_entries) > self.max_context_entries:
            await self._cleanup_old_contexts()
        
        self.logger.debug(f"Added context entry {context_id} of type {context_type.value}")
        return context_id
    
    async def update_user_preferences(self, 
                                    user_id: str,
                                    preferences: Dict[str, Any],
                                    session_id: Optional[str] = None):
        """Update user preferences based on behavior and explicit settings."""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {}
        
        # Merge with existing preferences
        existing = self.user_contexts[user_id].get("preferences", {})
        existing.update(preferences)
        self.user_contexts[user_id]["preferences"] = existing
        
        # Store as context entry
        await self.add_context_entry(
            context_type=ContextType.USER_PREFERENCE,
            scope=ContextScope.USER,
            content={"user_id": user_id, "preferences": existing},
            tags={"user_preference", user_id},
            source="system_learning"
        )
        
        self.logger.info(f"Updated preferences for user {user_id}: {list(preferences.keys())}")
    
    async def learn_from_feedback(self,
                                session_id: str,
                                turn_id: str, 
                                feedback: Dict[str, Any]):
        """Learn from user feedback on system responses."""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # Find the relevant turn
        turn = None
        for t in session.turns:
            if t.turn_id == turn_id:
                turn = t
                break
        
        if not turn:
            return
        
        # Update turn with feedback
        turn.user_satisfaction = feedback.get("satisfaction", 0.5)
        turn.success_indicators.update(feedback)
        
        # Learn patterns from feedback
        await self._learn_from_feedback(turn, feedback, session)
        
        self.learning_events += 1
        self.logger.info(f"Learned from feedback for turn {turn_id}: satisfaction={turn.user_satisfaction}")
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation session."""
        if session_id not in self.active_sessions:
            return {}
        
        session = self.active_sessions[session_id]
        
        # Analyze session patterns
        intent_distribution = {}
        domain_distribution = {}
        avg_satisfaction = 0.0
        satisfaction_count = 0
        
        for turn in session.turns:
            if turn.intent_analysis:
                intent = turn.intent_analysis.primary_intent.value
                domain = turn.intent_analysis.technical_domain.value
                
                intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
                domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
            
            if turn.user_satisfaction is not None:
                avg_satisfaction += turn.user_satisfaction
                satisfaction_count += 1
        
        if satisfaction_count > 0:
            avg_satisfaction /= satisfaction_count
        
        return {
            "session_id": session_id,
            "duration_minutes": (session.last_activity - session.start_time).total_seconds() / 60,
            "total_turns": len(session.turns),
            "intent_distribution": intent_distribution,
            "domain_distribution": domain_distribution,
            "average_satisfaction": avg_satisfaction,
            "active": session.active,
            "user_id": session.user_id,
            "project_context": session.project_context
        }
    
    async def close_session(self, session_id: str):
        """Close and finalize a conversation session."""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        session.active = False
        
        # Extract learnings from the session
        await self._extract_session_learnings(session)
        
        # Move to inactive sessions (optional - could be implemented)
        del self.active_sessions[session_id]
        
        self.logger.info(f"Closed session {session_id} with {len(session.turns)} turns")
    
    async def _load_session_context(self, session: ConversationSession):
        """Load relevant context for a new session."""
        # Load user preferences
        if session.user_id and session.user_id in self.user_contexts:
            user_prefs = self.user_contexts[session.user_id]
            session.session_context["user_preferences"] = user_prefs
        
        # Load project context
        if session.project_context:
            project_contexts = [
                ctx for ctx in self.context_entries.values()
                if ctx.context_type == ContextType.PROJECT_CONTEXT
                and session.project_context in ctx.tags
            ]
            session.session_context["project_contexts"] = project_contexts
        
        # Load relevant learned patterns
        session.session_context["learned_patterns"] = self.learned_patterns.copy()
    
    async def _calculate_relevance_score(self,
                                       context: ContextEntry,
                                       current_input: str,
                                       intent_analysis: Optional[IntentAnalysis],
                                       session: ConversationSession) -> float:
        """Calculate relevance score for a context entry."""
        score = 0.0
        
        # Base score by context type
        type_scores = {
            ContextType.USER_PREFERENCE: 0.3,
            ContextType.TECHNICAL_CONTEXT: 0.5,
            ContextType.PROJECT_CONTEXT: 0.4,
            ContextType.CONVERSATION_HISTORY: 0.6,
            ContextType.LEARNED_PATTERN: 0.4,
            ContextType.SUCCESS_PATTERN: 0.7,
            ContextType.ERROR_PATTERN: 0.5
        }
        score += type_scores.get(context.context_type, 0.2)
        
        # Keyword overlap
        if "keywords" in context.content:
            context_keywords = set(context.content["keywords"])
            input_words = set(current_input.lower().split())
            overlap = len(context_keywords.intersection(input_words))
            score += overlap * 0.1
        
        # Intent matching
        if intent_analysis and "intent" in context.content:
            context_intent = context.content.get("intent")
            if context_intent == intent_analysis.primary_intent.value:
                score += 0.3
            elif context_intent in [i.value for i in intent_analysis.secondary_intents]:
                score += 0.2
        
        # Domain matching
        if intent_analysis and "domain" in context.content:
            context_domain = context.content.get("domain")
            if context_domain == intent_analysis.technical_domain.value:
                score += 0.2
        
        # Recency bonus
        age_hours = (datetime.now() - context.last_accessed).total_seconds() / 3600
        if age_hours < 1:
            score += 0.2
        elif age_hours < 24:
            score += 0.1
        
        # Frequency bonus (popular contexts)
        if context.access_count > 5:
            score += 0.1
        
        # Scope relevance
        if context.scope == ContextScope.SESSION:
            score += 0.3  # Current session is highly relevant
        elif context.scope == ContextScope.USER and session.user_id:
            score += 0.2
        elif context.scope == ContextScope.PROJECT and session.project_context:
            score += 0.2
        
        # Confidence factor
        score *= context.confidence
        
        return min(score, 1.0)
    
    async def _learn_from_turn(self, turn: ConversationTurn, session: ConversationSession):
        """Learn patterns from a conversation turn."""
        if not turn.intent_analysis:
            return
        
        # Extract learning data
        learning_data = {
            "user_input": turn.user_input,
            "intent": turn.intent_analysis.primary_intent.value,
            "domain": turn.intent_analysis.technical_domain.value,
            "complexity": turn.intent_analysis.complexity_level,
            "keywords": turn.intent_analysis.keywords[:10],
            "technologies": turn.intent_analysis.technologies,
            "timestamp": turn.timestamp.isoformat()
        }
        
        # Store as conversation history context
        await self.add_context_entry(
            context_type=ContextType.CONVERSATION_HISTORY,
            scope=ContextScope.SESSION,
            content=learning_data,
            session_id=session.session_id,
            tags={"conversation", session.session_id},
            source="system_learning"
        )
        
        # Update learned patterns
        intent_key = turn.intent_analysis.primary_intent.value
        if intent_key not in self.learned_patterns:
            self.learned_patterns[intent_key] = {
                "count": 0,
                "common_keywords": {},
                "common_domains": {},
                "avg_complexity": 0.0
            }
        
        pattern = self.learned_patterns[intent_key]
        pattern["count"] += 1
        
        # Update common keywords
        for keyword in turn.intent_analysis.keywords[:5]:
            pattern["common_keywords"][keyword] = pattern["common_keywords"].get(keyword, 0) + 1
        
        # Update common domains
        domain = turn.intent_analysis.technical_domain.value
        pattern["common_domains"][domain] = pattern["common_domains"].get(domain, 0) + 1
    
    async def _learn_from_feedback(self, 
                                 turn: ConversationTurn,
                                 feedback: Dict[str, Any],
                                 session: ConversationSession):
        """Learn from user feedback."""
        satisfaction = feedback.get("satisfaction", 0.5)
        
        if not turn.intent_analysis:
            return
        
        # Create learning entry
        pattern_data = {
            "user_input": turn.user_input[:200],  # Truncate for privacy
            "intent": turn.intent_analysis.primary_intent.value,
            "domain": turn.intent_analysis.technical_domain.value,
            "satisfaction": satisfaction,
            "feedback": feedback,
            "system_response_length": len(turn.system_response),
            "keywords": turn.intent_analysis.keywords[:5]
        }
        
        # Store as success or error pattern
        if satisfaction >= 0.7:
            self.success_patterns.append(pattern_data)
            context_type = ContextType.SUCCESS_PATTERN
        elif satisfaction <= 0.3:
            self.error_patterns.append(pattern_data)
            context_type = ContextType.ERROR_PATTERN
        else:
            return  # Neutral feedback - don't store
        
        # Limit pattern history
        if len(self.success_patterns) > 100:
            self.success_patterns = self.success_patterns[-100:]
        if len(self.error_patterns) > 50:
            self.error_patterns = self.error_patterns[-50:]
        
        # Store as context entry
        await self.add_context_entry(
            context_type=context_type,
            scope=ContextScope.GLOBAL,
            content=pattern_data,
            tags={"feedback", "learning"},
            source="user_feedback"
        )
    
    async def _extract_session_learnings(self, session: ConversationSession):
        """Extract learnings from a completed session."""
        if not session.turns:
            return
        
        # Analyze session-level patterns
        successful_turns = [
            turn for turn in session.turns 
            if turn.user_satisfaction and turn.user_satisfaction >= 0.7
        ]
        
        if successful_turns:
            # Find common patterns in successful interactions
            common_intents = {}
            common_keywords = {}
            
            for turn in successful_turns:
                if turn.intent_analysis:
                    intent = turn.intent_analysis.primary_intent.value
                    common_intents[intent] = common_intents.get(intent, 0) + 1
                    
                    for keyword in turn.intent_analysis.keywords[:3]:
                        common_keywords[keyword] = common_keywords.get(keyword, 0) + 1
            
            # Store session learning
            session_learning = {
                "session_id": session.session_id,
                "successful_intents": common_intents,
                "successful_keywords": common_keywords,
                "total_turns": len(session.turns),
                "successful_turns": len(successful_turns),
                "user_id": session.user_id,
                "project_context": session.project_context
            }
            
            await self.add_context_entry(
                context_type=ContextType.LEARNED_PATTERN,
                scope=ContextScope.USER if session.user_id else ContextScope.GLOBAL,
                content=session_learning,
                tags={"session_learning", session.session_id},
                source="system_learning"
            )
    
    async def _cleanup_old_contexts(self):
        """Clean up old and expired context entries."""
        now = datetime.now()
        to_remove = []
        
        for context_id, context in self.context_entries.items():
            # Remove expired contexts
            if context.expires_at and context.expires_at < now:
                to_remove.append(context_id)
                continue
            
            # Remove least accessed contexts if over limit
            if (context.access_count == 0 and 
                (now - context.last_accessed).total_seconds() > 7 * 24 * 3600):  # 7 days
                to_remove.append(context_id)
        
        # Remove oldest contexts if still over limit
        if len(self.context_entries) - len(to_remove) > self.max_context_entries:
            remaining = [(cid, ctx) for cid, ctx in self.context_entries.items() if cid not in to_remove]
            remaining.sort(key=lambda x: x[1].last_accessed)
            
            excess_count = len(remaining) - self.max_context_entries
            for i in range(excess_count):
                to_remove.append(remaining[i][0])
        
        # Remove selected contexts
        for context_id in to_remove:
            del self.context_entries[context_id]
        
        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old context entries")
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get conversation context statistics."""
        active_contexts_by_type = {}
        active_contexts_by_scope = {}
        
        for context in self.context_entries.values():
            ctx_type = context.context_type.value
            ctx_scope = context.scope.value
            
            active_contexts_by_type[ctx_type] = active_contexts_by_type.get(ctx_type, 0) + 1
            active_contexts_by_scope[ctx_scope] = active_contexts_by_scope.get(ctx_scope, 0) + 1
        
        return {
            "total_contexts_stored": self.total_contexts_stored,
            "active_contexts": len(self.context_entries),
            "active_sessions": len(self.active_sessions),
            "total_sessions": self.total_sessions,
            "context_hits": self.context_hits,
            "learning_events": self.learning_events,
            "contexts_by_type": active_contexts_by_type,
            "contexts_by_scope": active_contexts_by_scope,
            "success_patterns": len(self.success_patterns),
            "error_patterns": len(self.error_patterns),
            "learned_patterns": len(self.learned_patterns)
        }