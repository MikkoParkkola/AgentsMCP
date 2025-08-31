"""
Communication Interceptor for Strict Orchestrator Architecture

Intercepts all agent communications to prevent direct user exposure and ensure
strict adherence to the orchestrator-only communication pattern.

Key Functions:
- Intercept agent responses before they reach user interface
- Sanitize agent identifiers and direct references  
- Convert agent status messages to orchestrator perspective
- Maintain complete communication isolation
- Log agent interactions for orchestrator analysis
"""

import logging
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class InterceptedResponse:
    """Response after interception and processing."""
    intercepted: bool
    processed_response: Dict[str, Any]
    original_agent: str
    timestamp: datetime
    sanitization_applied: List[str]


class CommunicationInterceptor:
    """
    Intercepts and sanitizes all agent communications to enforce orchestrator-only architecture.
    
    This component is critical for maintaining communication isolation and ensuring users
    never see raw agent outputs that could break the orchestrator illusion.
    """
    
    def __init__(self):
        """Initialize the communication interceptor."""
        self.interceptions_count = 0
        self.agent_message_log: List[Dict] = []
        self.sanitization_patterns = self._build_sanitization_patterns()
        self.blocked_patterns = self._build_blocked_patterns()
        
        # Statistics
        self.interceptions_by_agent = {}
        self.sanitization_stats = {}
        
        logger.info("Communication interceptor initialized")
    
    def _build_sanitization_patterns(self) -> List[Dict]:
        """Build patterns for sanitizing agent communications."""
        return [
            {
                "name": "agent_identifier_removal",
                "patterns": [
                    r'🧩\s*\w+\s*:\s*',  # Remove "🧩 agent_name:" patterns
                    r'Agent\s+\w+\s*:\s*',  # Remove "Agent X:" patterns
                    r'\[\w+\s+Agent\]\s*',  # Remove "[X Agent]" patterns
                ],
                "replacement": ""
            },
            
            {
                "name": "agent_self_reference_removal",
                "patterns": [
                    r'I am (an? )?\w+ agent',
                    r'As (an? )?\w+ agent',
                    r'From my perspective as \w+',
                    r'Speaking as (an? )?\w+ agent'
                ],
                "replacement": "I"
            },
            
            {
                "name": "status_message_conversion",
                "patterns": [
                    r'=🎯\s*Orchestrating.*',  # Convert orchestration messages
                    r'Agent\s+\w+\s+is\s+(starting|working|completed)',
                ],
                "replacement": "Processing your request..."
            },
            
            {
                "name": "capability_statements",
                "patterns": [
                    r'I\'m\s+\w+\s+and\s+I\s+can',
                    r'I\'m\s+specialized\s+in',
                    r'My\s+capabilities\s+include'
                ],
                "replacement": "I can"
            }
        ]
    
    def _build_blocked_patterns(self) -> List[str]:
        """Build patterns that should be completely blocked from user view."""
        return [
            r'Error\s+in\s+agent\s+\w+',  # Internal agent errors
            r'Agent\s+\w+\s+failed',  # Agent failure messages  
            r'MCP\s+.*\s+not\s+available',  # MCP system messages
            r'Task\s+classification:',  # Internal task analysis
            r'Delegation\s+to\s+agent',  # Delegation messages
        ]
    
    def intercept_response(self, agent_id: str, response: str, 
                          metadata: Dict[str, Any]) -> InterceptedResponse:
        """
        Intercept an agent response and sanitize it for orchestrator use.
        
        This is the main interception point that all agent communications must pass through.
        """
        self.interceptions_count += 1
        self.interceptions_by_agent[agent_id] = self.interceptions_by_agent.get(agent_id, 0) + 1
        
        start_time = time.time()
        sanitization_applied = []
        
        # Log the original response for analysis
        self._log_agent_message(agent_id, response, metadata)
        
        # Check if response should be completely blocked
        if self._should_block_response(response):
            logger.debug(f"Blocked direct agent response from {agent_id}")
            sanitization_applied.append("response_blocked")
            
            return InterceptedResponse(
                intercepted=True,
                processed_response={
                    "sanitized_content": "",  # Blocked content
                    "agent_metadata": {"agent_id": agent_id, "blocked": True},
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                },
                original_agent=agent_id,
                timestamp=datetime.now(),
                sanitization_applied=sanitization_applied
            )
        
        # Apply sanitization patterns
        sanitized_content = response
        
        for pattern_group in self.sanitization_patterns:
            for pattern in pattern_group["patterns"]:
                if re.search(pattern, sanitized_content, re.IGNORECASE):
                    sanitized_content = re.sub(
                        pattern, pattern_group["replacement"], 
                        sanitized_content, flags=re.IGNORECASE
                    )
                    sanitization_applied.append(pattern_group["name"])
                    
                    # Update stats
                    stat_key = pattern_group["name"]
                    self.sanitization_stats[stat_key] = self.sanitization_stats.get(stat_key, 0) + 1
        
        # Clean up formatting issues from sanitization
        sanitized_content = self._clean_post_sanitization(sanitized_content)
        
        # Ensure content doesn't start with agent-like markers
        sanitized_content = self._ensure_orchestrator_voice(sanitized_content)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.debug(f"Intercepted and sanitized response from {agent_id} "
                    f"({processing_time}ms, {len(sanitization_applied)} changes)")
        
        return InterceptedResponse(
            intercepted=True,
            processed_response={
                "sanitized_content": sanitized_content,
                "agent_metadata": {
                    "agent_id": agent_id,
                    "original_length": len(response),
                    "sanitized_length": len(sanitized_content),
                    "sanitization_count": len(sanitization_applied)
                },
                "processing_time_ms": processing_time
            },
            original_agent=agent_id,
            timestamp=datetime.now(),
            sanitization_applied=sanitization_applied
        )
    
    def _should_block_response(self, response: str) -> bool:
        """Check if response should be completely blocked from user view."""
        for pattern in self.blocked_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        
        # Block responses that are purely agent identification
        if len(response.strip()) < 20 and any(
            marker in response.lower() 
            for marker in ["hello! how can i assist", "agent", "🧩"]
        ):
            return True
        
        return False
    
    def _clean_post_sanitization(self, content: str) -> str:
        """Clean up formatting issues after sanitization."""
        # Remove double spaces
        content = re.sub(r'\s+', ' ', content)
        
        # Remove leading/trailing whitespace
        content = content.strip()
        
        # Fix capitalization after period + space
        content = re.sub(r'(\.\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), content)
        
        # Ensure content ends with proper punctuation
        if content and not content.endswith(('.', '!', '?', ':')):
            content += '.'
        
        return content
    
    def _ensure_orchestrator_voice(self, content: str) -> str:
        """Ensure content sounds like it's coming from the orchestrator."""
        if not content:
            return content
        
        # If content starts with certain agent-like patterns, convert to orchestrator voice
        agent_starters = [
            (r'^(Hello!?\s+)?How can I assist', 'I can help'),
            (r'^I\'m here to help', 'I can help'),
            (r'^Let me help you with', 'I can help you with'),
            (r'^I\'ll be happy to', 'I\'d be happy to'),
        ]
        
        for pattern, replacement in agent_starters:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def _log_agent_message(self, agent_id: str, response: str, metadata: Dict[str, Any]):
        """Log agent message for analysis (internal use only)."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "response_length": len(response),
            "response_preview": response[:100] + "..." if len(response) > 100 else response,
            "metadata": metadata
        }
        
        self.agent_message_log.append(log_entry)
        
        # Keep log size manageable (last 1000 entries)
        if len(self.agent_message_log) > 1000:
            self.agent_message_log = self.agent_message_log[-1000:]
    
    def intercept_status_message(self, agent_id: str, status: str) -> Optional[str]:
        """
        Intercept agent status messages and convert to orchestrator perspective.
        
        Returns orchestrator-appropriate status or None if message should be suppressed.
        """
        # Block most agent status messages from user view
        blocked_status_patterns = [
            r'Agent\s+\w+\s+starting',
            r'Connecting to \w+',
            r'Loading \w+ model',
            r'Task assigned to',
            r'Delegation complete'
        ]
        
        for pattern in blocked_status_patterns:
            if re.search(pattern, status, re.IGNORECASE):
                return None  # Suppress this status message
        
        # Convert remaining status messages to orchestrator perspective
        status_conversions = [
            (r'Processing.*', 'Working on your request...'),
            (r'Analyzing.*', 'Analyzing your request...'),
            (r'Generating.*', 'Preparing your response...'),
            (r'Complete.*', 'Ready'),
        ]
        
        for pattern, replacement in status_conversions:
            if re.search(pattern, status, re.IGNORECASE):
                return replacement
        
        # Default: suppress unknown status messages
        return None
    
    def get_interception_stats(self) -> Dict[str, Any]:
        """Get communication interception statistics."""
        return {
            "total_interceptions": self.interceptions_count,
            "interceptions_by_agent": self.interceptions_by_agent,
            "sanitization_stats": self.sanitization_stats,
            "log_entries": len(self.agent_message_log),
            "patterns_configured": {
                "sanitization_patterns": len(self.sanitization_patterns),
                "blocked_patterns": len(self.blocked_patterns)
            }
        }
    
    def get_agent_communication_summary(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of agent communications (for orchestrator analysis only)."""
        if agent_id:
            # Filter for specific agent
            agent_logs = [log for log in self.agent_message_log if log["agent_id"] == agent_id]
        else:
            agent_logs = self.agent_message_log
        
        if not agent_logs:
            return {"message_count": 0, "agents": []}
        
        return {
            "message_count": len(agent_logs),
            "time_range": {
                "earliest": agent_logs[0]["timestamp"],
                "latest": agent_logs[-1]["timestamp"]
            },
            "agents": list(set(log["agent_id"] for log in agent_logs)),
            "average_response_length": sum(log["response_length"] for log in agent_logs) / len(agent_logs)
        }
    
    def clear_logs(self):
        """Clear agent message logs (for testing or reset purposes)."""
        self.agent_message_log.clear()
        logger.info("Agent communication logs cleared")