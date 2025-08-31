"""
Response Synthesizer for Orchestrator Communication

Consolidates multiple agent responses into single coherent user-facing responses.
Ensures users see unified, orchestrator-perspective communications rather than
individual agent outputs.

Synthesis Strategies:
- SUMMARIZE: Consolidate key points from all agents
- BEST_RESPONSE: Select the best single agent response  
- COLLABORATIVE: Merge complementary agent responses into unified answer
- CONSENSUS: Find agreement points across agent responses
"""

import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SynthesisStrategy(Enum):
    """Strategies for synthesizing multiple agent responses."""
    SUMMARIZE = "summarize"
    BEST_RESPONSE = "best_response" 
    COLLABORATIVE = "collaborative"
    CONSENSUS = "consensus"


@dataclass
class SynthesisResult:
    """Result of response synthesis."""
    synthesized_response: str
    synthesis_metadata: Dict[str, Any]
    confidence_score: float
    source_agents: List[str]


class ResponseSynthesizer:
    """
    Synthesizes multiple agent responses into unified user-facing responses.
    
    This component is crucial for maintaining the orchestrator-only communication
    architecture by ensuring users never see raw agent outputs.
    """
    
    def __init__(self):
        """Initialize the response synthesizer."""
        self.synthesis_stats = {
            SynthesisStrategy.SUMMARIZE: 0,
            SynthesisStrategy.BEST_RESPONSE: 0,
            SynthesisStrategy.COLLABORATIVE: 0,
            SynthesisStrategy.CONSENSUS: 0
        }
        self.total_syntheses = 0
        
        # Agent reliability scoring (learned over time)
        self.agent_reliability = {
            "codex": 0.85,
            "claude": 0.90,
            "ollama": 0.75
        }
        
        # Common patterns to remove from agent responses
        self.agent_identifiers = [
            r'🧩\s*\w+\s*:',  # Agent emoji identifiers
            r'Agent\s+\w+\s*:',  # "Agent X:" patterns
            r'I\'m\s+\w+\s+agent',  # "I'm X agent" patterns
            r'As\s+\w+\s+agent',  # "As X agent" patterns
        ]
    
    async def synthesize_responses(self, agent_responses: Dict[str, str], 
                                 original_request: str,
                                 strategy: SynthesisStrategy) -> SynthesisResult:
        """Synthesize multiple agent responses into a unified response."""
        self.total_syntheses += 1
        self.synthesis_stats[strategy] += 1
        
        if not agent_responses:
            return SynthesisResult(
                synthesized_response="I apologize, but I wasn't able to generate a response for your request. Could you please try rephrasing it?",
                synthesis_metadata={"method_used": "empty_fallback", "agent_count": 0},
                confidence_score=0.0,
                source_agents=[]
            )
        
        # Clean all agent responses first
        cleaned_responses = {
            agent: self._clean_agent_response(response)
            for agent, response in agent_responses.items()
        }
        
        # Apply synthesis strategy
        if strategy == SynthesisStrategy.SUMMARIZE:
            result = await self._synthesize_summarize(cleaned_responses, original_request)
        elif strategy == SynthesisStrategy.BEST_RESPONSE:
            result = await self._synthesize_best_response(cleaned_responses, original_request)
        elif strategy == SynthesisStrategy.COLLABORATIVE:
            result = await self._synthesize_collaborative(cleaned_responses, original_request)
        elif strategy == SynthesisStrategy.CONSENSUS:
            result = await self._synthesize_consensus(cleaned_responses, original_request)
        else:
            # Fallback to summarize
            result = await self._synthesize_summarize(cleaned_responses, original_request)
        
        # Add orchestrator framing
        result.synthesized_response = self._add_orchestrator_framing(
            result.synthesized_response, original_request, len(agent_responses)
        )
        
        logger.debug(f"Synthesized response using {strategy.value} strategy "
                    f"(confidence: {result.confidence_score:.2f})")
        
        return result
    
    def _clean_agent_response(self, response: str) -> str:
        """Clean agent identifiers and markers from response."""
        cleaned = response
        
        # Remove agent identifier patterns
        for pattern in self.agent_identifiers:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove common agent self-references
        agent_refs = [
            r'I am (an? )?\w+ agent',
            r'As (an? )?\w+ agent',
            r'From my perspective as',
            r'Speaking as (an? )?\w+'
        ]
        
        for pattern in agent_refs:
            cleaned = re.sub(pattern, 'I', cleaned, flags=re.IGNORECASE)
        
        # Clean up any double spaces or line breaks
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    async def _synthesize_summarize(self, responses: Dict[str, str], 
                                  original_request: str) -> SynthesisResult:
        """Summarize key points from all agent responses."""
        if len(responses) == 1:
            # Single response - just clean and return
            agent, response = list(responses.items())[0]
            return SynthesisResult(
                synthesized_response=response,
                synthesis_metadata={
                    "method_used": "single_response_passthrough",
                    "source_agent": agent
                },
                confidence_score=self.agent_reliability.get(agent, 0.7),
                source_agents=[agent]
            )
        
        # Multiple responses - create summary
        key_points = []
        all_agents = list(responses.keys())
        
        # Extract key information from each response
        for agent, response in responses.items():
            if response and len(response.strip()) > 10:
                # Take first substantial sentence or key point
                sentences = response.split('.')
                if sentences:
                    key_point = sentences[0].strip()
                    if len(key_point) > 20:
                        key_points.append(key_point)
        
        if not key_points:
            # Fallback if no good points extracted
            synthesized = f"I can help you with that request. {list(responses.values())[0][:100]}..."
        else:
            # Combine key points into coherent summary
            synthesized = f"I can help with your request. {' '.join(key_points)}."
        
        confidence = min(0.8, sum(self.agent_reliability.get(a, 0.7) for a in all_agents) / len(all_agents))
        
        return SynthesisResult(
            synthesized_response=synthesized,
            synthesis_metadata={
                "method_used": "summarize",
                "key_points_count": len(key_points),
                "source_agents": all_agents
            },
            confidence_score=confidence,
            source_agents=all_agents
        )
    
    async def _synthesize_best_response(self, responses: Dict[str, str],
                                      original_request: str) -> SynthesisResult:
        """Select the best single agent response."""
        if not responses:
            return SynthesisResult(
                synthesized_response="I'm unable to provide a response right now.",
                synthesis_metadata={"method_used": "empty_fallback"},
                confidence_score=0.0,
                source_agents=[]
            )
        
        # Score each response
        best_agent = None
        best_response = None
        best_score = 0.0
        
        for agent, response in responses.items():
            score = self._score_response_quality(response, agent, original_request)
            if score > best_score:
                best_score = score
                best_agent = agent
                best_response = response
        
        return SynthesisResult(
            synthesized_response=best_response or "I'm working on that request.",
            synthesis_metadata={
                "method_used": "best_response",
                "selected_agent": best_agent,
                "quality_score": best_score
            },
            confidence_score=best_score,
            source_agents=[best_agent] if best_agent else []
        )
    
    async def _synthesize_collaborative(self, responses: Dict[str, str],
                                      original_request: str) -> SynthesisResult:
        """Merge complementary agent responses into unified answer."""
        if len(responses) == 1:
            return await self._synthesize_best_response(responses, original_request)
        
        # Identify complementary sections
        code_sections = []
        explanation_sections = []
        action_sections = []
        
        for agent, response in responses.items():
            if self._contains_code(response):
                code_sections.append(response)
            elif self._contains_explanation(response):
                explanation_sections.append(response)
            else:
                action_sections.append(response)
        
        # Build collaborative response
        parts = []
        
        if explanation_sections:
            parts.append(explanation_sections[0])  # Best explanation
        
        if code_sections:
            parts.append("Here's the implementation:")
            parts.append(code_sections[0])  # Best code
        
        if action_sections and not explanation_sections:
            parts.append(action_sections[0])  # Action guidance if no explanation
        
        if not parts:
            # Fallback to first available response
            parts.append(list(responses.values())[0])
        
        synthesized = " ".join(parts)
        confidence = 0.85  # High confidence for collaborative synthesis
        
        return SynthesisResult(
            synthesized_response=synthesized,
            synthesis_metadata={
                "method_used": "collaborative",
                "sections_combined": len(parts),
                "has_code": len(code_sections) > 0,
                "has_explanation": len(explanation_sections) > 0
            },
            confidence_score=confidence,
            source_agents=list(responses.keys())
        )
    
    async def _synthesize_consensus(self, responses: Dict[str, str],
                                  original_request: str) -> SynthesisResult:
        """Find agreement points across agent responses."""
        if len(responses) <= 1:
            return await self._synthesize_best_response(responses, original_request)
        
        # Find common themes or agreement points
        response_words = []
        for response in responses.values():
            words = set(response.lower().split())
            response_words.append(words)
        
        # Find intersection of key words (consensus points)
        if len(response_words) >= 2:
            common_words = response_words[0]
            for word_set in response_words[1:]:
                common_words = common_words.intersection(word_set)
            
            # Filter to meaningful words (remove common stopwords)
            stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            meaningful_consensus = [word for word in common_words if word not in stopwords and len(word) > 2]
        else:
            meaningful_consensus = []
        
        # If we found consensus points, use them
        if meaningful_consensus:
            base_response = list(responses.values())[0]  # Use first response as base
            synthesized = f"Based on analysis, {base_response}"
            confidence = 0.90
        else:
            # No clear consensus - fall back to best response
            return await self._synthesize_best_response(responses, original_request)
        
        return SynthesisResult(
            synthesized_response=synthesized,
            synthesis_metadata={
                "method_used": "consensus",
                "consensus_points": len(meaningful_consensus),
                "common_themes": meaningful_consensus[:5]  # Top 5 consensus words
            },
            confidence_score=confidence,
            source_agents=list(responses.keys())
        )
    
    def _score_response_quality(self, response: str, agent: str, original_request: str) -> float:
        """Score the quality of an agent response."""
        if not response or len(response.strip()) < 10:
            return 0.0
        
        score = self.agent_reliability.get(agent, 0.7)
        
        # Bonus for length (more comprehensive)
        if len(response) > 100:
            score += 0.1
        
        # Bonus for code content if request seems to need code
        if self._request_needs_code(original_request) and self._contains_code(response):
            score += 0.15
        
        # Bonus for explanation if request asks for explanation
        if self._request_needs_explanation(original_request) and self._contains_explanation(response):
            score += 0.15
        
        # Penalty for error indicators
        error_indicators = ['error', 'failed', 'unable', 'cannot', 'sorry']
        if any(indicator in response.lower() for indicator in error_indicators):
            score -= 0.2
        
        return min(1.0, score)
    
    def _contains_code(self, response: str) -> bool:
        """Check if response contains code."""
        code_indicators = ['```', 'def ', 'function ', 'class ', 'import ', 'from ', 'console.log', '#!/']
        return any(indicator in response for indicator in code_indicators)
    
    def _contains_explanation(self, response: str) -> bool:
        """Check if response contains explanations."""
        explanation_indicators = ['explain', 'because', 'this means', 'in other words', 'for example']
        return any(indicator in response.lower() for indicator in explanation_indicators)
    
    def _request_needs_code(self, request: str) -> bool:
        """Check if original request needs code."""
        code_request_words = ['write', 'create', 'implement', 'code', 'function', 'script', 'program']
        return any(word in request.lower() for word in code_request_words)
    
    def _request_needs_explanation(self, request: str) -> bool:
        """Check if original request needs explanation."""
        explanation_request_words = ['explain', 'describe', 'what is', 'how does', 'why', 'help me understand']
        return any(word in request.lower() for word in explanation_request_words)
    
    def _add_orchestrator_framing(self, response: str, original_request: str, agent_count: int) -> str:
        """Add orchestrator perspective framing to synthesized response."""
        # Don't add framing if response already has orchestrator voice
        if response.startswith("I ") or response.startswith("Let me ") or response.startswith("I can "):
            return response
        
        # Add appropriate orchestrator introduction based on complexity
        if agent_count > 1:
            # Multi-agent response
            intro = "I've analyzed your request and here's what I can help you with: "
        else:
            # Single agent or simple response  
            intro = "I can help you with that. "
        
        # Ensure smooth flow
        if response.startswith(("Here", "This", "The", "Based")):
            return f"{intro}{response}"
        else:
            return f"{intro}{response}"
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get synthesis performance statistics."""
        return {
            "total_syntheses": self.total_syntheses,
            "strategy_usage": {k.value: v for k, v in self.synthesis_stats.items()},
            "agent_reliability": self.agent_reliability
        }