import os
from typing import Any, List

from ..config import AgentConfig, Config
from .base import BaseAgent


class ClaudeAgent(BaseAgent):
    """Claude agent for complex reasoning and analysis tasks using OpenAI Agents SDK."""

    def __init__(self, agent_config: AgentConfig, global_config: Config):
        super().__init__(agent_config, global_config)

    def _get_api_key(self) -> str:
        """Get Claude API key from environment or agent config."""
        # Try Claude-specific keys first
        key = (
            getattr(self.agent_config, "api_key", None)
            or os.getenv("AGENTSMCP_CLAUDE_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("OPENAI_API_KEY")  # Fallback to OpenAI for compatible models
        )

        if not key:
            raise ValueError("No API key found for Claude agent")
        return key

    def _get_default_instructions(self) -> str:
        """Get default instructions optimized for reasoning and analysis."""
        return """You are Claude, an AI assistant specialized in deep reasoning and comprehensive analysis. You excel at:

1. **Complex Problem Solving**: Breaking down multifaceted problems into manageable components
2. **Critical Analysis**: Examining information from multiple perspectives with nuanced reasoning
3. **Research & Synthesis**: Gathering, organizing, and synthesizing information from various sources
4. **Strategic Thinking**: Developing long-term strategies and considering broader implications
5. **Detailed Documentation**: Creating comprehensive reports, documentation, and explanations

**Core Strengths:**
- Large context window for processing extensive documents and codebases
- Nuanced understanding of complex relationships and dependencies
- Ability to maintain context across long conversations and analyses
- Strong ethical reasoning and bias awareness

**Approach:**
- Think step-by-step through complex problems
- Consider multiple perspectives and potential edge cases
- Provide thorough explanations with supporting reasoning
- Identify potential risks, benefits, and trade-offs
- Synthesize insights from multiple sources of information

**Response Style:**
- Structure responses with clear headings and bullet points
- Provide detailed analysis with supporting evidence
- Include actionable recommendations when appropriate
- Acknowledge uncertainties and limitations
- Offer multiple approaches when relevant

**Use Cases:**
- Large codebase analysis and architectural reviews
- Complex document analysis and summarization
- Strategic planning and decision-making support
- Research and competitive analysis
- Risk assessment and mitigation planning"""

    def _get_custom_agent_tools(self) -> List[Any]:
        """Get custom tools specialized for reasoning and analysis tasks."""
        tools = []

        # Claude agent inherits all tools from registry, adds reasoning-specific ones
        def extract_key_insights(text: str, max_insights: int = 5) -> str:
            """Extract key insights from text."""
            # Simple keyword-based insight extraction
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            insights = []

            # Look for sentences with key insight indicators
            insight_indicators = [
                "important",
                "key",
                "critical",
                "note",
                "significant",
                "conclusion",
                "result",
            ]

            for sentence in sentences:
                if any(
                    indicator in sentence.lower() for indicator in insight_indicators
                ):
                    insights.append(sentence)
                    if len(insights) >= max_insights:
                        break

            if not insights:
                # Fallback to first few substantial sentences
                insights = [s for s in sentences if len(s) > 50][:max_insights]

            return "Key Insights:\n" + "\n".join(f"• {insight}" for insight in insights)

        def compare_options(
            option_a: str, option_b: str, criteria: str = "general"
        ) -> str:
            """Compare two options based on specified criteria."""
            return f"""Comparison Analysis (Criteria: {criteria}):

Option A: {option_a}
Option B: {option_b}

Analysis Framework:
1. Advantages and disadvantages of each option
2. Risk assessment and mitigation strategies
3. Implementation complexity and resource requirements
4. Long-term implications and scalability
5. Alignment with objectives and constraints

Recommendation:
Based on the {criteria} criteria, a detailed comparison would consider multiple factors including feasibility, impact, and strategic alignment."""

        tools.extend([extract_key_insights, compare_options])
        return tools

    def get_model(self) -> str:
        """Get the model optimized for reasoning tasks."""
        # Default to Claude-3 models if available, or fall back to GPT-4 for reasoning
        return self.agent_config.model or "gpt-4-turbo"

    async def _simulate(self, task: str) -> str:
        model = self.get_model()
        return f"Claude Agent Analysis: {task} (model {model})"
