"""Structured prompt engineering system for professional AI interactions."""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Categories of request types for role selection."""
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    CODE_REVIEW = "code_review"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    TROUBLESHOOTING = "troubleshooting"
    IMPLEMENTATION = "implementation"
    OPTIMIZATION = "optimization"
    SECURITY_AUDIT = "security_audit"
    GENERAL = "general"


@dataclass
class PromptComponent:
    """Individual component of a structured prompt."""
    role: str
    goal: str
    context: Dict[str, Any]
    task: str
    constraints: List[str]
    output_format: Dict[str, Any]
    examples: Optional[List[str]] = None
    instruction: Optional[str] = None


class StructuredPromptBuilder:
    """Builder for creating structured, professional prompts."""

    @staticmethod
    def classify_request(user_input: str) -> RequestType:
        """Classify the user request to select appropriate role and template."""
        input_lower = user_input.lower()
        
        # Keywords mapping to request types
        keywords_map = {
            RequestType.COMPETITIVE_ANALYSIS: [
                'competitive', 'competition', 'competitor', 'advantage', 'market position',
                'differentiation', 'compare', 'comparison', 'vs', 'versus', 'edge',
                'unique selling', 'positioning', 'market analysis'
            ],
            RequestType.CODE_REVIEW: [
                'review', 'code review', 'feedback', 'improve code', 'best practices',
                'refactor', 'clean code', 'quality', 'bugs', 'issues'
            ],
            RequestType.ARCHITECTURE: [
                'architecture', 'design', 'system design', 'structure', 'patterns',
                'scalability', 'modular', 'components', 'dependencies'
            ],
            RequestType.DOCUMENTATION: [
                'document', 'documentation', 'readme', 'guide', 'tutorial',
                'manual', 'write docs', 'create documentation'
            ],
            RequestType.TROUBLESHOOTING: [
                'debug', 'error', 'bug', 'issue', 'problem', 'fix', 'troubleshoot',
                'not working', 'broken', 'fails', 'crash'
            ],
            RequestType.IMPLEMENTATION: [
                'implement', 'build', 'create', 'develop', 'add feature',
                'make', 'write code', 'coding', 'program'
            ],
            RequestType.OPTIMIZATION: [
                'optimize', 'performance', 'speed up', 'efficient', 'faster',
                'memory', 'resource usage', 'benchmark', 'improve performance'
            ],
            RequestType.SECURITY_AUDIT: [
                'security', 'vulnerability', 'secure', 'audit', 'penetration',
                'threat', 'risk', 'exploit', 'safe'
            ]
        }
        
        # Score each request type based on keyword matches
        scores = {}
        for req_type, keywords in keywords_map.items():
            score = sum(1 for keyword in keywords if keyword in input_lower)
            if score > 0:
                scores[req_type] = score
        
        # Return the highest scoring type, or GENERAL if no clear match
        if scores:
            return max(scores, key=scores.get)
        return RequestType.GENERAL

    @staticmethod
    def build_role_template(request_type: RequestType) -> Dict[str, str]:
        """Build role-specific template based on request type."""
        
        templates = {
            RequestType.COMPETITIVE_ANALYSIS: {
                "role": "You are an experienced software architect, business strategist, and competitive analyst with deep expertise in technology markets and product differentiation.",
                "goal": "Analyze this project's competitive advantages and market positioning with specific evidence and actionable insights.",
                "task_prefix": "Provide comprehensive competitive analysis",
                "constraints_template": [
                    "Focus on technical differentiation and unique capabilities",
                    "Use evidence from the codebase and project structure",
                    "Identify 3-5 key competitive advantages",
                    "Provide actionable recommendations"
                ],
                "output_structure": {
                    "structure": "markdown with clear sections",
                    "sections": [
                        "Executive Summary", 
                        "Technical Advantages", 
                        "Competitive Positioning", 
                        "Market Differentiation",
                        "Recommendations"
                    ]
                }
            },
            RequestType.CODE_REVIEW: {
                "role": "You are a senior software engineer and code quality expert with extensive experience in best practices, security, and performance optimization.",
                "goal": "Provide thorough code review with specific improvements and rationale.",
                "task_prefix": "Review the code and provide detailed feedback",
                "constraints_template": [
                    "Focus on correctness, security, performance, and maintainability",
                    "Provide specific examples and suggestions",
                    "Prioritize issues by severity",
                    "Include positive observations"
                ],
                "output_structure": {
                    "structure": "structured review format",
                    "sections": [
                        "Summary", 
                        "Critical Issues", 
                        "Improvements", 
                        "Best Practices",
                        "Recommendations"
                    ]
                }
            },
            RequestType.ARCHITECTURE: {
                "role": "You are a system architect and software design expert with deep knowledge of design patterns, scalability, and maintainable systems.",
                "goal": "Design or analyze system architecture with clear rationale and best practices.",
                "task_prefix": "Analyze the system architecture",
                "constraints_template": [
                    "Consider scalability, maintainability, and extensibility",
                    "Identify design patterns and architectural decisions",
                    "Assess component relationships and dependencies",
                    "Provide improvement recommendations"
                ],
                "output_structure": {
                    "structure": "architectural analysis format",
                    "sections": [
                        "Architecture Overview", 
                        "Component Analysis", 
                        "Design Patterns", 
                        "Dependencies",
                        "Recommendations"
                    ]
                }
            },
            RequestType.TROUBLESHOOTING: {
                "role": "You are a debugging expert and system diagnostician with extensive experience in identifying and resolving technical issues.",
                "goal": "Identify the root cause of the issue and provide step-by-step resolution.",
                "task_prefix": "Diagnose and resolve the technical issue",
                "constraints_template": [
                    "Identify root cause, not just symptoms",
                    "Provide step-by-step resolution steps",
                    "Consider edge cases and prevention",
                    "Include verification steps"
                ],
                "output_structure": {
                    "structure": "troubleshooting guide format",
                    "sections": [
                        "Issue Analysis", 
                        "Root Cause", 
                        "Resolution Steps", 
                        "Verification",
                        "Prevention"
                    ]
                }
            },
            RequestType.GENERAL: {
                "role": "You are an expert software engineer and technical advisor with broad knowledge across multiple domains.",
                "goal": "Provide comprehensive and actionable assistance based on the specific request.",
                "task_prefix": "Address the user's request comprehensively",
                "constraints_template": [
                    "Provide accurate and detailed information",
                    "Include relevant context and examples",
                    "Consider best practices and standards",
                    "Offer practical solutions"
                ],
                "output_structure": {
                    "structure": "well-organized response",
                    "sections": ["Analysis", "Solution", "Implementation", "Recommendations"]
                }
            }
        }
        
        return templates.get(request_type, templates[RequestType.GENERAL])

    @staticmethod
    def build_structured_prompt(
        user_input: str,
        context: Dict[str, Any],
        request_type: Optional[RequestType] = None
    ) -> PromptComponent:
        """Build a structured prompt component from user input and context."""
        
        # Classify request type if not provided
        if request_type is None:
            request_type = StructuredPromptBuilder.classify_request(user_input)
        
        # Get role template
        template = StructuredPromptBuilder.build_role_template(request_type)
        
        # Build task from user input and template
        task = f"{template['task_prefix']}: {user_input}"
        
        # Build constraints
        constraints = template['constraints_template'].copy()
        if context.get('project_info'):
            constraints.append("Consider the current project context and structure")
        if context.get('conversation_history'):
            constraints.append("Build upon previous conversation context where relevant")
        
        # Create the structured prompt component
        return PromptComponent(
            role=template['role'],
            goal=template['goal'],
            context=context,
            task=task,
            constraints=constraints,
            output_format=template['output_structure'],
            instruction=f"Based on all the above components, {template['task_prefix'].lower()} with specific evidence and actionable insights."
        )

    @staticmethod
    def render_prompt(component: PromptComponent) -> str:
        """Convert structured prompt component to effective text format for LLM."""
        
        # Start with role assignment
        prompt_parts = [f"{component.role}"]
        
        # Add goal
        prompt_parts.append(f"\n**GOAL:** {component.goal}")
        
        # Add context sections
        if component.context:
            prompt_parts.append("\n**CONTEXT:**")
            
            # Project information
            if component.context.get('project_info'):
                prompt_parts.append(f"{component.context['project_info']}")
            
            # Conversation history
            if component.context.get('conversation_history'):
                prompt_parts.append(f"\n{component.context['conversation_history']}")
            
            # Additional context
            for key, value in component.context.items():
                if key not in ['project_info', 'conversation_history'] and value:
                    prompt_parts.append(f"\n**{key.upper().replace('_', ' ')}:** {value}")
        
        # Add specific task
        prompt_parts.append(f"\n**TASK:** {component.task}")
        
        # Add constraints
        if component.constraints:
            prompt_parts.append("\n**CONSTRAINTS:**")
            for constraint in component.constraints:
                prompt_parts.append(f"- {constraint}")
        
        # Add output format specification
        if component.output_format:
            prompt_parts.append("\n**OUTPUT FORMAT:**")
            if component.output_format.get('structure'):
                prompt_parts.append(f"- Structure: {component.output_format['structure']}")
            if component.output_format.get('sections'):
                sections = component.output_format['sections']
                prompt_parts.append(f"- Required sections: {', '.join(sections)}")
        
        # Add examples if provided
        if component.examples:
            prompt_parts.append("\n**EXAMPLES:**")
            for i, example in enumerate(component.examples, 1):
                prompt_parts.append(f"{i}. {example}")
        
        # Add final instruction
        if component.instruction:
            prompt_parts.append(f"\n**INSTRUCTION:** {component.instruction}")
        
        return "\n".join(prompt_parts)

    @staticmethod
    def optimize_prompt_with_structure(
        user_input: str,
        context: Dict[str, Any],
        request_type: Optional[RequestType] = None
    ) -> str:
        """Main entry point for structured prompt optimization."""
        
        try:
            # Build structured prompt component
            component = StructuredPromptBuilder.build_structured_prompt(
                user_input, context, request_type
            )
            
            # Render to effective text format
            structured_prompt = StructuredPromptBuilder.render_prompt(component)
            
            return structured_prompt
            
        except Exception as e:
            logger.error(f"Error in structured prompt optimization: {e}")
            # Fallback to basic enhancement
            return f"Please help with the following request, considering the provided context:\n\n{user_input}"


# Convenience functions for common use cases
def optimize_competitive_analysis_prompt(user_input: str, context: Dict[str, Any]) -> str:
    """Optimize prompt specifically for competitive analysis requests."""
    return StructuredPromptBuilder.optimize_prompt_with_structure(
        user_input, context, RequestType.COMPETITIVE_ANALYSIS
    )

def optimize_code_review_prompt(user_input: str, context: Dict[str, Any]) -> str:
    """Optimize prompt specifically for code review requests."""
    return StructuredPromptBuilder.optimize_prompt_with_structure(
        user_input, context, RequestType.CODE_REVIEW
    )

def optimize_architecture_prompt(user_input: str, context: Dict[str, Any]) -> str:
    """Optimize prompt specifically for architecture analysis requests."""
    return StructuredPromptBuilder.optimize_prompt_with_structure(
        user_input, context, RequestType.ARCHITECTURE
    )