"""
Prompt Optimizer - Applies prompt engineering best practices

Transforms user prompts into optimized versions following established
prompt engineering principles including:
- Clear structure and formatting
- Specific instructions and constraints  
- Context enhancement
- Example provision when helpful
- Role/persona specification
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from .intent_analyzer import IntentAnalysis, IntentType, TechnicalDomain, UrgencyLevel

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Prompt optimization strategies."""
    STRUCTURE_ENHANCEMENT = "structure_enhancement"
    CONTEXT_ENRICHMENT = "context_enrichment" 
    INSTRUCTION_CLARIFICATION = "instruction_clarification"
    EXAMPLE_PROVISION = "example_provision"
    CONSTRAINT_SPECIFICATION = "constraint_specification"
    ROLE_DEFINITION = "role_definition"
    FORMAT_STANDARDIZATION = "format_standardization"


class OptimizationLevel(Enum):
    """Levels of optimization to apply."""
    MINIMAL = "minimal"      # Basic cleanup and structure
    STANDARD = "standard"    # Standard best practices
    ENHANCED = "enhanced"    # Advanced optimization
    COMPREHENSIVE = "comprehensive"  # Full optimization suite


@dataclass
class OptimizationRule:
    """A single optimization rule."""
    name: str
    strategy: OptimizationStrategy
    pattern: Optional[str] = None  # Regex pattern to match
    applies_to_intent: Set[IntentType] = field(default_factory=set)
    applies_to_domain: Set[TechnicalDomain] = field(default_factory=set) 
    priority: int = 0  # Higher numbers = higher priority
    confidence_threshold: float = 0.0  # Only apply if confidence is above this
    description: str = ""


@dataclass
class OptimizedPrompt:
    """Result of prompt optimization."""
    original_prompt: str
    optimized_prompt: str
    optimizations_applied: List[str]
    estimated_improvement: float  # 0.0-1.0 scale
    optimization_level: OptimizationLevel
    processing_time_ms: int
    structure_score: float = 0.0  # How well structured (0-1)
    clarity_score: float = 0.0   # How clear (0-1)
    completeness_score: float = 0.0  # How complete (0-1)
    metadata: Dict = field(default_factory=dict)


class PromptOptimizer:
    """
    Advanced prompt optimizer applying engineering best practices.
    
    Transforms user prompts to be more effective for AI systems while
    preserving the original intent and adding helpful structure.
    """
    
    def __init__(self, default_level: OptimizationLevel = OptimizationLevel.STANDARD):
        """Initialize the prompt optimizer."""
        self.default_level = default_level
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Build optimization components
        self.optimization_rules = self._build_optimization_rules()
        self.template_library = self._build_template_library()
        self.enhancement_patterns = self._build_enhancement_patterns()
        self.role_definitions = self._build_role_definitions()
        self.format_templates = self._build_format_templates()
        
        # Statistics
        self.prompts_optimized = 0
        self.optimization_stats = {strategy: 0 for strategy in OptimizationStrategy}
        
        self.logger.info(f"PromptOptimizer initialized with level: {default_level.value}")
    
    async def optimize_prompt(self, 
                            prompt: str,
                            intent_analysis: IntentAnalysis,
                            level: Optional[OptimizationLevel] = None,
                            preserve_style: bool = True,
                            add_examples: bool = True,
                            context: Optional[Dict] = None) -> OptimizedPrompt:
        """
        Optimize a user prompt based on intent analysis and best practices.
        
        Args:
            prompt: Original user prompt
            intent_analysis: Analysis of user intent
            level: Optimization level to apply
            preserve_style: Whether to preserve user's communication style
            add_examples: Whether to add helpful examples
            context: Additional context for optimization
        
        Returns:
            OptimizedPrompt with original, optimized version and metadata
        """
        start_time = asyncio.get_event_loop().time()
        self.prompts_optimized += 1
        
        if level is None:
            level = self.default_level
        
        if context is None:
            context = {}
        
        # Start with original prompt
        optimized = prompt.strip()
        applied_optimizations = []
        
        # Apply optimizations based on level
        if level in [OptimizationLevel.MINIMAL, OptimizationLevel.STANDARD, 
                    OptimizationLevel.ENHANCED, OptimizationLevel.COMPREHENSIVE]:
            optimized, minimal_opts = await self._apply_minimal_optimization(optimized, intent_analysis)
            applied_optimizations.extend(minimal_opts)
        
        if level in [OptimizationLevel.STANDARD, OptimizationLevel.ENHANCED, OptimizationLevel.COMPREHENSIVE]:
            optimized, standard_opts = await self._apply_standard_optimization(
                optimized, intent_analysis, preserve_style, context
            )
            applied_optimizations.extend(standard_opts)
        
        if level in [OptimizationLevel.ENHANCED, OptimizationLevel.COMPREHENSIVE]:
            optimized, enhanced_opts = await self._apply_enhanced_optimization(
                optimized, intent_analysis, add_examples, context
            )
            applied_optimizations.extend(enhanced_opts)
        
        if level == OptimizationLevel.COMPREHENSIVE:
            optimized, comprehensive_opts = await self._apply_comprehensive_optimization(
                optimized, intent_analysis, context
            )
            applied_optimizations.extend(comprehensive_opts)
        
        # Calculate quality scores
        structure_score = self._calculate_structure_score(optimized, intent_analysis)
        clarity_score = self._calculate_clarity_score(optimized, intent_analysis)
        completeness_score = self._calculate_completeness_score(optimized, intent_analysis)
        
        # Estimate improvement
        estimated_improvement = self._estimate_improvement(
            prompt, optimized, applied_optimizations, intent_analysis
        )
        
        processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        # Update statistics
        for opt in applied_optimizations:
            if opt in [strategy.value for strategy in OptimizationStrategy]:
                self.optimization_stats[OptimizationStrategy(opt)] += 1
        
        result = OptimizedPrompt(
            original_prompt=prompt,
            optimized_prompt=optimized,
            optimizations_applied=applied_optimizations,
            estimated_improvement=estimated_improvement,
            optimization_level=level,
            processing_time_ms=processing_time,
            structure_score=structure_score,
            clarity_score=clarity_score,
            completeness_score=completeness_score,
            metadata={
                "original_length": len(prompt),
                "optimized_length": len(optimized),
                "length_change": len(optimized) - len(prompt),
                "intent": intent_analysis.primary_intent.value,
                "domain": intent_analysis.technical_domain.value
            }
        )
        
        self.logger.debug(f"Prompt optimized in {processing_time}ms, improvement: {estimated_improvement:.2f}")
        
        return result
    
    async def _apply_minimal_optimization(self, prompt: str, intent_analysis: IntentAnalysis) -> Tuple[str, List[str]]:
        """Apply minimal optimization - basic cleanup and structure."""
        optimized = prompt
        applied = []
        
        # 1. Clean up formatting
        original_optimized = optimized
        optimized = self._clean_formatting(optimized)
        if optimized != original_optimized:
            applied.append("formatting_cleanup")
        
        # 2. Fix basic grammar and punctuation
        original_optimized = optimized
        optimized = self._fix_basic_grammar(optimized)
        if optimized != original_optimized:
            applied.append("grammar_fix")
        
        # 3. Add basic structure for very unstructured requests
        if self._is_unstructured(optimized):
            optimized = self._add_basic_structure(optimized, intent_analysis)
            applied.append("basic_structure")
        
        return optimized, applied
    
    async def _apply_standard_optimization(self, 
                                         prompt: str, 
                                         intent_analysis: IntentAnalysis,
                                         preserve_style: bool,
                                         context: Dict) -> Tuple[str, List[str]]:
        """Apply standard optimization - best practices."""
        optimized = prompt
        applied = []
        
        # 1. Enhance instruction clarity
        original_optimized = optimized
        optimized = self._enhance_instruction_clarity(optimized, intent_analysis)
        if optimized != original_optimized:
            applied.append("instruction_clarification")
        
        # 2. Add context if missing
        if self._needs_context_enhancement(optimized, intent_analysis):
            optimized = self._add_context_enhancement(optimized, intent_analysis, context)
            applied.append("context_enrichment")
        
        # 3. Specify constraints and requirements
        if intent_analysis.has_constraints or intent_analysis.success_criteria:
            optimized = self._enhance_constraints(optimized, intent_analysis)
            applied.append("constraint_specification")
        
        # 4. Add role specification for complex tasks
        if self._should_add_role(intent_analysis):
            optimized = self._add_role_specification(optimized, intent_analysis)
            applied.append("role_definition")
        
        return optimized, applied
    
    async def _apply_enhanced_optimization(self, 
                                         prompt: str,
                                         intent_analysis: IntentAnalysis,
                                         add_examples: bool,
                                         context: Dict) -> Tuple[str, List[str]]:
        """Apply enhanced optimization - advanced techniques."""
        optimized = prompt
        applied = []
        
        # 1. Add examples when helpful
        if add_examples and self._should_add_examples(intent_analysis):
            examples = self._generate_relevant_examples(intent_analysis)
            if examples:
                optimized = self._add_examples(optimized, examples)
                applied.append("example_provision")
        
        # 2. Enhance technical specifications
        if intent_analysis.technical_domain != TechnicalDomain.NON_TECHNICAL:
            optimized = self._enhance_technical_specs(optimized, intent_analysis)
            applied.append("technical_enhancement")
        
        # 3. Add output format specification
        optimized = self._add_output_format(optimized, intent_analysis)
            applied.append("format_standardization")
        
        # 4. Improve prompt structure with sections
        optimized = self._improve_structure_with_sections(optimized, intent_analysis)
        applied.append("structure_enhancement")
        
        return optimized, applied
    
    async def _apply_comprehensive_optimization(self, 
                                              prompt: str,
                                              intent_analysis: IntentAnalysis,
                                              context: Dict) -> Tuple[str, List[str]]:
        """Apply comprehensive optimization - full suite."""
        optimized = prompt
        applied = []
        
        # 1. Add comprehensive persona and context
        optimized = self._add_comprehensive_persona(optimized, intent_analysis)
        applied.append("comprehensive_persona")
        
        # 2. Add quality requirements and success criteria
        optimized = self._add_quality_requirements(optimized, intent_analysis)
        applied.append("quality_requirements")
        
        # 3. Add thinking framework for complex tasks
        if intent_analysis.complexity_level in ["high", "very_high"]:
            optimized = self._add_thinking_framework(optimized, intent_analysis)
            applied.append("thinking_framework")
        
        # 4. Add validation and error handling guidance
        optimized = self._add_validation_guidance(optimized, intent_analysis)
        applied.append("validation_guidance")
        
        return optimized, applied
    
    def _clean_formatting(self, prompt: str) -> str:
        """Clean up basic formatting issues."""
        # Remove excessive whitespace
        prompt = re.sub(r'\s+', ' ', prompt)
        
        # Fix spacing around punctuation
        prompt = re.sub(r'\s+([.!?])', r'\1', prompt)
        prompt = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', prompt)
        
        # Remove trailing/leading whitespace
        prompt = prompt.strip()
        
        return prompt
    
    def _fix_basic_grammar(self, prompt: str) -> str:
        """Fix basic grammar issues."""
        # Capitalize first letter
        if prompt and not prompt[0].isupper():
            prompt = prompt[0].upper() + prompt[1:]
        
        # Ensure proper sentence ending
        if prompt and prompt[-1] not in '.!?':
            prompt += '.'
        
        # Fix common contractions
        contractions = {
            r'\bi\b': 'I',
            r'\bcant\b': "can't",
            r'\bdont\b': "don't", 
            r'\bwont\b': "won't",
            r'\byour\s+welcome\b': "you're welcome"
        }
        
        for pattern, replacement in contractions.items():
            prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
        
        return prompt
    
    def _is_unstructured(self, prompt: str) -> bool:
        """Check if prompt lacks basic structure."""
        # Very short prompts (< 10 words) might need structure
        word_count = len(prompt.split())
        if word_count < 5:
            return True
        
        # No clear action verbs
        action_verbs = re.findall(r'\b(create|build|make|write|analyze|fix|help|show|explain)\b', prompt, re.IGNORECASE)
        if not action_verbs:
            return True
        
        # Just greetings or questions without clear intent
        if re.match(r'^\s*(hello|hi|hey|what|how)\b.*\?\s*$', prompt, re.IGNORECASE):
            return True
        
        return False
    
    def _add_basic_structure(self, prompt: str, intent_analysis: IntentAnalysis) -> str:
        """Add basic structure to unstructured prompts.""" 
        if intent_analysis.primary_intent == IntentType.INFORMATION_SEEKING:
            if not prompt.endswith('?'):
                prompt += "?"
            return f"Please help me understand: {prompt}"
        
        elif intent_analysis.primary_intent == IntentType.TASK_EXECUTION:
            return f"I need you to {prompt.lower().strip('.')}"
        
        elif intent_analysis.primary_intent == IntentType.PROBLEM_SOLVING:
            return f"I'm having an issue: {prompt.lower().strip('.')}. Can you help me resolve this?"
        
        return prompt
    
    def _enhance_instruction_clarity(self, prompt: str, intent_analysis: IntentAnalysis) -> str:
        """Make instructions clearer and more specific."""
        # Replace vague terms with specific ones
        vague_replacements = {
            r'\bdo something\b': 'complete the specific task',
            r'\bmake it better\b': 'improve the quality and functionality',
            r'\bfix it\b': 'identify and resolve the issues',
            r'\bhelp me\b': 'provide assistance to',
            r'\bthing\b': 'component',
            r'\bstuff\b': 'elements'
        }
        
        for pattern, replacement in vague_replacements.items():
            prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
        
        # Add action-oriented language for task execution
        if intent_analysis.primary_intent == IntentType.TASK_EXECUTION:
            if not re.search(r'\b(please|can you|would you|i need you to)\b', prompt, re.IGNORECASE):
                prompt = f"Please {prompt.lower().strip('.')}"
        
        return prompt
    
    def _needs_context_enhancement(self, prompt: str, intent_analysis: IntentAnalysis) -> bool:
        """Check if prompt needs context enhancement."""
        return (
            len(intent_analysis.missing_context) > 0 or
            len(intent_analysis.ambiguous_terms) > 0 or
            (intent_analysis.technical_domain != TechnicalDomain.NON_TECHNICAL and
             len(intent_analysis.technologies) == 0)
        )
    
    def _add_context_enhancement(self, prompt: str, intent_analysis: IntentAnalysis, context: Dict) -> str:
        """Add helpful context to the prompt."""
        enhancements = []
        
        # Add domain context
        if intent_analysis.technical_domain != TechnicalDomain.NON_TECHNICAL:
            domain_name = intent_analysis.technical_domain.value.replace('_', ' ')
            enhancements.append(f"Context: This is a {domain_name} task.")
        
        # Add complexity context
        if intent_analysis.complexity_level in ["high", "very_high"]:
            enhancements.append("Note: This appears to be a complex task that may require multiple steps.")
        
        # Add urgency context
        if intent_analysis.urgency in [UrgencyLevel.HIGH, UrgencyLevel.CRITICAL]:
            urgency_text = "high priority" if intent_analysis.urgency == UrgencyLevel.HIGH else "critical"
            enhancements.append(f"Priority: This is a {urgency_text} request.")
        
        if enhancements:
            return f"{prompt}\n\n{' '.join(enhancements)}"
        
        return prompt
    
    def _enhance_constraints(self, prompt: str, intent_analysis: IntentAnalysis) -> str:
        """Enhance constraint and requirement specification."""
        additions = []
        
        # Add success criteria if identified
        if intent_analysis.success_criteria:
            criteria_text = "; ".join(intent_analysis.success_criteria[:3])
            additions.append(f"Success criteria: {criteria_text}")
        
        # Add constraint information
        if intent_analysis.constraint_types:
            constraint_text = ", ".join(intent_analysis.constraint_types)
            additions.append(f"Constraints to consider: {constraint_text}")
        
        # Add quality requirements for code tasks
        if intent_analysis.technical_domain in [TechnicalDomain.SOFTWARE_DEVELOPMENT, TechnicalDomain.WEB_DEVELOPMENT]:
            additions.append("Please ensure the solution follows best practices for code quality, security, and maintainability.")
        
        if additions:
            return f"{prompt}\n\nRequirements:\n" + "\n".join([f"- {addition}" for addition in additions])
        
        return prompt
    
    def _should_add_role(self, intent_analysis: IntentAnalysis) -> bool:
        """Determine if role specification would be helpful."""
        return (
            intent_analysis.technical_domain != TechnicalDomain.NON_TECHNICAL or
            intent_analysis.complexity_level in ["high", "very_high"] or
            intent_analysis.primary_intent in [IntentType.ANALYSIS_REVIEW, IntentType.CREATIVE_GENERATION]
        )
    
    def _add_role_specification(self, prompt: str, intent_analysis: IntentAnalysis) -> str:
        """Add appropriate role specification."""
        domain = intent_analysis.technical_domain
        intent = intent_analysis.primary_intent
        
        # Select appropriate role
        role = "helpful assistant"
        
        if domain == TechnicalDomain.SOFTWARE_DEVELOPMENT:
            role = "expert software developer"
        elif domain == TechnicalDomain.WEB_DEVELOPMENT:
            role = "experienced web developer"
        elif domain == TechnicalDomain.DATA_SCIENCE:
            role = "skilled data scientist"
        elif domain == TechnicalDomain.DEVOPS_INFRASTRUCTURE:
            role = "DevOps engineer"
        elif domain == TechnicalDomain.SECURITY_PRIVACY:
            role = "cybersecurity expert"
        elif intent == IntentType.ANALYSIS_REVIEW:
            role = "thorough analyst"
        elif intent == IntentType.CREATIVE_GENERATION:
            role = "creative problem solver"
        
        if role != "helpful assistant":
            return f"Acting as an {role}, {prompt.lower().strip('.')}"
        
        return prompt
    
    def _should_add_examples(self, intent_analysis: IntentAnalysis) -> bool:
        """Determine if examples would be helpful.""" 
        return (
            intent_analysis.primary_intent in [IntentType.TASK_EXECUTION, IntentType.LEARNING_TUTORIAL] or
            intent_analysis.complexity_level in ["medium", "high"] or
            len(intent_analysis.ambiguous_terms) > 0
        )
    
    def _generate_relevant_examples(self, intent_analysis: IntentAnalysis) -> List[str]:
        """Generate relevant examples based on intent analysis."""
        examples = []
        
        # Examples based on technical domain
        if intent_analysis.technical_domain == TechnicalDomain.SOFTWARE_DEVELOPMENT:
            if "function" in intent_analysis.keywords:
                examples.append("For example, a function that processes user input and returns validated data.")
        
        # Examples based on intent type
        if intent_analysis.primary_intent == IntentType.ANALYSIS_REVIEW:
            examples.append("Include specific findings, recommendations, and potential risks or improvements.")
        
        return examples[:2]  # Limit to 2 examples
    
    def _add_examples(self, prompt: str, examples: List[str]) -> str:
        """Add examples to the prompt."""
        if not examples:
            return prompt
        
        example_text = "\n".join([f"Example: {ex}" for ex in examples])
        return f"{prompt}\n\n{example_text}"
    
    def _enhance_technical_specs(self, prompt: str, intent_analysis: IntentAnalysis) -> str:
        """Enhance technical specifications in the prompt."""
        additions = []
        
        # Add technology context if available
        if intent_analysis.technologies:
            tech_list = ", ".join(intent_analysis.technologies[:5])
            additions.append(f"Technologies involved: {tech_list}")
        
        # Add domain-specific guidance
        domain_guidance = {
            TechnicalDomain.SOFTWARE_DEVELOPMENT: "Please include proper error handling, documentation, and testing considerations.",
            TechnicalDomain.WEB_DEVELOPMENT: "Consider responsive design, accessibility, and cross-browser compatibility.",
            TechnicalDomain.DATA_SCIENCE: "Include data validation, statistical significance, and interpretation of results.",
            TechnicalDomain.SECURITY_PRIVACY: "Prioritize security best practices and compliance requirements."
        }
        
        if intent_analysis.technical_domain in domain_guidance:
            additions.append(domain_guidance[intent_analysis.technical_domain])
        
        if additions:
            return f"{prompt}\n\nTechnical considerations:\n" + "\n".join([f"- {addition}" for addition in additions])
        
        return prompt
    
    def _add_output_format(self, prompt: str, intent_analysis: IntentAnalysis) -> str:
        """Add output format specification."""
        format_specs = []
        
        # Format based on intent
        if intent_analysis.primary_intent == IntentType.ANALYSIS_REVIEW:
            format_specs.append("Provide a structured analysis with clear sections for findings and recommendations.")
        
        elif intent_analysis.primary_intent == IntentType.TASK_EXECUTION:
            if intent_analysis.technical_domain in [TechnicalDomain.SOFTWARE_DEVELOPMENT, TechnicalDomain.WEB_DEVELOPMENT]:
                format_specs.append("Include working code with clear comments and explanation.")
        
        elif intent_analysis.primary_intent == IntentType.LEARNING_TUTORIAL:
            format_specs.append("Structure as a step-by-step guide with clear explanations.")
        
        # General formatting
        if intent_analysis.complexity_level in ["high", "very_high"]:
            format_specs.append("Use clear headings and organize information logically.")
        
        if format_specs:
            return f"{prompt}\n\nOutput format: {' '.join(format_specs)}"
        
        return prompt
    
    def _improve_structure_with_sections(self, prompt: str, intent_analysis: IntentAnalysis) -> str:
        """Improve prompt structure by adding sections."""
        # For complex tasks, suggest breaking into sections
        if intent_analysis.complexity_level in ["high", "very_high"] and len(prompt.split()) > 50:
            return f"{prompt}\n\nPlease organize your response into clear sections addressing each aspect of this request."
        
        return prompt
    
    def _add_comprehensive_persona(self, prompt: str, intent_analysis: IntentAnalysis) -> str:
        """Add comprehensive persona and context."""
        domain = intent_analysis.technical_domain
        
        persona_templates = {
            TechnicalDomain.SOFTWARE_DEVELOPMENT: "You are a senior software engineer with expertise in multiple programming languages and frameworks. You follow industry best practices and write clean, maintainable code.",
            TechnicalDomain.WEB_DEVELOPMENT: "You are a full-stack web developer with extensive experience in modern web technologies. You prioritize user experience, performance, and accessibility.",
            TechnicalDomain.DATA_SCIENCE: "You are a data scientist with strong analytical skills and experience in statistical modeling. You ensure data quality and provide actionable insights.",
            TechnicalDomain.DEVOPS_INFRASTRUCTURE: "You are a DevOps engineer focused on automation, scalability, and reliability. You implement infrastructure as code and follow security best practices.",
        }
        
        if domain in persona_templates:
            return f"{persona_templates[domain]}\n\n{prompt}"
        
        return prompt
    
    def _add_quality_requirements(self, prompt: str, intent_analysis: IntentAnalysis) -> str:
        """Add comprehensive quality requirements.""" 
        quality_requirements = [
            "Ensure high quality in your response",
            "Double-check for accuracy and completeness",
            "Provide clear explanations for your reasoning"
        ]
        
        # Add domain-specific quality requirements
        if intent_analysis.technical_domain == TechnicalDomain.SOFTWARE_DEVELOPMENT:
            quality_requirements.extend([
                "Follow coding best practices and standards",
                "Include appropriate error handling",
                "Consider performance and scalability"
            ])
        
        requirements_text = "\n".join([f"- {req}" for req in quality_requirements])
        return f"{prompt}\n\nQuality requirements:\n{requirements_text}"
    
    def _add_thinking_framework(self, prompt: str, intent_analysis: IntentAnalysis) -> str:
        """Add thinking framework for complex tasks."""
        framework = """
Please approach this systematically:
1. Break down the problem/request into key components
2. Consider potential approaches and trade-offs  
3. Identify dependencies and constraints
4. Provide a clear, step-by-step solution
5. Include validation and testing considerations
"""
        return f"{prompt}\n\n{framework}"
    
    def _add_validation_guidance(self, prompt: str, intent_analysis: IntentAnalysis) -> str:
        """Add validation and error handling guidance."""
        if intent_analysis.primary_intent in [IntentType.TASK_EXECUTION, IntentType.PROBLEM_SOLVING]:
            validation_text = "\n\nPlease include validation steps and explain how to verify the solution works correctly."
            return prompt + validation_text
        
        return prompt
    
    def _calculate_structure_score(self, prompt: str, intent_analysis: IntentAnalysis) -> float:
        """Calculate how well structured the prompt is (0-1)."""
        score = 0.0
        
        # Check for clear sections/organization
        if re.search(r'(requirements|context|example)', prompt, re.IGNORECASE):
            score += 0.3
        
        # Check for proper formatting
        if '\n' in prompt:
            score += 0.2
        
        # Check for specific instructions
        action_verbs = len(re.findall(r'\b(please|create|build|analyze|explain|provide)\b', prompt, re.IGNORECASE))
        score += min(action_verbs * 0.1, 0.3)
        
        # Check for constraints/requirements
        if re.search(r'(must|should|need to|requirement)', prompt, re.IGNORECASE):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_clarity_score(self, prompt: str, intent_analysis: IntentAnalysis) -> float:
        """Calculate how clear the prompt is (0-1)."""
        score = 0.5  # Base score
        
        # Penalty for ambiguous terms
        score -= len(intent_analysis.ambiguous_terms) * 0.05
        
        # Bonus for specific terms
        specific_terms = len(intent_analysis.technologies) + len(intent_analysis.entities)
        score += min(specific_terms * 0.05, 0.2)
        
        # Bonus for clear action verbs
        score += min(len(intent_analysis.action_verbs) * 0.1, 0.3)
        
        return max(min(score, 1.0), 0.0)
    
    def _calculate_completeness_score(self, prompt: str, intent_analysis: IntentAnalysis) -> float:
        """Calculate how complete the prompt is (0-1)."""
        score = 0.3  # Base score
        
        # Bonus for success criteria
        if intent_analysis.success_criteria:
            score += 0.2
        
        # Bonus for constraints
        if intent_analysis.has_constraints:
            score += 0.15
        
        # Bonus for context information
        if not intent_analysis.missing_context:
            score += 0.2
        
        # Bonus for technical specifications
        if intent_analysis.technologies:
            score += 0.15
        
        return min(score, 1.0)
    
    def _estimate_improvement(self, original: str, optimized: str, optimizations: List[str], intent_analysis: IntentAnalysis) -> float:
        """Estimate the improvement from optimization (0-1 scale)."""
        base_improvement = len(optimizations) * 0.1
        
        # Length improvement (moderate)
        length_ratio = len(optimized) / max(len(original), 1)
        if 1.2 <= length_ratio <= 2.0:  # Good expansion
            base_improvement += 0.1
        elif length_ratio > 2.0:  # Too much expansion
            base_improvement -= 0.05
        
        # Specific optimization bonuses
        if "structure_enhancement" in optimizations:
            base_improvement += 0.15
        if "context_enrichment" in optimizations:
            base_improvement += 0.1
        if "role_definition" in optimizations:
            base_improvement += 0.08
        
        # Intent-based bonuses
        if intent_analysis.primary_intent == IntentType.TASK_EXECUTION and "constraint_specification" in optimizations:
            base_improvement += 0.1
        
        return min(base_improvement, 1.0)
    
    def _build_optimization_rules(self) -> List[OptimizationRule]:
        """Build library of optimization rules."""
        return [
            OptimizationRule(
                name="add_role_for_technical_tasks",
                strategy=OptimizationStrategy.ROLE_DEFINITION,
                applies_to_domain={TechnicalDomain.SOFTWARE_DEVELOPMENT, TechnicalDomain.WEB_DEVELOPMENT},
                priority=8,
                description="Add role specification for technical tasks"
            ),
            OptimizationRule(
                name="structure_complex_tasks",
                strategy=OptimizationStrategy.STRUCTURE_ENHANCEMENT,
                applies_to_intent={IntentType.TASK_EXECUTION},
                priority=7,
                description="Add structure to complex task requests"
            ),
            OptimizationRule(
                name="clarify_vague_instructions",
                strategy=OptimizationStrategy.INSTRUCTION_CLARIFICATION,
                pattern=r'\b(thing|stuff|it|this)\b',
                priority=9,
                description="Clarify vague terms and references"
            )
        ]
    
    def _build_template_library(self) -> Dict[str, str]:
        """Build template library for different use cases."""
        return {
            "code_task": """As an expert {language} developer, please {task}.

Requirements:
- Follow best practices and coding standards
- Include proper error handling
- Add clear comments and documentation
- Consider performance and maintainability

Please provide working code with explanation.""",
            
            "analysis_task": """Please provide a comprehensive analysis of {subject}.

Structure your response with:
1. Executive summary
2. Key findings
3. Detailed analysis
4. Recommendations
5. Risk considerations

Include specific examples and actionable insights.""",
            
            "learning_task": """Please create a step-by-step guide for {topic}.

Format:
- Clear learning objectives
- Prerequisites (if any)
- Step-by-step instructions
- Examples and practice exercises
- Common pitfalls to avoid
- Further resources"""
        }
    
    def _build_enhancement_patterns(self) -> Dict[str, Dict]:
        """Build patterns for prompt enhancement."""
        return {
            "vague_terms": {
                "patterns": [r'\b(thing|stuff|it|this|that)\b'],
                "replacements": ["component", "element", "the specific item", "the mentioned item", "the referenced item"]
            },
            "weak_verbs": {
                "patterns": [r'\b(do|make|get|have)\b'],
                "replacements": ["create", "implement", "obtain", "establish"]
            }
        }
    
    def _build_role_definitions(self) -> Dict[TechnicalDomain, str]:
        """Build role definitions by technical domain."""
        return {
            TechnicalDomain.SOFTWARE_DEVELOPMENT: "expert software developer with deep knowledge of programming languages, design patterns, and best practices",
            TechnicalDomain.WEB_DEVELOPMENT: "experienced full-stack web developer proficient in modern frameworks and web technologies",
            TechnicalDomain.DATA_SCIENCE: "skilled data scientist with expertise in statistical analysis, machine learning, and data visualization",
            TechnicalDomain.DEVOPS_INFRASTRUCTURE: "DevOps engineer specializing in automation, cloud infrastructure, and system reliability",
            TechnicalDomain.SECURITY_PRIVACY: "cybersecurity expert focused on threat analysis, secure coding, and compliance"
        }
    
    def _build_format_templates(self) -> Dict[IntentType, str]:
        """Build format templates by intent type."""
        return {
            IntentType.TASK_EXECUTION: "Please provide a complete solution including implementation details, testing approach, and deployment considerations.",
            IntentType.ANALYSIS_REVIEW: "Structure your analysis with clear sections, specific findings, and actionable recommendations.",
            IntentType.PROBLEM_SOLVING: "Identify the root cause, explain your debugging approach, and provide a comprehensive solution.",
            IntentType.LEARNING_TUTORIAL: "Create a structured learning path with examples, exercises, and progressive complexity."
        }
    
    def get_optimizer_stats(self) -> Dict:
        """Get prompt optimizer statistics."""
        return {
            "prompts_optimized": self.prompts_optimized,
            "optimization_distribution": {
                strategy.value: count for strategy, count in self.optimization_stats.items()
            },
            "most_common_optimization": max(self.optimization_stats.items(), key=lambda x: x[1])[0].value if self.prompts_optimized > 0 else None
        }