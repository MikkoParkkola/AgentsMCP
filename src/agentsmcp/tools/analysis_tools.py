"""Analysis tools for OpenAI Agents SDK integration."""

import re
from typing import Any, Dict

from .base_tools import BaseTool, tool_registry


class TextAnalysisTool(BaseTool):
    """Tool for analyzing text content."""

    def __init__(self):
        super().__init__(
            name="analyze_text",
            description="Analyze text content and provide insights about structure, complexity, and characteristics.",
        )

    def execute(self, text: str, analysis_type: str = "comprehensive") -> str:
        """Analyze text content."""
        try:
            if not text.strip():
                return "Error: Empty text provided for analysis"

            # Basic metrics
            word_count = len(text.split())
            char_count = len(text)
            char_count_no_spaces = len(text.replace(" ", ""))
            line_count = len(text.split("\n"))
            paragraph_count = len([p for p in text.split("\n\n") if p.strip()])
            sentence_count = len([s for s in re.split(r"[.!?]+", text) if s.strip()])

            # Complexity analysis
            avg_words_per_sentence = word_count / max(sentence_count, 1)
            avg_chars_per_word = char_count_no_spaces / max(word_count, 1)

            # Content analysis
            technical_terms = len(
                [
                    word
                    for word in text.split()
                    if any(
                        indicator in word.lower()
                        for indicator in [
                            "api",
                            "http",
                            "json",
                            "xml",
                            "sql",
                            "html",
                            "css",
                            "js",
                        ]
                    )
                ]
            )

            code_indicators = (
                text.count("```")
                + text.count("def ")
                + text.count("class ")
                + text.count("function")
            )

            # Readability assessment
            complexity_score = "Low"
            if avg_words_per_sentence > 20 or avg_chars_per_word > 6:
                complexity_score = "High"
            elif avg_words_per_sentence > 15 or avg_chars_per_word > 5:
                complexity_score = "Medium"

            # Determine document characteristics
            doc_type = (
                "Technical/Code" 
                if code_indicators > 0 or technical_terms > word_count * 0.1 
                else "General Text"
            )
            structure_quality = (
                "Well-structured" 
                if paragraph_count > 1 and line_count > paragraph_count * 2 
                else "Dense"
            )
            appears_technical = (
                "Appears to be technical documentation or code." 
                if technical_terms > 5 or code_indicators > 0 
                else "Appears to be general text content."
            )

            result = f"""Text Analysis Report ({analysis_type}):

STRUCTURE METRICS:
- Characters: {char_count:,} (excluding spaces: {char_count_no_spaces:,})
- Words: {word_count:,}
- Sentences: {sentence_count:,}
- Lines: {line_count:,}
- Paragraphs: {paragraph_count:,}

READABILITY ANALYSIS:
- Average words per sentence: {avg_words_per_sentence:.1f}
- Average characters per word: {avg_chars_per_word:.1f}
- Complexity score: {complexity_score}

CONTENT CHARACTERISTICS:
- Technical terms detected: {technical_terms}
- Code indicators: {code_indicators}
- Document type: {doc_type}
- Structure quality: {structure_quality}

SUMMARY:
This text contains {word_count:,} words across {sentence_count:,} sentences with {complexity_score.lower()} complexity.
{appears_technical}"""

            self.logger.debug(
                f"Analyzed text: {word_count} words, {char_count} characters"
            )
            return result

        except Exception as e:
            self.logger.exception("Error analyzing text")
            return f"Error analyzing text: {str(e)}"

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text content to analyze"},
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis to perform",
                    "enum": ["basic", "comprehensive", "readability"],
                    "default": "comprehensive",
                },
            },
            "required": ["text"],
        }


class CodeAnalysisTool(BaseTool):
    """Tool for analyzing code structure and quality."""

    def __init__(self):
        super().__init__(
            name="analyze_code",
            description="Analyze code structure, quality metrics, and provide insights about maintainability.",
        )

    def execute(self, code: str, language: str = "python") -> str:
        """Analyze code structure and quality."""
        try:
            if not code.strip():
                return "Error: Empty code provided for analysis"

            lines = code.split("\n")
            total_lines = len(lines)
            non_empty_lines = len([line for line in lines if line.strip()])

            # Language-specific comment detection
            comment_patterns = {
                "python": [r"^\s*#", r'^\s*"""', r"^\s*\'\'\'"],
                "javascript": [r"^\s*//", r"^\s*/\*", r"^\s*\*"],
                "java": [r"^\s*//", r"^\s*/\*", r"^\s*\*"],
                "c": [r"^\s*//", r"^\s*/\*", r"^\s*\*"],
                "cpp": [r"^\s*//", r"^\s*/\*", r"^\s*\*"],
            }

            patterns = comment_patterns.get(language.lower(), [r"^\s*#", r"^\s*//"])
            comment_lines = 0
            for line in lines:
                if any(re.match(pattern, line) for pattern in patterns):
                    comment_lines += 1

            # Code complexity indicators
            function_defs = len(
                re.findall(r"\b(def|function|func)\s+\w+", code, re.IGNORECASE)
            )
            class_defs = len(
                re.findall(r"\b(class|interface)\s+\w+", code, re.IGNORECASE)
            )
            import_statements = len(
                re.findall(r"\b(import|from|include|using)\s+", code, re.IGNORECASE)
            )

            # Control flow complexity
            control_structures = len(
                re.findall(
                    r"\b(if|else|elif|for|while|switch|try|catch|finally)\b",
                    code,
                    re.IGNORECASE,
                )
            )

            # Calculate metrics
            comment_ratio = comment_lines / max(non_empty_lines, 1) * 100
            complexity_score = control_structures / max(non_empty_lines, 1) * 100

            # Quality assessment
            quality_scores = []
            if comment_ratio >= 20:
                quality_scores.append("Well-documented")
            elif comment_ratio >= 10:
                quality_scores.append("Moderately documented")
            else:
                quality_scores.append("Needs more documentation")

            if complexity_score < 10:
                quality_scores.append("Low complexity")
            elif complexity_score < 20:
                quality_scores.append("Medium complexity")
            else:
                quality_scores.append("High complexity")

            size_category = (
                "Large"
                if total_lines > 500
                else "Medium"
                if total_lines > 100
                else "Small"
            )

            # Determine assessment categories
            if comment_ratio >= 20:
                doc_assessment = "Excellent"
            elif comment_ratio >= 15:
                doc_assessment = "Good" 
            elif comment_ratio >= 10:
                doc_assessment = "Adequate"
            else:
                doc_assessment = "Needs improvement"

            structure_assessment = (
                "Well-organized" if function_defs > 0 or class_defs > 0 else "Script-style"
            )
            complexity_assessment = (
                "Manageable" if complexity_score < 15 else "Review recommended"
            )
            
            recommendations = self._generate_recommendations(
                comment_ratio, complexity_score, total_lines, function_defs, class_defs
            )

            result = f"""Code Analysis Report ({language}):

STRUCTURE METRICS:
- Total lines: {total_lines:,}
- Code lines (non-empty): {non_empty_lines:,}
- Comment lines: {comment_lines:,}
- Comment ratio: {comment_ratio:.1f}%

CODE ORGANIZATION:
- Functions/Methods: {function_defs}
- Classes/Interfaces: {class_defs}
- Import statements: {import_statements}
- Control structures: {control_structures}

COMPLEXITY ANALYSIS:
- Control flow complexity: {complexity_score:.1f}%
- Size category: {size_category}
- Quality indicators: {", ".join(quality_scores)}

MAINTAINABILITY ASSESSMENT:
- Documentation: {doc_assessment}
- Structure: {structure_assessment}
- Complexity: {complexity_assessment}

RECOMMENDATIONS:
{recommendations}"""

            self.logger.debug(
                f"Analyzed code: {total_lines} lines, {language} language"
            )
            return result

        except Exception as e:
            self.logger.exception("Error analyzing code")
            return f"Error analyzing code: {str(e)}"

    def _generate_recommendations(
        self,
        comment_ratio: float,
        complexity_score: float,
        total_lines: int,
        functions: int,
        classes: int,
    ) -> str:
        """Generate code improvement recommendations."""
        recommendations = []

        if comment_ratio < 10:
            recommendations.append("• Add more comments to improve code documentation")

        if complexity_score > 20:
            recommendations.append(
                "• Consider breaking down complex functions to reduce complexity"
            )

        if total_lines > 500 and (functions == 0 and classes == 0):
            recommendations.append(
                "• Consider organizing code into functions or classes for better structure"
            )

        if total_lines > 300:
            recommendations.append(
                "• Consider splitting into multiple modules for better maintainability"
            )

        if not recommendations:
            recommendations.append("• Code structure and quality appear to be good")

        return "\n".join(recommendations)

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code content to analyze"},
                "language": {
                    "type": "string",
                    "description": "Programming language of the code",
                    "enum": [
                        "python",
                        "javascript",
                        "java",
                        "c",
                        "cpp",
                        "go",
                        "rust",
                        "typescript",
                    ],
                    "default": "python",
                },
            },
            "required": ["code"],
        }


# Create and register tool instances
text_analysis_tool = TextAnalysisTool()
code_analysis_tool = CodeAnalysisTool()

tool_registry.register(text_analysis_tool)
tool_registry.register(code_analysis_tool)
