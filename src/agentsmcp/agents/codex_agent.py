from typing import Any, List

from ..config import AgentConfig, Config
from .base import BaseAgent


class CodexAgent(BaseAgent):
    """Codex agent for code generation and analysis tasks using OpenAI Agents SDK."""

    def __init__(self, agent_config: AgentConfig, global_config: Config):
        super().__init__(agent_config, global_config)

    def _get_default_instructions(self) -> str:
        """Get default instructions optimized for coding tasks."""
        return """You are Codex, an expert programming assistant specialized in:

1. **Code Generation**: Write clean, efficient, and well-documented code
2. **Code Analysis**: Review and analyze existing code for bugs, improvements, and optimization
3. **Debugging**: Identify and fix issues in code
4. **Architecture**: Design software architecture and system design
5. **Best Practices**: Follow coding standards and industry best practices

**Guidelines:**
- Write production-ready code with proper error handling
- Include clear documentation and comments
- Follow security best practices
- Optimize for maintainability and performance
- Provide step-by-step explanations for complex logic

**Code Quality Standards:**
- Use appropriate design patterns
- Follow language-specific conventions
- Include comprehensive tests when requested
- Consider edge cases and error scenarios
- Write self-documenting code with meaningful variable names

**Response Format:**
- Start with a brief explanation of your approach
- Provide clean, well-formatted code
- Include usage examples when appropriate
- Explain any complex logic or algorithms used
- Suggest improvements or alternatives when relevant"""

    def _get_custom_agent_tools(self) -> List[Any]:
        """Get custom tools specialized for coding tasks."""
        tools = []

        # Codex agent inherits all tools from registry, but can add custom ones here
        # The registry already provides read_file, write_file, and code analysis tools

        # Add any Codex-specific tools here if needed
        def lint_code(code: str, language: str = "python") -> str:
            """Perform basic linting on code."""
            lines = code.split("\n")
            issues = []

            for i, line in enumerate(lines, 1):
                # Basic linting rules
                if len(line) > 100:
                    issues.append(f"Line {i}: Line too long ({len(line)} chars)")
                if line.rstrip() != line:
                    issues.append(f"Line {i}: Trailing whitespace")
                if "\t" in line and "    " in line:
                    issues.append(f"Line {i}: Mixed tabs and spaces")

            if not issues:
                return f"Code linting complete: No issues found in {len(lines)} lines"

            return f"Code linting found {len(issues)} issues:\n" + "\n".join(
                issues[:10]
            )

        tools.append(lint_code)
        return tools

    def get_model(self) -> str:
        """Get the model optimized for coding tasks."""
        # Use the configured model or default to a code-optimized model
        return self.agent_config.model or "gpt-4-turbo"

    async def _simulate(self, task: str) -> str:
        model = self.get_model()
        return f"Codex Simulation: {task} (model {model})"

    async def cleanup(self):
        """Ensure session resources are cleared for CodexAgent."""
        # Clear any session identifiers/state to satisfy cleanup semantics in tests
        if hasattr(self, "session_id"):
            self.session_id = None  # type: ignore[assignment]
        await super().cleanup()
