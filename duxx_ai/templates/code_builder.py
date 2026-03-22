"""Software engineering agent template.

Provides an AI software engineer capable of writing clean code,
creating tests, debugging, refactoring, and applying best practices
across multiple programming languages.

Usage::

    from duxx_ai.templates.code_builder import CodeBuilderAgent

    agent = CodeBuilderAgent.create()
    result = await agent.run("Refactor the authentication module to use dependency injection.")
"""

from __future__ import annotations

from duxx_ai.core.agent import AgentConfig
from duxx_ai.core.llm import LLMConfig
from duxx_ai.core.tool import Tool
from duxx_ai.governance.guardrails import (
    ContentFilterGuardrail,
    GuardrailChain,
    PromptInjectionGuardrail,
)
from duxx_ai.tools.builtin import get_builtin_tools

from ._base import AgentTemplate

_SYSTEM_PROMPT = """\
You are a software engineering agent. Your core capabilities include:

1. **Clean Code**: Write well-structured, readable, and maintainable code
   following language-specific conventions and SOLID principles. Use
   meaningful names, small functions, and clear abstractions.
2. **Testing**: Create comprehensive test suites including unit tests,
   integration tests, and edge case coverage. Apply TDD methodology
   when appropriate. Use mocking and fixtures effectively.
3. **Debugging**: Systematically diagnose and fix bugs using root cause
   analysis. Add logging, reproduce issues, trace execution paths, and
   validate fixes with tests.
4. **Refactoring**: Improve code structure without changing behavior.
   Identify code smells, reduce complexity, extract reusable components,
   and improve performance.
5. **Best Practices**: Apply design patterns, follow the principle of
   least surprise, write idiomatic code, handle errors gracefully, and
   document public APIs.
6. **Code Review**: Analyze code for correctness, security vulnerabilities,
   performance issues, and adherence to coding standards.

Guidelines:
- Always explain your reasoning and design decisions.
- Write code that is self-documenting with clear variable and function names.
- Include error handling and input validation in all code.
- Follow the DRY (Don't Repeat Yourself) principle.
- Consider edge cases and boundary conditions.
- Write tests before or alongside implementation code.
- Never execute destructive system commands.
- Prefer safe, reversible operations over irreversible ones.
"""

_BLOCKED_PATTERNS = [
    "rm -rf /",
    "DROP TABLE",
    "FORMAT C:",
]


class CodeBuilderAgent(AgentTemplate):
    """Software engineering agent for code creation and maintenance.

    Provides AI-powered software engineering support with code writing,
    testing, debugging, and refactoring capabilities. Includes content
    filtering to block dangerous command patterns.

    Usage::

        agent = CodeBuilderAgent.create()
        result = await agent.run("Write unit tests for the user service module.")
    """

    name: str = "code_builder"
    description: str = "Software engineer — clean code, tests, debugging, refactoring, and best practices."
    category: str = "Engineering"
    default_tools: list[str] = [
        "python_exec",
        "bash_exec",
        "read_file",
        "write_file",
        "list_files",
        "web_request",
    ]

    @classmethod
    def _build_config(cls, llm_config: LLMConfig | None = None, **kwargs) -> AgentConfig:
        return AgentConfig(
            name=cls.name,
            description=cls.description,
            system_prompt=_SYSTEM_PROMPT,
            llm_config=llm_config or LLMConfig(),
            **kwargs,
        )

    @classmethod
    def _build_tools(cls, extra_tools: list[Tool] | None = None) -> list[Tool]:
        tools = get_builtin_tools([
            "python_exec",
            "bash_exec",
            "read_file",
            "write_file",
            "list_files",
            "web_request",
        ])
        try:
            from duxx_ai.tools.engineering import MODULE_TOOLS

            tools.extend(MODULE_TOOLS.values())
        except ImportError:
            pass
        if extra_tools:
            tools.extend(extra_tools)
        return tools

    @classmethod
    def _build_guardrails(cls) -> GuardrailChain:
        chain = GuardrailChain()
        chain.add(PromptInjectionGuardrail())
        chain.add(ContentFilterGuardrail(blocked_patterns=_BLOCKED_PATTERNS))
        return chain
