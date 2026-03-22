"""Base template for enterprise agent creation.

All agent templates extend AgentTemplate to provide standardized
agent construction with sensible defaults for tools, guardrails,
and configuration.

Usage::

    from duxx_ai.templates._base import AgentTemplate

    class MyAgent(AgentTemplate):
        name = "my_agent"
        description = "Custom agent template"
        category = "general"
        default_tools = ["calculator"]

        @classmethod
        def _build_config(cls, llm_config=None, **kwargs):
            ...
"""

from __future__ import annotations

from typing import Any

from duxx_ai.core.agent import Agent, AgentConfig
from duxx_ai.core.llm import LLMConfig
from duxx_ai.core.tool import Tool
from duxx_ai.governance.guardrails import GuardrailChain


class AgentTemplate:
    """Base class for all enterprise agent templates.

    Provides a standardized interface for creating pre-configured agents
    with appropriate tools, guardrails, and system prompts for specific
    enterprise use cases.

    Subclasses must implement ``_build_config`` and ``_build_tools``.
    """

    name: str = "base"
    description: str = ""
    category: str = "general"
    default_tools: list[str] = []

    @classmethod
    def create(
        cls,
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        guardrails: GuardrailChain | None = None,
        **kwargs,
    ) -> Agent:
        """Create a fully configured agent from this template.

        Args:
            llm_config: Optional LLM configuration override.
            tools: Optional extra tools to include alongside defaults.
            guardrails: Optional guardrail chain override.
            **kwargs: Additional keyword arguments forwarded to ``_build_config``.

        Returns:
            A ready-to-use ``Agent`` instance.
        """
        config = cls._build_config(llm_config, **kwargs)
        all_tools = cls._build_tools(tools)
        chain = guardrails or cls._build_guardrails()
        return Agent(config=config, tools=all_tools, guardrails=chain)

    @classmethod
    def _build_config(cls, llm_config=None, **kwargs) -> AgentConfig:
        """Build the agent configuration. Must be implemented by subclasses."""
        raise NotImplementedError

    @classmethod
    def _build_tools(cls, extra=None) -> list[Tool]:
        """Build the tool list. Must be implemented by subclasses."""
        raise NotImplementedError

    @classmethod
    def _build_guardrails(cls) -> GuardrailChain:
        """Build the guardrail chain. Override for custom guardrails."""
        return GuardrailChain()

    @classmethod
    def info(cls) -> dict[str, Any]:
        """Return metadata about this template.

        Returns:
            Dictionary with name, description, category, and default_tools.
        """
        return {
            "name": cls.name,
            "description": cls.description,
            "category": cls.category,
            "default_tools": cls.default_tools,
        }
