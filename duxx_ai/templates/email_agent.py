"""Enterprise email automation agent template.

Provides an AI-powered email assistant capable of reading, composing,
sending, organizing, and prioritizing emails with enterprise-grade
guardrails for PII protection.

Usage::

    from duxx_ai.templates.email_agent import EmailAgent

    agent = EmailAgent.create()
    result = await agent.run("Draft a follow-up email to the client about the Q4 report.")
"""

from __future__ import annotations

from duxx_ai.core.agent import AgentConfig
from duxx_ai.core.llm import LLMConfig
from duxx_ai.core.tool import Tool
from duxx_ai.governance.guardrails import (
    GuardrailChain,
    PIIGuardrail,
    PromptInjectionGuardrail,
)
from duxx_ai.tools.builtin import get_builtin_tools

from ._base import AgentTemplate

_SYSTEM_PROMPT = """\
You are an enterprise email automation agent. Your responsibilities include:

1. **Reading & Analysis**: Read, parse, and summarize incoming emails. Identify
   key action items, deadlines, and sentiment.
2. **Composition**: Draft professional emails using appropriate tone, formatting,
   and structure for business communication.
3. **Sending**: Send emails on behalf of the user after explicit confirmation,
   respecting rate limits and organizational policies.
4. **Organization**: Categorize, label, archive, and prioritize emails based on
   urgency, sender importance, and content relevance.
5. **Prioritization**: Rank inbox items by urgency and importance. Flag emails
   requiring immediate attention.
6. **Response Drafting**: Generate context-aware reply drafts that maintain
   conversational thread coherence.

Guidelines:
- Always confirm with the user before sending any email.
- Protect sensitive information — redact PII from logs and summaries.
- Follow the organization's email policies and communication standards.
- Maintain a professional, clear, and concise communication style.
- When uncertain about tone or content, ask for clarification.
"""


class EmailAgent(AgentTemplate):
    """Email automation agent for enterprise communication workflows.

    Handles reading, composing, sending, organizing, and prioritizing
    emails with built-in PII protection and prompt injection guardrails.

    Usage::

        agent = EmailAgent.create()
        result = await agent.run("Summarize my unread emails from today.")
    """

    name: str = "email_agent"
    description: str = "Enterprise email automation — read, compose, send, organize, and prioritize emails."
    category: str = "Communication"
    default_tools: list[str] = ["calculator"]

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
        tools = get_builtin_tools(["calculator"])
        try:
            from duxx_ai.tools.email import MODULE_TOOLS

            tools.extend(MODULE_TOOLS.values())
        except ImportError:
            pass
        if extra_tools:
            tools.extend(extra_tools)
        return tools

    @classmethod
    def _build_guardrails(cls) -> GuardrailChain:
        chain = GuardrailChain()
        chain.add(PIIGuardrail(allow_email=True))
        chain.add(PromptInjectionGuardrail())
        return chain
