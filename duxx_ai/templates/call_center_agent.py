"""AI call center agent template.

Provides an empathetic, SOP-driven customer service agent capable of
handling inquiries, accessing customer records, escalating issues, and
resolving common support scenarios.

Usage::

    from duxx_ai.templates.call_center_agent import CallCenterAgent

    agent = CallCenterAgent.create()
    result = await agent.run("Customer is asking about a refund for order #12345.")
"""

from __future__ import annotations

from duxx_ai.core.agent import AgentConfig
from duxx_ai.core.llm import LLMConfig
from duxx_ai.core.tool import Tool
from duxx_ai.governance.guardrails import (
    GuardrailChain,
    PIIGuardrail,
    PromptInjectionGuardrail,
    TopicGuardrail,
)
from duxx_ai.tools.builtin import get_builtin_tools

from ._base import AgentTemplate

_SYSTEM_PROMPT = """\
You are an AI call center agent designed to handle customer inquiries with
empathy, professionalism, and efficiency. Your core responsibilities include:

1. **Customer Inquiry Handling**: Address questions about products, services,
   billing, accounts, orders, refunds, and shipping with accurate, helpful
   responses.
2. **Empathetic Communication**: Always acknowledge the customer's feelings
   and frustrations. Use active listening cues and maintain a warm, supportive
   tone throughout the interaction.
3. **Customer Record Access**: Look up customer accounts, order histories,
   and relevant records to provide personalized assistance.
4. **Escalation Management**: Recognize when an issue exceeds your authority
   or capability and escalate to the appropriate human agent or department.
   Always explain the escalation process to the customer.
5. **SOP Adherence**: Follow standard operating procedures for common
   scenarios including returns, refunds, billing disputes, account changes,
   and service cancellations.
6. **Resolution Tracking**: Document all interactions, resolutions, and
   follow-up actions required.

Guidelines:
- Never share sensitive customer data with unauthorized parties.
- Stay within approved topics: customer service, support, billing, accounts,
  orders, refunds, and shipping.
- If you cannot resolve an issue, escalate — never guess or provide inaccurate
  information.
- Maintain composure and professionalism regardless of customer demeanor.
- Follow data retention and privacy policies at all times.
"""

_ALLOWED_TOPICS = [
    "customer service",
    "support",
    "billing",
    "account",
    "order",
    "refund",
    "shipping",
]


class CallCenterAgent(AgentTemplate):
    """AI call center agent for customer service workflows.

    Handles customer inquiries with empathy and SOP adherence, with
    topic-restricted guardrails to keep conversations on-scope.

    Usage::

        agent = CallCenterAgent.create()
        result = await agent.run("Customer wants to return a defective product.")
    """

    name: str = "call_center_agent"
    description: str = "AI call center — handle inquiries with empathy, access customer records, escalate, and follow SOPs."
    category: str = "Customer Service"
    default_tools: list[str] = ["read_file", "web_request", "json_query"]

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
        tools = get_builtin_tools(["read_file", "web_request", "json_query"])
        try:
            from duxx_ai.tools.customer_service import MODULE_TOOLS

            tools.extend(MODULE_TOOLS.values())
        except ImportError:
            pass
        if extra_tools:
            tools.extend(extra_tools)
        return tools

    @classmethod
    def _build_guardrails(cls) -> GuardrailChain:
        chain = GuardrailChain()
        chain.add(PIIGuardrail())
        chain.add(PromptInjectionGuardrail())
        chain.add(TopicGuardrail(allowed_topics=_ALLOWED_TOPICS))
        return chain
