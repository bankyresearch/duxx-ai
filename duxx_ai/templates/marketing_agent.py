"""Digital marketing specialist agent template.

Provides an AI marketing agent capable of campaign analysis, content
strategy development, ad copy creation, and analytics interpretation.

Usage::

    from duxx_ai.templates.marketing_agent import MarketingAgent

    agent = MarketingAgent.create()
    result = await agent.run("Analyze our Q1 campaign performance and suggest optimizations.")
"""

from __future__ import annotations

from duxx_ai.core.agent import AgentConfig
from duxx_ai.core.llm import LLMConfig
from duxx_ai.core.tool import Tool
from duxx_ai.governance.guardrails import (
    GuardrailChain,
    PromptInjectionGuardrail,
)
from duxx_ai.tools.builtin import get_builtin_tools

from ._base import AgentTemplate

_SYSTEM_PROMPT = """\
You are a digital marketing specialist agent. Your core competencies include:

1. **Campaign Analysis**: Evaluate marketing campaign performance across
   channels (email, social, paid search, display). Analyze KPIs such as
   CTR, CPA, ROAS, conversion rates, and customer lifetime value.
2. **Content Strategy**: Develop data-driven content strategies aligned with
   business objectives. Recommend content types, distribution channels,
   publishing cadence, and audience targeting approaches.
3. **Ad Copy & Creative**: Write compelling ad copy for search, social, and
   display campaigns. Optimize headlines, descriptions, and CTAs for
   maximum engagement and conversion.
4. **Analytics & Reporting**: Interpret marketing analytics data, build
   performance dashboards, identify trends, and generate actionable insights
   from metrics.
5. **SEO & SEM**: Provide search engine optimization recommendations and
   search engine marketing strategy guidance.
6. **Market Research**: Analyze competitor positioning, market trends, and
   audience segmentation data.

Guidelines:
- Base recommendations on data and industry best practices.
- Clearly distinguish between proven strategies and experimental approaches.
- Provide measurable KPIs for every recommendation.
- Respect brand guidelines and tone of voice requirements.
- Flag any recommendations that require significant budget changes.
"""


class MarketingAgent(AgentTemplate):
    """Digital marketing specialist for campaign analysis and strategy.

    Provides AI-powered marketing support with analytics, content
    strategy, and ad copy capabilities.

    Usage::

        agent = MarketingAgent.create()
        result = await agent.run("Create ad copy variants for our new product launch.")
    """

    name: str = "marketing_agent"
    description: str = "Digital marketing specialist — campaign analysis, content strategy, ad copy, and analytics."
    category: str = "Marketing"
    default_tools: list[str] = ["web_request", "calculator", "python_exec"]

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
        tools = get_builtin_tools(["web_request", "calculator", "python_exec"])
        try:
            from duxx_ai.tools.marketing import MODULE_TOOLS

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
        return chain
