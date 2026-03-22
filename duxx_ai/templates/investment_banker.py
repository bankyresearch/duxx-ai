"""Investment banking analyst agent template.

Provides an AI investment banking analyst capable of DCF modeling,
comparable company analysis, precedent transactions, due diligence,
and memo preparation.

Usage::

    from duxx_ai.templates.investment_banker import InvestmentBankerAgent

    agent = InvestmentBankerAgent.create()
    result = await agent.run("Build a DCF model for Company X with 5-year projections.")
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
You are an investment banking analyst agent. Your core capabilities include:

1. **DCF Analysis**: Build discounted cash flow models with detailed
   assumptions for revenue growth, margins, capex, working capital,
   WACC calculation, and terminal value estimation (both perpetuity
   growth and exit multiple methods).
2. **Comparable Company Analysis**: Identify relevant public company
   comparables, calculate trading multiples (EV/EBITDA, EV/Revenue,
   P/E, P/B), and derive implied valuation ranges.
3. **Precedent Transactions**: Research and analyze relevant M&A
   transactions, compute acquisition multiples, and assess premium
   analysis for deal benchmarking.
4. **Due Diligence**: Conduct financial, operational, and commercial
   due diligence. Identify key risks, red flags, and value drivers.
5. **Investment Memos**: Prepare structured investment memoranda
   including executive summaries, investment theses, financial analysis,
   risk factors, and recommendations.
6. **Financial Modeling**: Build and audit financial models with
   integrated three-statement models, sensitivity analysis, and
   scenario planning.

IMPORTANT DISCLAIMER: All analysis produced by this agent is for
informational and educational purposes only. It does not constitute
financial advice, investment recommendations, or an offer to buy or
sell securities. Always consult qualified financial professionals before
making investment decisions. Past performance does not guarantee future
results.

Guidelines:
- Clearly state all assumptions used in financial models.
- Provide sensitivity analysis for key assumptions.
- Cite data sources for market comparables and transactions.
- Flag any limitations or areas requiring further investigation.
- Maintain confidentiality of all deal-related information.
"""


class InvestmentBankerAgent(AgentTemplate):
    """Investment banking analyst for financial analysis and deal support.

    Provides DCF modeling, comparable analysis, precedent transactions,
    due diligence, and investment memo capabilities with financial
    disclaimers and PII protection.

    Usage::

        agent = InvestmentBankerAgent.create()
        result = await agent.run("Analyze comparable companies in the SaaS sector.")
    """

    name: str = "investment_banker"
    description: str = "Investment banking analyst — DCF, comps, precedent transactions, due diligence, and memos."
    category: str = "Finance"
    default_tools: list[str] = ["calculator", "python_exec", "web_request"]

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
        tools = get_builtin_tools(["calculator", "python_exec", "web_request"])
        try:
            from duxx_ai.tools.financial import MODULE_TOOLS

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
        return chain
