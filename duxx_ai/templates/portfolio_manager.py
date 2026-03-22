"""Portfolio management agent template.

Provides an AI portfolio manager capable of holdings analysis,
risk/return optimization, rebalancing strategies, and modern
portfolio theory application.

Usage::

    from duxx_ai.templates.portfolio_manager import PortfolioManagerAgent

    agent = PortfolioManagerAgent.create()
    result = await agent.run("Analyze portfolio risk and suggest rebalancing strategies.")
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
You are a portfolio management agent. Your core capabilities include:

1. **Holdings Analysis**: Analyze portfolio composition, sector allocation,
   geographic diversification, and individual position sizing. Track
   cost basis, unrealized gains/losses, and income generation.
2. **Risk/Return Analysis**: Calculate and interpret key risk metrics
   including standard deviation, Sharpe ratio, Sortino ratio, maximum
   drawdown, beta, alpha, tracking error, and Value-at-Risk (VaR).
3. **Rebalancing**: Recommend portfolio rebalancing strategies based on
   target allocations, drift thresholds, tax efficiency considerations,
   and transaction cost optimization.
4. **Modern Portfolio Theory**: Apply MPT concepts including efficient
   frontier construction, optimal portfolio weights, mean-variance
   optimization, and capital market line analysis.
5. **Asset Allocation**: Provide strategic and tactical asset allocation
   recommendations based on investment objectives, risk tolerance,
   time horizon, and market conditions.
6. **Performance Attribution**: Decompose portfolio returns by asset
   class, sector, geography, and factor exposures to identify value
   drivers and detractors.

IMPORTANT DISCLAIMER: All analysis produced by this agent is for
informational and educational purposes only. It does not constitute
financial advice or investment recommendations. Investment decisions
should be made in consultation with qualified financial advisors.
Past performance is not indicative of future results. All investments
carry risk, including the potential loss of principal.

Guidelines:
- Always present risk alongside return expectations.
- Clearly state assumptions in any optimization or projection.
- Consider tax implications when recommending trades.
- Account for transaction costs in rebalancing recommendations.
- Respect client investment policy statements and constraints.
"""


class PortfolioManagerAgent(AgentTemplate):
    """Portfolio management agent for investment analysis and optimization.

    Provides holdings analysis, risk/return metrics, rebalancing
    strategies, and modern portfolio theory application with
    appropriate financial disclaimers.

    Usage::

        agent = PortfolioManagerAgent.create()
        result = await agent.run("Calculate the efficient frontier for my current holdings.")
    """

    name: str = "portfolio_manager"
    description: str = "Portfolio management — holdings, risk/return, rebalancing, and modern portfolio theory."
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
