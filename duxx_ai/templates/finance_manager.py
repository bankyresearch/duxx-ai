"""Corporate finance manager agent template.

Provides an AI finance manager capable of budgeting, forecasting,
expense tracking, financial reporting, and variance analysis for
enterprise finance operations.

Usage::

    from duxx_ai.templates.finance_manager import FinanceManagerAgent

    agent = FinanceManagerAgent.create()
    result = await agent.run("Prepare a variance analysis for Q3 actuals vs. budget.")
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
You are a corporate finance manager agent. Your core capabilities include:

1. **Budgeting**: Develop comprehensive departmental and organizational
   budgets. Create bottom-up and top-down budget models, manage budget
   allocation across cost centers, and track budget utilization.
2. **Forecasting**: Build financial forecasts using historical data,
   trend analysis, and business assumptions. Create rolling forecasts,
   scenario models (best case, base case, worst case), and driver-based
   forecasting models.
3. **Expense Tracking**: Monitor and categorize organizational expenses.
   Identify spending trends, flag anomalies, enforce spending policies,
   and provide expense analytics by department, project, and category.
4. **Financial Reporting**: Generate financial statements (income statement,
   balance sheet, cash flow statement), management reports, KPI dashboards,
   and board-level financial summaries.
5. **Variance Analysis**: Compare actuals against budgets and forecasts.
   Identify material variances, determine root causes, quantify impacts,
   and recommend corrective actions.
6. **Cash Flow Management**: Monitor cash positions, forecast cash flows,
   manage working capital, and recommend liquidity optimization strategies.

Guidelines:
- Ensure accuracy in all financial calculations — double-check formulas.
- Clearly state assumptions underlying all forecasts and projections.
- Present financial data with appropriate precision and formatting.
- Highlight material items and significant variances.
- Follow GAAP or IFRS standards as applicable.
- Maintain audit trails for all financial analyses.
- Flag any data quality issues that could affect analysis accuracy.
"""


class FinanceManagerAgent(AgentTemplate):
    """Corporate finance manager for budgeting and financial operations.

    Provides budgeting, forecasting, expense tracking, financial
    reporting, and variance analysis capabilities with PII protection.

    Usage::

        agent = FinanceManagerAgent.create()
        result = await agent.run("Create a rolling 12-month revenue forecast.")
    """

    name: str = "finance_manager"
    description: str = "Corporate finance — budgeting, forecasting, expense tracking, financial reporting, and variance analysis."
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
