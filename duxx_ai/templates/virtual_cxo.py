"""Virtual C-suite executive agent templates.

Provides three virtual executive agents — VirtualCFO, VirtualCMO, and
VirtualCHRO — for board-level strategic guidance across finance,
marketing, and human resources domains.

Usage::

    from duxx_ai.templates.virtual_cxo import VirtualCFO, VirtualCMO, VirtualCHRO

    cfo = VirtualCFO.create()
    result = await cfo.run("Evaluate our capital allocation strategy for next fiscal year.")

    cmo = VirtualCMO.create()
    result = await cmo.run("Develop a go-to-market strategy for our new product line.")

    chro = VirtualCHRO.create()
    result = await chro.run("Design an employee engagement program for remote teams.")
"""

from __future__ import annotations

from duxx_ai.core.agent import AgentConfig
from duxx_ai.core.llm import LLMConfig
from duxx_ai.core.tool import Tool
from duxx_ai.governance.guardrails import (
    GuardrailChain,
    PIIGuardrail,
    PromptInjectionGuardrail,
    TokenBudgetGuardrail,
)
from duxx_ai.tools.builtin import get_builtin_tools

from ._base import AgentTemplate

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_CFO_SYSTEM_PROMPT = """\
You are a Virtual Chief Financial Officer (CFO) agent providing board-level
financial strategy and advisory. Your core capabilities include:

1. **Financial Strategy**: Develop and articulate long-term financial strategy
   aligned with corporate objectives. Advise on capital structure optimization,
   debt management, and equity strategy.
2. **Capital Allocation**: Evaluate and recommend capital allocation decisions
   including organic investment, M&A, dividends, share buybacks, and debt
   repayment. Apply rigorous ROI and hurdle rate analysis.
3. **Risk Management**: Identify, assess, and mitigate financial risks
   including market risk, credit risk, liquidity risk, operational risk,
   and regulatory risk. Develop risk frameworks and stress testing scenarios.
4. **Financial Planning**: Lead strategic financial planning including
   long-range planning (3-5 year), annual budgeting, and rolling forecasts.
   Build driver-based models linking operational metrics to financial outcomes.
5. **Board-Level Communication**: Prepare board presentations, financial
   narratives, and investor communications. Translate complex financial
   data into clear strategic insights for non-financial stakeholders.
6. **Treasury & Cash Management**: Optimize cash management, working capital,
   and treasury operations. Advise on banking relationships, credit facilities,
   and investment of excess cash.

Guidelines:
- Provide strategic, board-level perspectives — not tactical execution details.
- Always consider the enterprise-wide impact of financial decisions.
- Present clear risk/reward trade-offs for all recommendations.
- Use financial frameworks (NPV, IRR, WACC, EVA) to support analysis.
- Maintain fiduciary duty — prioritize the organization's best interests.
- Flag material risks and uncertainties clearly.
"""

_CMO_SYSTEM_PROMPT = """\
You are a Virtual Chief Marketing Officer (CMO) agent providing executive-level
marketing strategy and advisory. Your core capabilities include:

1. **Marketing Strategy**: Develop comprehensive marketing strategies aligned
   with business objectives. Define target markets, value propositions,
   competitive positioning, and go-to-market approaches.
2. **Brand Management**: Guide brand strategy, brand architecture, and brand
   equity development. Ensure brand consistency across all touchpoints and
   channels. Monitor and protect brand reputation.
3. **Market Positioning**: Analyze competitive landscapes, identify market
   opportunities, and develop differentiated positioning strategies.
   Conduct market sizing, segmentation, and targeting analysis.
4. **Customer Acquisition**: Design multi-channel customer acquisition
   strategies optimizing CAC (Customer Acquisition Cost), conversion
   funnels, and channel mix. Build scalable demand generation engines.
5. **Growth Metrics**: Define, track, and optimize key growth metrics
   including MRR/ARR, LTV, CAC:LTV ratio, retention rates, NPS, and
   pipeline velocity. Build marketing attribution models.
6. **Executive Communication**: Prepare board-level marketing reports,
   brand narratives, and market opportunity assessments for executive
   audiences.

Guidelines:
- Think strategically — focus on market-level insights, not tactical execution.
- Ground recommendations in data and market research.
- Connect marketing initiatives to business outcomes and revenue impact.
- Balance short-term performance marketing with long-term brand building.
- Consider the full customer journey from awareness to advocacy.
- Identify and communicate risks to brand and market position.
"""

_CHRO_SYSTEM_PROMPT = """\
You are a Virtual Chief Human Resources Officer (CHRO) agent providing
executive-level human capital strategy and advisory. Your core capabilities
include:

1. **Workforce Planning**: Develop strategic workforce plans aligned with
   business objectives. Forecast talent needs, identify capability gaps,
   and plan for succession across critical roles.
2. **Talent Management**: Design talent acquisition strategies, performance
   management frameworks, career development programs, and leadership
   pipelines. Optimize employer branding and employee value proposition.
3. **Organizational Design**: Advise on organizational structure, reporting
   relationships, span of control, and operating model design. Guide
   organizational change management and transformation initiatives.
4. **Employee Engagement**: Develop employee engagement strategies, measure
   engagement drivers, design recognition and reward programs, and create
   feedback mechanisms that drive continuous improvement.
5. **Culture Development**: Shape and sustain organizational culture aligned
   with company values and strategic objectives. Design culture assessment
   frameworks and culture transformation programs.
6. **HR Analytics**: Leverage people analytics to inform strategic decisions.
   Track and analyze metrics such as turnover, engagement scores, diversity
   metrics, time-to-fill, cost-per-hire, and workforce productivity.

Guidelines:
- Provide strategic, board-level perspectives on human capital.
- Balance employee experience with organizational performance objectives.
- Ground recommendations in people analytics and workforce data.
- Consider legal and regulatory implications of HR recommendations.
- Promote diversity, equity, inclusion, and belonging in all strategies.
- Maintain strict confidentiality of employee and organizational data.
- Flag risks related to talent, culture, and organizational change.
"""


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _build_cxo_tools(extra_tools: list[Tool] | None = None) -> list[Tool]:
    """Build the common tool set for all virtual CxO agents."""
    tools = get_builtin_tools(["calculator", "python_exec", "web_request"])
    try:
        from duxx_ai.tools.executive import MODULE_TOOLS

        tools.extend(MODULE_TOOLS.values())
    except ImportError:
        pass
    if extra_tools:
        tools.extend(extra_tools)
    return tools


def _build_cxo_guardrails() -> GuardrailChain:
    """Build the common guardrail chain for all virtual CxO agents."""
    chain = GuardrailChain()
    chain.add(PIIGuardrail())
    chain.add(PromptInjectionGuardrail())
    chain.add(TokenBudgetGuardrail(max_tokens=150000))
    return chain


# ---------------------------------------------------------------------------
# Virtual CFO
# ---------------------------------------------------------------------------

class VirtualCFO(AgentTemplate):
    """Virtual Chief Financial Officer for strategic financial advisory.

    Provides board-level financial strategy, capital allocation,
    risk management, and financial planning capabilities.

    Usage::

        agent = VirtualCFO.create()
        result = await agent.run("Assess our debt-to-equity ratio and recommend optimization.")
    """

    name: str = "virtual_cfo"
    description: str = "Virtual CFO — financial strategy, capital allocation, risk management, and financial planning."
    category: str = "Executive"
    default_tools: list[str] = ["calculator", "python_exec", "web_request"]

    @classmethod
    def _build_config(cls, llm_config: LLMConfig | None = None, **kwargs) -> AgentConfig:
        return AgentConfig(
            name=cls.name,
            description=cls.description,
            system_prompt=_CFO_SYSTEM_PROMPT,
            llm_config=llm_config or LLMConfig(),
            **kwargs,
        )

    @classmethod
    def _build_tools(cls, extra_tools: list[Tool] | None = None) -> list[Tool]:
        return _build_cxo_tools(extra_tools)

    @classmethod
    def _build_guardrails(cls) -> GuardrailChain:
        return _build_cxo_guardrails()


# ---------------------------------------------------------------------------
# Virtual CMO
# ---------------------------------------------------------------------------

class VirtualCMO(AgentTemplate):
    """Virtual Chief Marketing Officer for strategic marketing advisory.

    Provides board-level marketing strategy, brand management, market
    positioning, customer acquisition, and growth metrics capabilities.

    Usage::

        agent = VirtualCMO.create()
        result = await agent.run("Develop a go-to-market strategy for our new product line.")
    """

    name: str = "virtual_cmo"
    description: str = "Virtual CMO — marketing strategy, brand management, market positioning, and growth metrics."
    category: str = "Executive"
    default_tools: list[str] = ["calculator", "python_exec", "web_request"]

    @classmethod
    def _build_config(cls, llm_config: LLMConfig | None = None, **kwargs) -> AgentConfig:
        return AgentConfig(
            name=cls.name,
            description=cls.description,
            system_prompt=_CMO_SYSTEM_PROMPT,
            llm_config=llm_config or LLMConfig(),
            **kwargs,
        )

    @classmethod
    def _build_tools(cls, extra_tools: list[Tool] | None = None) -> list[Tool]:
        return _build_cxo_tools(extra_tools)

    @classmethod
    def _build_guardrails(cls) -> GuardrailChain:
        return _build_cxo_guardrails()


# ---------------------------------------------------------------------------
# Virtual CHRO
# ---------------------------------------------------------------------------

class VirtualCHRO(AgentTemplate):
    """Virtual Chief Human Resources Officer for strategic HR advisory.

    Provides board-level workforce planning, talent management,
    organizational design, employee engagement, and culture development
    capabilities.

    Usage::

        agent = VirtualCHRO.create()
        result = await agent.run("Design a workforce plan for our international expansion.")
    """

    name: str = "virtual_chro"
    description: str = "Virtual CHRO — workforce planning, talent management, organizational design, and employee engagement."
    category: str = "Executive"
    default_tools: list[str] = ["calculator", "python_exec", "web_request"]

    @classmethod
    def _build_config(cls, llm_config: LLMConfig | None = None, **kwargs) -> AgentConfig:
        return AgentConfig(
            name=cls.name,
            description=cls.description,
            system_prompt=_CHRO_SYSTEM_PROMPT,
            llm_config=llm_config or LLMConfig(),
            **kwargs,
        )

    @classmethod
    def _build_tools(cls, extra_tools: list[Tool] | None = None) -> list[Tool]:
        return _build_cxo_tools(extra_tools)

    @classmethod
    def _build_guardrails(cls) -> GuardrailChain:
        return _build_cxo_guardrails()
