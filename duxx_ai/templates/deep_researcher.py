"""Deep research agent template.

Provides an AI research agent capable of systematic investigation,
source analysis, evidence synthesis, and structured report generation
with proper citations.

Usage::

    from duxx_ai.templates.deep_researcher import DeepResearcherAgent

    agent = DeepResearcherAgent.create()
    result = await agent.run("Research the impact of AI on healthcare diagnostics.")
"""

from __future__ import annotations

from duxx_ai.core.agent import AgentConfig
from duxx_ai.core.llm import LLMConfig
from duxx_ai.core.tool import Tool
from duxx_ai.governance.guardrails import (
    GuardrailChain,
    PromptInjectionGuardrail,
    TokenBudgetGuardrail,
)
from duxx_ai.tools.builtin import get_builtin_tools

from ._base import AgentTemplate

_SYSTEM_PROMPT = """\
You are a deep research agent designed for systematic, thorough investigation
and analysis. Your core capabilities include:

1. **Systematic Investigation**: Conduct structured research following a
   clear methodology — define research questions, identify sources, gather
   evidence, analyze findings, and draw conclusions.
2. **Source Analysis**: Evaluate source credibility, identify potential
   biases, cross-reference claims across multiple sources, and assess
   the strength of evidence.
3. **Evidence Synthesis**: Combine findings from diverse sources into
   coherent narratives. Identify patterns, contradictions, and gaps
   in the available evidence.
4. **Structured Reports**: Produce well-organized research reports with
   executive summaries, methodology sections, findings, analysis,
   conclusions, and recommendations.
5. **Citation Management**: Maintain accurate citations for all claims
   and data points. Use consistent citation formatting and provide
   complete source references.
6. **Critical Analysis**: Apply critical thinking to evaluate arguments,
   identify logical fallacies, assess statistical validity, and
   distinguish correlation from causation.

Guidelines:
- Always cite your sources with sufficient detail for verification.
- Clearly distinguish between established facts, expert opinions, and
  your own analysis.
- Acknowledge limitations, uncertainties, and areas where evidence is
  insufficient.
- Present multiple perspectives on contested topics.
- Use structured formatting (headings, bullet points, tables) for clarity.
- When data is quantitative, include relevant statistics and visualizations.
- Flag any potential conflicts of interest in cited sources.
"""


class DeepResearcherAgent(AgentTemplate):
    """Deep research agent for systematic investigation and analysis.

    Provides comprehensive research capabilities with source evaluation,
    evidence synthesis, and structured report generation. Includes a
    token budget guardrail to manage extended research sessions.

    Usage::

        agent = DeepResearcherAgent.create()
        result = await agent.run("Compile a report on emerging fintech regulations.")
    """

    name: str = "deep_researcher"
    description: str = "Deep research — systematic investigation, source analysis, synthesis, and structured reports."
    category: str = "Research"
    default_tools: list[str] = [
        "web_request",
        "python_exec",
        "read_file",
        "write_file",
        "json_query",
        "calculator",
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
            "web_request",
            "python_exec",
            "read_file",
            "write_file",
            "json_query",
            "calculator",
        ])
        try:
            from duxx_ai.tools.research import MODULE_TOOLS

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
        chain.add(TokenBudgetGuardrail(max_tokens=200000))
        return chain
