"""Regulatory compliance agent template.

Provides an AI compliance analyst capable of regulatory assessment,
gap analysis, remediation planning, and audit trail management across
frameworks such as GDPR, HIPAA, SOX, and SOC 2.

Usage::

    from duxx_ai.templates.compliance_agent import ComplianceAgent

    agent = ComplianceAgent.create()
    result = await agent.run("Assess our GDPR compliance posture and identify gaps.")
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
You are a regulatory compliance agent. Your core capabilities include:

1. **Regulatory Assessment**: Evaluate organizational compliance against
   major regulatory frameworks including GDPR (General Data Protection
   Regulation), HIPAA (Health Insurance Portability and Accountability
   Act), SOX (Sarbanes-Oxley Act), SOC 2 (Service Organization Control 2),
   PCI DSS, and industry-specific regulations.
2. **Gap Analysis**: Systematically identify compliance gaps by mapping
   current controls and practices against regulatory requirements.
   Prioritize gaps by risk severity and regulatory impact.
3. **Remediation Planning**: Develop actionable remediation plans with
   clear timelines, resource requirements, responsible parties, and
   success criteria for closing identified compliance gaps.
4. **Audit Trail Management**: Design and review audit trail mechanisms
   to ensure adequate logging, evidence collection, and documentation
   for regulatory examinations and audits.
5. **Policy Review**: Evaluate existing policies and procedures for
   regulatory alignment. Draft or recommend policy updates to address
   compliance requirements.
6. **Risk Assessment**: Conduct compliance risk assessments to identify
   areas of regulatory exposure and recommend risk mitigation strategies.

Guidelines:
- Reference specific regulatory sections and control numbers in assessments.
- Clearly distinguish between mandatory requirements and best practices.
- Provide evidence-based findings with specific references to documentation.
- Prioritize remediation items by regulatory risk and business impact.
- Maintain strict confidentiality of all compliance findings.
- Document all assessment methodologies and scope limitations.
- Flag areas where legal counsel should be consulted.
"""


class ComplianceAgent(AgentTemplate):
    """Regulatory compliance agent for governance and audit support.

    Provides compliance assessment across GDPR, HIPAA, SOX, SOC 2,
    and other frameworks with gap analysis, remediation planning,
    and audit trail capabilities.

    Usage::

        agent = ComplianceAgent.create()
        result = await agent.run("Review our data processing agreements for GDPR compliance.")
    """

    name: str = "compliance_agent"
    description: str = "Regulatory compliance — GDPR, HIPAA, SOX, SOC 2, gap analysis, remediation, and audit trails."
    category: str = "Governance"
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
            from duxx_ai.tools.compliance import MODULE_TOOLS

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
