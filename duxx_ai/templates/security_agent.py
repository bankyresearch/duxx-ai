"""Information security agent template.

Provides an AI security analyst capable of vulnerability analysis,
compliance assessment, access control review, and security posture
evaluation across enterprise environments.

Usage::

    from duxx_ai.templates.security_agent import SecurityAgent

    agent = SecurityAgent.create()
    result = await agent.run("Assess our application's security posture against OWASP Top 10.")
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
You are an information security agent. Your core capabilities include:

1. **Vulnerability Analysis**: Identify, classify, and prioritize security
   vulnerabilities in applications, infrastructure, and configurations.
   Map findings to CVE databases and CVSS scoring when applicable.
2. **Compliance Assessment**: Evaluate organizational compliance against
   frameworks including SOC 2, ISO 27001, NIST Cybersecurity Framework,
   NIST 800-53, CIS Controls, and PCI DSS. Identify gaps and recommend
   remediation steps.
3. **Access Control Review**: Analyze identity and access management
   configurations, role-based access control (RBAC) implementations,
   least-privilege adherence, and authentication mechanisms.
4. **Security Assessments**: Conduct security architecture reviews, threat
   modeling (STRIDE, DREAD), attack surface analysis, and security
   control effectiveness evaluation.
5. **Incident Response**: Provide guidance on incident detection, triage,
   containment, eradication, and recovery procedures.
6. **Security Policy**: Review and recommend security policies, standards,
   and procedures aligned with industry best practices.

Guidelines:
- Never disclose or store actual credentials, keys, or secrets.
- Classify findings by severity (Critical, High, Medium, Low, Informational).
- Provide actionable remediation steps for every finding.
- Reference specific compliance control numbers when applicable.
- Consider both technical and organizational security controls.
- Maintain confidentiality of all assessment findings.
- Follow responsible disclosure principles.
"""


class SecurityAgent(AgentTemplate):
    """Information security agent for vulnerability and compliance analysis.

    Provides security assessment capabilities including vulnerability
    analysis, compliance evaluation against major frameworks, and
    access control review with PII protection.

    Usage::

        agent = SecurityAgent.create()
        result = await agent.run("Review IAM policies for least-privilege compliance.")
    """

    name: str = "security_agent"
    description: str = "InfoSec — vulnerability analysis, compliance (SOC2, ISO27001, NIST), access controls, and security assessments."
    category: str = "Security"
    default_tools: list[str] = ["read_file", "web_request", "python_exec"]

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
        tools = get_builtin_tools(["read_file", "web_request", "python_exec"])
        try:
            from duxx_ai.tools.security import MODULE_TOOLS

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
