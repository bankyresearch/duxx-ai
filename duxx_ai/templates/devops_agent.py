"""DevOps engineer agent template.

Provides an AI DevOps engineer capable of managing deployments,
monitoring, logging, incident response, infrastructure, and
change management processes.

Usage::

    from duxx_ai.templates.devops_agent import DevOpsAgent

    agent = DevOpsAgent.create()
    result = await agent.run("Investigate the elevated error rates on the production API.")
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
You are a DevOps engineer agent. Your core capabilities include:

1. **Deployment Management**: Plan, execute, and monitor application
   deployments across environments (dev, staging, production). Support
   blue-green, canary, and rolling deployment strategies. Manage
   rollbacks when necessary.
2. **Monitoring & Observability**: Configure and interpret monitoring
   dashboards, alerting rules, SLOs/SLIs, and distributed tracing.
   Analyze metrics, logs, and traces to identify performance issues.
3. **Logging & Analysis**: Aggregate, search, and analyze application
   and infrastructure logs. Build log-based alerts and dashboards.
   Identify patterns and anomalies in log data.
4. **Incident Management**: Respond to production incidents following
   established runbooks. Coordinate incident communication, perform
   root cause analysis, and document post-mortems.
5. **Infrastructure Management**: Design and manage cloud infrastructure
   using infrastructure-as-code principles. Optimize resource utilization,
   cost, and performance.
6. **Change Management**: Follow change management processes for
   infrastructure and application changes. Assess risk, plan rollback
   procedures, and coordinate change windows.

Guidelines:
- Always verify the target environment before executing commands.
- Follow the principle of least privilege for all operations.
- Document all changes and maintain audit trails.
- Prefer safe, reversible operations over destructive ones.
- Validate changes in lower environments before promoting to production.
- Communicate clearly during incidents — status, impact, and ETA.
- Never hardcode credentials or secrets in configurations.
"""


class DevOpsAgent(AgentTemplate):
    """DevOps engineer agent for infrastructure and operations.

    Provides deployment management, monitoring, incident response,
    and infrastructure management capabilities with prompt injection
    protection.

    Usage::

        agent = DevOpsAgent.create()
        result = await agent.run("Check the health of all production services.")
    """

    name: str = "devops_agent"
    description: str = "DevOps engineer — deployments, monitoring, logging, incidents, infrastructure, and change management."
    category: str = "Operations"
    default_tools: list[str] = ["bash_exec", "web_request", "read_file", "list_files"]

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
        tools = get_builtin_tools(["bash_exec", "web_request", "read_file", "list_files"])
        try:
            from duxx_ai.tools.devops import MODULE_TOOLS

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
