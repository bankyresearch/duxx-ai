"""Enterprise agent templates for Duxx AI.

This module provides pre-configured agent templates for common enterprise
use cases. Each template bundles appropriate system prompts, tools, and
guardrails for its domain.

Usage::

    from duxx_ai.templates import TEMPLATES, EmailAgent, CodeBuilderAgent

    # List available templates
    for name, cls in TEMPLATES.items():
        print(cls.info())

    # Create an agent from a template
    agent = EmailAgent.create()
    result = await agent.run("Summarize my unread emails.")

    # Create by name from the registry
    agent_cls = TEMPLATES["code_builder"]
    agent = agent_cls.create()
"""

from duxx_ai.templates._base import AgentTemplate
from duxx_ai.templates.call_center_agent import CallCenterAgent
from duxx_ai.templates.code_builder import CodeBuilderAgent
from duxx_ai.templates.compliance_agent import ComplianceAgent
from duxx_ai.templates.deep_researcher import DeepResearcherAgent
from duxx_ai.templates.devops_agent import DevOpsAgent
from duxx_ai.templates.email_agent import EmailAgent
from duxx_ai.templates.finance_manager import FinanceManagerAgent
from duxx_ai.templates.investment_banker import InvestmentBankerAgent
from duxx_ai.templates.marketing_agent import MarketingAgent
from duxx_ai.templates.portfolio_manager import PortfolioManagerAgent
from duxx_ai.templates.security_agent import SecurityAgent
from duxx_ai.templates.virtual_cxo import VirtualCFO, VirtualCHRO, VirtualCMO

# ---------------------------------------------------------------------------
# Template registry — maps template name to its class
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, type[AgentTemplate]] = {
    EmailAgent.name: EmailAgent,
    CallCenterAgent.name: CallCenterAgent,
    MarketingAgent.name: MarketingAgent,
    InvestmentBankerAgent.name: InvestmentBankerAgent,
    PortfolioManagerAgent.name: PortfolioManagerAgent,
    DeepResearcherAgent.name: DeepResearcherAgent,
    CodeBuilderAgent.name: CodeBuilderAgent,
    SecurityAgent.name: SecurityAgent,
    DevOpsAgent.name: DevOpsAgent,
    ComplianceAgent.name: ComplianceAgent,
    FinanceManagerAgent.name: FinanceManagerAgent,
    VirtualCFO.name: VirtualCFO,
    VirtualCMO.name: VirtualCMO,
    VirtualCHRO.name: VirtualCHRO,
}

__all__ = [
    # Base
    "AgentTemplate",
    # Templates
    "CallCenterAgent",
    "CodeBuilderAgent",
    "ComplianceAgent",
    "DeepResearcherAgent",
    "DevOpsAgent",
    "EmailAgent",
    "FinanceManagerAgent",
    "InvestmentBankerAgent",
    "MarketingAgent",
    "PortfolioManagerAgent",
    "SecurityAgent",
    "VirtualCFO",
    "VirtualCHRO",
    "VirtualCMO",
    # Registry
    "TEMPLATES",
]
