"""Duxx AI — Enterprise Agentic SDK.

Build, fine-tune, orchestrate, and govern AI agents at scale.
"""

__version__ = "0.2.0"

from duxx_ai.core.agent import Agent, AgentConfig
from duxx_ai.core.tool import Tool, tool
from duxx_ai.core.message import Message, Role
from duxx_ai.orchestration.graph import Graph, Node, Edge
from duxx_ai.orchestration.crew import Crew, CrewAgent
from duxx_ai.router.adaptive import AdaptiveRouter
from duxx_ai.governance.guardrails import Guardrail, GuardrailChain
from duxx_ai.governance.audit import AuditLog
from duxx_ai.memory.manager import MemoryManager
from duxx_ai.observability.tracer import Tracer, OTelExporter
from duxx_ai.observability.evaluator import AgentEvaluator, EvalCase, EvalResult
from duxx_ai.orchestration.graph import GraphInterrupt, append_reducer, sum_reducer, merge_dict_reducer

__all__ = [
    "Agent",
    "AgentConfig",
    "Tool",
    "tool",
    "Message",
    "Role",
    "Graph",
    "Node",
    "Edge",
    "Crew",
    "CrewAgent",
    "AdaptiveRouter",
    "Guardrail",
    "GuardrailChain",
    "AuditLog",
    "MemoryManager",
    "Tracer",
    "OTelExporter",
    "AgentEvaluator",
    "EvalCase",
    "EvalResult",
    "GraphInterrupt",
    "append_reducer",
    "sum_reducer",
    "merge_dict_reducer",
]
