"""Core abstractions for Duxx AI agents, tools, and messages."""

from duxx_ai.core.agent import Agent, AgentConfig, AgentState
from duxx_ai.core.tool import Tool, ToolParameter, tool
from duxx_ai.core.message import Message, Conversation
from duxx_ai.core.llm import LLMConfig, LLMResponse, LLMProvider, LLMCache, RateLimiter, CachedProvider
from duxx_ai.core.patterns import (
    ReActAgent, ReActTrace, ThoughtStep,
    AgentHandoff, HandoffResult,
    SelfImprovingAgent, Reward, Skill,
    TeachableAgent,
    EvaluatorOptimizer, OptimizationResult,
    OrchestratorWorker, WorkerResult,
    ParallelGuardrails, GuardrailMode, GuardrailCheckResult,
    AgenticRAG, RAGResult,
)

__all__ = [
    # Agent
    "Agent", "AgentConfig", "AgentState",
    # Tool
    "Tool", "ToolParameter", "tool",
    # Message
    "Message", "Conversation",
    # LLM
    "LLMConfig", "LLMResponse", "LLMProvider", "LLMCache", "RateLimiter", "CachedProvider",
    # Patterns
    "ReActAgent", "ReActTrace", "ThoughtStep",
    "AgentHandoff", "HandoffResult",
    "SelfImprovingAgent", "Reward", "Skill",
    "TeachableAgent",
    "EvaluatorOptimizer", "OptimizationResult",
    "OrchestratorWorker", "WorkerResult",
    "ParallelGuardrails", "GuardrailMode", "GuardrailCheckResult",
    "AgenticRAG", "RAGResult",
]
