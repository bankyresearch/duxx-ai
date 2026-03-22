"""Adaptive model router — routes tasks to the optimal model based on complexity, cost, and capability."""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from duxx_ai.core.llm import LLMConfig, LLMResponse, create_provider
from duxx_ai.core.message import Conversation

logger = logging.getLogger(__name__)


class ModelTier(BaseModel):
    name: str
    config: LLMConfig
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_complexity: float = 1.0  # 0.0 = simplest, 1.0 = most complex
    avg_latency_ms: float = 0.0
    capabilities: list[str] = Field(default_factory=list)  # e.g., ["tool_use", "vision", "code"]


class RoutingDecision(BaseModel):
    selected_tier: str
    reason: str
    estimated_cost: float = 0.0
    complexity_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingStats(BaseModel):
    total_requests: int = 0
    requests_per_tier: dict[str, int] = Field(default_factory=dict)
    total_cost: float = 0.0
    cost_per_tier: dict[str, float] = Field(default_factory=dict)
    avg_latency_per_tier: dict[str, float] = Field(default_factory=dict)


class ComplexityEstimator:
    """Estimates task complexity to inform routing decisions."""

    # Indicators of higher complexity
    COMPLEX_KEYWORDS = [
        "analyze", "compare", "synthesize", "evaluate", "design", "architect",
        "debug", "optimize", "refactor", "multi-step", "reason", "explain why",
        "trade-off", "pros and cons", "strategy",
    ]

    SIMPLE_KEYWORDS = [
        "list", "format", "convert", "translate", "summarize", "extract",
        "lookup", "define", "calculate", "count",
    ]

    def estimate(self, text: str, conversation: Conversation | None = None) -> float:
        text_lower = text.lower()
        score = 0.3  # base complexity

        # Keyword analysis
        for kw in self.COMPLEX_KEYWORDS:
            if kw in text_lower:
                score += 0.08
        for kw in self.SIMPLE_KEYWORDS:
            if kw in text_lower:
                score -= 0.05

        # Length heuristic — longer prompts tend to be more complex
        word_count = len(text.split())
        if word_count > 200:
            score += 0.15
        elif word_count > 100:
            score += 0.08

        # Conversation depth
        if conversation and len(conversation.messages) > 6:
            score += 0.1

        # Tool use indicators
        if any(w in text_lower for w in ["code", "function", "api", "database", "query"]):
            score += 0.1

        return max(0.0, min(1.0, score))


class AdaptiveRouter:
    """Routes requests to the optimal model tier based on complexity and cost constraints."""

    def __init__(
        self,
        tiers: list[ModelTier] | None = None,
        budget_limit: float | None = None,
        prefer_speed: bool = False,
    ) -> None:
        self.tiers: dict[str, ModelTier] = {}
        self.estimator = ComplexityEstimator()
        self.stats = RoutingStats()
        self.budget_limit = budget_limit
        self.prefer_speed = prefer_speed

        for tier in tiers or []:
            self.add_tier(tier)

    def add_tier(self, tier: ModelTier) -> AdaptiveRouter:
        self.tiers[tier.name] = tier
        return self

    def route(self, text: str, conversation: Conversation | None = None, required_capabilities: list[str] | None = None) -> RoutingDecision:
        complexity = self.estimator.estimate(text, conversation)
        candidates = list(self.tiers.values())

        # Filter by required capabilities
        if required_capabilities:
            candidates = [
                t for t in candidates
                if all(cap in t.capabilities for cap in required_capabilities)
            ]

        if not candidates:
            # Fallback to highest tier
            candidates = sorted(self.tiers.values(), key=lambda t: t.max_complexity, reverse=True)

        # Budget check
        if self.budget_limit is not None and self.stats.total_cost >= self.budget_limit:
            # Force cheapest tier
            candidates.sort(key=lambda t: t.cost_per_1k_input)
            selected = candidates[0]
            return RoutingDecision(
                selected_tier=selected.name,
                reason="Budget limit reached, using cheapest tier",
                complexity_score=complexity,
            )

        # Match complexity to tier
        candidates.sort(key=lambda t: t.max_complexity)
        selected = candidates[-1]  # default to highest

        for tier in candidates:
            if tier.max_complexity >= complexity:
                selected = tier
                break

        # Speed preference
        if self.prefer_speed and len(candidates) > 1:
            selected = min(candidates, key=lambda t: t.avg_latency_ms if t.max_complexity >= complexity * 0.8 else float("inf"))

        return RoutingDecision(
            selected_tier=selected.name,
            reason=f"Complexity {complexity:.2f} routed to {selected.name}",
            complexity_score=complexity,
            estimated_cost=selected.cost_per_1k_input * len(text.split()) / 750,
        )

    async def complete(
        self,
        text: str,
        conversation: Conversation | None = None,
        required_capabilities: list[str] | None = None,
    ) -> tuple[LLMResponse, RoutingDecision]:
        decision = self.route(text, conversation, required_capabilities)
        tier = self.tiers[decision.selected_tier]
        provider = create_provider(tier.config)

        conv = conversation or Conversation()
        from duxx_ai.core.message import Message, Role
        conv.add(Message(role=Role.USER, content=text))

        start = time.monotonic()
        response = await provider.complete(conv)
        latency = (time.monotonic() - start) * 1000

        # Update stats
        self.stats.total_requests += 1
        self.stats.requests_per_tier[tier.name] = self.stats.requests_per_tier.get(tier.name, 0) + 1
        tokens = response.usage.get("total_tokens", 0)
        cost = (tokens / 1000) * tier.cost_per_1k_input
        self.stats.total_cost += cost
        self.stats.cost_per_tier[tier.name] = self.stats.cost_per_tier.get(tier.name, 0) + cost

        # Running average latency
        prev = self.stats.avg_latency_per_tier.get(tier.name, latency)
        count = self.stats.requests_per_tier[tier.name]
        self.stats.avg_latency_per_tier[tier.name] = prev + (latency - prev) / count

        return response, decision
