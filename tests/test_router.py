"""Tests for the adaptive model router."""

from duxx_ai.core.llm import LLMConfig
from duxx_ai.router.adaptive import AdaptiveRouter, ComplexityEstimator, ModelTier


class TestComplexityEstimator:
    def test_simple_query(self):
        est = ComplexityEstimator()
        score = est.estimate("What is 2+2?")
        assert score < 0.4

    def test_complex_query(self):
        est = ComplexityEstimator()
        score = est.estimate(
            "Analyze the trade-offs between microservices and monolithic architecture "
            "considering scalability, maintenance, and deployment complexity"
        )
        assert score > 0.4

    def test_code_query(self):
        est = ComplexityEstimator()
        score = est.estimate("Debug this function that queries the database API")
        assert score > 0.3


class TestAdaptiveRouter:
    def setup_method(self):
        self.router = AdaptiveRouter(
            tiers=[
                ModelTier(
                    name="small",
                    config=LLMConfig(model="small-model"),
                    max_complexity=0.3,
                    cost_per_1k_input=0.0001,
                ),
                ModelTier(
                    name="medium",
                    config=LLMConfig(model="medium-model"),
                    max_complexity=0.6,
                    cost_per_1k_input=0.001,
                    capabilities=["tool_use"],
                ),
                ModelTier(
                    name="large",
                    config=LLMConfig(model="large-model"),
                    max_complexity=1.0,
                    cost_per_1k_input=0.01,
                    capabilities=["tool_use", "vision", "code"],
                ),
            ]
        )

    def test_simple_routes_to_small(self):
        decision = self.router.route("List the days of the week")
        assert decision.selected_tier == "small"

    def test_complex_routes_to_large(self):
        decision = self.router.route(
            "Analyze and compare the architectural trade-offs between event-driven "
            "and request-response patterns for a high-throughput financial trading system, "
            "considering latency, fault tolerance, and regulatory compliance requirements"
        )
        assert decision.selected_tier in ("medium", "large")

    def test_capability_filter(self):
        decision = self.router.route("What is 1+1?", required_capabilities=["vision"])
        assert decision.selected_tier == "large"  # Only large has vision

    def test_budget_limit(self):
        router = AdaptiveRouter(
            tiers=[
                ModelTier(name="cheap", config=LLMConfig(), max_complexity=0.3, cost_per_1k_input=0.0001),
                ModelTier(name="expensive", config=LLMConfig(), max_complexity=1.0, cost_per_1k_input=0.1),
            ],
            budget_limit=0.0,  # Zero budget
        )
        decision = router.route("Complex analysis task")
        assert decision.selected_tier == "cheap"
        assert "budget" in decision.reason.lower()
