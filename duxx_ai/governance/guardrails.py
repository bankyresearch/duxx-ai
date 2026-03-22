"""Guardrails system for input/output safety and policy enforcement."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    passed: bool = True
    reason: str = ""
    guardrail_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Guardrail(ABC):
    """Base guardrail that checks input or output text."""

    name: str = "base_guardrail"

    @abstractmethod
    async def check(self, text: str, direction: str = "input") -> GuardrailResult: ...


class ContentFilterGuardrail(Guardrail):
    """Blocks messages containing prohibited patterns."""

    name = "content_filter"

    def __init__(self, blocked_patterns: list[str] | None = None) -> None:
        self.blocked_patterns = blocked_patterns or []

    async def check(self, text: str, direction: str = "input") -> GuardrailResult:
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    reason=f"Content matched blocked pattern: {pattern}",
                    guardrail_name=self.name,
                )
        return GuardrailResult(passed=True, guardrail_name=self.name)


class PIIGuardrail(Guardrail):
    """Detects and blocks personally identifiable information."""

    name = "pii_filter"

    PII_PATTERNS = [
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
        (r"\b\d{16}\b", "credit card number"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email address"),
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone number"),
    ]

    def __init__(self, allow_email: bool = False) -> None:
        self.allow_email = allow_email

    async def check(self, text: str, direction: str = "input") -> GuardrailResult:
        for pattern, label in self.PII_PATTERNS:
            if label == "email address" and self.allow_email:
                continue
            if re.search(pattern, text):
                return GuardrailResult(
                    passed=False,
                    reason=f"Detected PII ({label}) in {direction}",
                    guardrail_name=self.name,
                )
        return GuardrailResult(passed=True, guardrail_name=self.name)


class PromptInjectionGuardrail(Guardrail):
    """Detects common prompt injection patterns."""

    name = "prompt_injection"

    INJECTION_PATTERNS = [
        r"ignore (?:all )?(?:previous|above) instructions",
        r"you are now",
        r"new instructions:",
        r"system prompt:",
        r"forget (?:all )?(?:previous|your) (?:instructions|rules)",
        r"do not follow",
        r"override (?:all )?(?:safety|rules|guidelines)",
    ]

    async def check(self, text: str, direction: str = "input") -> GuardrailResult:
        if direction != "input":
            return GuardrailResult(passed=True, guardrail_name=self.name)

        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    reason=f"Potential prompt injection detected",
                    guardrail_name=self.name,
                )
        return GuardrailResult(passed=True, guardrail_name=self.name)


class TokenBudgetGuardrail(Guardrail):
    """Enforces token budget limits."""

    name = "token_budget"

    def __init__(self, max_tokens: int = 100_000) -> None:
        self.max_tokens = max_tokens
        self.used_tokens = 0

    async def check(self, text: str, direction: str = "input") -> GuardrailResult:
        # Rough estimate: 1 token ~= 4 characters
        estimated = len(text) // 4
        self.used_tokens += estimated
        if self.used_tokens > self.max_tokens:
            return GuardrailResult(
                passed=False,
                reason=f"Token budget exceeded: {self.used_tokens}/{self.max_tokens}",
                guardrail_name=self.name,
                metadata={"used": self.used_tokens, "limit": self.max_tokens},
            )
        return GuardrailResult(passed=True, guardrail_name=self.name)


class TopicGuardrail(Guardrail):
    """Restricts conversation to allowed topics."""

    name = "topic_filter"

    def __init__(self, allowed_topics: list[str] | None = None, blocked_topics: list[str] | None = None) -> None:
        self.allowed_topics = allowed_topics or []
        self.blocked_topics = blocked_topics or []

    async def check(self, text: str, direction: str = "input") -> GuardrailResult:
        text_lower = text.lower()

        # Check blocked topics first
        for topic in self.blocked_topics:
            if topic.lower() in text_lower:
                return GuardrailResult(
                    passed=False,
                    reason=f"Blocked topic detected: {topic}",
                    guardrail_name=self.name,
                )

        # If allowed_topics is set and non-empty, only allow messages
        # that match at least one allowed topic
        if self.allowed_topics:
            matched = any(topic.lower() in text_lower for topic in self.allowed_topics)
            if not matched:
                return GuardrailResult(
                    passed=False,
                    reason=f"Message does not match any allowed topic. Allowed: {self.allowed_topics}",
                    guardrail_name=self.name,
                )

        return GuardrailResult(passed=True, guardrail_name=self.name)


class GuardrailChain:
    """Runs multiple guardrails in sequence."""

    def __init__(self, guardrails: list[Guardrail] | None = None) -> None:
        self.guardrails = guardrails or []

    def add(self, guardrail: Guardrail) -> GuardrailChain:
        self.guardrails.append(guardrail)
        return self

    async def check_input(self, text: str) -> GuardrailResult:
        for g in self.guardrails:
            result = await g.check(text, direction="input")
            if not result.passed:
                return result
        return GuardrailResult(passed=True)

    async def check_output(self, text: str) -> GuardrailResult:
        for g in self.guardrails:
            result = await g.check(text, direction="output")
            if not result.passed:
                return result
        return GuardrailResult(passed=True)
