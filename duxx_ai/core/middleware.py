"""Middleware — prompt caching, content moderation, and request/response hooks.

Middleware wraps LLM calls to add cross-cutting concerns like caching,
moderation, logging, and rate limiting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from duxx_ai.core.message import Conversation

logger = logging.getLogger(__name__)


class Middleware(ABC):
    """Base middleware class. Wraps LLM calls."""

    @abstractmethod
    async def before_call(self, conversation: Conversation, system_prompt: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Called before LLM call. Returns modified metadata."""
        ...

    @abstractmethod
    async def after_call(self, response: str, metadata: dict[str, Any]) -> str:
        """Called after LLM call. Returns modified response."""
        ...


class PromptCacheMiddleware(Middleware):
    """Cache LLM responses for identical prompts.

    Supports TTL-based expiration and configurable cache key generation.

    Usage:
        cache_mw = PromptCacheMiddleware(ttl_seconds=300)
        # Use with Agent or directly
    """

    def __init__(self, ttl_seconds: float = 300, max_entries: int = 1000) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._cache: dict[str, tuple[float, str]] = {}
        self._hits = 0
        self._misses = 0

    def _make_key(self, conversation: Conversation, system_prompt: str) -> str:
        msgs = [(m.role.value, m.content) for m in conversation.messages[-5:]]
        raw = json.dumps({"msgs": msgs, "system": system_prompt}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    async def before_call(self, conversation: Conversation, system_prompt: str, metadata: dict[str, Any]) -> dict[str, Any]:
        key = self._make_key(conversation, system_prompt)
        if key in self._cache:
            cached_time, cached_response = self._cache[key]
            if time.time() - cached_time < self.ttl:
                self._hits += 1
                metadata["_cached"] = True
                metadata["_cached_response"] = cached_response
                return metadata
        self._misses += 1
        metadata["_cache_key"] = key
        return metadata

    async def after_call(self, response: str, metadata: dict[str, Any]) -> str:
        key = metadata.get("_cache_key")
        if key and not metadata.get("_cached"):
            self._cache[key] = (time.time(), response)
            # Evict oldest if over limit
            if len(self._cache) > self.max_entries:
                oldest = min(self._cache.items(), key=lambda x: x[1][0])
                del self._cache[oldest[0]]
        return response

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


class ContentModerationMiddleware(Middleware):
    """Moderate LLM input/output for harmful content.

    Checks for profanity, PII, prompt injection, and custom patterns.

    Usage:
        mod = ContentModerationMiddleware(block_pii=True, block_profanity=True)
    """

    PROFANITY_PATTERNS = [
        r"\b(damn|hell|crap)\b",  # Mild — customize as needed
    ]

    PII_PATTERNS = [
        r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    ]

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+",
        r"forget\s+(everything|all)",
        r"new\s+instructions?\s*:",
        r"system\s*:\s*",
    ]

    def __init__(
        self,
        block_pii: bool = True,
        block_profanity: bool = False,
        block_injection: bool = True,
        custom_blocked_patterns: list[str] | None = None,
        action: str = "block",  # "block", "redact", "warn"
    ) -> None:
        self.block_pii = block_pii
        self.block_profanity = block_profanity
        self.block_injection = block_injection
        self.custom_patterns = custom_blocked_patterns or []
        self.action = action
        self._violations: list[dict[str, Any]] = []

    async def before_call(self, conversation: Conversation, system_prompt: str, metadata: dict[str, Any]) -> dict[str, Any]:
        if conversation.messages:
            last_msg = conversation.messages[-1].content
            violations = self._check(last_msg)
            if violations:
                self._violations.extend(violations)
                if self.action == "block":
                    metadata["_blocked"] = True
                    metadata["_block_reason"] = f"Content moderation: {', '.join(v['type'] for v in violations)}"
                elif self.action == "redact":
                    metadata["_redacted_input"] = self._redact(last_msg)
        return metadata

    async def after_call(self, response: str, metadata: dict[str, Any]) -> str:
        violations = self._check(response)
        if violations:
            self._violations.extend(violations)
            if self.action == "block":
                return "[Response blocked by content moderation]"
            elif self.action == "redact":
                return self._redact(response)
        return response

    def _check(self, text: str) -> list[dict[str, Any]]:
        violations = []
        if self.block_pii:
            for pattern in self.PII_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append({"type": "pii", "pattern": pattern})
        if self.block_profanity:
            for pattern in self.PROFANITY_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append({"type": "profanity", "pattern": pattern})
        if self.block_injection:
            for pattern in self.INJECTION_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append({"type": "prompt_injection", "pattern": pattern})
        for pattern in self.custom_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({"type": "custom", "pattern": pattern})
        return violations

    def _redact(self, text: str) -> str:
        for pattern in self.PII_PATTERNS:
            text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
        return text

    @property
    def violations(self) -> list[dict[str, Any]]:
        return list(self._violations)


class LoggingMiddleware(Middleware):
    """Log all LLM calls with timing and metadata.

    Usage:
        log_mw = LoggingMiddleware(log_prompts=True, log_responses=False)
    """

    def __init__(self, log_prompts: bool = True, log_responses: bool = False, logger_name: str = "duxx_ai.llm") -> None:
        self._logger = logging.getLogger(logger_name)
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self._call_count = 0

    async def before_call(self, conversation: Conversation, system_prompt: str, metadata: dict[str, Any]) -> dict[str, Any]:
        self._call_count += 1
        metadata["_call_id"] = self._call_count
        metadata["_start_time"] = time.time()
        if self.log_prompts and conversation.messages:
            self._logger.info(f"LLM Call #{self._call_count}: {conversation.messages[-1].content[:100]}...")
        return metadata

    async def after_call(self, response: str, metadata: dict[str, Any]) -> str:
        duration = time.time() - metadata.get("_start_time", time.time())
        call_id = metadata.get("_call_id", 0)
        self._logger.info(f"LLM Call #{call_id} completed in {duration:.2f}s ({len(response)} chars)")
        if self.log_responses:
            self._logger.debug(f"Response: {response[:200]}...")
        return response


class RateLimitMiddleware(Middleware):
    """Enforce rate limits on LLM calls.

    Usage:
        rate_mw = RateLimitMiddleware(max_calls_per_minute=60)
    """

    def __init__(self, max_calls_per_minute: int = 60) -> None:
        self.max_rpm = max_calls_per_minute
        self._calls: list[float] = []

    async def before_call(self, conversation: Conversation, system_prompt: str, metadata: dict[str, Any]) -> dict[str, Any]:
        import asyncio
        now = time.time()
        # Remove calls older than 60 seconds
        self._calls = [t for t in self._calls if now - t < 60]
        if len(self._calls) >= self.max_rpm:
            wait_time = 60 - (now - self._calls[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached ({self.max_rpm}/min). Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
        self._calls.append(time.time())
        return metadata

    async def after_call(self, response: str, metadata: dict[str, Any]) -> str:
        return response


class MiddlewareChain:
    """Chain multiple middleware together.

    Usage:
        chain = MiddlewareChain([
            PromptCacheMiddleware(ttl_seconds=300),
            ContentModerationMiddleware(block_pii=True),
            LoggingMiddleware(),
        ])
        metadata = await chain.before_call(conversation, system_prompt, {})
        response = await chain.after_call(response, metadata)
    """

    def __init__(self, middlewares: list[Middleware] | None = None) -> None:
        self.middlewares = middlewares or []

    def add(self, middleware: Middleware) -> MiddlewareChain:
        self.middlewares.append(middleware)
        return self

    async def before_call(self, conversation: Conversation, system_prompt: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = metadata or {}
        for mw in self.middlewares:
            metadata = await mw.before_call(conversation, system_prompt, metadata)
            if metadata.get("_blocked"):
                break
        return metadata

    async def after_call(self, response: str, metadata: dict[str, Any]) -> str:
        for mw in reversed(self.middlewares):
            response = await mw.after_call(response, metadata)
        return response
