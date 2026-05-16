"""Traffic-splitting between a prod prompt tag and a canary tag.

The router answers ONE question on every agent turn:

    "Which version of ``prompt_name`` should I render this turn with?"

Defaults to the version tagged ``prod``. When a ``canary`` tag exists,
a configurable fraction of traffic is steered to the canary so live
A/B numbers can accumulate in the eval registry. The decision is
deterministic per ``(tenant, query_hash, seed)`` triple — same request
always sees the same arm, so any external sticky-session layer
behaves correctly.

This module is purely a *picker*: it does not call the LLM. It returns
the version number to use plus a label (`"prod"` / `"canary"`) so
downstream code can tag the resulting trace + eval score correctly.

Talks to ``duxx-server`` over RESP via the standard ``redis`` client.
The Phase 7.2 commands it uses are read-only — no writes from this
module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PromptChoice:
    """One traffic-split decision for a single agent turn."""

    name: str
    version: int
    tag: str  # "prod" or "canary"
    content: str


class PromptRouter:
    """Picks which version of ``prompt_name`` to use this turn.

    Parameters
    ----------
    client:
        A connected ``redis.Redis`` instance pointed at ``duxx-server``.
    prompt_name:
        The prompt under self-improvement. ``PROMPT.GET name prod``
        must already exist when the agent starts.
    canary_traffic_pct:
        Fraction of traffic to steer to the ``canary`` tag (if it
        exists). 0.0 disables canary entirely. 0.10 is a safe default.
    seed:
        Hashing seed so multiple ``PromptRouter`` instances in
        different processes pick consistently. Same seed + same
        (tenant, query) always picks the same arm.
    """

    def __init__(
        self,
        client: Any,
        prompt_name: str,
        *,
        canary_traffic_pct: float = 0.10,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= canary_traffic_pct <= 1.0:
            raise ValueError(
                f"canary_traffic_pct must be in [0, 1], got {canary_traffic_pct}"
            )
        self._client = client
        self.prompt_name = prompt_name
        self.canary_traffic_pct = canary_traffic_pct
        self.seed = seed

    # ---------------------------------------------------------- public

    def pick(
        self,
        *,
        tenant: str = "default",
        query: str = "",
    ) -> PromptChoice:
        """Decide which prompt version this turn should use.

        Strictly deterministic for a given ``(tenant, query, seed)``
        triple so traffic-splitting is sticky across retries.
        """
        prod = self._fetch_tagged("prod")
        if prod is None:
            raise RuntimeError(
                f"PromptRouter: prompt {self.prompt_name!r} has no 'prod' tag. "
                "Call `client.execute_command('PROMPT.TAG', name, version, 'prod')` "
                "before starting the self-improvement loop."
            )

        if self.canary_traffic_pct <= 0.0:
            return prod

        canary = self._fetch_tagged("canary")
        if canary is None or canary.version == prod.version:
            return prod

        if self._roll(tenant, query) < self.canary_traffic_pct:
            return canary
        return prod

    # ---------------------------------------------------------- helpers

    def _fetch_tagged(self, tag: str) -> PromptChoice | None:
        try:
            raw = self._client.execute_command(
                "PROMPT.GET", self.prompt_name, tag
            )
        except Exception as exc:  # noqa: BLE001 — RESP errors are stringly typed
            logger.warning(
                "PromptRouter: PROMPT.GET %s %s failed: %s",
                self.prompt_name,
                tag,
                exc,
            )
            return None
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        return PromptChoice(
            name=payload["name"],
            version=int(payload["version"]),
            tag=tag,
            content=payload["content"],
        )

    def _roll(self, tenant: str, query: str) -> float:
        """Hash the (tenant, query, seed) tuple into [0, 1).

        Hashing rather than ``random.random()`` so two replicas of the
        same agent process steer the same request to the same arm.
        ``query`` may be empty for callers without one; we fall back
        to a fresh random number in that case.
        """
        if not query:
            return random.random()
        h = hashlib.sha1(
            f"{self.seed}\0{tenant}\0{query}".encode("utf-8"),
            usedforsecurity=False,
        ).digest()
        # Take the first 4 bytes as a big-endian uint32 → 32-bit float in [0, 1)
        return int.from_bytes(h[:4], "big") / 2**32
