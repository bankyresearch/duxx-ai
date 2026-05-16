"""The improvement loop — mines failures, proposes canaries, promotes winners.

Runs in a daemon thread inside the agent process. Every ``tick_secs``
seconds (default 60) it executes one *cycle*:

1. **Mine failures** — call ``EVAL.CLUSTER_FAILURES`` on the prod run.
   The returned clusters are groups of semantically-similar failing
   outputs from the live traffic the agent has been serving.

2. **Propose candidate** — if there's no active canary and we have a
   cluster of failures, ask the configured :class:`CandidateGenerator`
   for a revised prompt. If it declines (no useful proposal),
   we skip and try again next tick.

3. **Register canary** — write the proposal via ``PROMPT.PUT`` and
   tag the new version with ``canary``. The router will start steering
   ``canary_traffic_pct`` of traffic to it on the next agent turn.

4. **Evaluate canary** — once the canary has accumulated at least
   ``min_canary_samples`` scored rows, compute its pass-rate. Compare
   against the prod pass-rate over the same recent window. Two
   outcomes:

   * **Promote**: canary pass-rate ≥ prod pass-rate + ``promote_threshold``
     → move the ``prod`` tag onto the canary version. The previous prod
     gets archived (its run id is closed). On the next agent turn the
     router serves the new prompt to 100% of traffic.
   * **Retire**: canary pass-rate < prod pass-rate − ``retire_threshold``
     → untag the canary and forget it. Prod is unaffected.
   * **Wait**: anything in between → keep collecting samples.

The loop is single-threaded. It owns a ``threading.Event`` you can
``set()`` from the outside to wake it up early (used in tests).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from .candidate import CandidateGenerator, FailureSample
from .scorer import EvalRunRegistry

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """Tunables for :class:`ImprovementLoop`."""

    tick_secs: float = 60.0
    min_canary_samples: int = 20
    min_prod_samples: int = 20
    promote_threshold: float = 0.03  # +3% pass-rate to promote
    retire_threshold: float = 0.05  # -5% pass-rate to retire
    score_threshold: float = 0.5  # row counts as "failure" below this
    sim_threshold: float = 0.8
    max_clusters: int = 5


class ImprovementLoop:
    """Background thread that runs the mine → propose → promote cycle.

    Parameters
    ----------
    client:
        Connected ``redis.Redis`` against ``duxx-server``.
    prompt_name:
        The prompt under self-improvement.
    eval_runs:
        Shared :class:`EvalRunRegistry`. The loop reads its open
        run ids to drive ``EVAL.CLUSTER_FAILURES`` + ``EVAL.LIST``
        comparisons.
    candidate_generator:
        Plug-in that turns a cluster into a revised prompt.
    config:
        Tunables. Defaults are conservative — bias toward fewer
        promotions, more samples per decision.
    """

    def __init__(
        self,
        client: Any,
        *,
        prompt_name: str,
        eval_runs: EvalRunRegistry,
        candidate_generator: CandidateGenerator,
        config: LoopConfig | None = None,
    ) -> None:
        self._client = client
        self.prompt_name = prompt_name
        self._runs = eval_runs
        self._gen = candidate_generator
        self.config = config or LoopConfig()

        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Surfaced for telemetry / tests.
        self.stats = {
            "ticks": 0,
            "candidates_proposed": 0,
            "promotions": 0,
            "retirements": 0,
            "errors": 0,
        }

    # ---------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="duxx-self-improving-loop", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def trigger(self) -> None:
        """Wake the loop immediately for one cycle. Test hook."""
        self._wake.set()

    def cycle_once(self) -> None:
        """Run exactly one cycle, synchronously. Test hook / cron path."""
        try:
            self._cycle()
        except Exception as exc:  # noqa: BLE001
            self.stats["errors"] += 1
            logger.exception("improvement loop cycle failed: %s", exc)

    # ---------------------------------------------------------- internals

    def _run(self) -> None:
        while not self._stop.is_set():
            self.cycle_once()
            # Sleep up to tick_secs, but wake early when triggered.
            self._wake.wait(timeout=self.config.tick_secs)
            self._wake.clear()

    def _cycle(self) -> None:
        self.stats["ticks"] += 1
        prod = self._current_tagged_version("prod")
        if prod is None:
            return  # caller hasn't tagged a prod version yet — nothing to do
        canary = self._current_tagged_version("canary")

        if canary is None or canary == prod:
            self._maybe_propose_canary(prod_version=prod)
        else:
            self._maybe_decide_canary(prod_version=prod, canary_version=canary)

    # ----- propose ---------------------------------------------------

    def _maybe_propose_canary(self, *, prod_version: int) -> None:
        run_id = self._runs.run_id_for(self.prompt_name, prod_version)
        clusters = self._cluster_failures(run_id)
        if not clusters:
            return

        # Convert the largest cluster into FailureSample rows.
        top = clusters[0]
        failures = [
            FailureSample(
                row_id=m["row_id"],
                score=float(m["score"]),
                output_text=m.get("output_text") or top.get("representative_text", ""),
            )
            for m in top.get("members", [])
        ]
        if not failures:
            return

        current_content = self._fetch_content(prod_version)
        proposal = self._gen.propose(
            prompt_name=self.prompt_name,
            current_content=current_content,
            current_version=prod_version,
            failures=failures,
        )
        if proposal is None:
            return

        new_version = int(
            self._client.execute_command(
                "PROMPT.PUT",
                self.prompt_name,
                proposal.content,
                json.dumps(
                    {
                        "source": "duxx_ai.self_improving",
                        "rationale": proposal.rationale,
                        "based_on_version": proposal.based_on_version,
                        "based_on_cluster_size": proposal.based_on_cluster_size,
                    }
                ),
            )
        )
        self._client.execute_command(
            "PROMPT.TAG", self.prompt_name, str(new_version), "canary"
        )
        self.stats["candidates_proposed"] += 1
        logger.info(
            "self_improving: proposed canary v%s for prompt %r (based on v%s, %s failures): %s",
            new_version,
            self.prompt_name,
            prod_version,
            proposal.based_on_cluster_size,
            proposal.rationale,
        )

    # ----- decide ----------------------------------------------------

    def _maybe_decide_canary(
        self, *, prod_version: int, canary_version: int
    ) -> None:
        prod_summary = self._pass_rate(self.prompt_name, prod_version)
        canary_summary = self._pass_rate(self.prompt_name, canary_version)
        if prod_summary is None or canary_summary is None:
            return
        prod_n, prod_rate = prod_summary
        canary_n, canary_rate = canary_summary

        if (
            prod_n < self.config.min_prod_samples
            or canary_n < self.config.min_canary_samples
        ):
            return  # not enough data yet — wait

        delta = canary_rate - prod_rate
        if delta >= self.config.promote_threshold:
            self._promote(canary_version=canary_version, delta=delta)
        elif delta <= -self.config.retire_threshold:
            self._retire(canary_version=canary_version, delta=delta)
        # else: keep collecting

    def _promote(self, *, canary_version: int, delta: float) -> None:
        # Move the prod tag onto the canary, drop the canary tag.
        self._client.execute_command(
            "PROMPT.TAG", self.prompt_name, str(canary_version), "prod"
        )
        self._client.execute_command(
            "PROMPT.UNTAG", self.prompt_name, "canary"
        )
        self.stats["promotions"] += 1
        logger.info(
            "self_improving: PROMOTED prompt %r canary v%s to prod (Δ pass-rate = %+.4f)",
            self.prompt_name,
            canary_version,
            delta,
        )

    def _retire(self, *, canary_version: int, delta: float) -> None:
        self._client.execute_command(
            "PROMPT.UNTAG", self.prompt_name, "canary"
        )
        self.stats["retirements"] += 1
        logger.info(
            "self_improving: RETIRED prompt %r canary v%s (Δ pass-rate = %+.4f)",
            self.prompt_name,
            canary_version,
            delta,
        )

    # ----- RESP helpers ---------------------------------------------

    def _current_tagged_version(self, tag: str) -> int | None:
        try:
            raw = self._client.execute_command(
                "PROMPT.GET", self.prompt_name, tag
            )
        except Exception:  # noqa: BLE001
            return None
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        return int(json.loads(raw)["version"])

    def _fetch_content(self, version: int) -> str:
        raw = self._client.execute_command(
            "PROMPT.GET", self.prompt_name, str(version)
        )
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        return json.loads(raw)["content"]

    def _cluster_failures(self, run_id: str) -> list[dict]:
        try:
            raw = self._client.execute_command(
                "EVAL.CLUSTER_FAILURES",
                run_id,
                f"{self.config.score_threshold:.4f}",
                f"{self.config.sim_threshold:.4f}",
                str(self.config.max_clusters),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("EVAL.CLUSTER_FAILURES failed for %s: %s", run_id, exc)
            return []
        return [_decode(b) for b in (raw or [])]

    def _pass_rate(
        self, prompt_name: str, prompt_version: int
    ) -> tuple[int, float] | None:
        """Average score over every scored row in this version's run."""
        run_id = self._runs.run_id_for(prompt_name, prompt_version)
        try:
            raw = self._client.execute_command("EVAL.SCORES", run_id) or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("EVAL.SCORES failed for %s: %s", run_id, exc)
            return None
        scores = [_decode(b) for b in raw]
        if not scores:
            return None
        n = len(scores)
        pass_rate = sum(
            1 for s in scores if float(s.get("score", 0.0)) >= 0.5
        ) / n
        return n, pass_rate


def _decode(blob) -> dict:
    """RESP bulk → dict. Same trick the duxx-ai server facade uses."""
    if blob is None:
        return {}
    if isinstance(blob, (bytes, bytearray)):
        blob = blob.decode("utf-8")
    if isinstance(blob, str):
        return json.loads(blob)
    return blob
