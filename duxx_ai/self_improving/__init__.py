"""Self-improving agents — duxx-ai's headline differentiator.

This module wires four DuxxDB Phase 7 primitives — ``PROMPT.*``,
``EVAL.*``, ``TRACE.*``, ``COST.*`` — into a closed feedback loop
that detects which prompts fail in production, drafts improvements,
canary-routes a slice of traffic to them, and promotes winners.
Automatically. While the agent serves real traffic.

The loop is short enough to fit on one slide:

::

    every agent turn ──► EVAL.SCORE (live continuous eval)
                            │
                            ▼
    background tick ──► EVAL.CLUSTER_FAILURES (mine semantic clusters)
                            │
                            ▼
                       LlmCandidateGenerator  (or any plug-in)
                            │
                            ▼
                       PROMPT.PUT + PROMPT.TAG canary
                            │
                            ▼
    next agent turn ──► PromptRouter steers ~10% to canary
                            │
                            ▼
    enough samples? ──► compare canary vs prod pass-rate
                            │
                            ▼
                       PROMPT.TAG prod  (promote canary)
                       or PROMPT.UNTAG canary  (retire)

Nothing in the rest of duxx-ai (or in LangChain/CrewAI/Portkey) has
all four primitives in one transactional store, which is why the
loop closes here. Every other framework needs three SaaS products
glued together to do part of this.

Worked example
--------------

.. code-block:: python

    import redis
    from duxx_ai.self_improving import SelfImprovingAgent

    client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    # First-time setup: seed the prompt and tag a "prod" version.
    v = int(client.execute_command(
        "PROMPT.PUT", "refund_classifier",
        "You are a refund-classification agent. ...",
    ))
    client.execute_command("PROMPT.TAG", "refund_classifier", str(v), "prod")

    def chat(messages):
        # Plug any LLM in here.
        return call_openai(messages)

    def score_refund_reply(user_msg: str, agent_reply: str) -> float:
        # Caller-owned rubric.
        return 1.0 if "REFUND" in agent_reply.upper() else 0.0

    agent = SelfImprovingAgent(
        client=client,
        prompt_name="refund_classifier",
        chat=chat,
        scorer=score_refund_reply,
    )
    agent.start()

    # Now use it as a normal agent. The loop keeps improving the
    # prompt behind your back as long as the process is alive.
    print(agent.run("I want a refund for order #9910"))
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from .candidate import (
    CandidateGenerator,
    CandidateProposal,
    FailureSample,
    LlmCandidateGenerator,
    StaticAppendGenerator,
)
from .loop import ImprovementLoop, LoopConfig
from .router import PromptChoice, PromptRouter
from .scorer import (
    EvalRunRegistry,
    LlmJudgeScorer,
    Scorer,
    TurnRecorder,
    TurnResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SelfImprovingAgent",
    # Components, exposed for callers who want to plug in their own pieces.
    "PromptRouter",
    "PromptChoice",
    "EvalRunRegistry",
    "TurnRecorder",
    "TurnResult",
    "Scorer",
    "LlmJudgeScorer",
    "CandidateGenerator",
    "LlmCandidateGenerator",
    "StaticAppendGenerator",
    "FailureSample",
    "CandidateProposal",
    "ImprovementLoop",
    "LoopConfig",
]


class SelfImprovingAgent:
    """An LLM-backed agent that improves its prompt in production.

    Conceptually:

    * Every ``run(user_input)`` reads the currently-tagged ``prod``
      (or canary) version of ``prompt_name`` from DuxxDB, calls the
      provided ``chat`` callable with the rendered messages, scores
      the reply, and writes the score back via ``EVAL.SCORE``.
    * A background :class:`ImprovementLoop` watches the eval stream,
      drafts candidate prompts when failure clusters emerge, tags
      them as canaries, and promotes them after enough A/B samples.

    Designed to be drop-in over an existing chat call. Single
    ``run`` returns the agent reply as a string; the score plus
    routing metadata live on :attr:`last_turn` so callers can log
    them or surface in telemetry.

    Parameters
    ----------
    client:
        Connected ``redis.Redis`` against ``duxx-server``.
    prompt_name:
        Name of the prompt under self-improvement. A version tagged
        ``prod`` must already exist — call
        ``PROMPT.PUT name content`` + ``PROMPT.TAG name version prod``
        once during setup.
    chat:
        Callable ``(messages: list[dict]) -> str``. The standard
        OpenAI-style chat shape: ``{"role": "...", "content": "..."}``.
        duxx-ai stays provider-agnostic — pass any wrapper that
        returns the assistant's text.
    scorer:
        Callable ``(user_input, agent_reply) -> float`` in ``[0, 1]``.
        If ``None``, defaults to :class:`LlmJudgeScorer` over the
        same ``chat`` callable. Pass exact_match / regex / custom
        rubrics here for cheaper deterministic scoring.
    candidate_generator:
        Plug-in that proposes a new prompt from a failure cluster.
        Defaults to :class:`LlmCandidateGenerator` over ``chat``.
    canary_traffic_pct:
        Fraction of live traffic to steer to a canary version
        (when one exists). 0.0 disables canary entirely. Default
        0.10.
    loop_config:
        Override the default loop tunables (promotion threshold,
        sample sizes, tick rate, etc.). See :class:`LoopConfig`.
    autostart:
        If True (default), the background loop starts on
        construction. Pass False to run cycles manually via
        :meth:`cycle_once` — useful in tests.
    """

    def __init__(
        self,
        client: Any,
        *,
        prompt_name: str,
        chat: Callable[[list[dict]], str],
        scorer: Scorer | None = None,
        candidate_generator: CandidateGenerator | None = None,
        canary_traffic_pct: float = 0.10,
        loop_config: LoopConfig | None = None,
        autostart: bool = True,
    ) -> None:
        self._client = client
        self.prompt_name = prompt_name
        self._chat = chat
        self._scorer: Scorer = scorer or LlmJudgeScorer(chat)
        self.router = PromptRouter(
            client,
            prompt_name=prompt_name,
            canary_traffic_pct=canary_traffic_pct,
        )
        self.eval_runs = EvalRunRegistry(client)
        self.recorder = TurnRecorder(client, self.eval_runs)
        self.loop = ImprovementLoop(
            client,
            prompt_name=prompt_name,
            eval_runs=self.eval_runs,
            candidate_generator=candidate_generator or LlmCandidateGenerator(chat),
            config=loop_config,
        )
        self.last_turn: TurnResult | None = None
        if autostart:
            self.start()

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Start the background improvement loop."""
        self.loop.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the background loop. Pending ``EVAL.SCORE`` writes are NOT cancelled."""
        self.loop.stop(timeout=timeout)

    def cycle_once(self) -> None:
        """Run exactly one loop cycle synchronously.

        Useful in tests, or in a cron-driven setup where you'd
        rather not have a background thread. Same code path as the
        background loop.
        """
        self.loop.cycle_once()

    # ----------------------------------------------------------- run

    def run(
        self,
        user_input: str,
        *,
        tenant: str = "default",
        history: list[dict] | None = None,
    ) -> str:
        """Serve one agent turn.

        Renders the current ``prod`` (or, with probability
        ``canary_traffic_pct``, the current ``canary``) version of
        ``prompt_name`` as the system message, prepends it to the
        supplied ``history``, calls ``chat``, and returns the reply.
        The score is recorded under :attr:`last_turn`.

        ``user_input`` and ``tenant`` together drive the
        traffic-split hash so retries land on the same arm.
        """
        choice = self.router.pick(tenant=tenant, query=user_input)
        messages: list[dict] = [{"role": "system", "content": choice.content}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        try:
            reply = self._chat(messages)
        except Exception:
            # Score 0 + re-raise so caller can decide. We don't
            # silently swallow upstream LLM failures.
            self.last_turn = self.recorder.record(
                prompt_name=choice.name,
                prompt_version=choice.version,
                prompt_tag=choice.tag,
                input_text=user_input,
                output_text="",
                score=0.0,
                notes={"error": "chat_call_failed"},
            )
            raise

        try:
            score = float(self._scorer(user_input, reply))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "self_improving: scorer raised, recording 0.0: %s", exc
            )
            score = 0.0

        self.last_turn = self.recorder.record(
            prompt_name=choice.name,
            prompt_version=choice.version,
            prompt_tag=choice.tag,
            input_text=user_input,
            output_text=reply,
            score=score,
        )
        return reply

    # ----------------------------------------------------------- introspection

    @property
    def stats(self) -> dict:
        """Live counters from the background loop."""
        return dict(self.loop.stats)

    def __enter__(self) -> "SelfImprovingAgent":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.stop()
