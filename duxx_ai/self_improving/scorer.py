"""Scoring + result-recording for the self-improvement loop.

Every agent turn that goes through :class:`SelfImprovingAgent`
flows through this module:

1. Pre-run: the turn is opened against an :class:`EvalRunHandle`
   (one ``EVAL.START`` per (prompt_name, prompt_version) bucket).
2. Post-run: the supplied scorer function rates the agent's
   reply on ``[0, 1]``. The score plus the output text are
   appended via ``EVAL.SCORE``. The output text is what
   ``EVAL.CLUSTER_FAILURES`` will later use for semantic clustering
   of failure modes.

The default scorer is a thin LLM-as-judge implementation
(:class:`LlmJudgeScorer`). Callers can supply any callable
``(input_text, output_text) -> float`` — exact-match, regex,
custom rubric, or a remote scoring service.

Run reuse: we open one eval run per (prompt_name, prompt_version)
combination per :class:`SelfImprovingAgent` instance, so a Phase 7
``EVAL.COMPARE prod_run canary_run`` directly gives us the
prod-vs-canary delta the loop needs to decide on promotion.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


# A scorer is just a callable. Kept as a Protocol so callers can pass
# bound methods, lambdas, or class instances interchangeably.
class Scorer(Protocol):
    def __call__(self, input_text: str, output_text: str) -> float: ...


@dataclass
class TurnResult:
    """Outcome of one ``SelfImprovingAgent.run`` call."""

    run_id: str
    prompt_name: str
    prompt_version: int
    prompt_tag: str  # "prod" or "canary"
    score: float
    output_text: str


class EvalRunRegistry:
    """Cache of (prompt_name, prompt_version, scorer_name) → run_id.

    Opens one eval run per (prompt, version) bucket the first time
    we see it, then reuses it for every subsequent turn against
    that version. This is what lets ``EVAL.COMPARE`` between the
    prod and canary versions Just Work — the canary's run_id is
    the one we've been scoring against the whole time.
    """

    def __init__(
        self,
        client: Any,
        *,
        dataset_name: str = "self_improving.live",
        dataset_version: int = 1,
        scorer_name: str = "self_improving.scorer",
        model_label: str = "agent",
    ) -> None:
        self._client = client
        self._dataset_name = dataset_name
        self._dataset_version = dataset_version
        self._scorer_name = scorer_name
        self._model_label = model_label
        # (prompt_name, prompt_version) → run_id
        self._cache: dict[tuple[str, int], str] = {}

    def run_id_for(self, prompt_name: str, prompt_version: int) -> str:
        cache_key = (prompt_name, prompt_version)
        if cache_key in self._cache:
            return self._cache[cache_key]

        raw = self._client.execute_command(
            "EVAL.START",
            self._dataset_name,
            str(self._dataset_version),
            prompt_name,
            str(prompt_version),
            self._model_label,
            self._scorer_name,
            json.dumps(
                {
                    "source": "duxx_ai.self_improving",
                    "kind": "live_continuous_eval",
                }
            ),
        )
        run_id = (
            raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        )
        self._cache[cache_key] = run_id
        return run_id

    def open_run_ids(self) -> dict[tuple[str, int], str]:
        """Snapshot of every (prompt, version) → run_id this instance opened.

        Used by the promotion loop to call ``EVAL.COMPARE`` between
        prod and canary runs without re-deriving the ids.
        """
        return dict(self._cache)


# Cap for the input_text we persist into eval-score notes. Bigger
# than the typical user message, small enough not to bloat the
# eval table. Truncated text is still useful to the LLM rewriter.
INPUT_TEXT_NOTE_CAP = 4096


class TurnRecorder:
    """One ``EVAL.SCORE`` write per agent turn.

    Generates a stable ``row_id`` so re-scoring the same input
    (e.g. on retry) overwrites rather than double-counts. Best-effort
    on RESP errors — a logging WARNING is the worst case, the agent
    keeps serving traffic.

    Two row-id strategies, in order of preference:

    * **Idempotency key** (recommended). Callers pass
      ``idempotency_key=`` on :meth:`record` — typically a request /
      message id from upstream, or the SHA1 of the input. Retrying
      the same turn produces the SAME ``row_id`` and ``EVAL.SCORE``
      overwrites in place, so retries don't double-count.
    * **Sequence fallback.** When no key is supplied we fall back to
      a per-bucket counter (``live-<version>-<seq>``). This preserves
      v0.2.x behavior for callers that never opted in, but DOES
      double-count on retry — which is exactly the bug review
      comment P2 flagged. Pass an idempotency key whenever you
      have one.
    """

    def __init__(
        self,
        client: Any,
        run_registry: EvalRunRegistry,
        *,
        drop_on_error: bool = True,
        input_text_note_cap: int = INPUT_TEXT_NOTE_CAP,
    ) -> None:
        self._client = client
        self._runs = run_registry
        self._drop_on_error = drop_on_error
        self._input_text_cap = max(0, int(input_text_note_cap))
        # row_id sequencer per (prompt, version) bucket so ids stay
        # roughly chronological within a run when no idempotency key
        # is supplied.
        self._row_seq: dict[tuple[str, int], int] = {}

    def record(
        self,
        *,
        prompt_name: str,
        prompt_version: int,
        prompt_tag: str,
        input_text: str,
        output_text: str,
        score: float,
        notes: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> TurnResult:
        """Score one turn and persist it to the eval registry.

        Pass ``idempotency_key`` (request id / message id / hash of
        the input) so retries overwrite rather than double-count.

        Returns the :class:`TurnResult` so the caller can log it to
        OpenTelemetry, surface it in the response, or feed it into
        a downstream RL signal.
        """
        run_id = self._runs.run_id_for(prompt_name, prompt_version)
        cache_key = (prompt_name, prompt_version)
        row_id = self._row_id_for(
            cache_key=cache_key,
            idempotency_key=idempotency_key,
        )

        annotated = dict(notes or {})
        annotated.setdefault("prompt_tag", prompt_tag)
        # Persist the ORIGINAL user input alongside the agent's
        # output, capped to avoid bloat. The candidate-prompt
        # generator reads this back at mining time so it can see
        # WHAT the user asked, not just what the agent got wrong.
        # (Review comment P1.)
        if input_text:
            annotated.setdefault(
                "input_text",
                input_text[: self._input_text_cap] if self._input_text_cap else "",
            )
            annotated.setdefault("input_text_len", len(input_text))
        if idempotency_key is not None:
            annotated.setdefault("idempotency_key", idempotency_key)

        try:
            self._client.execute_command(
                "EVAL.SCORE",
                run_id,
                row_id,
                f"{float(score):.6f}",
                output_text or "-",
                json.dumps(annotated),
            )
        except Exception as exc:  # noqa: BLE001
            if self._drop_on_error:
                logger.warning(
                    "TurnRecorder: EVAL.SCORE failed for %s/%s: %s",
                    prompt_name,
                    prompt_version,
                    exc,
                )
            else:
                raise

        return TurnResult(
            run_id=run_id,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            prompt_tag=prompt_tag,
            score=float(score),
            output_text=output_text,
        )

    def _row_id_for(
        self,
        *,
        cache_key: tuple[str, int],
        idempotency_key: str | None,
    ) -> str:
        if idempotency_key:
            # SHA1 keeps the row id short, ASCII-safe, and stable
            # across processes — two replicas seeing the same
            # idempotency key produce the same row_id, so the
            # second EVAL.SCORE overwrites the first.
            digest = hashlib.sha1(
                idempotency_key.encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:16]
            return f"idem-{cache_key[1]}-{digest}"
        seq = self._row_seq.get(cache_key, 0)
        self._row_seq[cache_key] = seq + 1
        return f"live-{cache_key[1]}-{seq}"


# ---------------------------------------------------------------- judges


class LlmJudgeScorer:
    """Default LLM-as-judge scorer.

    Callers wrap any chat callable ``(messages: list[dict]) -> str``
    so we don't take a hard dependency on a specific provider SDK.
    The judge prompt asks for a single integer ``0`` / ``1`` / ``2``
    which we normalize to ``0.0`` / ``0.5`` / ``1.0``.

    The judge is intentionally rubric-free in the default form. For
    domain-specific evaluation, pass a custom callable instead.
    """

    DEFAULT_SYSTEM = (
        "You are a strict evaluator. Read the user input and the agent "
        "response, then output exactly one integer:\n"
        "  2 = correct + helpful\n"
        "  1 = partially correct, salvageable\n"
        "  0 = incorrect or unhelpful\n"
        "Output ONLY the digit, nothing else."
    )

    def __init__(
        self,
        chat: Callable[[list[dict]], str],
        *,
        system_prompt: str = DEFAULT_SYSTEM,
    ) -> None:
        self._chat = chat
        self._system_prompt = system_prompt

    def __call__(self, input_text: str, output_text: str) -> float:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    f"USER INPUT:\n{input_text}\n\n"
                    f"AGENT RESPONSE:\n{output_text}"
                ),
            },
        ]
        raw = self._chat(messages).strip()
        # First char is what we asked for. Fail closed (0.0) if the
        # judge produced something unparseable.
        head = raw[:1]
        return {"2": 1.0, "1": 0.5, "0": 0.0}.get(head, 0.0)
