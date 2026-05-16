"""Candidate-prompt generation from failure clusters.

The promotion loop calls ``EVAL.CLUSTER_FAILURES`` on the prod run.
Each returned cluster is a group of semantically-similar failing
outputs. This module turns those clusters into a *candidate prompt*
that's hopefully more robust against the pattern of failures the
cluster captured.

We deliberately keep two implementations:

* :class:`LlmCandidateGenerator` — asks the caller's chat model to
  rewrite the prompt given a sample of failing rows. Default.
* :class:`StaticAppendGenerator` — appends a fixed instruction
  block. Useful in unit tests, demos with no LLM available, or as
  a sanity baseline against the LLM rewriter.

Both implementations return ``None`` when they have nothing useful
to propose (e.g. cluster too small, LLM declined to change the
prompt). The loop treats ``None`` as "skip this cycle, try again
later" — never as an error.

The generator does NOT write the candidate to the prompt registry.
That's the loop's job, so the registry write happens under the
same lock as the canary-tag assignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


@dataclass
class FailureSample:
    """One row inside a failure cluster, passed to candidate generators.

    Carries both the agent's wrong reply (``output_text``) AND the
    original user input that triggered it (``input_text``). The
    candidate generator NEEDS both: knowing what the user asked is
    what lets the LLM rewriter propose a meaningful rule. Without
    the input, the rewriter only sees repeated wrong answers.
    """

    row_id: str
    score: float
    output_text: str
    input_text: str = ""


@dataclass
class CandidateProposal:
    """One generated alternative prompt, awaiting registration."""

    content: str
    rationale: str
    based_on_version: int
    based_on_cluster_size: int


class CandidateGenerator(Protocol):
    def propose(
        self,
        *,
        prompt_name: str,
        current_content: str,
        current_version: int,
        failures: list[FailureSample],
    ) -> CandidateProposal | None: ...


# ---------------------------------------------------------------- LLM


class LlmCandidateGenerator:
    """Asks the caller's chat model to rewrite the prompt.

    The rewriter is given:

    * the current prompt content (as the agent saw it),
    * a sample of failure rows (max ``max_examples``, cheapest LLM
      cost while still being representative),
    * an instruction to PROPOSE A REWRITE that would handle the
      observed failures, or to RETURN UNCHANGED if it can't see an
      improvement.

    Parsing: we accept either ``<<<...>>>`` fences or a literal
    `"UNCHANGED"` marker. Anything else is rejected as malformed
    and the loop treats this cycle as a no-op.
    """

    DEFAULT_SYSTEM = (
        "You are a senior prompt engineer. You will be shown the current "
        "version of a production prompt, plus several recent failing "
        "responses the prompt produced. Your job is to propose a "
        "REVISED prompt that would have handled those failure cases.\n\n"
        "RULES:\n"
        "- Keep the rewrite focused. Do not change the core task.\n"
        "- Add a short rules section ONLY if the failures share a pattern.\n"
        "- If the failures look like LLM noise rather than a prompt bug, "
        "output the literal token UNCHANGED on its own line.\n\n"
        "OUTPUT FORMAT:\n"
        "First line: ONE-sentence rationale.\n"
        "Then fenced revised prompt between <<< and >>> on their own lines.\n"
        "If no change is warranted, output exactly UNCHANGED instead of "
        "the fenced block."
    )

    def __init__(
        self,
        chat: Callable[[list[dict]], str],
        *,
        system_prompt: str = DEFAULT_SYSTEM,
        max_examples: int = 6,
        min_cluster_size: int = 2,
    ) -> None:
        self._chat = chat
        self._system_prompt = system_prompt
        self._max_examples = max_examples
        self._min_cluster_size = min_cluster_size

    def propose(
        self,
        *,
        prompt_name: str,
        current_content: str,
        current_version: int,
        failures: list[FailureSample],
    ) -> CandidateProposal | None:
        if len(failures) < self._min_cluster_size:
            return None

        sample = failures[: self._max_examples]
        # Each failure example shows BOTH the user input that caused
        # the failure AND the agent's wrong reply. The rewriter needs
        # both to spot patterns (e.g. "users say 'return' instead of
        # 'refund' and the agent then misclassifies").
        def _fmt(idx: int, f: FailureSample) -> str:
            inp = (f.input_text or "(input not captured)").strip()
            out = (f.output_text or "(empty output)").strip()
            return (
                f"FAILURE {idx + 1} (score={f.score:.2f})\n"
                f"  USER INPUT: {inp}\n"
                f"  AGENT REPLY: {out}"
            )

        examples_block = "\n\n".join(_fmt(i, f) for i, f in enumerate(sample))
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    f"PROMPT NAME: {prompt_name}\n"
                    f"CURRENT VERSION: {current_version}\n\n"
                    f"CURRENT PROMPT:\n<<<\n{current_content}\n>>>\n\n"
                    f"RECENT FAILURES ({len(failures)} in cluster, showing "
                    f"{len(sample)}):\n{examples_block}"
                ),
            },
        ]

        try:
            raw = self._chat(messages).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LlmCandidateGenerator: chat call failed for %s/%s: %s",
                prompt_name,
                current_version,
                exc,
            )
            return None

        if "UNCHANGED" in raw and "<<<" not in raw:
            return None

        rationale, revised = _parse_response(raw)
        if revised is None or revised.strip() == current_content.strip():
            return None

        return CandidateProposal(
            content=revised,
            rationale=rationale,
            based_on_version=current_version,
            based_on_cluster_size=len(failures),
        )


def _parse_response(raw: str) -> tuple[str, str | None]:
    """Pull the rationale + fenced revised prompt out of the LLM reply."""
    lines = raw.splitlines()
    rationale = ""
    body: list[str] = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped == "<<<":
            in_fence = True
            continue
        if stripped == ">>>":
            in_fence = False
            continue
        if in_fence:
            body.append(line)
        elif not rationale and stripped:
            rationale = stripped
    if not body:
        return rationale, None
    return rationale, "\n".join(body).strip()


# ---------------------------------------------------------------- static


class StaticAppendGenerator:
    """Deterministic fallback: append a fixed instruction to the prompt.

    Doesn't see the failure rows; just appends ``addendum``. Useful
    as a sanity baseline and in tests where calling an LLM would
    couple unrelated state. ``min_cluster_size`` still gates the
    "do we even bother" decision so we don't churn canaries.
    """

    def __init__(
        self,
        addendum: str,
        *,
        min_cluster_size: int = 2,
        rationale: str = "static append-only candidate (baseline)",
    ) -> None:
        self._addendum = addendum
        self._min_cluster_size = min_cluster_size
        self._rationale = rationale

    def propose(
        self,
        *,
        prompt_name: str,
        current_content: str,
        current_version: int,
        failures: list[FailureSample],
    ) -> CandidateProposal | None:
        if len(failures) < self._min_cluster_size:
            return None
        revised = f"{current_content.rstrip()}\n\n{self._addendum.strip()}"
        if revised == current_content:
            return None
        return CandidateProposal(
            content=revised,
            rationale=self._rationale,
            based_on_version=current_version,
            based_on_cluster_size=len(failures),
        )
