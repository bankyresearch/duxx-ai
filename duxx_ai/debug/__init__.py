"""Causal trace replay debugger for duxx-ai.

Two-phase workflow:

1. **Capture** during normal agent runs — wrap your chat/tool
   callables in :class:`CapturingChat` / :class:`CapturingTool`.
   Every invocation flows to DuxxDB under a stable ``trace_id``.
2. **Replay** later — instantiate a :class:`TraceReplayer` against
   that ``trace_id``. Inspect the captured invocations, build
   per-step overrides (swap model, swap prompt, inject output,
   set temperature, skip), and re-execute. Diff the result.

DuxxDB owns the ``REPLAY.*`` wire protocol at the storage layer, so
this is the first agent framework where deterministic re-execution
with per-step overrides is a primitive, not a screenshot.

.. code-block:: python

    import redis
    from duxx_ai.debug import (
        CapturingChat,
        TraceReplayer,
        swap_prompt,
        swap_model,
        inject_output,
    )

    client = redis.Redis(host="localhost", port=6379, decode_responses=True)

    # Phase 1: record a run.
    chat = CapturingChat(
        base=your_real_chat,
        client=client,
        trace_id="trace-2024-01-01-abc",
        model="gpt-4o-mini",
        prompt_name="refund_classifier",
        prompt_version=7,
    )
    chat([{"role": "user", "content": "I want a refund"}])
    # (run continues...)

    # Phase 2: debug it later.
    replayer = TraceReplayer(client, "trace-2024-01-01-abc")
    print(f"captured {len(replayer.invocations)} invocations")

    # Try the same run with a different prompt version.
    result = replayer.replay(
        chat=your_real_chat,
        overrides=[swap_prompt(at_idx=0, prompt_name="refund_classifier",
                               prompt_version=8)],
    )
    diff = result.diff()
    for step in diff.steps:
        if step.differs:
            print(step.idx, step.original_output, "→", step.replayed_output)
"""

from __future__ import annotations

from .capture import CapturingChat, CapturingTool
from .replay import (
    CapturedInvocation,
    CapturedSession,
    DiffStep,
    ReplayDiff,
    ReplayResult,
    TraceReplayer,
    inject_output,
    set_temperature,
    skip,
    swap_model,
    swap_prompt,
)

__all__ = [
    # Capture
    "CapturingChat",
    "CapturingTool",
    # Replay
    "TraceReplayer",
    "ReplayResult",
    "ReplayDiff",
    "DiffStep",
    "CapturedSession",
    "CapturedInvocation",
    # Override builders
    "swap_model",
    "swap_prompt",
    "set_temperature",
    "inject_output",
    "skip",
]
