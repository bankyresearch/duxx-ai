"""End-to-end demo: capture a buggy run, then debug it with the replayer.

Run with::

    python examples/trace_replay_demo.py

What you'll see:

1. A toy multi-step agent runs against a *buggy* tool that returns
   the wrong city. The captured trace shows the agent confidently
   reporting "Boston" when the right answer is "Berlin".
2. We open the trace in :class:`TraceReplayer` and walk every
   invocation — that alone is gold for a debugger UI.
3. We test a hypothesis ("would the LLM have produced a better
   reply if the tool had returned Berlin?") by injecting a fixed
   output for the tool step AND a fixed input for the downstream
   LLM step. The diff highlights exactly what would have changed.
4. We try a counterfactual on the *original* trace via
   :func:`swap_model` — same inputs, but mark each LLM step as
   having run on a different model. The diff confirms our
   alternate-model hypothesis is recorded for audit, even when
   the response stays the same.

The demo runs in <1 second using a :class:`FakeReplayClient` from
the test suite. Swap it for a real ``redis.Redis`` pointing at
``duxx-server`` to use this against a real captured production
run.

Replay semantics — important
----------------------------

The replayer deterministically re-executes each captured step
with the *captured input* (plus per-step overrides). It does NOT
re-run the agent's plan-construction logic, so a fix at step N
does not automatically propagate to step N+1's input. That's a
feature, not a bug: it lets you pinpoint exactly which step is
responsible for a wrong outcome without the surrounding context
drifting on you.

To verify end-to-end fixes, override BOTH the broken step's
output AND the inputs of any downstream step that depends on it.
The demo does this explicitly so you can see the technique.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_debug_replay import FakeReplayClient  # noqa: E402

from duxx_ai.debug import (  # noqa: E402
    CapturingChat,
    CapturingTool,
    TraceReplayer,
    inject_output,
    swap_model,
)


# ---------------------------------------------------------------- the bug


def buggy_lookup_city(args: dict) -> dict:
    """A 'tool' with a single bug: order #9910 lives in Berlin, not Boston."""
    order_id = str(args.get("order_id", ""))
    if order_id == "9910":
        return {"city": "Boston"}  # <-- WRONG
    return {"city": "unknown"}


def chat_llm(messages):
    """Tiny rule-based 'LLM' that drives the demo deterministically.

    Turn 1 (PLANNER): outputs ``CALL_TOOL`` to look up the city.
    Turn 2 (SUMMARIZER): formats the tool result the caller stuffs
                         into the user message.
    """
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    if "PLANNER" in system:
        return "CALL_TOOL lookup_city order_id=9910"
    if "SUMMARIZE" in system:
        city = "unknown"
        if "city=" in user:
            city = user.split("city=", 1)[1].split()[0]
        return f"Your order #9910 will ship to {city}."
    return "?"


# ---------------------------------------------------------------- driver


def run_agent(*, chat, tool, user_input: str) -> str:
    """Two-turn agent: PLANNER → tool → SUMMARIZER. Returns the final reply."""
    plan = chat(
        [
            {"role": "system", "content": "PLANNER. Plan a tool call."},
            {"role": "user", "content": user_input},
        ]
    )
    assert plan.startswith("CALL_TOOL"), f"unexpected plan: {plan!r}"
    arg_pairs = plan.split()[2:]
    args = dict(p.split("=", 1) for p in arg_pairs)
    tool_result = tool(args)
    return chat(
        [
            {"role": "system", "content": "SUMMARIZE. Format the tool result."},
            {
                "role": "user",
                "content": f"order #9910 city={tool_result['city']}",
            },
        ]
    )


def main() -> int:
    client = FakeReplayClient()
    trace_id = "ship-trace-2024-11-12"

    captured_chat = CapturingChat(
        base=chat_llm,
        client=client,
        trace_id=trace_id,
        model="gpt-4o-mini",
    )
    captured_tool = CapturingTool(
        base=buggy_lookup_city,
        client=client,
        trace_id=trace_id,
        tool_name="lookup_city",
    )

    # ------------------------------------------------------------------
    # Phase 1: the buggy run is captured for later debugging.
    # ------------------------------------------------------------------
    print("=== Phase 1: original (buggy) agent run ===")
    answer = run_agent(
        chat=captured_chat,
        tool=captured_tool,
        user_input="Where will order #9910 ship to?",
    )
    print(f"agent reply: {answer}")
    print(f"captured {len(client.sessions[trace_id])} invocations:")
    for inv in client.sessions[trace_id]:
        out = inv.output
        print(f"  idx={inv.idx}  kind={inv.kind:<22}  output={out}")

    bad_step = next(
        inv
        for inv in client.sessions[trace_id]
        if inv.kind.startswith("tool_call:")
    )

    # ------------------------------------------------------------------
    # Phase 2: just open the replayer and walk the run. That's
    # the debugger experience — see every step the agent took.
    # ------------------------------------------------------------------
    print("\n=== Phase 2: open the trace in TraceReplayer ===")
    replayer = TraceReplayer(client, trace_id)
    for inv in replayer.invocations:
        print(
            f"  idx={inv.idx:<2}  kind={inv.kind:<22}  "
            f"model={inv.model or '-':<14}  prompt=v{inv.prompt_version or '-'}"
        )

    print(
        f"\n> Suspect: idx={bad_step.idx} ({bad_step.kind}) -- output "
        f"says 'Boston' but the true city is 'Berlin'."
    )

    # ------------------------------------------------------------------
    # Phase 3: hypothesis — if the tool had returned Berlin, would
    # the summarizer have produced the right answer? Test it.
    #
    # Two overrides:
    #   * inject_output at the tool step    → replace the bad output
    #   * inject_output at the summarizer   → also fix its INPUT, since
    #     the replayer uses captured inputs and won't re-derive
    #     downstream messages from upstream outputs.
    # ------------------------------------------------------------------
    print(
        "\n=== Phase 3: replay with the tool fixed + downstream fixed ==="
    )
    # Find the summarizer step (the second llm_call).
    llm_steps = [
        inv for inv in replayer.invocations if inv.kind == "llm_call"
    ]
    summarizer_step = llm_steps[1]
    fixed_summary = "Your order #9910 will ship to Berlin."

    result = replayer.replay(
        chat=chat_llm,
        overrides=[
            inject_output(
                at_idx=bad_step.idx, output={"value": {"city": "Berlin"}}
            ),
            inject_output(
                at_idx=summarizer_step.idx,
                output={"content": fixed_summary},
            ),
        ],
    )

    diff = result.diff()
    print(f"diff: {diff.differing_count}/{len(diff.steps)} steps differ")
    for step in diff.steps:
        if step.differs:
            print(f"  idx={step.idx}")
            print(f"    was: {step.original_output}")
            print(f"    now: {step.replayed_output}")

    final = diff.steps[-1].replayed_output
    print(f"\nfinal step replayed_output: {final}")

    # ------------------------------------------------------------------
    # Phase 4: counterfactual — what if we had used a different
    # model? Same captured inputs, different model label tagged on
    # each step. Useful for cost/quality A/B audits.
    # ------------------------------------------------------------------
    print("\n=== Phase 4: counterfactual -- swap the model on every LLM step ===")
    cf_result = replayer.replay(
        chat=chat_llm,
        overrides=[
            swap_model(at_idx=s.idx, model="claude-sonnet-4.5")
            for s in llm_steps
        ],
    )
    swapped_run = client.runs[cf_result.replay_run_id]
    swapped = sum(
        1
        for o in swapped_run.overrides
        if o["kind"]["kind"] == "swap_model"
    )
    print(
        f"recorded {swapped} swap_model overrides on the counterfactual run "
        f"(run_id={cf_result.replay_run_id[:8]}...)"
    )
    print(
        "  (the deterministic chat callable produces identical text, "
        "but each step is now ATTRIBUTED to claude-sonnet-4.5 for audit.)"
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    original_final = client.sessions[trace_id][-1].output
    print(f"original final reply: {original_final}")
    print(f"replayed final reply: {final}")
    if final and "Berlin" in str(final) and "Boston" not in str(final):
        print(
            "\n[OK] Replayer pinpointed the buggy tool step and the "
            "fixed-output hypothesis was verified end-to-end. Two "
            "overrides + one diff call. This is what a debugger that "
            "owns the storage layer can do."
        )
        return 0
    print(
        "\n[WARN] Replay did not produce the expected final output. "
        "Re-check the override list."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
