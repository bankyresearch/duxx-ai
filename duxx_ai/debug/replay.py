"""TraceReplayer — scrub any past agent run, edit any call, re-run.

The piece that makes duxx-ai's debugging story unique: deterministic
re-execution of a captured agent run with per-invocation overrides.
LangSmith stores traces but you can't re-execute with edits;
their "replay" is video-playback semantics. DuxxDB owns the
REPLAY.* protocol at the storage layer, and this module exposes
it as a Pythonic debugger.

Workflow
--------

1. **Capture** during a normal agent run via
   :class:`duxx_ai.debug.CapturingChat`. This emits one
   ``REPLAY.CAPTURE`` row per LLM/tool call against a stable
   ``trace_id``.
2. **Inspect** the captured invocations:

   .. code-block:: python

       replayer = TraceReplayer(client=redis_client, trace_id="...")
       for inv in replayer.invocations:
           print(inv.idx, inv.kind, inv.input)

3. **Build overrides** to test alternate decisions:

   .. code-block:: python

       from duxx_ai.debug import swap_model, swap_prompt, inject_output

       overrides = [
           swap_model(at_idx=0, model="claude-sonnet-4.5"),
           swap_prompt(at_idx=2, prompt_name="refund_v2", prompt_version=5),
           inject_output(at_idx=4, output={"role": "assistant", "content": "FORCED"}),
       ]

4. **Replay** — duxx-ai drives the loop, calling your chat callable
   for each non-injected step:

   .. code-block:: python

       result = replayer.replay(overrides=overrides, chat=my_chat)
       print(result.replay_run_id)

5. **Diff** the original vs the replay:

   .. code-block:: python

       diff = result.diff()
       for step in diff.steps:
           if step.differs:
               print(step.idx, "WAS:", step.original_output)
               print(step.idx, "NOW:", step.replayed_output)
       print(f"{diff.differing_count} of {len(diff.steps)} steps differ")

Or step through one invocation at a time for an interactive
debugger (see :meth:`TraceReplayer.walk`).
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- types


@dataclass
class CapturedInvocation:
    """One row from a captured replay session."""

    idx: int
    kind: str
    span_id: str
    model: str | None
    prompt_name: str | None
    prompt_version: int | None
    input: Any
    output: Any
    metadata: Any
    recorded_at_unix_ns: int


@dataclass
class CapturedSession:
    """A whole captured agent run, in invocation order."""

    trace_id: str
    invocations: list[CapturedInvocation]
    fingerprint: str
    captured_at_unix_ns: int


# Override builders — typed helpers so callers don't write JSON by hand.


def swap_model(*, at_idx: int, model: str) -> dict:
    """Tell the replay to use ``model`` at this step instead of the original."""
    return {"at_idx": at_idx, "kind": {"kind": "swap_model", "model": model}}


def swap_prompt(*, at_idx: int, prompt_name: str, prompt_version: int) -> dict:
    """Swap the prompt-registry pointer at this step."""
    return {
        "at_idx": at_idx,
        "kind": {
            "kind": "swap_prompt",
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
        },
    }


def set_temperature(*, at_idx: int, temperature: float) -> dict:
    """Override the LLM ``temperature`` at this step (only if ``input`` is an object)."""
    return {
        "at_idx": at_idx,
        "kind": {"kind": "set_temperature", "temperature": float(temperature)},
    }


def inject_output(*, at_idx: int, output: Any) -> dict:
    """Pretend the LLM at this step returned ``output``. Skips re-execution."""
    return {
        "at_idx": at_idx,
        "kind": {"kind": "inject_output", "output": output},
    }


def skip(*, at_idx: int) -> dict:
    """Skip this step entirely. The diff will show it as omitted."""
    return {"at_idx": at_idx, "kind": {"kind": "skip"}}


# ---------------------------------------------------------------- diff


@dataclass
class DiffStep:
    """One step in a replay diff."""

    idx: int
    original_output: Any
    replayed_output: Any
    differs: bool


@dataclass
class ReplayDiff:
    """Per-step + summary diff between an original capture and one replay."""

    source_trace_id: str
    replay_run_id: str
    steps: list[DiffStep]
    differing_count: int


# ---------------------------------------------------------------- result


@dataclass
class ReplayResult:
    """One completed replay run.

    Returned by :meth:`TraceReplayer.replay`. Holds the new
    ``run_id`` plus a ``diff()`` helper so callers can ask
    "what changed?" in one call.
    """

    replayer: "TraceReplayer"
    replay_run_id: str
    mode: str
    steps_executed: int
    overrides_applied: list[dict] = field(default_factory=list)

    def diff(self) -> ReplayDiff:
        """Compute the per-step diff against the source session."""
        return self.replayer.diff(self.replay_run_id)


# ---------------------------------------------------------------- replayer


class TraceReplayer:
    """Time-travel debugger for a captured agent run.

    Parameters
    ----------
    client:
        Connected ``redis.Redis`` against ``duxx-server``.
    trace_id:
        The ``trace_id`` used at capture time. Must already exist
        on the daemon — call :class:`CapturingChat` upstream first.
    """

    def __init__(self, client: Any, trace_id: str) -> None:
        self._client = client
        self.trace_id = trace_id
        self._session: CapturedSession | None = None
        # One warn per override kind so a noisy session doesn't
        # flood the log when the user's chat callable can't carry
        # the overridden field.
        self._warned_drops: set[str] = set()

    # ------------------------------------------------------------ inspect

    @property
    def session(self) -> CapturedSession:
        """The captured session, fetched lazily on first access."""
        if self._session is None:
            self._session = self._fetch_session()
        return self._session

    @property
    def invocations(self) -> list[CapturedInvocation]:
        return self.session.invocations

    def reload(self) -> CapturedSession:
        """Force a fresh fetch from the daemon (e.g. after more captures)."""
        self._session = self._fetch_session()
        return self._session

    # ------------------------------------------------------------ replay

    def replay(
        self,
        *,
        overrides: list[dict] | None = None,
        chat: Callable[[list[dict]], str] | None = None,
        mode: str = "live",
        metadata: dict | None = None,
    ) -> ReplayResult:
        """Replay the captured session, optionally with overrides.

        Parameters
        ----------
        overrides:
            List of override dicts built via :func:`swap_model`,
            :func:`swap_prompt`, :func:`set_temperature`,
            :func:`inject_output`, :func:`skip`. ``None`` replays
            the original verbatim.
        chat:
            ``(messages) -> str`` callable. Required in ``"live"``
            and ``"stepped"`` modes — that's what re-executes
            each LLM step. Ignored in ``"cached"`` mode (which
            just plays the captured outputs back).
        mode:
            ``"live"`` (default) — execute each step end-to-end.
            ``"stepped"`` — same, but you control the loop via
            :meth:`walk`.
            ``"cached"`` — play captured outputs back. Useful for
            replay-as-fixture in tests.
        metadata:
            Free-form annotation stored on the replay run.
        """
        if mode not in {"live", "stepped", "cached"}:
            raise ValueError(
                f"replay mode must be live | stepped | cached, got {mode!r}"
            )
        # Both ``live`` and ``stepped`` re-execute LLM steps through
        # ``chat`` — interactive manual stepping is handled by
        # :meth:`walk`, not by ``replay(mode='stepped')``. Without a
        # chat callable the loop would just break on the first
        # invocation and finalize an empty run, which is worse than
        # failing loudly.
        if mode in {"live", "stepped"} and chat is None:
            raise ValueError(
                f"{mode} replay requires a `chat` callable. "
                "For interactive single-stepping use replayer.walk() instead."
            )

        run_id = self._start_run(overrides=overrides, mode=mode, metadata=metadata)

        # Build a client-side index of overrides by step. The daemon
        # also has its own copy; ours is used to decide whether to
        # invoke the chat callable for each yielded step (InjectOutput
        # short-circuits, Skip is invisible to us, everything else
        # re-executes through `chat`).
        overrides_by_idx: dict[int, dict] = {
            int(o["at_idx"]): o["kind"] for o in (overrides or [])
        }

        # Introspect the chat callable so we can pass the right
        # extra kwargs (model / prompt_name / prompt_version /
        # temperature) when the daemon hands back an overridden
        # invocation. Without this propagation, swap_model /
        # swap_prompt / set_temperature would never reach the LLM.
        chat_kwargs = _supported_chat_kwargs(chat) if chat is not None else set()

        steps_executed = 0
        if mode == "cached":
            # Daemon pre-populates outputs from the captured session;
            # nothing to do client-side.
            steps_executed = self._cached_step_count(run_id)
        else:
            # Drive REPLAY.STEP + REPLAY.RECORD until the daemon returns None.
            for invocation in self._step_loop(run_id):
                if chat is None:
                    break
                output = self._execute_step(
                    invocation,
                    chat=chat,
                    override=overrides_by_idx.get(invocation.idx),
                    chat_kwargs=chat_kwargs,
                )
                self._record_output(
                    run_id=run_id,
                    idx=invocation.idx,
                    output=output,
                )
                steps_executed += 1

        self._complete_run(run_id)
        return ReplayResult(
            replayer=self,
            replay_run_id=run_id,
            mode=mode,
            steps_executed=steps_executed,
            overrides_applied=list(overrides or []),
        )

    def walk(
        self,
        *,
        overrides: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> Iterator[tuple[str, CapturedInvocation, Callable[[Any], None]]]:
        """Interactive replay: yield one step at a time.

        Yields tuples of ``(run_id, invocation, record_callable)``.
        The caller executes the step however they like (interactive
        REPL, alternate model, etc.) and invokes ``record_callable``
        with the produced output to advance to the next step.

        Generator finishes when the daemon returns ``None`` from
        ``REPLAY.STEP``. The replay run is auto-completed at that
        point.

        .. code-block:: python

            for run_id, inv, record in replayer.walk(overrides=[...]):
                print(inv.idx, inv.kind, inv.input)
                user_choice = prompt_user_for_choice()  # your UI
                record(user_choice)
        """
        run_id = self._start_run(
            overrides=overrides, mode="stepped", metadata=metadata
        )

        def make_recorder(idx: int):
            def _record(output: Any) -> None:
                self._record_output(run_id=run_id, idx=idx, output=output)

            return _record

        try:
            for invocation in self._step_loop(run_id):
                yield run_id, invocation, make_recorder(invocation.idx)
        finally:
            self._complete_run(run_id)

    # ------------------------------------------------------------ diff

    def diff(self, replay_run_id: str) -> ReplayDiff:
        """Compute per-step diff between the captured session and a replay run."""
        raw = self._client.execute_command(
            "REPLAY.DIFF", self.trace_id, replay_run_id
        )
        decoded = _decode(raw)
        steps = []
        for s in decoded.get("per_step", []):
            steps.append(
                DiffStep(
                    idx=int(s.get("idx", 0)),
                    original_output=s.get("original_output"),
                    replayed_output=s.get("replayed_output"),
                    differs=bool(s.get("differs", False)),
                )
            )
        return ReplayDiff(
            source_trace_id=self.trace_id,
            replay_run_id=replay_run_id,
            steps=steps,
            differing_count=int(decoded.get("differing_count", 0)),
        )

    # ------------------------------------------------------------ RESP helpers

    def _fetch_session(self) -> CapturedSession:
        raw = self._client.execute_command(
            "REPLAY.GET_SESSION", self.trace_id
        )
        if raw is None:
            raise LookupError(
                f"no captured session for trace_id={self.trace_id!r}. "
                "Capture the run with CapturingChat first."
            )
        body = _decode(raw)
        invocations = [
            _invocation_from_dict(d) for d in body.get("invocations", [])
        ]
        return CapturedSession(
            trace_id=body["trace_id"],
            invocations=invocations,
            fingerprint=body.get("fingerprint", ""),
            captured_at_unix_ns=int(body.get("captured_at_unix_ns", 0)),
        )

    def _start_run(
        self,
        *,
        overrides: list[dict] | None,
        mode: str,
        metadata: dict | None,
    ) -> str:
        raw = self._client.execute_command(
            "REPLAY.START",
            self.trace_id,
            mode,
            json.dumps(overrides or []),
            json.dumps(metadata or {}),
        )
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        return str(raw)

    def _step_loop(self, run_id: str) -> Iterator[CapturedInvocation]:
        while True:
            raw = self._client.execute_command("REPLAY.STEP", run_id)
            if raw is None:
                return
            decoded = _decode(raw)
            yield _invocation_from_dict(decoded)

    def _execute_step(
        self,
        invocation: CapturedInvocation,
        *,
        chat: Callable[..., str],
        override: dict | None = None,
        chat_kwargs: set[str] | None = None,
    ) -> Any:
        """Call the user's chat callable for one LLM step.

        Re-execution semantics:

        * ``inject_output`` — short-circuit. We use the injected
          payload and never call ``chat``. The daemon already
          recorded the injection itself, so our REPLAY.RECORD is
          an idempotent re-write.
        * everything else — invoke ``chat`` with the (possibly
          override-mutated) messages, plus any overridden
          ``model`` / ``prompt_name`` / ``prompt_version`` /
          ``temperature`` as keyword arguments the chat callable
          declares in its signature. Return value is wrapped as
          ``{"content": reply}``.

        ``chat_kwargs`` is the set of keyword-arg names the chat
        callable accepts (introspected once in :meth:`replay`).
        Callables with ``**kwargs`` get every override; legacy
        ``def chat(messages)`` callables get none and stay
        backward-compatible. When swap_model / swap_prompt /
        set_temperature overrides exist but the chat signature
        wouldn't receive them, we warn so callers know the
        override is metadata-only on this run.

        Note: the captured original output sits on ``invocation.output``
        as historical context. It is NOT a signal to skip
        re-execution — re-execution is the whole point of a live
        replay.
        """
        if override is not None and override.get("kind") == "inject_output":
            return override.get("output", {})

        # Only ``llm_call`` steps go through the chat callable. For
        # every other kind (``tool_call:*``, ``other:*``) we replay
        # the captured output verbatim. This lets common
        # counterfactuals — "swap only the LLM model" — work
        # without forcing the caller to also supply a tool runner.
        # Callers who DO want different tool behavior provide an
        # ``inject_output`` override at the tool step.
        if invocation.kind != "llm_call":
            return invocation.output if invocation.output is not None else {}

        messages = _extract_messages(invocation.input)
        kwargs = self._chat_kwargs_for(
            invocation=invocation,
            override=override,
            accepted=chat_kwargs or set(),
        )
        reply = chat(messages, **kwargs)
        return {"content": reply}

    def _chat_kwargs_for(
        self,
        *,
        invocation: CapturedInvocation,
        override: dict | None,
        accepted: set[str],
    ) -> dict[str, Any]:
        """Compute the extra kwargs to pass to the chat callable.

        Pulls overridden fields off the invocation REPLAY.STEP
        returned (the daemon already applied SwapModel / SwapPrompt
        / SetTemperature when it built that). Only keys the chat
        callable's signature accepts make it into the returned
        dict; the rest are dropped silently for callables that
        weren't built to receive them.

        If the override list asks for a swap but the chat signature
        can't carry it, log a WARNING once so callers know the
        override is metadata-only.
        """
        kwargs: dict[str, Any] = {}
        if invocation.model is not None and "model" in accepted:
            kwargs["model"] = invocation.model
        if invocation.prompt_name is not None and "prompt_name" in accepted:
            kwargs["prompt_name"] = invocation.prompt_name
        if invocation.prompt_version is not None and "prompt_version" in accepted:
            kwargs["prompt_version"] = invocation.prompt_version
        # set_temperature mutates invocation.input["temperature"];
        # surface it as a top-level kwarg for chat callables that
        # support one.
        if "temperature" in accepted and isinstance(invocation.input, dict):
            temp = invocation.input.get("temperature")
            if temp is not None:
                kwargs["temperature"] = temp

        # Warn once per override kind when the user asked for a
        # swap the chat callable won't receive.
        if override is not None:
            kind = override.get("kind", "")
            if kind == "swap_model" and "model" not in accepted:
                self._warn_drop("swap_model", "model")
            elif kind == "swap_prompt" and (
                "prompt_name" not in accepted
                and "prompt_version" not in accepted
            ):
                self._warn_drop("swap_prompt", "prompt_name / prompt_version")
            elif kind == "set_temperature" and "temperature" not in accepted:
                self._warn_drop("set_temperature", "temperature")
        return kwargs

    def _warn_drop(self, override_kind: str, missing_kwarg: str) -> None:
        if override_kind in self._warned_drops:
            return
        self._warned_drops.add(override_kind)
        logger.warning(
            "TraceReplayer: %s override is set but the chat callable's "
            "signature doesn't accept %r — the override will be RECORDED "
            "on the replay run for audit but will NOT change what the "
            "LLM sees. Add the kwarg to your chat function to propagate it.",
            override_kind,
            missing_kwarg,
        )

    def _record_output(self, *, run_id: str, idx: int, output: Any) -> None:
        # output must be JSON-encodable for the daemon
        payload = output if isinstance(output, (dict, list)) else {"value": output}
        self._client.execute_command(
            "REPLAY.RECORD",
            run_id,
            str(idx),
            json.dumps(payload),
        )

    def _complete_run(self, run_id: str) -> None:
        try:
            self._client.execute_command("REPLAY.COMPLETE", run_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TraceReplayer: REPLAY.COMPLETE failed for %s: %s", run_id, exc
            )

    def _cached_step_count(self, run_id: str) -> int:
        """Step count for cached replays — read off the run handle."""
        raw = self._client.execute_command("REPLAY.GET_RUN", run_id)
        decoded = _decode(raw) if raw is not None else {}
        return int(decoded.get("current_idx", 0)) or len(self.invocations)


# ---------------------------------------------------------------- helpers


def _decode(blob) -> dict:
    if blob is None:
        return {}
    if isinstance(blob, (bytes, bytearray)):
        blob = blob.decode("utf-8")
    if isinstance(blob, str):
        return json.loads(blob)
    return blob


def _invocation_from_dict(d: dict) -> CapturedInvocation:
    return CapturedInvocation(
        idx=int(d.get("idx", 0)),
        kind=str(d.get("kind", "")),
        span_id=str(d.get("span_id", "")),
        model=d.get("model"),
        prompt_name=d.get("prompt_name"),
        prompt_version=d.get("prompt_version"),
        input=d.get("input"),
        output=d.get("output"),
        metadata=d.get("metadata"),
        recorded_at_unix_ns=int(d.get("recorded_at_unix_ns", 0)),
    )


def _extract_messages(invocation_input: Any) -> list[dict]:
    """Pull a chat messages list out of whatever shape was captured.

    :class:`CapturingChat` stores ``{"messages": [...]}``, but
    arbitrary callers may have captured a different shape. We
    accept ``{"messages": [...]}``, raw lists, or anything with a
    ``messages`` key.
    """
    if isinstance(invocation_input, dict) and "messages" in invocation_input:
        return list(invocation_input["messages"])
    if isinstance(invocation_input, list):
        return list(invocation_input)
    raise ValueError(
        f"can't extract chat messages from captured input: {type(invocation_input).__name__}"
    )


_REPLAYABLE_CHAT_KWARGS = (
    "model",
    "prompt_name",
    "prompt_version",
    "temperature",
)


def _supported_chat_kwargs(chat: Callable[..., Any] | None) -> set[str]:
    """Which override-related kwargs does this chat callable accept?

    Introspects ``chat``'s signature once per ``replay()`` call.
    Returns the subset of ``("model", "prompt_name", "prompt_version",
    "temperature")`` we'll be able to pass through. Callables with
    a ``**kwargs`` parameter get all four; legacy ``def chat(messages)``
    callables get an empty set and stay backward-compatible.

    Returns the empty set for builtins / C functions we can't
    introspect — same conservative posture as legacy callables.
    """
    if chat is None:
        return set()
    try:
        sig = inspect.signature(chat)
    except (ValueError, TypeError):
        return set()
    params = sig.parameters
    if any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ):
        return set(_REPLAYABLE_CHAT_KWARGS)
    return {name for name in _REPLAYABLE_CHAT_KWARGS if name in params}
