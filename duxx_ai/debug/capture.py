"""Capture LLM/tool invocations into DuxxDB during normal agent runs.

A captured trace is the *prerequisite* for replay. Without it,
:class:`TraceReplayer` has nothing to step through. This module
provides the recording side: drop-in wrappers around your chat /
tool callables that emit ``REPLAY.CAPTURE`` rows as the agent runs.

Usage
-----

.. code-block:: python

    import redis
    from duxx_ai.debug import CapturingChat

    client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    chat = CapturingChat(
        base=your_real_chat,
        client=client,
        trace_id="trace-2024-11-12-abc",
        # optional — links the capture to a PROMPT registry version
        prompt_name="refund_classifier",
        prompt_version=7,
        model="gpt-4o-mini",
    )

    # Use as normal — every call is captured.
    reply = chat([{"role": "user", "content": "I want a refund"}])

Each call writes one ``REPLAY.CAPTURE`` row carrying the input
messages, the model that ran, and (after the call returns) the
output text. The captured rows form the session that
:class:`TraceReplayer` later steps through.

Design notes
------------

* **Provider-agnostic.** The wrapper takes any ``(messages) -> str``
  chat callable. No SDK pin.
* **Errors stay errors.** If the wrapped chat raises, we still emit
  one capture row (with the exception class name) so the failing
  step is replayable. The exception re-raises so the caller can
  decide retry / circuit-break policy.
* **No span correlation required.** The capture is keyed by
  ``trace_id``; if your tracer already produces span ids, pass
  ``span_id_fn`` to link them.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CapturingChat:
    """A chat callable that captures every invocation to DuxxDB.

    Parameters
    ----------
    base:
        The chat callable you'd otherwise call directly:
        ``base(messages: list[dict]) -> str``.
    client:
        Connected ``redis.Redis`` against ``duxx-server``.
    trace_id:
        Stable identifier for the run being captured. Reuse the same
        id across multiple chat calls in one agent turn / session so
        ``REPLAY.GET_SESSION`` returns the whole sequence.
    model:
        Optional label stored on each captured invocation. Used by
        ``SwapModel`` overrides at replay time.
    prompt_name / prompt_version:
        Optional pointers to a PROMPT registry entry. ``SwapPrompt``
        overrides use these at replay time to know what to swap.
    span_id_fn:
        Optional zero-arg callable that returns the active span id
        from your tracer. If supplied, each captured invocation
        carries the span id so cross-store queries can join.
    drop_on_capture_error:
        If True (default), failures to write the capture row are
        logged at WARNING and the chat call returns normally.
        Set False for fail-loud testing.
    """

    def __init__(
        self,
        base: Callable[[list[dict]], str],
        *,
        client: Any,
        trace_id: str,
        model: str | None = None,
        prompt_name: str | None = None,
        prompt_version: int | None = None,
        span_id_fn: Callable[[], str] | None = None,
        drop_on_capture_error: bool = True,
    ) -> None:
        self._base = base
        self._client = client
        self.trace_id = trace_id
        self.model = model
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version
        self._span_id_fn = span_id_fn
        self._drop_on_capture_error = drop_on_capture_error

    def __call__(self, messages: list[dict]) -> str:
        started = time.time_ns()
        span_id = ""
        if self._span_id_fn is not None:
            try:
                span_id = self._span_id_fn() or ""
            except Exception:  # noqa: BLE001
                span_id = ""

        try:
            reply = self._base(messages)
        except Exception as exc:  # noqa: BLE001 — captures across providers
            # Capture the FAILED call so the replay UI can still
            # show the user where it broke. Re-raise after.
            self._emit(
                started=started,
                span_id=span_id,
                kind="llm_call",
                input_payload={"messages": messages},
                output_payload=None,
                metadata={
                    "error": True,
                    "error_class": exc.__class__.__name__,
                    "error_msg": str(exc)[:512],
                },
            )
            raise

        self._emit(
            started=started,
            span_id=span_id,
            kind="llm_call",
            input_payload={"messages": messages},
            output_payload={"content": reply},
            metadata=None,
        )
        return reply

    # ------------------------------------------------------------ internals

    def _emit(
        self,
        *,
        started: int,
        span_id: str,
        kind: str,
        input_payload: dict,
        output_payload: dict | None,
        metadata: dict | None,
    ) -> None:
        invocation: dict[str, Any] = {
            # idx is auto-assigned by REPLAY.CAPTURE based on insertion order
            "idx": 0,
            "span_id": span_id,
            "kind": kind,
            "input": input_payload,
            "recorded_at_unix_ns": started,
        }
        if output_payload is not None:
            invocation["output"] = output_payload
        if self.model is not None:
            invocation["model"] = self.model
        if self.prompt_name is not None:
            invocation["prompt_name"] = self.prompt_name
        if self.prompt_version is not None:
            invocation["prompt_version"] = self.prompt_version
        if metadata:
            invocation["metadata"] = metadata

        try:
            self._client.execute_command(
                "REPLAY.CAPTURE",
                self.trace_id,
                json.dumps(invocation),
            )
        except Exception as exc:  # noqa: BLE001
            if self._drop_on_capture_error:
                logger.warning(
                    "CapturingChat: REPLAY.CAPTURE failed for trace %s: %s",
                    self.trace_id,
                    exc,
                )
            else:
                raise


class CapturingTool:
    """Same shape as :class:`CapturingChat` but for tool callables.

    A tool callable here is anything that takes a dict of arguments
    and returns a JSON-serializable result. Captured under
    ``kind = "tool_call:<name>"`` so ``REPLAY.GET_SESSION`` shows
    LLM calls and tool calls in chronological order.
    """

    def __init__(
        self,
        base: Callable[[dict], Any],
        *,
        client: Any,
        trace_id: str,
        tool_name: str,
        span_id_fn: Callable[[], str] | None = None,
        drop_on_capture_error: bool = True,
    ) -> None:
        self._base = base
        self._client = client
        self.trace_id = trace_id
        self.tool_name = tool_name
        self._span_id_fn = span_id_fn
        self._drop_on_capture_error = drop_on_capture_error

    def __call__(self, args: dict) -> Any:
        started = time.time_ns()
        span_id = ""
        if self._span_id_fn is not None:
            try:
                span_id = self._span_id_fn() or ""
            except Exception:  # noqa: BLE001
                span_id = ""

        try:
            result = self._base(args)
        except Exception as exc:  # noqa: BLE001
            self._emit(
                started=started,
                span_id=span_id,
                input_payload={"args": args},
                output_payload=None,
                metadata={
                    "error": True,
                    "error_class": exc.__class__.__name__,
                    "error_msg": str(exc)[:512],
                },
            )
            raise

        # Wrap non-dict outputs so the captured payload is JSON-shaped.
        if isinstance(result, (dict, list)):
            output_payload = {"value": result}
        else:
            output_payload = {"value": result}

        self._emit(
            started=started,
            span_id=span_id,
            input_payload={"args": args},
            output_payload=output_payload,
            metadata=None,
        )
        return result

    def _emit(
        self,
        *,
        started: int,
        span_id: str,
        input_payload: dict,
        output_payload: dict | None,
        metadata: dict | None,
    ) -> None:
        invocation: dict[str, Any] = {
            "idx": 0,
            "span_id": span_id,
            "kind": f"tool_call:{self.tool_name}",
            "input": input_payload,
            "recorded_at_unix_ns": started,
        }
        if output_payload is not None:
            invocation["output"] = output_payload
        if metadata:
            invocation["metadata"] = metadata

        try:
            self._client.execute_command(
                "REPLAY.CAPTURE",
                self.trace_id,
                json.dumps(invocation),
            )
        except Exception as exc:  # noqa: BLE001
            if self._drop_on_capture_error:
                logger.warning(
                    "CapturingTool: REPLAY.CAPTURE failed for trace %s tool %s: %s",
                    self.trace_id,
                    self.tool_name,
                    exc,
                )
            else:
                raise
