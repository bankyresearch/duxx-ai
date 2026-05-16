"""DuxxDB-backed tracer exporter.

Flushes every :class:`duxx_ai.observability.tracer.Span` to a running
``duxx-server`` daemon via its Phase 7.1 ``TRACE.*`` RESP commands.
Once flushed, spans become queryable from any RESP client (or from
DuxxDB's gRPC / MCP surfaces) and live for as long as the daemon
keeps them:

* ``TRACE.GET trace_id``        — the full tree for a trace
* ``TRACE.SUBTREE span_id``     — descendants of a node
* ``TRACE.THREAD thread_id``    — every span across every trace in
                                   a multi-turn conversation
* ``TRACE.SEARCH filter_json``  — name / time / status / kind filters
* ``PSUBSCRIBE trace.*``        — live tail of every recorded span

Install the optional dependency once::

    pip install "redis>=5"            # or pip install duxx-ai[duxxdb-server]

Then plug the exporter into the existing tracer wiring::

    from duxx_ai.observability import Tracer
    from duxx_ai.observability.duxx_exporter import DuxxExporter

    exporter = DuxxExporter(url="redis://:$TOKEN@localhost:6379")
    tracer = Tracer(exporters=[exporter])

The exporter is a strict superset of :class:`OTelExporter` for
agent-observability workloads — same span shape, same OTLP-style
attributes JSON, plus thread reconstruction and reactive live tail.
See https://github.com/bankyresearch/duxxdb for the engine.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from duxx_ai.observability.tracer import Span as DuxxAISpan
    from duxx_ai.observability.tracer import Trace as DuxxAITrace

from duxx_ai.observability.tracer import TracerExporter

logger = logging.getLogger(__name__)


class DuxxExporter(TracerExporter):
    """Sends every span in a finished trace to a remote ``duxx-server``.

    Parameters
    ----------
    url:
        ``redis://[:token@]host:port`` for plain RESP or ``rediss://``
        for TLS. The token segment becomes the ``AUTH`` payload that
        DuxxDB's Phase 6.1 auth gate validates with a constant-time
        compare.
    thread_id:
        Optional thread identifier attached to every exported span so
        ``TRACE.THREAD`` can pull a whole multi-turn conversation back
        in one call. If ``None``, each trace stands alone.
    timeout:
        Per-call socket timeout in seconds. Default 5s.
    drop_on_error:
        If ``True`` (default), wire-level errors are logged at WARNING
        and the export is dropped — the application keeps running.
        If ``False``, errors propagate so callers can fail loud.

    Notes
    -----
    duxx-ai's :class:`Span` is roughly OTel-shaped already
    (``id``, ``parent_id``, ``trace_id``, ``start_time``, ``end_time``,
    ``attributes``, ``status``). This exporter maps that shape to
    DuxxDB's :class:`duxx_trace::Span` field-for-field, converting
    seconds-since-epoch timestamps to nanoseconds (the OTel /
    OpenInference convention).
    """

    def __init__(
        self,
        url: str,
        *,
        thread_id: str | None = None,
        timeout: float = 5.0,
        drop_on_error: bool = True,
    ) -> None:
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only when extra missing
            raise ImportError(
                "DuxxExporter requires the 'redis' package. "
                "Install with: pip install redis>=5  (or pip install duxx-ai[duxxdb-server])"
            ) from exc

        self._redis = redis
        self._client: Any = redis.from_url(
            url, decode_responses=True, socket_timeout=timeout
        )
        self.thread_id = thread_id
        self.drop_on_error = drop_on_error

        # Probe so misconfiguration surfaces at construction, not on
        # the first export. Best-effort; on AUTH-required servers a
        # missing token still bubbles up here clearly.
        try:
            self._client.execute_command("PING")
        except Exception as exc:  # pragma: no cover
            raise ConnectionError(
                f"Could not reach duxx-server at {url}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # TracerExporter protocol
    # ------------------------------------------------------------------

    def export(self, trace: DuxxAITrace) -> None:
        for span in trace.spans:
            try:
                self._record_span(trace.id, span)
            except Exception as exc:
                if self.drop_on_error:
                    logger.warning("DuxxExporter failed to record span %s: %s", span.id, exc)
                    return
                raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_span(self, trace_id: str, span: DuxxAISpan) -> None:
        # duxx_ai stores parent_id as None for roots; the wire wants
        # "-" / "null" / "" to mean the same thing.
        parent = span.parent_id or "-"

        # Stringify attributes. DuxxDB's TRACE.RECORD accepts an
        # OTLP-style JSON blob; "-" / "" mean empty.
        attrs_payload: dict[str, Any] = dict(span.attributes)
        if span.events:
            # OTel keeps events as a separate field; we encode them
            # inline under "events" so search + JSON-path queries see
            # them.
            attrs_payload["events"] = [
                {
                    "name": e.name,
                    "timestamp_ns": int(e.timestamp * 1_000_000_000),
                    "attributes": e.attributes,
                }
                for e in span.events
            ]
        attrs_json = json.dumps(attrs_payload, default=str)

        start_ns = int(span.start_time * 1_000_000_000)
        # end_unix_ns can be omitted for still-open spans by passing
        # "-" / empty. duxx_ai always closes a span via the contextmanager
        # before the trace flushes, so end_time should be set; defensive
        # `or "-"` covers the edge case.
        end_ns_raw = span.end_time
        end_ns = "-" if end_ns_raw is None else str(int(end_ns_raw * 1_000_000_000))

        status = "ok" if span.status == "ok" else "error"
        thread = self.thread_id or "-"

        # All eleven positional args. DuxxDB accepts the trailing ones
        # as optional but pass them all so the wire shape is explicit.
        self._client.execute_command(
            "TRACE.RECORD",
            trace_id,
            span.id,
            parent,
            span.name,
            attrs_json,
            str(start_ns),
            end_ns,
            status,
            "internal",  # kind — duxx_ai doesn't track it today
            thread,
        )

    # ------------------------------------------------------------------
    # Convenience escape hatch
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """The underlying ``redis.Redis`` client. For callers that want
        to issue ``TRACE.GET`` / ``PSUBSCRIBE trace.*`` directly."""

        return self._client


__all__ = ["DuxxExporter"]
