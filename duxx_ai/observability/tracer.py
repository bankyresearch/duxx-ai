"""Observability — OpenTelemetry-compatible tracing with cost tracking and OTel bridge."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Generator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SpanEvent(BaseModel):
    name: str
    timestamp: float = Field(default_factory=time.time)
    attributes: dict[str, Any] = Field(default_factory=dict)


class Span(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    parent_id: str | None = None
    trace_id: str = ""
    start_time: float = Field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)
    status: str = "ok"

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append(SpanEvent(name=name, attributes=attrs))

    def end(self) -> None:
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


class Trace(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    spans: list[Span] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def root_span(self) -> Span | None:
        for s in self.spans:
            if s.parent_id is None:
                return s
        return self.spans[0] if self.spans else None

    @property
    def duration_ms(self) -> float:
        root = self.root_span
        return root.duration_ms if root else 0.0

    def get_token_usage(self) -> dict[str, int]:
        """Aggregate token usage across all spans in this trace."""
        total: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for span in self.spans:
            tokens = span.attributes.get("tokens", {})
            if isinstance(tokens, dict):
                total["input_tokens"] += tokens.get("input_tokens", 0) + tokens.get("prompt_tokens", 0)
                total["output_tokens"] += tokens.get("output_tokens", 0) + tokens.get("completion_tokens", 0)
                total["total_tokens"] += tokens.get("total_tokens", 0)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.id,
            "spans": [s.model_dump() for s in self.spans],
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
            "token_usage": self.get_token_usage(),
        }


class TracerExporter:
    """Base class for exporting traces to external systems."""

    def export(self, trace: Trace) -> None:
        pass


class ConsoleExporter(TracerExporter):
    def export(self, trace: Trace) -> None:
        from rich.console import Console
        from rich.tree import Tree

        console = Console()
        root = trace.root_span
        if root is None:
            return

        tree = Tree(f"[bold]{root.name}[/bold] ({root.duration_ms:.1f}ms)")
        for span in trace.spans:
            if span.id != root.id:
                status_icon = "[green]OK[/green]" if span.status == "ok" else "[red]ERR[/red]"
                tree.add(f"{span.name} ({span.duration_ms:.1f}ms) {status_icon}")

        usage = trace.get_token_usage()
        if usage["total_tokens"] > 0:
            tree.add(f"[dim]Tokens: {usage['total_tokens']} (in={usage['input_tokens']}, out={usage['output_tokens']})[/dim]")

        console.print(tree)


class JSONExporter(TracerExporter):
    def __init__(self, filepath: str = "traces.jsonl") -> None:
        self.filepath = filepath

    def export(self, trace: Trace) -> None:
        with open(self.filepath, "a") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")


class OTelExporter(TracerExporter):
    """Export traces to OpenTelemetry-compatible backends (Jaeger, Zipkin, etc.).

    Requires: pip install opentelemetry-sdk opentelemetry-exporter-otlp
    """

    def __init__(self, service_name: str = "duxx_ai", endpoint: str | None = None) -> None:
        self.service_name = service_name
        self.endpoint = endpoint
        self._otel_tracer: Any = None
        self._init_otel()

    def _init_otel(self) -> None:
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)

            if self.endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                    from opentelemetry.sdk.trace.export import BatchSpanProcessor
                    exporter = OTLPSpanExporter(endpoint=self.endpoint)
                    provider.add_span_processor(BatchSpanProcessor(exporter))
                except ImportError:
                    logger.warning("OTLP exporter not available. Install opentelemetry-exporter-otlp.")

            otel_trace.set_tracer_provider(provider)
            self._otel_tracer = otel_trace.get_tracer("duxx_ai")
        except ImportError:
            logger.warning("OpenTelemetry SDK not available. Install opentelemetry-sdk.")

    def export(self, trace: Trace) -> None:
        if self._otel_tracer is None:
            return
        try:
            from opentelemetry import trace as otel_trace

            root = trace.root_span
            if root is None:
                return

            with self._otel_tracer.start_as_current_span(root.name) as otel_root:
                for key, value in root.attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        otel_root.set_attribute(f"duxx_ai.{key}", value)

                for span in trace.spans:
                    if span.id == root.id:
                        continue
                    with self._otel_tracer.start_as_current_span(span.name) as otel_span:
                        for key, value in span.attributes.items():
                            if isinstance(value, (str, int, float, bool)):
                                otel_span.set_attribute(f"duxx_ai.{key}", value)
                        if span.status == "error":
                            otel_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR))
        except Exception as e:
            logger.warning(f"Failed to export to OTel: {e}")


class Tracer:
    """Manages traces and spans for a session."""

    def __init__(self, exporters: list[TracerExporter] | None = None) -> None:
        self.exporters = exporters or []
        self.traces: list[Trace] = []
        self._current_trace: Trace | None = None
        self._span_stack: list[Span] = []

    @contextmanager
    def span(self, name: str) -> Generator[Span, None, None]:
        if self._current_trace is None:
            self._current_trace = Trace()
            self.traces.append(self._current_trace)

        parent_id = self._span_stack[-1].id if self._span_stack else None
        s = Span(name=name, parent_id=parent_id, trace_id=self._current_trace.id)
        self._current_trace.spans.append(s)
        self._span_stack.append(s)

        try:
            yield s
        except Exception as e:
            s.status = "error"
            s.set_attribute("exception", str(e))
            raise
        finally:
            s.end()
            self._span_stack.pop()
            if not self._span_stack:
                self._flush_trace()

    def _flush_trace(self) -> None:
        if self._current_trace:
            for exporter in self.exporters:
                try:
                    exporter.export(self._current_trace)
                except Exception as e:
                    logger.warning(f"Exporter {type(exporter).__name__} failed: {e}")
            self._current_trace = None

    def get_cost_summary(self) -> dict[str, Any]:
        """Get aggregated metrics across all traces."""
        total_spans = 0
        total_duration = 0.0
        total_tokens: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        for trace in self.traces:
            total_spans += len(trace.spans)
            total_duration += trace.duration_ms
            usage = trace.get_token_usage()
            for key in total_tokens:
                total_tokens[key] += usage.get(key, 0)

        return {
            "total_traces": len(self.traces),
            "total_spans": total_spans,
            "total_duration_ms": total_duration,
            "total_tokens": total_tokens["total_tokens"],
            "input_tokens": total_tokens["input_tokens"],
            "output_tokens": total_tokens["output_tokens"],
        }
