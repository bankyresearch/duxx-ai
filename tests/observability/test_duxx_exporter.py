"""Unit + integration tests for :class:`DuxxExporter`.

The unit tests use a fake redis client that records every command —
they run on every CI pass.

The single integration test is skipped unless ``DUXX_SERVER_URL`` is
set and the ``redis`` extra is installed. Spin one up locally with::

    docker run --rm -d -p 6379:6379 ghcr.io/bankyresearch/duxxdb:v0.1.1
    DUXX_SERVER_URL=redis://localhost:6379 pytest \\
        tests/observability/test_duxx_exporter.py -v
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from types import ModuleType
from typing import Any

import pytest

from duxx_ai.observability.tracer import Span, SpanEvent, Trace, Tracer

# --------------------------------------------------------------------------
# Fake redis client — captures every TRACE.RECORD command so the test
# can inspect the wire shape without a real server.
# --------------------------------------------------------------------------


class _FakeClient:
    def __init__(self) -> None:
        self.commands: list[tuple[Any, ...]] = []

    def execute_command(self, *args: Any) -> Any:
        self.commands.append(args)
        # Mimic PING handshake response.
        if args and str(args[0]).upper() == "PING":
            return "PONG"
        return "OK"


class _FakeRedisModule(ModuleType):
    """Stand-in for the ``redis`` package; provides ``from_url`` only."""

    def __init__(self) -> None:
        super().__init__("redis")
        self.last_client: _FakeClient | None = None

    def from_url(self, url: str, **kwargs: Any) -> _FakeClient:  # noqa: ARG002
        self.last_client = _FakeClient()
        return self.last_client


@pytest.fixture()
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> _FakeRedisModule:
    """Install a fake `redis` module so ``import redis`` resolves to it."""

    mod = _FakeRedisModule()
    monkeypatch.setitem(sys.modules, "redis", mod)
    # Make sure any cached duxx_exporter is reloaded so its import picks
    # up the monkeypatched redis.
    if "duxx_ai.observability.duxx_exporter" in sys.modules:
        del sys.modules["duxx_ai.observability.duxx_exporter"]
    return mod


# --------------------------------------------------------------------------
# Unit tests — work on every CI pass
# --------------------------------------------------------------------------


def test_exporter_pings_on_init(fake_redis: _FakeRedisModule) -> None:
    from duxx_ai.observability.duxx_exporter import DuxxExporter

    DuxxExporter("redis://localhost:6379")
    client = fake_redis.last_client
    assert client is not None
    cmds = [c[0] for c in client.commands]
    assert "PING" in cmds


def test_export_emits_one_trace_record_per_span(
    fake_redis: _FakeRedisModule,
) -> None:
    from duxx_ai.observability.duxx_exporter import DuxxExporter

    exporter = DuxxExporter("redis://localhost:6379")
    fake_redis.last_client.commands.clear()  # type: ignore[union-attr]

    trace = Trace(id="trace-abc")
    root = Span(id="span-1", name="agent.run", trace_id="trace-abc",
                start_time=1.0, end_time=2.0,
                attributes={"model": "gpt-4o"})
    child = Span(id="span-2", name="llm.call", parent_id="span-1",
                 trace_id="trace-abc",
                 start_time=1.1, end_time=1.5,
                 attributes={"tokens": 421})
    trace.spans = [root, child]

    exporter.export(trace)

    cmds = fake_redis.last_client.commands  # type: ignore[union-attr]
    record_cmds = [c for c in cmds if c[0] == "TRACE.RECORD"]
    assert len(record_cmds) == 2


def test_export_wire_shape_matches_trace_record_protocol(
    fake_redis: _FakeRedisModule,
) -> None:
    from duxx_ai.observability.duxx_exporter import DuxxExporter

    exporter = DuxxExporter("redis://localhost:6379", thread_id="session-7")
    fake_redis.last_client.commands.clear()  # type: ignore[union-attr]

    trace = Trace(id="trace-xyz")
    root = Span(
        id="span-1",
        name="agent.run",
        trace_id="trace-xyz",
        start_time=1.0,
        end_time=2.0,
        attributes={"model": "gpt-4o", "tokens_in": 421},
    )
    trace.spans = [root]
    exporter.export(trace)

    cmds = fake_redis.last_client.commands  # type: ignore[union-attr]
    assert cmds[-1][0] == "TRACE.RECORD"
    (
        _,
        trace_id,
        span_id,
        parent,
        name,
        attrs_json,
        start_ns,
        end_ns,
        status,
        kind,
        thread,
    ) = cmds[-1]
    assert trace_id == "trace-xyz"
    assert span_id == "span-1"
    assert parent == "-"  # root span
    assert name == "agent.run"
    attrs = json.loads(attrs_json)
    assert attrs["model"] == "gpt-4o"
    assert attrs["tokens_in"] == 421
    assert int(start_ns) == 1_000_000_000
    assert int(end_ns) == 2_000_000_000
    assert status == "ok"
    assert kind == "internal"
    assert thread == "session-7"


def test_error_status_propagates_to_trace_record(
    fake_redis: _FakeRedisModule,
) -> None:
    from duxx_ai.observability.duxx_exporter import DuxxExporter

    exporter = DuxxExporter("redis://localhost:6379")
    fake_redis.last_client.commands.clear()  # type: ignore[union-attr]

    trace = Trace(id="trace-err")
    s = Span(
        id="span-1",
        name="tool.broken",
        trace_id="trace-err",
        start_time=1.0,
        end_time=2.0,
        status="error",
        attributes={"exception": "boom"},
    )
    trace.spans = [s]
    exporter.export(trace)

    cmds = fake_redis.last_client.commands  # type: ignore[union-attr]
    record = next(c for c in cmds if c[0] == "TRACE.RECORD")
    assert record[8] == "error"  # status field


def test_events_inlined_into_attributes(
    fake_redis: _FakeRedisModule,
) -> None:
    from duxx_ai.observability.duxx_exporter import DuxxExporter

    exporter = DuxxExporter("redis://localhost:6379")
    fake_redis.last_client.commands.clear()  # type: ignore[union-attr]

    trace = Trace(id="trace-events")
    s = Span(
        id="span-1",
        name="agent.run",
        trace_id="trace-events",
        start_time=1.0,
        end_time=2.0,
        events=[SpanEvent(name="received_token", attributes={"token": "hello"})],
    )
    trace.spans = [s]
    exporter.export(trace)

    cmds = fake_redis.last_client.commands  # type: ignore[union-attr]
    record = next(c for c in cmds if c[0] == "TRACE.RECORD")
    attrs = json.loads(record[5])
    assert "events" in attrs
    assert attrs["events"][0]["name"] == "received_token"
    assert attrs["events"][0]["attributes"]["token"] == "hello"


def test_drop_on_error_swallows_wire_failure(
    fake_redis: _FakeRedisModule,
) -> None:
    from duxx_ai.observability.duxx_exporter import DuxxExporter

    exporter = DuxxExporter("redis://localhost:6379", drop_on_error=True)

    # Replace the client with one that raises on any TRACE.RECORD.
    class BrokenClient(_FakeClient):
        def execute_command(self, *args: Any) -> Any:
            if str(args[0]) == "TRACE.RECORD":
                raise ConnectionError("simulated network drop")
            return super().execute_command(*args)

    exporter._client = BrokenClient()
    trace = Trace(id="trace-1")
    trace.spans = [
        Span(id="s1", name="x", trace_id="trace-1", start_time=1.0, end_time=2.0)
    ]
    # Must not raise.
    exporter.export(trace)


def test_drop_on_error_false_raises(
    fake_redis: _FakeRedisModule,
) -> None:
    from duxx_ai.observability.duxx_exporter import DuxxExporter

    exporter = DuxxExporter("redis://localhost:6379", drop_on_error=False)

    class BrokenClient(_FakeClient):
        def execute_command(self, *args: Any) -> Any:
            if str(args[0]) == "TRACE.RECORD":
                raise ConnectionError("simulated network drop")
            return super().execute_command(*args)

    exporter._client = BrokenClient()
    trace = Trace(id="trace-1")
    trace.spans = [
        Span(id="s1", name="x", trace_id="trace-1", start_time=1.0, end_time=2.0)
    ]
    with pytest.raises(ConnectionError):
        exporter.export(trace)


def test_tracer_integration_flushes_on_span_exit(
    fake_redis: _FakeRedisModule,
) -> None:
    """End-to-end: a normal Tracer.span() block must result in
    TRACE.RECORD commands once the outermost span closes."""

    from duxx_ai.observability.duxx_exporter import DuxxExporter

    exporter = DuxxExporter("redis://localhost:6379")
    fake_redis.last_client.commands.clear()  # type: ignore[union-attr]
    tracer = Tracer(exporters=[exporter])

    with tracer.span("agent.run") as s:
        s.set_attribute("model", "gpt-4o")
        with tracer.span("llm.call") as inner:
            inner.set_attribute("tokens", 12)

    cmds = fake_redis.last_client.commands  # type: ignore[union-attr]
    record_cmds = [c for c in cmds if c[0] == "TRACE.RECORD"]
    assert len(record_cmds) == 2
    # The root span has no parent on the wire.
    parents = [c[3] for c in record_cmds]
    assert "-" in parents


# --------------------------------------------------------------------------
# Integration test — needs a running duxx-server
# --------------------------------------------------------------------------


def _has_redis() -> bool:
    try:
        importlib.import_module("redis")
        return True
    except ImportError:
        return False


_INTEGRATION_URL = os.environ.get("DUXX_SERVER_URL")


@pytest.mark.skipif(
    not (_has_redis() and _INTEGRATION_URL),
    reason="set DUXX_SERVER_URL + install redis to run integration test",
)
def test_integration_real_duxx_server_round_trip() -> None:
    """Records two spans against a real duxx-server, then reads them
    back with TRACE.GET to confirm the wire roundtrip."""

    from duxx_ai.observability.duxx_exporter import DuxxExporter

    assert _INTEGRATION_URL is not None
    exporter = DuxxExporter(_INTEGRATION_URL, thread_id="pytest-thread")

    trace_id = f"pytest-trace-{int(time.time() * 1000)}"
    trace = Trace(id=trace_id)
    root = Span(id=f"{trace_id}-root", name="agent.run",
                trace_id=trace_id, start_time=time.time(),
                end_time=time.time() + 0.001)
    trace.spans = [root]
    exporter.export(trace)

    rows = exporter.client.execute_command("TRACE.GET", trace_id) or []
    assert len(rows) >= 1, "duxx-server did not echo any spans back"
    body = json.loads(rows[0])
    assert body["name"] == "agent.run"
