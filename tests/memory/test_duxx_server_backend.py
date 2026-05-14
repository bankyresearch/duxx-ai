"""Integration tests for :class:`DuxxServerBackend` (remote duxx-server).

Skipped unless ``DUXX_SERVER_URL`` is set in the environment AND the
``redis`` client is importable. Example invocation::

    docker run --rm -d -p 6379:6379 ghcr.io/bankyresearch/duxxdb:latest
    DUXX_SERVER_URL=redis://localhost:6379 pytest \\
        tests/memory/test_duxx_server_backend.py -v
"""

from __future__ import annotations

import importlib
import os

import pytest

from duxx_ai.memory import MemoryEntry


def _has_redis() -> bool:
    try:
        importlib.import_module("redis")
        return True
    except ImportError:
        return False


def _server_url() -> str | None:
    return os.environ.get("DUXX_SERVER_URL")


pytestmark = pytest.mark.skipif(
    not (_has_redis() and _server_url()),
    reason=(
        "install the duxxdb-server extra AND set DUXX_SERVER_URL to a "
        "running duxx-server to run these tests."
    ),
)


def test_remember_recall_roundtrip() -> None:
    from duxx_ai.memory.storage.duxx_server import DuxxServerBackend

    backend = DuxxServerBackend(_server_url(), dim=32, agent_id="pytest-user")
    backend.store(
        MemoryEntry(
            content="The wallet was lost at the cafe",
            memory_type="episodic",
            agent_id="pytest-user",
        )
    )
    hits = backend.recall("wallet", agent_id="pytest-user", k=3)
    assert len(hits) >= 1
    assert any("wallet" in h.content.lower() for h in hits)


def test_stats_exposes_counters() -> None:
    from duxx_ai.memory.storage.duxx_server import DuxxServerBackend

    backend = DuxxServerBackend(_server_url(), dim=32, agent_id="pytest-stats")
    backend.store(
        MemoryEntry(content="x", memory_type="episodic", agent_id="pytest-stats")
    )
    s = backend.stats()
    assert s["count"] >= 1


def test_underlying_client_is_exposed() -> None:
    from duxx_ai.memory.storage.duxx_server import DuxxServerBackend

    backend = DuxxServerBackend(_server_url(), dim=32, agent_id="pytest-raw")
    assert backend.client is not None
    pong = backend.client.execute_command("PING")
    assert pong in ("PONG", b"PONG")
