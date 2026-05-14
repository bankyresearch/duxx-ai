"""Integration tests for :class:`DuxxBackend` (embedded DuxxDB).

These tests are skipped when the ``duxxdb`` extra is not installed.
To run them locally::

    pip install -e .[duxxdb]
    pytest tests/memory/test_duxx_local_backend.py -v
"""

from __future__ import annotations

import importlib
import tempfile

import pytest

from duxx_ai.memory import MemoryEntry, MemoryManager
from duxx_ai.memory.storage import MemoryBackend


def _has_duxxdb() -> bool:
    try:
        importlib.import_module("duxxdb")
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_duxxdb(),
    reason="install the duxxdb extra to run these tests: pip install -e .[duxxdb]",
)


def test_backend_satisfies_protocol() -> None:
    from duxx_ai.memory.storage.duxx_local import DuxxBackend

    backend = DuxxBackend(dim=4)
    assert isinstance(backend, MemoryBackend)


def test_store_then_recall_in_memory() -> None:
    from duxx_ai.memory.storage.duxx_local import DuxxBackend

    backend = DuxxBackend(dim=4)
    backend.store(
        MemoryEntry(
            content="I lost my wallet at the cafe",
            memory_type="episodic",
            agent_id="alice",
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
    )
    backend.store(
        MemoryEntry(
            content="My favorite color is blue",
            memory_type="episodic",
            agent_id="alice",
            embedding=[0.0, 1.0, 0.0, 0.0],
        )
    )
    hits = backend.recall(
        "wallet",
        agent_id="alice",
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        k=2,
    )
    assert len(hits) >= 1
    assert any("wallet" in h.content for h in hits)


def test_persistent_storage_survives_close_and_reopen() -> None:
    """Persist with ``storage="dir:..."``, drop the backend, then open
    the same dir from a fresh backend instance and confirm the row
    survives.

    Requires ``duxxdb>=0.1.1`` (added the ``open_at`` PyO3 binding).
    Older wheels skip with a clear upgrade hint; the pyproject extra
    pins ``>=0.1.0,<0.2.0`` to keep the install surface forgiving.
    """

    import duxxdb  # type: ignore[import-not-found]

    if not hasattr(duxxdb.MemoryStore, "open_at"):
        pytest.skip(
            "duxxdb < 0.1.1 does not expose MemoryStore.open_at; "
            "upgrade with: pip install -U 'duxxdb>=0.1.1'"
        )

    from duxx_ai.memory.storage.duxx_local import DuxxBackend

    with tempfile.TemporaryDirectory() as tmp:
        b1 = DuxxBackend(dim=4, storage=f"dir:{tmp}")
        b1.store(
            MemoryEntry(
                content="persistent memory",
                memory_type="episodic",
                agent_id="alice",
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
        )
        del b1

        b2 = DuxxBackend(dim=4, storage=f"dir:{tmp}")
        hits = b2.recall(
            "persistent",
            agent_id="alice",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            k=1,
        )
        assert len(hits) >= 1
        assert "persistent" in hits[0].content.lower()


def test_dim_mismatch_raises_value_error() -> None:
    from duxx_ai.memory.storage.duxx_local import DuxxBackend

    backend = DuxxBackend(dim=4)
    with pytest.raises(ValueError, match="dim"):
        backend.store(
            MemoryEntry(
                content="x", memory_type="episodic", embedding=[1.0, 2.0]
            )
        )


def test_memory_manager_routes_through_duxx_backend() -> None:
    from duxx_ai.memory.storage.duxx_local import DuxxBackend

    backend = DuxxBackend(dim=4)
    mgr = MemoryManager(backend=backend, agent_id="alice")
    mgr.remember(
        "test routed through DuxxBackend",
        memory_type="episodic",
        embedding=[1.0, 0.0, 0.0, 0.0],
    )
    hits = mgr.recall("routed", query_embedding=[1.0, 0.0, 0.0, 0.0], k=1)
    assert len(hits) >= 1
    assert "routed" in hits[0].content.lower()


def test_stats_exposes_count() -> None:
    from duxx_ai.memory.storage.duxx_local import DuxxBackend

    backend = DuxxBackend(dim=4)
    backend.store(
        MemoryEntry(content="x", memory_type="episodic", embedding=[1.0, 0.0, 0.0, 0.0])
    )
    backend.store(
        MemoryEntry(content="y", memory_type="episodic", embedding=[0.0, 1.0, 0.0, 0.0])
    )
    s = backend.stats()
    assert s["count"] == 2
