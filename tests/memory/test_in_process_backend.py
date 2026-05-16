"""Tests for the default :class:`InProcessBackend`."""

from __future__ import annotations

import tempfile
from pathlib import Path

from duxx_ai.memory import MemoryEntry, MemoryManager
from duxx_ai.memory.storage import InProcessBackend, MemoryBackend


def test_backend_satisfies_protocol() -> None:
    """InProcessBackend must satisfy MemoryBackend via runtime_checkable."""

    assert isinstance(InProcessBackend(), MemoryBackend)


def test_store_then_get_roundtrip() -> None:
    backend = InProcessBackend()
    entry = MemoryEntry(content="hello world", memory_type="episodic")
    sid = backend.store(entry)
    assert sid == entry.id
    got = backend.get(sid)
    assert got is not None
    assert got.content == "hello world"
    assert got.access_count == 1


def test_keyword_recall_orders_by_overlap() -> None:
    backend = InProcessBackend()
    backend.store(MemoryEntry(content="the cat sat on the mat", memory_type="episodic"))
    backend.store(MemoryEntry(content="the dog ran in the park", memory_type="episodic"))
    backend.store(MemoryEntry(content="completely unrelated content", memory_type="episodic"))

    hits = backend.recall("cat on mat", k=2)
    assert len(hits) == 2
    assert "cat" in hits[0].content.lower()


def test_vector_recall_uses_cosine_when_embedding_provided() -> None:
    backend = InProcessBackend()
    backend.store(
        MemoryEntry(content="a", memory_type="episodic", embedding=[1.0, 0.0, 0.0])
    )
    backend.store(
        MemoryEntry(content="b", memory_type="episodic", embedding=[0.0, 1.0, 0.0])
    )
    backend.store(
        MemoryEntry(content="c", memory_type="episodic", embedding=[0.9, 0.1, 0.0])
    )
    hits = backend.recall(
        "query", query_embedding=[1.0, 0.0, 0.0], k=2
    )
    assert len(hits) == 2
    assert hits[0].content == "a"


def test_memory_type_filter() -> None:
    backend = InProcessBackend()
    backend.store(MemoryEntry(content="ep", memory_type="episodic"))
    backend.store(MemoryEntry(content="sem", memory_type="semantic"))
    hits = backend.recall("ep", memory_type="episodic", k=10)
    assert all(h.memory_type == "episodic" for h in hits)


def test_agent_id_filter() -> None:
    backend = InProcessBackend()
    backend.store(MemoryEntry(content="alice msg", memory_type="episodic", agent_id="alice"))
    backend.store(MemoryEntry(content="bob msg", memory_type="episodic", agent_id="bob"))
    hits = backend.recall("msg", agent_id="alice", k=10)
    contents = [h.content for h in hits]
    assert "alice msg" in contents
    assert "bob msg" not in contents


def test_delete_removes_entry() -> None:
    backend = InProcessBackend()
    sid = backend.store(MemoryEntry(content="x", memory_type="episodic"))
    assert backend.delete(sid) is True
    assert backend.get(sid) is None
    assert backend.delete(sid) is False


def test_stats_tracks_count_and_recalls() -> None:
    backend = InProcessBackend()
    assert backend.stats()["count"] == 0
    backend.store(MemoryEntry(content="x", memory_type="episodic"))
    backend.store(MemoryEntry(content="y", memory_type="episodic"))
    backend.recall("x", k=1)
    s = backend.stats()
    assert s["count"] == 2
    assert s["recalls"] == 1


def test_jsonl_persistence_roundtrips_across_instances() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mem.jsonl"
        b1 = InProcessBackend(storage_path=str(path))
        b1.store(MemoryEntry(content="persistent", memory_type="episodic", agent_id="alice"))
        assert path.exists()
        # New instance loads the same file.
        b2 = InProcessBackend(storage_path=str(path))
        assert b2.stats()["count"] == 1
        hits = b2.recall("persistent", k=1)
        assert len(hits) == 1
        assert hits[0].content == "persistent"


def test_per_tier_eviction_cap() -> None:
    backend = InProcessBackend(max_items=2)
    backend.store(MemoryEntry(content="ep1", memory_type="episodic"))
    backend.store(MemoryEntry(content="ep2", memory_type="episodic"))
    backend.store(MemoryEntry(content="ep3", memory_type="episodic"))
    backend.store(MemoryEntry(content="sem1", memory_type="semantic"))
    s = backend.stats()
    # Episodic capped to 2, semantic untouched.
    assert s["evictions"] >= 1
    assert s["count"] == 3  # 2 episodic + 1 semantic


def test_memory_manager_uses_backend_when_provided() -> None:
    backend = InProcessBackend()
    mgr = MemoryManager(backend=backend, agent_id="alice")
    e = mgr.remember("alice lost her wallet", memory_type="episodic")
    assert e.agent_id == "alice"
    # The backend now owns the entry.
    assert backend.stats()["count"] == 1
    hits = mgr.recall("wallet", k=1)
    assert len(hits) == 1
    assert "wallet" in hits[0].content


def test_memory_manager_legacy_path_still_works_without_backend() -> None:
    mgr = MemoryManager()
    e = mgr.remember("legacy episodic content", memory_type="episodic")
    assert e.memory_type == "episodic"
    # In legacy mode the entry lives on the EpisodicMemory tier.
    assert any(
        ep.content == "legacy episodic content" for ep in mgr.episodic.episodes
    )
