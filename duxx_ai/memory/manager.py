"""Multi-tier memory system: working, episodic, semantic, procedural, and shared memory.

As of v0.31, :class:`MemoryManager` also accepts an optional pluggable
``backend`` (see :mod:`duxx_ai.memory.storage`) that routes
:meth:`MemoryManager.remember` and :meth:`MemoryManager.recall` through
a real storage engine such as embedded DuxxDB or a remote
``duxx-server``. The five tier classes below are preserved verbatim
for backward compatibility and are still used when no backend is
configured.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from duxx_ai.memory.storage.backend import MemoryBackend


class MemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    memory_type: str  # working, episodic, semantic, procedural, shared
    agent_id: str = ""
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)
    importance: float = 0.5  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: float = Field(default_factory=time.time)
    ttl: float | None = None  # Time-to-live in seconds (None = permanent)

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


class WorkingMemory:
    """Short-term transient memory for the current task context."""

    def __init__(self, max_items: int = 50) -> None:
        self.items: dict[str, MemoryEntry] = {}
        self.max_items = max_items

    def store(self, key: str, content: str, **metadata: Any) -> MemoryEntry:
        entry = MemoryEntry(
            content=content, memory_type="working", metadata=metadata, ttl=3600
        )
        self.items[key] = entry
        self._evict_if_needed()
        return entry

    def recall(self, key: str) -> str | None:
        entry = self.items.get(key)
        if entry and not entry.is_expired:
            entry.access_count += 1
            entry.last_accessed = time.time()
            return entry.content
        if entry and entry.is_expired:
            del self.items[key]
        return None

    def clear(self) -> None:
        self.items.clear()

    def _evict_if_needed(self) -> None:
        while len(self.items) > self.max_items:
            # Evict least recently accessed
            oldest_key = min(self.items, key=lambda k: self.items[k].last_accessed)
            del self.items[oldest_key]


class EpisodicMemory:
    """Long-term memory of past task executions and interactions."""

    def __init__(self, storage_path: str | None = None) -> None:
        self.episodes: list[MemoryEntry] = []
        self.storage_path = Path(storage_path) if storage_path else None

    def record(self, content: str, agent_id: str = "", importance: float = 0.5, **metadata: Any) -> MemoryEntry:
        entry = MemoryEntry(
            content=content,
            memory_type="episodic",
            agent_id=agent_id,
            importance=importance,
            metadata=metadata,
        )
        self.episodes.append(entry)
        if self.storage_path:
            self._persist(entry)
        return entry

    def recall(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Simple keyword-based recall. For production, use embedding similarity."""
        query_lower = query.lower()
        scored = []
        for ep in self.episodes:
            if ep.is_expired:
                continue
            # Simple relevance: keyword overlap + recency + importance
            words = set(query_lower.split())
            content_words = set(ep.content.lower().split())
            overlap = len(words & content_words) / max(len(words), 1)
            recency = 1.0 / (1.0 + (time.time() - ep.timestamp) / 86400)
            score = overlap * 0.5 + recency * 0.3 + ep.importance * 0.2
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [ep for _, ep in scored[:top_k]]
        for ep in results:
            ep.access_count += 1
            ep.last_accessed = time.time()
        return results

    def _persist(self, entry: MemoryEntry) -> None:
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "a") as f:
                f.write(entry.model_dump_json() + "\n")


class SemanticMemory:
    """Persistent knowledge store — facts, domain knowledge, learned information."""

    def __init__(self) -> None:
        self.facts: dict[str, MemoryEntry] = {}

    def store(self, key: str, content: str, importance: float = 0.7, **metadata: Any) -> MemoryEntry:
        entry = MemoryEntry(
            content=content, memory_type="semantic", importance=importance, metadata=metadata
        )
        self.facts[key] = entry
        return entry

    def recall(self, key: str) -> str | None:
        entry = self.facts.get(key)
        if entry:
            entry.access_count += 1
            entry.last_accessed = time.time()
            return entry.content
        return None

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        query_lower = query.lower()
        results = []
        for entry in self.facts.values():
            if query_lower in entry.content.lower() or any(
                query_lower in str(v).lower() for v in entry.metadata.values()
            ):
                results.append(entry)
        return results[:top_k]


class ProceduralMemory:
    """Stores learned workflows and procedures that agents can replay."""

    def __init__(self) -> None:
        self.procedures: dict[str, list[dict[str, Any]]] = {}

    def record_procedure(self, name: str, steps: list[dict[str, Any]]) -> None:
        self.procedures[name] = steps

    def get_procedure(self, name: str) -> list[dict[str, Any]] | None:
        return self.procedures.get(name)

    def list_procedures(self) -> list[str]:
        return list(self.procedures.keys())


class SharedMemory:
    """Cross-agent shared memory for multi-agent coordination."""

    def __init__(self) -> None:
        self.store: dict[str, MemoryEntry] = {}
        self._locks: dict[str, str] = {}  # key -> agent_id holding lock

    def write(self, key: str, content: str, agent_id: str = "", **metadata: Any) -> MemoryEntry:
        if key in self._locks and self._locks[key] != agent_id:
            raise ValueError(f"Key '{key}' is locked by agent '{self._locks[key]}'")
        entry = MemoryEntry(
            content=content, memory_type="shared", agent_id=agent_id, metadata=metadata
        )
        self.store[key] = entry
        return entry

    def read(self, key: str) -> str | None:
        entry = self.store.get(key)
        if entry:
            entry.access_count += 1
            return entry.content
        return None

    def lock(self, key: str, agent_id: str) -> bool:
        if key in self._locks and self._locks[key] != agent_id:
            return False
        self._locks[key] = agent_id
        return True

    def unlock(self, key: str, agent_id: str) -> bool:
        if self._locks.get(key) == agent_id:
            del self._locks[key]
            return True
        return False


class MemoryManager:
    """Unified memory manager that coordinates all memory tiers.

    Parameters
    ----------
    storage_dir:
        Optional directory for episodic-tier JSONL persistence
        (unchanged from v0.30).
    backend:
        Optional pluggable backend (v0.31+). When provided,
        :meth:`remember` and :meth:`recall` route through the backend
        instead of the legacy tier classes. The five tier instances
        (``working``, ``episodic``, ``semantic``, ``procedural``,
        ``shared``) remain on the manager for backward compatibility
        and continue to work in their original mode.

        See :mod:`duxx_ai.memory.storage` for the three built-in
        backends: :class:`InProcessBackend`, :class:`DuxxBackend`,
        :class:`DuxxServerBackend`.
    agent_id:
        Default ``agent_id`` attached to every entry stored through
        :meth:`remember`. Override per-call.
    """

    def __init__(
        self,
        storage_dir: str | None = None,
        *,
        backend: MemoryBackend | None = None,
        agent_id: str = "",
    ) -> None:
        ep_path = f"{storage_dir}/episodic.jsonl" if storage_dir else None
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(storage_path=ep_path)
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.shared = SharedMemory()
        self.backend: MemoryBackend | None = backend
        self.agent_id: str = agent_id

    # ------------------------------------------------------------------
    # v0.31: backend-routed unified API
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        memory_type: str = "episodic",
        importance: float = 0.5,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        ttl: float | None = None,
        agent_id: str | None = None,
    ) -> MemoryEntry:
        """Store one entry, routing through the configured backend.

        Falls back to the matching legacy tier when no backend is set,
        so v0.30 callers see no behavior change.
        """

        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            agent_id=agent_id if agent_id is not None else self.agent_id,
            importance=importance,
            embedding=embedding or [],
            metadata=metadata or {},
            ttl=ttl,
        )

        if self.backend is not None:
            self.backend.store(entry)
            return entry

        # Legacy fallback — pick the tier by memory_type.
        if memory_type == "episodic":
            return self.episodic.record(
                content,
                agent_id=entry.agent_id,
                importance=importance,
                **(metadata or {}),
            )
        if memory_type == "semantic":
            key = (metadata or {}).get(
                "key",
                hashlib.sha256(content.encode()).hexdigest()[:12],
            )
            return self.semantic.store(key, content, importance=importance)
        if memory_type == "working":
            key = (metadata or {}).get(
                "key", f"task_{entry.agent_id}_{time.time()}"
            )
            return self.working.store(key, content, **(metadata or {}))
        if memory_type == "shared":
            key = (metadata or {}).get("key", f"shared_{time.time()}")
            return self.shared.write(
                key, content, agent_id=entry.agent_id, **(metadata or {})
            )
        # Unknown tier: store it raw in semantic by content hash.
        key = hashlib.sha256(content.encode()).hexdigest()[:12]
        return self.semantic.store(key, content, importance=importance)

    def recall(
        self,
        query: str,
        *,
        k: int = 10,
        memory_type: str | None = None,
        query_embedding: list[float] | None = None,
        agent_id: str | None = None,
    ) -> list[MemoryEntry]:
        """Cross-tier hybrid recall, routed through the configured backend.

        Falls back to legacy :meth:`recall_all` when no backend is set.
        """

        if self.backend is not None:
            return self.backend.recall(
                query,
                agent_id=agent_id if agent_id is not None else (self.agent_id or None),
                memory_type=memory_type,
                k=k,
                query_embedding=query_embedding,
            )
        return self.recall_all(query, top_k=k)

    # ------------------------------------------------------------------
    # v0.30 API — preserved verbatim
    # ------------------------------------------------------------------

    def auto_store(self, content: str, agent_id: str = "", context: str = "task") -> None:
        """Automatically decide which memory tier to use."""
        if context == "task":
            self.working.store(f"task_{agent_id}_{time.time()}", content)
        elif context == "result":
            self.episodic.record(content, agent_id=agent_id, importance=0.6)
        elif context == "fact":
            key = hashlib.sha256(content.encode()).hexdigest()[:12]
            self.semantic.store(key, content)
        elif context == "procedure":
            self.procedural.record_procedure(f"proc_{time.time()}", [{"action": content}])

    def recall_all(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search across all memory tiers (legacy v0.30 path)."""
        results: list[MemoryEntry] = []
        results.extend(self.episodic.recall(query, top_k=top_k))
        results.extend(self.semantic.search(query, top_k=top_k))
        # Sort by relevance (importance * recency)
        results.sort(key=lambda e: e.importance * (1.0 / (1.0 + (time.time() - e.timestamp) / 86400)), reverse=True)
        return results[:top_k]
