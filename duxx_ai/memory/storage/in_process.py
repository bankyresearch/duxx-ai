"""The default backend — pure-Python dict + optional JSON-file
persistence. Equivalent to duxx-ai's pre-0.31 in-memory behavior.

No external dependencies. Suitable for tests, local dev, single-process
agents. Switch to :class:`~duxx_ai.memory.storage.duxx_local.DuxxBackend`
or :class:`~duxx_ai.memory.storage.duxx_server.DuxxServerBackend` once
you outgrow it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duxx_ai.memory.manager import MemoryEntry


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain Python cosine similarity. Returns 0.0 if either side is empty."""

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))


class InProcessBackend:
    """Backend implementing :class:`~duxx_ai.memory.storage.MemoryBackend`
    on top of a single in-memory dict plus optional append-only JSON
    persistence.

    Parameters
    ----------
    storage_path:
        If given, each :meth:`store` call appends the entry as a JSON
        line to this file. On construction, existing entries in the
        file are loaded.
    max_items:
        Per-``memory_type`` LRU cap. ``None`` disables eviction.
    """

    def __init__(
        self,
        *,
        storage_path: str | Path | None = None,
        max_items: int | None = None,
    ) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._stats: dict[str, int] = {"count": 0, "evictions": 0, "recalls": 0}
        self.storage_path: Path | None = (
            Path(storage_path) if storage_path is not None else None
        )
        self.max_items: int | None = max_items

        if self.storage_path is not None and self.storage_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # MemoryBackend protocol surface
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> str:
        self._entries[entry.id] = entry
        self._stats["count"] = len(self._entries)
        if self.storage_path is not None:
            self._append(entry)
        self._evict_if_needed(entry.memory_type)
        return entry.id

    def get(self, id: str) -> MemoryEntry | None:
        entry = self._entries.get(id)
        if entry is None:
            return None
        if entry.is_expired:
            del self._entries[id]
            self._stats["count"] = len(self._entries)
            return None
        entry.access_count += 1
        entry.last_accessed = time.time()
        return entry

    def recall(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        memory_type: str | None = None,
        k: int = 10,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryEntry]:
        self._stats["recalls"] = self._stats.get("recalls", 0) + 1
        query_lower = query.lower()
        q_words = set(query_lower.split())
        now = time.time()
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in list(self._entries.values()):
            if entry.is_expired:
                del self._entries[entry.id]
                continue
            if memory_type is not None and entry.memory_type != memory_type:
                continue
            if agent_id is not None and entry.agent_id and entry.agent_id != agent_id:
                continue

            if query_embedding is not None and entry.embedding:
                # Pure vector score, normalized to [0, 1].
                score = (_cosine(query_embedding, entry.embedding) + 1.0) / 2.0
            else:
                content_words = set(entry.content.lower().split())
                overlap = len(q_words & content_words) / max(len(q_words), 1)
                recency = 1.0 / (1.0 + (now - entry.timestamp) / 86400)
                score = overlap * 0.5 + recency * 0.3 + entry.importance * 0.2

            scored.append((score, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: list[MemoryEntry] = []
        for _, entry in scored[:k]:
            entry.access_count += 1
            entry.last_accessed = now
            results.append(entry)
        self._stats["count"] = len(self._entries)
        return results

    def delete(self, id: str) -> bool:
        if id in self._entries:
            del self._entries[id]
            self._stats["count"] = len(self._entries)
            return True
        return False

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self, memory_type: str) -> None:
        if self.max_items is None:
            return
        same_tier = [e for e in self._entries.values() if e.memory_type == memory_type]
        while len(same_tier) > self.max_items:
            victim = min(same_tier, key=lambda e: e.last_accessed)
            del self._entries[victim.id]
            self._stats["evictions"] = self._stats.get("evictions", 0) + 1
            same_tier = [
                e for e in self._entries.values() if e.memory_type == memory_type
            ]
        self._stats["count"] = len(self._entries)

    def _append(self, entry: MemoryEntry) -> None:
        assert self.storage_path is not None
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def _load(self) -> None:
        from duxx_ai.memory.manager import MemoryEntry  # local: avoid cycle

        assert self.storage_path is not None
        with open(self.storage_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = MemoryEntry.model_validate_json(line)
                self._entries[entry.id] = entry
        self._stats["count"] = len(self._entries)
