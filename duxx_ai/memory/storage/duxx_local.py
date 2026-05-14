"""Embedded DuxxDB backend — uses the ``duxxdb`` pip package directly
in-process. Sub-ms hybrid recall (vector + BM25) and persistence.

Install with::

    pip install duxx-ai[duxxdb]

See https://github.com/bankyresearch/duxxdb for the underlying engine.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from duxx_ai.memory.manager import MemoryEntry


_DEFAULT_KEY = "default"


class DuxxBackend:
    """Backend implementing :class:`~duxx_ai.memory.storage.MemoryBackend`
    on top of an embedded :class:`duxxdb.MemoryStore`.

    Parameters
    ----------
    dim:
        Embedding dimension. Must match every embedding handed to
        :meth:`store` / :meth:`recall`. Default 1536 (OpenAI
        ``text-embedding-3-small``).
    capacity:
        Initial soft capacity. The store grows beyond this on demand
        unless ``max_memories`` is set.
    storage:
        ``None`` for in-memory, ``"dir:./path"`` for persistent. The
        persistent mode requires a ``duxxdb`` wheel that exposes
        ``MemoryStore.open_at``; older wheels raise
        :class:`NotImplementedError`.
    max_memories:
        Hard cap on stored rows; oldest evicted FIFO. Requires a wheel
        with ``MemoryStore.set_max_rows``.
    embedder:
        Optional callable ``str -> list[float]``. If provided,
        :meth:`recall` and :meth:`store` can be called without an
        explicit ``query_embedding`` / ``entry.embedding`` and will
        invoke this function on the fly.
    """

    def __init__(
        self,
        *,
        dim: int = 1536,
        capacity: int = 100_000,
        storage: str | None = None,
        max_memories: int | None = None,
        embedder: Callable[[str], list[float]] | None = None,
    ) -> None:
        try:
            import duxxdb  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only when extra missing
            raise ImportError(
                "DuxxBackend requires the 'duxxdb' package. "
                "Install with: pip install duxx-ai[duxxdb]"
            ) from exc

        self._duxxdb = duxxdb
        self._dim = dim
        self._embedder = embedder
        self._storage = storage
        self._max_memories = max_memories

        store_cls = duxxdb.MemoryStore
        if storage is not None and storage.startswith("dir:"):
            if not hasattr(store_cls, "open_at"):
                raise NotImplementedError(
                    "The installed duxxdb wheel does not expose "
                    "MemoryStore.open_at; persistent storage is only "
                    "available in newer wheels. Upgrade duxxdb or pass "
                    "storage=None for in-memory mode."
                )
            self._store: Any = store_cls.open_at(
                dim=dim, capacity=capacity, dir=storage[len("dir:") :]
            )
        else:
            self._store = store_cls(dim, capacity)

        if max_memories is not None:
            if not hasattr(self._store, "set_max_rows"):
                raise NotImplementedError(
                    "The installed duxxdb wheel does not expose "
                    "MemoryStore.set_max_rows; max_memories is "
                    "unavailable. Upgrade duxxdb or drop the kwarg."
                )
            self._store.set_max_rows(max_memories)

        # Local mirror of metadata that duxxdb doesn't track (timestamps,
        # importance, metadata, etc.). Keyed by string-form id.
        self._meta: dict[str, MemoryEntry] = {}
        self._stats: dict[str, int] = {"count": 0, "recalls": 0}

    # ------------------------------------------------------------------
    # MemoryBackend protocol surface
    # ------------------------------------------------------------------

    def store(self, entry: "MemoryEntry") -> str:
        emb = entry.embedding or self._embed(entry.content)
        if emb is None:
            raise ValueError(
                "embedding not provided on MemoryEntry and no embedder "
                "configured on DuxxBackend"
            )
        if len(emb) != self._dim:
            raise ValueError(
                f"embedding has dim {len(emb)}, store expects {self._dim}"
            )
        key = entry.agent_id or _DEFAULT_KEY
        new_id = self._store.remember(key=key, text=entry.content, embedding=list(emb))
        sid = str(new_id)
        entry.id = sid
        if not entry.embedding:
            entry.embedding = list(emb)
        self._meta[sid] = entry
        self._stats["count"] = len(self._meta)
        return sid

    def get(self, id: str) -> "MemoryEntry | None":
        entry = self._meta.get(id)
        if entry is None:
            return None
        if entry.is_expired:
            del self._meta[id]
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
    ) -> list["MemoryEntry"]:
        self._stats["recalls"] = self._stats.get("recalls", 0) + 1
        emb = query_embedding if query_embedding is not None else self._embed(query)
        if emb is None:
            raise ValueError(
                "query_embedding not provided and no embedder configured "
                "on DuxxBackend"
            )
        if len(emb) != self._dim:
            raise ValueError(
                f"embedding has dim {len(emb)}, store expects {self._dim}"
            )
        key = agent_id or _DEFAULT_KEY
        # duxxdb may return fewer hits if rows are filtered by tier;
        # over-fetch and trim post-filter.
        fetch_k = k * 5 if memory_type is not None else k
        hits = self._store.recall(key=key, query=query, embedding=list(emb), k=fetch_k)

        out: list[MemoryEntry] = []
        for hit in hits:
            sid = str(hit.id)
            entry = self._meta.get(sid)
            if entry is None:
                # Hit refers to a row we don't have local metadata for
                # (e.g., persistent store reopened in a fresh process).
                entry = self._reconstruct(hit)
                self._meta[sid] = entry
            if memory_type is not None and entry.memory_type != memory_type:
                continue
            entry.access_count += 1
            entry.last_accessed = time.time()
            out.append(entry)
            if len(out) >= k:
                break
        return out

    def delete(self, id: str) -> bool:
        # The current duxxdb wheel has no row-level delete on
        # MemoryStore; we simply forget the metadata. The row will be
        # evicted by ``max_memories`` LRU or process exit. When a future
        # wheel adds ``MemoryStore.forget``, wire it up here.
        if id in self._meta:
            del self._meta[id]
            self._stats["count"] = len(self._meta)
            if hasattr(self._store, "forget"):
                try:
                    self._store.forget(int(id))  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - best-effort
                    pass
            return True
        return False

    def stats(self) -> dict[str, int]:
        out = dict(self._stats)
        try:
            out["rows"] = len(self._store)
        except TypeError:
            pass
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float] | None:
        if self._embedder is None:
            return None
        return list(self._embedder(text))

    def _reconstruct(self, hit: Any) -> "MemoryEntry":
        from duxx_ai.memory.manager import MemoryEntry  # local: avoid cycle

        return MemoryEntry(
            id=str(hit.id),
            content=hit.text,
            memory_type="semantic",
            agent_id=getattr(hit, "key", "") or "",
            metadata={"score": float(hit.score)},
        )
