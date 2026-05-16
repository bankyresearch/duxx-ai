"""Remote-DuxxDB backend — talks to a ``duxx-server`` daemon over the
RESP protocol via the ``redis`` Python client.

Install with::

    pip install duxx-ai[duxxdb-server]

Use this backend for multi-worker fleets that share state, for
production deployments that want TLS + auth + Prometheus + graceful
shutdown out of the box, or for any case where the storage lifetime
needs to outlive the Python process.

See https://github.com/bankyresearch/duxxdb for the underlying engine.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from duxx_ai.memory.manager import MemoryEntry


_DEFAULT_KEY = "default"


class DuxxServerBackend:
    """Backend implementing :class:`~duxx_ai.memory.storage.MemoryBackend`
    against a remote ``duxx-server`` daemon.

    Parameters
    ----------
    url:
        ``redis://[:token@]host:port`` for plain RESP or
        ``rediss://...`` for TLS. The ``token`` segment becomes
        the ``AUTH`` payload that DuxxDB's Phase 6.1 auth gate
        validates with a constant-time compare.
    dim:
        Embedding dimension. The server doesn't enforce this on the
        wire, but the local embedder must agree with the server's
        startup dim or recall will refuse the request.
    agent_id:
        Default key used for ``REMEMBER`` and ``RECALL`` when an entry
        carries no explicit ``agent_id``. Override per-call.
    embedder:
        Optional callable ``str -> list[float]``. The server only sees
        text; embeddings are computed client-side and passed alongside.
        If ``None``, the caller must supply embeddings in every
        :meth:`store` and :meth:`recall`.
    timeout:
        Per-call socket timeout in seconds.

    Notes
    -----
    The server's standard ``REMEMBER key text`` / ``RECALL key query k``
    commands do not currently accept caller-side embeddings on the
    wire; the server embeds with whichever provider was configured at
    boot time. The ``embedder`` kwarg here is reserved for a future
    ``REMEMBER.WITH_EMBEDDING`` extension and is currently informational.
    """

    def __init__(
        self,
        url: str,
        *,
        dim: int = 1536,
        agent_id: str = _DEFAULT_KEY,
        embedder: Callable[[str], list[float]] | None = None,
        timeout: float = 5.0,
    ) -> None:
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only when extra missing
            raise ImportError(
                "DuxxServerBackend requires the 'redis' package. "
                "Install with: pip install duxx-ai[duxxdb-server]"
            ) from exc

        self._redis = redis
        self._client: Any = redis.from_url(
            url, decode_responses=True, socket_timeout=timeout
        )
        # Best-effort handshake. If a token is required and missing the
        # server will return NOAUTH on the first non-PING command, and
        # the user gets a clear error then.
        try:
            self._client.execute_command("PING")
        except Exception as exc:  # pragma: no cover - probe is best-effort
            raise ConnectionError(
                f"Could not reach duxx-server at {url}: {exc}"
            ) from exc

        self._dim = dim
        self._default_agent_id = agent_id
        self._embedder = embedder
        # Mirror of metadata that the wire doesn't carry. Keyed by
        # string-form id returned by REMEMBER.
        self._meta: dict[str, MemoryEntry] = {}
        self._stats: dict[str, int] = {"count": 0, "recalls": 0}

    # ------------------------------------------------------------------
    # MemoryBackend protocol surface
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> str:
        key = entry.agent_id or self._default_agent_id
        new_id = self._client.execute_command("REMEMBER", key, entry.content)
        sid = str(new_id)
        entry.id = sid
        self._meta[sid] = entry
        self._stats["count"] = len(self._meta)
        return sid

    def get(self, id: str) -> MemoryEntry | None:
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
        query_embedding: list[float] | None = None,  # noqa: ARG002 (server embeds)
    ) -> list[MemoryEntry]:
        self._stats["recalls"] = self._stats.get("recalls", 0) + 1
        key = agent_id or self._default_agent_id

        # Over-fetch when we will post-filter by memory_type.
        fetch_k = k * 5 if memory_type is not None else k
        rows = self._client.execute_command("RECALL", key, query, fetch_k) or []
        out: list[MemoryEntry] = []
        for row in rows:
            # RESP shape: [int_id, "score_string", "text"]
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            row_id, _score, text = row[0], row[1], row[2]
            sid = str(row_id)
            entry = self._meta.get(sid)
            if entry is None:
                entry = self._reconstruct(sid, text, key)
                self._meta[sid] = entry
            if memory_type is not None and entry.memory_type != memory_type:
                continue
            entry.access_count += 1
            entry.last_accessed = time.time()
            out.append(entry)
            if len(out) >= k:
                break
        self._stats["count"] = len(self._meta)
        return out

    def delete(self, id: str) -> bool:
        # The current RESP surface does not expose a row-level delete
        # for the memory store. We drop metadata locally; the row will
        # be evicted by the server-side ``--max-memories`` policy.
        if id in self._meta:
            del self._meta[id]
            self._stats["count"] = len(self._meta)
            return True
        return False

    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reconstruct(self, sid: str, text: str, key: str) -> MemoryEntry:
        from duxx_ai.memory.manager import MemoryEntry  # local: avoid cycle

        return MemoryEntry(
            id=sid,
            content=str(text),
            memory_type="semantic",
            agent_id=key,
        )

    # ------------------------------------------------------------------
    # Convenience escape hatch
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        """The underlying ``redis.Redis`` client, for callers that need
        to issue raw RESP commands (e.g. ``PSUBSCRIBE memory.*`` for
        live updates or ``TRACE.RECORD`` for Phase 7.1 observability)."""

        return self._client
