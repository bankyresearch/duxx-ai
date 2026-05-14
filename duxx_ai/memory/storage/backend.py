"""The :class:`MemoryBackend` protocol — one interface, many storage
engines.

Every method is sync; an async variant ships in v0.32. The protocol is
:func:`typing.runtime_checkable`, so ``isinstance(x, MemoryBackend)``
works for duck-typed third-party backends.

See ``docs/DUXX_STACK_INTEGRATION.md`` in the DuxxDB repo for the full
design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid circular import at runtime
    from duxx_ai.memory.manager import MemoryEntry


@runtime_checkable
class MemoryBackend(Protocol):
    """Pluggable backend for :class:`duxx_ai.memory.MemoryManager` and
    the five tier proxies (``working``, ``episodic``, ``semantic``,
    ``procedural``, ``shared``).

    Implementations need only the five methods below. Tier scoping is
    handled in the manager layer by passing ``memory_type`` through.
    """

    def store(self, entry: "MemoryEntry") -> str:
        """Insert an entry. Returns its id (the backend MAY assign a
        new id and overwrite ``entry.id``)."""

    def get(self, id: str) -> "MemoryEntry | None":
        """Point lookup. Returns ``None`` if not found or expired."""

    def recall(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        memory_type: str | None = None,
        k: int = 10,
        query_embedding: list[float] | None = None,
    ) -> list["MemoryEntry"]:
        """Hybrid recall.

        * If ``query_embedding`` is provided, it is used as the vector
          input. Otherwise the backend may embed ``query`` itself (if
          it has a configured embedder) or fall back to keyword scoring.
        * ``agent_id`` and ``memory_type`` are filter scopes — pass
          ``None`` to disable filtering on that axis.
        * Returns at most ``k`` entries, ordered by score descending.
        """

    def delete(self, id: str) -> bool:
        """Remove an entry. Returns ``True`` if it existed."""

    def stats(self) -> dict[str, int]:
        """Operational counters — at minimum ``count``. Backends may
        add ``evictions``, ``bytes_used``, ``recalls``, etc."""
