"""Memory system: working, episodic, semantic, procedural, and shared memory.

As of v0.31, ``MemoryManager`` also accepts a pluggable storage
:class:`~duxx_ai.memory.storage.MemoryBackend` that routes
:meth:`MemoryManager.remember` and :meth:`MemoryManager.recall` through
a real storage engine (embedded DuxxDB, remote ``duxx-server``, or the
default in-process dict).
"""

from __future__ import annotations

from duxx_ai.memory.manager import (
    EpisodicMemory,
    MemoryEntry,
    MemoryManager,
    ProceduralMemory,
    SemanticMemory,
    SharedMemory,
    WorkingMemory,
)
from duxx_ai.memory.storage import (
    InProcessBackend,
    MemoryBackend,
)

__all__ = [
    "EpisodicMemory",
    "InProcessBackend",
    "MemoryBackend",
    "MemoryEntry",
    "MemoryManager",
    "ProceduralMemory",
    "SemanticMemory",
    "SharedMemory",
    "WorkingMemory",
]


def __getattr__(name: str) -> object:
    """Lazy-load the optional backends so importing :mod:`duxx_ai.memory`
    never requires the ``duxxdb`` / ``redis`` extras to be installed."""

    if name == "DuxxBackend":
        from duxx_ai.memory.storage import DuxxBackend

        return DuxxBackend
    if name == "DuxxServerBackend":
        from duxx_ai.memory.storage import DuxxServerBackend

        return DuxxServerBackend
    raise AttributeError(f"module 'duxx_ai.memory' has no attribute {name!r}")
