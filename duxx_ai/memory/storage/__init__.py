"""Pluggable memory backends for :class:`duxx_ai.memory.MemoryManager`.

Three backends ship out of the box:

* :class:`InProcessBackend` — pure-Python dict + JSON-file persistence.
  Default. Zero extra dependencies. Same behavior as duxx-ai <= 0.30.
* :class:`DuxxBackend` — embedded DuxxDB via the ``duxxdb`` pip
  package. Sub-ms hybrid recall, persistent if ``storage="dir:..."``
  is supplied (requires a duxxdb wheel that exposes ``open_at``).
* :class:`DuxxServerBackend` — talks to a remote ``duxx-server``
  daemon over the RESP protocol via the ``redis`` Python client.

See ``docs/DUXX_STACK_INTEGRATION.md`` in the DuxxDB repo for the
design rationale.
"""

from __future__ import annotations

from duxx_ai.memory.storage.backend import MemoryBackend
from duxx_ai.memory.storage.in_process import InProcessBackend

__all__ = [
    "MemoryBackend",
    "InProcessBackend",
    "DuxxBackend",
    "DuxxServerBackend",
]


def __getattr__(name: str) -> object:
    """Lazy import so the optional ``duxxdb`` / ``redis`` packages are
    only imported when their backend is actually referenced."""

    if name == "DuxxBackend":
        from duxx_ai.memory.storage.duxx_local import DuxxBackend

        return DuxxBackend
    if name == "DuxxServerBackend":
        from duxx_ai.memory.storage.duxx_server import DuxxServerBackend

        return DuxxServerBackend
    raise AttributeError(f"module 'duxx_ai.memory.storage' has no attribute {name!r}")
