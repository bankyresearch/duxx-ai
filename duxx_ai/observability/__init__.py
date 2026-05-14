"""Observability: OpenTelemetry-based tracing, evaluation, metrics, and cost tracking.

As of v0.31, a :class:`DuxxExporter` ships in this package — point it
at a running ``duxx-server`` and every finished trace lands in
DuxxDB's Phase 7.1 ``TRACE.*`` surface, where it can be queried by
``TRACE.GET`` / ``TRACE.SUBTREE`` / ``TRACE.THREAD`` / ``TRACE.SEARCH``
and streamed live via ``PSUBSCRIBE trace.*``."""

from __future__ import annotations


def __getattr__(name: str) -> object:
    """Lazy import so the optional ``redis`` package is only imported
    when :class:`DuxxExporter` is actually referenced."""

    if name == "DuxxExporter":
        from duxx_ai.observability.duxx_exporter import DuxxExporter

        return DuxxExporter
    raise AttributeError(f"module 'duxx_ai.observability' has no attribute {name!r}")


__all__ = ["DuxxExporter"]
