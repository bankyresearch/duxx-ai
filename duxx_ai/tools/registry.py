"""Central registry for all domain tool libraries."""

from __future__ import annotations

from duxx_ai.core.tool import Tool
from duxx_ai.tools.builtin import BUILTIN_TOOLS


# Global registry: domain_name -> {tool_name: Tool}
DOMAIN_TOOLS: dict[str, dict[str, Tool]] = {}


def register_domain(domain: str, tools: dict[str, Tool]) -> None:
    """Register a collection of tools under a domain name."""
    DOMAIN_TOOLS[domain] = tools


def get_tools(
    names: list[str] | None = None,
    domains: list[str] | None = None,
) -> list[Tool]:
    """Get tools by name and/or domain.

    Args:
        names: If provided, only return tools whose names are in this list.
        domains: If provided, only search within these domains.
                 If None, search all registered domains plus builtins.

    Returns:
        List of matching Tool objects.
    """
    # Collect candidate pools
    pools: dict[str, Tool] = {}

    if domains is None:
        # Include builtins and all domains
        pools.update(BUILTIN_TOOLS)
        for domain_tools in DOMAIN_TOOLS.values():
            pools.update(domain_tools)
    else:
        for domain in domains:
            if domain == "builtin":
                pools.update(BUILTIN_TOOLS)
            elif domain in DOMAIN_TOOLS:
                pools.update(DOMAIN_TOOLS[domain])

    if names is None:
        return list(pools.values())

    return [pools[n] for n in names if n in pools]


def list_domains() -> list[str]:
    """Return all registered domain names."""
    _auto_import_domains()
    return list(DOMAIN_TOOLS.keys())


def _auto_import_domains() -> None:
    """Lazily import all domain tool modules to trigger registration."""
    if DOMAIN_TOOLS:
        return  # Already populated
    domain_modules = [
        "duxx_ai.tools.email",
        "duxx_ai.tools.calendar",
        "duxx_ai.tools.database",
        "duxx_ai.tools.api",
        "duxx_ai.tools.document",
        "duxx_ai.tools.financial",
        "duxx_ai.tools.security",
        "duxx_ai.tools.devops",
        "duxx_ai.tools.analytics",
    ]
    import importlib
    for mod in domain_modules:
        try:
            importlib.import_module(mod)
        except Exception:
            pass
