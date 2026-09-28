"""total-agent-memory (TAM) memory provider for Hermes Agent.

This directory is both the pip package and the Hermes directory plugin (plugin.yaml sits next to
this file); Hermes recognises it by the MemoryProvider / register_memory_provider marker below.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"


def register(ctx: Any) -> None:
    """Hermes plugin entry point (directory plugin and ``hermes_agent.memory_providers`` entry point)."""
    from .provider import TamMemoryProvider

    ctx.register_memory_provider(TamMemoryProvider())


__all__ = ["__version__", "register"]
