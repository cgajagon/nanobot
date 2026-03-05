"""Memory system for persistent agent memory.

This package decomposes the monolithic memory module into focused sub-modules
while preserving backward-compatible imports::

    from nanobot.agent.memory import MemoryStore      # primary public API
    from nanobot.agent.memory import _Mem0Adapter      # internal, used by tests
"""

from .mem0_adapter import _Mem0Adapter, _Mem0RuntimeInfo
from .reranker import CrossEncoderReranker
from .store import MemoryStore

__all__ = [
    "MemoryStore",
    "CrossEncoderReranker",
    "_Mem0Adapter",
    "_Mem0RuntimeInfo",
]
