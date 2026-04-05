"""Memory graph subpackage."""

from __future__ import annotations

from nanobot.memory._text import normalize_entity_name

__all__: list[str] = []


def _norm(name: str) -> str:
    """Canonical name normalisation: delegates to normalize_entity_name.

    Package-private helper used by graph.py and graph_traversal.py.
    """
    return normalize_entity_name(name)
