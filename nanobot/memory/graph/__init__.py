"""Memory graph subpackage."""

from __future__ import annotations

__all__: list[str] = []


def _norm(name: str) -> str:
    """Canonical name normalisation: strip, lower, spaces to underscores.

    Package-private helper used by graph.py and graph_traversal.py.
    """
    return name.strip().lower().replace(" ", "_")
