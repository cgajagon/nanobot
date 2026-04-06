"""Entity linking — alias resolution and name normalisation.

Responsible for mapping shorthand names, abbreviations, and spelling
variants to their canonical form *before* the classifier runs.

Separated from the classifier so that alias tables can grow
independently (e.g. populated from profile data or external sources)
without touching scoring logic.
"""

from __future__ import annotations

from nanobot.memory._text import normalize_entity_name

# ---------------------------------------------------------------------------
# Alias map — shorthand → canonical name
# ---------------------------------------------------------------------------

ALIAS_MAP: dict[str, str] = {
    # Databases
    "pg": "postgresql",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "es": "elasticsearch",
    # Technologies
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "tf": "terraform",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "python3": "python",
    "gh": "github",
    "gh_actions": "github actions",
    # Environments
    "prod": "production",
    "dev": "development",
    "preprod": "pre-production",
    "pre-prod": "pre-production",
}


def resolve_alias(name: str) -> str:
    """Map known shorthand/alias to its canonical entity name.

    Returns the original name (stripped) if no alias is registered.
    """
    key = normalize_entity_name(name)
    return ALIAS_MAP.get(key, name.strip())
