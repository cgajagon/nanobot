"""Text normalization, timestamp, and coercion helpers for the memory subsystem.

Pure functions shared across memory subpackages.  Renamed from ``helpers.py``
(prohibited filename) per CLAUDE.md conventions.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float) -> float:
    """Coerce *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_str_list(value: Any) -> list[str]:
    """Coerce *value* to a list of non-empty stripped strings."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _to_datetime(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string to *datetime*, returning ``None`` on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------


def _norm_text(value: str) -> str:
    """Lowercase, strip, and collapse whitespace."""
    return re.sub(r"\s+", " ", value.strip().lower())


# ---------------------------------------------------------------------------
# Entity name normalisation
# ---------------------------------------------------------------------------

# Regex for possessive suffixes (straight and smart quotes)
_POSSESSIVE_RE = re.compile(r"['\u2018\u2019]s$", re.IGNORECASE)

# Titles stripped only at start of name
_TITLE_RE = re.compile(r"^(?:dr|mr|mrs|ms|prof)\.?\s+", re.IGNORECASE)

# Punctuation to strip (keep hyphens, underscores, alphanumeric, spaces)
_ENTITY_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name to its canonical form.

    Pipeline: NFKC -> strip -> strip possessives -> strip titles ->
    strip punctuation (preserve hyphens/underscores) -> lowercase ->
    collapse whitespace -> spaces to underscores.
    """
    if not name or not name.strip():
        return ""
    text = unicodedata.normalize("NFKC", name)
    text = text.strip()
    text = _POSSESSIVE_RE.sub("", text)
    text = _TITLE_RE.sub("", text)
    text = _ENTITY_PUNCT_RE.sub("", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ", "_")
    return text


def _tokenize(value: str) -> set[str]:
    """Split *value* into lowercased alphanumeric tokens (len > 1)."""
    return {t for t in re.findall(r"[a-zA-Z0-9_\-]+", value.lower()) if len(t) > 1}


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token count approximation (1 token ~ 4 chars)."""
    value = str(text or "")
    if not value:
        return 0
    return max(1, len(value) // 4)


# ---------------------------------------------------------------------------
# Substring matching
# ---------------------------------------------------------------------------


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    """Return ``True`` if *text* contains any of the *needles* (case-insensitive)."""
    lowered = str(text or "").lower()
    return any(needle in lowered for needle in needles)


# ---------------------------------------------------------------------------
# Prompt injection sanitization
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = re.compile(
    r"(?i)(ignore\s+(?:all\s+)?previous\s+instructions|"
    r"you\s+are\s+now\s+a|"
    r"new\s+system\s+prompt|"
    r"override\s+instructions|"
    r"\[system\]|\[instruction\])",
)


def _sanitize_for_prompt(text: str) -> str:
    """Escape text that will be injected into LLM prompts.

    Strips patterns commonly used in prompt injection attacks:
    system-instruction overrides, role declarations, and markdown
    that could break prompt structure.
    """
    # Remove lines that look like system instruction overrides
    text = _INJECTION_PATTERNS.sub("[filtered]", text)
    # Escape markdown heading markers that could break prompt sections
    text = re.sub(r"^(#{1,4})\s", r"\\\1 ", text, flags=re.MULTILINE)
    # Escape triple-backtick code fences
    text = text.replace("```", "\\`\\`\\`")
    # Escape horizontal rules that could visually separate sections
    text = re.sub(r"^---+\s*$", "\\---", text, flags=re.MULTILINE)
    return text
