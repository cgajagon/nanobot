"""Agent registry for multi-agent routing.

Maps role names to ``AgentRoleConfig`` instances so the coordinator
can look up which agent should handle a given message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from nanobot.config.schema import AgentRoleConfig


class AgentRegistry:
    """Registry of named agent roles.

    Thin wrapper around a dict that maps role names to their configs.
    The coordinator uses this to resolve classification results to
    concrete agent configurations.
    """

    def __init__(self, default_role: str = "general") -> None:
        self._roles: dict[str, AgentRoleConfig] = {}
        self._default_role = default_role

    def register(self, role: AgentRoleConfig) -> None:
        """Register an agent role config by its name."""
        if role.name in self._roles:
            logger.debug("Overriding existing agent role: {}", role.name)
        self._roles[role.name] = role

    def get(self, name: str) -> AgentRoleConfig | None:
        """Look up a role by name. Returns None if not found."""
        return self._roles.get(name)

    def get_default(self) -> AgentRoleConfig | None:
        """Return the default (fallback) role."""
        return self._roles.get(self._default_role)

    def list_roles(self) -> list[AgentRoleConfig]:
        """Return all registered roles (enabled only)."""
        return [r for r in self._roles.values() if r.enabled]

    def role_names(self) -> list[str]:
        """Return names of all enabled roles."""
        return [r.name for r in self._roles.values() if r.enabled]

    def __len__(self) -> int:
        return len(self._roles)

    def __contains__(self, name: str) -> bool:
        return name in self._roles
