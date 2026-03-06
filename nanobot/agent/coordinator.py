"""Multi-agent coordinator with LLM-based intent classification.

The ``Coordinator`` sits between the message bus and agent processing.
It classifies each inbound message into one of several specialized roles
and returns the matching ``AgentRoleConfig`` for the agent loop to use.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.registry import AgentRegistry
from nanobot.config.schema import AgentRoleConfig

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


# ------------------------------------------------------------------
# Built-in role definitions (used when user doesn't configure roles)
# ------------------------------------------------------------------

DEFAULT_ROLES: list[AgentRoleConfig] = [
    AgentRoleConfig(
        name="code",
        description="Code generation, debugging, refactoring, and programming tasks.",
        system_prompt=(
            "You are a senior software engineer. Focus on writing clean, correct, "
            "well-tested code. Prefer concrete implementations over explanations."
        ),
    ),
    AgentRoleConfig(
        name="research",
        description="Web search, document analysis, codebase exploration, and fact-finding.",
        system_prompt=(
            "You are a research specialist. Gather information thoroughly, cite sources, "
            "and present findings in a structured format."
        ),
        denied_tools=["write_file", "edit_file"],
    ),
    AgentRoleConfig(
        name="writing",
        description="Documentation, emails, summaries, and content creation.",
        system_prompt=(
            "You are a skilled technical writer. Produce clear, well-structured prose. "
            "Match the appropriate tone and format for the audience."
        ),
        denied_tools=["exec"],
    ),
    AgentRoleConfig(
        name="system",
        description="Shell commands, deployment, infrastructure, and DevOps tasks.",
        system_prompt=(
            "You are a systems engineer and DevOps specialist. Execute commands carefully, "
            "verify results, and explain what each step does."
        ),
    ),
    AgentRoleConfig(
        name="pm",
        description=(
            "Project planning, task breakdown, milestone tracking, "
            "sprint management, and progress coordination."
        ),
        system_prompt=(
            "You are a project manager. Break down goals into actionable steps, "
            "track progress, identify blockers, and coordinate deliverables."
        ),
        denied_tools=["exec"],
    ),
    AgentRoleConfig(
        name="general",
        description="General-purpose assistant for tasks that don't fit other specialists.",
        system_prompt="",
    ),
]


def build_default_registry(default_role: str = "general") -> AgentRegistry:
    """Create a registry pre-loaded with the built-in agent roles."""
    registry = AgentRegistry(default_role=default_role)
    for role in DEFAULT_ROLES:
        registry.register(role)
    return registry


# ------------------------------------------------------------------
# Coordinator
# ------------------------------------------------------------------

_CLASSIFY_SYSTEM = (
    "You are a message router. Given a user message, decide which specialist agent "
    "should handle it. Reply with ONLY a JSON object: "
    '{\"role\": \"<name>\", \"confidence\": <0.0-1.0>}. '
    "confidence 1.0 = very certain, 0.0 = no idea. "
    "Do not include any other text."
)


class Coordinator:
    """LLM-based message router that classifies intent and selects an agent role."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: AgentRegistry,
        *,
        classifier_model: str | None = None,
        default_role: str = "general",
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._classifier_model = classifier_model
        self._default_role = default_role

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    def _build_classify_prompt(self, message: str) -> str:
        """Build the classification user prompt listing available roles."""
        roles = self._registry.list_roles()
        role_lines = "\n".join(f"- **{r.name}**: {r.description}" for r in roles)
        return (
            f"Available agents:\n{role_lines}\n\n"
            f"User message:\n{message}\n\n"
            f"Which agent should handle this? Reply with {{\"role\": \"<name>\"}}."
        )

    async def classify(self, message: str) -> tuple[str, float]:
        """Classify a message and return ``(role_name, confidence)``.

        Uses a lightweight LLM call. Falls back to *default_role* on any
        error or unrecognised response.
        """
        model = self._classifier_model or self._provider.get_default_model()
        user_prompt = self._build_classify_prompt(message)

        t0 = time.monotonic()
        try:
            response = await self._provider.chat(
                messages=[
                    {"role": "system", "content": _CLASSIFY_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                tools=None,
                model=model,
                temperature=0.0,
                max_tokens=64,
            )
            raw = (response.content or "").strip()
            parsed_role, confidence = self._parse_response(raw)
            role_name = parsed_role if parsed_role in self._registry else self._default_role
            latency_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "Coordinator classified → {} (confidence={:.2f}, latency={:.0f}ms, raw: {})",
                role_name,
                confidence,
                latency_ms,
                raw,
            )
            return role_name, confidence
        except Exception:
            logger.warning("Coordinator classification failed, using default role")
            return self._default_role, 0.0

    def _parse_response(self, raw: str) -> tuple[str, float]:
        """Extract role name and confidence from the classifier's raw response.

        Returns ``(role_name, confidence)``.
        """
        # Try JSON parse first
        try:
            data: dict[str, Any] = json.loads(raw)
            if isinstance(data, dict) and "role" in data:
                role = str(data["role"]).strip().lower()
                confidence = float(data.get("confidence", 1.0))
                return role, min(max(confidence, 0.0), 1.0)
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback: look for a known role name in the raw text
        lower = raw.lower()
        for name in self._registry.role_names():
            if name in lower:
                return name, 0.5  # Text-scan match gets moderate confidence
        return self._default_role, 0.0

    async def route(self, message: str) -> AgentRoleConfig:
        """Classify message and return the matching role config."""
        role_name, _confidence = await self.classify(message)
        role = self._registry.get(role_name)
        if role is None:
            role = self._registry.get_default()
        if role is None:
            # Should never happen if registry has defaults, but be safe
            return AgentRoleConfig(name=self._default_role, description="General assistant")
        return role

    def route_direct(self, role_name: str) -> AgentRoleConfig | None:
        """Look up a role by name without LLM classification.

        Returns ``None`` when *role_name* is not found in the registry.
        """
        return self._registry.get(role_name)
