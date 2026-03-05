"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        memory_retrieval_k: int = 6,
        memory_token_budget: int = 900,
        memory_uncertainty_threshold: float = 0.6,
        memory_enable_contradiction_check: bool = True,
        memory_rollout_mode: str = "enabled",
        memory_type_separation_enabled: bool = True,
        memory_router_enabled: bool = True,
        memory_reflection_enabled: bool = True,
        memory_shadow_mode: bool = False,
        memory_shadow_sample_rate: float = 0.2,
        memory_vector_health_enabled: bool = True,
        memory_auto_reindex_on_empty_vector: bool = True,
        memory_history_fallback_enabled: bool = False,
        memory_fallback_allowed_sources: list[str] | None = None,
        memory_fallback_max_summary_chars: int = 280,
        memory_rollout_gate_min_recall_at_k: float = 0.55,
        memory_rollout_gate_min_precision_at_k: float = 0.25,
        memory_rollout_gate_max_avg_memory_context_tokens: float = 1400.0,
        memory_rollout_gate_max_history_fallback_ratio: float = 0.05,
        brave_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.memory_retrieval_k = memory_retrieval_k
        self.memory_token_budget = memory_token_budget
        self.memory_uncertainty_threshold = memory_uncertainty_threshold
        self.memory_enable_contradiction_check = memory_enable_contradiction_check
        self.memory_rollout_overrides = {
            "memory_rollout_mode": memory_rollout_mode,
            "memory_type_separation_enabled": memory_type_separation_enabled,
            "memory_router_enabled": memory_router_enabled,
            "memory_reflection_enabled": memory_reflection_enabled,
            "memory_shadow_mode": memory_shadow_mode,
            "memory_shadow_sample_rate": memory_shadow_sample_rate,
            "memory_vector_health_enabled": memory_vector_health_enabled,
            "memory_auto_reindex_on_empty_vector": memory_auto_reindex_on_empty_vector,
            "memory_history_fallback_enabled": memory_history_fallback_enabled,
            "memory_fallback_allowed_sources": memory_fallback_allowed_sources or ["profile", "events", "mem0_get_all"],
            "memory_fallback_max_summary_chars": memory_fallback_max_summary_chars,
            "rollout_gates": {
                "min_recall_at_k": memory_rollout_gate_min_recall_at_k,
                "min_precision_at_k": memory_rollout_gate_min_precision_at_k,
                "max_avg_memory_context_tokens": memory_rollout_gate_max_avg_memory_context_tokens,
                "max_history_fallback_ratio": memory_rollout_gate_max_history_fallback_ratio,
            },
        }
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(
            workspace,
            memory_retrieval_k=self.memory_retrieval_k,
            memory_token_budget=self.memory_token_budget,
            memory_rollout_overrides=self.memory_rollout_overrides,
        )
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: dict[str, asyncio.Lock] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id, message_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        clean = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
        if not clean:
            return None
        # Strip common reasoning prefixes that sometimes leak into final answers.
        # Remove leading lines like "analysis", "assistantanalysis", etc.
        while True:
            stripped = re.sub(r"^(assistant\s*)?analysis\b[^\n]*\n?", "", clean, flags=re.IGNORECASE).lstrip()
            if stripped == clean:
                break
            clean = stripped
        return clean or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            val = next(iter(tc.arguments.values()), None) if tc.arguments else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _classify_action_intent(text: str) -> tuple[str, str]:
        """
        Classify message into coarse intent classes and domain.

        Returns:
            (intent_class, domain)
            intent_class: informational | state-read | state-write
            domain: email | filesystem | web | system | generic
        """
        content = (text or "").strip().lower()
        if not content:
            return "informational", "generic"

        domain = "generic"
        if re.search(r"\b(email(s)?|inbox(es)?|unread|gmail|imap|smtp)\b", content):
            domain = "email"
        elif re.search(r"\b(file(s)?|folder(s)?|directory(ies)?|path(s)?|workspace)\b", content):
            domain = "filesystem"
        elif re.search(r"\b(web|website(s)?|url(s)?|internet|online)\b", content):
            domain = "web"
        elif re.search(r"\b(log(s)?|process(es)?|service(s)?|system|status|health)\b", content):
            domain = "system"

        write_pattern = (
            r"\b(send|write|edit|delete|update|create|run|execute|schedule|set|start|stop|restart|apply)\b"
        )
        read_pattern = (
            r"\b(check|list|show|fetch|find|search|read|inspect|monitor|get|summarize)\b"
        )
        external_state_pattern = (
            r"\b(inbox(es)?|unread|email(s)?|gmail|imap|smtp|file(s)?|folder(s)?|directory(ies)?|path(s)?|"
            r"log(s)?|status|service(s)?|process(es)?|url(s)?|website(s)?)\b"
        )

        if re.search(write_pattern, content):
            return "state-write", domain
        if re.search(read_pattern, content) and (domain != "generic" or re.search(external_state_pattern, content)):
            return "state-read", domain
        return "informational", domain

    @staticmethod
    def _should_require_tool_evidence(intent_class: str, domain: str) -> bool:
        if intent_class == "state-write":
            return True
        if intent_class == "state-read" and domain != "generic":
            return True
        return False

    @staticmethod
    def _tool_requirement_nudge(intent_class: str, domain: str) -> str:
        tool_hints = {
            "email": "Prefer tools that actually fetch/send email state for this chat.",
            "filesystem": "Prefer filesystem tools (read_file/list_dir/exec) for concrete evidence.",
            "web": "Prefer web_search/web_fetch for concrete evidence.",
            "system": "Prefer exec/log inspection tools for concrete evidence.",
        }
        domain_hint = tool_hints.get(domain, "Use at least one appropriate tool before finalizing.")
        return (
            "For this turn, you must execute at least one tool call before giving a final answer. "
            "Do not answer from memory alone. "
            f"{domain_hint} "
            "If tools fail, report the concrete error output from this turn."
        )

    @staticmethod
    def _user_requested_write(text: str) -> bool:
        """Heuristic: user explicitly asked to create/save/export a file or script."""
        content = (text or "").lower()
        if not content:
            return False
        # Direct verbs with file-ish objects or explicit paths/extensions.
        if re.search(
            r"\b(create|write|save|store|export|dump|generate|make)\b.*\b(file|script|report|note|doc|document|"
            r"markdown|md|json|csv|txt|log)\b",
            content,
        ):
            return True
        if re.search(r"\b(save|write|dump|export)\b.*\b(to|into)\b.*\b(/|~|\\w+\\.(sh|py|js|ts|md|txt|json|csv|log))\b", content):
            return True
        return False

    @staticmethod
    def _filter_write_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove write-capable tools from tool definitions."""
        write_tools = {"write_file", "edit_file", "cron"}
        filtered: list[dict[str, Any]] = []
        for tool in tools:
            name = (tool.get("function") or {}).get("name")
            if name in write_tools:
                continue
            filtered.append(tool)
        return filtered

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        require_tool_evidence: bool = False,
        tool_requirement_nudge: str | None = None,
        allow_write_tools: bool = True,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        turn_tool_calls = 0
        nudged_for_tools = False
        nudged_for_final = False
        disable_tools_next = False

        while iteration < self.max_iterations:
            iteration += 1

            tools_def = None if disable_tools_next else self.tools.get_definitions()
            if tools_def is not None and not allow_write_tools:
                tools_def = self._filter_write_tools(tools_def)
            disable_tools_next = False
            response = await self.provider.chat(
                messages=messages,
                tools=tools_def,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                # Suppress draft content when tool calls are present; keep as reasoning if useful.
                reasoning = response.reasoning_content or response.content
                messages = self.context.add_assistant_message(
                    messages, None, tool_call_dicts,
                    reasoning_content=reasoning,
                )

                for tool_call in response.tool_calls:
                    turn_tool_calls += 1
                    tools_used.append(tool_call.name)
                    # Normalize common arg mistakes (model frequently uses "cmd" for exec).
                    if tool_call.name == "exec" and isinstance(tool_call.arguments, dict):
                        if "command" not in tool_call.arguments and "cmd" in tool_call.arguments:
                            tool_call.arguments["command"] = tool_call.arguments.pop("cmd")
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                if require_tool_evidence and turn_tool_calls == 0 and not nudged_for_tools:
                    nudged_for_tools = True
                    messages.append(
                        {
                            "role": "system",
                            "content": tool_requirement_nudge or (
                                "For this turn, call at least one tool before finalizing your answer."
                            ),
                        }
                    )
                    logger.info("Action intent required tool evidence; retrying once with tool-use nudge.")
                    continue
                if not final_content and turn_tool_calls > 0 and not nudged_for_final:
                    nudged_for_final = True
                    disable_tools_next = True
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "You have already used tools in this turn. "
                                "Now produce the final answer summarizing the tool results. "
                                "Do not call any more tools."
                            ),
                        }
                    )
                    logger.info("Tool results present but no final text; retrying once for final answer.")
                    continue
                final_content = self._strip_think(response.content)
                messages = self.context.add_assistant_message(
                    messages,
                    final_content,
                    reasoning_content=response.reasoning_content,
                )
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )
            messages = self.context.add_assistant_message(messages, final_content)

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                try:
                    response = await self._process_message(msg)
                    if response is not None:
                        await self.bus.publish_outbound(response)
                    elif msg.channel in {"cli", "telegram"}:
                        # Emit an empty outbound to let channel-side state (e.g. Telegram typing indicator)
                        # flush even when the assistant only used tool-driven delivery in this turn.
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id, content="", metadata=msg.metadata or {},
                        ))
                except Exception as e:
                    logger.error("Error processing message: {}", e)
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except RuntimeError:
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None
        try:
            await self.provider.aclose()
        except Exception as e:
            logger.debug("Provider cleanup failed: {}", e)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    def _get_consolidation_lock(self, session_key: str) -> asyncio.Lock:
        lock = self._consolidation_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._consolidation_locks[session_key] = lock
        return lock

    def _prune_consolidation_lock(self, session_key: str, lock: asyncio.Lock) -> None:
        """Drop lock entry if no longer in use."""
        if not lock.locked():
            self._consolidation_locks.pop(session_key, None)

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        content = (text or "").strip().lower()
        if not content:
            return False
        if "?" in content:
            return True
        starters = (
            "what ", "which ", "who ", "when ", "where ", "why ", "how ",
            "is ", "are ", "do ", "does ", "did ", "can ", "could ",
            "should ", "would ", "will ",
        )
        return content.startswith(starters)

    def _estimate_grounding_confidence(self, query: str) -> float:
        try:
            items = self.context.memory.retrieve(
                query,
                top_k=1,
            )
        except Exception:
            return 0.0
        if not items:
            return 0.0
        top = items[0]
        try:
            score = float(top.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        return max(0.0, min(1.0, score))

    def _should_force_verification(self, text: str) -> bool:
        if not self._looks_like_question(text):
            return False
        confidence = self._estimate_grounding_confidence(text)
        return confidence < self.memory_uncertainty_threshold

    @staticmethod
    def _build_no_answer_explanation(user_text: str, messages: list[dict[str, Any]]) -> str:
        """Explain why the agent could not produce an answer on this turn."""
        tool_results = [m for m in messages if m.get("role") == "tool"]
        last_tool = tool_results[-1] if tool_results else None
        last_tool_name = str(last_tool.get("name", "")) if last_tool else ""
        last_tool_content = str(last_tool.get("content", "")) if last_tool else ""
        lowered = last_tool_content.lower()

        reasons: list[str] = []
        if not tool_results:
            reasons.append("I did not get usable evidence from tools or memory retrieval.")
        if "exit code: 1" in lowered or "no such file" in lowered or "not found" in lowered:
            reasons.append(
                f"My last check with `{last_tool_name or 'a tool'}` returned no matching data."
            )
        if "permission denied" in lowered:
            reasons.append("The lookup failed due to a local permission error.")
        if "insufficient_quota" in lowered or "429" in lowered:
            reasons.append("A provider quota/rate limit blocked part of the retrieval.")
        if not reasons:
            reasons.append("The model returned no final answer text after tool execution.")

        question = (user_text or "").strip()
        help_line = (
            "Please share the fact directly and I can save it to memory."
            if question
            else "Please restate your question and I will retry with explicit verification."
        )

        primary_reason = reasons[0]
        return f"Sorry, I couldn't answer that just now. {primary_reason} {help_line}"

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=self.memory_window)
            skill_names = self.context.skills.detect_relevant_skills(msg.content)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content,
                skill_names=skill_names,
                channel=channel,
                chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._get_consolidation_lock(session.key)
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        archived = await self._consolidate_memory(temp, archive_all=True)
                        if not archived:
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)
                self._prune_consolidation_lock(session.key, lock)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")

        memory_store = self.context.memory

        conflict_reply = memory_store.handle_user_conflict_reply(msg.content)
        if conflict_reply.get("handled"):
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=str(conflict_reply.get("message", "")),
            )

        try:
            correction_result = memory_store.apply_live_user_correction(
                msg.content,
                channel=msg.channel,
                chat_id=msg.chat_id,
                enable_contradiction_check=self.memory_enable_contradiction_check,
            )
            if correction_result.get("question"):
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=str(correction_result.get("question", "")),
                )
        except Exception:
            logger.exception("Live correction capture failed")

        # Proactively ask for unresolved conflicts discovered during prior consolidation turns.
        pending_conflict_question = memory_store.ask_user_for_conflict()
        if pending_conflict_question:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=pending_conflict_question,
            )

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._get_consolidation_lock(session.key)

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    self._prune_consolidation_lock(session.key, lock)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        intent_class, intent_domain = self._classify_action_intent(msg.content)
        require_tool_evidence = self._should_require_tool_evidence(intent_class, intent_domain)
        allow_write_tools = (intent_class == "state-write") or self._user_requested_write(msg.content)
        verify_before_answer = self._should_force_verification(msg.content) or require_tool_evidence
        skill_names = self.context.skills.detect_relevant_skills(msg.content)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            skill_names=skill_names,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            verify_before_answer=verify_before_answer,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, tools_used, all_msgs = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            require_tool_evidence=require_tool_evidence,
            tool_requirement_nudge=self._tool_requirement_nudge(intent_class, intent_domain),
            allow_write_tools=allow_write_tools,
        )

        if final_content is None:
            final_content = self._build_no_answer_explanation(msg.content, all_msgs)
            # Ensure fallback responses are recorded in the session log.
            all_msgs = self.context.add_assistant_message(all_msgs, final_content)
        elif require_tool_evidence and not tools_used:
            final_content = (
                "I could not complete that action because no tool execution succeeded in this turn. "
                "Please retry, and I will return results grounded in live tool output."
            )

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return None

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    _TOOL_RESULT_MAX_CHARS = 500

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = {k: v for k, v in m.items() if k != "reasoning_content"}
            if entry.get("role") == "tool" and isinstance(entry.get("content"), str):
                content = entry["content"]
                if len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        return await self.context.memory.consolidate(
            session, self.provider, self.model,
            archive_all=archive_all,
            memory_window=self.memory_window,
            enable_contradiction_check=self.memory_enable_contradiction_check,
        )

    def _fallback_archive_snapshot(self, snapshot: list[dict]) -> bool:
        """Fallback archival used by /new when AI consolidation fails."""
        try:
            lines: list[str] = []
            for m in snapshot:
                content = m.get("content")
                if not content:
                    continue
                tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
                timestamp = str(m.get("timestamp", "?"))[:16]
                role = str(m.get("role", "unknown")).upper()
                lines.append(f"[{timestamp}] {role}{tools}: {content}")

            if not lines:
                return True

            header = (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] "
                f"Fallback archive from /new ({len(lines)} messages)"
            )
            entry = header + "\n" + "\n".join(lines)
            self.context.memory.append_history(entry)
            logger.warning("/new used fallback archival: {} messages", len(lines))
            return True
        except Exception:
            logger.exception("Fallback archival failed")
            return False

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
