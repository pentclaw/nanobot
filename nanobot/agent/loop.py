"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryConsolidator
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

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        temperature: float | None = None,
        max_tokens: int | None = None,
        memory_window: int | None = None,
        search_api_key: str | None = None,
        search_engine: str = "tavily",
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
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
        self.context_window_tokens = context_window_tokens
        self.temperature = (
            temperature
            if temperature is not None
            else getattr(provider.generation, "temperature", 0.7)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else getattr(provider.generation, "max_tokens", 4096)
        )
        self.memory_window = memory_window
        # Keep backward compatibility with legacy brave_api_key wiring.
        self.search_api_key = search_api_key or brave_api_key
        self.search_engine = search_engine
        self.reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else getattr(provider.generation, "reasoning_effort", None)
        )
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            search_api_key=self.search_api_key,
            search_engine=self.search_engine,
            brave_api_key=self.search_api_key,
            web_proxy=self.web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
        )
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
            path_append=self.exec_config.path_append,
        ))
        # Web tools
        self.tools.register(WebSearchTool(
            api_key=self.search_api_key,
            engine=self.search_engine,
            proxy=self.web_proxy,
        ))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
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
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _extract_think(text: str | None) -> str | None:
        """Extract text inside <think>…</think> blocks."""
        if not text:
            return None
        blocks = [b.strip() for b in re.findall(r"<think>([\s\S]*?)</think>", text) if b and b.strip()]
        if not blocks:
            return None
        return "\n".join(blocks)

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint with all arguments, e.g. 'exec(command="ls", working_dir="/tmp")'."""
        _max_val = 40

        def _fmt_val(val: Any) -> str:
            if val is None:
                return "None"
            s = val if isinstance(val, str) else repr(val)
            if len(s) > _max_val:
                return s[:_max_val] + "…"
            return s

        def _fmt(tc) -> str:
            args = tc.arguments[0] if isinstance(tc.arguments, list) and tc.arguments else tc.arguments
            if not args or not isinstance(args, dict):
                return tc.name + "()"
            parts = [f'{k}={_fmt_val(v)}' for k, v in args.items()]
            return f"{tc.name}({', '.join(parts)})"
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _reasoning_summary(text: str | None, max_len: int = 240) -> str | None:
        """Return a compact one-line reasoning summary for progress UI."""
        if not text:
            return None
        clean = re.sub(r"\s+", " ", text).strip()
        # Remove markdown heading markers to avoid accidental heading rendering in stream cards.
        clean = re.sub(r"(^|\s)#{1,6}\s+", r"\1", clean)
        if not clean:
            return None
        if len(clean) > max_len:
            clean = clean[:max_len] + "…"
        return clean

    @staticmethod
    def _pick_reasoning_for_rich_stream(
        reasoning_content: str | None,
        content: str | None,
        *,
        has_tool_calls: bool,
    ) -> tuple[str | None, str]:
        """Pick best reasoning snippet for rich stream UI and return (summary, source)."""
        if reasoning_content:
            summary = AgentLoop._reasoning_summary(reasoning_content)
            if summary:
                return summary, "reasoning_content"

        think_text = AgentLoop._extract_think(content)
        if think_text:
            summary = AgentLoop._reasoning_summary(think_text)
            if summary:
                return summary, "think_tag"

        if has_tool_calls:
            fallback = AgentLoop._strip_think(content)
            if fallback:
                summary = AgentLoop._reasoning_summary(fallback)
                if summary:
                    return summary, "content_fallback"

        return None, "none"

    @staticmethod
    def _tool_result_summary(result: Any, max_len: int = 180) -> str:
        """Return a compact one-line summary of tool output."""
        if result is None:
            return "No output"
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return "No output"
        if len(clean) > max_len:
            clean = clean[:max_len] + "…"
        return clean

    @staticmethod
    def _tool_params_summary(arguments: dict[str, Any] | None) -> str:
        """Return concise key=value pairs for tool arguments."""
        if not arguments:
            return "(none)"
        max_val = 40
        parts: list[str] = []
        for k, v in arguments.items():
            sval = v if isinstance(v, str) else repr(v)
            if len(sval) > max_val:
                sval = sval[:max_val] + "…"
            parts.append(f"{k}={sval}")
        return ", ".join(parts)

    @staticmethod
    def _as_text_code_block(text: str) -> str:
        """Wrap text in a safe markdown text code block."""
        safe = (text or "").replace("```", "'''")
        return f"```text\n{safe}\n```"

    @staticmethod
    def _render_tool_entry(name: str, arguments: dict[str, Any] | None, summary: str) -> str:
        """Render one tool execution item with separated fields."""
        params_text = AgentLoop._tool_params_summary(arguments)
        return (
            f"🔧 Tool: **{name}**\n"
            "Params:\n"
            f"{AgentLoop._as_text_code_block(params_text)}\n"
            "Summary:\n"
            f"{AgentLoop._as_text_code_block(summary)}\n\n"
        )

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        stream_callback: Callable[[str], Any] | None = None,
        stream_ui: str | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        rich_stream = bool(stream_callback and stream_ui == "feishu_chat_sections")
        show_tool_details = True
        if rich_stream and self.channels_config is not None:
            show_tool_details = bool(self.channels_config.send_tool_hints)
        analysis_header_sent = False
        final_header_sent = False
        missing_reasoning_noted = False
        section_gap = "\n\n"

        async def _emit(text: str) -> None:
            if not stream_callback or not text:
                return
            res = stream_callback(text)
            if asyncio.iscoroutine(res):
                await res

        while iteration < self.max_iterations:
            iteration += 1

            tool_defs = self.tools.get_definitions()
            if stream_callback:
                # Use streaming provider
                full_content = ""
                full_reasoning = ""
                tool_calls: list = []

                async for chunk in self.provider.stream(
                    messages=messages,
                    tools=tool_defs,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                ):
                    if chunk.content:
                        full_content += chunk.content
                        if not rich_stream:
                            await _emit(chunk.content)
                    if chunk.reasoning_content:
                        full_reasoning += chunk.reasoning_content
                    if chunk.tool_calls:
                        tool_calls.extend(chunk.tool_calls)

                from nanobot.providers.base import LLMResponse
                response = LLMResponse(
                    content=full_content if full_content else None,
                    reasoning_content=full_reasoning if full_reasoning else None,
                    tool_calls=tool_calls,
                )
            else:
                response = await self.provider.chat_with_retry(
                    messages=messages,
                    tools=tool_defs,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )

            if response.has_tool_calls:
                if on_progress and not stream_callback:
                    thoughts = [
                        self._strip_think(response.content),
                        response.reasoning_content,
                        *(
                            f"Thinking [{b.get('signature', '...')}]:\n{b.get('thought', '...')}"
                            for b in (response.thinking_blocks or [])
                            if isinstance(b, dict) and "signature" in b
                        ),
                    ]
                    combined_thoughts = "\n\n".join(filter(None, thoughts))
                    if combined_thoughts:
                        await on_progress(combined_thoughts)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)
                elif rich_stream:
                    if not analysis_header_sent:
                        heading = "🧠 **Reasoning Log & 🔧 Tool Execution**" if show_tool_details else "🧠 **Reasoning Log**"
                        await _emit(f"{heading}{section_gap}")
                        analysis_header_sent = True
                    reasoning, source = self._pick_reasoning_for_rich_stream(
                        response.reasoning_content,
                        response.content,
                        has_tool_calls=True,
                    )
                    logger.debug("Rich stream reasoning source (tool round): {}", source)
                    if reasoning:
                        await _emit(f"🤔 {reasoning}\n")
                    elif not missing_reasoning_noted:
                        await _emit("🤔 No reasoning details were returned by the model.\n")
                        missing_reasoning_noted = True

                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    if rich_stream and show_tool_details:
                        entry = self._render_tool_entry(
                            tool_call.name,
                            tool_call.arguments,
                            self._tool_result_summary(result),
                        )
                        await _emit(entry)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )

            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                if on_progress and clean and not stream_callback:
                    await on_progress(clean)
                elif rich_stream:
                    reasoning, source = self._pick_reasoning_for_rich_stream(
                        response.reasoning_content,
                        response.content,
                        has_tool_calls=False,
                    )
                    logger.debug("Rich stream reasoning source (final round): {}", source)
                    if reasoning:
                        if not analysis_header_sent:
                            await _emit(f"🧠 **Reasoning Log**{section_gap}")
                        await _emit(f"🤔 {reasoning}\n")
                    if not final_header_sent:
                        await _emit(f"{section_gap}📌 **Final Output**{section_gap}---{section_gap}")
                        final_header_sent = True
                    if clean:
                        await _emit(clean)
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            stream_callback = None
            if msg.stream_id:
                stream_callback = self.bus.get_stream_callback(msg.stream_id)

            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)
            elif cmd == "/restart":
                await self._handle_restart(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg, stream_callback=stream_callback))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)
            if msg.stream_id and cmd in {"/stop", "/restart"}:
                self.bus.mark_stream_done(msg.stream_id)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _handle_restart(self, msg: InboundMessage) -> None:
        """Restart the process in-place via os.execv."""
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        ))

        async def _do_restart():
            await asyncio.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)

        asyncio.create_task(_do_restart())

    async def _dispatch(
        self,
        msg: InboundMessage,
        stream_callback: Callable[[str], Any] | None = None,
    ) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg, stream_callback=stream_callback)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))
            finally:
                if msg.stream_id:
                    self.bus.mark_stream_done(msg.stream_id)

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        stream_callback: Callable[[str], Any] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            try:
                if not await self.memory_consolidator.archive_unconsolidated(session):
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Memory archival failed, session not cleared. Please try again.",
                    )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            # Session key was explicitly reset; drop its lock mapping to avoid stale retention.
            self._consolidation_locks.pop(session.key, None)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            lines = [
                "🐈 nanobot commands:",
                "/new — Start a new conversation",
                "/stop — Stop the current task",
                "/restart — Restart the bot",
                "/help — Show available commands",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines),
            )
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            stream_callback=stream_callback,
            stream_ui=(msg.metadata or {}).get("_stream_ui"),
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        # Mark stream as done so channel can close streaming session
        if msg.stream_id:
            self.bus.mark_stream_done(msg.stream_id)

        # If streaming was used, content was already delivered via callback
        if stream_callback:
            return None

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        stream_callback: Callable[[str], Any] | None = None,
        stream_ui: str | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        metadata: dict[str, Any] = {}
        if stream_ui:
            metadata["_stream_ui"] = stream_ui
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content,
            metadata=metadata,
        )
        # Keep direct invocations (cron/heartbeat/CLI) serialized with channel-driven
        # dispatches to avoid shared per-turn tool state races.
        async with self._processing_lock:
            response = await self._process_message(
                msg, session_key=session_key, on_progress=on_progress, stream_callback=stream_callback
            )
        return response.content if response else ""
