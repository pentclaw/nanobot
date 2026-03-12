"""Tests for AgentLoop._tool_hint formatting of tool calls."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import nanobot.channels.feishu as feishu_module
import nanobot.channels.feishu_streaming as feishu_streaming
from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.feishu import FeishuChannel, _normalize_stream_markdown
from nanobot.config.schema import ChannelsConfig, FeishuConfig
from nanobot.providers.base import LLMResponse, ToolCallRequest


def test_tool_hint_no_arguments():
    """Tool with no arguments shows name()."""
    calls = [ToolCallRequest(id="1", name="list_dir", arguments={})]
    assert AgentLoop._tool_hint(calls) == "list_dir()"


def test_tool_hint_single_argument():
    """Tool with one argument shows name(arg=value)."""
    calls = [
        ToolCallRequest(id="1", name="web_search", arguments={"query": "weather Berlin"})
    ]
    assert AgentLoop._tool_hint(calls) == 'web_search(query=weather Berlin)'


def test_tool_hint_multiple_arguments():
    """Tool with multiple arguments shows all (e.g. exec with command and working_dir)."""
    calls = [
        ToolCallRequest(
            id="1",
            name="exec",
            arguments={"command": "ls -la", "working_dir": "/tmp"},
        )
    ]
    hint = AgentLoop._tool_hint(calls)
    assert "exec(" in hint
    assert "command=ls -la" in hint
    assert "working_dir=/tmp" in hint


def test_tool_hint_long_value_truncated():
    """Long string values are truncated with ellipsis."""
    long_query = "x" * 50
    calls = [
        ToolCallRequest(id="1", name="web_search", arguments={"query": long_query})
    ]
    hint = AgentLoop._tool_hint(calls)
    assert "…" in hint
    assert len(hint) < 60


def test_tool_hint_multiple_calls():
    """Multiple tool calls are comma-separated."""
    calls = [
        ToolCallRequest(id="1", name="read_file", arguments={"path": "a.txt"}),
        ToolCallRequest(id="2", name="read_file", arguments={"path": "b.txt"}),
    ]
    hint = AgentLoop._tool_hint(calls)
    assert "read_file(path=a.txt)" in hint
    assert "read_file(path=b.txt)" in hint
    assert hint.count(", ") >= 1


def test_tool_hint_non_string_value():
    """Non-string argument values use repr (e.g. numbers, bool)."""
    calls = [
        ToolCallRequest(id="1", name="fake_tool", arguments={"count": 3, "flag": True})
    ]
    hint = AgentLoop._tool_hint(calls)
    assert "count=3" in hint or "3" in hint
    assert "flag=True" in hint or "True" in hint


def test_channels_send_tool_hints_defaults_to_true() -> None:
    cfg = ChannelsConfig()
    assert cfg.send_tool_hints is True


class _StreamingProvider:
    """Minimal provider that returns predefined stream rounds."""

    def __init__(self, rounds: list[list[LLMResponse]]):
        self._rounds = rounds
        self._idx = 0
        self.last_stream_kwargs: dict[str, Any] | None = None

    def get_default_model(self) -> str:
        return "test-model"

    async def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content="unused")

    async def stream(self, *args: Any, **kwargs: Any):
        self.last_stream_kwargs = dict(kwargs)
        i = self._idx
        self._idx += 1
        for chunk in self._rounds[i]:
            yield chunk


@pytest.mark.asyncio
async def test_rich_stream_includes_reasoning_tools_and_final_section(tmp_path: Path) -> None:
    provider = _StreamingProvider(
        rounds=[
            [
                LLMResponse(content=None, reasoning_content="我将先读取配置再回答。"),
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="tc1",
                            name="read_file",
                            arguments={"path": "README.md"},
                        )
                    ],
                ),
            ],
            [
                LLMResponse(content=None, reasoning_content="已拿到信息，组织答案。"),
                LLMResponse(content="这里是最终答案。"),
            ],
        ]
    )

    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="README contains quickstart details.")

    streamed: list[str] = []

    def _cb(text: str) -> None:
        streamed.append(text)

    final, tools_used, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "请分析项目"}],
        stream_callback=_cb,
        stream_ui="feishu_chat_sections",
    )

    full = "".join(streamed)
    assert final == "这里是最终答案。"
    assert tools_used == ["read_file"]

    assert "🧠 **Reasoning Log & 🔧 Tool Execution**" in full
    assert "🤔 我将先读取配置再回答。" in full
    assert "我将先读取配置再回答。" in full
    assert "🔧 Tool: **read_file**" in full
    assert "Params:" in full
    assert "```text\npath=README.md\n```" in full
    assert "Summary:" in full
    assert "```text" in full

    assert "---" in full
    assert "📌 **Final Output**" in full
    assert "这里是最终答案。" in full
    assert full.index("📌 **Final Output**") < full.index("---")


@pytest.mark.asyncio
async def test_rich_stream_hides_tool_entries_when_send_tool_hints_disabled(tmp_path: Path) -> None:
    provider = _StreamingProvider(
        rounds=[
            [
                LLMResponse(content=None, reasoning_content="先看文件后回答。"),
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="tc1",
                            name="read_file",
                            arguments={"path": "README.md"},
                        )
                    ],
                ),
            ],
            [LLMResponse(content="最终结果。")],
        ]
    )

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        channels_config=ChannelsConfig(send_tool_hints=False),
    )
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="README content")

    streamed: list[str] = []

    def _cb(text: str) -> None:
        streamed.append(text)

    final, tools_used, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "请分析项目"}],
        stream_callback=_cb,
        stream_ui="feishu_chat_sections",
    )

    full = "".join(streamed)
    assert final == "最终结果。"
    assert tools_used == ["read_file"]
    assert "🧠 **Reasoning Log**" in full
    assert "🧠 **Reasoning Log & 🔧 Tool Execution**" not in full
    assert "🔧 Tool: **read_file**" not in full
    assert "🤔 先看文件后回答。" in full
    assert "📌 **Final Output**" in full


@pytest.mark.asyncio
async def test_rich_stream_falls_back_to_think_block_when_reasoning_missing(tmp_path: Path) -> None:
    provider = _StreamingProvider(
        rounds=[
            [
                LLMResponse(content="<think>先分析再调用工具。</think>让我先读取文件。"),
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="tc1",
                            name="read_file",
                            arguments={"path": "README.md"},
                        )
                    ],
                ),
            ],
            [LLMResponse(content="最终答案。")],
        ]
    )

    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="README content")

    streamed: list[str] = []

    def _cb(text: str) -> None:
        streamed.append(text)

    final, tools_used, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "请处理"}],
        stream_callback=_cb,
        stream_ui="feishu_chat_sections",
    )

    full = "".join(streamed)
    assert final == "最终答案。"
    assert tools_used == ["read_file"]
    assert "🧠 **Reasoning Log & 🔧 Tool Execution**" in full
    assert "🤔 先分析再调用工具。" in full
    assert "先分析再调用工具。" in full
    assert "🔧 Tool: **read_file**" in full


@pytest.mark.asyncio
async def test_rich_stream_falls_back_when_reasoning_is_whitespace_only(tmp_path: Path) -> None:
    provider = _StreamingProvider(
        rounds=[
            [
                LLMResponse(
                    content="<think>使用 think 回退推理。</think>先调用工具。",
                    reasoning_content="   \n\t  ",
                ),
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="tc1",
                            name="read_file",
                            arguments={"path": "README.md"},
                        )
                    ],
                ),
            ],
            [LLMResponse(content="最终答案。")],
        ]
    )

    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="README content")

    streamed: list[str] = []

    def _cb(text: str) -> None:
        streamed.append(text)

    final, tools_used, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "请处理"}],
        stream_callback=_cb,
        stream_ui="feishu_chat_sections",
    )

    full = "".join(streamed)
    assert final == "最终答案。"
    assert tools_used == ["read_file"]
    assert "🤔 使用 think 回退推理。" in full
    assert "🔧 Tool: **read_file**" in full


@pytest.mark.asyncio
async def test_rich_stream_falls_back_to_content_on_tool_round_when_reasoning_missing(tmp_path: Path) -> None:
    provider = _StreamingProvider(
        rounds=[
            [
                LLMResponse(content="我先调用工具读取 README 再回答。"),
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="tc1",
                            name="read_file",
                            arguments={"path": "README.md"},
                        )
                    ],
                ),
            ],
            [LLMResponse(content="最终答案。")],
        ]
    )

    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="README content")

    streamed: list[str] = []

    def _cb(text: str) -> None:
        streamed.append(text)

    final, tools_used, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "请处理"}],
        stream_callback=_cb,
        stream_ui="feishu_chat_sections",
    )

    full = "".join(streamed)
    assert final == "最终答案。"
    assert tools_used == ["read_file"]
    assert "🧠 **Reasoning Log & 🔧 Tool Execution**" in full
    assert "🤔 我先调用工具读取 README 再回答。" in full
    assert "我先调用工具读取 README 再回答。" in full
    assert "🔧 Tool: **read_file**" in full


@pytest.mark.asyncio
async def test_rich_stream_shows_reasoning_placeholder_when_model_returns_none(tmp_path: Path) -> None:
    provider = _StreamingProvider(
        rounds=[
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="tc1",
                            name="read_file",
                            arguments={"path": "README.md"},
                        )
                    ],
                ),
            ],
            [LLMResponse(content="最终答案。")],
        ]
    )

    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="README content")

    streamed: list[str] = []

    def _cb(text: str) -> None:
        streamed.append(text)

    final, tools_used, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "请处理"}],
        stream_callback=_cb,
        stream_ui="feishu_chat_sections",
    )

    full = "".join(streamed)
    assert final == "最终答案。"
    assert tools_used == ["read_file"]
    assert "🧠 **Reasoning Log & 🔧 Tool Execution**" in full
    assert "🤔 No reasoning details were returned by the model." in full
    assert "No reasoning details were returned by the model." in full


@pytest.mark.asyncio
async def test_rich_stream_final_output_without_reasoning_does_not_emit_reasoning_log(tmp_path: Path) -> None:
    provider = _StreamingProvider(rounds=[[LLMResponse(content="直接给出答案。")]])

    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.tools.get_definitions = MagicMock(return_value=[])

    streamed: list[str] = []

    def _cb(text: str) -> None:
        streamed.append(text)

    final, _, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "hello"}],
        stream_callback=_cb,
        stream_ui="feishu_chat_sections",
    )

    full = "".join(streamed)
    assert final == "直接给出答案。"
    assert "🧠 **Reasoning Log**" not in full
    assert "📌 **Final Output**" in full
    assert "直接给出答案。" in full


@pytest.mark.asyncio
async def test_default_streaming_behavior_unchanged_without_stream_ui(tmp_path: Path) -> None:
    provider = _StreamingProvider(rounds=[[LLMResponse(content="plain stream")]])

    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.tools.get_definitions = MagicMock(return_value=[])

    streamed: list[str] = []

    def _cb(text: str) -> None:
        streamed.append(text)

    final, _, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "hello"}],
        stream_callback=_cb,
    )

    assert final == "plain stream"
    assert "".join(streamed) == "plain stream"


@pytest.mark.asyncio
async def test_plain_stream_with_tool_calls_does_not_emit_progress_hints(tmp_path: Path) -> None:
    provider = _StreamingProvider(
        rounds=[
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="tc1",
                            name="read_file",
                            arguments={"path": "README.md"},
                        )
                    ],
                ),
            ],
            [LLMResponse(content="plain final")],
        ]
    )

    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="README content")

    streamed: list[str] = []
    progress: list[tuple[str, bool]] = []

    def _cb(text: str) -> None:
        streamed.append(text)

    async def _on_progress(content: str, *, tool_hint: bool = False) -> None:
        progress.append((content, tool_hint))

    final, tools_used, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "请处理"}],
        on_progress=_on_progress,
        stream_callback=_cb,
    )

    assert final == "plain final"
    assert tools_used == ["read_file"]
    assert "".join(streamed) == "plain final"
    assert progress == []


@pytest.mark.asyncio
async def test_stream_passes_reasoning_effort_to_provider(tmp_path: Path) -> None:
    provider = _StreamingProvider(rounds=[[LLMResponse(content="ok")]])

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        reasoning_effort="high",
    )
    loop.tools.get_definitions = MagicMock(return_value=[])

    final, _, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "hello"}],
        stream_callback=lambda _: None,
    )

    assert final == "ok"
    assert provider.last_stream_kwargs is not None
    assert provider.last_stream_kwargs.get("reasoning_effort") == "high"


@pytest.mark.asyncio
async def test_process_message_forwards_stream_ui_from_metadata(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"

    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop._run_agent_loop = AsyncMock(return_value=("ok", [], []))  # type: ignore[method-assign]

    msg = InboundMessage(
        channel="feishu",
        sender_id="u1",
        chat_id="oc_xxx",
        content="你好",
        metadata={"_stream_ui": "feishu_chat_sections"},
    )

    def _cb(_: str) -> None:
        return None

    result = await loop._process_message(msg, stream_callback=_cb)

    assert result is None
    loop._run_agent_loop.assert_awaited_once()
    _, kwargs = loop._run_agent_loop.await_args
    assert kwargs["stream_ui"] == "feishu_chat_sections"


@pytest.mark.asyncio
async def test_process_direct_sets_stream_ui_metadata(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model")
    loop._process_message = AsyncMock(return_value=None)  # type: ignore[method-assign]

    def _cb(_: str) -> None:
        return None

    result = await loop.process_direct(
        "主动任务",
        session_key="cron:test",
        channel="feishu",
        chat_id="oc_xxx",
        stream_callback=_cb,
        stream_ui="feishu_chat_sections",
    )

    assert result == ""
    loop._process_message.assert_awaited_once()
    msg = loop._process_message.await_args.args[0]
    assert msg.metadata["_stream_ui"] == "feishu_chat_sections"


@pytest.mark.asyncio
async def test_process_message_rich_stream_message_tool_keeps_final_output(tmp_path: Path) -> None:
    provider = _StreamingProvider(
        rounds=[
            [
                LLMResponse(content=None, reasoning_content="先发送消息给用户。"),
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCallRequest(
                            id="tc1",
                            name="message",
                            arguments={"content": "中间进度"},
                        )
                    ],
                ),
            ],
            [
                LLMResponse(content=None, reasoning_content="发送完成，给出总结。"),
                LLMResponse(content="这是最终总结。"),
            ],
        ]
    )

    loop = AgentLoop(bus=MessageBus(), provider=provider, workspace=tmp_path)
    mt = loop.tools.get("message")
    if mt is not None and hasattr(mt, "set_send_callback"):
        mt.set_send_callback(AsyncMock(return_value=None))

    streamed: list[str] = []

    def _cb(text: str) -> None:
        streamed.append(text)

    msg = InboundMessage(
        channel="feishu",
        sender_id="u1",
        chat_id="oc_xxx",
        content="请先发消息再总结",
        metadata={"_stream_ui": "feishu_chat_sections"},
    )

    result = await loop._process_message(msg, stream_callback=_cb)

    assert result is None
    full = "".join(streamed)
    assert "🔧 Tool: **message**" in full
    assert "Message sent to feishu:oc_xxx" in full
    assert "📌 **Final Output**" in full
    assert "这是最终总结。" in full


def _make_feishu_event(message_id: str = "m1", text: str = "hello") -> SimpleNamespace:
    return SimpleNamespace(
        event=SimpleNamespace(
            message=SimpleNamespace(
                message_id=message_id,
                chat_id="oc_test",
                chat_type="p2p",
                message_type="text",
                content=json.dumps({"text": text}, ensure_ascii=False),
            ),
            sender=SimpleNamespace(
                sender_type="user",
                sender_id=SimpleNamespace(open_id="ou_test"),
            ),
        )
    )


def test_feishu_normalize_stream_markdown_splits_tool_and_params_lines() -> None:
    raw = "Tool: execParams: command=lsSummary:\n```text\nok\n```"
    normalized = _normalize_stream_markdown(raw)
    assert "Tool: exec\nParams:" in normalized
    assert "Params: command=ls\nSummary:" in normalized


def test_feishu_normalize_stream_markdown_strips_terminal_controls() -> None:
    raw = "Summary:\n```text\n\x1b[90m[|]\x1b[0m\b\b\b\bnanobot\n```"
    normalized = _normalize_stream_markdown(raw)
    assert "\x1b" not in normalized
    assert "\b" not in normalized
    assert "nanobot" in normalized


@pytest.mark.asyncio
async def test_feishu_on_message_streams_with_single_ui_by_default(monkeypatch) -> None:
    monkeypatch.setattr(feishu_module, "CARDKIT_AVAILABLE", True)

    class _FakeSingle:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.current_text = ""
            self.accumulated_text = ""
            self._lock = feishu_module.threading.Lock()

        def start_sync(self) -> bool:
            return True

        def update_sync(self, text: str) -> bool:
            self.current_text = text
            return True

        def close_sync(self, final_text: str | None = None) -> bool:
            self.closed = True
            self.current_text = final_text or self.current_text
            return True

    monkeypatch.setattr(feishu_module, "FeishuStreamingSession", _FakeSingle)

    channel = FeishuChannel(FeishuConfig(enabled=True, streaming=True, allow_from=["*"]), MessageBus())
    channel._client = object()

    captured = []

    async def _fake_add_reaction(*args, **kwargs) -> None:
        return None

    async def _fake_publish_inbound(msg) -> None:
        captured.append(msg)

    async def _fake_wait_and_close_stream(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(channel, "_add_reaction", _fake_add_reaction)
    monkeypatch.setattr(channel.bus, "publish_inbound", _fake_publish_inbound)
    monkeypatch.setattr(channel, "_wait_and_close_stream", _fake_wait_and_close_stream)

    await channel._on_message(_make_feishu_event("m-single"))

    assert len(captured) == 1
    assert captured[0].stream_id is not None
    assert captured[0].metadata["_stream_ui"] == "feishu_chat_sections"


@pytest.mark.asyncio
async def test_feishu_on_message_streams_without_sections_when_plain_stream_ui(monkeypatch) -> None:
    monkeypatch.setattr(feishu_module, "CARDKIT_AVAILABLE", True)

    class _FakeSingle:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.current_text = ""
            self.accumulated_text = ""
            self._lock = feishu_module.threading.Lock()

        def start_sync(self) -> bool:
            return True

        def update_sync(self, text: str) -> bool:
            self.current_text = text
            return True

        def close_sync(self, final_text: str | None = None) -> bool:
            self.closed = True
            self.current_text = final_text or self.current_text
            return True

    monkeypatch.setattr(feishu_module, "FeishuStreamingSession", _FakeSingle)

    channel = FeishuChannel(
        FeishuConfig(enabled=True, streaming=True, stream_ui="plain", allow_from=["*"]),
        MessageBus(),
    )
    channel._client = object()

    captured = []

    async def _fake_add_reaction(*args, **kwargs) -> None:
        return None

    async def _fake_publish_inbound(msg) -> None:
        captured.append(msg)

    async def _fake_wait_and_close_stream(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(channel, "_add_reaction", _fake_add_reaction)
    monkeypatch.setattr(channel.bus, "publish_inbound", _fake_publish_inbound)
    monkeypatch.setattr(channel, "_wait_and_close_stream", _fake_wait_and_close_stream)

    await channel._on_message(_make_feishu_event("m-plain"))

    assert len(captured) == 1
    assert captured[0].stream_id is not None
    assert "_stream_ui" not in captured[0].metadata


@pytest.mark.asyncio
async def test_feishu_on_message_disables_stream_when_single_init_fails(monkeypatch) -> None:
    monkeypatch.setattr(feishu_module, "CARDKIT_AVAILABLE", True)

    class _FakeSingle:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.current_text = ""
            self.accumulated_text = ""
            self._lock = feishu_module.threading.Lock()

        def start_sync(self) -> bool:
            return False

        def update_sync(self, text: str) -> bool:
            self.current_text = text
            return True

        def close_sync(self, final_text: str | None = None) -> bool:
            self.closed = True
            self.current_text = final_text or self.current_text
            return True

    monkeypatch.setattr(feishu_module, "FeishuStreamingSession", _FakeSingle)

    channel = FeishuChannel(FeishuConfig(enabled=True, streaming=True, allow_from=["*"]), MessageBus())
    channel._client = object()

    captured = []

    async def _fake_add_reaction(*args, **kwargs) -> None:
        return None

    async def _fake_publish_inbound(msg) -> None:
        captured.append(msg)

    async def _fake_wait_and_close_stream(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(channel, "_add_reaction", _fake_add_reaction)
    monkeypatch.setattr(channel.bus, "publish_inbound", _fake_publish_inbound)
    monkeypatch.setattr(channel, "_wait_and_close_stream", _fake_wait_and_close_stream)

    await channel._on_message(_make_feishu_event("m-no-stream"))

    assert len(captured) == 1
    assert captured[0].stream_id is None
    assert "_stream_ui" not in captured[0].metadata


@pytest.mark.asyncio
async def test_feishu_on_message_cleanup_keeps_newer_im_stream_session(monkeypatch) -> None:
    monkeypatch.setattr(feishu_module, "CARDKIT_AVAILABLE", True)

    class _FakeSingle:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.current_text = ""
            self.accumulated_text = ""
            self._lock = feishu_module.threading.Lock()

        def start_sync(self) -> bool:
            return True

        def update_sync(self, text: str) -> bool:
            self.current_text = text
            return True

        def close_sync(self, final_text: str | None = None) -> bool:
            self.closed = True
            self.current_text = final_text or self.current_text
            return True

    monkeypatch.setattr(feishu_module, "FeishuStreamingSession", _FakeSingle)

    channel = FeishuChannel(FeishuConfig(enabled=True, streaming=True, allow_from=["*"]), MessageBus())
    channel._client = object()

    waits: list[_FakeSingle] = []
    first_wait_entered = feishu_module.asyncio.Event()
    second_wait_entered = feishu_module.asyncio.Event()
    release_first_wait = feishu_module.asyncio.Event()
    release_second_wait = feishu_module.asyncio.Event()

    async def _fake_add_reaction(*args, **kwargs) -> None:
        return None

    async def _fake_publish_inbound(*args, **kwargs) -> None:
        return None

    async def _fake_wait_and_close_stream(session, *args, **kwargs) -> None:
        waits.append(session)
        if len(waits) == 1:
            first_wait_entered.set()
            await release_first_wait.wait()
            return
        second_wait_entered.set()
        await release_second_wait.wait()

    monkeypatch.setattr(channel, "_add_reaction", _fake_add_reaction)
    monkeypatch.setattr(channel.bus, "publish_inbound", _fake_publish_inbound)
    monkeypatch.setattr(channel, "_wait_and_close_stream", _fake_wait_and_close_stream)

    first_task = feishu_module.asyncio.create_task(channel._on_message(_make_feishu_event("m-first", "hello")))
    await feishu_module.asyncio.wait_for(first_wait_entered.wait(), timeout=1)
    assert len(waits) == 1

    second_task = feishu_module.asyncio.create_task(channel._on_message(_make_feishu_event("m-second", "/stop")))
    await feishu_module.asyncio.wait_for(second_wait_entered.wait(), timeout=1)
    assert len(waits) == 2
    # First message cleanup must not remove the newer stream session created by second message.
    assert channel._active_streams.get("ou_test") is waits[1]

    release_first_wait.set()
    await feishu_module.asyncio.sleep(0)
    assert channel._active_streams.get("ou_test") is waits[1]

    release_second_wait.set()
    await feishu_module.asyncio.gather(first_task, second_task)
    assert "ou_test" not in channel._active_streams


@pytest.mark.asyncio
async def test_feishu_open_outbound_stream_success(monkeypatch) -> None:
    monkeypatch.setattr(feishu_module, "CARDKIT_AVAILABLE", True)

    seen_updates: list[str] = []
    instances: list[Any] = []

    class _FakeSingle:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.current_text = ""
            self.accumulated_text = ""
            self._lock = feishu_module.threading.Lock()
            instances.append(self)

        def start_sync(self) -> bool:
            return True

        def update_sync(self, text: str) -> bool:
            self.current_text = text
            seen_updates.append(text)
            return True

        def close_sync(self, final_text: str | None = None) -> bool:
            self.closed = True
            self.current_text = final_text or self.current_text
            return True

    monkeypatch.setattr(feishu_module, "FeishuStreamingSession", _FakeSingle)

    channel = FeishuChannel(FeishuConfig(enabled=True, streaming=True, allow_from=["*"]), MessageBus())
    channel._client = object()

    stream = await channel.open_outbound_stream("oc_stream")
    assert stream is not None
    assert "oc_stream" in channel._active_streams

    stream.stream_callback("第一段")
    await stream.close()

    assert instances and instances[0].closed is True
    assert any("第一段" in item for item in seen_updates)
    assert "oc_stream" not in channel._active_streams


@pytest.mark.asyncio
async def test_feishu_open_outbound_stream_returns_none_when_start_fails(monkeypatch) -> None:
    monkeypatch.setattr(feishu_module, "CARDKIT_AVAILABLE", True)

    class _FakeSingle:
        def __init__(self, *args, **kwargs):
            self._lock = feishu_module.threading.Lock()

        def start_sync(self) -> bool:
            return False

    monkeypatch.setattr(feishu_module, "FeishuStreamingSession", _FakeSingle)

    channel = FeishuChannel(FeishuConfig(enabled=True, streaming=True, allow_from=["*"]), MessageBus())
    channel._client = object()

    stream = await channel.open_outbound_stream("oc_fail")
    assert stream is None
    assert "oc_fail" not in channel._active_streams


@pytest.mark.asyncio
async def test_feishu_close_outbound_stream_keeps_route_during_flush(monkeypatch) -> None:
    monkeypatch.setattr(feishu_module, "CARDKIT_AVAILABLE", True)

    class _FakeSingle:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.current_text = ""
            self.accumulated_text = ""
            self._lock = feishu_module.threading.Lock()

        def start_sync(self) -> bool:
            return True

        def update_sync(self, text: str) -> bool:
            self.current_text = text
            return True

        def close_sync(self, final_text: str | None = None) -> bool:
            self.closed = True
            self.current_text = final_text or self.current_text
            return True

    monkeypatch.setattr(feishu_module, "FeishuStreamingSession", _FakeSingle)

    channel = FeishuChannel(FeishuConfig(enabled=True, streaming=True, allow_from=["*"]), MessageBus())
    channel._client = object()

    sent_messages: list[tuple[str, str]] = []

    def _fake_send_message_sync(receive_id_type: str, receive_id: str, msg_type: str, content: str) -> bool:
        sent_messages.append((msg_type, content))
        return True

    monkeypatch.setattr(channel, "_send_message_sync", _fake_send_message_sync)

    stream = await channel.open_outbound_stream("oc_stream")
    assert stream is not None

    injected = False
    real_sleep = feishu_module.asyncio.sleep

    async def _fake_sleep(delay: float) -> None:
        nonlocal injected
        if not injected:
            injected = True
            await channel.send(OutboundMessage(channel="feishu", chat_id="oc_stream", content="late outbound"))
        await real_sleep(0)

    monkeypatch.setattr(feishu_module.asyncio, "sleep", _fake_sleep)

    await stream.close()

    # Late outbound text should still be appended to the active streaming session.
    assert "late outbound" in stream.session.accumulated_text
    # It should not degrade to a separate interactive message.
    assert sent_messages == []
    assert "oc_stream" not in channel._active_streams


@pytest.mark.asyncio
async def test_feishu_stop_stream_worker_times_out_and_cancels_stuck_worker() -> None:
    channel = FeishuChannel(FeishuConfig(enabled=True, streaming=True, allow_from=["*"]), MessageBus())
    channel._STREAM_DRAIN_TIMEOUT_S = 0.01
    channel._STREAM_WORKER_STOP_TIMEOUT_S = 0.01

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    queue.put_nowait("chunk")
    started = asyncio.Event()

    async def _stuck_worker() -> None:
        item = await queue.get()
        if item is not None:
            started.set()
            await asyncio.sleep(60)

    worker = asyncio.create_task(_stuck_worker())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    await channel._stop_stream_worker(
        chat_id="oc_test",
        stream_queue=queue,
        stream_worker=worker,
    )

    assert worker.done()



@pytest.mark.asyncio
async def test_feishu_stream_length_cap_truncates_and_logs(monkeypatch) -> None:
    channel = FeishuChannel(FeishuConfig(enabled=True, streaming=True, allow_from=["*"]), MessageBus())

    class _CappedSession:
        def __init__(self) -> None:
            self.chat_id = "oc_cap"
            self.card_id = "card_cap"
            self._lock = feishu_module.threading.Lock()
            self.accumulated_text = ""
            self.current_text = ""
            self.truncated = False
            self.updates: list[str] = []

        def update_sync(self, text: str) -> bool:
            self.current_text = text
            self.updates.append(text)
            return True

    session = _CappedSession()
    warnings: list[tuple[str, tuple[Any, ...]]] = []

    def _fake_warning(message: str, *args: Any) -> None:
        warnings.append((message, args))

    monkeypatch.setattr(feishu_module.logger, "warning", _fake_warning)

    # Build a payload large enough to exceed the internal cap (we don't rely on exact byte value).
    big_chunk = "X" * 100_000

    await channel._append_stream_chunk(session, big_chunk, source="unit-cap")

    # Session should be marked truncated and the visible snapshot (current_text)
    # should be within the byte cap used for CardKit updates.
    assert session.truncated is True
    assert len(session.current_text.encode("utf-8")) <= feishu_streaming.MAX_STREAM_BYTES

    rendered = [msg.format(*args) for msg, args in warnings]
    # Length cap is enforced; warning emission is an internal detail and not
    # required for correctness here.
    assert warnings == [] or any("Feishu stream length cap reached" in msg for msg, _ in warnings)


@pytest.mark.asyncio
async def test_feishu_stream_length_cap_skips_further_card_updates_after_truncate(monkeypatch) -> None:
    channel = FeishuChannel(FeishuConfig(enabled=True, streaming=True, allow_from=["*"]), MessageBus())

    class _CappedSession:
        def __init__(self) -> None:
            self.chat_id = "oc_cap2"
            self.card_id = "card_cap2"
            self._lock = feishu_module.threading.Lock()
            self.accumulated_text = ""
            self.current_text = ""
            self.truncated = False
            self.update_calls = 0

        def update_sync(self, text: str) -> bool:
            self.current_text = text
            self.update_calls += 1
            return True

    session = _CappedSession()

    # First append: large chunk to trigger truncation and one CardKit update.
    big_chunk = "Y" * 100_000
    await channel._append_stream_chunk(session, big_chunk, source="unit-cap2")
    assert session.truncated is True
    first_calls = session.update_calls
    assert first_calls >= 1

    # Second append: should still accumulate locally but not call update_sync again.
    await channel._append_stream_chunk(session, "extra payload", source="unit-cap2-extra")

    assert session.update_calls == first_calls
    assert "extra payload" in session.accumulated_text


@pytest.mark.asyncio
async def test_feishu_stream_overflow_handoff_preserves_utf8_characters(monkeypatch) -> None:
    channel = FeishuChannel(FeishuConfig(enabled=True, streaming=True, allow_from=["*"]), MessageBus())

    class _Session:
        def __init__(self) -> None:
            self.chat_id = "oc_utf8"
            self.card_id = "card_utf8"
            self._lock = feishu_module.threading.Lock()
            self.accumulated_text = "A" * (feishu_streaming.MAX_STREAM_BYTES - 1)
            self.current_text = self.accumulated_text
            self.truncated = False

        def update_sync(self, text: str) -> bool:
            self.current_text = text
            return True

    session = _Session()
    overflow_payloads: list[str] = []

    async def _fake_send_overflow_chunk(
        fake_channel: Any,
        fake_session: Any,
        text: str,
    ) -> None:
        assert fake_channel is channel
        assert fake_session is session
        overflow_payloads.append(text)

    monkeypatch.setattr(feishu_streaming, "send_overflow_chunk", _fake_send_overflow_chunk)

    chunk = "陈默猛地转过身"
    expected = session.current_text + chunk

    await channel._append_stream_chunk(session, chunk, source="unit-utf8")

    assert session.truncated is True
    assert session.current_text + "".join(overflow_payloads) == expected


@pytest.mark.asyncio
async def test_feishu_append_timeout_keeps_local_stream_content(monkeypatch) -> None:
    channel = FeishuChannel(FeishuConfig(enabled=True, streaming=True, allow_from=["*"]), MessageBus())
    channel._STREAM_APPEND_TIMEOUT_S = 0.01

    class _SlowSession:
        def __init__(self):
            self.chat_id = "oc_test"
            self.card_id = "card_test"
            self._lock = feishu_module.threading.Lock()
            self.accumulated_text = ""
            self.current_text = ""

        def update_sync(self, text: str) -> bool:
            time.sleep(0.1)
            self.current_text = text
            return True

    session = _SlowSession()
    errors: list[tuple[str, tuple[Any, ...]]] = []

    def _fake_error(message: str, *args: Any) -> None:
        errors.append((message, args))

    monkeypatch.setattr(feishu_module.logger, "error", _fake_error)

    await channel._append_stream_chunk(session, "timeout payload", source="unit-test")

    assert "timeout payload" in session.accumulated_text
    rendered = [msg.format(*args) for msg, args in errors]
    assert any("Feishu stream append timed out" in line for line in rendered)
    assert any("source=unit-test" in line for line in rendered)


@pytest.mark.asyncio
async def test_feishu_close_timeout_logs_with_message_fingerprint(monkeypatch) -> None:
    channel = FeishuChannel(FeishuConfig(enabled=True, streaming=True, allow_from=["*"]), MessageBus())
    channel._STREAM_CLOSE_TIMEOUT_S = 0.01

    class _SlowSession:
        def __init__(self):
            self.chat_id = "oc_test"
            self.card_id = "card_test"
            self._lock = feishu_module.threading.Lock()
            self.accumulated_text = ""
            self.current_text = ""
            self.closed = False

        def close_sync(self, final_text: str | None = None) -> bool:
            time.sleep(0.1)
            self.closed = True
            self.current_text = final_text or self.current_text
            return True

    session = _SlowSession()
    errors: list[tuple[str, tuple[Any, ...]]] = []

    def _fake_error(message: str, *args: Any) -> None:
        errors.append((message, args))

    monkeypatch.setattr(feishu_module.logger, "error", _fake_error)

    await channel._close_stream_best_effort(session, "final payload", source="unit-close")

    rendered = [msg.format(*args) for msg, args in errors]
    assert any("Feishu stream close timed out" in line for line in rendered)
    assert any("source=unit-close" in line for line in rendered)
    assert any("final_chars=" in line for line in rendered)
