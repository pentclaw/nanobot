"""Streaming helpers for Feishu CardKit sessions.

This module isolates CardKit-specific streaming logic so ``feishu.py`` can stay
close to upstream structure while preserving local streaming behavior.
"""

import asyncio
import json
import re
import threading
import time
import uuid
from typing import Any

from loguru import logger

import importlib.util

FEISHU_AVAILABLE = importlib.util.find_spec("lark_oapi") is not None
CARDKIT_AVAILABLE = False

# Approximate safety cap for CardKit streaming content.
# Feishu card payloads are limited to ~30KB; we keep a margin.
MAX_STREAM_BYTES = 24 * 1024

if FEISHU_AVAILABLE:
    try:
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody
    except ImportError:
        FEISHU_AVAILABLE = False
    else:
        try:
            from lark_oapi.api.cardkit.v1 import (
                ContentCardElementRequest,
                ContentCardElementRequestBody,
                CreateCardRequest,
                CreateCardRequestBody,
                SettingsCardRequest,
                SettingsCardRequestBody,
            )
            CARDKIT_AVAILABLE = True
        except ImportError:
            CARDKIT_AVAILABLE = False

_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _sanitize_stream_text(text: str) -> str:
    """Strip terminal control sequences that can break card rendering."""
    if not text:
        return ""
    cleaned = _ANSI_ESCAPE_RE.sub("", text)
    cleaned = _CONTROL_CHAR_RE.sub("", cleaned)
    return cleaned


def _normalize_stream_markdown(text: str) -> str:
    """Normalize streamed markdown to reduce Feishu rendering glitches."""
    if not text:
        return ""
    normalized = _sanitize_stream_text(text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(Tool:\s*[^\n]+)Params:", r"\1\nParams:", normalized)
    normalized = re.sub(r"(Params:\s*[^\n]+)Summary:", r"\1\nSummary:", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def split_utf8_text_for_byte_limit(text: str, byte_limit: int) -> tuple[str, str]:
    """Split text at a valid UTF-8 boundary within the given byte limit."""
    if not text or byte_limit <= 0:
        return "", text

    encoded = text.encode("utf-8")
    if len(encoded) <= byte_limit:
        return text, ""

    cut = min(byte_limit, len(encoded))
    while cut > 0:
        try:
            prefix = encoded[:cut].decode("utf-8")
            remainder = encoded[cut:].decode("utf-8")
            return prefix, remainder
        except UnicodeDecodeError:
            cut -= 1

    return "", text


class FeishuStreamingSession:
    """Manages a streaming card session using CardKit streaming API."""

    ELEMENT_ID = "streaming_content"

    def __init__(self, client: Any, chat_id: str, receive_id_type: str, title: str | None = None):
        self.client = client
        self.chat_id = chat_id
        self.receive_id_type = receive_id_type
        self.title = title
        self.card_id: str | None = None
        self.accumulated_text = ""
        self.current_text = ""
        self.closed = False
        self.last_update_time = 0.0
        self.pending_text: str | None = None
        self._sequence = 0
        self._lock = threading.Lock()
        # Mark when this session has hit the safe length cap; further updates
        # will still accumulate locally but stop updating CardKit.
        self.truncated = False

    def _render_stream_text(self, text: str) -> str:
        body = _normalize_stream_markdown(text or "")
        if self.title:
            return f"**{self.title}**\n\n{body}" if body else f"**{self.title}**"
        return body or "..."

    def _build_streaming_card_json(self) -> str:
        """Build Card JSON 2.0 with streaming mode enabled."""
        card = {
            "schema": "2.0",
            "config": {
                "streaming_mode": True,
                "summary": {"content": self.title or "[生成中...]"},
                "streaming_config": {
                    "print_frequency_ms": {"default": 50},
                    "print_step": {"default": 2},
                    "print_strategy": "fast",
                },
            },
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "content": self._render_stream_text("..."),
                        "element_id": self.ELEMENT_ID,
                    }
                ]
            },
        }
        return json.dumps(card, ensure_ascii=False)

    def start_sync(self) -> bool:
        """Create card entity and send it."""
        if self.card_id or not CARDKIT_AVAILABLE:
            return bool(self.card_id)
        try:
            create_req = CreateCardRequest.builder().request_body(
                CreateCardRequestBody.builder().type("card_json").data(self._build_streaming_card_json()).build()
            ).build()
            create_resp = self.client.cardkit.v1.card.create(create_req)
            if not create_resp.success():
                logger.warning("Failed to create card: {}", create_resp.msg)
                return False
            self.card_id = create_resp.data.card_id

            msg_content = json.dumps({"type": "card", "data": {"card_id": self.card_id}}, ensure_ascii=False)
            send_req = CreateMessageRequest.builder().receive_id_type(self.receive_id_type).request_body(
                CreateMessageRequestBody.builder()
                .receive_id(self.chat_id)
                .msg_type("interactive")
                .content(msg_content)
                .build()
            ).build()
            send_resp = self.client.im.v1.message.create(send_req)
            return send_resp.success()
        except Exception as e:
            logger.error("Error starting streaming session: {}", e)
            return False

    def update_sync(self, text: str) -> bool:
        """Update card content with throttling."""
        if self.closed or not self.card_id:
            return False
        with self._lock:
            now = time.time() * 1000
            if now - self.last_update_time < 100:
                self.pending_text = text
                return True
            self.pending_text = text
            self._sequence += 1
            seq = self._sequence
            self.current_text = text
            self.last_update_time = now
        try:
            req = ContentCardElementRequest.builder().card_id(self.card_id).element_id(self.ELEMENT_ID).request_body(
                ContentCardElementRequestBody.builder()
                .content(self._render_stream_text(text))
                .uuid(str(uuid.uuid4()))
                .sequence(seq)
                .build()
            ).build()
            return self.client.cardkit.v1.card_element.content(req).success()
        except Exception:
            return False

    def close_sync(self, final_text: str | None = None) -> bool:
        """Close streaming mode and finalize card."""
        if self.closed:
            return True
        self.closed = True
        if not self.card_id:
            return False
        text = final_text or self.pending_text or self.current_text or "No updates."
        try:
            with self._lock:
                self._sequence += 1
                seq = self._sequence
            content_req = ContentCardElementRequest.builder().card_id(self.card_id).element_id(self.ELEMENT_ID).request_body(
                ContentCardElementRequestBody.builder()
                .content(self._render_stream_text(text))
                .uuid(str(uuid.uuid4()))
                .sequence(seq)
                .build()
            ).build()
            self.client.cardkit.v1.card_element.content(content_req)
            settings = {"config": {"streaming_mode": False, "summary": {"content": ""}}}
            settings_req = SettingsCardRequest.builder().card_id(self.card_id).request_body(
                SettingsCardRequestBody.builder()
                .settings(json.dumps(settings, ensure_ascii=False))
                .uuid(str(uuid.uuid4()))
                .sequence(seq + 1)
                .build()
            ).build()
            return self.client.cardkit.v1.card.settings(settings_req).success()
        except Exception as e:
            logger.error("Error closing streaming session: {}", e)
            return False


class FeishuOutboundStream:
    """Handle for a proactive Feishu streaming session."""

    def __init__(
        self,
        channel: Any,
        chat_id: str,
        session: FeishuStreamingSession,
        stream_queue: asyncio.Queue[str | None],
        stream_worker: asyncio.Task[Any],
        stream_callback,
    ):
        self._channel = channel
        self.chat_id = chat_id
        self.session = session
        self.stream_queue = stream_queue
        self.stream_worker = stream_worker
        self.stream_callback = stream_callback
        self._closed = False

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._channel._close_outbound_stream(self)


def stream_append(session: FeishuStreamingSession, text: str) -> str:
    """Append text to local stream buffer and return snapshot."""
    with session._lock:
        session.accumulated_text += text
        return session.accumulated_text


def remove_active_stream_if_current(
    active_streams: dict[str, FeishuStreamingSession], chat_id: str, session: FeishuStreamingSession
) -> None:
    """Remove stream mapping only if it still points to this session."""
    if active_streams.get(chat_id) is session:
        active_streams.pop(chat_id, None)


async def append_stream_chunk(
    session: FeishuStreamingSession,
    chunk: str,
    *,
    source: str,
    append_timeout_s: float,
) -> None:
    """Append one chunk and push update with timeout protection and length cap.

    We keep a safety margin via MAX_STREAM_BYTES for the CardKit payload. Once
    the cap is reached, we stop updating CardKit for this session while still
    allowing callers to accumulate the full logical output elsewhere.
    """
    clean_chunk = _sanitize_stream_text(str(chunk or ""))
    if not clean_chunk:
        return

    # If we've already hit the cap, skip CardKit updates entirely.
    if getattr(session, "truncated", False):
        return

    current = getattr(session, "accumulated_text", "")
    current_bytes = len(current.encode("utf-8"))
    chunk_bytes = clean_chunk.encode("utf-8")

    # If we're already at or above the cap, mark truncated and do not update.
    if current_bytes >= MAX_STREAM_BYTES:
        session.truncated = True
        logger.warning(
            "Feishu stream length cap reached: chat={}, card_id={}, source={}",
            getattr(session, "chat_id", "unknown"),
            getattr(session, "card_id", "unknown"),
            source,
        )
        return

    remaining = MAX_STREAM_BYTES - current_bytes

    # If the new chunk alone would exceed the cap, only take the prefix that fits.
    if len(chunk_bytes) > remaining:
        kept_text, _ = split_utf8_text_for_byte_limit(clean_chunk, remaining)
        snapshot = stream_append(session, kept_text)
        session.truncated = True
        logger.warning(
            "Feishu stream length cap reached: chat={}, card_id={}, source={}",
            getattr(session, "chat_id", "unknown"),
            getattr(session, "card_id", "unknown"),
            source,
        )
    else:
        snapshot = stream_append(session, clean_chunk)

    loop = asyncio.get_running_loop()
    try:
        ok = await asyncio.wait_for(
            asyncio.shield(loop.run_in_executor(None, session.update_sync, snapshot)),
            timeout=append_timeout_s,
        )
        if not ok:
            logger.error(
                "Feishu stream append failed: chat={}, card_id={}, source={}, chunk_chars={}",
                getattr(session, "chat_id", "unknown"),
                getattr(session, "card_id", "unknown"),
                source,
                len(clean_chunk),
            )
    except asyncio.TimeoutError:
        logger.error(
            "Feishu stream append timed out: chat={}, card_id={}, source={}, chunk_chars={}",
            getattr(session, "chat_id", "unknown"),
            getattr(session, "card_id", "unknown"),
            source,
            len(clean_chunk),
        )
    except Exception as e:
        logger.error(
            "Feishu stream append exception: chat={}, card_id={}, source={}, chunk_chars={}, err={}",
            getattr(session, "chat_id", "unknown"),
            getattr(session, "card_id", "unknown"),
            source,
            len(clean_chunk),
            e,
        )


async def send_overflow_chunk(channel: Any, session: FeishuStreamingSession, text: str) -> None:
    """Send overflow text as normal interactive cards and append locally."""
    overflow = _normalize_stream_markdown(text)
    stream_append(session, overflow)

    receive_id_type = getattr(session, "receive_id_type", None)
    chat_id = getattr(session, "chat_id", None)
    if not (getattr(channel, "_client", None) and receive_id_type and chat_id):
        return

    elements = channel._build_card_elements(overflow)
    loop = asyncio.get_running_loop()
    for chunk_elems in channel._split_elements_by_table_limit(elements):
        card = {"config": {"wide_screen_mode": True}, "elements": chunk_elems}
        await loop.run_in_executor(
            None,
            channel._send_message_sync,
            receive_id_type,
            chat_id,
            "interactive",
            json.dumps(card, ensure_ascii=False),
        )


async def append_stream_chunk_with_overflow(
    channel: Any,
    session: FeishuStreamingSession,
    chunk: str,
    *,
    source: str,
    append_timeout_s: float,
) -> None:
    """Append a chunk, routing overflow to normal messages after stream cap."""
    try:
        raw_text = str(chunk or "")
    except Exception:
        raw_text = ""
    if not raw_text:
        return

    # If already capped, route all subsequent content to overflow path.
    if getattr(session, "truncated", False):
        await send_overflow_chunk(channel, session, raw_text)
        return

    current = getattr(session, "accumulated_text", "")
    current_bytes = len(current.encode("utf-8"))
    remaining = MAX_STREAM_BYTES - current_bytes

    if remaining <= 0:
        session.truncated = True
        logger.warning(
            "Feishu stream length cap reached: chat={}, source={}",
            getattr(session, "chat_id", "unknown"),
            source,
        )
        await send_overflow_chunk(channel, session, raw_text)
        return

    chunk_bytes = raw_text.encode("utf-8")
    if len(chunk_bytes) <= remaining:
        await append_stream_chunk(
            session,
            raw_text,
            source=source,
            append_timeout_s=append_timeout_s,
        )
        if len(chunk_bytes) == remaining:
            session.truncated = True
        return

    prefix, remainder = split_utf8_text_for_byte_limit(raw_text, remaining)
    await append_stream_chunk(
        session,
        prefix,
        source=source,
        append_timeout_s=append_timeout_s,
    )
    session.truncated = True
    logger.warning(
        "Feishu stream length cap reached: chat={}, source={}",
        getattr(session, "chat_id", "unknown"),
        source,
    )

    if remainder:
        await send_overflow_chunk(channel, session, remainder)


async def close_stream_best_effort(
    session: FeishuStreamingSession,
    final_text: str,
    *,
    source: str,
    close_timeout_s: float,
) -> None:
    """Best-effort close without retries."""
    loop = asyncio.get_running_loop()
    final_chars = len(final_text or "")
    close_future = loop.run_in_executor(None, session.close_sync, final_text)
    try:
        ok = await asyncio.wait_for(asyncio.shield(close_future), timeout=close_timeout_s)
        if not ok:
            logger.error(
                "Feishu stream close failed: chat={}, card_id={}, source={}, final_chars={}",
                session.chat_id,
                session.card_id,
                source,
                final_chars,
            )
    except asyncio.TimeoutError:
        logger.error(
            "Feishu stream close timed out: chat={}, card_id={}, source={}, final_chars={}",
            session.chat_id,
            session.card_id,
            source,
            final_chars,
        )

        def _on_done(fut: asyncio.Future[Any]) -> None:
            try:
                ok_late = fut.result()
                if not ok_late:
                    logger.error(
                        "Feishu stream close late-failed: chat={}, card_id={}, source={}, final_chars={}",
                        session.chat_id,
                        session.card_id,
                        source,
                        final_chars,
                    )
            except Exception as late_err:
                logger.error(
                    "Feishu stream close late-exception: chat={}, card_id={}, source={}, final_chars={}, err={}",
                    session.chat_id,
                    session.card_id,
                    source,
                    final_chars,
                    late_err,
                )

        close_future.add_done_callback(_on_done)
    except Exception as e:
        logger.error(
            "Feishu stream close exception: chat={}, card_id={}, source={}, final_chars={}, err={}",
            session.chat_id,
            session.card_id,
            source,
            final_chars,
            e,
        )


async def stop_stream_worker(
    *,
    chat_id: str,
    stream_queue: asyncio.Queue[str | None],
    stream_worker: asyncio.Task[Any],
    drain_timeout_s: float,
    worker_stop_timeout_s: float,
) -> None:
    """Drain stream queue best-effort and stop worker without hanging forever."""
    try:
        await asyncio.wait_for(stream_queue.join(), timeout=drain_timeout_s)
    except asyncio.TimeoutError:
        logger.warning("Feishu stream queue drain timed out for chat {}", chat_id)

    stream_queue.put_nowait(None)
    try:
        await asyncio.wait_for(stream_worker, timeout=worker_stop_timeout_s)
    except asyncio.TimeoutError:
        logger.warning("Feishu stream worker stop timed out for chat {}", chat_id)
        stream_worker.cancel()
        await asyncio.gather(stream_worker, return_exceptions=True)
    except Exception:
        stream_worker.cancel()
        await asyncio.gather(stream_worker, return_exceptions=True)


async def open_outbound_stream(
    channel: Any,
    chat_id: str,
    title: str | None,
    *,
    cardkit_available: bool,
    session_cls: type[FeishuStreamingSession],
) -> FeishuOutboundStream | None:
    """Open a proactive streaming card session for outbound jobs (cron/heartbeat)."""
    if not channel._client or not channel.config.streaming or not cardkit_available:
        return None

    if chat_id in channel._active_streams:
        logger.warning("Feishu stream already active for chat {}", chat_id)
        return None

    receive_id_type = "chat_id" if chat_id.startswith("oc_") else "open_id"
    loop = asyncio.get_running_loop()
    session = session_cls(
        client=channel._client,
        chat_id=chat_id,
        receive_id_type=receive_id_type,
        title=title,
    )

    started = await loop.run_in_executor(None, session.start_sync)
    if not started:
        return None

    channel._active_streams[chat_id] = session
    stream_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _drain_stream_queue() -> None:
        while True:
            chunk = await stream_queue.get()
            try:
                if chunk is None:
                    return
                await append_stream_chunk(
                    session,
                    chunk,
                    source="outbound_stream",
                    append_timeout_s=channel._STREAM_APPEND_TIMEOUT_S,
                )
            finally:
                stream_queue.task_done()

    stream_worker = asyncio.create_task(_drain_stream_queue())

    def stream_callback(chunk: str) -> None:
        try:
            text = str(chunk or "")
            if text:
                stream_queue.put_nowait(text)
        except RuntimeError:
            pass

    return FeishuOutboundStream(
        channel=channel,
        chat_id=chat_id,
        session=session,
        stream_queue=stream_queue,
        stream_worker=stream_worker,
        stream_callback=stream_callback,
    )


async def close_outbound_stream(channel: Any, stream: FeishuOutboundStream) -> None:
    """Drain and close a proactive streaming session."""
    try:
        await stop_stream_worker(
            chat_id=stream.chat_id,
            stream_queue=stream.stream_queue,
            stream_worker=stream.stream_worker,
            drain_timeout_s=channel._STREAM_DRAIN_TIMEOUT_S,
            worker_stop_timeout_s=channel._STREAM_WORKER_STOP_TIMEOUT_S,
        )

        # Keep session routable briefly so late outbound sends in the same turn
        # still append to this stream card instead of becoming separate messages.
        await asyncio.sleep(2)

        if stream.session.closed:
            return
        final_text = getattr(stream.session, "accumulated_text", None) or stream.session.current_text
        await close_stream_best_effort(
            stream.session,
            final_text,
            source="outbound_close",
            close_timeout_s=channel._STREAM_CLOSE_TIMEOUT_S,
        )
    finally:
        remove_active_stream_if_current(channel._active_streams, stream.chat_id, stream.session)


async def wait_and_close_stream(
    channel: Any,
    session: FeishuStreamingSession,
    stream_id: str,
    *,
    stream_queue: asyncio.Queue[str | None] | None = None,
    stream_worker: asyncio.Task[Any] | None = None,
) -> None:
    """Wait for agent loop to finish, then close streaming session."""
    await channel.bus.wait_stream_done(stream_id, timeout=300)
    if stream_queue is not None and stream_worker is not None:
        await stop_stream_worker(
            chat_id=session.chat_id,
            stream_queue=stream_queue,
            stream_worker=stream_worker,
            drain_timeout_s=channel._STREAM_DRAIN_TIMEOUT_S,
            worker_stop_timeout_s=channel._STREAM_WORKER_STOP_TIMEOUT_S,
        )
    await asyncio.sleep(2)  # drain pending outbound (e.g. media from tool calls)
    if session.closed:
        return
    final_text = getattr(session, "accumulated_text", None) or session.current_text
    if getattr(session, "truncated", False):
        # When truncated, rely on the last visible snapshot instead of the full
        # accumulated logical text to keep the final card payload within cap.
        final_text = session.current_text
    await close_stream_best_effort(
        session,
        final_text,
        source=f"inbound_close:{stream_id}",
        close_timeout_s=channel._STREAM_CLOSE_TIMEOUT_S,
    )
