# Feishu Streaming UI Config Guide

This document describes how Feishu streaming output is controlled in the current codebase.

## Scope

These settings affect Feishu streaming presentation and progress visibility:

- `channels.send_progress`
- `channels.send_tool_hints`
- `channels.feishu.streaming`
- `channels.feishu.stream_ui` (`sections` or `plain`)

## Default Behavior (Current Schema)

Defaults from `nanobot/config/schema.py`:

- `channels.send_progress = true`
- `channels.send_tool_hints = true`
- `channels.feishu.streaming = true`
- `channels.feishu.stream_ui = "sections"`

With default settings, Feishu uses sectioned streaming UI:

- `🧠 Reasoning Log`
- `🔧 Tool Execution` details (when tool hints are enabled)
- `📌 Final Output`

## What Each Setting Controls

### `channels.send_progress`

Global switch for non-tool progress messages (`_progress=true` and `_tool_hint=false`).

- `true`: allow regular progress messages
- `false`: drop regular progress messages

Note: this does not directly control Feishu section headers in `sections` mode.

### `channels.send_tool_hints`

Global switch for tool hint progress messages (`_progress=true` and `_tool_hint=true`).

- `true`: allow tool hint messages
- `false`: drop tool hint messages

In Feishu `sections` mode, this also controls whether detailed tool blocks are rendered:

- `true`: show `🔧 Tool / Params / Summary`
- `false`: hide tool detail blocks

### `channels.feishu.streaming`

Feishu CardKit streaming on/off.

- `true`: use streaming card session when available
- `false`: send regular (non-streaming) responses

### `channels.feishu.stream_ui`

Feishu streaming presentation mode.

- `sections`: structured sections (`Reasoning`, optional `Tool`, `Final`)
- `plain`: raw streaming text only (no section headings)

## Mode Comparison

### `stream_ui = sections`

- Uses sectioned UI (`feishu_chat_sections`)
- Can show reasoning and final section headers
- Tool detail visibility depends on `send_tool_hints`

### `stream_ui = plain`

- No section UI metadata
- Streams plain text output only
- No `Reasoning Log` or `Final Output` section headers

## Recommended Presets

### 1) Keep default rich sections

```json
{
  "channels": {
    "send_progress": true,
    "send_tool_hints": true,
    "feishu": {
      "streaming": true,
      "stream_ui": "sections"
    }
  }
}
```

### 2) Plain streaming only (no reasoning/tool/final sections)

```json
{
  "channels": {
    "send_progress": true,
    "send_tool_hints": false,
    "feishu": {
      "streaming": true,
      "stream_ui": "plain"
    }
  }
}
```

## Notes

- `sections`/`plain` is a Feishu-specific display mode selector.
- `send_progress` and `send_tool_hints` are global channel-level switches shared by all channels.
