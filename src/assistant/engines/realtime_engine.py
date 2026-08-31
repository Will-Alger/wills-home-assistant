"""OpenAI Realtime voice engine (gpt-realtime family) — M4 bake-off candidate.

One conversation = one WebSocket session, opened only after the local wake
word fires (idle listening costs nothing and stays private). The model does
speech natively — STT, reasoning, and TTS collapse into the session — while
our parts stay ours: wake gating, Home Assistant tools, the close-vs-stay-open
contract (an `end_conversation` tool + instructions), and cost metering.

Half-duplex on purpose: mic audio is NOT sent while assistant audio plays
(desktop speakers + mic = echo chaos without WebRTC-style AEC). Barge-in =
say the wake phrase mid-reply; it cancels the response locally and instantly.

Audio is PCM 16-bit mono at 24 kHz — the API supports only rate 24000
(verified against openai SDK types, 2026-08-31).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from scipy.signal import resample_poly

from assistant.brain.tools import TOOL_DEFINITIONS, ToolExecutor
from assistant.home.base import HomeApi, device_table

REALTIME_RATE = 24_000
FRAME_SAMPLES_24K = 1920  # 80 ms

# $/1M tokens, gpt-realtime-2.1 (pricing page, 2026-08; approximate meter —
# cached tokens all billed at the cached-audio rate for simplicity).
_PRICE_AUDIO_IN, _PRICE_AUDIO_OUT = 32.0, 64.0
_PRICE_TEXT_IN, _PRICE_TEXT_OUT = 4.0, 24.0
_PRICE_CACHED = 0.40

_INSTRUCTIONS = """\
You are {name}, the voice assistant in {owner}'s home. Anyone in the room may \
talk to you; {owner} is the household owner. You are as much good company as \
you are a home controller: chat, opinions, and thinking out loud are first-\
class, not just commands. Speak naturally and concisely — brief confirmations \
for commands, real conversation can breathe.

You control the home through tools. No canned routines: interpret intent and \
decide. Prefer area targets, and batch every lighting change into ONE \
set_lights call. If something is beyond your tools (music is not wired up \
yet), say so honestly.

Devices:
{devices}

Ending: when the interaction is clearly over — the speaker used a wrap-up \
phrase ("that's all", "thanks, that's it", "never mind"), or a one-shot \
command finished and invites nothing more — say a brief closing word, then \
call end_conversation. During a flowing conversation, never call \
end_conversation: only the speaker ends a live conversation. Never say the \
phrase "{wake_phrase}".
"""

_END_TOOL = {
    "type": "function",
    "name": "end_conversation",
    "description": (
        "Close this conversation and return the assistant to sleep. Call it "
        "after your brief goodbye when the interaction is clearly over. Never "
        "call it mid-conversation."
    ),
    "parameters": {"type": "object", "properties": {}},
}


def realtime_tools() -> list[dict[str, Any]]:
    """Our Anthropic-shaped tool defs, converted to Realtime's function shape."""
    converted = [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        }
        for tool in TOOL_DEFINITIONS
    ]
    return [*converted, _END_TOOL]


@dataclass
class SessionStats:
    responses: int = 0
    cost_usd: float = 0.0
    tool_calls: list[str] = field(default_factory=list)
    ended_by: str = "unknown"


def _usage_cost(usage: Any) -> float:
    def n(obj: Any, name: str) -> int:
        return int(getattr(obj, name, 0) or 0)

    in_det = getattr(usage, "input_token_details", None)
    out_det = getattr(usage, "output_token_details", None)
    cached = n(in_det, "cached_tokens")
    audio_in = max(n(in_det, "audio_tokens") - cached, 0)
    text_in = n(in_det, "text_tokens")
    return (
        audio_in * _PRICE_AUDIO_IN
        + text_in * _PRICE_TEXT_IN
        + cached * _PRICE_CACHED
        + n(out_det, "audio_tokens") * _PRICE_AUDIO_OUT
        + n(out_det, "text_tokens") * _PRICE_TEXT_OUT
    ) / 1_000_000


def downsample_24k_to_16k(frame_24k: bytes) -> bytes:
    """24 kHz int16 frame -> 16 kHz, for feeding the wake detector mid-session."""
    samples = np.frombuffer(frame_24k, dtype=np.int16).astype(np.float32)
    out = resample_poly(samples, up=2, down=3)
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


class RealtimeEngine:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice: str,
        home: HomeApi,
        owner: str,
        name: str = "Jarvis",
        wake_phrase: str = "hey jarvis",
        idle_timeout_s: float = 20.0,
        usage_log: Path | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._voice = voice
        self._home = home
        self._executor = ToolExecutor(home)
        self._owner = owner
        self._name = name
        self._wake_phrase = wake_phrase
        self._idle_timeout_s = idle_timeout_s
        self._usage_log = usage_log

    async def _session_config(self, transcription: bool) -> dict[str, Any]:
        instructions = _INSTRUCTIONS.format(
            name=self._name,
            owner=self._owner,
            wake_phrase=self._wake_phrase,
            devices=device_table(await self._home.get_lights()),
        )
        audio_in: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
            "turn_detection": {"type": "semantic_vad"},
        }
        if transcription:
            audio_in["transcription"] = {"model": "whisper-1"}
        return {
            "type": "realtime",
            "instructions": instructions,
            "tools": realtime_tools(),
            "tool_choice": "auto",
            "output_modalities": ["audio"],
            "audio": {
                "input": audio_in,
                "output": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                    "voice": self._voice,
                },
            },
        }

    async def _handle_response_done(self, connection: Any, event: Any, stats: SessionStats) -> bool:
        """Execute any function calls; returns True when end_conversation fired."""
        response = getattr(event, "response", None)
        usage = getattr(response, "usage", None)
        if usage is not None:
            cost = _usage_cost(usage)
            stats.cost_usd += cost
            self._log_usage(cost, usage)
        stats.responses += 1

        closing = False
        outputs: list[dict[str, Any]] = []
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", "") != "function_call":
                continue
            call_name = item.name
            stats.tool_calls.append(call_name)
            if call_name == "end_conversation":
                closing = True
                continue
            try:
                args = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result_text, is_error = await self._executor.execute(call_name, args)
            outputs.append(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({"error" if is_error else "result": result_text}),
                    },
                }
            )
        for message in outputs:
            await connection.send(message)
        if outputs and not closing:
            await connection.send({"type": "response.create"})
        return closing

    def _log_usage(self, cost: float, usage: Any) -> None:
        if self._usage_log is None:
            return
        in_det = getattr(usage, "input_token_details", None)
        out_det = getattr(usage, "output_token_details", None)
        entry = {
            "ts": time.time(),
            "engine": "realtime",
            "model": self._model,
            "cost_usd": round(cost, 6),
            "input_audio_tokens": getattr(in_det, "audio_tokens", 0),
            "input_text_tokens": getattr(in_det, "text_tokens", 0),
            "cached_tokens": getattr(in_det, "cached_tokens", 0),
            "output_audio_tokens": getattr(out_det, "audio_tokens", 0),
            "output_text_tokens": getattr(out_det, "text_tokens", 0),
        }
        with self._usage_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    # ── text probe: no audio devices, proves the whole engine ──────────────

    async def text_probe(self, text: str) -> tuple[str, bytes, SessionStats]:
        """Send one typed command; returns (assistant transcript, 24k pcm, stats)."""
        stats = SessionStats()
        transcript_parts: list[str] = []
        audio = bytearray()
        async with self._client.realtime.connect(model=self._model) as connection:
            await connection.send(
                {"type": "session.update", "session": await self._session_config(False)}
            )
            await connection.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
            await connection.send({"type": "response.create"})
            while True:
                event = await connection.recv()
                kind = event.type
                if kind.endswith("audio.delta") and "transcript" not in kind:
                    audio.extend(base64.b64decode(event.delta))
                elif kind.endswith("audio_transcript.delta"):
                    transcript_parts.append(event.delta)
                elif kind == "response.done":
                    closing = await self._handle_response_done(connection, event, stats)
                    output = getattr(event.response, "output", None) or []
                    had_calls = any(
                        getattr(item, "type", "") == "function_call"
                        for item in output
                        if getattr(item, "name", "") != "end_conversation"
                    )
                    if closing or not had_calls:
                        stats.ended_by = "end_conversation" if closing else "response complete"
                        break
                elif kind == "error":
                    raise RuntimeError(f"Realtime error: {getattr(event, 'error', event)}")
        return "".join(transcript_parts).strip(), bytes(audio), stats

    # ── live voice conversation ─────────────────────────────────────────────

    async def run_conversation(self, mic: Any, speaker: Any, wake: Any, ui: Any) -> SessionStats:
        """One wake-to-close conversation. `mic` must be a 24 kHz source."""
        stats = SessionStats()
        speaking = False
        response_active = False
        closing = False
        transcription = True
        last_activity = time.monotonic()
        ended = asyncio.Event()

        async with self._client.realtime.connect(model=self._model) as connection:
            await connection.send(
                {"type": "session.update", "session": await self._session_config(True)}
            )

            async def pump_mic() -> None:
                nonlocal speaking
                while True:
                    frame = await mic.get_frame()
                    if speaking:
                        # Half-duplex: don't feed our own voice back. But keep
                        # watching for the wake phrase = instant barge-in.
                        if wake is not None and wake.detect(downsample_24k_to_16k(frame)):
                            speaker.clear()
                            if response_active:
                                await connection.send({"type": "response.cancel"})
                            speaking = False
                            mic.drain()
                            ui.interrupted()
                        continue
                    await connection.send(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(frame).decode("ascii"),
                        }
                    )

            async def finish_playback(close_after: bool) -> None:
                nonlocal speaking
                await speaker.wait_idle()
                speaking = False
                mic.drain()
                if close_after:
                    ended.set()
                else:
                    ui.listening()

            async def receive() -> None:
                nonlocal speaking, response_active, closing, transcription, last_activity
                while True:
                    event = await connection.recv()
                    kind = event.type
                    last_activity = time.monotonic()
                    if kind.endswith("audio.delta") and "transcript" not in kind:
                        speaking = True
                        speaker.enqueue(base64.b64decode(event.delta))
                    elif kind == "response.created":
                        response_active = True
                    elif kind.endswith("audio_transcript.done"):
                        ui.assistant_said(getattr(event, "transcript", ""))
                    elif kind == "conversation.item.input_audio_transcription.completed":
                        ui.user_said(getattr(event, "transcript", ""))
                    elif kind == "input_audio_buffer.speech_started":
                        ui.user_speaking()
                    elif kind == "response.done":
                        response_active = False
                        closing = await self._handle_response_done(connection, event, stats)
                        asyncio.create_task(finish_playback(closing))
                    elif kind == "error":
                        message = str(getattr(event, "error", event))
                        if transcription and "transcription" in message.lower():
                            # Transcription config rejected: retry without it.
                            transcription = False
                            await connection.send(
                                {
                                    "type": "session.update",
                                    "session": await self._session_config(False),
                                }
                            )
                        else:
                            ui.error(message)

            async def idle_watchdog() -> None:
                while True:
                    await asyncio.sleep(1.0)
                    quiet = time.monotonic() - last_activity
                    if not speaking and not response_active and quiet > self._idle_timeout_s:
                        stats.ended_by = "idle timeout"
                        ended.set()
                        return

            tasks = [
                asyncio.create_task(pump_mic()),
                asyncio.create_task(receive()),
                asyncio.create_task(idle_watchdog()),
            ]
            try:
                await ended.wait()
                if stats.ended_by == "unknown":
                    stats.ended_by = "end_conversation"
            finally:
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with contextlib.suppress(Exception):
                        await task
        return stats
