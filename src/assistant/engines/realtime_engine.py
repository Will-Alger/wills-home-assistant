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
from assistant.home.base import HomeApi, device_table, media_table
from assistant.memory import MemoryStore

REALTIME_RATE = 24_000
FRAME_SAMPLES_24K = 1920  # 80 ms
FALLBACK_VOICE = "marin"  # used automatically when the configured voice is gated

# $/1M tokens, gpt-realtime-2.1 (pricing page, 2026-08; approximate meter —
# cached tokens all billed at the cached-audio rate for simplicity).
_PRICE_AUDIO_IN, _PRICE_AUDIO_OUT = 32.0, 64.0
_PRICE_TEXT_IN, _PRICE_TEXT_OUT = 4.0, 24.0
_PRICE_CACHED = 0.40

_INSTRUCTIONS = """\
You are {name}, the voice assistant in {owner}'s home. Anyone in the room may \
talk to you; {owner} is the household owner. You are as much good company as \
you are a home controller: chat, opinions, and thinking out loud are first-\
class, not just commands. Default to BRIEF: commands get a few words ("Done." \
"Hallway's dimmed.") — never a room-by-room recap, never unsolicited \
follow-up suggestions. In conversation, match the speaker's energy but stay \
compact: a sentence or two unless asked to go deeper.

You control the home through tools. No canned routines: interpret intent and \
decide. Prefer area targets, and batch every lighting change into ONE \
set_lights call. Music: play_music takes plain names (playlist/artist/track) \
— for open-ended asks ("something chill") pick a fitting artist or track and \
set radio_mode; starting can take a few seconds, so don't declare failure \
hastily. If playback fails or the speakers' TV is off, media_control turn_on \
the TV first, then retry once. The TV can open apps via launch_app. The home holds \
MORE than the lights and media listed below — thermostats, switches, scenes, \
sensors, weather: discover with search_entities, read with get_entity, act \
via ha_call_service (the escape hatch — prefer the dedicated tools whenever \
one fits). If something is truly beyond your tools, say so honestly.

Lights:
{devices}

Media players:
{media}

Standing preferences ({owner}'s, apply them automatically, no announcement):
{preferences}

Memory: when the speaker states a durable preference ("from now on…", \
"I always want…", "call me…"), store it with remember(kind="preference"). \
If a new preference updates or contradicts a stored one, forget the old id \
first and store the new — never keep both versions. \
Things they ask you to keep for later go in remember(kind="fact"); answer \
"what do you remember?" via list_memories, and delete with forget after \
checking ids. Store only what the speaker deliberately tells you — never \
ambient chatter. You cannot yet react to events ("when the sun sets…") — \
only to what is said to you; say so honestly if asked.

Ending: when the interaction is clearly over — the speaker used a wrap-up \
phrase ("that's all", "thanks, that's it", "never mind"), or a one-shot \
command finished and invites nothing more — say a brief closing word, then \
call end_conversation. During a flowing conversation, never call \
end_conversation: only the speaker ends a live conversation. Never say the \
phrase "{wake_phrase}".
{extra}"""

MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "remember",
        "description": (
            "Store a lasting memory. kind='preference' for standing "
            "instructions applied automatically in every future conversation; "
            "kind='fact' for things to recall later on request."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["preference", "fact"]},
                "text": {"type": "string", "description": "one self-contained sentence"},
            },
            "required": ["kind", "text"],
        },
    },
    {
        "type": "function",
        "name": "list_memories",
        "description": "List all stored preferences and facts with their ids.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "forget",
        "description": "Delete one stored memory by id (see list_memories first).",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
    },
]
_MEMORY_TOOL_NAMES = {tool["name"] for tool in MEMORY_TOOLS}

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
        talk_over: bool = False,
        eagerness: str = "high",
        extra_instructions: str = "",
        memory: MemoryStore | None = None,
        usage_log: Path | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._voice = voice  # may be swapped to FALLBACK_VOICE during _configure
        self.voice_note: str | None = None
        self._home = home
        self._executor = ToolExecutor(home)
        self._owner = owner
        self._name = name
        self._wake_phrase = wake_phrase
        self._idle_timeout_s = idle_timeout_s
        self._talk_over = talk_over  # headphones only: mic streams during playback
        self._eagerness = eagerness  # semantic VAD: how fast it decides you're done
        self._extra_instructions = extra_instructions
        self._memory = memory
        self._instructions_stale = False  # a preference changed mid-session
        self._transcription_model: str | None = None  # what _configure settled on
        self._usage_log = usage_log

    async def _session_config(self, transcription_model: str | None) -> dict[str, Any]:
        extra = f"\n{self._extra_instructions}\n" if self._extra_instructions else ""
        instructions = _INSTRUCTIONS.format(
            name=self._name,
            owner=self._owner,
            wake_phrase=self._wake_phrase,
            devices=device_table(await self._home.get_lights()),
            media=media_table(await self._home.media_players()),
            preferences=(
                self._memory.preferences_text() if self._memory else "(memory not enabled)"
            ),
            extra=extra,
        )
        audio_in: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
            "turn_detection": {"type": "semantic_vad", "eagerness": self._eagerness},
        }
        if transcription_model:
            audio_in["transcription"] = {"model": transcription_model}
        tools = realtime_tools() + (MEMORY_TOOLS if self._memory else [])
        return {
            "type": "realtime",
            "instructions": instructions,
            "tools": tools,
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

    async def _configure(self, connection: Any, transcription: bool) -> None:
        """Send session config and wait for acceptance before any audio flows —
        otherwise a rejected update leaves a session running with no tools.
        Degrades gracefully: a gated voice falls back; transcription steps
        down streaming model -> whisper -> none.
        """
        # mini-transcribe streams deltas (live typing); whisper-1 is the
        # widely-available fallback (transcript arrives only at end of turn).
        transcribers: list[str | None] = (
            ["gpt-4o-mini-transcribe", "whisper-1", None] if transcription else [None]
        )
        for _attempt in range(6):
            await connection.send(
                {"type": "session.update", "session": await self._session_config(transcribers[0])}
            )
            resend = False
            while not resend:
                event = await connection.recv()
                kind = event.type
                if kind == "session.updated":
                    self._transcription_model = transcribers[0]
                    return
                if kind == "error":
                    message = str(getattr(event, "error", event)).lower()
                    if "voice" in message and self._voice != FALLBACK_VOICE:
                        self.voice_note = (
                            f"voice '{self._voice}' not available yet — using {FALLBACK_VOICE}"
                        )
                        self._voice = FALLBACK_VOICE
                        resend = True
                    elif "transcri" in message and len(transcribers) > 1:
                        transcribers.pop(0)
                        resend = True
                    else:
                        raise RuntimeError(f"Realtime session config rejected: {message}")
                # anything else (session.created, ...) is ignored during setup
        raise RuntimeError("Realtime session config could not be applied")

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
            if call_name in _MEMORY_TOOL_NAMES:
                result_text, is_error = self._execute_memory(call_name, args)
            else:
                result_text, is_error = await self._executor.execute(call_name, args)
            tool_hook = getattr(self, "_ui_tool_hook", None)
            if tool_hook is not None:
                tool_hook(call_name, result_text, is_error)
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
        if self._instructions_stale:
            # A preference changed: refresh instructions so it applies to the
            # rest of THIS conversation, not just future ones.
            self._instructions_stale = False
            await connection.send(
                {
                    "type": "session.update",
                    "session": await self._session_config(self._transcription_model),
                }
            )
        if outputs and not closing:
            await connection.send({"type": "response.create"})
        return closing

    def _execute_memory(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        if self._memory is None:
            return "memory is not enabled", True
        try:
            if name == "remember":
                item = self._memory.add(str(args.get("kind", "")), str(args.get("text", "")))
                if item.kind == "preference":
                    self._instructions_stale = True
                return f"stored (id {item.id})", False
            if name == "list_memories":
                items = self._memory.items()
                if not items:
                    return "nothing stored yet", False
                return json.dumps(
                    [
                        {"id": i.id, "kind": i.kind, "text": i.text, "since": i.created}
                        for i in items
                    ]
                ), False
            if name == "forget":
                item_id = int(args.get("id", -1))
                if self._memory.forget(item_id):
                    self._instructions_stale = True
                    return f"forgot id {item_id}", False
                return f"no memory with id {item_id}", True
            return f"unknown memory tool {name}", True
        except (ValueError, TypeError) as err:
            return f"memory error: {err}", True

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
            await self._configure(connection, transcription=False)
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
        last_activity = time.monotonic()
        ended = asyncio.Event()

        self._ui_tool_hook = getattr(ui, "tool", None)  # observability: show tool outcomes
        async with self._client.realtime.connect(model=self._model) as connection:
            await self._configure(connection, transcription=True)
            if self.voice_note:
                note = getattr(ui, "note", None)
                if note is not None:
                    note(self.voice_note)
                    self.voice_note = None

            async def pump_mic() -> None:
                nonlocal speaking
                while True:
                    frame = await mic.get_frame()
                    if speaking and not self._talk_over:
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

            pending: list[asyncio.Task] = []

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
                nonlocal speaking, response_active, closing, last_activity
                heard = ""  # live accumulation of the user's words
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
                    elif kind == "conversation.item.input_audio_transcription.delta":
                        heard += getattr(event, "delta", "") or ""
                        ui.user_partial(heard)
                    elif kind == "conversation.item.input_audio_transcription.completed":
                        heard = ""
                        ui.user_said(getattr(event, "transcript", ""))
                    elif kind == "input_audio_buffer.speech_started":
                        if self._talk_over and speaking:
                            # Talk-over interrupt: you spoke, it stops.
                            speaker.clear()
                            if response_active:
                                await connection.send({"type": "response.cancel"})
                            speaking = False
                            ui.interrupted()
                        ui.user_speaking()
                    elif kind == "response.done":
                        response_active = False
                        closing = await self._handle_response_done(connection, event, stats)
                        pending.append(asyncio.create_task(finish_playback(closing)))
                    elif kind == "error":
                        ui.error(str(getattr(event, "error", event)))

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
                for task in [*tasks, *pending]:
                    task.cancel()
                for task in [*tasks, *pending]:
                    # CancelledError is a BaseException, not Exception — it must
                    # be suppressed explicitly or teardown masquerades as Ctrl+C
                    # and kills the whole app (the "closed when I said that's
                    # all" bug).
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
        return stats
