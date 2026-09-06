"""A scripted stand-in for the OpenAI Realtime connection.

`FakeClient.realtime.connect(model=...)` yields a `FakeConnection` that
records everything the engine sends and replies with scripted events. It
answers the session handshake automatically and, for each `response.create`
it receives, emits one audio delta and a `response.done` with no output —
enough to exercise the engine's session lifecycle without a network. A
scripted user turn is the real event order: speech_started, speech_stopped,
then the finished transcription. `hold_response` leaves a reply open — she is
still speaking — until the test calls `finish_response()`, which is how a
barge-in gets something to cut into.

Replay tests (tests/test_replay.py) want the timing to be theirs: set
`auto_reply = False` and script every event by hand with `push`,
`push_audio` and `push_response_done`, spacing them with real sleeps.
`InstantSpeaker` plays nothing at all; `DelayedSpeaker` plays in real time,
so what was HEARD and what was generated can finally disagree.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np

from assistant.engines.realtime_engine import FRAME_SAMPLES_24K, REALTIME_RATE


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        # When set, the "owner" says this right after the FIRST response ends
        # (a scripted user turn: speech_started + a finished transcription).
        self.say_after_response: str | None = None
        self._said = False
        # False: response.create is only recorded, never answered — the test
        # scripts every event itself, with its own delays between them.
        self.auto_reply = True
        # True: a response.cancel is answered with the cancelled response's
        # done event, as the server does (off by default: older scripts count
        # their done events by hand).
        self.ack_cancel = False
        # When set, a reply starts but never finishes until the test calls
        # finish_response() — the only way to hold the engine in "she is
        # speaking" long enough to interrupt her on purpose.
        self.hold_response = False

    async def send(self, event: dict[str, Any]) -> None:
        self.sent.append(event)
        if event["type"] == "session.update":
            self._events.put_nowait(SimpleNamespace(type="session.updated"))
        elif event["type"] == "response.cancel" and self.ack_cancel:
            # the server answers a cancel with a done event for the cancelled response
            self._events.put_nowait(
                SimpleNamespace(type="response.done", response=SimpleNamespace(output=[], usage=None))
            )
        elif event["type"] == "response.create" and self.auto_reply:
            self._reply("Heads up: the build finished.")
            if self.hold_response:
                return  # she is mid-sentence until the test says otherwise
            if self.say_after_response and not self._said:
                self._said = True
                self._events.put_nowait(SimpleNamespace(type="input_audio_buffer.speech_started"))
                self._events.put_nowait(SimpleNamespace(type="input_audio_buffer.speech_stopped"))
                self._events.put_nowait(SimpleNamespace(type="input_audio_buffer.committed"))
                self._events.put_nowait(
                    SimpleNamespace(
                        type="conversation.item.input_audio_transcription.completed",
                        transcript=self.say_after_response,
                    )
                )

    def _reply(
        self,
        transcript: str,
        *,
        calls: list[tuple[str, dict[str, Any]]] | None = None,
        audio: bool = True,
    ) -> None:
        """One response: created, (audio, transcript,) done — with any
        function calls listed on the done event, the way the server does it.
        Under `hold_response` the done event waits for `finish_response()`."""
        self._events.put_nowait(SimpleNamespace(type="response.created"))
        if audio:
            self._events.put_nowait(
                SimpleNamespace(
                    type="response.output_audio.delta",
                    item_id="item_1",
                    delta=base64.b64encode(b"\x00\x00" * 240).decode("ascii"),
                )
            )
        if transcript:
            self._events.put_nowait(
                SimpleNamespace(type="response.output_audio_transcript.done", transcript=transcript)
            )
        if self.hold_response:
            return
        self.finish_response(calls=calls)

    def finish_response(self, *, calls: list[tuple[str, dict[str, Any]]] | None = None) -> None:
        """End the reply the engine is waiting on (see `hold_response`)."""
        output = [
            SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args), call_id=f"call_{i}")
            for i, (name, args) in enumerate(calls or [])
        ]
        self._events.put_nowait(
            SimpleNamespace(type="response.done", response=SimpleNamespace(output=output, usage=None))
        )

    def user_says(
        self,
        text: str,
        *,
        reply: str | None = None,
        calls: list[tuple[str, dict[str, Any]]] | None = None,
        audio: bool = True,
    ) -> None:
        """Script a user turn at a moment the test chooses (say_after_response
        fires one immediately instead, in the same batch as the reply). With
        `reply`, the server answers it the way VAD-created responses do; `calls`
        puts function calls on that response, `audio=False` makes it a
        tool-only response with no speech."""
        self._events.put_nowait(SimpleNamespace(type="input_audio_buffer.speech_started"))
        self._events.put_nowait(SimpleNamespace(type="input_audio_buffer.speech_stopped"))
        self._events.put_nowait(SimpleNamespace(type="input_audio_buffer.committed"))  # the turn ended
        self._events.put_nowait(
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed", transcript=text
            )
        )
        if reply is not None or calls:
            self._reply(reply or "", calls=calls, audio=audio)

    fail_recv_after: int | None = None  # the socket dies on this recv (1-based)
    _recv_count = 0

    def push(self, event_type: str, **fields: Any) -> None:
        """One scripted event, at the moment the test chooses — so a test can
        put real delays between the steps of a turn and time them."""
        self._events.put_nowait(SimpleNamespace(type=event_type, **fields))

    def push_audio(self, ms: int, *, item_id: str = "item_1") -> None:
        """`ms` milliseconds of (silent) assistant audio in one delta — enough
        of it that a real-time speaker takes real time to play it out."""
        pcm = b"\x00\x00" * (24 * ms)  # 24 kHz mono int16: 24 samples a millisecond
        self.push(
            "response.output_audio.delta",
            item_id=item_id,
            delta=base64.b64encode(pcm).decode("ascii"),
        )

    def fail_now(self, error: BaseException | None = None) -> None:
        """The socket dies right here — reaching the receiver even when it is
        already parked in recv(), which is how a connection really drops."""
        self._events.put_nowait(error or ConnectionError("socket closed"))

    def push_response_done(self, *calls: tuple[str, str, dict[str, Any]]) -> None:
        """A finished response, optionally with function calls to execute:
        each is (call_id, name, arguments)."""
        output = [
            SimpleNamespace(
                type="function_call", call_id=call_id, name=name, arguments=json.dumps(args)
            )
            for call_id, name, args in calls
        ]
        self.push("response.done", response=SimpleNamespace(output=output, usage=None))

    async def recv(self) -> Any:
        self._recv_count += 1
        if self.fail_recv_after is not None and self._recv_count >= self.fail_recv_after:
            raise ConnectionError("socket closed")
        event = await self._events.get()
        if isinstance(event, BaseException):
            raise event
        return event

    def kinds(self) -> list[str]:
        return [e["type"] for e in self.sent]


class FakeClient:
    def __init__(self) -> None:
        self.connection = FakeConnection()
        self.realtime = self

    @asynccontextmanager
    async def connect(self, *, model: str):
        yield self.connection


class NeverMic:
    """A mic that never produces a frame (the announcement needs none)."""

    drained = 0

    async def get_frame(self) -> bytes:
        await asyncio.sleep(3600)
        return b""

    def drain(self) -> None:
        pass


class LoudMic:
    """A 24 kHz mic that never stops talking — every frame is well clear of
    the engine's silence floor, so a hold counts as speech."""

    def __init__(self, amplitude: int = 8000, interval_s: float = 0.01) -> None:
        t = np.arange(FRAME_SAMPLES_24K) / REALTIME_RATE
        self._frame = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.int16).tobytes()
        self._interval = interval_s
        self.drained = 0

    async def get_frame(self) -> bytes:
        await asyncio.sleep(self._interval)
        return self._frame

    def drain(self) -> None:
        self.drained += 1


class QuietRoomMic(LoudMic):
    """The same mic in an empty room: frames arrive, none of them is speech."""

    def __init__(self, interval_s: float = 0.01) -> None:
        super().__init__(amplitude=0, interval_s=interval_s)


class InstantSpeaker:
    def __init__(self, played_ms: int = 0, drain_s: float = 0.0) -> None:
        self.chunks: list[bytes] = []
        self.items: list[str] = []  # begin_item calls, in order
        self.current_item = ""
        self._played_ms = played_ms  # what played_ms() reports for any item
        self._drain_s = drain_s  # >0: pretend the audio takes this long to play out

    def enqueue(self, pcm: bytes) -> None:
        self.chunks.append(pcm)

    def begin_item(self, item_id: str) -> None:
        if item_id and item_id != self.current_item:
            self.current_item = item_id
            self.items.append(item_id)

    def played_ms(self, item_id: str = "") -> int:
        return self._played_ms

    def clear(self) -> None:
        self.chunks.clear()
        self._drain_s = 0.0  # a barge-in: nothing left to play out
        self.paused = False

    paused = False
    pauses = 0  # how often playback was held for a possible interruption
    resumes = 0

    def pause(self) -> None:
        self.paused = True
        self.pauses += 1

    def resume(self) -> None:
        self.paused = False
        self.resumes += 1

    async def wait_idle(self, tail_s: float = 0.0) -> None:
        # A paused speaker does not drain: only unpaused wall time counts.
        # (Measured, not assumed: on Windows a short asyncio.sleep can return
        # at once, so the loop must never count its own iterations.)
        loop = asyncio.get_running_loop()
        remaining = self._drain_s
        last = loop.time()
        while remaining > 0:
            await asyncio.sleep(0.01)
            now = loop.time()
            if not self.paused:
                remaining -= now - last
            last = now


BYTES_PER_SECOND = 48_000  # 24 kHz mono int16: one second of her voice


class DelayedSpeaker:
    """A speaker that plays in real time, so heard and generated can differ.

    Audio drains at `BYTES_PER_SECOND` (`scale` times faster when a test
    wants the same shape in less wall clock; the millisecond figures stay
    true to the 24 kHz stream either way). The accounting mirrors
    `assistant.audio.speaker.Speaker`: `begin_item` marks where an item's
    audio starts, `played_ms` reports only what has actually drained, and
    `clear()` — a barge-in — drops the unplayed tail and pulls the item's
    recorded end back to what was heard.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self._rate = BYTES_PER_SECOND * scale  # bytes of playout per real second
        self.chunks: list[bytes] = []
        self.items: list[str] = []  # begin_item calls, in order
        self.current_item = ""
        self.clears = 0  # barge-ins
        self._enqueued = 0
        self._played = 0.0  # bytes actually drained (fractional: no rounding drift)
        self._spans: dict[str, tuple[int, int]] = {}  # item id -> (start, end) bytes
        self._at = time.monotonic()

    def _drain(self) -> None:
        """Catch the playhead up with the clock (the audio callback's job)."""
        now = time.monotonic()
        if self._played < self._enqueued:
            self._played = min(float(self._enqueued), self._played + (now - self._at) * self._rate)
        self._at = now

    def enqueue(self, pcm: bytes) -> None:
        self._drain()
        self.chunks.append(pcm)
        self._enqueued += len(pcm)
        if self.current_item and pcm:
            start, _end = self._spans.get(self.current_item, (self._enqueued - len(pcm), 0))
            self._spans[self.current_item] = (start, self._enqueued)

    def begin_item(self, item_id: str) -> None:
        if not item_id or item_id == self.current_item:
            return
        self._drain()
        self.current_item = item_id
        self._spans.setdefault(item_id, (self._enqueued, self._enqueued))
        self.items.append(item_id)

    def played_ms(self, item_id: str = "") -> int:
        """Milliseconds of this item the room has heard (0 if it is unknown)."""
        self._drain()
        span = self._spans.get(item_id or self.current_item)
        if span is None:
            return 0
        start, end = span
        heard = min(max(self._played - start, 0.0), float(end - start))
        return int(heard * 1000 / BYTES_PER_SECOND)

    @property
    def pending_seconds(self) -> float:
        """Wall-clock time left before the queue runs dry."""
        self._drain()
        return max(self._enqueued - self._played, 0.0) / self._rate

    def clear(self) -> None:
        self._drain()
        self.clears += 1
        self._enqueued = int(self._played)
        span = self._spans.get(self.current_item)
        if span is not None:
            start, end = span
            self._spans[self.current_item] = (start, max(start, min(end, self._enqueued)))

    async def wait_idle(self, tail_s: float = 0.0) -> None:
        while self.pending_seconds > 0:
            # capped, so a barge-in's clear() is noticed at once and not one
            # whole reply later
            await asyncio.sleep(min(self.pending_seconds, 0.02))
        if tail_s:
            await asyncio.sleep(tail_s)


class QuietUi:
    def __init__(self) -> None:
        self.notes: list[str] = []
        self.interruptions = 0

    def note(self, message: str) -> None:
        self.notes.append(message)

    def listening(self) -> None: ...
    def user_speaking(self) -> None: ...
    def user_partial(self, heard: str) -> None: ...
    def user_said(self, transcript: str) -> None: ...
    def assistant_said(self, transcript: str) -> None: ...
    def interrupted(self) -> None:
        self.interruptions += 1

    def tool(self, name: str, result: str, is_error: bool) -> None: ...
    def error(self, message: str) -> None: ...
