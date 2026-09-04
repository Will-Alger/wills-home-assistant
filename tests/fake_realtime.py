"""A scripted stand-in for the OpenAI Realtime connection.

`FakeClient.realtime.connect(model=...)` yields a `FakeConnection` that
records everything the engine sends and replies with scripted events. It
answers the session handshake automatically and, for each `response.create`
it receives, emits one audio delta and a `response.done` with no output —
enough to exercise the engine's session lifecycle without a network. A
scripted user turn is the real event order: speech_started, speech_stopped,
then the finished transcription. `hold_response` keeps a response open so a
test can interrupt her; the mics and speakers here stand in for the room.
"""

from __future__ import annotations

import asyncio
import base64
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
        # When set, a response starts but never finishes until the test calls
        # finish_response() — the only way to hold the engine in "she is
        # speaking" long enough to interrupt her on purpose.
        self.hold_response = False

    async def send(self, event: dict[str, Any]) -> None:
        self.sent.append(event)
        if event["type"] == "session.update":
            self._events.put_nowait(SimpleNamespace(type="session.updated"))
        elif event["type"] == "response.create":
            self._events.put_nowait(SimpleNamespace(type="response.created"))
            self._events.put_nowait(
                SimpleNamespace(
                    type="response.output_audio.delta",
                    delta=base64.b64encode(b"\x00\x00" * 240).decode("ascii"),
                )
            )
            self._events.put_nowait(
                SimpleNamespace(
                    type="response.output_audio_transcript.done",
                    transcript="Heads up: the build finished.",
                )
            )
            if self.hold_response:
                return
            self.finish_response()
            if self.say_after_response and not self._said:
                self._said = True
                self._events.put_nowait(SimpleNamespace(type="input_audio_buffer.speech_started"))
                self._events.put_nowait(SimpleNamespace(type="input_audio_buffer.speech_stopped"))
                self._events.put_nowait(
                    SimpleNamespace(
                        type="conversation.item.input_audio_transcription.completed",
                        transcript=self.say_after_response,
                    )
                )

    def finish_response(self) -> None:
        """End the response the engine is waiting on (see hold_response)."""
        self._events.put_nowait(
            SimpleNamespace(type="response.done", response=SimpleNamespace(output=[], usage=None))
        )

    def user_says(self, text: str) -> None:
        """Script a user turn at a moment the test chooses (say_after_response
        fires one immediately instead, in the same batch as the reply)."""
        self._events.put_nowait(SimpleNamespace(type="input_audio_buffer.speech_started"))
        self._events.put_nowait(SimpleNamespace(type="input_audio_buffer.speech_stopped"))
        self._events.put_nowait(
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed", transcript=text
            )
        )

    async def recv(self) -> Any:
        return await self._events.get()

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
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def enqueue(self, pcm: bytes) -> None:
        self.chunks.append(pcm)

    def clear(self) -> None:
        self.chunks.clear()

    async def wait_idle(self, tail_s: float = 0.0) -> None:
        await asyncio.sleep(0)


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
