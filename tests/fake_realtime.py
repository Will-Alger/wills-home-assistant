"""A scripted stand-in for the OpenAI Realtime connection.

`FakeClient.realtime.connect(model=...)` yields a `FakeConnection` that
records everything the engine sends and replies with scripted events. It
answers the session handshake automatically and, for each `response.create`
it receives, emits one audio delta and a `response.done` with no output —
enough to exercise the engine's session lifecycle without a network.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        # When set, the "owner" says this right after the FIRST response ends
        # (a scripted user turn: speech_started + a finished transcription).
        self.say_after_response: str | None = None
        self._said = False

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
            self._events.put_nowait(
                SimpleNamespace(
                    type="response.done",
                    response=SimpleNamespace(output=[], usage=None),
                )
            )
            if self.say_after_response and not self._said:
                self._said = True
                self._events.put_nowait(SimpleNamespace(type="input_audio_buffer.speech_started"))
                self._events.put_nowait(
                    SimpleNamespace(
                        type="conversation.item.input_audio_transcription.completed",
                        transcript=self.say_after_response,
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

    async def get_frame(self) -> bytes:
        await asyncio.sleep(3600)
        return b""

    def drain(self) -> None:
        pass


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

    def note(self, message: str) -> None:
        self.notes.append(message)

    def listening(self) -> None: ...
    def user_speaking(self) -> None: ...
    def user_partial(self, heard: str) -> None: ...
    def user_said(self, transcript: str) -> None: ...
    def assistant_said(self, transcript: str) -> None: ...
    def interrupted(self) -> None: ...
    def tool(self, name: str, result: str, is_error: bool) -> None: ...
    def error(self, message: str) -> None: ...
