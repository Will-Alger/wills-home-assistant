"""A scripted stand-in for the OpenAI Realtime connection.

`FakeClient.realtime.connect(model=...)` yields a `FakeConnection` that
records everything the engine sends and replies with scripted events. It
answers the session handshake automatically and, for each `response.create`
it receives, emits one audio delta and a `response.done` with no output —
enough to exercise the engine's session lifecycle without a network. A
scripted user turn is the real event order: speech_started, speech_stopped,
then the finished transcription.
"""

from __future__ import annotations

import asyncio
import base64
import json
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
            self._reply("Heads up: the build finished.")
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

    def _reply(
        self,
        transcript: str,
        *,
        calls: list[tuple[str, dict[str, Any]]] | None = None,
        audio: bool = True,
    ) -> None:
        """One response: created, (audio, transcript,) done — with any
        function calls listed on the done event, the way the server does it."""
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

    async def wait_idle(self, tail_s: float = 0.0) -> None:
        await asyncio.sleep(self._drain_s)


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
