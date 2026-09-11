"""A scripted GPT-Live socket for the engine tests.

The engine sends plain event dicts and iterates the connection for server
events; the fake records what was sent, answers the handshake and the
acknowledgments the way the server does, and lets a test push the rest at
the moments it chooses: her voice as loud audio deltas with word-level
transcript fragments, his words as fragments, backend delegations with
function calls, usage, and the close.
"""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np

from tests.fake_realtime import InstantSpeaker

RATE = 24_000


def loud_pcm(ms: int, amplitude: int = 2000) -> bytes:
    """Her voice: an alternating ±amplitude signal (RMS = amplitude)."""
    n = RATE * ms // 1000
    return ((np.arange(n) % 2 * 2 - 1) * amplitude).astype(np.int16).tobytes()


def silent_pcm(ms: int) -> bytes:
    return b"\x00\x00" * (RATE * ms // 1000)


class FakeLiveConnection:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self.session_id = "live_fake"
        self.reject_start: dict[str, str] | None = None  # {"message", "code", "param"} instead of started
        self.seconds = 0.0  # what session.closed reports
        self.on_response_create: Any = None  # called with the connection after each response.create

    # ── what the engine sends ──────────────────────────────────────────────

    async def send(self, event: dict[str, Any]) -> None:
        self.sent.append(event)
        kind = event["type"]
        if kind == "session.start":
            if self.reject_start is not None:
                rejection, self.reject_start = self.reject_start, None  # once: the retry gets in
                self.push("error", error=SimpleNamespace(type="invalid_request_error", **rejection))
            else:
                self.push("session.started", session=SimpleNamespace(id=self.session_id, expires_at=1_800_000_000))
        elif kind == "session.update":
            self.push("session.updated", session=SimpleNamespace(id=self.session_id))
        elif kind in ("session.instructions.append", "session.commentary.append", "session.thinking.append"):
            self.push(kind.replace(".append", ".appended"), start_ms=0, end_ms=0, client_event_id=None)
        elif kind == "session.input_audio.mute":
            self.push("session.input_audio.muted")
        elif kind == "session.input_audio.unmute":
            self.push("session.input_audio.unmuted")
        elif kind == "session.close":
            self.close("close_requested")
        elif kind == "response.create" and self.on_response_create is not None:
            self.on_response_create(self)

    # ── what the test pushes ───────────────────────────────────────────────

    def push(self, event_type: str, **fields: Any) -> None:
        self._events.put_nowait(SimpleNamespace(type=event_type, **fields))

    def audio(self, pcm: bytes) -> None:
        self.push("session.output_audio.delta", delta=base64.b64encode(pcm).decode("ascii"))

    def she_speaks(self, ms: int, transcript: str = "", *, start_ms: int | None = None, chunk_ms: int = 100) -> int:
        """`ms` of her voice in 100 ms deltas, with the transcript as word
        fragments on the 200 ms grid from `start_ms`. Returns the end stamp."""
        start = self._stamp if start_ms is None else start_ms
        words = transcript.split()
        for i, word in enumerate(words):
            piece = word if i == 0 else " " + word
            self.push("session.output_transcript.delta", delta=piece, start_ms=start + i * 200, end_ms=start + (i + 1) * 200)
        for _ in range(max(ms // chunk_ms, 1)):
            self.audio(loud_pcm(chunk_ms))
        self._stamp = start + max(ms, len(words) * 200)
        return self._stamp

    def silence(self, ms: int, chunk_ms: int = 100) -> None:
        """The stream between her words."""
        for _ in range(max(ms // chunk_ms, 1)):
            self.audio(silent_pcm(chunk_ms))
        self._stamp += ms

    _stamp = 0  # the session timeline the scripted fragments run on

    def owner_says(self, text: str, *, start_ms: int | None = None, step: int = 200) -> int:
        start = self._stamp if start_ms is None else start_ms
        for i, word in enumerate(text.split()):
            piece = word if i == 0 else " " + word
            self.push("session.input_transcript.delta", delta=piece, start_ms=start + i * step, end_ms=start + (i + 1) * step)
        self._stamp = start + len(text.split()) * step
        return self._stamp

    def backend(
        self,
        delegation_id: str = "d1",
        *,
        calls: list[tuple[str, str, dict[str, Any]]] | None = None,
        text: str = "",
        usage: dict[str, Any] | None = None,
        created: bool = True,
        response_id: str = "resp_1",
    ) -> None:
        """A delegation and its backend response: created, any function
        calls as done output items, then completed with usage."""
        if created:
            self.push(
                "session.delegation.created", offset_ms=self._stamp,
                delegation=SimpleNamespace(id=delegation_id, target="responses", type="delegation", response_id=response_id),
            )
        self.push("response.event", delegation_id=delegation_id, event={"type": "response.created", "response": {"id": response_id}})
        if text:
            self.push("response.event", delegation_id=delegation_id, event={"type": "response.output_text.delta", "delta": text})
        for call_id, name, args in calls or []:
            self.push(
                "response.event", delegation_id=delegation_id,
                event={"type": "response.output_item.done", "item": {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args)}},
            )
        self.push(
            "response.event", delegation_id=delegation_id,
            event={"type": "response.completed", "response": {"id": response_id, "usage": usage or {"input_tokens": 1000, "output_tokens": 50, "input_tokens_details": {"cached_tokens": 800}}}},
        )

    def usage(self, seconds: float, ratio: float | None = None) -> None:
        self.seconds = seconds
        self.push(
            "session.usage.updated", usage=SimpleNamespace(seconds=seconds),
            context_window=None if ratio is None else SimpleNamespace(usage_ratio=ratio),
        )

    def close(self, reason: str = "close_requested") -> None:
        self.push("session.closed", reason=reason, usage=SimpleNamespace(seconds=self.seconds), session=SimpleNamespace(id=self.session_id))

    def fail_now(self, error: BaseException | None = None) -> None:
        self._events.put_nowait(error or ConnectionError("socket closed"))

    async def __aiter__(self):
        while True:
            event = await self._events.get()
            if isinstance(event, BaseException):
                raise event
            yield event

    # ── reading what was sent ──────────────────────────────────────────────

    def kinds(self) -> list[str]:
        return [e["type"] for e in self.sent]

    def tool_outputs(self) -> list[dict[str, Any]]:
        return [json.loads(e["item"]["output"]) for e in self.sent if e["type"] == "response.item.create"]

    def instructions(self) -> list[str]:
        return [e["content"] for e in self.sent if e["type"] == "session.instructions.append"]

    def commentary(self) -> list[str]:
        return [e["content"] for e in self.sent if e["type"] == "session.commentary.append"]

    def audio_frames(self) -> list[bytes]:
        return [base64.b64decode(e["audio"]) for e in self.sent if e["type"] == "session.input_audio.append"]

    def start_config(self) -> dict[str, Any]:
        return next(e["session"] for e in self.sent if e["type"] == "session.start")


class FakeLiveClient:
    def __init__(self) -> None:
        self.connection = FakeLiveConnection()
        self.live = self
        self.connects = 0

    @asynccontextmanager
    async def connect(self):
        self.connects += 1
        yield self.connection


class LevelSpeaker(InstantSpeaker):
    """An instant speaker that can say how loud she is playing."""

    def __init__(self, level: float = 0.0, **kw: Any) -> None:
        super().__init__(**kw)
        self.level = level

    def played_level(self, window_s: float = 0.3) -> float:
        return self.level
