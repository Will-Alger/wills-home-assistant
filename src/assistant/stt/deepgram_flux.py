"""Deepgram Flux adapter (listen v2 websocket, deepgram-sdk >= 7).

Flux does semantic end-of-turn detection natively — a TurnInfo message with
event == "EndOfTurn" is the "speaker finished" signal, which removes the
usual DIY VAD/endpointing tuning. Wire surface verified against the installed
SDK source: connect(model, encoding, sample_rate, ...) -> AsyncV2SocketClient
with send_media(bytes), async iteration for messages, send_close_stream().
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from deepgram import AsyncDeepgramClient

from assistant.audio.base import SAMPLE_RATE
from assistant.stt.base import SttError, SttEvent


class DeepgramFlux:
    def __init__(self, api_key: str, *, eot_threshold: float | None = None) -> None:
        if not api_key:
            raise SttError("DEEPGRAM_API_KEY is not set (see .env.example).")
        self._client = AsyncDeepgramClient(api_key=api_key)
        self._eot_threshold = eot_threshold

    @asynccontextmanager
    async def stream(self):
        kwargs = {}
        if self._eot_threshold is not None:
            kwargs["eot_threshold"] = self._eot_threshold
        async with self._client.listen.v2.connect(
            model="flux-general-en",
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            **kwargs,
        ) as connection:
            yield _FluxStream(connection)


class _FluxStream:
    def __init__(self, connection) -> None:
        self._connection = connection

    async def send_audio(self, pcm: bytes) -> None:
        await self._connection.send_media(pcm)

    async def events(self) -> AsyncIterator[SttEvent]:
        async for message in self._connection:
            msg_type = getattr(message, "type", "")
            if msg_type == "TurnInfo":
                transcript = (message.transcript or "").strip()
                if message.event == "EndOfTurn":
                    if transcript:
                        yield SttEvent("final", transcript)
                    return  # empty end-of-turn = silence; caller decides what's next
                if message.event in ("Update", "StartOfTurn", "EagerEndOfTurn") and transcript:
                    yield SttEvent("partial", transcript)
            elif msg_type == "FatalError":
                raise SttError(f"Deepgram fatal error: {message}")
