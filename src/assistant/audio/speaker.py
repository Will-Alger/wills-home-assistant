"""Streaming speaker output: enqueue PCM chunks, a callback drains them.

Built for the realtime engine (24 kHz mono int16). `clear()` supports
interruption — dropping everything not yet played, instantly.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Self

import sounddevice as sd


class Speaker:
    def __init__(self, samplerate: int = 24_000) -> None:
        self._samplerate = samplerate
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._stream: sd.RawOutputStream | None = None

    async def __aenter__(self) -> Self:
        def callback(outdata, frames: int, _time, _status) -> None:  # PortAudio thread
            need = frames * 2  # int16 mono
            with self._lock:
                chunk = bytes(self._buffer[:need])
                del self._buffer[: len(chunk)]
            outdata[: len(chunk)] = chunk
            if len(chunk) < need:
                outdata[len(chunk) :] = b"\x00" * (need - len(chunk))  # underflow = silence

        self._stream = sd.RawOutputStream(
            samplerate=self._samplerate,
            dtype="int16",
            channels=1,
            callback=callback,
        )
        self._stream.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None

    def enqueue(self, pcm: bytes) -> None:
        with self._lock:
            self._buffer.extend(pcm)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    @property
    def pending_seconds(self) -> float:
        with self._lock:
            return len(self._buffer) / 2 / self._samplerate

    async def wait_idle(self, tail_s: float = 0.25) -> None:
        """Return once queued audio has (approximately) finished playing."""
        while self.pending_seconds > 0:
            await asyncio.sleep(0.05)
        await asyncio.sleep(tail_s)
