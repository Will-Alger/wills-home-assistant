"""Streaming speaker output: enqueue PCM chunks, a callback drains them.

Built for the realtime engine (24 kHz mono int16). `clear()` supports
interruption — dropping everything not yet played, instantly. The output
device is a saved name fragment (see devices.py); a chosen device that is
missing or won't open falls back to the system default with a note, so a
headset walking out of range never silences her.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Self

import sounddevice as sd

from assistant.audio import devices


class Speaker:
    def __init__(self, samplerate: int = 24_000, device: str = "") -> None:
        self._samplerate = samplerate
        self._spec = device or ""
        self._device = devices.find(self._spec, "output")
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._stream: sd.RawOutputStream | None = None
        self.device_note: str | None = None  # set when the chosen speaker was not used
        self.device_in_use = ""  # the speaker actually opened (the panel shows it)

    def _open(self, device: int | None) -> sd.RawOutputStream:
        def callback(outdata, frames: int, _time, _status) -> None:  # PortAudio thread
            need = frames * 2  # int16 mono
            with self._lock:
                chunk = bytes(self._buffer[:need])
                del self._buffer[: len(chunk)]
            outdata[: len(chunk)] = chunk
            if len(chunk) < need:
                outdata[len(chunk) :] = b"\x00" * (need - len(chunk))  # underflow = silence

        stream = sd.RawOutputStream(
            samplerate=self._samplerate,
            dtype="int16",
            channels=1,
            device=device,
            callback=callback,
        )
        stream.start()
        return stream

    async def __aenter__(self) -> Self:
        chosen = self._device
        if chosen is None and not devices.is_default(self._spec):
            self.device_note = f"speaker '{self._spec}' isn't plugged in — using the default"
        try:
            self._stream = self._open(chosen)
        except sd.PortAudioError as err:
            if chosen is None:
                raise
            self._stream = self._open(None)  # the chosen one won't open: the default, with a note
            self.device_note = f"speaker '{self._spec}' would not open ({err}) — using the default"
            chosen = None
        self.device_in_use = devices.describe("" if chosen is None else str(chosen), "output")
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

    async def wait_idle(self, tail_s: float = 0.1) -> None:
        """Return once queued audio has (approximately) finished playing."""
        while self.pending_seconds > 0:
            await asyncio.sleep(0.05)
        await asyncio.sleep(tail_s)
