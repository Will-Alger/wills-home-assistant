"""Microphone AudioSource backed by sounddevice (PortAudio)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Self

import sounddevice as sd

from assistant.audio.base import FRAME_SAMPLES, SAMPLE_RATE, AudioSourceClosed


def resolve_device(spec: str) -> int | str | None:
    """'' = system default; digits = device index; otherwise a name substring."""
    spec = spec.strip()
    if not spec:
        return None
    if spec.isdigit():
        return int(spec)
    return spec


def describe_device(spec: str) -> str:
    """Human name of the input device this spec resolves to."""
    try:
        resolved = resolve_device(spec)
        if resolved is None:
            return str(sd.query_devices(kind="input")["name"])
        return str(sd.query_devices(resolved)["name"])
    except Exception:  # noqa: BLE001 — a label, never worth crashing over
        return spec or "default"


class Microphone:
    """Continuous capture; frames buffer in an asyncio queue (drops when full)."""

    def __init__(self, device: str = "", queue_frames: int = 50) -> None:
        self._device = resolve_device(device)
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_frames)
        self._stream: sd.RawInputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> Self:
        self._loop = asyncio.get_running_loop()

        def callback(indata, _frames, _time, _status) -> None:  # PortAudio thread
            data = bytes(indata)
            assert self._loop is not None
            self._loop.call_soon_threadsafe(self._offer, data)

        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="int16",
            channels=1,
            device=self._device,
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

    def _offer(self, data: bytes) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(data)

    async def get_frame(self) -> bytes:
        if self._stream is None:
            raise AudioSourceClosed
        return await self._queue.get()

    def drain(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()
