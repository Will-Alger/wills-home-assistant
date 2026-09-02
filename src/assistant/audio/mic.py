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


# Endpoints that are never the room microphone: a Bluetooth headset's
# hands-free profile (Windows makes it the default input the moment a headset
# connects, and it won't open), virtual/loopback inputs, line-ins.
_NOT_A_MIC = ("headset", "hands-free", "steam", "stereo mix", "line in", "loopback")


def input_candidates() -> list[int]:
    """Input devices worth trying when the configured one will not open:
    real microphones first (WASAPI before the others), then anything else
    with an input channel that isn't a known non-microphone."""
    try:
        devices = sd.query_devices()
        apis = [str(api.get("name", "")).lower() for api in sd.query_hostapis()]
    except Exception:  # noqa: BLE001 — no PortAudio, no candidates
        return []
    wasapi = next((i for i, name in enumerate(apis) if "wasapi" in name), -1)
    mics: list[tuple[int, int]] = []
    rest: list[tuple[int, int]] = []
    for index, dev in enumerate(devices):
        if int(dev.get("max_input_channels", 0) or 0) <= 0:
            continue
        name = str(dev.get("name", "")).lower()
        if any(word in name for word in _NOT_A_MIC):
            continue
        rank = 0 if dev.get("hostapi") == wasapi else 1
        (mics if "mic" in name else rest).append((rank, index))
    return [i for _, i in sorted(mics)] + [i for _, i in sorted(rest)]


class Microphone:
    """Continuous capture; frames buffer in an asyncio queue (drops when full).

    Defaults to the wake-word format (16 kHz, 80 ms frames); the realtime
    engine opens it at 24 kHz instead — pass matching samplerate/frame_samples.
    """

    def __init__(
        self,
        device: str = "",
        queue_frames: int = 50,
        *,
        samplerate: int = SAMPLE_RATE,
        frame_samples: int = FRAME_SAMPLES,
    ) -> None:
        self._spec = device
        self._device = resolve_device(device)
        self._samplerate = samplerate
        self._frame_samples = frame_samples
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_frames)
        self._stream: sd.RawInputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.device_note: str | None = None  # set when a fallback device was used
        self.device_in_use = ""  # the mic actually opened (the panel shows it)

    def _open(self, device: int | str | None) -> sd.RawInputStream:
        def callback(indata, _frames, _time, _status) -> None:  # PortAudio thread
            data = bytes(indata)
            assert self._loop is not None
            self._loop.call_soon_threadsafe(self._offer, data)

        stream = sd.RawInputStream(
            samplerate=self._samplerate,
            blocksize=self._frame_samples,
            dtype="int16",
            channels=1,
            device=device,
            callback=callback,
        )
        stream.start()
        return stream

    async def __aenter__(self) -> Self:
        self._loop = asyncio.get_running_loop()
        try:
            self._stream = self._open(self._device)
            self.device_in_use = describe_device(self._spec)
            return self
        except sd.PortAudioError as err:
            # The configured/default input won't open (typical: a Bluetooth
            # headset just became Windows' default input). Try real mics
            # instead of looping on the error — she must keep hearing the room.
            for index in input_candidates():
                if index == self._device:
                    continue
                try:
                    self._stream = self._open(index)
                except sd.PortAudioError:
                    continue
                name = describe_device(str(index))
                self.device_in_use = name
                self.device_note = f"mic fallback: using '{name}' — the default input would not open ({err})"
                return self
            raise

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
