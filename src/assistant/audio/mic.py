"""Microphone AudioSource backed by sounddevice (PortAudio)."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Self

import sounddevice as sd

from assistant.audio import devices
from assistant.audio.base import FRAME_SAMPLES, SAMPLE_RATE, AudioSourceClosed


def resolve_device(spec: str) -> int | None:
    """'' / 'System default' = the default; digits = an index; otherwise a
    case-insensitive name fragment, WASAPI entry preferred (devices.find)."""
    return devices.find(spec, "input")


def describe_device(spec: str) -> str:
    """Human name of the input device this spec resolves to."""
    return devices.describe(spec, "input")


# Endpoints that are never the room microphone: a Bluetooth headset's
# hands-free profile (Windows makes it the default input the moment a headset
# connects, and it won't open), virtual/loopback inputs, line-ins.
_NOT_A_MIC = ("headset", "hands-free", "steam", "stereo mix", "line in", "loopback")


def not_a_mic(name: str) -> bool:
    """A virtual or hands-free input: it opens fine and hears nothing."""
    return any(word in name.lower() for word in _NOT_A_MIC)


def default_input_name() -> str:
    """What Windows calls the default input right now ("" when there is none)."""
    with contextlib.suppress(Exception):
        return " ".join(str(sd.query_devices(kind="input")["name"]).split())
    return ""


def input_candidates() -> list[int]:
    """Input devices worth trying when the configured one will not open:
    real microphones first (WASAPI before the others), then anything else
    with an input channel that isn't a known non-microphone."""
    try:
        found = sd.query_devices()
        apis = [str(api.get("name", "")).lower() for api in sd.query_hostapis()]
    except Exception:  # noqa: BLE001 — no PortAudio, no candidates
        return []
    wasapi = next((i for i, name in enumerate(apis) if "wasapi" in name), -1)
    mics: list[tuple[int, int]] = []
    rest: list[tuple[int, int]] = []
    for index, dev in enumerate(found):
        if int(dev.get("max_input_channels", 0) or 0) <= 0:
            continue
        name = " ".join(str(dev.get("name", "")).split()).lower()
        if not_a_mic(name) or name in devices._ALIASES:  # Sound Mapper is the default again
            continue
        rank = 0 if dev.get("hostapi") == wasapi else 1
        (mics if "mic" in name else rest).append((rank, index))
    return [i for _, i in sorted(mics)] + [i for _, i in sorted(rest)]


class Microphone:
    """Continuous capture; frames buffer in an asyncio queue (drops when full).

    Defaults to the wake-word format (16 kHz, 80 ms frames); the realtime
    engine opens it at 24 kHz instead — pass matching samplerate/frame_samples.

    Every frame is stamped with `time.monotonic()` in the PortAudio callback,
    at capture, so `ignore_before` can throw away the moment she spoke into
    the room and keep everything he said after it (audio/acks.py).
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
        self._queue: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue(maxsize=queue_frames)
        self._ignore_before = 0.0  # frames captured before this stamp are dropped
        self._stream: sd.RawInputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.device_note: str | None = None  # set when a fallback device was used
        self.device_in_use = ""  # the mic actually opened (the panel shows it)
        self.fallback = False  # not the device asked for: the runner keeps re-scanning

    def _open(self, device: int | str | None) -> sd.RawInputStream:
        def callback(indata, _frames, _time, _status) -> None:  # PortAudio thread
            captured = time.monotonic()  # when the room made this sound, not when we read it
            data = bytes(indata)
            assert self._loop is not None
            self._loop.call_soon_threadsafe(self._offer, captured, data)

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
        return await self.open()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    async def open(self) -> Self:
        self._loop = asyncio.get_running_loop()
        missing = self._device is None and not devices.is_default(self._spec)
        if missing:
            # the saved microphone isn't plugged in right now: the default, and say so
            self.device_note = f"microphone '{self._spec.strip()}' isn't plugged in — using the default"
            self.fallback = True
        default = default_input_name() if self._device is None else ""
        if default and not_a_mic(default):
            # Windows' default input is Steam's streaming mic or a headset's
            # hands-free endpoint: it opens without a murmur and she is deaf.
            # A real microphone first; the default only as the last resort.
            lead = f"microphone '{self._spec.strip()}' isn't plugged in and " if missing else ""
            self.fallback = True
            for index in input_candidates():
                try:
                    self._stream = self._open(index)
                except sd.PortAudioError:
                    continue
                self.device_in_use = describe_device(str(index))
                self.device_note = (
                    f"{lead}the default input '{devices.pretty(default)}' isn't a microphone "
                    f"— using '{self.device_in_use}'"
                )
                return self
            self.device_note = (
                f"{lead}the default input '{devices.pretty(default)}' isn't a microphone and no "
                "other microphone would open — she can't hear the room until one is plugged in "
                "or picked in settings"
            )
        try:
            self._stream = self._open(self._device)
            self.device_in_use = describe_device("" if self._device is None else str(self._device))
            return self
        except sd.PortAudioError as err:
            # The chosen entry refused — WASAPI's shared mode will not resample
            # to 16 kHz — so the same microphone through another host API
            # first: that is the device he asked for, not a fallback.
            if self._device is not None:
                for index in devices.twins(self._device, "input"):
                    try:
                        self._stream = self._open(index)
                    except sd.PortAudioError:
                        continue
                    self.device_in_use = describe_device(str(index))
                    return self
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
                self.fallback = True
                return self
            raise

    async def close(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None

    def _offer(self, captured: float, data: bytes) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait((captured, data))

    def ignore_before(self, deadline: float) -> None:
        """Drop every frame captured before `deadline` (a `time.monotonic()`
        stamp). This is how her spoken wake acknowledgment does not become his
        turn: it comes out of the speaker and straight back in here, and
        silence-based turn detection would hand it to the server as speech.
        Frames captured AFTER the deadline are delivered exactly as they
        always were — the command he gives the instant she stops must still
        reach the session first. Only ever moves forward."""
        self._ignore_before = max(self._ignore_before, deadline)

    async def get_frame(self) -> bytes:
        if self._stream is None:
            raise AudioSourceClosed
        while True:
            captured, frame = await self._queue.get()
            if captured >= self._ignore_before:
                return frame

    def drain(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()
