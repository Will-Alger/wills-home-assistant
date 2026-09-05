"""Streaming speaker output: enqueue PCM chunks, a callback drains them.

Built for the realtime engine (24 kHz mono int16). `clear()` supports
interruption — dropping everything not yet played, instantly. The output
device is a saved name fragment (see devices.py). A chosen entry that
refuses the rate is opened through the same device's other host API (no
note: same speaker). A chosen device that is missing or won't open falls
back with a note — to the system default, unless Windows' default is a
virtual endpoint (Steam's streaming speakers: audible only inside a Steam
client), in which case real speakers or headphones come first. A headset
walking out of range must never silence her.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Self

import sounddevice as sd

from assistant.audio import devices

# Outputs nobody hears in the room: Steam's streaming endpoints and their
# "Wave" twins, loopbacks, a headset's phone-quality hands-free link.
_NOT_A_SPEAKER = ("steam", "wave", "loopback", "hands-free")


def not_a_speaker(name: str) -> bool:
    return any(word in name.lower() for word in _NOT_A_SPEAKER)


def default_output_name() -> str:
    """What Windows calls the default output right now ("" when there is none)."""
    with contextlib.suppress(Exception):
        return " ".join(str(sd.query_devices(kind="output")["name"]).split())
    return ""


def output_candidates() -> list[int]:
    """Outputs worth trying when the chosen one is gone and the default is a
    virtual device: named speakers and headphones first, then the rest, in
    the pickers' order (best host API, then name)."""
    rows = [row for row in devices.entries("output") if not not_a_speaker(str(row["name"]))]
    rows.sort(key=lambda row: 0 if any(w in str(row["name"]).lower() for w in ("speakers", "headphones")) else 1)
    return [int(row["index"]) for row in rows]


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

    def _try(self, index: int) -> bool:
        try:
            self._stream = self._open(index)
        except sd.PortAudioError:
            return False
        self.device_in_use = devices.describe(str(index), "output")
        return True

    async def __aenter__(self) -> Self:
        chosen = self._device
        problems: list[str] = []
        if chosen is None and not devices.is_default(self._spec):
            problems.append(f"speaker '{self._spec}' isn't plugged in")
        if chosen is not None:
            try:
                self._stream = self._open(chosen)
                self.device_in_use = devices.describe(str(chosen), "output")
                return self
            except sd.PortAudioError as err:
                if any(self._try(index) for index in devices.twins(chosen, "output")):
                    return self  # the same speaker, through a host API that resamples
                problems.append(f"speaker '{self._spec}' would not open ({err})")
        default = default_output_name()
        if default and not_a_speaker(default):
            # Windows' default output is Steam's streaming device: she would
            # talk into a void. Real speakers first, the default as a last resort.
            for index in output_candidates():
                if index != chosen and self._try(index):
                    problems.append(f"the default output '{devices.pretty(default)}' isn't a speaker")
                    self.device_note = "; ".join(problems) + f" — using '{self.device_in_use}'"
                    return self
            problems.append(f"the default output '{devices.pretty(default)}' isn't a speaker and nothing else would open")
        self._stream = self._open(None)  # the default, whatever it is; raises only if even that fails
        self.device_in_use = devices.describe("", "output")
        self.device_note = "; ".join(problems) + " — using the default" if problems else None
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
