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
import time
from collections import deque
from collections.abc import Callable
from typing import Self

import numpy as np
import sounddevice as sd

from assistant.audio import devices

_STALL_S = 1.0  # audio waiting, no callback for this long: the device is gone

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
    """One output stream, kept open across idle and conversation (the runner
    owns it): chimes and her voice share it, so nothing races to open a
    device, and a Bluetooth speaker never sees a fresh stream mid-sentence.

    Per-item accounting: `begin_item(id)` marks where an assistant item's
    audio starts in the byte stream; `played_ms(id)` says how much of it the
    callback has actually consumed — what he heard, not what was generated.
    That number is what `conversation.item.truncate` needs after a barge-in.
    """

    def __init__(self, samplerate: int = 24_000, device: str = "") -> None:
        self._samplerate = samplerate
        self._spec = device or ""
        self._device = devices.find(self._spec, "output")
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._stream: sd.RawOutputStream | None = None
        self.device_note: str | None = None  # set when the chosen speaker was not used
        self.device_in_use = ""  # the speaker actually opened (the panel shows it)
        self._enqueued = 0  # bytes ever enqueued (minus what clear() dropped)
        self._consumed = 0  # bytes the callback has played
        self._items: dict[str, tuple[int, int]] = {}  # item id -> (start, end) byte offsets
        self.current_item = ""  # the assistant item whose audio is being enqueued
        self._last_callback = time.monotonic()
        self._marks: list[tuple[int, Callable[[], None]]] = []  # (byte position, callback)
        self._paused = False
        # What the device has been playing, as RMS per callback chunk with its
        # time: the engine's talk-over guard reads it to know how loud her
        # own voice is right now, and so how loud the mic's echo of it will be.
        self._played_levels: deque[tuple[float, float]] = deque(maxlen=64)

    def played_level(self, window_s: float = 0.3) -> float:
        """The loudest chunk the device pulled in the last `window_s` seconds
        (int16 RMS; 0.0 in silence). The mic hears the room a little after
        the device plays it, so the guard asks for a window, not an instant."""
        now = time.monotonic()
        levels = [rms for at, rms in list(self._played_levels) if now - at <= window_s]
        return max(levels) if levels else 0.0

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    @property
    def stalled(self) -> bool:
        """Audio is waiting but the device stopped asking for it (a Bluetooth
        speaker that walked away): the runner should reopen."""
        with self._lock:
            waiting = bool(self._buffer)
        return self.is_open and waiting and time.monotonic() - self._last_callback > _STALL_S

    @property
    def enqueued(self) -> int:
        """Bytes enqueued so far: a position to hand to notify_when_played."""
        with self._lock:
            return self._enqueued

    def notify_when_played(self, position: int, fn: Callable[[], None]) -> None:
        """Call `fn` (from the audio thread, once) the instant the callback
        pulls the byte at `position` — the moment a chime could be heard,
        not the moment it was queued. Dropped by clear() if never reached."""
        with self._lock:
            if position < self._consumed:
                fire = True
            else:
                self._marks.append((position, fn))
                fire = False
        if fire:
            with contextlib.suppress(Exception):
                fn()

    def pause(self) -> None:
        """Hold playback where it is (a possible interruption): the callback
        plays silence, the audio waits, played_ms stops moving."""
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        """A false alarm: carry on from the paused position."""
        with self._lock:
            self._paused = False

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def _consume(self, need: int) -> bytes:
        """The callback's read: the next `need` bytes, silence-padded."""
        self._last_callback = time.monotonic()
        due: list[Callable[[], None]] = []
        with self._lock:
            if self._paused:
                chunk = b""  # held: the device gets silence, the audio waits
            else:
                chunk = bytes(self._buffer[:need])
                del self._buffer[: len(chunk)]
                self._consumed += len(chunk)
                if self._marks and chunk:
                    due = [fn for pos, fn in self._marks if pos < self._consumed]
                    self._marks = [(pos, fn) for pos, fn in self._marks if pos >= self._consumed]
        for fn in due:
            with contextlib.suppress(Exception):
                fn()
        if len(chunk) < need:
            chunk += b"\x00" * (need - len(chunk))  # underflow = silence
        with contextlib.suppress(Exception):
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            self._played_levels.append((time.monotonic(), float(np.sqrt(np.mean(samples * samples)))))
        tap = self.tap
        if tap is not None:
            with contextlib.suppress(Exception):
                tap(chunk)
        return chunk

    def _open(self, device: int | None) -> sd.RawOutputStream:
        def callback(outdata, frames: int, _time, _status) -> None:  # PortAudio thread
            outdata[:] = self._consume(frames * 2)  # int16 mono

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
        return await self.open()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def open(self) -> Self:
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

    async def close(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None

    # A session recording: every chunk the device pulled, silence included,
    # so the track is real time — what was heard when, not what was queued.
    # Called on the PortAudio thread: it must only stash bytes.
    tap: Callable[[bytes], None] | None = None

    def enqueue(self, pcm: bytes) -> None:
        with self._lock:
            self._buffer.extend(pcm)
            self._enqueued += len(pcm)
            if self.current_item:
                start, _end = self._items.get(self.current_item, (self._enqueued - len(pcm), 0))
                self._items[self.current_item] = (start, self._enqueued)

    def begin_item(self, item_id: str) -> None:
        """Audio for this assistant item starts here (idempotent for the same id)."""
        if not item_id or item_id == self.current_item:
            return
        with self._lock:
            self.current_item = item_id
            self._items.setdefault(item_id, (self._enqueued, self._enqueued))
            if len(self._items) > 50:  # a long session: forget the oldest
                for old in list(self._items)[:-25]:
                    del self._items[old]

    def played_ms(self, item_id: str = "") -> int:
        """How much of the item he has actually heard, in milliseconds
        (0 when the item is unknown)."""
        item_id = item_id or self.current_item
        with self._lock:
            span = self._items.get(item_id)
            if span is None:
                return 0
            start, end = span
            heard = min(max(self._consumed - start, 0), end - start)
        return int(heard * 1000 / (2 * self._samplerate))

    def clear(self) -> None:
        """Drop everything not yet played (a barge-in). The current item's
        recorded end moves back to what was heard, so played_ms stays honest."""
        with self._lock:
            dropped = len(self._buffer)
            self._buffer.clear()
            self._enqueued -= dropped
            self._paused = False
            self._marks = [(pos, fn) for pos, fn in self._marks if pos < self._enqueued]
            if self.current_item in self._items:
                start, end = self._items[self.current_item]
                self._items[self.current_item] = (start, max(start, min(end, self._enqueued)))

    @property
    def pending_seconds(self) -> float:
        with self._lock:
            return len(self._buffer) / 2 / self._samplerate

    async def wait_idle(self, tail_s: float = 0.1) -> None:
        """Return once queued audio has (approximately) finished playing — or
        once the device has stopped taking it, so a vanished speaker can
        never hang a conversation."""
        while self.pending_seconds > 0:
            if self.stalled:
                self.clear()
                return
            await asyncio.sleep(0.05)
        await asyncio.sleep(tail_s)
