"""How loud the room is, right now — the one number behind the panel's bar.

RMS per 80 ms frame (the frames already flowing to the model; no extra
capture), mapped onto the range a voice actually lives in at a desk mic and
smoothed with an instant attack and a short decay: the bar jumps the moment
he speaks and falls flat about a second after he stops. It is a level meter,
not a waveform — one honest scalar the Settings panel can draw from its own
thread.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

# int16 RMS. The floor is ordinary room tone at a desk mic (below it the bar
# reads flat — see _ABS_FLOOR in the realtime engine, which calls the same
# region "not speech"); the ceiling is a normal voice close to the mic.
FLOOR = 300.0
CEILING = 6_000.0

_DECAY = 0.7  # per frame: full to flat in about a second (12.5 frames)
_FLAT = 0.02  # below this the bar is simply flat


def rms(frame: bytes) -> float:
    """Int16 RMS of one PCM frame (0.0 when there are no samples)."""
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0


def normalize(level: float) -> float:
    """int16 RMS -> 0..1 bar height, logarithmic because hearing is: quiet
    speech reads about a third, a normal voice about three quarters."""
    if level <= FLOOR:
        return 0.0
    return min(1.0, math.log(level / FLOOR) / math.log(CEILING / FLOOR))


class LevelMeter:
    """Frames in, bar height out. Rises instantly and falls by a fixed decay
    per frame, so a pause between words dips rather than flickers to nothing."""

    def __init__(self, *, decay: float = _DECAY) -> None:
        self._decay = decay
        self.value = 0.0

    def feed(self, frame: bytes) -> float:
        """One 80 ms frame of int16 PCM."""
        return self.push(rms(frame))

    def push(self, level: float) -> float:
        """One frame's int16 RMS, already measured."""
        return self._set(max(normalize(level), self.value * self._decay))

    def idle(self) -> float:
        """No frame of his this tick (she is speaking, the mic is muted):
        keep falling. Her own voice must never move his bar."""
        return self._set(self.value * self._decay)

    def reset(self) -> None:
        self.value = 0.0

    def _set(self, value: float) -> float:
        self.value = 0.0 if value < _FLAT else value
        return self.value


_RECENT_S = 6.0  # how far back the ring remembers
_SPEECH_MIN = 250.0  # int16 RMS a voice at a desk mic clears even when quiet
_SPEECH_OVER_FLOOR = 2.5  # ...and how far above the room's own level it must sit
_FLOOR_WINDOW_S = 2.0  # the room's level is read from this long before the moment asked about


class RecentLevels:
    """The last few seconds of the room as (when, RMS) pairs, so a question
    about a moment that has already passed — did he keep talking after the
    wake word? — can be answered without owning a single frame. The frames
    themselves keep flowing to whoever reads the microphone."""

    def __init__(self, keep_s: float = _RECENT_S) -> None:
        self._keep_s = keep_s
        self._levels: deque[tuple[float, float]] = deque()

    def push(self, captured: float, level: float) -> None:
        self._levels.append((captured, level))
        cutoff = captured - self._keep_s
        while self._levels and self._levels[0][0] < cutoff:
            self._levels.popleft()

    def floor(self, before: float, window_s: float = _FLOOR_WINDOW_S) -> float:
        """The room's own level just before `before`: the median RMS there,
        so a television or music in the room raises the bar rather than
        passing for him."""
        window = sorted(level for at, level in self._levels if before - window_s <= at < before)
        return window[len(window) // 2] if window else 0.0

    def heard(
        self,
        since: float,
        until: float | None = None,
        *,
        before: float | None = None,
        frames: int = 2,
    ) -> bool:
        """Speech between `since` and `until`: at least `frames` frames well
        above both the absolute floor and the room's level before `before`
        (the wake word's own moment, so the room is read from before it)."""
        floor = self.floor(before if before is not None else since)
        bar = max(_SPEECH_MIN, floor * _SPEECH_OVER_FLOOR)
        loud = sum(
            1 for at, level in self._levels
            if since <= at and (until is None or at <= until) and level >= bar
        )
        return loud >= frames
