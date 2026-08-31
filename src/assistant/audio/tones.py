"""Earcons: tiny locally generated tones (this is not TTS — that's M4).

The rising tone doubles as the privacy "listening" cue from the design notes.
All playback is fire-and-forget and failure-tolerant: no output device means
no sound, never a crash.
"""

from __future__ import annotations

import contextlib

import numpy as np
import sounddevice as sd

_RATE = 22_050


def _tone(freqs: list[float], ms: int = 110) -> np.ndarray:
    pieces = []
    for freq in freqs:
        t = np.linspace(0, ms / 1000, int(_RATE * ms / 1000), endpoint=False)
        wave = 0.25 * np.sin(2 * np.pi * freq * t)
        fade = np.minimum(1, np.minimum(t, t[-1] - t) / 0.012)  # declick
        pieces.append(wave * fade)
    return np.concatenate(pieces).astype(np.float32)


_SOUNDS = {
    "wake": _tone([660, 990]),  # rising: I'm listening
    "close": _tone([660, 440]),  # falling: conversation over
    "error": _tone([220], ms=250),  # low buzz
}


def play(kind: str) -> None:
    with contextlib.suppress(Exception):
        sd.play(_SOUNDS[kind], _RATE, blocking=False)
