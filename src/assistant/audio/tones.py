"""Earcons: tiny locally synthesized chimes (this is not TTS — that's M4).

Not beeps — small bell tones: a few harmonic partials, an 8 ms attack, and a
long exponential decay, overlapped so two notes ring into each other. That's
the whole difference between "cheap electronics" and "she heard me."

The rising chime doubles as the privacy "listening" cue from the design notes.
All playback is fire-and-forget and failure-tolerant: no output device means
no sound, never a crash.
"""

from __future__ import annotations

import contextlib

import numpy as np
import sounddevice as sd

_RATE = 22_050


def _note(freq: float, ms: int, amp: float, decay: float) -> np.ndarray:
    """One bell-like note: fundamental + soft octave + a whisper of the 12th,
    fast attack, exponential ring-out."""
    t = np.linspace(0, ms / 1000, int(_RATE * ms / 1000), endpoint=False)
    wave = (
        1.00 * np.sin(2 * np.pi * freq * t)
        + 0.35 * np.sin(2 * np.pi * freq * 2 * t)
        + 0.12 * np.sin(2 * np.pi * freq * 3 * t)
    )
    envelope = np.minimum(t / 0.008, 1.0) * np.exp(-decay * t)
    return (amp / 1.47) * wave * envelope


def _chime(
    notes: list[float], *, ms_each: int = 340, stagger_ms: int = 95,
    amp: float = 0.34, decay: float = 7.0,
) -> np.ndarray:
    """Overlap-add the notes so each starts while the previous still rings."""
    step = int(_RATE * stagger_ms / 1000)
    length = step * (len(notes) - 1) + int(_RATE * ms_each / 1000)
    out = np.zeros(length, dtype=np.float64)
    for i, freq in enumerate(notes):
        rendered = _note(freq, ms_each, amp, decay)
        out[i * step : i * step + len(rendered)] += rendered
    peak = np.max(np.abs(out)) or 1.0
    if peak > 0.85:
        out *= 0.85 / peak  # headroom: never clip when notes stack
    return out.astype(np.float32)


_SOUNDS = {
    # rising fourth, bright: I'm listening
    "wake": _chime([659.3, 880.0]),
    # falling, softer and longer ring: going back to sleep
    "close": _chime([880.0, 659.3, 523.3], amp=0.26, decay=5.0, ms_each=420),
    # low, brief, minor-ish: something went wrong
    "error": _chime([220.0, 185.0], amp=0.24, decay=9.0, ms_each=260),
}


def play(kind: str) -> None:
    with contextlib.suppress(Exception):
        sd.play(_SOUNDS[kind], _RATE, blocking=False)
