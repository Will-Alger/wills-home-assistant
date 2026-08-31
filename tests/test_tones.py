"""Earcon synthesis sanity — no audio device needed."""

from __future__ import annotations

import numpy as np

from assistant.audio.tones import _SOUNDS


def test_all_earcons_are_well_formed() -> None:
    for kind in ("wake", "close", "error"):
        sound = _SOUNDS[kind]
        assert sound.dtype == np.float32
        assert len(sound) > 2000  # audible, not a click
        assert np.max(np.abs(sound)) <= 0.85  # headroom, never clips
        assert np.max(np.abs(sound)) > 0.05  # actually audible


def test_wake_rises_and_close_falls() -> None:
    # crude spectral check: the loudest early moment vs late moment frequency
    def dominant_hz(chunk: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        return float(np.fft.rfftfreq(len(chunk), 1 / 22_050)[int(np.argmax(spectrum))])

    wake = _SOUNDS["wake"]
    assert dominant_hz(wake[:1500]) < dominant_hz(wake[-6000:-4500])
    close = _SOUNDS["close"]
    assert dominant_hz(close[:1500]) > dominant_hz(close[-7000:-5500])
