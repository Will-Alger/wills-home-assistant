"""Earcon synthesis sanity — no audio device needed."""

from __future__ import annotations

import numpy as np

from assistant.audio.tones import _SOUNDS


def test_all_earcons_are_well_formed() -> None:
    for kind in ("wake", "close", "error", "working"):
        sound = _SOUNDS[kind]
        assert sound.dtype == np.float32
        assert len(sound) > 2000  # audible, not a click
        assert np.max(np.abs(sound)) <= 0.85  # headroom, never clips
        assert np.max(np.abs(sound)) > 0.05  # actually audible


def test_pcm_renders_at_arbitrary_rates() -> None:
    from assistant.audio.tones import pcm

    data = pcm("close", 24_000)
    assert len(data) % 2 == 0 and len(data) > 20_000  # int16 mono, audibly long
    as_int = np.frombuffer(data, dtype=np.int16)
    assert np.max(np.abs(as_int)) <= int(0.85 * 32767) + 1  # same headroom rule


def test_wake_rises_and_close_falls() -> None:
    # crude spectral check: the loudest early moment vs late moment frequency
    def dominant_hz(chunk: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        return float(np.fft.rfftfreq(len(chunk), 1 / 22_050)[int(np.argmax(spectrum))])

    wake = _SOUNDS["wake"]
    assert dominant_hz(wake[:1500]) < dominant_hz(wake[-6000:-4500])
    close = _SOUNDS["close"]
    assert dominant_hz(close[:1500]) > dominant_hz(close[-7000:-5500])


def test_the_working_tick_is_lower_and_quieter_than_the_wake_ding() -> None:
    """It says "still here" under a running tool — it must never be mistaken
    for the ding that says "your turn"."""
    working, wake = _SOUNDS["working"], _SOUNDS["wake"]
    assert np.max(np.abs(working)) < 0.5 * np.max(np.abs(wake))  # clearly softer
    assert len(working) < len(wake)  # and shorter: a tick, not a chime

    def dominant_hz(chunk: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        return float(np.fft.rfftfreq(len(chunk), 1 / 22_050)[int(np.argmax(spectrum))])

    assert dominant_hz(working) < dominant_hz(wake[:1500])  # and lower
