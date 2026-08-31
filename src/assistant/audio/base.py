"""Audio layer contract.

Everything speaks 16 kHz mono 16-bit PCM in 80 ms frames — the native format
of both openWakeWord and Deepgram Flux, so no resampling happens anywhere.
The AudioSource protocol is the seam where a network slides in later
(satellite devices) and where a fake slides in for tests.
"""

from __future__ import annotations

from typing import Protocol

SAMPLE_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 1280
FRAME_BYTES = FRAME_SAMPLES * 2  # int16


class AudioSourceClosed(Exception):
    """Raised by get_frame() when the source has ended."""


class AudioSource(Protocol):
    async def get_frame(self) -> bytes:
        """Next 80 ms frame of 16 kHz mono int16 PCM; raises AudioSourceClosed at end."""
        ...

    def drain(self) -> None:
        """Discard buffered frames (e.g. audio captured while the brain was busy)."""
        ...
