"""Four sentences she can say with the network unplugged.

Everything else she says is generated in a datacentre: when the Realtime
socket will not open there is nothing to speak with, and today she goes
quiet behind an error tone. These four lines are rendered once (see
`scripts/render_fallbacks.py`), committed as WAVs under `assets/voice/`,
and played straight off the disk — no key, no socket, no model.

Two rules, both learned from the earcons (`cues.py`):

- The line goes through the session speaker whenever one is open. A fresh
  `sd.play` stream loses the race against a live PortAudio stream on
  Windows, so a line queued on the stream she already owns is the only one
  reliably heard mid-conversation.
- One failure gets one line. Two paths can see the same collapse — the
  engine's supervisor and the runner's recovery loop — and she must not
  say two things about it, so a second line inside `min_gap_s` is dropped.

An asset that is missing or unreadable is silence, never a crash: a
fallback that can take the app down is worse than no fallback at all.
"""

from __future__ import annotations

import contextlib
import importlib
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from assistant.config import code_root

RATE = 24_000  # what the session speaker takes (realtime_engine.REALTIME_RATE)

# kind -> the words. The render script speaks exactly these.
LINES: dict[str, str] = {
    "voice_service": "I can't reach my voice service right now.",
    "home_down": "The home isn't answering.",
    "failed": "Something failed. Check the log.",
    "moment": "One moment.",
}

# kind -> the file under assets/voice/
FILES: dict[str, str] = {
    "voice_service": "voice-service.wav",
    "home_down": "home-down.wav",
    "failed": "failed.wav",
    "moment": "moment.wav",
}


def voice_dir() -> Path:
    """Where the rendered lines live — beside the code, not in the data
    directory: they are part of the build, the same in every worktree."""
    return code_root() / "assets" / "voice"


def to_pcm(frames: bytes, *, channels: int, source_rate: int, rate: int) -> bytes:
    """Mono 16-bit PCM at `rate`, from whatever the WAV happened to hold.

    Linear interpolation is enough: these are four short sentences played
    once in a while, not a stream, and a re-render with another tool must
    not need a matching sample rate to be audible."""
    samples = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if source_rate != rate and len(samples):
        wanted = int(len(samples) * rate / source_rate)
        samples = np.interp(
            np.linspace(0, len(samples), wanted, endpoint=False),
            np.arange(len(samples)),
            samples.astype(np.float64),
        )
    return samples.astype("<i2").tobytes()


def read_wav(path: Path, rate: int = RATE) -> bytes:
    """One rendered line as mono int16 PCM at `rate`."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path.name} is not 16-bit audio")
        channels, source_rate = wav.getnchannels(), wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    return to_pcm(frames, channels=channels, source_rate=source_rate, rate=rate)


def write_wav(path: Path, pcm: bytes, rate: int = RATE) -> None:
    """Write mono int16 PCM as a WAV — what the render script commits."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)


_UNREACHABLE: tuple[type[BaseException], ...] = ()


def _unreachable_errors() -> tuple[type[BaseException], ...]:
    """The exception types that mean "the voice service was not reachable".

    OSError covers what the websocket layer raises with the cable out
    (`socket.gaierror`, connection refused/reset, and `TimeoutError`, which
    subclasses OSError since 3.10). The other three are imported by name
    because they are not: the OpenAI SDK wraps transport failures in
    `APIConnectionError` (checked in the installed SDK), httpx has its own
    `TransportError` tree, and `websockets` raises `WebSocketException` for
    a handshake that never completed. Deliberately NOT here:
    `sounddevice.PortAudioError`, which subclasses Exception alone — a
    microphone that will not open is not the network."""
    global _UNREACHABLE
    if not _UNREACHABLE:
        found: list[type[BaseException]] = [OSError]
        for module, name in (
            ("openai", "APIConnectionError"),
            ("httpx", "TransportError"),
            ("websockets.exceptions", "WebSocketException"),
        ):
            with contextlib.suppress(Exception):  # an absent package is not a failure
                found.append(getattr(importlib.import_module(module), name))
        _UNREACHABLE = tuple(found)
    return _UNREACHABLE


def kind_for(err: BaseException) -> str:
    """Which line a dead cycle deserves: "I can't reach my voice service"
    when nothing could be reached, otherwise "Something failed". The
    `__cause__` chain is walked because the SDK re-raises (`raise ... from`)
    and the real reason is usually one link down."""
    seen: set[int] = set()
    cause: BaseException | None = err
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, _unreachable_errors()):
            return "voice_service"
        cause = cause.__cause__
    return "failed"


def _speak_locally(pcm: bytes, rate: int) -> None:
    """No session speaker open (she was idle): a throwaway stream on the
    chosen output, the same two-try walk the earcons do."""
    import sounddevice as sd

    from assistant.audio import tones

    samples = np.frombuffer(pcm, dtype="<i2")
    for device in dict.fromkeys((tones.output_device(), None)):
        with contextlib.suppress(Exception):  # no output device is never a crash
            sd.play(samples, rate, blocking=False, device=device)
            return


class SpokenFallbacks:
    """The four lines, and the one-line-per-failure rule."""

    def __init__(
        self,
        directory: Path | None = None,
        *,
        rate: int = RATE,
        min_gap_s: float = 8.0,
        play: Callable[[bytes, int], None] = _speak_locally,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._dir = directory if directory is not None else voice_dir()
        self._rate = rate
        self._min_gap = min_gap_s
        self._play = play
        self._now = now
        self._cache: dict[str, bytes] = {}
        self._last = float("-inf")
        self.played: list[str] = []  # the last few file names, newest last

    def missing(self) -> list[str]:
        """The lines that are not on disk (an incomplete render)."""
        return [name for name in FILES.values() if not (self._dir / name).is_file()]

    def note(self) -> str:
        """One sentence for the boot log when she cannot speak her fallbacks."""
        gone = self.missing()
        if not gone:
            return ""
        return (
            f"the spoken fallbacks are missing ({', '.join(gone)}) — a failure will be "
            "an error tone only; re-render with scripts/render_fallbacks.py"
        )

    def _pcm(self, kind: str) -> bytes:
        if kind not in self._cache:
            self._cache[kind] = b""
            with contextlib.suppress(Exception):  # a missing line is silence
                self._cache[kind] = read_wav(self._dir / FILES[kind], self._rate)
        return self._cache[kind]

    def say(self, kind: str, speaker: Any | None = None) -> str:
        """Say one line. Returns the file name played, or "" when nothing
        was — an unknown kind, an asset that would not load, or a second
        line for a failure that already has one."""
        if kind not in FILES:
            return ""
        now = self._now()
        if now - self._last < self._min_gap:
            return ""  # one failure, one line
        pcm = self._pcm(kind)
        if not pcm:
            return ""
        self._last = now
        name = FILES[kind]
        self.played.append(name)
        del self.played[:-20]  # a days-long process keeps a window, not a history
        spoken = False
        if speaker is not None and getattr(speaker, "is_open", True):
            with contextlib.suppress(Exception):
                speaker.enqueue(pcm)
                spoken = True
        if not spoken:
            with contextlib.suppress(Exception):
                self._play(pcm, self._rate)
        return name

    def after_failure(self, err: BaseException, speaker: Any | None = None) -> str:
        """The runner's recovery path: one line for the cycle that just died."""
        return self.say(kind_for(err), speaker)
