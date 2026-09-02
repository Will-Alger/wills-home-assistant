"""The capture cues: you always know whether she is hearing you.

One ding when a listening window opens — the wake, and every follow-up turn
in the same conversation — a smaller, falling one when it closes normally,
and the low error tone when capture fails, times out, or the session errors.
A failure never plays the listening ding and clears the listening flag, so
neither the tone nor the Settings panel can tell you she's listening when
she isn't.

Two rules learned the hard way on Windows: a tone goes through the session
speaker whenever one is open (a fresh sd.play stream loses the race against a
live PortAudio stream), and repeat calls inside one window are ignored — a
multi-step reply raises "now listening" several times and must ding once.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from assistant.audio import tones


class VoiceCues:
    def __init__(
        self,
        *,
        rate: int = 24_000,
        status: Any | None = None,
        play: Callable[[str], None] = tones.play,
        render: Callable[[str, int], bytes] = tones.pcm,
    ) -> None:
        self._rate = rate
        self._status = status
        self._play = play
        self._render = render
        self.listening = False
        self.played: list[str] = []  # the last few earcons, newest last

    def _sound(self, kind: str, speaker: Any | None) -> None:
        self.played.append(kind)
        del self.played[:-20]  # a days-long process keeps a window, not a history
        with contextlib.suppress(Exception):  # no output device is never a crash
            if speaker is not None:
                speaker.enqueue(self._render(kind, self._rate))
            else:
                self._play(kind)

    def _state(self, state: str) -> None:
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status.set_state(state)
                self._status.set_listening(self.listening)

    def start(self, speaker: Any | None = None) -> bool:
        """A listening window opened. False when one already was."""
        if self.listening:
            return False
        self.listening = True
        self._sound("wake", speaker)
        self._state("listening")
        return True

    def end(self, speaker: Any | None = None) -> bool:
        """The listening window closed normally (you stopped talking)."""
        if not self.listening:
            return False
        self.listening = False
        self._sound("listen_end", speaker)
        self._state("working")
        return True

    def error(self, message: str = "", speaker: Any | None = None) -> None:
        """Capture failed, timed out, or the session errored."""
        self.listening = False
        self._sound("error", speaker)
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status.error(message or "capture failed")
                self._status.set_listening(False)

    def session_end(self, speaker: Any | None = None) -> None:
        """The conversation is over: the goodbye chime, back to idle."""
        self.listening = False
        self._sound("close", speaker)
        self._state("idle")

    def reset(self) -> None:
        """Drop the listening flag without a sound (the session is closing)."""
        self.listening = False
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status.set_listening(False)
