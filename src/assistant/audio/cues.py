"""The capture cues: you always know whether she is hearing you.

One ding when a listening window opens — the wake, a push-to-talk hold, and
every follow-up turn in the same conversation — a smaller, falling one when it
closes normally, and the low error tone when capture fails, times out, or the
session errors. While she is off running a tool and the room would hear
nothing at all, a soft low tick repeats (the engine decides when:
realtime_engine.working_cue).

Or, in her voice (`voices`, from audio/acks.py — WAKE_ACK=voice): the wake is
answered by the acknowledgment ("Yes?"), the end of his turn by "Mm-hm.", a
slow tool by "One moment." and then "Still on it.", a failure by "Sorry,
something went wrong." — and the two dings that had no words fall silent: the
window that re-opens after her reply (she is simply waiting, as a person
would) and the goodbye chime (she has already said goodbye). Will: "anywhere
we have dings should be replaced with her audio feedback". Every voiced cue
returns how long it runs, so the engine can flag the microphone for her
echo the way the wake acknowledgment does.
A failure never plays the listening ding and clears the listening flag, so
neither the tone nor the Settings panel can tell you she's listening when
she isn't. The live microphone level rides the same path (`level`), so the
panel's bar can only move inside a window the flag says is open.

Two rules learned the hard way on Windows: a tone goes through the session
speaker whenever one is open (a fresh sd.play stream loses the race against a
live PortAudio stream), and repeat calls inside one window are ignored — a
multi-step reply raises "now listening" several times and must ding once.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Callable
from typing import Any

from assistant.audio import tones

# In her voice, which earcons fall silent: a listening window that re-opens
# after her reply is just her waiting (a person does not ding after
# answering), and a goodbye she has already said needs no chime after it.
_SILENT_WHEN_VOICED = frozenset({"wake", "close"})
_WORKING_LINES = 2  # "One moment." … "Still on it." — then the room is quiet on purpose


class VoiceCues:
    def __init__(
        self,
        *,
        rate: int = 24_000,
        status: Any | None = None,
        play: Callable[[str], None] = tones.play,
        render: Callable[[str, int], bytes] = tones.pcm,
        voices: dict[str, list[bytes]] | None = None,
        rng: Any | None = None,
    ) -> None:
        """`voices` maps an earcon kind to clips of her voice (audio/acks.py
        `load_cues`): with it, every cue that has a clip is spoken instead
        of rung, and the two that are better left silent are (see
        `_SILENT_WHEN_VOICED`). Without it, the tones."""
        self._rate = rate
        self._status = status
        self._play = play
        self._render = render
        self._voices = voices or {}
        self._rng = rng if rng is not None else random.Random()
        self._last_voice: dict[str, int] = {}  # kind -> index of the clip used last
        self._working_said = 0  # voiced working lines said in this stretch
        self.listening = False
        self.played: list[str] = []  # the last few earcons, newest last

    @property
    def voiced(self) -> bool:
        return bool(self._voices)

    def _sound(
        self, kind: str, speaker: Any | None, on_audible: Callable[[], None] | None = None
    ) -> float:
        """Ring (or say) the cue; returns how long her VOICE runs, in
        seconds — 0.0 for a tone or silence — so the engine can flag the
        microphone for her echo the way the wake acknowledgment does."""
        self.played.append(kind)
        del self.played[:-20]  # a days-long process keeps a window, not a history
        if self._voices and kind in _SILENT_WHEN_VOICED:
            return 0.0
        clips = self._voices.get(kind)
        if clips and speaker is not None:
            if kind == "working":
                if self._working_said >= min(_WORKING_LINES, len(clips)):
                    return 0.0  # she said she is on it, twice: the rest is patience
                index = self._working_said
                self._working_said += 1
            else:
                choices = [i for i in range(len(clips)) if i != self._last_voice.get(kind)] or [0]
                index = self._rng.choice(choices)
            self._last_voice[kind] = index
            pcm = clips[index]
            with contextlib.suppress(Exception):  # a speaker that won't take it: the tone below
                speaker.enqueue(pcm)
                return len(pcm) / 2 / self._rate
        with contextlib.suppress(Exception):  # no output device is never a crash
            if speaker is not None:
                # Mixed into a stream we don't own the callback of: nobody can
                # say when it was heard, so the latency log gets no stamp.
                speaker.enqueue(self._render(kind, self._rate))
            elif on_audible is not None:
                self._play(kind, on_audible)
            else:
                self._play(kind)
        return 0.0

    def _state(self, state: str) -> None:
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status.set_state(state)
                self._status.set_listening(self.listening)

    def start(
        self,
        speaker: Any | None = None,
        on_audible: Callable[[], None] | None = None,
        *,
        sound: bool = True,
    ) -> bool:
        """A listening window opened. False when one already was.

        `sound=False` when something else has already said so out loud — her
        spoken wake acknowledgment (audio/acks.py) answers in place of the
        ding — or when a window re-opens because he never stopped talking
        (a dropped fragment). Either way the flag, the panel state and the
        level meter behave exactly as they do behind a chime."""
        if self.listening:
            return False
        self.listening = True
        self._working_said = 0  # a new turn: the next slow tool may say so again
        if sound:
            self._sound("wake", speaker, on_audible)
        self._state("listening")
        return True

    def end(self, speaker: Any | None = None, *, sound: bool = True) -> bool:
        """The listening window closed normally (you stopped talking). With
        `sound=False` only the flag drops: the engine plays the falling tone
        a beat later, through `turn_over`, and only if the turn really stood
        — a pause mid-sentence used to ding at once."""
        if not self.listening:
            return False
        self.listening = False
        if sound:
            self._sound("listen_end", speaker)
        self._state("working")
        return True

    def wake_sound(self, speaker: Any | None = None) -> float:
        """The wake chime on its own: the window is already open (a deferred
        acknowledgment raised the flag at once) and no clip would load."""
        return self._sound("wake", speaker)

    def turn_over(self, speaker: Any | None = None) -> float:
        """The falling tone — or her "Mm-hm." — on its own, after the flag
        already dropped. Returns the seconds her voice runs (0.0 for a tone)."""
        return self._sound("listen_end", speaker)

    def level(self, value: float) -> None:
        """How loud the room is right now (0..1), for the panel's bar. Sound-
        less and cheap — the engine calls it once per 80 ms frame. The status
        drops it flat with the listening flag, so the bar cannot outlive the
        window it belongs to."""
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status.set_level(value)

    def working(self, speaker: Any | None = None) -> float:
        """She is away doing something and the room would otherwise be silent:
        the soft tick — or "One moment.", then "Still on it.", then nothing —
        and the panel says "working". Never a listening window — the flag is
        untouched, so a tick can't claim she is hearing you. Returns the
        seconds her voice runs (0.0 for a tone or silence)."""
        spoken = self._sound("working", speaker)
        self._state("working")
        return spoken

    def error(self, message: str = "", speaker: Any | None = None) -> float:
        """Capture failed, timed out, or the session errored. Returns the
        seconds her voice runs (0.0 for the tone)."""
        self.listening = False
        spoken = self._sound("error", speaker)
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status.error(message or "capture failed")
                self._status.set_listening(False)
        return spoken

    def idle(self) -> None:
        """Nobody is being listened to and nothing is being worked on, but the
        session is still open — push to talk between holds, or a hold so short
        it caught nothing. Silent on purpose: a tone here would be a lie."""
        self.listening = False
        self._working_said = 0
        self._state("idle")

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
