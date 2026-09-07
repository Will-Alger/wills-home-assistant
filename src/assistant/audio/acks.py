"""She answers her name in her own voice, before there is a session to ask.

The wake word used to get a rising chime. A chime says "heard you"; it does
not say "yes?" — and the Realtime session behind it takes about a second to
open, which is far too long to answer with. So the answer is pre-rendered:
eight short lines spoken once in her real voice (`scripts/render_acks.py`),
committed as WAVs under `assets/voice/ack/`, loaded at boot and queued on the
speaker the runner already owns the instant the wake fires. Nothing here has
a key, a socket or a model behind it.

Four rules, each paid for elsewhere in this repo:

- Through the SESSION SPEAKER. A fresh `sd.play` stream loses the race
  against a live PortAudio stream on Windows (`cues.py`, `fallbacks.py`).
- Short, varied, and never the same one twice in a row. It is heard twenty
  times a day, and every one of them lands over the top of whatever he is
  about to say next. "Morning." is weighted in before 11:00 and "Evening."
  after 18:00, so a greeting is only ever spoken when it is true.
- Her voice comes straight back in through the microphone, and with
  silence-based turn detection the server would answer it as if HE had
  spoken. The runner therefore flags everything captured before the clip
  ends plus `ECHO_TAIL_S` (`mic.suspect_before`): the engine drops the loud
  frames of that moment — her, off a loudspeaker — and keeps the quiet ones,
  because the command he says right over her "Yes?" is the whole reason for
  answering at all. A transcript that is exactly one of her lines is her
  echo too, and is never his turn (`spoken_lines`).
- A missing or unreadable clip is the old ding plus one boot note, never a
  crash: an acknowledgment that can take the app down is worse than a chime.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from assistant.audio.fallbacks import RATE, read_wav, voice_dir

ECHO_TAIL_S = 0.15  # after the clip has played: the room's own tail of it
MODES = ("voice", "ding", "off")

# slug (= the file name under assets/voice/ack/) -> the words. The render
# script speaks exactly these; {owner} is filled in at render time.
PHRASES: dict[str, str] = {
    "yes": "Yes?",
    "yes-owner": "Yes, {owner}?",
    "go-ahead": "Go ahead.",
    "listening": "Listening.",
    "mm-hm": "Mm-hm?",
    "im-here": "I'm here.",
    "morning": "Morning.",
    "evening": "Evening.",
}

# The two that are only true at their hour: slug -> [from, until) local hour.
GREETINGS: dict[str, tuple[int, int]] = {"morning": (0, 11), "evening": (18, 24)}
NEUTRAL: tuple[str, ...] = tuple(slug for slug in PHRASES if slug not in GREETINGS)
GREETING_WEIGHT = 3  # how much likelier the right greeting is than any one neutral line

# The other cues, in her voice instead of a tone (audio/cues.py): what she
# says when his turn is over and she is about to answer, while a tool keeps
# the room silent, and when something broke. Will: "anywhere we have dings
# should be replaced with her audio feedback". Rendered to assets/voice/cue/.
CUE_PHRASES: dict[str, str] = {
    "mm-hm-ok": "Mm-hm.",  # falling — "heard you" — unlike the wake ack's rising "Mm-hm?"
    "mm": "Mm.",
    "one-moment": "One moment.",
    "still-on-it": "Still on it.",
    "sorry": "Sorry, something went wrong.",
}
# earcon kind (cues.py) -> the slugs that may stand in for it, in order of
# use where order matters (a tool that runs long says the second one once).
CUE_VOICES: dict[str, tuple[str, ...]] = {
    "listen_end": ("mm-hm-ok", "mm"),
    "working": ("one-moment", "still-on-it"),
    "error": ("sorry",),
}
CUE_MAX_S: dict[str, float] = {"sorry": 2.6}  # the one line that is a sentence (takes run 2.2–2.4 s)


def ack_dir() -> Path:
    """Where the rendered acknowledgments live — beside the code, like the
    spoken fallbacks: part of the build, the same in every worktree."""
    return voice_dir() / "ack"


def cue_dir() -> Path:
    return voice_dir() / "cue"


def phrase(slug: str, owner: str = "Will") -> str:
    """What the clip with this slug says, with the owner's name filled in."""
    text = PHRASES.get(slug) or CUE_PHRASES[slug]
    return text.format(owner=owner)


def spoken_lines(owner: str = "Will") -> set[str]:
    """Every line she can say off the disk, normalised the way a transcript
    of it would be — so the engine can tell her own echo from his turn."""
    return {normalise(phrase(slug, owner)) for slug in (*PHRASES, *CUE_PHRASES)}


def normalise(text: str) -> str:
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text.lower()).split())


def load_cues(directory: Path | None = None, *, rate: int = RATE) -> tuple[dict[str, list[bytes]], list[str]]:
    """The voiced cues by earcon kind, and the clip files that are missing."""
    directory = directory if directory is not None else cue_dir()
    clips: dict[str, bytes] = {}
    gone: list[str] = []
    for slug in CUE_PHRASES:
        path = directory / f"{slug}.wav"
        pcm = b""
        with contextlib.suppress(Exception):
            pcm = read_wav(path, rate)
        if pcm:
            clips[slug] = pcm
        else:
            gone.append(path.name)
    voices = {
        kind: [clips[slug] for slug in slugs if slug in clips] for kind, slugs in CUE_VOICES.items()
    }
    return {kind: pcms for kind, pcms in voices.items() if pcms}, gone


class WakeAcks:
    """The clips, the pick, and the one-line boot note when they aren't there.

    `mode` is WAKE_ACK: "voice" (her own voice), "ding" (the old chime) or
    "off" (silence). Only "voice" reads the disk.
    """

    def __init__(
        self,
        directory: Path | None = None,
        *,
        mode: str = "voice",
        rate: int = RATE,
        clock: Callable[[], datetime] = datetime.now,
        rng: random.Random | None = None,
    ) -> None:
        self._dir = directory if directory is not None else ack_dir()
        self._rate = rate
        self._clock = clock
        self._rng = rng if rng is not None else random.Random()
        self._clips: dict[str, bytes] = {}
        self._gone: list[str] = []  # file names that would not load
        self._mode = (mode or "").strip().lower() or "voice"  # WAKE_ACK= is the default
        self._unknown = "" if self._mode in MODES else self._mode
        if self._unknown:
            self._mode = "voice"  # a typo must never cost him the acknowledgment
        self.last = ""  # the slug played last: never the same one twice in a row
        self.played: list[str] = []  # the last few slugs, newest last
        if self._mode == "voice":
            self._load()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def silent(self) -> bool:
        """WAKE_ACK=off: no clip and no ding either, on purpose."""
        return self._mode == "off"

    def _load(self) -> None:
        for slug in PHRASES:
            path = self._dir / f"{slug}.wav"
            pcm = b""
            with contextlib.suppress(Exception):  # a bad asset is a ding, not a crash
                pcm = read_wav(path, self._rate)
            if pcm:
                self._clips[slug] = pcm
            else:
                self._gone.append(path.name)

    def missing(self) -> list[str]:
        """The clips that are not on disk, or would not read (an incomplete render)."""
        return list(self._gone)

    def note(self) -> str:
        """One sentence for the boot log when she cannot answer in her own voice."""
        parts = []
        if self._unknown:
            parts.append(f"WAKE_ACK='{self._unknown}' is not voice, ding or off — using voice")
        if self._mode == "voice" and self._gone:
            parts.append(
                f"spoken wake acknowledgments missing ({', '.join(self._gone)}) — the wake ding "
                "stands in; re-render with scripts/render_acks.py"
            )
        return "; ".join(parts)

    def _pool(self) -> list[str]:
        """The slugs this hour may answer with, greetings weighted in."""
        hour = self._clock().hour
        pool = list(NEUTRAL)
        for slug, (start, until) in GREETINGS.items():
            if start <= hour < until:
                pool += [slug] * GREETING_WEIGHT
        return pool

    def pick(self) -> str:
        """Which clip this wake gets ("" when none would load): one of the
        loaded lines at random, the hour's greeting weighted in, never the
        one that answered the last wake."""
        pool = [slug for slug in self._pool() if slug in self._clips]
        fresh = [slug for slug in pool if slug != self.last]
        choices = fresh or pool  # one clip left on disk is better than silence
        return self._rng.choice(choices) if choices else ""

    def acknowledge(self, speaker: Any) -> float:
        """Answer the wake through `speaker`, and say how long the answer runs.

        Returns the clip's length in seconds — what the runner holds the
        microphone shut for — or 0.0 when nothing was said, which means the
        caller still owes the ding (unless `silent`)."""
        if self._mode != "voice" or speaker is None:
            return 0.0
        slug = self.pick()
        if not slug:
            return 0.0
        pcm = self._clips[slug]
        try:
            speaker.enqueue(pcm)
        except Exception:  # noqa: BLE001 — a speaker that won't take it falls back to the ding
            return 0.0
        self.last = slug
        self.played.append(slug)
        del self.played[:-20]  # a days-long process keeps a window, not a history
        return len(pcm) / 2 / self._rate
