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
    "yes-sir": "Yes, sir?",
    "sir": "Sir?",
    "mm-hm": "Mm-hm?",
    "go-ahead": "Go ahead.",
}
# Retired 2026-09-12 (Will: "'I'm here' feels fake and less Jarvis-like than
# 'Yes, sir?'"): "Yes, {owner}?", "Listening.", "I'm here.", "Morning.", "Evening.".

# Lines that are only true at their hour: slug -> [from, until) local hour.
# None at the moment; the machinery stays for the day one earns its place.
GREETINGS: dict[str, tuple[int, int]] = {}
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
        beat_s: float = 0.0,
        beat_jitter_s: float = 0.15,
    ) -> None:
        self._dir = directory if directory is not None else ack_dir()
        self._rate = rate
        self._clock = clock
        self._rng = rng if rng is not None else random.Random()
        # A person answers their name a beat after it is said; the clip is
        # ready 30 ms after the wake, which sounded eager (Will: "so fast
        # it's almost a little unnatural"). Silence queued ahead of the clip
        # is that beat, a little different each time; 0 = at once.
        self._beat_s = max(0.0, float(beat_s))
        self._beat_jitter_s = max(0.0, float(beat_jitter_s))
        self.last_beat_s = 0.0  # the pause before the clip that answered the last wake
        # slug -> its takes (slug.wav, slug-2.wav, …): the same words said
        # more than one way, so twenty wakes a day do not all sound alike
        self._clips: dict[str, list[bytes]] = {}
        self._gone: list[str] = []  # file names that would not load
        self._mode = (mode or "").strip().lower() or "voice"  # WAKE_ACK= is the default
        self._unknown = "" if self._mode in MODES else self._mode
        if self._unknown:
            self._mode = "voice"  # a typo must never cost him the acknowledgment
        self.last = ""  # the slug played last: never the same one twice in a row
        self.last_take: tuple[str, int] = ("", -1)  # ...and never the same take of it either
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
            paths = [self._dir / f"{slug}.wav"]
            with contextlib.suppress(OSError):
                paths += sorted(self._dir.glob(f"{slug}-[0-9]*.wav"))
            takes: list[bytes] = []
            for path in paths:
                pcm = b""
                with contextlib.suppress(Exception):  # a bad asset is a ding, not a crash
                    pcm = read_wav(path, self._rate)
                if pcm:
                    takes.append(pcm)
            if takes:
                self._clips[slug] = takes
            else:
                self._gone.append(f"{slug}.wav")

    def takes(self, slug: str) -> int:
        """How many ways she can say this line."""
        return len(self._clips.get(slug, ()))

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

    @property
    def beat_s(self) -> float:
        return self._beat_s

    def acknowledge(self, speaker: Any, *, beat_s: float | None = None) -> float:
        """Answer the wake through `speaker`, and say how long the answer runs.

        Returns the clip's length in seconds — what the runner holds the
        microphone shut for — or 0.0 when nothing was said, which means the
        caller still owes the ding (unless `silent`). `beat_s` overrides the
        pause queued ahead of the clip: 0 when the caller already waited it."""
        if self._mode != "voice" or speaker is None:
            return 0.0
        slug = self.pick()
        if not slug:
            return 0.0
        takes = self._clips[slug]
        index = self._rng.randrange(len(takes))
        if len(takes) > 1 and (slug, index) == self.last_take:
            index = (index + 1) % len(takes)  # the same take twice running is the old metronome
        pcm = takes[index]
        beat_s = self._beat_s if beat_s is None else max(0.0, float(beat_s))
        beat_s = beat_s + (self._rng.random() * self._beat_jitter_s if beat_s else 0.0)
        beat = b"\x00\x00" * int(beat_s * self._rate)
        try:
            if beat:
                speaker.enqueue(beat)  # the beat, on the same stream, so nothing can race it
            speaker.enqueue(pcm)
        except Exception:  # noqa: BLE001 — a speaker that won't take it falls back to the ding
            return 0.0
        self.last = slug
        self.last_take = (slug, index)
        self.last_beat_s = len(beat) / 2 / self._rate
        self.played.append(slug)
        del self.played[:-20]  # a days-long process keeps a window, not a history
        return (len(beat) + len(pcm)) / 2 / self._rate
