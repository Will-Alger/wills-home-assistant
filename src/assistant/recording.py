"""Record a session: both sides of the audio and every decision the engine
made, on one timeline.

Debugging her turn-taking from the log after the fact is guesswork — the
log says "the reply to the fragment is dropped", not what the microphone
actually heard at that moment. A recording is ground truth: `mic.wav` is the
raw microphone (every frame the pump saw, the gate's silence and the echo
window included — the events say what became of each), `speaker.wav` is
everything queued to the speaker (her voice, the chimes, the "Yes?"), and
`events.jsonl` is one row per engine decision with the numbers it used, `t`
in seconds from the start of the WAVs. `summary.json` is what the panel
lists: when, how long, how it ended, how many turns, and the owner's note.

Both WAVs start `preroll_s` BEFORE the wake — the last seconds of the room
while she was idle — so a false wake shows what set it off. Audio is written
as raw PCM while the session runs and turned into WAV at the end; a crash
leaves the PCM, and the next start repairs it into a readable file.

Opt-in (the panel's "Record sessions", or "Alexa, record this session"), and
it all lives in `data/recordings/`, which is never committed.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

_RATE = 24_000
_WIDTH = 2  # int16 mono
_PREROLL_S = 3.0
_KEEP = 50  # recordings kept; the oldest go when a new one starts

# The read-aloud test script: what to say, what should happen. Each step is
# stamped into the recording when the owner clicks "Next step", so the
# timeline can be read against the script.
SCRIPT: tuple[tuple[str, str], ...] = (
    (
        "Say nothing for ten seconds, with the room as it is.",
        "She must not wake. (A wake here is a false wake — note what the room was doing.)",
    ),
    (
        "\"Alexa\" — then say nothing at all.",
        "She answers \"Yes?\" and closes quietly on her own within about six seconds. No ding, no reply to the room.",
    ),
    (
        "\"Alexa, turn off the hallway light.\"",
        "A one-shot command: a brief confirmation, then she closes on her own.",
    ),
    (
        (
            "\"Alexa, I've been thinking about…\" (pause two seconds) \"…whether to repaint the living "
            "room, and\" (pause again) \"I'm not sure the cool white was a good idea.\""
        ),
        (
            "She waits through both pauses and answers ONCE, to the whole thought. A reply to a "
            "fragment, or a ding at a pause, is the cut-off bug."
        ),
    ),
    (
        "\"Alexa, let me think.\" Pause ten seconds. \"Okay, make it warmer.\"",
        "She waits the whole ten seconds without closing or answering, then acts on the second sentence.",
    ),
    (
        (
            "\"Alexa, explain how a heat pump works.\" Halfway through her answer, without the wake "
            "word, say: \"actually, never mind.\""
        ),
        "She stops within a beat of your voice and takes the new turn. Playback should hold, not stutter.",
    ),
    (
        "Ask her anything with a long answer, then cough or clap once while she speaks.",
        "At most a brief hold, then she carries on where she was. A cough that stops her is a false alarm.",
    ),
    (
        "\"Alexa, undo that.\"",
        "The hallway light comes back on (she remembers the last action across conversations).",
    ),
    (
        "\"Alexa, next time we talk, ask me how the demo went.\" Then: \"that's all.\"",
        "She confirms the follow-up, says a short goodbye, and closes — no listening ding after it.",
    ),
    (
        "Say \"Alexa\" again straight away.",
        "She answers at once. A dead window after a goodbye is a bug.",
    ),
    (
        "Hold Ctrl+Alt, say a command, let go.",
        "Same result as the wake word: the turn ends when you let go, not when the server decides.",
    ),
    (
        "\"Alexa, what are you waiting on me for?\"",
        "The demo question from step 9 is still pending, and she brings it up.",
    ),
)


class Recorder:
    """One recording at a time; armed or not, persisted beside the recordings."""

    def __init__(
        self,
        root: Path,
        *,
        rate: int = _RATE,
        preroll_s: float = _PREROLL_S,
        keep: int = _KEEP,
        now: Callable[[], float] = time.time,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root = Path(root)
        self.rate = rate
        self.keep = keep
        self._now = now
        self._clock = clock
        self._state = self.root / "state.json"
        self._lock = threading.Lock()
        frame_bytes = int(rate * 0.08) * _WIDTH  # the pump's 80 ms frames
        self._ring: deque[bytes] = deque(maxlen=max(1, int(preroll_s * rate * _WIDTH / frame_bytes)))
        self._folder: Path | None = None
        self._mic: Any = None
        self._spk: Any = None
        self._spk_pending = bytearray()  # played bytes stashed by the audio thread
        self._events: Any = None
        self._t0 = 0.0
        self._started = 0.0
        self._kind = ""
        self._pending: list[dict[str, Any]] = []  # script steps clicked before a session opened
        self._count = 0
        with contextlib.suppress(Exception):
            self.repair()

    # ── armed ──────────────────────────────────────────────────────────────

    @property
    def armed(self) -> bool:
        try:
            return bool(json.loads(self._state.read_text(encoding="utf-8")).get("armed"))
        except (OSError, ValueError):
            return False

    def arm(self, on: bool = True) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._state.write_text(json.dumps({"armed": bool(on)}), encoding="utf-8")
        if not on:
            self._ring.clear()

    def disarm(self) -> None:
        self.arm(False)

    @property
    def active(self) -> bool:
        return self._folder is not None

    @property
    def current(self) -> str:
        return self._folder.name if self._folder is not None else ""

    # ── the taps ───────────────────────────────────────────────────────────

    def mic(self, frame: bytes) -> None:
        """Every frame the microphone delivered — while idle it feeds the
        pre-roll ring, in a session it goes to disk."""
        if self._folder is None:
            if self.armed_cached():
                self._ring.append(frame)
            return
        with self._lock, contextlib.suppress(Exception):
            self._mic.write(frame)
            self._flush_spoken()

    def spoke(self, pcm: bytes) -> None:
        """Everything the speaker actually played (her voice, chimes, the
        "Yes?", and the silence between), as it played. Called on the audio
        thread, so it only stashes the bytes; the mic tap on the loop thread
        writes them out a few times a second."""
        if self._folder is None or not pcm:
            return
        with self._lock:
            self._spk_pending += pcm

    def _flush_spoken(self) -> None:
        """Under the lock: the speaker bytes stashed since the last write."""
        if self._spk_pending and self._spk is not None:
            with contextlib.suppress(Exception):
                self._spk.write(bytes(self._spk_pending))
                self._spk.flush()  # a crash keeps what was played up to the last frame
            self._spk_pending.clear()

    def event(self, kind: str, **fields: Any) -> None:
        """One timeline row. Anything, at any time: unknown fields are kept
        as they come, and a failure to write is never the engine's problem."""
        if self._folder is None:
            return
        row = {"t": round(self._clock() - self._t0, 3), "kind": kind, **fields}
        with self._lock, contextlib.suppress(Exception):
            self._flush_spoken()
            self._events.write(json.dumps(row, default=str) + "\n")
            self._events.flush()
            self._count += 1

    def step(self, n: int, say: str = "") -> None:
        """The owner reached step `n` of the script. Before a session opens
        it is held and stamped first thing when one does."""
        if self._folder is None:
            self._pending.append({"kind": "script_step", "n": n, "say": say})
            return
        self.event("script_step", n=n, say=say)

    # ── a session ──────────────────────────────────────────────────────────

    def begin(self, kind: str = "wake") -> str | None:
        """Open a recording (no-op unless armed, or when one is open)."""
        if self._folder is not None or not self.armed:
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        self._prune()
        name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(self._now()))
        folder = self.root / name
        n = 2
        while folder.exists():
            folder = self.root / f"{name}-{n}"
            n += 1
        folder.mkdir(parents=True)
        preroll = b"".join(self._ring)
        self._ring.clear()
        preroll_s = len(preroll) / (self.rate * _WIDTH)
        with self._lock:
            self._mic = (folder / "mic.pcm").open("ab")
            self._spk = (folder / "speaker.pcm").open("ab")
            self._events = (folder / "events.jsonl").open("a", encoding="utf-8")
            self._mic.write(preroll)
            self._spk.write(bytes(len(preroll)))  # silence: both sides share t = 0
            self._spk_pending.clear()
            self._folder = folder
            self._t0 = self._clock() - preroll_s
            self._started = self._now()
            self._kind = kind
            self._count = 0
        (folder / "summary.json").write_text(
            json.dumps(
                {
                    "name": folder.name,
                    "started": _iso(self._started),
                    "kind": kind,
                    "preroll_s": round(preroll_s, 3),
                    "rate": self.rate,
                    "ended_by": "recording",
                    "note": "",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.event("recording", session=kind, preroll_s=round(preroll_s, 3))
        for row in self._pending:
            self.event(**row)
        self._pending.clear()
        return folder.name

    def end(
        self,
        *,
        ended_by: str = "",
        transcript: list[tuple[str, str]] | None = None,
        replied: bool = False,
    ) -> dict[str, Any] | None:
        """Close the recording: WAVs out of the PCM, the summary written."""
        if self._folder is None:
            return None
        self.event("ended", by=ended_by)
        folder = self._folder
        with self._lock:
            self._flush_spoken()
            for handle in (self._mic, self._spk, self._events):
                with contextlib.suppress(Exception):
                    handle.close()
            self._folder = None
            self._mic = self._spk = self._events = None
        lines = [list(pair) for pair in (transcript or [])]
        summary = self._finalize(
            folder,
            ended_by=ended_by or "unknown",
            turns=sum(1 for role, _ in lines if role == "you"),
            replied=replied,
            transcript=lines,
            events=self._count,
        )
        return summary

    def _finalize(self, folder: Path, **fields: Any) -> dict[str, Any]:
        duration = 0.0
        for side in ("mic", "speaker"):
            pcm = folder / f"{side}.pcm"
            if pcm.exists():
                duration = max(duration, _to_wav(pcm, folder / f"{side}.wav", self.rate))
                with contextlib.suppress(OSError):
                    pcm.unlink()
        summary = self._read_summary(folder)
        summary.update(fields)
        summary["duration_s"] = round(duration, 1)
        summary.setdefault("note", "")
        with contextlib.suppress(OSError):
            (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def repair(self) -> int:
        """Finish recordings a crash left as raw PCM; returns how many."""
        fixed = 0
        if not self.root.exists():
            return 0
        for folder in self.root.iterdir():
            if folder.is_dir() and any(folder.glob("*.pcm")):
                self._finalize(folder, ended_by="crashed", events=_count_lines(folder / "events.jsonl"))
                fixed += 1
        return fixed

    def _prune(self) -> None:
        names = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        for name in names[: max(0, len(names) - self.keep + 1)]:
            shutil.rmtree(self.root / name, ignore_errors=True)

    # ── browsing ───────────────────────────────────────────────────────────

    def list(self) -> list[dict[str, Any]]:
        """Every finished recording, newest first."""
        out = []
        if not self.root.exists():
            return out
        for folder in sorted((p for p in self.root.iterdir() if p.is_dir()), reverse=True):
            if folder == self._folder:
                continue
            summary = self._read_summary(folder)
            if not summary:
                continue
            summary.setdefault("name", folder.name)
            summary["folder"] = str(folder)
            out.append(summary)
        return out

    def folder(self, name: str) -> Path | None:
        path = self.root / name
        return path if name and "/" not in name and "\\" not in name and path.is_dir() else None

    def note(self, name: str, text: str) -> bool:
        folder = self.folder(name)
        if folder is None:
            return False
        summary = self._read_summary(folder)
        summary["note"] = text.strip()
        (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return True

    def delete(self, name: str) -> bool:
        folder = self.folder(name)
        if folder is None or folder == self._folder:
            return False
        shutil.rmtree(folder, ignore_errors=True)
        return not folder.exists()

    def audio(self, name: str, side: str = "mic") -> tuple[bytes, int] | None:
        """(PCM16 mono bytes, rate) for the panel's play button."""
        folder = self.folder(name)
        if folder is None or side not in ("mic", "speaker"):
            return None
        path = folder / f"{side}.wav"
        try:
            with wave.open(str(path), "rb") as wav:
                return wav.readframes(wav.getnframes()), wav.getframerate()
        except (OSError, wave.Error):
            return None

    # ── internals ──────────────────────────────────────────────────────────

    _armed_at = 0.0
    _armed_value = False

    def armed_cached(self) -> bool:
        """The idle mic tap runs twelve times a second: read the flag from
        disk once a second, not once a frame."""
        now = self._clock()
        if now - self._armed_at > 1.0:
            self._armed_value = self.armed
            self._armed_at = now
        return self._armed_value

    @staticmethod
    def _read_summary(folder: Path) -> dict[str, Any]:
        try:
            data = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}


def _to_wav(pcm: Path, out: Path, rate: int) -> float:
    """Raw int16 mono -> WAV, streamed; returns the duration in seconds."""
    total = 0
    with wave.open(str(out), "wb") as wav, pcm.open("rb") as src:
        wav.setnchannels(1)
        wav.setsampwidth(_WIDTH)
        wav.setframerate(rate)
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            if len(chunk) % _WIDTH:
                chunk = chunk[: len(chunk) - len(chunk) % _WIDTH]
            wav.writeframes(chunk)
            total += len(chunk)
    return total / (rate * _WIDTH)


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def describe(summary: dict[str, Any]) -> str:
    """One line for the panel's list."""
    started = str(summary.get("started", summary.get("name", "")))
    duration = float(summary.get("duration_s", 0) or 0)
    turns = int(summary.get("turns", 0) or 0)
    ended = str(summary.get("ended_by", "") or "")
    kind = str(summary.get("kind", "") or "")
    note = str(summary.get("note", "") or "")
    bits = [started, f"{duration:.0f} s", kind, f"{turns} turn{'s' if turns != 1 else ''}", ended]
    line = " · ".join(b for b in bits if b)
    return f"{line} — {note}" if note else line
