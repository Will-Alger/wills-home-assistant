"""Record a session: both sides of the audio and every decision the engine
made, on one timeline, from "Start recording" until "End recording".

Debugging her turn-taking from the log after the fact is guesswork — the
log says "the reply to the fragment is dropped", not what the microphone
actually heard at that moment. A recording is ground truth: `mic.wav` is the
raw microphone, continuously, idle stretches included (so a false wake shows
what set it off, and a wake that never fired shows what she heard instead);
`speaker.wav` is everything the speaker played, as it played (her voice, the
chimes, the "Yes?", and the silence between); `events.jsonl` is one row per
engine decision with the numbers it used, `t` in seconds from the start of
both WAVs; `summary.json` lists the conversations inside it (kind, how each
ended, its transcript) and the owner's note.

The owner starts one from the panel ("Start recording", with a running
timer), by voice ("Alexa, record this"), or by clicking the test script's
first step; it runs across every conversation until "End recording" — or
`max_s`, so a forgotten one cannot fill the disk. Audio is written as raw
PCM while it runs and turned into WAV at the end; a crash leaves the PCM,
and the next start repairs it. Everything lives in `data/recordings/`,
which is never committed.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

_RATE = 24_000
_WIDTH = 2  # int16 mono
_KEEP = 50  # recordings kept; the oldest go when a new one starts
_MAX_S = 30 * 60  # a recording nobody ended stops itself here (~170 MB)

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
        "She answers \"Yes?\" and closes quietly on her own within about eight seconds. No ding, no reply to the room.",
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
    """One recording at a time, from start() to stop(), across conversations."""

    def __init__(
        self,
        root: Path,
        *,
        rate: int = _RATE,
        keep: int = _KEEP,
        max_s: float = _MAX_S,
        now: Callable[[], float] = time.time,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root = Path(root)
        self.rate = rate
        self.keep = keep
        self.max_s = max_s
        self._now = now
        self._clock = clock
        self._lock = threading.Lock()
        self._folder: Path | None = None
        self._mic: Any = None
        self._spk: Any = None
        self._spk_pending = bytearray()  # played bytes stashed by the audio thread
        self._events: Any = None
        self._t0 = 0.0
        self._started = 0.0
        self._count = 0
        self._steps = 0
        self._sessions: list[dict[str, Any]] = []
        self.stopped_by: str = ""  # why the last recording ended (the panel says so)
        with contextlib.suppress(Exception):
            self.repair()

    # ── state ──────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self._folder is not None

    @property
    def current(self) -> str:
        return self._folder.name if self._folder is not None else ""

    @property
    def elapsed(self) -> float:
        """Seconds since start() (0.0 when nothing is recording)."""
        return self._clock() - self._t0 if self._folder is not None else 0.0

    # ── the taps ───────────────────────────────────────────────────────────

    def mic(self, frame: bytes) -> None:
        """Every frame the microphone delivered — idle or in a conversation."""
        if self._folder is None:
            return
        with self._lock, contextlib.suppress(Exception):
            self._mic.write(frame)
            self._flush_spoken()
        if self._clock() - self._t0 > self.max_s:
            self.stop(reason=f"stopped itself after {self.max_s / 60:.0f} minutes")

    def spoke(self, pcm: bytes) -> None:
        """Everything the speaker actually played, as it played (her voice,
        chimes, the "Yes?", and the silence between). Called on the audio
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
        """The owner reached step `n` of the script."""
        if self._folder is None:
            return
        self._steps += 1
        self.event("script_step", n=n, say=say)

    # ── conversations inside the recording ─────────────────────────────────

    def session_started(self, kind: str = "wake") -> bool:
        """A conversation opened; True when it is being recorded."""
        if self._folder is None:
            return False
        self._sessions.append({"kind": kind, "t": round(self.elapsed, 3)})
        self.event("session", what="opened", session=kind)
        with contextlib.suppress(Exception):
            self._write_summary(self._folder, ended_by="recording")  # a crash mid-conversation still lists it
        return True

    def session_ended(
        self,
        *,
        ended_by: str = "",
        transcript: list[tuple[str, str]] | None = None,
        replied: bool = False,
    ) -> None:
        if self._folder is None:
            return
        lines = [list(pair) for pair in (transcript or [])]
        if self._sessions and "ended_by" not in self._sessions[-1]:
            self._sessions[-1].update(
                {
                    "t_end": round(self.elapsed, 3),
                    "ended_by": ended_by or "unknown",
                    "turns": sum(1 for role, _ in lines if role == "you"),
                    "replied": replied,
                    "transcript": lines,
                }
            )
        self.event("session", what="closed", by=ended_by, replied=replied)
        with contextlib.suppress(Exception):
            self._write_summary(self._folder, ended_by="recording")

    # ── start / stop ───────────────────────────────────────────────────────

    def start(self) -> str:
        """Open a recording; the name of the one already running if any."""
        if self._folder is not None:
            return self._folder.name
        self.root.mkdir(parents=True, exist_ok=True)
        self._prune()
        name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(self._now()))
        folder = self.root / name
        n = 2
        while folder.exists():
            folder = self.root / f"{name}-{n}"
            n += 1
        folder.mkdir(parents=True)
        with self._lock:
            self._mic = (folder / "mic.pcm").open("ab")
            self._spk = (folder / "speaker.pcm").open("ab")
            self._events = (folder / "events.jsonl").open("a", encoding="utf-8")
            self._spk_pending.clear()
            self._folder = folder
            self._t0 = self._clock()
            self._started = self._now()
            self._count = 0
            self._steps = 0
            self._sessions = []
            self.stopped_by = ""
        self._write_summary(folder, ended_by="recording")
        self.event("recording", what="started")
        return folder.name

    def stop(self, *, reason: str = "ended") -> dict[str, Any] | None:
        """Close the recording: WAVs out of the PCM, the summary written."""
        if self._folder is None:
            return None
        self.event("recording", what="stopped", by=reason)
        folder = self._folder
        with contextlib.suppress(Exception):
            self._write_summary(folder, ended_by=reason)  # the counts, while they are still ours
        with self._lock:
            self._flush_spoken()
            for handle in (self._mic, self._spk, self._events):
                with contextlib.suppress(Exception):
                    handle.close()
            self._folder = None
            self._mic = self._spk = self._events = None
        self.stopped_by = reason
        return self._finalize(folder, ended_by=reason)

    def _write_summary(self, folder: Path, **fields: Any) -> dict[str, Any]:
        summary = self._read_summary(folder)
        summary.update(
            {
                "name": folder.name,
                "started": _iso(self._started),
                "rate": self.rate,
                "sessions": self._sessions,
                "conversations": len(self._sessions),
                "turns": sum(int(s.get("turns", 0) or 0) for s in self._sessions),
                "steps": self._steps,
                "events": self._count,
                "elapsed_s": round(self.elapsed, 1),
                **fields,
            }
        )
        summary.setdefault("note", "")
        (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
        summary.setdefault("conversations", len(summary.get("sessions", []) or []))
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


def clock(seconds: float) -> str:
    """mm:ss (h:mm:ss past an hour) for the panel's timer."""
    seconds = max(0, int(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def describe(summary: dict[str, Any]) -> str:
    """One line for the panel's list."""
    started = str(summary.get("started", summary.get("name", "")))
    duration = float(summary.get("duration_s", 0) or 0)
    n = int(summary.get("conversations", 0) or 0)
    turns = int(summary.get("turns", 0) or 0)
    ended = str(summary.get("ended_by", "") or "")
    note = str(summary.get("note", "") or "")
    bits = [
        started,
        clock(duration),
        f"{n} conversation{'s' if n != 1 else ''}",
        f"{turns} turn{'s' if turns != 1 else ''}",
        ended if ended not in ("ended", "recording") else "",
    ]
    line = " · ".join(b for b in bits if b)
    return f"{line} — {note}" if note else line
