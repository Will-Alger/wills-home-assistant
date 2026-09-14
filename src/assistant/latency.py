"""The turn latency log: how long each step of a turn actually took.

Until now we tuned her timing by feel, and a missing activation chime could
not be told apart from a missed wake word or a slow connection. Every turn
now leaves one small JSON row in `logs/turns.jsonl` — no transcripts, no
audio, just numbers — and the wake words that ALMOST fired leave one too, so
the threshold can be argued from real household audio instead of memory.

Every time in a row is monotonic seconds since the wake fired, or None when
that step never happened. A conversation's activation is stamped once, on its
first turn; later turns in the same conversation carry only their own timings
(repeating the activation would double-count it in every median).

What the numbers honestly mean:

* `chime_audible` is the moment PortAudio's callback first pulled the chime
  out of the buffer — when it could be heard, not when it was queued.
* an interruption's seconds run from the frame the wake phrase was detected
  in to the moment our queue was cleared. Audio already handed to PortAudio
  is not ours to see yet (that is the `truncate` work in phase 3).
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# A frame this loud is the wake word being said badly (or someone else's
# name); below it is room noise nobody meant as an activation.
NEAR_MISS_FLOOR = 0.2
_MISS_SETTLE_S = 1.0  # scores wobble: one utterance is one row, at its peak

# Set on the first turn of a conversation only.
_ACTIVATION = ("wake_score", "chime_enqueued", "chime_audible", "mic_ready", "connected")
_TURN = ("first_speech", "speech_end", "transcript_at", "first_call", "first_audio", "playback_end")


class TurnTrace:
    """One conversation's stopwatch. Created at the wake, stamped by the
    runner and the engine as the turn unfolds, written when it closes.

    A trace with no log writes nothing — that is the default inside the
    engine, so a session nobody is measuring costs a few dict writes."""

    def __init__(
        self,
        log: LatencyLog | None = None,
        *,
        wake_score: float | None = None,
        now: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self._log = log
        self._now = now
        self._wall = wall
        self.t0 = now()
        self.started = wall()
        self.session: int | None = None  # the data/sessions.json row id
        self.ended_by = ""
        self.rows: list[dict[str, Any]] = []  # finished turns, oldest first
        self._closed = False
        self._row = self._blank(0)
        self._row["wake_score"] = None if wake_score is None else round(float(wake_score), 3)

    # ── stamping ───────────────────────────────────────────────────────────

    def _blank(self, turn: int) -> dict[str, Any]:
        row: dict[str, Any] = {"kind": "turn", "turn": turn}
        if turn == 0:
            row.update(dict.fromkeys(_ACTIVATION))
        row.update(dict.fromkeys(_TURN))
        row["tools"] = []
        row["interruptions"] = 0
        row["interrupt_gaps"] = []
        return row

    def _at(self) -> float:
        return round(self._now() - self.t0, 3)

    def stamp(self, field: str) -> None:
        """First one wins: the step happened when it FIRST happened.

        A field the row does not carry is ignored, so a late stamp — the chime
        watcher is on its own thread — can never spill an activation time onto
        a later turn."""
        if field in self._row and self._row[field] is None:
            self._row[field] = self._at()

    def audible(self) -> None:
        """The chime reached the speaker callback (called from its thread)."""
        self.stamp("chime_audible")

    def connected(self) -> None:
        self.stamp("connected")

    def speech_started(self) -> None:
        """A user turn began. The second one rolls the row over."""
        if self._row["first_speech"] is not None:
            self.rows.append(self._row)
            self._row = self._blank(self._row["turn"] + 1)
        self.stamp("first_speech")

    def speech_stopped(self) -> None:
        self.stamp("speech_end")

    def transcribed(self) -> None:
        self.stamp("transcript_at")

    def audio_delta(self) -> None:
        self.stamp("first_audio")

    @property
    def wake_score(self) -> float | None:
        """How sure the detector was of the wake that opened this trace
        (None: a push-to-talk or an announcement opened it, or a later turn)."""
        score = self._row.get("wake_score") if self._row.get("turn") == 0 else None
        return None if score is None else float(score)

    def first_call(self) -> None:
        """The backend's first tool call of the turn arrived — how long the
        model took to decide, before any tool ran."""
        self.stamp("first_call")

    def playback_done(self) -> None:
        # Last one wins: a tool turn drains the speaker several times and it
        # is the final drain that ended the speaking.
        self._row["playback_end"] = self._at()

    def tool(self, name: str, seconds: float) -> None:
        self._row["tools"].append([str(name), round(float(seconds), 3)])

    def interrupted(self, seconds: float) -> None:
        self._row["interruptions"] += 1
        self._row["interrupt_gaps"].append(round(max(0.0, float(seconds)), 3))

    # ── closing ────────────────────────────────────────────────────────────

    def finish(self, ended_by: str = "", session: int | None = None) -> list[dict[str, Any]]:
        """Close the open turn and write every row. Safe to call twice."""
        if self._closed:
            return self.rows
        self._closed = True
        self.rows.append(self._row)
        self.ended_by = str(ended_by)
        if session is not None:
            self.session = int(session)
        for index, row in enumerate(self.rows):
            last = row.get("playback_end") or row.get("first_audio") or row.get("speech_end") or 0.0
            row["ts"] = round(self.started + last, 3)
            row["ended_by"] = self.ended_by
            row["session"] = self.session
            self.rows[index] = _trim(row)
        if self._log is not None:
            self._log.append(self.rows)
        return self.rows

    def compact(self) -> dict[str, Any]:
        """The last turn's timings for the session row — the same numbers,
        without the nulls and without what the session row already knows."""
        if not self.rows:
            return {}
        row = dict(self.rows[-1])
        for field in ("kind", "ts", "ended_by", "session"):
            row.pop(field, None)
        return {k: v for k, v in row.items() if v is not None}

    def console_note(self) -> str:
        """'first audio 0.9 s' for the last turn — the number he feels."""
        if not self.rows:
            return ""
        row = self.rows[-1]
        seconds = _answer_seconds(row)
        if seconds is None and row.get("speech_end") is None:
            seconds = row.get("first_audio")  # an announcement: nobody spoke, so from the wake
        return "" if seconds is None else f"first audio {seconds:.1f} s"


def _trim(row: dict[str, Any]) -> dict[str, Any]:
    """Rows are read by eye as often as by code: drop the empty fields."""
    return {k: v for k, v in row.items() if not (v == [] or (k == "interruptions" and not v))}


class LatencyLog:
    """Appends rows to logs/turns.jsonl and reads them back for the report."""

    def __init__(
        self,
        path: Path,
        *,
        max_rows: int = 20_000,
        now: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self._path = Path(path)
        self._max_rows = max(100, max_rows)
        self._now = now
        self._wall = wall
        self._peak: float | None = None  # a near miss building up
        self._peak_at = 0.0
        self._appended = 0

    # ── writing ────────────────────────────────────────────────────────────

    def wake(self, score: float | None = None) -> TurnTrace:
        """The wake fired (or an announcement opened a session): a new trace."""
        self._peak = None  # the run-up belongs to the wake, not to a near miss
        return TurnTrace(self, wake_score=score, now=self._now, wall=self._wall)

    def near_miss(self, score: float, threshold: float) -> None:
        """One idle frame that did not fire. Scores rise and fall across an
        utterance, so we keep the peak and write ONE row once it settles."""
        if score >= threshold:
            self._peak = None  # loud enough; only the cooldown held it back
            return
        if score >= NEAR_MISS_FLOOR:
            self._peak = max(self._peak or 0.0, float(score))
            self._peak_at = self._now()
            return
        if self._peak is not None and self._now() - self._peak_at >= _MISS_SETTLE_S:
            peak, self._peak = self._peak, None
            self.append(
                [
                    {
                        "kind": "wake_miss",
                        "ts": round(self._wall(), 3),
                        "wake_score": round(peak, 3),
                        "threshold": round(float(threshold), 3),
                    }
                ]
            )

    def append(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with contextlib.suppress(OSError):  # a log that won't write never breaks a turn
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, default=str) + "\n")
        self._appended += len(rows)
        if self._appended >= 200:  # an always-on process: keep the file bounded
            self._appended = 0
            self._trim_file()

    def _trim_file(self) -> None:
        with contextlib.suppress(OSError):
            lines = self._path.read_text(encoding="utf-8").splitlines()
            if len(lines) <= self._max_rows:
                return
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text("\n".join(lines[-self._max_rows // 2 :]) + "\n", encoding="utf-8")
            os.replace(tmp, self._path)

    # ── reading ────────────────────────────────────────────────────────────

    def read(self, since: float | None = None, until: float | None = None) -> list[dict[str, Any]]:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        rows: list[dict[str, Any]] = []
        for line in lines[-self._max_rows :]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written tail line is not worth a crash
            if not isinstance(row, dict):
                continue
            ts = row.get("ts")
            if since is not None and (ts is None or ts < since):
                continue
            if until is not None and (ts is None or ts >= until):
                continue
            rows.append(row)
        return rows


# ── the spoken report ──────────────────────────────────────────────────────


def _answer_seconds(row: dict[str, Any]) -> float | None:
    """Speech end → her first audio: the gap he actually experiences."""
    end, audio = row.get("speech_end"), row.get("first_audio")
    if end is None or audio is None or audio < end:
        return None
    return audio - end


def _chime_seconds(row: dict[str, Any]) -> float | None:
    """Wake → the chime he can hear (falling back to when it was queued)."""
    heard = row.get("chime_audible")
    return heard if heard is not None else row.get("chime_enqueued")


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def since_label(spec: str) -> str:
    """The window, said the way she would say it."""
    text = (spec or "today").strip().lower()
    if text in ("today", "yesterday"):
        return text
    if text in ("week", "this week", "last 7 days"):
        return "this week"
    try:
        hours = float(text)
    except ValueError:
        return f"on {text}"
    return "in the last hour" if hours == 1 else f"in the last {hours:g} hours"


def latency_report(rows: list[dict[str, Any]], label: str = "today") -> str:
    """One or two sentences she can read aloud: the medians and the slowest
    tool, or a plain admission that nothing has been measured yet."""
    turns = [r for r in rows if r.get("kind") == "turn"]
    chimes = [s for s in (_chime_seconds(r) for r in turns) if s is not None]
    answers = [s for s in (_answer_seconds(r) for r in turns) if s is not None]
    tools = [
        (str(name), float(seconds))
        for row in turns
        for name, seconds in row.get("tools") or []
    ]
    if not turns or not (chimes or answers):
        return f"I haven't timed any turns {label} yet."

    clauses = []
    if (chime := _median(chimes)) is not None:
        clauses.append(f"my chime came {chime:.1f} seconds after the wake word")
    if (answer := _median(answers)) is not None:
        clauses.append(f"I started answering {answer:.1f} seconds after you stopped talking")
    turn_count = len(answers) or len(chimes)
    plural = "turn" if turn_count == 1 else "turns"
    first = (
        f"{label.capitalize()}, {' and '.join(clauses)} — "
        f"the median over {turn_count} {plural}."
    )
    if not tools:
        return f"{first} No tools ran, so nothing slowed me down."
    name, seconds = max(tools, key=lambda pair: pair[1])
    return f"{first} My slowest step was {name}, at {seconds:.1f} seconds."
