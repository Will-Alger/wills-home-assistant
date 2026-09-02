"""Presence: is the owner home? Fed by Home Assistant's person entity (the
companion app's GPS), debounced, with voice as proof.

Why she needs it: an announcement to an empty house is a wasted cent and a
lost message; a phone push while he is sitting on the couch is noise. Her
delivery policy (delivery.py) reads `state` and `settled()`; the Courier
turns transitions into the arrival welcome and follow-up triggers.

Rules:
- `observe()` records what HA (or a wake-word session) says; `tick()` promotes
  a change only after it has held for arrive_after_s / leave_after_s, so GPS
  flapping at the edge of the home zone never transitions.
- Talking to her IS being home: a wake session applies "home" at once.
- `settled()` adds a grace period after arriving (he is still parking) so the
  welcome and the held news are spoken together once he is in the door.
- `sync()` reads the entity on boot (no transition — she doesn't know what
  she missed) and on every websocket reconnect (a real change then goes
  through the normal debounce).
- `unknown` (no phone reporting) is treated as home by the policy: silence
  when he is home costs trust; a wasted announcement costs a cent.
Persisted in data/presence.json so "home since 6:12 PM" survives restarts.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_UNKNOWN = ("", "unknown", "unavailable", "none")


@dataclass(frozen=True)
class Transition:
    kind: str  # arrived | left
    at: float
    away_for_s: float | None = None  # arrived: how long he was gone


def _spoken_clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).astimezone().strftime("%I:%M %p").lstrip("0")


class Presence:
    def __init__(
        self,
        path: Path,
        entity: str,
        *,
        owner: str = "the owner",
        arrive_after_s: float = 60.0,
        leave_after_s: float = 300.0,
        settle_s: float = 90.0,
        now: Callable[[], float] = time.time,
        journal: Any | None = None,
    ) -> None:
        self._path = path
        self.entity = entity
        self._owner = owner
        self._arrive_after = max(0.0, arrive_after_s)
        self._leave_after = max(0.0, leave_after_s)
        self._settle = max(0.0, settle_s)
        self._now = now
        self._journal = journal
        self.state = "unknown"  # home | away | unknown
        self.since = now()
        self._candidate: str | None = None
        self._candidate_since = 0.0
        self.on_transition: list[Callable[[Transition], None]] = []
        self._load()

    # ── persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        state = str(data.get("state", "unknown"))
        if state in ("home", "away"):
            self.state = state
            self.since = float(data.get("since") or self._now())

    def _save(self) -> None:
        with contextlib.suppress(OSError):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"state": self.state, "since": self.since}), encoding="utf-8")
            os.replace(tmp, self._path)

    # ── observing ──────────────────────────────────────────────────────────

    @staticmethod
    def _wanted(raw_state: str | None) -> str | None:
        raw = str(raw_state or "").strip().lower()
        if raw in _UNKNOWN:
            return None
        return "home" if raw == "home" else "away"  # zone names ("Work") are away

    def observe(self, raw_state: str | None, ts: float | None = None, *, source: str = "ha") -> Transition | None:
        """Record what HA (or a voice session) says. Voice = home, at once."""
        want = self._wanted(raw_state)
        if want is None:
            return None
        ts = self._now() if ts is None else ts
        if source == "voice":
            self._candidate = None
            return self._apply("home", ts) if self.state != "home" else None
        if want == self.state:
            self._candidate = None  # a flap that came back: forget it
            return None
        if self._candidate != want:
            self._candidate = want
            self._candidate_since = ts
        return None

    def tick(self, now: float | None = None) -> Transition | None:
        """Promote a change that has held long enough."""
        if self._candidate is None:
            return None
        now = self._now() if now is None else now
        need = self._arrive_after if self._candidate == "home" else self._leave_after
        if now - self._candidate_since < need:
            return None
        return self._apply(self._candidate, now)

    def _apply(self, state: str, at: float) -> Transition | None:
        previous, previous_since = self.state, self.since
        self.state, self.since = state, at
        self._candidate = None
        self._save()
        if previous == "unknown" or previous == state:
            return None  # first knowledge: no story to tell
        kind = "arrived" if state == "home" else "left"
        away_for = (at - previous_since) if kind == "arrived" else None
        transition = Transition(kind=kind, at=at, away_for_s=away_for)
        if self._journal is not None:
            with contextlib.suppress(Exception):
                gone = f" after {away_for / 3600:.1f} h away" if away_for else ""
                self._journal.write("presence", f"{self._owner} {kind}{gone}", source=self.entity)
        for fn in list(self.on_transition):
            with contextlib.suppress(Exception):
                fn(transition)
        return transition

    async def sync(self, home: Any, *, boot: bool = False) -> None:
        """Read the entity now. Boot: take it as-is (no transition). Later
        (a websocket reconnect): a change goes through the normal debounce."""
        try:
            row = await home.get_entity(self.entity)
        except Exception:  # noqa: BLE001 — HA down: keep what we had
            return
        if not isinstance(row, dict):
            return
        want = self._wanted(row.get("state"))
        if want is None:
            return
        now = self._now()
        if boot or self.state == "unknown":
            since = now
            changed = row.get("last_changed")
            if changed:
                with contextlib.suppress(ValueError, TypeError):
                    since = datetime.fromisoformat(str(changed)).timestamp()
            self.state, self.since = want, min(since, now)
            self._candidate = None
            self._save()
            return
        self.observe(row.get("state"), now)

    # ── reading ────────────────────────────────────────────────────────────

    def settled(self, now: float | None = None) -> bool:
        """Home long enough for the welcome (he is in the door, not parking)."""
        if self.state != "home":
            return self.state == "unknown"
        now = self._now() if now is None else now
        return now - self.since >= self._settle

    @property
    def away_since(self) -> float | None:
        return self.since if self.state == "away" else None

    def describe(self, now: float | None = None) -> str:
        if self.state == "home":
            return f"{self._owner} is home (since {_spoken_clock(self.since)})"
        if self.state == "away":
            return f"{self._owner} is away (since {_spoken_clock(self.since)})"
        return f"{self._owner}'s whereabouts are unknown — his phone hasn't reported"
