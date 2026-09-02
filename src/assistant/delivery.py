"""How a notification reaches the owner: speak now, hold, or push to his
phone — decided deterministically, never by the model.

`DeliveryPolicy.decide(item, now)` is the Announcer's policy hook. It reads
presence (presence.py), quiet hours, a focus ("I'm on a call for an hour"),
and the per-kind preferences (`DeliverySettings`), in that order:

  inbox items      never speak; pushed only when urgent and he is away
  away             urgent → push; else the kind's away preference
                   (push | hold | inbox | journal; default: push for task,
                   question, watch, thought, system, followup; hold otherwise)
  home, unsettled  hold (he is still parking; welcome him once in the door)
  focus active     urgent → push, else hold
  quiet hours      urgent → speak, else the kind's quiet preference (hold)
  home             the kind's home preference (speak)

`Courier` is the one maintenance loop (2 s): presence debounce and the
arrival marker, phone pushes (with retries), and — every minute — escalation
of stale unread items, expiry, and applying inbox/journal preferences.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_PUSH_KINDS = frozenset({"task", "question", "watch", "thought", "system", "followup"})
_PUSH_RETRY_S = 60.0
_PUSH_MAX_TRIES = 3
_ALLOWED = {
    "away": ("push", "hold", "inbox", "journal"),
    "home": ("speak", "inbox", "hold", "journal"),
    "quiet": ("hold", "speak", "journal"),
}
_MEETING = re.compile(
    r"\b(meeting|call|standup|stand-up|1:1|1-1|interview|sync|demo|review)\b", re.IGNORECASE
)


class DeliverySettings:
    """Focus mode and per-kind preferences, persisted in data/delivery.json."""

    def __init__(self, path: Path, *, now: Callable[[], float] = time.time) -> None:
        self._path = path
        self._now = now
        self._focus: dict[str, Any] | None = None  # {"name", "until", "source"}
        self._prefs: dict[str, dict[str, str]] = {}  # kind -> {when: action}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        focus = data.get("focus")
        self._focus = dict(focus) if isinstance(focus, dict) and focus.get("until") else None
        prefs = data.get("preferences")
        if isinstance(prefs, dict):
            self._prefs = {
                str(k): {str(w): str(a) for w, a in v.items()} for k, v in prefs.items() if isinstance(v, dict)
            }

    def _save(self) -> None:
        with contextlib.suppress(OSError):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"focus": self._focus, "preferences": self._prefs}, indent=1), encoding="utf-8"
            )
            os.replace(tmp, self._path)

    # ── focus ──────────────────────────────────────────────────────────────

    def set_focus(self, name: str, minutes: float, *, source: str = "voice") -> dict[str, Any]:
        name = " ".join(str(name).split()) or "focus"
        minutes = max(1.0, float(minutes))
        self._focus = {"name": name, "until": self._now() + minutes * 60, "source": source}
        self._save()
        return dict(self._focus)

    def auto_focus(self, name: str, until: float) -> bool:
        """A calendar-derived focus; never overrides one the owner set by voice."""
        current = self.focus_active()
        if current is not None and current.get("source") == "voice":
            return False
        self._focus = {"name": name, "until": float(until), "source": "calendar"}
        self._save()
        return True

    def clear_focus(self, *, source: str | None = None) -> bool:
        if self._focus is None or (source is not None and self._focus.get("source") != source):
            return False
        self._focus = None
        self._save()
        return True

    def focus_active(self, now: float | None = None) -> dict[str, Any] | None:
        if self._focus is None:
            return None
        now = self._now() if now is None else now
        if float(self._focus.get("until", 0)) <= now:
            self._focus = None
            self._save()
            return None
        return dict(self._focus)

    # ── preferences ────────────────────────────────────────────────────────

    def set_preference(self, kind: str, when: str, action: str) -> None:
        kind, when, action = str(kind).strip().lower(), str(when).strip().lower(), str(action).strip().lower()
        if when not in _ALLOWED:
            raise ValueError("when must be away, home, or quiet")
        if action not in _ALLOWED[when]:
            raise ValueError(f"for {when} the action must be one of: {', '.join(_ALLOWED[when])}")
        if not kind:
            raise ValueError("a preference needs a notification kind (task, watch, action, ...)")
        self._prefs.setdefault(kind, {})[when] = action
        self._save()

    def clear_preference(self, kind: str, when: str | None = None) -> bool:
        kind = str(kind).strip().lower()
        if kind not in self._prefs:
            return False
        if when is None:
            del self._prefs[kind]
        else:
            self._prefs[kind].pop(str(when).strip().lower(), None)
            if not self._prefs[kind]:
                del self._prefs[kind]
        self._save()
        return True

    def preference(self, kind: str, when: str) -> str | None:
        """The owner's explicit choice for this kind and situation, if any."""
        for key in (kind, "*"):
            action = self._prefs.get(key, {}).get(when)
            if action:
                return action
        return None

    def describe(self) -> dict[str, Any]:
        now = self._now()
        focus = self.focus_active(now)
        return {
            "focus": (
                f"{focus['name']} for {max(0.0, (focus['until'] - now) / 60):.0f} more minutes"
                if focus
                else "none"
            ),
            "preferences": self._prefs,
        }

    def text(self) -> str:
        info = self.describe()
        parts = [f"focus: {info['focus']}"]
        for kind, rules in self._prefs.items():
            parts.append(f"{kind}: " + ", ".join(f"{w} → {a}" for w, a in rules.items()))
        return "; ".join(parts)


class DeliveryPolicy:
    def __init__(
        self,
        *,
        quiet: Callable[[float], bool],
        presence: Any | None = None,
        push_kinds: frozenset[str] = DEFAULT_PUSH_KINDS,
        settings: DeliverySettings | None = None,
    ) -> None:
        self._quiet = quiet
        self._presence = presence
        self._push_kinds = frozenset(push_kinds)
        self._settings = settings

    def _pref(self, kind: str, when: str, default: str) -> str:
        if self._settings is None:
            return default
        return self._settings.preference(kind, when) or default

    def decide(self, item: Any, now: float) -> str:
        """speak | hold | push | inbox | journal."""
        urgent = getattr(item, "priority", "normal") == "urgent"
        kind = str(getattr(item, "kind", ""))
        presence = self._presence
        away = presence is not None and presence.state == "away"
        if getattr(item, "mode", "speak") == "inbox":
            return "push" if (urgent and away) else "inbox"
        if away:
            if urgent:
                return "push"
            return self._pref(kind, "away", "push" if kind in self._push_kinds else "hold")
        if presence is not None and presence.state == "home" and not presence.settled(now):
            return "hold"
        if self._settings is not None and self._settings.focus_active(now) is not None:
            return "push" if urgent else "hold"
        if not urgent and self._quiet(now):
            return self._pref(kind, "quiet", "hold")
        if urgent:
            return "speak"
        return self._pref(kind, "home", "speak")


class Courier:
    """The one background loop for delivery: presence debounce, the arrival
    marker, phone pushes, escalation, expiry, and inbox/journal preferences."""

    def __init__(
        self,
        announcer: Any,
        *,
        presence: Any | None = None,
        pusher: Any | None = None,
        followups: Any | None = None,
        journal: Any | None = None,
        settings: DeliverySettings | None = None,
        calendar: Any | None = None,
        board: Any | None = None,
        owner: str = "the owner",
        escalate_after_s: float = 4 * 3600,
        unread_expire_s: float = 3 * 86400,
        nudge_after_s: float = 2 * 86400,
        focus_from_calendar: bool = False,
        now: Callable[[], float] = time.time,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._announcer = announcer
        self._presence = presence
        self._pusher = pusher
        self._followups = followups
        self._journal = journal
        self._settings = settings
        self._calendar = calendar
        self._board = board
        self._owner = owner
        self._escalate_after = escalate_after_s
        self._expire_after = unread_expire_s
        self._nudge_after = nudge_after_s
        self._focus_from_calendar = focus_from_calendar
        self._now = now
        self._log = log or (lambda _m: None)
        self._stop = asyncio.Event()
        self._push_tries: dict[int, tuple[int, float]] = {}  # id -> (tries, next_try)
        self._last_minute = 0.0
        self._last_calendar = 0.0
        self.transitions: list[Any] = []  # for tests/logs
        if presence is not None:
            presence.on_transition.append(self._on_transition)

    # ── presence ───────────────────────────────────────────────────────────

    def _on_transition(self, transition: Any) -> None:
        self.transitions.append(transition)
        self._log(f"presence: {self._owner} {transition.kind}")
        if self._followups is not None:
            with contextlib.suppress(Exception):
                self._followups.on_presence(transition)
        if transition.kind == "arrived" and self._announcer.pending():
            with contextlib.suppress(Exception):
                self._announcer.enqueue(
                    f"{self._owner} just got home.",
                    kind="presence",
                    ref=f"presence:arrived:{int(transition.at)}",
                    expires_in_s=600,
                )

    # ── pushes ─────────────────────────────────────────────────────────────

    async def _push_due(self, now: float) -> None:
        if self._pusher is None:
            return
        for item in self._announcer.pending_push(now):
            tries, next_try = self._push_tries.get(item.id, (0, 0.0))
            if now < next_try or tries >= _PUSH_MAX_TRIES:
                continue
            try:
                await self._pusher.push(item)
            except Exception as err:  # noqa: BLE001 — retry later, never crash
                self._push_tries[item.id] = (tries + 1, now + _PUSH_RETRY_S)
                self._log(f"phone push failed for notification {item.id}: {err}")
                continue
            self._push_tries.pop(item.id, None)
            self._announcer.mark_pushed(item.id)

    # ── the minute work: escalation, expiry, preferences ───────────────────

    async def _every_minute(self, now: float) -> None:
        apply = getattr(self._announcer, "apply_decisions", None)
        if apply is not None:
            with contextlib.suppress(Exception):
                apply(now)
        if self._board is not None:
            with contextlib.suppress(Exception):
                self._board.nudges(after_s=self._nudge_after)
        for item in list(self._announcer.unread()):
            age = now - (item.delivered or item.created)
            if age > self._expire_after:
                self._announcer.resolve_ids([item.id])
                if self._journal is not None:
                    with contextlib.suppress(Exception):
                        self._journal.write(
                            "notification", f"expired unread: {item.text[:120]}", source=item.kind,
                            data={"id": item.id},
                        )
                continue
            if (
                self._pusher is not None
                and item.pushed is None
                and item.priority != "urgent"
                and item.mode == "speak"
                and age > self._escalate_after
                and self._push_tries.get(item.id, (0, 0.0))[0] < _PUSH_MAX_TRIES
            ):
                try:
                    await self._pusher.push(item, level="passive")
                except Exception as err:  # noqa: BLE001
                    tries, _ = self._push_tries.get(item.id, (0, 0.0))
                    self._push_tries[item.id] = (tries + 1, now + _PUSH_RETRY_S)
                    self._log(f"escalation push failed for notification {item.id}: {err}")
                    continue
                self._announcer.mark_pushed(item.id)
        if self._journal is not None:
            with contextlib.suppress(Exception):
                self._journal.prune()

    async def _calendar_focus(self, now: float) -> None:
        if not (self._focus_from_calendar and self._calendar is not None and self._settings is not None):
            return
        from datetime import datetime, timedelta

        moment = datetime.fromtimestamp(now).astimezone()
        try:
            events = await self._calendar.list_events(moment - timedelta(hours=8), moment + timedelta(minutes=1))
        except Exception:  # noqa: BLE001 — the calendar being down never blocks delivery
            return
        busy_until: float | None = None
        for event in events:
            start, end = getattr(event, "start", None), getattr(event, "end", None)
            if getattr(event, "all_day", False) or not isinstance(start, datetime) or not isinstance(end, datetime):
                continue
            if start <= moment < end and _MEETING.search(str(getattr(event, "summary", ""))):
                busy_until = max(busy_until or 0.0, end.timestamp())
        if busy_until is not None:
            self._settings.auto_focus("meeting", busy_until)
        else:
            self._settings.clear_focus(source="calendar")

    # ── loop ───────────────────────────────────────────────────────────────

    async def tick(self) -> None:
        now = self._now()
        if self._presence is not None:
            with contextlib.suppress(Exception):
                self._presence.tick(now)
        if self._followups is not None:
            with contextlib.suppress(Exception):
                self._followups.tick(now)
        await self._push_due(now)
        if now - self._last_minute >= 60.0:
            self._last_minute = now
            await self._every_minute(now)
        if now - self._last_calendar >= 300.0:
            self._last_calendar = now
            await self._calendar_focus(now)

    async def run(self, interval_s: float = 2.0) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                await self.tick()
            with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=interval_s)

    def stop(self) -> None:
        self._stop.set()
