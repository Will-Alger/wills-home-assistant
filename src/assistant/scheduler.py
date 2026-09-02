"""Timers, alarms, and scheduled actions — things that happen later.

Owner-set, so they are URGENT: they speak even inside quiet hours. A timer or
alarm fires as an announcement (chime + her voice); a scheduled action runs a
home tool at the right moment and announces the result briefly. Everything is
persisted in data/schedule.json and survives restarts; recurring items compute
their next occurrence from a local "HH:MM" and a day list.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
KINDS = ("timer", "alarm", "reminder", "action", "briefing")


@dataclass
class Item:
    id: int
    kind: str  # timer | alarm | reminder | action
    label: str
    fire_at: float | None = None  # one-shot moment (timers, one-off reminders/actions)
    at: str = ""  # "HH:MM" local for recurring alarms/actions
    days: list[str] = field(default_factory=list)  # empty with `at` = every day
    message: str = ""  # what she says (reminders/alarms; action results otherwise)
    action: dict[str, Any] = field(default_factory=dict)  # {"tool": ..., "input": {...}}
    priority: str = "urgent"
    active: bool = True
    snoozed_until: float | None = None
    last_fired: float | None = None
    fired: int = 0
    created: float = field(default_factory=time.time)

    @property
    def recurring(self) -> bool:
        return bool(self.at)


def _hhmm(text: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", text or "")
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return (h, mi) if 0 <= h < 24 and 0 <= mi < 60 else None


def _next_occurrence(at: str, days: list[str], after: float) -> float | None:
    """The first local `at` on an allowed day strictly after `after`."""
    parsed = _hhmm(at)
    if parsed is None:
        return None
    hour, minute = parsed
    allowed = {d.lower()[:3] for d in days} or set(_DAYS)
    base = datetime.fromtimestamp(after).astimezone()
    for offset in range(8):
        day = (base + timedelta(days=offset)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        if day.timestamp() > after and _DAYS[day.weekday()] in allowed:
            return day.timestamp()
    return None


def spoken_time(ts: float) -> str:
    when = datetime.fromtimestamp(ts).astimezone()
    today = datetime.now(when.tzinfo).date()
    clock = when.strftime("%I:%M %p").lstrip("0")
    if when.date() == today:
        return f"today at {clock}"
    if when.date() == today + timedelta(days=1):
        return f"tomorrow at {clock}"
    return when.strftime("%a %b ") + str(when.day) + f" at {clock}"


class Scheduler:
    def __init__(
        self,
        path: Path,
        *,
        announcer: Any | None = None,
        executor: Any | None = None,
        now: Callable[[], float] = time.time,
        journal: Any | None = None,
    ) -> None:
        self._path = path
        self._announcer = announcer
        self._journal = journal
        self._executor = executor  # ToolExecutor for scheduled actions
        self.briefing: Callable[[], Any] | None = None  # async () -> str, set by the app
        self._now = now
        self._next_id = 1
        self._items: list[Item] = []
        self._stop = asyncio.Event()
        self._load()

    # ── persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._next_id = int(data.get("next_id", 1))
        known = set(Item.__dataclass_fields__)
        for row in data.get("items", []):
            if isinstance(row, dict) and "id" in row and "kind" in row:
                self._items.append(Item(**{k: v for k, v in row.items() if k in known}))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"next_id": self._next_id, "items": [asdict(i) for i in self._items]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, self._path)

    # ── creating ───────────────────────────────────────────────────────────

    def set_timer(self, seconds: float, label: str = "") -> Item:
        if seconds <= 0:
            raise ValueError("a timer needs a positive length")
        item = Item(
            id=self._next_id,
            kind="timer",
            label=" ".join(str(label).split()) or "timer",
            fire_at=self._now() + seconds,
            message="",
            created=self._now(),
        )
        return self._add(item)

    def set_alarm(self, at: str, *, days: list[str] | None = None, label: str = "", once: bool = False) -> Item:
        if _hhmm(at) is None:
            raise ValueError(f"an alarm time must look like 07:00 (got {at!r})")
        item = Item(
            id=self._next_id,
            kind="alarm",
            label=" ".join(str(label).split()) or "alarm",
            at="" if once else at,
            fire_at=_next_occurrence(at, days or [], self._now()) if once else None,
            days=[str(d) for d in (days or [])],
            created=self._now(),
        )
        return self._add(item)

    def schedule(
        self,
        *,
        kind: str,
        label: str,
        at: str = "",
        in_seconds: float | None = None,
        days: list[str] | None = None,
        message: str = "",
        action: dict[str, Any] | None = None,
        repeat: bool = False,
    ) -> Item:
        """A reminder (spoken) or an action (a home tool call) at a time or
        after a delay, optionally repeating daily / on given days."""
        if kind not in ("reminder", "action", "briefing"):
            raise ValueError("kind must be reminder, action, or briefing")
        if kind == "action" and not (action and action.get("tool")):
            raise ValueError("an action needs a tool to call")
        if kind == "reminder" and not message:
            raise ValueError("a reminder needs the message to say")
        if in_seconds is not None:
            if in_seconds <= 0:
                raise ValueError("the delay must be positive")
            fire_at, at_text = self._now() + in_seconds, ""
        else:
            if _hhmm(at) is None:
                raise ValueError(f"the time must look like 18:00 (got {at!r})")
            fire_at = None if repeat else _next_occurrence(at, days or [], self._now())
            at_text = at if repeat else ""
        item = Item(
            id=self._next_id,
            kind=kind,
            label=" ".join(str(label).split()) or kind,
            fire_at=fire_at,
            at=at_text,
            days=[str(d) for d in (days or [])],
            message=" ".join(str(message).split()),
            action=dict(action or {}),
            priority="urgent" if kind in ("reminder", "briefing") else "normal",
            created=self._now(),
        )
        return self._add(item)

    def _add(self, item: Item) -> Item:
        self._next_id += 1
        self._items.append(item)
        self._save()
        return item

    # ── managing ───────────────────────────────────────────────────────────

    def active(self) -> list[Item]:
        return [i for i in self._items if i.active]

    def get(self, item_id: int) -> Item | None:
        return next((i for i in self._items if i.id == int(item_id)), None)

    def cancel(self, item_id: int) -> Item | None:
        item = self.get(item_id)
        if item is None or not item.active:
            return None
        item.active = False
        self._save()
        return item

    def snooze(self, item_id: int | None, minutes: float) -> Item | None:
        """Push an alarm/reminder out; with no id, the one that fired last."""
        if item_id is None:
            fired = [i for i in self._items if i.kind in ("alarm", "reminder") and i.last_fired]
            item = max(fired, key=lambda i: i.last_fired or 0) if fired else None
        else:
            item = self.get(item_id)
        if item is None:
            return None
        item.snoozed_until = self._now() + max(1.0, minutes) * 60
        item.active = True
        self._save()
        return item

    def next_fire(self, item: Item) -> float | None:
        if not item.active:
            return None
        if item.snoozed_until and item.snoozed_until > (item.last_fired or 0):
            return item.snoozed_until
        if item.fire_at is not None:
            return item.fire_at
        if item.recurring:
            return _next_occurrence(item.at, item.days, item.last_fired or item.created)
        return None

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for item in self.active():
            nxt = self.next_fire(item)
            rows.append(
                {
                    "id": item.id,
                    "kind": item.kind,
                    "label": item.label,
                    "next": spoken_time(nxt) if nxt else "unscheduled",
                    "repeats": (", ".join(item.days) if item.days else "every day") if item.recurring else "",
                    **({"says": item.message} if item.message else {}),
                    **({"does": item.action.get("tool")} if item.action else {}),
                }
            )
        rows.sort(key=lambda r: r["next"])
        return rows

    # ── firing ─────────────────────────────────────────────────────────────

    async def tick(self) -> list[str]:
        """Fire everything due; returns what was announced (for tests/logs)."""
        now = self._now()
        said: list[str] = []
        for item in self.active():
            when = self.next_fire(item)
            if when is None or when > now:
                continue
            said.append(await self._fire(item, now))
        return said

    async def _fire(self, item: Item, now: float) -> str:
        item.fired += 1
        item.last_fired = now
        item.snoozed_until = None
        if not item.recurring:
            item.active = False
        if item.kind == "timer":
            text = f"Your {item.label} is up." if item.label != "timer" else "Your timer is up."
        elif item.kind == "alarm":
            text = (
                f"It's {datetime.fromtimestamp(now).astimezone().strftime('%I:%M %p').lstrip('0')} — "
                f"your {item.label}." + (" Say 'snooze' for ten more minutes." if item.label else "")
            )
        elif item.kind == "reminder":
            text = f"Reminder: {item.message}"
        elif item.kind == "briefing":
            text = await self._compose_briefing(item)
        else:
            text = await self._run_action(item)
        self._save()
        if self._journal is not None:
            with contextlib.suppress(Exception):
                self._journal.write("schedule", text, source=item.kind, data={"id": item.id})
        if self._announcer is not None:
            with contextlib.suppress(Exception):
                self._announcer.enqueue(
                    text,
                    kind=item.kind,
                    ref=f"schedule:{item.id}:{int(now)}",
                    priority=item.priority,
                    expires_in_s=6 * 3600,
                )
        return text

    async def _compose_briefing(self, item: Item) -> str:
        if self.briefing is None:
            return "Good morning. I don't have a briefing source wired up yet."
        try:
            text = await self.briefing()
        except Exception as err:  # noqa: BLE001 — a broken briefing still says something
            return f"Good morning. I couldn't put your briefing together: {str(err) or type(err).__name__}."
        return " ".join(str(text).split()) or "Good morning. Nothing on the books today."

    async def _run_action(self, item: Item) -> str:
        tool = str(item.action.get("tool", ""))
        payload = dict(item.action.get("input") or {})
        if self._executor is None:
            return f"I couldn't run the scheduled {item.label}: no home connection."
        try:
            result, is_error = await self._executor.execute(tool, payload)
        except Exception as err:  # noqa: BLE001 — a bad action must not kill the loop
            result, is_error = f"{type(err).__name__}: {err}", True
        if is_error:
            return f"The scheduled {item.label} failed: {str(result)[:160]}"
        return f"Done: {item.label}." + (f" {item.message}" if item.message else "")

    async def run(self, interval_s: float = 1.0) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                await self.tick()
            with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=interval_s)

    def stop(self) -> None:
        self._stop.set()
