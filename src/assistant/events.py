"""Event reactivity: she watches the house and speaks up when something happens.

Home Assistant pushes every state change over its websocket (verified live
2026-09-02 against HA 2026.8.3: `auth_required` → `auth` → `auth_ok`, then
`subscribe_events` with `event_type: state_changed`; each event carries
`entity_id`, `old_state`, `new_state`). A Watch is a standing rule the owner
set by voice — "tell me when the front door opens after 11pm" — evaluated
here deterministically; a hit becomes an announcement (announce.py), so the
same quiet-hours / urgency / delivery rules apply.
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
from datetime import datetime
from pathlib import Path
from typing import Any

_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass
class Watch:
    id: int
    entity_id: str  # exact id, or a fragment ("front_door")
    message: str  # what she says; {entity} and {state} are filled in
    to_state: str = ""  # "" = any change
    from_state: str = ""
    after: str = ""  # "HH:MM" local; with `before` forms a window (wraps overnight)
    before: str = ""
    days: list[str] = field(default_factory=list)  # ["mon", ...]; empty = every day
    once: bool = True  # fire once then retire
    priority: str = "normal"  # normal | urgent (urgent ignores quiet hours)
    active: bool = True
    created: float = field(default_factory=time.time)
    fired: int = 0
    last_fired: float | None = None


def _minutes(hhmm: str) -> int | None:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", hhmm or "")
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def in_window(after: str, before: str, when: datetime) -> bool:
    start, end = _minutes(after), _minutes(before)
    minute = when.hour * 60 + when.minute
    if start is None and end is None:
        return True
    if start is None:
        return minute < end  # type: ignore[operator]
    if end is None:
        return minute >= start
    if start <= end:
        return start <= minute < end
    return minute >= start or minute < end  # overnight wrap


def matches(watch: Watch, entity_id: str, old: str | None, new: str | None, now: float) -> bool:
    if not watch.active:
        return False
    needle = watch.entity_id.strip().lower()
    if needle != entity_id.lower() and needle not in entity_id.lower():
        return False
    if old == new:
        return False  # attribute-only updates are not a state change
    if watch.to_state and (new or "").lower() != watch.to_state.lower():
        return False
    if watch.from_state and (old or "").lower() != watch.from_state.lower():
        return False
    when = datetime.fromtimestamp(now).astimezone()
    if watch.days and _DAYS[when.weekday()] not in [d.lower()[:3] for d in watch.days]:
        return False
    return in_window(watch.after, watch.before, when)


def render(watch: Watch, entity_id: str, new: str | None) -> str:
    pretty = entity_id.split(".", 1)[-1].replace("_", " ")
    return watch.message.replace("{entity}", pretty).replace("{state}", str(new or "unknown"))


class WatchStore:
    def __init__(self, path: Path, *, now: Callable[[], float] = time.time) -> None:
        self._path = path
        self._now = now
        self._next_id = 1
        self._watches: list[Watch] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._next_id = int(data.get("next_id", 1))
        known = set(Watch.__dataclass_fields__)
        for row in data.get("watches", []):
            if isinstance(row, dict) and "id" in row and "entity_id" in row:
                self._watches.append(Watch(**{k: v for k, v in row.items() if k in known}))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"next_id": self._next_id, "watches": [asdict(w) for w in self._watches]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, self._path)

    def add(self, **fields: Any) -> Watch:
        entity = " ".join(str(fields.get("entity_id", "")).split())
        message = " ".join(str(fields.get("message", "")).split())
        if not entity or not message:
            raise ValueError("a watch needs an entity and what to say")
        for key in ("after", "before"):
            value = str(fields.get(key, "") or "")
            if value and _minutes(value) is None:
                raise ValueError(f"{key} must look like 23:00 (got {value!r})")
        watch = Watch(
            id=self._next_id,
            entity_id=entity,
            message=message,
            to_state=str(fields.get("to_state", "") or ""),
            from_state=str(fields.get("from_state", "") or ""),
            after=str(fields.get("after", "") or ""),
            before=str(fields.get("before", "") or ""),
            days=[str(d) for d in (fields.get("days") or [])],
            once=bool(fields.get("once", True)),
            priority="urgent" if fields.get("priority") == "urgent" else "normal",
            created=self._now(),
        )
        self._next_id += 1
        self._watches.append(watch)
        self._save()
        return watch

    def active(self) -> list[Watch]:
        return [w for w in self._watches if w.active]

    def all(self) -> list[Watch]:
        return list(self._watches)

    def cancel(self, watch_id: int) -> Watch | None:
        for w in self._watches:
            if w.id == int(watch_id) and w.active:
                w.active = False
                self._save()
                return w
        return None

    def evaluate(self, entity_id: str, old: str | None, new: str | None) -> list[tuple[Watch, str]]:
        """Which watches fire for this state change, with their rendered text."""
        now = self._now()
        hits: list[tuple[Watch, str]] = []
        for w in self._watches:
            if matches(w, entity_id, old, new, now):
                w.fired += 1
                w.last_fired = now
                if w.once:
                    w.active = False
                hits.append((w, render(w, entity_id, new)))
        if hits:
            self._save()
        return hits

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "id": w.id,
                "entity": w.entity_id,
                "when": " ".join(
                    part
                    for part in (
                        f"turns {w.to_state}" if w.to_state else "changes",
                        f"from {w.from_state}" if w.from_state else "",
                        f"after {w.after}" if w.after else "",
                        f"before {w.before}" if w.before else "",
                        f"on {', '.join(w.days)}" if w.days else "",
                    )
                    if part
                ),
                "say": w.message,
                "once": w.once,
                "priority": w.priority,
                "fired": w.fired,
            }
            for w in self.active()
        ]


class EventWatcher:
    """Keeps a websocket to Home Assistant open and feeds state changes to the
    watch store; hits become announcements. Reconnects with backoff forever."""

    def __init__(
        self,
        url: str,
        token: str,
        store: WatchStore,
        announcer: Any,
        *,
        connector: Callable[..., Any] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._url = url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
        self._token = token
        self._store = store
        self._announcer = announcer
        self._connector = connector
        self._log = log or (lambda _m: None)
        self._stop = asyncio.Event()
        self.events_seen = 0

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        backoff = 5.0
        while not self._stop.is_set():
            try:
                await self._session()
                backoff = 5.0
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 — never let the house-watch die
                self._log(f"event watcher: {type(err).__name__}: {err} — retrying in {backoff:.0f}s")
            if self._stop.is_set():
                return
            with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=backoff)
            backoff = min(backoff * 2, 60.0)

    async def _session(self) -> None:
        connector = self._connector
        if connector is None:
            import websockets

            connector = websockets.connect
        async with connector(f"{self._url}/api/websocket") as ws:
            hello = json.loads(await ws.recv())
            if hello.get("type") != "auth_required":
                raise RuntimeError(f"unexpected greeting {hello!r}")
            await ws.send(json.dumps({"type": "auth", "access_token": self._token}))
            auth = json.loads(await ws.recv())
            if auth.get("type") != "auth_ok":
                raise RuntimeError(f"Home Assistant refused the token: {auth.get('message', auth)}")
            await ws.send(json.dumps({"id": 1, "type": "subscribe_events", "event_type": "state_changed"}))
            self._log("event watcher: connected, watching state changes")
            while not self._stop.is_set():
                raw = await ws.recv()
                if raw is None:
                    return
                self.handle(raw)

    def handle(self, raw: str) -> list[str]:
        """Feed one websocket message; returns the announcement texts it produced."""
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if msg.get("type") != "event":
            return []
        data = (msg.get("event") or {}).get("data") or {}
        entity_id = str(data.get("entity_id") or "")
        if not entity_id:
            return []
        old = (data.get("old_state") or {}).get("state")
        new = (data.get("new_state") or {}).get("state")
        self.events_seen += 1
        said: list[str] = []
        for watch, text in self._store.evaluate(entity_id, old, new):
            with contextlib.suppress(Exception):
                self._announcer.enqueue(
                    text,
                    kind="watch",
                    ref=f"watch:{watch.id}:{int(time.time())}",
                    priority=watch.priority,
                    expires_in_s=12 * 3600,
                )
            said.append(text)
        return said
