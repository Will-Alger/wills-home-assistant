"""Announcements: things the assistant says on her own, later.

The endgame loop needs her to speak up when something finishes — a build,
a rollback, one day a timer or a sensor — without anyone having said the
wake word. This is the persisted queue behind that:

- `enqueue()` from inside the app (task completion, milestones).
- `write_inbox()` / the inbox directory for OTHER processes (the watchdog is
  stdlib-only and must never import the app to tell her it rolled back).
- `due()` is cheap enough to call on every 80 ms idle frame.
- Delivery is marked only after she actually spoke (`mark_delivered`), with
  backoff between attempts, so a dead network yields a late announcement
  rather than a lost one. Duplicates beat losses.
- Quiet hours hold normal announcements for morning; urgent ones bypass.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_BACKOFF_S = (0.0, 60.0, 300.0, 900.0)  # by attempt number; then cancelled
_SWEEP_EVERY_S = 2.0
_DUE_CACHE_S = 1.0


@dataclass
class Announcement:
    id: int
    text: str
    kind: str = "task"  # task | milestone | system | timer | sensor
    ref: str = ""  # dedupe key while undelivered, e.g. "job:calendar-3edbeb:done"
    priority: str = "normal"  # normal | urgent (urgent ignores quiet hours)
    created: float = field(default_factory=time.time)
    expires: float | None = None
    attempts: int = 0
    next_attempt: float = 0.0
    delivered: float | None = None
    cancelled: bool = False

    @property
    def open(self) -> bool:
        return self.delivered is None and not self.cancelled


def parse_quiet_hours(spec: str) -> tuple[int, int] | None:
    """'23:00-08:00' -> (start_minute, end_minute); '' -> None. Wraps overnight."""
    spec = (spec or "").strip()
    if not spec:
        return None
    try:
        start, end = spec.split("-", 1)
        sh, sm = (int(x) for x in start.strip().split(":"))
        eh, em = (int(x) for x in end.strip().split(":"))
    except ValueError as err:
        raise ValueError(f"quiet hours must look like 23:00-08:00, got {spec!r}") from err
    return sh * 60 + sm, eh * 60 + em


def in_quiet_hours(window: tuple[int, int] | None, when: datetime) -> bool:
    if window is None:
        return False
    start, end = window
    minute = when.hour * 60 + when.minute
    if start == end:
        return False
    if start < end:
        return start <= minute < end
    return minute >= start or minute < end  # overnight wrap


def write_inbox(
    data_dir: Path, text: str, *, kind: str = "system", ref: str = "", priority: str = "normal"
) -> Path:
    """Drop an announcement for the app from another process (stdlib only).
    Written to a temp name then renamed, so a half-written file is never read."""
    inbox = data_dir / "announcements" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    final = inbox / f"{time.time_ns()}.json"
    tmp = final.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"text": text, "kind": kind, "ref": ref, "priority": priority}),
        encoding="utf-8",
    )
    os.replace(tmp, final)
    return final


class Announcer:
    def __init__(
        self,
        path: Path,
        *,
        quiet_hours: str = "",
        max_attempts: int = 4,
        keep: int = 200,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._path = path
        self._inbox = path.parent / "announcements" / "inbox"
        self._quiet = parse_quiet_hours(quiet_hours)
        self._max_attempts = max(1, max_attempts)
        self._keep = keep
        self._now = now
        self._next_id = 1
        self._items: list[Announcement] = []
        self._last_sweep = 0.0
        self._due_cache: tuple[float, bool] = (-1.0, False)
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
        known = set(Announcement.__dataclass_fields__)
        for row in data.get("items", []):
            if isinstance(row, dict) and "id" in row and "text" in row:
                self._items.append(Announcement(**{k: v for k, v in row.items() if k in known}))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if len(self._items) > self._keep:
            closed = [a for a in self._items if not a.open]
            drop = {a.id for a in closed[: len(self._items) - self._keep]}
            self._items = [a for a in self._items if a.id not in drop]
        payload = {"next_id": self._next_id, "items": [asdict(a) for a in self._items]}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, self._path)

    # ── producing ──────────────────────────────────────────────────────────

    def enqueue(
        self,
        text: str,
        *,
        kind: str = "task",
        ref: str = "",
        priority: str = "normal",
        expires_in_s: float | None = None,
    ) -> Announcement:
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("an announcement needs text")
        if ref:
            for existing in self._items:
                if existing.open and existing.ref == ref:
                    return existing  # already queued; don't say it twice
        now = self._now()
        item = Announcement(
            id=self._next_id,
            text=text,
            kind=kind,
            ref=ref,
            priority="urgent" if priority == "urgent" else "normal",
            created=now,
            expires=(now + expires_in_s) if expires_in_s else None,
        )
        self._next_id += 1
        self._items.append(item)
        self._due_cache = (-1.0, False)
        self._save()
        return item

    def sweep_inbox(self) -> int:
        """Pull drops from other processes into the queue."""
        if not self._inbox.exists():
            return 0
        added = 0
        for path in sorted(self._inbox.glob("*.json")):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                self.enqueue(
                    str(row.get("text", "")),
                    kind=str(row.get("kind", "system")),
                    ref=str(row.get("ref", "")),
                    priority=str(row.get("priority", "normal")),
                )
                added += 1
            except (json.JSONDecodeError, OSError, ValueError, AttributeError):
                continue  # partial or junk file: leave it for the next sweep
            path.unlink(missing_ok=True)
        return added

    # ── consuming ──────────────────────────────────────────────────────────

    def is_quiet(self, now: float | None = None) -> bool:
        stamp = datetime.fromtimestamp(now if now is not None else self._now(), tz=UTC)
        return in_quiet_hours(self._quiet, stamp.astimezone())  # local wall clock

    def pending(self) -> list[Announcement]:
        now = self._now()
        return [a for a in self._items if a.open and (a.expires is None or a.expires > now)]

    def _deliverable(self, now: float) -> list[Announcement]:
        quiet = self.is_quiet(now)
        return [
            a
            for a in self.pending()
            if a.next_attempt <= now and (a.priority == "urgent" or not quiet)
        ]

    def due(self) -> bool:
        """Cheap enough for every idle frame: sweeps the inbox every 2 s and
        re-evaluates at most once per second."""
        now = self._now()
        if now - self._last_sweep >= _SWEEP_EVERY_S:
            self._last_sweep = now
            if self.sweep_inbox():
                self._due_cache = (-1.0, False)
        stamp, cached = self._due_cache
        if stamp >= 0 and now - stamp < _DUE_CACHE_S:
            return cached
        result = bool(self._deliverable(now))
        self._due_cache = (now, result)
        return result

    def take_due(self) -> list[Announcement]:
        """Hand out what should be spoken now, scheduling a retry for each in
        case delivery fails. Items past max attempts or expiry are cancelled."""
        now = self._now()
        changed = False
        for item in self._items:
            if item.open and item.expires is not None and item.expires <= now:
                item.cancelled = True
                changed = True
        out: list[Announcement] = []
        for item in self._deliverable(now):
            item.attempts += 1
            changed = True
            if item.attempts > self._max_attempts:
                item.cancelled = True
                continue
            item.next_attempt = now + _BACKOFF_S[min(item.attempts, len(_BACKOFF_S) - 1)]
            out.append(item)
        if changed:
            self._due_cache = (-1.0, False)
            self._save()
        return out

    def mark_delivered(self, ids: list[int]) -> None:
        now = self._now()
        wanted = set(ids)
        for item in self._items:
            if item.id in wanted and item.delivered is None:
                item.delivered = now
        self._due_cache = (-1.0, False)
        self._save()

    def cancel(self, ref_prefix: str) -> int:
        count = 0
        for item in self._items:
            if item.open and item.ref.startswith(ref_prefix):
                item.cancelled = True
                count += 1
        if count:
            self._due_cache = (-1.0, False)
            self._save()
        return count

    def history(self, since: float | None = None) -> list[dict[str, Any]]:
        rows = [asdict(a) for a in self._items if a.delivered is not None]
        if since is not None:
            rows = [r for r in rows if (r["delivered"] or 0) >= since]
        return rows
