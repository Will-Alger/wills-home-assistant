"""Announcements and notifications: things the assistant says on her own,
later — and whether the owner actually heard them.

The endgame loop needs her to speak up when something finishes — a build,
a rollback, a timer, a sensor — without anyone having said the wake word.
This is the persisted queue behind that and, since milestone 11, the
notification store:

- `enqueue()` from inside the app (task completion, milestones, watches...).
- `write_inbox()` / the inbox directory for OTHER processes (the watchdog is
  stdlib-only and must never import the app to tell her it rolled back).
- `due()` is cheap enough to call on every 80 ms idle frame.
- `delivered` means SPOKEN. It is set only after she actually said it
  (`mark_delivered`), with backoff between attempts, so a dead network yields
  a late announcement rather than a lost one. Duplicates beat losses.
- `read` means the owner consumed it: he replied, tapped it on his phone, or
  she listed it to him (`mark_read`). Spoken-but-unread items are recapped at
  the next conversation. `mode="inbox"` items never open a session; they wait
  to be listed. Timers, alarms and briefings are read the moment they are
  spoken — nobody wants "your pasta timer is up" in tomorrow's recap.
- A delivery `policy` (delivery.py) decides speak / hold / push per item from
  presence, quiet hours and focus. Without one, quiet hours hold normal
  items and urgent ones bypass — the original behavior.
- `subscribe()` observers (the journal, the phone) see every transition.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_BACKOFF_S = (0.0, 60.0, 300.0, 900.0)  # by attempt number; then parked in the inbox
_SWEEP_EVERY_S = 2.0
_DUE_CACHE_S = 1.0
_GROUP_WINDOW_S = 600.0  # repeats with the same explicit group merge within this
_EPHEMERAL_KINDS = frozenset({"timer", "alarm", "briefing", "presence"})  # read once spoken

Policy = Callable[["Announcement", float], str]  # -> speak | hold | push | inbox


@dataclass
class Announcement:
    id: int
    text: str
    kind: str = "task"  # task | milestone | system | timer | alarm | reminder | action | briefing | watch | thought | question | presence | followup | nudge
    ref: str = ""  # dedupe key while undelivered, e.g. "task:7:2:built"
    priority: str = "normal"  # normal | urgent (urgent ignores quiet hours)
    created: float = field(default_factory=time.time)
    expires: float | None = None
    attempts: int = 0
    next_attempt: float = 0.0
    delivered: float | None = None  # when she SPOKE it — not proof he heard it
    cancelled: bool = False
    # milestone 11: notifications
    mode: str = "speak"  # speak | inbox (inbox never opens a session)
    read: float | None = None  # when the owner consumed it
    pushed: float | None = None  # when it went to his phone
    resolved: float | None = None  # superseded (task approved, revision rebuilt, ...)
    group: str = ""  # explicit grouping key; "" = never grouped
    count: int = 1  # grouped repeats
    context: dict[str, Any] = field(default_factory=dict)  # {"task_id": 7} etc.
    actions: list[str] = field(default_factory=list)  # phone buttons: approve, later, ...

    @property
    def live(self) -> bool:
        """Still something for the owner: not read, resolved, or cancelled."""
        return not self.cancelled and self.resolved is None and self.read is None

    @property
    def open(self) -> bool:
        """Pending SPEECH: live, meant to be spoken, not spoken yet."""
        return self.live and self.mode == "speak" and self.delivered is None

    @property
    def unread(self) -> bool:
        return self.live

    @property
    def state(self) -> str:
        if self.cancelled:
            return "cancelled"
        if self.resolved is not None:
            return "resolved"
        if self.read is not None:
            return "read"
        if self.delivered is not None:
            return "spoken"
        if self.mode == "inbox":
            return "inbox"
        return "pending"


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


def spoken_stamp(ts: float) -> str:
    """'Tue 9:12 PM' — how a notification's time is read back."""
    when = datetime.fromtimestamp(ts).astimezone()
    return f"{when.strftime('%a')} {when.strftime('%I:%M %p').lstrip('0')}"


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
        policy: Policy | None = None,
    ) -> None:
        self._path = path
        self._inbox = path.parent / "announcements" / "inbox"
        self._quiet = parse_quiet_hours(quiet_hours)
        self._max_attempts = max(1, max_attempts)
        self._keep = keep
        self._now = now
        self._policy = policy
        self._subscribers: list[Callable[[Announcement, str], None]] = []
        self._next_id = 1
        self._items: list[Announcement] = []
        self._last_sweep = 0.0
        self._due_cache: tuple[float, bool] = (-1.0, False)
        self._load()

    # ── wiring ─────────────────────────────────────────────────────────────

    @property
    def policy(self) -> Policy | None:
        return self._policy

    @policy.setter
    def policy(self, value: Policy | None) -> None:
        self._policy = value
        self._due_cache = (-1.0, False)

    def subscribe(self, fn: Callable[[Announcement, str], None]) -> None:
        """Observe transitions: created, grouped, spoken, read, unread,
        pushed, resolved, cancelled, parked."""
        self._subscribers.append(fn)

    def _emit(self, item: Announcement, event: str) -> None:
        for fn in self._subscribers:
            with contextlib.suppress(Exception):
                fn(item, event)

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
            # trim only what the owner is finished with; an unread item must
            # keep its id (a phone tap or "mark it read" may still name it)
            closed = [a for a in self._items if not a.live]
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
        mode: str = "speak",
        group: str = "",
        context: dict[str, Any] | None = None,
        actions: Iterable[str] | None = None,
    ) -> Announcement:
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("an announcement needs text")
        now = self._now()
        if ref:
            for existing in self._items:
                if existing.open and existing.ref == ref:
                    return existing  # already queued; don't say it twice
        if group:
            for existing in self._items:
                if existing.live and existing.group == group and now - existing.created < _GROUP_WINDOW_S:
                    existing.count += 1
                    existing.text = text
                    existing.next_attempt = 0.0
                    self._due_cache = (-1.0, False)
                    self._save()
                    self._emit(existing, "grouped")
                    return existing
        item = Announcement(
            id=self._next_id,
            text=text,
            kind=kind,
            ref=ref,
            priority="urgent" if priority == "urgent" else "normal",
            created=now,
            expires=(now + expires_in_s) if expires_in_s else None,
            mode="inbox" if mode == "inbox" else "speak",
            group=str(group or ""),
            context=dict(context or {}),
            actions=[str(a) for a in (actions or [])],
        )
        self._next_id += 1
        self._items.append(item)
        self._due_cache = (-1.0, False)
        self._save()
        self._emit(item, "created")
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
                    mode=str(row.get("mode", "speak")),
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
        """Waiting to be SPOKEN (not expired)."""
        now = self._now()
        return [a for a in self._items if a.open and (a.expires is None or a.expires > now)]

    def _decision(self, item: Announcement, now: float, quiet: bool) -> str:
        if self._policy is not None:
            with contextlib.suppress(Exception):  # a broken policy must not silence her
                return str(self._policy(item, now))
        if quiet and item.priority != "urgent":
            return "hold"
        return "speak"

    def _deliverable(self, now: float) -> list[Announcement]:
        quiet = self.is_quiet(now)
        return [
            a
            for a in self.pending()
            if a.next_attempt <= now and self._decision(a, now, quiet) == "speak"
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
        case delivery fails. Expired items are cancelled; items past the
        attempt cap are parked in the inbox (never lost, just not spoken)."""
        now = self._now()
        changed = False
        for item in self._items:
            if item.open and item.expires is not None and item.expires <= now:
                item.cancelled = True
                changed = True
                self._emit(item, "cancelled")
        out: list[Announcement] = []
        for item in self._deliverable(now):
            item.attempts += 1
            changed = True
            if item.attempts > self._max_attempts:
                item.mode = "inbox"
                self._emit(item, "parked")
                continue
            item.next_attempt = now + _BACKOFF_S[min(item.attempts, len(_BACKOFF_S) - 1)]
            out.append(item)
        if changed:
            self._due_cache = (-1.0, False)
            self._save()
        return out

    def apply_decisions(self, now: float | None = None) -> int:
        """Preferences that change what an item IS: 'inbox' parks a speak
        item (never spoken, listed later); 'journal' resolves it at once
        (recorded, never raised). Returns how many changed."""
        if self._policy is None:
            return 0
        now = self._now() if now is None else now
        quiet = self.is_quiet(now)
        changed = 0
        for item in self._items:
            if not item.live:
                continue
            decision = self._decision(item, now, quiet)
            if decision == "journal":
                item.resolved = now
                changed += 1
                self._emit(item, "journaled")
            elif decision == "inbox" and item.mode == "speak" and item.delivered is None:
                item.mode = "inbox"
                changed += 1
                self._emit(item, "parked")
        if changed:
            self._due_cache = (-1.0, False)
            self._save()
        return changed

    def pending_push(self, now: float | None = None) -> list[Announcement]:
        """Live items the policy wants on his phone that haven't gone yet."""
        if self._policy is None:
            return []
        now = self._now() if now is None else now
        quiet = self.is_quiet(now)
        return [
            a
            for a in self._items
            if a.live and a.pushed is None and self._decision(a, now, quiet) == "push"
        ]

    # ── state transitions ──────────────────────────────────────────────────

    def get(self, item_id: int) -> Announcement | None:
        return next((a for a in self._items if a.id == int(item_id)), None)

    def _select(self, ids: Iterable[int]) -> list[Announcement]:
        wanted = {int(i) for i in ids}
        return [a for a in self._items if a.id in wanted]

    def mark_delivered(self, ids: list[int]) -> None:
        """She SPOKE these. Ephemeral kinds count as read too."""
        now = self._now()
        for item in self._select(ids):
            if item.delivered is None:
                item.delivered = now
                if item.kind in _EPHEMERAL_KINDS and item.read is None:
                    item.read = now
                self._emit(item, "spoken")
        self._due_cache = (-1.0, False)
        self._save()

    mark_spoken = mark_delivered

    def mark_read(self, ids: Iterable[int]) -> int:
        """The owner consumed these (replied, tapped, or was read the list)."""
        now = self._now()
        count = 0
        for item in self._select(ids):
            if item.read is None and not item.cancelled:
                item.read = now
                count += 1
                self._emit(item, "read")
        if count:
            self._due_cache = (-1.0, False)
            self._save()
        return count

    def mark_unread(self, ids: Iterable[int]) -> int:
        count = 0
        for item in self._select(ids):
            if item.read is not None:
                item.read = None
                count += 1
                self._emit(item, "unread")
        if count:
            self._due_cache = (-1.0, False)
            self._save()
        return count

    def mark_pushed(self, item_id: int) -> None:
        item = self.get(item_id)
        if item is not None and item.pushed is None:
            item.pushed = self._now()
            self._save()
            self._emit(item, "pushed")

    def resolve(self, ref_prefix: str) -> int:
        """Supersede live items by ref prefix (kept in history, not unread)."""
        now = self._now()
        count = 0
        for item in self._items:
            if item.live and ref_prefix and item.ref.startswith(ref_prefix):
                item.resolved = now
                count += 1
                self._emit(item, "resolved")
        if count:
            self._due_cache = (-1.0, False)
            self._save()
        return count

    def resolve_ids(self, ids: Iterable[int]) -> int:
        now = self._now()
        count = 0
        for item in self._select(ids):
            if item.live:
                item.resolved = now
                count += 1
                self._emit(item, "resolved")
        if count:
            self._due_cache = (-1.0, False)
            self._save()
        return count

    def cancel(self, ref_prefix: str) -> int:
        count = 0
        for item in self._items:
            if item.open and item.ref.startswith(ref_prefix):
                item.cancelled = True
                count += 1
                self._emit(item, "cancelled")
        if count:
            self._due_cache = (-1.0, False)
            self._save()
        return count

    # ── reading back ───────────────────────────────────────────────────────

    def unread(self, *, kinds: Iterable[str] | None = None) -> list[Announcement]:
        wanted = {k for k in kinds} if kinds else None
        now = self._now()
        rows = [
            a
            for a in self._items
            if a.live
            and (wanted is None or a.kind in wanted)
            and (a.expires is None or a.expires > now or a.delivered is not None)
        ]
        rows.sort(key=lambda a: a.delivered or a.created)
        return rows

    def unread_summary(self, limit: int = 3) -> str:
        """For her instructions: a count and a teaser, never the full texts.
        Items about to be spoken anyway (deliverable now) are left out."""
        now = self._now()
        soon = {a.id for a in self._deliverable(now)}
        rows = [a for a in self.unread() if a.id not in soon]
        if not rows:
            return "none"
        oldest = min(a.delivered or a.created for a in rows)
        head = "; ".join(f"[{a.id}] {a.kind}: {a.text[:70]}" for a in rows[:limit])
        more = f" — and {len(rows) - limit} more" if len(rows) > limit else ""
        return f"{len(rows)} unread since {spoken_stamp(oldest)}: {head}{more}"

    def last_spoken(self) -> Announcement | None:
        spoken = [a for a in self._items if a.delivered is not None]
        return max(spoken, key=lambda a: a.delivered or 0) if spoken else None

    def to_row(self, item: Announcement) -> dict[str, Any]:
        stamp = item.delivered or item.created
        row: dict[str, Any] = {
            "id": item.id,
            "when": spoken_stamp(stamp),
            "kind": item.kind,
            "text": item.text[:240],
            "state": item.state,
        }
        if item.count > 1:
            row["count"] = item.count
        if item.pushed is not None:
            row["on_phone"] = True
        return row

    def items(
        self,
        since: float | None = None,
        until: float | None = None,
        *,
        unread_only: bool = False,
        kinds: Iterable[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Rows for the owner, oldest first within the window."""
        wanted = {k for k in kinds} if kinds else None
        rows = []
        for a in self._items:
            if a.cancelled or (unread_only and not a.live):
                continue
            if wanted is not None and a.kind not in wanted:
                continue
            stamp = a.delivered or a.created
            if since is not None and stamp < since:
                continue
            if until is not None and stamp >= until:
                continue
            rows.append(a)
        rows.sort(key=lambda a: a.delivered or a.created)
        return [self.to_row(a) for a in rows[-limit:]]

    def history(self, since: float | None = None) -> list[dict[str, Any]]:
        rows = [asdict(a) for a in self._items if a.delivered is not None]
        if since is not None:
            rows = [r for r in rows if (r["delivered"] or 0) >= since]
        return rows
